# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 x BF16 -> FP32 router GEMM for Solar Open2 on ROCm.

The router has to hand top-k selection FP32 logits, so the ROCm path runs
``torch.mm(x, w.t(), out_dtype=torch.float32)``. Measured under graph replay on
MI355X that costs 7.29 us at one token for a 2.5 MiB weight - 343 GB/s, and
4.2 us of it is above the 3.1 us per-kernel floor. The same weight through the
BF16 skinny GEMM takes 2.46 us, but its BF16 output would round the logits and
can flip near-tie expert selection, so it is not a substitute.

This kernel keeps the FP32 accumulate and the FP32 output and just streams the
weight with a plain tiled loop, which is all the shape needs.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _router_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    stride_xm,
    stride_wn,
    stride_om,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // n_blocks
    pid_n = pid % n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_mask = k < K
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


# Tuned on MI355X for (320 x 4096): 5.86 us at one token against 7.35 us
# for torch.mm(out_dtype=float32), and never slower up to 128 tokens.
# BLOCK_N=8 is 46 us - below tl.dot's 16-wide MFMA tile it falls off a
# cliff - and BLOCK_N=32 halves the workgroup count to 10 of 256 CUs.
BLOCK_N = int(os.environ.get("SGLANG_SOLAR_OPEN2_ROUTER_GEMV_BLOCK_N", "16"))
BLOCK_K = int(os.environ.get("SGLANG_SOLAR_OPEN2_ROUTER_GEMV_BLOCK_K", "256"))
NUM_WARPS = int(os.environ.get("SGLANG_SOLAR_OPEN2_ROUTER_GEMV_WARPS", "2"))
NUM_STAGES = int(os.environ.get("SGLANG_SOLAR_OPEN2_ROUTER_GEMV_STAGES", "2"))
# tl.dot needs M>=16; one token still pays for a 16-row tile, which is free
# here because the kernel is weight-bandwidth bound.
BLOCK_M = 16


def router_gemv_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` in FP32 from BF16 operands."""
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    _router_gemv_kernel[grid](
        x,
        weight,
        out,
        m,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        N=n,
        K=k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


def supports(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.dim() == 2
        and weight.dim() == 2
        and x.shape[0] > 0
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.shape[1] == weight.shape[1]
        and x.shape[1] % BLOCK_K == 0
    )
