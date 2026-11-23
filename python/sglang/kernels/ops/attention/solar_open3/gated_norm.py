# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Upstage AI.
"""SolarOpen3 self-gate kernels carried from the simple-trainer vLLM adapter.

Both matmul accumulators stay fp32 through SiLU/sigmoid before the bf16 stores.
The caller supplies its graph token ceiling; large or unsupported shapes use
vLLM's original bf16 two-linear chain.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.utils.custom_op import register_custom_op

_SELF_GATE_FMA_MAX_TOKENS = 6
_SELF_GATE_FMA_WARPS = 4
_SELF_GATE_DOT_STAGES = 6
_SELF_GATE_DOT_UP_STAGES = 3
_SELF_GATE_DOT_WIDE_BLOCK_T_ABOVE = 144
_WIDE_TILE_SM_MAJOR = 9
_SM_MAJOR: int | None = None
_PDL_SUPPORTED: bool | None = None


def _sm_major() -> int:
    global _SM_MAJOR
    if _SM_MAJOR is None:
        try:
            _SM_MAJOR = int(torch.cuda.get_device_capability()[0])
        except Exception:
            _SM_MAJOR = 0
    return _SM_MAJOR


def _pdl_supported() -> bool:
    global _PDL_SUPPORTED
    if _PDL_SUPPORTED is None:
        _PDL_SUPPORTED = bool(is_arch_support_pdl())
    return _PDL_SUPPORTED


@triton.jit
def _self_gate_down_silu_kernel(
    y_ptr,
    wd_ptr,
    h_ptr,
    num_tokens,
    stride_y_m: tl.constexpr,
    stride_wd_r: tl.constexpr,
    stride_h_m: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_DOT: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    pid_t = tl.program_id(0)
    pid_r = tl.program_id(1)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t < num_tokens
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    if USE_DOT:
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
        acc = tl.zeros((BLOCK_T, BLOCK_R), tl.float32)
        for k0 in range(0, hidden_size, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                y_ptr + t[:, None] * stride_y_m + k[None, :],
                mask=t_mask[:, None],
                other=0.0,
            )
            b = tl.load(wd_ptr + r[:, None] * stride_wd_r + k[None, :])
            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        k = tl.arange(0, BLOCK_K)
        b = tl.load(wd_ptr + r[:, None] * stride_wd_r + k[None, :]).to(tl.float32)
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
        a = tl.load(
            y_ptr + t[:, None] * stride_y_m + k[None, :],
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        acc = tl.sum(a[:, None, :] * b[None, :, :], axis=2)
    tl.store(
        h_ptr + t[:, None] * stride_h_m + r[None, :],
        (acc * tl.sigmoid(acc)).to(h_ptr.dtype.element_ty),
        mask=t_mask[:, None],
    )


@triton.jit
def _self_gate_up_sigmoid_kernel(
    y_ptr,
    h_ptr,
    wu_ptr,
    out_ptr,
    num_tokens,
    stride_y_m: tl.constexpr,
    stride_h_m: tl.constexpr,
    stride_wu_d: tl.constexpr,
    stride_out_m: tl.constexpr,
    rank: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_RK: tl.constexpr,
    USE_DOT: tl.constexpr,
    launch_pdl: tl.constexpr = False,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t < num_tokens
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    if USE_DOT:
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
        acc = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
        for r0 in range(0, rank, BLOCK_RK):
            rr = r0 + tl.arange(0, BLOCK_RK)
            a = tl.load(
                h_ptr + t[:, None] * stride_h_m + rr[None, :],
                mask=t_mask[:, None],
                other=0.0,
            )
            b = tl.load(wu_ptr + d[:, None] * stride_wu_d + rr[None, :])
            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)
    else:
        rr = tl.arange(0, BLOCK_RK)
        b = tl.load(wu_ptr + d[:, None] * stride_wu_d + rr[None, :]).to(tl.float32)
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
        a = tl.load(
            h_ptr + t[:, None] * stride_h_m + rr[None, :],
            mask=t_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(a[:, None, :] * b[None, :, :], axis=2)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    y = tl.load(
        y_ptr + t[:, None] * stride_y_m + d[None, :], mask=t_mask[:, None], other=0.0
    ).to(tl.float32)
    tl.store(
        out_ptr + t[:, None] * stride_out_m + d[None, :],
        (y * tl.sigmoid(acc)).to(out_ptr.dtype.element_ty),
        mask=t_mask[:, None],
    )


def _self_gate_tiles(num_tokens: int, hidden_size: int, rank: int) -> dict | None:
    if hidden_size & hidden_size - 1 or rank & rank - 1:
        return None
    if num_tokens <= _SELF_GATE_FMA_MAX_TOKENS:
        block_r = max(1, 4096 // hidden_size)
        block_d = max(1, 4096 // rank)
        if rank % block_r or hidden_size % block_d:
            return None
        return dict(
            BLOCK_T=1,
            BLOCK_R=block_r,
            BLOCK_K=hidden_size,
            BLOCK_D=block_d,
            BLOCK_RK=rank,
            USE_DOT=False,
            num_warps=_SELF_GATE_FMA_WARPS,
            num_stages=2,
            num_stages_up=2,
        )
    if rank % 16 or hidden_size % 64:
        return None
    wide = (
        num_tokens > _SELF_GATE_DOT_WIDE_BLOCK_T_ABOVE
        and _sm_major() >= _WIDE_TILE_SM_MAJOR
    )
    return dict(
        BLOCK_T=64 if wide else 16,
        BLOCK_R=16,
        BLOCK_K=min(hidden_size, 128),
        BLOCK_D=64,
        BLOCK_RK=min(rank, 128),
        USE_DOT=True,
        num_warps=4,
        num_stages=_SELF_GATE_DOT_STAGES,
        num_stages_up=_SELF_GATE_DOT_UP_STAGES,
    )


def _fused_self_gate_launch(
    y: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    h: torch.Tensor,
    out: torch.Tensor,
    token_cap: int = 512,
) -> None:
    assert y.ndim == 2, (
        "the served trunk always hands apply_gate a flat [num_tokens, D]"
    )
    num_tokens, hidden_size = y.shape
    rank = w_down.shape[0]
    assert y.stride(-1) == 1 and out.stride(-1) == 1 and (h.stride(-1) == 1)
    assert w_down.stride(-1) == 1 and w_up.stride(-1) == 1
    assert w_down.dtype == y.dtype and w_up.dtype == y.dtype, (
        f"gate weights ({w_down.dtype}, {w_up.dtype}) must match the activation dtype ({y.dtype}); the fused gate does not cast"
    )
    if num_tokens == 0:
        return
    tiles = (
        None
        if num_tokens > token_cap or y.dtype == torch.float32
        else _self_gate_tiles(num_tokens, hidden_size, rank)
    )
    if tiles is None:
        torch.mm(y, w_down.t(), out=h)
        torch.nn.functional.silu(h, inplace=True)
        torch.mm(h, w_up.t(), out=out)
        out.sigmoid_()
        out.mul_(y)
        return
    assert tiles["USE_DOT"] or (
        tiles["BLOCK_K"] == hidden_size and tiles["BLOCK_RK"] == rank
    )
    pdl = {"launch_pdl": True} if _pdl_supported() else {}
    num_warps, num_stages = (tiles["num_warps"], tiles["num_stages"])
    num_stages_up = tiles["num_stages_up"]
    grid_t = triton.cdiv(num_tokens, tiles["BLOCK_T"])
    _self_gate_down_silu_kernel[grid_t, rank // tiles["BLOCK_R"]](
        y,
        w_down,
        h,
        num_tokens,
        y.stride(0),
        w_down.stride(0),
        h.stride(0),
        hidden_size,
        BLOCK_T=tiles["BLOCK_T"],
        BLOCK_R=tiles["BLOCK_R"],
        BLOCK_K=tiles["BLOCK_K"],
        USE_DOT=tiles["USE_DOT"],
        num_warps=num_warps,
        num_stages=num_stages,
        **pdl,
    )
    _self_gate_up_sigmoid_kernel[grid_t, hidden_size // tiles["BLOCK_D"]](
        y,
        h,
        w_up,
        out,
        num_tokens,
        y.stride(0),
        h.stride(0),
        w_up.stride(0),
        out.stride(0),
        rank,
        BLOCK_T=tiles["BLOCK_T"],
        BLOCK_D=tiles["BLOCK_D"],
        BLOCK_RK=tiles["BLOCK_RK"],
        USE_DOT=tiles["USE_DOT"],
        num_warps=num_warps,
        num_stages=num_stages_up,
        **pdl,
    )


@register_custom_op(out_shape="y")
def fused_self_gate(
    y: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    token_cap: int,
) -> torch.Tensor:
    out = torch.empty_like(y)
    h = y.new_empty((y.shape[0], w_down.shape[0]))
    _fused_self_gate_launch(y, w_down, w_up, h, out, token_cap=token_cap)
    return out
