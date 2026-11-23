"""The trainer's arithmetic for steps where SGLang's own kernels compute something else.

Each function reproduces what the compiled training forward (simple-trainer, torch.compile per
block, Inductor defaults) runs for one op, so serving computes the function the weights were
trained with. The training op each one reproduces is named in its docstring.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rne_bf16_bits(f):
    """fp32 -> bf16 bit pattern (int32 in [0, 0xFFFF]), round to nearest even: ``cvt.rn.bf16.f32``
    for finite input, written in integers so it does not depend on the conversion instruction."""
    u = f.to(tl.int32, bitcast=True)
    u = u + (0x7FFF + ((u >> 16) & 1))
    return (u >> 16) & 0xFFFF


@triton.jit
def _bf16_bits_to_f32(b):
    return (b << 16).to(tl.float32, bitcast=True)


@triton.jit
def _trainer_router_kernel(
    logits_ptr,
    w_ptr,
    ids_ptr,
    M,
    stride_lm,
    stride_wm,
    stride_im,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    COL_BITS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_E)
    rmask = rows < M
    cmask = cols < E
    x = tl.load(
        logits_ptr + rows[:, None] * stride_lm + cols[None, :],
        mask=rmask[:, None] & cmask[None, :],
        other=0.0,
    ).to(tl.float32)
    # scores = sigmoid(fp32 logits) stored as bf16 (Inductor: tl.sigmoid, rounded at the bf16 store)
    s_bits = _rne_bf16_bits(tl.sigmoid(x))
    # quack top-k order as a unique integer key: score desc, then the lower expert id
    MASK: tl.constexpr = (1 << COL_BITS) - 1
    key = (s_bits << COL_BITS) | (MASK - cols[None, :])
    key = tl.where(cmask[None, :], key, -1)
    kk = tl.arange(0, BLOCK_K)
    ids = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)
    g = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    S = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in tl.static_range(K):
        kmax = tl.max(key, axis=1)
        col = MASK - (kmax & MASK)
        val = _bf16_bits_to_f32(kmax >> COL_BITS)
        S = S + val  # fp32 left fold in slot order
        ids = tl.where(kk[None, :] == k, col[:, None], ids)
        g = tl.where(kk[None, :] == k, val[:, None], g)
        key = tl.where(key == kmax[:, None], -1, key)
    # the sum stored as bf16, + fp32(1e-20), fp32 '/' (div.full), stored as bf16
    den = _bf16_bits_to_f32(_rne_bf16_bits(S)) + 1e-20
    w = _bf16_bits_to_f32(_rne_bf16_bits(g / den[:, None]))
    smask = rmask[:, None] & (kk[None, :] < K)
    tl.store(w_ptr + rows[:, None] * stride_wm + kk[None, :], w, mask=smask)
    tl.store(ids_ptr + rows[:, None] * stride_im + kk[None, :], ids, mask=smask)


def trainer_router(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The trainer's ``TokenChoiceTopKRouter`` (sigmoid, ``router_topk_backend='quack'``,
    ``route_norm``, route_scale 1, no expert bias) from its bf16 gate logits:
    ``s = bf16(sigmoid(fp32(logits)))``; the top_k largest ``s``, ties to the lower expert id, in
    that slot order; ``w = bf16(s / (bf16(sum s) + 1e-20))``.

    Returns weights fp32 [M, top_k] holding those bf16 values and ids int32 [M, top_k]. No host
    sync and static shapes, so it runs under CUDA-graph capture.
    """
    assert router_logits.dtype == torch.bfloat16 and router_logits.dim() == 2
    router_logits = router_logits.contiguous()
    M, E = router_logits.shape
    col_bits = max(1, (E - 1).bit_length())
    assert 16 + col_bits <= 31, E
    w = torch.empty((M, top_k), dtype=torch.float32, device=router_logits.device)
    ids = torch.empty((M, top_k), dtype=torch.int32, device=router_logits.device)
    if M:
        _trainer_router_kernel[(M,)](
            router_logits,
            w,
            ids,
            M,
            router_logits.stride(0),
            w.stride(0),
            ids.stride(0),
            E=E,
            BLOCK_E=triton.next_power_of_2(E),
            COL_BITS=col_bits,
            K=top_k,
            BLOCK_K=triton.next_power_of_2(top_k),
            BLOCK_M=1,
            num_warps=4,
        )
    return w, ids
