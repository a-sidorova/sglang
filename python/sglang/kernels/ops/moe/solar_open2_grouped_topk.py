# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-kernel biased grouped top-k for Solar Open2 on ROCm.

AITER's ``biased_grouped_topk`` costs 6.96 us at one token and 17.16 us at 128
(measured under graph replay on MI355X), against a 3.1 us floor for any kernel
whose input depends on the previous kernel's output. At 48 layers that is
0.33 ms of a 5.37 ms single-request decode step spent selecting 8 of 320
experts for one token. This kernel does the whole selection - sigmoid, expert
bias, group top-2 sum, group top-k mask, expert top-k, renormalise, scale - in
one program per token, so the cost is the floor plus one 320-wide pass.

Numerics follow ``grouped_topk`` in ``grouped_topk_router.py`` exactly:
selection uses the biased scores, the returned weights are the unbiased
sigmoid scores, and ties resolve to the lower expert index.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _biased_grouped_topk_kernel(
    gating_ptr,
    bias_ptr,
    out_w_ptr,
    out_i_ptr,
    stride_gm,
    stride_wm,
    stride_im,
    scale,
    n_experts: tl.constexpr,
    n_group: tl.constexpr,
    topk_group: tl.constexpr,
    topk: tl.constexpr,
    grp_size: tl.constexpr,
    GRP_PAD: tl.constexpr,
    RENORM: tl.constexpr,
):
    tok = tl.program_id(0)

    g_idx = tl.arange(0, n_group)[:, None]
    e_idx = tl.arange(0, GRP_PAD)[None, :]
    expert = g_idx * grp_size + e_idx
    valid = (e_idx < grp_size) & (expert < n_experts)

    logits = tl.load(gating_ptr + tok * stride_gm + expert, mask=valid, other=0.0).to(
        tl.float32
    )
    scores = tl.sigmoid(logits)
    bias = tl.load(bias_ptr + expert, mask=valid, other=0.0).to(tl.float32)
    biased = tl.where(valid, scores + bias, float("-inf"))

    # Group score = sum of the two largest biased scores in the group.
    top1 = tl.max(biased, axis=1)
    is_top1 = biased == top1[:, None]
    # Only blank one element even if the max repeats inside a group.
    first_top1 = tl.cumsum(is_top1.to(tl.int32), axis=1) == 1
    without_top1 = tl.where(is_top1 & first_top1, float("-inf"), biased)
    top2 = tl.max(without_top1, axis=1)
    gscore = top1 + top2

    # Keep the topk_group best groups.
    garange = tl.arange(0, n_group)
    gsel = tl.zeros([n_group], dtype=tl.int1)
    for _ in tl.static_range(topk_group):
        gbest = tl.argmax(gscore, axis=0)
        gsel = gsel | (garange == gbest)
        gscore = tl.where(garange == gbest, float("-inf"), gscore)

    cand = tl.where(gsel[:, None] & valid, biased, float("-inf"))
    cand_flat = tl.reshape(cand, [n_group * GRP_PAD])
    score_flat = tl.reshape(tl.where(valid, scores, 0.0), [n_group * GRP_PAD])
    pos = tl.arange(0, n_group * GRP_PAD)

    # Accumulate the selected weights/ids in registers. Storing each slot and
    # reloading it for the renormalise pass costs a round trip that is a large
    # share of a kernel this small.
    koffs = tl.arange(0, topk)
    acc_w = tl.zeros([topk], dtype=tl.float32)
    acc_i = tl.zeros([topk], dtype=tl.int32)
    wsum = 0.0
    for k in tl.static_range(topk):
        best = tl.argmax(cand_flat, axis=0)
        hit = pos == best
        w = tl.sum(tl.where(hit, score_flat, 0.0), axis=0)
        eid = (best // GRP_PAD) * grp_size + (best % GRP_PAD)
        slot = koffs == k
        acc_w = tl.where(slot, w, acc_w)
        acc_i = tl.where(slot, eid.to(tl.int32), acc_i)
        wsum += w
        cand_flat = tl.where(hit, float("-inf"), cand_flat)

    inv = scale / tl.maximum(wsum, 1.0e-20) if RENORM else scale
    tl.store(out_i_ptr + tok * stride_im + koffs, acc_i)
    tl.store(out_w_ptr + tok * stride_wm + koffs, acc_w * inv)


# One workgroup per token over a <=2048-element tile: a single
# wavefront keeps every reduction intra-wave (no LDS round trip),
# which matters because the 8 top-k passes each reduce the tile.
NUM_WARPS = int(os.environ.get("SGLANG_SOLAR_OPEN2_TRITON_TOPK_WARPS", "1"))


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 1 else 1


def solar_open2_biased_grouped_topk(
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> None:
    """In-place replacement for ``rocm_aiter_ops.biased_grouped_topk``."""
    num_tokens, n_experts = gating_output.shape
    topk = topk_ids.shape[1]
    grp_size = n_experts // num_expert_group
    _biased_grouped_topk_kernel[(num_tokens,)](
        gating_output,
        e_score_correction_bias,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        float(routed_scaling_factor),
        n_experts=n_experts,
        n_group=_next_pow2(num_expert_group),
        topk_group=topk_group,
        topk=topk,
        grp_size=grp_size,
        GRP_PAD=_next_pow2(grp_size),
        RENORM=bool(renormalize),
        num_warps=NUM_WARPS,
    )


def supports(
    gating_output: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
) -> bool:
    """Shape guard: the kernel materialises one padded group tile per token."""
    if (
        gating_output.ndim != 2
        or topk_ids.ndim != 2
        or not gating_output.is_cuda
        or topk_ids.device != gating_output.device
        or gating_output.shape[0] != topk_ids.shape[0]
        or gating_output.shape[0] == 0
        or num_expert_group <= 0
        or topk_group <= 0
    ):
        return False
    n_experts = gating_output.shape[1]
    if n_experts % num_expert_group:
        return False
    group_size = n_experts // num_expert_group
    if group_size < 2 or not 0 < topk_ids.shape[1] <= topk_group * group_size:
        return False
    if _next_pow2(num_expert_group) != num_expert_group:
        return False
    tile = num_expert_group * _next_pow2(n_experts // num_expert_group)
    if tile > 2048 or _next_pow2(topk_ids.shape[1]) != topk_ids.shape[1]:
        return False
    return (
        gating_output.is_contiguous()
        and topk_weights_ok(topk_ids)
        and topk_group <= num_expert_group
    )


def topk_weights_ok(topk_ids: torch.Tensor) -> bool:
    return topk_ids.dtype == torch.int32
