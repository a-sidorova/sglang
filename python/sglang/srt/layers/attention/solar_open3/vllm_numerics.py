# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Upstage AI.
"""Single-group fp32 sigmoid routing matching vLLM's fused grouped top-k."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _vllm_router_kernel(
    logits,
    bias,
    weights,
    indices,
    rows,
    stride_lm: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_im: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    expert = tl.arange(0, BLOCK_E)
    raw = tl.load(logits + row * stride_lm + expert, expert < E, other=0).to(tl.float32)
    # vLLM's grouped_topk CUDA kernel uses sigmoid_accurate/tanhf.
    scores = 0.5 * libdevice.tanh(0.5 * raw) + 0.5
    candidates = scores
    if HAS_BIAS:
        candidates += tl.load(bias + expert, expert < E, other=0).to(tl.float32)
    candidates = tl.where(expert < E, candidates, float("-inf"))
    slots = tl.arange(0, 32)
    selected = tl.full((32,), 0.0, tl.float32)
    selected_ids = tl.full((32,), 0, tl.int32)
    for slot in tl.static_range(K):
        best = tl.max(candidates, 0)
        idx = tl.min(
            tl.where((candidates == best) & (expert < E), expert, 2147483647), 0
        )
        # NaN comparisons can leave the INT_MAX sentinel. Keep the expert ID
        # in bounds while preserving the NaN weight for downstream validation.
        idx = tl.minimum(idx, E - 1)
        value = tl.sum(tl.where(expert == idx, scores, 0.0), 0)
        selected = tl.where(slots == slot, value, selected)
        selected_ids = tl.where(slots == slot, idx, selected_ids)
        candidates = tl.where(expert == idx, float("-inf"), candidates)
    scale = SCALE
    if RENORMALIZE:
        scale = scale / (tl.sum(selected, 0) + 1e-20)
    tl.store(weights + row * stride_wm + slots, selected * scale, slots < K)
    tl.store(indices + row * stride_im + slots, selected_ids, slots < K)


def vllm_router(
    logits: torch.Tensor,
    top_k: int,
    correction_bias: torch.Tensor | None = None,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep scores and renormalized weights fp32; exact ties use lower IDs.

    SolarOpen3 computes gate logits in fp32, matching the original vLLM model.
    The bias affects selection only, never the returned unbiased weights.
    """
    if logits.ndim != 2 or not logits.is_cuda:
        raise ValueError(
            "vLLM-compatible SolarOpen3 routing requires a CUDA logits matrix"
        )
    rows, experts = logits.shape
    if not 1 <= top_k <= min(experts, 32):
        raise ValueError(f"top_k={top_k} must be in 1..min({experts}, 32)")
    if correction_bias is not None and (
        correction_bias.shape != (experts,) or correction_bias.device != logits.device
    ):
        raise ValueError(
            "correction_bias must have one entry per expert on the logits device"
        )
    logits = logits.contiguous()
    bias = correction_bias.contiguous() if correction_bias is not None else logits
    weights = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
    indices = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    if rows:
        _vllm_router_kernel[(rows,)](
            logits,
            bias,
            weights,
            indices,
            rows,
            logits.stride(0),
            weights.stride(0),
            indices.stride(0),
            E=experts,
            K=top_k,
            BLOCK_E=triton.next_power_of_2(experts),
            HAS_BIAS=correction_bias is not None,
            RENORMALIZE=renormalize,
            SCALE=float(routed_scaling_factor),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return weights, indices
