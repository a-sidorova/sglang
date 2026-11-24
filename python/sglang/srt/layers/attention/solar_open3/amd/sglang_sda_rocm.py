# Copyright 2026 Upstage AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Solar Open3 SDA prefill on ROCm: SGLang's KDA backend contract over aiter's kernel.

FlashSDA (the CUDA plugin that serves the exact-erase SDA prefill on sm90/sm100)
cannot run on ROCm: its kernels are TileLang with tcgen05 / wgmma / TMA. On ROCm the
kernel is aiter's ``flash_sda_fwd``
(``aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_sda.py``).

* ``SdaRocmPrefill`` is set on an SDA layer's ``RadixLinearAttention`` (the
  ``sda_prefill`` attribute) by ``SolarOpen3KimiDeltaAttention``; the KDA backend's
  ``forward_extend`` calls its ``extend`` for that layer instead of the KDA kernel
  dispatcher -- every prefill / extend / mixed batch. It takes the dispatcher's
  arguments: RAW gate plus ``A_log`` / ``dt_bias``, ACTIVATED beta, the state pool
  ``[slots, H, V, K]`` read at ``cache_indices`` and overwritten with the final
  state (rows ``-1`` read zero, nothing written), optional ``track_state`` /
  ``track_chunk_idx`` snapshot and per-64-chunk states.
* Decode and target-verify stay on SGLang's Triton recurrent kernel, which runs
  the delta rule in fp32 with the channel-wise gate (G = 8 exactly, the gate is
  block-constant). For exact erase it only needs beta capped at ``2 - 2^-6``;
  ``beta_logit_cap`` gives the logit clamp the model applies where beta is
  produced (FlashSDA's ``sglang_beta_cap``).
"""

from __future__ import annotations

import functools
import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

_UNCHECKED_ENV = "SGLANG_SDA_ROCM_UNCHECKED"
# aiter is imported where it is used, not here, so this module imports with any aiter;
# a missing flash_sda is reported by the users below and by
# SolarOpen3ForCausalLM._refuse_unserved_exact_erase.
_KERNEL_MODULE = "aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_sda"


def _kernel_module():
    """aiter's flash_sda module, or a RuntimeError naming what is missing."""
    try:
        from aiter.ops.triton._triton_kernels.chunk_delta_attn import flash_sda
    except ImportError as err:
        raise RuntimeError(
            f"Solar Open3 SDA (ROCm) needs aiter's flash_sda kernel ({_KERNEL_MODULE}), "
            "but it is not importable; install an aiter that provides it"
        ) from err
    return flash_sda
_logged = False


class SdaRocmPrefill:
    """One SDA layer's prefill kernel: ``extend`` takes what the KDA backend hands
    its kernel dispatcher and returns what the dispatcher returns (``o``, or
    ``(o, h)`` with ``return_intermediate_states``)."""

    def __init__(self, groups: int, exact_erase: bool):
        if not isinstance(exact_erase, bool):
            raise TypeError(f"exact_erase must be a bool, got {exact_erase!r}")
        self.groups = int(groups)
        self.exact_erase = exact_erase
        self._groups_checked = False

    def __repr__(self) -> str:
        return f"SdaRocmPrefill(groups={self.groups}, exact_erase={self.exact_erase})"

    def _check_groups(self, g: torch.Tensor, dt_bias: torch.Tensor | None) -> None:
        """First call: the gate (and dt_bias) must be constant inside every group of
        ``K / groups`` channels, or reading one lane per group is wrong. One host sync
        per layer per process; ``SGLANG_SDA_ROCM_UNCHECKED=1`` skips it
        (``--load-format dummy`` timing runs only: the output is meaningless)."""
        if self._groups_checked:
            return
        if os.environ.get(_UNCHECKED_ENV) != "1":
            gw = g.shape[-1] // self.groups
            blk = g.unflatten(-1, (self.groups, gw))
            ok = bool(torch.equal(blk, blk[..., :1].expand_as(blk)))
            if ok and dt_bias is not None:
                b = dt_bias.reshape(-1, self.groups, gw)
                ok = bool(torch.equal(b, b[..., :1].expand_as(b)))
            if not ok:
                raise ValueError(
                    f"Solar Open3 SDA (ROCm): the gate is not constant within {self.groups} "
                    f"groups of {gw} channels; the checkpoint does not carry the SDA group "
                    f"structure (sda_decay_groups={self.groups}). Set {_UNCHECKED_ENV}=1 only "
                    "for dummy-weight timing."
                )
        self._groups_checked = True

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        lower_bound: float | None = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        track_state: torch.Tensor | None = None,
        track_chunk_idx: torch.Tensor | None = None,
        **kwargs,
    ):
        """``h`` (``[1, NT64, H, V, K]``, the state before every 64-token chunk) is
        returned as ``None`` when ``track_state`` is given: the backend then reads the
        snapshot and never ``h``."""
        global _logged
        del kwargs  # beta_scale (already applied), chunk plans, checkpoints: not used
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Solar Open3 SDA (ROCm): prefill called under graph capture; its varlen "
                "plan is host-side. Use --cuda-graph-backend-prefill disabled or tc_piecewise."
            )
        if lower_bound is not None:
            raise ValueError("Solar Open3 SDA (ROCm): the KDA safe gate (lower_bound) is not SDA")
        if beta_is_raw:
            raise ValueError(
                "Solar Open3 SDA (ROCm): a raw beta logit reached the chunked prefill; "
                "the layer activates beta before every non-decode forward"
            )
        if q.dim() != 4 or q.shape[0] != 1:
            raise ValueError(
                f"Solar Open3 SDA (ROCm): expected q [1, T, H, K], got {tuple(q.shape)}"
            )
        T = q.shape[1]
        # Capture-bucket padding on the gate / beta (tc_piecewise dummy forwards).
        if g.shape[1] > T:
            g = g[:, :T]
        if beta.shape[1] > T:
            beta = beta[:, :T]
        if g.shape[:3] != q.shape[:3] or beta.shape[:3] != q.shape[:3]:
            raise ValueError(
                f"Solar Open3 SDA (ROCm): gate {tuple(g.shape)} / beta {tuple(beta.shape)} "
                f"do not match q {tuple(q.shape)}"
            )
        K = q.shape[3]
        if g.shape[-1] != K:
            raise ValueError(
                f"Solar Open3 SDA (ROCm): expected the {K}-lane gate, got {g.shape[-1]}"
            )
        self._check_groups(g, dt_bias)

        flash_sda_fwd = _kernel_module().flash_sda_fwd
        o = torch.empty(q.shape[:3] + (v.shape[-1],), dtype=q.dtype, device=q.device)
        _, _, h = flash_sda_fwd(
            q, k, v, g, beta, A_log, dt_bias, K**-0.5,
            groups=self.groups,
            exact_erase=self.exact_erase,
            cu_seqlens=query_start_loc,
            state_cache=ssm_states,
            state_indices=cache_indices.reshape(-1),
            state_v_first=True,
            track_state=track_state,
            track_chunk_idx=track_chunk_idx,
            output_h=return_intermediate_states and track_state is None,
            out=o,
        )  # fmt: skip
        if not _logged:
            logger.info("Solar Open3 SDA (ROCm): prefill served by aiter flash_sda (%r)", self)
            _logged = True
        if return_intermediate_states:
            return o, h
        return o


@functools.lru_cache(maxsize=8)
def beta_logit_cap(beta_scale: float, device_type: str = "cuda") -> float:
    """The smallest fp32 logit whose ``beta_scale * sigmoid`` reaches the exact-erase
    cap ``2 - 2^-6`` on the device (plus 2e-5 of margin, ~2.6 output ulps, so the
    device's sigmoid may round down). Clamping the logit there leaves every beta
    below the cap untouched and maps every one above it to the cap (the prefill
    kernel's own ``min`` makes that exact); decode and verify then erase with at
    most the cap, as training did."""
    beta_max = _kernel_module().SDA_BETA_MAX_BF16  # the cap the prefill kernel applies
    p = beta_max / float(beta_scale)
    if not 0.0 < p < 1.0:
        raise ValueError(f"beta_scale {beta_scale} cannot reach the cap {beta_max}")
    x = math.log(p / (1.0 - p)) + 2e-5
    dev = device_type if torch.cuda.is_available() else "cpu"
    t = torch.tensor([x], dtype=torch.float32, device=dev)
    # Walk up by ulps until the device's activation reaches the cap.
    for _ in range(64):
        if float(torch.sigmoid(t) * beta_scale) >= beta_max:
            break
        t = torch.nextafter(t, torch.tensor([math.inf], device=dev))
    return float(t)
