# Copyright 2026 Upstage AI.
"""Solar Open3 SDA linear attention on ROCm (MI300 / MI355X).

The exact-erase grouped-decay prefill kernel is aiter's ``flash_sda_fwd``
(``aiter/ops/triton/_triton_kernels/chunk_delta_attn/flash_sda.py``).
``SdaRocmPrefill`` adapts it to the KDA backend's prefill call for one SDA layer;
``beta_logit_cap`` is the decode-side exact-erase cap.
"""

from .sglang_sda_rocm import SdaRocmPrefill, beta_logit_cap

__all__ = ["SdaRocmPrefill", "beta_logit_cap"]
