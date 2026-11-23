# Copyright 2026 Upstage AI.
"""Legacy model paths used by installed FlashSDA integrations."""

# The deployed plugin wraps both paths. Keep the same class object so its
# existing idempotence guard also prevents wrapping the native loader twice.
from sglang.srt.models.solar_open2 import SolarOpen2ForCausalLM

__all__ = ["SolarOpen2ForCausalLM", "SolarOpen2KDAAttention"]


def __getattr__(name):
    # Import lazily: FlashSDA can resolve this path while loading model modules.
    if name == "SolarOpen2KDAAttention":
        from sglang.srt.models.solar_open3 import SolarOpen3KimiDeltaAttention

        return SolarOpen3KimiDeltaAttention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
