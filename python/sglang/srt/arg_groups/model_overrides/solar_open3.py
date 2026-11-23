# SPDX-License-Identifier: Apache-2.0
"""Solar Open3 shares Solar Open2's reasoning and tool-call formats."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)


@_register_for("SolarOpen3ForCausalLM")
def _solar_open3_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    return {
        field: "solar_open2"
        for field in ("reasoning_parser", "tool_call_parser")
        if getattr(cfg, field) is None
    }
