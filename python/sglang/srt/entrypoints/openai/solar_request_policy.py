# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Upstage AI.
"""Solar Open2/Open3 request adaptation shared by Chat and Responses serving."""

from sglang.srt.parser.reasoning_parser import SolarOpen2Detector


def prepare_request(request, reasoning_parser, tool_call_parser) -> None:
    """Forward template effort and tool settings without replacing explicit budgets."""
    if any(
        parser in ("solar_open2", "solar_open3")
        for parser in (reasoning_parser, tool_call_parser)
    ):
        request.custom_params = {
            **(request.custom_params or {}),
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        # Use the rendered template effort for the token budget, while
        # preserving an explicit custom_params budget override.
        effort = (request.chat_template_kwargs or {}).get(
            "reasoning_effort", request.reasoning_effort
        )
        if isinstance(effort, str):
            request.custom_params.setdefault("reasoning_effort", effort)


def observe_prompt(serving, request, result, thinking_mode) -> bool:
    """Use the rendered Solar prompt; preserve other parsers' reasoning decision."""
    if serving.reasoning_parser in ("solar_open2", "solar_open3"):
        thinking_mode = SolarOpen2Detector.set_initial_state_from_prompt(
            request,
            result.prompt_ids if len(result.prompt_ids) else result.prompt,
            serving.tokenizer_manager.tokenizer,
        )
    return thinking_mode


def carry_prompt_state(request, processed_messages, reasoning_parser) -> None:
    """Carry the temporary Chat request's observed state into the Responses request."""
    if reasoning_parser in ("solar_open2", "solar_open3"):
        request._solar_open2_prompt_in_reasoning = processed_messages.require_reasoning


def reasoning_enabled(request, reasoning_parser) -> bool | None:
    """Return Solar's initial channel, or None so the caller keeps its own policy."""
    if reasoning_parser in ("solar_open2", "solar_open3"):
        return SolarOpen2Detector.generation_starts_in_reasoning(request)
    return None
