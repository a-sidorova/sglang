# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning detector shared by Solar Open2 and Solar Open3."""

import os
import re
from typing import Optional

from sglang.srt.parser.reasoning_parser import (
    BaseReasoningFormatDetector,
    StreamingParseResult,
)


class SolarOpen2Detector(BaseReasoningFormatDetector):
    """Solar's prompt prefills <|think:start|>; only the end tag is generated.

    Adapted from vLLM's SolarOpen2ReasoningParser. SGLang keeps calling the
    text detector after reasoning ends, so no token-ID boundary hook is needed.
    """

    # Fallback for callers without a rendered prompt; the checkpoint opens
    # thinking only for these efforts. Serving uses the actual prompt below.
    OPEN_THINK_EFFORTS = frozenset({"medium", "high", "xhigh", "max"})
    DEFAULT_EFFORT_ENV = "SOLAR_OPEN2_PARSER_DEFAULT_EFFORT"
    _embedded_tool_call_pattern = re.compile(
        r"<\|tool_call:start\|>.+?\n"
        r"(?:<\|tool_arg:start\|>.*?<\|tool_arg:end\|>\n?)*"
        r"<\|tool_call:end\|>",
        re.DOTALL,
    )

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request=None,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
        tokenizer=None,
    ):
        prompt_state = getattr(request, "_solar_open2_prompt_in_reasoning", None)
        input_ids = getattr(request, "input_ids", None)
        if prompt_state is None and input_ids is not None and tokenizer is not None:
            prompt_state = self.set_initial_state_from_prompt(
                request, input_ids, tokenizer
            )
        if force_reasoning is None:
            force_reasoning = self.generation_starts_in_reasoning(request)
        super().__init__(
            "<|think:start|>",
            "<|think:end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="solar_open2",
            force_nonempty_content=force_nonempty_content,
        )
        if prompt_state is not None:
            # The rendered prompt outranks effort guesses and history markers.
            self._in_reasoning = prompt_state

    @classmethod
    def set_initial_state_from_prompt(cls, request, prompt, tokenizer=None) -> bool:
        if not isinstance(prompt, str):
            prompt = tokenizer.decode(prompt, skip_special_tokens=False)
        in_reasoning = prompt.rfind("<|think:start|>") > prompt.rfind("<|think:end|>")
        # Private, request-local state; it stays outside the public API schema.
        request._solar_open2_prompt_in_reasoning = in_reasoning
        return in_reasoning

    @classmethod
    def generation_starts_in_reasoning(cls, request=None) -> bool:
        prompt_state = getattr(request, "_solar_open2_prompt_in_reasoning", None)
        if prompt_state is not None:
            return prompt_state
        if (
            getattr(request, "continue_final_message", False)
            or not getattr(request, "add_generation_prompt", True)
            or getattr(request, "chat_template", None) is not None
        ):
            return False
        # SGLang renders template kwargs over the request's effort field.
        kwargs = getattr(request, "chat_template_kwargs", None)
        if isinstance(kwargs, dict) and "reasoning_effort" in kwargs:
            effort = kwargs["reasoning_effort"]
        else:
            effort = getattr(request, "reasoning_effort", None)
            if effort is None:
                effort = getattr(getattr(request, "reasoning", None), "effort", None)
            if effort is None:
                effort = os.environ.get(cls.DEFAULT_EFFORT_ENV) or "high"
        return isinstance(effort, str) and effort in cls.OPEN_THINK_EFFORTS

    def _detect_and_parse_impl(self, text: str) -> StreamingParseResult:
        if self.think_end_token in text:
            reasoning, content = text.split(self.think_end_token, 1)
        elif self._in_reasoning:
            reasoning, content = text, ""
        else:
            return StreamingParseResult(normal_text=text)

        # vLLM recovers complete calls emitted inside reasoning in non-streaming
        # responses. Append them after real content to preserve its text prefix.
        if "<|tool_call:start|>" in reasoning:
            blocks = []

            def collect(match):
                blocks.append(match.group(0))
                return ""

            remaining = self._embedded_tool_call_pattern.sub(collect, reasoning)
            if blocks:
                reasoning = remaining.strip()
                content = "\n".join(part for part in [content, *blocks] if part)
        return StreamingParseResult(normal_text=content, reasoning_text=reasoning)

    def _parse_streaming_increment_impl(self, new_text: str) -> StreamingParseResult:
        if not self._in_reasoning:
            return StreamingParseResult(normal_text=new_text)
        self._buffer += new_text
        if self.think_end_token in self._buffer:
            reasoning, content = self._buffer.split(self.think_end_token, 1)
            self._buffer = ""
            self._in_reasoning = False
            return StreamingParseResult(normal_text=content, reasoning_text=reasoning)
        if not self.stream_reasoning:
            return StreamingParseResult()
        holdback = self._ends_with_partial_token(self._buffer, self.think_end_token)
        end = len(self._buffer) - holdback
        reasoning, self._buffer = self._buffer[:end], self._buffer[end:]
        return StreamingParseResult(reasoning_text=reasoning)
