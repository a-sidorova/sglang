# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""Solar Open2 token constraints, adapted from vLLM's logits processor."""

import json
import logging
import os
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
NEG_INF = -float("inf")
_BUDGETS = {"low": 4096, "medium": 16384, "high": 32768, "xhigh": 65536, "max": 131072}
_FORCE_NEWLINE = "force_reasoning_newline"
_FORCE_END = "force_think_end"


@dataclass(frozen=True)
class SolarOpen2TokenIds:
    im_start: int | None = 128
    im_end: int | None = 129
    think_start: int | None = 130
    think_end: int | None = 131
    im_content: int | None = 132
    tool_start: int | None = 133
    tool_end: int | None = 134
    tool_call_start: int | None = 135
    tool_call_end: int | None = 136
    tool_arg_start: int | None = 137
    tool_arg_value: int | None = 138
    tool_arg_end: int | None = 139
    tool_response_start: int | None = 140
    tool_response_end: int | None = 141


class SolarOpen2State(Enum):
    REASONING = "reasoning"
    CONTENT = "content"
    TOOL_CALL_BEGIN = "tool_call_begin"
    TOOL_CALL_NAME = "tool_call_name"
    TOOL_ARG_BEGIN = "tool_arg_begin"
    TOOL_ARG_NAME = "tool_arg_name"
    TOOL_ARG_VALUE_BEGIN = "tool_arg_value_begin"
    TOOL_ARG_VALUE = "tool_arg_value"
    TOOL_ARG_END = "tool_arg_end"
    TOOL_CALL_END = "tool_call_end"


_MASK_SPEC = {
    SolarOpen2State.REASONING: (("think_end",), True),
    SolarOpen2State.TOOL_CALL_BEGIN: ((), True),
    SolarOpen2State.TOOL_CALL_NAME: (("tool_arg_start", "tool_call_end"), True),
    SolarOpen2State.TOOL_ARG_BEGIN: ((), True),
    SolarOpen2State.TOOL_ARG_NAME: (("tool_arg_value",), True),
    SolarOpen2State.TOOL_ARG_VALUE_BEGIN: (("tool_arg_end",), True),
    SolarOpen2State.TOOL_ARG_VALUE: (("tool_arg_end",), True),
    SolarOpen2State.TOOL_ARG_END: (("tool_arg_start", "tool_call_end"), True),
    SolarOpen2State.TOOL_CALL_END: (("tool_call_start", "im_end"), False),
}


def _token_id(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid token id")
    token_id = int(value)
    if token_id < 0:
        raise ValueError("token id must be non-negative")
    return token_id


def _env_ids(name):
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON list of ints") from exc
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a JSON list of ints")
    # vLLM ignores booleans/nulls in ID lists, but rejects negative IDs.
    return frozenset(
        _token_id(value)
        for value in values
        if isinstance(value, int) and not isinstance(value, bool)
    )


def _checkpoint_vocab(tokenizer_path=None):
    """Read token IDs without initializing a tokenizer (skip-tokenizer-init)."""
    paths = [tokenizer_path] if tokenizer_path else []
    if not paths:
        from sglang.srt.runtime_context import get_context, get_model, get_serving

        context = get_context()
        if context.is_config_namespace_published("serving"):
            paths.append(get_serving().tokenizer_path)
        if context.is_config_namespace_published("model"):
            paths.append(get_model().model_path)
    vocab = {}
    for path in dict.fromkeys(path for path in paths if path):
        directory = Path(path)
        if directory.is_file():
            directory = directory.parent
        local_vocab = {}
        for name in (
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
        ):
            try:
                with (directory / name).open(encoding="utf-8") as file:
                    data = json.load(file)
            except (FileNotFoundError, NotADirectoryError):
                continue
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Failed to read Solar Open2 token IDs from %s: %s", name, exc
                )
                continue
            if not isinstance(data, dict):
                continue
            decoder = data.get("added_tokens_decoder")
            if isinstance(decoder, dict):
                for token_id, token in decoder.items():
                    text = token.get("content") if isinstance(token, dict) else token
                    if isinstance(text, str):
                        local_vocab[text] = _token_id(token_id)
            added_tokens = data.get("added_tokens")
            if isinstance(added_tokens, list):
                for token in added_tokens:
                    if (
                        isinstance(token, dict)
                        and isinstance(token.get("content"), str)
                        and token.get("id") is not None
                    ):
                        local_vocab[token["content"]] = _token_id(token["id"])
            model = data.get("model")
            if isinstance(model, dict) and isinstance(model.get("vocab"), dict):
                local_vocab.update(model["vocab"])
        for token, token_id in local_vocab.items():
            vocab.setdefault(token, token_id)
    return vocab


def resolve_effort_reasoning_budget(effort: str | None) -> int:
    no_reasoning = frozenset(
        item.strip().lower()
        for item in os.environ.get(
            "SOLAR_REASONING_BUDGET_NO_REASONING_EFFORTS", "none,minimal"
        ).split(",")
        if item.strip()
    )
    normalized = (effort or "").strip().lower()
    if normalized not in no_reasoning and normalized not in _BUDGETS:
        normalized = (
            (os.environ.get("SOLAR_REASONING_BUDGET_DEFAULT_EFFORT") or "high")
            .strip()
            .lower()
        )
        if normalized not in no_reasoning and normalized not in _BUDGETS:
            normalized = "high"
    if normalized in no_reasoning:
        return 0
    default = _BUDGETS[normalized]
    env_name = f"SOLAR_REASONING_BUDGET_{normalized.upper()}"
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value < 0:
            raise ValueError("must be non-negative")
        return value
    except ValueError:
        logger.warning(
            "Ignoring malformed %s=%r; using budget %d", env_name, raw, default
        )
        return default


@dataclass(frozen=True)
class SolarOpen2Config:
    token_ids: SolarOpen2TokenIds
    extra_eos_token_ids: frozenset[int]
    leading_forbidden_ids: frozenset[int]
    reasoning_end_newline_id: int | None
    hard_reasoning_budget: int | None
    eos_overridden: bool = False

    @classmethod
    def from_tokenizer(cls, tokenizer=None, tokenizer_path=None):
        vocab = (
            tokenizer.get_vocab()
            if tokenizer is not None
            else _checkpoint_vocab(tokenizer_path)
        )
        values = {}
        aliases = {}
        for field in fields(SolarOpen2TokenIds):
            group, marker = field.name.rsplit("_", 1)
            text = f"<|{group}:{marker}|>"
            aliases[text] = field.name
            if text in vocab:
                values[field.name] = _token_id(vocab[text])
        raw = os.environ.get("SOLAR_OPEN2_TOKEN_IDS")
        if raw:
            try:
                overrides = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("SOLAR_OPEN2_TOKEN_IDS must be a JSON object") from exc
            if not isinstance(overrides, dict):
                raise ValueError("SOLAR_OPEN2_TOKEN_IDS must be a JSON object")
            names = {field.name for field in fields(SolarOpen2TokenIds)}
            for key, value in overrides.items():
                name = aliases.get(key, key)
                if name not in names:
                    raise ValueError(f"Unknown Solar Open2 token field: {key}")
                values[name] = _token_id(value)
        token_ids = SolarOpen2TokenIds(**values)

        eos = _env_ids("SOLAR_OPEN2_EOS_TOKEN_IDS")
        eos_overridden = eos is not None
        if eos is None:
            eos = set(getattr(tokenizer, "additional_stop_token_ids", None) or ())
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                eos.add(_token_id(eos_id))
            if tokenizer is None:
                eos.add(vocab.get("<|endoftext|>", 2))
            eos = frozenset(eos)

        leading = _env_ids("SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS")
        if leading is None:
            leading = frozenset(
                token_id
                for token, token_id in vocab.items()
                if token and (set(token) == {"Ċ"} or token in ("\n", "\n\n"))
            ) or frozenset({4294, 4372})
        newline = vocab.get("Ċ", vocab.get("\n")) if vocab else 4294

        name = "SOLAR_REASONING_BUDGET_HARD_LIMIT"
        raw_limit = os.environ.get(name)
        try:
            hard_limit = int(raw_limit) if raw_limit else 131072
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer number of tokens") from exc
        if hard_limit < 0:
            raise ValueError(f"{name} must be non-negative")
        return cls(token_ids, eos, leading, newline, hard_limit or None, eos_overridden)


@lru_cache(maxsize=16)
def _forbidden_table(token_ids, extra_eos_token_ids, block_tool_call_start=False):
    controls = {
        getattr(token_ids, field.name)
        for field in fields(SolarOpen2TokenIds)
        if getattr(token_ids, field.name) is not None
    }
    extra_eos_token_ids = extra_eos_token_ids - controls
    table = {}
    for state in SolarOpen2State:
        for progress in (False, True):
            if state is SolarOpen2State.CONTENT:
                allowed = (
                    ("tool_call_start", "im_end") if progress else ("tool_call_start",)
                )
                mask_eos = not progress
            else:
                allowed, mask_eos = _MASK_SPEC[state]
            forbidden = controls - {getattr(token_ids, name) for name in allowed}
            if mask_eos:
                forbidden.update(extra_eos_token_ids)
            if block_tool_call_start and token_ids.tool_call_start is not None:
                forbidden.add(token_ids.tool_call_start)
            table[state, progress] = tuple(sorted(forbidden))
    return table


def _rindex(values, needle):
    if needle is not None:
        for idx in range(len(values) - 1, -1, -1):
            if values[idx] == needle:
                return idx
    return -1


class SolarOpen2TokenFSMEnforcer:
    """Request-local token state; only newly generated reasoning consumes budget."""

    def __init__(
        self,
        prompt_token_ids,
        token_ids: SolarOpen2TokenIds,
        has_structured_outputs=False,
        extra_eos_token_ids=frozenset(),
        reasoning_budget: int | None = None,
        leading_forbidden_ids=frozenset(),
        parallel_tool_calls=True,
        reasoning_end_newline_id: int | None = 4294,
    ):
        if reasoning_budget is not None and (
            isinstance(reasoning_budget, bool)
            or not isinstance(reasoning_budget, int)
            or reasoning_budget < 0
        ):
            raise ValueError("reasoning budget must be a non-negative integer or None")
        self.token_ids = token_ids
        start = _rindex(prompt_token_ids, token_ids.think_start)
        end = _rindex(prompt_token_ids, token_ids.think_end)
        self._initial_state = (
            SolarOpen2State.REASONING
            if token_ids.think_end is not None and start > end
            else SolarOpen2State.CONTENT
        )
        call_end = _rindex(prompt_token_ids, token_ids.tool_call_end)
        message_start = _rindex(prompt_token_ids, token_ids.im_start)
        # Only a call after the last message start belongs to a continued turn.
        self._initial_completed_call = 0 <= message_start < call_end
        self._initial_leading = bool(prompt_token_ids) and (
            token_ids.think_start is not None
            and prompt_token_ids[-1] == token_ids.think_start
        )
        self._has_structured_outputs = has_structured_outputs
        self._reasoning_budget = reasoning_budget
        self.reasoning_end_newline_id = reasoning_end_newline_id
        self._parallel_tool_calls = parallel_tool_calls
        self._controls = frozenset(
            getattr(token_ids, field.name)
            for field in fields(SolarOpen2TokenIds)
            if getattr(token_ids, field.name) is not None
        )
        self._forbidden = _forbidden_table(token_ids, extra_eos_token_ids)
        self._single_forbidden = _forbidden_table(token_ids, extra_eos_token_ids, True)
        self._leading_forbidden = tuple(
            sorted(
                set(self._forbidden[SolarOpen2State.REASONING, False])
                | set(leading_forbidden_ids)
            )
        )
        self._transitions = {}
        for name, state in (
            ("think_start", SolarOpen2State.REASONING),
            ("think_end", SolarOpen2State.CONTENT),
            ("tool_call_start", SolarOpen2State.TOOL_CALL_BEGIN),
            ("tool_call_end", SolarOpen2State.TOOL_CALL_END),
            ("tool_arg_start", SolarOpen2State.TOOL_ARG_BEGIN),
            ("tool_arg_value", SolarOpen2State.TOOL_ARG_VALUE_BEGIN),
            ("tool_arg_end", SolarOpen2State.TOOL_ARG_END),
        ):
            token_id = getattr(token_ids, name)
            if token_id is not None:
                self._transitions.setdefault(token_id, state)
        self._reset()

    def _reset(self):
        self._state = self._initial_state
        self._content_progress = self._initial_completed_call
        self._completed_tool_call = self._initial_completed_call
        self._reasoning_token_count = 0
        self._last_processed_len = 0
        self._prev_was_think_start = self._initial_leading

    def _process_token(self, token_id):
        previous = self._state
        if token_id in self._transitions:
            self._state = self._transitions[token_id]
        elif previous is SolarOpen2State.TOOL_CALL_BEGIN:
            self._state = SolarOpen2State.TOOL_CALL_NAME
        elif previous is SolarOpen2State.TOOL_ARG_BEGIN:
            self._state = SolarOpen2State.TOOL_ARG_NAME
        elif previous is SolarOpen2State.TOOL_ARG_VALUE_BEGIN:
            self._state = SolarOpen2State.TOOL_ARG_VALUE
        elif previous is SolarOpen2State.TOOL_CALL_END:
            self._state = SolarOpen2State.CONTENT
        if token_id == self.token_ids.think_start:
            self._reasoning_token_count = 0
        elif previous is self._state is SolarOpen2State.REASONING:
            self._reasoning_token_count += 1
        if token_id == self.token_ids.tool_call_end:
            self._content_progress = self._completed_tool_call = True
        elif (
            previous in (SolarOpen2State.CONTENT, SolarOpen2State.TOOL_CALL_END)
            and token_id not in self._controls
        ):
            self._content_progress = True
        self._prev_was_think_start = token_id == self.token_ids.think_start

    def advance_mask_ids(self, output_token_ids):
        if len(output_token_ids) < self._last_processed_len:
            self._reset()
        for idx in range(self._last_processed_len, len(output_token_ids)):
            self._process_token(output_token_ids[idx])
        self._last_processed_len = len(output_token_ids)
        state = self._state
        single = not self._parallel_tool_calls and self._completed_tool_call
        if self._has_structured_outputs and state is SolarOpen2State.CONTENT:
            start = self.token_ids.tool_call_start
            return (start,) if single and start is not None else None
        if state is SolarOpen2State.REASONING:
            budget = self._reasoning_budget
            if (
                budget is not None
                and self._reasoning_token_count >= budget
                and self.token_ids.think_end is not None
            ):
                if (
                    budget > 0
                    and self.reasoning_end_newline_id is not None
                    and output_token_ids
                    and output_token_ids[-1] != self.reasoning_end_newline_id
                ):
                    return _FORCE_NEWLINE
                return _FORCE_END
            if self._prev_was_think_start:
                return self._leading_forbidden
        table = self._single_forbidden if single else self._forbidden
        return table[state, self._content_progress]

    def __call__(self, output_token_ids, logits: torch.Tensor):
        directive = self.advance_mask_ids(output_token_ids)
        if directive is None:
            return logits
        vocab_size = logits.shape[-1]
        if isinstance(directive, str):
            end = self.token_ids.think_end
            if end is not None and 0 <= end < vocab_size:
                newline = self.reasoning_end_newline_id
                forced = end
                if (
                    directive == _FORCE_NEWLINE
                    and newline is not None
                    and 0 <= newline < vocab_size
                    and not torch.isneginf(logits[newline]).item()
                ):
                    forced = newline
                # Preserve penalties/grammar masks already applied by the sampler.
                keep = logits[forced].clone()
                logits.fill_(NEG_INF)
                logits[forced] = keep
                return logits
            directive = self._forbidden[SolarOpen2State.REASONING, False]
        indices = [token_id for token_id in directive if 0 <= token_id < vocab_size]
        if indices:
            logits[indices] = NEG_INF
        return logits
