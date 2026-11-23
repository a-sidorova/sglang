# SPDX-License-Identifier: Apache-2.0
"""Solar Open2/3's request-local template and reasoning-budget enforcement."""

from typing import Any

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.solar_open2 import (
    SolarOpen2Config,
    SolarOpen2TokenFSMEnforcer,
    resolve_effort_reasoning_budget,
)


class SolarOpen2LogitProcessor(CustomLogitProcessor):
    """Enforce Solar Open2/3's template using SGLang's custom processor API.

    Launch Solar Open2/3 with ``--enable-custom-logit-processor`` to apply it
    server-wide. Other models can select it per request with
    ``custom_logit_processor=SolarOpen2LogitProcessor.to_str()``.
    ``custom_params`` is optional (e.g. a dict containing ``reasoning_effort``).
    Parallel tool calls default to allowed; False/0 limits the turn to one call.
    The default effort is high; its budget is independent of max_new_tokens.
    """

    def __init__(self):
        self._tokenizer = None
        self._config = None

    @staticmethod
    def validate_params(params: dict[str, Any]) -> None:
        if not isinstance(params, dict):
            raise ValueError("Solar Open2 custom_params must be a dictionary")
        for key in ("disable_solar_open2_logits_processor", "parallel_tool_calls"):
            value = params.get(key)
            if value is not None and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer flag")
        effort = params.get("reasoning_effort")
        if effort is not None and not isinstance(effort, str):
            raise ValueError("reasoning_effort must be a string effort name")

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: list[dict[str, Any] | None] | None = None,
    ) -> torch.Tensor:
        for row, params in enumerate(custom_param_list or []):
            if not params:
                continue
            self.validate_params(params)
            if params.get("disable_solar_open2_logits_processor"):
                continue
            req = params.get("__req__")
            if req is None:
                continue
            enforcer = getattr(req, "_solar_open2_fsm", None)
            if enforcer is None:
                tokenizer = req.tokenizer
                if self._config is None or self._tokenizer is not tokenizer:
                    self._config = SolarOpen2Config.from_tokenizer(tokenizer)
                    self._tokenizer = tokenizer
                config = self._config
                budget = resolve_effort_reasoning_budget(params.get("reasoning_effort"))
                if config.hard_reasoning_budget is not None:
                    budget = min(budget, config.hard_reasoning_budget)
                eos = config.extra_eos_token_ids
                if not config.eos_overridden:
                    eos = eos | frozenset(req.eos_token_ids or ())
                sampling_params = req.sampling_params
                parallel = params.get("parallel_tool_calls")
                enforcer = SolarOpen2TokenFSMEnforcer(
                    req.origin_input_ids,
                    config.token_ids,
                    has_structured_outputs=any(
                        getattr(sampling_params, key, None) is not None
                        for key in ("json_schema", "regex", "ebnf", "structural_tag")
                    ),
                    extra_eos_token_ids=eos,
                    reasoning_budget=budget,
                    leading_forbidden_ids=config.leading_forbidden_ids,
                    parallel_tool_calls=parallel is None or parallel != 0,
                    reasoning_end_newline_id=config.reasoning_end_newline_id,
                )
                # Keep state on the request so filtering/merging batches and
                # deserializing another processor instance cannot mix histories.
                req._solar_open2_fsm = enforcer
            enforcer(req.output_ids, logits[row])
        return logits
