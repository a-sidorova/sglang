# Copyright 2026 Upstage AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SolarOpen3 scheduling, KDA numerics, and graph-capture checks."""

from __future__ import annotations

import functools
import inspect
import logging
import math
import os

logger = logging.getLogger(__name__)


def _prefill_capture_required(model_runner, capture) -> bool:
    """Keep Solar's strict fallback policy local to the adapter.

    Use the engine's backend/hidden-state accessors rather than reconstructing
    resolved settings from CLI flags. Integration tests exercise both policies.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup

    if (
        cuda_graph_setup.check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
        or model_runner.is_draft_worker
    ):
        return False
    # All remaining deliberate prefill skips return the eager runner. A missing
    # runner means an unsupported model or capture shape and must still fail.
    if capture.runner is None:
        return True
    if (
        model_runner.spec_algorithm.is_eagle()
        and cuda_graph_setup.get_server_return_hidden_states_mode()
        < CaptureHiddenMode.FULL
        and cuda_graph_setup.check_cuda_graph_backend(
            Phase.PREFILL, Backend.TC_PIECEWISE
        )
    ):
        return False
    if (
        model_runner.lora_manager is not None
        and not model_runner.lora_manager.supports_prefill_cuda_graph
    ):
        return False
    parallel = cuda_graph_setup.get_parallel()
    return not (
        cuda_graph_setup.check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
        and parallel.enable_prefill_cp
        and parallel.pp_size > 1
    )


def _decode_capture_required(model_runner, requested: bool) -> bool:
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup

    if not requested:
        return False
    if model_runner.device not in ("cuda", "musa", "cpu", "npu", "xpu") and not (
        cuda_graph_setup.current_platform.is_out_of_tree()
        and cuda_graph_setup.current_platform.support_cuda_graph()
    ):
        return False
    if (
        model_runner.spec_algorithm.is_speculative()
        and not model_runner.is_draft_worker
        and cuda_graph_setup.get_disagg().disaggregation_mode == "prefill"
    ):
        return False
    if not model_runner.is_generation:
        return False
    if (
        cuda_graph_setup.get_model().model_impl.lower()
        == cuda_graph_setup.ModelImpl.MINDSPORE
    ):
        return False
    if model_runner.device == "cpu":
        return cuda_graph_setup.get_flags().capture.enable_torch_compile
    return not cuda_graph_setup.check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)


def _require_cuda_graph_capture() -> None:
    """Require requested CUDA graphs while keeping shared engine sources unchanged.

    Solar owns the strict startup check and its deliberate-skip predicates.
    SOLAR_OPEN3_ALLOW_GRAPH_FALLBACK=1 retains the engine's usual behavior.
    """
    if os.environ.get("SOLAR_OPEN3_ALLOW_GRAPH_FALLBACK", "") == "1":
        return
    from sglang.srt.model_executor.model_runner_components import (
        cuda_graph_setup as _cgs,
    )

    if getattr(_cgs.capture_cuda_graphs, "_solar_open3_wrapped", False):
        return
    _inner = _cgs.capture_cuda_graphs

    @functools.wraps(_inner)
    def _checked(*args, **kwargs):
        bound = inspect.signature(_inner).bind(*args, **kwargs)
        bound.apply_defaults()
        model_runner = bound.arguments["model_runner"]
        decode_requested = bound.arguments["capture_decode_cuda_graph"]
        out = _inner(*args, **kwargs)
        bad = []
        for name in ("prefill", "decode"):
            phase = getattr(out, name, None)
            if phase is None:
                raise RuntimeError(
                    f"SGLang's capture result has no {name!r} phase; "
                    "the CUDA-graph check cannot inspect its capture outcome."
                )
            # Upstream returns the bare eager runner when no capture bucket
            # fits. Refuse this fallback before consumers expect GraphCapture.
            if phase is out.eager_runner:
                bad.append(name)
            elif phase.runner is None or phase.runner is out.eager_runner:
                required = (
                    _prefill_capture_required(model_runner, phase)
                    if name == "prefill"
                    else _decode_capture_required(model_runner, decode_requested)
                )
                if required:
                    bad.append(name)
        if bad:
            raise RuntimeError(
                "CUDA graph capture did not happen for: "
                + ", ".join(bad)
                + ". SGLang falls back "
                "to eager here and keeps serving, so the server would come up and answer every "
                "request at eager speed while reporting nothing -- any timing taken from it would "
                "be wrong. The server log above says which check refused the capture. Set "
                "SOLAR_OPEN3_ALLOW_GRAPH_FALLBACK=1 to serve anyway."
            )
        return out

    _checked._solar_open3_wrapped = True
    _cgs.capture_cuda_graphs = _checked


def exact_kda_prefill_diagonal() -> None:
    """Keep the KDA prefill off the fork's small-grid FUSE_DIAGONAL path, which mis-decays gates
    losing > ~87 nats inside a 16-row sub-chunk (trained SDA layers do). Only the `triton` prefill
    backend reaches the patched kernel (`flashkda` too: it falls back to it for every unbounded
    gate), so any other is refused. Evidence: .jobs/261012_so3_serving_options.
    """
    from sglang.srt.layers.attention.solar_open3.settings import runtime_options

    sa = (
        runtime_options()
    )  # set in every process that builds a model (ModelRunner needs it)
    backend = sa.linear_attn_prefill_backend or sa.linear_attn_backend
    if backend not in ("triton", "flashkda"):
        raise RuntimeError(
            f"solar_open3: --linear-attn-prefill-backend {backend!r} runs its own KDA prefill "
            "kernel, unchecked against the fast-decaying gates trained SDA layers carry. Serve "
            "with the default `triton` (FlashSDA also rebinds that one)."
        )
    from sglang.kernels.ops.attention.fla import kda as _kda

    if getattr(_kda.chunk_kda_fwd_intra, "_solar_open3_wrapped", False):
        return
    _inner = _kda.chunk_kda_fwd_intra
    if "fuse_diagonal" not in inspect.signature(_inner).parameters:
        raise RuntimeError(
            "solar_open3: sglang's chunk_kda_fwd_intra no longer takes `fuse_diagonal`. Re-check "
            "its fused-diagonal KDA prefill against fast-decaying gates (see "
            "plugin.exact_kda_prefill_diagonal) before serving SDA/KDA layers."
        )

    @functools.wraps(_inner)
    def _exact(*args, **kwargs):
        kwargs["fuse_diagonal"] = False
        return _inner(*args, **kwargs)

    _exact._solar_open3_wrapped = True
    _kda.chunk_kda_fwd_intra = _exact
    logger.info(
        "[solar_open3] KDA prefill: fused-diagonal path disabled (exact diagonal blocks)"
    )


def align_routed_prefill_chunks(q_block: int) -> None:
    """End every non-final prefill chunk of a request on the ABSOLUTE `q_block` grid.

    A routed layer elects once per `q_block` rows of the document; a chunk boundary inside a block
    makes each part elect from its own rows, so a prompt's logprobs would depend on what else was
    scheduled with it. SGLang's `truncation_align_size` does not cover this: it aligns the first
    chunk's length, not prefix + length, and leaves `add_chunked_req` alone. So both `PrefillAdder`
    entry points get a chunk budget that ends on the grid. Installed by the target model's
    constructor, which runs in the scheduler process (the TP worker is in-process); the draft's
    constructor does not call it.
    """
    from sglang.srt.layers.attention.solar_open3.settings import runtime_options
    from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

    sa = runtime_options()
    if (
        q_block > 1
        and not getattr(sa, "disable_radix_cache", False)
        and getattr(sa, "mamba_radix_cache_strategy", None) == "no_buffer"
    ):
        raise ValueError(
            "solar_open3 routed prefix caching requires an intermediate SDA state "
            "at a query-block boundary; --mamba-radix-cache-strategy no_buffer "
            "only keeps the final state, which may be inside a query block. "
            "Use extra_buffer (or auto), or --disable-radix-cache."
        )
    grid = math.lcm(q_block, int(sa.page_size or 1))
    if 0 < int(sa.chunked_prefill_size or 0) < grid:
        raise RuntimeError(
            f"solar_open3: --chunked-prefill-size {sa.chunked_prefill_size} is smaller than "
            f"kv_route_q_block={q_block} (lcm with --page-size {sa.page_size}: {grid}). A chunk "
            "must hold a whole query block for routed prefill to match training; raise it."
        )
    if getattr(PrefillAdder.add_one_req, "_solar_open3_wrapped", False):
        return
    one, chunked = PrefillAdder.add_one_req, PrefillAdder.add_chunked_req

    def _take(self, prefix, budget):
        """Tokens to the last grid line within `budget`, or to the next one if there is none."""
        g = math.lcm(q_block, self.page_size)
        n = (prefix + budget) // g * g - prefix
        return n if n > 0 else n + g

    @functools.wraps(one)
    def add_one_req(self, req, *args, **kwargs):
        rem = self.rem_chunk_tokens
        prefix = req.prefix_len + req.host_hit_length  # what it will be after load-back
        if (
            rem is None
            or self.ceil_paged_tokens(len(req.full_untruncated_fill_ids) - prefix)
            <= rem
        ):
            return one(self, req, *args, **kwargs)  # unchunked, or its last chunk
        n = _take(self, prefix, rem)
        if n > rem and (self.can_run_list or rem <= 0):
            # No grid line in what is left: end this step here, where SGLang's own truncation
            # would have ended it. Only a request alone in the step may overshoot, to progress.
            return AddReqResult.OTHER
        self.rem_chunk_tokens = n
        try:
            return one(self, req, *args, **kwargs)
        finally:
            self.rem_chunk_tokens += rem - n

    @functools.wraps(chunked)
    def add_chunked_req(self, req):
        rem, avail = self.rem_chunk_tokens, int(self.rem_total_tokens)
        prefix = req.prefix_len
        budget = min(rem, avail) if avail > 0 else rem  # SGLang's own pick
        if len(req.full_untruncated_fill_ids) - prefix <= budget:
            return chunked(self, req)
        n = _take(self, prefix, budget)
        # A request mid-prefill must move, so past `avail` by < one grid when no line fits in it
        # (SGLang itself takes a whole chunk once `avail` is spent).
        lend = n - avail if 0 < avail < n else 0
        self.rem_chunk_tokens = n
        self.memory_budget.total_offset -= lend
        try:
            return chunked(self, req)
        finally:
            self.rem_chunk_tokens += rem - n
            self.memory_budget.total_offset += lend

    add_one_req._solar_open3_wrapped = True
    PrefillAdder.add_one_req, PrefillAdder.add_chunked_req = (
        add_one_req,
        add_chunked_req,
    )
    logger.info(
        "[solar_open3] routed prefill: non-final chunks end on multiples of %d", grid
    )
