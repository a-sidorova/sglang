# Copyright 2026 Upstage AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Native SolarOpen3 inference: LSSA routed GQA over the SolarOpen2 hybrid trunk.

The model reuses SolarOpen2 KDA, MoE, layer communication, and weight loading.
Routed attention, depth attention-residuals, GatedNorm, and MTP integration are
implemented in this module and sglang.srt.layers.attention.solar_open3.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import os
from copy import copy
from typing import Optional

import torch
from torch import nn

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.configs.solar_open3 import (
    SolarOpen3Config,
    _is_kda_layer,
    _patch_config,
    pin_served_options,
    validate_precision_config,
)
from sglang.srt.distributed import (
    attention_tensor_model_parallel_all_reduce,
    attention_tensor_model_parallel_quant_all_reduce,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.solar_open3.precision import hold_fp32_params
from sglang.srt.layers.attention.solar_open3.routed_core import (
    _attn_res_fake,
    _attn_res_launch,
    _cp_attn_res_fake,
    _cp_attn_res_launch,
    _current_sm_major,
    _gluon_prefill_mod,
    _resolve_prefill_body,
    _routing_tristate,
    _validate_scorer_config,
    attn_res_writes_before,
    routed_decode_batched,
    routed_prefill_batched,
)
from sglang.srt.layers.attention.solar_open3.runtime import (
    _require_cuda_graph_capture,
    align_routed_prefill_chunks,
    exact_kda_prefill_diagonal,
)
from sglang.srt.layers.attention.solar_open3.settings import runtime_options
from sglang.srt.layers.attention.solar_open3.vllm_numerics import vllm_router
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopK,
    capture_routed_experts_if_allowed,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
    get_tc_piecewise_forward_context,
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.models.solar_open2 import (
    SolarOpen2Attention,
    SolarOpen2DecoderLayer,
    SolarOpen2ForCausalLM,
    SolarOpen2KimiDeltaAttention,
    SolarOpen2MoE,
    _apply_gqa_output_gate,
)
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import is_hip, make_layers
from sglang.srt.utils.common import BumpAllocator
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_FELL_BACK: set[str] = set()


def _fall_back(what: str, why: str) -> None:
    """A trainer-matched path that cannot run here leaves SGLang's own in place; say so once."""
    if what not in _FELL_BACK:
        _FELL_BACK.add(what)
        logger.warning("[solar_open3] %s off, SGLang's own path serves: %s", what, why)


# The layer-id / cache properties sglang reads off the config. solar_open3 splits GQA from KDA by
# `gqa_layers`, a field solar_open2 reads too, so the same patcher answers for both -- plus, for a
# LINEAR MTP head, one reserved state layer (see `_reserve_linear_draft_state_layer`).


def linear_draft_state_layer_id(config) -> int:
    """The mamba-pool layer id the linear MTP head's recurrent + conv state occupies.

    The HIGHEST TRUNK GQA INDEX. Those ids are free in the mamba pool -- a GQA layer keeps no
    recurrent state, so no trunk layer claims a slot under them -- and every consumer of the
    pool's layer list treats it as "which ids get a slot", never as "which trunk layers are
    linear" (checked against the four consumers in kv_cache_configurator and unified_memory_pool).

    ⚠️ It has to be INSIDE the rank's PIPELINE-PARALLEL range, because the pool builder keeps
    only ids that fall in it -- so one past the trunk, the obviously-collision-free choice, is
    filtered straight back out and the head silently gets no slot (measured, job 110659). The id
    this returns is GLOBAL, so it satisfies that on the LAST pp rank only; the earlier ranks would
    build a pool without it and refuse to serve. ``_refuse_linear_head_preconditions`` therefore
    refuses ``--pp-size > 1`` by name rather than hand back an id that holds on one rank in eight.

    It is a POOL id, not a module index: the head's decoder layer stays at index 0 (its weight keys
    are ``mtp.layers.0.*`` and the draft's own KV pool only knows index 0), and only the id its
    ``RadixLinearAttention`` reports is moved -- that id is what every pool lookup keys on.

    Falls back to 0 for an all-linear trunk, which has no free id; ``serves_linear_draft`` is False
    there, so 0 is only ever used as the sole layer of the head's own private pool.
    """
    free = [i for i in range(config.num_hidden_layers) if not _is_kda_layer(config, i)]
    return free[-1] if free else 0


def linear_draft_share_pool_opt_out() -> bool:
    """True when the operator asked for the head's state to stay in a PRIVATE pool.

    ONE predicate, because two places read it and they have to agree: `serves_linear_draft` stops
    reserving the layer, and `_refuse_linear_head_preconditions` recognises the same request and
    downgrades its refusal to the warning that says what the control arm is measuring. Parsed
    independently in each, a change to the accepted spelling in one makes a deliberate control-arm
    launch die on the very refusal that exists to explain it.
    """
    return os.environ.get("SO3_LINEAR_DRAFT_SHARE_POOL", "1") == "0"


def serves_linear_draft(config) -> bool:
    """True when this server runs the checkpoint's LINEAR MTP head as an EAGLE draft.

    Gated on the server args, not just the checkpoint: reserving the layer costs state-pool memory
    on every rank, and a server that is not speculating has no head to keep state for. Off outside
    a server (no global server args) and under ``SO3_LINEAR_DRAFT_SHARE_POOL=0``, which is the
    control arm that puts the head's state back in a private pool the radix cache never sees.
    """
    if getattr(config, "mtp_num_hidden_layers", 0) < 1:
        return False
    if linear_draft_share_pool_opt_out():
        return False
    if all(_is_kda_layer(config, i) for i in range(config.num_hidden_layers)):
        return False  # no free pool id to reserve (see linear_draft_state_layer_id)
    from sglang.srt.layers.attention.solar_open3.settings import runtime_options

    try:
        server_args = runtime_options()
    except ValueError:
        # THE ONE tolerated failure, caught by type rather than by `except Exception`:
        # `RuntimeContext.server_args` raises ValueError("Global server args is not set yet!")
        # outside a server (import time, unit tests), where there is nothing to serve and nothing
        # to reserve. Every other failure -- a renamed attribute, an sglang refactor, a
        # half-built process -- must surface here instead of turning into "do not reserve the
        # head's state layer", which is the silent-stale-state configuration this path exists to
        # prevent (R13: catch only to raise, never to fall back).
        return False
    algorithm = getattr(server_args, "speculative_algorithm", None)
    return str(algorithm or "").upper() == "EAGLE"


def with_linear_draft_layer(params, config):
    """``params`` plus the head's reserved state layer, when this server serves a linear draft.

    The single source for the reservation: TWO config classes can reach the pool builder -- the one
    registered here and the checkpoint's own through its ``auto_map`` -- and each has its own
    ``mamba2_cache_params``. Reserving in only one of them puts the head's state in a private pool
    on the other path, silently (see `_reserve_linear_draft_state_layer`).
    """
    if not serves_linear_draft(config):
        return params
    reserved = linear_draft_state_layer_id(config)
    if reserved in params.layers:
        return params
    return dataclasses.replace(params, layers=list(params.layers) + [reserved])


# Stamped on the wrapper `_reserve_linear_draft_state_layer` installs, so that re-reserving is a
# no-op while a property `_patch_config` put back is re-wrapped. See that function.
_DRAFT_LAYER_WRAPPER = "_solar_open3_draft_layer_reserved"


def _reserve_linear_draft_state_layer(config) -> None:
    """Append the head's state layer to what the mamba pool is built from.

    Sharing provides prefix-cache restoration (`MambaPool.copy_from`), interval snapshots
    (the ping-pong track), and fresh-slot zeroing (`clear_slots`). Only the first two depend on
    the radix cache. `HybridReqToTokenPool.alloc` marks every newly allocated slot for clearing,
    including under --disable-radix-cache, so the reservation is not gated on caching being on.
    All three walk the pool's layer axis. A head in a private pool receives none of them and may
    retain the previous occupant's state; the target still verifies its output, but stale state
    can affect acceptance. Reserving the layer lets the existing machinery carry the head too.

    Costs one layer of conv + temporal per SLOT -- ~200 KB at the 94B shape, and ~0.90 GB per
    rank across the whole pool (~2.4% of it) -- and only when a linear head is actually being
    served as a draft.

    IDEMPOTENT ON THE PROPERTY, NEVER ON A CLASS FLAG. ``_patch_config`` re-installs its own
    unwrapped ``mamba2_cache_params`` on this same class whenever it is handed an instance with no
    ``_linear_layer_ids`` yet -- which is EVERY fresh instance, because its own ``hasattr`` guard
    evaluates the property and that read raises on one. The draft config is exactly such an
    instance of the same trust-remote-code class the target came through, so a class flag saying
    "already reserved" would leave the class holding the PLAIN property with no way back: every
    later read returns the layer list without the reserved id, which is the silent unreserved-pool
    state this whole path exists to eliminate (job 110656). Marking the WRAPPER re-wraps whatever
    is installed now, and reading the property off the CLASS (never off an instance) never
    evaluates it.
    """
    cls = type(config)
    current = getattr(cls, "mamba2_cache_params", None)
    if not isinstance(current, property):
        raise RuntimeError(
            f"{cls.__name__}.mamba2_cache_params is {type(current).__name__}, not a property, so "
            "the linear draft's state layer cannot be reserved by wrapping it. `_patch_config` "
            "installs it as one and must run first (see `patch_config`)."
        )
    if getattr(current.fget, _DRAFT_LAYER_WRAPPER, False):
        return

    @property
    def mamba2_cache_params(self):
        return with_linear_draft_layer(current.fget(self), self)

    setattr(mamba2_cache_params.fget, _DRAFT_LAYER_WRAPPER, True)
    cls.mamba2_cache_params = mamba2_cache_params


HEAD_SWIGLU_CLAMP_OFF = "off"


def _bound(config, key: str, allow_off: bool = False):
    """A clamp / clip bound off the HF config: a number, None (no bound) or, for an MTP head field,
    "off" (``allow_off``)."""
    value = getattr(config, key, None)
    if value == HEAD_SWIGLU_CLAMP_OFF and not allow_off:
        raise ValueError(f'solar_open3: {key} has no "off" state, got {value!r}')
    return value


def resolve_head_swiglu_limit(trunk_limit, head_limit):
    """COPY of ``hf/configuration_solar_open3.py:resolve_head_swiglu_limit`` (this package is copied
    into an SGLang venv and cannot import the trainer; test_deepseek_clamp_export.py holds them equal):
    None follows the trunk, "off" is no clamp whatever the trunk does, a number is the head's own."""
    if head_limit == HEAD_SWIGLU_CLAMP_OFF:
        return None
    if head_limit is None:
        return trunk_limit
    return float(head_limit)


def resolve_mtp_swiglu_limit(config, key: str = "swiglu_limit"):
    return resolve_head_swiglu_limit(
        _bound(config, key),
        _bound(config, f"mtp_{key}", allow_off=True),
    )


def patch_config(config):
    """Pin and validate either config class SGLang may hand over; reserve the head's state layer."""
    validate_precision_config(config)
    if (
        getattr(config, "gqa_layers", None) is None
    ):  # the gqa_interval fallback is not served
        raise ValueError(
            "solar_open3 SGLang: config.json has no `gqa_layers` (the GQA/KDA split "
            "is read from that list only); re-export the checkpoint"
        )
    pin_served_options(config)
    if (
        "SolarOpen3ForCausalLMNextN" in (getattr(config, "architectures", None) or [])
        and getattr(config, "num_nextn_predict_layers", None)
        != config.mtp_num_hidden_layers
    ):
        raise ValueError(
            "solar_open3 SGLang: the draft config.json lacks `num_nextn_predict_layers` "
            "(SGLang then sizes the draft KV pool as the whole trunk); regenerate the "
            "draft dir with `python -m scripts.so3_mtp_draft`"
        )
    for key in ("shared_expert_swiglu_limit", "routed_out_clip", "shared_out_clip"):
        _bound(config, key)
    resolve_mtp_swiglu_limit(config, "shared_expert_swiglu_limit")
    for key, value in (
        ("swiglu_limit", _bound(config, "swiglu_limit")),
        ("mtp_swiglu_limit (resolved)", resolve_mtp_swiglu_limit(config)),
    ):
        if value is not None and float(value) != 10.0:
            raise NotImplementedError(
                f"solar_open3 SGLang serves a routed SwiGLU clamp only at 10 (SGLang's DeepSeek-V4 "
                f"clamp path asserts it); got {key}={value!r}. Use the vLLM adapter."
            )
    # Native configs already implement the cache contract. Evaluating their cache
    # property through hasattr would require a distributed group during discovery.
    if not isinstance(config, SolarOpen3Config):
        config = _patch_config(config)
    _reserve_linear_draft_state_layer(config)
    return config


_PREFILL_BODY_LOGGED: set = set()  # (body, why) pairs already logged, once per process


def _routing_flag(config, name: str, default: bool) -> bool:
    """A tri-state routing flag, read the way the converter writes it.

    Every one of these fields is exported explicitly, so JSON ``null`` means "this arm did not set
    it" and must fall back to the default rather than to ``False``. Reading a null as False is the
    failure this exists to prevent: it would serve a checkpoint that attended its anchors with
    the anchors removed, and the scores would still look reasonable.
    """
    v = getattr(config, name, None)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    raise ValueError(
        f"{name}={v!r} is not a boolean; pass true or false, or leave it unset."
    )


# ── the routed body behind one opaque node ──────────────────────────────────────────────────
#
# A compiling prefill backend (tc_piecewise) traces the model with Dynamo, and the routed body
# cannot be traced: it branches on per-request data pulled to the host (`if num_anchor:`, the
# per-request loop). Registering it as a custom op is what the vLLM adapter does for the same
# reason -- the tracer sees one node and never looks inside.
#
# Two things the op cannot take as arguments, and how each is recovered:
#   * `self`     -- the layer registers itself under its prefix, and the op looks it up by name.
#   * the batch  -- ForwardBatch is not a tensor, and SGLang's ForwardContext holds only the
#                   attention backend. It travels by the two channels `_current_batch` reads.
_ROUTED_LAYERS: dict[str, SolarOpen3Attention] = {}
# ⚠️ The registry key must identify the LAYER INSTANCE, not its module path. A path is not unique
# within a process: speculative decoding builds a target and a draft in the same process and both
# start at `prefix="model"`, so `model.layers.0.self_attn` names a layer in each. Keying on the
# path alone lets the second registration overwrite the first, and the custom op -- which receives
# only this string -- then runs one model's forward through the OTHER model's parameters, with no
# error. Today the draft's single layer is not a routed one, so the collision does not occur; that
# is an accident of the layer pattern, not a property anything enforces.
_ROUTED_SERIAL = itertools.count()

# The fallback channel, used when the engine has no tc_piecewise context of its own (eager, and
# the decode graph paths). Never read it before `get_tc_piecewise_forward_context()` -- see
# `_current_batch`.
_CURRENT_BATCH: list = []


def _current_batch():
    """The ForwardBatch this op is running under.

    ⚠️ The ENGINE's context has to win, and the order here is the whole point.

    `tc_piecewise` torch.compiles the outer `model.forward`, so `SolarOpen3Model.forward` --
    stash and all -- is traced rather than executed. Dynamo defers a mutation of a module-level
    list to AFTER the compiled region runs, so at the moment the op executes the stash is still
    empty and its last entry, if any, belongs to a previous forward. Reading it first is not a
    missing batch but a WRONG one: the op would attend the previous request's cache and return
    plausible tokens.

    `set_tc_piecewise_forward_context` is called outside the compiled region on all three paths
    that use it (capture, replay, eager runner), which is the same channel SGLang's own
    `radix_attention` resolves its batch through. The stash covers the paths that never set it.
    """
    ctx = get_tc_piecewise_forward_context()
    if ctx is not None and ctx.forward_batch is not None:
        return ctx.forward_batch
    if _CURRENT_BATCH:
        return _CURRENT_BATCH[-1]
    raise RuntimeError(
        "solar_open3_routed_attn ran with no forward batch: SGLang's tc_piecewise context is "
        "unset and SolarOpen3Model.forward has not stashed one. The op recovers the batch from "
        "one of those two, so it cannot be called from outside a model forward."
    )


# ⚠️ BOTH decorators are load-bearing, and for DIFFERENT reasons.
#
#   register_custom_op  hides the routed body from the TRACER. Dynamo sees one opaque node and
#                       never meets the host-side `if num_anchor:` branch inside.
#   register_split_op   hides it from the CAPTURE. `tc_piecewise` does not merely trace: it cuts
#                       the compiled function into pieces and records each piece as a CUDA graph
#                       (`cuda_piecewise_backend.py`, `torch.cuda.graph(...)`). The routed body
#                       reads the batch to the host, which is illegal mid-capture -- the op has to
#                       be a CUT POINT so it runs eager between two captured pieces.
#
# Without the split op the capture dies with cudaErrorStreamCaptureUnsupported. This is the same
# pairing SGLang puts on its own `unified_attention_with_output`.
# ⚠️ The name must be passed to BOTH: register_split_op defaults to the PYTHON function's name,
# which is underscore-prefixed here, and a split op that names nothing registered silently splits
# nothing.
@register_custom_op(op_name="solar_open3_routed_attn", mutates_args=["out"])
@register_split_op(op_name="solar_open3_routed_attn")
def _solar_open3_routed_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
    output_gate: Optional[torch.Tensor] = None,
) -> None:
    """Opaque to the tracer; `out` is pre-allocated by the caller and written in place."""
    _ROUTED_LAYERS[layer_name]._routed_attn_impl(
        q, k, v, out, _current_batch(), output_gate=output_gate
    )


# --- Attention-Residuals -------------------------------------------------------------------
#
# The kernels, launchers and fake impls are CARRIED in routed_core.py, extracted from the vLLM
# adapter like the routed bodies. Only the two lines below differ, and they are the one thing that
# cannot travel: vLLM registers these under `torch.ops.vllm.*` with `direct_register_custom_op`,
# so the wrappers that call them by name are rewritten here against `torch.ops.sglang.*`.
#
# `mutates_args` matches the vLLM registration exactly -- the op writes `prefix`, `blocks` and
# `output` in place. Getting that list wrong does not fail loudly: the tracer would be free to
# reorder or elide the writes and the model would serve a different function.
register_custom_op(
    op_name="solar_open3_attn_res",
    mutates_args=["prefix", "blocks", "output"],
    fake_impl=_attn_res_fake,
)(_attn_res_launch)
register_custom_op(
    op_name="solar_open3_cp_attn_res",
    mutates_args=["prefix", "blocks", "output"],
    fake_impl=_cp_attn_res_fake,
)(_cp_attn_res_launch)


def attn_res(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    scale_by_sources=False,
):
    """One fused Attention-Residuals sublayer step. `prefix` and `blocks` are MUTATED."""
    output = prefix.new_empty(prefix.shape)
    torch.ops.sglang.solar_open3_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
        scale_by_sources,
    )
    return output


def cp_attn_res(
    prefix,
    delta,
    blocks,
    score_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    output_norm_eps,
    scale_by_sources=False,
):
    """The `cp` scorer variant of the above. `prefix` and `blocks` are MUTATED."""
    output = prefix.new_empty(prefix.shape)
    torch.ops.sglang.solar_open3_cp_attn_res(
        prefix,
        delta,
        blocks,
        score_weight,
        output_norm_weight,
        output,
        num_blocks,
        block_write_idx,
        output_norm_eps,
        scale_by_sources,
    )
    return output


class SolarOpen3MoE(SolarOpen2MoE):
    """Open3's FP32 routing and runner scaling, with shared Open2 execution."""

    _use_open2_optimizations = False

    def __init__(
        self,
        config: SolarOpen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_idx: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        super().__init__(config, layer_idx, quant_config, prefix, alt_stream)
        self.routed_out_clip = None
        self.shared_out_clip = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if self.routed_out_clip is not None or self.shared_out_clip is not None:
            return self._forward_clipped(
                hidden_states, forward_batch, use_reduce_scatter
            )
        return super().forward(hidden_states, forward_batch, use_reduce_scatter)

    def _forward_clipped(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        """Clip each TP-reduced branch before adding routed and shared outputs."""
        if get_moe_a2a_backend().is_deepep() or use_reduce_scatter:
            raise NotImplementedError(
                "solar_open3 SGLang applies routed_out_clip / shared_out_clip only on the "
                "all-reduce MoE path; serve with --moe-a2a-backend none and without reduce-scatter."
            )
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        # The shared expert first, as in the parent: the routed experts run in place on hidden_states.
        shared = (
            self.shared_experts(hidden_states) if hidden_states.shape[0] > 0 else None
        )
        router_logits, _ = self.gate(hidden_states)
        out = self.experts(hidden_states, self.topk(hidden_states, router_logits))
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        if self.routed_out_clip is not None:
            out = _clip_token_norm(out, self.routed_out_clip)
        if shared is not None:
            if self.tp_size > 1:
                shared = tensor_model_parallel_all_reduce(shared)
            if self.shared_out_clip is not None:
                shared = _clip_token_norm(shared, self.shared_out_clip)
            out = out + shared
        return out.view(num_tokens, hidden_size)


class SolarOpen3KimiDeltaAttention(SolarOpen2KimiDeltaAttention):
    """Open3's precision and reduction policy over the shared Open2 KDA."""

    beta_in_fp32 = True

    def __init__(self, layer_idx, config, quant_config=None, prefix=""):
        # Native/remote Open3 configs may omit Kimi's explicit shape defaults.
        kda_config = copy(config)
        lac = dict(config.linear_attn_config or {})
        lac.setdefault("head_dim", config.head_dim)
        lac.setdefault("num_heads", config.num_attention_heads)
        lac.setdefault("short_conv_kernel_size", 4)
        kda_config.linear_attn_config = lac
        super().__init__(
            kda_config,
            layer_idx,
            quant_config,
            prefix,
            rms_norm_eps=config.rms_norm_eps,
            reduce_results=False,
        )
        self.config = config
        self.allow_neg_eigval = config.kda_allow_neg_eigval
        # FlashSDA's optional fold adapter checks this legacy flag. A single
        # registered conv parameter now owns the backend view; no aliases to check.
        self._conv_alias_checked = True
        self.round_sda_prefill_beta = getattr(
            config, "linear_attn", None
        ) == "sda" and not getattr(config, "sda_beta_fp32", False)
        # ROCm: this layer's prefill / extend / mixed batches run aiter's flash_sda
        # (grouped decay, the checkpoint's erase); the KDA backend calls
        # `self.attn.sda_prefill` instead of its kernel dispatcher. Decode and verify
        # stay on SGLang's Triton recurrent kernel, which activates the raw beta logit
        # itself with no cap: for exact erase the logit is clamped where it is
        # produced so every path erases with at most 2 - 2^-6, as training does
        # (FlashSDA's sglang_beta_cap on CUDA). Prefill output is unchanged by the
        # clamp: its kernel caps beta itself.
        self._sda_beta_logit_cap = None
        if is_hip() and getattr(config, "linear_attn", None) == "sda":
            from sglang.srt.layers.attention.solar_open3.amd import (
                SdaRocmPrefill,
                beta_logit_cap,
            )

            groups = getattr(config, "sda_decay_groups", None)
            if not groups:
                raise ValueError(
                    "solar_open3 SDA (ROCm): config.json has no `sda_decay_groups`; the "
                    "ROCm kernel reads one gate lane per decay group and needs the count"
                )
            self.attn.sda_prefill = SdaRocmPrefill(
                groups=int(groups), exact_erase=bool(config.exact_erase)
            )
            if config.exact_erase:
                self._sda_beta_logit_cap = beta_logit_cap(float(self.attn.beta_scale))

    def _cap_beta_logit(self, states):
        if self._sda_beta_logit_cap is None:
            return states
        qkv, beta, forget_gate, norm_gate = states
        return qkv, beta.float().clamp(max=self._sda_beta_logit_cap), forget_gate, norm_gate

    def forward_qkvbfg(self, hidden_states):
        return self._cap_beta_logit(super().forward_qkvbfg(hidden_states))

    def forward_qkvbfg_fused(self, hidden_states):
        return self._cap_beta_logit(super().forward_qkvbfg_fused(hidden_states))

    def prepare_fused_decode(self):
        # The ROCm Open2 kernel does not implement Open3's SDA erase/rounding
        # policy. Open3 keeps its installed KDA/FlashSDA recurrent backend.
        self.attn._solar_fused_decode_args = None

    def forward(self, hidden_states, positions, forward_batch, zero_allocator=None):
        return super().forward(hidden_states, positions, forward_batch)


class SolarOpen3Attention(SolarOpen2Attention):
    """GQA attention, routed (LSSA) on the layers the checkpoint says are routed.

    A routed layer reads, under ONE softmax over the union:
      * the subsampled anchor stream (one key per ``kv_subsample_stride`` positions),
      * the top-k route blocks at original resolution, ranked by the anchor logits,
      * its own block's raw span.
    ``kv_route_dedup_anchor`` drops, from the anchor half, the anchors whose unit was selected --
    at ``kv_subsample_pool='last'`` that anchor IS one of the unit's raw tokens.
    """

    def __init__(
        self,
        layer_idx: int,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=prefix,
            reduce_results=False,
        )
        # Keep the strided QKV views used by Open3's existing attention path.
        self.zero_copy_qkv = True
        self.fuse_routed_output_gate = True
        self._init_routing(config, layer_idx, prefix)
        # The path stays in the name so a traceback still says which layer this is; the serial is
        # what makes it unique.
        self.sparse_layer_name = f"{prefix}#{next(_ROUTED_SERIAL)}"
        if self.kv_route_layer:
            _ROUTED_LAYERS[self.sparse_layer_name] = self

    # ------------------------------------------------------------------ routing configuration
    def _init_routing(self, config, layer_idx: int, prefix: str) -> None:
        sub_layers = getattr(config, "kv_subsample_layers", None) or []
        route_layers = getattr(config, "kv_route_layers", None) or []
        # `is None`, not `or`: stride 0 is not "unset", and reading it as 1 would serve a layer
        # with no anchor stream at all. The vLLM adapter refuses the same value.
        _st = getattr(config, "kv_subsample_stride", None)
        self.kv_stride = 1 if _st is None else int(_st)
        self.kv_subsample_layer = layer_idx in sub_layers
        self.kv_route_layer = layer_idx in route_layers

        # The pre-rename `kv_route_sink_units` is refused, not skipped, on EVERY GQA layer (before
        # the non-routed early return below, the way the vLLM adapter's `_config_sink_tokens`
        # runs per layer): this adapter reads config.json straight from the engine's loader, so
        # the converter's refusal of that key never runs here, and a checkpoint exported under
        # the old spelling would otherwise serve WITHOUT the sink it trained -- silently, at -1.1p.
        _old = getattr(config, "kv_route_sink_units", None)
        if _old is not None:
            raise ValueError(
                f"{prefix}: config.json carries kv_route_sink_units={_old!r}, the pre-rename "
                "serving key; the sink is read from kv_route_sink_tokens (the span the trainer "
                "trained with). Re-export from the training config, or rename the key to the "
                "token span (units x kv_route_block)."
            )

        if self.kv_subsample_layer and self.kv_stride <= 1:
            raise ValueError(
                f"{prefix}: layer {layer_idx} is in config.kv_subsample_layers but "
                f"kv_subsample_stride is {self.kv_stride}; the layer would silently serve as "
                "plain full-causal attention, which is a different model from the trained one."
            )
        if self.kv_route_layer and not self.kv_subsample_layer:
            raise ValueError(
                f"{prefix}: layer {layer_idx} is in config.kv_route_layers but not in "
                "config.kv_subsample_layers. The router ranks units by the ANCHOR logits, so a "
                "routed layer without an anchor stream has nothing to rank with."
            )
        if not self.kv_route_layer:
            # A subsampled-but-unrouted layer is a different model from a full-causal one, and
            # this adapter does not implement it. Refuse rather than serve dense attention under
            # its name.
            if self.kv_subsample_layer:
                raise NotImplementedError(
                    f"{prefix}: layer {layer_idx} subsamples but does not route "
                    "(kv_subsample_layers without kv_route_layers). This adapter implements the "
                    "routed union only; serving it as full-causal attention would read every key "
                    "the trained model never saw, with no crash and no log line."
                )
            return

        # `is None`, not `or`: 0 is a value the config CHOSE, not an absent one, and the two must
        # not resolve the same way. How LARGE it has to be is decided below, against the units a
        # query block actually spans -- see the own-unit rule after the geometry is read.
        _tk = getattr(config, "kv_route_topk", None)
        self.kv_route_topk = 0 if _tk is None else int(_tk)
        if self.kv_route_topk < 0:
            raise ValueError(
                f"{prefix}: layer {layer_idx} is routed with kv_route_topk="
                f"{self.kv_route_topk}; the budget cannot be negative."
            )
        self.kv_route_block = _route_geometry(config, "kv_route_block", 512)
        self.kv_route_q_block = _route_geometry(config, "kv_route_q_block", 256)
        # The budget is the TOTAL: a query block's own units are forced into the ranking by
        # scoring +inf and then SPEND top-k slots. With fewer slots than own units the block keeps
        # some of its own and drops the rest -- and this body does not drop the same ones the
        # trainer did, so the served union differs from the trained one.
        #
        # ⚠️ Not `ceil(q_block / rb)`: a block starts at `b * q_block`, so when `rb` does not
        # divide `q_block` some blocks straddle one unit more. The offsets a block can start at
        # are the multiples of `gcd(q_block, rb)`, so take the worst. Ported verbatim in intent
        # from the vLLM adapter, which is where this rule was derived.
        if self.kv_route_topk:
            _g = math.gcd(self.kv_route_q_block, self.kv_route_block)
            _own = max(
                (_r + self.kv_route_q_block - 1) // self.kv_route_block
                - _r // self.kv_route_block
                + 1
                for _r in range(0, self.kv_route_block, _g)
            )
            if self.kv_route_topk < _own:
                raise ValueError(
                    f"{prefix}: kv_route_topk={self.kv_route_topk} is smaller than the {_own} "
                    f"routing units a query block of kv_route_q_block={self.kv_route_q_block} "
                    f"spans at kv_route_block={self.kv_route_block}. Those units are forced into "
                    "the ranking and spend top-k slots, so a smaller budget drops some of the "
                    "block's own units -- and this body does not drop the same ones the trainer "
                    f"did, so the served union differs from the trained one. Use kv_route_topk "
                    f">= {_own}, or 0 for anchors plus own units."
                )
        # Sink forcing (config.kv_route_sink_tokens -- the SAME key the trainer trains with,
        # passed through the converter verbatim): the units covering the request's first span
        # are forced into the selection the way the own units are -- +inf, then ranked in the
        # same budget. The trainer forces every DOCUMENT's first span; a serving request is one
        # document starting at position 0, so the leading ceil(tokens / kv_route_block) units
        # are the same set with no straddle. Absent/null = 0 = the pre-sink ranking bit-for-bit;
        # a checkpoint trained with the sink and served without it (or the reverse) is a
        # different function, so the key rides the checkpoint.
        _sk = getattr(config, "kv_route_sink_tokens", None)
        if _sk is not None and (
            isinstance(_sk, bool) or not isinstance(_sk, int) or _sk < 0
        ):
            raise ValueError(
                f"{prefix}: kv_route_sink_tokens={_sk!r} is not a non-negative integer. It is "
                "the token span of the request's head forced into the selection, so a value the "
                "ranking cannot spend is refused rather than read as a flag."
            )
        self.kv_route_sink_units = -(-(_sk or 0) // self.kv_route_block)
        if self.kv_route_sink_units:
            if not self.kv_route_topk:
                raise ValueError(
                    f"{prefix}: kv_route_sink_tokens={_sk} with "
                    "kv_route_topk=0: no ranking runs at topk 0 (the selection is the own units "
                    "in closed form), so the sink has no top-k to be forced into."
                )
            if self.kv_route_topk < _own + self.kv_route_sink_units:
                raise ValueError(
                    f"{prefix}: kv_route_topk={self.kv_route_topk} cannot hold the forced "
                    f"units: {_own} own + {self.kv_route_sink_units} sink "
                    f"(kv_route_sink_tokens={_sk} over kv_route_block={self.kv_route_block}) = "
                    f"{_own + self.kv_route_sink_units}. Both are forced by scoring +inf and "
                    "then ranked, so a smaller budget drops forced units "
                    "nondeterministically. Raise kv_route_topk or lower kv_route_sink_tokens."
                )
        self.kv_route_score_block = getattr(config, "kv_route_score_block", None)
        # ⚠️ Resolve and validate HERE, the way the vLLM adapter does. Storing the raw value left
        # the two call sites disagreeing: decode resolved it (`or "mean"`) and prefill -- the one
        # that actually uses it -- passed `None` through, so a checkpoint with
        # `kv_route_score_block` set and `kv_route_reduce` null raised inside the kernel launcher
        # on its first prefill instead of at startup. `None` is the "not set" sentinel; anything
        # outside the three reductions is a config error, not a default.
        _rd_raw = getattr(config, "kv_route_reduce", None)
        _rd = "mean" if _rd_raw is None else _rd_raw
        if _rd not in ("mean", "amax", "smax"):
            raise ValueError(
                f"{prefix}: kv_route_reduce={getattr(config, 'kv_route_reduce', None)!r} is not "
                "one of mean|amax|smax (None means 'mean')."
            )
        self.kv_route_reduce = _rd
        # Same reasoning as `kv_route_reduce` above, for the two knobs the routed core checks
        # only at its kernel launch. `head_reduce` is compared against "smax" and nothing else, so
        # a typo runs as `amax` and answers plausibly; `qhead_block` raises on the first prefill,
        # after the weights are loaded. Both belong at startup, where the operator can read them.
        _hr_raw = getattr(config, "kv_route_head_reduce", None)
        _hr = "amax" if _hr_raw is None else _hr_raw
        if _hr not in ("amax", "smax"):
            raise ValueError(
                f"{prefix}: kv_route_head_reduce={getattr(config, 'kv_route_head_reduce', None)!r} "
                "is not one of amax|smax (None means 'amax'). Anything else would run as 'amax' "
                "without a word."
            )
        self.kv_route_head_reduce = _hr
        # Per-row checkpoints rank units with each token's own query. The
        # per-head election uses smax/1; the group mean leaves both knobs unset.
        self.kv_route_per_row = _routing_flag(config, "kv_route_per_row", False)
        _qb_raw = getattr(config, "kv_route_qhead_block", None)
        _per_head = _qb_raw is not None and int(_qb_raw) == 1 and _rd_raw == "smax"
        if (
            self.kv_route_per_row
            and self.kv_route_layer
            and (
                self.kv_route_score_block is not None
                or not (_per_head or (_rd_raw is None and _qb_raw is None))
            )
        ):
            raise ValueError(
                f"{prefix}: kv_route_per_row ranks every row with its own query; kv_route_score_block (sub-query "
                "pooling) does not apply, and kv_route_reduce / kv_route_qhead_block are either unset (the GQA-group "
                "mean) or 'smax' / 1 (the per-head election) -- the trainer refuses the same config."
            )
        # The THREE CROSS rules the vLLM adapter checks at boot, and for the same reason: each one
        # is a router the checkpoint was never trained with, served without a word.
        if _hr == "smax" and _rd != "smax":
            raise ValueError(
                f"{prefix}: kv_route_head_reduce='smax' needs kv_route_reduce='smax', got "
                f"{_rd!r}; the head collapse maxes what that reduce normalises, so without it "
                "this is 'amax' on raw scores under a different name."
            )
        if _hr == "smax" and self.kv_route_score_block is None:
            raise ValueError(
                f"{prefix}: kv_route_head_reduce='smax' needs kv_route_score_block set: the "
                "pooled scoring branch folds the KV heads with amax and never reads this knob "
                "(`routed_core.py`, the `head_shared` fold), so the layer would rank on raw "
                "logits under the smax name. Set kv_route_score_block to the sub-query span the "
                "checkpoint was trained with, or drop kv_route_head_reduce."
            )
        _qb = getattr(config, "kv_route_qhead_block", None)
        self.kv_route_qhead_block = None  # set once here; 1 below when the knob is on
        if _qb is not None:
            if int(_qb) != 1:
                raise ValueError(
                    f"{prefix}: kv_route_qhead_block={_qb!r} is not supported; only 1 (every "
                    "query head votes) or None (the GQA group is averaged) are implemented."
                )
            # Stored as the int the check judged, not the raw value, and BEFORE the cross rules
            # below (as the vLLM adapter does): a config that carries "1" would otherwise pass
            # `int(_qb) != 1` and reach the kernels as a string, and a test that reads the
            # stored value past a later refusal must find it there.
            self.kv_route_qhead_block = 1
            if _rd != "smax":
                raise ValueError(
                    f"{prefix}: kv_route_qhead_block=1 with kv_route_reduce={_rd!r} does not do "
                    "what the name suggests: the per-head votes are combined by that reduce, and "
                    "only 'smax' normalises each vote before the max. With 'amax' a head whose "
                    "logits merely run large wins every unit; with 'mean' the split is undone."
                )
            if self.kv_route_score_block is None and not self.kv_route_per_row:
                raise ValueError(
                    f"{prefix}: kv_route_qhead_block=1 needs kv_route_score_block set: the pooled "
                    "scoring branch averages the GQA group and never reads this knob, so the "
                    "prefill would elect pooled while the decode elects per head. Set "
                    "kv_route_score_block=1 (the per-head recipe) or drop kv_route_qhead_block."
                )
        self.kv_route_head_shared = _routing_flag(config, "kv_route_head_shared", True)
        if not self.kv_route_head_shared and (_qb is not None or _hr == "smax"):
            raise ValueError(
                f"{prefix}: kv_route_head_shared=False cannot be combined with "
                f"kv_route_qhead_block={_qb!r} / kv_route_head_reduce={_hr!r}: both collapse the "
                "KV heads into one ranking, which is what per-head selection removes."
            )
        self.kv_anchor_in_attention = _routing_flag(
            config, "kv_anchor_in_attention", True
        )
        self.kv_route_dedup_anchor = _routing_flag(
            config, "kv_route_dedup_anchor", False
        )
        # Which body serves the per-head prefill (same election either way): a tri-state
        # DECLARATION -- null/absent lets the facts decide (and the log below says which), true
        # requires the fused gluon body or refuses to build, false serves the three-piece body.
        # An implementation choice, so the converter does not export it. Resolved below, before
        # anything that needs a live server, so the CPU startup guards reach the decision.
        self.kv_route_fused_prefill = _routing_tristate(
            getattr(config, "kv_route_fused_prefill", None), "kv_route_fused_prefill"
        )
        self.kv_route_prefill_body, self.kv_route_prefill_body_why = None, ""
        self.kv_route_prefill_scratch_bytes = 0

        _pool_raw = getattr(config, "kv_subsample_pool", None)
        pool = "last" if _pool_raw is None else _pool_raw
        # ⚠️ Checked against what the CHECKPOINT carries, because the failure is silent: an arm
        # trained without the pair attends its selected tokens twice here and still scores
        # plausibly. There is no switch to open it -- reproducing such an arm means serving it
        # with a build from before the rule, not opening the rule.
        legacy = []
        if not self.kv_route_dedup_anchor:
            legacy.append("kv_route_dedup_anchor=False")
        if pool != "last":
            legacy.append(f"kv_subsample_pool={pool!r}")
        if legacy:
            raise ValueError(
                f"{prefix}: routed arms require kv_route_dedup_anchor=True and "
                f"kv_subsample_pool='last'; this checkpoint has {', '.join(legacy)}. Without the "
                "pair an anchor enters the union twice -- once from the anchor half and once "
                "from the selected unit that holds it -- and those keys carry double weight "
                "through the softmax (0.19-0.45 relative on the prefill shapes in "
                "tests/model/solar_open3/test_boundary_paged_kv_gpu.py). Retrain the arm, or "
                "serve it with a build from before the rule."
            )
        if self.kv_route_block % self.kv_stride:
            raise ValueError(
                f"{prefix}: kv_route_dedup_anchor needs kv_route_block ({self.kv_route_block}) "
                f"divisible by kv_subsample_stride ({self.kv_stride}): one anchor page must hold "
                "exactly one routing unit's anchors for the page table to express the drop."
            )
        # ⚠️ NOT refused when `kv_route_q_block % kv_route_block`. That relation picks the FUSED
        # own-and-routed launch; it is not a correctness condition. The vLLM adapter says so where
        # it dispatches -- "`2 * rb >= q_block` is a DISPATCH choice ... the boundary pass below
        # now sizes its own window from `q_block`, so a wider block is served either way" -- and
        # refusing here turned a served configuration into a refused one on this engine alone.
        if self.kv_route_q_block % self.kv_route_block:
            logger.info(
                "[solar_open3] %s: kv_route_q_block=%d is not a multiple of kv_route_block=%d, so "
                "the routed prefill takes the eager boundary pass instead of the fused one. Same "
                "keys, slower launch.",
                prefix,
                self.kv_route_q_block,
                self.kv_route_block,
            )

        # Router and union variants this adapter does not implement. Compared against the DEFAULT
        # rather than against presence, because the converter exports every one of them.
        for field, default in (
            ("kv_route_anchor_weight", None),
            ("kv_route_selected_bias", 0.0),
            ("kv_route_selected_beta", None),
            ("kv_route_own_window", 0),
            ("kv_route_key_dim", 0),
            ("kv_route_anchor_topk", 0),
            ("kv_route_anchor_topk_decode", 0),
            ("kv_route_anchor_tile", None),
        ):
            value = getattr(config, field, default)
            if value is None:
                value = default
            if value != default:
                raise ValueError(
                    f"{prefix}: {field}={value!r} is not implemented by this adapter (default "
                    f"{default!r}). It changes which keys a query reads or how their scores "
                    "combine, so the served model would differ from the trained one with no "
                    "other symptom."
                )
        # The prefill BODY, fixed here for the process (the vLLM adapter's rule, carried:
        # `_resolve_prefill_body`). Every TP rank derives the same answer from the same config.
        self.kv_route_prefill_body, self.kv_route_prefill_body_why = (
            _resolve_prefill_body(
                declared=self.kv_route_fused_prefill,
                # Before the routing budget, every begun unit is selected.
                score_block=None
                if self.kv_route_per_row
                else self.kv_route_score_block,
                qhead_block=None
                if self.kv_route_per_row
                else self.kv_route_qhead_block,
                reduce="mean" if self.kv_route_per_row else self.kv_route_reduce,
                topk=self.kv_route_topk,
                head_shared=self.kv_route_head_shared,
                dedup_anchor=self.kv_route_dedup_anchor,
                stride=self.kv_stride,
                rb=self.kv_route_block,
                q_block=self.kv_route_q_block,
                # Set by the parent constructor before this runs; the startup-guard tests drive
                # `_init_routing` on a bare object, so the config's value stands in there.
                head_dim=getattr(self, "head_dim", None)
                or getattr(config, "head_dim", None),
                sm_major=_current_sm_major(),
            )
        )
        _bkey = (self.kv_route_prefill_body, self.kv_route_prefill_body_why)
        if _bkey not in _PREFILL_BODY_LOGGED:
            _PREFILL_BODY_LOGGED.add(_bkey)
            logger.info(
                "[solar_open3] %s: routed prefill body=%s (%s) kv_route_fused_prefill=%s",
                prefix,
                self.kv_route_prefill_body,
                self.kv_route_prefill_body_why,
                {None: "auto", True: "true", False: "false"}[
                    self.kv_route_fused_prefill
                ],
            )

        # The ceiling the decode grid is sized from: what the ENGINE was launched with, so it is
        # constant for the process and a CUDA graph can capture it. Serving a 1M engine at 256K
        # therefore pays fourfold on everything scaled by the unit count.
        _sa = runtime_options()
        ctx = getattr(_sa, "context_length", None)
        self.route_pad_positions = int(
            ctx or getattr(config, "max_position_embeddings", 0) or 0
        )
        # The fused gluon prefill's per-chunk scratch, bounded at this engine's ceiling and
        # chunk size. SGLang has no memory-profile run to reserve it in, so it is printed: leave
        # it under `--mem-fraction-static` (the vLLM adapter reserves the same bound in its
        # profile run).
        self.kv_route_prefill_scratch_bytes = 0
        if self.kv_route_prefill_body == "gluon":
            _cp = int(getattr(_sa, "chunked_prefill_size", 0) or 0)
            _chunk = _cp if _cp > 0 else self.route_pad_positions
            self.kv_route_prefill_scratch_bytes = (
                _gluon_prefill_mod().prefill_scratch_bytes(
                    hq=self.num_heads,
                    hkv=self.num_kv_heads,
                    d=self.head_dim,
                    stride=self.kv_stride,
                    rb=self.kv_route_block,
                    q_block=self.kv_route_q_block,
                    topk=self.kv_route_topk,
                    max_positions=self.route_pad_positions,
                    chunk_tokens=_chunk,
                )
            )
            if ("gluon-scratch",) not in _PREFILL_BODY_LOGGED:
                _PREFILL_BODY_LOGGED.add(("gluon-scratch",))
                logger.info(
                    "[solar_open3] %s: the fused gluon prefill allocates up to %.2f GiB "
                    "per chunk (bound at %d positions, chunk %d) outside the KV pool -- "
                    "leave it under --mem-fraction-static",
                    prefix,
                    self.kv_route_prefill_scratch_bytes / 2**30,
                    self.route_pad_positions,
                    _chunk,
                )
        # ⚠️ Speculative decoding writes PAST the context limit on purpose: a verify step appends
        # `speculative_num_draft_tokens` positions to a request that may already sit at the limit.
        # SGLang widens its own position-indexed table by exactly that amount
        # (`mem_cache/kv_cache_configurator.py`: `extra_max_context_len += ...num_draft_tokens`),
        # so the routed grid follows the same rule. Sizing it to the context alone would leave the
        # last rows of such a request with no route unit of their own -- silently, since the grid
        # is a captured constant and nothing downstream re-derives it.
        _refuse_speculative_tree(_sa, prefix)
        # ⚠️ Round UP to a whole route block. The only consumer floors this
        # (`routed_core`: `n_units = pad_positions // rb`), so adding the draft window alone is
        # absorbed whenever `(ctx % rb) + draft < rb` -- which is every configuration here
        # (ctx 4096, rb 128, draft 2: 32 units either way). The extra positions have to buy a
        # whole unit or they buy nothing.
        _draft = getattr(_sa, "speculative_num_draft_tokens", None)
        if _draft:
            _rb = int(self.kv_route_block)
            _need = self.route_pad_positions + int(_draft)
            self.route_pad_positions = -(-_need // _rb) * _rb
        if self.route_pad_positions <= 0:
            raise ValueError(
                f"{prefix}: cannot size the routed decode grid -- neither --context-length nor "
                "config.max_position_embeddings gives a position ceiling."
            )

        # The cache PAGE is what the routed bodies address by, and it is a server argument, not
        # a model property. Two constraints, both from how the anchor grid is formed:
        #   * an anchor sits at offset ``stride-1`` of every stride block, so the grid is a slice
        #     of the cache only while the page is a multiple of the stride;
        #   * a routing unit is named by page id, so ``gcd(page, kv_route_block)`` must also be a
        #     multiple of the stride (``subpage_cache`` re-addresses when the page does not divide
        #     the unit).
        # Checked here, at startup, because the alternative is a ValueError on the first request
        # after the checkpoint has loaded.
        self.page_size = int(runtime_options().page_size or 1)
        _g = math.gcd(self.page_size, self.kv_route_block)
        if self.page_size % self.kv_stride or _g % self.kv_stride:
            raise ValueError(
                f"{prefix}: --page-size {self.page_size} cannot serve this arm. The routed "
                f"anchors sit every kv_subsample_stride={self.kv_stride} positions, so the page "
                f"the routing reads -- the page itself, or gcd(page, kv_route_block="
                f"{self.kv_route_block})={_g} when the page does not divide the unit -- must be a "
                f"multiple of {self.kv_stride}. Launch with --page-size "
                f"{self.kv_stride * (self.kv_route_block // self.kv_stride)} or another multiple "
                f"of {self.kv_stride} that divides or is divided by {self.kv_route_block}."
            )

        # PREFILL graph backends: tc_piecewise works, full and breakable do not. All three were
        # measured.
        #
        #   tc_piecewise  ✅ supported. It traces with Dynamo, and the routed body is behind a
        #                 custom op (`torch.ops.sglang.solar_open3_routed_attn`), so the tracer
        #                 sees one opaque node and never meets the host branches inside. This is
        #                 the arrangement the vLLM adapter uses, for the same reason.
        #   full          ❌ cudaErrorStreamCaptureUnsupported. It captures one whole-forward
        #                 graph, and the routed body walks the batch request by request on the
        #                 host; CUDA refuses a device-to-host read mid-capture. A custom op does
        #                 not help -- it hides the work from the TRACER, not from the CAPTURE.
        #   breakable     ❌ the same violation; its break points are not where those reads are.
        #
        # DECODE graphs are unaffected and stay on under every mode.
        #
        # SOLAR_OPEN3_ALLOW_PREFILL_GRAPH=1 lifts the refusal, for probing the two that fail.
        _cg = getattr(runtime_options(), "cuda_graph_config", None)
        _pre = getattr(getattr(_cg, "prefill", None), "backend", "disabled")
        if (
            _pre not in (None, "disabled", "tc_piecewise")
            and os.environ.get("SOLAR_OPEN3_ALLOW_PREFILL_GRAPH", "") != "1"
        ):
            raise ValueError(
                f"{prefix}: prefill CUDA graph backend {_pre!r} captures one whole-forward graph, "
                "and the routed prefill walks the batch on the host inside it -- CUDA refuses the "
                "read mid-capture. Use --cuda-graph-backend-prefill=tc_piecewise (supported: the "
                "routed body is a custom op, so Dynamo never traces into it) or =disabled."
            )

        # TP: the router's ballots are folded across ranks before the top-k, so every rank picks
        # the SAME units and the union is one model rather than one per rank.
        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        if self.total_num_kv_heads >= tp_size:
            self.kv_head_start = tp_rank * self.num_kv_heads
        else:
            self.kv_head_start = tp_rank // (tp_size // self.total_num_kv_heads)
        self.qhead_start = tp_rank * self.num_heads
        group = self.total_num_heads // self.total_num_kv_heads
        self._route_writes_score = self.qhead_start % group == 0
        if tp_size > 1:
            self._route_all_reduce = tensor_model_parallel_all_reduce

            def _max_all_reduce(x: torch.Tensor) -> torch.Tensor:
                torch.distributed.all_reduce(
                    x,
                    op=torch.distributed.ReduceOp.MAX,
                    group=get_parallel().tp_group.device_group,
                )
                return x

            self._route_max_all_reduce = _max_all_reduce
        else:
            self._route_all_reduce = None
            self._route_max_all_reduce = None

    # ------------------------------------------------------------------------------- forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        if not self.kv_route_layer:
            return super().forward(hidden_states, positions, forward_batch, **kwargs)

        qkv, gate = self.project_qkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # ⚠️ zeros, not empty: the early returns inside write nothing and the caller must still
        # see a defined tensor. It is also what `mutates_args=["out"]` declares.
        attn_output = torch.zeros(
            q.shape[0], self.num_heads * self.head_dim, dtype=q.dtype, device=q.device
        )
        fuse_gate = (
            self.fuse_routed_output_gate
            and self.use_gqa_gate
            and q.is_cuda
            and torch.version.hip is None
            and q.dtype == torch.bfloat16
        )
        if fuse_gate and not self.fused_qkvg:
            gate, _ = self.g_proj(hidden_states)
        if fuse_gate:
            torch.ops.sglang.solar_open3_routed_attn(
                q, k, v, attn_output, self.sparse_layer_name, gate
            )
        else:
            torch.ops.sglang.solar_open3_routed_attn(
                q, k, v, attn_output, self.sparse_layer_name
            )
        if self.use_gqa_gate and not fuse_gate:
            if not self.fused_qkvg:
                gate, _ = self.g_proj(hidden_states)
            attn_output = _apply_gqa_output_gate(attn_output, gate)
        output, _ = self.o_proj(attn_output)
        return output

    def _routed_attn_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        forward_batch: ForwardBatch,
        output_gate: Optional[torch.Tensor] = None,
    ) -> None:
        """The routed union, written into the caller's `out`. Reached only through the custom op."""
        n = q.shape[0]
        if n == 0:
            return

        # The pools live on the ModelRunner, not on the batch; the forward context is how a
        # model reaches them (``deepseek_v4`` does the same, for the same reason -- its indexer
        # also reads the cache outside the attention backend).
        pool = get_token_to_kv_pool()
        kv_shape = (-1, self.num_kv_heads, self.head_dim)
        pool.set_kv_buffer(
            self.attn, forward_batch.out_cache_loc, k.view(kv_shape), v.view(kv_shape)
        )

        page = self.page_size
        key_cache = pool.get_key_buffer(self.attn.layer_id).view(
            -1, page, self.num_kv_heads, self.head_dim
        )
        value_cache = pool.get_value_buffer(self.attn.layer_id).view(
            -1, page, self.num_kv_heads, self.head_dim
        )

        stride, q_block, route_block = (
            self.kv_stride,
            self.kv_route_q_block,
            self.kv_route_block,
        )
        # Page ids, not token slot ids. The anchor grid is a SLICE of the cache -- an anchor sits
        # at offset ``stride-1`` in every stride block -- so the routed bodies address the cache
        # by page and need the page to be a multiple of the stride. ``req_to_token`` is per
        # position, so the page table is its every-``page``-th column divided by the page.
        req_to_token = get_req_to_token_pool().req_to_token
        if not torch.compiler.is_compiling():
            # Diagnostics, not model math. Calling them from a traced region is an error under a
            # compiling prefill backend even when they are marked disabled, and the branch is
            # resolved at trace time so nothing of it reaches the graph.
            self._check_kv_roundtrip(
                forward_batch, key_cache, k.view(kv_shape), req_to_token
            )

        # DECODE and TARGET_VERIFY both take the shape-static path. Verify is a decode with
        # `num_tokens_per_req` query rows instead of one: the drafted tokens are already in the KV
        # cache when this runs, and each row's own position excludes the drafts that follow it, so
        # the causal window is right without any extra masking.
        #
        # ⭐ This is what makes speculative decoding capturable. The extend path below reads
        # per-request positions to the HOST to size its grid, which a CUDA graph forbids -- and
        # SGLang captures verify as a DECODE graph (`decode_cuda_graph_runner.py`: speculative
        # decoding forces `capture_forward_mode = TARGET_VERIFY`), where `register_split_op` has no
        # effect because the `full` backend has no cut points. Routing verify here instead keeps
        # every host read out of the captured region.
        verify = forward_batch.forward_mode.is_target_verify()
        if forward_batch.forward_mode.is_decode() or verify:
            # One query row per request on decode; `num_tokens_per_req` rows on verify, laid out
            # request-major. Gather current page IDs directly into one row per query, including
            # verify's repeated rows, without materializing an intermediate token-slot table.
            per_req = 1
            if verify:
                per_req = int(
                    getattr(forward_batch.spec_info, "num_tokens_per_req", 0) or 0
                )
                batch_size = forward_batch.req_pool_indices.shape[0]
                if per_req <= 0 or batch_size * per_req != n:
                    raise RuntimeError(
                        f"{self.attn.layer_id}: target verify carries {n} query rows over "
                        f"{batch_size} requests, which is not a whole number of rows per request "
                        f"(spec_info.num_tokens_per_req={per_req!r}). The router places rows by "
                        "that count, so it cannot lay this batch out."
                    )
                # Row positions, on the device. `positions` is what the speculative worker filled
                # for these rows; `seq_lens - 1` is the decode-only shortcut and would give every
                # row of a request the same position.
                pos_t = forward_batch.positions[:n]
            else:
                pos_t = forward_batch.seq_lens - 1
            bt = _step_page_table(forward_batch, req_to_token, page, per_req)
            if not torch.compiler.is_compiling():
                self._check_grid_fits(bt, page, key_cache.shape[0])
            decoded = routed_decode_batched(
                q.view(n, self.num_heads, self.head_dim),
                None,  # positions stay on the device; no host readback
                bt,
                key_cache,
                value_cache,
                page,
                self.num_kv_heads,
                self.scaling,
                stride,
                q_block,
                route_block,
                self.kv_route_topk,
                anchor_in_union=self.kv_anchor_in_attention,
                dedup_anchor=self.kv_route_dedup_anchor,
                pos_t=pos_t,
                pad_positions=self.route_pad_positions,
                score_block=self.kv_route_score_block,
                reduce=self.kv_route_reduce,
                head_reduce=self.kv_route_head_reduce,
                qhead_block=self.kv_route_qhead_block,
                head_shared=self.kv_route_head_shared,
                total_kv_heads=self.total_num_kv_heads,
                kv_head_start=self.kv_head_start,
                group_size=self.total_num_heads // self.total_num_kv_heads,
                all_reduce=self._route_all_reduce,
                max_reduce=self._route_max_all_reduce,
                plan_cache=_step_plan(forward_batch, n),
                sink_units=self.kv_route_sink_units,
                per_row=self.kv_route_per_row,
                # SGLang allocates K and V as two tensors per layer, so the fused decode is
                # handed the pair rather than one interleaved cache. See routed_core.py's
                # ADAPTATIONS note.
                kv_cache=(key_cache, value_cache),
                output_gate=output_gate,
                output=out if output_gate is not None else None,
            )
            if output_gate is None:
                out[:] = decoded
            return

        # Extend (prefill, possibly chunked). Positions inside a request ascend and are
        # contiguous in the chunk, which is what lets one call cover every query block of it.
        ext_lens = forward_batch.extend_seq_lens_cpu
        if ext_lens is None:
            raise RuntimeError(
                f"{self.attn.layer_id}: routed prefill needs extend_seq_lens_cpu on the forward "
                "batch to place each request's rows; this forward mode "
                f"({forward_batch.forward_mode}) supplied none, so the router cannot tell which "
                "rows belong to which request."
            )
        plan = _step_plan(forward_batch, n)
        # The chunk's first POSITION comes from the batch, not from arithmetic on the lengths.
        # `seq_len - extend_len` is the same number for every request the scheduler builds, but it
        # is a restatement of an invariant rather than the value itself -- and a batch that does
        # not satisfy it (a startup warmup carries 16k rows all at position 0) would be attended
        # against the wrong causal window with nothing to say so. Read once per step and shared by
        # every routed layer, so it costs one device-to-host read for the whole forward.
        p0s = _step_chunk_starts(forward_batch, ext_lens)
        start = 0
        for i, ext in enumerate(ext_lens):
            ext = int(ext)
            if ext == 0:
                continue
            s, e = start, start + ext
            start = e
            p0 = p0s[i]  # first position of this chunk, as the batch says
            # Positions 0 .. total-1 exist after this chunk. Taken from the chunk's own extent, not
            # from `seq_lens`: on an ordinary extend the two agree by construction (`seq_lens` is
            # prefix + extend and `p0` IS the prefix), but under TARGET_VERIFY `seq_lens` is the
            # accepted length and does not count the draft tokens this batch is verifying -- a
            # 5-token sequence reported 3, which left the anchor pool empty for a chunk that
            # reaches position 4 and tripped the A=0 guard. The positions are authoritative.
            total = p0 + ext
            num_anchor = total // stride
            if num_anchor:
                # Anchor POSITIONS are always the window's last slot; only the CONTENT differs
                # between pool="last" and pool="mean".
                apos = torch.arange(
                    stride - 1, num_anchor * stride, stride, device=q.device
                )
            else:
                # Shorter than one stride: no anchors. It still attends, because a query's own
                # route block is always selected.
                apos = torch.zeros(0, dtype=torch.long, device=q.device)
            n_pages = (total + page - 1) // page
            bt_row = (
                req_to_token[forward_batch.req_pool_indices[i], ::page][:n_pages]
                // page
            )

            if self.kv_route_per_row and total - 1 >= self.kv_route_topk * route_block:
                # config.kv_route_per_row: the chunk routes per row with training's router and attends through the
                # FA4 anchor half + the selected half over the paged cache. A chunk wholly before the budget selects
                # every begun unit under either rule and keeps the pooled body below.
                from sglang.kernels.ops.attention.solar_open3.row_routed import (
                    row_routed_prefill,
                )

                if ("prefill-rowroute",) not in _PREFILL_BODY_LOGGED:
                    _PREFILL_BODY_LOGGED.add(("prefill-rowroute",))
                    logger.info(
                        "[solar_open3] kv_route_per_row: prompt chunks reaching the routed budget take the "
                        "per-row prefill (row_routed_attn)"
                    )
                attended = row_routed_prefill(
                    q[s:e].view(ext, self.num_heads, self.head_dim),
                    key_cache,
                    value_cache,
                    bt_row,
                    p0,
                    stride=stride,
                    route_block=route_block,
                    q_block=q_block,
                    kv_block=128,
                    topk=self.kv_route_topk,
                    sink_tokens=self.kv_route_sink_units * route_block,
                    scale=self.scaling,
                    dedup=self.kv_route_dedup_anchor,
                    score_reduce=self._route_max_all_reduce,
                    cache=plan,
                    cache_key=i,
                    all_reduce=self._route_all_reduce,
                    group_size=self.total_num_heads // self.total_num_kv_heads,
                    kv_head_start=self.kv_head_start,
                    total_kv_heads=self.total_num_kv_heads,
                    qhead=self.kv_route_qhead_block == 1,
                    output_gate=output_gate[s:e] if output_gate is not None else None,
                    output=out[s:e] if output_gate is not None else None,
                ).reshape(ext, -1)
                if output_gate is None:
                    out[s:e] = attended
                continue

            def _kf_at(pos2d, _bt=bt_row, _page=page):
                pg, off = _bt[pos2d // _page], pos2d % _page
                return key_cache[pg, off], value_cache[pg, off]

            attended = routed_prefill_batched(
                q[s:e],
                p0,
                apos,
                _kf_at,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.scaling,
                stride,
                q_block,
                route_block,
                self.kv_route_topk,
                anchor_in_union=self.kv_anchor_in_attention,
                dedup_anchor=self.kv_route_dedup_anchor,
                score_block=None
                if self.kv_route_per_row
                else self.kv_route_score_block,
                reduce="mean" if self.kv_route_per_row else self.kv_route_reduce,
                head_reduce=self.kv_route_head_reduce,
                qhead_block=None
                if self.kv_route_per_row
                else self.kv_route_qhead_block,
                fused_prefill=self.kv_route_prefill_body == "gluon",
                paged_kv=(key_cache, value_cache, bt_row, page),
                total_kv_heads=self.total_num_kv_heads,
                kv_head_start=self.kv_head_start,
                group_size=self.total_num_heads // self.total_num_kv_heads,
                all_reduce=self._route_all_reduce,
                max_reduce=self._route_max_all_reduce,
                writes_score=self._route_writes_score,
                qhead_start=self.qhead_start,
                head_shared=self.kv_route_head_shared,
                plan_cache=plan,
                sink_units=self.kv_route_sink_units,
                output_gate=output_gate[s:e] if output_gate is not None else None,
                output=out[s:e] if output_gate is not None else None,
            )
            if output_gate is None:
                out[s:e] = attended
        # Under tc_piecewise the token count is PADDED up to a captured size, so the batch
        # legitimately carries rows past what extend_seq_lens_cpu accounts for. Those are the
        # engine's padding: it discards them, and the zeros `out` was allocated with are what
        # SGLang's own attention leaves there too. `is_in_tc_piecewise_cuda_graph()` is true for
        # both capture and replay, which is exactly the regime that pads.
        #
        # The check stays hard everywhere else, and stays hard in BOTH directions here: a short
        # cover outside that regime is a real defect -- a row no routed call touched keeps those
        # zeros, and zeros are a perfectly plausible attention output, so the layer would
        # contribute nothing for that token and the answer would still read as fluent. `start > n`
        # is never padding; it means a routed call ran past the end of the query rows.
        if start != n and not (start < n and is_in_tc_piecewise_cuda_graph()):
            raise RuntimeError(
                f"layer {self.attn.layer_id}: extend_seq_lens_cpu sums to {start} but the batch "
                f"carries {n} query rows. {abs(n - start)} row(s) would have been left "
                f"{'unattended' if start < n else 'over-read'}. (A short cover is accepted only "
                "as tc_piecewise capture/replay padding, and this forward is not in one.)"
            )

    def _check_grid_fits(self, bt: torch.Tensor, page: int, num_pages: int) -> None:
        """Once per layer: the decode grid cannot reach past the page table.

        The grid is sized from the engine's position ceiling so it does not move with the sequence
        length -- a captured graph needs that -- and the kernels MASK the units past the row
        rather than skipping them, so every lane the grid covers still ADDRESSES the table. If the
        table is narrower than the ceiling, those lanes read out of bounds, and an out-of-bounds
        read inside a captured graph surfaces as an illegal access with no line number.
        """
        if getattr(self, "_grid_checked", False):
            return
        self._grid_checked = True
        need_pages = -(-self.route_pad_positions // page)
        logger.info(
            "[solar_open3] routed decode grid: pad_positions=%d page=%d -> %d pages needed; "
            "page table is %s; cache holds %d pages",
            self.route_pad_positions,
            page,
            need_pages,
            tuple(bt.shape),
            num_pages,
        )
        if bt.shape[1] < need_pages:
            raise ValueError(
                f"layer {self.attn.layer_id}: the routed decode grid is sized for "
                f"{self.route_pad_positions} positions ({need_pages} pages) but the page table is "
                f"only {bt.shape[1]} pages wide, because req_to_token holds "
                f"{bt.shape[1] * page} positions per request. The masked lanes would read past "
                "the table. Launch with --context-length no larger than that, or raise the "
                "request pool's max context length."
            )

    def _check_kv_roundtrip(
        self, forward_batch, key_cache, k_written, req_to_token
    ) -> None:
        """Once per layer: read back what was just written, through the anchor gather's own math.

        ``page_size`` is a SERVER argument -- the default attention backend raised it to 64 on the
        first run -- while this adapter addresses the cache by position, on the premise that
        ``req_to_token`` holds one slot id per token whatever the page size is. Nothing in the
        config states that premise, and if it were wrong the layer would still run and still
        return fluent text; it would just attend to the wrong keys. So it is proved once, against
        the cache, rather than assumed.
        """
        if getattr(self, "_kv_checked", False):
            return
        # EXTEND only. What this proves -- that a KV slot id reaches, through the page arithmetic,
        # the bytes set_kv_buffer just wrote -- is a property of the layout, so proving it once on
        # a real prefill is enough. Decode batches are where the graph runner's warmups live, and
        # those carry fabricated slots that the comparison cannot judge: it read one as a layout
        # failure and killed the server before the first request.
        if not forward_batch.forward_mode.is_extend():
            return
        if torch.cuda.is_current_stream_capturing():
            # Every line below reads the device to the host, which CUDA refuses mid-capture.
            # Nothing is lost: the layout does not depend on the batch, so the first uncaptured
            # step performs the check.
            return
        n = min(64, k_written.shape[0])
        if n == 0:
            return
        slots = forward_batch.out_cache_loc[:n]
        if bool((slots < 0).any()):
            return  # a warmup batch that owns no slots proves nothing
        # The check compares row i against the slot row i was written to, so it only means
        # anything while each row HAS its own slot. A startup warmup batch writes every padded
        # token to the same sink page, and reading that back returns the last write for all of
        # them -- a mismatch that says nothing about the layout. Wait for a batch with distinct
        # slots rather than reporting the padding as a layout failure.
        if int(torch.unique(slots).numel()) != int(slots.numel()):
            return
        self._kv_checked = True
        page = self.page_size
        got = key_cache[slots // page, slots % page]
        if not torch.equal(got, k_written[:n]):
            raise RuntimeError(
                f"layer {self.attn.layer_id}: KV round trip failed -- reading the cache at the "
                "slot ids req_to_token holds does not return what set_kv_buffer just wrote. The "
                "routed bodies address anchors and selected units exactly this way, so they "
                f"would attend to the wrong keys and still produce plausible scores (page_size="
                f"{page})."
            )
        # ...and that a POSITION reaches the same place through the page table, which is what the
        # routed bodies are actually handed. Page-aligned allocation is what makes
        # `req_to_token[req, ::page] // page` a page table at all; nothing states it.
        req = int(forward_batch.req_pool_indices[0])
        total = (
            int(forward_batch.seq_lens_cpu[0])
            if forward_batch.seq_lens_cpu is not None
            else 0
        )
        m = min(total, 4 * page)
        # A request whose table row is still all zeros has not been allocated -- another warmup
        # shape -- and comparing against it would compare two zeros and prove nothing.
        if m > 0 and bool(req_to_token[req, :m].any()):
            pos = torch.arange(m, device=k_written.device)
            slot_by_pos = req_to_token[req, :m].long()
            pg = (req_to_token[req, ::page][: (m + page - 1) // page] // page).long()
            slot_by_page = pg[pos // page] * page + (pos % page)
            if not torch.equal(slot_by_pos, slot_by_page):
                bad = int((slot_by_pos != slot_by_page).nonzero()[0])
                raise RuntimeError(
                    f"layer {self.attn.layer_id}: position {bad} of request {req} lives in KV "
                    f"slot {int(slot_by_pos[bad])}, but the page table this adapter builds sends "
                    f"it to slot {int(slot_by_page[bad])}. Allocation is not page-aligned at "
                    f"--page-size {page}, so a routing unit named by page id does not hold the "
                    "positions the router thinks it holds."
                )


def _refuse_speculative_tree(sa, prefix: str) -> None:
    """Refuse a tree-shaped speculative proposal at STARTUP, not per batch.

    The routed decode addresses the cache BY POSITION, which is exact only for a LINEAR draft
    chain. A tree gives siblings the same position while the cache is written by tree index
    (`assign_extend_cache_locs_uniform`), so a row would attend to its sibling's key and miss its
    own -- silently, and only on the routed layers.

    This is decided from server arguments, which are constant for the process, so it belongs here
    rather than in the forward: a per-batch check costs a lookup on every layer of every step and
    tells the operator only after the weights are loaded. The batch itself carries the tree only in
    `custom_mask`, which this path never reads, so an absent width must not read as "linear".
    """
    algo = getattr(sa, "speculative_algorithm", None)
    if algo in (None, "", "NONE"):
        return
    # ⚠️ An ABSENT topk is not a tree. SGLang normalises the knob per algorithm before the model
    # is built -- EAGLE/EAGLE3/DFLASH/DSPARK all set it, and NGRAM sets it from
    # `speculative_ngram_max_bfs_breadth` -- so the only algorithms that arrive with `None` are the
    # ones whose proposal is a linear chain (FROZEN_KV_MTP is the shipped example). Defaulting a
    # missing width to "tree" refused those servers outright.
    topk = getattr(sa, "speculative_eagle_topk", None)
    width = 1 if topk is None else int(topk)
    if width > 1:
        raise NotImplementedError(
            f"{prefix}: solar_open3 routed attention serves a LINEAR speculative draft chain only, "
            f"but this server proposes a tree (algorithm={algo!r}, eagle topk={topk!r}). Rows that "
            "share a position would attend to a sibling's key instead of their own, with no error. "
            "Serve with --speculative-eagle-topk 1."
        )


def _step_chunk_starts(forward_batch: ForwardBatch, ext_lens) -> list[int]:
    """The first position of every request's chunk, read from the batch once per forward.

    One gather and one device-to-host read for the whole step, shared by all ten routed layers --
    the alternative is either the same read per layer, or trusting an invariant the batch does not
    have to satisfy.

    Cached on the batch, which the decode page table deliberately is NOT: this runs only on the
    extend path, and the routed prefill is never captured in a CUDA graph (``_init_routing``
    refuses that configuration), so there is no replay against a reused batch object to go stale
    against.
    """
    v = getattr(forward_batch, "_solar_open3_p0", None)
    if v is not None and len(v) != len(ext_lens):
        v = None  # a different batch shape on the same object
    if v is None:
        idx, at = [], 0
        for ext in ext_lens:
            idx.append(at)
            at += int(ext)
        pos = forward_batch.positions
        v = (
            [int(x) for x in pos[torch.tensor(idx, device=pos.device)].tolist()]
            if idx and pos is not None
            else [0] * len(idx)
        )
        try:
            forward_batch._solar_open3_p0 = v
        except AttributeError:
            pass
    return v


def _step_plan(forward_batch: ForwardBatch, n: int) -> dict:
    """The routed bodies' per-step scratch, shared by every routed layer of this forward.

    The plans they cache (block arithmetic, decode segmentation) depend on the STEP, not on the
    layer, so ten routed layers would otherwise each rebuild the same thing.

    ⚠️ Keyed on the token count. The CUDA graph runner captures one graph per batch size against a
    ForwardBatch object it REUSES, so anything cached on that object outlives the capture it was
    built for: a plan built while capturing 183 rows would be baked into the graph captured for 1,
    and the shapes it names are not that graph's. The token count is what distinguishes them, and
    it is the same guard the vLLM adapter puts on its own step plan.
    """
    plan = getattr(forward_batch, "_solar_open3_plan", None)
    if plan is None or plan.get("__n") != n:
        plan = {"__n": n}
        try:
            forward_batch._solar_open3_plan = plan
        except AttributeError:  # slotted batch: correct, just not shared
            pass
    return plan


def _step_page_table(
    forward_batch: ForwardBatch,
    req_to_token: torch.Tensor,
    page: int,
    rows_per_req: int = 1,
) -> torch.Tensor:
    """Full-width PAGE ids, one row per query (repeated per request on verify).

    Full width on purpose. The decode grid is sized from the engine's position ceiling so it does
    not move with the sequence length -- which is what a captured graph needs -- and the kernels
    mask the padded units rather than skipping them, so the table has to be addressable wherever
    the grid reaches. A narrower gather would turn masked lanes into out-of-bounds reads.

    ``req_to_token[:, ::page]`` is a strided VIEW, so only the ``(batch, pages)`` result is
    materialised -- at page 64 that is 1/64th of what a per-position table would copy.

    ⚠️ NOT cached on the batch, unlike the chunk starts. A decode step is what CUDA graphs
    capture, and a captured graph is replayed against a ForwardBatch the runner reuses: a table
    cached on that object would name the pages the requests owned when the graph was captured, and
    every later replay would attend to another request's cache with nothing to say so. The gather
    is cheap and unconditional for exactly that reason -- the vLLM adapter reads its block table
    every step on the same grounds.
    """
    if req_to_token.is_cuda and torch.version.hip is None:
        from sglang.kernels.ops.attention.solar_open3.page_table import (
            solar_open3_page_table,
        )

        return solar_open3_page_table(
            req_to_token, forward_batch.req_pool_indices, page, rows_per_req
        )
    table = req_to_token[:, ::page][forward_batch.req_pool_indices] // page
    return table.repeat_interleave(rows_per_req, dim=0) if rows_per_req != 1 else table


def _route_geometry(config, name: str, default: int) -> int:
    """A routing WIDTH off the config: absent takes `default`, anything else is used as written.

    Ported from the vLLM adapter so the two engines accept the same configurations. ``or`` would
    read 0 as absent and hand back the default -- a different unit width or query tile than the
    checkpoint was trained with, chosen silently. The width decides which keys a query reads, so a
    value the layer cannot build is named here instead.
    """
    v = getattr(config, name, None)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(
            f"{name}={v!r} is not a positive integer. It sets the routing geometry -- the unit "
            f"width and the query tile the router scores -- so a value the layer cannot build is "
            f"refused rather than replaced by the default {default} the checkpoint never used."
        )
    return v


def _refuse_attn_res_under_communication(layer, prefix: str) -> None:
    """AttnRes may bypass boundaries only when every bound row layout is full.

    Check the resolved upstream plans after layer_stack binds them. AttnRes
    still completes its own attention sum; its fused reads cannot perform the
    token transport otherwise owned by these boundaries.
    """
    moved = []
    for name in ("attn_boundary", "ffn_boundary"):
        boundary = getattr(layer, name)
        for variant, edges in boundary.plan.edges.items():
            for side in ("incoming", "outgoing"):
                edge = getattr(edges, side)
                layouts = (
                    edge.produced.layout,
                    edge.need.layout,
                    edge.residual,
                    edge.residual_to,
                )
                if any(layout.sharded for layout in layouts):
                    moved.append(f"{name}/{variant.name}/{side}")
    if is_dp_attention_enabled():
        moved.append("data-parallel attention is enabled")
    if moved:
        raise NotImplementedError(
            f"{prefix}: Attention-Residuals cannot run in this parallel layout ("
            + ", ".join(moved)
            + "). Its fused residual reads bypass boundary token transport. "
            "Serve without data-parallel attention or an all-to-all MoE backend, "
            "or use the standard-residual path."
        )


class SolarOpen3GatedRMSNorm(RMSNorm):
    """Qwen3.8 GatedNorm for the SGLang adapter: RMSNorm followed by a low-rank self-gate.

    ``out = y * sigmoid(W_up SiLU(W_down y))`` where ``y`` is the normalised read. The RMSNorm half
    is inherited and, on the AttnRes path, is not even run here: the fused depth-mix kernel applies
    it as its epilogue and hands `apply_gate` the already-normalised tensor -- exactly the split the
    vLLM adapter uses.

    ``ReplicatedLinear``, not the column/row pair the vLLM adapter uses: the gate is a d -> r -> d
    bottleneck with r = d/8, so a full copy per rank costs 2 MB per gate (80 MB for a 40-layer
    MoE-read-only checkpoint) and buys back the all-reduce that a row-parallel ``gate_up`` would put
    between the rank reduction and the sigmoid. The sigmoid is not linear, so that all-reduce cannot
    be deferred; replicating keeps every rank's gate bit-identical to the single-GPU one.
    """

    def __init__(
        self, hidden_size: int, gate_hidden_size: int, eps: float, prefix: str
    ) -> None:
        super().__init__(hidden_size, eps=eps)
        # No quant_config parameter on purpose (yoonniverse review #11): the gate is a forward-pass modulator whose output
        # multiplies the residual stream elementwise, and quantising it would change the FUNCTION
        # rather than the arithmetic of a matmul. The trainer keeps it in bf16 for the same reason.
        self.gate_down = ReplicatedLinear(
            hidden_size,
            gate_hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate_down",
        )
        self.gate_up = ReplicatedLinear(
            gate_hidden_size,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate_up",
        )

        graph = get_exec().graph.cuda_graph_config
        decode = getattr(graph, "decode", None)
        cap = getattr(decode, "max_bs", None)
        if not cap:
            cap = max(getattr(decode, "bs", None) or [512])
        self.gate_token_cap = min(cap, 512) if cap > 0 else 512
        self.gate_fused = (
            os.environ.get("SOLAR_OPEN3_FUSED_GATE", "1") != "0"
            and get_parallel().tp_size == 1
            and not get_exec().deterministic.enable_deterministic_inference
        )

    def apply_gate(self, normalized: torch.Tensor) -> torch.Tensor:
        """Use vLLM's fused rounding on TP1 CUDA shapes below the graph ceiling."""
        if (
            self.gate_fused
            and normalized.is_cuda
            and type(self.gate_down) is ReplicatedLinear
            and type(self.gate_up) is ReplicatedLinear
        ):
            from sglang.kernels.ops.attention.solar_open3.gated_norm import (
                fused_self_gate,
            )

            return fused_self_gate(
                normalized,
                self.gate_down.weight,
                self.gate_up.weight,
                self.gate_token_cap,
            )
        gate, _ = self.gate_down(normalized)
        gate, _ = self.gate_up(torch.nn.functional.silu(gate))
        return normalized * torch.sigmoid(gate)

    # `RMSNorm.forward` is deliberately NOT overridden. It is SGLang's dispatch point (backend
    # forcing, and a wider signature than `(x, residual)` on newer releases), and nothing on the
    # gated path calls it: GatedNorm requires AttnRes, and there the depth-mix kernel applies the
    # RMSNorm half and the layer calls `apply_gate` on its output. Overriding it would put a
    # narrower signature on a method SGLang may call with more arguments, for a path that cannot
    # be reached (Codex review, low). This module is a gate CARRIER for the AttnRes path plus the
    # ordinary norm SGLang already knows how to run.


class _ClampedSiluAndMul(nn.Module):
    """``SiluAndMul`` on the [T, 2I] gate|up halves with the trainer's DeepSeek clamp:
    ``silu(gate.clamp(max=L)) * up.clamp(-L, L)`` (components/feedforward.py)."""

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and torch.version.hip is None
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.is_contiguous()
            and x.shape[-1] % 32 == 0
            and self.swiglu_limit > 0
        ):
            from sglang.kernels.ops.activation.activation import (
                silu_and_mul_with_activation_rounding,
            )

            # Preserve the native SiLU intermediate rounding before multiplying.
            return silu_and_mul_with_activation_rounding(
                x, clamp_limit=self.swiglu_limit
            )
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate.clamp(max=self.swiglu_limit)) * up.clamp(
            min=-self.swiglu_limit, max=self.swiglu_limit
        )


def _clip_token_norm(y: torch.Tensor, limit: float) -> torch.Tensor:
    """The trainer's ``clip_token_norm`` (components/moe.py)."""
    n = y.float().norm(dim=-1, keepdim=True)
    return y * (limit / n.clamp_min(limit)).to(y.dtype)


def _clamp_moe(moe: SolarOpen3MoE, config) -> None:
    """Give the MoE the parent layer built the trainer's clamps and clips, in place: none adds a
    parameter, and rebuilding would allocate every expert twice. The routed bound goes on the
    ``MoeRunnerConfig`` FusedMoE shares with its runner."""
    routed = _bound(config, "swiglu_limit")
    if routed is not None:
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )

        if isinstance(moe.experts.quant_method, CompressedTensorsFusedMoEMethod):
            # compressed-tensors delegates create_moe_runner to the layer's scheme.
            runner = moe.experts.scheme.runner
        else:
            runner = moe.experts.quant_method.runner
        if not runner.runner_backend.is_triton():
            raise NotImplementedError(
                f"solar_open3 SGLang applies swiglu_limit={routed} only through the triton MoE runner "
                f"(--moe-runner-backend triton), not {runner.runner_backend!r}."
            )
        moe.experts.moe_runner_config.swiglu_limit = routed
        assert (
            runner.config is runner.runner_core.config is moe.experts.moe_runner_config
        ), (
            "MoE runner holds a copy of MoeRunnerConfig: the routed SwiGLU clamp would not reach the kernel"
        )
    # BF16 Sonic applies SwiGLU directly to the FP32 GEMM accumulator. Keep
    # canonical expert weights; the Triton epilogue pairs gate/up while loading.
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    method = moe.experts.quant_method
    if (
        isinstance(method, UnquantizedFusedMoEMethod)
        and method.runner.runner_backend.is_triton()
        and moe.experts.w13_weight.dtype == torch.bfloat16
        and moe.experts.w13_weight.device.type == "cuda"
        and torch.version.hip is None
    ):
        moe.experts.moe_runner_config.activation = "silu_fp32_accum"
    shared = _bound(config, "shared_expert_swiglu_limit")
    if shared is not None:
        moe.shared_experts.act_fn = _ClampedSiluAndMul(shared)
    moe.routed_out_clip = _bound(config, "routed_out_clip")
    moe.shared_out_clip = _bound(config, "shared_out_clip")


class _VllmRouterTopK(TopK):
    """vLLM's fp32 sigmoid/top-k/normalization over the fp32 router GEMM."""

    def forward_cuda(
        self,
        hidden_states,
        router_logits,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        if (
            num_token_non_padded is not None
            or expert_location_dispatch_info is not None
        ):
            return super().forward_cuda(
                hidden_states,
                router_logits.float(),
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
        config = self.topk_config
        # The Triton runner scales the routed expert sum. Scaling these weights
        # as well would apply routed_scaling_factor twice.
        weights, ids = vllm_router(
            router_logits,
            config.top_k,
            config.correction_bias,
            config.renormalize,
        )
        capture_routed_experts_if_allowed(config, self.layer_id, ids)
        get_global_expert_distribution_recorder().on_select_experts(topk_ids=ids)
        return StandardTopKOutput(weights, ids, router_logits.float())

    forward_native = forward_cuda

    def _torch_compile_forward(self, num_tokens):
        return None


def _vllm_router_moe(moe: SolarOpen3MoE) -> None:
    config = moe.topk.topk_config
    if not (
        config.use_grouped_topk
        and config.num_expert_group == 1
        and config.topk_group == 1
        and config.num_fused_shared_experts == 0
        and config.custom_routing_function is None
        and not config.apply_routed_scaling_factor_on_output
        and not moe.topk.enable_waterfill
        and get_moe_runner_backend().is_triton()
    ):
        raise NotImplementedError(
            "SolarOpen3 vLLM-compatible routing requires single-group Triton MoE without fused shared experts"
        )
    moe.topk.__class__ = _VllmRouterTopK


class SolarOpen3DecoderLayer(SolarOpen2DecoderLayer):
    """Shared Open2 stages with Open3 attention, MoE, and residual policies."""

    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        # Attention-Residuals: one depth-mix (pseudo-query + key-norm gain) per SUBLAYER. Both are
        # replicated across TP ranks. Built here only when the checkpoint asks for it, so the
        # standard-residual path keeps exactly the parameters it had.
        self.use_attn_res = getattr(config, "attn_res_block_size", None) is not None
        self.use_cp_attn_res = (
            self.use_attn_res and getattr(config, "attn_res_scorer", "static") == "cp"
        )
        # A pure forward-pass constant, no weights and no TP implications: it multiplies the mix by
        # its source count so the fused RMSNorm's epsilon is not amplified by K^2.
        self.attn_res_scale = self.use_attn_res and bool(
            getattr(config, "attn_res_scale_by_sources", False)
        )
        # GatedNorm: WHICH AttnRes reads carry the low-rank self-gate. A config without the key
        # means both -- the shape of every gated checkpoint exported before the knob existed. The
        # prenorms are REPLACED (not wrapped) so the checkpoint keys stay `<read>_layernorm.weight`
        # plus the two gate matrices, exactly as the trainer writes them.
        gated = self.use_attn_res and bool(
            getattr(config, "attn_res_gated_norm", False)
        )
        gated_reads = set(
            getattr(config, "attn_res_gated_norm_reads", None) or ("attn", "moe")
        )
        self.gate_attn_read = gated and "attn" in gated_reads
        self.gate_moe_read = gated and "moe" in gated_reads
        if gated:
            gate_hidden = (
                getattr(config, "attn_res_gated_norm_dim", None)
                or config.hidden_size // 8
            )
            if self.gate_attn_read:
                self.input_layernorm = SolarOpen3GatedRMSNorm(
                    config.hidden_size,
                    gate_hidden,
                    config.rms_norm_eps,
                    prefix=f"{prefix}.input_layernorm",
                )
            if self.gate_moe_read:
                self.post_attention_layernorm = SolarOpen3GatedRMSNorm(
                    config.hidden_size,
                    gate_hidden,
                    config.rms_norm_eps,
                    prefix=f"{prefix}.post_attention_layernorm",
                )
        if self.use_attn_res:
            block = config.attn_res_block_size
            # Both attention modules build `o_proj` with `reduce_results=False`: under tensor
            # parallelism every rank leaves the sublayer with a PARTIAL sum, and it is the layer
            # FFN boundary's input that all-reduces it before the post-attention norm.
            # `forward_attn_res` bypasses the boundaries, so it has to do that reduce itself --
            # at TP=1 the omission is invisible (one rank, whole sum), at TP=8 the trunk answers
            # in digits. The MoE reduces its own output and needs nothing here.
            #
            # Keyed on the ATTENTION tp group, which is what the FFN input keys on and what the
            # collective below actually reduces over. It equals the TP world size only while
            # attention is not data-parallel -- refused after the boundaries bind -- so reading the world size
            # instead would be a trap for whoever lifts that refusal.
            self.attn_res_all_reduce = get_parallel().attn_tp_size > 1
            template = self.input_layernorm.weight
            if self.use_cp_attn_res:
                num_sources = (
                    attn_res_writes_before(2 * config.num_hidden_layers, block) + 1
                )
                qdim = config.attn_res_scorer_dim
                for who in ("self_attention", "mlp"):
                    setattr(
                        self,
                        f"{who}_res_query_proj",
                        nn.Parameter(
                            torch.empty(
                                qdim,
                                config.hidden_size,
                                dtype=template.dtype,
                                device=template.device,
                            )
                        ),
                    )
                    # DERIVED, not loaded: filled from E @ Wq after the checkpoint is in.
                    # Non-persistent so it never appears in a saved state dict.
                    self.register_buffer(
                        f"{who}_res_score_weight",
                        torch.empty(
                            num_sources,
                            config.hidden_size,
                            dtype=template.dtype,
                            device=template.device,
                        ),
                        persistent=False,
                    )
            else:
                for who in ("self_attention", "mlp"):
                    setattr(
                        self,
                        f"{who}_res_norm",
                        RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                    )
                    setattr(
                        self,
                        f"{who}_res_proj",
                        ReplicatedLinear(
                            config.hidden_size,
                            1,
                            bias=False,
                            quant_config=None,
                            prefix=f"{prefix}.{who}_res_proj",
                        ),
                    )
            # Global SUBLAYER indices: this layer's attention is 2i, its MLP 2i+1. Everything the
            # fused op needs per sublayer is a Python int fixed at construction, so it specialises
            # at torch.compile trace time and becomes a Triton constexpr.
            u_attn, u_mlp = 2 * layer_idx, 2 * layer_idx + 1
            self.attn_res_nb_attn = attn_res_writes_before(u_attn, block)
            self.attn_res_nb_mlp = attn_res_writes_before(u_mlp, block)
            # u == 0 banks the token embedding; every later multiple of the block size closes the
            # open block. The written column is always the current bank width, which is also
            # ``num_blocks`` for that call -- so the snapshot IS the prefix the mix attends over.
            self.attn_res_widx_attn = (
                self.attn_res_nb_attn if (u_attn % block) == 0 else -1
            )
            self.attn_res_widx_mlp = (
                self.attn_res_nb_mlp if (u_mlp % block) == 0 else -1
            )

    def _create_attention(self, config, layer_idx, quant_config, prefix):
        self.is_kda = _is_kda_layer(config, layer_idx)
        attention_cls = (
            SolarOpen3KimiDeltaAttention if self.is_kda else SolarOpen3Attention
        )
        return attention_cls(
            layer_idx=layer_idx,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _create_moe(self, config, layer_idx, quant_config, prefix):
        moe = SolarOpen3MoE(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=self.alt_stream,
        )
        _clamp_moe(moe, config)
        _vllm_router_moe(moe)
        return moe

    def _forward_attention(
        self, positions, hidden_states, forward_batch, zero_allocator
    ):
        if forward_batch.forward_mode.is_idle():
            return hidden_states
        if not self.is_kda:
            return self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
            )
        return super()._forward_attention(
            positions, hidden_states, forward_batch, zero_allocator
        )

    def forward(
        self, positions, hidden_states, forward_batch, residual, zero_allocator
    ):
        hidden_states, stream = ResidualStream.from_handoff(
            hidden_states, residual, PLAIN_ADD
        )
        forward_batch.residual_stream = stream
        hidden_states = super().forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        hidden_states, residual = stream.export(hidden_states)
        forward_batch.residual_stream = None
        return hidden_states, residual

    def forward_attn_res(
        self, positions, forward_batch, prefix, delta, blocks, zero_allocator
    ):
        """One decoder layer under Attention-Residuals, on the fused three-tensor state.

        ``prefix`` ``[T, H]`` is the open block's running residual sum, ``delta`` the previous
        sublayer's output not yet added into it (the op does the add), and ``blocks`` ``[T, N, H]``
        the depth-source bank, written in place at block boundaries. Returns the new
        ``(prefix, delta)``; ``blocks`` is mutated.

        Each sublayer is ONE fused launch that does the residual add, the block snapshot, the depth
        mix and the following RMSNorm -- so ``input_layernorm`` / ``post_attention_layernorm`` are
        loaded and their gains passed to the kernel, but their ``forward`` is never called.

        When a sublayer closes a block the running prefix RESETS to that sublayer's output rather
        than accumulating (``delta = None``), because the pre-existing prefix was just snapshotted
        into the bank. Getting this wrong double-counts a block or grows the state with depth.

        Kept as a SEPARATE method rather than a widened ``forward``: the standard-residual path
        then stays byte-identical, and SGLang's ``(hidden_states, residual)`` layer contract is
        untouched.
        """
        # --- attention sublayer ---
        if self.use_cp_attn_res:
            hidden_states = cp_attn_res(
                prefix,
                delta,
                blocks,
                self.self_attention_res_score_weight,
                self.input_layernorm.weight,
                self.attn_res_nb_attn,
                self.attn_res_widx_attn,
                self.input_layernorm.variance_epsilon,
                self.attn_res_scale,
            )
        else:
            hidden_states = attn_res(
                prefix,
                delta,
                blocks,
                self.self_attention_res_norm.weight,
                self.self_attention_res_proj.weight.squeeze(0),
                self.input_layernorm.weight,
                self.attn_res_nb_attn,
                self.attn_res_widx_attn,
                self.self_attention_res_norm.variance_epsilon,
                self.input_layernorm.variance_epsilon,
                self.attn_res_scale,
            )
        # The kernel applied the RMSNorm half as its epilogue; the gate is the rest of the read.
        if self.gate_attn_read:
            hidden_states = self.input_layernorm.apply_gate(hidden_states)
        # KDA carries recurrent state and needs the step's scratch allocator; the routed layers
        # take the three-argument form. Same split as the inherited layer forward.
        if self.is_kda:
            attn_output = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )
        else:
            attn_output = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        if self.attn_res_all_reduce:
            # What the FFN boundary input does for the standard path (see __init__),
            # INCLUDING its choice of collective: it quantizes a non-decode forward when
            # communication quantization is on, and always did -- picking the plain one
            # unconditionally would silently opt this path out of a feature the operator enabled
            # and leave its prefill numerics differing from the standard-residual path it exists
            # to reproduce. (Server args refuse the flag off NPU today, so this is a latent trap
            # rather than a live one; it costs one branch to not leave it armed.)
            if (
                not forward_batch.forward_mode.is_decode_or_idle()
                and get_exec().comm.enable_quant_communications
            ):
                attn_output = attention_tensor_model_parallel_quant_all_reduce(
                    attn_output
                )
            else:
                attn_output = attention_tensor_model_parallel_all_reduce(attn_output)
        if self.attn_res_widx_attn >= 0:  # the block was banked -> start a fresh one
            prefix, delta = attn_output, None
        else:
            delta = attn_output

        # --- MLP/MoE sublayer ---
        if self.use_cp_attn_res:
            hidden_states = cp_attn_res(
                prefix,
                delta,
                blocks,
                self.mlp_res_score_weight,
                self.post_attention_layernorm.weight,
                self.attn_res_nb_mlp,
                self.attn_res_widx_mlp,
                self.post_attention_layernorm.variance_epsilon,
                self.attn_res_scale,
            )
        else:
            hidden_states = attn_res(
                prefix,
                delta,
                blocks,
                self.mlp_res_norm.weight,
                self.mlp_res_proj.weight.squeeze(0),
                self.post_attention_layernorm.weight,
                self.attn_res_nb_mlp,
                self.attn_res_widx_mlp,
                # Its OWN norm's eps -- numerically identical to the attention branch's (every norm
                # is built from rms_norm_eps) but semantically this norm's value.
                self.mlp_res_norm.variance_epsilon,
                self.post_attention_layernorm.variance_epsilon,
                self.attn_res_scale,
            )
        if self.gate_moe_read:
            hidden_states = self.post_attention_layernorm.apply_gate(hidden_states)
        mlp_output = self.mlp(hidden_states)
        if self.attn_res_widx_mlp >= 0:
            prefix, delta = mlp_output, None
        else:
            delta = mlp_output
        return prefix, delta


class SolarOpen3Model(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_parallel().pp_group
        self.dflash_layers_to_capture: Optional[list[int]] = None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.alt_stream = torch.cuda.Stream()
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SolarOpen3DecoderLayer(
                layer_idx=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
            stage_facts=lambda idx: SolarOpen3DecoderLayer.stage_facts(config, idx),
        )
        for idx in range(self.start_layer, self.end_layer):
            layer = self.layers[idx]
            if layer.use_attn_res:
                _refuse_attn_res_under_communication(layer, f"{prefix}.layers.{idx}")

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # Attention-Residuals, model level: the depth-source bank's width and the final
        # aggregation's own scorer. Built only when the checkpoint asks for it.
        self.use_attn_res = getattr(config, "attn_res_block_size", None) is not None
        # ``mtp_hidden_prenorm``: the MTP draft reads the trunk hidden BEFORE ``self.norm``, handed
        # to the LogitsProcessor as ``hidden_states_before_norm`` (the hidden it stores for the
        # draft) while logits still come from the normed hidden. Must match training; configs
        # without the field were trained post-norm. Declared and type-checked in ``config.py``.
        self.return_prenorm = config.mtp_hidden_prenorm
        # Logged because nothing else shows it: the main logits are the same either way, so an
        # adapter that predates the field serves a prenorm checkpoint with correct scores and
        # only a lower draft acceptance. A boot log without this line is that older adapter.
        logger.info(
            "[solar_open3] mtp_hidden_prenorm=%s (MTP draft reads the trunk hidden %s the "
            "final norm; from config.json, not the checkpoint)",
            self.return_prenorm,
            "before" if self.return_prenorm else "after",
        )
        # Logged for the same reason: nothing else in the boot log says which SDA erase is served.
        logger.info(
            "[solar_open3] exact_erase=%s (SDA erase from config.json; served by the "
            "%s, checked in post_load_weights)",
            config.exact_erase,
            "aiter flash_sda kernel (layers/attention/solar_open3/amd)"
            if is_hip() and getattr(config, "linear_attn", None) == "sda"
            else "FlashSDA plugin",
        )
        self.use_cp_attn_res = (
            self.use_attn_res and getattr(config, "attn_res_scorer", "static") == "cp"
        )
        self.attn_res_scale = self.use_attn_res and bool(
            getattr(config, "attn_res_scale_by_sources", False)
        )
        # CP's score buffers are DERIVED after loading, so a forward before that fold would mix
        # depth sources with uninitialised memory and never raise. Non-CP is loaded by definition.
        self._cp_attn_res_loaded = not self.use_cp_attn_res
        # Same fail-closed device for the gate matrices (hancheolcho review): a partial runtime
        # update leaves layers on two checkpoints, and sigmoid maps any mix into (0, 1) with no
        # error. False from `load_weights` entry until every gate matrix has been seen.
        self._gated_norm_loaded = not (
            self.use_attn_res and bool(getattr(config, "attn_res_gated_norm", False))
        )
        if self.use_attn_res:
            if self.pp_group.world_size > 1:
                raise NotImplementedError(
                    "solar_open3 AttnRes does not support pipeline parallel: the depth-source bank "
                    "would have to ride PPProxyTensors across ranks. Use tensor parallel (the "
                    "depth mix is replicated and TP-exact)."
                )
            # Bank width EXCLUDES the still-open prefix; every mix sees num_blocks + 1 sources.
            self.num_attn_res_blocks = attn_res_writes_before(
                2 * config.num_hidden_layers, config.attn_res_block_size
            )
            num_sources = self.num_attn_res_blocks + 1
            template = self.norm.weight
            if self.use_cp_attn_res:
                qdim = config.attn_res_scorer_dim
                self.attn_res_key_embed = nn.Parameter(
                    torch.empty(
                        num_sources, qdim, dtype=template.dtype, device=template.device
                    )
                )
                self.output_attn_res_query_proj = nn.Parameter(
                    torch.empty(
                        qdim,
                        config.hidden_size,
                        dtype=template.dtype,
                        device=template.device,
                    )
                )
                self.register_buffer(
                    "output_attn_res_score_weight",
                    torch.empty(
                        num_sources,
                        config.hidden_size,
                        dtype=template.dtype,
                        device=template.device,
                    ),
                    persistent=False,
                )
            else:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if get_parallel().pp_group.is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_tokens(input_ids)
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        total_num_layers = self.end_layer - self.start_layer
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        aux_hidden_states = []

        # True only on the AttnRes path, where the output norm is folded into the final
        # aggregation. The standard-residual path leaves it False and normalises as before.
        _attn_res_normed = False

        # The fallback channel for the batch (see `_current_batch`): used on the paths where the
        # engine sets no tc_piecewise context of its own. A stack rather than a slot: a draft
        # worker or an overlapped step can nest one forward inside another, and the op must see
        # ITS OWN batch, not the outer one.
        # ⚠️ Under tc_piecewise this whole function is TRACED, so the append below does not
        # happen when the op runs -- which is exactly why the engine's context is read first.
        _CURRENT_BATCH.append(forward_batch)
        try:
            if self.use_attn_res:
                if self.use_cp_attn_res and not self._cp_attn_res_loaded:
                    raise RuntimeError(
                        "CP depth routers were not folded from the checkpoint. The score buffers "
                        "are derived after load_weights, so serving now would mix depth sources "
                        "with uninitialised memory and produce plausible tokens. Every loader "
                        "SGLang ships reaches the fold through load_weights or post_load_weights; "
                        "a path that reaches this guard is one that calls neither."
                    )
                if not self._gated_norm_loaded:
                    raise RuntimeError(
                        "AttnRes GatedNorm gate matrices were not fully loaded. A partially "
                        "updated gate still returns sigmoid(...) in (0, 1), so serving now would "
                        "scale the residual stream by a plausible number with no error."
                    )
                # ONE preallocated [T, N, H] depth-source bank plus a running prefix. The width is
                # a compile-time constant and the allocation happens once per forward, which is
                # what keeps this CUDA-graph safe. ``new_empty`` is safe uninitialised: sublayer 0
                # banks the token embedding into column 0 and reads no column before it is written.
                blocks = hidden_states.new_empty(
                    hidden_states.shape[0],
                    self.num_attn_res_blocks,
                    hidden_states.shape[1],
                )
                prefix, delta = hidden_states, None
                for i in range(self.start_layer, self.end_layer):
                    with get_global_expert_distribution_recorder().with_current_layer(
                        i
                    ):
                        prefix, delta = self.layers[i].forward_attn_res(
                            positions,
                            forward_batch,
                            prefix,
                            delta,
                            blocks,
                            zero_allocator,
                        )
                    if (
                        self.dflash_layers_to_capture is not None
                        and i in self.dflash_layers_to_capture
                    ):
                        # The layer output is the open block's residual stream.
                        # Later AttnRes reads update prefix in place.
                        aux_hidden_states.append(
                            prefix.clone() if delta is None else prefix + delta
                        )
                # Final aggregation over every banked source plus the still-open prefix, with the
                # model output norm FOLDED IN -- self.norm applies here either way, so folding it
                # removes one unfused RMSNorm per forward. The return is still post-norm, matching
                # the standard-residual path, so the caller below is unchanged.
                # Under ``return_prenorm`` nothing is folded: the draft needs the pre-norm hidden,
                # and the epilogue below applies the norm unfused.
                out_norm = None if self.return_prenorm else self.norm.weight
                if self.use_cp_attn_res:
                    hidden_states = cp_attn_res(
                        prefix,
                        delta,
                        blocks,
                        self.output_attn_res_score_weight,
                        out_norm,
                        self.num_attn_res_blocks,
                        -1,
                        self.norm.variance_epsilon,
                        self.attn_res_scale,
                    )
                else:
                    hidden_states = attn_res(
                        prefix,
                        delta,
                        blocks,
                        self.output_attn_res_norm.weight,
                        self.output_attn_res_proj.weight.squeeze(0),
                        out_norm,
                        self.num_attn_res_blocks,
                        -1,
                        self.output_attn_res_norm.variance_epsilon,
                        self.norm.variance_epsilon,
                        self.attn_res_scale,
                    )
                # ⚠️ `self.norm` is already FOLDED INTO the aggregation above, so the epilogue
                # must not apply it again. Setting `residual = None` is not enough -- that branch
                # calls `self.norm(hidden_states)`, which would normalise twice and change the
                # function without failing. The flag below makes the epilogue skip it.
                residual, _attn_res_normed = None, not self.return_prenorm
            else:
                for i in range(self.start_layer, self.end_layer):
                    with get_global_expert_distribution_recorder().with_current_layer(
                        i
                    ):
                        hidden_states, residual = self.layers[i](
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            residual=residual,
                            zero_allocator=zero_allocator,
                        )
                    if (
                        self.dflash_layers_to_capture is not None
                        and i in self.dflash_layers_to_capture
                    ):
                        aux_hidden_states.append(
                            hidden_states.clone()
                            if residual is None
                            else hidden_states + residual
                        )
        finally:
            _CURRENT_BATCH.pop()

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        before_norm = None
        if self.return_prenorm and hidden_states.shape[0] != 0:
            before_norm = (
                hidden_states if residual is None else hidden_states + residual
            )
        if hidden_states.shape[0] != 0 and not _attn_res_normed:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                normed = self.norm(hidden_states, residual)
                # RMSNorm returns (x, residual); under fp8 per-tensor quant fusion it returns a
                # third element, so take the first rather than unpacking a fixed arity. Reject a
                # non-tuple: that means `self.norm` is a PPMissingLayer, which returns its input
                # unchanged and would drop the final norm without failing.
                if not isinstance(normed, tuple) or len(normed) < 2:
                    raise RuntimeError(
                        f"self.norm ({type(self.norm).__name__}) returned "
                        f"{type(normed).__name__}, expected (x, residual)"
                    )
                hidden_states = normed[0]
        out = hidden_states, before_norm
        if self.dflash_layers_to_capture is not None:
            return out, aux_hidden_states
        return out


def _trainer_sda(config) -> None:
    """The trainer's gate / beta / q,k arithmetic goes into FlashSDA's SGLang prefill and the KDA decode
    kernel only where both run: FlashSDA hooked the prefill (FLASHSDA_SGLANG=1 with the flash_sda plugin)
    and the decode is the triton backend's regular kernel, which an allow_neg_eigval model takes (at a
    beta_scale of 1 SGLang packs the decode into another kernel). Otherwise the arithmetic SGLang or
    FlashSDA has serves both halves.

    The swap is the arithmetic of a training without exact erase. A checkpoint trained with it (FlashSDA
    PR #7; config.json ``exact_erase: true``) caps beta at 2 - 2^-6 and erases with the fp32 key, which the
    swap's bf16 key and uncapped beta would break (|1 - beta k.k| > 1 again); FlashSDA's prefill and the
    fork's fp32 decode serve it."""
    from sglang.srt.layers.attention.linear.kernels import kda_triton

    args = runtime_options()
    decode = args.linear_attn_decode_backend or args.linear_attn_backend
    # The checkpoint's own (remote-code) config class has no exact_erase_written; it has exact_erase
    # exactly when config.json wrote the key.
    if getattr(config, "exact_erase", False):
        _fall_back(
            "trainer SDA arithmetic",
            "the checkpoint was trained with exact erase (config.json "
            "exact_erase: true)",
        )
    elif not getattr(config, "exact_erase_written", hasattr(config, "exact_erase")):
        _fall_back(
            "trainer SDA arithmetic",
            "config.json has no exact_erase key, so the checkpoint may have "
            "been trained with exact erase; re-convert it with the current "
            "converter, which writes the key",
        )
    elif getattr(
        config, "sda_beta_fp32", False
    ):  # the swap's decode rounds beta to bf16, as does the prefill
        _fall_back(
            "trainer SDA arithmetic",
            "the checkpoint was trained with an fp32 SDA beta (config.json "
            "sda_beta_fp32: true)",
        )
    elif not kda_triton.chunk_kda.__module__.startswith("flash_sda"):
        _fall_back(
            "trainer SDA arithmetic",
            "FlashSDA does not serve the prefill (FLASHSDA_SGLANG=1 with the "
            "flash_sda plugin)",
        )
    elif decode != "triton" or not config.kda_allow_neg_eigval:
        _fall_back(
            "trainer SDA arithmetic",
            f"the decode runs another kernel than the one it replaces (the "
            f"{decode} backend, or the packed one at a beta_scale of 1)",
        )
    elif (
        getattr(args, "linear_attn_verify_backend", None) or "triton"
    ) != "triton":  # unset follows decode
        _fall_back(
            "trainer SDA arithmetic",
            f"the speculative target-verify runs the "
            f"{args.linear_attn_verify_backend} backend, not the triton kernel "
            f"it replaces",
        )
    else:
        from sglang.srt.layers.attention.solar_open3.trainer_sda import install

        try:
            install()
        except (
            ImportError,
            RuntimeError,
        ) as e:  # checked before any swap: nothing changed
            _fall_back("trainer SDA arithmetic", str(e))


class SolarOpen3ForCausalLM(SolarOpen2ForCausalLM):
    """Open3's trunk and loading policies over native Open2 checkpoint packing."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # The parent's __init__ would build a solar_open2 trunk -- a whole model -- before this
        # one replaced it, so its body is re-stated here against SolarOpen3Model rather than
        # called. nn.Module.__init__ is what the parent would have called first.
        nn.Module.__init__(self)
        config = patch_config(config)
        _refuse_unimplemented(config)
        _require_registration()
        _require_cuda_graph_capture()
        exact_kda_prefill_diagonal()
        if getattr(config, "kv_route_layers", None):
            align_routed_prefill_chunks(
                _route_geometry(config, "kv_route_q_block", 256)
            )
        if getattr(config, "linear_attn", None) == "sda":
            _trainer_sda(config)
        self.config = config
        self.quant_config = quant_config
        self.model = SolarOpen3Model(
            config, quant_config, prefix=maybe_prefix(prefix, "model")
        )
        hold_fp32_params(self.model, config)
        self.pp_group = get_parallel().pp_group
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
                use_attn_tp_group=runtime_options().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config=config, logit_scale=logit_scale)
        self.capture_aux_hidden_states = False

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture zero-based layer outputs, as in SolarOpen2 DFlash checkpoints."""
        if self.pp_group.world_size > 1:
            raise NotImplementedError("DFLASH aux hidden capture requires PP=1.")
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        self.model.dflash_layers_to_capture = list(layer_ids)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """The parent's, on the pinned config (read before ``patch_config`` runs)."""
        return super().get_model_config_for_expert_location(pin_served_options(config))

    # --- Speculative decoding (MTP): what the EAGLE worker asks the TARGET for --------------
    #
    # The draft shares the target's token embedding and LM head rather than carrying its own -- the
    # exported `mtp.*` do not include them. `eagle_worker_v2.init_lm_head` calls this on the target
    # and hands the pair to the draft's `set_embed_and_head`. Without it the server dies at startup
    # with AttributeError, which is at least loud.

    @torch.no_grad()  # as on the parent's forward: an inference call must not record autograd
    def forward(
        self,
        input_ids,
        positions,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        """The parent's forward, plus the pre-norm hidden the MTP draft reads under
        ``mtp_hidden_prenorm`` (``None`` otherwise, which is the parent's behaviour)."""
        out = self.model(
            input_ids, positions, forward_batch, inputs_embeds, pp_proxy_tensors
        )
        if not self.pp_group.is_last_rank:
            return out
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            out, aux_hidden_states = out
        hidden_states, before_norm = out
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states=aux_hidden_states,
            # LogitsProcessor gives pre-norm hidden states precedence over aux
            # features. DFLASH needs the captured layer outputs instead.
            hidden_states_before_norm=(
                before_norm if aux_hidden_states is None else None
            ),
        )

    def get_embed_and_head(self):
        """Used by the eagle_worker."""
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker."""
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # --- Attention-Residuals: the checkpoint-side half -------------------------------------
    #
    # The STATIC scorer loads every tensor it uses. The CP scorer does not: the kernel consumes one
    # `[n_src, hidden]` score matrix per consumer, and the checkpoint ships the FACTORS it is built
    # from (a shared key table `E` and a per-consumer query projection `Wq`). So the fold below has
    # to run after loading, and until it does the score buffers hold whatever `torch.empty` left --
    # which would mix depth sources with garbage and never raise.

    def _cp_required_weight_names(self) -> set[str]:
        """Every CP tensor the checkpoint must supply -- the TRAINING factorisation, not the fold."""
        required = {"model.attn_res_key_embed", "model.output_attn_res_query_proj"}
        for layer_idx in range(self.config.num_hidden_layers):
            required.update(
                {
                    f"model.layers.{layer_idx}.self_attention_res_query_proj",
                    f"model.layers.{layer_idx}.mlp_res_query_proj",
                }
            )
        return required

    def _static_required_weight_names(self) -> set[str]:
        """Every static-scorer tensor the checkpoint must supply: per-layer norm/proj for both
        sublayer mixes, plus the output mix's pair -- ``4 * num_hidden_layers + 2``."""
        required = {
            "model.output_attn_res_norm.weight",
            "model.output_attn_res_proj.weight",
        }
        for layer_idx in range(self.config.num_hidden_layers):
            required.update(
                {
                    f"model.layers.{layer_idx}.self_attention_res_norm.weight",
                    f"model.layers.{layer_idx}.self_attention_res_proj.weight",
                    f"model.layers.{layer_idx}.mlp_res_norm.weight",
                    f"model.layers.{layer_idx}.mlp_res_proj.weight",
                }
            )
        return required

    def _gated_norm_required_weight_names(self) -> set[str]:
        """The gate matrices this checkpoint must supply, for the gated reads ONLY.

        The same post-condition the CP and static mixes carry, for the same reason: a quantized load
        disables strict weight tracking and these `ReplicatedLinear` layers carry
        `quant_config=None`, so a gate the loader skipped would reach the first forward as
        uninitialised memory -- and `sigmoid` maps whatever it is fed into (0, 1), so the server
        would scale the residual stream by a plausible number and answer fluently. A gn-moe
        checkpoint has HALF these names, which is why the read set is consulted rather than assumed.
        """
        cfg = getattr(self, "config", None)
        if cfg is None:
            return set()
        if not bool(getattr(cfg, "attn_res_gated_norm", False)):
            return set()
        if getattr(cfg, "attn_res_block_size", None) is None:
            return set()
        reads = set(getattr(cfg, "attn_res_gated_norm_reads", None) or ("attn", "moe"))
        norms = [
            n
            for read, n in (
                ("attn", "input_layernorm"),
                ("moe", "post_attention_layernorm"),
            )
            if read in reads
        ]
        return {
            f"model.layers.{i}.{norm}.{mat}.weight"
            for i in range(cfg.num_hidden_layers)
            for norm in norms
            for mat in ("gate_down", "gate_up")
        }

    def _finalize_loaded_weights(self):
        # Prepare KDA and run native loader plugins before the Open3 checks below.
        # CP/GatedNorm finalization must wait until their weights are complete.
        super().post_load_weights()

    def post_load_weights(self) -> None:
        """Derive what the checkpoint stores factorised. SGLang's own post-load hook.

        The CP scorer keeps `E` and `Wq` on disk and attends through `A = E @ Wq / sqrt(qdim)`,
        which only exists once the weights are in. `load_weights` used to fold inline, which left
        every loader that bypasses it -- `loader.py`'s own list is "dummy / sharded state / remote
        instance / remote fs" -- serving with the score buffers still empty. The forward guard
        refuses that, so it surfaced as a server that would not start rather than as wrong routes,
        but the feature simply did not work on those paths.

        Idempotent: folding twice recomputes the same matrix from the same parameters.
        """
        self._check_served_erase()
        if not getattr(self.model, "use_cp_attn_res", False):
            return
        self._fold_cp_routers()
        self.model._cp_attn_res_loaded = True

    def _check_served_erase(self) -> None:
        """Serve the SDA erase the checkpoint was trained with (``config.exact_erase``).

        The FlashSDA plugin (``FLASHSDA_SGLANG=1``) wraps the parent's ``load_weights`` and
        ``post_load_weights`` and installs its prefill kernel there with the model config's mode.
        This class overrides ``post_load_weights``, so it calls the parent's first: loaders that
        bypass ``load_weights`` (dummy / sharded state / remote) then still reach the plugin. An
        exact-erase checkpoint is refused unless the adapter is installed with that mode -- stock
        KDA, or a FlashSDA older than the switch, would serve the bf16 erase without an error."""
        super().post_load_weights()
        self._refuse_unserved_exact_erase()

    def _refuse_unserved_exact_erase(self) -> None:
        """Refuse ``config.exact_erase=True`` unless the FlashSDA prefill kernel is installed exact.

        Both load paths reach it: ``post_load_weights`` (the loaders that bypass ``load_weights``,
        and the CP AttnRes fold inside ``load_weights``) and the end of the first ``load_weights``
        for every other model -- SGLang's ``DefaultModelLoader`` calls only ``load_weights``."""
        if not self.config.exact_erase:
            return
        if is_hip() and getattr(self.config, "linear_attn", None) == "sda":
            # Every SDA layer carries aiter's exact-erase prefill on ROCm; it has to exist.
            try:
                from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_sda import (  # noqa: F401
                    flash_sda_fwd,
                )
            except ImportError as err:
                raise RuntimeError(
                    "config.exact_erase=True needs aiter's flash_sda kernel on ROCm "
                    "(aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_sda), but it "
                    "is not importable in this worker; install an aiter that provides it"
                ) from err
            return
        try:
            from flash_sda.integrations import sglang_kernel
        except ImportError as err:
            raise RuntimeError(
                "config.exact_erase=True needs the FlashSDA SGLang plugin, but flash_sda is not "
                "importable in this worker; install FlashSDA and start with FLASHSDA_SGLANG=1"
            ) from err
        installed = getattr(sglang_kernel, "installed_exact_erase", None)
        if installed is None:
            raise RuntimeError(
                "config.exact_erase=True but the installed FlashSDA predates the exact-erase switch "
                "(FlashSDA #11); install the SDA_PIN of setup/run.sh"
            )
        if installed() is not True:
            raise RuntimeError(
                f"config.exact_erase=True but the FlashSDA prefill kernel is installed with "
                f"exact_erase={installed()} (None: not installed -- start with FLASHSDA_SGLANG=1); "
                "SGLang would serve this checkpoint with the bf16 erase"
            )

    def _fold_cp_routers(self) -> None:
        """Build each consumer's ``A = E @ Wq / sqrt(qdim)`` once, after the checkpoint is in.

        The trainer computes ``(Wq x) . e_i / sqrt(qdim)`` -- two matmuls per consumer. Folding the
        two constant factors gives the same function in ONE matmul against a matrix that is also
        much smaller than Wq at the shipped rank, which is what the fused kernel consumes.

        Done in fp32 and cast back: the product is a contraction over ``qdim`` and accumulating it
        in bf16 would lose precision the trained factors still carry. It is exact in exact
        arithmetic but NOT bitwise against the trainer's two-matmul order, so parity against the HF
        reference is a tolerance check rather than an equality one.
        """
        model = self.model
        scale = self.config.attn_res_scorer_dim**-0.5
        key = model.attn_res_key_embed.data.float()  # [n_src, qdim]
        for layer in model.layers:
            if not getattr(layer, "use_cp_attn_res", False):
                continue
            for owner in ("self_attention", "mlp"):
                wq = getattr(
                    layer, f"{owner}_res_query_proj"
                ).data.float()  # [qdim, dim]
                folded = getattr(layer, f"{owner}_res_score_weight")
                folded.copy_((key @ wq * scale).to(folded.dtype))
        out = model.output_attn_res_score_weight
        out.copy_(
            (key @ model.output_attn_res_query_proj.data.float() * scale).to(out.dtype)
        )

    def load_weights(self, weights):
        """The parent's loader, plus the AttnRes post-conditions.

        ⚠️ The parent is what actually loads -- solar_open3 adds no trunk parameters, so replacing
        its loader would be wrong. This only brackets it.
        """
        cp_was_loaded = getattr(self.model, "_cp_attn_res_loaded", False)
        if getattr(self.model, "use_cp_attn_res", False):
            # Loading can fail after a partial mutation. Invalidate FIRST so every failure is
            # fail-closed instead of leaving a partially refreshed router serviceable.
            self.model._cp_attn_res_loaded = False
        if self._gated_norm_required_weight_names():
            self.model._gated_norm_loaded = (
                False  # same rule for the gate (hancheolcho review)
            )

        # solar_open3 ships the MTP head under the `mtp.` prefix, and a separate draft model
        # loads those. The trunk loader must skip them: the parent would otherwise report them as
        # unexpected, or worse, try to place them into trunk modules that do not exist.
        # ⚠️ Do NOT take the loaded names from the parent's return value.
        # `SolarOpen2ForCausalLM.load_weights` does not return loaded names, so
        # `loaded` is always None. Guarding the checks below with `and names` therefore disabled
        # them entirely: CP went on to fold uninitialised `torch.empty` and mark itself loaded, and
        # the static mix reached its first forward on uninitialised memory. Both served fluent text.
        #
        # The names are recorded here instead, as the checkpoint offers them. That answers the
        # question these checks actually ask -- "does this checkpoint carry the AttnRes tensors?"
        # -- and it does not depend on the parent's bookkeeping.
        names: set[str] = set()

        def _seen(pairs):
            for args in pairs:
                name = args[0]
                if not name.startswith("mtp."):
                    names.add(name)
                    yield args

        try:
            loaded = super().load_weights(_seen(weights))
        except KeyError as exc:
            # The parent indexes `params_dict[name]` directly, so a checkpoint that carries a gate
            # this config did not build dies with the bare tensor name -- and the runtime update
            # path puts that message in its response (hancheolcho review F-35). Only OUR names
            # are re-raised with the cause; anything else stays the parent's failure.
            key = exc.args[0] if exc.args else ""
            if not (
                isinstance(key, str)
                and key.endswith((".gate_down.weight", ".gate_up.weight"))
            ):
                raise
            raise ValueError(
                f"checkpoint has the AttnRes GatedNorm matrix {key!r} but this model built no "
                "gate for it: config.json sets attn_res_gated_norm="
                f"{getattr(self.config, 'attn_res_gated_norm', None)!r} and "
                f"attn_res_gated_norm_reads={getattr(self.config, 'attn_res_gated_norm_reads', None)!r}. "
                "`reads` decides which norms own a gate, so both keys are properties of the "
                "checkpoint: set them to what it was trained with, or serve a checkpoint whose "
                "norms carry no gate."
            ) from exc
        if loaded is not None:
            names |= set(loaded)

        # ⚠️ Completeness is a question about the INITIAL load only. SGLang's streaming updaters
        # call `load_weights` once per chunk (`weight_updater.py`: distributed, bucketed, and the
        # checkpoint engine's per-bucket loader), and a chunk carries whatever names it was given.
        # Demanding the full set on every call raises on the first chunk, and because the
        # invalidation above has already run, the flag stays False and every later forward refuses
        # -- an engine that only a restart recovers. So the set is accumulated across calls and
        # enforced once.
        first_load = not getattr(self, "_solar_open3_loaded_once", False)
        self._solar_open3_loaded_once = True
        seen_all = getattr(self, "_solar_open3_seen_names", None)
        if seen_all is None:
            seen_all = self._solar_open3_seen_names = set()
        seen_all |= names

        # Gate completeness BEFORE the CP branch (hancheolcho review): it needs only `seen_all`,
        # and `post_load_weights()` below re-derives the score matrices and marks CP loaded. The
        # invalidation at the top of this method exists so a failed load stays fail-closed, and
        # raising after that fold would leave the routers derived from a load the caller was
        # told had failed.
        gate_missing = sorted(
            self._gated_norm_required_weight_names().difference(seen_all)
        )
        if gate_missing and first_load:
            raise ValueError(
                "AttnRes GatedNorm checkpoint loading did not populate every gate matrix. A gate "
                "left uninitialised still returns sigmoid(...) in (0, 1), so this would serve a "
                f"scaled residual stream with no error. Missing: {gate_missing}"
            )
        if not gate_missing:
            # Every gate matrix has been seen (this call or an earlier chunk): the forward guard
            # may serve again. A chunk that fails inside the parent never reaches this line.
            self.model._gated_norm_loaded = True

        if getattr(self.model, "use_cp_attn_res", False):
            required = self._cp_required_weight_names()
            missing = sorted(required.difference(seen_all))
            if missing and first_load:
                raise ValueError(
                    "CP Attention-Residuals checkpoint loading did not populate every query "
                    f"projection / key table. Missing: {missing}"
                )
            # Re-fold after a call that TOUCHED a CP tensor: the score buffers are derived, so an
            # update that refreshes `E` or `Wq` leaves them holding the previous fold. A chunk that
            # carried none of them changed nothing to fold, and folding again would walk every
            # layer for no reason.
            if not missing:
                if first_load or names & required or not cp_was_loaded:
                    self.post_load_weights()
                else:
                    self.model._cp_attn_res_loaded = True
        elif getattr(self.model, "use_attn_res", False):
            # The same explicit check for the STATIC scorer. ⚠️ A quantized load disables strict
            # weight tracking and the static mix's ReplicatedLinear layers carry
            # `quant_config=None`, so an un-loaded ``*_res_proj.weight`` would reach the first
            # forward as uninitialised memory.
            missing = sorted(self._static_required_weight_names().difference(seen_all))
            if missing and first_load:
                raise ValueError(
                    "static Attention-Residuals checkpoint loading did not populate every "
                    f"res-norm / res-proj tensor. Missing: {missing}"
                )
        if first_load and not getattr(self.model, "use_cp_attn_res", False):
            # The CP branch above refuses through `post_load_weights()`; every other model gets
            # the check here, after the parent's (plugin-wrapped) loader has installed the kernel.
            self._refuse_unserved_exact_erase()
        return loaded


def _require_registration() -> None:
    """Require the native config registration before constructing recurrent layers."""
    from sglang.srt.configs.linear_attn_model_registry import (
        get_linear_attn_spec_by_arch,
    )

    if get_linear_attn_spec_by_arch("SolarOpen3ForCausalLM") is None:
        raise RuntimeError(
            "SolarOpen3ForCausalLM has no hybrid-attention specification. "
            "Import sglang.srt.configs.solar_open3 before resolving server arguments."
        )


def _refuse_unimplemented(config) -> None:
    """Refuse the two solar_open3 axes this adapter does not serve, by name.

    Both are OFF in every arm served today, and both change the trunk rather than an option:
    silently ignoring either would answer with a different model.
    """
    # Attention-Residuals IS implemented here: the fused kernels are carried in routed_core.py and
    # the depth-mix path is `SolarOpen3DecoderLayer.forward_attn_res` plus the model-level branch.
    # Both scorers are served. The `cp` routers are DERIVED after load, which `post_load_weights`
    # does; the forward guard refuses a forward that reaches it before that fold.
    if getattr(config, "attn_res_block_size", None) is not None:
        _validate_scorer_config(config)
    # GatedNorm IS served: `SolarOpen3GatedRMSNorm` replaces the prenorm of each read named by
    # `attn_res_gated_norm_reads`, and the depth-mix kernels keep applying the RMSNorm half. The one
    # shape to refuse is the gate WITHOUT AttnRes -- there is no read to gate. The trainer and the
    # HF config reject that too, so no such checkpoint can be exported; this is the serving-side
    # copy of that rule.
    if (
        bool(getattr(config, "attn_res_gated_norm", False))
        and getattr(config, "attn_res_block_size", None) is None
    ):
        raise NotImplementedError(
            "solar_open3 SGLang: attn_res_gated_norm=True without AttnRes has no read to gate"
        )
    # ...and refuse the gate under FlashRL's reloading FP8 loader (Codex review, high). Its
    # `SKIP_QUANTIZATION_PARAMS` is a substring list of EXACT names ("model.norm.weight", ".bias",
    # ...); nothing in it matches `...post_attention_layernorm.gate_down.weight`, so a reload
    # quantizes the gate matrices, and the unquantized ReplicatedLinear loader then copies fp8
    # qvalues into a bf16 parameter and never applies the separately stored scale. The gate would
    # be silently wrong -- sigmoid maps any garbage into (0, 1) and the server keeps answering.
    if bool(getattr(config, "attn_res_gated_norm", False)):
        try:
            load_format = getattr(runtime_options(), "load_format", None)
        except Exception:  # noqa: BLE001 -- no server args outside a running server
            load_format = None
        if load_format == "flash_rl":
            raise NotImplementedError(
                "solar_open3 SGLang: --load-format flash_rl quantizes every weight whose name its "
                "skip-list does not match, and the GatedNorm gate matrices "
                "(*.gate_down.weight / *.gate_up.weight) are not on that list. An RL reload would "
                "write fp8 values into the bf16 gate and drop the scale, which serves a wrong gate "
                "with no error. Serve gated checkpoints with the default loader."
            )
    # mtp_num_hidden_layers > 0 is SERVED: solar_open3_mtp.py registers SolarOpen3ForCausalLMNextN
    # as an SGLang draft model, and this trunk loader skips the `mtp.*` half so the two read the
    # same checkpoint without colliding. Speculative decoding is opt-in at the server:
    #   --speculative-algorithm EAGLE --speculative-draft-model-path <draft dir>
    # Without those flags the head is simply not loaded and the trunk serves on its own, which is
    # why carrying one is no longer a reason to refuse.


EntryClass = SolarOpen3ForCausalLM
