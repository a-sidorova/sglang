# Copyright 2026 Upstage AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Native SolarOpen3 configuration and hybrid-attention registration."""

from __future__ import annotations

from typing import List, Optional

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.runtime_context import get_parallel

# The inherited solar_open2 trunk reads these as plain attributes, and exports no longer write them
# (the converter refuses them on), so they are set to the one value served. An old config.json
# carries exactly these values; every other dropped key is read by neither the fork nor this adapter.
SERVED_FIXED_OPTIONS = {
    "use_rope": False,  # SolarOpen2Attention's rotary embedding (+ rope_theta/_scaling)
    "use_gqa_gate_bias": False,  # the GQA output gate's g_proj bias
    "kda_use_full_proj": False,  # SolarOpen3KimiDeltaAttention's low-rank vs full f/g projections
    "n_group": 1,  # the MoE's grouped top-k, and the EPLB expert-location config
    "topk_group": 1,
    "tie_word_embeddings": False,  # SGLang's loader; transformers 5 no longer defaults it
}


def pin_served_options(config):
    """Set ``SERVED_FIXED_OPTIONS`` on ``config``, whichever class carries it (idempotent)."""
    for key, value in SERVED_FIXED_OPTIONS.items():
        setattr(config, key, value)
    return config


# `exact_erase`'s default: an export from before the converter wrote the key, told apart from an explicit
# value (JSON `null` is not one and is refused).
_UNWRITTEN = object()
FP32_PARAM_KINDS = (
    "A_log",
    "attn_res_key_embed",
    "dt_bias",
    "norm_gain",
    "output_attn_res_query_proj",
)


def validate_precision_config(config) -> None:
    """Validate checkpoint precision policy for native and remote configs."""
    value = getattr(config, "attn_res_key_fp32", False)
    if not isinstance(value, bool):
        raise ValueError(f"attn_res_key_fp32 must be a JSON boolean, got {value!r}")
    kinds = getattr(config, "fp32_params", None)
    if kinds is not None and (
        not isinstance(kinds, (list, tuple))
        or any(
            not isinstance(kind, str) or kind not in FP32_PARAM_KINDS for kind in kinds
        )
    ):
        raise ValueError(
            f"fp32_params must be a list of {list(FP32_PARAM_KINDS)} or None, got {kinds!r}"
        )


def _patch_config(config):
    """Ensure HF-loaded config has required methods/properties for sglang."""
    if hasattr(config, "is_kda_layer") and hasattr(config, "mamba2_cache_params"):
        return config

    # Compute layer lists once
    num_layers = config.num_hidden_layers
    gqa_layers = getattr(config, "gqa_layers", None)
    gqa_interval = getattr(config, "gqa_interval", 3)

    kda_layer_set = set()
    gqa_layer_set = set()
    for i in range(num_layers):
        if gqa_layers is not None:
            is_gqa = i in gqa_layers
        else:
            is_gqa = (i + 1) % gqa_interval == 0
        if is_gqa:
            gqa_layer_set.add(i)
        else:
            kda_layer_set.add(i)

    config._kda_layer_set = kda_layer_set
    config._linear_layer_ids = sorted(kda_layer_set)
    config._full_attention_layer_ids = sorted(gqa_layer_set)

    def is_kda_layer(self, layer_idx):
        return layer_idx in self._kda_layer_set

    @property
    def linear_layer_ids(self):
        return self._linear_layer_ids

    @property
    def full_attention_layer_ids(self):
        return self._full_attention_layer_ids

    @property
    def is_linear_attn(self):
        return len(self._linear_layer_ids) > 0

    @property
    def mamba2_cache_params(self):
        lac = getattr(self, "linear_attn_config", None) or {}
        num_heads = lac.get("num_heads", self.num_attention_heads)
        head_dim = lac.get(
            "head_dim",
            getattr(self, "head_dim", self.hidden_size // self.num_attention_heads),
        )
        conv_kernel_size = lac.get("short_conv_kernel_size", 4)

        shape = KimiLinearStateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_kernel_size=conv_kernel_size,
        )
        return KimiLinearCacheParams(shape=shape, layers=self._linear_layer_ids)

    # Monkey-patch the config instance's class
    config_cls = type(config)
    config_cls.is_kda_layer = is_kda_layer
    config_cls.linear_layer_ids = linear_layer_ids
    config_cls.full_attention_layer_ids = full_attention_layer_ids
    config_cls.is_linear_attn = is_linear_attn
    config_cls.mamba2_cache_params = mamba2_cache_params
    return config


def _is_gqa_layer(config, layer_idx: int) -> bool:
    """Check if a layer is GQA (not KDA). Works with any config object."""
    gqa_layers = getattr(config, "gqa_layers", None)
    if gqa_layers is not None:
        return layer_idx in gqa_layers
    gqa_interval = getattr(config, "gqa_interval", 3)
    return (layer_idx + 1) % gqa_interval == 0


def _is_kda_layer(config, layer_idx: int) -> bool:
    return not _is_gqa_layer(config, layer_idx)


class SolarOpen3Config(PretrainedConfig):
    model_type = "solar_open3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="solar_open3",
        vocab_size=196608,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=10240,
        moe_intermediate_size: int = 1280,
        num_hidden_layers=48,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=2,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=1048576,
        # GQA. `gqa_layers` is the ONLY layer-type source (`patch_config` refuses a config without
        # it); the interval fallback the solar_open2 trunk still carries is never reached.
        gqa_layers: Optional[List[int]] = None,
        use_gqa_gate: bool = True,
        # KDA
        linear_attn_config: Optional[dict] = None,
        kda_allow_neg_eigval: bool = True,
        # MoE
        n_routed_experts: Optional[int] = 128,
        n_shared_experts: Optional[int] = 1,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        # LSSA -- subsampled keys, and the routed top-k on top of them. Every one of these is
        # exported explicitly by the converter, so "absent" and "null" both mean unset.
        kv_subsample_layers: Optional[List[int]] = None,
        kv_subsample_stride: Optional[int] = None,
        kv_subsample_pool: Optional[str] = None,
        kv_route_layers: Optional[List[int]] = None,
        kv_route_topk: Optional[int] = None,
        kv_route_block: Optional[int] = None,
        kv_route_q_block: Optional[int] = None,
        kv_route_score_block: Optional[int] = None,
        kv_route_reduce: Optional[str] = None,
        kv_route_head_reduce: Optional[str] = None,
        kv_route_head_shared: Optional[bool] = None,
        kv_route_qhead_block: Optional[int] = None,
        kv_route_per_row: Optional[bool] = None,
        kv_anchor_in_attention: Optional[bool] = None,
        kv_route_dedup_anchor: Optional[bool] = None,
        # Token span of every document's head forced into the routed selection (the trainer's
        # key, passed through config.json verbatim; null/absent = 0 = the pre-sink ranking).
        # Declared like the keys above so it never rides `**kwargs`.
        kv_route_sink_tokens: Optional[int] = None,
        kv_route_fused_prefill: Optional[bool] = None,
        # Depth attention-residuals and the MTP draft head. Both are OFF in every arm this
        # adapter serves; they are named so a checkpoint that turns one on is refused by
        # ``solar_open3.py`` rather than served as a different model.
        attn_res_block_size: Optional[int] = None,
        attn_res_scorer: str = "static",
        attn_res_scorer_dim: int = 0,
        attn_res_scale_by_sources: bool = False,
        # Preserve the parameter kinds and folded CP scores used in FP32 during training.
        attn_res_key_fp32: bool = False,
        fp32_params: Optional[List[str]] = None,
        # GatedNorm on the AttnRes reads (PR #1374). Declared for the same reason as the four
        # keys above: a key that rides `**kwargs` arrives as raw JSON, and `getattr(cfg, k, None)`
        # then cannot tell JSON `null` from "absent" the way the trainer/HF readers do.
        attn_res_gated_norm: bool = False,
        attn_res_gated_norm_dim: Optional[int] = None,
        attn_res_gated_norm_reads: Optional[list] = None,
        mtp_num_hidden_layers: int = 0,
        # The trunk hidden the MTP draft reads at step 0: False = post-final-norm, True = before it.
        # Declared so it never rides `**kwargs`: there JSON `null` and the string "false" would
        # reach the adapter as raw values and be served as the wrong hidden. Checked below.
        mtp_hidden_prenorm: bool = False,
        # The SDA erase the linear-attention weights were trained with (FlashSDA ``exact_erase``,
        # written by the converter; absent = False, the bf16 erase). The FlashSDA plugin reads it at
        # model load and serves that erase. Declared and checked like mtp_hidden_prenorm. Absent is
        # recorded in `exact_erase_written`: such an export may still be exact-erase trained.
        exact_erase: bool = _UNWRITTEN,
        # Clamp / clip bounds added to the HF config after the other clamp fields; declared so no
        # new HF field rides `**kwargs` (the field-parity test). solar_open3.py serves them as it
        # does the older `swiglu_limit` family that still arrives through `**kwargs`.
        mtp_shared_expert_swiglu_limit=None,
        routed_out_clip: Optional[float] = None,
        shared_out_clip: Optional[float] = None,
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings

        self.gqa_layers = gqa_layers
        self.use_gqa_gate = use_gqa_gate

        self.linear_attn_config = linear_attn_config
        self.kda_allow_neg_eigval = kda_allow_neg_eigval

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace

        self.kv_subsample_layers = kv_subsample_layers
        self.kv_subsample_stride = kv_subsample_stride
        self.kv_subsample_pool = kv_subsample_pool
        self.kv_route_layers = kv_route_layers
        self.kv_route_topk = kv_route_topk
        self.kv_route_block = kv_route_block
        self.kv_route_q_block = kv_route_q_block
        self.kv_route_score_block = kv_route_score_block
        self.kv_route_reduce = kv_route_reduce
        self.kv_route_head_reduce = kv_route_head_reduce
        self.kv_route_head_shared = kv_route_head_shared
        self.kv_route_qhead_block = kv_route_qhead_block
        self.kv_route_per_row = kv_route_per_row
        self.kv_anchor_in_attention = kv_anchor_in_attention
        self.kv_route_dedup_anchor = kv_route_dedup_anchor
        self.kv_route_sink_tokens = kv_route_sink_tokens
        self.kv_route_fused_prefill = kv_route_fused_prefill

        self.attn_res_block_size = attn_res_block_size
        self.attn_res_scorer = attn_res_scorer
        self.attn_res_scorer_dim = attn_res_scorer_dim
        self.attn_res_scale_by_sources = attn_res_scale_by_sources
        self.attn_res_key_fp32 = attn_res_key_fp32
        self.fp32_params = fp32_params
        validate_precision_config(self)
        self.attn_res_gated_norm = attn_res_gated_norm
        self.attn_res_gated_norm_dim = attn_res_gated_norm_dim
        self.attn_res_gated_norm_reads = attn_res_gated_norm_reads
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        if not isinstance(mtp_hidden_prenorm, bool):
            raise ValueError(
                f"mtp_hidden_prenorm must be a JSON boolean (true/false), got {mtp_hidden_prenorm!r} "
                f"({type(mtp_hidden_prenorm).__name__}); it selects which trunk hidden the MTP draft "
                "reads, so it must be written as the training config had it"
            )
        if mtp_hidden_prenorm and mtp_num_hidden_layers <= 0:
            raise ValueError(
                "mtp_hidden_prenorm=true requires mtp_num_hidden_layers > 0: this config "
                "builds no MTP head"
            )
        self.mtp_hidden_prenorm = mtp_hidden_prenorm
        self.exact_erase_written = exact_erase is not _UNWRITTEN
        if not self.exact_erase_written:
            exact_erase = False
        if not isinstance(exact_erase, bool):
            raise ValueError(
                f"exact_erase must be a JSON boolean (true/false), got {exact_erase!r} "
                f"({type(exact_erase).__name__}); it selects the SDA erase the weights were trained "
                "with, so it must be written as the training config had it"
            )
        self.exact_erase = exact_erase
        self.mtp_shared_expert_swiglu_limit = mtp_shared_expert_swiglu_limit
        self.routed_out_clip = routed_out_clip
        self.shared_out_clip = shared_out_clip

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # After `super().__init__` (which sets an old export's leftover keys): SGLang reads this
        # class before any model is built. `gqa_layers` is checked in `patch_config` instead.
        pin_served_options(self)
        # SGLang sizes the draft KV pool from this; unset, the draft counts as the whole trunk.
        if "SolarOpen3ForCausalLMNextN" in (self.architectures or []):
            if mtp_num_hidden_layers < 1:
                raise ValueError(
                    "a SolarOpen3ForCausalLMNextN draft config needs mtp_num_hidden_layers "
                    f">= 1, got {mtp_num_hidden_layers}: the target has no MTP head"
                )
            self.num_nextn_predict_layers = mtp_num_hidden_layers

    def _is_gqa_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.gqa_layers

    def is_kda_layer(self, layer_idx: int) -> bool:
        return not self._is_gqa_layer(layer_idx)

    @property
    def linear_layer_ids(self) -> List[int]:
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def full_attention_layer_ids(self) -> List[int]:
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def is_linear_attn(self) -> bool:
        return len(self.linear_layer_ids) > 0

    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        lac = self.linear_attn_config or {}
        num_heads = lac.get("num_heads", self.num_attention_heads)
        head_dim = lac.get("head_dim", self.head_dim)
        conv_kernel_size = lac.get("short_conv_kernel_size", 4)
        shape = KimiLinearStateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_kernel_size=conv_kernel_size,
        )
        from sglang.srt.models.solar_open3 import with_linear_draft_layer

        # The linear MTP head's state layer rides in this list when one is served; see
        # `solar_open3.with_linear_draft_layer`.
        return with_linear_draft_layer(
            KimiLinearCacheParams(shape=shape, layers=self.linear_layer_ids), self
        )


from sglang.srt.configs.linear_attn_model_registry import (  # noqa: E402
    LinearAttnModelSpec,
    get_linear_attn_spec_by_arch,
    register_linear_attn_model,
)


def _is_solar_open3_config(config) -> bool:
    """Accept a solar_open3 config, whichever class carries it.

    The converter exports an ``auto_map`` naming the checkpoint's own
    ``configuration_solar_open3.SolarOpen3Config``, so HF hands sglang THAT class and an
    ``isinstance`` against the one registered here fails. The registry falls back to matching
    ``arch_names``, and this predicate then normalises the foreign instance the same way
    ``solar_open2`` does -- by giving it the layer-id and cache properties sglang reads.
    """
    if getattr(config, "model_type", None) != "solar_open3" and not isinstance(
        config, SolarOpen3Config
    ):
        return False
    from sglang.srt.models.solar_open3 import patch_config

    # Unconditionally, on BOTH classes: the patch is idempotent, and skipping it for the class
    # registered here left the linear head's state layer unreserved on that path -- which the pool
    # builder then honoured by giving the head a private pool, with nothing in the log but one
    # line saying so (measured, job 110656).
    patch_config(config)
    return True


def _draft_backend_wrapper(runner, backend, phase: str):
    """The fork's draft hook, resolved lazily so importing this module stays light."""
    from sglang.srt.layers.attention.solar_open3.linear_draft import (
        draft_backend_wrapper,
    )

    return draft_backend_wrapper(runner, backend, phase)


# The hook is a field the in-house fork added; a fork without it still serves the trunk, so its
# absence must not break registration -- only the (linear) MTP head refuses it, by name, later.
_SPEC_EXTRAS = (
    {"draft_backend_wrapper": _draft_backend_wrapper}
    if "draft_backend_wrapper"
    in getattr(LinearAttnModelSpec, "__dataclass_fields__", {})
    else {}
)


def _matches_runner_config(config, *, draft):
    is_draft = "SolarOpen3ForCausalLMNextN" in (
        getattr(config, "architectures", None) or []
    )
    return is_draft == draft and _is_solar_open3_config(config)


def _mamba_cache_alignment(config) -> int:
    # A cached prefix must contain whole routed query blocks: the election for
    # a partial block would otherwise lose the cached rows' query votes. Keep
    # this separate from mamba_chunk_size, which indexes the kernel's states.
    if not getattr(config, "kv_route_layers", None):
        return 1
    block = getattr(config, "kv_route_q_block", None)
    if block is None:
        return 256
    if isinstance(block, bool) or not isinstance(block, int) or block < 1:
        raise ValueError(f"kv_route_q_block={block!r} must be a positive integer")
    return block


# The initial draft runner also goes through SGLang's generic attention factory.
# It labels its full-attention layer [0], which is unrelated to the trunk's GQA
# layer IDs. Only the target may use the trunk-only state-commit wrapper. The
# dedicated draft hook below still supplies the linear head's real backends.
for _arch, _draft in (
    ("SolarOpen3ForCausalLM", False),
    ("SolarOpen3ForCausalLMNextN", True),
):
    if get_linear_attn_spec_by_arch(_arch) is None:
        register_linear_attn_model(
            LinearAttnModelSpec(
                config_class=SolarOpen3Config,
                backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
                hybrid_backend_class_name=(
                    None
                    if _draft
                    else "sglang.srt.layers.attention.solar_open3.linear_draft.SolarOpen3HybridAttnBackend"
                ),
                arch_names=[_arch],
                uses_mamba_radix_cache=True,
                support_mamba_cache=True,
                support_mamba_cache_extra_buffer=True,
                config_predicate=lambda config, draft=_draft: _matches_runner_config(
                    config, draft=draft
                ),
                mamba_cache_alignment=_mamba_cache_alignment,
                **_SPEC_EXTRAS,
            )
        )

try:  # pragma: no cover - depends on HF import order
    from transformers import AutoConfig

    AutoConfig.register("solar_open3", SolarOpen3Config)
except ValueError:
    # AutoConfig may already contain this native class during repeated imports.
    pass
