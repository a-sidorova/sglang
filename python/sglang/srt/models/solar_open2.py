# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 Upstage AI.
# Adapted from vLLM's inference-only Solar Open2 implementation.

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.solar_open2 import SolarOpen2Config
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.layer_boundary import append_stages, declare_attn, declare_ffn
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelBatchedLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedColumnParallelRepeatedLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.solar_open2 import (
    SolarOpen2FusedRouter,
    SolarOpen2RouterGate,
)
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.kimi_linear import (
    KimiDecoderLayer,
    KimiDeltaAttention,
    KimiLinearForCausalLM,
    KimiLinearModel,
)
from sglang.srt.models.solar import SolarMLP as SolarOpen2MLP
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import (
    add_prefix,
    get_bool_env_var,
    is_cuda_alike,
    is_hip,
    make_layers,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


def _apply_gqa_output_gate(output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Preserve the BF16 sigmoid rounding when fusing the output gate."""
    if not torch.compiler.is_compiling() and output.is_cuda and not is_hip():
        from sglang.kernels.ops.attention import mla_output_gate

        if mla_output_gate.covered(output, gate):
            return mla_output_gate.kimi_k3_mla_output_gate(output, gate)
    return output * gate.sigmoid()


class SolarOpen2Attention(nn.Module):
    def __init__(
        self,
        config: SolarOpen2Config,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        reduce_results: bool = True,
        allow_qkvg_fusion: bool = True,
    ):
        super().__init__()
        parallel = get_parallel()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attn_tp_size, self.attn_tp_rank = (
            parallel.attn_tp_size,
            parallel.attn_tp_rank,
        )
        tp_size = self.attn_tp_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.head_dim = config.head_dim
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.scaling = self.head_dim**-0.5
        attention_bias = getattr(config, "attention_bias", False)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            parallel_group="attn_tp",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            parallel_group="attn_tp",
            reduce_results=reduce_results,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = None
        if config.use_rope:
            rope_theta, rope_scaling = get_rope_config(config)
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=int(
                    self.head_dim
                    * (getattr(config, "rope_parameters", None) or {}).get(
                        "partial_rotary_factor", 0.5
                    )
                ),
                max_position=config.max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
        self.use_gqa_gate = config.use_gqa_gate
        if self.use_gqa_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                config.num_attention_heads * self.head_dim,
                bias=config.use_gqa_gate_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
                parallel_group="attn_tp",
            )
        self.fused_qkvg = (
            allow_qkvg_fusion
            and get_bool_env_var("SGLANG_SOLAR_OPEN2_DENSE_FUSION")
            and self.use_gqa_gate
            and config.num_key_value_heads >= tp_size
            and not attention_bias
            and not config.use_gqa_gate_bias
            and isinstance(self.qkv_proj.quant_method, UnquantizedLinearMethod)
            and isinstance(self.g_proj.quant_method, UnquantizedLinearMethod)
        )
        if self.fused_qkvg:
            self.qkvg_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [
                    config.num_attention_heads * self.head_dim,
                    config.num_key_value_heads * self.head_dim,
                    config.num_key_value_heads * self.head_dim,
                    config.num_attention_heads * self.head_dim,
                ],
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.qkvg_proj",
                parallel_group="attn_tp",
            )
            del self.qkv_proj, self.g_proj
        self.zero_copy_qkv = _use_aiter and get_bool_env_var(
            "SGLANG_SOLAR_OPEN2_ZERO_COPY_QKV"
        )

    def project_qkv(self, hidden_states):
        """Project QKV, returning the output gate only when it is fused."""
        if self.fused_qkvg:
            qkvg, _ = self.qkvg_proj(hidden_states)
            return qkvg.split([self.q_size + 2 * self.kv_size, self.q_size], dim=-1)
        qkv, _ = self.qkv_proj(hidden_states)
        return qkv, None

    def forward(self, hidden_states, positions, forward_batch, **kwargs):
        qkv, gate = self.project_qkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if not self.zero_copy_qkv:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if self.use_qk_norm:
            q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(
                q.shape
            )
            k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(
                k.shape
            )
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v, forward_batch)
        if self.use_gqa_gate:
            if not self.fused_qkvg:
                gate, _ = self.g_proj(hidden_states)
            output = _apply_gqa_output_gate(output, gate)
        return self.o_proj(output)[0]


class SolarOpen2KimiDeltaAttention(KimiDeltaAttention):
    beta_in_fp32 = False
    round_sda_prefill_beta = False

    def __init__(
        self,
        config,
        layer_idx,
        quant_config=None,
        prefix="",
        *,
        rms_norm_eps=1e-5,
        reduce_results: bool = True,
    ):
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            no_kda_lora=config.kda_use_full_proj,
            shard_on_attn_tp=True,
            rms_norm_eps=rms_norm_eps,
            reduce_results=reduce_results,
        )
        self.attn.beta_scale = 2.0 if config.kda_allow_neg_eigval else 1.0
        # HF's dtype describes the checkpoint; the loader's default dtype owns
        # execution. The factored BMM must match its preceding projection.
        if self.do_fuse_qkvbfg and not self.no_kda_lora:
            self.fused_fg_b_proj.to(dtype=self.fused_qkvbfg_a_proj.weight.dtype)
        if get_bool_env_var("SGLANG_SOLAR_OPEN2_DENSE_FUSION"):
            self._fuse_unquantized_projections(config)

    def _fuse_unquantized_projections(self, config):
        # A globally quantized MoE checkpoint can still exclude all attention
        # projections. Inspect the resolved methods before using BF16 packing.
        if self.do_fuse_qkvbfg:
            if not self.no_kda_lora or self.fuse_no_lora_beta:
                return
            modules = [self.fused_qkvbfg_proj, self.b_proj]
        else:
            modules = [
                self.qkv_proj,
                self.b_proj,
                self.f_a_proj,
                self.g_a_proj,
                self.f_b_proj,
                self.g_b_proj,
            ]
        if not all(
            isinstance(m.quant_method, UnquantizedLinearMethod) for m in modules
        ):
            return
        projection = self.head_dim * self.num_heads
        local = projection // self.shard_tp_size
        if self.no_kda_lora:
            self.qkvbfg_sizes = [projection] * 3 + [self.num_heads] + [projection] * 2
            self.fused_qkvbfg_proj = MergedColumnParallelLinear(
                self.hidden_size,
                self.qkvbfg_sizes,
                bias=False,
                quant_config=None,
                prefix=f"{self.prefix}.fused_qkvbfg_proj",
                parallel_group="attn_tp",
            )
            self.split_sizes = [3 * local, self.local_num_heads, local, local]
            self.fuse_no_lora_beta = True
            del self.b_proj
        else:
            self.qkvb_sizes = [projection] * 3 + [self.num_heads]
            self.fg_sizes = [self.head_dim] * 2
            self.fused_qkvbfg_a_proj = MergedColumnParallelRepeatedLinear(
                self.hidden_size,
                self.qkvb_sizes,
                self.fg_sizes,
                quant_config=None,
                prefix=f"{self.prefix}.fused_qkvbfg_a_proj",
                parallel_group="attn_tp",
            )
            self.split_sizes = [3 * local, self.local_num_heads, 2 * self.head_dim]
            self.fused_fg_b_proj = ColumnParallelBatchedLinear(
                2,
                self.head_dim,
                projection,
                dtype=self.fused_qkvbfg_a_proj.weight.dtype,
                parallel_group="attn_tp",
            )
            del self.qkv_proj, self.b_proj, self.f_a_proj, self.g_a_proj
            del self.f_b_proj, self.g_b_proj
        self.do_fuse_qkvbfg = True

    def prepare_fused_decode(self):
        from sglang.kernels.ops.attention import solar_open2_kda

        solar_open2_kda.stage_weights(self)

    def forward(self, hidden_states, positions, forward_batch, **kwargs):
        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, norm_gate = self.forward_qkvbfg_fused(
                hidden_states
            )
        else:
            mixed_qkv, beta, forget_gate, norm_gate = self.forward_qkvbfg(hidden_states)
        if self.beta_in_fp32:
            beta = beta.float()
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            forget_gate = forget_gate.unflatten(-1, (-1, self.head_dim)).unsqueeze(0)
            beta = beta.float().sigmoid() * self.attn.beta_scale
            if self.round_sda_prefill_beta:
                beta = beta.to(mixed_qkv.dtype)
        fused_decode = (
            forward_batch.forward_mode.is_decode()
            and getattr(self.attn, "_solar_fused_decode_args", None) is not None
        )
        if fused_decode:
            self.attn._solar_onorm_gate = norm_gate
            self.attn._solar_onorm_consumed = False
        try:
            output = self.attn(
                forward_batch,
                mixed_qkv=mixed_qkv,
                a=forget_gate,
                b=beta.unsqueeze(0),
            )
        finally:
            if fused_decode:
                self.attn._solar_onorm_gate = None
        if not (fused_decode and self.attn._solar_onorm_consumed):
            output = self.o_norm(output, norm_gate.unflatten(-1, (-1, self.head_dim)))
        return self.o_proj(output.squeeze(0).flatten(-2))[0]


def _load_kda_weight(name, weight, attention, params):
    """Pack checkpoint KDA projections and convs into their live parameters.

    Return the destination name, or None for non-packed parameters. Both Solar
    loaders use the same mapping, including quantization scales and RL updates.
    """
    if attention.no_kda_lora:
        mapping = [(".fused_qkvbfg_proj", f".{p}_proj", i) for i, p in enumerate("qkv")]
        if attention.fuse_no_lora_beta:
            mapping.append((".fused_qkvbfg_proj", ".b_proj", 3))
        offset = 4 if attention.fuse_no_lora_beta else 3
        mapping.extend(
            [
                (".fused_qkvbfg_proj", f".{p}_proj", offset + i)
                for i, p in enumerate("fg")
            ]
        )
    elif attention.do_fuse_qkvbfg:
        mapping = [
            (".fused_qkvbfg_a_proj", f".{p}_proj", i)
            for i, p in enumerate(("q", "k", "v", "b", "f_a", "g_a"))
        ] + [(".fused_fg_b_proj", f".{p}_b_proj", i) for i, p in enumerate("fg")]
    else:
        mapping = [(".qkv_proj", f".{p}_proj", p) for p in "qkv"]
    mapping.extend((".qkv_conv1d", f".{p}_conv1d", i) for i, p in enumerate("qkv"))
    for target_module, source, shard_id in mapping:
        target = name.replace(source, target_module)
        if source in name and target in params:
            param = params[target]
            param.weight_loader(param, weight, shard_id)
            return target
    return None


class SolarOpen2MoE(nn.Module):
    # Open3 keeps FP32 routing and its standard runner's scaling boundary.
    _use_open2_optimizations = True
    _scale_in_experts = False

    def __init__(
        self, config, layer_idx, quant_config=None, prefix="", alt_stream=None
    ):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream
        use_deepep = get_moe_a2a_backend().is_deepep()
        self._scale_in_experts = not self._use_open2_optimizations and not use_deepep
        self.gate = SolarOpen2RouterGate(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
            allow_bf16_weight=self._use_open2_optimizations,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        self.num_fused_shared_experts = 0
        if (
            self._use_open2_optimizations
            and _use_aiter
            and not use_deepep
            and get_bool_env_var("SGLANG_SOLAR_OPEN2_FUSE_SHARED_EXPERTS")
            and quant_config is None
            and get_parallel().moe_ep_size == 1
        ):
            self.num_fused_shared_experts = config.n_shared_experts or 0
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.n_routed_experts + self.num_fused_shared_experts,
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_idx,
            quant_config=quant_config,
            activation=config.hidden_act,
            # Standard Open3 scales before the runner's BF16 cast. DeepEP
            # runners may ignore this factor, so scale after combine instead.
            routed_scaling_factor=(
                self.routed_scaling_factor if self._scale_in_experts else 1.0
            ),
            prefix=f"{prefix}.experts",
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_id=layer_idx,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )
        self.shared_experts = None
        if config.n_shared_experts and not self.num_fused_shared_experts:
            self.shared_experts = SolarOpen2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                # DeepEP combines routed experts across EP ranks. Shared experts
                # must be complete locally because no TP reduction follows it.
                **(dict(parallel_group="replicated") if use_deepep else {}),
            )
        self.fused_router = (
            SolarOpen2FusedRouter(self.experts, config)
            if self._use_open2_optimizations
            else None
        )

    def _forward_routed(self, hidden_states, forward_batch=None):
        if hidden_states.shape[0] == 0:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        else:
            router_logits = self.gate(hidden_states)[0]
            fused_router = getattr(self, "fused_router", None)
            if fused_router is not None and forward_batch is None:
                output = fused_router(
                    hidden_states, router_logits, self.gate.e_score_correction_bias
                )
                if output is not None:
                    # The AITER fused router already applies routed scaling.
                    return output
            routing_kwargs = {}
            if forward_batch is not None:
                routing_kwargs = dict(
                    num_token_non_padded=forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_idx
                    ),
                )
            topk_output = self.topk(hidden_states, router_logits, **routing_kwargs)
        output = self.experts(hidden_states, topk_output)
        if (
            not self._scale_in_experts
            and self.routed_scaling_factor is not None
            and self.routed_scaling_factor != 1.0
            and not (
                self.experts.should_fuse_routed_scaling_factor_in_topk or _use_aiter
            )
        ):
            output = output * self.routed_scaling_factor
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if get_moe_a2a_backend().is_deepep():
            return self._forward_deepep(hidden_states, forward_batch)

        shared_output = None
        alt_stream = getattr(self, "alt_stream", None)
        if (
            alt_stream is not None
            and self.shared_experts is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            current_stream = torch.cuda.current_stream()
            alt_stream.wait_stream(current_stream)
            shared_output = self.shared_experts(hidden_states.clone())
            with torch.cuda.stream(alt_stream):
                output = self._forward_routed(hidden_states)
            current_stream.wait_stream(alt_stream)
        else:
            if self.shared_experts is not None and hidden_states.shape[0] > 0:
                # FusedMoE may overwrite its input; evaluate the shared branch first.
                shared_output = self.shared_experts(hidden_states)
            output = self._forward_routed(hidden_states)
        if shared_output is not None:
            output = output + shared_output
        if self.tp_size > 1 and not use_reduce_scatter:
            output = tensor_model_parallel_all_reduce(output)
        return output

    def _forward_deepep(self, hidden_states, forward_batch):
        if hidden_states.shape[0] > 0 and forward_batch is None:
            raise ValueError("DeepEP MoE requires a ForwardBatch for token dispatch")
        shared_output = None
        if self.shared_experts is not None and hidden_states.shape[0] > 0:
            shared_output = self.shared_experts(hidden_states)
        output = self._forward_routed(hidden_states, forward_batch)
        if shared_output is not None:
            output = output + shared_output
        # DeepEP combine reduces routed experts; the shared branch is replicated.
        return output


class SolarOpen2DecoderLayer(KimiDecoderLayer):
    def __init__(
        self, config, layer_idx, quant_config=None, prefix="", alt_stream=None
    ):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream
        self.self_attn = self._create_attention(
            config, layer_idx, quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = self._create_mlp(
            config, layer_idx, quant_config, prefix=f"{prefix}.mlp"
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        attn, ffn = self.stage_facts(config, layer_idx)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm),
            (ffn, self.post_attention_layernorm),
        )

    def _create_attention(self, config, layer_idx, quant_config, prefix):
        attention_cls = (
            SolarOpen2Attention
            if layer_idx in config.gqa_layers
            else SolarOpen2KimiDeltaAttention
        )
        return attention_cls(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=prefix,
            # The FFN input boundary completes the attention's output sum.
            reduce_results=False,
        )

    def _create_mlp(self, config, layer_idx, quant_config, prefix):
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        ):
            return self._create_moe(config, layer_idx, quant_config, prefix)
        return SolarOpen2MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _create_moe(self, config, layer_idx, quant_config, prefix):
        return SolarOpen2MoE(
            config, layer_idx, quant_config, prefix=prefix, alt_stream=self.alt_stream
        )

    def _forward_attention(
        self, positions, hidden_states, forward_batch, zero_allocator
    ):
        return self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

    def forward(self, positions, hidden_states, forward_batch, zero_allocator):
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self._forward_attention(
            positions, hidden_states, forward_batch, zero_allocator
        )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states, forward_batch)
        return self.ffn_boundary.finish(hidden_states, forward_batch)

    @staticmethod
    def stage_facts(config, layer_idx):
        sparse = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
        )
        next_sparse = (
            config.n_routed_experts is not None
            and layer_idx + 1 >= config.first_k_dense_replace
            and layer_idx + 1 < config.num_hidden_layers
        )
        return declare_attn(), declare_ffn(
            sparse=sparse, next_layer_sparse=next_sparse, output_complete=True
        )


class SolarOpen2Model(KimiLinearModel):
    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.dspark_layers_to_capture = None
        self.alt_stream = get_stream("alt") if is_cuda_alike() else None
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SolarOpen2DecoderLayer(
                config, idx, quant_config, prefix, alt_stream=self.alt_stream
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
            stage_facts=lambda idx: SolarOpen2DecoderLayer.stage_facts(config, idx),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()


class SolarOpen2ForCausalLM(KimiLinearForCausalLM):
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        if config.n_routed_experts is None:
            return None
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkvg_proj": ["q_proj", "k_proj", "v_proj", "g_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkvbfg_proj": ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj"],
        "fused_qkvbfg_a_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
        "qkv_conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
    }

    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.model = SolarOpen2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.start_layer, self.end_layer = self.model.start_layer, self.model.end_layer
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
            if config.tie_word_embeddings and self.pp_group.is_first_rank:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config=config)
        self.capture_aux_hidden_states = False

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.pp_group.world_size > 1:
            raise NotImplementedError("DFLASH aux hidden capture requires PP=1.")
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        # KimiLinearModel already captures the residual stream after each layer,
        # matching DFlash's zero-based target layer-output ids.
        self.set_dspark_layers_to_capture(layer_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".qkvg_proj", ".q_proj", 0),
            (".qkvg_proj", ".k_proj", 1),
            (".qkvg_proj", ".v_proj", 2),
            (".qkvg_proj", ".g_proj", 3),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        expert_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts
                + max(
                    (
                        getattr(
                            getattr(layer, "mlp", None), "num_fused_shared_experts", 0
                        )
                        for layer in self.model.layers
                    ),
                    default=0,
                ),
            )
            if getattr(self.config, "n_routed_experts", None)
            else []
        )
        params = dict(self.named_parameters())
        for args in self._shared_expert_weights(weights):
            name, weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            layer_idx = get_layer_id(name)
            if self._is_non_local_pp_weight(name):
                continue
            if layer_idx is not None and layer_idx >= self.config.num_hidden_layers:
                # Checkpoints may also contain next-token prediction layers.
                continue
            if "rotary_emb." in name:
                continue
            if ".mlp.experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name in name:
                        target = name.replace(weight_name, param_name)
                        param = params[target]
                        param.weight_loader(
                            param,
                            weight,
                            target,
                            expert_id=expert_id,
                            shard_id=shard_id,
                        )
                        break
                else:
                    raise KeyError(f"Unrecognized Solar Open2 expert weight: {name}")
                continue
            # Resolve cache scales before packing projection weights.
            name = maybe_remap_kv_scale_name(name, params)
            if name is None:
                continue
            if layer_idx is not None and ".self_attn." in name:
                attention = self.model.layers[layer_idx].self_attn
                # An MTP head can use KDA where the trunk config selects GQA.
                if isinstance(attention, SolarOpen2KimiDeltaAttention):
                    if _load_kda_weight(name, weight, attention, params) is not None:
                        continue
            for target_module, source, shard_id in stacked_mapping:
                target = name.replace(source, target_module)
                if source not in name or target not in params:
                    continue
                param = params[target]
                param.weight_loader(param, weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params:
                    continue
                param = params[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight, **kwargs)
        self._finalize_loaded_weights()

    def _finalize_loaded_weights(self):
        self.post_load_weights()

    def _shared_expert_weights(self, weights):
        for args in weights:
            name, weight = args[:2]
            if ".mlp.shared_experts." not in name or self._is_non_local_pp_weight(name):
                yield args
                continue
            layer_idx = get_layer_id(name)
            if layer_idx is None or layer_idx >= len(self.model.layers):
                yield args
                continue
            mlp = self.model.layers[layer_idx].mlp
            count = getattr(mlp, "num_fused_shared_experts", 0)
            if not count:
                yield args
                continue
            axis = 1 if ".down_proj." in name else 0
            for shared_idx, shard in enumerate(weight.chunk(count, dim=axis)):
                target = name.replace(
                    ".shared_experts.",
                    f".experts.{self.config.n_routed_experts + shared_idx}.",
                )
                yield (target, shard, *args[2:])

    def post_load_weights(self):
        for layer in self.model.layers:
            attention = getattr(layer, "self_attn", None)
            if isinstance(attention, SolarOpen2KimiDeltaAttention):
                attention.prepare_fused_decode()


EntryClass = SolarOpen2ForCausalLM
