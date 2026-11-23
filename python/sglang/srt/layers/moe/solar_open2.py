# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Solar Open2 router projections with FP32 logits."""

import os
from functools import cache

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils.custom_op import register_custom_op


@cache
def _mm_supports_fp32_out(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    probe = torch.zeros((16, 16), dtype=torch.bfloat16, device=device)
    try:
        return torch.mm(probe, probe, out_dtype=torch.float32).dtype == torch.float32
    except (RuntimeError, TypeError):
        return False


def _router_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


def _router_gemm_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    max_tokens = int(os.environ.get("SGLANG_SOLAR_OPEN2_ROUTER_GEMV_MAX_TOKENS", "4"))
    if (
        os.environ.get("SGLANG_SOLAR_OPEN2_ROUTER_GEMV", "torch") == "triton"
        and x.shape[0] <= max_tokens
    ):
        from sglang.kernels.ops.moe import solar_open2_router_gemv

        if solar_open2_router_gemv.supports(x, weight):
            return solar_open2_router_gemv.router_gemv_fp32(x, weight)
    return torch.mm(x, weight.t(), out_dtype=torch.float32)


@register_custom_op(fake_impl=_router_gemm_fake)
def solar_open2_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _router_gemm_impl(x, weight)


class SolarOpen2RouterGate(ReplicatedLinear):
    """Opt-in BF16 checkpoint weights with FP32 accumulation and output."""

    def __init__(self, *args, allow_bf16_weight=True, **kwargs):
        mode = (
            os.environ.get("SGLANG_SOLAR_OPEN2_ROUTER_BF16_WEIGHT", "0")
            if allow_bf16_weight
            else "0"
        )
        modes = {"0": "off", "1": "fp32out", "2": "bf16out"}
        mode = modes.get(mode, mode)
        if mode not in ("off", "fp32out", "bf16out"):
            raise ValueError(f"Unsupported Solar Open2 router weight mode: {mode}")
        kwargs["params_dtype"] = torch.float32 if mode == "off" else torch.bfloat16
        super().__init__(*args, **kwargs)
        self.bf16_output = mode == "bf16out"
        self.fp32_output_supported = mode == "fp32out" and _mm_supports_fp32_out(
            self.weight.device
        )

    def forward(self, x: torch.Tensor):
        if self.weight.dtype == torch.bfloat16 and x.dtype == torch.bfloat16:
            if self.bf16_output:
                return F.linear(x, self.weight).float(), None
            if self.fp32_output_supported:
                return solar_open2_router_gemm(x, self.weight), None
        return linear_bf16_fp32(x, self.weight.float()), None


@cache
def _fused_router_ops():
    try:
        from aiter.fused_moe import fused_moe_router, fused_moe_router_supported

        return fused_moe_router, fused_moe_router_supported
    except ImportError:
        return None


class SolarOpen2FusedRouter:
    """AITER routing/quantization fusion for supported Quark TP-only layers."""

    def __init__(self, experts, config):
        self.experts = experts
        self.config = config
        self.ops = None
        if os.environ.get("SGLANG_SOLAR_OPEN2_AITER_FUSED_ROUTER", "0") != "1":
            return
        from sglang.srt.layers.moe import get_moe_a2a_backend
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils import get_bool_env_var, is_hip

        if (
            not is_hip()
            or not get_bool_env_var("SGLANG_USE_AITER")
            or get_parallel().moe_ep_size != 1
            or not get_moe_a2a_backend().is_none()
            or experts.num_fused_shared_experts
            or config.hidden_act != "silu"
        ):
            return
        scheme = getattr(experts, "scheme", None)
        runner = getattr(scheme, "runner", None)
        if runner is None or not runner.runner_backend.is_aiter():
            return
        from aiter import QuantType

        scheme_name = type(scheme).__name__
        if scheme_name == "QuarkW4A4MXFp4MoE" and hasattr(torch, "float4_e2m1fn_x2"):
            self.quant_type = QuantType.per_1x32
        elif (
            scheme_name == "QuarkW8A8FP8MoE"
            and scheme.is_weight_per_channel
            and not scheme.is_static_input_scheme
        ):
            self.quant_type = QuantType.per_Token
        else:
            return
        self.ops = _fused_router_ops()

    def __call__(self, hidden_states, router_logits, correction_bias):
        if (
            self.ops is None
            or hidden_states.dtype != torch.bfloat16
            or router_logits.dtype != torch.float32
            or correction_bias.dtype != torch.float32
            or hidden_states.shape[0] == 0
            or hidden_states.shape[-1] != self.experts.hidden_size
        ):
            return None
        from aiter import ActivationType, QuantType

        op, supported = self.ops
        layer, config = self.experts, self.config
        w1, w2 = layer.w13_weight, layer.w2_weight
        if self.quant_type == QuantType.per_1x32:
            w1, w2 = w1.view(torch.float4_e2m1fn_x2), w2.view(torch.float4_e2m1fn_x2)
            if getattr(layer.w13_weight, "is_shuffled", False):
                w1.is_shuffled = w2.is_shuffled = True
        kwargs = dict(
            quant_type=self.quant_type,
            activation=ActivationType.Silu,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            gate_mode="separated",
        )
        if not supported(hidden_states, w1, w2, config.num_experts_per_tok, **kwargs):
            return None
        return op(
            hidden_states,
            router_logits,
            correction_bias,
            w1,
            w2,
            config.num_experts_per_tok,
            **kwargs,
            need_renorm=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=getattr(layer, "w13_input_scale", None),
            a2_scale=getattr(layer, "w2_input_scale", None),
        )
