# Copyright 2026 Upstage AI.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-selected FP32 parameters for the SolarOpen3 trunk and MTP head."""

import types

import torch
from torch import nn

from sglang.srt.configs.solar_open3 import (
    FP32_PARAM_KINDS,
    validate_precision_config,
)
from sglang.srt.distributed import (
    attention_tensor_model_parallel_all_reduce,
    moe_tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import RMSNorm


@torch.compile(dynamic=True, fullgraph=True)
def _rmsnorm_fp32_gain(x: torch.Tensor, weight: torch.Tensor, eps: float):
    hidden = x.float()
    return (
        hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps) * weight
    ).to(x.dtype)


@torch.compile(dynamic=True, fullgraph=True)
def _add_rmsnorm_fp32_gain(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
):
    hidden = x.float() + residual.float()
    output = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps) * weight
    return output.to(x.dtype), hidden.to(x.dtype)


def _held_gain_rmsnorm_forward(
    self, x, residual=None, post_residual_addition=None, quant_linear=None
):
    # The usual CUDA kernels read the gain as the activation dtype. Use the
    # native FP32 multiply and single output rounding for a held FP32 gain.
    if (
        post_residual_addition is not None
        or self.variance_size_override is not None
        or self.cast_x_before_out_mul
        or self.fp32_residual
        or self.override_orig_dtype is not None
        or x.shape[-1] != self.hidden_size
        or x.numel() == 0
    ):
        return self.forward_native(x, residual, post_residual_addition, quant_linear)
    x = x.contiguous()
    if residual is None:
        return _rmsnorm_fp32_gain(x, self.weight, self.variance_epsilon)
    return _add_rmsnorm_fp32_gain(x, residual, self.weight, self.variance_epsilon)


def _held_gain_allreduce_forward(
    self, x, residual=None, post_residual_addition=None, use_attn_tp_group=True
):
    # Communication fusion otherwise bypasses forward_cuda and its gain dtype
    # handling. Keep the same collective group and normalize separately.
    reduce = (
        attention_tensor_model_parallel_all_reduce
        if use_attn_tp_group
        else moe_tensor_model_parallel_all_reduce
    )
    if residual is not None:
        x = reduce(x)
    return self(x, residual, post_residual_addition)


def hold_fp32_params(model: nn.Module, config) -> None:
    """Apply the checkpoint's precision policy after construction, before loading.

    Preserve Parameter identity and loader metadata. Only the requested kinds
    are promoted; ordinary projections and existing FP32 state stay untouched.
    The output CP query can be selected to preserve checkpoint precision before
    folding; other CP queries keep their activation dtype. Derived score buffers
    use FP32 when attn_res_key_fp32 is enabled.
    """
    validate_precision_config(config)
    kinds = set(getattr(config, "fp32_params", None) or ())
    key_fp32 = getattr(config, "attn_res_key_fp32", False)
    for module in model.modules():
        held_gain = False
        for name, param in module.named_parameters(recurse=False):
            kind = (
                "norm_gain"
                if name == "weight"
                and param.ndim == 1
                and "Norm" in type(module).__name__
                else name
                if name in FP32_PARAM_KINDS
                else None
            )
            if kind in kinds:
                param.data = param.data.float()
                held_gain = held_gain or name == "weight"
        if held_gain and isinstance(module, RMSNorm):
            module.forward_cuda = types.MethodType(_held_gain_rmsnorm_forward, module)
            module.forward_with_allreduce_fusion = types.MethodType(
                _held_gain_allreduce_forward, module
            )
        if key_fp32:
            for name, buffer in module.named_buffers(recurse=False):
                if name.endswith("_res_score_weight"):
                    module._buffers[name] = buffer.float()
