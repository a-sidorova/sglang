# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Upstage AI.
"""Shared epilogue for SolarOpen3 attention merges; match the CUDA BF16 gate."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rounded_output_gate(value, GATE, offset, mask):
    # Keep BOTH rounding boundaries from bf16(attention) * bf16(sigmoid(gate)).
    x = value.to(tl.bfloat16).to(tl.float32)
    g = tl.load(GATE + offset, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    return x * sigmoid


def _validate_output_gate(q, gate, output):
    if gate is not None:
        if (
            not q.is_cuda
            or torch.version.hip is not None
            or q.dtype != torch.bfloat16
            or gate.dtype != q.dtype
            or gate.device != q.device
            or gate.ndim != 2
            or (gate.shape[0] != q.shape[0] or gate.numel() != q.numel())
            or gate.stride(1) != 1
        ):
            raise ValueError(
                "merge gate requires CUDA BF16 queries and a matching row-strided 2D gate"
            )
    if output is not None and (
        output.numel() != q.numel()
        or output.dtype != q.dtype
        or output.device != q.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "merge output must be a contiguous query-sized buffer on the same device/dtype"
        )


def _finish_output(result, gate, output):
    """Keep the existing standalone gate on bodies without a fused merge epilogue."""
    if gate is not None:
        from sglang.kernels.ops.attention import mla_output_gate

        result = result.reshape(gate.shape)
        result = (
            mla_output_gate.kimi_k3_mla_output_gate(result, gate)
            if mla_output_gate.covered(result, gate)
            else result * gate.sigmoid()
        )
    if output is not None:
        output.copy_(result.reshape_as(output))
        return output
    return result
