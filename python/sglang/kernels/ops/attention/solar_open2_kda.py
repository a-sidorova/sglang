# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional ROCm fused convolution, KDA recurrence, and gated output norm."""

import logging
import os
import tempfile
from functools import cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
SUPPORTED_HEADS = (8, 12, 16, 24, 48, 96)


@cache
def available() -> bool:
    if (
        os.environ.get("SGLANG_SOLAR_OPEN2_FUSED_KDA_DECODE", "0") == "0"
        or torch.version.hip is None
        or not torch.cuda.is_available()
    ):
        return False
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        return False
    try:
        library = os.environ.get("SGLANG_SOLAR_OPEN2_KDA_LIB")
        if library:
            torch.ops.load_library(library)
        else:
            from torch.utils.cpp_extension import _get_build_directory, load

            source = (
                Path(__file__).resolve().parents[2]
                / "jit/csrc/attention/solar_open2_kda.cu"
            )
            name = "sglang_solar_open2_kda"
            build_dir = Path(_get_build_directory(name, verbose=False))
            staged_source = build_dir / source.name
            contents = source.read_bytes()
            if not staged_source.exists() or staged_source.read_bytes() != contents:
                with tempfile.NamedTemporaryFile(dir=build_dir, delete=False) as temp:
                    temp.write(contents)
                os.replace(temp.name, staged_source)
            load(
                name=name,
                sources=[str(staged_source)],
                build_directory=str(build_dir),
                extra_cuda_cflags=["-O3"],
                is_python_module=False,
            )
        return hasattr(torch.ops.sglang, "solar_open2_kda_decode")
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Solar Open2 fused KDA unavailable; using Triton: %s", exc)
        return False


def stage_weights(attention) -> None:
    layer = attention.attn
    layer._solar_fused_decode_args = None
    if (
        attention.head_dim != 128
        or attention.conv_size != 4
        or attention.local_num_heads not in SUPPORTED_HEADS
        or attention.qkv_conv1d.weight.dtype != torch.float32
        or not attention.qkv_conv1d.weight.is_cuda
        or not available()
    ):
        return
    width = attention.local_num_heads * attention.head_dim
    conv_weight = (
        attention.qkv_conv1d.weight.detach()
        .reshape(3, width, 4)
        .permute(0, 2, 1)
        .contiguous()
    )
    norm_weight = attention.o_norm.weight.detach().float().contiguous()
    layer._solar_fused_decode_args = (
        conv_weight,
        norm_weight,
        attention.o_norm.eps,
    )


def covered(qkv, a, b, conv_state, state, indices, gate) -> bool:
    if state.ndim != 4 or qkv.ndim != 2:
        return False
    batch, channels = qkv.shape
    heads = state.shape[1]
    dim = heads * 128
    return (
        0 < batch <= 65535
        and heads in SUPPORTED_HEADS
        and state.shape[2:] == (128, 128)
        and channels == 3 * dim
        and a.shape == (batch, dim)
        and b.shape == (1, batch, heads)
        and gate.shape == (batch, dim)
        and conv_state.ndim == 3
        and conv_state.shape[1:] == (3, channels)
        and conv_state.stride(1) == channels
        and conv_state.stride(2) == 1
        and state.shape[0] == conv_state.shape[0]
        and state.stride()[1:] == (128 * 128, 128, 1)
        and indices.shape == (batch,)
        and indices.dtype == torch.int32
        and indices.is_contiguous()
        and state.dtype == torch.float32
        and all(
            t.dtype == torch.bfloat16 and t.stride(-1) == 1
            for t in (qkv, a, b, conv_state, gate)
        )
        and all(
            t.device == qkv.device for t in (a, b, conv_state, state, indices, gate)
        )
        and qkv.is_cuda
    )


def run(layer, qkv, a, b, conv_state, state, indices, gate):
    conv_weight, norm_weight, norm_eps = layer._solar_fused_decode_args
    heads = layer.num_v_heads
    output = qkv.new_empty((1, qkv.shape[0], heads, 128))
    torch.ops.sglang.solar_open2_kda_decode(
        qkv,
        conv_weight,
        None,
        conv_state.transpose(1, 2),
        a.unflatten(-1, (heads, 128)).unsqueeze(0),
        b,
        layer.A_log.reshape(-1),
        layer.dt_bias,
        indices,
        state,
        output,
        layer.lower_bound,
        gate.unflatten(-1, (heads, 128)),
        norm_weight,
        norm_eps,
        layer.beta_scale,
    )
    return output
