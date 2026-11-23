"""Fused SDA projections must use the loader's resolved execution dtype."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.configs.solar_open3 import SolarOpen3Config
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models import solar_open3
from sglang.srt.runtime_context import get_parallel


@torch.no_grad()
def check_fused_projection_uses_execution_dtype(
    monkeypatch, device, tp_size, config_dtype, runtime_dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = SolarOpen3Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=8,
        gqa_layers=[],
        linear_attn_config={"head_dim": 8, "num_heads": 4},
        dtype=config_dtype,
    )
    # Isolate distributed bookkeeping; keep the real projection constructors,
    # checkpoint loaders and matrix multiplies, including TP sharding.
    with (
        get_parallel().override(
            tp_rank=tp_size - 1,
            tp_size=tp_size,
            moe_tp_size=tp_size,
            attn_tp_rank=tp_size - 1,
            attn_tp_size=tp_size,
        ),
        set_default_torch_dtype(runtime_dtype),
        torch.device(device),
    ):
        attention = solar_open3.SolarOpen3KimiDeltaAttention(0, config)
    a_proj = attention.fused_qkvbfg_a_proj
    b_proj = attention.fused_fg_b_proj
    assert a_proj.weight.dtype == b_proj.weight.dtype == runtime_dtype
    assert config.dtype == config_dtype
    # These state parameters deliberately retain FP32 precision.
    assert attention.qkv_conv1d.weight.dtype == attention.dt_bias.dtype == torch.float32
    assert attention.A_log.dtype == torch.float32

    generator = torch.Generator(device=device).manual_seed(147959)
    checkpoint_dtype = config_dtype or torch.float32
    for shard, size in enumerate([32, 32, 32, 4, 8, 8]):
        weight = torch.randn(size, 32, device=device, generator=generator).to(
            checkpoint_dtype
        )
        a_proj.weight_loader(a_proj.weight, weight, shard)
    for shard in range(2):
        weight = torch.randn(32, 8, device=device, generator=generator).to(
            checkpoint_dtype
        )
        b_proj.weight_loader(b_proj.weight, weight, shard)

    hidden = torch.randn(5, 32, device=device, generator=generator).to(runtime_dtype)
    fused = F.linear(hidden, a_proj.weight)
    qkv, beta, fg = fused.split(attention.split_sizes, dim=-1)
    f, g = fg.split(8, dim=-1)
    expected = (qkv, beta, F.linear(f, b_proj.weight[0]), F.linear(g, b_proj.weight[1]))

    def check(actual):
        for output, reference in zip(actual, expected, strict=True):
            assert output.dtype == runtime_dtype
            torch.testing.assert_close(output, reference)

    check(attention.forward_qkvbfg_fused(hidden))
    if device == "cuda":
        # Reproduce the startup failure inside a real CUDA graph as well.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                attention.forward_qkvbfg_fused(hidden)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = attention.forward_qkvbfg_fused(hidden)
        graph.replay()
        check(captured)
