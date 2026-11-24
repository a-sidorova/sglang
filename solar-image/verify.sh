#!/usr/bin/env bash
# Runtime verification of the image. Needs a GPU: `import aiter` shells out to
# rocminfo, so none of this can run during `docker build`.
#
#   docker run --rm --entrypoint verify-image \
#     --device=/dev/kfd --device=/dev/dri --group-add video \
#     -e HIP_VISIBLE_DEVICES=0 upstage-sglang-rocm:v0.5.21-rocm724
set -euo pipefail

set -a; . /sgl-workspace/image-pins.env; set +a
cat /sgl-workspace/image-versions.txt
echo

python3 - <<'PY'
import importlib.metadata as md
import os
import time

import torch

print(f"torch            {torch.__version__}")
print(f"triton           {md.version('triton')}")
print(f"flydsl           {md.version('flydsl')}")
print(f"sglang-kernel    {md.version('sglang-kernel')}")

# --- AITER: the pinned branch, from /sgl-workspace/aiter -------------------
jit_dir = "/sgl-workspace/aiter/aiter/jit"
so_before = {f: os.path.getmtime(os.path.join(jit_dir, f))
             for f in os.listdir(jit_dir) if f.endswith(".so")}

import aiter
from aiter._version import __version__ as aiter_scm_version
from aiter.jit.utils.chip_info import get_gfx

assert md.version("amd-aiter") == os.environ["AITER_VERSION"], md.version("amd-aiter")
assert aiter_scm_version == os.environ["AITER_VERSION"], aiter_scm_version
assert aiter.__file__ == "/sgl-workspace/aiter/aiter/__init__.py", aiter.__file__
print(f"amd-aiter        {md.version('amd-aiter')}  ({os.path.dirname(aiter.__file__)})")
print(f"gfx              {get_gfx()}")
assert get_gfx() == "gfx950", get_gfx()

# Modules that exist on the branch but not in the base image's AITER: the image
# must carry the branch, not a leftover of the base.
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_sda import flash_sda_fwd
from aiter.ops.fused_moe_router import *  # noqa: F401,F403
print("branch modules   chunk_delta_attn.flash_sda, fused_moe_router: import OK")

# Two prebuilt HIP modules (module_rmsnorm, module_activation) must run
# straight from the image: no compile, i.e. no new or rebuilt .so.
x = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")
w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
t0 = time.perf_counter()
y = aiter.rms_norm(x, w, 1e-6)
ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float())
torch.testing.assert_close(y.float(), ref, atol=5e-2, rtol=2e-2)

g = torch.randn(64, 2 * 4096, dtype=torch.bfloat16, device="cuda")
out = torch.empty(64, 4096, dtype=torch.bfloat16, device="cuda")
aiter.silu_and_mul(out, g)
a, b = g.float().chunk(2, dim=-1)
torch.testing.assert_close(out.float(), torch.nn.functional.silu(a) * b, atol=5e-2, rtol=2e-2)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

so_after = {f: os.path.getmtime(os.path.join(jit_dir, f))
            for f in os.listdir(jit_dir) if f.endswith(".so")}
rebuilt = sorted(f for f in so_after if so_before.get(f) != so_after[f])
print(f"aiter kernels    rms_norm, silu_and_mul correct in {dt:.1f}s; prebuilt .so: {len(so_before)}")
assert not rebuilt, f"JIT compiled at runtime (not prebuilt in the image): {rebuilt}"

# --- sglang: the pinned branch, from /sgl-workspace/sglang -----------------
import sglang

assert md.version("sglang") == os.environ["SGLANG_VERSION"], md.version("sglang")
assert sglang.__file__ == "/sgl-workspace/sglang/python/sglang/__init__.py", sglang.__file__
print(f"sglang           {md.version('sglang')}  ({os.path.dirname(sglang.__file__)})")

# The sglang side of AITER: its attention backend and MoE paths import aiter.
import sglang.srt.layers.attention.aiter_backend  # noqa: F401
import sglang.srt.layers.moe.fused_moe_triton  # noqa: F401
from sglang.srt.server_args import ServerArgs  # noqa: F401
print("sglang imports   aiter_backend, fused_moe_triton, server_args: OK")
PY

python3 -m sglang.launch_server --help > /dev/null
echo "sglang.launch_server --help: OK"
echo
echo OK
