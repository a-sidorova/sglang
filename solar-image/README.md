# SGLang + AITER ROCm image (MI355X)

The stock `lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261014` image with its
**sglang** and **AITER** replaced by our branches. Everything else in the base
image (torch, triton, flydsl, sglang-kernel, sglang-router, mori, Mooncake,
nixl, ROCm) stays as it is.

## Inputs

| | Value |
|---|---|
| Base image | `lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261014`, pinned by digest `sha256:2764f925…` |
| sglang | `solar-open3-rocm` @ `1eefe1b304` (`bab9c81fea` + own commits) |
| sglang remote | `github.com/a-sidorova/sglang-upstage` (**private**) |
| AITER | `asidorov/port/solar-4.5` @ `990cfa2ab` (`v0.1.23` + 7 commits) |
| AITER remote | `github.com/sshlyapn/aiter` |
| composable_kernel | not pinned separately: the submodule commit AITER `990cfa2ab` records (`af9e1d1f1`, same as the base image) |
| Target arch | `gfx950` (MI355X only) |
| torch / triton / flydsl | `2.11.0+rocm7.2` / `3.7.0+amd.rocm7.2.0.git89002410` / `0.3.4.1` (base image, not reinstalled) |

Installed versions in the image:

| Package | Version | Editable from |
|---|---|---|
| `sglang` | `0.5.21.dev20261014+g1eefe1b304` | `/sgl-workspace/sglang/python` |
| `amd-aiter` | `0.1.24.dev7+g990cfa2ab` | `/sgl-workspace/aiter` |

Both are installed at the same paths and in the same way as the base image
installs its own copies, so anything that relies on the base image layout
(`PYTHONPATH=/sgl-workspace/aiter` in `/etc/bash.bashrc`, editable installs)
keeps working. `/sgl-workspace/image-versions.txt` records what went in.

## Build

Requires Docker on a ROCm host. No GPU needed for the build. The host has no
`buildx`, so use the legacy builder (`DOCKER_BUILDKIT=0`).

```bash
# sglang (this repo) and aiter checked out side by side:
#   <ws>/sglang   branch solar-open3-rocm
#   <ws>/aiter    branch asidorov/port/solar-4.5 (with its submodule)
cd <ws>/sglang/solar-image
./fetch-sources.sh
DOCKER_BUILDKIT=0 docker build -t upstage-sglang-rocm:v0.5.21-rocm724 .
```

`fetch-sources.sh` exports both trees from git at the pinned commits:

- sglang from the checkout this directory lives in (`SGLANG_GIT` to
  override), falling back to fetching the commit from the private remote;
- AITER from an `aiter` checkout next to the sglang one (`AITER_GIT` to
  override), falling back to `github.com/sshlyapn/aiter`;
- `3rdparty/composable_kernel` at the commit the AITER commit records, from
  the AITER checkout's own submodule (`<ws>/aiter/3rdparty/composable_kernel`),
  falling back to `github.com/ROCm/composable_kernel`.

It writes `sglang-src/`, `aiter-src/`, `pins.env` (commits and the versions to
install them under) and an md5 manifest of every file in each tree
(`sglang-src.md5`, `aiter-src.md5`). All of these are build inputs and
git-ignored; re-running it on another host gives byte-identical inputs.

To build from newer commits, update `SGLANG_COMMIT` / `AITER_COMMIT` (and
`AITER_VERSION` from `git describe --tags --long`) in `fetch-sources.sh`, and `ENV AITER_COMMIT` in the
`Dockerfile` (the build fails if it does not match `pins.env`). The commits
must be reachable from the remotes, or present in the local checkouts.

### Build arguments

| Arg | Default | |
|---|---|---|
| `AITER_PREBUILD` | `1` | `1` prebuilds the AITER JIT modules exactly as the base image does (`PREBUILD_KERNELS=1`: everything except tuners and non-`fmha_v3` MHA). `0` skips it: a much faster build, but every fresh container compiles each AITER module the first time it is used. |
| `AITER_GPU_ARCHS` | `gfx950` | The arch AITER modules are compiled for. |

### What the build does

1. Records the base image's torch / triton / flydsl / sglang-kernel /
   sglang-router / mori versions and its list of prebuilt AITER modules.
2. **AITER**: the base image ships AITER `e7d2453f2` (`v0.1.23` + 183), the
   branch is `v0.1.23` + 7, so this is a full replacement, not an overlay. The
   old `/sgl-workspace/aiter` is uninstalled and removed in its own layer (no
   stale sources or `.so` survive), the branch is copied in, prebuilt with
   `PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python setup.py build_ext --inplace`,
   then installed with `pip install --no-deps --no-build-isolation
   --config-settings editable_mode=compat -e .` - the same sequence as the base
   image. `AITER_USE_SYSTEM_TRITON=1` (set by the base image) keeps its triton.
3. **sglang**: the old `/sgl-workspace/sglang` is uninstalled and removed, the
   branch is copied in, `python/pyproject_other.toml` becomes
   `python/pyproject.toml` (as in the base image), and it is installed with
   `pip install --no-deps -e python`. `--no-deps` keeps the base image's
   dependency set untouched. The branch adds only Python and sglang JIT
   sources (compiled at runtime), so the base image's AOT `sglang-kernel` is
   still valid.
4. **Build-time checks** (`build_check.py`), failing the build on any miss:
   - `sglang` and `amd-aiter` are installed once each, at the pinned versions,
     editable from `/sgl-workspace/sglang/python` and `/sgl-workspace/aiter`;
   - every file of both trees is byte-identical to the pinned commit (the
     md5 manifests); the only intended change is sglang's `pyproject.toml`,
     which must equal `pyproject_other.toml`;
   - torch, triton, flydsl, sglang-kernel, sglang-router and mori are still
     the base image's versions;
   - with `AITER_PREBUILD=1`, every module the base image shipped prebuilt and
     the branch still defines is prebuilt again;
   - `ENV AITER_COMMIT` matches the pin.

## Verify

Needs a GPU (`import aiter` calls `rocminfo`). Pick a free GPU with
`HIP_VISIBLE_DEVICES`.

```bash
docker run --rm --entrypoint verify-image \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=0 upstage-sglang-rocm:v0.5.21-rocm724
```

It checks:

- `amd-aiter` is the pinned version, imported from `/sgl-workspace/aiter`,
  the GPU is `gfx950`, and branch-only AITER modules
  (`chunk_delta_attn.flash_sda`, `fused_moe_router`) import - i.e. this is the
  branch, not the base image's AITER;
- two prebuilt HIP modules (`rms_norm`, `silu_and_mul`) run, match a torch
  reference, and no `.so` is compiled or rebuilt at runtime;
- `sglang` is the pinned version, imported from
  `/sgl-workspace/sglang/python`, its AITER-dependent modules import, and
  `python -m sglang.launch_server --help` runs.

Ends with `OK`.

## Running a container

```bash
docker run -it --name <name> \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 32G --security-opt seccomp=unconfined \
  -v <models>:<models> \
  upstage-sglang-rocm:v0.5.21-rocm724 bash
```

For development, both packages are editable, so a checkout can be mounted
over `/sgl-workspace/sglang` or `/sgl-workspace/aiter` without reinstalling
(AITER modules whose sources changed are then rebuilt by its JIT on first use).

## Files

| File | |
|---|---|
| `Dockerfile` | The image (steps above). |
| `fetch-sources.sh` | Pins; exports `sglang-src/`, `aiter-src/`, `pins.env` and the md5 manifests. |
| `build_check.py` | Build-time checks, run by the Dockerfile. |
| `verify.sh` | Runtime (GPU) checks, baked in as `verify-image`. |
