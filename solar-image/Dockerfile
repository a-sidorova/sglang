# SGLang ROCm image: the base lmsysorg/sglang-rocm v0.5.21 image with its
# sglang and AITER replaced by our branches.
#
#   sglang  a-sidorova/sglang-upstage solar-open3-rocm @ 1eefe1b304
#           (bab9c81fea + own commits)
#   aiter   sshlyapn/aiter asidorov/port/solar-4.5 @ 990cfa2ab
#           (v0.1.23 + 7 commits), composable_kernel @ af9e1d1f1
#
# Everything else (torch, triton, flydsl, sglang-kernel, sglang-router, mori,
# Mooncake, nixl, ...) is the base image's, untouched.
#
# Build:
#   ./fetch-sources.sh          # regenerates sglang-src/, aiter-src/, pins.env
#   docker build -t upstage-sglang-rocm:v0.5.21-rocm724 .

# Pinned by digest, not tag: a re-pushed tag would otherwise change the image
# silently. This digest is the tag as of 2026-11-24.
FROM lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261014@sha256:2764f9253c7d9ddca736817f27ae03e3f62fa255b95f45d6c5e465f6c6844e73

# The base image builds AITER for this arch only (GPU_ARCH_LIST=gfx950).
ARG AITER_GPU_ARCHS=gfx950
# 1 (default) prebuilds the AITER JIT modules exactly as the base image does
# (PREBUILD_KERNELS=1: everything except tuners and non-fmha_v3 MHA), so a fresh
# container does not JIT them on first use. 0 skips it: a much faster build, but
# every container then compiles each module the first time it is called.
ARG AITER_PREBUILD=1

COPY pins.env sglang-src.md5 aiter-src.md5 /tmp/pins/

# ---------------------------------------------------------------------------
# 0. Record what the base image has, to check against after the swap: the
#    versions of the packages that must stay untouched, and the AITER modules
#    it ships prebuilt.
# ---------------------------------------------------------------------------
RUN set -eu; \
    python3 -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ('torch','triton','flydsl','sglang-kernel','sglang-router','amd_mori')]" \
        > /tmp/pins/base-packages.txt; \
    ls /sgl-workspace/aiter/aiter/jit/ | sed -n 's/\.so$//p' > /tmp/pins/base-prebuilt.txt; \
    cat /tmp/pins/base-packages.txt; \
    echo "base AITER prebuilt modules: $(wc -l < /tmp/pins/base-prebuilt.txt)"

# ---------------------------------------------------------------------------
# 1. AITER. Not an overlay: the base ships e7d2453f2 (v0.1.23 + 183), the
#    branch is v0.1.23 + 7, so the old tree is removed wholesale (no stale
#    sources or .so survive) and the branch is installed the way the base image
#    installs AITER: build_ext --inplace, then an editable (compat) install at
#    the same path, /sgl-workspace/aiter. AITER_USE_SYSTEM_TRITON=1 is set by the
#    base image, so the install keeps its triton.
#
#    The old tree is removed in its own layer before the COPY: COPY onto it
#    would merge the two trees.
# ---------------------------------------------------------------------------
RUN set -eu; \
    pip uninstall -y amd-aiter; \
    rm -rf /sgl-workspace/aiter
COPY aiter-src/ /sgl-workspace/aiter/
RUN set -eu; \
    . /tmp/pins/pins.env; \
    cd /sgl-workspace/aiter; \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_AITER=${AITER_VERSION}; \
    if [ "${AITER_PREBUILD}" = 1 ]; then \
        PREBUILD_KERNELS=1 GPU_ARCHS=${AITER_GPU_ARCHS} python3 setup.py build_ext --inplace; \
    fi; \
    GPU_ARCHS=${AITER_GPU_ARCHS} pip install --no-deps --no-build-isolation \
        --config-settings editable_mode=compat -e .; \
    rm -rf /root/.cache/pip

# ---------------------------------------------------------------------------
# 2. sglang, the way the base image installs it: pyproject_other.toml (the
#    ROCm dependency set) as pyproject.toml, editable from
#    /sgl-workspace/sglang/python. --no-deps: the base image's dependency set
#    is kept as is. The branch only adds Python and sglang JIT (.cu, compiled at
#    runtime) sources, so the base image's AOT sglang-kernel stays valid.
# ---------------------------------------------------------------------------
RUN set -eu; \
    pip uninstall -y sglang; \
    rm -rf /sgl-workspace/sglang
COPY sglang-src/ /sgl-workspace/sglang/
RUN set -eu; \
    . /tmp/pins/pins.env; \
    cd /sgl-workspace/sglang; \
    cp python/pyproject_other.toml python/pyproject.toml; \
    SETUPTOOLS_SCM_PRETEND_VERSION=${SGLANG_VERSION} \
        pip --no-cache-dir install --no-deps -e python

# ---------------------------------------------------------------------------
# 3. Fail the build if anything did not land. Build-time checks only:
#    `docker build` has no GPU, and `import aiter` shells out to rocminfo. The
#    runtime checks live in verify.sh, baked in as `verify-image`.
#
#    - both packages are installed at the pinned versions, editable, from
#      /sgl-workspace;
#    - every source file is byte-identical to the pinned commit (the manifests
#      fetch-sources.sh wrote); sglang's pyproject.toml is the one file
#      replaced on purpose, by pyproject_other.toml;
#    - torch, triton, flydsl, sglang-kernel, sglang-router and mori are the
#      base image's versions;
#    - with AITER_PREBUILD=1, every module the base image shipped prebuilt and
#      the branch still defines is prebuilt again.
# ---------------------------------------------------------------------------
# The base image's env names the AITER it was built with; keep it truthful.
# build_check.py asserts it matches pins.env.
ENV AITER_COMMIT=990cfa2ab6f55678cd1cbb3804ed8f9fe36bc3df
ENV AITER_COMMIT_DEFAULT=990cfa2ab6f55678cd1cbb3804ed8f9fe36bc3df
COPY build_check.py /tmp/pins/build_check.py
RUN set -eu; \
    . /tmp/pins/pins.env; \
    python3 /tmp/pins/build_check.py /tmp/pins "${AITER_PREBUILD}"; \
    { \
      echo "sglang: ${SGLANG_VERSION} (a-sidorova/sglang-upstage solar-open3-rocm @ ${SGLANG_COMMIT})"; \
      echo "aiter: ${AITER_VERSION} (sshlyapn/aiter asidorov/port/solar-4.5 @ ${AITER_COMMIT}, composable_kernel @ ${CK_COMMIT})"; \
      echo "aiter prebuilt: PREBUILD_KERNELS=${AITER_PREBUILD}, GPU_ARCHS=${AITER_GPU_ARCHS}"; \
      echo "base: lmsysorg/sglang-rocm:v0.5.21-rocm724-mi35x-20261014"; \
    } > /sgl-workspace/image-versions.txt; \
    cp /tmp/pins/pins.env /sgl-workspace/image-pins.env; \
    cat /sgl-workspace/image-versions.txt; \
    rm -rf /tmp/pins

COPY verify.sh /usr/local/bin/verify-image
RUN chmod +x /usr/local/bin/verify-image

WORKDIR /sgl-workspace/sglang
