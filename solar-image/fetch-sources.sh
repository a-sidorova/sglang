#!/usr/bin/env bash
# Regenerate the build inputs the Dockerfile consumes, from git pins.
#
#   sglang-src/      the sglang branch, exported with `git archive`
#   aiter-src/       the aiter branch, exported with `git archive`, plus its
#                    composable_kernel submodule at the commit the branch records
#                    (from the aiter checkout's own submodule)
#   pins.env         commits and the package versions to install them under
#   *-src.md5        md5 of every file in each tree; docker build checks the
#                    installed trees against them
#
# Both trees are pinned by commit, so a re-run on another host produces
# byte-identical inputs or fails loudly.
#
#   ./fetch-sources.sh && docker build -t upstage-sglang-rocm:v0.5.21-rocm724 .
#
# Sources are taken from local checkouts when they have the pinned commits
# (SGLANG_GIT, default: the repo this directory lives in; AITER_GIT, default:
# an `aiter` checkout next to it), else fetched from the remotes.
set -euo pipefail
cd "$(dirname "$0")"

# --- pins ------------------------------------------------------------------
SGLANG_REMOTE=https://github.com/a-sidorova/sglang-upstage.git
SGLANG_REF=solar-open3-rocm
SGLANG_COMMIT=1eefe1b304fe70df2c99d89bc6043665a793401c
# The base image's sglang is installed as 0.5.21.dev20261014+gbab9c81fea;
# the branch is bab9c81fea plus its own commits.
SGLANG_VERSION=0.5.21.dev20261014+g${SGLANG_COMMIT:0:10}

AITER_REMOTE=https://github.com/sshlyapn/aiter.git
AITER_REF=asidorov/port/solar-4.5
AITER_COMMIT=990cfa2ab6f55678cd1cbb3804ed8f9fe36bc3df
# `git describe` = v0.1.23-7-g990cfa2ab; this is what setuptools_scm derives
# from it in a checkout.
AITER_VERSION=0.1.24.dev7+g${AITER_COMMIT:0:9}
# 3rdparty/composable_kernel is not pinned separately: it is the submodule
# commit $AITER_COMMIT records, read from the aiter tree below.
CK_REMOTE=https://github.com/ROCm/composable_kernel.git

has_commit() { git -C "$1" cat-file -e "$2^{commit}" 2>/dev/null; }

# Make sure $1 (a git dir) has commit $3, fetching it from $2 if not.
# Fetches the commit itself, not the ref, so a later push cannot change it.
ensure_commit() {
    local repo=$1 remote=$2 commit=$3 ref=${4:-}
    has_commit "$repo" "$commit" && return 0
    echo "    $commit not in $repo; fetching from $remote"
    git -C "$repo" fetch -q "$remote" "$commit" 2>/dev/null ||
        { [ -n "$ref" ] && git -C "$repo" fetch -q "$remote" "$ref"; } || true
    has_commit "$repo" "$commit" || {
        echo "could not obtain $commit from $remote" >&2
        exit 1
    }
}

# md5 of every regular file in a tree, paths relative to its root.
manifest() {
    (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 md5sum)
}

# --- sglang ----------------------------------------------------------------
echo "==> sglang-src/  $SGLANG_COMMIT"
SGLANG_GIT=${SGLANG_GIT:-$(git rev-parse --show-toplevel)}
ensure_commit "$SGLANG_GIT" "$SGLANG_REMOTE" "$SGLANG_COMMIT" "$SGLANG_REF"
rm -rf sglang-src && mkdir sglang-src
git -C "$SGLANG_GIT" archive "$SGLANG_COMMIT" | tar -x -C sglang-src
# The base image installs sglang from pyproject_other.toml (the ROCm/"other"
# dependency set); the Dockerfile does the same.
[ -f sglang-src/python/pyproject_other.toml ] || {
    echo "sglang-src/python/pyproject_other.toml missing at $SGLANG_COMMIT" >&2
    exit 1
}
manifest sglang-src > sglang-src.md5

# --- aiter -----------------------------------------------------------------
echo "==> aiter-src/  $AITER_COMMIT"
AITER_GIT=${AITER_GIT:-$SGLANG_GIT/../aiter}
if [ -d "$AITER_GIT" ] && has_commit "$AITER_GIT" "$AITER_COMMIT"; then
    echo "    from local checkout $(cd "$AITER_GIT" && pwd)"
else
    echo "    fetching from $AITER_REMOTE"
    AITER_GIT=$(mktemp -d)
    trap 'rm -rf "$AITER_GIT"' EXIT
    git -C "$AITER_GIT" init -q
    ensure_commit "$AITER_GIT" "$AITER_REMOTE" "$AITER_COMMIT" "$AITER_REF"
fi
rm -rf aiter-src && mkdir aiter-src
git -C "$AITER_GIT" archive "$AITER_COMMIT" | tar -x -C aiter-src

CK_COMMIT=$(git -C "$AITER_GIT" ls-tree "$AITER_COMMIT" 3rdparty/composable_kernel | awk '{print $3}')
[ -n "$CK_COMMIT" ] || { echo "$AITER_COMMIT has no 3rdparty/composable_kernel" >&2; exit 1; }
echo "==> aiter-src/3rdparty/composable_kernel/  $CK_COMMIT"
CK_GIT=$AITER_GIT/3rdparty/composable_kernel
if [ -e "$CK_GIT/.git" ] && has_commit "$CK_GIT" "$CK_COMMIT"; then
    echo "    from local checkout $(cd "$CK_GIT" && pwd)"
else
    echo "    fetching from $CK_REMOTE"
    CK_GIT=$(mktemp -d)
    git -C "$CK_GIT" init -q
    ensure_commit "$CK_GIT" "$CK_REMOTE" "$CK_COMMIT"
fi
rmdir aiter-src/3rdparty/composable_kernel 2>/dev/null || true
mkdir -p aiter-src/3rdparty/composable_kernel
git -C "$CK_GIT" archive "$CK_COMMIT" | tar -x -C aiter-src/3rdparty/composable_kernel
manifest aiter-src > aiter-src.md5

# --- pins ------------------------------------------------------------------
cat > pins.env <<EOF
SGLANG_COMMIT=$SGLANG_COMMIT
SGLANG_VERSION=$SGLANG_VERSION
AITER_COMMIT=$AITER_COMMIT
AITER_VERSION=$AITER_VERSION
CK_COMMIT=$CK_COMMIT
EOF

echo
printf '    sglang  %s  (%s files)\n' "$SGLANG_VERSION" "$(wc -l < sglang-src.md5)"
printf '    aiter   %s  (%s files)\n' "$AITER_VERSION" "$(wc -l < aiter-src.md5)"
echo
echo "Sources ready. Next:  docker build -t upstage-sglang-rocm:v0.5.21-rocm724 ."
