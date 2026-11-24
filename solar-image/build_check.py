"""Build-time checks for the image (no GPU): run from the Dockerfile.

    python3 build_check.py <pins dir> <AITER_PREBUILD>

<pins dir> holds pins.env, sglang-src.md5, aiter-src.md5 (from
fetch-sources.sh) and base-packages.txt, base-prebuilt.txt (recorded from the
base image before the swap). Nothing here imports aiter or sglang: `import
aiter` shells out to rocminfo, which needs a GPU. Those checks are in verify.sh.
"""

import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys

PINS_DIR, PREBUILD = sys.argv[1], sys.argv[2] == "1"
SGLANG_DIR = "/sgl-workspace/sglang"
AITER_DIR = "/sgl-workspace/aiter"

pins = dict(
    line.strip().split("=", 1)
    for line in open(os.path.join(PINS_DIR, "pins.env"))
    if "=" in line
)
errors = []


def check(ok, msg):
    print(f"[{'ok' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok:
        errors.append(msg)


# --- installed versions and locations --------------------------------------
for dist, want, src in (
    ("sglang", pins["SGLANG_VERSION"], f"{SGLANG_DIR}/python"),
    ("amd-aiter", pins["AITER_VERSION"], AITER_DIR),
):
    got = md.version(dist)
    check(got == want, f"{dist} version {got} (want {want})")
    # PEP 660 records the editable source in direct_url.json.
    direct = json.loads(md.distribution(dist).read_text("direct_url.json") or "{}")
    url = direct.get("url", "")
    editable = direct.get("dir_info", {}).get("editable", False)
    check(
        editable and url == f"file://{src}",
        f"{dist} installed editable from {url} (want file://{src})",
    )

# Exactly one copy of each: a leftover dist-info from the base would shadow it.
out = subprocess.run(
    [sys.executable, "-m", "pip", "list", "--format=json"],
    check=True, capture_output=True, text=True,
).stdout
names = [p["name"].lower() for p in json.loads(out)]
for dist in ("sglang", "amd-aiter"):
    check(names.count(dist) == 1, f"{dist} installed once ({names.count(dist)})")

check(
    os.environ.get("AITER_COMMIT") == pins["AITER_COMMIT"],
    f"ENV AITER_COMMIT={os.environ.get('AITER_COMMIT')} matches the pin",
)

# --- source trees byte-identical to the pinned commits ---------------------
def verify_tree(root, manifest, allowed_changes=()):
    bad = []
    n = 0
    for line in open(manifest):
        want, rel = line.rstrip("\n").split("  ", 1)
        n += 1
        if rel in allowed_changes:
            continue
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            bad.append(f"missing {rel}")
            continue
        with open(path, "rb") as f:
            got = hashlib.file_digest(f, "md5").hexdigest()
        if got != want:
            bad.append(f"modified {rel}")
    check(not bad, f"{root}: {n} files match the pinned commit"
          + (f"; {len(bad)} differ: {bad[:10]}" if bad else ""))


verify_tree(
    SGLANG_DIR,
    os.path.join(PINS_DIR, "sglang-src.md5"),
    allowed_changes=("./python/pyproject.toml",),
)
check(
    open(f"{SGLANG_DIR}/python/pyproject.toml").read()
    == open(f"{SGLANG_DIR}/python/pyproject_other.toml").read(),
    "sglang pyproject.toml is pyproject_other.toml",
)
verify_tree(AITER_DIR, os.path.join(PINS_DIR, "aiter-src.md5"))

# --- the rest of the base image is untouched --------------------------------
for line in open(os.path.join(PINS_DIR, "base-packages.txt")):
    dist, want = line.split()
    got = md.version(dist)
    check(got == want, f"{dist} {got} (base image: {want})")

# --- prebuilt AITER modules -------------------------------------------------
jit_dir = os.path.join(AITER_DIR, "aiter", "jit")
built = sorted(f[:-3] for f in os.listdir(jit_dir) if f.endswith(".so"))
print(f"AITER prebuilt modules: {len(built)}", flush=True)
if PREBUILD:
    defined = set(json.load(open(os.path.join(jit_dir, "optCompilerConfig.json"))))
    base = [m.strip() for m in open(os.path.join(PINS_DIR, "base-prebuilt.txt"))]
    # Modules the base shipped and the branch still defines must be prebuilt
    # again. The rest are newer than the branch (base AITER is 176 commits
    # ahead) or recipe-generated variants (mha_varlen_*, moe_ck2stages_*),
    # which are reported, not required.
    missing = [m for m in base if m in defined and m not in built]
    check(not missing, f"every base-prebuilt module the branch defines is prebuilt "
          f"({len([m for m in base if m in defined])})"
          + (f"; missing {missing}" if missing else ""))
    others = sorted(set(base) - defined - set(built))
    if others:
        print(f"[note] base-prebuilt, not built here ({len(others)}): {others}")
    check(len(built) > 0, "AITER_PREBUILD=1 produced .so files")
else:
    print("[note] AITER_PREBUILD=0: modules JIT-compile on first use")

if errors:
    sys.exit(f"{len(errors)} check(s) failed")
print("build checks OK")
