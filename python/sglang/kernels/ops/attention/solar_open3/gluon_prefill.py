# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Upstage AI.
"""The fused per-head routed prefill for solar_open3, in three attention pieces:

  * the ANCHOR pass -- a Gluon (sm100) attention kernel over every closed anchor that also emits
    the per-head election scores from the same QK tiles (the generated section below);
  * the SELECTED pass -- the raw tokens of the elected non-own units, minus the rows that are
    themselves anchors (FA4 varlen, ``_a`` variables);
  * the OWN pass -- the query block's own span, causal, minus its anchor rows (``_own_half_kernel``,
    ``_b`` variables);

merged by log-sum-exp. Attending "all anchors" first and removing anchor rows from the raw
pieces is the served dedup union exactly (pool="last": an anchor IS its raw token), so the
anchor pass can run before the election it feeds -- no selection before the anchor pass, no
LSE subtraction.

Deployed next to ``solar_open3.py`` by ``install.py``. Which body serves the prefill is decided
once at initialisation (``_resolve_prefill_body`` in the adapter): a deployment that asks for
the per-head election on an sm100 part takes THIS body or refuses to build -- nothing falls
back at runtime, and an exception raised in here propagates.

The anchor-pass KERNEL lives in generated sibling modules -- ``solar_open3_gluon_pass1_t37``
(triton 3.6 / 3.7) and ``solar_open3_gluon_pass1_t38`` (triton 3.8), written by the generators
in ``gluon_gen/`` from the official Gluon attention example of the matching release; edit the
GENERATOR, never those modules (the drift test fails the build). This module is hand-maintained,
imports no gluon at load time, and selects the kernel by triton version (``load_kernel``).
"""

# ── the anchor-pass kernel: generated, version-selected, imported on demand ───────────────
# The kernel modules (`solar_open3_gluon_pass1_t37.py` for triton 3.6 / 3.7,
# `solar_open3_gluon_pass1_t38.py` for 3.8) are generated from the official Gluon attention
# example of the matching triton release and import triton's EXPERIMENTAL gluon API. This
# module imports neither at load time, so the adapter's body resolver can read the kernel's
# shape from here on any triton and any part; `load_kernel()` imports the one this triton has
# -- or raises an ImportError that names the version and the supported set -- and the resolver
# calls it at initialisation whenever the fused body is the answer.
import importlib

import torch
import triton
import triton.language as tl

# The kernel's shape, for the glue, the adapter's dispatch gate and the tests to read instead
# of restating it. Every generated kernel module carries the same constants; `load_kernel`
# asserts they agree.
PASS1_BLOCK_M = 256  # query tile == one query block: the election gate is per tile
PASS1_BLOCK_N = 128  # anchor tile
ANCHORS_PER_UNIT = (
    32  # the score epilogue folds 32-anchor segments: one routing unit each
)
SM_MAJOR = 10  # tcgen05 / TMEM code -- sm100 parts only
PASS1_HEAD_DIM = 128  # the served head dim; the kernel carries HEAD_DIM branches for
# others, unmeasured -- the adapter refuses them at init
# triton (major, minor) -> the kernel module that runs on it.
KERNEL_BY_TRITON = {
    (3, 6): "solar_open3_gluon_pass1_t37",
    (3, 7): "solar_open3_gluon_pass1_t37",
    (3, 8): "solar_open3_gluon_pass1_t38",
}
_KERNEL = None


def triton_version() -> tuple:
    return tuple(int(x) for x in triton.__version__.split(".")[:2])


def load_kernel():
    """The anchor-pass kernel module for the installed triton, memoised. Raises ImportError --
    naming the triton version, the supported set and the module tried -- when there is none
    or it does not import; the adapter's body resolver turns that into a boot refusal."""
    global _KERNEL
    if _KERNEL is None:
        ver = triton_version()
        name = KERNEL_BY_TRITON.get(ver)
        if name is None:
            raise ImportError(
                f"no gluon anchor-pass kernel for triton {triton.__version__}: the generated "
                f"kernels cover {sorted('.'.join(map(str, v)) for v in KERNEL_BY_TRITON)} "
                "(gluon is an experimental API and moves between releases). Pin triton, or "
                "generate a kernel for this release from its Gluon attention example."
            )
        full = (__name__.rsplit(".", 1)[0] + "." + name) if "." in __name__ else name
        try:
            k = importlib.import_module(full)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001 -- reported as an import failure, by name
            raise ImportError(
                f"{full} did not import under triton {triton.__version__}: "
                f"{type(e).__name__}: {e}"
            ) from e
        for c in (
            "PASS1_BLOCK_M",
            "PASS1_BLOCK_N",
            "ANCHORS_PER_UNIT",
            "SM_MAJOR",
            "PASS1_HEAD_DIM",
        ):
            assert getattr(k, c) == globals()[c], (full, c, getattr(k, c), globals()[c])
        _KERNEL = k
    return _KERNEL


def pass1_forward(*args, **kwargs):
    """The anchor pass, through the kernel `load_kernel` selects (see its docstring there)."""
    return load_kernel().pass1_forward(*args, **kwargs)


# ── glue: the assembled routed prefill ─────────────────────────────────────────────────────
# Everything below consumes the anchor pass above. The adapter hands in its own `_vote_reduce`
# and `_varlen_lse`, so the ballot fold and the FA4 call are the SAME objects the served
# three-piece body uses -- one implementation to drift, not two.

LN2 = 0.6931471805599453
# Elements of the anchor pass's fp32 score buffer per call (hq * query rows * padded units).
# 2^29 = 2 GiB keeps the measured 262k serving shape in ONE pass call (24 heads x 8192 rows x
# 2048 units = 1.5 GiB); a 1M context's last chunk (6 GiB in one call) runs in groups, and the
# loop drops each group's buffer before the next call so the PEAK is one call's buffer.
US_ELEMENT_BUDGET = 1 << 29
# The own pass's row tile. Independent of the generated section's PASS1_BLOCK_M by ownership
# (that constant is rewritten by the generator), so the relation the launch relies on is pinned
# at import: a query block is a whole number of own-pass tiles.
OWN_ROW_TILE = 128
assert PASS1_BLOCK_M % OWN_ROW_TILE == 0, (PASS1_BLOCK_M, OWN_ROW_TILE)


def prefill_scratch_bytes(
    *, hq, hkv, d, stride, rb, q_block, topk, max_positions, chunk_tokens
):
    """An upper bound on what one call of `routed_prefill_gluon` allocates, in bytes, at the
    worst chunk of a `max_positions` context served in `chunk_tokens` chunks.

    None of it is visible to the engine's memory profile (the routed op returns before any
    body runs there), so the adapter reserves this bound during that profile and prints it at
    boot. Every term mirrors an allocation below by name; the sum is a bound, not a peak --
    the groups of the anchor pass free their score buffer before the next group's, and the
    selected pass frees its gathers before the own pass allocates its own, but the bound
    charges all of them. bf16 = 2 B, fp32 = 4 B.
    """
    A = max_positions // stride  # every closed anchor
    a_pad = -(-max(A, PASS1_BLOCK_N) // PASS1_BLOCK_N) * PASS1_BLOCK_N
    u_pad = a_pad // (rb // stride)
    NQ = (
        -(-(chunk_tokens + q_block - 1) // q_block)
    ) * q_block  # pad0 up to q_block-1 rows
    n_qb = NQ // q_block
    G = max(1, US_ELEMENT_BUDGET // (hq * q_block * u_pad))
    rows_g = min(n_qb, G) * q_block  # one anchor-pass group
    kpu = rb - rb // stride  # kept raw tokens per unit
    n_free = max(0, min(topk, max_positions // rb + 1) - 1)  # selected non-own units
    kb = q_block - q_block // stride  # own span's kept columns
    b = 0
    b += 2 * hkv * a_pad * d * 2  # ka, va  (bf16)
    b += NQ * hq * d * 2  # qpad     (bf16)
    b += hq * rows_g * d * 2 + hq * rows_g * 4  # pass1 o, m  (one group)
    b += hq * rows_g * u_pad * 4  # pass1 us    (one group)
    b += hq * NQ * d * 2 + hq * NQ * 4  # o1 (cat), lse1
    b += 2 * n_qb * n_free * kpu * hkv * d * 2  # kg_a, vg_a  (bf16)
    b += NQ * hq * d * 4 + hq * NQ * 4  # o_a, lse_a  (fp32)
    b += 2 * n_qb * kb * hkv * d * 2  # kg, vg      (bf16)
    b += NQ * hq * d * 4 + hq * NQ * 4  # o_b, lse_b  (fp32)
    b += NQ * hq * d * 2  # out_pad     (bf16)
    return int(b)


@triton.jit
def _gather_kv_kernel(
    KC,
    VC,
    BT,
    POS,
    KO,
    VO,
    s_pg,
    s_row,
    s_h,
    s_d,
    so_n,
    so_h,
    page,
    n,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Positions -> cache rows through the block table, k and v in one launch, ANY source
    strides (the paged cache is a view with a doubled page stride) and any dense output layout
    (strides passed, so one kernel fills both the (n, hkv, d) raw shape and the (hkv, A, d)
    anchor shape). torch's fancy indexing falls to its slow elementwise path on exactly this
    shape -- two index tensors x strided source -- measured 3x this kernel per row."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < n
    pos = tl.load(POS + i, mask=ok, other=0)
    pg = pos // page
    ppg = tl.load(BT + pg, mask=ok, other=0).to(tl.int64)
    off = pos - pg * page
    d_off = tl.arange(0, D)
    src = ppg[:, None] * s_pg + off[:, None] * s_row + d_off[None, :] * s_d
    dst = i[:, None] * so_n + d_off[None, :]
    for h in tl.static_range(HKV):
        k = tl.load(KC + src + h * s_h, mask=ok[:, None], other=0.0)
        tl.store(KO + dst + h * so_h, k, mask=ok[:, None])
        v = tl.load(VC + src + h * s_h, mask=ok[:, None], other=0.0)
        tl.store(VO + dst + h * so_h, v, mask=ok[:, None])


def _gather_rows(kc, vc, btrow, page, pos, ko, vo, so_n, so_h):
    assert vc.stride() == kc.stride(), "the cache halves must share a layout"
    n = pos.numel()
    BLOCK = 128
    _gather_kv_kernel[(triton.cdiv(n, BLOCK),)](
        kc,
        vc,
        btrow,
        pos,
        ko,
        vo,
        kc.stride(0),
        kc.stride(1),
        kc.stride(2),
        kc.stride(3),
        so_n,
        so_h,
        page,
        n,
        HKV=kc.shape[2],
        D=kc.shape[3],
        BLOCK=BLOCK,
        num_warps=4,
    )


@triton.jit
def _own_half_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    sq_r,
    sq_h,
    sq_d,
    sk_r,
    sk_h,
    sk_d,
    so_r,
    so_h,
    so_d,
    sl_h,
    sl_r,
    scale,
    GQA: tl.constexpr,
    STRIDE: tl.constexpr,
    QB: tl.constexpr,
    RT: tl.constexpr,
    KB: tl.constexpr,
    KBP: tl.constexpr,
    D: tl.constexpr,
):
    """The OWN pass: per (query block, row tile, head) attention over the block's own kept raw
    columns.

    K/V hold the gathered kept-own columns, KB per block, block-major; the caller's blocks are
    STRIDE-aligned, so a row at block offset r sees the first (r+1) - (r+1)//STRIDE of them —
    always >= 1 (offset 0 is kept), so no empty-row guard is needed.
    """
    pid, h = tl.program_id(0), tl.program_id(1)
    b, t = pid // (QB // RT), pid % (QB // RT)
    rows = t * RT + tl.arange(0, RT)
    grow = b * QB + rows
    d_off = tl.arange(0, D)
    q = tl.load(Q + grow[:, None] * sq_r + h * sq_h + d_off[None, :] * sq_d)
    c = tl.arange(0, KBP)
    kmask = c < KB
    kv = h // GQA
    k = tl.load(
        K + (b * KB + c)[:, None] * sk_r + kv * sk_h + d_off[None, :] * sk_d,
        mask=kmask[:, None],
        other=0.0,
    )
    sc = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
    lim = (rows + 1) - (rows + 1) // STRIDE
    vis = kmask[None, :] & (c[None, :] < lim[:, None])
    sc = tl.where(vis, sc, float("-inf"))
    m = tl.max(sc, 1)
    p = tl.where(vis, tl.exp(sc - m[:, None]), 0.0)
    l = tl.sum(p, 1)
    v = tl.load(
        V + (b * KB + c)[:, None] * sk_r + kv * sk_h + d_off[None, :] * sk_d,
        mask=kmask[:, None],
        other=0.0,
    )
    o = tl.dot(p.to(v.dtype), v).to(tl.float32) / l[:, None]
    tl.store(O + grow[:, None] * so_r + h * so_h + d_off[None, :] * so_d, o)
    tl.store(LSE + h * sl_h + grow * sl_r, m + tl.log(l))


from sglang.kernels.ops.attention.solar_open3.output_gate import _rounded_output_gate


@triton.jit
def _merge3_kernel(
    O1,
    L1,
    O2,
    L2,
    O3,
    L3,
    OUT,
    s1r,
    s1h,
    s1d,
    s2r,
    s2h,
    s2d,
    s3r,
    s3h,
    s3d,
    sl1h,
    sl1r,
    sl2h,
    sl2r,
    sl3h,
    sl3r,
    sor,
    D: tl.constexpr,
    GATE=None,
    stride_g: tl.constexpr = 0,
    PAD0: tl.constexpr = 0,
    TOKENS: tl.constexpr = 0,
    HAS_GATE: tl.constexpr = False,
):
    """One (row, head): fold three (o, natural-log lse) halves. A half at lse=-inf contributes
    nothing even when its o is NaN (empty-softmax rows); all three empty comes out zero."""
    r, h = tl.program_id(0), tl.program_id(1)
    l1 = tl.load(L1 + h * sl1h + r * sl1r)
    l2 = tl.load(L2 + h * sl2h + r * sl2r)
    l3 = tl.load(L3 + h * sl3h + r * sl3r)
    m = tl.maximum(tl.maximum(l1, l2), l3)
    mf = tl.where(m == float("-inf"), 0.0, m)
    w1 = tl.where(l1 == float("-inf"), 0.0, tl.exp(l1 - mf))
    w2 = tl.where(l2 == float("-inf"), 0.0, tl.exp(l2 - mf))
    w3 = tl.where(l3 == float("-inf"), 0.0, tl.exp(l3 - mf))
    den = tl.maximum(w1 + w2 + w3, 1e-30)
    d_off = tl.arange(0, D)
    o1 = tl.load(O1 + r * s1r + h * s1h + d_off * s1d).to(tl.float32)
    o2 = tl.load(O2 + r * s2r + h * s2h + d_off * s2d).to(tl.float32)
    o3 = tl.load(O3 + r * s3r + h * s3h + d_off * s3d).to(tl.float32)
    o1 = tl.where(w1 > 0.0, o1, 0.0)
    o2 = tl.where(w2 > 0.0, o2, 0.0)
    o3 = tl.where(w3 > 0.0, o3, 0.0)
    o = (o1 * w1 + o2 * w2 + o3 * w3) / den
    if HAS_GATE:
        live = (r >= PAD0) & (r < PAD0 + TOKENS)
        o = _rounded_output_gate(o, GATE, (r - PAD0) * stride_g + h * D + d_off, live)
        tl.store(OUT + (r - PAD0) * sor + h * D + d_off, o, live)
    else:
        tl.store(OUT + r * sor + h * D + d_off, o)


def routed_prefill_gluon(
    q,  # (T, hq*d) bf16, this chunk's rows at positions p0..p0+T-1
    p0,
    *,
    kc,
    vc,  # the paged cache halves, (pages, page, hkv, d) -- VIEWS with
    # arbitrary strides (the sub-page re-address keeps the unbound
    # tensor's doubled page stride), so every gather goes through
    # `_gather_kv_kernel` by (physical page, offset), never a reshape.
    btrow,  # this request's block table row (logical page -> physical page)
    page,  # cache page size (the SUB-PAGE the routed path reads, = rb-divisor)
    num_heads,
    num_kv_heads,
    head_dim,
    scaling,
    stride,
    q_block,
    route_block,
    topk,
    sink_units,
    vote_reduce,  # the adapter's `_vote_reduce`
    varlen_lse,  # the adapter's `_varlen_lse`
    max_reduce=None,  # TP: the adapter's MAX all-reduce; None below tp=2
    apos=None,  # the caller's anchor positions, when it has them: bound to the
    # closed form below by count (a host shape read, no sync)
    return_idx=False,
    output_gate=None,
    output=None,
):
    """The fused per-head routed prefill of one chunk (see the module docstring for the three
    pieces). The anchor pass runs first and yields the election scores; the raw pieces follow
    from the identity  (anchors \\ selected-units' anchors) ∪ selected-raw  ==
    all-anchors ∪ (selected-raw \\ anchor-position rows)  -- no selection before the anchor
    pass, no LSE subtraction.

    The ELECTION is the served one: US carries the same block-end-gated per-(head, row, unit)
    maxima `_prefill_score` emits (bit-identical, measured), the fold is `vote_reduce`, the
    ranking runs on the fold's fp32 output exactly as the adapter's `us` buffer does (the dtype
    is part of the election: bf16-rounded ties break differently on 54 of 96 ranking-dependent
    blocks at U=256, measured), and the mask/own/sink/topk lines are transcribed from
    `routed_prefill_batched`.

    ⚠️ No pageable H2D, boolean indexing, or masked_select in here: each is a device sync per
    chunk (a pipeline flush). Every per-block quantity is a closed form of (qb0, n_qb, pe) on a
    device arange; segment lengths are host arithmetic.

    Tensor parallelism is ONE collective: `max_reduce` over the fold's fp32 output. A query head
    lives whole on one rank under either sharding (KV heads partitioned, or a KV head replicated
    with its query group split), so each head's softmax ballot is final where it is computed,
    and the election's own fold across heads is a max -- the MAX of the ranks' local folds IS
    the election, exactly, with no query to reassemble and no `writes_score` bookkeeping (MAX
    is idempotent, so a replicated KV head reaching two ranks changes nothing). The three
    attention pieces then run on the rank's own heads. Nothing in the control flow reads the
    reduced value, so every rank runs the same number of collectives in the same order.
    """
    dev = q.device
    T = q.shape[0]
    hq, hkv, d = num_heads, num_kv_heads, head_dim
    gqa = hq // hkv
    rb = route_block
    pe = p0 + T - 1
    qb0, qb1 = p0 // q_block, pe // q_block
    n_qb = qb1 - qb0 + 1
    A = (pe + 1) // stride
    assert q_block == PASS1_BLOCK_M, "the anchor pass's election gate is its query tile"
    assert q_block % rb == 0 and rb % stride == 0 and topk
    # The three-piece body gathers by the caller's `apos` VALUES; this body derives the anchor
    # set from (pe, stride) in closed form (the kernel's election gate and the unit fold are
    # closed forms of the same pair, so deriving them from `apos` would bind nothing). The two
    # are equal by construction under pool="last"; this pins it. Count only: comparing values
    # would be a device readback per chunk.
    assert apos is None or apos.numel() == A, (
        f"the caller's anchor pool has {apos.numel()} anchors, the closed form (pe+1)//stride "
        f"gives {A}: the two bodies would attend different anchor sets"
    )

    # ── padded block grid, both layouts ─────────────────────────────────────
    pad0 = p0 - qb0 * q_block
    NQ = n_qb * q_block
    qh = q.view(T, hq, d)
    qpad = qh.new_zeros(NQ, hq, d)
    qpad[pad0 : pad0 + T] = qh
    q0_t = (torch.arange(qb0, qb1 + 1, device=dev)) * q_block
    pl_t = (q0_t + q_block - 1).clamp(max=pe)
    present = torch.zeros(NQ, device=dev)
    present[pad0 : pad0 + T] = 1.0

    n_units = pe // rb + 1
    a_pad = -(-max(A, PASS1_BLOCK_N) // PASS1_BLOCK_N) * PASS1_BLOCK_N
    u_pad = a_pad // (rb // stride)
    if A == 0:
        # No closed anchor yet (the chunk ends before position stride-1: the first tokens of a
        # request shorter than the stride). The anchor half is empty and every unit score is
        # -inf; the own pass and the merge below serve the chunk unchanged, and the cross-rank
        # MAX is still issued so the collective count stays rank-uniform. Same body, no switch.
        o1 = qpad.new_zeros(hq, NQ, d)
        lse1 = torch.full((hq, NQ), float("-inf"), device=dev, dtype=torch.float32)
        us = torch.full(
            (n_qb, 1, n_units), float("-inf"), device=dev, dtype=torch.float32
        )
    else:
        # ── the anchor pass: anchor attention + election scores ─────────────────
        # Anchors gathered STRAIGHT into the pass's padded (hkv, A_pad, d) layout -- one kernel,
        # no intermediate copies; only the pad tail is zeroed (the pass reads it as zero keys and
        # values), the gather overwrites everything below A.
        ka = kc.new_empty(hkv, a_pad, d)
        va = vc.new_empty(hkv, a_pad, d)
        if a_pad > A:
            ka[:, A:].zero_()
            va[:, A:].zero_()
        apos_t = torch.arange(A, device=dev) * stride + (stride - 1)
        _gather_rows(
            kc, vc, btrow, page, apos_t, ka, va, so_n=ka.stride(1), so_h=ka.stride(0)
        )

        # The pass's score buffer is (hq * rows, u_pad) fp32 -- at a 1M context's last chunk that
        # is 6 GiB for one call, which the engine's memory profile never sees (routing returns
        # early there). The query blocks are independent (own gate, own fold each), so the pass
        # runs in groups of blocks under `US_ELEMENT_BUDGET`, the anchors gathered once; the
        # three-piece body caps its score buffer the same way (`_SUB_SCORE_BUDGET`). The gathered
        # anchors above are the OTHER allocation the profile never sees: 2 x hkv x a_pad x d bf16
        # per chunk (512 MiB at that 1M chunk), uncapped -- the pass takes dense K/V, and the
        # three-piece body reads its anchors in place through the page table instead. An
        # out-of-memory here propagates: the body was fixed at init, nothing falls back.
        G = max(1, US_ELEMENT_BUDGET // (hq * q_block * u_pad))
        o1_parts, lse1_parts, us_parts = [], [], []
        for g0 in range(0, n_qb, G):
            g1 = min(g0 + G, n_qb)
            r0, r1 = g0 * q_block, g1 * q_block
            q_g = qpad[r0:r1].permute(1, 0, 2).contiguous()  # (hq, NQg, d)
            o1_g, m1_g, us_g = pass1_forward(
                q_g, ka, va, (qb0 + g0) * q_block, scaling, gqa, stride, rb, a_real=A
            )
            o1_parts.append(o1_g[:, : r1 - r0])  # (hq, NQg, d) bf16
            lse1_parts.append(m1_g[:, : r1 - r0] * LN2)  # (hq, NQg) natural log
            # The anchor pass stores US in the fold's own row order -- (query block, kv head, gqa
            # head, row-in-block) -- so the (blocks, votes, units) view is free. The vote ORDER
            # differs from the served (kv, row, gqa): legal, because each vote's smax is
            # independent and the fold is an amax across votes -- only `cnt_vote` must pair with
            # its vote, built here in the same order. The fold runs over the PADDED unit axis:
            # padded columns are -inf everywhere and fold to -inf, which the `vals == -inf`
            # sentinel line strips like any unit no vote scored.
            sq = us_g.view(-1, hkv * gqa * q_block, u_pad)[: g1 - g0]
            cnt_vote = (
                present[r0:r1]
                .view(g1 - g0, 1, q_block)
                .expand(g1 - g0, hkv * gqa, q_block)
                .reshape(g1 - g0, -1)
                .contiguous()
            )
            us_parts.append(vote_reduce(sq, cnt_vote, reduce="smax"))
            del (
                us_g,
                sq,
                o1_g,
                m1_g,
            )  # the next call's buffer must reuse this block, not sit beside it
        o1 = o1_parts[0] if len(o1_parts) == 1 else torch.cat(o1_parts, 1)
        lse1 = lse1_parts[0] if len(lse1_parts) == 1 else torch.cat(lse1_parts, 1)
        # A row with no closed anchor (position < stride-1) comes back NaN, not -inf, and
        # `maximum(NaN, x)` in a merge would keep it. Empty means -inf here.
        lse1 = torch.where(
            torch.isnan(lse1), torch.full_like(lse1, float("-inf")), lse1
        )

        # ── the served fold's output -> the served mask/own/sink/topk ───────────
        # Ranked in the fold's fp32, as the adapter's `us` buffer is: the dtype is part of the
        # election (bf16-rounded ties break by index differently -- 54 of 96 ranking-dependent
        # blocks at U=256, measured), and the decode ranks its `usf` in fp32 too.
        us = (us_parts[0] if len(us_parts) == 1 else torch.cat(us_parts, 0))[:, None, :]
        if u_pad < n_units:  # a unit past every closed
            us = torch.nn.functional.pad(
                us,
                (0, n_units - u_pad),  # anchor scores -inf
                value=float("-inf"),
            )
    us = us[..., :n_units]
    if max_reduce is not None:
        # The cross-rank combine (see the docstring): the max of finished ballots, taken on
        # the fp32 fold BEFORE the forcing lines below, which are rank-invariant. `-inf` slots
        # (padded units, units past the row) survive a MAX unchanged.
        us = max_reduce(us.contiguous())

    unit = torch.arange(n_units, device=dev)
    _pl, _q0b = pl_t[:, None, None], q0_t[:, None, None]
    us = us.masked_fill(unit * rb > _pl, float("-inf"))
    own = (unit * rb <= _pl) & ((unit + 1) * rb > _q0b)
    if sink_units:
        own = own | (unit < sink_units) & (unit * rb <= _pl)
    us = us.masked_fill(own, float("inf"))
    k_eff = min(topk, n_units)
    vals, idx = us.topk(k_eff, -1)
    idx = idx.masked_fill(vals == float("-inf"), n_units)

    # ── the SELECTED pass: non-own selected units' raw tokens minus anchor rows ──
    # Segment LENGTHS are host arithmetic (the adapter's own `nrf` closed form): sorted
    # ascending, the non-own selected units are exactly the first  min(topk, nua) - n_own
    # entries -- own units rank above own_lo and sentinels above everything. Only unit
    # IDENTITIES flow device-side; nothing is read back.
    p_last = [min((b + 1) * q_block - 1, pe) for b in range(qb0, qb1 + 1)]
    q0_list = [b * q_block for b in range(qb0, qb1 + 1)]
    nua = [pl // rb + 1 for pl in p_last]
    n_own = [x - q0 // rb for x, q0 in zip(nua, q0_list)]
    n_free = [min(topk, x) - o for x, o in zip(nua, n_own)]
    kpu = rb - rb // stride
    max_cnt = max(n_free)
    if (
        max_cnt > 0
    ):  # `> 0`, as `c <= 0` below: a negative count is the same guard's case
        sel_sorted = idx[:, 0].sort(-1).values[:, :max_cnt]  # non-own come first
        keep_rb = torch.arange(kpu, device=dev)  # closed form, no nonzero
        keep_rb = (keep_rb // (stride - 1)) * stride + keep_rb % (stride - 1)
        pos = (sel_sorted[:, :, None] * rb + keep_rb[None, None, :]).clamp(
            min=0, max=pe
        )
        pos = pos.reshape(n_qb, -1)
        # Prefix slices with host-known lengths, one cat kernel; an empty block keeps one dummy
        # unit so no varlen segment is zero-length (a zero seqlen poisons the kernel), and its
        # lse is forced back to -inf below.
        gat = torch.cat([pos[i, : max(c, 1) * kpu] for i, c in enumerate(n_free)])
        cnt_t = (pl_t // rb + 1).clamp(max=topk) - (
            pl_t // rb + 1 - q0_t // rb
        )  # == n_free, device-side
        cu_k = torch.cat([cnt_t.new_zeros(1), (cnt_t.clamp(min=1) * kpu).cumsum(0)]).to(
            torch.int32
        )
        cu_q = torch.arange(n_qb + 1, device=dev, dtype=torch.int32) * q_block
        kg_a = kc.new_empty(gat.numel(), hkv, d)
        vg_a = vc.new_empty(gat.numel(), hkv, d)
        _gather_rows(
            kc,
            vc,
            btrow,
            page,
            gat,
            kg_a,
            vg_a,
            so_n=kg_a.stride(0),
            so_h=kg_a.stride(1),
        )
        o_a, lse_a = varlen_lse(
            qpad,
            kg_a,
            vg_a,
            cu_q=cu_q,
            max_q=q_block,
            cu_k=cu_k,
            max_k=max_cnt * kpu,
            scale=scaling,
        )
        for i, c in enumerate(n_free):  # dummy segments out
            if (
                c <= 0
            ):  # `<= 0`, as the three-piece body's `nrf` guard: a negative count
                lse_a[:, i * q_block : (i + 1) * q_block] = float(
                    "-inf"
                )  # also kept one dummy
    else:
        o_a = torch.zeros(NQ, hq, d, device=dev, dtype=torch.float32)
        lse_a = torch.full((hq, NQ), float("-inf"), device=dev, dtype=torch.float32)

    # ── the OWN pass: the own span's kept raw columns, causal ───────────────
    kb = q_block - q_block // stride
    keep = torch.arange(kb, device=dev)
    keep = (keep // (stride - 1)) * stride + keep % (
        stride - 1
    )  # kept offsets, closed form
    colpos = (q0_t[:, None] + keep[None, :]).clamp(max=pe).reshape(-1)
    kg = kc.new_empty(colpos.numel(), hkv, d)  # (n_qb*KB, hkv, d)
    vg = vc.new_empty(colpos.numel(), hkv, d)
    _gather_rows(
        kc, vc, btrow, page, colpos, kg, vg, so_n=kg.stride(0), so_h=kg.stride(1)
    )
    o_b = torch.empty(NQ, hq, d, device=dev, dtype=torch.float32)
    lse_b = torch.empty(hq, NQ, device=dev, dtype=torch.float32)
    RT = OWN_ROW_TILE
    assert q_block % RT == 0, (
        f"the own pass tiles each query block by {RT} rows: q_block={q_block} would leave "
        f"{q_block % RT} rows of o_b/lse_b unwritten (torch.empty) for the merge"
    )
    _own_half_kernel[(n_qb * (q_block // RT), hq)](
        qpad,
        kg,
        vg,
        o_b,
        lse_b,
        qpad.stride(0),
        qpad.stride(1),
        qpad.stride(2),
        kg.stride(0),
        kg.stride(1),
        kg.stride(2),
        o_b.stride(0),
        o_b.stride(1),
        o_b.stride(2),
        lse_b.stride(0),
        lse_b.stride(1),
        scaling,
        GQA=gqa,
        STRIDE=stride,
        QB=q_block,
        RT=RT,
        KB=kb,
        KBP=triton.next_power_of_2(kb),
        D=d,
        num_warps=8,
    )

    # ── merge (rounded to the query dtype, as the served merge is) ──────────
    out_pad = (
        torch.empty(NQ, hq * d, device=dev, dtype=q.dtype)
        if output_gate is None
        else (
            torch.empty(T, hq * d, device=dev, dtype=q.dtype)
            if output is None
            else output
        )
    )
    _merge3_kernel[(NQ, hq)](
        o1,
        lse1,
        o_a,
        lse_a,
        o_b,
        lse_b,
        out_pad,
        o1.stride(1),
        o1.stride(0),
        o1.stride(2),  # o1 is (hq, NQ, d)
        o_a.stride(0),
        o_a.stride(1),
        o_a.stride(2),
        o_b.stride(0),
        o_b.stride(1),
        o_b.stride(2),
        lse1.stride(0),
        lse1.stride(1),
        lse_a.stride(0),
        lse_a.stride(1),
        lse_b.stride(0),
        lse_b.stride(1),
        out_pad.stride(0),
        D=d,
        GATE=output_gate,
        stride_g=output_gate.stride(0) if output_gate is not None else 0,
        PAD0=pad0,
        TOKENS=T,
        HAS_GATE=output_gate is not None,
        num_warps=1,
    )
    out = out_pad[pad0 : pad0 + T] if output_gate is None else out_pad
    if output is not None and output_gate is None:
        output.copy_(out)
        out = output
    return (out, idx) if return_idx else out
