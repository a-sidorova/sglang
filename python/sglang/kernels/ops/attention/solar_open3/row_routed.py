# Copyright 2026 Upstage AI.
# SPDX-License-Identifier: Apache-2.0
"""Per-row SolarOpen3 routing and FA4 prefill over SGLang's paged KV cache.

Serving-only functions carried from simple-trainer's
simple_trainer/models/components/row_routed_attn.py at
59bd6fba62c1d689a257a9f9d66f318437fbfeb9. The router uses each row's own query;
per-head ballots follow the trainer's exact softmax recurrence. The causal
anchor and selected-token halves merge under one softmax, with anchor dedup.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.solar_open3.output_gate import (
    _rounded_output_gate,
    _validate_output_gate,
)

_LOG2E = 1.4426950408889634
ROUTE_CFG = dict(bs=64, warps=8, stages=3)
QHEAD_CFG = dict(warps=4, stages=2, m=128, rows_per_program=None)
QHEAD_CFG["rows_per_program"] = lambda gp: max(1, QHEAD_CFG["m"] // gp)
FA4SEL_RPB = 42
_CUTE_SEL_MASKS: dict = {}
_CUTE_MASKS: dict = {}


@triton.jit
def _rr_merge2_kernel(
    OA,
    LA,
    OS,
    LS,
    OUT,
    LSE,
    T,
    HQ: tl.constexpr,
    D: tl.constexpr,
    HB: tl.constexpr,
    GATE=None,
    stride_g: tl.constexpr = 0,
    HAS_GATE: tl.constexpr = False,
):
    """o = LSE-merge of the anchor half (OA, LA ``(1, Hq, T)``) and the selected half (OS, LS ``(Hq, T)``)."""
    r = tl.program_id(0)
    hh = tl.program_id(1) * HB + tl.arange(0, HB)
    hm = hh < HQ  # a rank's head count need not divide by HB (SO3 at TP 4/8)
    dd = tl.arange(0, D)
    la = tl.load(LA + hh.to(tl.int64) * T + r, mask=hm, other=float("-inf"))
    ls = tl.load(LS + hh.to(tl.int64) * T + r, mask=hm, other=float("-inf"))
    mx = tl.maximum(la, ls)
    mxs = tl.where(mx == float("-inf"), 0.0, mx)
    wa = tl.exp(la - mxs)
    ws = tl.exp(ls - mxs)
    den = wa + ws
    off = (r.to(tl.int64) * HQ + hh)[:, None] * D + dd[None, :]
    oa = tl.where(
        (la == float("-inf"))[:, None],
        0.0,
        tl.load(OA + off, mask=hm[:, None], other=0.0).to(tl.float32),
    )
    os_ = tl.where(
        (ls == float("-inf"))[:, None],
        0.0,
        tl.load(OS + off, mask=hm[:, None], other=0.0).to(tl.float32),
    )
    o = (wa[:, None] * oa + ws[:, None] * os_) / tl.where(den == 0.0, 1.0, den)[:, None]
    if HAS_GATE:
        o = _rounded_output_gate(
            o, GATE, r * stride_g + hh[:, None] * D + dd[None, :], hm[:, None]
        )
    tl.store(OUT + off, o.to(OUT.dtype.element_ty), mask=hm[:, None])
    tl.store(
        LSE + hh.to(tl.int64) * T + r,
        tl.where(den == 0.0, float("-inf"), mxs + tl.log(den)),
        mask=hm,
    )


@triton.jit
def _kv_unit_gather_kernel(
    KC,
    VC,
    BT,
    KP,
    VP,
    NTOK,
    NVALID,
    PAGE,
    sKb,
    sKt,
    HD: tl.constexpr,
    BT_: tl.constexpr,
):
    """K / V of tokens 0 .. NTOK-1 of one request out of the paged cache into contiguous (NTOK, Hkv * D) buffers
    (tokens past NVALID repeat the last written one: only ever masked). One program = BT_ tokens."""
    t = tl.program_id(0) * BT_ + tl.arange(0, BT_)
    tm = t < NTOK
    tc = tl.minimum(t, NVALID - 1)
    pg = tl.load(BT + tc // PAGE, mask=tm, other=0).to(tl.int64)
    src = pg * sKb + (tc % PAGE).to(tl.int64) * sKt
    cc = tl.arange(0, HD)
    dst = t.to(tl.int64)[:, None] * HD + cc[None, :]
    tl.store(
        KP + dst,
        tl.load(KC + src[:, None] + cc[None, :], mask=tm[:, None]),
        mask=tm[:, None],
    )
    tl.store(
        VP + dst,
        tl.load(VC + src[:, None] + cc[None, :], mask=tm[:, None]),
        mask=tm[:, None],
    )


@triton.jit
def _route_serve_kernel(
    Q,
    KA,
    ADM,
    SINK,
    OUT,
    ULO,
    UHI,
    FLO,
    FHI,
    T,
    ROFF,
    A,
    NQB,
    q_start,
    sQr,
    sQh,
    sA,
    sH,
    s_am,
    s_aa,
    s_sm,
    so_r,
    scale,
    QB: tl.constexpr,
    RB: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    APU: tl.constexpr,
    UB: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """The serving router in one kernel: ``_row_route_score_kernel`` + the GQA head mean (fp32 head-order sum,
    round-to-nearest divide, bf16 -- the ``_route_head_mean`` contract, so the scores are those of the training
    router) read straight from the chunk's raw ``q`` ``(T, Hq, D)`` and the anchors ``(A, Hkv, D)`` by stride: no
    tile padding, transpose or head-mean passes. Rows are tile-padded coordinates; row r is q row r - ROFF.
    ``q`` may be a view of the fused qkv projection (SGLang splits it without a copy): rows and heads by stride."""
    pid = tl.program_id(0)
    r0 = pid * BS
    qb = r0 // QB
    rows = r0 + tl.arange(0, BS)
    qr = rows - ROFF
    rmask = (qr >= 0) & (qr < T)
    dd = tl.arange(0, D)
    G: tl.constexpr = HQ // HKV
    ulo = tl.load(ULO + qb)
    uhi = tl.load(UHI + qb)
    NA: tl.constexpr = UB * APU
    for h in tl.static_range(HKV):
        qacc = tl.load(
            Q + (qr.to(tl.int64) * sQr + h * G * sQh)[:, None] + dd[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        for g in tl.static_range(1, G):
            qacc = qacc + tl.load(
                Q + (qr.to(tl.int64) * sQr + (h * G + g) * sQh)[:, None] + dd[None, :],
                mask=rmask[:, None],
                other=0.0,
            ).to(tl.float32)
        q = tl.div_rn(qacc, tl.full(qacc.shape, G, tl.float32)).to(Q.dtype.element_ty)
        for u0 in range(ulo, uhi, UB):
            an = u0 * APU + tl.arange(0, NA)
            amask = an < tl.minimum(uhi * APU, A)
            gate = tl.load(ADM + qb * s_am + an * s_aa, mask=amask, other=0)
            k = tl.load(
                KA + an[:, None].to(tl.int64) * sA + h * sH + dd[None, :],
                mask=amask[:, None],
                other=0.0,
            )
            sc = (tl.dot(q, tl.trans(k)).to(tl.float32) * scale).to(tl.float32)
            sc = tl.where((gate[None, :] != 0) & amask[None, :], sc, float("-inf")).to(
                tl.float32
            )
            m = tl.max(tl.reshape(sc, (BS, UB, APU)), axis=2)
            uu = u0 + tl.arange(0, UB)
            optr = OUT + rows[:, None].to(tl.int64) * so_r + uu[None, :]
            om = (uu < uhi)[None, :] & (rows >= 0)[:, None]
            if h > 0:
                m = tl.maximum(m, tl.load(optr, mask=om, other=float("-inf")))
            tl.store(optr, m, mask=om)
    flo = tl.load(FLO + qb)
    fhi = tl.load(FHI + qb)
    qb_first = q_start + qb * QB
    p = qb_first + (rows - qb * QB)
    own_lo = qb_first // RB
    for u0 in range(flo, fhi, UB):
        uu = u0 + tl.arange(0, UB)
        optr = OUT + rows[:, None].to(tl.int64) * so_r + uu[None, :]
        om = (uu < fhi)[None, :] & (rows >= 0)[:, None]
        v = tl.load(optr, mask=om, other=float("-inf"))
        adm = (uu[None, :] * RB) <= p[:, None]
        forced = adm & (uu[None, :] >= own_lo)
        if HAS_SINK:
            sk = tl.load(SINK + qb * s_sm + uu, mask=uu < fhi, other=0)
            forced = forced | (adm & (sk[None, :] != 0))
        v = tl.where(adm, v, float("-inf"))
        v = tl.where(forced, float("inf"), v)
        tl.store(optr, v, mask=om)


@triton.jit
def _qhead_tile_max(
    q,
    KA,
    ADM,
    b,
    h,
    u0,
    uhi,
    A_LIM,
    qb,
    p_m,
    sk_b,
    sk_h,
    sk_a,
    s_ab,
    s_am,
    s_aa,
    scale,
    D: tl.constexpr,
    BM: tl.constexpr,
    UB: tl.constexpr,
    APU: tl.constexpr,
    STRIDE: tl.constexpr,
    MASKED: tl.constexpr,
):
    """(BM, UB): per (row x query head, unit of the tile at ``u0``) the max of ``q . k`` over the unit's admitted
    anchors (-inf if none). ``MASKED`` applies the tile gate and each row's causal bound; the caller passes False
    for tiles wholly before the program's first row inside its document, where every anchor is admitted."""
    NA: tl.constexpr = UB * APU
    dd = tl.arange(0, D)
    an = u0 * APU + tl.arange(0, NA)
    amask = (an < uhi * APU) & (an < A_LIM)
    k = tl.load(
        KA + b * sk_b + h * sk_h + an[:, None].to(tl.int64) * sk_a + dd[None, :],
        mask=amask[:, None],
        other=0.0,
    )
    sc = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
    if MASKED:
        gate = tl.load(ADM + b * s_ab + qb * s_am + an * s_aa, mask=amask, other=0)
        ok = (amask & (gate != 0))[None, :] & (
            (an * STRIDE + STRIDE - 1)[None, :] <= p_m[:, None]
        )
    else:
        ok = amask[None, :] & (p_m >= 0)[:, None]
    return tl.max(tl.reshape(tl.where(ok, sc, float("-inf")), (BM, UB, APU)), axis=2)


@triton.jit
def _ballot_stats_tile(um, m_run, z, UB: tl.constexpr):
    """The election's (M, Z) recurrence over one tile of UB units: the canonical order -- tiles of UB units
    from the document's first unit, the tile max first, then each unit's term added in unit order -- that the
    serving decode's ``EXACT`` stats repeat, so all paths hold bit-identical (M, Z)."""
    m_new = tl.maximum(m_run, tl.max(um, 1))
    # exactly 1 where the max did not move (incl. -inf == -inf): the decode walks trailing all--inf tiles
    alpha = tl.where(m_new == m_run, 1.0, tl.exp(m_run - m_new))
    e = tl.where(um == float("-inf"), 0.0, tl.exp(um - m_new[:, None]))
    z = z * alpha
    cols = tl.arange(0, UB)
    for j in tl.static_range(UB):
        z = z + tl.sum(tl.where((cols == j)[None, :], e, 0.0), 1)
    return m_new, z


@triton.jit
def _ballot_fold_tile(
    um,
    m_run,
    z,
    OUT,
    b,
    so_b,
    so_r,
    rows,
    rmask,
    u0,
    uhi,
    h,
    BS: tl.constexpr,
    GP: tl.constexpr,
    UB: tl.constexpr,
):
    """The tile's ballots ``exp(s - M) / Z``, max over the program's query heads and (through OUT, this
    program's rows only) over the KV heads already folded."""
    bal = tl.where(
        um == float("-inf"), float("-inf"), tl.exp(um - m_run[:, None]) / z[:, None]
    )
    m = tl.max(tl.reshape(bal, (BS, GP, UB)), axis=1)  # (BS, UB)
    uu = u0 + tl.arange(0, UB)
    optr = OUT + b * so_b + rows[:, None].to(tl.int64) * so_r + uu[None, :]
    om = rmask[:, None] & (uu < uhi)[None, :]
    if h > 0:
        m = tl.maximum(m, tl.load(optr, mask=om, other=float("-inf")))
    tl.store(optr, m, mask=om)


@triton.jit
def _row_route_qhead_kernel(
    Q,
    KA,
    ADM,
    SINK,
    OUT,
    ULO,
    UHI,
    FLO,
    FHI,
    n_rows,
    ROFF,
    T_VALID,
    A_LIM,
    NQB,
    q_start,
    q_block_stride,
    sq_b,
    sq_h,
    sq_r,
    sk_b,
    sk_h,
    sk_a,
    s_ab,
    s_am,
    s_aa,
    s_sb,
    s_sm,
    so_b,
    so_r,
    scale,
    QB: tl.constexpr,
    RB: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    APU: tl.constexpr,
    UB: tl.constexpr,
    STRIDE: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """PER-HEAD election (cfg.kv_route_qhead_block=1 with kv_route_per_row), training's and the serving
    prefill's: per row and QUERY head, a unit's raw score is the max of ``q_h . k`` over the unit's anchors at
    or before the row; each head's scores are softmaxed over the units (a ballot), and the unit's score is its
    best ballot over every head -- the serving decode's election, one top-k per row shared by all heads.

    The ballot is ``exp(s - M) / Z`` (the decode fold's expression), with ``M`` / ``Z`` accumulated in one
    canonical order (``_ballot_stats_tile``) that the decode's per-row stats repeat. Two passes over the
    anchors (stats, then ballots), no per-head buffer: a program packs BS rows x GP (the GQA group, padded to a
    power of two) query heads into the MMA's M. Tiles wholly before the program's first row skip the masks.
    Then the same -inf / +inf forcing as ``_row_route_score_kernel``, AFTER the fold (a +inf before a softmax
    would one-hot a ballot). Rows are tile-padded coordinates; row r reads query row r - ROFF (serving
    chunks), valid below T_VALID."""
    pid = tl.program_id(0)
    b = tl.program_id(1)
    r0 = pid * BS
    qb = r0 // QB
    rows = r0 + tl.arange(0, BS)
    rmask = rows < n_rows
    qb_first = q_start + (qb * q_block_stride) * QB
    p = qb_first + (rows - qb * QB)  # each row's global position
    BM: tl.constexpr = BS * GP
    mm = tl.arange(0, BM)
    g = mm % GP
    rows_m = r0 + mm // GP
    qr_m = rows_m - ROFF
    valid_m = (g < G) & (qr_m >= 0) & (qr_m < T_VALID) & (rows_m < n_rows)
    p_m = tl.where(
        valid_m, qb_first + (rows_m - qb * QB), -1
    )  # -1: a padded head / row admits nothing
    dd = tl.arange(0, D)
    ulo = tl.load(ULO + b * NQB + qb)
    uhi = tl.load(UHI + b * NQB + qb)
    # Tiles in [ulo + UB, split) lie wholly before the program's first row and inside the tile's document range:
    # every anchor is admitted, no masks. The first tile stays masked -- a document starting mid-unit leaves the
    # previous document's anchors in unit ulo -- and so does everything from the program's first row on.
    first = tl.minimum(ulo + UB, uhi)
    split = tl.maximum(
        ulo
        + (
            tl.maximum(tl.minimum(r0 + qb_first - qb * QB, uhi * RB) // RB - ulo, 0)
            // UB
        )
        * UB,
        first,
    )
    for h in tl.static_range(HKV):
        q = tl.load(
            Q
            + b * sq_b
            + (h * G + g)[:, None] * sq_h
            + qr_m[:, None].to(tl.int64) * sq_r
            + dd[None, :],
            mask=valid_m[:, None],
            other=0.0,
        )
        m_run = tl.full((BM,), float("-inf"), tl.float32)
        z = tl.zeros((BM,), tl.float32)
        # pass 1: (M, Z), tiles in order (the canonical recurrence)
        for u0 in range(ulo, first, UB):
            um = _qhead_tile_max(
                q,
                KA,
                ADM,
                b,
                h,
                u0,
                uhi,
                A_LIM,
                qb,
                p_m,
                sk_b,
                sk_h,
                sk_a,
                s_ab,
                s_am,
                s_aa,
                scale,
                D,
                BM,
                UB,
                APU,
                STRIDE,
                True,
            )
            m_run, z = _ballot_stats_tile(um, m_run, z, UB)
        for u0 in range(first, split, UB):
            um = _qhead_tile_max(
                q,
                KA,
                ADM,
                b,
                h,
                u0,
                uhi,
                A_LIM,
                qb,
                p_m,
                sk_b,
                sk_h,
                sk_a,
                s_ab,
                s_am,
                s_aa,
                scale,
                D,
                BM,
                UB,
                APU,
                STRIDE,
                False,
            )
            m_run, z = _ballot_stats_tile(um, m_run, z, UB)
        for u0 in range(split, uhi, UB):
            um = _qhead_tile_max(
                q,
                KA,
                ADM,
                b,
                h,
                u0,
                uhi,
                A_LIM,
                qb,
                p_m,
                sk_b,
                sk_h,
                sk_a,
                s_ab,
                s_am,
                s_aa,
                scale,
                D,
                BM,
                UB,
                APU,
                STRIDE,
                True,
            )
            m_run, z = _ballot_stats_tile(um, m_run, z, UB)
        # pass 2: ballots, folded over heads
        for u0 in range(ulo, first, UB):
            um = _qhead_tile_max(
                q,
                KA,
                ADM,
                b,
                h,
                u0,
                uhi,
                A_LIM,
                qb,
                p_m,
                sk_b,
                sk_h,
                sk_a,
                s_ab,
                s_am,
                s_aa,
                scale,
                D,
                BM,
                UB,
                APU,
                STRIDE,
                True,
            )
            _ballot_fold_tile(
                um, m_run, z, OUT, b, so_b, so_r, rows, rmask, u0, uhi, h, BS, GP, UB
            )
        for u0 in range(first, split, UB):
            um = _qhead_tile_max(
                q,
                KA,
                ADM,
                b,
                h,
                u0,
                uhi,
                A_LIM,
                qb,
                p_m,
                sk_b,
                sk_h,
                sk_a,
                s_ab,
                s_am,
                s_aa,
                scale,
                D,
                BM,
                UB,
                APU,
                STRIDE,
                False,
            )
            _ballot_fold_tile(
                um, m_run, z, OUT, b, so_b, so_r, rows, rmask, u0, uhi, h, BS, GP, UB
            )
        for u0 in range(split, uhi, UB):
            um = _qhead_tile_max(
                q,
                KA,
                ADM,
                b,
                h,
                u0,
                uhi,
                A_LIM,
                qb,
                p_m,
                sk_b,
                sk_h,
                sk_a,
                s_ab,
                s_am,
                s_aa,
                scale,
                D,
                BM,
                UB,
                APU,
                STRIDE,
                True,
            )
            _ballot_fold_tile(
                um, m_run, z, OUT, b, so_b, so_r, rows, rmask, u0, uhi, h, BS, GP, UB
            )
    flo = tl.load(FLO + b * NQB + qb)
    fhi = tl.load(FHI + b * NQB + qb)
    own_lo = qb_first // RB
    for u0 in range(flo, fhi, UB):
        uu = u0 + tl.arange(0, UB)
        optr = OUT + b * so_b + rows[:, None].to(tl.int64) * so_r + uu[None, :]
        om = rmask[:, None] & (uu < fhi)[None, :]
        v = tl.load(optr, mask=om, other=float("-inf"))
        adm = (uu[None, :] * RB) <= p[:, None]
        forced = adm & (uu[None, :] >= own_lo)
        if HAS_SINK:
            sk = tl.load(SINK + b * s_sb + qb * s_sm + uu, mask=uu < fhi, other=0)
            forced = forced | (adm & (sk[None, :] != 0))
        v = tl.where(adm, v, float("-inf"))
        v = tl.where(forced, float("inf"), v)
        tl.store(optr, v, mask=om)


def _selected_cute_mask(
    rpb: int, rb: int, stride: int, dedup: bool, *, use_vec32: bool
):
    """The selected half on FA4's paged path: batch b = a block of ``rpb`` rows, its KV = the block's UNION of units as
    ``rb``-token pages (page table = the union list). Key kv of batch b is token ``unit * rb + kv % rb`` of unit
    ``ulist[b, kv // rb]``; it is admitted for row ``b * rpb + q`` iff that row chose the unit (``tab``), it is causal
    (``<= gpos[row]``) and, with dedup, not an anchor position. The unit index is constant over each page, so the
    derived lookups are constant over FA4's 32-column mask chunks."""
    key = (rpb, rb, stride, dedup, use_vec32)
    if key in _CUTE_SEL_MASKS:
        return _CUTE_SEL_MASKS[key]
    import operator

    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def ssa_to_fragment(ssa_value, dtype):
        if cutlass.const_expr(ssa_value.dtype != dtype):
            ssa_value = ssa_value.to(dtype)
        frag = cute.make_rmem_tensor(ssa_value.shape, dtype)
        frag.store(ssa_value)
        return frag

    @cute.jit
    def mask_mod(
        b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars=None
    ):
        ul = aux_tensors[0]
        tab = aux_tensors[1]
        gp = aux_tensors[2]
        fb = ssa_to_fragment(b_idx, cutlass.Int32)
        fpg = ssa_to_fragment(kv_idx // rb, cutlass.Int32)
        tu = cute.make_rmem_tensor(cute.size(fb.shape), cutlass.Int32)
        for i in cutlass.range(cute.size(tu.shape), unroll_full=True):
            tu[i] = ul[fb[i], fpg[i]]
        u = tu.load()
        row = b_idx * rpb + q_idx
        fr = ssa_to_fragment(row, cutlass.Int32)
        fu = ssa_to_fragment(u, cutlass.Int32)
        tb = cute.make_rmem_tensor(cute.size(fr.shape), cutlass.Int32)
        tp = cute.make_rmem_tensor(cute.size(fr.shape), cutlass.Int32)
        for i in cutlass.range(cute.size(tb.shape), unroll_full=True):
            tb[i] = tab[fr[i], fu[i]]
            tp[i] = gp[fr[i]]
        tok = u * rb + kv_idx % rb
        ok = operator.ne(tb.load(), 0) & operator.le(tok, tp.load())
        if cutlass.const_expr(dedup):
            ok = ok & operator.ne(tok % stride, stride - 1)
        return ok

    mask_mod.__cute_hash__ = (
        f"solar_rowroute_selected_paged_mask_{rpb}_{rb}_{stride}_{int(dedup)}"
    )
    if use_vec32 and 32 % stride == 0 and rb % 32 == 0:
        # VECTORIZED (FA4 hands 32 adjacent kv of one row, wants one packed keep word): within a 32-column chunk the
        # unit, the row's bit and the dedup pattern are constants -- one lookup each instead of three per element.
        from flash_attn.cute import utils as fa_utils

        pat = 0
        for j in range(32):
            if not dedup or j % stride != stride - 1:
                pat |= 1 << j

        @cute.jit
        def mask_vec(
            b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars=None
        ):
            ul = aux_tensors[0]
            tab = aux_tensors[1]
            gp = aux_tensors[2]
            fb = ssa_to_fragment(b_idx, cutlass.Int32)
            fq = ssa_to_fragment(q_idx, cutlass.Int32)
            fk = ssa_to_fragment(kv_idx, cutlass.Int32)
            b = fb[0]
            kv0 = fk[0]
            u = ul[b, kv0 // rb]
            row = b * rpb + fq[0]
            tok0 = u * rb + kv0 % rb
            lim = gp[row] - tok0 + 1  # columns kept: tok0 + j <= gpos[row]
            sh = min(max(32 - lim, 0), 32)  # DSL builtins, as r2p_bitmask_below
            keep = fa_utils.shr_u32(cutlass.Uint32(0xFFFFFFFF), cutlass.Uint32(sh))
            keep = keep & cutlass.Uint32(pat)
            out = cute.make_rmem_tensor(1, cutlass.Uint32)
            out[0] = keep if tab[row, u] != 0 else cutlass.Uint32(0)
            return out.load()

        mask_vec.__vec_size__ = 32
        mask_vec.__cute_hash__ = mask_mod.__cute_hash__ + "_vec32"
        mask_mod = mask_vec
    _CUTE_SEL_MASKS[key] = mask_mod
    return mask_mod


def selected_forward_fa4paged(
    q,
    sel,
    gpos,
    o_a,
    lse_a,
    *,
    stride,
    route_block,
    dedup,
    scale,
    NU,
    pages,
    output_gate=None,
    output=None,
):
    """Selected half on FA4 itself (paged KV, GQA packing, mask_mod), merged with the anchor half. ONE request:
    ``q`` ``(1, T, Hq, D)``, ``sel`` ``(1, T, KS)``, ``pages`` = the request's prefix K / V, contiguous ``(NU *
    route_block, Hkv * D)`` (positions past the written prefix are masked by causality). Rows go in blocks of ``rpb`` (42 rows x 6 heads = FA4's 256-row tile); each block attends to the
    union of its rows' units (real activations: 55.6 units at 42 rows vs 32 per row) with its own rows' bits."""
    from flash_attn.cute.interface import _flash_attn_fwd

    _, T, HQ, D = q.shape
    HKV = pages[0].shape[-1] // D
    rpb = FA4SEL_RPB
    n_blk = (T + rpb - 1) // rpb
    Tp = n_blk * rpb
    dev = q.device
    tab = torch.zeros(Tp + rpb, NU + 1, dtype=torch.int32, device=dev)
    tab[:T].scatter_(1, torch.where(sel[0] >= 0, sel[0], NU).long(), 1)
    tab = tab[:, :NU].contiguous()
    union = tab[:Tp].view(n_blk, rpb, NU).amax(1)
    ucnt = union.sum(-1, dtype=torch.int32)
    ulist = (
        torch.argsort(union, dim=-1, descending=True, stable=True)
        .to(torch.int32)
        .contiguous()
    )
    gp = torch.full((Tp + rpb,), -1, dtype=torch.int32, device=dev)
    gp[:T] = gpos
    kp, vp = (t.view(NU, route_block, HKV, D) for t in pages)  # rb-token unit pages
    qb = q[0] if Tp == T else torch.cat([q[0], q.new_zeros(Tp - T, HQ, D)], 0)
    qb = qb.view(n_blk, rpb, HQ, D)
    # [:2]: FA4 4.0.0b18 returns (out, lse), b19 (the SGLang venvs') appends (p, row_max)
    o_s, lse_s = _flash_attn_fwd(
        qb,
        kp,
        vp,
        page_table=ulist,
        seqused_k=ucnt * route_block,
        softmax_scale=scale,
        return_lse=True,
        mask_mod=_selected_cute_mask(
            rpb,
            route_block,
            stride,
            dedup,
            # Limit vec32 to data-center Blackwell (B200/B300, CC 10.x).
            use_vec32=torch.cuda.get_device_capability(q.device)[0] == 10,
        ),
        aux_tensors=[ulist, tab, gp],
    )[:2]
    o_s = o_s.view(Tp, HQ, D)[:T].contiguous()
    lse_s = lse_s.permute(1, 0, 2).reshape(HQ, Tp)[:, :T].contiguous()  # (Hq, T)
    out = torch.empty_like(q) if output is None else output.view_as(q)
    lse = torch.empty((1, HQ, T), device=dev, dtype=torch.float32)
    _rr_merge2_kernel[(T, triton.cdiv(HQ, 4))](
        o_a,
        lse_a,
        o_s,
        lse_s,
        out,
        lse,
        T,
        HQ=HQ,
        D=D,
        HB=4,
        num_warps=2,
        GATE=output_gate,
        stride_g=output_gate.stride(0) if output_gate is not None else 0,
        HAS_GATE=output_gate is not None,
    )
    return out, lse


def route_ranges(
    sel_adm,
    sink_mask,
    *,
    n_units,
    group,
    q_block,
    kv_block,
    q_start=0,
    q_block_stride=1,
):
    """Per query block: the unit range holding admissible anchors ``[ulo, uhi)`` (what the score kernel walks) and the
    range it masks / forces ``[flo, fhi)`` (adds the own and sink units). Positions-only: one build serves every
    routed layer of a step (serving caches it per step)."""
    B, n_qb, _ = sel_adm.shape
    dev = sel_adm.device
    ua = sel_adm.unflatten(-1, (n_units, group)).any(-1)  # (B, n_qb, units)
    has = ua.any(-1)
    ulo = torch.where(has, ua.int().argmax(-1), 0).to(torch.int32)
    uhi = torch.where(has, n_units - ua.flip(-1).int().argmax(-1), 0).to(torch.int32)
    qb_first = q_start + (torch.arange(n_qb, device=dev) * q_block_stride) * q_block
    own_lo = (qb_first // kv_block).to(torch.int32)[None].expand(B, -1)
    own_hi = (
        ((qb_first + q_block - 1) // kv_block + 1).to(torch.int32)[None].expand(B, -1)
    )
    flo, fhi = torch.minimum(ulo, own_lo), torch.maximum(uhi, own_hi)
    if sink_mask is not None:
        sh = sink_mask.any(-1)
        slo = torch.where(sh, sink_mask.int().argmax(-1), n_units).to(torch.int32)
        shi = torch.where(sh, n_units - sink_mask.flip(-1).int().argmax(-1), 0).to(
            torch.int32
        )
        flo, fhi = torch.minimum(flo, slo), torch.maximum(fhi, shi)
    return (
        ulo.contiguous(),
        uhi.contiguous(),
        flo.clamp(0, n_units).contiguous(),
        fhi.clamp(0, n_units).contiguous(),
    )


def _anchor_cute_mask(stride: int):
    """``anchor_pos(kv) <= qpos[q] AND qdoc[q] == anchor_doc[kv]`` as a CuteDSL mask_mod; aux tensors: each query
    row's global position and document ``(B, L)`` and each anchor's document ``(B, A)``. Per-row tables rather than
    a compiled-in query offset, so ONE compiled mask serves every CP rank, layout and serving chunk offset. Every
    gather is indexed by q / kv themselves (never by a value derived from kv -- the FA4 chunking defect
    ``build_anchor_causal_mask`` documents), the form inductor emits for the flex mask."""
    if stride in _CUTE_MASKS:
        return _CUTE_MASKS[stride]
    import operator

    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def ssa_to_fragment(ssa_value, dtype):
        # torch._inductor's helper (absent in the serving venv's torch): an SSA vector into a register fragment
        if cutlass.const_expr(ssa_value.dtype != dtype):
            ssa_value = ssa_value.to(dtype)
        frag = cute.make_rmem_tensor(ssa_value.shape, dtype)
        frag.store(ssa_value)
        return frag

    @cute.jit
    def mask_mod(
        b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors, aux_scalars=None
    ):
        qpos = aux_tensors[0]
        qdoc = aux_tensors[1]
        adoc = aux_tensors[2]
        fb = ssa_to_fragment(b_idx, cutlass.Int32)
        fq = ssa_to_fragment(q_idx, cutlass.Int32)
        t0 = cute.make_rmem_tensor(cute.size(fb.shape), cutlass.Int32)
        for i in cutlass.range(cute.size(t0.shape), unroll_full=True):
            t0[i] = qpos[fb[i], fq[i]]
        t1 = cute.make_rmem_tensor(cute.size(fb.shape), cutlass.Int32)
        for i in cutlass.range(cute.size(t1.shape), unroll_full=True):
            t1[i] = qdoc[fb[i], fq[i]]
        fb2 = ssa_to_fragment(b_idx, cutlass.Int32)
        fk = ssa_to_fragment(kv_idx, cutlass.Int32)
        t2 = cute.make_rmem_tensor(cute.size(fb2.shape), cutlass.Int32)
        for i in cutlass.range(cute.size(t2.shape), unroll_full=True):
            t2[i] = adoc[fb2[i], fk[i]]
        causal = operator.le(kv_idx * stride + (stride - 1), t0.load())
        return causal & operator.eq(t1.load(), t2.load())

    mask_mod.__cute_hash__ = f"solar_rowroute_anchor_mask_rowtab_{stride}"
    _CUTE_MASKS[stride] = mask_mod
    return mask_mod


def _anchor_fwd_fa4(q, ka, va, aux, blocks, *, stride, q_block, kv_block, scale):
    """The anchor half: FA4 over every causal same-document anchor. ``aux`` = (qpos, qdoc, adoc), see the mask."""
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_fwd

    kv_n, kv_i, fkv_n, fkv_i = blocks[:4]
    B, L, HQ, D = q.shape
    out = torch.zeros_like(q)
    lse = torch.full((B, HQ, L), float("-inf"), device=q.device, dtype=torch.float32)
    _flash_attn_fwd(
        q,
        ka,
        va,
        softmax_scale=scale,
        return_lse=True,
        mask_mod=_anchor_cute_mask(stride),
        block_sparse_tensors=BlockSparseTensorsTorch(
            kv_n, kv_i, fkv_n, fkv_i, block_size=(q_block, kv_block)
        ),
        out=out,
        lse=lse,
        aux_tensors=list(aux),
    )
    return out, lse


def _anchor_blocks_chunk(
    T: int, A: int, p0: int, *, stride: int, q_block: int, kv_block: int, device
):
    """The anchor half's block lists for ONE request's prefill chunk (rows at positions ``p0 .. p0 + T - 1``, the
    request's ``A`` anchors): tile i admits anchor block j when the block's first anchor is at or before the tile's
    last row, and takes it FULL when its last anchor is at or before the tile's first row (and the block is
    complete). The serving twin of ``make_concat_route_fn``'s ``a_full`` / ``a_part`` for one document."""
    n_qt, n_kb = (T + q_block - 1) // q_block, (A + kv_block - 1) // kv_block
    i = torch.arange(n_qt, device=device)
    j = torch.arange(n_kb, device=device)
    first = p0 + i * q_block
    last = torch.clamp(p0 + (i + 1) * q_block, max=p0 + T) - 1
    a_first = (j * kv_block) * stride + stride - 1
    a_last = (torch.clamp((j + 1) * kv_block, max=A) - 1) * stride + stride - 1
    adm = a_first[None, :] <= last[:, None]
    full = (
        adm & (a_last[None, :] <= first[:, None]) & ((j + 1) * kv_block <= A)[None, :]
    )
    part = adm & ~full
    f = lambda m: (
        m.sum(-1).to(torch.int32)[None, None].contiguous(),  # noqa: E731
        torch.argsort(m.int(), dim=-1, descending=True, stable=True)
        .to(torch.int32)[None, None]
        .contiguous(),
    )
    (pn, pi), (fn, fi) = f(part), f(full)
    return pn, pi, fn, fi


def route_rows_serving(
    q,
    ka,
    meta,
    *,
    p0,
    t0,
    n_qb,
    NU,
    A,
    topk,
    scale,
    q_block,
    route_block,
    stride,
    score_reduce=None,
    qhead=False,
):
    """The serving prefill's per-row router (= training's ``route_rows_index`` selection, see
    ``_route_serve_kernel``): q ``(T, Hq, D)`` raw rows at ``p0..``, anchors ``ka`` ``(A, Hkv, D)`` (a strided view
    is fine). Returns ``sel`` ``(1, T, kk)``. ``qhead``: the per-head election, training's kernel."""
    T, HQ, D = q.shape
    HKV = ka.shape[1]
    if q.stride(2) != 1:
        raise ValueError(
            f"route_rows_serving reads q rows and heads by stride but needs a unit-stride head dim "
            f"(got strides {tuple(q.stride())})"
        )
    ulo, uhi, flo, fhi = meta["ranges"]
    rows = n_qb * q_block
    sc = torch.full((rows, NU), float("-inf"), device=q.device, dtype=torch.float32)
    sink = meta["sink"]
    adm = meta["sel_adm"][0]
    if qhead:
        G = HQ // HKV
        GP = triton.next_power_of_2(G)
        bs = min(
            QHEAD_CFG["rows_per_program"](GP), q_block
        )  # a program never spans two query tiles
        _row_route_qhead_kernel[(rows // bs, 1)](
            q,
            ka,
            adm,
            sink[0] if sink is not None else adm,
            sc,
            ulo[0],
            uhi[0],
            flo[0],
            fhi[0],
            rows,
            p0 - t0,
            T,
            A,
            n_qb,
            t0,
            1,
            0,
            q.stride(1),
            q.stride(0),
            0,
            ka.stride(1),
            ka.stride(0),
            0,
            adm.stride(0),
            adm.stride(1),
            0,
            (sink[0].stride(0) if sink is not None else 0),
            0,
            sc.stride(0),
            scale,
            QB=q_block,
            RB=route_block,
            HKV=HKV,
            G=G,
            GP=GP,
            D=D,
            BS=bs,
            APU=route_block // stride,
            UB=max(1, 128 // (route_block // stride)),
            STRIDE=stride,
            HAS_SINK=sink is not None,
            num_warps=QHEAD_CFG["warps"],
            num_stages=QHEAD_CFG["stages"],
        )
    else:
        _route_serve_kernel[(rows // 64,)](
            q,
            ka,
            adm,
            sink[0] if sink is not None else adm,
            sc,
            ulo[0],
            uhi[0],
            flo[0],
            fhi[0],
            T,
            p0 - t0,
            A,
            n_qb,
            t0,
            q.stride(0),
            q.stride(1),
            ka.stride(0),
            ka.stride(1),
            adm.stride(0),
            adm.stride(1),
            (sink[0].stride(0) if sink is not None else 0),
            sc.stride(0),
            scale,
            QB=q_block,
            RB=route_block,
            HQ=HQ,
            HKV=HKV,
            D=D,
            BS=64,
            APU=route_block // stride,
            UB=max(1, 128 // (route_block // stride)),
            HAS_SINK=sink is not None,
            num_warps=ROUTE_CFG["warps"],
            num_stages=ROUTE_CFG["stages"],
        )
    if score_reduce is not None:
        sc = score_reduce(sc)
    sc = sc[p0 - t0 : p0 - t0 + T]
    kk = min(topk, NU)
    vals, idx = sc.topk(kk, dim=-1, sorted=False)
    return torch.where(vals > float("-inf"), idx, -1).to(torch.int32)[None]


def _prefill_meta(
    T: int,
    p0: int,
    bt_row,
    page: int,
    *,
    stride: int,
    route_block: int,
    q_block: int,
    kv_block: int,
    sink_tokens: int,
    device,
):
    """Everything ``row_routed_prefill`` needs that depends only on the chunk's positions and the request's page
    table -- built once per (step, request) and shared by every routed layer (the per-call glue was host-bound)."""
    pe = p0 + T - 1
    A = (pe + 1) // stride
    bt = bt_row.to(torch.int32).contiguous()
    t0 = (p0 // q_block) * q_block
    n_qb = (pe + 1 - t0 + q_block - 1) // q_block
    S_r = ((pe + 1 + route_block - 1) // route_block) * route_block
    A_r, NU = S_r // stride, S_r // route_block
    tile_last = t0 + (torch.arange(n_qb, device=device) + 1) * q_block - 1
    ap_r = torch.arange(A_r, device=device) * stride + stride - 1
    sel_adm = (
        (ap_r[None, :] <= tile_last[:, None])
        & (torch.arange(A_r, device=device) < A)[None, :]
    )[None]
    sel_adm = sel_adm.contiguous()
    sink = None
    if sink_tokens:
        sink = (
            (torch.arange(NU, device=device) * route_block < sink_tokens)[None, None]
            .expand(1, n_qb, NU)
            .contiguous()
        )
    group = route_block // stride
    ranges = route_ranges(
        sel_adm,
        sink,
        n_units=NU,
        group=group,
        q_block=q_block,
        kv_block=route_block,
        q_start=t0,
    )
    gpos = (p0 + torch.arange(T, device=device)).to(torch.int32)
    meta = dict(
        pe=pe,
        A=A,
        bt=bt,
        t0=t0,
        n_qb=n_qb,
        S_r=S_r,
        NU=NU,
        sel_adm=sel_adm,
        sink=sink,
        ranges=ranges,
        gpos=gpos,
    )
    if A:
        Tq = ((T + q_block - 1) // q_block) * q_block
        Tp = Tq + q_block
        Ap = ((A + kv_block - 1) // kv_block + 1) * kv_block
        qpos = torch.full((1, Tp), -1, dtype=torch.int32, device=device)
        qpos[0, :T] = gpos
        meta.update(
            Tq=Tq,
            blocks=_anchor_blocks_chunk(
                T,
                A,
                p0,
                stride=stride,
                q_block=q_block,
                kv_block=kv_block,
                device=device,
            ),
            aux=(
                qpos,
                torch.zeros(1, Tp, dtype=torch.int32, device=device),
                torch.zeros(1, Ap, dtype=torch.int32, device=device),
            ),
        )
    return meta


def row_routed_prefill(
    q,
    key_cache,
    value_cache,
    bt_row,
    p0: int,
    *,
    stride: int,
    route_block: int,
    q_block: int,
    kv_block: int,
    topk: int,
    sink_tokens: int,
    scale: float,
    dedup: bool,
    score_reduce=None,
    return_sel: bool = False,
    cache: dict | None = None,
    cache_key=None,
    all_reduce=None,
    group_size: int | None = None,
    kv_head_start: int = 0,
    total_kv_heads: int | None = None,
    qhead: bool = False,
    output_gate: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
):
    """cfg.kv_route_per_row serving PREFILL of one request chunk (vLLM / SGLang): the rows route themselves
    (``_route_serve_kernel``: training's scores and head-mean contract, so the same selections), attend through the
    FA4 anchor half over the request's anchors and through the FA4 paged selected half over a contiguous copy of the
    request's prefix K / V, merged by LSE. Without it a per-row prompt would need one decode-style call per row.

    ``q`` ``(T, Hq, D)``, rows at consecutive positions ``p0 .. p0 + T - 1``; ``key_cache`` / ``value_cache``
    ``(blocks, page, Hkv, D)`` holding this chunk's K/V already; ``bt_row`` the request's block-table row.
    ``score_reduce`` = MAX across tensor-parallel ranks of the router scores (each rank scores its own KV heads).
    ``group_size`` (the model's query heads per KV head) below this rank's: tensor parallelism wider than the KV head
    count REPLICATES a KV head and SPLITS its query group, so ``all_reduce`` (SUM over the TP group) reassembles the
    group's query sum before the head mean -- the pooled and decode routes do the same; ranks MAX-ing scores of their
    own group shares would rank units differently from training (a max of means is not the mean's max).
    ``qhead`` (cfg.kv_route_qhead_block=1): the per-head election -- every query head's ballot is whole on its rank,
    so no reassembly; ``score_reduce``'s MAX folds the ranks' finished ballots.
    ``cache`` (+ ``cache_key``, e.g. the request's index in the step) shares the positions-only metadata across the
    step's routed layers. Returns ``(T, Hq, D)``."""
    _validate_output_gate(q, output_gate, output)
    T, HQ, D = q.shape
    page, HKV = key_cache.shape[1], key_cache.shape[2]
    if not (
        route_block in (64, 128)
        and D in (64, 128)
        and q.dtype == key_cache.dtype == torch.bfloat16
    ):
        raise ValueError(
            f"row_routed_prefill needs a bf16 cache, head_dim 64/128 and route_block 64/128 (got "
            f"{q.dtype}/{key_cache.dtype}, {D}, {route_block})"
        )
    dev = q.device
    key = ("rr", cache_key, p0, T)
    meta = cache.get(key) if cache is not None else None
    if meta is None:
        meta = _prefill_meta(
            T,
            p0,
            bt_row,
            page,
            stride=stride,
            route_block=route_block,
            q_block=q_block,
            kv_block=kv_block,
            sink_tokens=sink_tokens,
            device=dev,
        )
        if cache is not None:
            cache[key] = meta
    A, t0, n_qb, NU, pe = meta["A"], meta["t0"], meta["n_qb"], meta["NU"], meta["pe"]
    # the request's prefix K / V, contiguous, once per call (the FA4 selected half's unit pages and the anchors)
    ntok = meta["S_r"]
    if (
        key_cache.stride(2) == D and key_cache.stride(3) == 1
    ):  # NHD: a token's heads are contiguous
        kcon = torch.empty(ntok, HKV * D, device=dev, dtype=key_cache.dtype)
        vcon = torch.empty_like(kcon)
        _kv_unit_gather_kernel[(triton.cdiv(ntok, 64),)](
            key_cache,
            value_cache,
            meta["bt"],
            kcon,
            vcon,
            ntok,
            pe + 1,
            page,
            key_cache.stride(0),
            key_cache.stride(1),
            HD=HKV * D,
            BT_=64,
            num_warps=4,
        )
    else:  # HND and other layouts
        tokc = torch.arange(ntok, device=dev).clamp(max=pe)
        blk_, off_ = meta["bt"][tokc // page].long(), tokc % page
        kcon = key_cache[blk_, off_].reshape(ntok, HKV * D)
        vcon = value_cache[blk_, off_].reshape(ntok, HKV * D)
    ka = kcon.view(ntok, HKV, D)[stride - 1 :: stride][:A]  # (A, Hkv, D) strided views
    va = vcon.view(ntok, HKV, D)[stride - 1 :: stride][:A]
    q_route = q
    if not qhead and group_size is not None and group_size != HQ // HKV:
        if all_reduce is None or total_kv_heads is None or group_size % (HQ // HKV):
            raise ValueError(
                f"row_routed_prefill: a query group split across ranks ({HQ // HKV} of {group_size} heads "
                f"here) needs all_reduce and total_kv_heads to reassemble it"
            )
        # fp32 head-order sum of this rank's share, summed across the group's ranks, then the contract's one divide
        qg = q.view(T, HKV, HQ // HKV, D)
        part = qg[:, :, 0].float()
        for g in range(1, HQ // HKV):
            part = part + qg[:, :, g].float()
        full = part.new_zeros(T, total_kv_heads, D)
        full[:, kv_head_start : kv_head_start + HKV] = part
        q_route = (
            (all_reduce(full)[:, kv_head_start : kv_head_start + HKV] / group_size)
            .to(q.dtype)
            .contiguous()
        )
    sel = route_rows_serving(
        q_route,
        ka,
        meta,
        p0=p0,
        t0=t0,
        n_qb=n_qb,
        NU=NU,
        A=A,
        topk=topk,
        scale=scale,
        q_block=q_block,
        route_block=route_block,
        stride=stride,
        score_reduce=score_reduce,
        qhead=qhead,
    )
    qb = q.unsqueeze(0).contiguous()  # (1, T, Hq, D)
    gpos = meta["gpos"]
    if A:
        # The query side is padded to whole q tiles (rows at position -1 admit no anchor; dropped below): FA4 takes a
        # different small-query path at seqlen_q < a tile, which faulted with block sparsity + mask_mod at T = 1 (a
        # chunk vLLM schedules when a request's last prompt token lands in a new step). The aux tables are padded
        # past whole tiles too: FA4 reads them over WHOLE tiles (rows past T, keys past A).
        Tq = meta["Tq"]
        qpad = qb if Tq == T else torch.cat([qb, qb.new_zeros(1, Tq - T, HQ, D)], 1)
        o_a, lse_a = _anchor_fwd_fa4(
            qpad,
            ka.contiguous().unsqueeze(0),
            va.contiguous().unsqueeze(0),
            meta["aux"],
            meta["blocks"],
            stride=stride,
            q_block=q_block,
            kv_block=kv_block,
            scale=scale,
        )
        o_a, lse_a = o_a[:, :T].contiguous(), lse_a[:, :, :T].contiguous()
    else:
        o_a = torch.zeros_like(qb)
        lse_a = torch.full((1, HQ, T), float("-inf"), device=dev, dtype=torch.float32)
    out, _ = selected_forward_fa4paged(
        qb,
        sel,
        gpos,
        o_a,
        lse_a,
        stride=stride,
        route_block=route_block,
        dedup=dedup,
        scale=scale,
        NU=NU,
        pages=(kcon, vcon),
        output_gate=output_gate,
        output=output,
    )
    return (out[0], sel) if return_sel else out[0]
