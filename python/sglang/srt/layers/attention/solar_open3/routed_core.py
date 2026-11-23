# Copyright 2026 Upstage AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SolarOpen3 LSSA routing, attention-residuals, and Triton kernels.

Adapted from the SolarOpen3 serving implementation in simple-trainer. The
attention kernels accept SGLang's separate key/value pools and position-based
request tables. Gluon prefill resolves exclusively to kernels shipped in SGLang.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.solar_open3.output_gate import (
    _finish_output,
    _rounded_output_gate,
    _validate_output_gate,
)

logger = logging.getLogger(__name__)
# SGLang adaptation: optional BF16 output-gate epilogues write the caller's buffer.
# The carried routing/softmax arithmetic and all ungated defaults stay unchanged.


# ─────────────────────────────────────────────────── everything below is CARRIED, not written
# Extracted from the vLLM adapter by reachability from the entry points named in
# tests/model/solar_open3/_routed_core_extract.py. Do not edit by hand: edit the vLLM
# adapter, then regenerate. test_sglang_core_matches_vllm.py fails the build on drift.
_SUB_SCORE_BUDGET = 1 << 26


_DEC_SPLITS = int(os.environ.get("SOLAR_OPEN3_DEC_SPLITS", "32"))


_DEC_SPLITS_FIXED = "SOLAR_OPEN3_DEC_SPLITS" in os.environ


_DEC_TARGET_PROGRAMS = int(os.environ.get("SOLAR_OPEN3_DEC_PROGRAMS", "2048"))


_DEC_BLOCK_A = int(os.environ.get("SOLAR_OPEN3_DEC_BLOCK_A", "64"))


_DEC_WARPS = int(os.environ.get("SOLAR_OPEN3_DEC_WARPS", "0") or 0)


_FUSED_LIVE_SPLIT = os.environ.get("SOLAR_OPEN3_FUSED_LIVE_SPLIT", "1") != "0"


_TILE_PAGE = os.environ.get("SOLAR_OPEN3_TILE_PAGE", "0") != "0"


_SCORE_TARGET_PROGRAMS = int(os.environ.get("SOLAR_OPEN3_SCORE_PROGRAMS", "2048"))


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _anchor_attn_combine(
        O,
        M,
        L,
        OUT,
        LSE,
        stride_on,
        stride_oh,
        stride_os,
        stride_od,
        stride_mn,
        stride_mh,
        stride_ms,
        stride_un,
        stride_uh,
        stride_ud,
        stride_ln,
        stride_lh,
        SPLITS: tl.constexpr,
        D: tl.constexpr,
        OUT_F32: tl.constexpr,
        GATE=None,
        stride_g: tl.constexpr = 0,
        HAS_GATE: tl.constexpr = False,
    ):
        """The splits, back into the one softmax their union defines.

        A request whose anchors were all masked -- every unit selected, or nothing generated yet --
        comes back with ``l == 0`` and ``m == -inf``. That row's output is zero and its lse is -inf,
        which is what makes the caller's merge drop it rather than divide by nothing.
        """
        n = tl.program_id(0)
        h = tl.program_id(1)
        s_off = tl.arange(0, SPLITS)
        d_off = tl.arange(0, D)

        m = tl.load(M + n * stride_mn + h * stride_mh + s_off * stride_ms)
        l = tl.load(L + n * stride_mn + h * stride_mh + s_off * stride_ms)
        g = tl.max(m)
        w = tl.where(m == float("-inf"), 0.0, tl.exp(m - g))
        l_tot = tl.sum(l * w)

        o = tl.load(
            O
            + n * stride_on
            + h * stride_oh
            + s_off[:, None] * stride_os
            + d_off[None, :] * stride_od
        )
        acc = tl.sum(o * w[:, None], axis=0)

        empty = l_tot == 0.0
        _res = tl.where(empty, 0.0, acc / l_tot)
        if HAS_GATE:
            _res = _rounded_output_gate(_res, GATE, n * stride_g + h * D + d_off, True)
        tl.store(
            OUT + n * stride_un + h * stride_uh + d_off * stride_ud,
            _res if OUT_F32 else _res.to(OUT.dtype.element_ty),
        )
        tl.store(
            LSE + n * stride_ln + h * stride_lh,
            tl.where(empty, float("-inf"), g + tl.log(l_tot)),
        )

    @triton.jit
    def _qhead_ballot_stats(
        US,
        MZ,
        PL,
        stride_ur,
        stride_uu,
        n_units,
        ROWS: tl.constexpr,
        RB: tl.constexpr,
        BLOCK_U: tl.constexpr,
        EXACT: tl.constexpr = False,
        UBK: tl.constexpr = 4,
    ):
        """Per-ballot softmax denominators: row r of ``US`` -> ``MZ[r] = (max, sumexp)``.

        ``EXACT`` (cfg.kv_route_per_row checkpoints): the trainer's canonical recurrence instead
        (``row_routed_attn._ballot_stats_tile``) -- tiles of ``UBK`` units from unit 0, the tile max first, then
        each unit's term added in unit order -- so the decode holds bit-identical (M, Z) to training's and the
        per-row prefill's election and all three elect the same units.

        One program per (row, KV head, query head) ballot; the unit axis is walked in tiles with
        the usual online max/sum, so an all--inf ballot (a row with no rankable unit) comes out
        as ``(-inf, 0)`` -- the pair the fold below reads as "this ballot casts no mass", which
        is exactly what the torch tail's ``ok``-zeroing did.
        """
        r = tl.program_id(0)
        if EXACT:
            # Keep the canonical tiles and addition order, but skip trailing all--inf
            # tiles: their max is unchanged, alpha is 1, and every added term is 0.
            # PL is read on device so a captured graph follows each row's live length.
            seq_len = tl.maximum(tl.load(PL + r // ROWS) + 1, 0)
            live_units = tl.minimum(tl.cdiv(seq_len, RB), n_units).to(tl.int32)
            cols = tl.arange(0, UBK)
            m_ex = float("-inf")
            z_ex = 0.0
            for u0 in range(0, live_units, UBK):
                off = u0 + cols
                s = tl.load(
                    US + r * stride_ur + off * stride_uu,
                    mask=off < n_units,
                    other=float("-inf"),
                )
                m_new = tl.maximum(m_ex, tl.max(s))
                alpha = tl.where(m_new == m_ex, 1.0, tl.exp(m_ex - m_new))
                e = tl.where(s == float("-inf"), 0.0, tl.exp(s - m_new))
                z_ex = z_ex * alpha
                for jj in tl.static_range(UBK):
                    z_ex = z_ex + tl.sum(tl.where(cols == jj, e, 0.0))
                m_ex = m_new
            tl.store(MZ + r * 2 + 0, m_ex)
            tl.store(MZ + r * 2 + 1, z_ex)
        else:
            m_run = float("-inf")
            z_run = 0.0
            for u0 in range(0, n_units, BLOCK_U):
                off = u0 + tl.arange(0, BLOCK_U)
                s = tl.load(
                    US + r * stride_ur + off * stride_uu,
                    mask=off < n_units,
                    other=float("-inf"),
                )
                m_new = tl.maximum(m_run, tl.max(s))
                # exp(-inf - -inf) is nan, not 0: keep the rescale factor finite while nothing has
                # been accumulated yet (z_run is 0 there, so any finite factor is exact).
                alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_run - m_new))
                z_run = z_run * alpha + tl.sum(
                    tl.where(s == float("-inf"), 0.0, tl.exp(s - m_new))
                )
                m_run = m_new
            tl.store(MZ + r * 2 + 0, m_run)
            tl.store(MZ + r * 2 + 1, z_run)

    @triton.jit
    def _qhead_ballot_fold(
        US,
        MZ,
        PL,
        Q0,
        OUT,
        stride_un,
        stride_ur,
        stride_uu,
        stride_on,
        stride_ou,
        n_units,
        rb,
        ROWS: tl.constexpr,
        BLOCK_U: tl.constexpr,
    ):
        """``usf[n, u] = max over the ballots of exp(us - M)/Z``, own/past forcing included.

        The served form of the torch tail this replaces, one launch instead of eight: for every
        unit tile the ``ROWS`` ballots of the request are walked with their (M, Z) pairs, a dead
        ballot (M = -inf) contributes 0, a unit no ballot scored at all is -inf rather than the
        0.0 the max started from (as ``_vote_combine_kernel``'s ``fin``), and the fold's result
        is then sent to -inf past the row and +inf on the own units -- the same order, on the
        same device values.
        """
        n = tl.program_id(0)
        t = tl.program_id(1)
        off = t * BLOCK_U + tl.arange(0, BLOCK_U)
        ok_u = off < n_units
        pl = tl.load(PL + n)
        q0 = tl.load(Q0 + n)
        acc = tl.zeros((BLOCK_U,), tl.float32)
        fin = tl.zeros((BLOCK_U,), tl.int32)
        for r in range(ROWS):
            m = tl.load(MZ + (n * ROWS + r) * 2 + 0)
            z = tl.load(MZ + (n * ROWS + r) * 2 + 1)
            s = tl.load(
                US + n * stride_un + r * stride_ur + off * stride_uu,
                mask=ok_u,
                other=float("-inf"),
            )
            live = (m != float("-inf")) & (s != float("-inf"))
            b = tl.where(live, tl.exp(s - m) / z, 0.0)
            acc = tl.maximum(acc, b)
            fin = fin | live.to(tl.int32)
        # A unit no ballot gave a finite score is -inf, not the 0.0 the max started from: as
        # `_vote_combine_kernel`'s `fin`, so the downstream `vals == -inf` guard sees it (a
        # probability of exactly 0.0 would take a top-k slot). Unreachable on the served path
        # (such a unit is the own unit, forced +inf below), kept so both folds are one function.
        acc = tl.where(fin != 0, acc, float("-inf"))
        acc = tl.where(off * rb > pl, float("-inf"), acc)
        acc = tl.where((off * rb <= pl) & ((off + 1) * rb > q0), float("inf"), acc)
        tl.store(OUT + n * stride_on + off * stride_ou, acc, mask=ok_u)

    @triton.jit
    def _route_anchor_part(
        Q,
        QP,
        KV,
        VV,
        BTAB,
        PL,
        Q0,
        O,
        M,
        L,
        US,
        stride_qn,
        stride_qh,
        stride_qd,
        qp_sn,
        qp_sh,
        qp_sd,
        s_blk,
        s_half,
        s_slot,
        s_head,
        s_dim,
        stride_bn,
        stride_bp,
        stride_on,
        stride_oh,
        stride_os,
        stride_od,
        stride_mn,
        stride_mh,
        stride_ms,
        stride_un,
        stride_uh,
        stride_uu,
        stride_ug,
        scale,
        a_max,
        n_units,
        n_pages,
        rb,
        APG: tl.constexpr,
        STRIDE: tl.constexpr,
        PAGE: tl.constexpr,
        GQA: tl.constexpr,
        BLOCK_H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_A: tl.constexpr,
        PER: tl.constexpr,
        QP_GIVEN: tl.constexpr,
        SPLITS: tl.constexpr = 1,
        LIVE_SPLIT: tl.constexpr = False,
        TILE_IN_PAGE: tl.constexpr = False,
        QHEAD_US: tl.constexpr = False,
    ):
        """The anchor half of the union AND the routing scores, from one pass over the anchors.

        The ranking is a dot of the same anchor tile against the group-pooled query, and the
        attention's dot already runs on `BLOCK_H` rows of which only `GQA` are real. So the pooled
        query rides one of the padding rows and the score comes out of that matmul -- the anchor
        keys are read once here rather than once to rank and once to attend.

        ``QHEAD_US`` (`kv_route_qhead_block=1`) swaps WHICH rows of that matmul the ranking reads:
        the per-head unit maxima come off the GQA REAL rows instead of the pooled padding row, and
        ``US`` gains a head axis (``stride_ug``; zero and unread when the flag is off). The stored
        scores are RAW -- no own-unit ``+inf`` -- because the election softmaxes each head's row
        over the units and a ``+inf`` before a softmax makes that ballot one-hot; the host tail
        forces the own units after the fold, the same "force after the reduce" order every other
        election in this family keeps. Units past the row still go to ``-inf`` here, and the host
        pre-fills the buffer with ``-inf`` so a split that never enters needs no store at all.

        The selected half and the combine stay in their own kernels, so the anchor axis keeps its
        splits.

        ⚠️ The SCORES are bf16 products, the dtype a `tl.dot` takes from its operands -- as the
        trainer and the prefill scorer compute them (against fp32 products about one unit in
        thirty-two differs at topk=32). The RANKING over those scores runs on fp32 buffers
        (`US`, `usf`); that dtype is part of the election and `test_election_ranking_dtype`
        pins it.
        """
        n = tl.program_id(0)
        kvh = tl.program_id(1)
        sp = tl.program_id(2)

        d_off = tl.arange(0, D)
        h_off = tl.arange(0, BLOCK_H)
        h_ok = h_off < GQA
        q = tl.load(
            Q
            + n * stride_qn
            + (kvh * GQA + h_off)[:, None] * stride_qh
            + d_off[None, :] * stride_qd,
            mask=h_ok[:, None],
            other=0.0,
        )  # (BLOCK_H, D)
        pl = tl.load(PL + n)
        q0 = tl.load(Q0 + n)

        # Row GQA carries the router's query: the mean of the group's rows, or, under tensor
        # parallelism above the KV head count, the mean the caller has already reassembled. The
        # per-head election has no pooled query -- its ballots are the real rows -- so the padding
        # row stays zero and is masked out of everything below.
        if not QHEAD_US:
            if QP_GIVEN:
                qp = tl.load(QP + n * qp_sn + kvh * qp_sh + d_off * qp_sd).to(q.dtype)
            else:
                # Same head-order contract as components.attention._route_head_mean. Keep the
                # loaded Q tile; selecting one row adds no new query loads or activation buffer.
                total = tl.zeros((D,), tl.float32)
                for g in tl.static_range(GQA):
                    value = tl.sum(
                        tl.where((h_off == g)[:, None], q.to(tl.float32), 0.0), 0
                    )
                    total = tl.inline_asm_elementwise(
                        "add.rn.f32 $0, $1, $2;",
                        "=r,r,r",
                        [total, value],
                        dtype=tl.float32,
                        is_pure=True,
                        pack=1,
                    )
                qp = tl.div_rn(total, GQA).to(q.dtype)
            q = tl.where((h_off == GQA)[:, None], qp[None, :], q)

        m_i = tl.full((BLOCK_H,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, D), tl.float32)

        # `UPT` units per tile. The tile is a whole number of them by the caller's check, so a
        # unit's anchors never straddle two tiles and its max is complete when the tile ends.
        UPT: tl.constexpr = BLOCK_A // APG
        # `PER` divides the CEILING's anchors, so a server opened above the live context hands
        # most programs a range entirely past the row. `LIVE_SPLIT` divides what the ROW has
        # instead. `pl` is a device value, so this holds under a capture: the grid and every shape
        # stay fixed and only the trip count moves.
        #
        # The span is rounded up to whole units, so a split's last tile can reach past it. `keep`
        # carries `off < span`, which keeps those lanes out of the loads and the attention.
        #
        # ⚠️ The score store is bounded to this split's own units (below). Unbounded, a tile
        # reaching past the span writes -inf by unit id over the real maximum the NEXT split has
        # already put there, dropping a live unit out of the top-k.
        if LIVE_SPLIT:
            # The anchors this row HAS, divided over the programs. No `live` bound is kept past
            # this point: `enter` skips a split whose anchors are all past the row, and `keep`
            # masks the tail of the last one, so nothing reads it.
            #
            # ⚠️ Sized to reach the row's OWN unit as well, which the anchor count alone does not.
            # The splits together cover `SPLITS * span / APG` units, and the row's own unit
            # `pl // (APG * STRIDE)` lands exactly ON that bound whenever the anchor count divides
            # evenly -- every `pl % (SPLITS * APG * STRIDE) < STRIDE - 1`, so `STRIDE - 1` of every
            # `SPLITS * route_block` positions: 7 in 4096 at `kv_subsample_stride=8`,
            # `kv_route_block=512`, 8 splits. No split then stores that unit's score, the -inf prefill
            # survives, and the sentinel mask drops it from the top-k. Decode has no boundary pass
            # and the host does not re-insert the own unit after the top-k, so that step attends
            # everything EXCEPT its own most recent tokens -- silently, and at a cadence that moves
            # with the batch because `splits` is sized from it. At `pl < STRIDE - 1` the anchor
            # count is 0, so the same gap empties the selection entirely.
            _live = tl.minimum((pl + 1) // STRIDE, a_max)
            _own = tl.minimum((pl // (APG * STRIDE) + 1) * APG, a_max)
            span = (
                ((tl.maximum(_live, _own) + SPLITS - 1) // SPLITS + APG - 1) // APG
            ) * APG
        else:
            span = PER
        base_j = sp * span
        first = (base_j + 1) * STRIDE - 1
        # A split whose anchors are all past the row contributes no mass, so its loads are
        # skipped. It enters on its first POSITION rather than its first ANCHOR: a unit's anchors
        # begin `stride` tokens inside it, so a row that has just entered a unit has none of them,
        # and entering on the anchor would skip the unit holding the row's own query block -- the
        # one unit that is forced whatever it scored.
        if LIVE_SPLIT:
            enter = base_j * STRIDE <= pl
        else:
            enter = first <= pl
        for a0 in range(0, span, BLOCK_A):
            ut = (base_j + a0) // APG + tl.arange(0, UPT)
            start = ut * rb
            # The forced own unit, decided outside the load: a row within `stride - 1` of a
            # unit's start has that unit's first anchor past it, so the split holding it can be
            # one this program skips and the unit still has to be taken.
            own = (start <= pl) & ((ut + 1) * rb > q0)
            if enter:
                off = a0 + tl.arange(0, BLOCK_A)
                j = base_j + off
                pos = (j + 1) * STRIDE - 1
                u = j // APG
                keep = (off < span) & (j < a_max) & (pos <= pl) & (u < n_units)
                pg_i = pos // PAGE
                keep = keep & (pg_i < n_pages)
                if TILE_IN_PAGE:
                    # Consecutive anchors are `STRIDE` apart, so a tile spans
                    # `(BLOCK_A - 1) * STRIDE` tokens, which the caller has checked is under a
                    # page. It therefore touches the page of its first anchor and at most the next
                    # one, so selecting between the two by each row's page index is exact -- two
                    # dependent loads a tile instead of `BLOCK_A` of them.
                    _i0 = tl.minimum(
                        ((base_j + a0 + 1) * STRIDE - 1) // PAGE, n_pages - 1
                    )
                    _i1 = tl.minimum(
                        ((base_j + a0 + BLOCK_A) * STRIDE - 1) // PAGE, n_pages - 1
                    )
                    _g0 = tl.load(BTAB + n * stride_bn + _i0 * stride_bp).to(tl.int64)
                    _g1 = tl.load(BTAB + n * stride_bn + _i1 * stride_bp).to(tl.int64)
                    # A row masked out of `keep` may land on neither; its k and v loads are masked
                    # too, so the page it is given does not matter.
                    pg = tl.where(pg_i == _i0, _g0, _g1)
                else:
                    pg = tl.load(
                        BTAB + n * stride_bn + pg_i * stride_bp, mask=keep, other=0
                    ).to(tl.int64)
                base = pg * s_blk + (pos % PAGE) * s_slot + kvh * s_head
                cols = d_off[None, :] * s_dim
                k = tl.load(KV + base[:, None] + cols, mask=keep[:, None], other=0.0)
                # Issued before the ranking. The two loads are the tile's only global reads and
                # do not depend on each other; putting the score store between them would leave
                # this one waiting behind a write.
                v = tl.load(
                    VV + base[:, None] + s_half + cols, mask=keep[:, None], other=0.0
                )
                sc = tl.dot(q, tl.trans(k)).to(tl.float32) * scale  # (BLOCK_H, BLOCK_A)

                if QHEAD_US:
                    # -- the per-head ranking, off the REAL rows of the same product. Raw maxima
                    # only (own forcing happens after the host's softmax fold, see the docstring);
                    # ⚠️ the store mask keeps this split's units exactly as the pooled store below
                    # does -- unbounded, a tile reaching past the span writes -inf by unit id over
                    # the maximum the NEXT split already put there (fix 3 in the README of
                    # .jobs/260830_lssa_fused_decode: a live unit dropped out of the top-k).
                    hs = tl.where(keep[None, :] & h_ok[:, None], sc, float("-inf"))
                    hmax = tl.max(
                        tl.reshape(hs, (BLOCK_H, UPT, APG)), 2
                    )  # (BLOCK_H, UPT)
                    hmax = tl.where((start > pl)[None, :], float("-inf"), hmax)
                    tl.store(
                        US
                        + n * stride_un
                        + kvh * stride_uh
                        + h_off[:, None] * stride_ug
                        + ut[None, :] * stride_uu,
                        hmax,
                        mask=h_ok[:, None]
                        & ((ut < n_units) & (ut * APG < base_j + span))[None, :],
                    )
                else:
                    # -- the ranking, off row GQA of the same product.
                    rs = tl.sum(
                        tl.where((h_off == GQA)[:, None], sc, 0.0), 0
                    )  # (BLOCK_A,)
                    rs = tl.where(keep, rs, float("-inf"))
                    umax = tl.max(tl.reshape(rs, (UPT, APG)), 1)  # (UPT,)
                    # A unit past the row is not rankable; the one holding the row's own query
                    # block is taken whatever it scored. -inf first, then +inf, as
                    # `_decode_score_unit` does.
                    umax = tl.where(start > pl, float("-inf"), umax)
                    umax = tl.where(own, float("inf"), umax)
                    tl.store(
                        US + n * stride_un + kvh * stride_uh + ut * stride_uu,
                        umax,
                        mask=(ut < n_units) & (ut * APG < base_j + span),
                    )

                # -- and the attention, over the real rows of the same product.
                sc = tl.where(keep[None, :] & h_ok[:, None], sc, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(sc, axis=1))
                alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
                pr = tl.where(sc == float("-inf"), 0.0, tl.exp(sc - m_new[:, None]))
                acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v).to(tl.float32)
                l_i = l_i * alpha + tl.sum(pr, axis=1)
                m_i = m_new
            else:
                # Every anchor of this split is past the row. It contributes no mass and no
                # maximum, which is what the accumulators already hold -- but the SCORES still
                # have to be written, or the ranking reads whatever the buffer was allocated with.
                # The per-head buffer arrives pre-filled with -inf (its host always allocates
                # `torch.full`), so it alone may skip the store.
                if not QHEAD_US:
                    umax = tl.where(own, float("inf"), float("-inf"))
                    tl.store(
                        US + n * stride_un + kvh * stride_uh + ut * stride_uu,
                        tl.broadcast_to(umax, (UPT,)),
                        mask=(ut < n_units) & (ut * APG < base_j + span),
                    )

        tl.store(
            O
            + n * stride_on
            + (kvh * GQA + h_off)[:, None] * stride_oh
            + sp * stride_os
            + d_off[None, :] * stride_od,
            acc,
            mask=h_ok[:, None],
        )
        tl.store(
            M + n * stride_mn + (kvh * GQA + h_off) * stride_mh + sp * stride_ms,
            m_i,
            mask=h_ok,
        )
        tl.store(
            L + n * stride_mn + (kvh * GQA + h_off) * stride_mh + sp * stride_ms,
            l_i,
            mask=h_ok,
        )

    @triton.jit
    def _selected_attn_part(
        Q,
        KV,
        VV,
        BTAB,
        PL,
        O,
        M,
        L,
        SIDX,
        stride_qn,
        stride_qh,
        stride_qd,
        s_blk,
        s_half,
        s_slot,
        s_head,
        s_dim,
        stride_bn,
        stride_bp,
        stride_on,
        stride_oh,
        stride_os,
        stride_od,
        stride_mn,
        stride_mh,
        stride_ms,
        stride_in,
        stride_ik,
        scale,
        n_units,
        n_pages,
        sp0,
        STRIDE: tl.constexpr,
        PAGE: tl.constexpr,
        GQA: tl.constexpr,
        BLOCK_H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_A: tl.constexpr,
        RB: tl.constexpr,
        K_EFF: tl.constexpr,
        PER_S: tl.constexpr,
        DEDUP: tl.constexpr,
    ):
        """The selected route blocks, into split slots `sp0 +` this program's own.

        Its partials share the buffer the anchor pass writes, so the combine that follows sees one
        set of splits over the whole union and no second merge is needed. `dedup_anchor` is
        expressed here, on the narrow side: the anchor pass has taken every anchor, so this loop
        skips the positions that ARE anchors.
        """
        n = tl.program_id(0)
        kvh = tl.program_id(1)
        s = tl.program_id(2)

        d_off = tl.arange(0, D)
        h_off = tl.arange(0, BLOCK_H)
        h_ok = h_off < GQA
        q = tl.load(
            Q
            + n * stride_qn
            + (kvh * GQA + h_off)[:, None] * stride_qh
            + d_off[None, :] * stride_qd,
            mask=h_ok[:, None],
            other=0.0,
        )
        pl = tl.load(PL + n)

        m_i = tl.full((BLOCK_H,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, D), tl.float32)

        for b0 in range(0, PER_S, BLOCK_A):
            offs = b0 + tl.arange(0, BLOCK_A)
            t = s * PER_S + offs
            ui = t // RB
            usel = tl.load(
                SIDX + n * stride_in + ui * stride_ik,
                mask=(offs < PER_S) & (t < K_EFF * RB) & (ui < K_EFF),
                other=n_units,
            )
            pos = usel * RB + (t % RB)
            keep = (offs < PER_S) & (t < K_EFF * RB) & (usel < n_units) & (pos <= pl)
            if DEDUP:
                keep = keep & ((pos % STRIDE) != STRIDE - 1)
            pg_i = pos // PAGE
            keep = keep & (pg_i < n_pages)
            pg = tl.load(
                BTAB + n * stride_bn + pg_i * stride_bp, mask=keep, other=0
            ).to(tl.int64)
            base = pg * s_blk + (pos % PAGE) * s_slot + kvh * s_head
            cols = d_off[None, :] * s_dim
            k = tl.load(KV + base[:, None] + cols, mask=keep[:, None], other=0.0)
            v = tl.load(
                VV + base[:, None] + s_half + cols, mask=keep[:, None], other=0.0
            )
            sc = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
            sc = tl.where(keep[None, :] & h_ok[:, None], sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
            pr = tl.where(sc == float("-inf"), 0.0, tl.exp(sc - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(pr, axis=1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v).to(tl.float32)
            m_i = m_new

        tl.store(
            O
            + n * stride_on
            + (kvh * GQA + h_off)[:, None] * stride_oh
            + (sp0 + s) * stride_os
            + d_off[None, :] * stride_od,
            acc,
            mask=h_ok[:, None],
        )
        tl.store(
            M + n * stride_mn + (kvh * GQA + h_off) * stride_mh + (sp0 + s) * stride_ms,
            m_i,
            mask=h_ok,
        )
        tl.store(
            L + n * stride_mn + (kvh * GQA + h_off) * stride_mh + (sp0 + s) * stride_ms,
            l_i,
            mask=h_ok,
        )

    @triton.jit
    def _topk_pack(v, u):
        """(score, unit) -> one int64 ordering by score, then by ASCENDING unit on a tie.

        The float32 bits are zero-extended into an int64 first: the sortable map needs the
        constants 0x80000000 and 0xffffffff and neither fits in int32. Zero-extended, a negative
        float is exactly the half at or above 0x80000000. The key is then re-centred to a signed
        32-bit range before the shift, because `key << 32` would otherwise set bit 63 for every
        POSITIVE float and order the positives below the negatives.

        -0.0 is normalised away first: the bit map puts +0.0 above -0.0 while a numeric comparison
        calls them equal and leaves the tie to the index. It is the one pair where the two
        disagree.
        """
        v = tl.where(v == 0.0, 0.0, v)
        b = v.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        key = b ^ tl.where(b >= 0x80000000, 0xFFFFFFFF, 0x80000000)
        return ((key - 0x80000000) << 32) | ((~u.to(tl.int64)) & 0xFFFFFFFF)

    @triton.jit
    def _chunk_topk_kernel(
        US,
        CAND,
        stride_un,
        stride_uu,
        stride_cn,
        n_units,
        K: tl.constexpr,
        TILE: tl.constexpr,
    ):
        """Each (row, chunk) keeps its own best K. The overall best K are a subset of the union."""
        n, c = tl.program_id(0), tl.program_id(1)
        u = c * TILE + tl.arange(0, TILE)
        v = tl.load(
            US + n * stride_un + u * stride_uu, mask=u < n_units, other=float("-inf")
        )
        tl.store(
            CAND + n * stride_cn + c * K + tl.arange(0, K), tl.topk(_topk_pack(v, u), K)
        )

    @triton.jit
    def _merge_topk_kernel(
        CAND,
        OUT,
        VAL,
        stride_cn,
        stride_on,
        n_cand,
        K: tl.constexpr,
        CAND_P: tl.constexpr,
    ):
        """The chunks' candidates, back down to K, unpacked into the pair torch.topk returns.

        `tl.topk` takes a power-of-two input, so the read is padded to one with the sentinel,
        which loses to every real packed value. An arithmetic shift recovers the signed key
        exactly: the low 32 bits it discards are non-negative and below 2**32.
        """
        n = tl.program_id(0)
        j = tl.arange(0, CAND_P)
        c = tl.load(
            CAND + n * stride_cn + j, mask=j < n_cand, other=-9223372036854775808
        )
        best = tl.topk(c, K)
        k_off = tl.arange(0, K)
        tl.store(OUT + n * stride_on + k_off, ((~best) & 0xFFFFFFFF).to(tl.int32))
        key = (best >> 32) + 0x80000000
        b = key ^ tl.where(key >= 0x80000000, 0x80000000, 0xFFFFFFFF)
        tl.store(
            VAL + n * stride_on + k_off,
            tl.where(b >= 0x80000000, b - 0x100000000, b)
            .to(tl.int32)
            .to(tl.float32, bitcast=True),
        )

    @triton.jit
    def _dedup_page_table_kernel(
        OL,
        UNITS,
        ABT,
        BTAB,
        NKEPT,
        s_uq,
        s_us,
        s_bq,
        s_bs,
        s_nq,
        n_units,
        n_pages,
        k_eff,
        NU: tl.constexpr,
        KP: tl.constexpr,
        UPP: tl.constexpr,
    ):
        """The dedup page table in one launch: keep mask, stable compaction, page expansion.

        The compaction is a prefix sum, not a sort: sorting a BOOLEAN key descending and stably is
        a stable partition, so a kept unit lands at the count of kept units before it and a dropped
        one at `n_kept` plus the count of dropped units before it. That makes the whole sequence
        one kernel instead of the nine the torch form launches.
        """
        q = tl.program_id(0)
        h = tl.program_id(1)
        u = tl.arange(0, NU)
        valid = u < n_units
        keep = valid & (u < tl.load(OL + q))

        kj = tl.arange(0, KP)
        units = tl.load(UNITS + q * s_uq + h * s_us + kj, mask=kj < k_eff, other=-1)
        # Two kinds of index arrive outside [0, n_units): the `-1` this load pads a lane with when
        # `k_eff` is not a power of two, and the `n_units` sentinel a row carries where the ranking
        # could not fill a slot. Neither selects anything, so neither may drop a page -- and
        # clamping them into range is exactly how they would: `-1` becomes unit 0 and the sentinel
        # becomes the last unit. Mask them out of the comparison instead.
        in_range = (units >= 0) & (units < n_units)
        keep = keep & (
            tl.sum(((u[:, None] == units[None, :]) & in_range[None, :]).to(tl.int32), 1)
            == 0
        )

        ki = keep.to(tl.int32)
        di = (valid & (keep == 0)).to(tl.int32)
        n_kept = tl.sum(ki, 0)
        rank = tl.where(keep, tl.cumsum(ki, 0) - ki, n_kept + tl.cumsum(di, 0) - di)

        for j in tl.static_range(UPP):
            # The pages past the real ones still have to name a readable entry -- `seqused_a`, not
            # the table, is what cuts each block to its own width.
            src = tl.minimum(u * UPP + j, tl.maximum(n_pages - 1, 0))
            tl.store(
                BTAB + q * s_bq + h * s_bs + rank * UPP + j,
                tl.load(ABT + src, mask=valid, other=0),
                mask=valid,
            )
        tl.store(NKEPT + q * s_nq + h, n_kept)

    def _dedup_page_table_torch(ol_t, units, abt, n_units, n_pages, upp):
        """The same table in plain torch: the REFERENCE, and the only path off CUDA.

        Serving never reaches this -- the routed prefill runs on CUDA or is refused long before
        here -- so it is not a fallback for the kernel. It is the definition the kernel is checked
        against (`test_dedup_page_table_gpu.py` asserts they agree bit for bit) and the form the
        set-semantics tests drive without a GPU. Changing one without the other is what the parity
        test exists to catch.
        """
        n_qb, nsel, _ = units.shape
        # The caller's `if dedup_anchor and n_pages:` is what keeps an empty page axis out of here,
        # and this says so rather than leaving it to an index error further down. A request shorter
        # than one stride window has no anchors and nothing to deduplicate; reaching this with none
        # killed eight engines mid-evaluation once, on the short prompts only.
        if n_units <= 0 or n_pages <= 0:
            raise RuntimeError(
                f"dedup page table on an empty anchor axis (n_units={n_units}, "
                f"n_pages={n_pages}): a request with no anchors has nothing to deduplicate, and "
                f"the caller is expected to skip this branch entirely."
            )
        pg = torch.arange(n_units, device=units.device)
        keep = (
            (pg[None, None] < ol_t[:, None, None]).expand(n_qb, nsel, n_units).clone()
        )
        # Out-of-range ids (the `n_units` sentinel, and anything below zero) must drop nothing --
        # see the kernel. Send them to a scratch column that is thrown away, rather than clamping
        # them onto a real unit.
        scratch = torch.cat([keep, keep.new_ones(n_qb, nsel, 1)], dim=2)
        scratch.scatter_(
            2,
            torch.where((units >= 0) & (units < n_units), units.long(), n_units),
            False,
        )
        keep = scratch[..., :n_units].contiguous()
        order = keep.int().argsort(dim=2, descending=True, stable=True)
        btab = (
            abt[
                (order[..., None] * upp + torch.arange(upp, device=units.device))
                .flatten(2)
                .clamp(max=max(n_pages - 1, 0))
            ]
            .to(torch.int32)
            .contiguous()
        )
        return btab, keep.sum(2).to(torch.int32)

    def dedup_page_table(ol_t, units, abt, n_units, n_pages, upp):
        """(page table, kept units per block) for `routed_prefill_batched`'s dedup branch."""
        if units.device.type != "cuda":
            return _dedup_page_table_torch(ol_t, units, abt, n_units, n_pages, upp)
        n_qb, nsel, k_eff = units.shape
        btab = torch.empty(
            (n_qb, nsel, n_units * upp), dtype=torch.int32, device=units.device
        )
        nkept = torch.empty((n_qb, nsel), dtype=torch.int32, device=units.device)
        _dedup_page_table_kernel[(n_qb, nsel)](
            ol_t,
            units,
            abt,
            btab,
            nkept,
            units.stride(0),
            units.stride(1),
            btab.stride(0),
            btab.stride(1),
            nkept.stride(0),
            n_units,
            n_pages,
            k_eff,
            NU=triton.next_power_of_2(n_units),
            KP=triton.next_power_of_2(k_eff),
            UPP=upp,
        )
        return btab, nkept

    @triton.jit
    def _selected_page_table_kernel(
        IDX,
        BTROW,
        UNITS,
        BTAB,
        s_iq,
        s_is,
        s_uq,
        s_us,
        s_bq,
        s_bs,
        k_eff,
        lp_max,
        KP: tl.constexpr,
        UPP: tl.constexpr,
    ):
        """The selected region's page table in one launch: sort, page expansion, clamp, gather.

        The torch form is `sort` -> `arange` -> `mul` -> `add` -> `clamp` -> `index` -> `cast`, one
        launch a line, and the engine profile prices the scope at 9 kernels per routed layer-pass
        for 2.5 us of GPU time each -- roughly 3.9x more host time issuing than device time doing.
        The sort is also shape-dependent, so the region cannot be captured as one graph either.

        `tl.sort` takes a power-of-two block, so the read pads to one with a sentinel that sorts
        above every real unit id; the first `k_eff` sorted slots are then exactly the ascending
        selection. A unit is `UPP` consecutive pages, so the page ids follow from the sorted value
        in closed form and the whole expansion is arithmetic on a register.
        """
        q = tl.program_id(0)
        h = tl.program_id(1)
        kj = tl.arange(0, KP)
        m = kj < k_eff
        u = tl.load(IDX + q * s_iq + h * s_is + kj, mask=m, other=9223372036854775807)
        u = tl.sort(u)
        tl.store(UNITS + q * s_uq + h * s_us + kj, u, mask=m)
        # The pad lanes store nothing, but their address arithmetic still runs; keeping the
        # sentinel would overflow the multiply below.
        u = tl.where(m, u, 0)
        for j in tl.static_range(UPP):
            # A block whose admissible units are fewer than the budget carries sentinel ids past
            # the last unit. Their pages still have to name a readable entry -- `seqused_k`, not
            # the table, is what cuts each block to its own width -- so they clamp to the request's
            # last page rather than running off the block-table row.
            lp = tl.minimum(tl.maximum(u * UPP + j, 0), lp_max)
            tl.store(
                BTAB + q * s_bq + h * s_bs + kj * UPP + j,
                tl.load(BTROW + lp, mask=m, other=0).to(tl.int32),
                mask=m,
            )

    def _selected_page_table_torch(idx, btrow, ppu, lp_max):
        """The same pair in plain torch: the REFERENCE, and the only path off CUDA.

        Serving never reaches this -- the routed prefill runs on CUDA or is refused long before
        here -- so it is not a fallback for the kernel. It is the definition the kernel is checked
        against (`test_selected_page_table_gpu.py` asserts they agree bit for bit). Changing one
        without the other is what that parity test exists to catch.
        """
        units = idx.sort(-1).values
        lp = (units[..., None] * ppu + torch.arange(ppu, device=idx.device)).flatten(2)
        return units, btrow[lp.clamp(min=0, max=lp_max)].to(torch.int32).contiguous()

    def selected_page_table(idx, btrow, ppu, lp_max):
        """(sorted selected units, page table) for `routed_prefill_batched`'s selected region."""
        if ppu < 1:
            raise ValueError(
                f"selected_page_table got pages-per-unit {ppu} for idx {tuple(idx.shape)}: a "
                "routing unit is at least one page, so the caller's route_block is smaller than "
                "the cache page it was handed. `subpage_cache` splits the page to fix that -- "
                "call it before this."
            )
        if idx.device.type != "cuda":
            return _selected_page_table_torch(idx, btrow, ppu, lp_max)
        if btrow.dim() != 1 or btrow.stride(0) != 1:
            # The kernel addresses the row as `BTROW + page`. A strided or multi-row table would
            # read the wrong entries and still return a full table, so it is refused here.
            raise ValueError(
                f"selected_page_table needs one contiguous block-table row, got shape "
                f"{tuple(btrow.shape)} stride {tuple(btrow.stride())}. Pass the request's own row "
                "(`block_table[i]`), or `.contiguous()` it first."
            )
        n_qb, nsel, k_eff = idx.shape
        units = torch.empty((n_qb, nsel, k_eff), dtype=idx.dtype, device=idx.device)
        btab = torch.empty(
            (n_qb, nsel, k_eff * ppu), dtype=torch.int32, device=idx.device
        )
        _selected_page_table_kernel[(n_qb, nsel)](
            idx,
            btrow,
            units,
            btab,
            idx.stride(0),
            idx.stride(1),
            units.stride(0),
            units.stride(1),
            btab.stride(0),
            btab.stride(1),
            k_eff,
            lp_max,
            KP=triton.next_power_of_2(k_eff),
            UPP=ppu,
        )
        return units, btab

    @triton.jit
    def _raw_page_table_kernel(
        BTAB_R,
        BTROW,
        Q0,
        NRF0,
        PL,
        BTAB_RO,
        SEQ_RO,
        s_rq,
        s_oq,
        bs,
        npg,
        wp,
        sel_w,
        lp_max,
        WP: tl.constexpr,
    ):
        """The read-once branch's interleaved page table in one launch.

        Flash gets ONE table per query block: the selected region's pages up to `nrf0 // bs`, the
        block's own pages after it. Every column is one load whose address is a closed form of the
        column index, so there is no intermediate to build.
        """
        q = tl.program_id(0)
        col = tl.arange(0, WP)
        m = col < wp
        q0 = tl.load(Q0 + q)
        nrf0 = tl.load(NRF0 + q)
        spg = nrf0 // bs
        # Both clamps are the torch form's. A column past either list still has to name a readable
        # page: `seqused`, not the table, is what cuts the row to its real width.
        sel = tl.load(BTAB_R + q * s_rq + tl.minimum(col, sel_w - 1), mask=m, other=0)
        own_off = tl.minimum(tl.maximum(col - spg, 0), npg - 1)
        own = tl.load(
            BTROW + tl.minimum(tl.maximum(q0 // bs + own_off, 0), lp_max),
            mask=m,
            other=0,
        )
        tl.store(
            BTAB_RO + q * s_oq + col, tl.where(col < spg, sel, own).to(tl.int32), mask=m
        )
        tl.store(SEQ_RO + q, (nrf0 + (tl.load(PL + q) - q0 + 1)).to(tl.int32))

    def _raw_page_table_torch(btab_r, btrow, q0_t, nrf0_t, pl_t, bs, npg, wp, lp_max):
        """The same table in plain torch: the definition `_raw_page_table_kernel` is checked
        against by `test_raw_page_table_gpu.py`, and the only path off CUDA.
        """
        if npg < 1:
            raise ValueError(
                f"a query block of this width holds no whole page: q_block // block_size = {npg}. "
                "The read-once branch interleaves the block's OWN pages with the selected ones, "
                "and there are none to interleave. Widen q_block or narrow the page."
            )
        dev = btab_r.device
        n_qb = btab_r.shape[0]
        lpg = ((q0_t[:, None] // bs) + torch.arange(npg, device=dev)).clamp(
            min=0, max=lp_max
        )
        btab_own = btrow[lpg].to(torch.int32)
        spg_t = (nrf0_t // bs)[:, None]
        col = torch.arange(wp, device=dev)[None]
        btab_ro = torch.where(
            col < spg_t,
            btab_r.gather(1, col.clamp(max=btab_r.shape[1] - 1).expand(n_qb, -1)),
            btab_own.gather(1, (col - spg_t).clamp(min=0, max=npg - 1)),
        ).contiguous()
        return btab_ro, (nrf0_t + (pl_t - q0_t + 1)).to(torch.int32)

    def raw_page_table(btab_r, btrow, q0_t, nrf0_t, pl_t, bs, npg, wp, lp_max):
        """(interleaved page table, seqused) for the fused read-once attention call."""
        if btab_r.device.type != "cuda":
            return _raw_page_table_torch(
                btab_r, btrow, q0_t, nrf0_t, pl_t, bs, npg, wp, lp_max
            )
        if npg < 1:
            raise ValueError(
                f"a query block of this width holds no whole page: q_block // block_size = {npg}. "
                "The read-once branch interleaves the block's OWN pages with the selected ones, "
                "and there are none to interleave. Widen q_block or narrow the page."
            )
        if btrow.dim() != 1 or not btrow.is_contiguous():
            # The kernel addresses `BTROW + page`. A strided or multi-row table would return a
            # different table with no error, which is the failure this arm keeps hitting.
            raise ValueError(
                f"the block-table row must be one contiguous 1-D tensor, got shape "
                f"{tuple(btrow.shape)} contiguous={btrow.is_contiguous()}."
            )
        n_qb = btab_r.shape[0]
        out = torch.empty((n_qb, wp), dtype=torch.int32, device=btab_r.device)
        seq = torch.empty((n_qb,), dtype=torch.int32, device=btab_r.device)
        _raw_page_table_kernel[(n_qb,)](
            btab_r,
            btrow,
            q0_t,
            nrf0_t,
            pl_t,
            out,
            seq,
            btab_r.stride(0),
            out.stride(0),
            bs,
            npg,
            wp,
            btab_r.shape[1],
            lp_max,
            WP=triton.next_power_of_2(wp),
        )
        return out, seq

    _HAS_DECODE_SCORE = True
except ImportError as _e:
    # ONLY triton. This `try` spans every kernel definition above it, so anything else raising
    # ImportError in those lines is a bug in this file and must not be read as "triton is absent".
    if (_e.name or "").split(".")[0] != "triton":
        raise
    _HAS_DECODE_SCORE = False
    logger.warning(
        "[solar_open3] triton is not importable (%s). Layers that neither route nor subsample "
        "serve normally; a routed layer refuses at its first request, because every scorer and "
        "every routed attention body in this file is a triton kernel.",
        _e,
    )


if _HAS_DECODE_SCORE:

    @triton.jit
    def _prefill_score_kernel(
        Q,
        KA,
        BTAB,
        PL,
        OUT,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kp,
        stride_kr,
        stride_kc,
        stride_oh,
        stride_os,
        stride_ou,
        scale,
        n_sub,
        n_anchor,
        n_units,
        VT,
        APG: tl.constexpr,
        APP: tl.constexpr,
        STRIDE: tl.constexpr,
        RB: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        UPT: tl.constexpr,
        NG: tl.constexpr,
        PER_ANCHOR: tl.constexpr,
    ):
        """One program per (head, unit GROUP, sub-query tile) -> each unit's best score.

        Unit ``u`` owns anchors ``[u*APG, (u+1)*APG)``, contiguously, so a program's keys are a
        run of rows of the anchor tensor and its ids are a closed form of the program's own.
        Neither a padded transposed copy of that tensor nor a per-key admissibility mask is built.

        It scores the ANCHORS, not the cache rows they were pooled from -- the same tensor the
        anchor attention attends, so the ranking and the attention see one thing. At
        ``kv_subsample_pool="last"`` an anchor IS its cache row and the two coincide; at
        "mean"/"wmean"/"attn" it is a reduction over the stride window, and ranking the raw row
        would rank something no pass attends.

        ``UPT`` units at a time, so the dot's N axis fills a tensor-core tile: one unit holds only
        `route_block / stride` anchors. Sixteen units exceeds the tensor memory limit.

        The fold stays a reshape because the units are contiguous in the anchor axis: ``UPT*APG``
        anchors come back as ``(BLOCK_S, UPT, APG)`` and the max over the last axis is each unit's
        score. No atomics, no cross-program reduction.

        The gate has two forms because the two callers it replaces gate differently, and the
        difference is a different ranking rather than a rounding:

        * ``PER_ANCHOR`` -- the sub-query path (`kv_route_score_block` set, which is how m100 is
          served). It builds ``adm = apos <= p_last`` per anchor, so an anchor past the block's
          last row never enters the max.
        * otherwise -- the pooled path, whose gate is per UNIT and applied to the finished score.
          A unit straddling ``p_last`` is therefore ranked on its later anchors too.

        ``n_anchor`` bounds the pool in both: an anchor exists while its index is below it.
        """
        h = tl.program_id(0)
        g = tl.program_id(1)
        # The sub-query axis is cut across programs as well as looped: (head x unit group) alone
        # is ~512 programs on a 148-SM part, each walking every vote in series, and the per-head
        # election multiplies the votes by the group size -- the serial walk was its whole cost.
        # Each z-program re-reads the tile's keys (~128KB), which the cut keeps trivial.
        v = tl.program_id(2)
        d_off = tl.arange(0, D)
        u_off = g * UPT + tl.arange(0, UPT)
        u_ok = u_off < n_units

        j = g * UPT * APG + tl.arange(0, UPT * APG)  # anchor index
        pos = (j + 1) * STRIDE - 1  # only the gate needs the position
        ok = j < n_anchor

        # Read ONCE and held. The sub-query axis is the loop below rather than a grid dimension,
        # so a program loads its keys once and every query tile reuses them -- which is what lets
        # them be read at stride through the paged cache instead of out of a gathered pool.
        # The keys are `cache[:, stride-1::stride]`, a view whose pages hold `APP` anchors each and
        # whose page axis is PHYSICAL, so a logical anchor reaches its row through the block table.
        # Two strides rather than one, because the cache arrives as one half of an unbound
        # (blocks, 2, page, ...) tensor and its page stride is twice its span.
        _lp = j // APP
        _pg = tl.load(BTAB + _lp, mask=ok, other=0).to(tl.int64)
        k = tl.load(
            KA
            + _pg[:, None] * stride_kp
            + (j - _lp * APP)[:, None] * stride_kr
            + (h * D + d_off)[None, :] * stride_kc,
            mask=ok[:, None],
            other=0.0,
        )  # (UPT*APG, D)
        kt = tl.trans(k)
        # No early exit for a unit group past every row: the keys are already loaded, and the gate
        # below writes the same -inf.
        for s0 in range(v * VT, tl.minimum((v + 1) * VT, n_sub), BLOCK_S):
            s_off = s0 + tl.arange(0, BLOCK_S)
            s_ok = (s_off < n_sub) & (s_off < (v + 1) * VT)
            pl = tl.load(PL + s_off // NG, mask=s_ok, other=-1)
            # `tl.dot` takes one dtype. Narrowing the query is a no-op for callers that already
            # hand it over in the cache's dtype, and widening the keys instead would double what
            # the dot reads.
            q = tl.load(
                Q
                + h * stride_qh
                + s_off[:, None] * stride_qs
                + d_off[None, :] * stride_qd,
                mask=s_ok[:, None],
                other=0.0,
            )
            sc = tl.dot(q.to(k.dtype), kt).to(tl.float32)  # (BLOCK_S, UPT*APG)
            vis = ok[None, :]
            if PER_ANCHOR:
                vis = vis & (pos[None, :] <= pl[:, None])
            sc = tl.where(vis, sc, float("-inf"))
            # Scale AFTER the fold: a positive constant commutes with max even through rounding
            # (both are monotone), so this is the same bits on 1/APG of the multiplies.
            acc = tl.max(tl.reshape(sc, (BLOCK_S, UPT, APG)), axis=2) * scale
            if not PER_ANCHOR:
                # The pooled caller's `masked_fill`, on the finished score rather than per anchor
                # -- the two rank a straddling unit differently.
                acc = tl.where(u_off[None, :] * RB <= pl[:, None], acc, float("-inf"))
            tl.store(
                OUT
                + h * stride_oh
                + s_off[:, None] * stride_os
                + u_off[None, :] * stride_ou,
                acc,
                mask=s_ok[:, None] & u_ok[None, :],
            )

    @triton.jit
    def _qpool_kernel(
        QPAD,
        CNT,
        OUT,
        n_sub,
        sb,
        s_qr,
        s_qh,
        s_qg,
        s_qd,
        s_ob,
        s_os,
        s_oh,
        s_od,
        D: tl.constexpr,
        DMAX: tl.constexpr,
        GQA: tl.constexpr,
        SB: tl.constexpr,
        HALF: tl.constexpr,
    ):
        """(rows, kv heads, gqa, head_dim) -> (query blocks, sub-queries, kv heads, head_dim) fp32.

        It is a sum over the score block, a divide by the block's present-row count, and a mean
        over the GQA group, in one launch instead of three.

        The normalization/head-mean contract is components.attention._route_head_mean's
        docstring. This adapter is copied into a serving venv, so it cannot import the trainer.
        ⚠️ Each reduction ROUNDS TO THE INPUT DTYPE before the next begins, and only the result is
        upcast. Pooling in fp32 throughout instead flips 12-15% of near-tie unit rankings, so every
        cast below is deliberate.
        """
        b, s, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        dd = tl.arange(0, DMAX)
        ok = dd < D
        cnt = tl.maximum(tl.load(CNT + b * n_sub + s).to(tl.float32), 1.0)
        acc = tl.zeros((DMAX,), tl.float32)
        for g in tl.static_range(GQA):
            ssum = tl.zeros((DMAX,), tl.float32)
            # Sequential over the score block: `static_range` would unroll a wide block into a
            # compile that never finishes, and the order is the one a strided reduction takes.
            for r in range(SB):
                row = (b * n_sub + s) * sb + r
                ssum += tl.load(
                    QPAD + row * s_qr + h * s_qh + g * s_qg + dd * s_qd,
                    mask=ok,
                    other=0.0,
                ).to(tl.float32)
            # `sum` on a half tensor accumulates in fp32 and casts back; the divide then promotes,
            # divides and casts back again. Both roundings happen, in that order.
            sr = ssum.to(tl.float16) if HALF else ssum.to(tl.bfloat16)
            q = tl.div_rn(sr.to(tl.float32), cnt)
            value = (q.to(tl.float16) if HALF else q.to(tl.bfloat16)).to(tl.float32)
            acc = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                "=r,r,r",
                [acc, value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        m = tl.div_rn(acc, GQA)
        out = (m.to(tl.float16) if HALF else m.to(tl.bfloat16)).to(tl.float32)
        tl.store(OUT + b * s_ob + s * s_os + h * s_oh + dd * s_od, out, mask=ok)


_SCORE_SHAPE: dict = {}


if _HAS_DECODE_SCORE:

    @triton.jit
    def _vote_stats_kernel(
        SQ,
        CNT,
        MOUT,
        ZOUT,
        n_units,
        n_vote,
        cnt_per_row,
        stride_r,
        BLOCK_U: tl.constexpr,
    ):
        """Per (row, vote): the max and the exp-sum over the WHOLE unit axis, in one pass.

        `smax` normalises each vote's ballot over every unit before the votes are compared, so the
        denominator spans an axis the combine kernel tiles across. Reducing it here costs one read
        of the score and leaves the combine kernel with a scalar per vote.

        Running max and rescaled sum, as a softmax does online: a second pass over the units would
        double the read for a value the first pass already saw.
        """
        r = tl.program_id(0)  # flat (row, vote), row-major
        if tl.load(CNT + ((r // n_vote) // cnt_per_row) * n_vote + (r % n_vote)) == 0:
            # An empty sub-query bin: every unit is -inf, the ballot is not cast, and the combine
            # kernel reads Z == 0 as "this vote contributes nothing" rather than dividing by it.
            tl.store(MOUT + r, float("-inf"))
            tl.store(ZOUT + r, 0.0)
            return
        m = float("-inf")
        z = 0.0
        for u0 in range(0, n_units, BLOCK_U):
            u = u0 + tl.arange(0, BLOCK_U)
            ok = u < n_units
            v = tl.load(SQ + r * stride_r + u, mask=ok, other=float("-inf"))
            m_new = tl.maximum(m, tl.max(v))
            # `m_new` is -inf only when every unit seen so far is -inf, and then `z` is still 0 and
            # the rescale below would be 0 * exp(inf). Guard the rescale, not the result.
            z = tl.where(
                m_new == float("-inf"),
                0.0,
                z * tl.exp(m - m_new) + tl.sum(tl.where(ok, tl.exp(v - m_new), 0.0)),
            )
            m = m_new
        tl.store(MOUT + r, m)
        tl.store(ZOUT + r, z)

    @triton.jit
    def _vote_combine_kernel(
        SQ,
        CNT,
        MIN_,
        ZIN,
        OUT,
        COUT,
        n_units,
        n_vote,
        cnt_per_row,
        stride_r,
        stride_or,
        REDUCE: tl.constexpr,
        BLOCK_U: tl.constexpr,
    ):
        """(rows, votes, units) -> (rows, VS, units): mask the empty bins, reduce over the votes.

        The vote axis is a loop over one tile of units, so only the tile is live and the score is
        read once instead of being materialised at each step of a mask / softmax / reduce chain.

        A unit no vote could score must stay -inf, but `smax` returns a probability and would put
        it at exactly 0.0, where the downstream `vals == -inf` guard would let it win a top-k slot.
        So finiteness is tracked alongside the reduce rather than read back off the result.

        The grid's z extent `VS` cuts the VOTE axis so the grid can outgrow the row count. It is
        read at run time, not as a constexpr, because it follows the chunk's q-block count, which
        changes from step to step. Every reduce here is associative over votes, so `VS` pieces
        plus a combine is exact. Each piece writes its own row of `OUT`; `mean` also writes its
        count to `COUT`, because a piece's average cannot be averaged again without one.
        """
        r, vc, VS = tl.program_id(0), tl.program_id(2), tl.num_programs(2)
        u = tl.program_id(1) * BLOCK_U + tl.arange(0, BLOCK_U)
        ok = u < n_units
        acc = (
            tl.zeros((BLOCK_U,), tl.float32)
            if REDUCE != 1
            else tl.full((BLOCK_U,), float("-inf"), tl.float32)
        )
        cnt_f = tl.zeros((BLOCK_U,), tl.float32)
        fin = tl.zeros((BLOCK_U,), tl.int1)
        _per = (n_vote + VS - 1) // VS
        for v in range(vc * _per, min((vc + 1) * _per, n_vote)):
            if tl.load(CNT + (r // cnt_per_row) * n_vote + v) != 0:
                sv = tl.load(
                    SQ + (r * n_vote + v) * stride_r + u, mask=ok, other=float("-inf")
                )
                f = ok & (sv > float("-inf"))
                fin = fin | f
                if REDUCE == 0:  # smax
                    z = tl.load(ZIN + r * n_vote + v)
                    mm = tl.load(MIN_ + r * n_vote + v)
                    p = tl.where(f & (z > 0.0), tl.exp(sv - mm) / z, 0.0)
                    acc = tl.maximum(acc, p)
                elif REDUCE == 1:  # amax
                    acc = tl.maximum(acc, tl.where(f, sv, float("-inf")))
                else:  # mean over the finite votes
                    acc += tl.where(f, sv, 0.0)
                    cnt_f += tl.where(f, 1.0, 0.0)
        # `mean` divides in the combine, not here: a piece holds a partial SUM and its count, and
        # dividing first would average the averages.
        _o = (r * VS + vc) * stride_or + u
        if REDUCE == 2:
            tl.store(OUT + _o, acc, mask=ok)
            tl.store(COUT + _o, cnt_f, mask=ok)
        else:
            tl.store(OUT + _o, tl.where(fin, acc, float("-inf")), mask=ok)


_VOTE_REDUCE_CODE = {"smax": 0, "amax": 1, "mean": 2}


def _vote_reduce(sq, cnt, *, reduce):
    """(R, V, U) scores + (R', V) present-row counts -> (R, U), fp32.

    One kernel, or two for `smax`, which needs a per-vote denominator over the whole unit axis.
    `R` folds every axis the reduce does NOT run over -- query blocks, and the KV heads when
    `kv_route_head_shared=False` keeps them -- so one call serves both selections. `cnt` may be
    narrower than `R`, where one count row is shared by the heads; `R` must then be a whole
    multiple of it.
    """
    if reduce not in _VOTE_REDUCE_CODE:
        raise ValueError(
            f"kv_route_reduce={reduce!r} is not one of {sorted(_VOTE_REDUCE_CODE)}; refusing "
            "rather than reducing by something the checkpoint was not trained with."
        )
    if sq.dim() != 3 or cnt.dim() != 2:
        raise ValueError(
            f"the vote reduce takes a 3-D score and a 2-D count, got {tuple(sq.shape)} and "
            f"{tuple(cnt.shape)}. Fold every axis the reduce does not run over into the first."
        )
    R, V, U = sq.shape
    if cnt.shape[1] != V or R % cnt.shape[0]:
        raise ValueError(
            f"score rows {R} x votes {V} do not line up with counts {tuple(cnt.shape)}: the vote "
            f"widths must match and the rows must be a whole multiple of the count rows (the "
            f"per-head path shares one count row across the KV heads)."
        )
    if not sq.is_contiguous() or not cnt.is_contiguous():
        raise ValueError(
            "the vote reduce indexes the score and the counts as flat rows, so both must be "
            "contiguous; a permuted view would be read in the wrong order and silently rank on "
            "the wrong unit."
        )
    cpr = R // cnt.shape[0]
    code = _VOTE_REDUCE_CODE[reduce]
    if code == 0:
        mm = sq.new_empty(R * V)
        zz = sq.new_empty(R * V)
        _vote_stats_kernel[(R * V,)](sq, cnt, mm, zz, U, V, cpr, U, BLOCK_U=1024)
    else:
        mm = zz = sq
    # Rows are query blocks -- 32 at the served shape -- so `rows x unit tiles` leaves a 148-SM
    # part running 64 programs, each walking every vote in series. Cutting the vote axis is what
    # fills it, and it is exact because every reduce here is associative over votes. Measured at
    # the served shape, output identical: 166.5 us shipped, 24.3 at 8 pieces and BLOCK_U=512.
    # More pieces than there are votes would leave programs with nothing to do.
    # 8 pieces filled the DECODE shape (R=32, hundreds of programs); the prefill sub-query
    # election arrives with a handful of query blocks and thousands of votes, where 8 leaves the
    # device near-idle and each program walking ~V/8 votes in series. Scale the cut to fill
    # `_SCORE_TARGET_PROGRAMS`, floored at the measured decode choice so that shape is untouched.
    # Exact for `smax` and `amax` (the pieces fold with a max, associative and idempotent); the
    # `mean` arm re-adds fp32 partial sums, so its cut is a reassociation -- within the repo's
    # reorder band, but not bit-identical across cuts.
    _progs = R * (-(-U // 512))
    VS = max(1, min(V, max(8, -(-_SCORE_TARGET_PROGRAMS // max(1, _progs)))))
    BLOCK_U = 512
    part = sq.new_empty(R * VS, U)
    cpart = sq.new_empty(R * VS, U) if code == 2 else part
    _vote_combine_kernel[(R, -(-U // BLOCK_U), VS)](
        sq,
        cnt,
        mm,
        zz,
        part,
        cpart,
        U,
        V,
        cpr,
        U,
        part.stride(0),
        REDUCE=code,
        BLOCK_U=BLOCK_U,
    )
    if VS == 1:
        out = (
            part.view(R, U)
            if code != 2
            else torch.where(
                cpart.view(R, U) > 0.0,
                part.view(R, U) / cpart.view(R, U).clamp(min=1.0),
                torch.full_like(part.view(R, U), float("-inf")),
            )
        )
        return out
    _p = part.view(R, VS, U)
    if code == 2:
        # A piece holds a partial SUM and its count: totals first, then one division. Averaging
        # the pieces' averages would weight a piece that saw two votes like one that saw a hundred.
        _c = cpart.view(R, VS, U).sum(1)
        return torch.where(
            _c > 0.0, _p.sum(1) / _c.clamp(min=1.0), torch.full_like(_c, float("-inf"))
        )
    return _p.amax(1)


def _pool_query(qpad, cnt_sub, *, n_qb, n_sub, hkv, gqa, d, score_block):
    """Pool the padded query down to one vector per (query block, sub-query, kv head), fp32.

    Normalization and head reduction follow components.attention._route_head_mean's
    contract on both dispatch paths. Small grids retain the tensor sequence sum (a wide
    serial walk underfills the GPU), then use the same kernel with SB=1 for the fixed mean.
    The sequence sum itself may round differently between tree and serial reductions;
    the fixed head mean does not claim to remove that independent difference.
    """
    if qpad.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"the routed prefill's query pooling takes bf16 or fp16, got {qpad.dtype}. The pool "
            f"rounds to the INPUT dtype between reductions, so a dtype it cannot name would pool "
            f"in a different order than the checkpoint was selected under. Cast the query, or add "
            f"the dtype to the kernel."
        )
    if _HAS_DECODE_SCORE and qpad.is_cuda:
        if n_qb * n_sub * hkv < 256:
            qpad = qpad.view(n_qb, n_sub, score_block, hkv, gqa, d).sum(2)
            qpad = qpad.reshape(n_qb * n_sub, hkv, gqa, d)
            score_block = 1
        out = qpad.new_empty((n_qb, n_sub, hkv, d), dtype=torch.float32)
        _qpool_kernel[(n_qb, n_sub, hkv)](
            qpad,
            cnt_sub,
            out,
            n_sub,
            score_block,
            qpad.stride(0),
            qpad.stride(1),
            qpad.stride(2),
            qpad.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            D=d,
            DMAX=triton.next_power_of_2(d),
            GQA=gqa,
            SB=score_block,
            HALF=qpad.dtype == torch.float16,
            num_warps=4,
        )
        return out
    pooled = qpad.view(n_qb, n_sub, score_block, hkv, gqa, d).sum(2)
    heads = (pooled.float() / cnt_sub.clamp(min=1).float()[..., None, None, None]).to(
        qpad.dtype
    )
    total = heads[:, :, :, 0].float()
    for g in range(1, gqa):
        total = total + heads[:, :, :, g].float()
    return (total / gqa).to(qpad.dtype).float()


def _prefill_score(
    q,
    ka_view,
    btab_row,
    pl_t,
    *,
    n_units,
    apg,
    stride,
    rb,
    d,
    scale,
    n_sub_per_block,
    n_anchor,
    per_anchor,
):
    """(H, n_sub, D) query x the anchor grid -> (H, n_sub, n_units) unit scores, fp32.

    `pl_t` is one last-position per QUERY BLOCK; sub-query `s` belongs to block
    `s // n_sub_per_block`. `ka_view` is `cache[:, stride-1::stride]` flattened to
    (pages, anchors_per_page, hkv*d) -- the anchors where they lie, which is the same tensor the
    anchor attention reads; `btab_row` is this request's block table, which turns a logical anchor
    page into the physical one. `n_anchor` is how many anchors the chunk closed, shared by every
    query block.
    """
    H, n_sub = q.shape[0], q.shape[1]
    out = torch.empty((H, n_sub, n_units), dtype=torch.float32, device=q.device)
    # Tile candidates, in measured order rather than by width: `UPT` widens the dot's N,
    # `BLOCK_S` trades the query tile against occupancy, and `num_warps` cuts the tile across the
    # block, so the three are chosen together. The list is walked down because a shape can ask for
    # more tensor memory than the part has, which depends on the caller's dtypes and head count.
    #
    # ⚠️ These cannot be ranked on a standalone bench of this kernel. A bench gives every query
    # block the same last position so every unit is live, while the served path cuts each block's
    # unit axis at its own position and the work is triangular; ranking on the bench picked a shape
    # that measured slower in the engine.
    _u_max = max(1, triton.next_power_of_2(n_units))
    _b_max = max(16, min(256, triton.next_power_of_2(n_sub)))
    _shapes = [
        (u, b, w)
        for u, b, w in (
            (2, 64, 8),
            (4, 64, 8),
            (8, 128, 8),
            (2, 128, 8),
            (4, 128, 8),
            (8, 64, 8),
            (1, 64, 8),
            (1, 128, 8),
            (2, 32, 8),
            (1, 32, 8),
            (1, 16, 8),
        )
        if u <= _u_max and b <= _b_max
    ]
    if not _shapes:
        # A shape always exists -- 16 is the narrowest tile `tl.dot` takes and 1 unit is the
        # fewest there are -- so an empty list means the bounds above are wrong, not that the
        # device is short.
        raise RuntimeError(
            f"no prefill scorer tile candidates for n_units={n_units}, n_sub={n_sub} "
            f"(bounds: units<={_u_max}, sub-queries<={_b_max}) -- the bounds are wrong."
        )
    _last = None
    for UPT, BLOCK_S, _WARPS in _shapes:
        # Cut the sub-query loop across a z axis too: (head x unit group) alone under-fills the
        # device and every extra vote then runs in series (the per-head election is x`gqa` votes).
        # Sized for `_SCORE_TARGET_PROGRAMS`; each z-program re-reads its ~128KB key tile, which
        # is nothing.
        _ng = H * (-(-n_units // UPT))
        _zt = max(
            1, min(-(-n_sub // BLOCK_S), -(-_SCORE_TARGET_PROGRAMS // max(1, _ng)))
        )
        _vt = -(-(-(-n_sub // _zt)) // BLOCK_S) * BLOCK_S
        _zt = -(-n_sub // _vt)
        try:
            _prefill_score_kernel[(H, -(-n_units // UPT), _zt)](
                q,
                ka_view,
                btab_row,
                pl_t,
                out,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                ka_view.stride(0),
                ka_view.stride(1),
                ka_view.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                scale,
                n_sub,
                n_anchor,
                n_units,
                _vt,
                APG=apg,
                APP=ka_view.shape[1],
                STRIDE=stride,
                RB=rb,
                D=d,
                BLOCK_S=BLOCK_S,
                UPT=UPT,
                NG=n_sub_per_block,
                PER_ANCHOR=per_anchor,
                num_warps=_WARPS,
                num_stages=2,
            )
            if _SCORE_SHAPE.get(id(q.device)) != (UPT, BLOCK_S, _WARPS):
                _SCORE_SHAPE[id(q.device)] = (UPT, BLOCK_S, _WARPS)
                logger.debug(
                    "[solar_open3] prefill scorer tile: %d units x %d sub-queries, "
                    "%d warps (dot N = %d)",
                    UPT,
                    BLOCK_S,
                    _WARPS,
                    UPT * apg,
                )
            return out
        except triton.runtime.errors.OutOfResources as e:
            _last = e
    raise RuntimeError(
        f"no prefill scorer tile fits: every shape from {_shapes[0]} down to {_shapes[-1]} "
        f"(units x sub-queries) asks for more than the device has at n_units={n_units}, "
        f"n_sub={n_sub}. Last error: {_last}"
    ) from _last


_FUSED_TOPK_TILE = 1024


def _fused_route_decode(
    q2,
    qp,
    kv,
    vv,
    btab,
    pl_t,
    *,
    a_max,
    n_units,
    apg,
    stride,
    page,
    rb,
    hkv,
    scale,
    topk,
    q_block,
    dedup,
    out_dtype=torch.float32,
    max_reduce=None,
    tkv=None,
    sink_units=0,
    qhead=False,
    group_size=None,
    exact_ballot=False,
    output_gate=None,
    output=None,
):
    """The routed decode with the anchor keys read ONCE.

    Four launches: the anchor half and the ranking together, the top-k, the selected half, and the
    combine that folds every split into one softmax. The selected half writes into the SAME partial
    buffer as the anchor half, past its splits, so the combine sees one set of splits and there is
    no second merge.
    """
    # ⚠️ The kernels address V with the KEY's strides (`s_half=0`), which is exact only while
    # the two caches are the same shape and layout. SGLang hands over two independent tensors, so
    # nothing in the type system says they still are; a pool layout that changes one of them would
    # read V from the wrong addresses and score routes on garbage without raising.
    if vv.shape != kv.shape or vv.stride() != kv.stride():
        raise ValueError(
            "the routed decode addresses the value cache with the key cache's strides, so the two "
            f"must share a layout; got key {tuple(kv.shape)}/{kv.stride()} and value "
            f"{tuple(vv.shape)}/{vv.stride()}."
        )
    n, hq, d = q2.shape
    gqa = hq // hkv
    dev = q2.device
    # Same rule as `_anchor_attn_paged`.
    if not _DEC_SPLITS_FIXED:
        _want = -(-_DEC_TARGET_PROGRAMS // max(1, n * hkv))
        splits = max(8, min(64, triton.next_power_of_2(_want)))
    else:
        splits = _DEC_SPLITS
    # Never slice the anchor axis below one UNIT. A split shorter than `apg` would put a unit's
    # anchors in two splits, and neither split's max would be that unit's. At the serving ceiling
    # there are thousands of units and this changes nothing; it is short contexts -- where the
    # anchors a row has are fewer than `splits * apg` -- that the cap keeps servable.
    splits = max(1, min(splits, a_max // apg))
    per = triton.next_power_of_2((a_max + splits - 1) // splits)
    # The tile has to hold a whole number of units too. `apg` is a power of two here (route_block
    # and stride both are), so rounding the tile up to a multiple of it keeps it a power of two.
    block_a = max(apg, min(_DEC_BLOCK_A, per))
    if block_a % apg or per % apg:
        raise ValueError(
            f"the fused route decode needs its anchor tile and split to hold whole units: "
            f"tile={block_a}, split={per}, anchors per unit={apg}, anchors={a_max}. "
            f"Lower SOLAR_OPEN3_DEC_BLOCK_A to a multiple of {apg}."
        )
    k_eff = min(topk, n_units) if topk else n_units
    LIVE_SPLIT = _FUSED_LIVE_SPLIT
    # The selected region rides its own splits, sized so a program's tile is the same width the
    # anchor pass uses -- it is a sixteenth of the anchor axis at this model's shape.
    sel_w = k_eff * rb
    splits_s = max(1, min(splits, -(-sel_w // block_a)))
    per_s = triton.next_power_of_2((sel_w + splits_s - 1) // splits_s)

    # The combine reads its split axis with `tl.arange`, which takes a power of two, and the two
    # halves' split counts do not add to one. The buffer is widened to the next power of two and
    # the surplus slots are given the state an empty split leaves -- no maximum, no mass, no
    # output -- so the combine folds them in and they change nothing.
    splits_t = triton.next_power_of_2(splits + splits_s)
    o = torch.empty((n, hq, splits_t, d), device=dev, dtype=torch.float32)
    m = torch.empty((n, hq, splits_t), device=dev, dtype=torch.float32)
    l = torch.empty_like(m)
    if splits_t > splits + splits_s:
        o[:, :, splits + splits_s :].zero_()
        m[:, :, splits + splits_s :].fill_(float("-inf"))
        l[:, :, splits + splits_s :].zero_()
    # With the live split the kernel only visits the anchors the row has, so units past the row
    # are never written and the buffer has to arrive holding what the ranking wants for them.
    # The per-head buffer is ALWAYS pre-filled: its kernel skips the empty-split store entirely
    # (see the kernel's else branch), so -inf has to be what an unwritten slot reads as.
    if qhead:
        us = torch.full(
            (n, hkv, gqa, n_units), float("-inf"), device=dev, dtype=torch.float32
        )
    else:
        us = (
            torch.full(
                (n, hkv, n_units), float("-inf"), device=dev, dtype=torch.float32
            )
            if LIVE_SPLIT
            else torch.empty((n, hkv, n_units), device=dev, dtype=torch.float32)
        )
    q0_t = (pl_t // q_block) * q_block
    qp_t = qp if qp is not None else us.flatten(2)[:, :, :1]

    _route_anchor_part[(n, hkv, splits)](
        q2,
        qp_t,
        kv,
        vv,
        btab,
        pl_t,
        q0_t,
        o,
        m,
        l,
        us,
        q2.stride(0),
        q2.stride(1),
        q2.stride(2),
        qp_t.stride(0),
        qp_t.stride(1),
        qp_t.stride(2),
        kv.stride(0),
        0,
        kv.stride(1),
        kv.stride(2),
        kv.stride(3),
        btab.stride(0),
        btab.stride(1),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        us.stride(0),
        us.stride(1),
        us.stride(-1),
        us.stride(2) if qhead else 0,
        scale,
        a_max,
        n_units,
        btab.shape[1],
        rb,
        APG=apg,
        STRIDE=stride,
        PAGE=page,
        GQA=gqa,
        BLOCK_H=max(16, triton.next_power_of_2(gqa + 1)),
        D=d,
        BLOCK_A=block_a,
        PER=per,
        QP_GIVEN=qp is not None,
        SPLITS=splits,
        LIVE_SPLIT=LIVE_SPLIT,
        # The per-tile page id is only exact while a tile of `block_a` anchors fits inside one
        # page; otherwise the per-row path stays.
        TILE_IN_PAGE=(_TILE_PAGE and block_a * stride <= page),
        QHEAD_US=qhead,
        **({"num_warps": _DEC_WARPS} if _DEC_WARPS else {}),
    )
    if qhead:
        # The per-head election, on the raw maxima the kernel just emitted: one softmax ballot
        # per (row, KV head, query head) over the UNIT axis, then the unit's best ballot over
        # both head axes -- the same fold the prefill sub-query branch applies at T rows. Two
        # launches, not the eight torch ops they replace: at ~9 elementwise kernels per routed
        # layer the fold's LAUNCHES were the decode's whole +8% over pooled, while its bytes
        # never mattered. Semantics unchanged from that torch tail, and pinned against it by
        # `test_qhead_fold_kernel_matches_the_torch_tail`:
        #   * a ballot with no rankable unit comes out of the stats pass as (M=-inf, Z=0) and
        #     casts no mass -- the `ok`-zeroing (a NaN otherwise);
        #   * a unit past the row goes back to -inf AFTER the fold: its ballot mass 0.0 is a
        #     valid score the -inf sentinel would let through, and a row with fewer live units
        #     than the budget would then select pages past itself;
        #   * own forcing lands after the fold, never before the softmax (a +inf there would
        #     one-hot that head's ballot) -- the "force after the reduce" contract.
        _rows = hkv * gqa
        _uflat = us.view(n * _rows, n_units)
        _mz = torch.empty(n * _rows * 2, device=dev, dtype=torch.float32)
        _qhead_ballot_stats[(n * _rows,)](
            _uflat,
            _mz,
            pl_t,
            _uflat.stride(0),
            _uflat.stride(1),
            n_units,
            ROWS=_rows,
            RB=rb,
            BLOCK_U=1024,
            EXACT=exact_ballot,
            UBK=max(1, 128 // apg),
        )
        usf = torch.empty(n, n_units, device=dev, dtype=torch.float32)
        _qhead_ballot_fold[(n, triton.cdiv(n_units, 1024))](
            us,
            _mz,
            pl_t,
            q0_t,
            usf,
            us.stride(0),
            us.stride(2),
            us.stride(3),
            usf.stride(0),
            usf.stride(1),
            n_units,
            rb,
            ROWS=_rows,
            BLOCK_U=1024,
        )
    else:
        # The KV heads fold with a max, which is what `kv_route_head_reduce="amax"` asks for and
        # what every rank's cross-rank combine already is.
        usf = us.amax(1) if hkv > 1 else us[:, 0]
    # ...and across the ranks that hold the rest of them. `us` only ever carries this rank's KV
    # heads, so without this the top-k below is taken over a SUBSET of the heads the checkpoint
    # ranks with: every rank selects a different set of units, o_proj sums partial outputs of
    # different unions, and nothing raises. `max` of the local `max`es is the global `max`, and
    # the guard is the same one the prefill scorer uses -- when this rank already holds every KV
    # head there is nothing to combine. It serves the per-head election too: a query head lives
    # whole on its rank, so its softmax ballot is rank-local and the cross-rank max combines
    # finished ballots. (`head_reduce="smax"` stays refused -- its fold is not a max.)
    # ⚠️ For the per-head election the head test is NOT the split test: at ONE total KV head
    # every rank holds that head (tkv == hkv) while the query group is still split `tp` ways,
    # and each rank's ballots then cover only its own query heads. The election is over ALL
    # query heads, so the ranks combine whenever the group is split -- the same predicate the
    # prefill's per-head SUM gather asks (`group_size != gqa`). The max is idempotent, so a
    # combine that turns out unnecessary changes nothing.
    _split_group = qhead and group_size is not None and group_size != gqa
    if max_reduce is not None and tkv is not None and (tkv != hkv or _split_group):
        usf = max_reduce(usf.contiguous())
    if sink_units:
        # Sink forcing at the same station the own unit is forced (the kernel scored it +inf):
        # after the head fold and the TP max-reduce, before the ranking. Gated by the row's live
        # length so the selection stays well-formed: the grid is sized from max_model_len, so a
        # young request has units past its row on the grid, and an ungated +inf would rank one
        # into `sidx`. It cannot reach the output -- `_selected_attn_part` keeps only
        # `pos <= pl` and masks the loads, and a row that short has fewer candidates than the
        # budget, so no real unit is displaced either -- but the trainer never selects a unit
        # past a block's history and the served selection should not either. Plain tensor ops
        # on device-side `pl_t`, so the cudagraph capture path serves it unchanged.
        _su = torch.arange(min(sink_units, n_units), device=dev)
        # In place: `usf` is this call's own fold (or the `us[:, 0]` view of a tensor nothing
        # reads again), so one kernel and no `(rows, sink_units)` temporary.
        usf[:, : _su.numel()].masked_fill_(
            _su[None, :] * rb <= pl_t[:, None], float("inf")
        )
    if _use_fused_topk():
        _val, _idx = topk_units(usf, k_eff)
    else:
        _val, _idx = torch.topk(usf, k_eff, -1)
    # A slot the ranking could not fill holds the sentinel one past the last unit; the selected
    # pass masks it out rather than clamping it onto a real unit.
    sidx = _idx.masked_fill(_val == float("-inf"), n_units).sort(-1).values.contiguous()

    _selected_attn_part[(n, hkv, splits_s)](
        q2,
        kv,
        vv,
        btab,
        pl_t,
        o,
        m,
        l,
        sidx,
        q2.stride(0),
        q2.stride(1),
        q2.stride(2),
        kv.stride(0),
        0,
        kv.stride(1),
        kv.stride(2),
        kv.stride(3),
        btab.stride(0),
        btab.stride(1),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        sidx.stride(0),
        sidx.stride(1),
        scale,
        n_units,
        btab.shape[1],
        splits,
        STRIDE=stride,
        PAGE=page,
        GQA=gqa,
        BLOCK_H=max(16, triton.next_power_of_2(gqa)),
        D=d,
        BLOCK_A=block_a,
        RB=rb,
        K_EFF=k_eff,
        PER_S=max(16, per_s),
        DEDUP=dedup,
        **({"num_warps": _DEC_WARPS} if _DEC_WARPS else {}),
    )

    out = (
        torch.empty((n, hq, d), device=dev, dtype=out_dtype)
        if output is None
        else output.view(n, hq, d)
    )
    lse = torch.empty((n, hq), device=dev, dtype=torch.float32)
    _anchor_attn_combine[(n, hq)](
        o,
        m,
        l,
        out,
        lse,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        SPLITS=splits_t,
        D=d,
        OUT_F32=out_dtype == torch.float32,
        GATE=output_gate,
        stride_g=output_gate.stride(0) if output_gate is not None else 0,
        HAS_GATE=output_gate is not None,
    )
    return out.reshape(n, hq * d)


def _use_fused_topk() -> bool:
    """Whether `topk_units` can run: only triton's absence says no. It handles any width itself,
    so no k is ever sent to `torch.topk` instead -- which is why this takes none.
    """
    return _HAS_DECODE_SCORE


def topk_units(us: torch.Tensor, k: int):
    """`(..., n_units)` scores -> `(values, indices)`, the pair `torch.topk(us, k, -1)` returns.

    Two passes: every chunk of `_FUSED_TOPK_TILE` units keeps its own best k, then those
    candidates are cut back to k. The overall top-k is a subset of the union of the per-chunk
    top-k, so this is exact, and the scores are read once.
    """
    n_units = us.shape[-1]
    # `tl.topk` takes a power-of-two width of at least two, so a narrower `k` is ranked at the
    # next one up and sliced: the result comes back sorted, so the first `k` of a wider ranking
    # ARE the top `k`.
    kp = max(2, triton.next_power_of_2(k))
    if kp > _FUSED_TOPK_TILE:
        raise ValueError(
            f"a top-k of {k} needs a {kp}-wide ranking, over the {_FUSED_TOPK_TILE} units a chunk "
            f"holds. Raise _FUSED_TOPK_TILE, or rank fewer units."
        )
    flat = us.reshape(-1, n_units)
    n_rows = flat.shape[0]
    n_chunks = triton.cdiv(n_units, _FUSED_TOPK_TILE)
    cand = torch.empty((n_rows, n_chunks * kp), dtype=torch.int64, device=us.device)
    _chunk_topk_kernel[(n_rows, n_chunks)](
        flat,
        cand,
        flat.stride(0),
        flat.stride(1),
        cand.stride(0),
        n_units,
        K=kp,
        TILE=_FUSED_TOPK_TILE,
    )
    idx = torch.empty((n_rows, kp), dtype=torch.int32, device=us.device)
    val = torch.empty((n_rows, kp), dtype=torch.float32, device=us.device)
    _merge_topk_kernel[(n_rows,)](
        cand,
        idx,
        val,
        cand.stride(0),
        kp,
        n_chunks * kp,
        K=kp,
        CAND_P=triton.next_power_of_2(n_chunks * kp),
    )
    return (
        val[:, :k].view(*us.shape[:-1], k),
        idx[:, :k].view(*us.shape[:-1], k).to(torch.int64),
    )


try:
    from flash_attn.cute.interface import flash_attn_varlen_func as _FA4_VARLEN
except Exception:  # pragma: no cover - venv without flash-attn-4
    _FA4_VARLEN = None


_VARLEN = _FA4_VARLEN


def _varlen_lse(
    q, k, v, *, cu_q, max_q, cu_k, max_k, seqused=None, btab=None, scale, causal=False
):
    """(out, lse) from whichever varlen kernel is present.

    ``causal`` is BOTTOM-RIGHT aligned in both kernels: key ``j`` is masked for query ``i`` of a
    segment iff ``j > i + (k_len - q_len)``. The boundary pass relies on that alignment rather
    than on q_len == k_len -- see the own-half fast path in ``routed_prefill_batched``.

    FA4 returns lse as (num_heads, total_q) in natural log -- verified against torch.logsumexp
    (3.6e-5; vLLM's own varlen was 1.9e-6 on the same check, from when both were candidates).
    """
    return _FA4_VARLEN(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max_q,
        cu_seqlens_k=cu_k,
        max_seqlen_k=max_k,
        seqused_k=seqused,
        page_table=btab,
        softmax_scale=scale,
        causal=causal,
        return_lse=True,
    )


_ROUTE_LOGGED: set = set()


_PAD_ROUTE = int(os.environ.get("SOLAR_OPEN3_PAD_ROUTE", "0") or 0)


def _fused_decode_ok(
    *,
    head_shared,
    anchor_in_union,
    anchor_topk,
    qhead_block,
    head_reduce,
    route_block,
    stride,
) -> str:
    """ "" if the fused decode serves this configuration, else why it does not.

    Most entries below are features the fused body does not express: it ranks once per request,
    puts every anchor in the union, caps nothing, and ranks off the rows its attention computes --
    the group-pooled padding row, or each query head's own (``kv_route_qhead_block=1``). The
    reason is returned rather than a bool so the caller can name what refused.
    """
    if not _HAS_DECODE_SCORE:
        return "triton is not importable (every kernel this decode launches is a triton kernel)"
    if not head_shared:
        return "kv_route_head_shared=False (the fused body selects once per request)"
    if not anchor_in_union:
        # Unreachable through a built layer -- the constructor refuses this first. Kept so the
        # decode gate still names every shape it cannot express, rather than depending on a check
        # in another class to have run.
        return (
            "kv_anchor_in_attention=False (the anchor pass IS the union's anchor half)"
        )
    if anchor_topk:
        return "kv_route_anchor_topk (the anchor pass attends every anchor it scores)"
    if qhead_block is not None and qhead_block != 1:
        # 1 is served: the kernel emits per-head unit maxima off the rows its attention already
        # computes (`QHEAD_US`), and the host folds the ballots. Any other value has no shape.
        return f"kv_route_qhead_block={qhead_block} (only 1, every head votes, is expressible)"
    if head_reduce == "smax":
        return "kv_route_head_reduce='smax' (the heads fold with a max, in registers)"
    if route_block % stride:
        return f"route_block={route_block} is not a multiple of stride={stride}"
    return ""


_GLUON_MOD = None  # the deployed gluon module, memoised on success only


GLUON_SM_MAJOR = 10


def _current_sm_major(dev=None):
    """The compute-capability major of the device the layer runs on, or None off CUDA."""
    if not torch.cuda.is_available():
        return None
    if dev is not None and dev.type != "cuda":
        return None
    return int(torch.cuda.get_device_capability(dev)[0])


def _routing_tristate(value, name: str):
    """A three-way routing declaration: None (absent or null -- the facts decide), True, False.

    ``bool()`` accepts anything, so the value is read the way `_routing_flag` reads the two-way
    switches: a boolean, the string form of one, or null; anything else is refused by name
    rather than rounded to True.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(
        f"{name}={value!r} is not a boolean. null (or absent) lets the part decide the prefill "
        "body and logs it, true requires the fused gluon body (and refuses to build where it "
        "cannot serve), false serves the three-piece body."
    )


def _gluon_prefill_mod():
    """Resolve the Gluon implementation shipped with SGLang."""
    global _GLUON_MOD
    if _GLUON_MOD is None:
        from sglang.kernels.ops.attention.solar_open3 import gluon_prefill

        _GLUON_MOD = gluon_prefill
    return _GLUON_MOD


def _resolve_prefill_body(
    *,
    declared,
    score_block,
    qhead_block,
    reduce,
    topk,
    head_shared,
    dedup_anchor,
    stride,
    rb,
    q_block,
    head_dim,
    sm_major,
):
    """Which body serves the prefill: ``("gluon" | "three-piece", why)``. Decided once.

    Ordered as the review that asked for it lays it out. Deployer FACTS come first and pick
    the three-piece body with a reason to log (a declaration of ``true`` turns each into a
    refusal, since a declaration that cannot be honoured must not be served silently):
    ``kv_route_fused_prefill=false``; not the per-head election; nothing to elect (topk 0);
    not an sm100 part. Past those the deployment asked for the per-head election on the part
    that serves it, and anything that keeps the fused body from serving is UNMET INTENT --
    the module not shipped, a shape the pass does not fold -- refused by name, under a null
    and a true declaration alike. Called from the layer constructor (every TP rank derives the
    same answer from the same config, so no collective is involved) and repeated at the top of
    the fused branch of `routed_prefill_batched` for the tests and probes that call the body
    directly.
    """
    if declared is False:
        return "three-piece", "kv_route_fused_prefill=false"

    def _fact(why):
        if declared is True:
            raise ValueError(
                f"kv_route_fused_prefill=true declares the fused gluon prefill, but {why}. "
                "Leave the key unset to let the part decide, or set it false."
            )
        return "three-piece", why

    if not (score_block == 1 and qhead_block == 1 and reduce == "smax"):
        return _fact(
            "this is not the per-head election (the fused body serves "
            "kv_route_score_block=1, kv_route_qhead_block=1, kv_route_reduce='smax')"
        )
    if not topk:
        return _fact("kv_route_topk=0 elects nothing (own units only)")
    if sm_major is None:
        return _fact("there is no CUDA device (the pass is a CUDA kernel)")
    if sm_major != GLUON_SM_MAJOR:
        return _fact(
            f"this is an sm{sm_major}x part (the pass is sm{GLUON_SM_MAJOR}0 tcgen05 "
            "code)"
        )
    gm = _gluon_prefill_mod()  # ImportError, by name, if it is not shipped

    def _unmet(what):
        raise NotImplementedError(
            f"the per-head election on an sm{sm_major}0 part takes the fused gluon prefill, "
            f"but {what}. Set kv_route_fused_prefill=false to serve the three-piece body (same "
            "election, slower)."
        )

    if not head_shared:
        _unmet("kv_route_head_shared=False (the pass elects once per query block)")
    if not dedup_anchor:
        _unmet("kv_route_dedup_anchor=False (the split IS the dedup union)")
    if stride <= 1 or rb % stride:
        _unmet(
            f"kv_subsample_stride={stride} / kv_route_block={rb} (stride must divide the "
            "unit)"
        )
    if q_block % rb:
        # The own pass covers exactly one query block; a unit straddling two blocks would leave
        # its tokens before q0 in no piece. The three-piece body widens its own window instead.
        _unmet(
            f"kv_route_q_block={q_block} is not a multiple of kv_route_block={rb} (the own "
            "pass covers exactly one query block)"
        )
    if q_block != gm.PASS1_BLOCK_M:
        _unmet(
            f"kv_route_q_block={q_block} != the pass's query tile ({gm.PASS1_BLOCK_M})"
        )
    if rb // stride != gm.ANCHORS_PER_UNIT:
        _unmet(
            f"kv_route_block/stride={rb // stride} anchors per unit != the pass's "
            f"{gm.ANCHORS_PER_UNIT}"
        )
    if head_dim != gm.PASS1_HEAD_DIM:
        _unmet(f"head_dim={head_dim} != the pass's {gm.PASS1_HEAD_DIM}")
    if sm_major != gm.SM_MAJOR:
        _unmet(f"the module is sm{gm.SM_MAJOR}0 code and this is an sm{sm_major}0 part")
    # Last: the kernel for THIS triton. Gluon is an experimental API and the kernel is generated
    # per release (3.6/3.7 and 3.8 today); a triton with no kernel, or one whose kernel does not
    # import, is an ImportError naming the version -- at init, not at the first prefill.
    gm.load_kernel()
    return (
        "gluon",
        f"the per-head election on an sm{sm_major}0 part (triton {gm.triton.__version__})",
    )


_FUSE_RO = os.environ.get("SOLAR_OPEN3_FUSE_RO", "1") != "0"


_ROUTE_OVERLAP = os.environ.get("SOLAR_OPEN3_ROUTE_OVERLAP", "1") != "0"


_ROUTE_STREAM: dict = {}


def _anchor_stream(dev):
    """One reusable side stream per device."""
    st = _ROUTE_STREAM.get(dev.index)
    if st is None:
        st = torch.cuda.Stream(device=dev)
        _ROUTE_STREAM[dev.index] = st
    return st


def routed_prefill_batched(
    q: torch.Tensor,  # (T, num_heads * head_dim) this request-chunk's query rows
    p0: int,  # position of the first row (rows are CONSECUTIVE: p0 .. p0+T-1)
    apos: torch.Tensor,  # (A,) anchor positions ((j+1)*stride - 1)
    kf_at,  # (pos int64 tensor (n, w)) -> (k, v) each (n, w, Hkv, D) own raw
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scaling: float,
    stride: int,
    q_block: int,
    route_block: int,
    topk: int,
    anchor_in_union: bool = True,  # `kv_anchor_in_attention`; False is refused at build
    dedup_anchor: bool = False,  # `kv_route_dedup_anchor`; drop selected units' anchors
    score_block: int
    | None = None,  # `kv_route_score_block`; None -> one pooled query per block
    reduce: str = "mean",  # `kv_route_reduce`; {"mean","amax","smax"} over sub-queries
    head_reduce: str = "amax",  # `kv_route_head_reduce`; how the KV-head ballots fold
    qhead_block: int
    | None = None,  # `kv_route_qhead_block`; 1 -> each query head votes on its own
    total_kv_heads: int | None = None,
    kv_head_start: int = 0,
    group_size: int
    | None = None,  # TP: FULL query-group size (None -> q's local ratio)
    all_reduce=None,
    max_reduce=None,  # TP: MAX all-reduce; folds the head axis without moving it
    writes_score: bool = True,  # TP: does this rank contribute its heads' score?
    qhead_start: int = 0,  # TP: global index of this rank's first query head
    head_shared: bool = True,  # False -> each KV head selects on its own
    paged_kv=None,  # (key_cache, value_cache, bt_row, block_size). Lets the
    # selected region and the boundary own half read the cache
    # through a page table instead of being gathered into it.
    plan_cache=None,  # the STEP's dict; see `_pkey` below
    sink_units: int = 0,  # from `kv_route_sink_tokens` (resolved to units at init);
    # leading units forced like the own ones, inside the same
    # top-k budget (0 = the pre-sink ranking, bit-for-bit)
    fused_prefill: bool = False,  # the body `_resolve_prefill_body` fixed at init: True =
    # the fused gluon body serves this call or raises. The
    # adapters pass `kv_route_prefill_body == "gluon"`; direct
    # callers (tests, probes) declare it -- the default is the
    # three-piece body, which every configuration serves.
    output_gate: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Every query block of a prefill chunk in a handful of kernels, not a python loop.

    The per-block loop cost ~25 launches x 1,024 blocks per layer even with the readbacks
    removed. Batched, the layer is: ONE routing matmul for every block's scores, ONE paged
    varlen flash over the shared anchor pool (each block reads its own prefix -- no copy, no
    GQA expansion; the kernel takes 24q/4kv natively), ONE packed varlen flash over the
    selected raw tokens (4-head, 1/6 the copy of the expanded buffer), a grouped dense pass
    for the row-dependent boundary (<= 64 anchors + the own raw block), and an LSE merge.
    One softmax over the whole union; measured 0.41 s vs 0.97 s (loop) vs 2.55 s
    (dense mask) per 256k layer.

    Positions must be consecutive (a prefill chunk always is; the caller checks). Zero-length
    flash segments are UNDEFINED -- a seqused of 0 poisons the lse of other segments on this
    kernel -- so empty segments get one dummy key and their lse is forced to -inf on the host
    side of the merge, where emptiness is plain arithmetic.
    """
    if qhead_block is not None and score_block is None:
        # The constructor refuses this pair; repeated here because tests and probes call this
        # body directly (the decode body carries the same guard).
        raise ValueError(
            "kv_route_qhead_block=1 needs kv_route_score_block set: the pooled scoring branch averages the GQA group and never reads this knob, so the prefill would elect pooled while the decode elects per head. Set kv_route_score_block=1 (the per-head recipe) or drop kv_route_qhead_block."
        )
    dev = q.device
    T = q.shape[0]
    hq, hkv, d = num_heads, num_kv_heads, head_dim
    gqa = hq // hkv
    A = apos.shape[0]
    pe = p0 + T - 1
    qb0, qb1 = p0 // q_block, pe // q_block
    n_qb = qb1 - qb0 + 1
    rb = route_block
    if fused_prefill:
        # The fused body is DECLARED by the caller (the layer resolved it at init with the same
        # function); re-resolved here, before any device work, for the tests and probes that
        # call this body directly. A declaration the facts cannot honour is refused, not served
        # on the other body.
        if all_reduce is not None and max_reduce is None:
            raise ValueError(
                "the fused gluon prefill under tensor parallelism needs the MAX all-reduce "
                "(`max_reduce`): it combines the ranks' finished ballots with one; the SUM "
                "gather alone cannot serve it."
            )
        if paged_kv is None:
            raise ValueError(
                "the fused gluon prefill reads the paged cache; pass `paged_kv`"
            )
        _resolve_prefill_body(
            declared=True,
            score_block=score_block,
            qhead_block=qhead_block,
            reduce=reduce,
            topk=topk,
            head_shared=head_shared,
            dedup_anchor=dedup_anchor,
            stride=stride,
            rb=rb,
            q_block=q_block,
            head_dim=d,
            sm_major=_current_sm_major(dev),
        )
    # The scorer reads the cache where it lies -- an anchor's position is a closed form of the
    # unit id -- so the routing unit has to be a whole number of pages, as the selected region
    # does: a unit straddling a page cannot be named by page id.
    if not anchor_in_union:
        # Unreachable through a built layer -- the constructor refuses this first. Named here too
        # so prefill answers like the decode gate, which lists every shape it cannot express
        # rather than trusting a check in another class to have run. Ignoring the flag instead
        # would put the anchors in the union while the caller believes it asked for the opposite.
        raise NotImplementedError(
            "kv_anchor_in_attention=False is not served by routed prefill: the anchor pass scores "
            "the units and attends them off one read of the anchor keys, so there is no form "
            "where the anchors rank units without entering the union. Serve this arm with a build "
            "that carries the separate scoring pass, or retrain it with the anchors in attention."
        )
    if not _HAS_DECODE_SCORE:
        raise RuntimeError(
            "routed prefill needs triton to score the units and it is not importable. The router "
            "ranks the anchors through a triton kernel; there is no second scorer to fall back "
            "to. Install triton, or serve this checkpoint on a build that has it."
        )
    if sink_units and not topk:
        # The constructor refuses this pair; repeated here because tests and probes call this
        # body directly, and the topk == 0 branch below builds the own-only selection in closed
        # form -- the sink would be silently dropped, not served.
        raise ValueError(
            f"sink_units={sink_units} with topk=0: no ranking runs at topk 0, so the sink has "
            "no top-k to be forced into."
        )
    if paged_kv is None:
        raise ValueError(
            "routed prefill was called without `paged_kv`, so the selected region has no cache to "
            "read. Pass (key_cache, value_cache, block_table_row, block_size); the caller in "
            "SolarOpen3Attention already does."
        )
    _validate_output_gate(q, output_gate, output)
    # When vLLM's page does not divide the routing unit, the cache is re-addressed at a page that
    # does: a view of the same storage. Everything below reads the page size out of this tuple.
    paged_kv = subpage_cache(paged_kv[0], paged_kv[1], paged_kv[2], rb)
    # An anchor closes the window ending at `(j + 1) * stride - 1`, so within a cache page the
    # anchors sit at offsets `stride-1, 2*stride-1, ...`. That makes the anchor grid a slice of the
    # cache rather than a gather out of it, which needs the page to be a multiple of the stride.
    if paged_kv[3] % stride:
        raise ValueError(
            f"the cache page size {paged_kv[3]} is not a multiple of kv_subsample_stride="
            f"{stride}, so the anchors are not at the same offsets in every page and the grid is "
            f"not a slice of the cache. This is the page the routing reads, which is vLLM's own "
            f"or a sub-page of it -- gcd(page, kv_route_block={rb}) -- so both have to be "
            f"multiples of {stride}. Serve with a --block-size that is."
        )
    _apage = paged_kv[3] // stride  # anchors per page of the view
    _kview = paged_kv[0][:, stride - 1 :: stride]
    _vview = paged_kv[1][:, stride - 1 :: stride]
    _abt = paged_kv[2]  # anchor page `pa` is cache page `pa`
    # (pages, anchors_per_page, hkv*d) -- `flatten(2)` on the trailing head/dim axes, which ARE
    # contiguous, so the sliced path stays a view.
    _ka3 = _kview.flatten(2)

    # ── the fused gluon prefill: one pass computes the anchor half AND the election ──────────
    # The body was fixed at initialisation (`_resolve_prefill_body`); it is re-resolved here
    # for the tests and probes that call this body directly, and nothing below falls back: an
    # exception in the fused body is a fault in that body and propagates (the decode's
    # "Refused, not absorbed" rule). The reference for the fused body is the rest of THIS
    # function -- identical election (fp32 ranking included) and merged output inside the
    # repo's reorder band (rel 2e-2); `test_gluon_prefill_parity_gpu` pins both. A chunk with
    # no closed anchor (A == 0: the first tokens of a request shorter than the stride) is
    # served by the fused body too, own pass only, so no request ever changes bodies.
    if fused_prefill:
        with torch.profiler.record_function("route.gluon"):
            _kc_g, _vc_g, _btr_g, _bs_g = paged_kv
            _out = _gluon_prefill_mod().routed_prefill_gluon(
                q,
                p0,
                apos=apos,
                kc=_kc_g,
                vc=_vc_g,
                btrow=_btr_g,
                page=_bs_g,
                num_heads=hq,
                num_kv_heads=hkv,
                head_dim=d,
                scaling=scaling,
                stride=stride,
                q_block=q_block,
                route_block=rb,
                topk=topk,
                sink_units=sink_units,
                vote_reduce=_vote_reduce,
                varlen_lse=_varlen_lse,
                max_reduce=max_reduce,
                output_gate=output_gate,
                output=output,
            )
        if ("gluon-served",) not in _ROUTE_LOGGED:
            # The observed counterpart of the boot line's `prefill_body=gluon`: a decision
            # became a served chunk. Once per process.
            _ROUTE_LOGGED.add(("gluon-served",))
            logger.info(
                "[solar_open3] the fused gluon prefill served its first chunk "
                "(T=%d, closed anchors=%d)",
                T,
                A,
            )
        return _out

    # ── per-block arithmetic, all host, no readbacks ────────────────────────
    blk = list(range(qb0, qb1 + 1))
    q0 = [b * q_block for b in blk]
    p_last = [min((b + 1) * q_block - 1, pe) for b in blk]
    r0 = [max(x - p0, 0) for x in q0]  # row range of each block
    r1 = [min(x + q_block - p0, T) for x in q0]
    naf = [min(x // stride, A) for x in q0]  # anchors visible to EVERY row
    nav = [min((pl + 1) // stride, A) for pl in p_last]  # anchors visible to ANY row
    nua = [pl // rb + 1 for pl in p_last]
    own_lo = [x // rb for x in q0]
    # topk is the TOTAL budget, own units included -- they are forced in by scoring +inf and then
    # spend slots in the same top-k, exactly as training counts it (`route_anchor_topk`). It is
    # not a free budget on top of the own units: at own=2 and topk=32 the freely chosen units are
    # 30, not 32. At topk == 0 the selection is the own units alone, `nrf` comes out 0, and only
    # the boundary pass reads them.
    nsel = [min(topk, x) if topk else x - ol for x, ol in zip(nua, own_lo)]
    S_i = [ns * rb - (x * rb - (pl + 1)) for ns, x, pl in zip(nsel, nua, p_last)]
    nrf = [si - (pl + 1 - ol * rb) for si, pl, ol in zip(S_i, p_last, own_lo)]
    nrf1 = [max(x, 1) for x in nrf]

    # A prefix sum over host integers stays on the host: a cumsum on a temporary tensor would
    # pay one element read per query block to get back here.
    _cuq_l = [0] + r1

    # ── routing: one matmul for every block ─────────────────────────────────
    # Profiler phases, so a profile can price the scoring matmul, the anchor attend and the
    # selected region separately. Entered and exited by hand because they end at the section
    # boundaries below rather than at a block boundary.
    _rf = torch.profiler.record_function("route.score")
    _rf.__enter__()
    # Query pooling over the rows PRESENT (serving's contract): pad the chunk to the block
    # grid, sum, divide by the true row count.
    pad0 = p0 - qb0 * q_block
    qh = q.view(T, hkv, gqa, d)
    qpad = qh.new_zeros(n_qb * q_block, hkv, gqa, d)
    qpad[pad0 : pad0 + T] = qh
    # Every query block in the plan holds at least one present row, and the pooled path DIVIDES
    # by this: a zero would make the pooled query inf/nan and the top-k would rank on garbage
    # rather than raise. The sub-query path clamps instead, because an empty sub-query BIN is
    # ordinary there (a chunk boundary leaves some) and is masked out afterwards.
    _rows = [b - a for a, b in zip(r0, r1)]
    if min(_rows) <= 0:
        raise ValueError(
            f"query block decomposition produced an empty block: rows per block {_rows} for "
            f"p0={p0}, T={T}, q_block={q_block}. Pooling would divide by zero and rank on nan."
        )
    tkv = total_kv_heads or hkv
    _nsel = 1 if head_shared else hkv
    # ONE transfer for every host-computed array this call hands the GPU, sliced into views
    # afterwards. `torch.tensor(list, device=cuda)` stages through pageable host memory and blocks
    # until its copy retires, so one transfer replaces eleven stalls. The pack is flat rather than
    # a rectangle because the cu_seqlens prefix sum is one element longer than the rest.
    # ⚠️ Order is the contract: the tuple unpack below names them in this order.
    _packs = [
        p_last,
        q0,
        nrf1,
        naf,
        nav,
        own_lo,
        _rows,
        [max(x, 0) for x in nrf],
        nsel,
        [1 if x <= 0 else 0 for x in nrf],
        _cuq_l,
    ]
    # And ONE transfer for the whole STEP, not one per routed layer: every element of `_packs` is
    # a function of (p0, T, A) and the routing config, so the step's routed layers all want the
    # same bytes. `plan_cache` is the step's dict, filled by the first routed layer to arrive.
    # ⚠️ The views are shared, so nothing below may write through them; the dedup branch REBINDS
    # `naf_t`/`nav_t` instead. The host lists are rebuilt per layer, which costs no sync.
    _pkey = (p0, T, A, stride, q_block, rb, topk)
    _views = plan_cache.get(_pkey) if plan_cache is not None else None
    if _views is None:
        _flat, _spans, _n = [], [], 0
        for _lst in _packs:
            _spans.append((_n, _n + len(_lst)))
            _flat.extend(_lst)
            _n += len(_lst)
        _dev_pack = torch.tensor(_flat, device=dev)
        _views = [_dev_pack[a:b] for a, b in _spans]
        if plan_cache is not None:
            plan_cache[_pkey] = _views
    (
        pl_t,
        q0_t,
        nrf_t,
        naf_t,
        nav_t,
        ol_t,
        rows_t,
        nrf0_t,
        nsel_t,
        nrf_le0_t,
        _cuq_t,
    ) = _views
    cnt = rows_t.to(torch.float32)
    cu_q = _cuq_t.to(torch.int32)
    seqused_a = naf_t.clamp(min=1).to(torch.int32)
    group = rb // stride
    # The unit grid spans POSITIONS, not anchors: unit u covers [u*rb, (u+1)*rb). Sizing it from
    # the anchor count would drop the last unit whenever the chunk ends less than one stride into a
    # new route block, and that unit is the block's OWN. The anchor count is still a floor, because
    # every one of the A columns has to land in the grid for the view to be exact.
    n_units = max(max(pl // rb + 1 for pl in p_last), (A + group - 1) // group)
    if _PAD_ROUTE:
        n_units = max(n_units, _PAD_ROUTE // rb)
    if A == 0:
        # A chunk shorter than one stride closes no anchor, so nothing can rank the units -- but
        # it also holds a single unit, its OWN, which the forcing below pins regardless of score.
        # A flat score is therefore exact here and only here, which the guard enforces: an empty
        # pool on a longer chunk would let the top-k fall back to index order with no error.
        if max(p_last) >= stride:
            raise ValueError(
                f"routed prefill got no anchors (A=0) for a chunk reaching position "
                f"{max(p_last)}, which is at or past stride={stride} -- the anchor pool should "
                f"hold {(max(p_last) + 1) // stride}. Nothing can rank the units here."
            )
        n_units = max(max(p_last) // rb + 1, 1)
        us = torch.zeros(n_qb, n_units, device=dev, dtype=torch.float32)
    elif score_block is not None:
        # `kv_route_score_block`: score each run of `score_block` rows on its own instead of
        # pooling the whole block into one query. The kernel collapses anchors to units inside, so
        # only (n_qb, n_sub, n_units) ever exists -- the un-reduced score would be far larger.
        if q_block % score_block:
            # The view below would fail with a bare shape error naming neither knob.
            raise ValueError(
                f"kv_route_score_block={score_block} does not divide q_block={q_block}; the "
                f"sub-query grid and the query block disagree, which would shift every unit "
                f"score. Training refuses the same config."
            )
        if reduce not in ("mean", "amax", "smax"):
            # The dispatch below is if/elif/else, so an unrecognised value would quietly take the
            # mean branch and serve a ranking the checkpoint was not trained with.
            raise ValueError(
                f"kv_route_reduce={reduce!r} is not one of mean|amax|smax; refusing rather than "
                f"falling through to the mean reduce, which ranks units differently."
            )
        n_sub = q_block // score_block
        present = torch.zeros(n_qb * q_block, device=dev, dtype=torch.float32)
        present[pad0 : pad0 + T] = 1.0
        cnt_sub = present.view(n_qb, n_sub, score_block).sum(-1)  # (n_qb, n_sub)
        # Rows and then the GQA group, averaged in the INPUT dtype with only the result upcast.
        # fp32-first flips near-tie rankings against the trained selection.
        _qflat = None
        if qhead_block is None:
            qsub = _pool_query(
                qpad,
                cnt_sub,
                n_qb=n_qb,
                n_sub=n_sub,
                hkv=hkv,
                gqa=gqa,
                d=d,
                score_block=score_block,
            )  # (n_qb,n_sub,hkv,d)
            n_vote = n_sub
        else:
            if score_block == 1:
                # The pool over a length-1 span is the row itself: the sum is a squeeze and the
                # divide is by 1 (an absent row is a zero in qpad and a zero count downstream,
                # both exactly as the general arm leaves them). Skipping it drops the widest
                # copies this branch made -- at the served shape they were most of its cost.
                qsub = qpad.view(n_qb, n_sub, hkv, gqa, d)
            else:
                qsub = (
                    qpad.view(n_qb, n_sub, score_block, hkv, gqa, d).sum(2)
                    / cnt_sub.clamp(min=1).to(qpad.dtype)[..., None, None, None]
                )
            # Each query head of the GQA group votes on its own instead of being averaged into
            # the group's mean, so a minority retrieval head is not drowned out. The head axis is
            # folded INTO the sub-query axis, so every reduce below applies unchanged and `smax`
            # normalises per (row, head) rather than per row.
            #
            # ⚠️ The score tensor grows by `gqa`, and vLLM's memory profile run never sees this
            # allocation -- the routed op returns early when there is no attention metadata.
            if qhead_block != 1:
                raise ValueError(
                    f"kv_route_qhead_block={qhead_block} is not supported; only 1 (every query "
                    f"head votes) or None (the GQA group is averaged) are implemented. A value "
                    f"between the two would need the group split into blocks, which nothing "
                    f"reads yet."
                )
            # fp32 only where the TP sum needs exact accumulation across ranks; without one the
            # scorer casts the query straight back to the key's dtype, and the upcast was a full
            # fp32 copy of the widest tensor this branch builds, once per chunk.
            qv = qsub.permute(0, 1, 3, 2, 4)  # (n_qb, n_sub, gqa, hkv, d)
            if all_reduce is not None and group_size is not None and group_size != gqa:
                qv = qv.float()
            votes = gqa
            if all_reduce is not None and group_size is not None and group_size != gqa:
                # Each query head is its own ballot, so a split group reassembles by
                # CONCATENATION on the vote axis: disjoint slots of a group_size-wide buffer plus
                # a SUM reduce. Summing the ballots themselves would collapse them to `gqa`.
                off = qhead_start % group_size
                qf = qv.new_zeros((n_qb, n_sub, group_size, tkv, d))
                qf[:, :, off : off + gqa, kv_head_start : kv_head_start + hkv] = qv
                qv = all_reduce(qf)[:, :, :, kv_head_start : kv_head_start + hkv]
                votes = group_size
            cnt_sub = cnt_sub.repeat_interleave(votes, dim=1)  # (n_qb, n_sub*votes)
            n_vote = n_sub * votes
            if all_reduce is not None and group_size is not None and group_size != gqa:
                qsub = qv.reshape(n_qb, n_sub * votes, hkv, d).contiguous()
            if all_reduce is None or group_size is None or group_size == gqa:
                # Lay the whole chunk out ONCE in the kernel's own (kv head, rows, dim) order:
                # every per-chunk `permute(...).contiguous()` below then becomes a free slice of
                # this buffer. bf16 because the kernel casts to the key's dtype regardless; the
                # TP paths keep the per-chunk copies, whose fp32 they need.
                _qflat = (
                    qv.permute(3, 0, 1, 2, 4)
                    .reshape(hkv, n_qb * n_vote, d)
                    .to(torch.bfloat16)
                    .contiguous()
                )
        if qhead_block is None and group_size is not None and group_size != gqa:
            # tp > total_kv_heads REPLICATES a KV head and SPLITS its query group, so `gqa` is
            # this rank's share of `group_size` heads and the pool above divided by the wrong
            # count. The group is reassembled BEFORE the kernel, by summing the QUERY: the scorer's
            # amax over a unit's anchors sits inside the dot, so it is not linear in the score.
            qsub = qsub * (gqa / group_size)
            if all_reduce is not None:
                qfull = qsub.new_zeros((n_qb, n_vote, tkv, d))
                qfull[:, :, kv_head_start : kv_head_start + hkv] = qsub
                qsub = all_reduce(qfull)[
                    :, :, kv_head_start : kv_head_start + hkv
                ].contiguous()
        # Query blocks are independent through this whole stage -- the score is per row, the
        # softmax is over the UNIT axis, and every reduction stays inside one query block -- so it
        # tiles over query blocks with no accumulator, each tile writing its own slice of `us`.
        # Untiled, the live set scales with the chunk's block count and the worker OOMs during a
        # long prefill.
        #
        # The empty-bin counts tile with whatever axis the votes live on: with the KV heads folded
        # in, a query block carries hkv*n_vote votes laid out KV-head-major, so `repeat` and not
        # `repeat_interleave` -- the reshape below puts the KV axis OUTSIDE the vote axis.
        #
        # A plain `amax` over the KV heads commutes with a MAX across ranks, so that fold stays
        # rank-local and only the collapsed tile crosses the wire. `smax` and the query-head split
        # need the per-head scores themselves on every rank, so they gather with a SUM instead.
        if not head_shared and (qhead_block is not None or head_reduce == "smax"):
            raise ValueError(
                f"kv_route_head_shared=False cannot be combined with "
                f"kv_route_qhead_block={qhead_block!r} / kv_route_head_reduce={head_reduce!r}: "
                "both collapse the KV heads into one ranking, which is what per-head selection "
                "removes. All three are serving-only knobs the training config does not carry, "
                "so a checkpoint that sets this pair was configured by hand."
            )
        _fold_ok = (
            qhead_block is None and head_reduce != "smax" and max_reduce is not None
        )
        _gather = all_reduce is not None and tkv != hkv and not _fold_ok
        # How many KV heads reach the collapse below: this rank's share, unless the SUM gather
        # widened the axis to the global count first.
        hs = tkv if _gather else hkv
        cnt_vote = cnt_sub if qhead_block is None else cnt_sub.repeat(1, hs)
        # The budget counts PER-KV-HEAD elements, the convention the shared path above set: its
        # buffer is (hkv, m*n_sub, U) and the divisor is n_sub*U. The per-head buffer is
        # (hkv, m*n_vote, U) -- same shape family -- so its divisor is n_vote*U, NOT hs*n_vote*U:
        # counting the KV axis here (as this line once did) quartered every chunk, and the extra
        # chunks re-launched the scorer over the same anchors for nothing. Under the SUM gather
        # the tile's widest buffer is `sqf`, tkv (not hkv) heads wide -- tkv*budget elements,
        # the same 1 GiB cap as tp=1's hkv*budget `sq` -- and the permute-reshape copy below
        # doubles either, so the per-tile peak is the same ~2 GiB at every tp (review 3-1).
        _votes = n_vote
        _GQ = max(1, _SUB_SCORE_BUDGET // max(1, _votes * n_units))
        # `_nsel` rankings per query block: one when every head shares it, one per KV head when
        # `kv_route_head_shared=False`. The axis is carried even at 1, so both cases take one path.
        # fp32 BY NAME, not by inheritance: `qsub.new_empty` was fp32 only because the TP upcast
        # above once ran unconditionally. The top-k below breaks bf16-rounded ties by index
        # differently from fp32 (54 of 96 ranking-dependent blocks at U=256, measured), so the
        # ranking dtype is part of the election; it matches the decode's fp32 `usf`, and
        # `test_election_ranking_dtype` pins both.
        us = torch.empty(
            (n_qb, _nsel, n_units), device=qsub.device, dtype=torch.float32
        )
        for _q0 in range(0, n_qb, _GQ):
            _q1 = min(_q0 + _GQ, n_qb)
            _m = _q1 - _q0
            # The anchors are read where they lie: their positions are a closed form of the unit
            # id, so neither a padded transposed pool nor a per-key mask is built. The kernel folds
            # anchors into units over several units at once, which is what fills the dot.
            sq = _prefill_score(
                # bf16, not the fp32 the pooling left behind: the kernel casts the query to the
                # KEY's dtype anyway, so fp32 here buys no precision and costs twice the bytes. The
                # cast is here rather than at the pool because the TP paths above want fp32.
                _qflat[:, _q0 * n_vote : _q1 * n_vote]
                if qhead_block is not None and _qflat is not None
                else qsub[_q0:_q1]
                .permute(2, 0, 1, 3)
                .reshape(hkv, _m * n_vote, d)
                .to(torch.bfloat16)
                .contiguous(),
                _ka3,
                _abt,
                pl_t[_q0:_q1],
                n_units=n_units,
                apg=group,
                stride=stride,
                rb=rb,
                d=d,
                scale=scaling,
                n_sub_per_block=n_vote,
                n_anchor=A,
                per_anchor=True,
            ).view(hkv, _m, n_vote, n_units)
            if _gather:
                # `smax` and the query-head split cannot be rebuilt from a SUM of per-rank
                # results, so the head axis is gathered HERE, between the kernel and the collapse.
                # `writes_score=False` leaves zeros: ranks replicating a KV head compute the same
                # score, so exactly one contributes and the SUM is a gather.
                sqf = sq.new_zeros((tkv, _m, n_vote, n_units))
                if writes_score:
                    sqf[kv_head_start : kv_head_start + hkv] = sq
                sq = all_reduce(sqf)
            if not head_shared:
                # `kv_route_head_shared=False`: no collapse. Every KV head keeps its OWN ranking
                # and the union is built per head below, so the head axis moves next to the query
                # blocks and the sub-query reduce runs inside it. Under TP the gathered score spans
                # every head, so this rank takes back the slice it holds.
                sq = sq if not _gather else sq[kv_head_start : kv_head_start + hkv]
                sq = sq.permute(1, 0, 2, 3)  # (_m, hkv, n_vote, U)
            elif qhead_block is None:
                # `amax` maxes RAW logits, so a KV head whose scores merely run large wins every
                # unit; `smax` normalizes each head over the unit axis first so the heads are
                # comparable. Same reduction the sub-query axis gets below, one axis over.
                if head_reduce == "smax":
                    _ok = sq.isfinite().any(-1, keepdim=True)
                    sq = torch.where(
                        _ok, torch.softmax(sq, dim=-1), torch.zeros_like(sq)
                    ).amax(0)
                else:
                    # A separate pass over the score the kernel just wrote. ⚠️ Folding it into
                    # the scorer instead costs more end to end: the kernel loses the grid's head
                    # axis and its keys' residency. Rank a change here against the served path,
                    # not against a standalone bench of the two kernels.
                    sq = sq.amax(0)  # heads: amax
                    if _fold_ok and tkv != hkv:
                        sq = max_reduce(
                            sq.contiguous()
                        )  # ...and across the other ranks'
            else:
                # ⚠️ The KV-head collapse sits AFTER the unit-axis softmax here, which is the
                # point of splitting the query heads: collapsing first maxes raw logits, so a KV
                # head whose scores merely run large wins every unit. The KV axis is folded into
                # the vote axis instead, and the `amax` over votes then picks the most confident
                # head rather than the loudest.
                sq = sq.permute(1, 0, 2, 3).reshape(_m, hs * n_vote, n_units)
            # ⚠️ An empty sub-query bin is a whole row of -inf and is the COMMON case, since a
            # chunk boundary leaves some bins with no rows. Every reduce below ignores those rows:
            # one -inf let through collapses the unit axis and the top-k degenerates to index
            # order, silently, because -inf is a valid float.
            # One call, whatever the reduce and whichever axes are still standing: the vote axis
            # is always second-to-last and everything left of it folds into the row index, so the
            # per-head selection and the shared one take the same path and cannot drift.
            sq = sq.reshape(-1, sq.shape[-2], n_units).contiguous()
            us[_q0:_q1] = _vote_reduce(
                sq,
                cnt_vote[_q0:_q1].contiguous(),
                reduce=reduce,
            ).view(_m, _nsel, n_units)
            sq = None  # drop the tile before the next allocates
    else:
        # One sub-query per block, with the same fixed normalization/head mean as the
        # sub-query path. Retain the per-head input-dtype rounding before averaging heads.
        qp = _pool_query(
            qpad,
            cnt[:, None],
            n_qb=n_qb,
            n_sub=1,
            hkv=hkv,
            gqa=gqa,
            d=d,
            score_block=q_block,
        )[:, 0]
        if all_reduce is not None and group_size is not None and group_size != gqa:
            # tp > total_kv_heads REPLICATES a KV head across ranks and SPLITS its query group, so
            # this rank's mean is over the wrong denominator. Reassemble the full-group query HERE,
            # ahead of the score, the way the sub-query branch and the decode route do: the kernel
            # folds a unit's anchors with a MAX, and a max of sums is not a sum of maxima, so ranks
            # that scored their own share and added the results would rank units differently from
            # one rank holding the whole group.
            _qz = qp.new_zeros(n_qb, tkv, d)
            _qz[:, kv_head_start : kv_head_start + hkv] = qp * (gqa / group_size)
            qp = all_reduce(_qz)[:, kv_head_start : kv_head_start + hkv].contiguous()
        # The same kernel the sub-query branch uses, at ONE vote per query block -- which is what
        # pooling is -- reading the anchors where they lie.
        us = _prefill_score(
            # bf16 for the same reason the sub-query branch gives: the kernel casts to the key's
            # dtype anyway.
            qp.transpose(0, 1).to(torch.bfloat16).contiguous(),  # (hkv, n_qb, D)
            _ka3,
            _abt,
            pl_t,
            n_units=n_units,
            apg=group,
            stride=stride,
            rb=rb,
            d=d,
            scale=scaling,
            n_sub_per_block=1,
            n_anchor=A,
            # Per anchor, as training gates it -- see the note at the unit mask below.
            per_anchor=True,
        ).transpose(0, 1)  # (n_qb, hkv, U)
        # An anchor past this block's last query must not score it. The kernel folds anchors into
        # units, so the cut lands on the UNIT grid rather than on the anchor axis.
        _u = torch.arange(n_units, device=dev)
        us = us.masked_fill((_u * rb > pl_t[:, None, None]), float("-inf"))
        if all_reduce is not None and tkv != hkv:
            # Heads are sharded: gather the axis before the collapse, as the sub-query branch does.
            # `writes_score=False` leaves zeros: the query above is whole on every rank, so ranks
            # replicating a KV head hold the SAME score and exactly one of them contributes it,
            # which keeps this SUM a gather instead of a multiply.
            _f = us.new_zeros(n_qb, tkv, n_units)
            if writes_score:
                _f[:, kv_head_start : kv_head_start + hkv] = us
            us = all_reduce(_f)
        if head_shared:
            us = us.amax(1, keepdim=True)  # (n_qb, 1, U) -- see `_nsel`
        else:
            # `kv_route_head_shared=False`: no collapse, every KV head keeps its own ranking.
            # Under TP the score above spans every head, so take back this rank's slice.
            us = (
                us if us.shape[1] == hkv else us[:, kv_head_start : kv_head_start + hkv]
            )
    unit = torch.arange(n_units, device=dev)
    _pl, _q0b = pl_t[:, None, None], q0_t[:, None, None]
    _live = unit * rb <= _pl  # the block has this unit behind it
    us = us.masked_fill(~_live, float("-inf"))
    own = _live & ((unit + 1) * rb > _q0b)
    if sink_units:
        # Sink forcing, same mechanism as own: +inf, then ranked in the same budget (the init
        # guard has already sized topk for both). Gated by the block's live length so the
        # selection stays well-formed: a query block early in the chunk may not have
        # `sink_units` units behind it yet, and an ungated +inf would rank a FUTURE unit for it
        # -- inert at the output (the selected pass is causal) and displacing nothing (such a
        # block has fewer candidates than the budget), but a unit the trainer's `admissible`
        # never selects.
        own = own | (unit < sink_units) & _live
    us = us.masked_fill(own, float("inf"))
    if topk:
        k_eff = min(topk, n_units)
        vals, idx = us.topk(k_eff, -1)
        # Blocks with fewer admissible units than k: the -inf slots carry garbage indices. Send
        # them past every real unit so the sorted grid keeps them behind the own span, where the
        # per-block S_i cut (valid_raw) can never reach them.
        idx = idx.masked_fill(vals == float("-inf"), n_units)
    else:
        # topk == 0: the selection is exactly the forced own units, built in closed form rather
        # than ranked. ⚠️ Ranking here would be WRONG, not just wasteful: the top-k returns one
        # width for every row, so a row needing fewer own units fills the surplus with real FOREIGN
        # units, which sort ahead of the own ones and displace the row's own position.
        k_eff = max(nsel)
        _ol_t, _ns_t = ol_t[:, None, None], nsel_t[:, None, None]
        _own_slot = torch.arange(k_eff, device=dev)
        idx = torch.where(_own_slot < _ns_t, _ol_t + _own_slot, n_units).expand(
            n_qb, _nsel, k_eff
        )

    _rf.__exit__(None, None, None)
    # ── selected raw tokens ─────────────────────────────────────────────────
    _rf = torch.profiler.record_function("route.assemble")
    _rf.__enter__()
    # A route block is a whole number of pages -- the cache was re-addressed above if vLLM's page
    # did not divide it -- so the selection is expressible as a page table and the union never has
    # to be gathered into a dense buffer.
    #
    # The valid tokens are the FIRST `nrf1[i]` of the sorted-unit grid, which is what `seqused_k`
    # expresses. The pages of the -inf sentinel units still have to hold a readable index, so they
    # clamp to the last real page rather than running off the block table.
    if idx.shape[1] != _nsel:
        # A mismatched ranking axis would still RESHAPE below, since the page table is built by
        # element count, and the heads would read each other's pages.
        raise AssertionError(
            f"the router produced {idx.shape[1]} rankings for {_nsel} expected "
            f"(kv_route_head_shared={head_shared}, num_kv_heads={hkv}). One of the score branches "
            "collapsed the KV heads when it should not have, or the reverse."
        )
    _kc2, _vc2, _btrow2, _bs2 = paged_kv
    # Sorting the selection and turning it into pages is ONE kernel; `_selected_page_table_torch`
    # is the same expression in torch, held to the same bytes by the parity test.
    _units, _btab_sel = (
        selected_page_table(  # (n_qb, _nsel, k_eff), (.., k_eff * rb // _bs2)
            idx, _btrow2, rb // _bs2, max(pe, 0) // _bs2
        )
    )
    _sel_paged = (_btab_sel, nrf_t.to(torch.int32), _kc2, _vc2)

    _rf.__exit__(None, None, None)
    # ── three attention pieces ──────────────────────────────────────────────
    # `dedup_anchor` is expressed in the PAGE TABLE, not in the data: an anchor page holds exactly
    # one routing unit's anchors, so "drop this unit's anchors" is "leave its page out of the
    # table". The anchors stay read in place, which is what lets every query block of the chunk
    # carry its own selection off one shared view.
    #
    # That view is the one the scorer ranks, handed to flash as a paged tensor of `_apage`-anchor
    # pages, addressed by the request's own block table because anchor page `pa` is cache page
    # `pa`. Nothing is gathered, padded or copied.
    n_pages = (A + _apage - 1) // _apage
    # `n_pages` is 0 for a request that has not reached its first anchor. The branch is skipped
    # rather than guarded per line: with no anchors there is nothing to deduplicate.
    if dedup_anchor and n_pages:
        # Keep a unit's page iff the unit is BELOW the query block's own units and the router did
        # not select it. The `own_lo` bound is what makes every kept page whole and visible to
        # every row: a unit straddling the block is one of its own, always selected, and read at
        # full resolution from the selected region instead. A unit spans `_upp` pages, so keeping
        # or dropping one is keeping or dropping its run of pages.
        _upp = max((route_block // stride) // _apage, 1)
        _nu = (n_pages + _upp - 1) // _upp
        # `_nsel` rankings: one when the heads share it, one per KV head when they do not. Kept
        # units come first and stay in ascending order, and each becomes its run of pages. One
        # kernel: the mask, the stable compaction and the page expansion fuse.
        btab, _kept = dedup_page_table(ol_t, _units, _abt, _nu, n_pages, _upp)
        _nka = _kept * (_upp * _apage)  # (n_qb, _nsel) kept per block
        seqused_a = _nka.clamp(min=1).to(torch.int32)
        # ⚠️ `_nka` is NOT read back: a host branch on it would stall the CPU until every queued
        # kernel retires. The kernel's `max_k` takes the static upper bound below instead of the
        # realized max, and "is this block's anchor segment empty" stays a GPU mask (`naf_t == 0`).
        # The collapse to one width per block is exact because every head keeps the same unit count.
        naf_t = _nka.amin(1).to(naf_t.dtype)
        # The gap the boundary pass exists for -- anchors inside the block's own span -- is empty
        # under dedup, because those anchors belong to the block's own units and their pages were
        # just dropped. naf == nav makes `avalid` all-False below.
        nav_t = naf_t
    else:
        btab = (
            _abt[: max(n_pages, 1)]
            .to(torch.int32)[None, None]
            .expand(n_qb, _nsel, -1)
            .contiguous()
        )
        seqused_a = seqused_a[:, None].expand(n_qb, _nsel).contiguous()
    # An upper bound on the anchor keys any block reads, known without looking at the device:
    # dedup only ever DROPS pages, so the padded pool bounds it. `seqused_a` still cuts each block
    # to its real width, so a loose bound costs a wider tile plan, not wrong keys.
    _max_k_a = max(n_pages * _apage, 1)
    # ⚠️ With the overlap on, this marker spans the LAUNCH, not the kernel: the work runs beside
    # `route.raw` and a phase split will show both windows covering the same wall clock. Set
    # SOLAR_OPEN3_ROUTE_OVERLAP=0 to get attributable spans back.
    _ov = _ROUTE_OVERLAP and dev.type == "cuda"
    _cur = torch.cuda.current_stream(dev) if _ov else None
    _sa = _anchor_stream(dev) if _ov else None
    if _ov:
        _sa.wait_stream(_cur)  # the pool and its page table were built on `_cur`
    with (
        torch.profiler.record_function("route.anchor"),
        torch.cuda.stream(_sa) if _ov else contextlib.nullcontext(),
    ):
        # ⚠️ `.contiguous()` on both slices: flash asserts `seqused_k.stride(0) == 1` and bakes
        # the page table's leading stride into its compile key, and a column of a
        # (query blocks, rankings, ...) tensor has neither unless there is exactly one ranking.
        if head_shared:
            o_a, lse_a = _varlen_lse(
                q.view(T, hq, d),
                _kview,
                _vview,
                cu_q=cu_q,
                max_q=q_block,
                cu_k=None,
                max_k=_max_k_a,
                seqused=seqused_a[:, 0].contiguous(),
                btab=btab[:, 0].contiguous(),
                scale=scaling,
            )
        else:
            # One call per KV head: its own page table, its own slice of the view, and the query
            # heads of its GQA group. Same arithmetic as the shared call; what grows is the number
            # of launches.
            _qv = q.view(T, hq, d)
            _oa, _la = [], []
            for _h in range(_nsel):
                _o, _l = _varlen_lse(
                    _qv[:, _h * gqa : (_h + 1) * gqa],
                    _kview[:, :, _h : _h + 1],
                    _vview[:, :, _h : _h + 1],
                    cu_q=cu_q,
                    max_q=q_block,
                    cu_k=None,
                    max_k=_max_k_a,
                    seqused=seqused_a[:, _h].contiguous(),
                    btab=btab[:, _h].contiguous(),
                    scale=scaling,
                )
                _oa.append(_o)
                _la.append(_l)
            o_a = torch.cat(_oa, 1)  # (T, hq, d)
            lse_a = torch.cat(_la, 0)  # (hq, T), as the kernel returns
    # ONE causal call over `[selected ; own]`. Both regions are pages of the SAME vLLM cache, so
    # one page table addresses them, and flash's bottom-right rule is exactly the mask they need:
    # with `k_len - q_len == nrf + max(p0 - q0, 0)`, an own key at local index `j` is admitted iff
    # `q0 + j - nrf <= max(p0, q0) + i`, while every selected key is admitted for every row.
    #
    # The two regions concatenate only because the FIRST one's width is a whole number of pages:
    # `nrf` is a multiple of `route_block`, which the page size divides.
    #
    # ⚠️ Requires dedup. Without it the boundary pass still owes the anchors inside the block's own
    # span, which belong to neither region.
    # ⚠️ Requires head_shared. The fused form reads ONE page table, `_sel_paged[0][:, 0]`, and
    # that axis is the ranking axis: `_nsel` is 1 when the heads share a selection and `hkv` when
    # they do not. Per-head, index 0 is KV head 0's selection applied to every head -- while the
    # anchor pass just above stays per-head, so the two halves of the same union would read
    # different units. The three-piece path below indexes the axis properly.
    # `2 * rb >= q_block` is a DISPATCH choice, not a correctness condition: the boundary pass
    # below now sizes its own window from `q_block`, so a wider block is served either way.
    _fused_ro = (
        _FUSE_RO
        and dedup_anchor
        and head_shared
        and A
        and q_block % rb == 0
        and 2 * rb >= q_block
        and q_block % _bs2 == 0
    )
    with torch.profiler.record_function("route.raw"):
        if _fused_ro:
            _kc2, _vc2, _btrow2, _bs2 = paged_kv
            _btab_r = _sel_paged[0][:, 0]
            # The TRUE selected width, not `nrf1`: `nrf1` borrows a key to keep a zero-length
            # varlen segment out of the kernel, and this segment can never be empty.
            # The own pages start at a different column per block, so the two page lists are
            # interleaved by position rather than concatenated -- page ids only, so nothing is
            # gathered out of the cache.
            _npg = q_block // _bs2
            _wp = max(max(x, 0) for x in nrf) // _bs2 + _npg
            _btab_ro, _seq_ro = raw_page_table(
                _btab_r,
                _btrow2,
                q0_t,
                nrf0_t,
                pl_t,
                _bs2,
                _npg,
                _wp,
                max(pe, 0) // _bs2,
            )
            o_r, lse_r = _varlen_lse(
                q.view(T, hq, d),
                _kc2,
                _vc2,
                cu_q=cu_q,
                max_q=q_block,
                cu_k=None,
                max_k=_wp * _bs2,
                seqused=_seq_ro,
                btab=_btab_ro,
                scale=scaling,
                causal=True,
            )
        else:
            _btab_r, _seq_r, _kc2, _vc2 = _sel_paged
            if head_shared:
                o_r, lse_r = _varlen_lse(
                    q.view(T, hq, d),
                    _kc2,
                    _vc2,
                    cu_q=cu_q,
                    max_q=q_block,
                    cu_k=None,
                    max_k=max(nrf1),
                    seqused=_seq_r,
                    btab=_btab_r[:, 0],
                    scale=scaling,
                )
            else:
                # One call per KV head: its own page table, its own slice of the cache, and the
                # query heads of its GQA group.
                #
                # The alternative is folding the head into the varlen SEGMENT axis, which would
                # need q laid out as (total_q * hkv, gqa, d): a real copy of the whole query at
                # this size. The slices below are views, and flash reads a strided KV-head slice
                # of a paged cache bit-identically (measured).
                _qv = q.view(T, hq, d)
                _os, _ls = [], []
                for _h in range(_nsel):
                    _o, _l = _varlen_lse(
                        _qv[:, _h * gqa : (_h + 1) * gqa],
                        _kc2[:, :, _h : _h + 1],
                        _vc2[:, :, _h : _h + 1],
                        cu_q=cu_q,
                        max_q=q_block,
                        cu_k=None,
                        max_k=max(nrf1),
                        seqused=_seq_r,
                        btab=_btab_r[:, _h],
                        scale=scaling,
                    )
                    _os.append(_o)
                    _ls.append(_l)
                o_r = torch.cat(_os, 1)  # (T, hq, d)
                lse_r = torch.cat(_ls, 0)  # (hq, T), as the kernel returns

    # boundary: <= (q_block // stride) gap anchors + the own raw span, grouped dense
    gapw = q_block // stride
    gi = torch.arange(gapw, device=dev)
    # Not allocated on the fused path, which leaves this buffer nothing to hold.
    o_b = lse_b = None
    if not _fused_ro:
        o_b = q.new_zeros(n_qb * q_block, hq, d, dtype=torch.float32)
        lse_b = q.new_full((n_qb * q_block, hq), float("-inf"), dtype=torch.float32)
    # ── boundary own half: flash, not a score tensor ──────────────────────
    _rf = torch.profiler.record_function("route.boundary")
    _rf.__enter__()
    # The own span is a SQUARE CAUSAL attention, so flash serves it instead of the eager score
    # tensor below -- and it returns the lse for free. `q_block % route_block == 0` puts the own
    # span's start exactly at the query block's, and with the queries in the real-row varlen
    # layout `k_len - q_len == max(p0 - q0, 0)`, so flash's bottom-right causal rule reduces to
    # causality on POSITIONS. That holds for a chunk starting mid-block and for a partial trailing
    # block, not only for whole ones.
    _own_fa = None
    if (
        not _fused_ro
        and paged_kv is not None
        and A
        and _VARLEN is not None
        and q_block % rb == 0
        and 2 * rb >= q_block
    ):
        _kc, _vc, _btrow, _bs = paged_kv
        if q_block % _bs == 0 and _bs > 0:
            _npg = q_block // _bs
            # Clamp to the last page the request actually has: a trailing query block asks for
            # pages past `pe`, whose block-table slots are unallocated. `seqused_k` keeps those
            # keys out of the attention, but the page ENTRY is still handed to the kernel.
            _lpg = ((q0_t[:, None] // _bs) + torch.arange(_npg, device=dev)).clamp(
                min=0, max=max(pe, 0) // _bs
            )
            _btab_own = _btrow[_lpg].to(torch.int32).contiguous()  # (n_qb, npg)
            # k_len per block: the causal extent, which is also what keeps the read inside the
            # pages that exist -- p_last never leaves [q0, q0 + q_block).
            _seq_own = (pl_t - q0_t + 1).to(torch.int32)
            _o_own, _l_own = _varlen_lse(
                q.view(T, hq, d),
                _kc,
                _vc,
                cu_q=cu_q,
                max_q=q_block,
                cu_k=None,
                max_k=q_block,
                seqused=_seq_own,
                btab=_btab_own,
                scale=scaling,
                causal=True,
            )
            _own_fa = (_o_own, _l_own)
    # dedup_anchor empties the gap-anchor half and `_own_fa` takes the own half, so the grouped
    # fp32 pass has no columns left. Skipping it leaves `lse_b` at -inf, which the merge already
    # reads as "this piece contributed nothing".
    _skip_boundary = _fused_ro or (dedup_anchor and _own_fa is not None)
    _shape = (
        "prefill",
        bool(dedup_anchor),
        _apage,
        n_pages,
        _own_fa is not None,
        _skip_boundary,
    )
    if _shape not in _ROUTE_LOGGED:
        _ROUTE_LOGGED.add(_shape)
        logger.info(
            "[solar_open3] routed prefill: dedup=%s anchor_page=%d (%d pages) "
            "own_half=%s boundary_pass=%s",
            dedup_anchor,
            _apage,
            n_pages,
            "paged flash" if _own_fa is not None else "eager",
            "SKIPPED (dedup emptied the gap anchors)"
            if _skip_boundary
            else (
                "gap anchors only" if _own_fa is not None else "gap anchors + own raw"
            ),
        )
    # The boundary pass reads the own span from `ol * rb`, so it has to reach `p_last`. The widest
    # that gets is `q_block + max_b(q0 % rb)`, and `q0` walks multiples of `q_block`, so the
    # residue tops out at `rb - gcd(q_block, rb)`. A fixed `2 * rb` covers that only while a query
    # block spans at most two route blocks: at `kv_route_q_block=1536` with `kv_route_block=512`
    # the block's last 512 positions are read by nobody -- dedup dropped their anchors, and the
    # selected region excludes the own span.
    #
    # Exact rather than `q_block + rb`, which over-shoots: at (768, 512) and (384, 256) the fixed
    # `2 * rb` was already the right width, and rounding up there would widen the reduction on a
    # shape that was never wrong. `max` keeps every such shape, the shipped pairs included, on the
    # width they have today.
    _own_w = max(2 * rb, q_block + rb - math.gcd(q_block, rb))
    _kw = gapw if _own_fa is not None else ((gapw if A else 0) + _own_w)
    # Query blocks per grouped pass, sized by the SCORE TENSOR rather than fixed: `sc` below is
    # (nb, hkv, gqa, q_block, kw) fp32 and three of them are live at once, so a constant block
    # count would let the footprint scale with the chunk. ⚠️ It would also OOM with no warning,
    # because vLLM's memory profile run never sees this allocation -- the routed op returns early
    # when there is no attention metadata. `_SUB_SCORE_BUDGET` is the element budget.
    G = max(1, _SUB_SCORE_BUDGET // max(1, hkv * gqa * q_block * _kw))
    prow_grid = (
        q0_t[:, None] + torch.arange(q_block, device=dev)[None]
    )  # (n_qb, q_block)
    for g0 in () if _skip_boundary else range(0, n_qb, G):
        g1 = min(g0 + G, n_qb)
        nb = g1 - g0
        aidx = (naf_t[g0:g1, None] + gi[None]).clamp(max=max(A - 1, 0))
        avalid = naf_t[g0:g1, None] + gi[None] < nav_t[g0:g1, None]
        if _own_fa is not None:
            # Flash took the own half, so this pass carries the gap anchors alone.
            _kg, _vg = kf_at(apos[aidx])
            kb, vb, bpos, bval = _kg, _vg, apos[aidx], avalid
        elif A:
            rpos = ol_t[g0:g1, None] * rb + torch.arange(_own_w, device=dev)[None]
            rvalid = rpos <= pl_t[g0:g1, None]
            k_own, v_own = kf_at(rpos.clamp(max=max(pe, 0)))
            _kg, _vg = kf_at(apos[aidx])
            kb = torch.cat([_kg, k_own], 1)
            vb = torch.cat([_vg, v_own], 1)
            bpos = torch.cat([apos[aidx], rpos], 1)
            bval = torch.cat([avalid, rvalid], 1)
        else:
            rpos = ol_t[g0:g1, None] * rb + torch.arange(_own_w, device=dev)[None]
            rvalid = rpos <= pl_t[g0:g1, None]
            k_own, v_own = kf_at(rpos.clamp(max=max(pe, 0)))
            # No anchor exists to gather: `apos` has no row for `aidx` to clamp onto. `avalid` is
            # all-False here anyway, so the own span alone is the whole pass.
            kb, vb, bpos, bval = k_own, v_own, rpos, rvalid
        qg = qpad.view(n_qb, q_block, hkv, gqa, d)[g0:g1]
        # Scale the KEYS, not the scores: same product, and `kb` is far smaller than `sc`, so the
        # scalar folds into a cast that already runs instead of rewriting the score tensor.
        sc = torch.einsum("nqhgd,nkhd->nhgqk", qg.float(), kb.float() * scaling)
        vis = bval[:, None, :] & (bpos[:, None, :] <= prow_grid[g0:g1, :, None])
        sc.masked_fill_(~vis[:, None, None], float("-inf"))
        o_b[g0 * q_block : g1 * q_block] = torch.einsum(
            "nhgqk,nkhd->nqhgd", torch.softmax(sc, -1).nan_to_num_(), vb.float()
        ).reshape(nb * q_block, hq, d)
        lse_b[g0 * q_block : g1 * q_block] = (
            torch.logsumexp(sc, -1).permute(0, 3, 1, 2).reshape(nb * q_block, hq)
        )
    if not _fused_ro:
        o_b = o_b[pad0 : pad0 + T]
        lse_b = lse_b[pad0 : pad0 + T]
    if _own_fa is not None:
        # Put the two halves back into the single softmax the eager pass computed in one go.
        _o_own, _l_own = _own_fa
        o_b, lse_b = _merge_own_half(o_b, lse_b, _o_own, _l_own)
    _rf.__exit__(None, None, None)
    # ── LSE merge (host-known empty segments forced out) ────────────────────
    if _ov:
        _cur.wait_stream(_sa)
        # Without this the caching allocator may hand these blocks out while the side stream is
        # still writing them.
        o_a.record_stream(_cur)
        lse_a.record_stream(_cur)
    l_a, l_r = lse_a.T.clone(), lse_r.T.clone()  # (T, hq)
    # An empty segment's lse has to be forced out or the merge gives it weight. The test is a
    # masked write that runs unconditionally on the device, because `bool(tensor.any())` would
    # stall the pipeline for a cheaper branch.
    #
    # `<= 0`, not `== 0`, on the selected side: `nrf1` raises a NEGATIVE count to 1, so that
    # segment holds a key the block must not read and its lse would survive the merge.
    l_a[(naf_t == 0).repeat_interleave(rows_t)] = float("-inf")
    # Only the un-fused path can hand the selected segment a key the block must not read: the
    # fused region always holds the own span, so it is never empty and never borrows.
    if not _fused_ro:
        # `nrf` is host arithmetic -- no device read to remove here. The mask rides the same pack
        # so it costs no extra transfer.
        l_r[(nrf_le0_t == 1).repeat_interleave(rows_t)] = float("-inf")
    _rf = torch.profiler.record_function("route.merge")
    _rf.__enter__()
    if _fused_ro:
        mm = torch.maximum(l_a, l_r)
        wa, wr = (l_a - mm).exp(), (l_r - mm).exp()
        o = _merge_two(o_a, o_r, wa, wr, q.dtype)
    else:
        mm = torch.maximum(torch.maximum(l_a, l_r), lse_b)
        wa, wr, wb = (l_a - mm).exp(), (l_r - mm).exp(), (lse_b - mm).exp()
        o = _merge_three(o_a, o_r, o_b, wa, wr, wb, q.dtype)
    _rf.__exit__(None, None, None)
    return _finish_output(o.reshape(T, hq * d), output_gate, output)


def _decode_segments(qsl: list, n_seg: int, n: int, max_rows: int = 1) -> list:
    """Which segments of an engine step are decode rows, given its ``query_start_loc``.

    Being a decode row is a property of the SEGMENT, not of the step: a step that also prefills a
    chunk still holds decode rows, and under chunked prefill most generated tokens are in one.

    A decode segment carries 1 .. ``max_rows`` rows: one on plain decode, the last token plus its
    drafts on a speculative verify step (``max_rows = 1 + num_speculative_tokens``, the same
    threshold vLLM's own backends split decodes from prefills by). Each row is then routed alone
    at its own position, exactly as a non-speculative step would have routed it.

    ``qsl[i + 1] <= n`` drops cudagraph padding: a row at or past ``num_actual_tokens`` is not a
    request, and writing it would corrupt the output.
    """
    if n_seg <= 0:
        return []
    return [
        i
        for i in range(n_seg)
        if 1 <= qsl[i + 1] - qsl[i] <= max_rows and qsl[i + 1] <= n
    ]


def require_supported_attention(m) -> None:
    """Refuse a backend whose metadata lacks the three fields the routed bodies read.

    They need `query_start_loc` / `block_table` / `slot_mapping` to gather anchor keys and name a
    page. FlashAttention builds all three; FlashInfer keeps the same information in a CSR layout
    and `install.py` patches the three back on, so an unpatched venv is caught here.

    Checked on the METADATA rather than on `--attention-backend`, whose default is "auto".
    """
    missing = [
        f
        for f in ("query_start_loc", "block_table", "slot_mapping")
        if not hasattr(m, f)
    ]
    if missing:
        raise NotImplementedError(
            f"{type(m).__name__} does not expose {'/'.join(missing)}, which subsampled and routed "
            f"attention need to gather anchor keys and name a page. Serve with "
            f"--attention-backend FLASH_ATTN, which is the default this model is measured on; "
            f"FLASHINFER also works, but only on a venv carrying the install.py patches that "
            f"re-attach those fields."
        )


def subpage_view(kv_strides, page: int, route_block: int):
    """Re-address a KV cache at a finer page so a routing unit is a whole number of pages.

    vLLM does not let this model pick its page width. It is a hybrid -- GQA attention beside KDA
    linear attention -- and the allocator holds one page size, so the attention page is raised
    until it reaches the KDA state page. The result does not divide the trained routing unit, and
    no width satisfies both: the routing needs one dividing `route_block`, the allocator needs one
    at least as large as the KDA state.

    So the page is split instead. A sub-page of `g = gcd(page, route_block)` tokens divides both,
    and the kernels address the cache as `pg * s_blk + (pos % PAGE) * s_slot`, which takes it
    directly: give them `PAGE = g`, `s_blk = g * s_slot`, and a block table of sub-page ids.

        sub id = block * f + pos // g,  f = s_blk // (g * s_slot)

    `f` is the sub-page STRIDE of a block, not the count in it: a block holds `page // g` of them
    and the rest of the span is the value half, which `s_half` reaches separately. Only ids below
    that count are ever put in a table, so nothing addresses the gap -- which is why the SUB-PAGE
    ID SPACE cannot be an `as_strided` view: its bounds check covers indices no table will hold.
    (`subpage_cache` below does use `as_strided`, on the cache TENSORS, which carry no such gap.)

    Returns `(g, f, s_blk_eff, per_block)`.
    """
    s_blk, _s_half, s_slot = kv_strides[0], kv_strides[1], kv_strides[2]
    g = math.gcd(page, route_block)
    if page % g or route_block % g:
        raise ValueError(
            f"sub-page {g} divides neither the KV cache page ({page}) nor kv_route_block "
            f"({route_block}); gcd should make it divide both, so the strides handed in do not "
            "describe the layout this was derived from."
        )
    if s_blk % (g * s_slot):
        raise ValueError(
            f"the KV cache block stride ({s_blk}) is not a multiple of a {g}-token sub-page "
            f"({g} x {s_slot}), so a sub-page id cannot address it with one stride. The routed "
            f"bodies name a selected unit by page id and kv_route_block={route_block} is not a "
            f"multiple of the page ({page}), so the page has to be split; this layout does not "
            "allow it. Serve on a backend whose page divides kv_route_block."
        )
    return g, s_blk // (g * s_slot), g * s_slot, page // g


def subpage_block_table(bt: torch.Tensor, f: int, per_block: int) -> torch.Tensor:
    """`(..., pages)` block table -> `(..., pages * per_block)` of sub-page ids.

    Entry `i` of the row becomes `per_block` consecutive sub-pages of the block it named, in
    position order, so the row still reads as "the pages of this sequence, in the order the
    sequence runs". The leading axes are whatever the caller has: decode holds a row per request,
    prefill one row for the request it is chunking.
    """
    j = torch.arange(per_block, device=bt.device, dtype=bt.dtype)
    return (bt.to(torch.int64)[..., None] * f + j).flatten(-2).to(bt.dtype)


def subpage_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    bt: torch.Tensor,
    route_block: int,
):
    """A paged KV cache and its block table, re-addressed at a page that divides `route_block`.

    Splitting the page into `gcd(page, route_block)` tokens moves nothing: the returned caches are
    `as_strided` views of the same storage and the returned table names the same tokens in the same
    order, so every consumer keeps reading the cache in place.

    A no-op returning the arguments unchanged when the page already divides the unit. The table is
    rebuilt per call, which only the split path pays.

    Returns `(key_cache, value_cache, block_table, page)`.
    """
    page = key_cache.shape[1]
    if route_block % page == 0:
        return key_cache, value_cache, bt, page
    s_blk, s_slot = key_cache.stride(0), key_cache.stride(1)
    g, f, s_blk_eff, per_block = subpage_view((s_blk, 0, s_slot), page, route_block)
    # The last block contributes the `per_block` sub-pages a table can name, not the `f` its
    # stride spans: the rest of that span is the value half, and declaring it would put the value
    # view's tail past the end of the storage.
    n_sub = (key_cache.shape[0] - 1) * f + per_block
    size = (n_sub, g) + tuple(key_cache.shape[2:])
    stride = (s_blk_eff, s_slot) + tuple(key_cache.stride()[2:])
    return (
        key_cache.as_strided(size, stride),
        value_cache.as_strided(size, stride),
        subpage_block_table(bt, f, per_block),
        g,
    )


def routed_decode_batched(
    q2: torch.Tensor,  # (N, num_heads, head_dim) decode rows, each routed alone; a
    # speculative verify request contributes one per token
    p_last: list | None,  # per-row position (host ints); None on the capture path,
    # where nothing here reads a position on the host
    bt_rows: torch.Tensor,  # (N, max_blocks) the block-table row of each row's request
    key_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, head_dim)
    value_cache: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    scaling: float,
    stride: int,
    q_block: int,
    route_block: int,
    topk: int,
    anchor_in_union: bool = True,  # `kv_anchor_in_attention`; False is refused at build
    dedup_anchor: bool = False,  # `kv_route_dedup_anchor`; drop selected units' anchors
    score_block: int
    | None = None,  # `kv_route_score_block`; identity here -- see the guard below
    reduce: str = "mean",  # `kv_route_reduce`; reduces the query-head ballots
    head_reduce: str = "amax",  # `kv_route_head_reduce`; how the KV-head ballots fold
    qhead_block: int
    | None = None,  # `kv_route_qhead_block`; 1 -> each query head votes on its own
    pos_t: torch.Tensor | None = None,  # (N,) the same positions on the device
    pad_positions: int = 0,  # size the grid for this many positions, not for `p_last`
    anchor_topk: int = 0,  # serving cap (anchors); 0 = full anchor
    total_kv_heads: int
    | None = None,  # TP: global KV head count (None -> rank-local == global)
    kv_head_start: int = 0,  # TP: global index of this rank's first KV head
    group_size: int
    | None = None,  # TP: FULL query-group size (None -> q's local ratio)
    all_reduce=None,  # TP: SUM all-reduce callable across the TP group
    max_reduce=None,  # TP: MAX all-reduce; combines the ranks' route scores
    head_shared: bool = True,  # False -> each KV head selects, and attends, on its own
    plan_cache=None,  # the STEP's dict; see `_pkey` below
    kv_cache=None,  # (key_cache, value_cache), each (blocks, page, heads, dim)
    sink_units: int = 0,  # from `kv_route_sink_tokens` (resolved to units at
    # init); leading units forced like OWN, inside the
    # same top-k budget (0 = pre-sink ranking)
    per_row: bool = False,  # use the trainer's per-row ballot normalization
    output_gate: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Every decode request of an engine step in a handful of kernels: one routing kernel, the
    ranking, and the attention over the union.

    A decode row sees every union column -- anchors and selected tokens are all at positions
    <= p_last by construction -- so there is no boundary piece and no causal mask.
    """
    _validate_output_gate(q2, output_gate, output)
    if qhead_block is not None:
        if qhead_block != 1:
            raise ValueError(
                f"kv_route_qhead_block={qhead_block} is not supported; use 1 (every query head "
                f"votes) or omit it (the GQA group is averaged)."
            )
        if anchor_topk:
            raise NotImplementedError(
                "kv_route_qhead_block cannot be combined with the decode anchor cap "
                "(kv_route_anchor_topk / kv_route_anchor_topk_decode). The cap ranks ANCHORS on "
                "one pooled per-head score, and with the query heads split there is no such "
                "score. Drop the cap, or leave the heads pooled."
            )
        if reduce != "smax":
            # The constructor refuses this pair; repeated here because tests and probes call
            # this body directly. The per-head fold kernels normalise every ballot (smax) and
            # nothing else, while the prefill body follows `reduce` -- the two would elect on
            # different arithmetic under any other value.
            raise ValueError(
                f"kv_route_qhead_block=1 with kv_route_reduce={reduce!r}: the per-head decode "
                "folds softmax ballots (smax) only; the prefill would follow the other reduce."
            )
        if score_block is None and not per_row:
            # Likewise: without a sub-query span the prefill body takes its pooled branch and
            # never reads `qhead_block`, while this body would elect per head.
            raise ValueError(
                "kv_route_qhead_block=1 needs kv_route_score_block set: the pooled scoring branch averages the GQA group and never reads this knob, so the prefill would elect pooled while the decode elects per head. Set kv_route_score_block=1 (the per-head recipe) or drop kv_route_qhead_block."
            )
    if sink_units and not topk:
        # The constructor refuses this pair; repeated here because tests and probes call this
        # body directly (the prefill body carries the same guard). At topk 0 the ranking keeps
        # every unit, so the sink would be a silent no-op rather than the served function.
        raise ValueError(
            f"sink_units={sink_units} with topk=0: no ranking runs at topk 0, so the sink has "
            "no top-k to be forced into."
        )
    # `score_block` and `reduce` arrive for call-site uniformity with the prefill route. Under
    # the pooled election they are the identity here (they subdivide a QUERY BLOCK, and this
    # body gets one row per request); under the per-head election `reduce` is pinned to smax
    # just above, as the prefill's per-head branch requires.
    dev = q2.device
    N, hq, d = q2.shape
    hkv, rb = num_kv_heads, route_block
    # ⚠️ ONLY the selected region is re-addressed at a sub-page below. The scorer and the anchor
    # pass take `pos // PAGE` of the original page, which is already page-agnostic.
    _fused_no = _fused_decode_ok(
        head_shared=head_shared,
        anchor_in_union=anchor_in_union,
        anchor_topk=anchor_topk,
        qhead_block=qhead_block,
        head_reduce=head_reduce,
        route_block=rb,
        stride=stride,
    )
    if not head_shared:
        if qhead_block is not None or head_reduce == "smax":
            raise ValueError(
                f"kv_route_head_shared=False cannot be combined with "
                f"kv_route_qhead_block={qhead_block!r} / kv_route_head_reduce={head_reduce!r}: "
                "both collapse the KV heads into one ranking, which is what per-head selection "
                "removes. All three are serving-only knobs the training config does not carry, "
                "so a checkpoint that sets this pair was configured by hand."
            )
        if anchor_topk:
            raise NotImplementedError(
                "kv_route_head_shared=False with kv_route_anchor_topk is not implemented: the "
                "cap ranks anchor TILES on a score pooled over the KV heads, and with each head "
                "selecting on its own there is no such score. Unset kv_route_anchor_topk."
            )
        if dedup_anchor:
            raise NotImplementedError(
                "kv_route_head_shared=False with kv_route_dedup_anchor is not implemented here: "
                "dedup clears the anchor columns of a SELECTED unit, so with each head selecting "
                "differently the union width becomes per head as well. No released checkpoint "
                "carries this pair; one that does was configured by hand."
            )
    # The step's shape, host side and device side. Every value below is a function of `p_last` and
    # the routing config, so `plan_cache` -- the step's dict, filled by whichever routed layer runs
    # first -- lets the rest reuse it. Keyed on what it is derived FROM.
    if p_last is None:
        # Capture path. The grid is sized from the model's ceiling, so every SHAPE below is a
        # constant and only the per-request VALUES move; those are derived on the device, because
        # a readback cannot run inside a graph and a cached one would be the warmup's.
        assert pad_positions, (
            "routed_decode_batched was given no host positions and no ceiling to size the grid "
            "from. Pass pad_positions=<max_model_len>, or pass p_last."
        )
        _apg = rb // stride
        # Round UP: a ceiling that is not a whole number of route blocks still has positions in
        # its last, partial unit, and a floored grid leaves them with no unit to be selected into
        # -- including a row's own. The host branch below already rounds its anchor count up to
        # `_apg`; these two have to agree, and the shipped ceilings (256K, 1M) divide evenly, so
        # only an odd `--max-model-len` shows the difference.
        n_units = -(-pad_positions // rb)
        A_max = n_units * _apg
        # One compiled region, not twelve launches -- see `_decode_plan_eager`.
        pl_t, _ol_t, _ns_t, _S_t, q0_t = _decode_plan(pos_t, rb, q_block, topk)
        ns_i = nu_i = S_i = None
    else:
        _pkey = ("decode", tuple(p_last), stride, q_block, rb, topk, head_shared)
        _pt = plan_cache.get(_pkey) if plan_cache is not None else None
        if _pt is None:
            # `_apg` is the anchor page: the `route_block // stride` anchors of ONE routing unit. The
            # anchor axis is a whole number of them, so the attention can address it by page id and
            # `dedup_anchor` becomes "leave this unit's page out". Padded slots hold positions past
            # `p_last`, which the scoring mask already drops.
            _apg = rb // stride
            A_max = max((pl + 1) // stride for pl in p_last)
            if A_max % _apg:
                A_max += _apg - A_max % _apg
            nu_i = [pl // rb + 1 for pl in p_last]
            # ...and up to the unit grid the scorer ranks on. The grid spans POSITIONS, so it can hold
            # one unit more than the anchors do: the block the last row sits in, whose anchors are not
            # closed yet.
            A_max = max(A_max, max(nu_i) * _apg) if A_max else 0
            # ⚠️ Decode has no boundary pass, so the own units reach the row ONLY through the
            # selection: this floor is what keeps the row's own keys in the union.
            _own_lo_i = [(pl // q_block) * q_block // rb for pl in p_last]
            ns_i = [min(topk, x) if topk else x - ol for x, ol in zip(nu_i, _own_lo_i)]
            S_i = [
                ns * rb - (nu * rb - (pl + 1)) for ns, nu, pl in zip(ns_i, nu_i, p_last)
            ]
            # The ranking grid, which the scorer and the selection both size themselves from.
            n_units = max(max(nu_i), (A_max + _apg - 1) // _apg)
            _pad = pad_positions or _PAD_ROUTE
            if _pad and A_max:
                # Sized for the LONGEST request the model can take, so the shapes stop moving with
                # position, which is what a cudagraph capture needs. The padding costs shape, not
                # work: both kernels skip a split whose first anchor is past the row without loading
                # it, so the bytes moved still follow the real context.
                n_units = max(n_units, _pad // rb)
                A_max = n_units * _apg
            pl_t_host = torch.tensor(p_last, device=dev)
            _pt = (
                pl_t_host,
                torch.tensor(_own_lo_i, device=dev)[:, None, None],
                torch.tensor(ns_i, device=dev)[:, None, None],
                torch.tensor(S_i, device=dev),
                _apg,
                A_max,
                nu_i,
                ns_i,
                S_i,
                n_units,
                (pl_t_host // q_block) * q_block,
            )
            if plan_cache is not None:
                plan_cache[_pkey] = _pt
        pl_t, _ol_t, _ns_t, _S_t, _apg, A_max, nu_i, ns_i, S_i, n_units, q0_t = _pt

    # The anchor half and the ranking come from one pass over the anchor keys. `A_max` of zero is
    # a request shorter than one stride, which has no anchors to rank.
    if not _fused_no and A_max:
        # The pooled router query, reassembled across ranks when tensor parallelism split the GQA
        # group -- the only piece the kernel cannot derive from the heads it holds.
        #
        # The condition is the split itself, exactly as the prefill route asks it. Adding "and this
        # rank is missing a KV head" would look equivalent and is not: at ONE total KV head every
        # rank holds that head, so the head test is false while the group is still split, and each
        # rank would rank the units on its own share of the query.
        _qp = None
        _tkv = total_kv_heads or hkv
        if (
            qhead_block is None
            and all_reduce is not None
            and group_size is not None
            and group_size != hq // hkv
        ):
            # The per-head election has no pooled query to reassemble: every ballot is a single
            # head's, computed whole on the rank that holds it, and the cross-rank max in
            # `_fused_route_decode` combines the finished ballots.
            _g = hq // hkv
            # Training sums heads in fp32, then divides once after the TP sum.
            # A bf16-rounded local mean can change the hard top-k selection.
            _qg = q2.view(N, hkv, _g, d)
            _part = _qg[:, :, 0].float()
            for _j in range(1, _g):
                _part = _part + _qg[:, :, _j].float()
            _qz = _part.new_zeros(N, _tkv, d)
            _qz[:, kv_head_start : kv_head_start + hkv] = _part
            _qp = (
                all_reduce(_qz)[:, kv_head_start : kv_head_start + hkv] / group_size
            ).contiguous()
        _rf3 = torch.profiler.record_function("routed_attn.decode_fused")
        _rf3.__enter__()
        if kv_cache is None or len(kv_cache) != 2:
            raise ValueError(
                "the routed decode needs (key_cache, value_cache) to read the union; got "
                f"{type(kv_cache).__name__}. Pass the pair the caller already holds."
            )
        _out = _fused_route_decode(
            q2,
            _qp,
            kv_cache[0],
            kv_cache[1],
            bt_rows,
            pl_t,
            a_max=A_max,
            n_units=n_units,
            apg=_apg,
            stride=stride,
            page=block_size,
            rb=rb,
            hkv=hkv,
            scale=scaling,
            topk=topk,
            q_block=q_block,
            dedup=dedup_anchor,
            out_dtype=q2.dtype,
            max_reduce=max_reduce,
            tkv=_tkv,
            sink_units=sink_units,
            qhead=qhead_block is not None,
            group_size=group_size,
            exact_ballot=per_row,
            output_gate=output_gate,
            output=output,
        )
        _rf3.__exit__(None, None, None)
        return _out
    if _fused_no:
        # Refused, not absorbed: serving this shape on other arithmetic would change which route
        # blocks the checkpoint reads, and nothing in the output would say so.
        raise NotImplementedError(
            f"the routed decode cannot serve this configuration: {_fused_no}. Serve a checkpoint "
            "that meets it, or add a decode for this shape."
        )

    # No anchor has closed: every row is still inside its first stride window, so there is nothing
    # for the router to rank. The row's own route block is the whole selection -- the floor the
    # ranking forces anyway -- and the union is that block's raw tokens. (The sink asks for the
    # LEADING units, and the selection here IS the leading units, so it is served by construction.)
    #
    # vLLM's warmup and a health check are what reach here; a served request leaves this branch as
    # soon as its context passes `kv_subsample_stride` tokens.
    with torch.profiler.record_function("routed_attn.decode_no_anchor"):
        _kc_sub, _vc_sub, _bt_sub, _page_sub = subpage_cache(
            key_cache, value_cache, bt_rows, rb
        )
        k_eff = min(topk or max(ns_i), max(nu_i))
        # Ascending by construction, which is the order a page table has to name pages in.
        _idx = torch.arange(k_eff, device=dev).expand(N, k_eff)
        # A routing unit is a whole number of sub-pages, so the selection IS a block table and
        # flash reads the cache in place.
        _btab, _max_k = _sel_pages(_idx, _bt_sub, rb // _page_sub, _page_sub)
        out, _ = _varlen_lse(
            q2,
            _kc_sub,
            _vc_sub,
            cu_q=torch.arange(N + 1, dtype=torch.int32, device=dev),
            max_q=1,
            cu_k=None,
            max_k=_max_k,
            seqused=_S_t.to(torch.int32).clamp(min=1),
            btab=_btab,
            scale=scaling,
        )
    return _finish_output(
        out.float().to(q2.dtype).reshape(N, hq * d), output_gate, output
    )


def _merge_own_half_eager(o_b, lse_b, o_own, l_own):
    """LSE-combine the boundary's gap-anchor half with the own half flash returned.

    Both are in the real-row layout, so this is row-for-row. Flash gives lse as
    (num_heads, total_q) and the merge wants (total_q, num_heads).

    ⚠️ ``nan_to_num`` covers the ANCHOR half and is load-bearing: a query block whose gap-anchor
    columns are all masked comes back non-finite, and its weight here is 0, but ``nan * 0`` is
    ``nan``.

    Written as one expression because the arithmetic is memory-bound: what costs is how many times
    ``o_b`` is walked, not the flops.
    """
    l_own = l_own.T.contiguous().float()
    m = torch.maximum(lse_b, l_own)
    wb, wo = (lse_b - m).exp(), (l_own - m).exp()
    den = wb + wo
    o = (torch.nan_to_num(o_b) * wb[..., None] + o_own.float() * wo[..., None]) / den[
        ..., None
    ]
    return o, m + den.log()


_merge_own_half = torch.compile(_merge_own_half_eager, dynamic=True)


def _merge_two_eager(o_a, o_ro, wa, wro, out_dtype):
    """Union softmax over [anchors] and the fused [selected ; own] region.

    ``nan_to_num`` is load-bearing on the anchor side: a block whose anchor segment is empty
    carries ``-inf`` lse, its weight is 0, and ``nan * 0`` is ``nan``, not 0.

    The arithmetic runs in fp32 and the result is cast INSIDE the compiled region: the same number
    either way, but casting at the call site would write a full fp32 tensor and read it back.
    """
    return (
        (
            torch.nan_to_num(o_a.float()) * wa[..., None]
            + torch.nan_to_num(o_ro.float()) * wro[..., None]
        )
        / (wa + wro)[..., None]
    ).to(out_dtype)


_merge_two = torch.compile(_merge_two_eager, dynamic=True)


def _decode_plan_eager(pl_t, rb: int, q_block: int, topk: int):
    """The step's positions -> every per-request number the routed decode derives from them.

    Compiled rather than cached across layers: vLLM reuses the attention metadata across replays,
    so a cached tensor would be the warmup's. This way every step derives its own numbers and still
    pays for one kernel instead of twelve.

    Returns `(pl_t, own_lo, n_sel, S, q0)`; the middle two carry the two trailing axes their
    consumers index with.
    """
    nu = pl_t // rb + 1  # units the row reaches into
    q0 = (pl_t // q_block) * q_block  # first position of the row's own block
    ol = q0 // rb  # first of the row's OWN units
    ns = torch.clamp(nu, max=topk) if topk else nu - ol  # selected units
    s = ns * rb - (nu * rb - (pl_t + 1))  # selected TOKENS, last unit part-full
    return pl_t, ol[:, None, None], ns[:, None, None], s, q0


_decode_plan = torch.compile(_decode_plan_eager, dynamic=True)


def _sel_pages_eager(sel0, bt_rows, ppu, block_size):
    """The selected units as block-table entries.

    ``route_block`` is a whole number of pages, so a selected unit needs no gather at all: flash
    reads the cache in place through these page ids. Units the ranking left as the -inf sentinel
    land past the end of the block table; the clamp keeps the gather legal and ``seqused`` cuts
    them off -- the sentinel sorts last, so the columns it occupies are the trailing ones.
    """
    N = sel0.shape[0]
    pg = (sel0[:, :, None] * ppu + torch.arange(ppu, device=sel0.device)).reshape(N, -1)
    btab = torch.gather(bt_rows, 1, pg.clamp(min=0, max=bt_rows.shape[1] - 1)).to(
        torch.int32
    )
    return btab, pg.shape[1] * block_size


_sel_pages = torch.compile(_sel_pages_eager, dynamic=True)


def _merge_three_eager(o_a, o_r, o_b, wa, wr, wb, out_dtype):
    """The routed prefill's three pieces, back into the one softmax their union defines.

    ⚠️ ``nan_to_num`` on the ANCHOR and SELECTED outputs: flash returns non-finite rows for a
    fully masked segment, and weight 0 does not save them because ``nan * 0`` is ``nan``. ``o_b``
    arrives finite from the boundary merge.

    The traversals are the cost here, not the flops, which is why the cast to the output dtype
    happens inside the compiled region rather than at the call site.
    """
    return (
        (
            torch.nan_to_num(o_a.float()) * wa[..., None]
            + torch.nan_to_num(o_r.float()) * wr[..., None]
            + o_b * wb[..., None]
        )
        / (wa + wr + wb)[..., None]
    ).to(out_dtype)


_merge_three = torch.compile(_merge_three_eager, dynamic=True)


MAX_DEPTH_SOURCES = 32


def attn_res_writes_before(sublayer_idx: int, block_size: int) -> int:
    """Depth-sources already banked when global SUBLAYER ``sublayer_idx`` runs.

    A sublayer at global index ``u`` (attention of layer i is ``2i``, its MoE ``2i+1``) closes the
    open block iff ``u % block_size == 0``; ``u = 0`` has no open block yet and instead banks the
    token EMBEDDING as source 0, which is the same store. So the count of banked sources before
    ``u`` is ``1 + floor((u-1)/S)`` for ``u > 0``, and 0 at ``u = 0``.

    The rule is SUBLAYER-indexed, because for an odd block size a boundary lands on a MoE
    sublayer. The total bank width is ``attn_res_writes_before(2 * n_layers, S)``, and the final
    mix sees that many sources plus the open prefix.
    """
    if sublayer_idx <= 0:
        return 0
    return 1 + (sublayer_idx - 1) // block_size


def _validate_scorer_config(config) -> None:
    """Reject a scorer this server cannot serve, before any weight is touched."""
    scorer = getattr(config, "attn_res_scorer", "static")
    if scorer not in ("static", "cp"):
        raise NotImplementedError(
            f"solar_open3 vLLM serves attn_res_scorer 'static' or 'cp', got {scorer!r}"
        )
    # Carries no weights, so a value nothing can honour would be ignored rather than raise later.
    scale = getattr(config, "attn_res_scale_by_sources", False)
    if not isinstance(scale, bool):
        raise ValueError(f"attn_res_scale_by_sources must be a bool, got {scale!r}")
    if scale and getattr(config, "attn_res_block_size", None) is None:
        raise ValueError(
            "attn_res_scale_by_sources=True needs attn_res_block_size to enable "
            "Attention-Residuals; there is no depth mix to scale when it is None"
        )
    gated_norm = getattr(config, "attn_res_gated_norm", False)
    gated_norm_dim = getattr(config, "attn_res_gated_norm_dim", None)
    if not isinstance(gated_norm, bool):
        raise ValueError(f"attn_res_gated_norm must be a bool, got {gated_norm!r}")
    if gated_norm and getattr(config, "attn_res_block_size", None) is None:
        raise ValueError(
            "attn_res_gated_norm=True needs attn_res_block_size to enable Attention-Residuals"
        )
    if gated_norm_dim is not None and (
        not isinstance(gated_norm_dim, int)
        or isinstance(gated_norm_dim, bool)
        or gated_norm_dim <= 0
        or gated_norm_dim > config.hidden_size
    ):
        raise ValueError(
            f"attn_res_gated_norm_dim={gated_norm_dim!r} must be in 1..{config.hidden_size} "
            "or None"
        )
    if not gated_norm and gated_norm_dim is not None:
        raise ValueError(
            "attn_res_gated_norm_dim is set while attn_res_gated_norm=False"
        )
    if (
        gated_norm
        and (config.hidden_size // 8 if gated_norm_dim is None else gated_norm_dim) < 1
    ):
        raise ValueError(
            "attn_res_gated_norm_dim defaults to hidden_size // 8, which must be >= 1; "
            f"got hidden_size={config.hidden_size}"
        )
    # Which reads carry a gate. config.json spells it as a JSON list; a missing key means both,
    # which is what every gated checkpoint exported before the knob has. This one is NOT a
    # weightless constant: ("moe",) means half as many gate matrices exist, so getting it wrong
    # is a load-time failure rather than a silent numerical difference.
    # `is None`, not `or`: an EMPTY list has to reach the check below and be refused, not be
    # silently promoted to the default the way a missing key legitimately is. This validator runs
    # in `SolarOpen3Model.__init__` before any layer is built, so it is what the `or` fallbacks in
    # the decoder layer and the weight-name check are allowed to lean on.
    gated_norm_reads = getattr(config, "attn_res_gated_norm_reads", None)
    if gated_norm_reads is None:
        gated_norm_reads = ("attn", "moe")
    if isinstance(gated_norm_reads, list):
        gated_norm_reads = tuple(gated_norm_reads)
    if (
        not isinstance(gated_norm_reads, tuple)
        or not gated_norm_reads
        or len(set(gated_norm_reads)) != len(gated_norm_reads)
        or any(r not in ("attn", "moe") for r in gated_norm_reads)
    ):
        raise ValueError(
            "attn_res_gated_norm_reads must be a non-empty subset of ('attn', 'moe') with no "
            f"duplicates, got {gated_norm_reads!r}"
        )
    if not gated_norm and set(gated_norm_reads) != {"attn", "moe"}:
        raise ValueError(
            f"attn_res_gated_norm_reads={gated_norm_reads} narrows a gate that is off "
            "(attn_res_gated_norm=False)"
        )
    # The depth-source budget binds BOTH scorers (see MAX_DEPTH_SOURCES: the bank is per-source
    # activation memory regardless of who scores it), so it is checked before the static early
    # return -- a static-scorer config over the cap must refuse here, not OOM the KV allowance.
    block_size = getattr(config, "attn_res_block_size", None)
    if isinstance(block_size, int) and block_size >= 1:
        num_sources = (
            attn_res_writes_before(2 * config.num_hidden_layers, block_size) + 1
        )
        if num_sources > MAX_DEPTH_SOURCES:
            raise NotImplementedError(
                f"the fused depth-mix kernel serves at most {MAX_DEPTH_SOURCES} depth sources; "
                f"attn_res_block_size={block_size} over {config.num_hidden_layers} layers needs "
                f"{num_sources}"
            )
    if scorer != "cp":
        return
    block_size = getattr(config, "attn_res_block_size", None)
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError(
            f"attn_res_scorer='cp' needs a positive attn_res_block_size, got {block_size!r}"
        )
    qdim = getattr(config, "attn_res_scorer_dim", 0)
    if not isinstance(qdim, int) or not 0 < qdim <= config.hidden_size:
        raise ValueError(
            f"attn_res_scorer_dim={qdim!r} must be in 1..{config.hidden_size}"
        )


@triton.jit
def _attn_res_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    SCALE_BY_SOURCES: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    updated_prefix = tl.load(
        prefix_ptr + row_idx * stride_prefix_m + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + row_idx * stride_delta_m + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        updated_prefix += delta
        # Match the BF16 prefix-add result before using it as a residual source.
        updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix_ptr + row_idx * stride_prefix_m + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    if WRITE_BLOCK:
        tl.store(
            blocks_ptr
            + row_idx * stride_block_m
            + block_write_idx * stride_block_r
            + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    # With only the prefix source, the AttnRes softmax is exactly one.
    if num_blocks == 0:
        mixed = updated_prefix
    else:
        # Reloading avoids keeping the full prefix vector live across the loop.
        if HAS_DELTA:
            tl.debug_barrier()
        input_qk_weight = tl.load(
            norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32) * tl.load(
            qk_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        mixed = tl.zeros((BLOCK_D,), tl.float32)

        num_sources = num_blocks + 1
        for source_tile in range(tl.cdiv(num_sources, BLOCK_L)):
            source_offsets = source_tile * BLOCK_L + tl.arange(0, BLOCK_L)
            source_mask = source_offsets < num_sources
            is_prefix = source_offsets == num_blocks
            block_ptrs = (
                blocks_ptr
                + row_idx * stride_block_m
                + source_offsets[:, None] * stride_block_r
                + d_offsets[None, :]
            )
            prefix_ptrs = (
                prefix_ptr
                + row_idx * stride_prefix_m
                + source_offsets[:, None] * 0
                + d_offsets[None, :]
            )
            value_ptrs = tl.where(is_prefix[:, None], prefix_ptrs, block_ptrs)
            values = tl.load(
                value_ptrs,
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            reciprocal_std = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
            logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std
            scores = tl.where(source_mask, logits, -float("inf"))

            new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
            old_scale = tl.exp(max_logit - new_max_logit)
            block_scales = tl.exp(scores - new_max_logit)
            denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
            mixed = mixed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
            max_logit = new_max_logit

        if SCALE_BY_SOURCES:
            # Multiply the convex combination by K = num_sources. The mix is read only by the
            # RMSNorm fused below, which is scale-invariant EXCEPT through its epsilon, and the
            # sources partition the residual stream -- so an unscaled mix makes that norm run at
            # K^2 * eps (81x at block size 10, 441x at 4). Folding K into the norm GAIN instead
            # would be wrong: it has to land inside the rsqrt, not outside it.
            mixed = mixed * (num_sources / denominator)
        else:
            mixed = mixed / denominator
    output = mixed

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if APPLY_OUTPUT_NORM:
        output_reciprocal_std = tl.rsqrt(
            tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0) * (1.0 / hidden_size)
            + output_norm_eps
        )
        output_norm_weight = tl.load(
            output_norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        output = mixed * output_reciprocal_std * output_norm_weight
    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output,
        mask=d_mask,
    )


@triton.jit
def _cp_attn_res_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    score_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_score_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    output_norm_eps: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    SCALE_BY_SOURCES: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Direct dynamic-query depth router.

    Same online-softmax tiling as the static kernel above: sources are consumed ``BLOCK_L`` at a
    time against a running max and denominator, so each source is read once and the mix needs no
    materialised score vector. The one structural difference is where a logit comes from — the
    static op scores a source by its own RMSNorm'd content, this one by a per-source router row —
    so a tile loads a ``[BLOCK_L, BLOCK_D]`` router slab in addition to the values.

    An earlier revision scanned routers and values in two separate scalar loops to avoid holding
    both ``[K, H]`` slabs at once. Tiling gets the same bound from BLOCK_L instead, and the router
    slab is dead once its logits are reduced, so peak pressure is ONE slab either way — while the
    cross-warp reductions drop from ``2K`` to ``2*ceil(K/BLOCK_L)``. That is the whole cost
    difference against the static kernel, and at decode K is the width of a single tile.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    updated_prefix = tl.load(
        prefix_ptr + row_idx * stride_prefix_m + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + row_idx * stride_delta_m + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        updated_prefix += delta
        # The trainer next query consumes the storage-dtype result of the residual add.
        updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix_ptr + row_idx * stride_prefix_m + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    if WRITE_BLOCK:
        tl.store(
            blocks_ptr
            + row_idx * stride_block_m
            + block_write_idx * stride_block_r
            + d_offsets,
            updated_prefix,
            mask=d_mask,
        )

    # A single visible source has softmax weight exactly one and needs no router read.
    if num_blocks == 0:
        mixed = updated_prefix
    else:
        max_logit = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        mixed = tl.zeros((BLOCK_D,), tl.float32)

        num_sources = num_blocks + 1
        for source_tile in range(tl.cdiv(num_sources, BLOCK_L)):
            source_offsets = source_tile * BLOCK_L + tl.arange(0, BLOCK_L)
            source_mask = source_offsets < num_sources
            is_prefix = source_offsets == num_blocks
            # Scores: s_i = dot(updated_prefix, score_weight_i). _fold_cp_routers already baked the
            # 1/sqrt(qdim) scale into this matrix, so there is nothing to apply here.
            router = tl.load(
                score_weight_ptr
                + source_offsets[:, None] * stride_score_r
                + d_offsets[None, :],
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            logits = tl.sum(updated_prefix[None, :] * router, axis=1)
            scores = tl.where(source_mask, logits, -float("inf"))

            # Only BANKED sources are read; the open prefix is already in registers. The static
            # kernel reloads it through a pointer select because its logit needs the value anyway —
            # here that would be one extra [BLOCK_D] read per call for a vector we are holding, and
            # at prefill that is the whole cost of the rewrite (measured +9.2% at T=8192).
            values = tl.load(
                blocks_ptr
                + row_idx * stride_block_m
                + source_offsets[:, None] * stride_block_r
                + d_offsets[None, :],
                mask=(source_mask & ~is_prefix)[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            values = tl.where(is_prefix[:, None], updated_prefix[None, :], values)

            new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
            old_scale = tl.exp(max_logit - new_max_logit)
            block_scales = tl.exp(scores - new_max_logit)
            denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
            mixed = mixed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
            max_logit = new_max_logit

        if SCALE_BY_SOURCES:
            # Multiply the convex combination by K = num_sources. The mix is read only by the
            # RMSNorm fused below, which is scale-invariant EXCEPT through its epsilon, and the
            # sources partition the residual stream -- so an unscaled mix makes that norm run at
            # K^2 * eps (81x at block size 10, 441x at 4). Folding K into the norm GAIN instead
            # would be wrong: it has to land inside the rsqrt, not outside it.
            mixed = mixed * (num_sources / denominator)
        else:
            mixed = mixed / denominator

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    output = mixed
    if APPLY_OUTPUT_NORM:
        reciprocal_std = tl.rsqrt(
            tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0) * (1.0 / hidden_size)
            + output_norm_eps
        )
        output_norm_weight = tl.load(
            output_norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        output = mixed * reciprocal_std * output_norm_weight
    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output,
        mask=d_mask,
    )


@triton.jit
def _cp_attn_res_prefill_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    score_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_score_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    output_norm_eps: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    SCALE_BY_SOURCES: tl.constexpr,
    SCORE_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_PAD: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """The CP depth mix for a PREFILL chunk: every logit first, then the bank as a pure stream.

    Same program-per-row shape, same traffic and the same arithmetic as ``_cp_attn_res_kernel``,
    reordered. That kernel interleaves scoring and mixing per source tile, so every value load
    waits behind a cross-warp reduction and the online softmax rescales the accumulator on the
    way; at H=4096 that held the 17-source chunk at 2.4 TB/s against a ~7.5 TB/s floor (measured,
    `.jobs/260903_sglang_attnres_serving_cost/results/cp_*_sweep.txt`). Here phase 1 reduces the
    K logits from the router rows alone and takes the softmax over a ``[K_PAD]`` vector in
    registers; phase 2 is ``mixed += p_i * v_i`` with nothing in the loop but loads and FMAs, so
    the loads pipeline. Per call at T=8192, 17 sources: 537 us -> 291 us; the 129-call schedule of
    the 64-layer arm: 35.0 ms -> 19.9 ms per 8192-token chunk.

    Taken only above 512 tokens and with at least two banked sources; the fused kernel keeps
    decode, where a single program per row is latency-bound and its wide tile is the right shape.
    The mix is fp32 in source order (no rescaling), which changes the rounding of the softmax
    weights by 2^-24-scale amounts against the fused kernel -- both sit at bf16 rounding of the
    oracle (``tests/model/solar_open3/test_cp_attn_res_kernel_gpu.py``).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    updated_prefix = tl.load(
        prefix_ptr + row_idx * stride_prefix_m + d_offsets, mask=d_mask, other=0.0
    ).to(tl.float32)
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + row_idx * stride_delta_m + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        updated_prefix += delta
        # The trainer next query consumes the storage-dtype result of the residual add.
        updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix_ptr + row_idx * stride_prefix_m + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    if WRITE_BLOCK:
        tl.store(
            blocks_ptr
            + row_idx * stride_block_m
            + block_write_idx * stride_block_r
            + d_offsets,
            updated_prefix,
            mask=d_mask,
        )

    num_sources: tl.constexpr = num_blocks + 1
    source_ids = tl.arange(0, K_PAD)
    # ---- phase 1: the K logits, from the router rows alone. `_fold_cp_routers` baked 1/sqrt(qdim)
    # into the score matrix, so there is nothing to apply here. The K_PAD - K padding rows need no
    # mask of their own: they keep the -inf this initialises them to, because only entries a
    # `tile_mask` admits are ever overwritten.
    logits = tl.full((K_PAD,), -float("inf"), tl.float32)
    # Integer arithmetic, not tl.cdiv: on Triton 3.6 (the vLLM serving venv's pin) tl.cdiv of two
    # constexprs returns an int32 tensor, which static_range refuses as a bound; 3.7 folds it.
    for source_tile in tl.static_range((num_sources + SCORE_L - 1) // SCORE_L):
        tile_ids = source_tile * SCORE_L + tl.arange(0, SCORE_L)
        tile_mask = tile_ids < num_sources
        router = tl.load(
            score_weight_ptr + tile_ids[:, None] * stride_score_r + d_offsets[None, :],
            mask=tile_mask[:, None] & d_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        tile_logits = tl.sum(updated_prefix[None, :] * router, axis=1)  # [SCORE_L]
        hit = (source_ids[:, None] == tile_ids[None, :]) & tile_mask[
            None, :
        ]  # [K_PAD, SCORE_L]
        logits = tl.where(
            tl.sum(hit.to(tl.int32), axis=1) > 0,
            tl.sum(tl.where(hit, tile_logits[None, :], 0.0), axis=1),
            logits,
        )
    max_logit = tl.max(logits, axis=0)
    probs = tl.exp(logits - max_logit)
    probs = probs / tl.sum(probs, axis=0)  # zeros past num_sources
    # ---- phase 2: the mix. The open prefix is the LAST source and is already in registers; the
    # banked sources stream in with nothing between the loads.
    mixed = updated_prefix * tl.sum(
        tl.where(source_ids == num_blocks, probs, 0.0), axis=0
    )
    for source in tl.static_range(num_blocks):
        values = tl.load(
            blocks_ptr + row_idx * stride_block_m + source * stride_block_r + d_offsets,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        mixed += values * tl.sum(tl.where(source_ids == source, probs, 0.0), axis=0)
    if SCALE_BY_SOURCES:
        # See `_cp_attn_res_kernel`: K rides inside the rsqrt of the fused norm, by design.
        mixed = mixed * num_sources

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    output = mixed
    if APPLY_OUTPUT_NORM:
        reciprocal_std = tl.rsqrt(
            tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0) * (1.0 / hidden_size)
            + output_norm_eps
        )
        output_norm_weight = tl.load(
            output_norm_weight_ptr + d_offsets, mask=d_mask, other=0.0
        ).to(tl.float32)
        output = mixed * reciprocal_std * output_norm_weight
    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output,
        mask=d_mask,
    )


def _is_arch_support_pdl() -> bool:
    """Whether the current device supports PDL (Programmatic Dependent Launch).

    The vLLM adapter reads this off its platform layer. This module stays free of both engines'
    imports so the kernels can be extracted and tested on their own, so the same predicate is
    expressed in torch directly.

    PDL is CUDA-only: ROCm reports the gfx version as the capability (MI355X -> (9, 5)), and the
    kernels' ``gdc_*`` calls lower to PTX ``griddepcontrol.*``, which the AMDGPU assembler rejects.
    """
    if torch.version.cuda is None:
        return False
    try:
        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 9


def _attn_res_launch(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    output: torch.Tensor,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    scale_by_sources: bool,
) -> None:
    num_tokens, hidden_size = prefix.shape
    # Same budget assert as the CP launcher, on the same constant, so the validator and the two
    # launchers cannot drift apart.
    assert num_blocks + 1 <= MAX_DEPTH_SOURCES, (
        f"the depth-mix kernel serves at most {MAX_DEPTH_SOURCES} sources"
    )
    # Only last-dim contiguity is required; strides are passed explicitly, so row-padded
    # storage works.
    assert prefix.stride(-1) == 1
    assert delta is None or delta.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1
    assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
    if num_tokens == 0:
        return
    # Tile width over depth sources, re-tuned for this model's hidden size. At or below the decode
    # batch, take the smallest power-of-two tile covering every source in ONE iteration -- the +1
    # is the open prefix, which rides the source loop. Above 512 tokens the grid alone fills the
    # device and the wide tile's register pressure stops paying, so upstream's width is kept.
    # `num_blocks >= 1` keeps the single-source case on the one-wide branch.
    if num_tokens <= 512 and 1 <= num_blocks < 16:
        block_l, num_warps = triton.next_power_of_2(num_blocks + 1), 4
    elif num_blocks <= 1:
        block_l, num_warps = 1, 4
    elif num_tokens >= 256:
        block_l, num_warps = 4, 4
    elif num_blocks >= 16:
        block_l, num_warps = 8, 4
    else:
        block_l, num_warps = 4, 8
    _attn_res_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        num_blocks,
        hidden_size,
        block_write_idx,
        eps,
        output_norm_eps,
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        SCALE_BY_SOURCES=scale_by_sources,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        num_stages=2,
        launch_pdl=_is_arch_support_pdl(),
    )


def _cp_attn_res_launch(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    score_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    output: torch.Tensor,
    num_blocks: int,
    block_write_idx: int,
    output_norm_eps: float,
    scale_by_sources: bool,
) -> None:
    """Launch the direct dynamic-query router.

    ``score_weight`` is the FOLDED per-consumer matrix ``_fold_cp_routers`` derives at load time
    (``E @ Wq / sqrt(qdim)``), one row per possible depth source; ``num_blocks + 1`` selects the
    visible prefix, and the kernel applies no further scale because the fold baked
    ``1/sqrt(qdim)`` in. ⚠️ Only shape and stride are asserted, so an unfolded matrix would serve
    silently-wrong routes -- fill this from the derived buffers, never from raw weights.
    """
    num_tokens, hidden_size = prefix.shape
    assert prefix.stride(-1) == 1
    assert delta is None or delta.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert score_weight.ndim == 2 and score_weight.stride(-1) == 1
    assert score_weight.shape[1] == hidden_size
    assert score_weight.shape[0] >= num_blocks + 1
    assert score_weight.shape[0] <= MAX_DEPTH_SOURCES, (
        f"the depth-mix kernel serves at most {MAX_DEPTH_SOURCES} sources"
    )
    assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
    if num_tokens == 0:
        return

    # THREE branches, in the order they are tested. A real prefill chunk -- above 512 tokens with a
    # bank of two or more blocks -- goes to the scored-then-streamed kernel just below and none of
    # the tile-width reasoning that follows it applies. What is left for `_cp_attn_res_kernel` is
    # the decode batch and the degenerate prefill shapes (a bank of 0 or 1 block, which has nothing
    # to stream).
    #
    # Tile width over depth sources, for that kernel. At or below the decode batch, take the
    # smallest power-of-two tile covering every source in ONE iteration -- the +1 is the open
    # prefix, which rides the source loop as a register broadcast. That is where tiling pays: the
    # grid is small and the kernel is latency-bound on sequential cross-warp reductions. Above 512
    # tokens the tile is 1, because a CP iteration holds a [BLOCK_L, BLOCK_D] router slab on top of
    # the value slab and at prefill the grid already fills the device, so the extra register
    # pressure only costs occupancy.
    #
    # Both branches are measured, not reasoned: `.jobs/260807_attnres_inference/RESULTS.md` replays
    # the real 81-consumer schedule off a rotated source bank (jobs 65312/65314), where untiling
    # this costs +5.6% at the decode batch and the tile stops paying above it.
    if num_tokens > 512 and num_blocks >= 2:
        # A prefill chunk with a real bank: the scored-then-streamed kernel. Launch parameters
        # are the sweep's optimum at H=4096 on Triton 3.7.1 and 3.6.0 alike (4 warps; the 8-warp
        # and 16-warp variants and every multi-row tile measured slower); SCORE_L=4 scores four
        # router rows per reduction.
        _cp_attn_res_prefill_kernel[(num_tokens,)](
            prefix,
            delta,
            blocks,
            score_weight,
            output_norm_weight,
            output,
            prefix.stride(0),
            0 if delta is None else delta.stride(0),
            blocks.stride(0),
            blocks.stride(1),
            score_weight.stride(0),
            output.stride(0),
            num_blocks,
            hidden_size,
            block_write_idx,
            output_norm_eps,
            HAS_DELTA=delta is not None,
            WRITE_BLOCK=block_write_idx >= 0,
            APPLY_OUTPUT_NORM=output_norm_weight is not None,
            SCALE_BY_SOURCES=scale_by_sources,
            SCORE_L=4,
            BLOCK_D=triton.next_power_of_2(hidden_size),
            K_PAD=max(2, triton.next_power_of_2(num_blocks + 1)),
            num_warps=4,
            num_stages=3,
            launch_pdl=_is_arch_support_pdl(),
        )
        return
    if num_blocks <= 1 or num_tokens > 512:
        block_l = 1
    else:
        block_l = triton.next_power_of_2(num_blocks + 1)
    num_warps = 8 if num_tokens <= 512 else 4

    _cp_attn_res_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        score_weight,
        output_norm_weight,
        output,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        score_weight.stride(0),
        output.stride(0),
        num_blocks,
        hidden_size,
        block_write_idx,
        output_norm_eps,
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        SCALE_BY_SOURCES=scale_by_sources,
        # Size the tile from the VISIBLE prefix, not the physical [n_src, H] checkpoint matrix:
        # unused suffix rows are stored for every consumer, and letting them drive this constexpr
        # would reserve a 16-wide tile even for K=2/3/4/5..8.
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        num_stages=2,
        launch_pdl=_is_arch_support_pdl(),
    )


def _attn_res_fake(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    output: torch.Tensor,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
    scale_by_sources: bool,
) -> None:
    return None


def _cp_attn_res_fake(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    score_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    output: torch.Tensor,
    num_blocks: int,
    block_write_idx: int,
    output_norm_eps: float,
    scale_by_sources: bool,
) -> None:
    return None
