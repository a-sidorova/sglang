"""The trainer's arithmetic in the SDA (grouped-decay linear attention) layers, prefill and decode.

The trainer (flash_sda e1b0b122 under torch.compile) feeds its kernels the decay gate
``-exp(A_log) * where(z > 20, z, log1p(exp(z)))`` (libdevice exp / log1p, fp32), the 64-token
chunk-local cumsum of FlashSDA's TileLang ``_ccs``, ``bf16(2 * sigmoid(b))`` and
``bf16(x * rsqrt(sum(x^2) + 1e-6))`` for q and k. Serving differed:

- prefill (FlashSDA 8a4a109's SGLang adapter): after the first guarded calls of a layer its fast
  gate uses ``log(1 + exp(z))`` / ``exp(z)`` with ``tl.exp``, fuses the cumsum into a Triton
  kernel, and normalises q/k as ``x / sqrt(...)``;
- decode (SGLang cb838a0f8's ``fused_sigmoid_gating_delta_rule_update_kernel``): ``log(1 + exp(z))``
  (0 below z ~ -16.6), and beta, q and k kept in fp32 without the trainer's bf16 rounding.

``install`` points those names at the functions below. Each name is read at call time, so the swap
holds for every later call in this process, including decode graphs captured after it.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def _pick_act_train_kernel(
    g_ptr,
    bias_ptr,
    alog_ptr,
    out_ptr,
    rows,
    H,
    stride_t,
    stride_h,
    stride_d,
    WIDTH: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # FlashSDA 8a4a109 _fast_gate._pick_act_kernel; only the activation lines differ.
    pid = tl.program_id(0)
    r = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    valid = r < rows
    t = r // H
    h = r % H
    gi = tl.arange(0, G)
    offs = (
        t[:, None] * stride_t + h[:, None] * stride_h + (gi[None, :] * WIDTH) * stride_d
    )
    z = tl.load(g_ptr + offs, mask=valid[:, None], other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + h[:, None] * G + gi[None, :], mask=valid[:, None], other=0.0)
    a = tl.load(alog_ptr + h, mask=valid, other=0.0)
    z = z + b
    sp = tl.where(z > 20.0, z, tldevice.log1p(tldevice.exp(z)))
    y = -tldevice.exp(a)[:, None] * sp
    tl.store(out_ptr + r[:, None] * G + gi[None, :], y, mask=valid[:, None])


def pick_act_train(
    g: torch.Tensor, A_log: torch.Tensor, bias: torch.Tensor | None, groups: int
) -> torch.Tensor:
    """FlashSDA's ``_fast_gate.pick_act`` contract (``g`` [1, T, H, DK] raw, one lane read per group;
    ``A_log`` [H]; ``bias`` [H, G], [H*G] or None -> the activated grouped gate [1, T, H, G] fp32)
    with the trainer's activation."""
    assert g.dim() == 4 and g.shape[0] == 1, tuple(g.shape)
    _, T, H, DK = g.shape
    width = DK // groups
    assert width * groups == DK, (DK, groups)
    out = torch.empty(1, T, H, groups, device=g.device, dtype=torch.float32)
    if bias is None:
        bias_t = torch.zeros(H * groups, device=g.device, dtype=torch.float32)
    else:
        bias_t = bias.reshape(-1).to(torch.float32).contiguous()
        assert bias_t.numel() == H * groups, (bias_t.numel(), H, groups)
    a_log = A_log.reshape(-1).to(torch.float32).contiguous()
    rows = T * H
    _pick_act_train_kernel[(triton.cdiv(rows, 128),)](
        g,
        bias_t,
        a_log,
        out,
        rows,
        H,
        g.stride(1),
        g.stride(2),
        g.stride(3),
        WIDTH=width,
        G=groups,
        BLOCK=128,
        num_warps=4,
    )
    return out


@triton.jit
def _l2norm_rsqrt_kernel(X, Y, n_rows, eps, BLOCK_ROWS: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    mask = (rows < n_rows)[:, None]
    offs = rows[:, None] * D + tl.arange(0, D).to(tl.int64)[None, :]
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tldevice.rsqrt(tl.sum(x * x, axis=1) + eps)
    tl.store(Y + offs, (x * rstd[:, None]).to(Y.dtype.element_ty), mask=mask)


def l2norm_train(x: torch.Tensor, eps: float = 1e-6):
    """The trainer's q/k l2norm ``x * rsqrt(sum(x^2) + eps)`` (compiled flash_sda ``l2norm_fwd``),
    rows reduced 16 at a time as that compile lays them out, in the ``(y, rstd)`` contract of
    FlashSDA's ``sglang_kernel.l2norm_fwd``. ``x`` is contiguous (the caller makes it so)."""
    d = x.shape[-1]
    x2 = x.reshape(-1, d)
    y = torch.empty_like(x2)
    n = x2.shape[0]
    if n:
        _l2norm_rsqrt_kernel[(triton.cdiv(n, 16),)](
            x2, y, n, eps, BLOCK_ROWS=16, D=d, num_warps=8
        )
    return y.view_as(x), None


@triton.jit(do_not_specialize=["T"])
def decode_kernel_train(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    lower_bound,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    stride_h0_source,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    BETA_SCALE: tl.constexpr = 1.0,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    # ReplaySSM fused ring-write. Pointers stay None and CACHE_RING False for
    # decode / flag-off -> byte-identical. The gate ring layout follows IS_KDA
    # (see the store below).
    replayssm_rawv=None,
    replayssm_rawk=None,
    replayssm_g=None,
    replayssm_beta=None,
    stride_rawv_slot: tl.constexpr = 0,
    stride_rawk_slot: tl.constexpr = 0,
    stride_g_slot: tl.constexpr = 0,
    stride_beta_slot: tl.constexpr = 0,
    MAX_CACHE_LEN: tl.constexpr = 0,
    CACHE_RING: tl.constexpr = False,
):
    """SGLang cb838a0f8's ``fused_sigmoid_gating_delta_rule_update_kernel`` with the trainer's operands:
    the decay gate's softplus and exp(A_log) in libdevice as Inductor lowers them, beta rounded to
    bf16, and q/k l2-normalised as ``x * rsqrt(sum(x^2) + 1e-6)`` and rounded to bf16. Every other
    line is the fork's."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # Slot stride comes from the caller (h0_source.stride(0)): the state pool
        # may be an envelope-strided view (page-major / unified memory), where the
        # per-slot pitch spans ALL layers' state, not HV*K*V. int64: envelope
        # pitches overflow an int32 index product.
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * stride_h0_source
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled. int64: the buffer is
    # contiguous but `cache_idx * cache_steps * HV * K * V` can exceed int32 for
    # large slot counts.
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        x = b_a + b_dt_bias
        if USE_LOWER_BOUND:
            # KDA safe gate: lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
            b_g = lower_bound * tl.sigmoid(tl.exp(b_A_log) * x)
        else:
            # Compute g = -exp(A_log) * softplus(a + dt_bias)
            beta_x = softplus_beta * x
            # trainer: F.softplus under Inductor -> where(z > 20, z, log1p(exp(z))) in libdevice
            softplus_x = tl.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * tldevice.log1p(tldevice.exp(beta_x)),
                x,
            )
            b_g = -tldevice.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = (1.0 / (1.0 + tl.exp(-b_b))) * BETA_SCALE
        b_beta = b_beta.to(tl.bfloat16).to(
            tl.float32
        )  # the trainer hands bf16(2 * sigmoid(b)) to its kernels

        # fused ring-write: stash this step's raw inputs + in-kernel gate/beta
        # into the per-slot ring for the commit fold to replay. Must sit here --
        # b_k is still pre-l2norm, b_v still pre-delta, b_g/b_beta are formed,
        # so the fold's replay is bit-identical to the update below. rawk uses
        # the k-head i_h (shared across a GQA group); rawv/g/beta use the v-head
        # i_hv. step_idx < MAX_CACHE_LEN: absorb-inflated rows can exceed the
        # ring; the overflow steps are past the committable prefix, so drop them
        # (writing them would smash the next slot's ring).
        if CACHE_RING:
            ring_slot = tl.load(h0_indices + i_n).to(tl.int64)
            if ring_slot >= 0 and step_idx < MAX_CACHE_LEN:
                tl.store(
                    replayssm_rawv
                    + ring_slot * stride_rawv_slot
                    + i_hv * MAX_CACHE_LEN * V
                    + step_idx * V
                    + o_v,
                    b_v.to(replayssm_rawv.dtype.element_ty),
                    mask=mask_v,
                )
                if i_v == 0:
                    tl.store(
                        replayssm_rawk
                        + ring_slot * stride_rawk_slot
                        + i_h * MAX_CACHE_LEN * K
                        + step_idx * K
                        + o_k,
                        b_k.to(replayssm_rawk.dtype.element_ty),
                        mask=mask_k,
                    )
                    # b_g follows IS_KDA: KDA loads a/dt_bias with mask_k, so the
                    # gate is a per-K vector and the ring row is K wide; GDN's is
                    # a scalar per (head, step). The two layouts are not
                    # interchangeable -- storing one into the other's stride is a
                    # shape error, not a slow path -- and memory_pool.py sizes
                    # replayssm_g off the same is_kda test.
                    if IS_KDA:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN * K
                            + step_idx * K
                            + o_k,
                            b_g,
                            mask=mask_k,
                        )
                    else:
                        tl.store(
                            replayssm_g
                            + ring_slot * stride_g_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_g,
                        )
                    if i_k == 0:
                        tl.store(
                            replayssm_beta
                            + ring_slot * stride_beta_slot
                            + i_hv * MAX_CACHE_LEN
                            + step_idx,
                            b_beta,
                        )

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            # trainer: bf16(x * rsqrt(sum(x^2) + 1e-6)), as the compiled flash_sda l2norm
            b_q = (
                (b_q * tldevice.rsqrt(tl.sum(b_q * b_q) + 1e-6))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            b_k = (
                (b_k * tldevice.rsqrt(tl.sum(b_k * b_k) + 1e-6))
                .to(tl.bfloat16)
                .to(tl.float32)
            )

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n).to(tl.int64)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * stride_h0_source
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def install() -> None:
    """Swap the trainer's SDA arithmetic into FlashSDA's SGLang prefill and SGLang's KDA decode."""
    from flash_sda.integrations import _fast_gate, sglang_kernel

    from sglang.kernels.ops.attention.fla import (
        fused_sigmoid_gating_recurrent as decode,
    )

    targets = (
        (_fast_gate, "pick_act"),
        (sglang_kernel, "_FUSE_CUMSUM"),
        (sglang_kernel, "l2norm_fwd"),
        (sglang_kernel, "_kdag_act_compiled"),
        (decode, "fused_sigmoid_gating_delta_rule_update_kernel"),
    )
    for module, name in targets:
        if not hasattr(module, name):
            raise RuntimeError(
                f"solar_open3 SDA: {module.__name__}.{name} is gone; the trainer-numerics swap "
                "was written against FlashSDA 8a4a109 and SGLang cb838a0f8"
            )
    _fast_gate.pick_act = pick_act_train
    sglang_kernel._FUSE_CUMSUM = (
        False  # the cumsum then runs in chunk_kdag_fwd's TileLang _ccs, as trained
    )
    sglang_kernel.l2norm_fwd = l2norm_train
    sglang_kernel._kdag_act_compiled = lambda g, a_log, bias: pick_act_train(
        g, a_log, bias, g.shape[-1]
    )
    decode.fused_sigmoid_gating_delta_rule_update_kernel = decode_kernel_train
    logging.getLogger(__name__).info(
        "solar_open3: SDA gate, beta and q/k use the trainer's arithmetic"
    )
