# Copyright 2026 Upstage AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""solar_open3's linear-attention (``kda`` / ``sda``) MTP head as an SGLang EAGLE draft.

SGLang's EAGLE draft runs on a FULL-attention backend: ``DraftBackendFactory`` maps every backend
name, ``hybrid_linear_attn`` included, to one, and the draft's own KV pool holds "layer 0" as a
softmax layer. A head whose one decoder layer is KDA therefore has nowhere to keep its recurrent
and conv state, and no discipline for it across a speculative round. This module supplies both.
The fork exposes one hook for it -- ``LinearAttnModelSpec.draft_backend_wrapper`` -- which the
factory calls with each draft backend it built; nothing else in SGLang changes.

**Where the state lives.** Two slots per request, in two pools:

* the COMMITTED state, in the TARGET's own mamba pool at a reserved layer id
  (``solar_open3.linear_draft_state_layer_id`` -- the HIGHEST TRUNK GQA INDEX, which no trunk
  layer claims a slot under; one PAST the trunk is filtered straight back out by the pool
  builder). Being in that pool is what makes the head work with the radix cache: a prefix hit
  copies the slot (``MambaPool.copy_from``), a fresh request's slot is zeroed (``clear_slots``),
  and the interval-aligned snapshot the cache later reads is written by the ping-pong track --
  all three iterate the pool's layer axis, so the head is carried by machinery that already
  exists. Outside the pool it is carried by none of them and silently drafts from another
  request's state.
* the WORKING copy, in a private single-layer pool at the SAME slot ids. It is scratch that draft
  decode steps advance speculatively; it is never cached, forked or copied, so it has no business
  in the shared pool.

**The round.** For a request whose committed state is S_n:

1. draft DECODE steps (``speculative_num_steps - 1`` of them): each consumes one draft token and
   must see the state after the previous one. They run on the WORKING copy.
2. target TARGET_VERIFY accepts ``a`` of the drafted tokens. Its registered
   ``SolarOpen3HybridAttnBackend`` commits only the trunk layers: it must not copy the head's
   stale scratch into S_n before the head's own refresh.
3. draft DRAFT_EXTEND_V2: the head re-reads the whole draft block of ``D`` tokens with the target's
   hidden states. The new committed state must be S_{n+a}: the state after the ``a`` accepted
   tokens only, computed from S_n -- never from the working copy, which has consumed rejected
   drafts. So this pass runs exactly like the trunk's own verify: the recurrent kernel reads the
   COMMITTED slot with ``disable_state_update`` and snapshots every step into the speculative
   scratch, the conv update snapshots its window per step, and the accepted step ``a-1`` is
   scattered back into the committed slot -- together with the interval-crossing step, so the
   radix cache's snapshot of the head is written in the same pass the trunk's was. The working
   copy is then refreshed from the committed one.

Prefill-time draft extend (plain EXTEND) writes the committed slot through the ordinary chunked
path, lets the base track it like a trunk layer, and refreshes the working copy the same way.

**One approximation.** The head at position m consumes the trunk hidden at m together with token
m+1, so a snapshot at a radix boundary of m encodes the DONOR request's first uncached token. A
later request whose suffix diverges at m+1 restores a state polluted by that one token, at the
position a decaying state weights most. Correctness is untouched -- the target verifies every
token -- and the error decays; the cost is a little acceptance right after a hit. Making it exact
would mean caching the hidden state at the boundary and replaying that single update against the
new request's own token, which is a measurement away, not a guess.

**CUDA graphs.** Every input this module derives from is a graph-static buffer the runner refreshes
in place (``spec_info.num_accept_tokens``, ``forward_batch.seq_lens``, the backend's own
``state_indices_list`` and track buffer), and every derivation is a device op, so the commit and
the refresh record into the draft-extend graph and replay correctly -- except SGLang's single-launch
conv commit, whose pointer table is cached, so the head commits through ``_commit_in_graph``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from types import SimpleNamespace

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    fused_mamba_state_scatter_with_mask,
    scatter_mamba_states_after_mtp_verify,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_exec, get_spec

logger = logging.getLogger(__name__)


def _one_layer_view(caches, layer_index: int):
    """``caches`` narrowed to one layer, KEEPING the layer axis.

    ``MambaPool.State.at_layer_idx`` indexes the axis away, which is what the per-layer kernels
    want; the post-verify scatter indexes ``[:, slot]`` and needs it present. Narrowing is a view,
    so this names the same storage the head's verify kernel just wrote.
    """
    return SimpleNamespace(
        temporal=caches.temporal[layer_index : layer_index + 1],
        conv=[c[layer_index : layer_index + 1] for c in caches.conv],
        intermediate_ssm=caches.intermediate_ssm[layer_index : layer_index + 1],
        intermediate_conv_window=[
            c[layer_index : layer_index + 1] for c in caches.intermediate_conv_window
        ],
    )


def _commit_in_graph(caches, committed, steps, track_indices, track_steps) -> None:
    """``scatter_mamba_states_after_mtp_verify``, minus its single-launch conv path.

    That path launches from a pointer table kept in a ONE-entry module cache, freed on any other
    key. The draft-extend graph bakes in the head's table; the target's eager commit (another key)
    frees it, and once ``/flush_cache``'s ``empty_cache()`` unmaps it the next replay faults. The
    per-pair kernels (SGLang's own fallback, same order) take every pointer as a launch argument.
    """
    if caches.temporal.numel() > 0:
        fused_mamba_state_scatter_with_mask(
            caches.temporal, caches.intermediate_ssm, committed, steps
        )
        if track_indices is not None:
            fused_mamba_state_scatter_with_mask(
                caches.temporal, caches.intermediate_ssm, track_indices, track_steps
            )
    pairs = list(zip(caches.conv, caches.intermediate_conv_window))
    for conv_states, window in pairs:
        fused_conv_window_scatter_with_mask(conv_states, window, committed, steps)
    if track_indices is not None:
        for conv_states, window in pairs:
            fused_conv_window_scatter_with_mask(
                conv_states, window, track_indices, track_steps
            )


def chain_track_steps(
    pre_verify_seq_lens: torch.Tensor,
    accept: torch.Tensor,
    interval: int,
    committed: torch.Tensor,
) -> torch.Tensor:
    """Which drafted step to snapshot for the prefix cache, per request, or -1 for none.

    SGLang's rule, from ``spec_utils._verify_commit_step_indices``: a request whose sequence
    length crosses a ``mamba_track_interval`` boundary inside the accepted span snapshots the
    state at that boundary. Restricted to a CHAIN, which is the only shape this head drafts
    (topk 1), where SGLang's ``accept_index[req, i] - offset`` is the identity on ``i``.

    Pure and tensor-only so the two derivations can be compared directly:
    ``tests/model/solar_open3/test_sglang_linear_draft_track_plan.py``.
    """
    post = pre_verify_seq_lens + accept
    crossing = (pre_verify_seq_lens // interval) != (post // interval)
    point = post // interval * interval
    step = torch.clamp(point - pre_verify_seq_lens - 1, min=0).to(committed.dtype)
    return torch.where(crossing & (committed >= 0), step, torch.full_like(step, -1))


class LinearDraftState:
    """The head's recurrent/conv state: a committed slot and a working copy (module docstring)."""

    def __init__(self, runner, layer_id: int, expect_shared: bool) -> None:
        target_pool = runner.req_to_token_pool
        if not isinstance(target_pool, HybridReqToTokenPool):
            raise RuntimeError(
                "solar_open3 linear draft: the draft runner's req_to_token_pool is "
                f"{type(target_pool).__name__}, not the target's HybridReqToTokenPool. The head's "
                "state shares the target's request -> mamba-slot mapping and cannot exist "
                "without it."
            )
        server_args = get_spec()
        num_draft_tokens = server_args.speculative_num_draft_tokens
        if not num_draft_tokens:
            raise RuntimeError(
                "solar_open3 linear draft: --speculative-num-draft-tokens is unset; the "
                "post-verify scratch is sized by it."
            )
        self.layer_id = layer_id
        self.num_draft_tokens = int(num_draft_tokens)
        # The head's KDA shape is the trunk's linear_attn_config, so the target's own cache params
        # describe it; only the layer list changes.
        cache_params = dataclasses.replace(
            runner.model_config.hf_config.mamba2_cache_params, layers=[layer_id]
        )

        # Share only when the config reserved the head's layer in the target's pool.
        # An all-linear trunk or the explicit control-arm opt-out uses private committed state.
        # Membership alone is not ownership: an all-linear trunk uses fallback id 0,
        # which belongs to its first real layer. Only share a deliberately reserved layer.
        self.shared = expect_shared and layer_id in target_pool.mamba_map
        if expect_shared and not self.shared:
            raise RuntimeError(
                f"solar_open3 linear draft: the head's state layer {layer_id} is not in the "
                f"target's mamba pool (it holds {sorted(target_pool.mamba_map)[:4]}...). The "
                "config reserved it but the pool was built without it, so the head would fall "
                "back to a private pool the radix cache cannot carry -- the silent-stale-state "
                "case this adapter refuses. The reserved id must be inside the rank's "
                "pipeline-parallel layer range, and `serves_linear_draft` must hold at POOL BUILD "
                "time, not only when the head is constructed."
            )
        if self.shared:
            self.committed_pool: HybridReqToTokenPool = target_pool
        else:
            self.committed_pool = target_pool.clone_with_new_mamba(
                mamba_size=target_pool.mamba_pool.size,
                mamba_spec_state_size=target_pool.size,
                cache_params=cache_params,
                device=runner.device,
                enable_mamba_extra_buffer=False,
                draft_model_idx=layer_id,
                speculative_num_draft_tokens=self.num_draft_tokens,
                speculative_eagle_topk=server_args.speculative_eagle_topk,
            )

        # The working copy: same slot ids as the committed state, plus one sink row the padded
        # rows of a cuda-graph replay copy into, so every copy is a fixed-shape gather. No
        # speculative scratch -- the refresh runs on the committed pool, never on this one.
        self.num_slots = int(target_pool.mamba_pool.size)
        # MambaSlotAllocator hands out 1..size INCLUSIVE; row 0 is already reserved.
        # The enlarged working pool has one further row beyond every live target slot.
        self.sink = self.num_slots + 1
        self.working_pool = target_pool.clone_with_new_mamba(
            mamba_size=self.num_slots + 1,
            mamba_spec_state_size=target_pool.size,
            cache_params=cache_params,
            device=runner.device,
            enable_mamba_extra_buffer=False,
            draft_model_idx=layer_id,
            speculative_num_draft_tokens=None,
        )
        logger.info(
            "[solar_open3] linear draft head: state layer %d, %d slots, %d draft tokens of "
            "scratch, committed state %s",
            layer_id,
            self.num_slots,
            self.num_draft_tokens,
            "IN the target's pool (radix cache carries it)"
            if self.shared
            else "in a PRIVATE pool (radix cache does NOT carry it)",
        )

    def refresh_working(self, committed: torch.Tensor) -> None:
        """working[i] <- committed[i] for every row; a padded row (-1) copies into the sink."""
        valid = committed >= 0
        src = torch.where(valid, committed, torch.zeros_like(committed)).long()
        dst = torch.where(
            valid, committed, torch.full_like(committed, self.sink)
        ).long()
        c = self.committed_pool.mamba2_layer_cache(self.layer_id)
        w = self.working_pool.mamba2_layer_cache(self.layer_id)
        for c_conv, w_conv in zip(c.conv, w.conv):
            w_conv[dst] = c_conv[src]
        w.temporal[dst] = c.temporal[src]


class _RunnerWithPool:
    """The draft runner as ``KDAAttnBackend`` reads it, with the head's committed pool in place."""

    def __init__(self, runner, pool: HybridReqToTokenPool) -> None:
        self._runner = runner
        self.req_to_token_pool = pool

    def __getattr__(self, name: str):
        return getattr(self._runner, name)


class SolarOpen3LinearDraftBackend(KDAAttnBackend):
    """``KDAAttnBackend`` on the head's pools, plus the round discipline (module docstring)."""

    def __init__(self, runner, state: LinearDraftState) -> None:
        super().__init__(_RunnerWithPool(runner, state.committed_pool))
        self.state = state
        if self.topk > 1:
            raise NotImplementedError(
                "solar_open3 linear draft: --speculative-eagle-topk > 1 builds a tree; the head's "
                "recurrent state is a chain (one state per position) and cannot branch. Use topk 1."
            )

    # ---- which pool a forward touches ---------------------------------------------------------

    @contextlib.contextmanager
    def _on_working_pool(self):
        """Point the inherited forward at the working copy for the duration of one call."""
        saved = self.req_to_token_pool
        self.req_to_token_pool = self.state.working_pool
        try:
            yield
        finally:
            self.req_to_token_pool = saved

    def _track_mamba_state_decode(self, *args, **kwargs):
        """No-op: a draft decode step advances the WORKING copy.

        The inherited decode calls this whenever the batch carries a track mask, which would
        snapshot a speculative state into the slot the radix cache restores from -- a state that
        includes drafts the target has not accepted. The head is tracked from the post-verify
        refresh and from the prefill-time extend, both of which run on the committed slot.
        """
        return None

    # ---- metadata ------------------------------------------------------------------------------

    def _forward_metadata(self, forward_batch: ForwardBatch):
        md = super()._forward_metadata(forward_batch)
        mode = forward_batch.forward_mode
        if mode.is_decode_or_idle():
            # See _track_mamba_state_decode; nulling both keeps every tracking path off.
            md.mamba_track_indices = None
            md.has_mamba_track_mask = False
        elif mode.is_draft_extend_v2():
            # The base plans no linear metadata for this mode; the head runs it verify-shaped.
            num_tokens = forward_batch.input_ids.shape[0]
            draft_token_num = int(forward_batch.spec_info.num_tokens_per_req)
            md.query_start_loc = torch.arange(
                0,
                num_tokens + 1,
                step=draft_token_num,
                dtype=torch.int32,
                device=self.device,
            )
            md.is_target_verify = True
            md.draft_token_num = draft_token_num
            md.has_mamba_track_mask = (
                False  # the verify-shaped path tracks through the commit
            )
        # Plain EXTEND (the prefill-time draft extend) keeps the base's tracking: the head's state
        # is snapshotted at the same interval boundary as a trunk layer's, which is what a later
        # prefix hit restores.
        return md

    def _replay_metadata(
        self,
        bs,
        req_pool_indices,
        forward_mode,
        spec_info,
        seq_lens_cpu,
        num_padding=None,
        in_capture=False,
        mamba_track_indices=None,
    ):
        if forward_mode.is_draft_extend_v2():
            draft_token_num = int(spec_info.num_tokens_per_req)
            shim = SimpleNamespace(
                draft_token_num=draft_token_num,
                ragged_verify_layout=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
            )
            md = super()._replay_metadata(
                bs,
                req_pool_indices,
                ForwardMode.TARGET_VERIFY,
                shim,
                seq_lens_cpu,
                num_padding=num_padding,
                in_capture=in_capture,
                mamba_track_indices=mamba_track_indices,
            )
            md.is_target_verify = True
            md.draft_token_num = draft_token_num
            return md
        md = super()._replay_metadata(
            bs,
            req_pool_indices,
            forward_mode,
            spec_info,
            seq_lens_cpu,
            num_padding=num_padding,
            in_capture=in_capture,
            mamba_track_indices=None
            if forward_mode.is_decode_or_idle()
            else mamba_track_indices,
        )
        if forward_mode.is_decode_or_idle():
            md.mamba_track_indices = None
        return md

    # ---- forwards --------------------------------------------------------------------------

    def forward_decode(
        self, layer, forward_batch: ForwardBatch, mixed_qkv, a, b, **kwargs
    ):
        with self._on_working_pool():
            return super().forward_decode(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )

    def forward_extend(
        self, layer, forward_batch: ForwardBatch, mixed_qkv, a, b, **kwargs
    ):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return self._forward_draft_extend_v2(layer, forward_batch, mixed_qkv, a, b)
        if forward_batch.forward_mode.is_target_verify():
            raise RuntimeError(
                "solar_open3 linear draft: TARGET_VERIFY is a target-model mode"
            )
        # Prefill-time draft extend: the ordinary chunked path over the committed slot.
        out = super().forward_extend(layer, forward_batch, mixed_qkv, a, b, **kwargs)
        self.state.refresh_working(self.forward_metadata.mamba_cache_indices)
        return out

    def _track_plan(
        self,
        forward_batch: ForwardBatch,
        committed: torch.Tensor,
        accept: torch.Tensor,
        draft_token_num: int,
    ):
        """The radix-cache snapshot destination and step for this refresh, or ``(None, None)``.

        The rule is SGLang's own (``spec_utils._verify_commit_step_indices``): a request whose
        sequence length crosses a ``mamba_track_interval`` boundary inside the accepted span
        snapshots the state at that boundary into its ping-pong slot, and ``-1`` means no crossing.
        It is re-derived here rather than read off the target's commit because that commit runs
        BEFORE this pass, inside the verify; ``test_sglang_linear_draft_track_plan.py`` holds the
        two derivations, including the worker-to-refresh sequence lengths, to the same answer.

        Reduces to the step index because a chain is the only shape this head drafts (topk 1),
        where SGLang's ``accept_index[req, i] - offset`` is the identity.
        """
        track = self.forward_metadata.mamba_track_indices
        if track is None or not self.state.shared:
            return None, None
        bs = committed.shape[0]
        # prepare_for_draft_extend adds the whole draft block to PRE-VERIFY seq_lens.
        # Spec_v2 leaves batch.seq_lens unchanged by acceptance; post-verify consumers
        # compute batch.seq_lens + accept_lens separately. Undo only the draft block.
        pre = forward_batch.seq_lens[:bs] - draft_token_num
        return track[:bs], chain_track_steps(
            pre, accept, get_exec().mamba.mamba_track_interval, committed
        )

    def _forward_draft_extend_v2(
        self, layer, forward_batch: ForwardBatch, mixed_qkv, a, b
    ):
        """The post-verify refresh: verify-shaped over the committed slot, then commit ``a-1``.

        Mirrors ``KDAAttnBackend._forward_target_verify`` for a dense chain (topk 1, no ragged
        layout), with the commit that the trunk leaves to the worker done here -- the head is one
        layer, and ``spec_info.num_accept_tokens`` is in hand.
        """
        fm = self.forward_metadata
        draft_token_num = fm.draft_token_num
        # The refresh reads the window as EXACTLY the draft block: it commits step `accept - 1`
        # and asks the kernel for `draft_token_num` steps of a scratch the target sized to
        # `speculative_num_draft_tokens`. A worker that widens the window (multi-layer EAGLE's
        # boundary-KV fix prepends `num_steps - 1 + conv_width` front rows) breaks all three at
        # once, and silently -- the target verifies either way. `_refuse_linear_head_preconditions`
        # refuses that worker by name; this is the same fact held where it would do the damage.
        if draft_token_num != self.state.num_draft_tokens:
            raise RuntimeError(
                f"solar_open3 linear draft: DRAFT_EXTEND_V2 carries {draft_token_num} tokens per "
                f"request, not the draft block of {self.state.num_draft_tokens} the committed "
                "pool's speculative scratch was sized to. A widened draft-extend window would "
                "commit the wrong step and overrun that scratch with no symptom but lost "
                "acceptance; refusing instead."
            )
        cache_indices = fm.mamba_cache_indices
        query_start_loc = fm.query_start_loc
        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = cache.conv[0]
        ssm_states = cache.temporal
        intermediate_ssm = getattr(cache, "intermediate_ssm", None)
        if intermediate_ssm is None:
            raise RuntimeError(
                "solar_open3 linear draft: the head's committed pool carries no speculative "
                "scratch (MambaPool.SpeculativeState); it must be built with "
                "speculative_num_draft_tokens."
            )
        intermediate_conv_window = cache.intermediate_conv_window[0]

        num_tokens = mixed_qkv.shape[0]
        if num_tokens % draft_token_num != 0:
            raise RuntimeError(
                f"solar_open3 linear draft: DRAFT_EXTEND_V2 carries {num_tokens} tokens, not a "
                f"multiple of the draft block {draft_token_num}"
            )
        batch_size = num_tokens // draft_token_num
        rows = self.verify_intermediate_state_indices[:batch_size]
        committed = cache_indices[:batch_size]

        # Conv: per-step windows snapshotted; the persistent window advances by all D tokens and
        # the commit below restores the accepted step's.
        x = mixed_qkv.view(batch_size, draft_token_num, -1).transpose(1, 2)
        x = causal_conv1d_update(
            x,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=committed,
            intermediate_conv_window=intermediate_conv_window.transpose(-1, -2),
            intermediate_state_indices=rows,
        )
        qkv = x.transpose(1, 2).reshape(num_tokens, -1)
        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        # SSM: reads the committed state, never writes it; every step's post-state lands in the
        # scratch row of its batch position.
        core_attn_out = self.kernel_dispatcher.target_verify(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_states_buffer=intermediate_ssm,
            intermediate_state_indices=rows,
            cache_steps=draft_token_num,
            retrieve_parent_token=None,
            lower_bound=layer.lower_bound,
            beta_scale=getattr(layer, "beta_scale", 1.0),
        )

        # Commit: committed[i] <- scratch[row i, accept_len_i - 1], and, where this round crossed a
        # track boundary, the same scatter writes the ping-pong slot the radix cache reads. A
        # padded row (-1) is skipped by both.
        accept = forward_batch.spec_info.num_accept_tokens[:batch_size].to(torch.int32)
        last_correct = torch.where(
            committed >= 0, accept - 1, torch.full_like(accept, -1)
        )
        track_indices, track_steps = self._track_plan(
            forward_batch, committed, accept, draft_token_num
        )
        caches = self.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        layer_index = self.req_to_token_pool.mamba_map[layer.layer_id]
        _commit_in_graph(
            _one_layer_view(caches, layer_index),
            committed,
            last_correct,
            track_indices,
            track_steps,
        )
        self.state.refresh_working(committed)
        return core_attn_out


class SolarOpen3HybridAttnBackend(HybridLinearAttnBackend):
    """The target commits only trunk layers; the draft owns its reserved layer's commit.

    Prefix restore and fresh-slot clearing still see the complete pool. The target's verify
    never produces the head's intermediate states, so scattering that layer would replace its
    committed state with stale scratch before the head can run its post-verify refresh.
    """

    def __init__(self, full_attn_backend, linear_attn_backend, full_attn_layers):
        super().__init__(full_attn_backend, linear_attn_backend, full_attn_layers)
        pool = linear_attn_backend.req_to_token_pool
        reserved = [
            index
            for layer, index in pool.mamba_map.items()
            if layer in full_attn_layers
        ]
        # with_linear_draft_layer appends exactly one free GQA id. Keeping it last lets the
        # target scatter into a view, without a gather/copy of the trunk's recurrent states.
        if reserved and (len(reserved) != 1 or reserved[0] != len(pool.mamba_map) - 1):
            raise RuntimeError(
                "solar_open3: the draft state must be the last, sole reserved GQA layer in "
                f"the mamba pool; got reserved indices {reserved}, map {pool.mamba_map}."
            )
        self._trunk_state_layers = reserved[0] if reserved else None

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices,
        mamba_track_indices,
        mamba_steps_to_track,
        model,
        req_pool_indices=None,
    ):
        count = self._trunk_state_layers
        if count is None:
            return super().update_mamba_state_after_mtp_verify(
                last_correct_step_indices,
                mamba_track_indices,
                mamba_steps_to_track,
                model,
                req_pool_indices=req_pool_indices,
            )
        if count == 0:
            return
        linear = self.linear_attn_backend
        if mamba_track_indices is not None:
            mamba_track_indices = linear._translate_mamba_indices(mamba_track_indices)
        committed = linear.forward_metadata.mamba_cache_indices[
            : last_correct_step_indices.shape[0]
        ]
        caches = linear.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        trunk = SimpleNamespace(
            temporal=caches.temporal[:count],
            conv=[c[:count] for c in caches.conv],
            intermediate_ssm=caches.intermediate_ssm[:count],
            intermediate_conv_window=[
                c[:count] for c in caches.intermediate_conv_window
            ],
        )
        scatter_mamba_states_after_mtp_verify(
            trunk,
            committed,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )


class SolarOpen3LinearDraftHybrid(HybridLinearAttnBackend):
    """SGLang's hybrid dispatcher with the head's linear backend and no full-attention layers.

    Wraps one full-attention draft backend (a multi-step child, or the draft-extend one). The
    base skips linear metadata for DRAFT_EXTEND_V2 -- right for a trunk, wrong here -- so that
    hook is re-stated; everything the graph runners and worker read off the full backend is
    delegated.
    """

    def __init__(
        self, full_attn_backend, linear_attn_backend: SolarOpen3LinearDraftBackend
    ):
        super().__init__(full_attn_backend, linear_attn_backend, [])

    @property
    def graph_capture_backend(self):
        """The backend whose type decides draft-extend CUDA-graph support (fork hook)."""
        return self.full_attn_backend

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.full_attn_backend.init_forward_metadata(forward_batch)
        self.linear_attn_backend.init_forward_metadata(forward_batch)

    def __getattr__(self, name: str):
        if name in ("full_attn_backend", "linear_attn_backend"):
            raise AttributeError(name)
        return getattr(self.full_attn_backend, name)


class SolarOpen3LinearDraftMultiStep:
    """The multi-step draft container with the head's linear backend riding along.

    ``EAGLEDraftCudaGraphRunner`` and the eager draft loop drive the container through
    ``attn_backends[i]``, ``init_cuda_graph_state`` and the three metadata hooks; each is forwarded
    to the inner container and to the ONE linear backend (the decode steps share their metadata:
    the same working slots, one token per request).
    """

    def __init__(self, inner, linear_attn_backend: SolarOpen3LinearDraftBackend):
        self.inner = inner
        self.linear_attn_backend = linear_attn_backend
        self.attn_backends = [
            SolarOpen3LinearDraftHybrid(child, linear_attn_backend)
            for child in inner.attn_backends
        ]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.inner.init_forward_metadata(forward_batch)
        self.linear_attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.inner.init_cuda_graph_state(max_bs, max_num_tokens)
        self.linear_attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.inner.init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.linear_attn_backend.init_forward_metadata_out_graph(
            forward_batch, in_capture=in_capture
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.inner.init_forward_metadata_in_graph(forward_batch)
        self.linear_attn_backend.init_forward_metadata_in_graph(forward_batch)

    def __getattr__(self, name: str):
        if name in ("inner", "linear_attn_backend", "attn_backends"):
            raise AttributeError(name)
        return getattr(self.inner, name)


def draft_backend_wrapper(runner, backend, phase: str):
    """``LinearAttnModelSpec.draft_backend_wrapper``: hand the head's state to each draft backend.

    Called once per phase; both phases share one state object, created on first use by the draft
    model.
    """
    state = runner.model.linear_draft_state(runner)
    linear = SolarOpen3LinearDraftBackend(runner, state)
    if phase == "decode":
        return SolarOpen3LinearDraftMultiStep(backend, linear)
    if phase == "draft_extend":
        return SolarOpen3LinearDraftHybrid(backend, linear)
    raise ValueError(f"solar_open3 linear draft: unknown draft backend phase {phase!r}")
