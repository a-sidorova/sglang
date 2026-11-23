# Copyright 2026 Upstage AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""solar_open3's Multi-Token-Prediction head, served as an SGLang draft model.

SGLang runs MTP through its speculative-decoding path: the algorithm is EAGLE, and the "draft
model" is this one-layer head. Two facts make that work without patching SGLang.

* **How the class is FOUND.** ``ModelConfig._config_draft_model`` rewrites the draft
  ``architectures[0]`` only for a hardcoded list of names (DeepseekV3, GlmMoeDsa, ...). solar_open3
  is not on it, so the draft config's own ``architectures`` is used verbatim -- point
  ``--speculative-draft-model-path`` at a directory whose config says
  ``SolarOpen3ForCausalLMNextN`` and SGLang loads the native ``EntryClass`` below from its model registry. The hardcoded list is a convenience for checkpoints that do NOT name their draft
  arch; it is not a gate.
* **What it is HANDED.** The draft's ``forward`` gets the target's last hidden state through
  ``forward_batch.spec_info.hidden_states`` -- the same contract ``deepseek_nextn`` uses.

The head is ONE decoder layer reused across draft steps, plus the two pre-fc norms and the fc that
mixes the token embedding with the previous hidden state. Everything else is inherited.
"""

from __future__ import annotations

import copy
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.attention.solar_open3.precision import hold_fp32_params
from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import BumpAllocator

from .solar_open3 import (
    SolarOpen3DecoderLayer,
    SolarOpen3ForCausalLM,
    linear_draft_share_pool_opt_out,
    linear_draft_state_layer_id,
    patch_config,
    resolve_mtp_swiglu_limit,
    serves_linear_draft,
)

logger = logging.getLogger(__name__)


class SolarOpen3MultiTokenPredictor(nn.Module):
    """The head itself: embedding, two pre-fc norms, fc, one decoder layer, output norm."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # The inner attention is linear (an "sda" head arrives broadcast to plain KDA; the only
        # served head, see the converter), served through `solar_open3_linear_draft.py`: SGLang's
        # draft backends are full-attention only (`DraftBackendFactory` maps every name,
        # `hybrid_linear_attn` included, to one), so the head brings its own recurrent/conv state
        # pool and the fork's `draft_backend_wrapper` hook attaches it to each draft backend. Its
        # preconditions are refused HERE, by name, before any weight loads.
        _refuse_linear_head_preconditions(config)

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
        self.fc = ColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            gather_output=True,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        # ONE decoder layer, following SGLang's own hybrid-attention MTP head
        # (`models/qwen3_next_mtp.py`): derive a config, shrink it to the head's depth, and
        # override the knob that DECIDES THE LAYER TYPE -- there `full_attention_interval = 1`,
        # here `gqa_layers`, which is what `_is_kda_layer` reads. The layer picks its type from its
        # INDEX, and the index stays 0-based (the module index is what the weight keys and the
        # draft's pool see), so the config says the type -- and everything else the trunk decides
        # by index.
        mtp_config = copy.deepcopy(config)
        # The head's clamps by the trainer's head rule (None follows the trunk, "off" is none);
        # the out clips are the trunk's, as in the vLLM head.
        mtp_config.swiglu_limit = resolve_mtp_swiglu_limit(config)
        mtp_config.shared_expert_swiglu_limit = resolve_mtp_swiglu_limit(
            config, "shared_expert_swiglu_limit"
        )
        mtp_config.num_hidden_layers = self.num_mtp_layers
        # The type knob: `_is_kda_layer` reads the list, so [] makes the one layer KDA.
        mtp_config.gqa_layers = []
        # The trainer builds the head one PAST the trunk (`model.py:SolarMTPBlock`), so it is
        # always MoE. At index 0 here `first_k_dense_replace > 0` would build a dense MLP (the
        # `mtp.*.mlp.experts` weights then KeyError'd), so it is cleared for the head.
        mtp_config.first_k_dense_replace = 0

        with layer_stack():
            self.layers = nn.ModuleList(
                [
                    SolarOpen3DecoderLayer(
                        config=mtp_config,
                        layer_idx=idx,
                        quant_config=quant_config,
                        prefix=f"{prefix}.layers.{idx}",
                    )
                    for idx in range(self.num_mtp_layers)
                ]
            )

        # The head's recurrent + conv state lives in the TARGET's mamba pool at a reserved id so
        # the radix cache carries it (solar_open3_linear_draft, module docstring). Every pool
        # lookup keys on the id the layer's RadixLinearAttention reports, and nothing else reads
        # it, so re-pointing it here moves the state without touching the module index: the
        # weight keys stay `layers.0.*` and the draft's KV pool still sees index 0.
        state_layer = linear_draft_state_layer_id(config)
        for layer in self.layers:
            layer.self_attn.layer_idx = state_layer
            layer.self_attn.attn.layer_id = state_layer

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.prenorm = config.mtp_hidden_prenorm
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_ids,
        positions,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=(
                input_embeds.device if input_embeds is not None else input_ids.device
            ),
        )

        inputs_embeds = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        # The target's last hidden state, handed over by the speculative worker.
        hidden_states = forward_batch.spec_info.hidden_states
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]

        # ⚠️ Normalise the token embedding and the previous hidden state SEPARATELY, then
        # concatenate with the EMBEDDING FIRST and project 2*hidden -> hidden. Swapping the two
        # halves, or sharing one norm, changes the function without failing: fc still consumes a
        # 2*hidden vector and the head still emits plausible drafts, they are just wrong ones.
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))
        if isinstance(hidden_states, tuple):  # ColumnParallelLinear returns (out, bias)
            hidden_states = hidden_states[0]
        residual = None

        # One layer, reused across draft steps.
        hidden_states, residual = self.layers[0](
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=residual,
            zero_allocator=zero_allocator,
        )

        # `LogitsProcessor` hands the next draft step `hidden_states_before_norm`, or the post-norm
        # hidden when it is None (which kind: the trainer's `SolarMTPBlock`).
        before_norm = hidden_states if residual is None else hidden_states + residual
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, (before_norm if self.prenorm else None)


class SolarOpen3ForCausalLMNextN(SolarOpen3ForCausalLM):
    """The draft model SGLang loads. Named for the draft config's ``architectures[0]``."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # The parent builds the full trunk; the draft is one layer, so its body is re-stated here
        # rather than called -- the same reason SolarOpen3ForCausalLM re-states solar_open2's.
        nn.Module.__init__(self)
        config = patch_config(config)
        # ⚠️ NOT `_refuse_unimplemented`: that refuses `mtp_num_hidden_layers > 0`, which is
        # exactly what this class exists to serve. AttnRes is still refused for the head -- it is a
        # TRUNK-depth mechanism and the exported mtp.* weights carry no depth-mix tensors.
        if getattr(config, "attn_res_block_size", None) is not None:
            logger.info(
                "[solar_open3] MTP head keeps standard residuals; the trunk's AttnRes "
                "does not apply to it (no depth-mix tensors are exported under mtp.*)"
            )
            config = _without_attn_res(config)
        self.config = config
        self.quant_config = quant_config
        self.model = SolarOpen3MultiTokenPredictor(config, quant_config, prefix="model")
        hold_fp32_params(self.model, config)
        self.pp_group = get_parallel().pp_group
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix="lm_head",
        )
        self.logits_processor = LogitsProcessor(
            config=config, logit_scale=getattr(config, "logit_scale", 1.0)
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states, before_norm = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=before_norm,
        )

    def linear_draft_state(self, runner):
        """The head's recurrent-state pool for `solar_open3_linear_draft.draft_backend_wrapper`.

        Built once, on the first draft backend the fork hands over (both phases share it).
        """
        if getattr(self, "_linear_draft_state", None) is None:
            from sglang.srt.layers.attention.solar_open3.linear_draft import (
                LinearDraftState,
            )

            self._linear_draft_state = LinearDraftState(
                runner,
                layer_id=linear_draft_state_layer_id(self.config),
                expect_shared=serves_linear_draft(self.config),
            )
        return self._linear_draft_state

    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker: the draft shares the TARGET's embedding and LM head.

        The exported ``mtp.*`` carry neither, so the modules built here are placeholders until this
        replaces their weights. Skipping it leaves the head drafting from random parameters and
        scoring against a random LM head -- it accepts nothing and looks merely slow.
        """
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load the ``mtp.*`` half of a solar_open3 checkpoint.

        The trunk and the head live in ONE checkpoint: the trunk loader skips ``mtp.*`` and this
        one keeps only that prefix, so the two modules read the same files and neither sees the
        other's tensors.

        The parent loader maps the head by name: the exported `mtp.layers.0.self_attn.*` keys of
        the linear head are exactly a trunk KDA layer's (q/k/v_proj, the three conv1d, A_log,
        dt_bias, b_proj, f_a/f_b_proj, g_a/g_b_proj, o_norm, o_proj).
        """
        mtp_weights = []
        for args in weights:
            name, tensor = args[:2]
            if not name.startswith("mtp."):
                continue
            name = name[len("mtp.") :]
            # The exported head is one layer; its keys are layers.0.* whatever index the module was
            # constructed with, because the ModuleList still stores it at position 0.
            mtp_weights.append(
                (
                    f"model.{name}" if not name.startswith("model.") else name,
                    tensor,
                    *args[2:],
                )
            )
        if not mtp_weights:
            # Streaming updates can send a bucket containing only target weights.
            if getattr(self, "_solar_open3_mtp_loaded_once", False):
                return
            raise ValueError(
                "no `mtp.*` weights in this checkpoint. The draft model needs the MTP head that "
                "`mtp_num_hidden_layers > 0` says is there -- serving without it would draft from "
                "uninitialised parameters and accept nothing."
            )
        # Bypass the target's AttnRes completeness checks. Decoder factories
        # already install the head's FP32 router and clamp policy.
        loaded = super(SolarOpen3ForCausalLM, self).load_weights(mtp_weights)
        self._refuse_unserved_exact_erase()
        self._solar_open3_mtp_loaded_once = True
        return loaded


def _refuse_linear_head_preconditions(config) -> None:
    """What a linear head cannot do, refused by name at construction.

    A tree (`--speculative-eagle-topk > 1`) needs one state per branch; the head keeps one per
    position. And the radix cache only carries the head's state when that state sits in the
    target's mamba pool, which `serves_linear_draft` decides: outside it, a prefix hit leaves the
    head drafting from the previous occupant's state -- correct output, because the target
    verifies, and collapsed acceptance with no error anywhere. That is refused rather than served,
    unless the operator asked for it by name to measure what carrying the state is worth.
    """
    from sglang.srt.configs.linear_attn_model_registry import (
        get_linear_attn_spec_by_arch,
    )
    from sglang.srt.layers.attention.solar_open3.settings import runtime_options

    # Match DraftBackendFactory's lookup, including a stale adapter that registered a spec
    # without the hook even though this fork's dataclass supports it.
    spec = get_linear_attn_spec_by_arch("SolarOpen3ForCausalLMNextN")
    if spec is None or getattr(spec, "draft_backend_wrapper", None) is None:
        raise NotImplementedError(
            "SolarOpen3 MTP requires LinearAttnModelSpec.draft_backend_wrapper on "
            "the registered SolarOpen3ForCausalLMNextN specification. Use the "
            "SGLang checkout containing the native SolarOpen3 configuration."
        )

    sa = runtime_options()
    topk = getattr(sa, "speculative_eagle_topk", None)
    if topk is not None and topk > 1:
        raise NotImplementedError(
            "solar_open3 MTP: the linear head drafts a chain; "
            f"--speculative-eagle-topk={topk} asks for a tree. Use topk 1."
        )
    pp_size = getattr(sa, "pp_size", 1) or 1
    if pp_size > 1:
        raise NotImplementedError(
            "solar_open3 MTP: the linear head reserves ONE pool id for its state and "
            "`linear_draft_state_layer_id` picks it globally (the highest trunk GQA index), but "
            f"--pp-size={pp_size} builds the mamba pool per rank, keeping only ids inside that "
            "rank's layer range. The id would land in the LAST rank's range and nowhere else, so "
            "every earlier rank builds a pool without it and refuses to serve. Use --pp-size 1, "
            "or give the head a per-rank free GQA id first."
        )
    if getattr(sa, "enable_multi_layer_eagle", False):
        raise NotImplementedError(
            "solar_open3 MTP: the linear head cannot serve under "
            "--enable-multi-layer-eagle. That worker WIDENS the draft-extend window by "
            "`speculative_num_steps - 1 + conv_width` front rows (the boundary-KV fix, on by "
            "default) while advancing `seq_lens` by the draft block alone, and the head's "
            "post-verify refresh reads the window as exactly that block: it would commit step "
            "`accept - 1` where the accepted row is `front + accept - 1`, snapshot the radix "
            "cache from a position `front` tokens early, and ask the verify kernel for "
            "`D + front` steps of a scratch the target sized to D. Every symptom is silent -- the "
            "target verifies the tokens either way -- so this is refused rather than measured."
        )
    if getattr(sa, "enable_unified_memory", False):
        raise NotImplementedError(
            "solar_open3 MTP: the linear head does not support "
            "--enable-unified-memory yet. The head needs a private working-state pool "
            "with an extra sink slot and matching physical slot ids; a unified-pool clone "
            "has fixed capacity, translates indices, and shares its backing buffer. "
            "Serve without --enable-unified-memory."
        )
    if serves_linear_draft(config):
        if getattr(sa, "enable_linear_replayssm_spec", False):
            raise NotImplementedError(
                "solar_open3 MTP: the linear head with shared state does not support "
                "--enable-linear-replayssm-spec. That mode removes the per-step SSM scratch "
                "the head needs and commits the target through a different all-layer path. "
                "Serve without --enable-linear-replayssm-spec."
            )
        target_spec = get_linear_attn_spec_by_arch("SolarOpen3ForCausalLM")
        hybrid = getattr(target_spec, "hybrid_backend_class_name", None)
        if not hybrid or not hybrid.endswith(
            ".linear_draft.SolarOpen3HybridAttnBackend"
        ):
            raise NotImplementedError(
                "solar_open3 MTP: shared draft state requires SolarOpen3HybridAttnBackend "
                "on the registered target spec, so target verification cannot overwrite the "
                "head's committed state. Check the native SolarOpen3 target registration."
            )
    if getattr(sa, "disable_radix_cache", False):
        return
    if not serves_linear_draft(config):
        if linear_draft_share_pool_opt_out():
            logger.warning(
                "[solar_open3] SO3_LINEAR_DRAFT_SHARE_POOL=0 with the radix cache ON: the MTP "
                "head's state is kept in a private pool the cache does not carry, so after a "
                "prefix hit the head drafts from another request's state. Output stays correct "
                "(the target verifies); ACCEPTANCE AFTER A HIT IS NOT REPRESENTATIVE. This is the "
                "control arm, not a serving configuration."
            )
            return
        raise NotImplementedError(
            "solar_open3 MTP: the linear head with the radix cache on needs its state "
            "layer reserved in the target's mamba pool, and this server did not reserve one. "
            "Serve with --disable-radix-cache, or use the native registration where "
            "`serves_linear_draft` holds (EAGLE + a linear head + SO3_LINEAR_DRAFT_SHARE_POOL "
            "unset). Serving now would draft from a stale state after every prefix hit, with no "
            "error to show for it."
        )
    strategy = getattr(sa, "mamba_radix_cache_strategy", "auto")
    if strategy == "no_buffer":
        raise NotImplementedError(
            "solar_open3 MTP: the linear head needs --mamba-radix-cache-strategy "
            "extra_buffer (or auto, which resolves to it for this architecture). The `no_buffer` "
            "strategy has no interval-aligned snapshot, so a prefix boundary inside a GENERATED "
            "region would restore a head state that was never written -- the same silent "
            "acceptance loss the trunk's own KDA layers hit under that strategy."
        )


def _without_attn_res(config):
    """A shallow copy with AttnRes off, for the head only."""
    import copy

    out = copy.copy(config)
    out.attn_res_block_size = None
    return out


EntryClass = [SolarOpen3ForCausalLMNextN]
