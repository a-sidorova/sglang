# Copyright 2026 Upstage AI.
"""Resolved engine settings consumed by SolarOpen3's startup checks.

The checks also accept small namespace objects in tests. Build that view from
the runtime bags; ServerArgs itself retains the user's unresolved CLI input.
"""

from types import SimpleNamespace

from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_server_args,
    get_spec,
)


def runtime_options():
    # Preserve the normal pre-server ValueError used during config discovery.
    get_server_args()
    spec = get_spec()
    parallel = get_parallel()
    exec_ = get_exec()
    mamba = exec_.mamba
    graph = exec_.graph
    features = exec_.features
    schedule = get_schedule()
    memory = get_memory()
    model = get_model()
    return SimpleNamespace(
        speculative_algorithm=spec.speculative_algorithm,
        speculative_num_draft_tokens=spec.speculative_num_draft_tokens,
        speculative_eagle_topk=spec.speculative_eagle_topk,
        enable_multi_layer_eagle=spec.enable_multi_layer_eagle,
        pp_size=parallel.pp_size,
        enable_dp_lm_head=parallel.enable_dp_lm_head,
        enable_unified_memory=memory.enable_unified_memory,
        disable_radix_cache=memory.disable_radix_cache,
        enable_linear_replayssm_spec=mamba.enable_linear_replayssm_spec,
        mamba_radix_cache_strategy=mamba.mamba_radix_cache_strategy,
        linear_attn_backend=mamba.linear_attn_backend,
        linear_attn_prefill_backend=mamba.linear_attn_prefill_backend,
        linear_attn_decode_backend=mamba.linear_attn_decode_backend,
        linear_attn_verify_backend=mamba.linear_attn_verify_backend,
        context_length=model.context_length,
        load_format=model.load_format,
        page_size=schedule.page_size,
        chunked_prefill_size=schedule.chunked_prefill_size,
        cuda_graph_config=graph.cuda_graph_config,
        disable_decode_cuda_graph=graph.disable_decode_cuda_graph,
        disable_cuda_graph=graph.disable_cuda_graph,
        return_hidden_states_mode=features.return_hidden_states_mode,
        enable_return_hidden_states=features.enable_return_hidden_states,
        enable_lora=get_lora().enable_lora,
        disaggregation_mode=get_disagg().disaggregation_mode,
    )
