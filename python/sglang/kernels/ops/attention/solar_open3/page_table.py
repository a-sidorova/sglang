# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Upstage AI.
"""Build SolarOpen3's full-width decode/verify page table in one launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _page_table_kernel(
    REQ_TO_TOKEN,
    REQ_INDICES,
    OUT,
    stride_req: tl.constexpr,
    stride_token: tl.constexpr,
    stride_indices: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROWS_PER_REQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(REQ_INDICES + row * stride_indices).to(tl.int64)
    # Match PyTorch advanced indexing for valid negative request indices too.
    req = tl.where(req < 0, req + NUM_REQUESTS, req)
    token = tl.load(
        REQ_TO_TOKEN + req * stride_req + col.to(tl.int64) * PAGE_SIZE * stride_token,
        mask=col < NUM_PAGES,
        other=0,
    )
    # Triton integer division truncates; torch // floors even negative padding.
    page = token // PAGE_SIZE
    page -= ((token < 0) & (token % PAGE_SIZE != 0)).to(token.dtype)
    # Load each request once even when verification needs several query rows.
    for repeat in tl.static_range(ROWS_PER_REQ):
        offset = (row.to(tl.int64) * ROWS_PER_REQ + repeat) * NUM_PAGES + col
        tl.store(OUT + offset, page, col < NUM_PAGES)


def solar_open3_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_size: int,
    rows_per_req: int = 1,
) -> torch.Tensor:
    """Fuse ``req_to_token[:, ::page_size][indices] // page_size`` and row repeat.

    Retain the full sliced width, including padding: the routed attention grid
    can address masked columns past a request's current length. Read both inputs
    on every call/replay rather than caching values on a reused ForwardBatch.
    """
    if req_to_token.ndim != 2 or req_pool_indices.ndim != 1:
        raise ValueError("expected a 2D request-to-token map and 1D request indices")
    if not req_to_token.is_cuda or req_pool_indices.device != req_to_token.device:
        raise ValueError("page-table inputs must be on the same CUDA device")
    if req_to_token.dtype not in (
        torch.int32,
        torch.int64,
    ) or req_pool_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("page-table inputs must have int32 or int64 dtype")
    if page_size <= 0 or rows_per_req <= 0:
        raise ValueError("page_size and rows_per_req must be positive")
    num_pages = triton.cdiv(req_to_token.shape[1], page_size)
    rows = req_pool_indices.numel() * rows_per_req
    out = req_to_token.new_empty((rows, num_pages))
    if rows and num_pages:
        block = 256
        _page_table_kernel[(triton.cdiv(num_pages, block), req_pool_indices.numel())](
            req_to_token,
            req_pool_indices,
            out,
            req_to_token.stride(0),
            req_to_token.stride(1),
            req_pool_indices.stride(0),
            NUM_REQUESTS=req_to_token.shape[0],
            NUM_PAGES=num_pages,
            PAGE_SIZE=page_size,
            ROWS_PER_REQ=rows_per_req,
            BLOCK=block,
            num_warps=4,
        )
    return out
