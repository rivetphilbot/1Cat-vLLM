# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash sparse MLA on SM70 over the packed E4M3 latent cache (see kernel/glm_sparse_mla.cu)."""

import torch

try:
    from . import glm_sparse_mla_cuda as _C
except ImportError:  # standalone / JIT builds
    import glm_sparse_mla_cuda as _C

LATENT_DIM = 512
SLOT_BYTES = 520
HEADS_PER_CTA = 16
KEYS_PER_TILE = 32


def default_num_splits(index_width: int) -> int:
    """One CTA per 128 selected keys (16 CTAs for the 2048-key DSA top-k)."""
    return max(1, min(64, -(-index_width // 128)))


def alloc_workspace(num_tokens: int, num_heads: int, num_splits: int, device) -> tuple:
    o_part = torch.empty(
        (num_tokens, num_heads, num_splits, LATENT_DIM), dtype=torch.float32, device=device
    )
    ml = torch.empty((num_tokens, num_heads, num_splits, 2), dtype=torch.float32, device=device)
    return o_part, ml


def sparse_mla_fp8(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    out: torch.Tensor | None = None,
    num_splits: int | None = None,
    workspace: tuple | None = None,
) -> torch.Tensor:
    """q [T, H, 512] fp16; cache uint8 [blocks, block_size, 520]; indices int32 [T, W] global slots;
    lengths int32 [T]. Returns out [T, H, 512] fp16. Pass `out` and `workspace` for CUDA graphs."""
    if num_splits is None:
        num_splits = default_num_splits(indices.shape[1])
    if out is None:
        out = torch.empty_like(q)
    if workspace is None:
        workspace = alloc_workspace(q.shape[0], q.shape[1], num_splits, q.device)
    o_part, ml = workspace
    if q.stride(-1) != 1 or q.stride(1) % 8 or q.stride(0) % 8 or q.data_ptr() % 16:
        q = q.contiguous()  # the kernel loads 16-byte query vectors
    _C.sparse_mla_fp8_fwd(
        q, cache, indices, lengths.reshape(-1), float(scale), out, o_part, ml, num_splits
    )
    return out
