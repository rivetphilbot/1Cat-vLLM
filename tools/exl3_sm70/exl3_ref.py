"""Pure-torch reference decoder for EXL3 trellis-quantized linear weights.

Independent of exllamav3 at runtime; it is the oracle the SM70 kernels are tested against.

Format (per linear layer, weight W of shape (in_features k, out_features n), y = x @ W):
  trellis  int16 [k/16, n/16, 16*K]  K bits per weight, one 16x16 tile per (k/16, n/16) cell
  suh      fp16  [k]                 input sign/scale vector
  svh      fp16  [n]                 output sign/scale vector
  mcg/mul1 int32 scalar marker       selects the codebook (absent = 3INST)

Decode:
  1. Each tile is a tail-biting bitstream of 256*K bits. Storage is uint16 words with each
     uint32 pair half-swapped, so read as big-endian uint32 the stream is MSB-first.
  2. Position p (0..255) has state = the 16 stream bits ending at bit (p+1)*K, mod 256*K.
  3. value(p) = codebook(state), an fp16.
  4. Position p holds tile element perm[p], element index = k_local * 16 + n_local. perm is the
     tile order: exllamav3's SM80 tensor-core order by default, or an SM70 order.
  5. W = had_r(had_l(T) * suh[:, None]) * svh[None, :], with blockwise 128-point
     Hadamards (normalized by 1/sqrt(128)) on the input dim (left) and output dim (right).
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

MCG_MULT = 0xCBAC1FED
MUL1_MULT = 0x83DCD12D
HAD_DIM = 128


@lru_cache
def tile_perm_sm80() -> torch.Tensor:
    """exllamav3 tensor_core_perm: lane t's 8 positions -> m16n8k16 B-fragment elements."""
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = t // 4
        for j, c in enumerate((c0, c0 + 8)):
            for i, r in enumerate(rows):
                perm[t * 8 + j * 4 + i] = r * 16 + c
    return torch.tensor(perm, dtype=torch.long)


@lru_cache
def tile_perm_sm70_k8() -> torch.Tensor:
    """Candidate SM70 order: lane t owns 8 consecutive k-rows of one n-column."""
    perm = [((t % 2) * 8 + i) * 16 + (t // 2) for t in range(32) for i in range(8)]
    return torch.tensor(perm, dtype=torch.long)


def unpack_bits(trellis: torch.Tensor, K: int) -> torch.Tensor:
    """int16 [..., 16*K] -> uint8 bitstream [..., 256*K] (MSB-first)."""
    assert trellis.dtype == torch.int16 and trellis.shape[-1] == 16 * K
    w = trellis.to(torch.int32) & 0xFFFF
    w = w.view(*w.shape[:-1], 8 * K, 2).flip(-1).reshape(*w.shape[:-1], 16 * K)  # undo half-swap
    shifts = torch.arange(15, -1, -1, device=w.device, dtype=torch.int32)
    bits = (w.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*w.shape[:-1], 256 * K).to(torch.uint8)


def trellis_states(trellis: torch.Tensor, K: int) -> torch.Tensor:
    """int16 [..., 16*K] -> int64 states [..., 256], tail-biting 16-bit windows."""
    bits = unpack_bits(trellis, K).to(torch.int64)
    L = 256 * K
    ends = (torch.arange(256, device=bits.device) + 1) * K
    idx = (ends.unsqueeze(1) - 16 + torch.arange(16, device=bits.device)) % L  # [256, 16]
    window = bits[..., idx]  # [..., 256, 16]
    weights = 1 << torch.arange(15, -1, -1, device=bits.device, dtype=torch.int64)
    return (window * weights).sum(-1)


def _halves_add(y: torch.Tensor) -> torch.Tensor:
    lo = (y & 0xFFFF).to(torch.int16).view(torch.float16)
    hi = ((y >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
    return (lo.float() + hi.float()).to(torch.float16)


def decode_states(states: torch.Tensor, codebook: str) -> torch.Tensor:
    """int64 16-bit states -> fp16 codebook values (bit-exact with exllamav3 decode_3inst)."""
    x = states.to(torch.int64)
    if codebook == "3inst":
        x = (x * 89226354 + 64248484) & 0xFFFFFFFF
        return _halves_add((x & 0x8FFF8FFF) ^ 0x3B603B60)
    if codebook == "mcg":
        x = (x * MCG_MULT) & 0xFFFFFFFF
        return _halves_add((x & 0x8FFF8FFF) ^ 0x3B603B60)
    if codebook == "mul1":
        x = (x * MUL1_MULT) & 0xFFFFFFFF
        s = 0x6400 + sum((x >> (8 * i)) & 0xFF for i in range(4))  # dp4a byte sum + acc
        h = (s & 0xFFFF).to(torch.int16).view(torch.float16).float()
        k_inv = torch.tensor(0x1EEE, dtype=torch.int16).view(torch.float16).float()
        k_bias = torch.tensor(0xC931 - 0x10000, dtype=torch.int16).view(torch.float16).float()
        return (h * k_inv + k_bias).to(torch.float16)  # hfma: fp32 product+add, one rounding
    raise ValueError(codebook)


def decode_tiles(trellis: torch.Tensor, K: int, codebook: str, perm: torch.Tensor) -> torch.Tensor:
    """int16 [tk, tn, 16*K] -> fp16 inner weight [tk*16, tn*16] (tile basis, before transforms)."""
    tk, tn, _ = trellis.shape
    vals = decode_states(trellis_states(trellis, K), codebook)  # [tk, tn, 256] by position
    tiles = torch.empty_like(vals)
    tiles[..., perm.to(vals.device)] = vals
    return tiles.view(tk, tn, 16, 16).permute(0, 2, 1, 3).reshape(tk * 16, tn * 16)


@lru_cache
def hadamard(n: int = HAD_DIM, device: str = "cpu") -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(torch.float32).to(device)


def reconstruct(trellis, suh, svh, K: int, codebook: str, perm=None) -> torch.Tensor:
    """Full fp32 weight W (in_features, out_features) in the original basis."""
    perm = tile_perm_sm80() if perm is None else perm
    t = decode_tiles(trellis, K, codebook, perm).float()
    k, n = t.shape
    h = hadamard(HAD_DIM, str(t.device))
    t = (h @ t.view(k // HAD_DIM, HAD_DIM, n)).view(k, n)
    t = t * suh.float().unsqueeze(1)
    t = (t.view(k, n // HAD_DIM, HAD_DIM) @ h).view(k, n)
    return t * svh.float().unsqueeze(0)
