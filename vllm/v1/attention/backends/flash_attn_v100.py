# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash Attention V100 backend for SM70.

Prefill uses the dense Flash V100 kernel for strict no-prefix cases.
Decode uses a paged Flash V100 kernel that reads vLLM's KV cache directly.
"""

from __future__ import annotations

import os
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

logger = init_logger(__name__)

# Lazy imports: only resolve optional CUDA extensions when needed.
_flash_attn_func = None
_flash_attn_decode_paged = None
_flash_attn_prefill_paged = None
_paged_kv_utils = None
_flash_v100_dumped_decode_nan = False
_warned_prefill_fallback = False
_warned_feature_fallback = False
_warned_decode_fallback = False
_logged_prefill_flash = False
_logged_prefill_prefix_flash = False
_logged_prefill_smallq_decode = False
_logged_decode_flash = False
_logged_prefill_compare = False
_warned_paged_prefill_smem = False

# V100 dynamic shared-memory ceiling (bytes).
_FLASH_V100_MAX_SMEM = 98304

# Base (block-table-independent) shared memory of the PAGED prefill kernel,
# per head_dim, mirroring KernelConfig<D>::TOTAL_SMEM in
# flash-attention-v100/kernel/fused_mha_forward_paged.cu. The kernel ALSO
# stores the per-sequence block table in smem (extra = align128(max_num_blocks
# * 4)), so total smem grows with max_model_len / page_block_size. These bases
# are already ~84-96KB, leaving little headroom: long-context servers overflow
# the 96KB ceiling and must fall back to the gather+dense prefill path.
_PAGED_PREFILL_BASE_SMEM = {64: 81408, 128: 97792, 256: 93952, 512: 85888}


def _get_flash_ops():
    """Lazy-load flash_attn_v100 ops if available."""
    global _flash_attn_func, _flash_attn_decode_paged, _flash_attn_prefill_paged
    if (_flash_attn_func is None or _flash_attn_decode_paged is None
            or _flash_attn_prefill_paged is None):
        try:
            from flash_attn_v100 import (flash_attn_decode_paged,
                                         flash_attn_func,
                                         flash_attn_prefill_paged)

            _flash_attn_func = flash_attn_func
            _flash_attn_decode_paged = flash_attn_decode_paged
            _flash_attn_prefill_paged = flash_attn_prefill_paged
        except ImportError:
            _flash_attn_func = None
            _flash_attn_decode_paged = None
            _flash_attn_prefill_paged = None
    return _flash_attn_func, _flash_attn_decode_paged, _flash_attn_prefill_paged


def _get_paged_kv_utils():
    """Lazy-load paged KV extraction CUDA extension."""
    global _paged_kv_utils
    if _paged_kv_utils is None:
        try:
            import paged_kv_utils

            _paged_kv_utils = paged_kv_utils
        except ImportError:
            _paged_kv_utils = None
    return _paged_kv_utils


def _has_prefix_context(attn_metadata: TritonAttentionMetadata) -> bool:
    """Return True if any sequence has KV context before current query tokens."""
    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    if query_start_loc_cpu is not None and seq_lens_cpu is not None:
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        return bool(torch.any(query_lens != seq_lens_cpu).item())

    query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
    return not torch.equal(query_lens, attn_metadata.seq_lens)


def _extract_contiguous_kv_from_paged_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    total_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract contiguous K/V from paged KV cache.

    Uses the CUDA extension when available and falls back to a Python path.
    """

    paged_kv_utils = _get_paged_kv_utils()

    if isinstance(kv_cache, (list, tuple)):
        key_cache, value_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        elif kv_cache.shape[1] == 2:
            key_cache, value_cache = kv_cache.unbind(1)
        else:
            raise ValueError(
                f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
                "expected dimension 2 at axis 0 or 1"
            )
    # unbind() of the interleaved K/V cache is non-contiguous; the gather
    # kernel mis-reads the stride. Force contiguous before extraction.
    if not isinstance(kv_cache, (list, tuple)):
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()

    # DIAGNOSTIC: force the pure-PyTorch gather (correct by construction) to
    # test whether the CUDA paged_kv_to_contiguous kernel is producing wrong KV.
    if os.getenv("VLLM_FLASH_V100_FORCE_PY_GATHER", "0") == "1":
        paged_kv_utils = None

    # fp8 (uint8) caches: the CUDA gather kernel is typed for fp16, but a
    # gather is a bitwise copy — view 2 uint8 bytes as one fake half, run the
    # fast kernel, view the result back. Avoids the very slow per-block Python
    # loop (the dominant chunked-prefill cost for fp8-KV models like deckard).
    if (paged_kv_utils is not None and key_cache.dtype == torch.uint8
            and head_dim % 2 == 0
            and os.getenv("VLLM_FLASH_V100_NO_U8_GATHER", "0") != "1"
            and hasattr(paged_kv_utils, "paged_kv_to_contiguous")):
        kc_h = key_cache.view(torch.float16)    # [..., head_dim//2] halfs
        vc_h = value_cache.view(torch.float16)
        k_h, v_h = paged_kv_utils.paged_kv_to_contiguous(
            kc_h, vc_h, block_table, seq_lens)
        if total_tokens is None:
            total_tokens = int(seq_lens.sum().item())
        k_cont = k_h.view(torch.uint8).view(-1, num_kv_heads, head_dim)
        v_cont = v_h.view(torch.uint8).view(-1, num_kv_heads, head_dim)
        return k_cont[:total_tokens], v_cont[:total_tokens]

    if paged_kv_utils is not None and key_cache.dtype != torch.uint8:
        if hasattr(paged_kv_utils, "paged_kv_to_contiguous"):
            k_cont, v_cont = paged_kv_utils.paged_kv_to_contiguous(
                key_cache, value_cache, block_table, seq_lens)
        else:
            k_cont = paged_kv_utils.paged_to_contiguous(key_cache, block_table,
                                                        seq_lens)
            v_cont = paged_kv_utils.paged_to_contiguous(value_cache, block_table,
                                                        seq_lens)
        if total_tokens is None:
            total_tokens = int(seq_lens.sum().item())
        if os.getenv("VLLM_FLASH_V100_GATHER_COMPARE", "0") == "1":
            import sys
            bs = block_table.shape[0]
            tt = total_tokens
            kp = torch.empty((tt, num_kv_heads, head_dim),
                             dtype=key_cache.dtype, device=key_cache.device)
            off = 0
            for bi in range(bs):
                sl_ = int(seq_lens[bi].item())
                nb = (sl_ + block_size - 1) // block_size
                for blk in range(nb):
                    pb = int(block_table[bi, blk].item())
                    st = blk * block_size
                    en = min(st + block_size, sl_)
                    n = en - st
                    if off + n > tt:
                        n = tt - off
                    if n <= 0:
                        break
                    kp[off:off + n] = key_cache[pb, :n]
                    off += n
            kc_cuda = k_cont[:tt]
            diff = (kc_cuda.float() - kp.float()).abs()
            perpos = diff.reshape(tt, -1).max(dim=1).values
            nbad = int((perpos > 1e-3).sum())
            bad_first = (perpos > 1e-3).nonzero().flatten()[:5].tolist()
            print(f"GATHER_COMPARE tt={tt} seqlens={seq_lens[:3].tolist()} "
                  f"maxdiff={float(diff.max()):.4f} nbad={nbad}/{tt} "
                  f"firstbad={bad_first} cuda_nan={int(torch.isnan(kc_cuda).sum())}",
                  file=sys.stderr, flush=True)
        return k_cont[:total_tokens], v_cont[:total_tokens]

    # Slow Python fallback.
    batch_size = block_table.shape[0]
    if total_tokens is None:
        total_tokens = int(seq_lens.sum().item())

    k_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=key_cache.dtype,
        device=key_cache.device,
    )
    v_cont = torch.empty(
        (total_tokens, num_kv_heads, head_dim),
        dtype=value_cache.dtype,
        device=value_cache.device,
    )

    token_offset = 0
    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        if seq_len == 0:
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        for block_idx in range(num_blocks):
            physical_block_idx = int(block_table[batch_idx, block_idx].item())
            start_token = block_idx * block_size
            end_token = min(start_token + block_size, seq_len)
            n = end_token - start_token

            k_cont[token_offset:token_offset + n] = key_cache[physical_block_idx, :n]
            v_cont[token_offset:token_offset + n] = value_cache[physical_block_idx, :n]
            token_offset += n

    return k_cont, v_cont


def _fp8_dtype_from_cache_dtype(kv_cache_dtype: str) -> torch.dtype:
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FLASH_ATTN_V100 fp8 dtype: {kv_cache_dtype}")


def _dequantize_fp8_contiguous_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not kv_cache_dtype.startswith("fp8"):
        return key, value
    fp8_dtype = _fp8_dtype_from_cache_dtype(kv_cache_dtype)
    key = key.view(fp8_dtype).to(torch.float16) * k_scale
    value = value.view(fp8_dtype).to(torch.float16) * v_scale
    return key, value


class FlashAttnV100MetadataBuilder(TritonAttentionMetadataBuilder):
    """Attach CPU metadata for the dense prefill path."""

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        attn_metadata.max_model_len = self.vllm_config.model_config.max_model_len

        # The Triton builder shortens capture seq_lens to 1 so full graph
        # capture stays cheap. That is valid for single-token decode, but the
        # FA2 small-query MTP verifier replays a tiny causal prefill as paged
        # decode. Capturing that branch with seq_len < query_len creates
        # negative per-token decode lengths and can poison long-context graph
        # replay. Keep capture cheap while preserving a valid verifier shape.
        max_query_len = getattr(attn_metadata, "max_query_len", 1)
        if max_query_len > 1:
            attn_metadata.seq_lens.fill_(max_query_len)

        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        attn_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        attn_metadata.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        attn_metadata.causal = common_attn_metadata.causal
        attn_metadata.max_model_len = self.vllm_config.model_config.max_model_len
        return attn_metadata


class FlashAttnV100Impl(TritonAttentionImpl):
    """Flash Attention V100 implementation with strict fallback policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.flash_attn_func, self.flash_attn_decode_paged,
         self.flash_attn_prefill_paged) = _get_flash_ops()
        # V100 FA2 kernels consume fp16 Q. FP8 KV cache support is implemented
        # as storage compression only, with K/V dequantized inside FA2 kernels.
        self.supports_quant_query_input = False
        self.use_flash_v100 = self.flash_attn_func is not None
        self.use_flash_v100_decode = self.flash_attn_decode_paged is not None
        paged_prefill_enable = os.getenv("VLLM_FLASH_V100_ENABLE_PAGED_PREFILL")
        paged_prefill_disable = (
            os.getenv("VLLM_FLASH_V100_DISABLE_PAGED_PREFILL", "0") == "1")
        self.use_flash_v100_prefill_paged = (
            self.flash_attn_prefill_paged is not None
            and paged_prefill_enable != "0"
            and not paged_prefill_disable)
        self.smallq_decode_max_query_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q", "16"))
        self.smallq_decode_max_model_len = int(
            os.getenv("VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN",
                      "0"))
        self._decode_cache_k: torch.Tensor | None = None
        self._decode_cache_v: torch.Tensor | None = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

    def _reset_decode_cache(self) -> None:
        self._decode_cache_k = None
        self._decode_cache_v = None
        self._decode_cache_len = 0
        self._decode_cache_capacity = 0

    def _ensure_decode_cache_capacity(
        self,
        required_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_capacity >= required_len
            and self._decode_cache_k.shape[1] == num_kv_heads
            and self._decode_cache_k.shape[2] == head_dim
            and self._decode_cache_k.dtype == dtype
            and self._decode_cache_k.device == device
        ):
            return

        new_capacity = max(required_len, max(16, self._decode_cache_capacity * 2))
        new_k = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        new_v = torch.empty(
            (new_capacity, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        if (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and self._decode_cache_len > 0
        ):
            new_k[:self._decode_cache_len].copy_(
                self._decode_cache_k[:self._decode_cache_len]
            )
            new_v[:self._decode_cache_len].copy_(
                self._decode_cache_v[:self._decode_cache_len]
            )

        self._decode_cache_k = new_k
        self._decode_cache_v = new_v
        self._decode_cache_capacity = new_capacity

    def _get_decode_kv_single_seq(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        block_size: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(seq_lens_cpu[0])
        q_len = int(attn_metadata.num_actual_tokens)
        num_kv_heads = key.shape[1]

        cache_hit = (
            self._decode_cache_k is not None
            and self._decode_cache_v is not None
            and seq_len > self._decode_cache_len
            and seq_len - q_len == self._decode_cache_len
        )

        if not cache_hit:
            k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
                total_tokens=seq_len,
            )
            self._ensure_decode_cache_capacity(
                seq_len,
                num_kv_heads,
                head_dim,
                k_cont.dtype,
                k_cont.device,
            )
            assert self._decode_cache_k is not None
            assert self._decode_cache_v is not None
            self._decode_cache_k[:seq_len].copy_(k_cont)
            self._decode_cache_v[:seq_len].copy_(v_cont)
            self._decode_cache_len = seq_len
            return (
                self._decode_cache_k[:seq_len],
                self._decode_cache_v[:seq_len],
            )

        self._ensure_decode_cache_capacity(
            seq_len,
            num_kv_heads,
            head_dim,
            key.dtype,
            key.device,
        )
        assert self._decode_cache_k is not None
        assert self._decode_cache_v is not None
        self._decode_cache_k[self._decode_cache_len:seq_len].copy_(key[:q_len])
        self._decode_cache_v[self._decode_cache_len:seq_len].copy_(value[:q_len])
        self._decode_cache_len = seq_len
        return (
            self._decode_cache_k[:seq_len],
            self._decode_cache_v[:seq_len],
        )

    def _supports_flash_v100_path(self) -> bool:
        """Check whether current layer/config can run Flash V100 safely."""
        supported_kv_dtype = (
            not self.kv_cache_dtype.startswith("fp8")
            or self.kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2")
        )
        # Causal sliding-window (left>=0, right==0) is supported by the kernels
        # via the `window` param. Full attention is (-1, -1). Bidirectional
        # (right != 0) windows are not supported.
        supported_window = (
            self.sliding_window == (-1, -1)
            or self.sliding_window[1] == 0
        )
        return (
            self.use_flash_v100
            and self.attn_type == AttentionType.DECODER
            and self.alibi_slopes is None
            and self.logits_soft_cap == 0
            and self.sinks is None
            and supported_window
            and supported_kv_dtype
        )

    @property
    def _flash_window(self) -> int:
        """Number of attended tokens for the kernels (-1 = unlimited).

        self.sliding_window is (left, right) with left == sliding_window - 1,
        so the attended-token count is left + 1 == the model's sliding_window.
        """
        if self.sliding_window == (-1, -1):
            return -1
        return self.sliding_window[0] + 1

    def _small_query_decode_enabled(
        self,
        attn_metadata: TritonAttentionMetadata,
    ) -> bool:
        if (not getattr(attn_metadata, "causal", True)
                or not self.use_flash_v100_decode
                or self.smallq_decode_max_query_len <= 0):
            return False

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu",
                                      None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        if len(query_start_loc) <= 1:
            return False

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.max().item())
        max_model_len = getattr(attn_metadata, "max_model_len", 0)
        model_len_supported = (
            self.smallq_decode_max_model_len <= 0
            or max_model_len <= self.smallq_decode_max_model_len
        )
        return (max_query_len <= self.smallq_decode_max_query_len
                and model_len_supported)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward path.

        - Prefill: use dense Flash V100 only when there is no prefix context.
        - Decode: use paged Flash V100 when available, otherwise fall back.
        """
        global _logged_decode_flash, _logged_prefill_flash
        global _logged_prefill_prefix_flash
        global _warned_decode_fallback, _warned_prefill_fallback
        global _warned_feature_fallback

        if attn_metadata is None:
            assert output is not None
            return output.fill_(0)

        if not self._supports_flash_v100_path():
            if self.use_flash_v100 and not _warned_feature_fallback:
                logger.warning(
                    "FLASH_ATTN_V100 fallback to Triton due to unsupported "
                    "attention features or KV cache dtype."
                )
                _warned_feature_fallback = True
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        is_prefill = attn_metadata.max_query_len > 1
        is_capturing = query.is_cuda and torch.cuda.is_current_stream_capturing()
        # FA kernels now handle head_dim up to 512 for both decode and prefill
        # (split via small blocks to fit smem). 512 prefill stays on FA.
        flash_prefill_ok = self.head_size <= 512

        if is_prefill:
            if is_capturing:
                # CUDA graph capture uses dummy metadata whose seq_lens can
                # look like no-prefix prefill, while replayed MTP verification
                # is a uniform small-query decode over an existing KV prefix.
                # Capture the same small-query kernel branch that replay needs.
                smallq_decode = self._small_query_decode_enabled(attn_metadata)
                if smallq_decode:
                    return self._flash_v100_prefill_with_prefix(
                        layer,
                        query,
                        kv_cache,
                        attn_metadata,
                        output,
                    )
                if not _warned_prefill_fallback:
                    logger.warning(
                        "FLASH_ATTN_V100 prefill fallback during CUDA graph "
                        "capture. Using Triton path for capture safety."
                    )
                    _warned_prefill_fallback = True
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            has_prefix_context = _has_prefix_context(attn_metadata)
            smallq_decode = (
                has_prefix_context
                and self._small_query_decode_enabled(attn_metadata)
            )
            if has_prefix_context and not (smallq_decode or flash_prefill_ok):
                # 512-dim prefix/chunked prefill can't use the 256-cap prefill
                # kernel and isn't small-query; fall back to Triton prefill.
                return super().forward(
                    layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale, output_block_scale,
                )
            if has_prefix_context:
                if not _logged_prefill_prefix_flash:
                    if smallq_decode:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via small-query paged decode)."
                        )
                    elif self.use_flash_v100_prefill_paged:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via direct paged prefill kernel)."
                        )
                    else:
                        logger.info(
                            "FLASH_ATTN_V100 prefill path active "
                            "(prefix/chunked via paged-KV gather)."
                        )
                    _logged_prefill_prefix_flash = True
                self._reset_decode_cache()
                if os.getenv("VLLM_FLASH_V100_TRACE", "0") == "1":
                    _ln = getattr(layer, "layer_name", "?")
                    _qn = int(torch.isnan(query).sum().item())
                    _kn = int(torch.isnan(key).sum().item()) if key is not None else -1
                    _vn = int(torch.isnan(value).sum().item()) if value is not None else -1
                    _out = self._flash_v100_prefill_with_prefix(
                        layer, query, kv_cache, attn_metadata, output)
                    _on = int(torch.isnan(_out).sum().item())
                    if _qn or _kn > 0 or _vn > 0 or _on:
                        logger.warning(
                            "TRACE prefix-prefill layer=%s win=%s qnan=%d knan=%d "
                            "vnan=%d outnan=%d qshape=%s",
                            _ln, str(self._flash_window), _qn, _kn, _vn, _on,
                            tuple(query.shape),
                        )
                    return _out
                return self._flash_v100_prefill_with_prefix(
                    layer,
                    query,
                    kv_cache,
                    attn_metadata,
                    output,
                )
            if getattr(self, "kv_sharing_target_layer_name", None) is not None:
                # KV-shared layer (gemma E2B/E4B): it applies RoPE to Q only and
                # passes raw, un-normed/un-RoPE'd K/V; the real K/V live in the
                # TARGET layer's cache, aliased into kv_cache and already written
                # by that earlier layer this forward pass. Read them via the
                # prefix path (paged kernel when smem-safe, else gather+dense)
                # instead of the dense passed-K/V path, which would attend to
                # junk. (Decode already reads kv_cache directly.)
                self._reset_decode_cache()
                return self._flash_v100_prefill_with_prefix(
                    layer, query, kv_cache, attn_metadata, output
                )
            if not flash_prefill_ok:
                # 512-dim no-prefix prefill: dense kernel caps at 256 -> Triton.
                return super().forward(
                    layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale, output_block_scale,
                )
            if not _logged_prefill_flash:
                logger.info(
                    "FLASH_ATTN_V100 prefill path active (no prefix/chunked context)."
                )
                _logged_prefill_flash = True
            self._reset_decode_cache()
            return self._flash_v100_prefill(query, key, value, attn_metadata, output)

        if not self.use_flash_v100_decode:
            if self.use_flash_v100 and not _warned_decode_fallback:
                logger.warning(
                    "FLASH_ATTN_V100 decode fallback to Triton: paged decode op "
                    "is unavailable."
                )
                _warned_decode_fallback = True
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if not _logged_decode_flash:
            logger.info(
                "FLASH_ATTN_V100 decode path active (paged KV kernel, CUDA-graph safe)."
            )
            _logged_decode_flash = True
        return self._flash_v100_decode(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )

    def _flash_v100_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for no-prefix case (query_len == seq_len per sequence)."""
        causal = getattr(attn_metadata, "causal", True)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu if query_start_loc_cpu is not None else attn_metadata.query_start_loc
        )
        num_seqs = len(query_start_loc) - 1

        if num_seqs == 0:
            return output

        seq_lens = query_start_loc[1:] - query_start_loc[:-1]
        run_start = 0
        while run_start < num_seqs:
            run_seq_len = int(seq_lens[run_start].item())
            run_end = run_start + 1
            while (
                run_end < num_seqs
                and int(seq_lens[run_end].item()) == run_seq_len
            ):
                run_end += 1

            if run_seq_len > 0:
                tok_start = int(query_start_loc[run_start].item())
                tok_end = int(query_start_loc[run_end].item())
                batch_size = run_end - run_start

                q_batch = query[tok_start:tok_end].view(
                    batch_size, run_seq_len, query.shape[1], query.shape[2]
                )
                k_batch = key[tok_start:tok_end].view(
                    batch_size, run_seq_len, key.shape[1], key.shape[2]
                )
                v_batch = value[tok_start:tok_end].view(
                    batch_size, run_seq_len, value.shape[1], value.shape[2]
                )

                out_batch = self.flash_attn_func(
                    q_batch,
                    k_batch,
                    v_batch,
                    causal=causal,
                    softmax_scale=self.scale,
                    window_size=self.sliding_window,
                )
                out_view[tok_start:tok_end].copy_(
                    out_batch.view(tok_end - tok_start, out_batch.shape[2], out_batch.shape[3])
                )

            run_start = run_end

        return output

    def _flash_v100_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Decode path using Flash V100 directly over paged KV cache."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        if query.shape[0] == 0:
            return output

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)

        self.flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            softmax_scale=self.scale,
            out=out_view,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            window=self._flash_window,
        )
        if os.getenv("VLLM_FLASH_V100_NAN_DEBUG", "0") == "1":
            import sys
            qn = int(torch.isnan(query).sum())
            kn = int(torch.isnan(key_cache).sum())
            on = int(torch.isnan(out_view).sum())
            sl = attn_metadata.seq_lens
            print(f"DECODE qnan={qn} kcache_nan={kn} out_nan={on} "
                  f"seq_lens={sl[:3].tolist()} win={self._flash_window} "
                  f"qshape={list(query.shape)}", file=sys.stderr, flush=True)
            global _flash_v100_dumped_decode_nan
            if (on > 0 and qn == 0 and kn == 0
                    and not _flash_v100_dumped_decode_nan
                    and os.getenv("VLLM_FLASH_V100_DUMP_DECODE_NAN", "0") == "1"):
                dp = f"/home/jarvis/builds/decode_nan_dump.pt"
                torch.save({
                    "query": query.detach().cpu(),
                    "key_cache": key_cache.detach().cpu(),
                    "value_cache": value_cache.detach().cpu(),
                    "block_table": attn_metadata.block_table.detach().cpu(),
                    "seq_lens": attn_metadata.seq_lens.detach().cpu(),
                    "scale": float(self.scale),
                    "window": int(self._flash_window),
                    "k_scale": float(layer._k_scale_float),
                    "v_scale": float(layer._v_scale_float),
                    "kv_cache_dtype": self.kv_cache_dtype,
                    "out_nan": on,
                }, dp)
                print(f"DECODE_NAN_DUMP saved {dp}", file=sys.stderr, flush=True)
                _flash_v100_dumped_decode_nan = True
        return output

    def _flash_v100_small_query_prefill_as_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        query_start_loc: torch.Tensor,
        _seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Run small causal prefix-prefill queries through paged decode.

        MTP verification presents a tiny query span over a long KV prefix. The
        paged prefill kernel is correct, but its work scheduling is much more
        expensive for this shape and exceeds SM70 shared-memory limits at very
        long contexts. Treating every query token as an independent decode row
        with an increasing seq_len preserves the causal mask without exposing
        future draft tokens.
        """
        device = attn_metadata.seq_lens.device
        dtype = attn_metadata.seq_lens.dtype

        num_query_tokens = attn_metadata.num_actual_tokens
        query_start_loc_gpu = attn_metadata.query_start_loc
        query = query[:num_query_tokens]
        out_view = output[:num_query_tokens]
        query_lens_gpu = query_start_loc_gpu[1:] - query_start_loc_gpu[:-1]
        real_query_lens_gpu = query_lens_gpu
        real_num_query_tokens = query_start_loc_gpu[-1]
        num_seqs = query_lens_gpu.numel()
        if num_seqs > 0:
            # FULL CUDA graph replay may pad a 3-request MTP verifier batch
            # from 15 tokens to 20 tokens while query_start_loc still marks
            # only the 15 live tokens. Give the padded tail a dummy query span
            # so repeat_interleave keeps the captured graph shape. The padded
            # rows are masked below and must not read real KV cache entries.
            padding_tokens = torch.clamp(
                num_query_tokens - real_num_query_tokens,
                min=0,
            )
            query_lens_gpu = query_lens_gpu.clone()
            query_lens_gpu[-1] += padding_tokens

        seq_lens = attn_metadata.seq_lens[:num_seqs]
        effective_seq_lens = torch.maximum(
            seq_lens,
            real_query_lens_gpu.to(dtype=attn_metadata.seq_lens.dtype),
        )
        block_table = attn_metadata.block_table[:num_seqs].clamp_min(0)
        decode_block_table = torch.repeat_interleave(
            block_table,
            query_lens_gpu,
            dim=0,
            output_size=num_query_tokens,
        ).contiguous()
        seq_lens_rep = torch.repeat_interleave(
            effective_seq_lens,
            query_lens_gpu,
            output_size=num_query_tokens,
        )
        query_lens_rep = torch.repeat_interleave(
            real_query_lens_gpu.to(dtype=dtype),
            query_lens_gpu,
            output_size=num_query_tokens,
        )
        start_locs_rep = torch.repeat_interleave(
            query_start_loc_gpu[:-1].to(dtype=dtype),
            query_lens_gpu,
            output_size=num_query_tokens,
        )
        token_indices = torch.arange(
            num_query_tokens,
            device=device,
            dtype=dtype,
        )
        offsets = token_indices - start_locs_rep + 1
        decode_seq_lens = (seq_lens_rep - query_lens_rep + offsets).contiguous()
        padding_mask = token_indices >= real_num_query_tokens
        decode_seq_lens = torch.where(
            padding_mask,
            torch.zeros_like(decode_seq_lens),
            decode_seq_lens,
        ).contiguous()
        decode_block_table = torch.where(
            padding_mask[:, None],
            torch.zeros_like(decode_block_table),
            decode_block_table,
        ).contiguous()
        self.flash_attn_decode_paged(
            query,
            key_cache,
            value_cache,
            decode_block_table,
            decode_seq_lens,
            softmax_scale=self.scale,
            out=out_view,
            kv_cache_dtype=self.kv_cache_dtype,
            k_scale=float(layer._k_scale_float),
            v_scale=float(layer._v_scale_float),
            window=self._flash_window,
        )
        return output

    def _paged_prefill_smem_fits(
        self,
        attn_metadata: TritonAttentionMetadata,
    ) -> bool:
        """Whether the paged prefill kernel's smem fits V100's 96KB ceiling.

        The kernel copies the per-sequence block table into shared memory, so
        smem = TOTAL_SMEM[head_dim] + align128(max_num_blocks * 4). For long
        max_model_len the block table alone blows the budget (e.g. head_dim 256
        at 177k ctx needs ~135KB). When it does not fit, the caller uses the
        gather + dense prefill path, which is smem-safe at any context length.
        """
        base = _PAGED_PREFILL_BASE_SMEM.get(self.head_size)
        if base is None:
            return False
        block_table = getattr(attn_metadata, "block_table", None)
        if block_table is None or block_table.ndim < 2:
            return False
        max_num_blocks = int(block_table.shape[1])
        extra = (max_num_blocks * 4 + 127) & ~127
        return base + extra <= _FLASH_V100_MAX_SMEM

    def _py_dense_prefix_attn(self, q_seq, k_cont, v_cont, k_len, q_len,
                              causal):
        """Pure-PyTorch dense attention for prefix/chunked prefill.

        Bypasses the FA_V100 prefill kernels (which mis-handle this shape on
        SM70). q_seq:(q_len, nq, d); k/v_cont:(k_len, nkv, d). The query is the
        SUFFIX of the sequence: query row i is at absolute position
        (k_len - q_len + i). Builds a causal + sliding-window mask in fp32.
        """
        nq = q_seq.shape[-2]
        nkv = k_cont.shape[-2]
        device = q_seq.device
        q = q_seq.to(torch.float32).transpose(0, 1).unsqueeze(0)   # (1,nq,q_len,d)
        k = k_cont.to(torch.float32).transpose(0, 1).unsqueeze(0)  # (1,nkv,k_len,d)
        v = v_cont.to(torch.float32).transpose(0, 1).unsqueeze(0)
        if nq != nkv and nkv > 0:
            rep = nq // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        qpos = torch.arange(k_len - q_len, k_len, device=device).unsqueeze(1)
        kpos = torch.arange(k_len, device=device).unsqueeze(0)
        allowed = torch.ones((q_len, k_len), dtype=torch.bool, device=device)
        if causal:
            allowed &= (kpos <= qpos)
        win = self.sliding_window
        if (isinstance(win, (tuple, list)) and len(win) >= 1
                and win[0] is not None and win[0] >= 0):
            allowed &= (kpos > qpos - (int(win[0]) + 1))
        bias = torch.zeros((q_len, k_len), dtype=torch.float32, device=device)
        bias.masked_fill_(~allowed, float("-inf"))
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, scale=self.scale)            # (1,nq,q_len,d)
        if os.getenv("VLLM_FLASH_V100_NAN_DEBUG", "0") == "1":
            import sys
            qn = int(torch.isnan(q).sum()); kn = int(torch.isnan(k).sum())
            vn = int(torch.isnan(v).sum()); on = int(torch.isnan(out).sum())
            allmasked = int((~torch.isfinite(bias)).all(dim=-1).sum())
            if qn or kn or vn or on or allmasked:
                print(f"PYDENSE_NAN q={qn} k={kn} v={vn} out={on} "
                      f"fully_masked_rows={allmasked} qlen={q_len} klen={k_len} "
                      f"win={self.sliding_window}", file=sys.stderr, flush=True)
        return out.transpose(1, 2).to(q_seq.dtype)                # (1,q_len,nq,d)

    def _flash_v100_prefill_with_prefix(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill path for prefix/chunked context via gathered contiguous KV."""
        global _logged_prefill_compare, _logged_prefill_smallq_decode
        causal = getattr(attn_metadata, "causal", True)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]

        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        query_start_loc = (
            query_start_loc_cpu
            if query_start_loc_cpu is not None
            else attn_metadata.query_start_loc
        )
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        seq_lens = seq_lens_cpu if seq_lens_cpu is not None else attn_metadata.seq_lens
        num_seqs = len(query_start_loc) - 1

        if kv_cache.shape[0] == 2:
            key_cache, value_cache = kv_cache.unbind(0)
        else:
            key_cache, value_cache = kv_cache.unbind(1)
        # unbind() yields a STRIDED (non-contiguous) view of the interleaved
        # K/V cache (key_stride[0] = 2x contiguous). The paged-prefill kernel
        # mis-reads that stride -> garbage chunked output (decode handles it,
        # prefill does not). Force contiguous so the kernel sees the layout it
        # assumes. Cost is ~hundreds of MB per layer, affordable.
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]
        debug_compare = (os.getenv("VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE", "0")
                         == "1")
        # The paged prefill kernel stores the block table in smem; at long
        # max_model_len it overflows V100's 96KB. When it won't fit, use the
        # gather + dense path below (smem-safe at any context length).
        paged_smem_fits = self._paged_prefill_smem_fits(attn_metadata)
        if not paged_smem_fits:
            global _warned_paged_prefill_smem
            if not _warned_paged_prefill_smem:
                logger.info(
                    "FLASH_ATTN_V100 paged prefill smem exceeds 96KB at this "
                    "max_model_len; using gather+dense prefill (still FA)."
                )
                _warned_paged_prefill_smem = True

        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = int(query_lens.max().item()) if num_seqs > 0 else 0
        if (causal and self.use_flash_v100_decode
                and self.smallq_decode_max_query_len > 0
                and max_query_len <= self.smallq_decode_max_query_len
                and (self.smallq_decode_max_model_len <= 0
                     or getattr(attn_metadata, "max_model_len", 0)
                     <= self.smallq_decode_max_model_len)):
            if not _logged_prefill_smallq_decode:
                logger.info(
                    "FLASH_ATTN_V100 prefix prefill small-query path active "
                    "(paged decode verifier, max_query_len<=%d).",
                    self.smallq_decode_max_query_len,
                )
                _logged_prefill_smallq_decode = True
            return self._flash_v100_small_query_prefill_as_decode(
                layer,
                query,
                key_cache,
                value_cache,
                attn_metadata,
                output,
                query_start_loc,
                seq_lens,
            )

        for i in range(num_seqs):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue

            if self.use_flash_v100_prefill_paged and paged_smem_fits:
                out_seq = self.flash_attn_prefill_paged(
                    query[start:end].unsqueeze(0),
                    key_cache,
                    value_cache,
                    attn_metadata.block_table[i:i + 1],
                    attn_metadata.seq_lens[i:i + 1],
                    softmax_scale=self.scale,
                    kv_cache_dtype=self.kv_cache_dtype,
                    k_scale=float(layer._k_scale_float),
                    v_scale=float(layer._v_scale_float),
                    causal=causal,
                    window=self._flash_window,
                )
                if debug_compare and not _logged_prefill_compare:
                    seq_len = int(seq_lens[i].item())
                    k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                        kv_cache=kv_cache,
                        block_table=attn_metadata.block_table[i:i + 1],
                        seq_lens=attn_metadata.seq_lens[i:i + 1],
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        block_size=block_size,
                        total_tokens=seq_len,
                    )
                    k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                        k_cont,
                        v_cont,
                        self.kv_cache_dtype,
                        float(layer._k_scale_float),
                        float(layer._v_scale_float),
                    )
                    ref_out = self.flash_attn_func(
                        query[start:end].unsqueeze(0),
                        k_cont.unsqueeze(0),
                        v_cont.unsqueeze(0),
                        causal=causal,
                        softmax_scale=self.scale,
                        window_size=self.sliding_window,
                    )
                    diff = (out_seq - ref_out).abs()
                    nan_count = int(torch.isnan(out_seq).sum().item())
                    logger.warning(
                        "FLASH_ATTN_V100 debug prefix compare: "
                        "query_len=%d seq_len=%d max_diff=%.8f mean_diff=%.8f "
                        "nan_count=%d q_absmax=%.6f k_absmax=%.6f v_absmax=%.6f "
                        "kv_cache_shape=%s key_shape=%s key_stride=%s "
                        "value_stride=%s key_contig=%s value_contig=%s",
                        end - start,
                        seq_len,
                        float(diff.max().item()),
                        float(diff.mean().item()),
                        nan_count,
                        float(query[start:end].abs().max().item()),
                        float(k_cont.abs().max().item()),
                        float(v_cont.abs().max().item()),
                        tuple(kv_cache.shape),
                        tuple(key_cache.shape),
                        tuple(key_cache.stride()),
                        tuple(value_cache.stride()),
                        str(key_cache.is_contiguous()),
                        str(value_cache.is_contiguous()),
                    )
                    if nan_count > 0:
                        dump_path = (
                            f"/tmp/flash_v100_prefill_nan_dump_pid{os.getpid()}.pt"
                        )
                        torch.save(
                            {
                                "query": query[start:end].detach().cpu(),
                                "key_cache": key_cache.detach().cpu(),
                                "value_cache": value_cache.detach().cpu(),
                                "block_table": attn_metadata.block_table[
                                    i:i + 1].detach().cpu(),
                                "seq_lens": attn_metadata.seq_lens[
                                    i:i + 1].detach().cpu(),
                                "k_cont": k_cont.detach().cpu(),
                                "v_cont": v_cont.detach().cpu(),
                                "out_seq": out_seq.detach().cpu(),
                                "ref_out": ref_out.detach().cpu(),
                            },
                            dump_path,
                        )
                        logger.warning(
                            "FLASH_ATTN_V100 saved failing prefix prefill dump to %s",
                            dump_path,
                        )
                    _logged_prefill_compare = True
            else:
                seq_len = int(seq_lens[i].item())
                k_cont, v_cont = _extract_contiguous_kv_from_paged_cache(
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table[i:i + 1],
                    seq_lens=attn_metadata.seq_lens[i:i + 1],
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                    total_tokens=seq_len,
                )
                k_cont, v_cont = _dequantize_fp8_contiguous_kv(
                    k_cont,
                    v_cont,
                    self.kv_cache_dtype,
                    float(layer._k_scale_float),
                    float(layer._v_scale_float),
                )

                q_seq = query[start:end]
                q_len = q_seq.shape[0]
                k_len = k_cont.shape[0]
                if os.getenv("VLLM_FLASH_V100_PY_DENSE", "0") == "1":
                    out_seq = self._py_dense_prefix_attn(
                        q_seq, k_cont, v_cont, k_len, q_len, causal)
                elif (causal and 0 < q_len < k_len
                        and os.getenv("VLLM_FLASH_V100_NO_QPAD", "0") != "1"):
                    # flash_attn_func has no query-offset arg; with q_len<k_len
                    # the kernel can't tell the query is the SUFFIX of the
                    # sequence (prefix/chunked prefill), so its causal mask is
                    # misaligned -> wrong output. Pad the query so the real
                    # tokens occupy the LAST q_len rows (bottom-right causal),
                    # then drop the padded prefix from the output.
                    pad = k_len - q_len
                    q_pad = torch.cat(
                        [q_seq.new_zeros((pad,) + tuple(q_seq.shape[1:])), q_seq],
                        dim=0,
                    )
                    out_full = self.flash_attn_func(
                        q_pad.unsqueeze(0),
                        k_cont.unsqueeze(0),
                        v_cont.unsqueeze(0),
                        causal=True,
                        softmax_scale=self.scale,
                        window_size=self.sliding_window,
                    )
                    out_seq = out_full[:, pad:]
                else:
                    out_seq = self.flash_attn_func(
                        q_seq.unsqueeze(0),
                        k_cont.unsqueeze(0),
                        v_cont.unsqueeze(0),
                        causal=causal,
                        softmax_scale=self.scale,
                        window_size=self.sliding_window,
                    )
            out_view[start:end].copy_(out_seq.squeeze(0))

        return output


class FlashAttnV100Backend(TritonAttentionBackend):
    """Flash Attention V100 Backend."""

    # Keep vLLM unified KV cache update path.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_impl_cls():
        return FlashAttnV100Impl

    @staticmethod
    def get_builder_cls():
        return FlashAttnV100MetadataBuilder

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_V100"

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # Decode kernel handles 512 (head-dim-generic GEMV); prefill kernels
        # cap at 256 (smem), so 512 prefill falls back to Triton in forward().
        return [64, 128, 256, 512]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # NOTE(rivet): validate_configuration() calls supports_head_size(),
        # NOT get_supported_head_sizes(). The Volta FA DECODE kernel is
        # head-dim-generic and now handles 512 (gemma-4 global layers); the
        # PREFILL kernels still cap at 256 (512 tile blows 96KB smem), so
        # forward() routes 512-dim prefill to Triton while decode stays on FA.
        return head_size in (64, 128, 256, 512)
