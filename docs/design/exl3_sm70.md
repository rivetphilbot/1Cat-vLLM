# EXL3 routed experts on SM70 (V100) — design notes for an upstream contribution (draft, 2026-10-05)

Status: this branch. Developed out of tree on 1Cat main @ d30469863 and serving GLM-5.3-Flash TP4 on four V100-32GB there;
rebased onto current main and NOT yet re-qualified on the rebased tree.
Not yet submitted. AI assistance was used for all of it; every line needs human review before a PR (AGENTS.md).

## What it adds
A quantization method `exl3` for exact SM70 that serves checkpoints whose routed MoE experts are EXL3
(trellis-coded, exllamav3 / QTIP-derived) and whose other tensors are plain FP16/BF16:

| File | Role |
|---|---|
| `vllm/model_executor/layers/quantization/exl3.py` | `Exl3Config` (min capability 70), `Exl3SM70MoEMethod(FusedMoEMethodBase)` |
| `vllm/model_executor/layers/quantization/exl3_sm70_ops.py` | `Exl3ExpertBank`: per-layer tensors, scratch, launch chain (ctypes today) |
| `csrc/quantization/exl3_sm70/exl3_sm70.cu` -> `libexl3_sm70.so` (`tools/exl3_sm70/build_exl3_sm70.sh`) | kernels: trellis decode + `mma.m8n8k4` GEMV (K = 2/3/4), FWHT-128 glue, grouped prefill |
| `tools/exl3_sm70/repack_1cat.py` (+ `exl3_ref.py`, reference decoder) | exllamav3 conversion output -> checkpoint in the layout below |

Only the routed experts take the new path. Everything else in GLM-5.3 (KDA, sparse MLA, mHC, shared expert, dense
MLP, vision) runs the existing qualified SM70 routes, because the NVFP4 checkpoint 1Cat qualified also leaves those
tensors BF16.

## Format
EXL3 per tensor: `trellis` int16 `[k/16, n/16, 16K]`, `suh [k]`, `svh [n]` fp16, mcg codebook.
`W = diag(suh) · H · T · H · diag(svh)`, `H` = blockwise Sylvester-128 / sqrt(128), `T` = decoded tiles.
Decode of position p in a 16x16 tile: 16-bit state ending at bit (p+1)K of the tile's 256K-bit MSB-first stream
(tail-biting); value = fp16(lo + hi) of `(state * 0xCBAC1FED & 0x8fff8fff) ^ 0x3b603b60`.

**Tile order.** Standard EXL3 stores positions in the sm80 `m16n8k16` B-fragment order. This work uses a checkpoint
written in *column-major* order (`quantization_config.tile_order = "sm70_colmajor"`, position p -> row p % 16,
column p / 16): a lane's 8 weights for `m8n8k4` are then 8 consecutive positions, extracted from two aligned u32
words with shifts - no lane exchange. Measured quality-neutral at quantization time (-0.02% proxy error).
exllamav3 needs a one-line switch in `tensor_core_perm` to write it (and a tile-order-aware forward to calibrate
on it - found the hard way). Standard-order checkpoints are rejected by `Exl3Config` today.

**1Cat checkpoint layout** (per MoE layer, experts stacked):
`...mlp.experts.exl3_{gate,up}_trellis [E, H/16, I/16, 16K]`, `_suh [E, H]`, `_svh [E, I]`;
`...mlp.experts.exl3_down_trellis [E, I/16, H/16, 16K]`, `_suh [E, I]`, `_svh [E, H]`.
`quantization_config = {quant_method: exl3, bits, layer_bits: {layer: K}, codebook: mcg, tile_order}`.
TP slices the intermediate dimension (tile columns of gate/up, tile rows of down, and the matching sign vectors);
512-wide slices keep the 128-point Hadamards rank-local, so no cross-rank transform is needed.

## Kernels (one slot = one (token, expert) assignment)
1. `pre2`: `X' = FWHT128(x[token] * suh[e])` for gate and up in one launch (fp16 out).
2. `gemv2`: `Y' = X' · T[e]` for gate and up in one launch: per block a 32-column strip, 4 warps over the k tiles,
   decode 8 weights per lane per k-step, `mma.m8n8k4` with FP32 accumulation, deterministic in-block reduction.
3. `mid`: `H' = FWHT128(silu(FWHT128(Yg')·svh_g) · FWHT128(Yu')·svh_u · suh_d)`.
4. `gemv` (down), 5. `post`: `out[token] = Σ_j w_j · FWHT128(O'_j) · svh_d` (fixed order over the top-k).
Expert ids are read on the device; shapes are static per token count -> CUDA-graph capturable with no host sync.
Prefill (T >= 65, eager): slots are grouped by expert (sort + cummax + cumsum on GPU), each group of <= 8 slots
decodes its expert once and multiplies 8 rows (`gemvg`/`gemvg2`): 6-7x less decode work.
FWHT is register/warp-shuffle based (lane holds 4 elements; 2 in-lane stages + 5 shuffle stages), no shared memory.

## Evidence so far (GLM-5.3-Flash, 2-bit experts, 4x V100-SXM2-32GB TP4)
- Kernel vs pure-torch oracle (independent reference decoder), real tensors: 3.5e-4 relative (fp16 intermediates),
  per-rank slices and their sum, T = 1..3000, K = 2, 3, 4.
- vLLM vs exllamav3 reference of the same checkpoint: next-token argmax identical, top-5 identical, cached decode
  greedy tokens identical on the test prompts; 94% argmax agreement over 250 generated tokens (near-ties).
- 1Cat GLM gate shape (1024 in / 256 out, B1, graphs): 48.2 tok/s steady decode without speculation (greedy hash
  identical across runs); DFlash2: 116 tok/s on that prompt, 66-70 tok/s on realistic chat prompts
  (draft acceptance is lower against a 2-bit target: 0.77 first position).
- Expert path cost: 86 us/layer/rank at T=1 (3.6 ms per token over 42 layers), 6.6-8 us/token/layer in prefill.
- Vision works with `--language-model-only` removed (first SM70 vision result for this model, text-only was qualified).

## Gaps before this is PR-ready
- Native ops should be `torch.ops._C` fragments in `csrc/sm70_turbomind/ops/` (CMake, `register_fake`), not ctypes.
- Policy must move to `kernel_config.sm70_*`; no environment reads.
- Type A / Type B contracts per `sm70_v100_migration_control.md`: the decode itself is exact; FP32 accumulation
  through `m8n8k4` and the fp16 intermediates between stages need stated bounds.
- A TurboMind `Transform` (like #848) for the large-M GEMM would replace the grouped GEMV in prefill.
- Standard sm80-order checkpoints (scattered decode) are unsupported.
- CPU oracle + goldens as a first, standalone PR (the oracle exists: `ref/exl3_ref.py`).
- The memory accounting gap seen on long prompts (~0.8 GiB/GPU outside the profiler) is not EXL3-specific but shows
  up because this checkpoint fills the cards.
