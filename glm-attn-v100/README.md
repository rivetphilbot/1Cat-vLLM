# glm-attn-v100

A fork of `flash-attention-v100` (the Volta attention kernels 1Cat-vLLM ships) for GLM-5.3-Flash sparse MLA.

Why a fork: the 11 softmax-attention layers of GLM-5.3-Flash are DeepSeek-style sparse MLA. Every query attends to
its own top-2048 tokens (DSA indexer), and all query heads share one 512-wide latent per token (absorbed MLA, i.e.
MQA with head dim 512), stored as packed E4M3 with one power-of-two scale per 64 values. No flash-attention-v100
kernel accepts that shape: they take per-head K/V with head dim <= 256 and no per-query index list.

Layout:
- Everything copied from `flash-attention-v100` is byte-identical except package and extension names
  (`flash_attn_v100` -> `glm_attn_v100`, `flash_attn_v100_cuda` -> `glm_attn_v100_cuda`), so upstream V100 fixes
  can be diffed and pulled in: `diff -r flash-attention-v100/kernel glm-attn-v100/kernel`.
- GLM-specific kernels live in `kernel/glm_*.cu`, with Python entry points in `glm_attn_v100/glm_sparse_mla.py`.
