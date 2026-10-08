"""Old GLM FP8 GEMM routes (1Cat Triton + cuBLAS) vs glm_attn_v100 sparse MLA on one V100. Run inside a 1Cat tree:
    PYTHONPATH=<1cat tree with glm_attn_v100 built> python glm-attn-v100/tests/bench_vs_old_route.py"""
import torch, sys
from vllm.models.glm5next.sm70.fp8_kv import (
    sm70_glm5_fp8_kv_insert, sm70_glm5_sparse_attention_paged_fp8_gemm,
    sm70_glm5_sparse_attention_paged_fp8_batched_gemm, GLM5_FP8_KV_SLOT_BYTES)
from glm_attn_v100 import glm_sparse_mla as G
dev = torch.device("cuda"); torch.manual_seed(0)
H, W, D, BS, NB = 16, 2048, 512, 64, 2048            # 131K-slot cache
cache = torch.zeros((NB, BS, GLM5_FP8_KV_SLOT_BYTES), dtype=torch.uint8, device=dev)
lat = torch.randn((NB * BS, D), device=dev).half()
sm70_glm5_fp8_kv_insert(lat, cache, torch.arange(NB * BS, device=dev, dtype=torch.int64))
def timeit(fn, n=200):
    for _ in range(10): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n): fn()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n * 1000
for T in (1, 4, 8):
    q = (torch.randn((T, H, D), device=dev) * 0.05).half()
    idx = torch.stack([torch.randperm(NB * BS, device=dev)[:W] for _ in range(T)]).int()
    lens = torch.full((T,), W, dtype=torch.int32, device=dev)
    out = torch.empty_like(q)
    gk = torch.empty((T, W, D) if T > 1 else (W, D), dtype=torch.float16, device=dev)
    sc = torch.empty((T, H, W) if T > 1 else (H, W), dtype=torch.float16, device=dev); pr = torch.empty_like(sc)
    if T == 1:
        old = lambda: sm70_glm5_sparse_attention_paged_fp8_gemm(q, cache, idx, lens, 0.0625, out, gk, sc, pr)
    else:
        old = lambda: sm70_glm5_sparse_attention_paged_fp8_batched_gemm(q, cache, idx, lens, 0.0625, out, gk, sc, pr)
    t_old = timeit(old); o_old = out.clone()
    row = [f"T={T}: old {t_old:6.1f} us"]
    for s in (8, 16, 32, 64):
        ws = G.alloc_workspace(T, H, s, dev); o = torch.empty_like(q)
        t = timeit(lambda: G.sparse_mla_fp8(q, cache, idx, lens, 0.0625, out=o, num_splits=s, workspace=ws))
        row.append(f"s{s} {t:6.1f}")
    rel = float((o.float() - o_old.float()).norm() / o_old.float().norm())
    print(" | ".join(row), f"| new-vs-old rel {rel:.1e}", flush=True)
