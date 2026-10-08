# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for glm_attn_v100 sparse MLA (run on one V100):
    python tests/test_glm_sparse_mla.py
JIT-builds kernel/glm_sparse_mla.cu, then compares against an FP64 reference built from the exact
dequantized cache, and against an emulation of the FP16-score GEMM route it replaces."""

import math
import os
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D, GROUP, NG, SLOT = 512, 64, 8, 520

ext = load(
    name="glm_sparse_mla_cuda",
    sources=[os.path.join(ROOT, "kernel", "glm_sparse_mla.cu")],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "-std=c++17"],
    verbose=False,
)
sys.modules["glm_sparse_mla_cuda"] = ext
sys.path.insert(0, ROOT)
from glm_attn_v100 import glm_sparse_mla as G  # noqa: E402

dev = torch.device("cuda")


def make_cache(latent: torch.Tensor, block_size: int):
    """Pack fp32 latents [N, 512] exactly like _sm70_glm5_fp8_kv_insert_kernel (slots 0..N-1)."""
    n = latent.shape[0]
    blocks = -(-n // block_size)
    flat = torch.zeros((blocks, block_size * SLOT), dtype=torch.uint8, device=dev)
    grouped = latent.reshape(n, NG, GROUP).float()
    absmax = grouped.abs().amax(-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(absmax / 448.0))
    quant = (grouped / torch.exp2(exponent)[..., None]).clamp(-448.0, 448.0)
    data = quant.to(torch.float8_e4m3fn).view(torch.uint8).reshape(n, D)
    scales = (exponent + 127.0).clamp(0, 255).to(torch.uint8)
    for s in range(n):
        b, p = divmod(s, block_size)
        flat[b, p * D:(p + 1) * D] = data[s]
        flat[b, block_size * D + p * NG:block_size * D + (p + 1) * NG] = scales[s]
    return flat.view(blocks, block_size, SLOT)


def dequant(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Exact fp16 dequant of the given slots (mirrors the Triton gather kernel)."""
    blocks, block_size, _ = cache.shape
    flat = cache.reshape(blocks, block_size * SLOT)
    b, p = slots // block_size, slots % block_size
    cols = (p[:, None] * D + torch.arange(D, device=dev)[None, :])
    data = flat[b[:, None], cols]
    scol = block_size * D + p[:, None] * NG + torch.arange(NG, device=dev)[None, :]
    sc = flat[b[:, None], scol].float()
    scale = torch.where(sc == 0, torch.zeros_like(sc), torch.exp2(sc - 127.0)).half()
    vals = data.view(torch.float8_e4m3fn).float().half().reshape(-1, NG, GROUP)
    return (vals * scale[..., None]).reshape(-1, D)


def reference(q, cache, indices, lengths, scale):
    """FP64 attention over the exactly dequantized latents."""
    T, H, _ = q.shape
    num_slots = cache.shape[0] * cache.shape[1]
    out = torch.zeros((T, H, D), dtype=torch.float64, device=dev)
    for t in range(T):
        n = min(int(lengths[t]), indices.shape[1])
        idx = indices[t, :n]
        idx = idx[(idx >= 0) & (idx < num_slots)]
        if idx.numel() == 0:
            continue
        kv = dequant(cache, idx.long()).double()
        s = (q[t].double() @ kv.T) * scale
        out[t] = torch.softmax(s, -1) @ kv
    return out


def old_route(q, cache, indices, lengths, scale):
    """Emulates sm70_glm5_sparse_attention_paged_fp8_(batched_)gemm: FP16 unscaled scores + FP16 probs."""
    T, H, _ = q.shape
    num_slots = cache.shape[0] * cache.shape[1]
    out = torch.zeros((T, H, D), dtype=torch.float16, device=dev)
    for t in range(T):
        W = indices.shape[1]
        n = min(int(lengths[t]), W)
        slots = indices[t]
        valid = (torch.arange(W, device=dev) < n) & (slots >= 0) & (slots < num_slots)
        kv = dequant(cache, torch.where(valid, slots, 0).long())
        kv = torch.where(valid[:, None], kv, torch.zeros_like(kv))
        scores = torch.mm(q[t], kv.T)  # fp16 out
        s = torch.where(valid[None, :], scores.float() * scale, torch.full_like(scores.float(), -3.4e38))
        p = torch.softmax(s, -1)
        p = torch.where(valid[None, :], p, torch.zeros_like(p)).half()
        out[t] = torch.mm(p, kv)
    return out


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def run_case(name, T, H, W, n_cache, lengths, q_std, kv_std, num_splits=None, invalid_frac=0.0,
             block_size=64, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    latent = torch.randn((n_cache, D), generator=g, device=dev) * kv_std
    latent[:, :8] *= 6.0  # a few heavy channels, like real post-norm latents
    cache = make_cache(latent, block_size)
    q = (torch.randn((T, H, D), generator=g, device=dev) * q_std).half()
    indices = torch.stack([torch.randperm(n_cache, generator=g, device=dev)[:W] for _ in range(T)]).int()
    if invalid_frac:
        mask = torch.rand(indices.shape, generator=g, device=dev) < invalid_frac
        indices[mask] = -1
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(256.0)
    ref = reference(q, cache, indices, lengths_t, scale)
    new = G.sparse_mla_fp8(q, cache, indices, lengths_t, scale, num_splits=num_splits)
    old = old_route(q, cache, indices, lengths_t, scale)
    torch.cuda.synchronize()
    new_err, old_err = rel_l2(new, ref), rel_l2(old, ref)
    new_bad = int((~torch.isfinite(new)).sum())
    old_bad = int((~torch.isfinite(old)).sum())
    max_abs_logit = float(((q.double() @ dequant(cache, indices[0].clamp_min(0).long()).double().T)
                           * scale).abs().max())
    print(f"{name:34s} new rel {new_err:.2e} nonfinite {new_bad:5d} | old rel {old_err:.2e} "
          f"nonfinite {old_bad:5d} | max|logit| {max_abs_logit:8.1f}")
    return new_err, new_bad, new


def main():
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # torch default
    fails = []
    cases = [
        dict(name="decode T1 H16 W2048", T=1, H=16, W=2048, n_cache=4096, lengths=[2048], q_std=0.05, kv_std=1.0),
        dict(name="decode short len 37", T=1, H=16, W=2048, n_cache=4096, lengths=[37], q_std=0.05, kv_std=1.0),
        dict(name="verify T8 H32 mixed lengths", T=8, H=32, W=2048, n_cache=4096,
             lengths=[2048, 1, 0, 100, 2047, 512, 33, 2048], q_std=0.05, kv_std=1.0, invalid_frac=0.05),
        dict(name="decode splits=1", T=1, H=16, W=2048, n_cache=4096, lengths=[2048], q_std=0.05, kv_std=1.0, num_splits=1),
        dict(name="decode splits=64", T=1, H=16, W=2048, n_cache=4096, lengths=[2048], q_std=0.05, kv_std=1.0, num_splits=64),
        dict(name="sharp attention (big logits)", T=2, H=16, W=2048, n_cache=4096, lengths=[2048, 2048], q_std=0.6, kv_std=1.0),
        dict(name="hot scores (old route overflows)", T=2, H=16, W=2048, n_cache=4096, lengths=[2048, 2048], q_std=4.0, kv_std=2.0),
        dict(name="block_size 16, width 512", T=3, H=16, W=512, n_cache=1000, lengths=[512, 400, 1], q_std=0.1, kv_std=1.0, block_size=16),
    ]
    for c in cases:
        err, bad, _ = run_case(**c)
        if bad or err > 3e-3:
            fails.append(c["name"])

    # CUDA graph capture + replay must match eager.
    g = torch.Generator(device=dev).manual_seed(7)
    latent = torch.randn((4096, D), generator=g, device=dev)
    cache = make_cache(latent, 64)
    q = (torch.randn((4, 16, D), generator=g, device=dev) * 0.05).half()
    idx = torch.stack([torch.randperm(4096, generator=g, device=dev)[:2048] for _ in range(4)]).int()
    lens = torch.tensor([2048, 1000, 5, 2048], dtype=torch.int32, device=dev)
    splits = G.default_num_splits(2048)
    ws = G.alloc_workspace(4, 16, splits, dev)
    out = torch.empty_like(q)
    eager = G.sparse_mla_fp8(q, cache, idx, lens, 0.0625, num_splits=splits).clone()
    G.sparse_mla_fp8(q, cache, idx, lens, 0.0625, out=out, num_splits=splits, workspace=ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        G.sparse_mla_fp8(q, cache, idx, lens, 0.0625, out=out, num_splits=splits, workspace=ws)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    graph_ok = torch.equal(out, eager)
    print(f"cuda graph replay bitwise == eager: {graph_ok}")
    if not graph_ok:
        fails.append("cuda graph")

    # Timing (decode shape) vs the old GEMM route emulation.
    q1 = q[:1].contiguous(); i1 = idx[:1].contiguous(); l1 = lens[:1].contiguous()
    ws1 = G.alloc_workspace(1, 16, splits, dev); o1 = torch.empty_like(q1)
    for _ in range(20):
        G.sparse_mla_fp8(q1, cache, i1, l1, 0.0625, out=o1, num_splits=splits, workspace=ws1)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(200):
        G.sparse_mla_fp8(q1, cache, i1, l1, 0.0625, out=o1, num_splits=splits, workspace=ws1)
    ev1.record(); torch.cuda.synchronize()
    print(f"decode T1 H16 W2048: {ev0.elapsed_time(ev1) / 200 * 1000:.1f} us/call (splits={splits})")

    print("FAIL: " + ", ".join(fails) if fails else "ALL PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
