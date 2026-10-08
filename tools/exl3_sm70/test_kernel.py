"""Unit test: libexl3_sm70 vs the independent reference decoder on REAL expert tensors from the conversion.
usage: python test_kernel.py <qtensors.safetensors of one MoE layer> [n_tokens]"""
import os, sys, time, torch
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "..", "..", "vllm", "model_executor", "layers", "quantization"))
import exl3_ref as R
from exl3_sm70_ops import Exl3ExpertBank
from safetensors import safe_open
dev = torch.device("cuda:0"); torch.manual_seed(0)
path, T = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
E, TOPK, RANKS = int(sys.argv[3]) if len(sys.argv) > 3 else 12, 8, 4
f = safe_open(path, "pt"); keys = list(f.keys())
pre = [k for k in keys if k.endswith("experts.0.up_proj.trellis")][0].rsplit("experts.0.", 1)[0]
get = lambda e, p, n: f.get_tensor(f"{pre}experts.{e}.{p}_proj.{n}").to(dev)
raw = {p: {n: torch.stack([get(e, p, n) for e in range(E)]) for n in ("trellis", "suh", "svh")} for p in ("gate", "up", "down")}
k, n_full = raw["gate"]["trellis"].shape[1] * 16, raw["gate"]["trellis"].shape[2] * 16
KB = raw["gate"]["trellis"].shape[-1] // 16
W = {p: torch.stack([R.reconstruct(raw[p]["trellis"][e], raw[p]["suh"][e], raw[p]["svh"][e], KB, "mcg", R.tile_perm_sm70_k8()) for e in range(E)])
     for p in ("gate", "up", "down")}                                   # (E, in, out) fp32, original basis
print(f"layer file {path.split('/')[-1]} | K = {KB} bits | hidden {k} intermediate {n_full} | experts {E} | tokens {T}")
x = (torch.randn(T, k, device = dev) * 0.5).half()
ids = torch.stack([torch.randperm(E, device = dev)[:TOPK] for _ in range(T)]).int().contiguous()
w = torch.rand(T, TOPK, device = dev).float().contiguous()

def oracle(sl):
    out = torch.zeros(T, k, device = dev, dtype = torch.double)
    for t in range(T):
        for j in range(TOPK):
            e = ids[t, j].item(); xv = x[t].double()
            g = xv @ W["gate"][e][:, sl].double(); u = xv @ W["up"][e][:, sl].double()
            h = torch.nn.functional.silu(g) * u
            out[t] += w[t, j].double() * (h @ W["down"][e][sl, :].double())
    return out

full = oracle(slice(0, n_full)); total = torch.zeros_like(full); worst = 0.0
for r in range(RANKS):
    nt = n_full // 16 // RANKS; ts = slice(r * nt, (r + 1) * nt); ns = slice(r * nt * 16, (r + 1) * nt * 16)
    bank = Exl3ExpertBank(raw["gate"]["trellis"][:, :, ts], raw["gate"]["suh"], raw["gate"]["svh"][:, ns],
                          raw["up"]["trellis"][:, :, ts], raw["up"]["suh"], raw["up"]["svh"][:, ns],
                          raw["down"]["trellis"][:, ts], raw["down"]["suh"][:, ns], raw["down"]["svh"], TOPK)
    y = bank.forward(x, ids, w).double().clone(); torch.cuda.synchronize()
    ref = oracle(ns); e = ((y - ref).norm() / ref.norm()).item(); worst = max(worst, e)
    print(f"  rank {r}: kernel vs oracle rel err {e:.2e} | finite {torch.isfinite(y).all().item()}")
    total += y
e_full = ((total - full).norm() / full.norm()).item()
print(f"sum of {RANKS} rank partials vs full-width oracle: rel err {e_full:.2e}")
# large-T correctness (prefill shapes use the shared growing scratch): sample tokens vs the oracle
for Tb in (65, 1024, 3000):
    xb = (torch.randn(Tb, k, device = dev) * 0.5).half(); ib = torch.stack([torch.randperm(E, device = dev)[:TOPK] for _ in range(Tb)]).int().contiguous(); wb = torch.rand(Tb, TOPK, device = dev).contiguous()
    yb = bank.forward(xb, ib, wb).double().clone(); torch.cuda.synchronize(); errs = []
    for t in (0, 1, Tb // 2, Tb - 2, Tb - 1):
        ref = torch.zeros(k, device = dev, dtype = torch.double)
        for j in range(TOPK):
            e = ib[t, j].item(); xv = xb[t].double()
            h = torch.nn.functional.silu(xv @ W["gate"][e][:, ns].double()) * (xv @ W["up"][e][:, ns].double())
            ref += wb[t, j].double() * (h @ W["down"][e][ns, :].double())
        errs.append(((yb[t] - ref).norm() / ref.norm()).item())
    worst = max(worst, max(errs)); print(f"  T={Tb}: sampled tokens vs oracle, worst rel err {max(errs):.2e} | finite {torch.isfinite(yb).all().item()} | slots per decode group {bank.last_group_ratio}")
# timing, decode shape (T = 1) and a prefill-ish shape
for Tb in (1, 8, 256, 1024):
    xb = (torch.randn(Tb, k, device = dev) * 0.5).half(); ib = torch.stack([torch.randperm(E, device = dev)[:TOPK] for _ in range(Tb)]).int().contiguous(); wb = torch.rand(Tb, TOPK, device = dev)
    for _ in range(3): bank.forward(xb, ib, wb)
    torch.cuda.synchronize(); t0 = time.time(); N = 20 if Tb < 100 else 3
    for _ in range(N): bank.forward(xb, ib, wb)
    torch.cuda.synchronize(); dt = (time.time() - t0) / N
    print(f"  timing T={Tb:4d} (one rank, one layer): {dt * 1e3:8.3f} ms  -> {dt / Tb * 1e6:7.1f} us/token")
print("RESULT:", "PASS" if worst < 2e-3 and e_full < 2e-3 else "FAIL")
