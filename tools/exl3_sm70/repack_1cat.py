"""Repack the exllamav3 conversion output (per-module qtensors) into a checkpoint 1Cat-vLLM's glm5next loader reads.

- Non-expert tensors keep their HF names; exllamav3's fused KDA tensors are split back
  (self_attn.qkv_proj.weight -> q_proj/k_proj/v_proj, self_attn.conv1d.weight -> q/k/v_conv1d).
- Routed-expert EXL3 tensors are STACKED per layer for the SM70 MoE method:
    <layer>.mlp.experts.exl3_{gate,up,down}_{trellis,suh,svh}   ([288, ...], int16 / fp16)
- MTP layer (layers.45.*) and vision (model.visual.*) are optional (--mtp / --vision).
usage: python repack_1cat.py <qtensors_dir> <src_model_dir> <out_dir> [--layers N] [--mtp] [--vision]
"""
import sys, os, re, json, shutil, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import exl3_ref as R
dq_dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
from safetensors import safe_open
from safetensors.torch import save_file

qt, src, out = sys.argv[1:4]
opt = lambda n: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else None
import time
PAUSE = float(opt("--pause") or 0)
NL = int(opt("--layers") or 45); want_mtp = "--mtp" in sys.argv; want_vis = "--vision" in sys.argv
os.makedirs(out, exist_ok = True)
EXP = re.compile(r"(.*\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.(trellis|suh|svh|mcg|mul1)$")
total = 0
layer_bits = {}

def convert(path, name):
    global total
    dst = os.path.join(out, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0 and "--force" not in sys.argv:
        total += os.path.getsize(dst); return                  # incremental: module files are final once written
    res, bank, dq = {}, {}, {}
    is_mtp_file = bool(re.match(r"model\.language_model\.layers\.(4[5-9]|[5-9]\d)\b", name))
    with safe_open(path, "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            m = EXP.match(k)
            if m:
                pre, e, proj, kind = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                if kind == "mcg":
                    assert t.view(torch.uint32).item() == 0xCBAC1FED, "unexpected mcg multiplier"
                    continue
                assert kind != "mul1", "mul1 codebook not supported by the SM70 path"
                bank.setdefault((pre, proj, kind), {})[e] = t
            elif k.endswith("self_attn.qkv_proj.weight"):
                q, kk, v = t.chunk(3, 0); b = k[:-len("qkv_proj.weight")]
                res[b + "q_proj.weight"], res[b + "k_proj.weight"], res[b + "v_proj.weight"] = q.contiguous(), kk.contiguous(), v.contiguous()
            elif k.endswith("self_attn.conv1d.weight"):
                q, kk, v = t.chunk(3, 0); b = k[:-len("conv1d.weight")]
                res[b + "q_conv1d.weight"], res[b + "k_conv1d.weight"], res[b + "v_conv1d.weight"] = q.contiguous(), kk.contiguous(), v.contiguous()
            elif is_mtp_file and re.search(r"\.(trellis|suh|svh|mcg)$", k):
                # MTP layer (45): the draft loader (1Cat Glm5NextMTP) wants plain fp16 weights for everything except
                # the routed experts, so dequantize the attention / indexer / shared-expert EXL3 tensors here
                base, kind = k.rsplit(".", 1)
                dq.setdefault(base, {})[kind] = t
            else:
                res[k] = t
    for base, parts in dq.items():
        assert {"trellis", "suh", "svh"} <= set(parts), (base, parts.keys())
        tr = parts["trellis"].to(dq_dev); K = tr.shape[-1] // 16
        W = R.reconstruct(tr, parts["suh"].to(dq_dev), parts["svh"].to(dq_dev), K, "mcg", R.tile_perm_sm70_k8())   # (in, out)
        res[base + ".weight"] = W.T.contiguous().half().cpu()
        print("  dequantized " + base.split(".layers.")[1] + f" K{K} -> fp16 " + str(tuple(res[base + ".weight"].shape)), flush = True)
    for (pre, proj, kind), d in bank.items():
        E = len(d); assert sorted(d) == list(range(E)), f"expert gap in {pre}"
        st = torch.stack([d[e] for e in range(E)])
        if kind == "trellis":
            layer_bits.setdefault(int(pre.split(".layers.")[1].split(".")[0]), set()).add(st.shape[-1] // 16)
        assert st.dtype == (torch.int16 if kind == "trellis" else torch.half), (kind, st.dtype)
        res[f"{pre}.exl3_{proj}_{kind}"] = st
    nb = sum(v.numel() * v.element_size() for v in res.values()); total += nb
    save_file(res, dst + ".tmp", metadata = {"format": "pt"}); os.replace(dst + ".tmp", dst)
    if PAUSE: time.sleep(PAUSE)
    print(f"{name}: {len(res)} tensors, {nb / 2**30:.2f} GiB" + (f", {len(bank)} stacked expert tensors" if bank else ""), flush = True)

files = sorted(os.listdir(qt))
plan = []
for fn in files:
    key = fn[:-len(".safetensors")]
    m = re.match(r"model\.language_model\.layers\.(\d+)(\..*)?$", key)
    if m:
        L = int(m.group(1))
        if L >= 45:
            if want_mtp: plan.append(fn)
        elif L < NL: plan.append(fn)
    elif key.startswith("model.visual"):
        if want_vis: plan.append(fn)
    else:
        plan.append(fn)
print(f"repacking {len(plan)} of {len(files)} module files -> {out} (layers < {NL}, mtp {want_mtp}, vision {want_vis})", flush = True)
for fn in plan:
    convert(os.path.join(qt, fn), fn if fn.strip() else "empty.safetensors")

c = json.load(open(os.path.join(src, "config.json")))
tc = c["text_config"]
if NL != 45:
    tc["num_hidden_layers"] = NL
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        if k in tc: tc[k] = tc[k][:NL]
    la = tc.get("linear_attn_config") or {}
    for k in ("kda_layers", "full_attn_layers"):
        if k in la: la[k] = [x for x in la[k] if x < NL]
assert all(len(v) == 1 for v in layer_bits.values()), f"mixed K inside one layer is not supported: {layer_bits}"
prev = {}
if os.path.exists(os.path.join(out, "config.json")):       # incremental runs: keep layers recorded earlier
    prev = (json.load(open(os.path.join(out, "config.json"))).get("quantization_config") or {}).get("layer_bits") or {}
lb = {int(k): int(v) for k, v in prev.items()}; lb.update({k: next(iter(v)) for k, v in layer_bits.items()})
c["quantization_config"] = {"quant_method": "exl3", "bits": 2, "codebook": "mcg", "tile_order": "sm70_colmajor",
                            "layer_bits": {str(k): lb[k] for k in sorted(lb) if lb[k] != 2},
                            "scope": "glm53_routed_experts_only", "source": "zai-org/GLM-5.3-Flash (FP8)"}
json.dump(c, open(os.path.join(out, "config.json"), "w"), indent = 2)
for fn in os.listdir(src):
    if fn.endswith((".json", ".jinja", ".txt", ".md")) and fn not in ("config.json", "model.safetensors.index.json", "quantization_config.json"):
        shutil.copy(os.path.join(src, fn), os.path.join(out, fn))
print(f"done: {total / 2**30:.1f} GiB of tensors; config.json quantization_config = exl3")
