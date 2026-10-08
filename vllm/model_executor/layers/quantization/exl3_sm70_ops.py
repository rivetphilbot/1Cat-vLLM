"""ctypes facade for libexl3_sm70.so + a pure-torch oracle. No vLLM imports (unit-testable standalone)."""
import ctypes, os, torch

_lib = None
def lib():
    global _lib
    if _lib is None:
        path = os.environ.get("EXL3_SM70_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "libexl3_sm70.so"))
        _lib = ctypes.CDLL(path)
        vp, i, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
        _lib.exl3_sm70_gemv.argtypes = [vp, vp, vp, vp, i, i, i, ll, vp, i]
        _lib.exl3_sm70_pre.argtypes = [vp, vp, vp, vp, i, i, i, vp]
        _lib.exl3_sm70_mid.argtypes = [vp, vp, vp, vp, vp, vp, vp, i, i, vp]
        _lib.exl3_sm70_post.argtypes = [vp, vp, vp, vp, vp, i, i, i, vp]
        _lib.exl3_sm70_gemv2.argtypes = [vp, vp, vp, vp, vp, vp, vp, i, i, i, ll, vp, i]
        _lib.exl3_sm70_pre2.argtypes = [vp, vp, vp, vp, vp, vp, i, i, i, vp]
        _lib.exl3_sm70_gemvg.argtypes = [vp, vp, vp, vp, vp, i, i, i, ll, vp, i]
        _lib.exl3_sm70_gemvg2.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, i, i, i, ll, vp, i]
        for f in (_lib.exl3_sm70_gemv, _lib.exl3_sm70_pre, _lib.exl3_sm70_mid, _lib.exl3_sm70_post,
                  _lib.exl3_sm70_gemv2, _lib.exl3_sm70_pre2, _lib.exl3_sm70_gemvg, _lib.exl3_sm70_gemvg2):
            f.restype = i
    return _lib

def _stream(dev):
    return torch.cuda.current_stream(dev).cuda_stream

def _ck(rc, name):
    if rc != 0:
        raise RuntimeError(f"{name} failed (rc {rc})")


class Exl3ExpertBank:
    """One MoE layer's routed experts on one device (optionally one TP rank's slice).

    gate/up: trellis [E, k/16, n/16, 32] int16 (k = hidden, n = intermediate slice), suh [E, k], svh [E, n]
    down:    trellis [E, n/16, k/16, 32] int16 (rows = intermediate slice),           suh [E, n], svh [E, k]
    """
    def __init__(self, g_tr, g_suh, g_svh, u_tr, u_suh, u_svh, d_tr, d_suh, d_svh, topk):
        self.t = [x.contiguous() for x in (g_tr, g_suh, g_svh, u_tr, u_suh, u_svh, d_tr, d_suh, d_svh)]
        (self.g_tr, self.g_suh, self.g_svh, self.u_tr, self.u_suh, self.u_svh, self.d_tr, self.d_suh, self.d_svh) = self.t
        self.E, kt, nt, w = self.g_tr.shape
        assert w in (32, 48, 64) and self.g_tr.dtype == torch.int16, "K = 2, 3 or 4 (mcg codebook)"
        self.K = w // 16
        self.k, self.n, self.topk = kt * 16, nt * 16, topk
        assert self.d_tr.shape == (self.E, nt, kt, w) and self.u_tr.shape == self.g_tr.shape
        self.dev = self.g_tr.device
        self.gw = kt * nt * 8 * self.K  # u32 words per expert matrix (gate/up and down are the same size)

    # Scratch is shared by every bank of the same shape on a device (layers run one after another). Small token
    # counts (decode; CUDA-graph captured) get exact, permanent buffers; larger ones share one growing scratch.
    _scratch = {}
    SMALL_T = 64

    def buffers(self, S, T):
        small = T <= self.SMALL_T
        key = (self.dev, self.k, self.n, self.topk, T if small else "big")
        b = Exl3ExpertBank._scratch.get(key)
        if b is None or b["T"] < T:
            mk = lambda *shape, dt: torch.empty(*shape, dtype = dt, device = self.dev)
            b = dict(T = T, xg = mk(S, self.k, dt = torch.half), xu = mk(S, self.k, dt = torch.half),
                     yg = mk(S, self.n, dt = torch.float), yu = mk(S, self.n, dt = torch.float),
                     hd = mk(S, self.n, dt = torch.half), yd = mk(S, self.k, dt = torch.float),
                     out = mk(T, self.k, dt = torch.float))
            Exl3ExpertBank._scratch[key] = b
        return b

    GROUP_MIN_T = 65            # token counts at/above this use the grouped (decode-once-per-8-rows) GEMVs; eager only
    last_group_ratio = None     # slots per group in the last grouped call (diagnostic)
    _dump_n = 0

    def _groups(self, ids):
        """Group slots by expert, up to 8 per group. Returns (gexp [G] int32, rows [G, 8] int32 with -1 padding, G)."""
        S = ids.numel()
        es, order = torch.sort(ids.view(-1).long(), stable = True)
        pos = torch.arange(S, device = self.dev)
        first = torch.ones(S, dtype = torch.bool, device = self.dev); first[1:] = es[1:] != es[:-1]
        start = torch.cummax(torch.where(first, pos, torch.zeros_like(pos)), 0).values
        inrun = pos - start
        gid = torch.cumsum((inrun % 8 == 0).long(), 0) - 1
        G = int(gid[-1].item()) + 1
        rows = torch.full((G * 8,), -1, dtype = torch.int32, device = self.dev)
        rows[gid * 8 + inrun % 8] = order.int()
        gexp = torch.empty(G, dtype = torch.int32, device = self.dev)
        gexp[gid] = es.int()
        return gexp, rows, G

    def forward(self, x, topk_ids, topk_weights):
        """x [T, k] fp16, topk_ids [T, topk] int32, topk_weights [T, topk] fp32 -> [T, k] fp32 (this rank's partial)."""
        T = x.shape[0]; S = T * self.topk
        assert 2 * S <= 65535, "too many slots for one launch"
        assert x.dtype == torch.half and x.is_contiguous() and topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float
        assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
        b = self.buffers(S, T); L = lib(); st = _stream(self.dev); p = lambda t: t.data_ptr()
        eid = p(topk_ids)
        _ck(L.exl3_sm70_pre2(p(x), eid, p(self.g_suh), p(self.u_suh), p(b["xg"]), p(b["xu"]), S, self.k, self.topk, st), "pre2")
        if T >= self.GROUP_MIN_T:
            dump = os.environ.get("EXL3_DUMP_ROUTING")       # diagnostics: append this call's routing table
            if dump:
                torch.save(topk_ids.cpu(), f"{dump}/ids_{os.getpid()}_{Exl3ExpertBank._dump_n:06d}.pt"); Exl3ExpertBank._dump_n += 1
            gexp, rows, G = self._groups(topk_ids)
            Exl3ExpertBank.last_group_ratio = S / G
            _ck(L.exl3_sm70_gemvg2(p(self.g_tr), p(self.u_tr), p(gexp), p(rows), p(b["xg"]), p(b["xu"]), p(b["yg"]), p(b["yu"]), G,
                                   self.k, self.n, self.gw, st, self.K), "gemvg2")
            _ck(L.exl3_sm70_mid(p(b["yg"]), p(b["yu"]), eid, p(self.g_svh), p(self.u_svh), p(self.d_suh), p(b["hd"]), S, self.n, st), "mid")
            _ck(L.exl3_sm70_gemvg(p(self.d_tr), p(gexp), p(rows), p(b["hd"]), p(b["yd"]), G, self.n, self.k, self.gw, st, self.K), "gemvg(d)")
            _ck(L.exl3_sm70_post(p(b["yd"]), eid, p(topk_weights), p(self.d_svh), p(b["out"]), T, self.k, self.topk, st), "post")
            return b["out"] if b["T"] == T else b["out"][:T]
        _ck(L.exl3_sm70_gemv2(p(self.g_tr), p(self.u_tr), eid, p(b["xg"]), p(b["xu"]), p(b["yg"]), p(b["yu"]), S, self.k, self.n,
                              self.gw, st, self.K), "gemv2")
        _ck(L.exl3_sm70_mid(p(b["yg"]), p(b["yu"]), eid, p(self.g_svh), p(self.u_svh), p(self.d_suh), p(b["hd"]), S, self.n, st), "mid")
        _ck(L.exl3_sm70_gemv(p(self.d_tr), eid, p(b["hd"]), p(b["yd"]), S, self.n, self.k, self.gw, st, self.K), "gemv(d)")
        _ck(L.exl3_sm70_post(p(b["yd"]), eid, p(topk_weights), p(self.d_svh), p(b["out"]), T, self.k, self.topk, st), "post")
        return b["out"] if b["T"] == T else b["out"][:T]
