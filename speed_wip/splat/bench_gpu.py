"""Splat-memory read / write benchmark, plain PyTorch (no Triton / torch.compile), plan_v2/23 section 7.

DO NOT RUN ON THE GPU WHILE A TRAINING RUN IS USING IT. CPU smoke (op counts are size-independent):
    python -B bench_gpu.py --device cpu --tokens 1024 --slots 65536
GPU, the full micro-batch (B = 16, T = 512 -> 8,192 tokens), 1M splats, when the GPU is free:
    python -B bench_gpu.py --device cuda --config A
    python -B bench_gpu.py --device cuda --config B
    python -B bench_gpu.py --device cuda --config all       (A, B, the variants and the PK-row reference)

What one "read" is (all queries of all tokens in ONE batched op sequence):
  1. queries: p = x @ W_view -> (N*V, d_m)                         (one GEMM from the 448-d addressing input)
  2. cells:   T index tables (each: a fixed random rotation of the memory space, d_m split in halves, C cells per half);
              -||p_h - c||^2 for all tables and halves in one bmm, top-m per half (one topk), pair scores, top-kc cells
              per table (one topk) = a multi-table product-key lookup of CELLS, not rows (ivf2_toy.py: T = 4 tables
              recover most of the brute-force drift robustness that one table loses)
  3. gather:  each (table, cell) owns a bucket of `cap` POINTERS to splat slots -> T*kc*cap candidates (duplicates
              across tables masked after a sort); one index_select of the packed per-slot coefficients
              [1/sigma^2 (d_m), -2 mu/sigma^2 (d_m), c_i]  (c_i = sum mu^2/sigma^2 - 2 log alpha); an empty pointer hits
              a dummy slot with c = 1e9 (weight 0)
  4. kernel:  Mahalanobis for every candidate in ONE bmm: [p^2, p, 1] . coef (point read), or for S ray samples at
              once (ray read), then front-to-back transmittance over the samples (cumprod)
  5. top-k:   scored under no_grad; the k winners are re-scored WITH grad (autograd keeps only (Q, k, 2 d_m + 1));
              weights w = A * rho / sum rho  (A = 1 - exp(-sum rho), the opacity)
  6. value:   embedding_bag(V_table, slots, per_sample_weights = w)  (the same MemRead-style op as today)
  Backward: autograd through 5 for the query (views learn) and alpha; the value table gets an index_add_ into one
  preallocated grad buffer (as MemRead). mu / sigma get no gradient (rule-based writes).
Write (once per optimizer step, a subsample of positions): reinforce if the read already returns the target, else
write a new splat into a free slot and file it in its best cell of every table (the bucket's weakest pointer).
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

CONFIGS = {
    # A: aims at >= 10k tok/s. Mirrors today's read pattern (12 queries per token = 4 levels x 3 heads, a read after
    #    every pass). d_m 32 (truncated whitening), T = 4 index tables, 4 cells per table x 8 pointers = 128 candidates.
    "A": dict(V=12, d_m=32, T=4, C=512, cap=8, m=4, kc=4, k=4, dv=256, S=1, hops=1, reads=5, ray=False),
    # B: aims at ~20k tok/s. 4 queries per token (one per level), reads on 2 passes only, same index.
    "B": dict(V=4, d_m=32, T=4, C=512, cap=8, m=4, kc=4, k=4, dv=256, S=1, hops=1, reads=2, ray=False),
    # variants (to price each option)
    "A_T1": dict(V=12, d_m=32, T=1, C=512, cap=8, m=4, kc=16, k=4, dv=256, S=1, hops=1, reads=5, ray=False),
    "A_ray8": dict(V=12, d_m=32, T=4, C=512, cap=8, m=4, kc=4, k=4, dv=256, S=8, hops=1, reads=5, ray=True),
    "A_hop2": dict(V=12, d_m=32, T=4, C=512, cap=8, m=4, kc=4, k=4, dv=256, S=1, hops=2, reads=5, ray=False),
    "A_d16": dict(V=12, d_m=16, T=4, C=512, cap=8, m=4, kc=4, k=4, dv=256, S=1, hops=1, reads=5, ray=False),
    "B_ray4": dict(V=4, d_m=32, T=4, C=512, cap=8, m=4, kc=4, k=4, dv=256, S=4, hops=1, reads=2, ray=True),
}

SKIP = {"detach", "view", "_unsafe_view", "t", "expand", "unsqueeze", "squeeze", "select", "slice", "permute",
        "transpose", "alias", "as_strided", "split", "split_with_sizes", "reshape", "unbind", "_reshape_alias",
        "lift_fresh", "numel", "sym_size", "sym_stride", "size", "stride", "is_same_size"}


class OpCount(TorchDispatchMode):
    """counts every dispatched aten op (~ kernel launches; pure view / metadata ops are not counted)"""

    def __init__(self):
        super().__init__()
        self.n, self.by = 0, {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.overloadpacket.__name__)
        if name not in SKIP:
            self.n += 1
            self.by[name] = self.by.get(name, 0) + 1
        return func(*args, **(kwargs or {}))


class SplatTable:
    """n splats (+1 dummy) in a slot store; T index tables of C*C cells x cap pointers each."""

    def __init__(self, d_m, C, cap, dv, device, n=None, T=4, dtype=torch.float32, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.d_m, self.C, self.cap, self.dv, self.T = d_m, C, cap, dv, T
        h = d_m // 2
        n = n or (1 << 20)
        rot = [torch.eye(d_m)] + [torch.linalg.qr(torch.randn(d_m, d_m, generator=g))[0] for _ in range(T - 1)]
        self.rot = torch.stack(rot).to(device)                                          # (T, d, d)
        self.cent = torch.randn(T * 2, C, h, generator=g).to(device)                    # codebooks per table and half
        self.cn = (self.cent ** 2).sum(-1)                                              # (2T, C)
        mu = torch.randn(n, d_m, generator=g).to(device)
        isg = torch.full((n + 1, d_m), 1 / 0.3 ** 2, device=device)
        la = torch.zeros(n + 1, device=device)
        mu = torch.cat([mu, torch.zeros(1, d_m, device=device)])
        c = (mu * mu * isg).sum(1) - 2 * la
        c[n] = 1e9                                                                      # the dummy slot
        self.coef = torch.cat([isg, -2 * mu * isg, c[:, None]], 1).to(dtype)
        self.log_alpha = la
        self.V = torch.randn(n + 1, dv, generator=g).to(device) * 0.02
        self.G = torch.zeros_like(self.V)                                               # one preallocated grad buffer
        self.birth = torch.zeros(n + 1, device=device)
        self.n = n
        # file every splat in its best cell of every table (bucket overflow is dropped)
        self.ptr = torch.full((T, C * C * cap), n, dtype=torch.long, device=device)
        cells = self.cells(mu[:n], 1, 1)[..., 0]                                        # (T, n)
        for t in range(T):
            order = cells[t].argsort()
            cs = cells[t][order]
            first = torch.searchsorted(cs, cs, right=False)
            rank = torch.arange(n, device=device) - first
            ok = rank < cap
            self.ptr[t].index_copy_(0, cs[ok] * cap + rank[ok], order[ok])

    def cells(self, p, m, kc):
        """p (Q, d) -> (T, Q, kc) cell ids per table: one bmm for all tables and halves, two topk"""
        T, h = self.T, self.d_m // 2
        y = torch.einsum("qd,ted->tqe", p, self.rot)                                    # (T, Q, d)
        yh = y.reshape(T, -1, 2, h).permute(0, 2, 1, 3).reshape(T * 2, -1, h)           # (2T, Q, h)
        sc = 2 * torch.bmm(yh, self.cent.transpose(1, 2)) - self.cn[:, None]            # (2T, Q, C)
        v, i = sc.topk(m, dim=2)
        v, i = v.view(T, 2, -1, m), i.view(T, 2, -1, m)
        pair = (v[:, 0, :, :, None] + v[:, 1, :, None, :]).flatten(2)                    # (T, Q, m*m)
        cid = (i[:, 0, :, :, None] * self.C + i[:, 1, :, None, :]).flatten(2)
        return cid.gather(2, pair.topk(kc, dim=2).indices)


class ValueRead(torch.autograd.Function):
    """embedding_bag forward; backward = index_add_ into the preallocated buffer + d/dw (as memory.MemRead)"""

    @staticmethod
    def forward(ctx, w, V, G, slots):
        ctx.save_for_backward(w, slots)
        ctx.V, ctx.G = V, G
        return F.embedding_bag(slots, V, per_sample_weights=w, mode="sum")

    @staticmethod
    def backward(ctx, gout):
        w, slots = ctx.saved_tensors
        if ctx.G is not None:
            ctx.G.index_add_(0, slots.flatten(), (w[..., None] * gout[:, None]).reshape(-1, gout.shape[1]))
        gw = torch.bmm(F.embedding(slots, ctx.V), gout[:, :, None])[..., 0]
        return gw, None, None, None


def lookup(tab, p, m, kc):
    """p (Q, d_m) -> candidate slot ids (Q, T*kc*cap), duplicates across tables replaced by the dummy slot"""
    cells = tab.cells(p, m, kc)                                                         # (T, Q, kc)
    T, Q = cells.shape[:2]
    base = (cells * tab.cap)[..., None] + torch.arange(tab.cap, device=p.device)        # (T, Q, kc, cap)
    flat = base + (torch.arange(T, device=p.device) * tab.ptr.shape[1])[:, None, None, None]
    slots = tab.ptr.view(-1)[flat].permute(1, 0, 2, 3).reshape(Q, -1)
    if T > 1:                                                                           # de-duplicate (sort + diff)
        slots, _ = slots.sort(dim=1)
        dup = torch.zeros_like(slots, dtype=torch.bool)
        dup[:, 1:] = slots[:, 1:] == slots[:, :-1]
        slots = slots.masked_fill(dup, tab.n)
    return slots, cells


def kernel_logits(tab, p, slots, S=1, dirs=None, L=1.0):
    """log rho for every candidate: point (Q, n) or S ray samples (Q, S, n); one gather + one bmm"""
    coef = tab.coef[slots].float()                                                      # (Q, n, 2d+1)
    if S == 1:
        feat = torch.cat([p * p, p, torch.ones_like(p[:, :1])], 1)[:, None]             # (Q, 1, 2d+1)
        return -0.5 * torch.bmm(feat, coef.transpose(1, 2))[:, 0]
    t = torch.linspace(0, L, S, device=p.device)
    ps = p[:, None] + t[None, :, None] * dirs[:, None]                                  # (Q, S, d)
    feat = torch.cat([ps * ps, ps, torch.ones_like(ps[..., :1])], 2)
    return -0.5 * torch.bmm(feat, coef.transpose(1, 2))                                 # (Q, S, n)


def _ray_weights(lg):
    """lg (Q, S, n) log rho per sample and candidate -> per-candidate front-to-back compositing weights (Q, n)"""
    rho = lg.exp()
    sig = rho.sum(2)
    a_s = 1 - torch.exp(-sig)
    T = torch.cumprod(torch.cat([torch.ones_like(a_s[:, :1]), 1 - a_s[:, :-1]], 1), 1)
    return ((T * a_s)[..., None] * rho / sig.clamp_min(1e-12)[..., None]).sum(1)


def splat_read(tab, x, Wv, cfg, write=True):
    V, d_m, k = cfg["V"], cfg["d_m"], cfg["k"]
    N = x.shape[0]
    p = (x @ Wv).view(N * V, d_m)
    dirs = None
    for hop in range(cfg["hops"]):
        with torch.no_grad():
            pd = p.detach()
            slots, _ = lookup(tab, pd, cfg["m"], cfg["kc"])
            if cfg["ray"]:                                           # candidates along the segment: cells of its end
                dirs = F.normalize(pd.roll(1, 1), dim=1)             # (stand-in for a learned direction)
                s2, _ = lookup(tab, pd + dirs, cfg["m"], max(1, cfg["kc"] // 2))
                slots = torch.cat([slots[:, :slots.shape[1] // 2], s2[:, :slots.shape[1] // 2]], 1)
                sc = _ray_weights(kernel_logits(tab, pd, slots, S=cfg["S"], dirs=dirs))
            else:
                sc = kernel_logits(tab, pd, slots)
            sl = slots.gather(1, sc.topk(k, dim=1).indices)          # (Q, k)
        if cfg["ray"]:                                               # exact, with grad, on the k winners only
            w = _ray_weights(kernel_logits(tab, p, sl, S=cfg["S"], dirs=dirs))
        else:
            rho = kernel_logits(tab, p, sl).exp()
            tot = rho.sum(1, keepdim=True)
            w = rho * ((1 - torch.exp(-tot)) / tot.clamp_min(1e-12))
        if hop + 1 < cfg["hops"]:                                    # hop: the location estimate updates the query
            cf = tab.coef[sl].float()
            mu = -0.5 * cf[..., d_m:2 * d_m] / cf[..., :d_m]
            p = (w[..., None] * mu).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-12)
    r = ValueRead.apply(w, tab.V, tab.G if write else None, sl)
    return r.view(N, V, -1), sl, w


@torch.no_grad()
def splat_write(tab, p, target, sl, w, step, frac=0.02):
    """direct write / densify for a subsample of query points p (Q, d_m) with value targets (Q, dv):
    reinforce if the read already returns the target, else write the new splat into a free slot and file it in its
    best cell of each table, replacing the bucket's weakest (faded) pointer. Stale pointers of an overwritten slot
    are harmless (they point at a far-away splat with ~0 weight)."""
    Q = p.shape[0]
    n = max(1, int(Q * frac))
    pick = torch.randperm(Q, device=p.device)[:n]
    p, target, sl, w = p[pick], target[pick], sl[pick], w[pick]
    col = (w[..., None] * tab.V[sl]).sum(1)
    A = w.sum(1)
    same = (A > 0.5) & (F.cosine_similarity(col, target, dim=1) > 0.9)
    tab.log_alpha.index_add_(0, sl[:, 0], same.float() * 0.3)                           # reinforce the top splat
    cells = tab.cells(p, 1, 1)[..., 0]                                                  # (T, n) best cell per table
    pos0 = cells * tab.cap + (torch.arange(tab.T, device=p.device) * tab.ptr.shape[1])[:, None]   # (T, n)
    bucket = tab.ptr.view(-1)[pos0[..., None] + torch.arange(tab.cap, device=p.device)]  # (T, n, cap) slot ids
    la = tab.log_alpha[bucket] - 1e-3 * (step - tab.birth[bucket])                       # weakest = lowest faded alpha
    la = la.masked_fill(bucket == tab.n, -1e9)                                          # (empty pointers first)
    vict = la.argmin(2)                                                                 # (T, n)
    dst = torch.randint(0, tab.n, (n,), device=p.device)                                # a slot from the free list
    dst = torch.where(same, torch.full_like(dst, tab.n), dst)                           # (stand-in); reinforced -> dummy
    isg = torch.full_like(p, 1 / 0.3 ** 2)
    coef = torch.cat([isg, -2 * p * isg, (p * p * isg).sum(1, keepdim=True)], 1)
    tab.coef.index_copy_(0, dst, coef.to(tab.coef.dtype))
    tab.V.index_copy_(0, dst, target)
    tab.log_alpha.index_copy_(0, dst, torch.zeros_like(dst, dtype=tab.log_alpha.dtype))
    tab.birth.index_copy_(0, dst, torch.full_like(dst, float(step), dtype=tab.birth.dtype))
    tab.ptr.view(-1).index_copy_(0, (pos0 + vict).flatten(), dst.repeat(tab.T))
    tab.coef[tab.n, -1] = 1e9                                                           # keep the dummy a dummy


def pk_rows_read(Vt, G, x, K1, Wq, H=12, m=8, km=6, tau=0.03):
    """reference: today's product-key ROW read pattern (fp32, S = 512 per half, 64-d halves, top-8 / top-6)"""
    N = x.shape[0]
    q = F.normalize((x @ Wq).view(N * H, 2, 64), dim=2)
    s = torch.bmm(q.transpose(0, 1), K1.transpose(1, 2))                                # (2, NH, S)
    v, i = s.topk(m, dim=2)
    pair = (v[0][:, :, None] + v[1][:, None, :]).flatten(1)
    rid = (i[0][:, :, None] * K1.shape[1] + i[1][:, None, :]).flatten(1)
    pv, j = pair.topk(km, dim=1)
    rows = rid.gather(1, j)
    w = torch.softmax(pv / tau, dim=1)
    return ValueRead.apply(w, Vt, G, rows % Vt.shape[0]).view(N, H, -1)


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bench(name, cfg, dev, tokens, slots_total=None, reps=5):
    cfg = dict(cfg)
    C, cap = cfg["C"], cfg["cap"]
    n = slots_total or (1 << 20)
    if slots_total is not None:                                     # CPU smoke: shrink the table, keep the op graph
        C = max(8, int(math.sqrt(slots_total / 4)))
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tab = SplatTable(cfg["d_m"], C, cap, cfg["dv"], dev, n=n, T=cfg["T"])
    x = torch.randn(tokens, 448, device=dev, requires_grad=True)
    Wv = (torch.randn(448, cfg["V"] * cfg["d_m"], device=dev) / math.sqrt(448)).requires_grad_()
    res = {}

    def one():
        r, sl, w = splat_read(tab, x, Wv, cfg)
        r.float().pow(2).mean().backward()
        return r, sl, w

    one()                                                           # warm-up
    oc = OpCount()
    with oc:
        r, sl, w = one()
    res["ops_read_fwd_bwd"] = oc.n
    res["ops_by_type"] = dict(sorted(oc.by.items(), key=lambda kv: -kv[1])[:12])
    p = (x.detach() @ Wv.detach()).view(-1, cfg["d_m"])
    tgt = torch.randn(p.shape[0], cfg["dv"], device=dev)
    oc2 = OpCount()
    with oc2:
        splat_write(tab, p, tgt, sl, w.detach(), step=10)
    res["ops_write"] = oc2.n
    ts = []
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        one()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    res["ms_read_fwd_bwd"] = 1e3 * sorted(ts)[len(ts) // 2]
    sync(dev)
    t0 = time.perf_counter()
    splat_write(tab, p, tgt, sl, w.detach(), step=11)
    sync(dev)
    res["ms_write"] = 1e3 * (time.perf_counter() - t0)
    if dev == "cuda":
        res["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        res["table_gb"] = (tab.coef.numel() * tab.coef.element_size() + 2 * tab.V.numel() * 4
                           + tab.ptr.numel() * 8) / 1e9
    # per step: reads per forward x 2 micro-batches, x1.33 for the checkpoint recompute of the forward
    res["est_ms_step"] = res["ms_read_fwd_bwd"] * cfg["reads"] * 2 * 1.33 + res["ms_write"]
    res["est_ops_step"] = res["ops_read_fwd_bwd"] * cfg["reads"] * 2 * 1.33 + res["ops_write"]
    res["est_launch_ms_step_at_35us"] = res["est_ops_step"] * 0.035
    res.update(cfg=cfg, tokens=tokens, slots=tab.n, cells_per_half=C)
    print(name, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    del tab
    return res


def bench_pk(dev, tokens, rows=1 << 20, H=12, reps=5):
    S = 512
    Vt = torch.randn(rows, 256, device=dev) * 0.02
    G = torch.zeros_like(Vt)
    K1 = F.normalize(torch.randn(2, S, 64, device=dev), dim=2)
    x = torch.randn(tokens, 448, device=dev, requires_grad=True)
    Wq = (torch.randn(448, H * 128, device=dev) / math.sqrt(448)).requires_grad_()
    f = lambda: pk_rows_read(Vt, G, x, K1, Wq, H=H).pow(2).mean().backward()
    f()
    oc = OpCount()
    with oc:
        f()
    ts = []
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        f()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    res = dict(ops_read_fwd_bwd=oc.n, ms_read_fwd_bwd=1e3 * sorted(ts)[len(ts) // 2], tokens=tokens)
    print("pk_rows_ref", json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default="all")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--slots", type=int, default=None, help="shrink the table (CPU smoke)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    names = list(CONFIGS) if a.config == "all" else [a.config]
    out = {n: bench(n, CONFIGS[n], a.device, a.tokens, a.slots) for n in names}
    out["pk_rows_ref"] = bench_pk(a.device, a.tokens, rows=(a.slots or (1 << 20)))
    path = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bench_{a.device}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path)
