"""Shared loading / evaluation for alt2.py (the same protocol as alt.py and addr_inv/sim.py)."""
import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
from memory import pk_topk

D = r"C:\Eitan\Python\NewBrain\speed_wip\addr_inv"
c6 = torch.load(os.path.join(D, "cap_6000.pt"), map_location="cpu", weights_only=False)
c7 = torch.load(os.path.join(D, "cap_b.pt"), map_location="cpu", weights_only=False)
assert torch.equal(c6["X"], c7["X"])
st = c6["state"]
S, m, km = c6["S"], c6["m"], c6["km"]
X = c6["X"]; B, T = X.shape; N = B * T
L, H = st["wq_ema"].shape[:2]
seq = torch.arange(B).repeat_interleave(T); pos = torch.arange(T).repeat(B)
eot = (X == 0); doc = (seq * 1000 + (torch.cumsum(eot.long(), 1) - eot.long()).flatten())
FIT = (seq < B // 2).nonzero().squeeze(1); EV = (seq >= B // 2).nonzero().squeeze(1)
NE = EV.numel()
gen = torch.Generator().manual_seed(0)
seq_e, doc_e = seq[EV], doc[EV]


def lag_pairs(lag, n=4000):
    a = torch.randint(NE, (n * 3,), generator=gen); b = a + lag
    ok = (b < NE); a, b = a[ok], b[ok]
    ok = (seq_e[a] == seq_e[b]) & (doc_e[a] == doc_e[b])
    return a[ok][:n], b[ok][:n]


PAIRS = {f"lag{g}": lag_pairs(g) for g in (1, 16, 256)}
a_ = torch.randint(NE, (8000,), generator=gen); b_ = torch.randint(NE, (8000,), generator=gen)
ok = seq_e[a_] != seq_e[b_]; PAIRS["xdoc"] = (a_[ok][:4000], b_[ok][:4000])
SUBS = torch.randperm(NE, generator=gen)[:4000].view(4, 1000)
VERT = torch.tensor([[1.0 if (c >> b) & 1 else -1.0 for b in range(9)] for c in range(512)])


def level_sets(k, l):
    ci6 = [s["k"] for s in c6["sels"]].index(k); ci7 = [s["k"] for s in c7["sels"]].index(k)
    a6 = c6["sels"][ci6]["a"][l].float(); a7 = c7["sels"][ci7]["a"][l].float()
    nm = lambda a: (a - st["norm_m"][l]) / torch.sqrt(st["norm_v"][l] + 1e-5)
    gn = torch.Generator().manual_seed(100 + l)
    ae = a6[EV]
    noise = {r: ae + r * ae.norm(dim=-1, keepdim=True) * torch.randn(ae.shape, generator=gn) / math.sqrt(ae.size(1)) for r in (0.03, 0.10)}
    return {"fit": nm(a6[FIT]), "ev": nm(ae), "ev7": nm(a7[EV]), "n03": nm(noise[0.03]), "n10": nm(noise[0.10])}


def make_inputs(sets_raw, mode):
    if mode == "ema":
        return sets_raw
    fit = sets_raw["fit"]
    mu = fit.mean(0); xc = fit - mu
    ev, V = torch.linalg.eigh(xc.T @ xc / xc.size(0))
    ev = ev.clamp(min=0) + 1e-6 * float(ev.max())
    if mode == "zca":
        W = V @ torch.diag(ev ** -0.5) @ V.T
    else:
        r = int(mode[3:])
        W = V[:, -r:] @ torch.diag(ev[-r:] ** -0.5)
    return {n_: (x - mu) @ W for n_, x in sets_raw.items()}


def spherical_kmeans(x, k, iters=25, balanced=False):
    x = F.normalize(x, dim=-1)
    C = x[torch.randperm(x.size(0), generator=gen)[:k]].clone()
    for _ in range(iters):
        sc = x @ C.T
        if balanced:
            Q = torch.exp((sc - sc.max()) / 0.05)
            for _ in range(5):
                Q = Q / Q.sum(0, keepdim=True); Q = Q / Q.sum(1, keepdim=True)
            a = Q.argmax(1)
        else:
            a = sc.argmax(1)
        Cn = torch.zeros_like(C).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k)
        empty = (cnt == 0).nonzero().squeeze(1)
        if empty.numel():
            worst = sc.max(1).values.argsort()[: empty.numel()]
            Cn[empty] = x[worst]
        C = F.normalize(Cn, dim=-1)
    return C


def overlap(Ra, Rb):
    return float((Ra[:, :, None] == Rb[:, None, :]).any(-1).float().sum(-1).mean() / km)


def top1share(R):
    c = torch.bincount(R.flatten(), minlength=S * S).double()
    return float(torch.sort(c, descending=True).values[: int(0.01 * S * S)].sum() / c.sum())


def mi_halves(R):
    i, j = R.flatten() // S, R.flatten() % S
    def Hh(x):
        p = torch.bincount(x).double(); p = p[p > 0] / p.sum(); return float(-(p * p.log()).sum())
    sh = j[torch.randperm(j.numel(), generator=gen)]
    return (Hh(i) + Hh(j) - Hh(i * S + j)) - (Hh(i) + Hh(sh) - Hh(i * S + sh))


def evaluate(score):
    R = {}
    for n_ in ("ev", "ev7", "n03", "n10"):
        s1, s2 = score(n_)
        v1, I1 = torch.topk(s1, m, -1); v2, I2 = torch.topk(s2, m, -1)
        ii, jj, _ = pk_topk(v1, I1, v2, I2, S, km)
        R[n_] = ii * S + jj
        if n_ == "ev":
            dead = ((torch.bincount(I1.flatten(), minlength=S) == 0).float().mean()
                    + (torch.bincount(I2.flatten(), minlength=S) == 0).float().mean()) / 2
    return dict(dist=float(np.mean([torch.unique(R["ev"][s_]).numel() for s_ in SUBS])),
                dist7=float(np.mean([torch.unique(R["ev7"][s_]).numel() for s_ in SUBS])),
                distall=int(torch.unique(R["ev"]).numel()),
                top1=top1share(R["ev"]), dead=float(dead), mi_ij=mi_halves(R["ev"]),
                stab_n03=overlap(R["ev"], R["n03"]), stab_n10=overlap(R["ev"], R["n10"]),
                stab_drift=overlap(R["ev"], R["ev7"]),
                **{pn: overlap(R["ev"][a], R["ev"][b]) for pn, (a, b) in PAIRS.items()})


COLS = ["dist", "dist7", "distall", "top1", "dead", "mi_ij", "xdoc", "lag1", "lag16", "lag256", "stab_n03", "stab_n10", "stab_drift"]


def table(RES):
    print("| input | address | " + " | ".join(COLS) + " |")
    for key, v in RES.items():
        _, inp, ad = key.split("|")
        print(f"| {inp} | {ad} | " + " | ".join(f"{np.mean([x[c] for x in v]):.0f}" if c.startswith("dist") else f"{np.mean([x[c] for x in v]):.3f}" for c in COLS) + " |")
