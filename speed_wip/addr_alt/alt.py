"""Learning-free / structural address alternatives on the captured 6K / 7K thoughts (plan_v2/18). CPU, 4 threads.
usage: python -B alt.py OUT.json [pass=2]
Same protocol as addr_inv/sim.py (so the numbers are comparable with plan 17's table): everything that is fit (input
whitening statistics, k-means codebooks, bit thresholds / quantiles) is fit on the 6K thoughts of sequences 0-15 and then
FROZEN; evaluation on sequences 16-31: 6K thoughts ("ev"), the 7K thoughts of the same tokens through the 6K-fitted
addressing ("ev7" = real drift), and 6K thoughts + 3% / 10% relative Gaussian noise on a.

inputs  : ema (the current per-coordinate EMA norm a_hat) | pca{r} (PCA-whitening of a_hat truncated to the top r
          directions) | zca (full whitening, reference)
address : trained   - the run's W_q + trained EMA sub-keys (ema input only; reference)
          rp_pk     - fixed Gaussian projection (in -> 128), halves normalised, fixed random unit sub-keys (no learning)
          simhash   - per half 9 fixed random hyperplanes, thresholds at the fit median; the 512 buckets = sign codes;
                      multi-probe score z . v over the 512 hypercube vertices v (= a product key whose keys are fixed)
          fsq       - per half 3 fixed random orthogonal directions, each mapped to its fit-quantile u in (0,1) and cut
                      into 8 equal-mass levels (8^3 = 512 codes); score -|u - c|^2 to the code centres
          km_wq     - spherical k-means codebooks (S=512 per half) on the trained-W_q queries (keys = centroids)
          km_rp     - spherical k-means codebooks on the rp_pk queries
          kmbal_rp  - the same with balanced (Sinkhorn) assignment during the fit, nearest-centroid lookup after
"""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
from memory import pk_topk

t0 = time.time()
OUTF = sys.argv[1]
PASS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
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
log = lambda *a: print(*a, flush=True)

seq_e, pos_e, doc_e = seq[EV], pos[EV], doc[EV]
def lag_pairs(lag, n=4000):
    a = torch.randint(NE, (n * 3,), generator=gen); b = a + lag
    ok = (b < NE); a, b = a[ok], b[ok]
    ok = (seq_e[a] == seq_e[b]) & (doc_e[a] == doc_e[b])
    return a[ok][:n], b[ok][:n]
PAIRS = {f"lag{g}": lag_pairs(g) for g in (1, 16, 256)}
a_ = torch.randint(NE, (8000,), generator=gen); b_ = torch.randint(NE, (8000,), generator=gen)
ok = seq_e[a_] != seq_e[b_]; PAIRS["xdoc"] = (a_[ok][:4000], b_[ok][:4000])
SUBS = torch.randperm(NE, generator=gen)[:4000].view(4, 1000)

# the 512 hypercube vertices (9 bits) and the 8^3 FSQ code centres
VERT = torch.tensor([[1.0 if (c >> b) & 1 else -1.0 for b in range(9)] for c in range(512)])
CEN = torch.tensor([[(((c >> (3 * d)) & 7) + 0.5) / 8 for d in range(3)] for c in range(512)])


def topm_rows(s1, s2):
    """per-half score matrices (n, 512) -> top-m per half, product top-km rows, per-half top-m index lists."""
    v1, I1 = torch.topk(s1, m, -1); v2, I2 = torch.topk(s2, m, -1)
    ii, jj, _ = pk_topk(v1, I1, v2, I2, S, km)
    return I1, I2, ii * S + jj


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


def spherical_kmeans(x, k, iters=25, balanced=False):
    x = F.normalize(x, dim=-1)
    C = x[torch.randperm(x.size(0), generator=gen)[:k]].clone()
    for _ in range(iters):
        sc = x @ C.T
        if balanced:  # Sinkhorn equal-mass assignment (SwAV style), eps 0.05
            Q = torch.exp((sc - sc.max()) / 0.05)
            for _ in range(5):
                Q = Q / Q.sum(0, keepdim=True); Q = Q / Q.sum(1, keepdim=True)
            a = Q.argmax(1)
        else:
            a = sc.argmax(1)
        Cn = torch.zeros_like(C).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k)
        empty = (cnt == 0).nonzero().squeeze(1)
        if empty.numel():  # re-seed empty clusters at the worst-fit points
            worst = sc.max(1).values.argsort()[: empty.numel()]
            Cn[empty] = x[worst]
        C = F.normalize(Cn, dim=-1)
    return C


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
        W = V[:, -r:] @ torch.diag(ev[-r:] ** -0.5)        # top-r PCA directions, whitened (r dims)
    return {n_: (x - mu) @ W for n_, x in sets_raw.items()}


RES = {}
k = PASS
ci6 = [s["k"] for s in c6["sels"]].index(k); ci7 = [s["k"] for s in c7["sels"]].index(k)
INPUTS = ["ema", "pca16", "pca32", "pca64", "zca"]
for l in range(L):
    a6 = c6["sels"][ci6]["a"][l].float(); a7 = c7["sels"][ci7]["a"][l].float()
    nm = lambda a: (a - st["norm_m"][l]) / torch.sqrt(st["norm_v"][l] + 1e-5)
    gn = torch.Generator().manual_seed(100 + l)
    ae = a6[EV]
    noise = {r: ae + r * ae.norm(dim=-1, keepdim=True) * torch.randn(ae.shape, generator=gn) / math.sqrt(ae.size(1)) for r in (0.03, 0.10)}
    sets_raw = {"fit": nm(a6[FIT]), "ev": nm(ae), "ev7": nm(a7[EV]), "n03": nm(noise[0.03]), "n10": nm(noise[0.10])}
    for inp in INPUTS:
        sets = make_inputs(sets_raw, inp)
        din = sets["fit"].size(1)
        for h in range(H):
            g = torch.Generator().manual_seed(1000 * l + 10 * h + 7)
            P = torch.randn(din, 128, generator=g)                                   # rp_pk projection
            rk1 = F.normalize(torch.randn(S, 64, generator=g), dim=-1); rk2 = F.normalize(torch.randn(S, 64, generator=g), dim=-1)
            Hp = torch.randn(din, 18, generator=g)                                   # simhash hyperplanes (9 per half)
            Fq = torch.linalg.qr(torch.randn(din, max(din, 6), generator=g))[0][:, :6] if din >= 6 else None  # fsq dirs
            # fitted (then frozen) pieces
            hp_fit = sets["fit"] @ Hp
            hp_med = hp_fit.median(0).values; hp_sd = hp_fit.std(0)
            fq_fit = torch.sort(sets["fit"] @ Fq, 0).values                          # quantile tables (n_fit, 6)
            qrp = {n_: F.normalize(x @ P, dim=-1) for n_, x in sets.items()}
            Qrp = {n_: (F.normalize(x[:, :64], dim=-1), F.normalize(x[:, 64:], dim=-1)) for n_, x in qrp.items()}
            addrs = ["rp_pk", "simhash", "fsq", "km_rp", "kmbal_rp"]
            if inp == "ema":
                addrs = ["trained", "km_wq"] + addrs
                wq = st["wq_ema"][l, h]
                qw = {n_: F.normalize(x @ wq.T, dim=-1) for n_, x in sets.items()}
                Qw = {n_: (F.normalize(x[:, :64], dim=-1), F.normalize(x[:, 64:], dim=-1)) for n_, x in qw.items()}
            for ad in addrs:
                if ad in ("trained", "km_wq", "rp_pk", "km_rp", "kmbal_rp"):
                    Q = Qw if ad in ("trained", "km_wq") else Qrp
                    if ad == "trained":
                        K1, K2 = F.normalize(st["k1_ema"][l, h], dim=-1), F.normalize(st["k2_ema"][l, h], dim=-1)
                    elif ad == "rp_pk":
                        K1, K2 = rk1, rk2
                    else:
                        bal = ad.startswith("kmbal")
                        K1 = spherical_kmeans(Q["fit"][0], S, balanced=bal); K2 = spherical_kmeans(Q["fit"][1], S, balanced=bal)
                    score = lambda n_: (Q[n_][0] @ K1.T, Q[n_][1] @ K2.T)
                elif ad == "simhash":
                    def score(n_):
                        z = (sets[n_] @ Hp - hp_med) / hp_sd
                        return z[:, :9] @ VERT.T, z[:, 9:] @ VERT.T
                else:  # fsq on fit-quantiles
                    def score(n_):
                        p = sets[n_] @ Fq
                        u = torch.stack([torch.searchsorted(fq_fit[:, d].contiguous(), p[:, d].contiguous()).float()
                                         for d in range(6)], 1) / fq_fit.size(0)
                        u = u.clamp(1e-4, 1 - 1e-4)
                        s1 = -((u[:, None, :3] - CEN[None]) ** 2).sum(-1)
                        s2 = -((u[:, None, 3:] - CEN[None]) ** 2).sum(-1)
                        return s1, s2
                R = {}
                for n_ in ("ev", "ev7", "n03", "n10"):
                    I1, I2, R[n_] = topm_rows(*score(n_))
                    if n_ == "ev":
                        dead = ((torch.bincount(I1.flatten(), minlength=S) == 0).float().mean()
                                + (torch.bincount(I2.flatten(), minlength=S) == 0).float().mean()) / 2
                r = dict(dist=float(np.mean([torch.unique(R["ev"][s_]).numel() for s_ in SUBS])),
                         dist7=float(np.mean([torch.unique(R["ev7"][s_]).numel() for s_ in SUBS])),
                         distall=int(torch.unique(R["ev"]).numel()),
                         top1=top1share(R["ev"]), dead=float(dead), mi_ij=mi_halves(R["ev"]),
                         stab_n03=overlap(R["ev"], R["n03"]), stab_n10=overlap(R["ev"], R["n10"]),
                         stab_drift=overlap(R["ev"], R["ev7"]),
                         **{pn: overlap(R["ev"][a], R["ev"][b]) for pn, (a, b) in PAIRS.items()})
                RES.setdefault(f"{k}|{inp}|{ad}", []).append(dict(level=l, head=h, **r))
        log(f"level {l} input {inp} done {time.time() - t0:.0f}s")
    json.dump(RES, open(OUTF, "w"))
cols = ["dist", "dist7", "distall", "top1", "dead", "mi_ij", "xdoc", "lag1", "lag16", "lag256", "stab_n03", "stab_n10", "stab_drift"]
print("| input | address | " + " | ".join(cols) + " |")
for key, v in RES.items():
    _, inp, ad = key.split("|")
    print(f"| {inp} | {ad} | " + " | ".join(f"{np.mean([x[c] for x in v]):.0f}" if c.startswith("dist") else f"{np.mean([x[c] for x in v]):.3f}" for c in cols) + " |")
log("done %.0fs" % (time.time() - t0))
