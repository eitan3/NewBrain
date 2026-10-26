"""Offline simulation of address fixes on the captured 6K / 7K thoughts (plan_v2/17). CPU, 4 threads.
usage: sim.py OUT.json [passes=2] [tune]
Pipelines: norm (ema | zca | pow0.5 | shrink0.1 | shrink0.3) x halves (plain | whiten_q | rot_q) x keys
(trained | random | kpp_reseed | fP_opt | attract_only[ema/plain only]). Statistics, re-seeding and key optimisation are fit
on the 6K thoughts of sequences 0-15; everything is evaluated on sequences 16-31 (6K thoughts; 7K thoughts with the same
6K-fitted addressing; noisy 6K thoughts)."""
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
PASSES = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2").split(",")]
TUNE = len(sys.argv) > 3 and sys.argv[3] == "tune"
D = os.path.dirname(os.path.abspath(__file__))
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

# evaluation pairs (indices into EV)
seq_e, pos_e, doc_e = seq[EV], pos[EV], doc[EV]
def lag_pairs(lag, n=4000):
    a = torch.randint(NE, (n * 3,), generator=gen)
    b = a + lag
    ok = (b < NE)
    a, b = a[ok], b[ok]
    ok = (seq_e[a] == seq_e[b]) & (doc_e[a] == doc_e[b])
    return a[ok][:n], b[ok][:n]
PAIRS = {f"lag{g}": lag_pairs(g) for g in (1, 16, 256)}
a_ = torch.randint(NE, (8000,), generator=gen); b_ = torch.randint(NE, (8000,), generator=gen)
ok = seq_e[a_] != seq_e[b_]; PAIRS["xdoc"] = (a_[ok][:4000], b_[ok][:4000])
SUBS = torch.randperm(NE, generator=gen)[:4000].view(4, 1000)


def address(q1, q2, k1, k2):
    v1, I1 = torch.topk(q1 @ k1.T, m, -1)
    v2, I2 = torch.topk(q2 @ k2.T, m, -1)
    ii, jj, _ = pk_topk(v1, I1, v2, I2, S, km)
    return I1, I2, ii * S + jj


def overlap(Ra, Rb):
    return float((Ra[:, :, None] == Rb[:, None, :]).any(-1).float().sum(-1).mean() / km)


def whitener(x, mode):
    mu = x.mean(0)
    xc = x - mu
    C = xc.T @ xc / x.size(0)
    d = C.size(0)
    if mode.startswith("shrink"):
        lam_ = float(mode[6:])
        C = (1 - lam_) * C + lam_ * torch.trace(C) / d * torch.eye(d)
    ev, V = torch.linalg.eigh(C)
    ev = ev.clamp(min=0) + 1e-6 * float(ev.max())
    p = 0.25 if mode == "pow0.5" else 0.5
    W = V @ torch.diag(ev ** -p) @ V.T
    if mode == "pow0.5":  # keep the total variance comparable (trace of the output = trace of the input)
        W = W * math.sqrt(float(torch.trace(C)) / float((ev ** 0.5).sum()))
    return mu, W


def kpp_reseed(K, q, used):
    """k-means++-style: dead keys (unused on the fit half) are re-seeded one at a time from the fit queries, with
    probability proportional to (1 - max cos to the current live keys)^2."""
    K = K.clone()
    dead = (~used).nonzero().squeeze(1)
    if not dead.numel():
        return K
    live = K[used]
    best = (q @ live.T).max(1).values
    for i in dead.tolist():
        w = (1 - best).clamp(min=0) ** 2
        j = int(torch.multinomial(w + 1e-12, 1, generator=gen))
        K[i] = F.normalize(q[j] + 0.05 * torch.randn(64, generator=gen), dim=0)
        best = torch.maximum(best, q @ K[i])
    return K


def optimise(K1, K2, q1, q2, lam_u, steps=100, lr=0.01, tau_u=0.1, nsub=4096):
    """Simulated infancy key training on the fit queries: maximise the scores of each query's top-m sub-keys (the
    read's pull) + lam_u * S * sum_i f_i P_i (plan 15 (a)), per half; keys renormalised in the forward."""
    out = []
    for K, q in ((K1, q1), (K2, q2)):
        P_ = K.clone().requires_grad_(True)
        opt = torch.optim.Adam([P_], lr=lr)
        for s in range(steps):
            idx = torch.randint(q.size(0), (nsub,), generator=gen)
            qq = q[idx]
            Kn = F.normalize(P_, dim=-1)
            sc = qq @ Kn.T
            v, I = torch.topk(sc.detach(), m, -1)
            read = -sc.gather(1, I).mean()
            loss = read
            if lam_u:
                f = torch.bincount(I.flatten(), minlength=S).float() / I.numel()
                Pm = torch.softmax(sc / tau_u, -1).mean(0)
                loss = loss + lam_u * S * (f * Pm).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        out.append(F.normalize(P_.detach(), dim=-1))
    return out


def ustats_rows(R):
    c = torch.bincount(R.flatten(), minlength=S * S).double()
    top = torch.sort(c, descending=True).values[: int(0.01 * S * S)].sum() / c.sum()
    return float(top)


RES = {}
rk = {(l, h): (F.normalize(torch.randn(S, 64, generator=gen), dim=-1), F.normalize(torch.randn(S, 64, generator=gen), dim=-1))
      for l in range(L) for h in range(H)}
rot = {(l, h): torch.linalg.qr(torch.randn(128, 128, generator=gen))[0] for l in range(L) for h in range(H)}
NORMS = ["ema", "zca", "pow0.5", "shrink0.1", "shrink0.3"]
HALVES = ["plain", "whiten_q", "rot_q"]
KEYS = ["trained", "random", "kpp_reseed", "fP_opt", "attract_only"]
LAM_U = float(os.environ.get("LAM_U", "1.0"))
if TUNE:
    NORMS, HALVES = ["ema"], ["plain"]
for k in PASSES:
    ci6 = [s["k"] for s in c6["sels"]].index(k); ci7 = [s["k"] for s in c7["sels"]].index(k)
    for l in range(L if not TUNE else 1):
        a6 = c6["sels"][ci6]["a"][l].float(); a7 = c7["sels"][ci7]["a"][l].float()
        nm = lambda a: (a - st["norm_m"][l]) / torch.sqrt(st["norm_v"][l] + 1e-5)
        gn = torch.Generator().manual_seed(100 + l)
        ae = a6[EV]
        noise = {r: ae + r * ae.norm(dim=-1, keepdim=True) * torch.randn(ae.shape, generator=gn) / math.sqrt(ae.size(1)) for r in (0.03, 0.10)}
        sets_raw = {"fit": nm(a6[FIT]), "ev": nm(ae), "ev7": nm(a7[EV]), "n03": nm(noise[0.03]), "n10": nm(noise[0.10])}
        for norm in NORMS:
            if norm == "ema":
                sets = sets_raw
            else:
                mu, W = whitener(sets_raw["fit"], norm)
                sets = {n_: (x - mu) @ W for n_, x in sets_raw.items()}
            for h in range(H if not TUNE else 1):
                wq = st["wq_ema"][l, h]
                qn = {n_: F.normalize(x @ wq.T, dim=-1) for n_, x in sets.items()}
                for hv in HALVES:
                    if hv == "plain":
                        qq = qn
                    elif hv == "rot_q":
                        qq = {n_: x @ rot[(l, h)] for n_, x in qn.items()}
                    else:
                        mu_q = qn["fit"].mean(0); xc = qn["fit"] - mu_q
                        ev_, V_ = torch.linalg.eigh(xc.T @ xc / xc.size(0))
                        ev_ = ev_.clamp(min=0) + 1e-6 * float(ev_.max())
                        Wq = V_ @ torch.diag(ev_ ** -0.5) @ V_.T
                        qq = {n_: (x - mu_q) @ Wq for n_, x in qn.items()}
                    Q = {n_: (F.normalize(x[:, :64], dim=-1), F.normalize(x[:, 64:], dim=-1)) for n_, x in qq.items()}
                    k1t, k2t = F.normalize(st["k1_ema"][l, h], dim=-1), F.normalize(st["k2_ema"][l, h], dim=-1)
                    for kv in KEYS:
                        if kv == "attract_only" and not (norm == "ema" and hv == "plain"):
                            continue
                        if kv == "trained":
                            K1, K2 = k1t, k2t
                        elif kv == "random":
                            K1, K2 = rk[(l, h)]
                        elif kv == "kpp_reseed":
                            I1, I2, _ = address(*Q["fit"], k1t, k2t)
                            u1 = torch.bincount(I1.flatten(), minlength=S) > 0; u2 = torch.bincount(I2.flatten(), minlength=S) > 0
                            K1, K2 = kpp_reseed(k1t, Q["fit"][0], u1), kpp_reseed(k2t, Q["fit"][1], u2)
                        elif kv == "fP_opt":
                            K1, K2 = optimise(k1t, k2t, *Q["fit"], LAM_U)
                        else:
                            K1, K2 = optimise(k1t, k2t, *Q["fit"], 0.0)
                        R = {}
                        for n_ in ("ev", "ev7", "n03", "n10"):
                            I1, I2, R[n_] = address(*Q[n_], K1, K2)
                            if n_ == "ev":
                                dead = ((torch.bincount(I1.flatten(), minlength=S) == 0).float().mean()
                                        + (torch.bincount(I2.flatten(), minlength=S) == 0).float().mean()) / 2
                        # the fit-half read score (how well the keys match the queries)
                        If1 = torch.topk(Q["ev"][0] @ K1.T, m, -1).values.mean() + torch.topk(Q["ev"][1] @ K2.T, m, -1).values.mean()
                        r = dict(dist=float(np.mean([torch.unique(R["ev"][s_]).numel() for s_ in SUBS])),
                                 dist7=float(np.mean([torch.unique(R["ev7"][s_]).numel() for s_ in SUBS])),
                                 top1=ustats_rows(R["ev"]), dead=float(dead), score=float(If1) / 2,
                                 stab_n03=overlap(R["ev"], R["n03"]), stab_n10=overlap(R["ev"], R["n10"]),
                                 stab_drift=overlap(R["ev"], R["ev7"]),
                                 **{pn: overlap(R["ev"][a], R["ev"][b]) for pn, (a, b) in PAIRS.items()})
                        RES.setdefault(f"{k}|{norm}|{hv}|{kv}", []).append(dict(level=l, head=h, **r))
            log(f"pass {k} level {l} norm {norm} done {time.time() - t0:.0f}s")
        if TUNE:
            for key, v in RES.items():
                log(key, {kk: round(float(np.mean([x[kk] for x in v])), 3) for kk in v[0] if kk not in ("level", "head")})
json.dump(RES, open(OUTF, "w"))
log("done %.0fs" % (time.time() - t0))
