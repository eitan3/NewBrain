"""Stability when the fitted pieces keep updating (plan_v2/18). Pass p3. Same protocol as alt.py.
The "drift" of alt.py keeps the 6K-fitted statistics / codebooks FROZEN and only moves the thoughts. Here the fitted
pieces are also re-fitted on the 7K thoughts (fit half), i.e. what an online (EMA) estimate would have done over
1,000 steps before the freeze:
  pca32 (truncated PCA basis, 32 output coords)   - naive refit of the eigenbasis (sign / order ambiguity)
  pzca32 (V_r L_r^-1/2 V_r^T, stays 448-d)        - the basis-free form of the same truncated whitening
  pzca64
then a FIXED random projection + random keys (rp_pk) or k-means codebooks (km_rp, warm-started from the 6K codebook
for 5 Lloyd iterations on the 7K fit queries, like an EMA codebook would track).
Reported: dist, stab_n03, drift_frozen (6K-fitted addressing on 7K thoughts), drift_refit (7K-refit addressing on 7K
thoughts vs 6K-fitted addressing on 6K thoughts)."""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
import importlib.util
spec = importlib.util.spec_from_file_location("altbase", os.path.join(os.path.dirname(os.path.abspath(__file__)), "alt_common.py"))
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
t0 = time.time()
S, m, km, L, H = cm.S, cm.m, cm.km, cm.L, cm.H
FIT, EV = cm.FIT, cm.EV


def whitener(fit, mode):
    mu = fit.mean(0); xc = fit - mu
    ev, V = torch.linalg.eigh(xc.T @ xc / xc.size(0))
    ev = ev.clamp(min=0) + 1e-6 * float(ev.max())
    r = int(mode[-2:])
    W = V[:, -r:] @ torch.diag(ev[-r:] ** -0.5)
    if mode.startswith("pzca"):
        W = W @ V[:, -r:].T
    return mu, W


def rows(s1, s2):
    v1, I1 = torch.topk(s1, m, -1); v2, I2 = torch.topk(s2, m, -1)
    ii, jj, _ = cm.pk_topk(v1, I1, v2, I2, S, km)
    return ii * S + jj


def lloyd(C, x, iters):
    x = F.normalize(x, dim=-1)
    for _ in range(iters):
        a = (x @ C.T).argmax(1)
        Cn = torch.zeros_like(C).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=C.size(0))
        C = torch.where(cnt[:, None] > 0, F.normalize(Cn, dim=-1), C)
    return C


RES = {}
k = 2
ci6 = [s["k"] for s in cm.c6["sels"]].index(k); ci7 = [s["k"] for s in cm.c7["sels"]].index(k)
for l in range(L):
    nm = lambda a: (a - cm.st["norm_m"][l]) / torch.sqrt(cm.st["norm_v"][l] + 1e-5)
    a6 = nm(cm.c6["sels"][ci6]["a"][l].float()); a7 = nm(cm.c7["sels"][ci7]["a"][l].float())
    gn = torch.Generator().manual_seed(100 + l)
    raw6 = cm.c6["sels"][ci6]["a"][l].float()[EV]
    n03 = nm(raw6 + 0.03 * raw6.norm(dim=-1, keepdim=True) * torch.randn(raw6.shape, generator=gn) / math.sqrt(raw6.size(1)))
    for mode in ("pca32", "pzca32", "pzca64"):
        mu6, W6 = whitener(a6[FIT], mode); mu7, W7 = whitener(a7[FIT], mode)
        x6, x7f, x7r, xn = (a6[EV] - mu6) @ W6, (a7[EV] - mu6) @ W6, (a7[EV] - mu7) @ W7, (n03 - mu6) @ W6
        f6, f7 = (a6[FIT] - mu6) @ W6, (a7[FIT] - mu7) @ W7
        din = x6.size(1)
        for h in range(H):
            g = torch.Generator().manual_seed(1000 * l + 10 * h + 7)
            P = torch.randn(din, 128, generator=g)
            qh = lambda x: (F.normalize(F.normalize(x @ P, dim=-1)[:, :64], dim=-1), F.normalize(F.normalize(x @ P, dim=-1)[:, 64:], dim=-1))
            rk = (F.normalize(torch.randn(S, 64, generator=g), dim=-1), F.normalize(torch.randn(S, 64, generator=g), dim=-1))
            for ad in ("rp_pk", "km_rp"):
                if ad == "rp_pk":
                    K6 = K7 = rk
                else:
                    qf6, qf7 = qh(f6), qh(f7)
                    K6 = tuple(cm.spherical_kmeans(qf6[i], S) for i in range(2))
                    K7 = tuple(lloyd(K6[i], qf7[i], 5) for i in range(2))
                sc = lambda x, K: (qh(x)[0] @ K[0].T, qh(x)[1] @ K[1].T)
                R6, R7f, R7r, Rn = rows(*sc(x6, K6)), rows(*sc(x7f, K6)), rows(*sc(x7r, K7)), rows(*sc(xn, K6))
                r = dict(dist=float(np.mean([torch.unique(R6[s_]).numel() for s_ in cm.SUBS])),
                         stab_n03=cm.overlap(R6, Rn), drift_frozen=cm.overlap(R6, R7f), drift_refit=cm.overlap(R6, R7r),
                         lag1=cm.overlap(R6[cm.PAIRS["lag1"][0]], R6[cm.PAIRS["lag1"][1]]))
                RES.setdefault(f"{mode}|{ad}", []).append(dict(level=l, head=h, **r))
        print(f"level {l} {mode} done {time.time() - t0:.0f}s", flush=True)
for l in range(L):  # the real trained addressing: captured 6K rows vs captured 7K rows (thoughts AND W_q/keys/norm moved)
    for h in range(H):
        R6 = cm.c6["sels"][ci6]["rows"][l, h].long()[EV]; R7 = cm.c7["sels"][ci7]["rows"][l, h].long()[EV]
        RES.setdefault("ema|trained(real captures)", []).append(dict(level=l, head=h, dist=float(np.mean([torch.unique(R6[s_]).numel() for s_ in cm.SUBS])),
                       stab_n03=float("nan"), drift_frozen=float("nan"), drift_refit=cm.overlap(R6, R7),
                       lag1=cm.overlap(R6[cm.PAIRS["lag1"][0]], R6[cm.PAIRS["lag1"][1]])))
json.dump(RES, open(sys.argv[1], "w"))
print("| input | address | dist | lag1 | stab_n03 | drift_frozen | drift_refit |")
for key, v in RES.items():
    inp, ad = key.split("|")
    print(f"| {inp} | {ad} | {np.mean([x['dist'] for x in v]):.0f} | " + " | ".join(f"{np.mean([x[c] for x in v]):.3f}" for c in ("lag1", "stab_n03", "drift_frozen", "drift_refit")) + " |")
print("done %.0fs" % (time.time() - t0))
