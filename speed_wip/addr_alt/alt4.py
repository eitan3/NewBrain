"""Which refit hurts stability: the whitening statistics or the codebook? (plan_v2/18). Pass p3.
For pzca32 / pzca64 (+ fixed projection) and for the trained W_q queries (ema input), compare on the 7K thoughts:
  frozen     - 6K whitening + 6K codebook
  cb_refit   - 6K whitening, codebook warm-started from 6K and tracked for 5 Lloyd iterations on the 7K fit queries
  wh_refit   - whitening refit on 7K, 6K codebook kept
  wh_ema     - whitening statistics = 0.5/0.5 mix of the 6K and 7K covariance (a slower EMA), 6K codebook
  both       - both refit (= alt3's drift_refit)
overlap is always against the 6K-fitted addressing on the 6K thoughts."""
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


def stats(fit):
    mu = fit.mean(0); xc = fit - mu
    return mu, xc.T @ xc / xc.size(0)


def wfrom(mu, C, r):
    ev, V = torch.linalg.eigh(C)
    ev = ev.clamp(min=0) + 1e-6 * float(ev.max())
    return mu, V[:, -r:] @ torch.diag(ev[-r:] ** -0.5) @ V[:, -r:].T


def rows(s1, s2):
    v1, I1 = torch.topk(s1, m, -1); v2, I2 = torch.topk(s2, m, -1)
    ii, jj, _ = cm.pk_topk(v1, I1, v2, I2, S, km)
    return ii * S + jj


def lloyd(C, x, iters=5):
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
    s6, s7 = stats(a6[FIT]), stats(a7[FIT])
    for mode in ("wq", "pzca32", "pzca64"):
        if mode == "wq":
            Wh = {"6": (torch.zeros(a6.size(1)), torch.eye(a6.size(1)))}
            Wh["7"] = Wh["mix"] = Wh["6"]
        else:
            r = int(mode[-2:])
            Wh = {"6": wfrom(*s6, r), "7": wfrom(*s7, r), "mix": wfrom(0.5 * (s6[0] + s7[0]), 0.5 * (s6[1] + s7[1]), r)}
        for h in range(H):
            g = torch.Generator().manual_seed(1000 * l + 10 * h + 7)
            if mode == "wq":
                P = cm.st["wq_ema"][l, h].T
            else:
                P = torch.randn(a6.size(1), 128, generator=g)
            def qh(x, w):
                z = F.normalize(((x - w[0]) @ w[1]) @ P, dim=-1)
                return F.normalize(z[:, :64], dim=-1), F.normalize(z[:, 64:], dim=-1)
            qf6 = qh(a6[FIT], Wh["6"])
            K6 = tuple(cm.spherical_kmeans(qf6[i], S) for i in range(2))
            def addr(x, w, K):
                q = qh(x, w); return rows(q[0] @ K[0].T, q[1] @ K[1].T)
            R6 = addr(a6[EV], Wh["6"], K6)
            out = {}
            for var, wk, refit_cb in (("frozen", "6", False), ("cb_refit", "6", True), ("wh_refit", "7", False),
                                      ("wh_mix", "mix", False), ("both", "7", True)):
                if mode == "wq" and wk != "6":
                    continue
                K = K6
                if refit_cb:
                    qf7 = qh(a7[FIT], Wh[wk]); K = tuple(lloyd(K6[i], qf7[i]) for i in range(2))
                out[var] = cm.overlap(R6, addr(a7[EV], Wh[wk], K))
            out["dist"] = float(np.mean([torch.unique(R6[s_]).numel() for s_ in cm.SUBS]))
            RES.setdefault(mode, []).append(dict(level=l, head=h, **out))
        print(f"level {l} {mode} done {time.time() - t0:.0f}s", flush=True)
json.dump(RES, open(sys.argv[1], "w"))
cols = ["dist", "frozen", "cb_refit", "wh_refit", "wh_mix", "both"]
print("| input (k-means codebook) | " + " | ".join(cols) + " |")
for key, v in RES.items():
    print(f"| {key} | " + " | ".join((f"{np.mean([x[c] for x in v]):.0f}" if c == "dist" else f"{np.mean([x[c] for x in v]):.3f}") if c in v[0] else "-" for c in cols) + " |")
print("done %.0fs" % (time.time() - t0))
