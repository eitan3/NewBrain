"""Neighbour writes: the share of a position's 7K top-6 rows that were inside its 6K top-K' pairs (K' = 6/8/12/16),
i.e. what writing to the 7th..K'-th candidates would carry across the real drift. Frozen keys."""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
from memory import pk_topk
src = open("check17b.py").read()
c6 = torch.load("cap_6000.pt", map_location="cpu", weights_only=False); c7 = torch.load("cap_b.pt", map_location="cpu", weights_only=False)
st = c6["state"]; m = 8; STEPS, BS = 2000, 512
X = c6["X"]; B, T = X.shape; seq = torch.arange(B).repeat_interleave(T)
FIT = (seq < 16).nonzero().squeeze(1); EV = (seq >= 16).nonzero().squeeze(1)
ci6 = [s["k"] for s in c6["sels"]].index(2); ci7 = [s["k"] for s in c7["sels"]].index(2)
def kmeans(q, K0, seed, S, r=0.1):
    g = torch.Generator().manual_seed(seed); K = K0.clone(); win = torch.zeros(S)
    for s in range(STEPS):
        b = q[torch.randint(q.size(0), (BS,), generator=g)]
        I = torch.topk(b @ K.T, m, -1).indices; i1 = I[:, 0]
        sums = torch.zeros(S, 64).index_add_(0, i1, b); cnt = torch.bincount(i1, minlength=S).float(); has = cnt > 0
        K[has] = F.normalize((1 - r) * K[has] + r * F.normalize(sums[has] / cnt[has, None], dim=-1), dim=-1)
        win += torch.bincount(I.flatten(), minlength=S).float()
        if (s + 1) % 100 == 0:
            dead = (win == 0).nonzero().squeeze(1)
            if dead.numel():
                j = torch.randint(BS, (dead.numel(),), generator=g); K[dead] = F.normalize(b[j] + 0.05 * torch.randn(dead.numel(), 64, generator=g), dim=-1)
            win.zero_()
    return K
def rows(q1, q2, k1, k2, kk):
    S = k1.size(0); mm = max(m, int(math.ceil(math.sqrt(kk))) + 2)
    v1, I1 = torch.topk(q1 @ k1.T, mm, -1); v2, I2 = torch.topk(q2 @ k2.T, mm, -1)
    ii, jj, _ = pk_topk(v1, I1, v2, I2, S, kk)
    return ii * S + jj
RES = {}
for l in (0, 3):
    a6 = c6["sels"][ci6]["a"][l].float(); a7 = c7["sels"][ci7]["a"][l].float()
    nm = lambda a: (a - st["norm_m"][l]) / torch.sqrt(st["norm_v"][l] + 1e-5)
    for h in range(3):
        wq = st["wq_ema"][l, h]
        def qp(x):
            q = F.normalize(nm(x) @ wq.T, dim=-1); return F.normalize(q[:, :64], dim=-1), F.normalize(q[:, 64:], dim=-1)
        Qf, Qe, Q7 = qp(a6[FIT]), qp(a6[EV]), qp(a7[EV])
        for name, S in (("today_trained", 512), ("kmeans512", 512), ("kmeans64", 64)):
            if name == "today_trained":
                K = (F.normalize(st["k1_ema"][l, h], dim=-1), F.normalize(st["k2_ema"][l, h], dim=-1))
            else:
                g = torch.Generator().manual_seed(1000 * l + h + S)
                K = tuple(kmeans(Qf[hf], F.normalize(torch.randn(S, 64, generator=g), dim=-1), 7 * l + h + hf, S) for hf in (0, 1))
            R7 = rows(*Q7, *K, 6)
            for kk in (6, 8, 12, 16, 32):
                R6 = rows(*Qe, *K, kk)
                v = float((R7[:, :, None] == R6[:, None, :]).any(-1).float().mean())
                RES.setdefault(f"{name}|top{kk}", []).append(v)
        print(l, h, flush=True)
for k, v in RES.items():
    print(k, round(float(np.mean(v)), 3))
json.dump(RES, open("nbr.json", "w"))
