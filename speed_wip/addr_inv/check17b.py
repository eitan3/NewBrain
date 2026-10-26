"""Plan 17 correction: the dynamic key methods re-scored with DE-DUPLICATED distinct rows (keys with cos > 0.95 merged
into one), plus distinct k-means (top-1 assignment) keys. ema input, plain query, pass p3. usage: check17b.py OUT.json L H"""
import os, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch, torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
from memory import pk_topk
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim2.py")).read()
fn = src[src.index("def sinkhorn_idx"):src.index("INPUTS = [")]
OUT, l, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
S, m, km, STEPS, BS, LR = 512, 8, 6, 2000, 512, 0.003
exec(fn)
D = os.path.dirname(os.path.abspath(__file__))
c6 = torch.load(os.path.join(D, "cap_6000.pt"), map_location="cpu", weights_only=False)
c7 = torch.load(os.path.join(D, "cap_b.pt"), map_location="cpu", weights_only=False)
st = c6["state"]
X = c6["X"]; B, T = X.shape; seq = torch.arange(B).repeat_interleave(T)
FIT = (seq < 16).nonzero().squeeze(1); EV = (seq >= 16).nonzero().squeeze(1); NE = EV.numel()
gen = torch.Generator().manual_seed(0)
SUBS = torch.randperm(NE, generator=gen)[:4000].view(4, 1000)
ci6 = [s["k"] for s in c6["sels"]].index(2); ci7 = [s["k"] for s in c7["sels"]].index(2)
a6 = c6["sels"][ci6]["a"][l].float(); a7 = c7["sels"][ci7]["a"][l].float()
nm = lambda a: (a - st["norm_m"][l]) / torch.sqrt(st["norm_v"][l] + 1e-5)
wq = st["wq_ema"][l, h]


def qp(x):
    q = F.normalize(nm(x) @ wq.T, dim=-1)
    return F.normalize(q[:, :64], dim=-1), F.normalize(q[:, 64:], dim=-1)


gn = torch.Generator().manual_seed(100 + l)
ae = a6[EV]
Qf, Qe, Q7 = qp(a6[FIT]), qp(ae), qp(a7[EV])
Qn = qp(ae + 0.03 * ae.norm(dim=-1, keepdim=True) * torch.randn(ae.shape, generator=gn) / math.sqrt(ae.size(1)))


def kmeans(q, K0, seed, r=0.1):
    g = torch.Generator().manual_seed(seed); K = K0.clone(); win = torch.zeros(S)
    for s in range(STEPS):
        b = q[torch.randint(q.size(0), (BS,), generator=g)]
        I = torch.topk(b @ K.T, m, -1).indices
        i1 = I[:, 0]
        sums = torch.zeros(S, 64).index_add_(0, i1, b); cnt = torch.bincount(i1, minlength=S).float(); has = cnt > 0
        K[has] = F.normalize((1 - r) * K[has] + r * F.normalize(sums[has] / cnt[has, None], dim=-1), dim=-1)
        win += torch.bincount(I.flatten(), minlength=S).float()
        if (s + 1) % 100 == 0:
            dead = (win == 0).nonzero().squeeze(1)
            if dead.numel():
                j = torch.randint(BS, (dead.numel(),), generator=g)
                K[dead] = F.normalize(b[j] + 0.05 * torch.randn(dead.numel(), 64, generator=g), dim=-1)
            win.zero_()
    return K


def clusters(K, thr=0.95):
    A = (K @ K.T) > thr
    lab = torch.arange(S)
    for _ in range(S):
        new = torch.where(A, lab[None, :], S).min(1).values
        new = torch.minimum(new, lab)
        if torch.equal(new, lab):
            break
        lab = new
    return lab


def address(q1, q2, k1, k2):
    v1, I1 = torch.topk(q1 @ k1.T, m, -1); v2, I2 = torch.topk(q2 @ k2.T, m, -1)
    ii, jj, _ = pk_topk(v1, I1, v2, I2, S, km)
    return ii, jj


def ovl(Ra, Rb):
    return float((Ra[:, :, None] == Rb[:, None, :]).any(-1).float().sum(-1).mean() / km)


gk = torch.Generator().manual_seed(1000 * l + h)
r1, r2 = F.normalize(torch.randn(S, 64, generator=gk), dim=-1), F.normalize(torch.randn(S, 64, generator=gk), dim=-1)
RES = {}
for meth in ["trained_static", "random_static", "kmeans_top1+restart", "attract", "fP", "C_restart", "D_sinkhorn", "F_gumbel"]:
    if meth == "trained_static":
        K1, K2 = F.normalize(st["k1_ema"][l, h], dim=-1), F.normalize(st["k2_ema"][l, h], dim=-1)
    elif meth == "random_static":
        K1, K2 = r1, r2
    elif meth.startswith("kmeans"):
        K1, K2 = kmeans(Qf[0], r1, 7 * l + h), kmeans(Qf[1], r2, 7 * l + h + 3)
    else:
        K1, K2 = dynamics(meth, Qf[0], r1, 7 * l + h), dynamics(meth, Qf[1], r2, 7 * l + h + 3)
    c1, c2 = clusters(K1), clusters(K2)
    out = {}
    for name, Q in (("e", Qe), ("n", Qn), ("7", Q7)):
        ii, jj = address(*Q, K1, K2)
        out[name] = (ii * S + jj, c1[ii] * S + c2[jj])
    raw, ded = out["e"]
    RES[meth] = dict(level=l, head=h, n_eff_keys=(int(torch.unique(c1).numel()) + int(torch.unique(c2).numel())) / 2,
                     dist=float(np.mean([torch.unique(raw[s]).numel() for s in SUBS])),
                     dist_dedup=float(np.mean([torch.unique(ded[s]).numel() for s in SUBS])),
                     stab_n03=ovl(raw, out["n"][0]), stab_n03_dedup=ovl(ded, out["n"][1]),
                     drift=ovl(raw, out["7"][0]), drift_dedup=ovl(ded, out["7"][1]))
    print(meth, RES[meth], flush=True)
json.dump(RES, open(OUT, "w"))
