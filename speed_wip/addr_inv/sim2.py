"""plan_v2/17 extension: query BatchNorm (A), EMA k-means keys (B), dead-key restart (C), Sinkhorn selection (D),
Gumbel selection noise (F), the read-pull baseline and the f.P loss, each with input norm ema / ZCA (W) / shrink0.1
(partial W). Dynamic key methods start from RANDOM unit keys (a fresh run) and stream over the fit queries (seqs 0-15
of the 6K capture) for STEPS batches of BS; scoring as in sim.py on seqs 16-31 with plain top-m selection (the
post-infancy / post-freeze addressing). usage: sim2.py OUT.json [pass=2] [levels=0,3] [steps=1500]"""
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
KP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
LEVELS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,3").split(",")]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
BS, LR = 512, 0.003
D = os.path.dirname(os.path.abspath(__file__))
c6 = torch.load(os.path.join(D, "cap_6000.pt"), map_location="cpu", weights_only=False)
c7 = torch.load(os.path.join(D, "cap_b.pt"), map_location="cpu", weights_only=False)
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
seq_e, doc_e = seq[EV], doc[EV]


def lag_pairs(lag, n=4000):
    a = torch.randint(NE, (n * 3,), generator=gen); b = a + lag
    ok = b < NE; a, b = a[ok], b[ok]
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
    mu = x.mean(0); xc = x - mu
    C = xc.T @ xc / x.size(0); d = C.size(0)
    if mode.startswith("shrink"):
        lam_ = float(mode[6:]); C = (1 - lam_) * C + lam_ * torch.trace(C) / d * torch.eye(d)
    ev, V = torch.linalg.eigh(C)
    ev = ev.clamp(min=0) + 1e-6 * float(ev.max())
    return mu, V @ torch.diag(ev ** -0.5) @ V.T


def sinkhorn_idx(sc, eps=0.05, iters=3):
    Q = torch.exp((sc - sc.max()) / eps)
    for _ in range(iters):
        Q = Q / Q.sum(0, keepdim=True)    # every key gets the same mass
        Q = Q / Q.sum(1, keepdim=True)    # every token sums to 1
    return torch.topk(Q, m, -1).indices


def dynamics(method, q, K0, gseed):
    """Stream the fit queries q (n, 64) in batches; return the final normalised keys (S, 64)."""
    g = torch.Generator().manual_seed(gseed)
    order = torch.randperm(q.size(0), generator=g)
    n = q.size(0)
    if method == "B_ema_kmeans":
        K = K0.clone()
        for s in range(STEPS):
            b = q[order[(torch.arange(BS) + s * BS) % n]]
            I = torch.topk(b @ K.T, m, -1).indices.flatten()
            src = b.repeat_interleave(m, 0)
            sums = torch.zeros(S, 64).index_add_(0, I, src)
            cnt = torch.bincount(I, minlength=S).float()
            has = cnt > 0
            K[has] = F.normalize(0.99 * K[has] + 0.01 * sums[has] / cnt[has, None], dim=-1)
        return K
    P_ = K0.clone().requires_grad_(True)
    opt = torch.optim.Adam([P_], lr=LR)
    win = torch.zeros(S)
    for s in range(STEPS):
        b = q[order[(torch.arange(BS) + s * BS) % n]]
        Kn = F.normalize(P_, dim=-1)
        sc = b @ Kn.T
        with torch.no_grad():
            if method == "D_sinkhorn":
                I = sinkhorn_idx(sc.detach())
            elif method == "F_gumbel":
                gum = -torch.log(-torch.log(torch.rand(sc.shape, generator=g).clamp(1e-9, 1 - 1e-9)))
                I = torch.topk(sc.detach() + 0.05 * gum, m, -1).indices
            else:
                I = torch.topk(sc.detach(), m, -1).indices
        loss = -sc.gather(1, I).mean()                          # the read's pull on the selected keys
        if method == "fP":
            f = torch.bincount(I.flatten(), minlength=S).float() / I.numel()
            loss = loss + 0.1 * S * (f * torch.softmax(sc / 0.1, -1).mean(0)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if method == "C_restart":
            win += torch.bincount(I.flatten(), minlength=S).float()
            if (s + 1) % 100 == 0:
                dead = (win == 0).nonzero().squeeze(1)
                if dead.numel():
                    j = torch.randint(BS, (dead.numel(),), generator=g)
                    with torch.no_grad():
                        P_[dead] = F.normalize(b[j] + 0.05 * torch.randn(dead.numel(), 64, generator=g), dim=-1)
                    # fresh Adam moments for the restarted keys
                    stt = opt.state[P_]
                    stt["exp_avg"][dead] = 0; stt["exp_avg_sq"][dead] = 0
                win.zero_()
    return F.normalize(P_.detach(), dim=-1)


INPUTS = ["ema", "zca", "shrink0.1"]
QT = ["plain", "BN"]
KEYS = ["trained_static", "random_static", "attract", "fP", "B_ema_kmeans", "C_restart", "D_sinkhorn", "F_gumbel"]
RES = {}
ci6 = [s["k"] for s in c6["sels"]].index(KP); ci7 = [s["k"] for s in c7["sels"]].index(KP)
for l in LEVELS:
    a6 = c6["sels"][ci6]["a"][l].float(); a7 = c7["sels"][ci7]["a"][l].float()
    nm = lambda a: (a - st["norm_m"][l]) / torch.sqrt(st["norm_v"][l] + 1e-5)
    gn = torch.Generator().manual_seed(100 + l)
    ae = a6[EV]
    noise = {r: ae + r * ae.norm(dim=-1, keepdim=True) * torch.randn(ae.shape, generator=gn) / math.sqrt(ae.size(1)) for r in (0.03, 0.10)}
    raw = {"fit": nm(a6[FIT]), "ev": nm(ae), "ev7": nm(a7[EV]), "n03": nm(noise[0.03]), "n10": nm(noise[0.10])}
    for inp in INPUTS:
        if inp == "ema":
            sets = raw
        else:
            mu, W = whitener(raw["fit"], inp)
            sets = {k_: (x - mu) @ W for k_, x in raw.items()}
        for h in range(H):
            wq = st["wq_ema"][l, h]
            qr = {k_: x @ wq.T for k_, x in sets.items()}
            for qt in QT:
                if qt == "BN":   # BatchNorm on the raw query (running stats = the fit half), before the split
                    mu_q, sd_q = qr["fit"].mean(0), qr["fit"].std(0) + 1e-5
                    qq = {k_: (x - mu_q) / sd_q for k_, x in qr.items()}
                else:
                    qq = {k_: F.normalize(x, dim=-1) for k_, x in qr.items()}
                Q = {k_: (F.normalize(x[:, :64], dim=-1), F.normalize(x[:, 64:], dim=-1)) for k_, x in qq.items()}
                gk = torch.Generator().manual_seed(1000 * l + h)
                r1, r2 = F.normalize(torch.randn(S, 64, generator=gk), dim=-1), F.normalize(torch.randn(S, 64, generator=gk), dim=-1)
                for kv in KEYS:
                    if kv == "trained_static":
                        K1, K2 = F.normalize(st["k1_ema"][l, h], dim=-1), F.normalize(st["k2_ema"][l, h], dim=-1)
                    elif kv == "random_static":
                        K1, K2 = r1, r2
                    else:
                        K1 = dynamics(kv, Q["fit"][0], r1, 7 * l + h)
                        K2 = dynamics(kv, Q["fit"][1], r2, 7 * l + h + 3)
                    R = {}
                    for k_ in ("ev", "ev7", "n03", "n10"):
                        I1, I2, R[k_] = address(*Q[k_], K1, K2)
                        if k_ == "ev":
                            dead = ((torch.bincount(I1.flatten(), minlength=S) == 0).float().mean()
                                    + (torch.bincount(I2.flatten(), minlength=S) == 0).float().mean()) / 2
                    c = torch.bincount(R["ev"].flatten(), minlength=S * S).double()
                    r = dict(dist=float(np.mean([torch.unique(R["ev"][s_]).numel() for s_ in SUBS])),
                             dist7=float(np.mean([torch.unique(R["ev7"][s_]).numel() for s_ in SUBS])),
                             top1=float(torch.sort(c, descending=True).values[: int(0.01 * S * S)].sum() / c.sum()),
                             dead=float(dead),
                             score=float(torch.topk(Q["ev"][0] @ K1.T, m, -1).values.mean() + torch.topk(Q["ev"][1] @ K2.T, m, -1).values.mean()) / 2,
                             stab_n03=overlap(R["ev"], R["n03"]), stab_n10=overlap(R["ev"], R["n10"]), stab_drift=overlap(R["ev"], R["ev7"]),
                             **{pn: overlap(R["ev"][a], R["ev"][b]) for pn, (a, b) in PAIRS.items()})
                    RES.setdefault(f"{KP}|{inp}|{qt}|{kv}", []).append(dict(level=l, head=h, **r))
            log(f"level {l} input {inp} head {h} done {time.time() - t0:.0f}s")
        json.dump(RES, open(OUTF, "w"))
log("done %.0fs" % (time.time() - t0))
