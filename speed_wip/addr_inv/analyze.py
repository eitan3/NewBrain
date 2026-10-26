"""Offline analysis of a capture (capture.py). usage: analyze.py CAP.pt OUT.json [CAP_PREV.pt]
CPU, 4 threads. Everything is printed as markdown-ish tables and saved as JSON."""
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
cap = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
OUT = {"step": cap["step"]}
st = cap["state"]
SWAP = os.environ.get("STATE_FROM")   # address these thoughts with another capture's translator / norm / keys
if SWAP:
    st = dict(torch.load(SWAP, map_location="cpu", weights_only=False)["state"])
    st["wcount"] = cap["state"]["wcount"]
    OUT["state_from"] = SWAP
S, m, km = cap["S"], cap["m"], cap["km"]
X = cap["X"]                                    # (B, T)
B, T = X.shape
N = B * T
L, H = st["wq_ema"].shape[:2]
tok = X.flatten()
seq = torch.arange(B).repeat_interleave(T)
pos = torch.arange(T).repeat(B)
# documents: a new doc after each EOT (id 0) inside a window
eot = (X == 0)
docn = torch.cumsum(eot.long(), 1) - eot.long()                  # EOT belongs to the doc it ends
doc = (seq * 1000 + docn.flatten())
after = torch.full((B, T), 10 ** 6)                              # tokens since last EOT (or since window start = big)
for b in range(B):
    last = -10 ** 6
    for t in range(T):
        after[b, t] = t - last if last > -10 ** 6 else 10 ** 6
        if X[b, t] == 0:
            last = t
after = after.flatten()
gen = torch.Generator().manual_seed(0)
log = lambda *a: print(*a, flush=True)
log("step", cap["step"], "N", N, "calls", [s["k"] for s in cap["sels"]])

# token frequency rank (train.bin, first 50M tokens)
tr = np.memmap(r"X:\Datasets\wikipedia_en_20231101_wiki16k\train.bin", dtype=np.uint16, mode="r")[:50_000_000]
freq = np.bincount(np.asarray(tr), minlength=16384)
rank = np.empty(16384, np.int64); rank[np.argsort(-freq)] = np.arange(16384)
rank = torch.from_numpy(rank)
try:
    from tokenizers import Tokenizer
    TK = Tokenizer.from_file(r"X:\Datasets\wikipedia_en_20231101_wiki16k\tokenizer.json")
    dec = lambda i: TK.id_to_token(int(i))
except Exception:
    dec = str


# ------------------------------------------------------------ helpers
def ustats(c):
    """usage stats of a count vector over n bins."""
    c = c.double(); n = c.numel(); tot = c.sum()
    p = c / tot
    nz = p[p > 0]
    ent = float(-(nz * nz.log()).sum())
    cs = torch.sort(c).values
    idx = torch.arange(1, n + 1, dtype=torch.double)
    gini = float(((2 * idx - n - 1) * cs).sum() / (n * cs.sum()))
    d = torch.sort(c, descending=True).values
    t1, t10 = max(1, int(math.ceil(0.01 * n))), max(1, int(math.ceil(0.1 * n)))
    return dict(ent=ent / math.log(n), perp=math.exp(ent), gini=gini, top1=float(d[:t1].sum() / tot),
                top10=float(d[:t10].sum() / tot), distinct=int((c > 0).sum()), dead=float((c == 0).double().mean()))


def rand_baseline(n, draws, reps=1):
    c = torch.bincount(torch.randint(n, (draws,), generator=gen), minlength=n)
    return ustats(c)


def geom(Xm, npairs=20000):
    """participation ratio, erank (entropy of singular values), top-1/5/20 variance shares (centred), mean pairwise
    cosine of random positions (uncentred), resultant length of the unit vectors."""
    Xm = Xm.float()
    Xc = Xm - Xm.mean(0)
    lam = torch.linalg.eigvalsh(Xc.T @ Xc / Xm.size(0)).clamp(min=0).flip(0).double()
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    sv = lam.sqrt(); p = sv / sv.sum(); p = p[p > 0]
    er = float(torch.exp(-(p * p.log()).sum()))
    sh = [float(lam[:k].sum() / lam.sum()) for k in (1, 5, 20)]
    U = F.normalize(Xm, dim=-1)
    i = torch.randint(Xm.size(0), (npairs,), generator=gen); j = torch.randint(Xm.size(0), (npairs,), generator=gen)
    ok = i != j
    cos = float((U[i[ok]] * U[j[ok]]).sum(-1).mean())
    R = float(U.mean(0).norm())
    # share of the uncentred second moment along the mean direction
    msh = float((Xm.mean(0).norm() ** 2) / (Xm.pow(2).sum(-1).mean()))
    return dict(pr=pr, erank=er, dim=Xm.size(1), top1=sh[0], top5=sh[1], top20=sh[2], cos=cos, R=R, mean_share=msh)


def ranks(x):
    r = torch.empty_like(x, dtype=torch.double)
    r[torch.argsort(x)] = torch.arange(x.numel(), dtype=torch.double)
    return r


def spearman(x, y):
    rx, ry = ranks(x.double()), ranks(y.double())
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))


def queries(a_hat_l, wq_lh):
    qr = a_hat_l @ wq_lh.T
    q = F.normalize(qr, dim=-1)
    return F.normalize(q[:, :64], dim=-1), F.normalize(q[:, 64:], dim=-1)


def address(q1, q2, k1, k2):
    """q (N,64), k (S,64) normalised: top-m per half and top-km rows (local index i*S+j)."""
    v1, I1 = torch.topk(q1 @ k1.T, m + 1, -1)
    v2, I2 = torch.topk(q2 @ k2.T, m + 1, -1)
    ii, jj, _ = pk_topk(v1[:, :m], I1[:, :m], v2[:, :m], I2[:, :m], S, km)
    return I1, I2, ii * S + jj, v1, v2


def mi(x, y):
    """plug-in mutual information (nats) of two integer label vectors."""
    _, x = torch.unique(x, return_inverse=True); _, y = torch.unique(y, return_inverse=True)
    nx, ny = int(x.max()) + 1, int(y.max()) + 1
    j = torch.bincount(x * ny + y, minlength=nx * ny).double()
    p = j / j.sum(); px = p.view(nx, ny).sum(1); py = p.view(nx, ny).sum(0)
    def H(v): v = v[v > 0]; return float(-(v * v.log()).sum())
    return H(px) + H(py) - H(p), H(px), H(py)


def mi_corr(x, y, reps=3):
    v, hx, hy = mi(x, y)
    sh = np.mean([mi(x, y[torch.randperm(y.numel(), generator=gen)])[0] for _ in range(reps)])
    return v - sh, v, hx, hy


Kn1 = F.normalize(st["k1_ema"], dim=-1); Kn2 = F.normalize(st["k2_ema"], dim=-1)
sels = cap["sels"]
ks = [s["k"] for s in sels]
pas = lambda k: "w0" if k == -1 else f"p{k + 1}"

# ------------------------------------------------------------ recompute q per call (fp32) and verify
QA = {}   # (call idx) -> dict(a_hat (L,N,D), q1,q2 (L,H,N,64))
for ci, s in enumerate(sels):
    a = s["a"].float()
    a_hat = (a - st["norm_m"][:, None]) / torch.sqrt(st["norm_v"][:, None] + 1e-5)
    q1 = torch.empty(L, H, N, 64); q2 = torch.empty(L, H, N, 64)
    for l in range(L):
        for h in range(H):
            q1[l, h], q2[l, h] = queries(a_hat[l], st["wq_ema"][l, h])
    QA[ci] = dict(a=a, a_hat=a_hat, q1=q1, q2=q2)
if SWAP:  # recompute the selections with the swapped state
    for ci, s_ in enumerate(sels):
        I1a = torch.empty(L, H, N, m + 1, dtype=torch.long); I2a = torch.empty_like(I1a); Ra = torch.empty(L, H, N, km, dtype=torch.long)
        for l in range(L):
            for h in range(H):
                a1, a2, rr_, _, _ = address(QA[ci]["q1"][l, h], QA[ci]["q2"][l, h], Kn1[l, h], Kn2[l, h])
                I1a[l, h], I2a[l, h], Ra[l, h] = a1, a2, rr_ + l * S * S
        s_["I1"], s_["I2"], s_["rows"] = I1a, I2a, Ra
# verification on one call
s = sels[-1]; c = QA[len(sels) - 1]
I1, I2, rows, _, _ = address(c["q1"][1, 0], c["q2"][1, 0], Kn1[1, 0], Kn2[1, 0])
agree = float((I1[:, 0] == s["I1"][1, 0, :, 0].long()).float().mean())
rr = (s["rows"][1, 0].long() - 1 * S * S)
agree_rows = float((torch.sort(rows, -1).values == torch.sort(rr, -1).values).all(-1).float().mean())
log("verify top1 subkey agree", agree, "row sets agree", agree_rows)
OUT["verify"] = dict(top1=agree, rows=agree_rows)

# ------------------------------------------------------------ 1/5/6: usage per level, head, pass (captured selections)
U1 = {}
for ci, s in enumerate(sels):
    for l in range(L):
        for h in range(H):
            r = {}
            for hf, I in ((0, s["I1"]), (1, s["I2"])):
                Ix = I[l, h].long()
                r[f"h{hf + 1}_m"] = ustats(torch.bincount(Ix[:, :m].flatten(), minlength=S))
                r[f"h{hf + 1}_top1"] = ustats(torch.bincount(Ix[:, 0], minlength=S))
            rows = s["rows"][l, h].long() - l * S * S
            r["rows"] = ustats(torch.bincount(rows.flatten(), minlength=S * S))
            r["rows_top1"] = ustats(torch.bincount(rows[:, 0], minlength=S * S))
            # H4: halves' dependence on the selected pairs
            i, j = rows.flatten() // S, rows.flatten() % S
            mic, mip, hi, hj = mi_corr(i, j, reps=1)
            r["mi_ij"] = dict(mi_corr=mic, mi_plugin=mip, Hi=hi, Hj=hj)
            U1[(ci, l, h)] = r
OUT["base_rows"] = rand_baseline(S * S, N * km)
OUT["base_sub_m"] = rand_baseline(S, N * m)
OUT["base_sub_1"] = rand_baseline(S, N)
OUT["base_rows_top1"] = rand_baseline(S * S, N)


def avg(dlist, key):
    return float(np.mean([d[key] for d in dlist]))


log("\n## 1. usage (mean over heads) per level x pass; sub = top-m selections per half; rows = top-km pairs")
log("ckpt pass lvl | h1: ent gini top1% top10% dead | h2: ent dead | top1-subkey ent | rows: ent perp distinct top1% top10% | MI(i;j) corr")
tab1 = []
for ci, k in enumerate(ks):
    for l in range(L):
        rs = [U1[(ci, l, h)] for h in range(H)]
        row = dict(pass_=pas(k), level=l,
                   h1_ent=avg([r["h1_m"] for r in rs], "ent"), h1_gini=avg([r["h1_m"] for r in rs], "gini"),
                   h1_top1=avg([r["h1_m"] for r in rs], "top1"), h1_top10=avg([r["h1_m"] for r in rs], "top10"),
                   h1_dead=avg([r["h1_m"] for r in rs], "dead"),
                   h2_ent=avg([r["h2_m"] for r in rs], "ent"), h2_dead=avg([r["h2_m"] for r in rs], "dead"),
                   h2_gini=avg([r["h2_m"] for r in rs], "gini"), h2_top10=avg([r["h2_m"] for r in rs], "top10"),
                   t1_ent=(avg([r["h1_top1"] for r in rs], "ent") + avg([r["h2_top1"] for r in rs], "ent")) / 2,
                   t1_top1=(avg([r["h1_top1"] for r in rs], "top1") + avg([r["h2_top1"] for r in rs], "top1")) / 2,
                   t1_top10=(avg([r["h1_top1"] for r in rs], "top10") + avg([r["h2_top1"] for r in rs], "top10")) / 2,
                   r_ent=avg([r["rows"] for r in rs], "ent"), r_perp=avg([r["rows"] for r in rs], "perp"),
                   r_gini=avg([r["rows"] for r in rs], "gini"),
                   r_dist=avg([r["rows"] for r in rs], "distinct"), r_top1=avg([r["rows"] for r in rs], "top1"),
                   r_top10=avg([r["rows"] for r in rs], "top10"),
                   mi=float(np.mean([r["mi_ij"]["mi_corr"] for r in rs])),
                   Hi=float(np.mean([r["mi_ij"]["Hi"] for r in rs])), Hj=float(np.mean([r["mi_ij"]["Hj"] for r in rs])))
        tab1.append(row)
        log(f"{cap['step']} {row['pass_']} L{l} | {row['h1_ent']:.3f} {row['h1_gini']:.3f} {row['h1_top1']:.3f} {row['h1_top10']:.3f} {row['h1_dead']:.3f} | "
            f"{row['h2_ent']:.3f} {row['h2_dead']:.3f} | {row['t1_ent']:.3f} | {row['r_ent']:.3f} {row['r_perp']:.0f} {row['r_dist']:.0f} "
            f"{row['r_top1']:.3f} {row['r_top10']:.3f} | {row['mi']:.3f} (Hi {row['Hi']:.2f} Hj {row['Hj']:.2f})")
OUT["usage"] = tab1
OUT["usage_heads"] = {f"{ci},{l},{h}": v for (ci, l, h), v in U1.items()}
log("random baseline rows", OUT["base_rows"], "\nsub m", OUT["base_sub_m"], "\nsub top1", OUT["base_sub_1"])

# per level, pooled over heads and passes (the row table is per level: what write counts see)
log("\n## pooled per level (all heads, all passes)")
pool = []
for l in range(L):
    allr = torch.cat([sels[ci]["rows"][l].long().flatten() - l * S * S for ci in range(len(sels))])
    u = ustats(torch.bincount(allr, minlength=S * S))
    b = rand_baseline(S * S, allr.numel())
    pool.append(dict(level=l, **u, base_distinct=b["distinct"], base_ent=b["ent"], base_top1=b["top1"]))
    log(f"L{l} draws {allr.numel()} ent {u['ent']:.3f} (rand {b['ent']:.3f}) distinct {u['distinct']} (rand {b['distinct']}) "
        f"top1% {u['top1']:.3f} (rand {b['top1']:.3f}) top10% {u['top10']:.3f} gini {u['gini']:.3f}")
OUT["pooled_level"] = pool

# ------------------------------------------------------------ 2. diversity of a, a_hat, q1, q2, and keys
log("\n## 2. geometry: a, a_hat per level; q1/q2 per level (mean over heads); keys")
G2 = []
sub = torch.randperm(N, generator=gen)[:8192]
for ci, k in enumerate(ks):
    for l in range(L):
        ga = geom(QA[ci]["a"][l][sub]); gh = geom(QA[ci]["a_hat"][l][sub])
        gq = [geom(QA[ci][f"q{hf}"][l, h][sub]) for hf in (1, 2) for h in range(H)]
        gq = {kk: float(np.mean([g[kk] for g in gq])) for kk in gq[0]}
        G2.append(dict(pass_=pas(k), level=l, a=ga, a_hat=gh, q=gq))
        log(f"{pas(k)} L{l} a: PR {ga['pr']:.1f}/{ga['dim']} er {ga['erank']:.0f} top1/5/20 {ga['top1']:.2f}/{ga['top5']:.2f}/{ga['top20']:.2f} cos {ga['cos']:.3f} msh {ga['mean_share']:.2f}"
            f" | a_hat: PR {gh['pr']:.1f} er {gh['erank']:.0f} top1/5/20 {gh['top1']:.2f}/{gh['top5']:.2f}/{gh['top20']:.2f} cos {gh['cos']:.3f} msh {gh['mean_share']:.2f}"
            f" | q: PR {gq['pr']:.1f}/64 er {gq['erank']:.0f} top1/5/20 {gq['top1']:.2f}/{gq['top5']:.2f}/{gq['top20']:.2f} cos {gq['cos']:.3f} R {gq['R']:.3f}")
OUT["geom"] = G2
GK = []
for l in range(L):
    gk = [geom(K[l, h]) for K in (Kn1, Kn2) for h in range(H)]
    gk = {kk: float(np.mean([g[kk] for g in gk])) for kk in gk[0]}
    GK.append(gk)
    log(f"keys L{l}: PR {gk['pr']:.1f}/64 er {gk['erank']:.0f} cos {gk['cos']:.3f} R {gk['R']:.3f} top1 {gk['top1']:.2f}")
OUT["geom_keys"] = GK

# ------------------------------------------------------------ 3. key geometry vs usage (+ 8. drift online vs EMA)
log("\n## 3/8. per sub-key: usage group vs cos(key, mean query dir), max cos to any query, the m-th score, drift |k - k_ema|")
cnt_all = torch.zeros(L, H, 2, S)
for s in sels:
    for hf, I in ((0, s["I1"]), (1, s["I2"])):
        for l in range(L):
            for h in range(H):
                cnt_all[l, h, hf] += torch.bincount(I[l, h, :, :m].long().flatten(), minlength=S).float()
ci_mid = min(range(len(ks)), key=lambda i: abs(ks[i] - 2))
G3 = []
for l in range(L):
    acc = {g: dict(cmean=[], cmax=[], drift=[], rawn=[], U=[]) for g in ("dead", "low", "mid", "busy")}
    sp = []
    for h in range(H):
        for hf, Kn, Kr in ((0, Kn1, st["k1"]), (1, Kn2, st["k2"])):
            q = QA[ci_mid][f"q{hf + 1}"][l, h]
            qm = F.normalize(q.mean(0), dim=0)
            kn = Kn[l, h]
            cmean = kn @ qm
            cmax = (q[sub[:4096]] @ kn.T).max(0).values
            drift = (F.normalize(Kr[l, h], dim=-1) - kn).norm(dim=-1)
            rawn = Kr[l, h].norm(dim=-1)
            c = cnt_all[l, h, hf]
            mth = float((q[sub[:4096]] @ kn.T).topk(m, -1).values[:, -1].median())
            q90 = torch.quantile(c, 0.9)
            grp = torch.where(c == 0, 0, torch.where(c <= torch.median(c), 1, torch.where(c >= q90, 3, 2)))
            for gi, g in enumerate(("dead", "low", "mid", "busy")):
                msk = grp == gi
                for nm, v in (("cmean", cmean), ("cmax", cmax), ("drift", drift), ("rawn", rawn), ("U", st["U"][l, h, hf] * S)):
                    acc[g][nm].append(v[msk])
            sp.append(dict(cmean=spearman(c, cmean), cmax=spearman(c, cmax), drift=spearman(c, drift), mth=mth,
                           U_batch=spearman(c, st["U"][l, h, hf])))
    row = dict(level=l, **{f"{g}_{nm}": float(torch.cat(v[nm]).mean()) if torch.cat(v[nm]).numel() else float("nan")
                           for g, v in acc.items() for nm in v},
               **{f"n_{g}": int(torch.cat(acc[g]["cmean"]).numel()) for g in acc},
               **{f"sp_{kk}": float(np.mean([d[kk] for d in sp])) for kk in sp[0]})
    G3.append(row)
    log(f"L{l}: n dead/low/mid/busy {row['n_dead']}/{row['n_low']}/{row['n_mid']}/{row['n_busy']} | cos to q-mean "
        f"{row['dead_cmean']:.3f}/{row['low_cmean']:.3f}/{row['mid_cmean']:.3f}/{row['busy_cmean']:.3f} | max cos to queries "
        f"{row['dead_cmax']:.3f}/{row['low_cmax']:.3f}/{row['mid_cmax']:.3f}/{row['busy_cmax']:.3f} (median m-th score {row['sp_mth']:.3f}) | drift "
        f"{row['dead_drift']:.4f}/{row['low_drift']:.4f}/{row['mid_drift']:.4f}/{row['busy_drift']:.4f} | raw |k| "
        f"{row['dead_rawn']:.3f}/{row['low_rawn']:.3f}/{row['mid_rawn']:.3f}/{row['busy_rawn']:.3f} | U*S {row['dead_U']:.2f}/{row['low_U']:.2f}/{row['mid_U']:.2f}/{row['busy_U']:.2f} "
        f"| spearman(usage, cmean {row['sp_cmean']:.2f}, cmax {row['sp_cmax']:.2f}, drift {row['sp_drift']:.2f}, U {row['sp_U_batch']:.2f})")
OUT["keygeom"] = G3

# ------------------------------------------------------------ 4. content dependence
log("\n## 4. content: row-set overlap |R_a & R_b|/6 (mean over heads) for pair types; MI of top-1 sub-key")
def pairs(kind, n=6000):
    if kind.startswith("lag"):
        lag = int(kind[3:])
        a = torch.randint(N, (n * 2,), generator=gen); a = a[(pos[a] + lag) < T][:n]; return a, a + lag
    if kind == "sameDoc>64":
        a = torch.randint(N, (n * 4,), generator=gen); b = a + torch.randint(64, T, (a.numel(),), generator=gen)
        ok = (b < N); a, b = a[ok], b[ok]; ok = (seq[a] == seq[b]) & (doc[a] == doc[b]); return a[ok][:n], b[ok][:n]
    if kind == "diffSeq":
        a = torch.randint(N, (n * 2,), generator=gen); b = torch.randint(N, (n * 2,), generator=gen)
        ok = seq[a] != seq[b]; return a[ok][:n], b[ok][:n]
    if kind == "sameTok_diffSeq":
        a = torch.randint(N, (n * 20,), generator=gen); b = torch.randint(N, (n * 20,), generator=gen)
        out_a, out_b = [], []
        # match by token: for each a find a b with the same token in another sequence
        order = torch.argsort(tok); st_ = tok[order]
        for x in a.tolist():
            lo = int(torch.searchsorted(st_, tok[x])); hi = int(torch.searchsorted(st_, tok[x], right=True))
            if hi - lo < 2: continue
            y = int(order[lo + int(torch.randint(hi - lo, (), generator=gen))])
            if seq[y] != seq[x] and tok[x] != 0:
                out_a.append(x); out_b.append(y)
            if len(out_a) >= n: break
        return torch.tensor(out_a), torch.tensor(out_b)
    if kind == "samePos<4_diffSeq":
        t = torch.randint(4, (n,), generator=gen); s1 = torch.randint(B, (n,), generator=gen); s2 = torch.randint(B, (n,), generator=gen)
        ok = s1 != s2; return (s1 * T + t)[ok], (s2 * T + t)[ok]
    if kind == "samePos>=256_diffSeq":
        t = torch.randint(256, T, (n,), generator=gen); s1 = torch.randint(B, (n,), generator=gen); s2 = torch.randint(B, (n,), generator=gen)
        ok = s1 != s2; return (s1 * T + t)[ok], (s2 * T + t)[ok]
    if kind == "pos<4_vs_pos<4_sameSeq?":
        pass
kinds = ["lag1", "lag8", "lag64", "lag256", "sameDoc>64", "diffSeq", "sameTok_diffSeq", "samePos<4_diffSeq", "samePos>=256_diffSeq"]
PP = {kd: pairs(kd) for kd in kinds}
log("pairs per kind", {kd: int(v[0].numel()) for kd, v in PP.items()})
G4 = []
for ci, k in enumerate(ks):
    for l in range(L):
        R = sels[ci]["rows"][l].long()        # (H, N, km)
        row = dict(pass_=pas(k), level=l)
        for kd, (a, b) in PP.items():
            row[kd] = float((R[:, a, :, None] == R[:, b, None, :]).any(-1).float().sum(-1).mean() / km)
        # MI(top-1 sub-key half 1; doc) and (; token among top-50 tokens), shuffle-corrected, normalised by H(doc)
        top1 = sels[ci]["I1"][l, :, :, 0].long()
        mid = [mi_corr(top1[h], doc, reps=2) for h in range(H)]
        row["mi_doc"] = float(np.mean([x[0] for x in mid])); row["H_doc"] = mid[0][3]; row["H_top1"] = float(np.mean([x[2] for x in mid]))
        common = torch.bincount(tok, minlength=16384).topk(50).indices
        msk = torch.isin(tok, common) & (tok != 0)
        mit = [mi_corr(top1[h][msk], tok[msk], reps=2) for h in range(H)]
        row["mi_tok50"] = float(np.mean([x[0] for x in mit])); row["H_tok50"] = mit[0][3]
        mip = [mi_corr(top1[h], pos // 32, reps=2) for h in range(H)]
        row["mi_pos32"] = float(np.mean([x[0] for x in mip]))
        G4.append(row)
        log(f"{pas(k)} L{l} " + " ".join(f"{kd} {row[kd]:.3f}" for kd in kinds)
            + f" | MI(top1;doc) {row['mi_doc']:.3f}/H(top1) {row['H_top1']:.2f} H(doc) {row['H_doc']:.2f} | MI(top1;tok50) {row['mi_tok50']:.3f} (H tok {row['H_tok50']:.2f}) | MI(top1;pos/32) {row['mi_pos32']:.3f}")
OUT["content"] = G4

# ------------------------------------------------------------ 7 / c / f: counterfactual re-addressing
log("\n## 7/c/f. re-addressing the same thoughts (passes p1 and p4): sub-key ent (top-m), dead, top10%; rows ent, distinct (per head), top1%; pooled-level distinct")
def whiten(x, eps=1e-3):
    mu = x.mean(0); xc = x - mu
    lam, V = torch.linalg.eigh(xc.T @ xc / x.size(0))
    W = V @ torch.diag(1 / torch.sqrt(lam.clamp(min=0) + eps * lam.max())) @ V.T
    return xc @ W
rk1 = F.normalize(torch.randn(L, H, S, 64, generator=gen), dim=-1); rk2 = F.normalize(torch.randn(L, H, S, 64, generator=gen), dim=-1)
G7 = []
for ci in [i for i, k in enumerate(ks) if k in (0, 3)]:
    for l in range(L):
        ahw = whiten(QA[ci]["a_hat"][l])
        res = {}
        for v in ("trained", "random_keys", "reseed_dead", "centred_q", "centred_q+random_keys", "whitened_ahat", "whitened_ahat+random_keys", "raw_a(no_norm)"):
            subs, rws, allrows = [], [], []
            for h in range(H):
                q1, q2 = QA[ci]["q1"][l, h], QA[ci]["q2"][l, h]
                k1, k2 = Kn1[l, h], Kn2[l, h]
                if "whitened" in v:
                    q1, q2 = queries(ahw, st["wq_ema"][l, h])
                if v.startswith("raw_a"):
                    q1, q2 = queries(QA[ci]["a"][l], st["wq_ema"][l, h])
                if v.startswith("centred"):
                    q1 = F.normalize(q1 - q1.mean(0), dim=-1); q2 = F.normalize(q2 - q2.mean(0), dim=-1)
                if "random_keys" in v:
                    k1, k2 = rk1[l, h], rk2[l, h]
                if v == "reseed_dead":
                    k1, k2 = k1.clone(), k2.clone()
                    for kk, q, hf in ((k1, q1, 0), (k2, q2, 1)):
                        U = st["U"][l, h, hf]; dead = (U < 0.05 * U.mean()).nonzero().squeeze(1)
                        pick = torch.randint(N, (dead.numel(),), generator=gen)
                        kk[dead] = F.normalize(q[pick] + 0.05 * torch.randn(dead.numel(), 64, generator=gen), dim=-1)
                I1, I2, rows, _, _ = address(q1, q2, k1, k2)
                for I in (I1, I2):
                    subs.append(ustats(torch.bincount(I[:, :m].flatten(), minlength=S)))
                rws.append(ustats(torch.bincount(rows.flatten(), minlength=S * S)))
                allrows.append(rows.flatten())
            pl = ustats(torch.bincount(torch.cat(allrows), minlength=S * S))
            res[v] = dict(sub_ent=avg(subs, "ent"), sub_dead=avg(subs, "dead"), sub_top10=avg(subs, "top10"), sub_gini=avg(subs, "gini"),
                          row_ent=avg(rws, "ent"), row_dist=avg(rws, "distinct"), row_top1=avg(rws, "top1"), pool_dist=pl["distinct"], pool_top1=pl["top1"])
        G7.append(dict(pass_=pas(ks[ci]), level=l, **res))
        for v, r in res.items():
            log(f"{pas(ks[ci])} L{l} {v:28s} sub ent {r['sub_ent']:.3f} dead {r['sub_dead']:.3f} top10 {r['sub_top10']:.3f} | rows ent {r['row_ent']:.3f} distinct {r['row_dist']:.0f} top1% {r['row_top1']:.3f} | pooled distinct {r['pool_dist']}")
OUT["counterfactual"] = G7

# ------------------------------------------------------------ d. who uses the popular rows
log("\n## d. the top-50 rows by write count (per level): who selects them in this batch")
wc = st["wcount"].long().view(L, S * S)
GD = []
allk = torch.cat([torch.full((N,), k) for k in ks])
for l in range(L):
    top = wc[l].topk(50).indices
    hits_pos, hits_k, nsel = [], [], 0
    for ci, s in enumerate(sels):
        R = s["rows"][l].long() - l * S * S         # (H, N, km)
        hit = torch.isin(R, top)                    # (H, N, km)
        nsel += R.numel()
        hp = hit.any(-1).any(0)                     # positions with at least one popular row (any head)
        hits_pos.append(hp.nonzero().squeeze(1)); hits_k.append(torch.full((int(hp.sum()),), ks[ci]))
        if ci == 0:
            frac_sel = []
        frac_sel.append(float(hit.float().mean()))
    hp = torch.cat(hits_pos); hk = torch.cat(hits_k)
    base_k = torch.tensor(ks)
    d = dict(level=l, wc_top50_mean=float(wc[l][top].float().mean()), wc_median=float(wc[l].float().median()),
             wc_q90=float(wc[l].float().quantile(0.9)) if wc[l].numel() < 1 << 24 else None,
             share_of_selections=float(np.mean(frac_sel)), share_expected_uniform=50 / (S * S),
             share_positions=float(hp.numel() / (N * len(ks))))
    if hp.numel():
        tt = tok[hp]
        ct = torch.bincount(tt, minlength=16384)
        topt = ct.topk(10)
        d["top_tokens"] = [(dec(i), float(c) / hp.numel()) for c, i in zip(topt.values.tolist(), topt.indices.tolist())]
        d["tok_ent_bits"] = float(-(lambda p: (p[p > 0] * p[p > 0].log2()).sum())(ct.double() / ct.sum()))
        d["tok_distinct"] = int((ct > 0).sum())
        d["rank_median"] = float(rank[tt].double().median()); d["rank_median_all"] = float(rank[tok].double().median())
        d["share_eot_or_after<=2"] = float(((tt == 0) | (after[hp] <= 2)).double().mean())
        d["share_eot_or_after<=2_all"] = float(((tok == 0) | (after <= 2)).double().mean())
        d["share_pos<16"] = float((pos[hp] < 16).double().mean()); d["share_pos<16_all"] = 16 / T
        d["pass_hist"] = {pas(k): float((hk == k).double().mean()) for k in ks}
        d["n_seqs_hit"] = int(torch.unique(seq[hp]).numel())
        # tokens' share of positions vs their share in the batch
        bc = torch.bincount(tok, minlength=16384).double() / N
        d["top_tokens_batch_share"] = [float(bc[i]) for i in topt.indices.tolist()]
    GD.append(d)
    log(json.dumps(d, ensure_ascii=True))
# batch-top-50 vs wcount-top-50 overlap
for l in range(L):
    allr = torch.cat([s["rows"][l].long().flatten() - l * S * S for s in sels])
    bt = torch.bincount(allr, minlength=S * S).topk(50).indices
    GD[l]["overlap_batch_top50"] = int(torch.isin(bt, wc[l].topk(50).indices).sum())
    log(f"L{l} batch-top50 rows in wcount-top50: {GD[l]['overlap_batch_top50']}; share of all row draws on batch top-50: "
        f"{float(torch.bincount(allr, minlength=S * S).topk(50).values.sum() / allr.numel()):.4f}")
OUT["popular"] = GD

# ------------------------------------------------------------ e. gate vs popularity; rich-get-richer between snapshots
log("\n## e. gate vs row popularity (log(1 + wcount) of the 6 rows, averaged)")
GE = []
for rd in cap["reads"]:
    for l in range(L):
        R = rd["rows"][l].long() - l * S * S                  # (H, N, km)
        pop = torch.log1p(wc[l][R].double()).mean(-1)          # (H, N)
        g = rd["g"][:, l, :].T.double()                        # (H, N)
        dec_ = torch.quantile(pop.flatten(), torch.linspace(0, 1, 11, dtype=torch.double))
        bins = torch.bucketize(pop.flatten(), dec_[1:-1])
        gm = [float(g.flatten()[bins == b].mean()) for b in range(10)]
        GE.append(dict(pass_=pas(rd["k"]), level=l, sp=spearman(g.flatten(), pop.flatten()), gate_mean=float(g.mean()),
                       gate_by_pop_decile=gm))
        log(f"{pas(rd['k'])} L{l} gate mean {float(g.mean()):.3f} spearman(gate, popularity) {GE[-1]['sp']:.3f} gate low->high pop decile {gm[0]:.3f} {gm[4]:.3f} {gm[9]:.3f}")
OUT["gate"] = GE
if len(sys.argv) > 3:
    prev = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
    wp = prev["state"]["wcount"].long().view(L, S * S)
    GR = []
    for l in range(L):
        grow = (wc[l] - wp[l]).double()
        r = dict(level=l, step_prev=prev["step"], sp_count_growth=spearman(wp[l].double(), grow),
                 sp_log=spearman(torch.log1p(wp[l].double()), grow),
                 growth_top1pct_share=float(grow[wp[l].topk(int(0.01 * S * S)).indices].sum() / grow.sum()),
                 never_written_prev=float((wp[l] == 0).double().mean()),
                 never_written_then_written=float(((wp[l] == 0) & (wc[l] > 0)).double().mean()))
        GR.append(r)
        log(json.dumps(r))
    OUT["richer"] = GR

# ------------------------------------------------------------ g. concrete reads
log("\n## g1. 20 contexts (seq 0..19, position 400, pass p3): their top-6 rows per level")
ci3 = ks.index(2) if 2 in ks else 0
G1 = []
for l in range(L):
    pp = torch.arange(20) * T + 400
    R = sels[ci3]["rows"][l][:, pp].long() - l * S * S         # (H, 20, km)
    per_head = []
    for h in range(H):
        c = torch.bincount(R[h].flatten(), minlength=S * S)
        per_head.append(dict(distinct=int((c > 0).sum()), shared_rows=int((c >= 2).sum()), max_mult=int(c.max()),
                             contexts_with_shared=int(sum(bool((c[R[h, i]] >= 2).any()) for i in range(20)))))
    c = torch.bincount(R.flatten(), minlength=S * S)
    G1.append(dict(level=l, per_head=per_head, pooled_distinct=int((c > 0).sum()), pooled_of=int(R.numel()),
                   pooled_max_mult=int(c.max()), example_ctx_tokens=[dec(tok[p]) for p in pp[:5].tolist()]))
    log(f"L{l} pooled distinct {G1[-1]['pooled_distinct']}/{R.numel()} max multiplicity {int(c.max())} | per head " +
        "; ".join(f"h{h}: distinct {d['distinct']}/120 rows-shared {d['shared_rows']} max {d['max_mult']} ctx-with-shared {d['contexts_with_shared']}/20" for h, d in enumerate(per_head)))
OUT["g1"] = G1
log("\n## g2. distinct rows per 1000 random positions (6000 draws/head; 18000 pooled over heads) vs ideal uniform")
p1k = torch.randperm(N, generator=gen)[:1000]
ideal6 = S * S * (1 - (1 - 1 / (S * S)) ** 6000); ideal18 = S * S * (1 - (1 - 1 / (S * S)) ** 18000)
G2b = []
for ci, k in enumerate(ks):
    for l in range(L):
        R = sels[ci]["rows"][l][:, p1k].long()
        dh = [int(torch.unique(R[h]).numel()) for h in range(H)]
        dp = int(torch.unique(R).numel())
        G2b.append(dict(pass_=pas(k), level=l, per_head=float(np.mean(dh)), pooled=dp, ideal_head=ideal6, ideal_pooled=ideal18))
        log(f"{pas(k)} L{l} distinct/head {np.mean(dh):.0f} (ideal {ideal6:.0f})  pooled {dp} (ideal {ideal18:.0f})")
OUT["g2"] = G2b
json.dump(OUT, open(sys.argv[2], "w"), indent=1, default=float)
log("done %.0fs" % (time.time() - t0))
