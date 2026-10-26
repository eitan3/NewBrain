"""plan_v2/20 part 1: address stability across the REAL 6K -> 7K thought drift (pass p3), with frozen / restarted /
tracking / re-fit+migrated / coarse keys. Fixed addressing input: the 6K EMA norm and the 6K W_q (EMA), plain split.
Fit on sequences 0-15, score on sequences 16-31. usage: sim3.py OUT.json [levels=0,3] [steps=2000]"""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("NT", "4")))
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
from memory import pk_topk

t0 = time.time()
OUTF = sys.argv[1]
LEVELS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,3").split(",")]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
BS, LR, M_STAGES = 512, 0.003, 40
ASSIGN = os.environ.get("ASSIGN", "topm")   # top1: k-means / VQ assignment (keys stay distinct)
FITMODE = os.environ.get("FITMODE", "attract")  # kmeans: the 6K / 7K fits by top-1 spherical k-means (r 0.1) + restart
D = os.path.dirname(os.path.abspath(__file__))
c6 = torch.load(os.path.join(D, "cap_6000.pt"), map_location="cpu", weights_only=False)
c7 = torch.load(os.path.join(D, "cap_b.pt"), map_location="cpu", weights_only=False)
st = c6["state"]
m, km = c6["m"], c6["km"]
X = c6["X"]; B, T = X.shape; N = B * T
H = st["wq_ema"].shape[1]
seq = torch.arange(B).repeat_interleave(T)
eot = (X == 0); doc = (seq * 1000 + (torch.cumsum(eot.long(), 1) - eot.long()).flatten())
FIT = (seq < B // 2).nonzero().squeeze(1); EV = (seq >= B // 2).nonzero().squeeze(1)
NE, NF = EV.numel(), FIT.numel()
gen = torch.Generator().manual_seed(0)
log = lambda *a: print(*a, flush=True)
seq_e, doc_e = seq[EV], doc[EV]
a_ = torch.randint(NE, (8000,), generator=gen); b_ = torch.randint(NE, (8000,), generator=gen)
ok = seq_e[a_] != seq_e[b_]; XD = (a_[ok][:4000], b_[ok][:4000])
a_ = torch.randint(NE - 1, (6000,), generator=gen); ok = (seq_e[a_] == seq_e[a_ + 1]) & (doc_e[a_] == doc_e[a_ + 1]); L1 = (a_[ok][:4000], a_[ok][:4000] + 1)
SUBS = torch.randperm(NE, generator=gen)[:4000].view(4, 1000)
ci6 = [s["k"] for s in c6["sels"]].index(2); ci7 = [s["k"] for s in c7["sels"]].index(2)


def address(q1, q2, k1, k2):
    S = k1.size(0)
    v1, I1 = torch.topk(q1 @ k1.T, m, -1)
    v2, I2 = torch.topk(q2 @ k2.T, m, -1)
    ii, jj, _ = pk_topk(v1, I1, v2, I2, S, km)
    return I1, I2, ii * S + jj


def ovl(Ra, Rb):
    return float((Ra[:, :, None] == Rb[:, None, :]).any(-1).float().sum(-1).mean() / Ra.size(1))


def train_keys(qfn, K0, steps, seed, attract=True, restart=True, track_r=0.0, stages=None, assign=None):
    """Streamed key dynamics. qfn(idx, t) -> fit queries (n, 64) at drift fraction t. attract: Adam read pull;
    restart: keys unused for 100 steps jump to a batch query + N(0, 0.05); track_r: k-means EMA toward the mean of
    the assigned queries (no gradient). stages: t schedule (None = t = 0 throughout)."""
    g = torch.Generator().manual_seed(seed)
    S = K0.size(0)
    P_ = K0.clone().requires_grad_(attract)
    opt = torch.optim.Adam([P_], lr=LR) if attract else None
    win = torch.zeros(S)
    for s in range(steps):
        t = 0.0 if stages is None else stages[s]
        b = qfn(torch.randint(NF, (BS,), generator=g), t)
        Kn = F.normalize(P_, dim=-1)
        sc = b @ Kn.T
        I = torch.topk(sc.detach(), m, -1).indices
        Ia = I[:, :1] if (assign or ASSIGN) == "top1" else I
        if attract:
            loss = -sc.gather(1, I).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if track_r > 0:
                If = Ia.flatten()
                sums = torch.zeros(S, 64).index_add_(0, If, b.repeat_interleave(Ia.size(1), 0))
                cnt = torch.bincount(If, minlength=S).float(); has = cnt > 0
                P_[has] = F.normalize((1 - track_r) * F.normalize(P_[has], dim=-1) + track_r * F.normalize(sums[has] / cnt[has, None], dim=-1), dim=-1)
            if restart:
                win += torch.bincount(I.flatten(), minlength=S).float()
                if (s + 1) % 100 == 0:
                    dead = (win == 0).nonzero().squeeze(1)
                    if dead.numel():
                        j = torch.randint(BS, (dead.numel(),), generator=g)
                        P_[dead] = F.normalize(b[j] + 0.05 * torch.randn(dead.numel(), 64, generator=g), dim=-1)
                        if attract:
                            stt = opt.state[P_]; stt["exp_avg"][dead] = 0; stt["exp_avg_sq"][dead] = 0
                    win.zero_()
    return F.normalize(P_.detach(), dim=-1)


def keystats(K):
    c = K @ K.T; c.fill_diagonal_(-1)
    mx = c.max(1).values
    return dict(max_cos=float(mx.max()), dup95=int((mx > 0.95).sum()), dup99=int((mx > 0.99).sum()))


def metrics(Q6, K6, Q7, K7, tag):
    """Q6/Q7 = eval queries ((q1, q2)) at 6K / 7K; K6 the 6K keys, K7 the keys used at 7K."""
    A1, A2, R6 = address(*Q6, *K6)
    B1, B2, R7 = address(*Q7, *K7)
    S = K7[0].size(0)
    dead = ((torch.bincount(B1.flatten(), minlength=S) == 0).float().mean() + (torch.bincount(B2.flatten(), minlength=S) == 0).float().mean()) / 2
    ks = [keystats(K7[0]), keystats(K7[1])]
    return dict(dist=float(np.mean([torch.unique(R7[s_]).numel() for s_ in SUBS])), dead=float(dead),
                xdoc=ovl(R7[XD[0]], R7[XD[1]]), lag1=ovl(R7[L1[0]], R7[L1[1]]),
                row_stab=ovl(R6, R7), key_stab=(ovl(A1, B1) + ovl(A2, B2)) / 2,
                max_key_cos=max(k["max_cos"] for k in ks), dup95=sum(k["dup95"] for k in ks), dup99=sum(k["dup99"] for k in ks))


def migrate(Q6f, K6, Q7f, K7, Q6e, Q7e):
    """Re-fit migration: map each NEW sub-key to the OLD sub-key its fit contexts used (top-1 votes), rows as the
    product map; also a row-level vote map. Content carried = the mapped new rows that are in the position's old set."""
    out = {}
    S = K6[0].size(0)
    maps, purity = [], []
    for h_ in (0, 1):
        old = torch.topk(Q6f[h_] @ K6[h_].T, 1, -1).indices[:, 0]
        new = torch.topk(Q7f[h_] @ K7[h_].T, 1, -1).indices[:, 0]
        V = torch.zeros(S, S).index_put_((new, old), torch.ones(new.numel()), accumulate=True)
        mp = V.argmax(1)
        tot = V.sum(1)
        pur = V.max(1).values / tot.clamp(min=1)
        maps.append(torch.where(tot > 0, mp, torch.arange(S)))
        purity.append(pur[tot > 0])
    _, _, R6 = address(*Q6e, *K6)
    _, _, R7 = address(*Q7e, *K7)
    i, j = R7 // S, R7 % S
    Rm = maps[0][i] * S + maps[1][j]
    out["carried_key_map"] = ovl(R6, Rm)
    pur = torch.cat(purity)
    out["key_purity_mean"] = float(pur.mean()); out["key_conflict_share"] = float((pur < 0.5).float().mean())
    # row-level vote map from the fit half
    _, _, R6f = address(*Q6f, *K6)
    _, _, R7f = address(*Q7f, *K7)
    votes = {}
    nf = R6f.size(0)
    new_f = R7f.flatten(); old_f = R6f.repeat_interleave(km, 0)       # each new row of a position votes for its old rows
    key = new_f[:, None].expand(-1, km).flatten() * (S * S) + old_f.flatten()
    u, c = torch.unique(key, return_counts=True)
    nr, orow = u // (S * S), u % (S * S)
    order = torch.argsort(nr * 100000 + c, descending=True)
    nr, orow, c = nr[order], orow[order], c[order]
    first = torch.ones_like(nr, dtype=torch.bool); first[1:] = nr[1:] != nr[:-1]
    win_new, win_old, win_c = nr[first], orow[first], c[first]
    tot = torch.zeros(S * S).index_add_(0, nr, c.float())
    lut = torch.full((S * S,), -1, dtype=torch.long); lut[win_new] = win_old
    Rr = lut[R7]
    out["unmapped_rows_eval"] = float((Rr < 0).float().mean())
    out["carried_row_map"] = float(((Rr[:, :, None] == R6[:, None, :]) & (Rr[:, :, None] >= 0)).any(-1).float().sum(-1).mean() / km)
    purity_r = win_c.float() / tot[win_new]
    out["row_conflict_share"] = float((purity_r < 0.5).float().mean())
    cnt_old = torch.bincount(win_old, minlength=S * S)
    out["old_rows_copied_to_2plus"] = int((cnt_old >= 2).sum()); out["new_rows_mapped"] = int(win_new.numel())
    return out


RES = {}
for l in LEVELS:
    a6 = c6["sels"][ci6]["a"][l].float(); a7 = c7["sels"][ci7]["a"][l].float()
    nm = lambda a: (a - st["norm_m"][l]) / torch.sqrt(st["norm_v"][l] + 1e-5)
    # drift diagnostics (pass p3): cosine of a_hat 6K vs 7K, and kNN preservation in q space
    ah6, ah7 = nm(a6[EV]), nm(a7[EV])
    diag = dict(level=l, cos_ahat=float(F.cosine_similarity(ah6 - ah6.mean(0), ah7 - ah7.mean(0)).mean()),
                rel_change=float(((ah7 - ah6).norm(dim=-1) / ah6.norm(dim=-1)).median()))
    sub = torch.randperm(NE, generator=gen)[:2000]
    for h in [int(x) for x in os.environ.get("HEADS", "0,1,2").split(",")]:
        wq = st["wq_ema"][l, h]
        def qpair(x):
            q = F.normalize(nm(x) @ wq.T, dim=-1)
            return F.normalize(q[:, :64], dim=-1), F.normalize(q[:, 64:], dim=-1)
        if "diag_done" not in diag:
            diag["diag_done"] = 1
            q6 = torch.cat(qpair(a6[EV][sub]), -1); q7 = torch.cat(qpair(a7[EV][sub]), -1)
            nn6 = torch.topk(q6 @ q6.T, 11, -1).indices[:, 1:]; nn7 = torch.topk(q7 @ q7.T, 11, -1).indices[:, 1:]
            diag["knn10_preserved"] = float((nn6[:, :, None] == nn7[:, None, :]).any(-1).float().mean())
            diag["q_cos_same_pos"] = float((q6 * q7).sum(-1).mean() / 2)
            log(json.dumps(diag)); RES.setdefault("diag", []).append(diag)
        Q6e, Q7e = qpair(a6[EV]), qpair(a7[EV])
        Q6f, Q7f = qpair(a6[FIT]), qpair(a7[FIT])
        a6f, a7f = a6[FIT], a7[FIT]
        def qfn_half(hf):
            def f(idx, t):
                x = (1 - t) * a6f[idx] + t * a7f[idx]
                return qpair(x)[hf]
            return f
        stages = [min(1.0, (s // (STEPS // M_STAGES) + 1) / M_STAGES) for s in range(STEPS)]
        for S in (512, 64):
            gk = torch.Generator().manual_seed(1000 * l + h + S)
            r1, r2 = F.normalize(torch.randn(S, 64, generator=gk), dim=-1), F.normalize(torch.randn(S, 64, generator=gk), dim=-1)
            # the 6K-fitted keys (attract + restart, from random): the common starting point
            K6 = tuple((train_keys(qfn_half(hf), r, STEPS, 7 * l + h + hf) if FITMODE == "attract" else train_keys(qfn_half(hf), r, STEPS, 7 * l + h + hf, attract=False, track_r=0.1, assign="top1")) for hf, r in ((0, r1), (1, r2)))
            def run(tag, Kend):
                r = metrics(Q6e, K6, Q7e, Kend, tag)
                RES.setdefault(f"S{S}|{tag}", []).append(dict(level=l, head=h, **r))
            if S == 512:
                run("a0_today_trained_keys_frozen", (F.normalize(st["k1_ema"][l, h], dim=-1), F.normalize(st["k2_ema"][l, h], dim=-1)))
                # (a0 is measured against its own 6K addressing)
                RES[f"S{S}|a0_today_trained_keys_frozen"][-1].update(metrics(Q6e, (F.normalize(st["k1_ema"][l, h], dim=-1), F.normalize(st["k2_ema"][l, h], dim=-1)), Q7e,
                                                                              (F.normalize(st["k1_ema"][l, h], dim=-1), F.normalize(st["k2_ema"][l, h], dim=-1)), ""))
            run("a_frozen", K6)
            run("b_frozen+restart", tuple(train_keys(qfn_half(hf), K6[hf], STEPS, 11 * l + h + hf, attract=False, restart=True, stages=stages) for hf in (0, 1)))
            run("keep_training(attract+restart)", tuple(train_keys(qfn_half(hf), K6[hf], STEPS, 13 * l + h + hf, stages=stages, assign="topm") for hf in (0, 1)))
            rates = (0.001, 0.003, 0.01, 0.03, 0.1) if S == 512 else (0.003, 0.01, 0.1)
            for rr in rates:
                run(f"c_track_r{rr}", tuple(train_keys(qfn_half(hf), K6[hf], STEPS, 17 * l + h + hf, attract=False, restart=False, track_r=rr, stages=stages) for hf in (0, 1)))
                run(f"c_track_r{rr}+restart", tuple(train_keys(qfn_half(hf), K6[hf], STEPS, 19 * l + h + hf, attract=False, restart=True, track_r=rr, stages=stages) for hf in (0, 1)))
            # (d) re-fit from random at 7K (the 7K fit queries), then migrate
            def qfn7(hf):
                return lambda idx, t: qpair(a7f[idx])[hf]
            K7 = tuple((train_keys(qfn7(hf), r, STEPS, 23 * l + h + hf) if FITMODE == "attract" else train_keys(qfn7(hf), r, STEPS, 23 * l + h + hf, attract=False, track_r=0.1, assign="top1")) for hf, r in ((0, r1.flip(0)), (1, r2.flip(0))))
            run("d_refit7K_no_migration", K7)
            mig = migrate(Q6f, K6, Q7f, K7, Q6e, Q7e)
            RES[f"S{S}|d_refit7K_no_migration"][-1].update({f"mig_{k}": v for k, v in mig.items()})
            # (f) linear re-alignment of the 7K a_hat onto the 6K one (ridge, fit half): an addressing-side map
            if S == 512:
                X7, X6 = nm(a7f), nm(a6f)
                mu7, mu6 = X7.mean(0), X6.mean(0)
                A = torch.linalg.solve((X7 - mu7).T @ (X7 - mu7) + 1e-2 * NF * torch.eye(X7.size(1)), (X7 - mu7).T @ (X6 - mu6))
                q = F.normalize(((nm(a7[EV]) - mu7) @ A + mu6) @ wq.T, dim=-1)
                Qa = (F.normalize(q[:, :64], dim=-1), F.normalize(q[:, 64:], dim=-1))
                r = metrics(Q6e, K6, Qa, K6, "f")
                RES.setdefault(f"S{S}|f_linear_realign+frozen", []).append(dict(level=l, head=h, **r))
            log(f"level {l} head {h} S {S} done {time.time() - t0:.0f}s")
    json.dump(RES, open(OUTF, "w"))
log("done %.0fs" % (time.time() - t0))
