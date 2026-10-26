"""The splat-memory toy (plan_v2/23). Usage: python -B run_toy.py <exp> <drift>   exp: main|width|cap|interf|dim|fade
drift: rot|warp. Results: res_<exp>_<drift>.json (+ printed tables)."""
import json
import sys
import time

from common import *
from memories import KNN, PKRows, SplatField, SplatGrad

EXP, DK = sys.argv[1], sys.argv[2]
cal = json.load(open(os.path.join(HERE, "calib.json")))[DK]
ts = ThoughtSpace(0)
wh = Whitener(ts.sample(20000, gen(3)), 32)
fit_a = ts.sample(20000, gen(4))
drifts = [Drift(ts, DK, cal["p1"], cal["p2"], s, ell=cal.get("ell", 0.5)) for s in (11, 12)]
DV, BATCH = 64, 512
LEVELS = [0, 0.5, 1, 2]


def make_facts(N, seed=100, clustered=0, eps=0.0):
    g = gen(seed)
    if clustered:                                                      # groups of `clustered` facts, radius eps
        c = ts.sample(N // clustered, g)
        a = c.repeat_interleave(clustered, 0) + eps * ts.sample(N // clustered * clustered, g)
        a = a[torch.randperm(a.shape[0], generator=g)]
    else:
        a = ts.sample(N, g)
    v = F.normalize(torch.randn(a.shape[0], DV, generator=g), dim=1)
    return a, v


import torch.nn.functional as F


def test_idx(N, n=4096):
    if N <= n:
        return torch.arange(N)
    return torch.randperm(N, generator=gen(7))[:n].sort().values


def queries(a, idx, level):
    return drift_level(drifts, a[idx], level, noise_seed=1000)


def write_all(mem, a, v, **kw):
    t0 = time.time()
    for i in range(0, a.shape[0], BATCH):
        mem.write(a[i:i + BATCH], v[i:i + BATCH], **kw)
    return time.time() - t0


def evaluate(read_fn, a, v, idx, levels=LEVELS, extra=None):
    out = {}
    for lv in levels:
        q = queries(a, idx, lv)
        t0 = time.time()
        r = read_fn(q, a[idx]) if extra else read_fn(q)
        dt = time.time() - t0
        s = score(r, idx, v, a.shape[0])
        s["us_per_read"] = dt / len(idx) * 1e6
        out[str(lv)] = s
    return out


def fmt(name, res):
    cells = " | ".join(f"{res[str(l)]['r1']:.3f} / {res[str(l)]['cos']:.2f}" for l in LEVELS if str(l) in res)
    ret = res[str(LEVELS[2])] if str(LEVELS[2]) in res else None
    rr = f"{res['0']['r1_early']:.3f} / {res['0']['r1_late']:.3f}"
    rr1 = f"{ret['r1_early']:.3f} / {ret['r1_late']:.3f}" if ret else "-"
    return f"| {name} | {cells} | {rr} | {rr1} | {res['1']['us_per_read']:.0f} |"


def drift_len(a, idx, field):
    p0 = field.project(a[idx])[0]
    p1 = field.project(queries(a, idx, 1))[0]
    return (p1 - p0).norm(dim=1).median().item()


R = {}
print(f"drift {DK}: {cal}")

if EXP == "main":
    N = 16384
    a, v = make_facts(N)
    idx = test_idx(N)
    print("| memory | " + " | ".join(f"lvl {l}: r@1 / cos" for l in LEVELS) + " | early/late @0 | early/late @1 | us/read |")
    # (a) PK rows
    for n_it in (1, 3):
        m = PKRows(wh, 256, DV, fit_a, seed=1)
        tw = write_all(m, a, v, n_it=n_it)
        R[f"pk_rows_S256_it{n_it}"] = res = evaluate(m.read, a, v, idx)
        res["write_s"] = tw
        print(fmt(f"(a) PK rows 65K, gradient n_it={n_it}", res))
    # (b) kNN
    for nm, sp, k in (("white32 k1", wh, 1), ("white32 k8", wh, 8), ("raw448 k1", lambda x: x, 1)):
        m = KNN(sp, DV, 10 ** 9, k=k)
        tw = write_all(m, a, v)
        R[f"knn_{nm}"] = res = evaluate(m.read, a, v, idx)
        res["write_s"] = tw
        print(fmt(f"(b) kNN {nm}", res))
    # gradient splats (soft rows)
    for d_m in (16,):
        tmpl = SplatField(wh, d_m, DV, 1, n_views=1)
        m = SplatGrad(tmpl, 65536, ts.sample(65536, gen(9)), DV, K=8)
        tw = write_all(m, a, v, n_it=3)
        R[f"splatgrad_d{d_m}_65K"] = res = evaluate(m.read, a, v, idx)
        res["write_s"] = tw
        print(fmt(f"(c') splat-grad d{d_m} 65K fixed centres", res))
    # (c) splat field
    for d_m in (8, 16, 32):
        for V in (1, 3):
            f = SplatField(wh, d_m, DV, 200000, n_views=V, kappa=0.5)
            tw = write_all(f, a, v)
            L = drift_len(a, idx, f)
            base = f"splat_d{d_m}_v{V}"
            R[base + "_info"] = dict(M=f.M, write_s=tw, L=L, stats=f.stats)
            for mode, hops, rd in (("point", 1, None), ("point", 2, None), ("ray", 1, "proj"), ("ray", 2, "proj"),
                                   ("ray", 1, "oracle")):
                nm = f"{base}_{mode}{'-' + rd if rd else ''}_h{hops}"
                fn = lambda q, aw=None, mode=mode, hops=hops, rd=rd: f.read(q, mode=mode, hops=hops, L=L,
                                                                         ray_dir=rd or "proj", a_write=aw)
                R[nm] = res = evaluate(fn, a, v, idx, extra=(rd == "oracle"))
                print(fmt(f"(c) splat d{d_m} {V}v {mode}{'-' + rd if rd else ''} h{hops} (M={f.M})", res))

elif EXP == "width":
    # recall vs splat width (kappa x 3-NN spacing) vs drift level; d_m 16, 1 view, point read; plus kNN-soft temps
    N = 16384
    a, v = make_facts(N)
    idx = test_idx(N)
    LV = [0, 0.25, 0.5, 1, 2]
    for d_m in (8, 16):
        f = SplatField(wh, d_m, DV, 200000, n_views=1, kappa=1.0, shrink=False)
        write_all(f, a, v)
        L = drift_len(a, idx, f)
        sp = (f.ls.exp().mean()).item()
        for ws in (0.25, 0.5, 1, 2, 4, 8):
            R[f"d{d_m}_w{ws}"] = res = evaluate(lambda q: f.read(q, ws=ws), a, v, idx, levels=LV)
            res["sigma_over_drift"] = ws * sp / L
            print(f"d{d_m} width x{ws} (sigma/drift {ws * sp / L:.2f}): " +
                  " ".join(f"{l}: {res[str(l)]['r1']:.3f}/{res[str(l)]['cos']:.2f}" for l in LV))
        R[f"d{d_m}_info"] = dict(L=L, sigma_mean=sp)

elif EXP == "dim":
    # why low-dim: kNN-1 recall vs the memory-space dimension (truncated whitening r), and raw a
    N = 16384
    a, v = make_facts(N)
    idx = test_idx(N)
    for r in (2, 4, 8, 16, 32, 64, 128, 448):
        w = Whitener(ts.sample(20000, gen(3)), r)
        m = KNN(w, DV, 10 ** 9, k=1)
        write_all(m, a, v)
        R[f"white{r}"] = res = evaluate(m.read, a, v, idx)
        print(f"kNN-1 whitened r={r}: " + " ".join(f"{l}: {res[str(l)]['r1']:.3f}" for l in LEVELS))
    for r in (2, 4, 8, 16, 32, 64):                                   # PCA without whitening
        w = Whitener(ts.sample(20000, gen(3)), r)
        w.W = w.V[:, :r].T.contiguous()
        m = KNN(w, DV, 10 ** 9, k=1)
        write_all(m, a, v)
        R[f"pca{r}"] = res = evaluate(m.read, a, v, idx)
        print(f"kNN-1 PCA (no whitening) r={r}: " + " ".join(f"{l}: {res[str(l)]['r1']:.3f}" for l in LEVELS))
    m = KNN(lambda x: x, DV, 10 ** 9, k=1)
    write_all(m, a, v)
    R["raw448"] = res = evaluate(m.read, a, v, idx)
    print("kNN-1 raw 448: " + " ".join(f"{l}: {res[str(l)]['r1']:.3f}" for l in LEVELS))

elif EXP == "cap":
    LV = [0, 1]
    for N in (4096, 16384, 65536):
        a, v = make_facts(N, seed=200 + N)
        idx = test_idx(N)
        for S in (128, 256):
            m = PKRows(wh, S, DV, fit_a, seed=1)
            write_all(m, a, v, n_it=3)
            R[f"N{N}_pk{S * S}"] = res = evaluate(m.read, a, v, idx, levels=LV)
            print(f"N={N} PK rows {S * S}: " + " ".join(
                f"{l}: {res[str(l)]['r1']:.3f}/{res[str(l)]['cos']:.2f} (early {res[str(l)]['r1_early']:.3f})" for l in LV))
        for B in (16384, 65536):
            m = KNN(wh, DV, B, k=1)
            write_all(m, a, v)
            R[f"N{N}_knn{B}"] = res = evaluate(m.read, a, v, idx, levels=LV)
            print(f"N={N} kNN budget {B}: " + " ".join(
                f"{l}: {res[str(l)]['r1']:.3f} (early {res[str(l)]['r1_early']:.3f})" for l in LV))
            for V in (1, 3):
                f = SplatField(wh, 16, DV, B, n_views=V, kappa=0.5)
                write_all(f, a, v)
                R[f"N{N}_splat{B}_v{V}"] = res = evaluate(lambda q: f.read(q), a, v, idx, levels=LV)
                res["M"] = f.M
                print(f"N={N} splat d16 {V}v budget {B}: " + " ".join(
                    f"{l}: {res[str(l)]['r1']:.3f}/{res[str(l)]['cos']:.2f} (early {res[str(l)]['r1_early']:.3f})"
                    for l in LV))

elif EXP == "interf":
    # clustered facts: groups of 8 facts whose thoughts lie within eps (relative) of a centre, different values
    LV = [0, 1]
    N = 16384
    for eps in (0.05, 0.1, 0.2, 0.4):
        a, v = make_facts(N, seed=300, clustered=8, eps=eps)
        idx = test_idx(N)
        m = PKRows(wh, 256, DV, fit_a, seed=1)
        write_all(m, a, v, n_it=3)
        R[f"eps{eps}_pk"] = r1 = evaluate(m.read, a, v, idx, levels=LV)
        m = KNN(wh, DV, 10 ** 9, k=1)
        write_all(m, a, v)
        R[f"eps{eps}_knn"] = r2 = evaluate(m.read, a, v, idx, levels=LV)
        out = []
        for shrink in (False, True):
            f = SplatField(wh, 16, DV, 200000, n_views=1, kappa=0.5, shrink=shrink)
            write_all(f, a, v)
            R[f"eps{eps}_splat_shrink{int(shrink)}"] = r3 = evaluate(lambda q: f.read(q), a, v, idx, levels=LV)
            r3["stats"] = f.stats
            out.append(r3)
        print(f"eps {eps}: PK " + " ".join(f"{r1[str(l)]['r1']:.3f}" for l in LV) +
              " | kNN " + " ".join(f"{r2[str(l)]['r1']:.3f}" for l in LV) +
              " | splat " + " ".join(f"{out[0][str(l)]['r1']:.3f}" for l in LV) +
              " | splat+shrink " + " ".join(f"{out[1][str(l)]['r1']:.3f}" for l in LV) +
              f" (shrinks {out[1]['stats']['shrink']})")

elif EXP == "fade":
    # forgetting / consolidation: alpha fading on, with rehearsal (a second write of the first 25% of facts at the
    # half-drifted thought, i.e. the fact is met again later) -> early facts consolidate; others fade.
    N = 16384
    a, v = make_facts(N)
    idx = test_idx(N)
    LV = [0, 1]
    for fade in (0.0, 0.01, 0.03):
        for reh in (False, True):
            f = SplatField(wh, 16, DV, 200000, n_views=1, kappa=0.5, fade=fade)
            nb = N // BATCH
            for b in range(nb):
                f.write(a[b * BATCH:(b + 1) * BATCH], v[b * BATCH:(b + 1) * BATCH])
                if reh and b == nb // 2:                               # rehearse the first quarter, drifted by 0.5
                    q = N // 4
                    ah = drift_level(drifts, a[:q], 0.5, noise_seed=55)
                    for i in range(0, q, BATCH):
                        f.write(ah[i:i + BATCH], v[i:i + BATCH])
            R[f"fade{fade}_reh{int(reh)}"] = res = evaluate(lambda q: f.read(q), a, v, idx, levels=LV)
            res["M"], res["stats"] = f.M, f.stats
            print(f"fade {fade} rehearsal {reh}: M={f.M} " + " ".join(
                f"{l}: all {res[str(l)]['r1']:.3f} early {res[str(l)]['r1_early']:.3f} late {res[str(l)]['r1_late']:.3f}"
                for l in LV))
    # the same rehearsal for kNN (a second key) and PK rows (a second gradient write)
    m = KNN(wh, DV, 10 ** 9, k=1)
    write_all(m, a, v)
    m.write(drift_level(drifts, a[:N // 4], 0.5, noise_seed=55), v[:N // 4])
    R["knn_reh"] = res = evaluate(m.read, a, v, idx, levels=LV)
    print("kNN rehearsal: " + " ".join(f"{l}: all {res[str(l)]['r1']:.3f} early {res[str(l)]['r1_early']:.3f}" for l in LV))

json.dump(R, open(os.path.join(HERE, f"res_{EXP}_{DK}.json"), "w"), indent=1)
print("saved")
