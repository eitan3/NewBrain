"""Calibrate the synthetic drift to the measured 6K->7K numbers (plan 20, part 1):
median ||a'-a||/||a|| = 0.72 and 10-NN preserved = 0.40 (8,192 positions). Writes calib.json.
Also reports: centred cos, 10-NN in the whitened-32 space, and whether a LINEAR re-alignment (lstsq a'->a fitted on
other points) recovers the neighbourhoods / the positions (plan 20: real data, no: 0.199 vs 0.210 frozen)."""
import json
import os
import time

import math
from common import *

N, K, T_REL, T_NN = 8192, 10, 0.72, 0.40
ts = ThoughtSpace(0)
a = ts.sample(N, gen(1))
a_fit = ts.sample(N, gen(2))
wh = Whitener(ts.sample(20000, gen(3)), 32)
print(f"thought PR {pr(a):.2f}; top1/5/20 var share "
      f"{[round(float(ts.lam[:k].sum()), 3) for k in (1, 5, 20)]}")


def stats(dr, x=a):
    x1 = dr(x)
    return rel_change(x, x1), nn_preserved(x, x1, K), x1


def bisect(f, lo, hi, target, it=14, increasing=True):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


out = {}
t0 = time.time()
SIG_W = 0.1   # warp: a small fixed i.i.d. part; the smooth field carries the drift


def mk(kind, x, y):
    return Drift(ts, "rot", x, y, 11) if kind == "rot" else Drift(ts, "warp", x, SIG_W, 11, ell=y)


def solve(kind, grid, lo, hi, nn_increasing_in_y, logy):
    """for each x on the grid, bisect y so that NN10 = 0.40; then interpolate x at rel = 0.72 and re-solve y"""
    def y_for(x, it=12):
        f = (lambda u: stats(mk(kind, x, math.exp(u)))[1]) if logy else (lambda u: stats(mk(kind, x, u))[1])
        l, h = (math.log(lo), math.log(hi)) if logy else (lo, hi)
        u = bisect(f, l, h, T_NN, it=it, increasing=nn_increasing_in_y)
        return math.exp(u) if logy else u
    rows = []
    for x in grid:
        y = y_for(x)
        r, n, _ = stats(mk(kind, x, y))
        rows.append((x, y, r, n))
        print(f"{kind} x {x:.3f}: y {y:.4f} rel {r:.3f} nn {n:.3f} ({time.time() - t0:.0f}s)")
    for (x0, _, r0, _), (x1, _, r1, _) in zip(rows, rows[1:]):
        if (r0 - T_REL) * (r1 - T_REL) <= 0:
            x = x0 + (T_REL - r0) / (r1 - r0) * (x1 - x0)
            break
    else:
        x = min(rows, key=lambda r: abs(r[2] - T_REL))[0]
    y = y_for(x, it=16)
    return x, y, rows


th, sg, scan = solve("rot", [0.4, 0.55, 0.7, 0.85, 1.0], 0.0, 1.5, False, False)
out["rot"] = dict(p1=th, p2=sg, scan=scan)
amp, ell, scan = solve("warp", [0.5, 0.65, 0.8, 0.95, 1.1], 0.2, 50.0, True, True)
out["warp"] = dict(p1=amp, p2=SIG_W, ell=ell, scan=scan)
print(out)

# ---- report every model: rel, NN (a / whitened), centred cos, coherent share, linear re-alignment
for kind in ("rot", "warp"):
    p = out[kind]
    dr = Drift(ts, kind, p["p1"], p["p2"], 11, ell=p.get("ell", 0.5))
    rel, nn, a1 = stats(dr)
    cosc = torch.nn.functional.cosine_similarity(a - a.mean(0), a1 - a1.mean(0), dim=1).mean().item()
    w0, w1 = wh(a), wh(a1)
    nn_w = nn_preserved(w0, w1, K)
    rel_w = rel_change(w0, w1)
    # linear re-alignment: lstsq M with a1_fit @ M ~ a_fit, then apply to the eval points
    a1_fit = dr(a_fit, noise_seed=99)
    M = torch.linalg.lstsq(a1_fit, a_fit).solution
    a1r = a1 @ M
    rel_r, nn_r = rel_change(a, a1r), nn_preserved(a, a1r, K)
    # the same drift at the 2x level and at t = 0.5
    dr2 = Drift(ts, kind, p["p1"], p["p2"], 12, ell=p.get("ell", 0.5))
    a2 = dr2(a1)
    ah = dr(a, 0.5)
    p.update(rel=rel, nn10=nn, cos_centred=cosc, nn10_white32=nn_w, rel_white32=rel_w,
             realign_rel=rel_r, realign_nn10=nn_r,
             rel_2x=rel_change(a, a2), nn10_2x=nn_preserved(a, a2, K),
             rel_half=rel_change(a, ah), nn10_half=nn_preserved(a, ah, K))
    print(kind, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items() if k != "scan"})
out["N"], out["K"] = N, K
json.dump(out, open(os.path.join(HERE, "calib.json"), "w"), indent=1)
print(f"done {time.time() - t0:.0f}s")
