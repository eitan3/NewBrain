"""Does the GPU lookup (product-quantised cells, bench_gpu.lookup) keep the kNN-1 drift robustness?
Facts stored at their write-time key in the whitened-d_m space, bucketed by the nearest cell (i, j) of two per-half
Euclidean k-means codebooks (C per half). Query: top-m cells per half -> top-kc pairs -> exact NN among the cell
members. Reported: recall@1 (exact key match) vs brute-force kNN-1, per drift level, and the mean candidate count.
Usage: python -B ivf_toy.py <drift>"""
import json
import sys

from common import *

DK = sys.argv[1]
cal = json.load(open(os.path.join(HERE, "calib.json")))[DK]
ts = ThoughtSpace(0)
drifts = [Drift(ts, DK, cal["p1"], cal["p2"], s, ell=cal.get("ell", 0.5)) for s in (11, 12)]
N = 65536
a = ts.sample(N, gen(100))
idx = torch.randperm(N, generator=gen(7))[:4096]
R = {}


def kmeans(x, C, it=12, g=None):
    c = x[torch.randperm(x.shape[0], generator=g)[:C]].clone()
    for _ in range(it):
        asg = ((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]).argmin(1)
        cn = torch.zeros_like(c).index_add_(0, asg, x)
        cnt = torch.bincount(asg, minlength=C)
        dead = cnt == 0
        c = torch.where(dead[:, None], x[torch.randperm(x.shape[0], generator=g)[:C]], cn / cnt.clamp_min(1)[:, None])
    return c


for d_m in (16, 32):
    wh = Whitener(ts.sample(20000, gen(3)), d_m)
    keys = wh(a)
    h = d_m // 2
    fit = wh(ts.sample(50000, gen(4)))
    for C in (64, 256):
        cb = [kmeans(fit[:, :h], C, g=gen(5)), kmeans(fit[:, h:], C, g=gen(6))]
        near = lambda x, c: ((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None])
        cell = near(keys[:, :h], cb[0]).argmin(1) * C + near(keys[:, h:], cb[1]).argmin(1)
        for m, kc in ((2, 2), (4, 8), (8, 16), (8, 32)):
            row = {}
            for lv in (0, 1, 2):
                q = wh(drift_level(drifts, a[idx], lv, noise_seed=1000))
                s1, i1 = (-near(q[:, :h], cb[0])).topk(m, dim=1)
                s2, i2 = (-near(q[:, h:], cb[1])).topk(m, dim=1)
                pair = (s1[:, :, None] + s2[:, None, :]).flatten(1)
                cid = (i1[:, :, None] * C + i2[:, None, :]).flatten(1)
                cells = cid.gather(1, pair.topk(min(kc, m * m), dim=1).indices)             # (Q, kc)
                incell = (cell[idx][:, None] == cells).any(1)                               # own key reachable?
                # exact NN among the probed members: own key wins iff reachable and nearer than every probed member
                member = (cell[None, :, None] == cells[:, None, :]).any(2)                  # (Q, N) bool
                dist = ((q * q).sum(1)[:, None] - 2 * q @ keys.T + (keys * keys).sum(1)[None])
                dist = torch.where(member, dist, torch.full_like(dist, float("inf")))
                hit = (dist.argmin(1) == idx) & incell
                brute = ((q * q).sum(1)[:, None] - 2 * q @ keys.T + (keys * keys).sum(1)[None]).argmin(1) == idx
                row[lv] = dict(r1=hit.float().mean().item(), own_cell_probed=incell.float().mean().item(),
                               brute_r1=brute.float().mean().item(), cand=member.sum(1).float().mean().item())
            R[f"d{d_m}_C{C}_m{m}_kc{kc}"] = row
            print(f"d{d_m} C={C}^2 cells, m={m} kc={kc}: " + " | ".join(
                f"lvl {lv}: r1 {row[lv]['r1']:.3f} (brute {row[lv]['brute_r1']:.3f}, own cell probed "
                f"{row[lv]['own_cell_probed']:.3f}, cand {row[lv]['cand']:.0f})" for lv in (0, 1, 2)), flush=True)
json.dump(R, open(os.path.join(HERE, f"res_ivf_{DK}.json"), "w"), indent=1)
