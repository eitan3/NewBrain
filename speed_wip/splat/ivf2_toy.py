"""Index redundancy for the cell lookup (follow-up to ivf_toy.py): (a) write duplication: each key is filed in its top-w
cells; (b) T independent cell tables (each: its own random rotation of the whitened space before the halves, its own
codebooks), candidates = union. Probe: top-m per half, top-kc pairs per table. N = 65,536 facts in d_m = 32.
Reports recall@1 (own key found and nearest among the probed), own-key-probed, candidates per query (with
duplicates removed), and the slot cost (keys stored x w x T). Usage: python -B ivf2_toy.py <drift>"""
import json
import sys

from common import *
from memories import rand_orth

DK = sys.argv[1]
cal = json.load(open(os.path.join(HERE, "calib.json")))[DK]
ts = ThoughtSpace(0)
drifts = [Drift(ts, DK, cal["p1"], cal["p2"], s, ell=cal.get("ell", 0.5)) for s in (11, 12)]
N, d_m = int(sys.argv[2]) if len(sys.argv) > 2 else 65536, 32
CS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [64, 256]
CONF = None if len(sys.argv) <= 4 else [tuple(int(y) for y in x.split("-")) for x in sys.argv[4].split(",")]
a = ts.sample(N, gen(100))
idx = torch.randperm(N, generator=gen(7))[:2048]
wh = Whitener(ts.sample(20000, gen(3)), d_m)
keys = wh(a)
fit = wh(ts.sample(50000, gen(4)))
h = d_m // 2
near = lambda x, c: ((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None])


def kmeans(x, C, it=12, g=None):
    c = x[torch.randperm(x.shape[0], generator=g)[:C]].clone()
    for _ in range(it):
        asg = near(x, c).argmin(1)
        cn = torch.zeros_like(c).index_add_(0, asg, x)
        cnt = torch.bincount(asg, minlength=C)
        c = torch.where((cnt == 0)[:, None], x[torch.randperm(x.shape[0], generator=g)[:C]], cn / cnt.clamp_min(1)[:, None])
    return c


def table(seed, C):
    g = gen(seed)
    Rm = torch.eye(d_m) if seed == 0 else rand_orth(d_m, d_m, g)
    f = fit @ Rm.T
    return Rm, [kmeans(f[:, :h], C, g=g), kmeans(f[:, h:], C, g=g)]


def cells_of(x, tb, C, m, kc):
    Rm, cb = tb
    y = x @ Rm.T
    s1, i1 = (-near(y[:, :h], cb[0])).topk(m, dim=1)
    s2, i2 = (-near(y[:, h:], cb[1])).topk(m, dim=1)
    pair = (s1[:, :, None] + s2[:, None, :]).flatten(1)
    cid = (i1[:, :, None] * C + i2[:, None, :]).flatten(1)
    return cid.gather(1, pair.topk(min(kc, m * m), dim=1).indices)


R = {}
for C in CS:
    tabs = [table(s, C) for s in range(4)]
    for T, w, m, kc in CONF or ((1, 1, 4, 8), (1, 2, 4, 8), (1, 4, 4, 8), (2, 1, 4, 8), (4, 1, 4, 8), (2, 2, 4, 8),
                                (4, 2, 4, 8), (4, 4, 4, 8), (4, 1, 8, 16)):
        # membership matrix of the stored keys: (N, T*w) cell ids (table t's ids offset by t*C*C)
        store = torch.cat([cells_of(keys, tabs[t], C, 2, w) + t * C * C for t in range(T)], 1)
        row = {}
        for lv in (0, 1, 2):
            q = wh(drift_level(drifts, a[idx], lv, noise_seed=1000))
            probe = torch.cat([cells_of(q, tabs[t], C, m, kc) + t * C * C for t in range(T)], 1)   # (Q, T*kc)
            member = torch.zeros(len(idx), N, dtype=torch.bool)
            for j in range(store.shape[1]):
                member |= (store[None, :, j, None] == probe[:, None, :]).any(2)
            dist = near(q, keys)
            own = member[torch.arange(len(idx)), idx]
            hit = torch.where(member, dist, torch.full_like(dist, float("inf"))).argmin(1) == idx
            row[lv] = dict(r1=(hit & own).float().mean().item(), own=own.float().mean().item(),
                           cand=member.sum(1).float().mean().item())
        nm = f"C{C}_T{T}_w{w}_m{m}_kc{kc}"
        R[nm] = row
        print(f"{nm} (key slots x{T * w}): " + " | ".join(
            f"lvl {lv}: r1 {row[lv]['r1']:.3f} own {row[lv]['own']:.3f} cand {row[lv]['cand']:.0f}" for lv in (0, 1, 2)),
              flush=True)
json.dump(R, open(os.path.join(HERE, f"res_ivf2_{DK}_N{N}.json"), "w"), indent=1)
