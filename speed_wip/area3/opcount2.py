"""Area 3: more op-count variants (K = 6 and 10, one training micro-batch + after_step)."""
import sys, json
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\speed_wip\area3")
from opcount import build, measure

V = {
    "as run": (dict(), None),
    "L1": (dict(L=1), None),
    "L3": (dict(L=3), None),
    "read0 off": (dict(read0=0), None),
    "margin off": (dict(), lambda m: setattr(m.mem, "lam_m", 0.0)),
    "frozen (after t_f)": (dict(), lambda m: m.mem.freeze()),
    "no block ckpt": (dict(), lambda m: setattr(m, "mem_ckpt", False)),
    "no private ckpt": (dict(), lambda m: setattr(m.mem, "ckpt", False)),
    "H1 L1 no private": (dict(H=1, L=1, R=0), None),
}
res = {}
for name, (kw, f) in V.items():
    m = build(**kw)
    if f:
        f(m)
    for K in (6, 10):
        n, t, _ = measure(m, K, reps=1)
        res[f"{name} K={K}"] = n
        print(f"{name:24s} K={K:2d} ops {n:6d}", flush=True)
json.dump(res, open(r"C:\Eitan\Python\NewBrain\speed_wip\area3\opcount2.json", "w"), indent=1)
