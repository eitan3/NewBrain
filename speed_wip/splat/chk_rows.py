"""Third calibration check: frozen product-key row-set overlap under the synthetic 1x drift vs the real 0.21-0.22
(S = 512 fitted k-means on pzca32, plan 18 / 20) and 0.42 (S = 64)."""
import json, sys
from common import *
from memories import PKRows
ts = ThoughtSpace(0); wh = Whitener(ts.sample(20000, gen(3)), 32); fit = ts.sample(20000, gen(4))
a = ts.sample(8192, gen(1))
out = {}
for DK in ("rot", "warp"):
    c = json.load(open("calib.json"))[DK]
    dr = Drift(ts, DK, c["p1"], c["p2"], 11, ell=c.get("ell", 0.5))
    for S in (64, 512):
        m = PKRows(wh, S, 8, fit, H=3, seed=1)
        r0, _ = m.address(a)
        for t in (0.25, 0.5, 1.0):
            r1, _ = m.address(dr(a, t))
            ov = (r0[..., :, None] == r1[..., None, :]).any(-1).float().mean().item()
            out[f"{DK}_S{S}_t{t}"] = ov
            print(DK, S, t, round(ov, 3))
json.dump(out, open("chk_rows.json", "w"), indent=1)
