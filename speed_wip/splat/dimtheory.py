"""Quantified examples for plan_v2/23 section 2 (why the memory space must / need not be low-dim).
(1) distance concentration of iid N(0, I_d): the 1-NN / median distance ratio among N = 1M points, and the
    number of points inside 1.1x the 1-NN distance (how sharp a kernel must be);
(2) the synthetic thoughts (PR ~5.3) with the calibrated drift: drift length vs the 1-NN distance in the whitened-r
    space for N = 16K and 1M stored facts (1M by sampling 64 queries against 1M points);
(3) the analytic point-read retrieval share vs the kernel width h / drift s (isotropic, d dims):
    target E[exp(-|delta|^2 / 2h^2)] = (1 + s^2/h^2)^(-d/2); competitors N (h^2/(1+h^2))^(d/2) exp(-|p|^2/2(1+h^2)).
Writes dimtheory.json."""
import json

from common import *

torch.manual_seed(0)
out = {}
NQ = 64
print("(1) iid N(0, I_d), N = 1,048,576")
for d in (2, 4, 8, 16, 32, 64, 128, 448):
    q = torch.randn(NQ, d)
    dists = []
    for c in range(16):
        y = torch.randn(65536, d)
        dists.append(((q * q).sum(1)[:, None] - 2 * q @ y.T + (y * y).sum(1)[None]).clamp_min(0).sqrt())
    dd = torch.cat(dists, 1)
    nn = dd.min(1).values
    med = dd.median(1).values
    within = (dd < 1.1 * nn[:, None]).sum(1).float()
    out[f"iid_d{d}"] = dict(nn_over_median=(nn / med).median().item(), n_within_1p1_nn=within.median().item())
    print(f"d={d}: 1-NN / median distance {(nn / med).median():.3f}; points within 1.1x the 1-NN distance "
          f"{within.median():.0f}")

print("(2) synthetic thoughts + calibrated drift")
ts = ThoughtSpace(0)
cal = json.load(open(os.path.join(HERE, "calib.json")))
for DK in ("rot", "warp"):
    c = cal[DK]
    dr = Drift(ts, DK, c["p1"], c["p2"], 11, ell=c.get("ell", 0.5))
    qa = ts.sample(NQ, gen(50))
    qd = dr(qa)
    for r in (4, 8, 16, 32, 64, 448):
        wh = Whitener(ts.sample(20000, gen(3)), r) if r < 448 else (lambda x: x)
        q0, q1 = wh(qa), wh(qd)
        drift = (q1 - q0).norm(dim=1)
        res = {}
        for N in (16384, 1 << 20):
            nn = []
            for cidx in range(max(1, N // 65536)):
                y = wh(ts.sample(min(N, 65536), gen(1000 + cidx)))
                nn.append(((q0 * q0).sum(1)[:, None] - 2 * q0 @ y.T + (y * y).sum(1)[None]).clamp_min(0).sqrt()
                          .min(1).values)
            nn = torch.stack(nn).min(0).values
            res[N] = (drift / nn).median().item()
        out[f"{DK}_r{r}"] = dict(drift_over_nn_16K=res[16384], drift_over_nn_1M=res[1 << 20])
        print(f"{DK} r={r}: drift / 1-NN distance: 16K facts {res[16384]:.2f}, 1M facts {res[1 << 20]:.2f}")

print("(3) analytic point-read target share T/(T+C) vs h/s, isotropic d dims, N facts, |p|^2 = d")
for d in (8, 16, 32):
    for N in (16384, 1 << 20):
        for s in (0.25, 0.5):
            row = []
            for hs in (0.25, 0.5, 1, 2, 4):
                h = hs * s
                T = (1 + s * s / (h * h)) ** (-d / 2)
                C = N * (h * h / (1 + h * h)) ** (d / 2) * math.exp(-d / (2 * (1 + h * h)))
                row.append(T / (T + C))
            out[f"an_d{d}_N{N}_s{s}"] = row
            print(f"d={d} N={N} s={s}: share at h/s = 0.25,0.5,1,2,4: " + " ".join(f"{x:.3f}" for x in row))
json.dump(out, open(os.path.join(HERE, "dimtheory.json"), "w"), indent=1)
