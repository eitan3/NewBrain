"""Consolidation toy (plan_v2/24): which life-cycle mechanism keeps a key memory (kNN / narrow splats) useful while
the thoughts keep drifting? Usage: python -B consol_toy.py [calib]

World: 8 drift intervals x 4 steps (1 step ~ 250 training steps, 1 interval ~ 1,000). The drift is a stationary
warp process (a <- normalise(a + amp F_k(a) + sigma noise), fresh field each step), calibrated so that 4 steps give
the measured 72% relative change / 40% 10-NN kept. 400 new facts per step (12,800); 30% are "useful": they are
re-encountered (seen again WITH their value at their CURRENT thought) with p = 0.25 per step; the rest are one-shot.
Memory: one key per stored item in the frozen whitened-32 space (fitted at t = 0), read = argmax over keys of
-|q-k|^2 / 2h^2 + log alpha (narrow kernel ~ kNN-1 with opacity). Base rule for every policy: new fact -> spawn; a
re-encounter -> read; correct -> reinforce (alpha + 1, cap 8); wrong -> spawn a new key at the current thought.
Policies (alpha fading per step lambda0 = 0.25 = plan 21's 1e-3 per training step; prune alpha < 0.05; over budget:
evict the lowest alpha_eff, oldest first):
  a  global      plan 21: fade all until the switch at 80% (the LR-decay start), linear ramp over 2 steps, then no
                 fade, frozen keys
  b  maturity    per item: maturity m += 1 on a SPACED successful re-use (>= 2 steps since the last); fade
                 lambda0 * 4^-m (m >= 3 ~ permanent); no global switch
  b_re           b + re-encode on a confirmed recall: key <- the current thought (supervised: only after the read was
                 confirmed by the re-encounter)
  b_un           b + unsupervised re-encode: every step 2,000 reads at current thoughts of random born facts; a
                 confident read (margin > 2) moves its top key to the query (no label: can hijack)
  c  cascade     Benna & Fusi 2016: per item u = (u1, u2, u3), capacities (1, 4, 16), conductances 0.25 / 0.0625, leak
                 0.0156 from u3; a write / reinforce adds 1 to u1; alpha_eff = u1 (power-law forgetting)
  d  fade        constant mild fade lambda0 / 2, reinforce + rewrite on re-encounter, no consolidation
  e  sleep       b + at every interval end, items with m >= 1 are re-registered at their context's current thought
                 (needs a stored source pointer + a forward pass over those contexts)
  f  keep        no fade, evict the weakest when full
"""
import json
import math
import sys
import time

from common import *

ts = ThoughtSpace(0)
INT, SPI = 8, 4                      # drift intervals, steps per interval
T = INT * SPI
NEW, USEFUL, P_RE = 400, 0.3, 0.25
ELL = 1.21
CAL = os.path.join(HERE, "consol_calib.json")


def step_drift(a, k, amp, sig, ell=None):
    dr = Drift(ts, "warp", amp, sig, 5000 + k, ell=ell or ELL)
    b = dr.full(a, noise_seed=9000 + k)
    return b * (a.pow(2).sum(1).mean() / b.pow(2).sum(1).mean()).sqrt()     # stationary (OU-like) variance


def calibrate():
    """MAIN calibration: the drift is set by its effect on ADDRESSING, as measured: after 4 composed steps (one 1K-step
    interval) the frozen product-key row overlap (S = 512, plan 18's fitted k-means addressing) is 0.21 (plans 18 / 20),
    with the field length scale from the 10-NN calibration (ell 2.16) and sigma 0.05 per step; the amplitude is
    bisected. The relative change and the 10-NN survival it implies are reported. (A 4-step process cannot give both
    0.72 relative change and 0.21 overlap: at 0.72 the overlap is <= 0.05 for every length scale; that "harsh" variant
    is in consol_harsh/.)"""
    from memories import PKRows
    a = ts.sample(8192, gen(1))
    wh = Whitener(ts.sample(20000, gen(3)), 32)
    pk = PKRows(wh, 512, 8, ts.sample(20000, gen(4)), H=3, seed=1)
    r0, _ = pk.address(a)
    SIG, ell = 0.05, 2.16

    def run(amp, full=False):
        b = a
        for k in range(SPI):
            b = step_drift(b, k, amp, SIG, ell)
        r1, _ = pk.address(b)
        ov = (r0[..., :, None] == r1[..., None, :]).any(-1).float().mean().item()
        return rel_change(a, b), ov, nn_preserved(a, b, 10) if full else None

    lo, hi = 0.0, 0.6
    for _ in range(14):
        m = 0.5 * (lo + hi)
        if run(m)[1] > 0.21:
            lo = m
        else:
            hi = m
    amp = 0.5 * (lo + hi)
    r, ov, nn = run(amp, full=True)
    out = dict(amp=amp, sig=SIG, ell=ell, rel=r, row_overlap=ov, nn=nn)
    json.dump(out, open(CAL, "w"), indent=1)
    print(out)
    return out


class KeyMem:
    def __init__(self, policy, budget, h, lam0=0.25):
        self.p, self.budget, self.h, self.lam0 = policy, budget, h, lam0
        self.key = torch.zeros(0, 32)
        self.src = torch.zeros(0, dtype=torch.long)
        self.u = torch.zeros(0, 3)            # alpha in u[:, 0] (cascade: u1, u2, u3)
        self.m = torch.zeros(0)               # maturity
        self.tls = torch.zeros(0)             # step of the last successful re-use
        self.birth = torch.zeros(0)
        self.t = 0
        self.stats = dict(spawn=0, reinforce=0, reencode=0, sleep=0, evict=0, pruned=0, hijack=0)

    @property
    def alpha(self):
        return self.u[:, 0]

    def read(self, q):
        if self.key.shape[0] == 0:
            n = q.shape[0]
            return torch.full((n,), -1, dtype=torch.long), torch.zeros(n), torch.full((n,), -1, dtype=torch.long)
        # pure nearest key (opacity only breaks near-ties): the opacity governs survival, not the read
        sc = -sq_d(q, self.key) / (2 * self.h ** 2) + 1e-3 * self.alpha.clamp_min(1e-6).log()[None]
        v, j = sc.topk(min(2, sc.shape[1]), dim=1)
        margin = v[:, 0] - v[:, 1] if v.shape[1] > 1 else torch.full_like(v[:, 0], 99.0)
        return self.src[j[:, 0]], margin, j[:, 0]

    def spawn(self, q, f):
        n = q.shape[0]
        if n == 0:
            return
        u = torch.zeros(n, 3)
        u[:, 0] = 1.0
        self.key = torch.cat([self.key, q])
        self.src = torch.cat([self.src, f])
        self.u = torch.cat([self.u, u])
        self.m = torch.cat([self.m, torch.zeros(n)])
        self.tls = torch.cat([self.tls, torch.full((n,), -99.0)])
        self.birth = torch.cat([self.birth, torch.full((n,), float(self.t))])
        self.stats["spawn"] += n

    def reencounter(self, q, f):
        pred, _, j = self.read(q)
        ok = pred == f
        if ok.any():
            jj = j[ok]
            self.u[jj, 0] = (self.u[jj, 0] + 1).clamp(max=8.0)
            self.stats["reinforce"] += int(ok.sum())
            if self.p in ("b", "b_re", "b_un", "b_tr", "e"):
                spaced = (self.t - self.tls[jj]) >= 2
                self.m[jj[spaced]] += 1
                self.tls[jj] = float(self.t)
            if self.p == "b_re":                                   # re-encode on a confirmed recall (move)
                self.key[jj] = q[ok]
                self.stats["reencode"] += int(ok.sum())
        if self.p == "b_tr":                                       # trail: a confirmed recall ADDS a key at the
            self.stats["reencode"] += int(ok.sum())                # current thought (the old one keeps its own fate)
            self.spawn(q, f)
        else:
            self.spawn(q[~ok], f[~ok])

    def unsup(self, q, f):
        """confidence-only re-encode (b_un): no label is used; f only counts hijacks"""
        pred, margin, j = self.read(q)
        conf = margin > 2.0
        if conf.any():
            self.key[j[conf]] = q[conf]
            self.stats["reencode"] += int(conf.sum())
            self.stats["hijack"] += int((conf & (pred != f)).sum())

    def sleep(self, A_now):
        sel = self.m >= 1
        if sel.any():
            self.key[sel] = A_now[self.src[sel]]
            self.stats["sleep"] += int(sel.sum())

    def end_step(self, lam_global):
        p = self.p
        if p == "c":                                               # Benna-Fusi cascade, one Euler step
            C = torch.tensor([1.0, 4.0, 16.0])
            f12 = 0.25 * (self.u[:, 0] - self.u[:, 1])
            f23 = 0.0625 * (self.u[:, 1] - self.u[:, 2])
            out = 0.0156 * self.u[:, 2]
            self.u = self.u + torch.stack([-f12, f12 - f23, f23 - out], 1) / C
            alive = self.u.sum(1) > 0.05
        else:
            if p == "a":
                lam = lam_global
            elif p in ("b", "b_re", "b_un", "b_tr", "e"):
                lam = self.lam0 * 4.0 ** (-self.m)
            elif p == "d":
                lam = self.lam0 / 2
            else:
                lam = 0.0
            self.u[:, 0] = self.u[:, 0] * torch.exp(-torch.as_tensor(lam))
            alive = self.u[:, 0] > 0.05
        self.stats["pruned"] += int((~alive).sum())
        self._keep(alive)
        if self.key.shape[0] > self.budget:                        # evict the weakest (oldest first on ties)
            score = self.u.sum(1) + 1e-4 * self.birth
            keep = torch.zeros(self.key.shape[0], dtype=torch.bool)
            keep[score.argsort(descending=True)[:self.budget]] = True
            self.stats["evict"] += int((~keep).sum())
            self._keep(keep)
        self.t += 1

    def _keep(self, k):
        for n in ("key", "src", "u", "m", "tls", "birth"):
            setattr(self, n, getattr(self, n)[k])


def sq_d(x, y):
    return ((x * x).sum(1)[:, None] - 2 * x @ y.T + (y * y).sum(1)[None]).clamp_min(0)


def simulate(policy, budget, cal, decay_after=None, seed=0):
    g = gen(100 + seed)
    N = NEW * T
    A = ts.sample(N, g)                                            # base thoughts; a fact's thought is drawn at
    useful = torch.rand(N, generator=g) < USEFUL                    # birth and drifts afterwards
    wh = Whitener(ts.sample(20000, gen(3)), 32)
    h = 0.5 * math.sqrt(2 * 32) * 0.35                              # ~0.25-0.5 x the 1-NN spacing (narrow)
    mem = KeyMem(policy, budget, h)
    t_sw = int(0.8 * T)
    curve, out = [], {}
    early = (torch.arange(N) < NEW * SPI)                           # born in interval 0
    late = (torch.arange(N) >= NEW * (T - SPI))                     # born in the last interval
    for t in range(T):
        born = NEW * t
        amp, sig = cal["amp"], cal["sig"]
        if decay_after is not None and t >= int(0.8 * T):
            amp, sig = amp * decay_after, sig * decay_after
        if born:                                                    # the drift of this step for the born facts
            A[:born] = step_drift(A[:born], t, amp, sig, cal["ell"])
        W = wh(A)
        if born:                                                    # re-encounters of useful born facts
            r = (torch.rand(born, generator=g) < P_RE) & useful[:born]
            f = r.nonzero()[:, 0]
            mem.reencounter(W[f], f)
            if policy == "b_un":
                f2 = torch.randint(0, born, (2000,), generator=g)
                mem.unsup(W[f2], f2)
        f = torch.arange(born, born + NEW)
        mem.spawn(W[f], f)
        if policy == "e" and (t + 1) % SPI == 0:
            mem.sleep(W)
        lam = mem.lam0 if t < t_sw else mem.lam0 * max(0.0, 1 - (t - t_sw + 1) / 2)
        mem.end_step(lam)
        if (t + 1) % SPI == 0:                                      # probes at the end of every interval
            nb = born + NEW
            ids = torch.arange(nb)
            pred, margin, _ = mem.read(W[ids])
            hit = pred == ids
            res = {}
            for nm, msk in (("early_useful", early[:nb] & useful[:nb]), ("early_oneshot", early[:nb] & ~useful[:nb]),
                            ("all_useful", useful[:nb]), ("all_oneshot", ~useful[:nb]),
                            ("new_useful", (ids >= nb - NEW * SPI) & useful[:nb]),
                            ("new_oneshot", (ids >= nb - NEW * SPI) & ~useful[:nb])):
                res[nm] = hit[msk].float().mean().item() if msk.any() else float("nan")
            res["conf_wrong"] = ((~hit) & (margin > 2.0) & (pred >= 0)).float().mean().item()
            res["slots"] = mem.key.shape[0]
            res["mature"] = int((mem.m >= 1).sum()) if policy in ("b", "b_re", "b_un", "b_tr", "e") else None
            us = useful[mem.src]
            res["keys_per_useful"] = us.sum().item() / max(1, int(useful[:nb].sum()))
            res["keys_per_oneshot"] = (~us).sum().item() / max(1, int((~useful[:nb]).sum()))
            curve.append(res)
    out["curve"] = curve
    out["final"] = curve[-1]
    out["stats"] = mem.stats
    # interference: share of confident wrong reads among all born facts at the end; key writes per step
    out["key_writes_per_step"] = (mem.stats["spawn"] + mem.stats["reencode"] + mem.stats["sleep"]) / T
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calib" or not os.path.exists(CAL):
        calibrate()
    cal = json.load(open(CAL))
    print("calib", {k: v for k, v in cal.items() if k != "scan"}, flush=True)
    R = {}
    t0 = time.time()
    for scen, dec in (("const", None), ("lrdecay", 0.3)):
        for budget in (6000, 20000):
            for pol in ("a", "b", "b_re", "b_tr", "b_un", "c", "d", "e", "f"):
                o = simulate(pol, budget, cal, decay_after=dec)
                R[f"{scen}_B{budget}_{pol}"] = o
                fz = o["final"]
                cur = " ".join(f"{c['early_useful']:.2f}" for c in o["curve"])
                print(f"{scen} B={budget} {pol:5s}: early useful {fz['early_useful']:.3f} oneshot {fz['early_oneshot']:.3f}"
                      f" | all useful {fz['all_useful']:.3f} | new useful {fz['new_useful']:.3f} oneshot "
                      f"{fz['new_oneshot']:.3f} | conf-wrong {fz['conf_wrong']:.3f} | slots {fz['slots']} mature "
                      f"{fz['mature']} | keys/useful {fz['keys_per_useful']:.2f} keys/oneshot {fz['keys_per_oneshot']:.2f} | key writes/step {o['key_writes_per_step']:.0f} | early-useful over intervals: "
                      f"{cur} | {o['stats']} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(R, open(os.path.join(HERE, "res_consol.json"), "w"), indent=1)
    print("saved")
