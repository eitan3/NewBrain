"""Shared pieces of the splat-memory toy (plan_v2/23): synthetic thoughts, calibrated drift, metrics.

CPU only. Thoughts a in R^448 with a power-law spectrum lambda_i ~ i^-1.5 (PR ~5.6, top-1/5/20 variance shares
0.38/0.68/0.85, close to plan 16 T4 for a-hat). Drift models:
  rot   : a' = R_theta a + sigma * noise                      (the model the task asked for)
  warp  : a' = a + amp * F(a) + sigma * noise                 (a smooth random displacement field: locally coherent,
                                                              globally non-linear; matches plan 20's finding that a
                                                              linear re-alignment of 7K onto 6K does not help)
noise = sqrt(1-iso) * (data-shaped Gaussian) + sqrt(iso) * (isotropic Gaussian), both with total variance 1.
Drift level t in [0, 1] interpolates a + t (a' - a) (plan 20 streamed the real 6K->7K drift the same way);
"2x" composes two independent 1x drifts.
"""
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
D = 448
HERE = os.path.dirname(os.path.abspath(__file__))


def gen(seed):
    return torch.Generator().manual_seed(int(seed))


class ThoughtSpace:
    def __init__(self, seed=0, p=1.5):
        g = gen(seed)
        lam = torch.arange(1, D + 1, dtype=torch.float64) ** (-p)
        self.lam = (lam / lam.sum()).float()            # total variance 1
        U, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
        self.U = U.float()

    def shaped(self, n, g):
        return (torch.randn(n, D, generator=g) * self.lam.sqrt()) @ self.U.T

    def sample(self, n, g):
        return self.shaped(n, g)


def iso_noise(n, g):
    return torch.randn(n, D, generator=g) / math.sqrt(D)


class Drift:
    """One 1,000-step drift interval. kind 'rot' (p1 = theta) or 'warp' (p1 = amp); p2 = sigma (noise)."""

    def __init__(self, ts, kind, p1, p2, seed, iso=0.25, ell=0.5, F=256):
        self.ts, self.kind, self.p1, self.p2, self.iso, self.seed = ts, kind, p1, p2, iso, seed
        g = gen(seed)
        if kind == "rot":
            G = torch.randn(D, D, generator=g, dtype=torch.float64)
            A = (G - G.T) / math.sqrt(2 * D)
            self.R = torch.linalg.matrix_exp(p1 * A).float()
        else:
            self.Om = torch.randn(F, D, generator=g) / ell
            self.phi = torch.rand(F, generator=g) * 2 * math.pi
            self.B = torch.randn(D, F, generator=g)
            self.F = F
        self.noise_seed = seed + 7919

    def field(self, a):
        f = math.sqrt(2.0 / self.F) * torch.cos(a @ self.Om.T + self.phi)      # (n, F), var ~1 per feature
        h = f @ self.B.T                                                          # (n, D), var ~1 per coordinate
        return (h * self.ts.lam.sqrt()) @ self.ts.U.T                             # data-shaped, total var ~1

    def full(self, a, noise_seed=None):
        g = gen(self.noise_seed if noise_seed is None else noise_seed)
        n = a.shape[0]
        coh = a @ self.R.T if self.kind == "rot" else a + self.p1 * self.field(a)
        nz = math.sqrt(1 - self.iso) * self.ts.shaped(n, g) + math.sqrt(self.iso) * iso_noise(n, g)
        return coh + self.p2 * nz

    def __call__(self, a, t=1.0, noise_seed=None):
        if t == 0:
            return a.clone()
        return a + t * (self.full(a, noise_seed) - a)


def drift_level(drifts, a, level, noise_seed=0):
    """level: 0, 0.25, 0.5, 1 (interpolated within interval 1) or 2 (two composed intervals)."""
    if level <= 1:
        return drifts[0](a, level, noise_seed=noise_seed + 1)
    b = drifts[0](a, 1.0, noise_seed=noise_seed + 1)
    return drifts[1](b, level - 1, noise_seed=noise_seed + 2)


def knn_sets(x, k=10, chunk=2048):
    n = x.shape[0]
    sq = (x * x).sum(1)
    out = []
    for i in range(0, n, chunk):
        d = sq[i:i + chunk, None] - 2 * x[i:i + chunk] @ x.T + sq[None]
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")
        out.append(d.topk(k, largest=False).indices)
    return torch.cat(out)


def nn_preserved(x0, x1, k=10):
    a, b = knn_sets(x0, k), knn_sets(x1, k)
    # overlap of two index sets per row
    eq = (a[:, :, None] == b[:, None, :]).any(2).float().sum(1) / k
    return eq.mean().item()


def rel_change(x0, x1):
    return ((x1 - x0).norm(dim=1) / x0.norm(dim=1).clamp_min(1e-9)).median().item()


def pr(x):
    x = x - x.mean(0)
    ev = torch.linalg.eigvalsh(x.T @ x / x.shape[0]).clamp_min(0)
    return (ev.sum() ** 2 / (ev ** 2).sum()).item()


class Whitener:
    """Basis-free truncated whitening fitted on a thought sample (plan 18): coords = Lambda_r^-1/2 V_r^T a."""

    def __init__(self, sample, r=32):
        mu = sample.mean(0)
        C = (sample - mu).T @ (sample - mu) / sample.shape[0]
        ev, V = torch.linalg.eigh(C.double())
        ev, V = ev.flip(0).float(), V.flip(1).float()
        self.mu, self.ev, self.V = mu, ev, V
        self.set_r(r)

    def set_r(self, r):
        self.r = r
        self.W = (self.V[:, :r] / self.ev[:r].sqrt()).T                    # (r, D)

    def __call__(self, a):
        return (a - self.mu) @ self.W.T


def decode(r, vals, chunk=4096):
    """index of the nearest stored value by cosine"""
    rn = torch.nn.functional.normalize(r, dim=1)
    vn = torch.nn.functional.normalize(vals, dim=1)
    out = []
    for i in range(0, rn.shape[0], chunk):
        out.append((rn[i:i + chunk] @ vn.T).argmax(1))
    return torch.cat(out)


def score(r, idx, vals, n_facts):
    """recall@1 (decode among all facts' values) and value cosine, overall + early 10% / late 10% by write order"""
    pred = decode(r, vals)
    hit = (pred == idx).float()
    cos = torch.nn.functional.cosine_similarity(r, vals[idx], dim=1).nan_to_num(0.0)
    early = idx < n_facts // 10
    late = idx >= n_facts - n_facts // 10
    f = lambda m, x: x[m].mean().item() if m.any() else float("nan")
    return dict(r1=hit.mean().item(), cos=cos.mean().item(), r1_early=f(early, hit), r1_late=f(late, hit),
                cos_early=f(early, cos), cos_late=f(late, cos))
