"""The three memories of the toy (plan_v2/23 part 2).

PKRows     product-key row memory as in memory.py, fixed (addr-learn 0) addressing on the truncated whitening (plan 18
           best: pzca32 -> random projection to 128 -> halves -> spherical k-means sub-keys), H = 3 heads hashing into
           ONE table of S^2 rows, top-m 8 per half, top-k 6 pairs, omega = softmax(sigma / tau). Gradient-written: n_it
           steps of the squared-error gradient per batch with the per-fact normalised (delta-rule) step size, the best
           case of a gradient write.
KNN        a key-value store with direct writes; read = top-k by L2 in the whitened-32 space (or raw a), softmax weights.
SplatField Gaussian splats in a d_m-dim memory space: mu, diagonal sigma, opacity alpha, value v. Views = fixed
           projections of the truncated-whitened thought. Direct writes + densify (spawn), reinforce, conflict shrink,
           prune / fade. Reads: point or ray (front-to-back compositing with transmittance), 1 or 2 hops, views
           combined by their accumulated opacity.
SplatGrad  gradient-trained splats: fixed centres sampled from the thought distribution, value delta rule (soft rows).
"""
import math

import torch
import torch.nn.functional as F

from common import gen


def rand_orth(rows, cols, g):
    q, _ = torch.linalg.qr(torch.randn(cols, rows, generator=g, dtype=torch.float64))
    return q.T.float()                                                    # (rows, cols), orthonormal rows


def sq_dist(x, y):
    return ((x * x).sum(1)[:, None] - 2 * x @ y.T + (y * y).sum(1)[None]).clamp_min(0)


def topk_small(x, y, k, chunk=4096):
    """indices / squared distances of the k nearest rows of y for each row of x (chunked brute force)"""
    I, Dd = [], []
    k = min(k, y.shape[0])
    for i in range(0, x.shape[0], chunk):
        d = sq_dist(x[i:i + chunk], y)
        v, j = d.topk(k, largest=False)
        I.append(j)
        Dd.append(v)
    return torch.cat(I), torch.cat(Dd)


def skmeans(x, S, it=15, g=None):
    x = F.normalize(x, dim=1)
    C = x[torch.randperm(x.shape[0], generator=g)[:S]].clone()
    for _ in range(it):
        a = (x @ C.T).argmax(1)
        Cn = torch.zeros_like(C).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=S)
        dead = cnt == 0
        if dead.any():
            Cn[dead] = x[torch.randperm(x.shape[0], generator=g)[:int(dead.sum())]]
        C = F.normalize(Cn, dim=1)
    return C


# ----------------------------------------------------------------------------------------------- (a) product-key rows
class PKRows:
    def __init__(self, wh, S, dv, fit_a, H=3, m=8, km=6, tau=0.03, seed=0):
        g = gen(seed)
        self.wh, self.S, self.H, self.m, self.km, self.tau, self.dv = wh, S, H, m, km, tau, dv
        x = wh(fit_a)
        self.G, self.K1, self.K2 = [], [], []
        for h in range(H):
            G = rand_orth(128, wh.r, g) if wh.r >= 128 else rand_orth(wh.r, 128, g).T   # (128, r)
            q = x @ G.T
            self.G.append(G)
            self.K1.append(skmeans(q[:, :64], S, g=g))
            self.K2.append(skmeans(q[:, 64:], S, g=g))
        self.V = torch.randn(S * S, dv, generator=g) * 0.02

    def address(self, a):
        x = self.wh(a)
        rows, ws = [], []
        for h in range(self.H):
            q = x @ self.G[h].T
            s1 = F.normalize(q[:, :64], dim=1) @ self.K1[h].T
            s2 = F.normalize(q[:, 64:], dim=1) @ self.K2[h].T
            v1, i1 = s1.topk(self.m, dim=1)
            v2, i2 = s2.topk(self.m, dim=1)
            sig = (v1[:, :, None] + v2[:, None, :]).flatten(1)
            rr = (i1[:, :, None] * self.S + i2[:, None, :]).flatten(1)
            v, j = sig.topk(self.km, dim=1)
            rows.append(rr.gather(1, j))
            ws.append(torch.softmax(v / self.tau, dim=1))
        return torch.stack(rows, 1), torch.stack(ws, 1)                  # (n, H, km)

    def read_rows(self, rows, w):
        return (w[..., None] * self.V[rows]).sum((1, 2))

    def read(self, a):
        return self.read_rows(*self.address(a))

    def write(self, a, v, n_it=3, beta=1.0):
        rows, w = self.address(a)
        nrm = (w * w).sum((1, 2))                                         # delta-rule normaliser
        for _ in range(n_it):
            err = (v - self.read_rows(rows, w)) * (beta / nrm)[:, None]
            self.V.index_add_(0, rows.flatten(), (w[..., None] * err[:, None, None, :]).reshape(-1, self.dv))

    def flops_read(self):
        r = self.wh.r
        return 448 * r * 2 + self.H * (128 * r * 2 + 2 * self.S * 64 * 2 + 64 * 2 + self.km * self.dv * 2)


# ----------------------------------------------------------------------------------------------- (b) kNN key-value
class KNN:
    def __init__(self, space, dv, budget, k=8):
        self.space, self.k, self.budget = space, k, budget
        self.keys, self.vals = None, None
        self.h2 = None

    def write(self, a, v):
        kk = self.space(a)
        self.keys = kk if self.keys is None else torch.cat([self.keys, kk])
        self.vals = v.clone() if self.vals is None else torch.cat([self.vals, v])
        if self.keys.shape[0] > self.budget:                              # FIFO eviction
            self.keys, self.vals = self.keys[-self.budget:], self.vals[-self.budget:]
        self.h2 = None

    def _h2(self):
        if self.h2 is None:
            s = self.keys[torch.randperm(self.keys.shape[0], generator=gen(5))[:2048]]
            _, d = topk_small(s, self.keys, 2)
            self.h2 = d[:, 1].median().clamp_min(1e-6)
        return self.h2

    def read(self, a, k=None, temp=1.0):
        k = k or self.k
        q = self.space(a)
        j, d = topk_small(q, self.keys, k)
        if k == 1:
            return self.vals[j[:, 0]]
        w = torch.softmax(-d / (2 * self._h2() * temp), dim=1)
        return (w[..., None] * self.vals[j]).sum(1)

    def flops_read(self, d):
        return self.keys.shape[0] * d * 2 + self.k * self.vals.shape[1] * 2


# ----------------------------------------------------------------------------------------------- (c) splat field
class SplatField:
    def __init__(self, wh, d_m, dv, budget, n_views=1, kappa=0.5, alpha0=3.0, amax=6.0, smin=0.02, smax=3.0,
                 shared=True, fade=0.0, K=32, seed=0, shrink=True):
        g = gen(seed)
        self.wh, self.d_m, self.dv, self.budget, self.V = wh, d_m, dv, budget, n_views
        self.kappa, self.alpha0, self.amax, self.smin, self.smax = kappa, alpha0, amax, smin, smax
        self.shared, self.fade, self.K, self.shrink = shared, fade, K, shrink
        r = wh.r
        # view 0: the top d_m whitened principal directions; views 1..: random d_m-dim subspaces of the whitened r
        Qs = [torch.eye(r)[:d_m]] + [rand_orth(d_m, r, g) for _ in range(n_views - 1)]
        self.Q = torch.stack(Qs)                                          # (V, d_m, r)
        self.Qd = torch.stack([rand_orth(d_m, r, g) for _ in range(n_views)])   # ray directions (fixed projection)
        self.off = torch.zeros(n_views, d_m)
        if not shared:
            self.off[:, 0] = torch.arange(n_views) * 1e4                  # separate fields per view
        self.mu = torch.zeros(0, d_m)
        self.ls = torch.zeros(0, d_m)                                     # log sigma (diagonal)
        self.ls0 = torch.zeros(0, d_m)                                    # log sigma at birth (shrink floor)
        self.al = torch.zeros(0)
        self.val = torch.zeros(0, dv)
        self.cons = torch.zeros(0, dtype=torch.bool)
        self.tl = torch.zeros(0)                                          # step of the last touch (lazy fading)
        self.step = 0
        self.stats = dict(spawn=0, reinforce=0, shrink=0, pruned=0)

    @property
    def M(self):
        return self.mu.shape[0]

    def project(self, a):
        x = self.wh(a)
        return torch.einsum("nr,vdr->vnd", x, self.Q) + self.off[:, None]  # (V, n, d_m)

    def ray_dirs(self, a):
        x = self.wh(a)
        return F.normalize(torch.einsum("nr,vdr->vnd", x, self.Qd), dim=-1)

    def alpha_eff(self, idx=None):
        al, tl, cons = (self.al, self.tl, self.cons) if idx is None else (self.al[idx], self.tl[idx], self.cons[idx])
        if self.fade <= 0:
            return al
        return torch.where(cons, al, al * torch.exp(-self.fade * (self.step - tl)))

    # ---------------- kernels
    def _point(self, p, ws=1.0, K=None):
        """p (n, d_m). Returns the colour, opacity A, the location estimate mu-hat, candidate ids and rho."""
        K = K or self.K
        idx, _ = topk_small(p, self.mu, K)                                # Euclidean pre-select
        mu, sg = self.mu[idx], self.ls[idx].exp() * ws
        m = (((p[:, None] - mu) / sg) ** 2).sum(-1)                        # Mahalanobis^2
        lr = self.alpha_eff(idx).clamp_min(1e-30).log() - 0.5 * m           # log rho (log domain: no underflow)
        lse = lr.logsumexp(1)                                              # log sum rho  (log-confidence)
        wn = torch.softmax(lr, 1)                                          # rho / sum rho
        col = (wn[..., None] * self.val[idx]).sum(1)
        A = -torch.expm1(-lse.exp())                                       # 1 - exp(-sum rho)
        muh = (wn[..., None] * mu).sum(1)
        return col, A, muh, idx, lr.exp(), m, lse

    def _ray(self, o, d, L, S=8, ws=1.0, K=None):
        """front-to-back compositing along o + t d, t in [0, L] (S samples). Candidates: the K nearest splats to the
        segment (point-to-segment distance, two GEMMs)."""
        K = K or 2 * self.K
        n = o.shape[0]
        idx_l = []
        for i in range(0, n, 2048):
            oo, dd, LL = o[i:i + 2048], d[i:i + 2048], L[i:i + 2048]
            mo = self.mu @ dd.T                                           # (M, c)
            proj = (mo - (oo * dd).sum(1)[None]).T                        # (c, M) = (mu - o).d
            t = (proj / LL[:, None]).clamp(0, 1) * LL[:, None]
            d2 = sq_dist(oo, self.mu) - 2 * t * proj + t * t
            idx_l.append(d2.topk(min(K, self.M), largest=False).indices)
        idx = torch.cat(idx_l)
        mu, sg, al = self.mu[idx], self.ls[idx].exp() * ws, self.alpha_eff(idx)
        ts = torch.linspace(0, 1, S)[None, :, None] * L[:, None, None]
        ps = o[:, None] + ts * d[:, None]                                 # (n, S, d_m)
        m = (((ps[:, :, None] - mu[:, None]) / sg[:, None]) ** 2).sum(-1)  # (n, S, K)
        lr = al.clamp_min(1e-30).log()[:, None] - 0.5 * m                  # log rho (n, S, K)
        ls = lr.logsumexp(-1)                                              # log sigma_s
        sig = ls.exp()
        # log a_s = log(1 - exp(-sigma_s)) (= log sigma_s when sigma_s is tiny); log T_s = -sum_{s'<s} sigma_s'
        la = torch.where(ls < -10, ls, torch.log(-torch.expm1(-sig)).clamp_min(-1e30))
        lT = -torch.cat([torch.zeros(n, 1), sig[:, :-1].cumsum(1)], 1)
        lwS = lT + la                                                      # log of T_s a_s (n, S)
        lwi = (lwS[..., None] + lr - ls[..., None]).logsumexp(1)           # log per-candidate weight (n, K)
        wn = torch.softmax(lwi, 1)
        col = (wn[..., None] * self.val[idx]).sum(1)
        lse = lwS.logsumexp(1)                                             # log opacity of the ray
        muh = (wn[..., None] * mu).sum(1)
        return col, lse.exp().clamp(max=1.0), muh, lse

    def read(self, a, mode="point", hops=1, views=None, L=None, ray_dir="proj", a_write=None, hop_w=2.0, S=8,
             ws=1.0):
        V = views or self.V
        P = self.project(a)[:V]
        if mode == "ray":
            if ray_dir == "oracle":                                        # ceiling: toward the write-time location
                Dd = F.normalize(self.project(a_write)[:V] - P, dim=-1)
            else:
                Dd = self.ray_dirs(a)[:V]
        cols, conf = [], []
        for v in range(V):
            p = P[v]
            w1 = hop_w if hops == 2 else 1.0
            if mode == "ray":
                col, A, muh, lse = self._ray(p, Dd[v], torch.full((p.shape[0],), float(L)), S=S, ws=w1 * ws)
            else:
                col, A, muh, _, _, _, lse = self._point(p, ws=w1 * ws)
            if hops == 2:                                                  # hop 2: a point read at the location
                col, A, _, _, _, _, lse = self._point(muh, ws=ws)          # estimate (mean shift, coarse-to-fine)
            cols.append(col)
            conf.append(lse)
        cols, conf = torch.stack(cols), torch.stack(conf)                  # (V, n, dv), (V, n)
        wv = torch.softmax(conf, 0)                                        # views weighted by their log-confidence
        return (wv[..., None] * cols).sum(0)                               # (= opacity weighting when sum rho << 1)

    # ---------------- writes
    def write(self, a, v):
        self.step += 1
        P = self.project(a)
        B = a.shape[0]
        pts = P.reshape(-1, self.d_m)
        tv = v.repeat(self.V, 1)
        spawn = torch.ones(pts.shape[0], dtype=torch.bool)
        if self.M > 0:
            col, A, _, idx, rho, m, _ = self._point(pts)
            same = (A > 0.5) & (F.cosine_similarity(col, tv, dim=1) > 0.9)
            if same.any():                                                 # reinforce (consolidation signal)
                j = idx[same].gather(1, rho[same].argmax(1, keepdim=True))[:, 0]
                self.al[j] = (self.alpha_eff(j) + 1.0).clamp(max=self.amax)
                self.cons[j] = True
                self.tl[j] = self.step
                spawn &= ~same
                self.stats["reinforce"] += int(same.sum())
            if self.shrink:                                                # conflict: shrink splats with another
                cv = F.cosine_similarity(self.val[idx], tv[:, None], dim=-1)  # value that cover the new point
                bad = (m < 2.5 ** 2) & (cv < 0.5) & spawn[:, None]
                if bad.any():
                    fac = torch.ones(self.M)
                    f = (m.sqrt() / 2.5).clamp(min=0.3)
                    fac.scatter_reduce_(0, idx[bad], f[bad], reduce="amin")
                    # floor: never below 0.3 x the birth width (the shrinks compound otherwise)
                    self.ls = torch.maximum(self.ls + fac.log()[:, None], self.ls0 + math.log(0.3))
                    self.stats["shrink"] += int((fac < 1).sum())
        newp, newv = pts[spawn], tv[spawn]
        if newp.shape[0]:
            ref = torch.cat([self.mu, newp]) if self.M else newp
            j, d = topk_small(newp, ref, 4)
            sc = d[:, 1:].sqrt().mean(1)                                   # mean distance to the 3 nearest (3DGS)
            sg = (self.kappa * sc).clamp(self.smin, self.smax)
            self.mu = torch.cat([self.mu, newp])
            self.ls = torch.cat([self.ls, sg.log()[:, None].expand(-1, self.d_m)])
            self.ls0 = torch.cat([self.ls0, sg.log()[:, None].expand(-1, self.d_m)])
            self.al = torch.cat([self.al, torch.full((newp.shape[0],), self.alpha0)])
            self.val = torch.cat([self.val, newv])
            self.cons = torch.cat([self.cons, torch.zeros(newp.shape[0], dtype=torch.bool)])
            self.tl = torch.cat([self.tl, torch.full((newp.shape[0],), float(self.step))])
            self.stats["spawn"] += int(newp.shape[0])
        self.prune()

    def prune(self, thr=0.05):
        ae = self.alpha_eff()
        keep = ae > thr
        if self.M - int((~keep).sum()) > self.budget:                       # over budget: drop the weakest (oldest
            score = ae * (1 + self.cons.float()) + 1e-6 * self.tl          # on ties)
            order = score.argsort(descending=True)
            keep = torch.zeros(self.M, dtype=torch.bool)
            keep[order[:self.budget]] = True
            keep &= ae > thr
        if (~keep).any():
            self.stats["pruned"] += int((~keep).sum())
            for n in ("mu", "ls", "ls0", "al", "val", "cons", "tl"):
                setattr(self, n, getattr(self, n)[keep])

    def flops_read(self, mode="point", hops=1, views=1, S=8, M=None):
        """per query, with an index (IVF / grid) instead of brute force: candidates K, d_m, dv"""
        d, K, dv = self.d_m, self.K, self.dv
        per = (448 * self.wh.r * 2) / max(views, 1) + self.wh.r * d * 2
        if mode == "ray":
            k = 2 * K
            per += k * S * d * 4 + k * S * 4 + k * dv * 2
        else:
            per += K * d * 4 + K * dv * 2
        if hops == 2:
            per += K * d * 4 + K * dv * 2
        return views * per


# ----------------------------------------------------------------------------------------------- gradient splats
class SplatGrad:
    """M fixed Gaussian centres (sampled thoughts, view 0, d_m dims), sigma = kappa * 3-NN spacing, values written by
    the delta rule through the normalised kernel weights over the top-K centres (a soft-row memory)."""

    def __init__(self, field_like, M, centres_a, dv, K=8, kappa=1.0):
        self.f, self.K = field_like, K
        self.mu = field_like.project(centres_a)[0][:M]
        j, d = topk_small(self.mu, self.mu, 4)
        self.sg = (kappa * d[:, 1:].sqrt().mean(1)).clamp_min(1e-3)
        self.V = torch.zeros(self.mu.shape[0], dv)

    def address(self, a):
        p = self.f.project(a)[0]
        idx, d = topk_small(p, self.mu, self.K)
        m = d / self.sg[idx] ** 2
        w = torch.softmax(-0.5 * m, dim=1)
        return idx, w

    def read(self, a):
        idx, w = self.address(a)
        return (w[..., None] * self.V[idx]).sum(1)

    def write(self, a, v, n_it=3, beta=1.0):
        idx, w = self.address(a)
        nrm = (w * w).sum(1)
        for _ in range(n_it):
            r = (w[..., None] * self.V[idx]).sum(1)
            err = (v - r) * (beta / nrm)[:, None]
            self.V.index_add_(0, idx.flatten(), (w[..., None] * err[:, None]).reshape(-1, self.V.shape[1]))
