"""The wave memory's ADDRESS (plan_v2/26 sections 6, 8.3 and 8.4): a code over neurons computed from the brain's own thought.

    xf (N, D) --detach--> x_hat = ((xf - mu) / sd) @ rot.T                     (running statistics, fixed rotation)
      entry map:  L = side x side prototypes on a torus (an online SOM-like competitive map, "roughly the person");
                  the top-`entries` locations by -|x_hat - W_l|^2 + b_loc (b_loc: usage homeostasis)
      level 1..n: the residual of x_hat after the previous level's prototypes (level 1: x_hat - W[location 1]),
                  cut into `slices` slices of D / slices dims; per slice the top-`topk` of `codes` prototypes
                  (soft top-2: weights from the distance gap); the residual continues with the top-1 prototype

  This is the idealised coarse-to-fine cue "(c+d) SOM + 3 residual levels x 32 slices, top-2" of the W-bench
  (speed_wip/mem26/wbench.py, m_entry_code("som", 256, 32, 16, r=2, coarse_fine=3)): 0.524 cloze / 0.684 held top-1
  over passes 1-5 there, reproduced with this module's own in-step learning by tests_wave_bench.py.

Learning (update(), called by the memory inside the training step, on DETACHED thoughts; no gradient anywhere):
  * statistics: running mean / variance of xf (count-capped, so they follow a slow drift)
  * entry map: batched SOM step - each row's winner under a usage-homeostasis penalty, a Gaussian torus neighbourhood
    (sigma annealed sigma0 -> sigma1 over `anneal` updates, then held), every prototype moved toward its
    neighbourhood-weighted batch mean at rate 1 - exp(-eta * sum h); b_loc pushed toward equal read usage
  * residual levels: minibatch k-means per (level, slice) with count-capped per-prototype rates; prototypes whose use
    falls below dead_frac / codes are re-seeded on the batch's worst-coded rows of that slice (deterministic)
  * initialisation: the first update seeds the map and every level from that batch's rows (a seeded permutation)
  * ready once `warm` updates are done
  * two speeds (`fast` > 0): the rates above for the first `fast` updates (the warm-up, about the memory's first
    1,000 steps), then ONE slow per-update rate eta_slow for everything (the statistics, each prototype by its share
    of the batch, the location bias, each code by its use; no re-seeding), whatever the batch size, so the codes under
    which facts were stored stay put while still following a slow drift of the thoughts

Everything is batched over all rows (tokens x passes; the experts share one xf per token), deterministic under the
seed (the only random draws: the rotation and the initial permutation, from a generator seeded by `seed`). The state
is buffers (saved with the memory) plus host counters in the extra state.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_f(idx, k):
    """F.one_hot(idx, k).float() without its host sync (F.one_hot checks idx's min / max on the host): a scatter of
    1.0 into zeros, the same values (idx must lie in [0, k))."""
    return torch.zeros(*idx.shape, k, device=idx.device).scatter_(-1, idx[..., None], 1.0)


def _count(idx, k):
    """torch.bincount(idx, minlength=k).float() without its host sync (bincount reads idx's max): an index_add of
    1.0 (exact integer counts below 2^24)."""
    return torch.zeros(k, device=idx.device).index_add_(0, idx, torch.ones(idx.numel(), device=idx.device))


class Codes:
    """One batch of addresses. loc (n, E) long, loc_s (n, E) scores; code (n, levels * slices, r) long, code_w (same)
    float (soft weights, rows sum to 1), code_d (n, levels * slices, r) distances; feats (n, F) detached features;
    code_e (n, E, levels * slices) long or None: the top-1 codes of each entry's OWN residual (x_hat - W[loc[:, j]];
    entry 0 == code[..., 0]), built when the address's entry_keys is on (the column store's second entry)."""
    __slots__ = ("loc", "loc_s", "code", "code_w", "code_d", "feats", "code_e")

    def __init__(self, loc, loc_s, code, code_w, code_d, feats, code_e=None):
        self.loc, self.loc_s, self.code, self.code_w, self.code_d, self.feats = loc, loc_s, code, code_w, code_d, feats
        self.code_e = code_e

    def take(self, idx):
        return Codes(*(None if getattr(self, k) is None else getattr(self, k)[idx] for k in self.__slots__))

    def __len__(self):
        return self.loc.size(0)

    @staticmethod
    def cat(parts):
        return Codes(*(None if getattr(parts[0], k) is None else torch.cat([getattr(p, k) for p in parts])
                       for k in Codes.__slots__))

    @staticmethod
    def pick(per, best):
        """per: a list of Codes of the same rows; best (n,) -> row i from per[best[i]]."""
        ar = torch.arange(best.numel(), device=best.device)
        return Codes(*(None if getattr(per[0], k) is None else torch.stack([getattr(p, k) for p in per])[best, ar]
                       for k in Codes.__slots__))


FEAT_NAMES = ("loc_margin", "lev_margin", "res_frac")       # (features per level: margin and residual fraction)


class WaveAddress(nn.Module):
    def __init__(self, d, side=16, levels=3, slices=32, codes=16, topk=2, entries=2, seed=1234, warm=200,
                 anneal=2000, sigma0=3.0, sigma1=0.5, eta0=0.2, eta1=0.01, homeo=0.5, bias_rate=0.1, n_cap=4096,
                 stat_cap=1_000_000, dead_frac=0.1, soft_tau=0.5, fast=0, eta_slow=1e-4):
        super().__init__()
        assert d % slices == 0, f"wave address: d {d} is not a multiple of --wave-slices {slices}"
        self.d, self.side, self.L, self.levels, self.S, self.M = d, side, side * side, levels, slices, codes
        self.ds, self.r, self.E = d // slices, topk, entries
        self.seed, self.warm, self.anneal = int(seed), int(warm), int(anneal)
        self.sigma0, self.sigma1, self.eta0, self.eta1, self.homeo = sigma0, sigma1, eta0, eta1, homeo
        self.bias_rate, self.n_cap, self.stat_cap, self.dead_frac, self.soft_tau = (bias_rate, n_cap, stat_cap,
                                                                                    dead_frac, soft_tau)
        # two speeds: the rates above for the first `fast` updates (0 = always fast), then the slow ones (the codes of
        # stored facts must stay put: plan 26, address drift)
        self.fast, self.eta_slow = int(fast), float(eta_slow)
        g = torch.Generator().manual_seed(self.seed)
        rot = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0].float()
        self.register_buffer("rot", rot)                                   # fixed (from the seed)
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("var", torch.ones(d))
        self.register_buffer("n_stat", torch.zeros((), dtype=torch.float64))
        self.register_buffer("W", torch.zeros(self.L, d))                  # the entry map (SOM prototypes)
        self.register_buffer("som_use", torch.full((self.L,), 1.0 / self.L))
        self.register_buffer("loc_use", torch.full((self.L,), 1.0 / self.L))
        self.register_buffer("b_loc", torch.zeros(self.L))                 # read-side usage homeostasis
        self.register_buffer("w_hits", torch.zeros(self.L))                # (EMA of writes per place: learn_writes)
        self.register_buffer("sc", torch.zeros(()))                        # median |x_hat|^2 (homeostasis scale)
        self.register_buffer("books", torch.zeros(levels, slices, codes, self.ds))
        self.register_buffer("code_n", torch.zeros(levels, slices, codes))
        self.register_buffer("code_use", torch.full((levels, slices, codes), 1.0 / codes))
        yy, xx = torch.div(torch.arange(self.L), side, rounding_mode="floor"), torch.arange(self.L) % side
        dy = (yy[:, None] - yy[None]).abs()
        dx = (xx[:, None] - xx[None]).abs()
        dy, dx = torch.minimum(dy, side - dy), torch.minimum(dx, side - dx)
        self.register_buffer("tor_d2", (dy * dy + dx * dx).float(), persistent=False)   # (derived: rebuilt)
        self.n_upd = 0                      # host counter (extra state of the memory)
        self.frozen = False                 # (train-only mode: no learning)
        self.entry_keys = False             # (encode also codes each further entry's own residual: Codes.code_e)

    # ------------------------------------------------------------------ host state
    def host_state(self):
        return dict(n_upd=self.n_upd)

    def load_host_state(self, st):
        self.n_upd = int(st.get("n_upd", 0))

    @property
    def ready(self):
        return self.n_upd >= self.warm

    @property
    def n_units(self):
        """Code units per location: levels x slices x codes."""
        return self.levels * self.S * self.M

    # ------------------------------------------------------------------ encode (the read side)
    @torch.no_grad()
    def standardize(self, xf):
        x = xf.detach().float()
        return ((x - self.mu) * torch.rsqrt(self.var + 1e-6)) @ self.rot.T

    def _loc_scores(self, X):
        return 2 * X @ self.W.T - (self.W * self.W).sum(1)                 # = -|x - W|^2 + |x|^2

    @torch.no_grad()
    def encode(self, xf, standardized=False, chunk=8192):
        """xf (n, D) -> Codes. No gradient (detached)."""
        out = []
        for a in range(0, xf.size(0), chunk):
            out.append(self._encode(xf[a:a + chunk] if standardized else self.standardize(xf[a:a + chunk])))
        return out[0] if len(out) == 1 else Codes.cat(out)

    def _encode(self, X):
        n = X.size(0)
        s = self._loc_scores(X) + self.b_loc
        top = s.topk(min(max(self.E, 2), self.L), -1)
        loc, loc_s = top.indices[:, :self.E], top.values[:, :self.E]
        sd = s.std(-1).clamp(min=1e-6)
        feats = [(top.values[:, 0] - top.values[:, 1]) / sd]
        x2 = (X * X).sum(-1).clamp(min=1e-9)
        R = (X - self.W[loc[:, 0]]).view(n, self.S, self.ds)
        codes, ws, dists = [], [], []
        for lev in range(self.levels):
            B = self.books[lev]                                            # (S, M, ds)
            dd = ((R * R).sum(-1, keepdim=True) - 2 * torch.einsum("nsd,smd->nsm", R, B)
                  + (B * B).sum(-1)[None]).clamp(min=0)                     # (n, S, M)
            t = dd.topk(min(max(self.r, 2), self.M), -1, largest=False)
            d1 = t.values[..., :1]
            gap = (t.values[..., :self.r] - d1) / (self.soft_tau * d1.mean((1, 2), keepdim=True) + 1e-9)
            w = torch.softmax(-gap, -1)
            codes.append(t.indices[..., :self.r])
            ws.append(w)
            dists.append(t.values[..., :self.r])
            feats.append(((t.values[..., 1] - t.values[..., 0]) / (t.values[..., 0] + 1e-9)).clamp(max=10).mean(-1))
            R = R - B[torch.arange(self.S, device=X.device)[None], t.indices[..., 0]]
            feats.append(((R * R).sum((1, 2)) / x2).sqrt())
        code = torch.cat(codes, 1)
        code_e = None
        if self.entry_keys:                 # (entry j >= 1: its own residual x_hat - W[loc[:, j]], top-1 per level)
            per = [code[..., 0]]
            for j in range(1, loc.size(1)):
                per.append(self._top1_codes(X - self.W[loc[:, j]]))
            code_e = torch.stack(per, 1)                                   # (n, E, levels * S)
        return Codes(loc, loc_s, code, torch.cat(ws, 1), torch.cat(dists, 1), torch.stack(feats, -1), code_e)

    def _top1_codes(self, R):
        """residual (n, D) -> the top-1 code per (level, slice) (n, levels * S), the residual continuing per level."""
        n = R.size(0)
        R = R.view(n, self.S, self.ds)
        ar = torch.arange(self.S, device=R.device)[None]
        out = []
        for lev in range(self.levels):
            B = self.books[lev]
            dd = (R * R).sum(-1, keepdim=True) - 2 * torch.einsum("nsd,smd->nsm", R, B) + (B * B).sum(-1)[None]
            a = dd.argmin(-1)                                              # (first index on ties)
            out.append(a)
            R = R - B[ar, a]
        return torch.cat(out, 1)

    @property
    def n_feats(self):
        return 1 + 2 * self.levels

    # ------------------------------------------------------------------ learning (in-step, local, no gradient)
    @property
    def slow(self):
        return bool(self.fast) and self.n_upd >= self.fast

    def _sched(self):
        if self.slow:
            return self.sigma1, self.eta_slow
        f = min(1.0, self.n_upd / max(1, min(self.anneal, self.fast) if self.fast else self.anneal))
        return self.sigma0 * (self.sigma1 / self.sigma0) ** f, self.eta0 * (self.eta1 / self.eta0) ** f

    @torch.no_grad()
    def update(self, xf):
        """One competitive / homeostatic learning step on a batch of DETACHED thoughts xf (n, D)."""
        if self.frozen or xf.size(0) == 0:
            return
        x = xf.detach().float()
        n = x.size(0)
        # statistics (count-capped running mean / variance)
        bm, bv = x.mean(0), x.var(0, unbiased=False)
        n0 = float(self.n_stat)
        slow = self.slow
        f = self.eta_slow if slow else n / (min(n0, self.stat_cap) + n)        # (slow: a fixed per-update rate)
        if n0 == 0:
            self.mu.copy_(bm)
            self.var.copy_(bv)
        else:
            dm = bm - self.mu
            self.var.copy_((1 - f) * self.var + f * bv + f * (1 - f) * dm * dm)
            self.mu.add_(f * dm)
        self.n_stat.add_(n)
        X = self.standardize(x)
        med = (X * X).sum(-1).median()
        self.sc.copy_(med if self.n_upd == 0 else self.sc.lerp(med, 0.05))
        if self.n_upd == 0:
            self._init(X)
        sig, eta = self._sched()
        # ---- the entry map: a batched SOM step with usage homeostasis
        d2 = (X * X).sum(-1, keepdim=True) - 2 * X @ self.W.T + (self.W * self.W).sum(1)
        c = (d2 + self.homeo * self.sc * torch.log(self.som_use * self.L)).argmin(1)
        h = torch.exp(-self.tor_d2[c] / (2 * sig * sig))
        h = torch.where(h > 1e-3, h, torch.zeros_like(h))                  # (n, L)
        hs = h.sum(0)
        tgt = (h.T @ X) / hs.clamp(min=1e-9)[:, None]
        # (fast: each row moves its neighbourhood by eta; slow: each prototype moves by eta_slow per update in
        # proportion to its share of the batch, whatever the batch size)
        rate = 1 - torch.exp(-eta * (hs / max(1.0, n / self.L) if slow else hs))
        self.W.add_(rate[:, None] * (tgt - self.W))
        hist = _count(c, self.L) / n
        dec = 0.999 ** n
        self.som_use.mul_(dec).add_((1 - dec) * hist)
        # ---- the read-side location bias toward equal usage (the read's winner under the bias)
        s = self._loc_scores(X)
        win = (s + self.b_loc).argmax(1)
        use = _count(win, self.L) / n
        self.loc_use.lerp_(use, 0.3)
        br = self.eta_slow if slow else self.bias_rate
        self.b_loc.sub_(br * s.std() * torch.log((self.loc_use + 0.1 / self.L) * self.L))
        self.b_loc.sub_(self.b_loc.mean())
        # ---- the residual levels: minibatch k-means per (level, slice), dead prototypes re-seeded
        R = (X - self.W[win]).view(n, self.S, self.ds)
        ar = torch.arange(self.S, device=X.device)
        for lev in range(self.levels):
            B = self.books[lev]
            dd = ((R * R).sum(-1, keepdim=True) - 2 * torch.einsum("nsd,smd->nsm", R, B) + (B * B).sum(-1)[None])
            a = dd.argmin(-1)                                              # (n, S)
            oh = one_hot_f(a, self.M)                                      # (n, S, M)
            cnt = oh.sum(0)                                                # (S, M)
            sums = torch.einsum("nsm,nsd->smd", oh, R)
            new = self.code_n[lev] + cnt
            if slow:                                                       # (a fixed per-update rate, by use)
                rt = (self.eta_slow * (cnt / max(1.0, n / self.M)).clamp(max=4.0))[..., None]
            else:
                rt = (cnt / new.clamp(min=1))[..., None]
            B.add_(rt * (sums / cnt.clamp(min=1)[..., None] - B))
            self.code_n[lev].copy_(new.clamp(max=self.n_cap))
            self.code_use[lev].mul_(0.95).add_(0.05 * cnt / n)
            err = (R - B[ar[None], a]).pow(2).sum(-1)                      # (n, S) after the update
            dead = self.code_use[lev] < self.dead_frac / self.M            # (S, M)
            if slow:                                                       # (no re-seeding: codes must stay put)
                dead = torch.zeros_like(dead)
            k = min(self.M, n)
            worst = err.topk(k, 0).indices                                 # (k, S) rows, worst first
            rank = (dead.long().cumsum(-1) - 1).clamp(min=0, max=k - 1)    # the j-th dead code of a slice
            pick = worst.gather(0, rank.T).T                               # (S, M) row for each code
            seed = R[pick, ar[:, None]]                                    # (S, M, ds)
            B.copy_(torch.where(dead[..., None], seed, B))
            self.code_n[lev].copy_(torch.where(dead, torch.ones_like(cnt), self.code_n[lev]))
            self.code_use[lev].copy_(torch.where(dead, torch.full_like(cnt, 1.0 / self.M), self.code_use[lev]))
            a = torch.where(dead.any(-1)[None], ((R[:, :, None] - B[None]).pow(2).sum(-1)).argmin(-1), a)
            R = R - B[ar[None], a]
        self.n_upd += 1

    @torch.no_grad()
    def learn_writes(self, xf, beta, chunk=32, decay=0.99, mask=None):
        """Fix A (place balance): the entry map also learns on the WRITE load. Each written thought's winner under a
        write-count conscience, BMU = argmin |x_hat - P|^2 + beta sc log((wh + 1) / (mean wh + 1)) (wh: EMA of writes
        per place, x decay per chunk of `chunk` writes = one step's), moves its prototype toward its rows' mean at 1 - exp(-eta n) (eta: the fast rate
        in the warm-up, then the slow one: the two-speed rule); sums by a one-hot GEMM. No neighbourhood, no bias."""
        if self.frozen or xf.size(0) == 0 or self.n_upd == 0:
            return
        _, eta = self._sched()
        if mask is not None:              # (static shapes, no host sync: chunk a = the kept rows of rank 32 a ..
            rk = mask.long().cumsum(0) - 1                 # 32 a + 31 (== the compacted loop's chunks); the others weigh 0)
            cid = torch.where(mask, rk // chunk, torch.full_like(rk, -1))
            X = self.standardize(xf)
            for a in range((xf.size(0) + chunk - 1) // chunk):
                wa = (cid == a).float()
                d2 = (X * X).sum(-1, keepdim=True) - 2 * X @ self.W.T + (self.W * self.W).sum(1)
                pen = beta * self.sc * torch.log((self.w_hits + 1) / (self.w_hits.mean() + 1))
                c = (d2 + pen).argmin(1)
                oh = one_hot_f(c, self.L) * wa[:, None]
                cnt = oh.sum(0)
                tgt = (oh.T @ X) / cnt.clamp(min=1)[:, None]
                rate = 1 - torch.exp(-eta * cnt)
                self.W.add_(rate[:, None] * (tgt - self.W))
                self.w_hits.copy_(torch.where(wa.sum() > 0, self.w_hits * decay + cnt, self.w_hits))
            return
        for a in range(0, xf.size(0), chunk):          # (the conscience moves per chunk: one step's writes)
            X = self.standardize(xf[a:a + chunk])
            d2 = (X * X).sum(-1, keepdim=True) - 2 * X @ self.W.T + (self.W * self.W).sum(1)
            pen = beta * self.sc * torch.log((self.w_hits + 1) / (self.w_hits.mean() + 1))
            c = (d2 + pen).argmin(1)
            oh = one_hot_f(c, self.L)                                      # (n, L)
            cnt = oh.sum(0)
            tgt = (oh.T @ X) / cnt.clamp(min=1)[:, None]
            rate = 1 - torch.exp(-eta * cnt)
            self.W.add_(rate[:, None] * (tgt - self.W))
            self.w_hits.mul_(decay).add_(cnt)

    def _init(self, X):
        """Seed the map and every level from the first batch (a permutation from the seed)."""
        g = torch.Generator().manual_seed(self.seed + 1)
        n = X.size(0)
        perm = torch.randperm(n, generator=g)
        idx = perm[torch.arange(self.L) % n].to(X.device)
        self.W.copy_(X[idx])
        R = (X - self.W[self._loc_scores(X).argmax(1)]).view(n, self.S, self.ds)
        ar = torch.arange(self.S, device=X.device)
        for lev in range(self.levels):
            p = torch.stack([torch.randperm(n, generator=g)[torch.arange(self.M) % n] for _ in range(self.S)], 1)
            self.books[lev].copy_(R[p.to(X.device), ar[None]].transpose(0, 1))        # (S, M, ds)
            B = self.books[lev]
            a = ((R[:, :, None] - B[None]).pow(2).sum(-1)).argmin(-1)
            R = R - B[ar[None], a]

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def stats(self):
        u = self.loc_use.clamp(min=1e-12)
        su = self.code_use.clamp(min=1e-12)
        return dict(loc_use_entropy=float(-(u * u.log()).sum() / math.log(self.L)),
                    loc_use_max=float(self.loc_use.max() * self.L),
                    code_entropy=float((-(su * su.log()).sum(-1) / math.log(self.M)).mean()),
                    dead_codes=float((self.code_use < self.dead_frac / self.M).float().mean()),
                    n_upd=self.n_upd, ready=int(self.ready), slow=int(self.slow))
