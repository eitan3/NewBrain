"""The blob memory's read gate (plan_v2/25d_global_expert_blob_plan.md rev 2, section 5; phase 2).

One decision per (token, expert) covers the global + private read (D4). Inputs are all DETACHED (the gate never pushes
the thought):
  token features phi_t: xf (d) | side(s) | the previous pass's mu_l of every concept level (L x d, zeros when none) |
  the global match now (6: A_g, d1/h0^2, log(d2/d1), maturity/3, log(1 + age), empty flag) | the previous read
  (6: A_g, log(d2/d1)_g, d1/h0^2, mean / max A_p over the token's pairs, the fraction of its pairs that read) |
  "no previous read" flag | the previous pass's halting confidence + its "none" flag | the content-fit score
  f_g = sigmoid(n(xf) . n(W_cf u_g)) (computed inside score(), W_cf is a gate parameter);
  pair features m_{t,e}: the expert's private match (the same 6) + f_p = sigmoid(n(xf) . n(W_cf u_p)).
Each token group is RMS-normalised.

    z      = GELU(W1 phi + b1)                                  (shared trunk, hidden 32)
    g_hat  = (a0 + P c_e) . z + beta_e + (g0 + g_e) . m          (rank-4 per-expert head P c_e, per-expert bias beta_e)

Rows are preallocated for max_experts + 1 (the shared expert = the last row, as the private spaces); split births copy
a parent's rows AND its AdamW moments (copy_rows). The gate is NOT a submodule of the model: its parameters stay out of
make_opt / carry_optimizer / the global clip (BlobMemory holds it outside the module registry and saves it in its extra
state). It is trained after backward, only on the exploration pairs, by a separate small graph (post_backward):
Huber(g_hat - y) weighted 1 / propensity, y clipped at 5 robust sigmas; its own AdamW (lr 1e-3; weight decay 0.01 on
the per-expert head / match weights only).

Selection (causal, H2): open = (g_hat > lambda_e) or explore; lambda_e <- [lambda_e + 0.01 (rate_e - 0.5)]_+ after
each optimizer step from the step's in-window non-exploration pairs. Exploration: a Bernoulli draw per pair with a
known propensity per stratum of the global match A_g (4 strata, edges = an EMA of the quartiles; draw probabilities
proportional to the target shares 0.15 / 0.20 / 0.30 / 0.35, mean 5%): per pair, so an earlier position's draw never
depends on a later one (causal).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MATCH = 8     # match features per space (without the content-fit score)
N_PREV = 6      # previous-read features


def hash_uniform(a, b, salt):
    """A counter-based uniform in [0, 1) per (a, b) pair (int64 tensors) and a host salt: the draw of one (token,
    expert) pair never depends on any other pair (so exploration is causal), and a recompute gives the same draw."""
    m = 0xFFFFFFFF
    x = (a.long() * 0x9E3779B1 + b.long() * 0x85EBCA77 + int(salt)) & m
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & m
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & m
    x = x ^ (x >> 16)
    return x.double().div(4294967296.0).float()


def nan_quantile(x, q):
    """The q-quantile of the non-NaN entries of x (linear interpolation; NaN when there are none), sort-based and
    sync-free (torch.quantile checks q on the host, and a 0-d tensor index would call .item())."""
    x = x.flatten().float()
    if x.numel() == 0:
        return x.new_tensor(float("nan"))
    xs = x.sort().values                                                  # NaNs sort last
    n = (~torch.isnan(x)).sum()
    pos = q * (n - 1).clamp(min=0).float()
    lo = pos.floor().long().clamp(max=x.numel() - 1).view(1)
    hi = pos.ceil().long().clamp(max=x.numel() - 1).view(1)
    a, b = xs.gather(0, lo), xs.gather(0, hi)
    v = (a + (b - a) * (pos - pos.floor())).squeeze(0)
    return torch.where(n > 0, v, torch.full_like(v, float("nan")))


_CONST = {}


def _const(vals, device, dtype=None):
    """A cached device copy of a constant list (a fresh torch.tensor(list, device=cuda) is a blocking H2D copy)."""
    key = (tuple(vals), str(device), dtype)
    t = _CONST.get(key)
    if t is None:
        t = _CONST[key] = torch.tensor(list(vals), device=device, dtype=dtype)
    return t


def _rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


PRIOR_COLS = (6, 2, 7)   # log d1, log(d2/d1), log(d2 - d1): the match-quality prior's inputs
PRIOR_SIGN = (-1.0, 1.0, 1.0)
PRIOR_W0 = 0.01
YRING = 1 << 20     # (r6 G3: rows of the static (g_hat, y, ok) logging ring)
STATS_KEEP = 512    # (r6 G3: per-call open-rate rows kept for the diag line; older ones are folded into a sum)


def match_features(A, d1, d2, h0, mature, age, empty):
    """The 8 match features of one space (all (n,)): A, d1/h0^2, log(d2/d1), maturity/3, log(1 + age), empty flag,
    log d1, log(d2 - d1)."""
    h2 = h0 * h0
    ok = ~empty
    fin = ok & torch.isfinite(d2)
    r = torch.where(fin, torch.log((d2 + 1e-6) / (d1 + 1e-6)), torch.zeros_like(d1))
    f = torch.stack([A, torch.where(ok, d1 / h2, torch.zeros_like(d1)), r, torch.where(ok, mature / 3.0, 0 * d1),
                     torch.where(ok, torch.log1p(age.clamp(min=0)), 0 * d1), empty.float(),
                     torch.where(ok, torch.log(d1 + 1e-6), torch.zeros_like(d1)),
                     torch.where(fin, torch.log((d2 - d1).clamp(min=0) + 1e-6), torch.zeros_like(d1))], -1)
    return f.float()


class BlobGate(nn.Module):
    def __init__(self, d, d_side, levels, n_rows, hidden=32, rank=4, beta0=-0.02, seed=4321, warm=200):
        super().__init__()
        self.d, self.d_side, self.L, self.hidden, self.rank = d, d_side, levels, hidden, rank
        self.n_tok = d + d_side + levels * d + N_MATCH + N_PREV + 1 + 2 + 1     # + f_g
        self.n_m = N_MATCH + 1                                                 # + f_p
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(hidden, self.n_tok, generator=g) / math.sqrt(self.n_tok))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.a0 = nn.Parameter(torch.zeros(hidden))
        self.P = nn.Parameter(torch.randn(hidden, rank, generator=g) / math.sqrt(hidden))
        self.c = nn.Parameter(torch.zeros(n_rows, rank))
        self.beta = nn.Parameter(torch.full((n_rows,), float(beta0)))
        self.g0 = nn.Parameter(torch.zeros(self.n_m))
        self.ge = nn.Parameter(torch.zeros(n_rows, self.n_m))
        self.W_cf = nn.Parameter(torch.eye(d))
        # the match-quality prior (no labels, no fitted coefficients): fixed-sign unit weights on the standardised
        # (log d1, log d2/d1, log(d2 - d1)) of the global and the private match, learnable from there
        # (r6 G2: small, 0.01, so g_hat starts in the standardised target's units; was 0.5 = ~200x the target spread)
        self.pg = nn.Parameter(torch.tensor(PRIOR_SIGN) * PRIOR_W0)
        self.pp = nn.Parameter(torch.tensor(PRIOR_SIGN) * PRIOR_W0)
        self.register_buffer("fstat", torch.zeros(2, 3, 2))                  # running mean / var of the prior inputs
        self.register_buffer("fstat_n", torch.zeros((), dtype=torch.long))
        # the open-rate bias: set from the gate's own score quantile during `warm` steps (~gate_open0 open), then fixed
        self.register_buffer("open_bias", torch.zeros(()))
        # (r6 G2: EMA of the robust target scale 1.4826 MAD(y); the loss regresses y / y_sc with delta 1)
        self.register_buffer("y_sc", torch.zeros(()))
        self.register_buffer("warm_left", torch.tensor(int(warm), dtype=torch.long))
        self.register_buffer("lam", torch.zeros(n_rows))                     # per-expert thresholds
        self.register_buffer("cnt_open", torch.zeros(n_rows))                # this step: non-exploration pairs ...
        self.register_buffer("cnt_all", torch.zeros(n_rows))                 # ... above lambda / all of them
        self.register_buffer("edges", torch.tensor([0.25, 0.5, 0.75]))       # A_g strata edges (EMA of quartiles)
        self.register_buffer("rate_sum", torch.zeros(n_rows), persistent=False)   # (health: open rates per expert)
        self.register_buffer("rate_n", torch.zeros(n_rows), persistent=False)
        self.shares = (0.15, 0.20, 0.30, 0.35)
        self.opt = None
        self.lr, self.wd = 1e-3, 0.01
        self.beta0 = beta0

    # ------------------------------------------------------------------ optimizer (own; out of the brain's paths)
    def make_opt(self):
        head = [self.c, self.ge]
        rest = [p for p in self.parameters() if all(p is not h for h in head)]
        self.opt = torch.optim.AdamW([dict(params=rest, weight_decay=0.0), dict(params=head, weight_decay=self.wd)],
                                     lr=self.lr, betas=(0.9, 0.99))
        return self.opt

    @torch.no_grad()
    def copy_rows(self, src, dst):
        """Split births: the children get their parents' per-expert rows and AdamW moments; lambda too."""
        for p in (self.c, self.beta, self.ge):
            p.data[dst] = p.data[src]
            st = self.opt.state.get(p) if self.opt is not None else None
            if st:
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in st:
                        st[k][dst] = st[k][src]
        self.lam[dst] = self.lam[src]

    @torch.no_grad()
    def reset_rows(self, dst):
        """Ramp births: the shared head (zero per-expert parts, beta0, lambda 0), zero moments."""
        self.c.data[dst] = 0
        self.ge.data[dst] = 0
        self.beta.data[dst] = self.beta0
        self.lam[dst] = 0
        for p in (self.c, self.beta, self.ge):
            st = self.opt.state.get(p) if self.opt is not None else None
            if st:
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in st:
                        st[k][dst] = 0

    @torch.no_grad()
    def grow(self, n_rows):
        """More rows than preallocated (the shared row stays last): rebuilds the parameters and the optimizer moments."""
        S = self.beta.size(0)
        if n_rows <= S:
            return
        old_shared = S - 1
        extra = n_rows - S

        def ext(t, fill=0.0):
            new = torch.cat([t, t.new_full((extra, *t.shape[1:]), fill)])
            new[n_rows - 1] = t[old_shared]
            new[old_shared] = fill
            return new
        state = {}
        if self.opt is not None:
            for p in (self.c, self.beta, self.ge):
                st = self.opt.state.get(p)
                if st:
                    state[id(p)] = {k: (ext(v) if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in st.items()}
        olds = (self.c, self.beta, self.ge)
        self.c = nn.Parameter(ext(self.c.data))
        self.beta = nn.Parameter(ext(self.beta.data, self.beta0))
        self.ge = nn.Parameter(ext(self.ge.data))
        for n in ("lam", "cnt_open", "cnt_all", "rate_sum", "rate_n"):
            setattr(self, n, ext(getattr(self, n)))
        if self.opt is not None:
            other = {id(p): self.opt.state[p] for p in self.parameters() if p in self.opt.state and
                     all(p is not o for o in olds)}
            self.make_opt()
            for p, o in zip((self.c, self.beta, self.ge), olds):
                if id(o) in state:
                    self.opt.state[p] = state[id(o)]
            for p in self.parameters():
                if id(p) in other:
                    self.opt.state[p] = other[id(p)]

    # ------------------------------------------------------------------ features and score
    def token_features(self, xf, side, mus, gmatch, prev, noprev, halt, halt_none):
        """phi without the content-fit score (n, n_tok - 1), each group RMS-normalised; all inputs detached."""
        n = xf.size(0)
        parts = [_rms(xf.float())]
        if self.d_side:
            parts.append(_rms(side.reshape(n, -1).float()) if side is not None else parts[0].new_zeros(n, self.d_side))
        for l in range(self.L):
            m = mus[l] if mus is not None and l < len(mus) and mus[l] is not None else None
            parts.append(_rms(m.reshape(n, -1).float()) if m is not None else parts[0].new_zeros(n, self.d))
        parts += [gmatch.float(), prev.float(), noprev.float()[:, None], halt.float()[:, None],
                  halt_none.float()[:, None]]
        return torch.cat(parts, -1).detach()

    def fit_score(self, xf, u):
        """sigmoid(n(xf) . n(W_cf u)) per row."""
        return torch.sigmoid((F.normalize(xf.float(), dim=-1) * F.normalize(u.float() @ self.W_cf.t(), dim=-1)).sum(-1))

    def score(self, phi, xf_t, u_g, m, u_p, e, tok, bias=True):
        """phi (n_tok, F-1) token features, xf_t / u_g (n_tok, d); pair rows: m (n, 6), u_p (n, d), expert rows e (n,),
        token index tok (n,) -> g_hat (n,). bias: + open_bias (the selection's open-rate offset; r6 G1: the loss
        regresses the score WITHOUT it, so the warm-up's bias never feeds back into the gate's training)."""
        fg = self.fit_score(xf_t, u_g)
        z = F.gelu(torch.cat([phi, fg[:, None]], -1) @ self.W1.t() + self.b1)            # (n_tok, hidden)
        fp = self.fit_score(xf_t[tok], u_p)
        mm = torch.cat([m.float(), fp[:, None]], -1)
        a = self.a0 + self.c[e] @ self.P.t()                                              # (n, hidden)
        raw = (a * z[tok]).sum(-1) + self.beta[e] + (mm * (self.g0 + self.ge[e])).sum(-1) + self.prior(phi, m, tok)
        return raw + self.open_bias if bias else raw

    def _z(self, x, k):
        mu, var = self.fstat[k, :, 0], self.fstat[k, :, 1]
        return (x - mu) / (var.clamp(min=1e-6)).sqrt()

    def prior(self, phi, m, tok):
        """The match-quality prior: pg . z(global) + pp . z(private) on the standardised prior inputs."""
        off = self.n_tok - 1 - (N_MATCH + N_PREV + 1 + 2)                    # where the global match starts in phi
        gcols = _const([off + c for c in PRIOR_COLS], phi.device)
        pcols = _const(PRIOR_COLS, m.device)
        zg = self._z(phi.index_select(1, gcols), 0)[tok]
        zp = self._z(m.float().index_select(1, pcols), 1)
        return (zg * self.pg).sum(-1) + (zp * self.pp).sum(-1)

    @torch.no_grad()
    def observe_prior(self, gm, pm, rate=0.01):
        """Running mean / variance of the prior inputs (training forwards only; no labels)."""
        cols = _const(PRIOR_COLS, gm.device)
        for k, f in ((0, gm.index_select(1, cols)), (1, pm.index_select(1, cols))):
            mu, var = f.mean(0), f.var(0, unbiased=False)
            first = self.fstat_n == 0
            self.fstat[k, :, 0] = torch.where(first, mu, self.fstat[k, :, 0].lerp(mu, rate))
            self.fstat[k, :, 1] = torch.where(first, var, self.fstat[k, :, 1].lerp(var, rate))
        self.fstat_n += 1

    # ------------------------------------------------------------------ exploration and the threshold
    @torch.no_grad()
    def explore(self, A, frac, u):
        """Per-pair Bernoulli exploration with a known propensity per A_g stratum (mean `frac` when the strata hold a
        quarter each); u: the pairs' own uniforms (hash_uniform). Returns (mask, propensity)."""
        stratum = (A[:, None] > self.edges.to(A.device)[None]).sum(-1)                     # 0 = far .. 3 = near
        p = _const(self.shares, A.device)[stratum] * (frac / 0.25)
        return u < p, p

    @torch.no_grad()
    def count(self, e, ghat, explore, lam):
        """In-window bookkeeping of the step's open rates (non-exploration pairs)."""
        w = (~explore).float()
        self.cnt_all.index_add_(0, e, w)
        self.cnt_open.index_add_(0, e, w * (ghat > lam).float())

    @torch.no_grad()
    def update_lambda(self, step=0.01, target=0.5):
        rate = self.cnt_open / self.cnt_all.clamp(min=1)
        self.rate_sum += torch.where(self.cnt_all > 0, rate, torch.zeros_like(rate))
        self.rate_n += (self.cnt_all > 0).float()
        self.lam.copy_(torch.where(self.cnt_all > 0, (self.lam + step * (rate - target)).clamp(min=0), self.lam))
        self.cnt_open.zero_()
        self.cnt_all.zero_()

    @torch.no_grad()
    def update_edges(self, A, rate=0.01):
        """EMA of the quartiles of A_g over the step's pairs (no sync)."""
        if A.numel() >= 4:
            q = torch.stack([nan_quantile(A, f) for f in (0.25, 0.5, 0.75)])
            self.edges.lerp_(q.to(self.edges.device), rate)

    # ------------------------------------------------------------------ the post-backward update
    def loss(self, rows, y, w, delta):
        """The gate loss on stored exploration rows (a dict of detached tensors), target y fixed: sum w Huber / sum w."""
        g = self.score(rows["phi"], rows["xf"], rows["ug"], rows["m"], rows["up"], rows["e"], rows["tok"], bias=False)
        h = F.huber_loss(g, y, reduction="none", delta=float(delta)) if torch.is_tensor(delta) is False else \
            torch.where((g - y).abs() <= delta, 0.5 * (g - y) ** 2, delta * ((g - y).abs() - 0.5 * delta))
        return (w * h).sum() / w.sum().clamp(min=1e-12), g


# ====================================================================== the gated read (BlobMemory with a gate)
class Scale(torch.autograd.Function):
    """out = s * (r^g + r^p) for the compact open rows. Backward passes s g to both parts and writes, into the record's
    fixed rows (copy_: idempotent under a checkpoint recompute), b = <g, r> (= dl/ds), <g, r^g>, <g, r^p>, the content
    tests cos(-g + kappa r, v_j1) for the global and the private winner, and the placebo <g, r_rand>."""

    @staticmethod
    def forward(ctx, s, rg, rp, rec):
        ctx.save_for_backward(s, rg.to(torch.bfloat16), rp.to(torch.bfloat16))
        ctx.rec = rec
        return s[:, None] * (rg + rp)

    @staticmethod
    def backward(ctx, g):
        s, rg, rp = ctx.saved_tensors
        rec = ctx.rec
        if rec is not None:
            with torch.no_grad():
                g32, rg32, rp32 = g.float(), rg.float(), rp.float()
                bg, bp = (g32 * rg32).sum(-1), (g32 * rp32).sum(-1)
                r = rg32 + rp32
                bm = rec["bm"]
                want = -g32 + bm.kappa.float() * r
                vg = bm.glob.value.view(-1, bm.d)[rec["jg"]].float()
                vp = bm.priv.value.view(-1, bm.d)[rec["jp"]].float()
                cg = F.cosine_similarity(want, vg, dim=-1)
                cp = F.cosine_similarity(want, vp, dim=-1)
                vr = bm.glob.value.view(-1, bm.d)[rec["jr"]].float()
                vr = vr * (r.norm(dim=-1) / vr.norm(dim=-1).clamp(min=1e-30))[:, None]
                pl = (g32 * vr).sum(-1)
                rec["bk"].copy_(torch.stack([bg + bp, bg, bp, cg, cp, pl], -1))
                rec["seen"].fill_(1)
        sg = s[:, None].to(g.dtype) * g
        return None, sg, sg, None


def compact_causal(open_, pos, n_max):
    """The static read buffer: open rows kept in position order up to n_max (the overflow, latest positions first, is
    dropped: an earlier position never depends on a later one). Returns (rows (n_max,), kept mask of those rows,
    keep mask over all rows, n_overflow)."""
    n = open_.size(0)
    order = (pos * (n + 1) + torch.arange(n, device=pos.device)).argsort()            # by position, then row
    rank = open_[order].long().cumsum(0)
    keep_sorted = open_[order] & (rank <= n_max)
    keep = torch.zeros_like(open_).index_copy_(0, order, keep_sorted)
    rows = (~keep).to(torch.uint8).argsort(stable=True)[:n_max]
    if rows.numel() < n_max:                                            # (fewer rows than the buffer: pad with row 0)
        rows = torch.cat([rows, rows.new_zeros(n_max - rows.numel())])
        kr = torch.cat([keep[rows[:n]], keep.new_zeros(n_max - n)])
        return rows, kr, keep, (open_ & ~keep).sum()
    return rows, keep[rows], keep, (open_ & ~keep).sum()


def blend_parts(Ag, ug, Ap, up, zeta=None):
    """r^g = zeta max(A) pi_g u_g, r^p = zeta max(A) pi_p u_p (pi = A / (A_g + A_p)); zeros when both A are 0."""
    S = Ag + Ap
    pos = S > 0
    Ss = torch.where(pos, S, torch.ones_like(S))
    m = torch.maximum(Ag, Ap)
    wg = torch.where(pos, m * Ag / Ss, torch.zeros_like(S))[:, None]
    wp = torch.where(pos, m * Ap / Ss, torch.zeros_like(S))[:, None]
    rg, rp = wg * ug, wp * up
    if zeta is not None:
        rg, rp = rg * zeta, rp * zeta
    return rg, rp


def _feats(bm, space, flat, dd, h0):
    ok = torch.isfinite(dd[:, 0])
    d1 = torch.where(ok, dd[:, 0], torch.zeros_like(dd[:, 0]))
    d2 = dd[:, 1] if dd.size(1) > 1 else dd[:, 0]
    A = torch.where(ok, torch.exp(-d1 / (2 * h0 * h0)), torch.zeros_like(d1))
    j = flat[:, 0].clamp(min=0)                                          # (r6 C2: -1 = none; masked by ok)
    mt = space.mature.view(-1)[j].float()
    age = (bm.now - space.birth.view(-1)[j]).float()
    return A, match_features(A, d1, d2, h0, mt, age, ~ok)


def inject_gated(bm, xf, buf, x, side, e_k, p_k, t_k, call=None, pair_valid=None):
    """The gated read (plan 25d rev 2 sections 5-6): one decision per (token, expert) incl. the shared expert, a causal
    threshold, the static 55% buffers, exploration with s ~ U(0, 1), the Scale read-out. Returns (buf, x_shared)."""
    gate = bm.gate
    N, dev, n_pairs = xf.size(0), xf.device, e_k.size(0)
    E, C, D = buf.size(0), buf.size(1), buf.size(2)
    Sx = bm.shared_idx
    rec_on = bm._live() and torch.is_grad_enabled() and call is not None
    acc = torch.promote_types(xf.dtype, torch.float32)
    q = bm.address(xf.to(acc))
    sel = bm._cache.get(call) if call is not None else None
    kg, kp = bm.k_g, bm.k_p
    T = x.size(1) if x.dim() == 3 else N
    pos_tok = torch.arange(N, device=dev) % T
    if sel is None:
        with torch.no_grad():
            qd = q.detach().float()
            qfull = ((xf.detach().float() - bm.mu) / bm.sigma) if bm.rerank_full else None
            gi, gd = bm.search_global(qd, pos=pos_tok, qfull=qfull)
            qfull = qfull.float() if qfull is not None else None
            Q = bm._workspace("priv_q", (E, C, bm.kd), torch.float32, dev)
            Q.zero_()
            Q.index_put_((e_k, p_k), qd[t_k])
            pf, pd = bm._rerank_priv(qd[t_k], e_k, bm.search_private(Q, E)[e_k, p_k], bm.cand)
            sh = torch.full((N,), Sx, dtype=torch.long, device=dev)
            sf, sd = bm._rerank_priv(qd, sh, bm.search_shared(qd), bm.cand)
            Ag0, gm = _feats(bm, bm.glob, gi, gd, bm.h0_g.float())
            Ap0, pm = _feats(bm, bm.priv, pf, pd, bm.h0_p.float())
            As0, smt = _feats(bm, bm.priv, sf, sd, bm.h0_p.float())
            prev = bm._prev if (bm._prev is not None and bm._prev.size(0) == N) else None
            noprev = torch.ones(N, device=dev) if prev is None else bm._prev_none
            prev = torch.zeros(N, 6, device=dev) if prev is None else prev
            # (r6 A2: the halting confidence was only computed by training forwards (with targets), so eval saw
            # halt 0 / "none" at every pass: a different gate policy. The features are kept as constants (halt 0,
            # none 1) in train AND eval: same input width, identical policy.)
            halt = torch.zeros(N, device=dev)
            hnone = torch.ones(N, device=dev)
            phi = gate.token_features(xf.detach(), side, bm.prev_mu, gm, prev, noprev, halt, hnone)
            ug0, _, _ = bm._read(qd, bm.glob, gi[:, :kg], bm.h0_g, grad=False, qfull=qfull, stats=False)  # (no-grad
            up0, _, _ = bm._read(qd[t_k], bm.priv, pf[:, :kp], bm.h0_p, grad=False, stats=False)  # reads for the
            us0, _, _ = bm._read(qd, bm.priv, sf[:, :kp], bm.h0_p, grad=False, stats=False)       # content-fit)
            if rec_on:
                gate.observe_prior(gm, torch.cat([pm, smt]))
            e_all = torch.cat([e_k, sh])
            tok_all = torch.cat([t_k, torch.arange(N, device=dev)])
            m_all = torch.cat([pm, smt])
            u_all = torch.cat([up0, us0])
            ghat = gate.score(phi, xf.detach().float(), ug0, m_all, u_all, e_all, tok_all)
            if rec_on:   # (the open-rate bias's warm-up: the gate's own score quantile, no labels)
                raw = torch.where(torch.ones_like(ghat, dtype=torch.bool) if pair_valid is None else
                                  torch.cat([pair_valid, torch.ones(N, dtype=torch.bool, device=dev)]),
                                  ghat - gate.open_bias, torch.full_like(ghat, float("nan")))
                bm._gq.append(nan_quantile(raw, 1.0 - bm.gate_open0))
            lam = gate.lam[e_all]
            salt = bm._salt + 7919 * (int(call) if call is not None else 0)
            if rec_on and bm.explore_frac > 0:
                explore, prop = gate.explore(Ag0[tok_all], bm.explore_frac, hash_uniform(tok_all, e_all, salt))
            else:
                explore, prop = torch.zeros_like(ghat, dtype=torch.bool), torch.ones_like(ghat)
            valid = torch.ones_like(explore)
            if pair_valid is not None:
                valid = torch.cat([pair_valid, torch.ones(N, dtype=torch.bool, device=dev)])
            open_ = ((ghat > lam) | explore) & valid
            if bm.ko:
                open_ = torch.zeros_like(open_)
            nr = max(1, math.ceil(bm.buf_frac * n_pairs))
            ns = max(1, math.ceil(bm.buf_frac * N))
            o_r, k_r, keep_r, ov_r = compact_causal(open_[:n_pairs], pos_tok[t_k], nr)
            o_s, k_s, keep_s, ov_s = compact_causal(open_[n_pairs:], pos_tok, ns)
            if rec_on:   # (the read scale of an exploration pair: its own uniform, a second salt)
                u = hash_uniform(tok_all, e_all, salt + 104729)
                s_all = torch.where(explore, u, torch.ones_like(ghat))
            else:
                s_all = torch.ones_like(ghat)
            sel = dict(gi=gi, gd=gd, pf=pf, pd=pd, sf=sf, sd=sd, o_r=o_r, k_r=k_r, o_s=o_s, k_s=k_s,
                       s_r=s_all[:n_pairs][o_r], s_s=s_all[n_pairs:][o_s], ghat=ghat, open=open_)
            npair = torch.zeros(N, device=dev).index_add_(0, t_k, torch.ones(n_pairs, device=dev))
            sum_ap = torch.zeros(N, device=dev).index_add_(0, t_k, Ap0)
            max_ap = torch.zeros(N, device=dev).scatter_reduce_(0, t_k, Ap0, "amax")
            nopen = torch.zeros(N, device=dev).index_add_(0, t_k, keep_r.float())
            sel["prev"] = torch.stack([Ag0, gm[:, 2], gm[:, 1], sum_ap / npair.clamp(min=1), max_ap,
                                       nopen / npair.clamp(min=1)], -1)
            sel["prev_none"] = ((nopen + keep_s.float()) == 0).float()
            if rec_on:
                ex = explore & valid
                gate.count(e_all, ghat, ex | ~valid, lam)
                bm._stats_step.append(torch.stack([
                    open_[:n_pairs].float().sum(), _const([float(n_pairs)], dev)[0], open_[n_pairs:].float().sum(),
                    _const([float(N)], dev)[0], (ov_r + ov_s).float(), ex.float().sum()]))
                if len(bm._stats_step) > STATS_KEEP:   # (r6 G3: bounded; the older rows live on as one sum)
                    old = bm._stats_step.pop(0)
                    bm._stats_acc = old if bm._stats_acc is None else bm._stats_acc + old
                bm._edges_A.append(Ag0)
                n_ex = max(1, math.ceil(2.5 * bm.explore_frac * (n_pairs + N)) + 8)
                kept_all = torch.cat([keep_r, keep_s])
                exk = ex & kept_all
                xo = (~exk).to(torch.uint8).argsort(stable=True)[:n_ex]
                if xo.numel() < n_ex:
                    xo = torch.cat([xo, xo.new_zeros(n_ex - xo.numel())])
                    okx = torch.cat([exk[xo[:exk.numel()]], exk.new_zeros(n_ex - exk.numel())])
                else:
                    okx = exk[xo]
                tk = tok_all[xo]
                sel["gate_rows"] = dict(phi=phi[tk], xf=xf.detach().float()[tk], ug=ug0[tk], m=m_all[xo], up=u_all[xo],
                                        e=e_all[xo], tok=torch.arange(n_ex, device=dev), prop=prop[xo], ok=okx,
                                        ghat=ghat[xo], row=xo)
            bm._count(dict(gi=gi, pf=pf, sf=sf, mr=keep_r, ms=keep_s), t_k)
        if call is not None and torch.is_grad_enabled():
            bm._cache[call] = sel
    bm._prev, bm._prev_none = sel["prev"], sel["prev_none"]
    o_r, o_s = sel["o_r"], sel["o_s"]
    tr = t_k[o_r]
    with torch.autocast(dev.type, enabled=False):
        qf = ((xf.to(acc) - bm.mu.to(acc)) / bm.sigma.to(acc)) if bm.xfull is not None else None
        ug, Ag, dg = bm._read(q, bm.glob, sel["gi"][:, :kg], bm.h0_g, qfull=qf)
        up, Ap, dp = bm._read(q[tr], bm.priv, sel["pf"][o_r, :kp], bm.h0_p)
        us, As, ds = bm._read(q[o_s], bm.priv, sel["sf"][o_s, :kp], bm.h0_p)
        zeta = bm.inj_scale.to(acc) if bm.inj_scale is not None else None
        rg_r, rp_r = blend_parts(Ag[tr], ug[tr], Ap, up, zeta)
        rg_s, rp_s = blend_parts(Ag[o_s], ug[o_s], As, us, zeta)
        s_r = sel["s_r"].to(acc) * sel["k_r"].to(acc)
        s_s = sel["s_s"].to(acc) * sel["k_s"].to(acc)
        rec_r = rec_s = None
        if rec_on:
            rec_r = bm._make_rec(call, "r", sel, o_r, tr, q, Ag[tr], Ap, sel["gi"][tr, 0], sel["pf"][o_r, 0],
                                 sel["gd"][tr], sel["pd"][o_r], s_r, rg_r, rp_r, n_pairs)
            rec_s = bm._make_rec(call, "s", sel, o_s, o_s, q, Ag[o_s], As, sel["gi"][o_s, 0], sel["sf"][o_s, 0],
                                 sel["gd"][o_s], sel["sd"][o_s], s_s, rg_s, rp_s, n_pairs)
        out_r = Scale.apply(s_r, rg_r, rp_r, rec_r)
        out_s = Scale.apply(s_s, rg_s, rp_s, rec_s)
        buf = buf.index_put((e_k[o_r], p_k[o_r]), out_r.to(buf.dtype), accumulate=True)
        xs = x.reshape(N, D).index_put((o_s,), out_s.to(x.dtype), accumulate=True).view(x.shape)
        bm.last = dict(A_g=Ag.detach(), d1_g=dg.detach(), A_p=Ap.detach(), d1_p=dp.detach(), A_s=As.detach(),
                       read_frac=sel["k_r"].float().sum() / max(1, n_pairs))
    return buf, xs


# ====================================================================== after backward / after the optimizer step
def _nanmedian(x):
    return nan_quantile(x, 0.5)


def _yring_add(bm, rows):
    """(r6 G3) the static (g_hat without bias, y in robust units, ok) ring: YRING rows, a host pointer, no growth."""
    if bm._yring is None or bm._yring.device != rows.device:
        bm._yring, bm._yring_n = torch.zeros(YRING, 3, device=rows.device), 0
    n = min(rows.size(0), YRING)
    idx = (torch.arange(n, device=rows.device) + bm._yring_n) % YRING
    bm._yring.index_copy_(0, idx, rows[:n])
    bm._yring_n += n


def after_backward(bm):
    """After a micro-batch's backward (idempotent read-outs are in the records): the gate targets and the gate's
    post-backward update (exploration rows only), the curvature statistics, confirmation candidates, strength credits,
    the kernel-width statistics, the placebo. No host sync."""
    recs, bm._recs = bm._recs, {}
    if not recs:
        return None
    gate = bm.gate
    kap = bm.kappa.float()
    out = []
    for call, rr in recs.items():
        parts = [rr.get("r"), rr.get("s")]
        if any(p is None for p in parts):
            continue
        sel = rr["sel"]
        n_tot = rr["n_total"]
        dev = parts[0]["bk"].device
        bfull = torch.zeros(n_tot + 1, device=dev)
        sfull = torch.ones(n_tot + 1, device=dev)
        r2full = torch.zeros(n_tot + 1, device=dev)
        seenfull = torch.zeros(n_tot + 1, device=dev)
        for rec in parts:
            ok = rec["k"] & (rec["seen"] > 0)
            b = rec["bk"][:, 0]
            bg, bp, cg, cp, pl = rec["bk"][:, 1], rec["bk"][:, 2], rec["bk"][:, 3], rec["bk"][:, 4], rec["bk"][:, 5]
            ex = rec["ex"] & ok
            dest = torch.where(ok, rec["orig"], torch.full_like(rec["orig"], n_tot))
            bfull.index_copy_(0, dest, b)
            sfull.index_copy_(0, dest, rec["s"])
            r2full.index_copy_(0, dest, rec["r2"])
            seenfull.index_copy_(0, dest, ok.float())
            # curvature: least squares of b = dl/ds on x = s |r|^2 over the exploration rows
            x, y = rec["s"] * rec["r2"], b
            w = ex.float()
            bm._kstats += torch.stack([w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum()])
            # the curvature-corrected credit (confirmation / strength) per part
            Gg = -bg + 0.5 * kap * rec["rg2"]
            Gp = -bp + 0.5 * kap * rec["rp2"]
            G = Gg + Gp
            base = ok & ~ex
            thr = nan_quantile(torch.where(base, G, torch.full_like(G, float("nan"))), 0.75)
            top = base & (G > 0) & (G >= torch.nan_to_num(thr, nan=float("inf")))
            okg_ = ok & rec["vg"]                                          # (r6 C2: a real global winner)
            cg_ok = top & okg_ & (Gg > 0) & (rec["w1g"] >= bm.conf_w1) & (rec["ratio_g"] >= bm.conf_margin) & \
                (rec["Ag"] >= bm.conf_A) & (cg > bm.conf_cos)
            cp_ok = top & (Gp > 0) & (rec["w1p"] >= bm.conf_w1) & (rec["ratio_p"] >= bm.conf_margin) & \
                (rec["Ap"] >= bm.conf_A) & (cp > bm.conf_cos)
            bm._conf["g"].append((rec["jg"], rec["q"], Gg, cg_ok))
            bm._conf["p"].append((rec["jp"], rec["q"], Gp, cp_ok))
            bm._credit["g"].append((rec["jg"], Gg.clamp(min=0) * okg_))      # (applied at the step end: the
            bm._credit["p"].append((rec["jp"], Gp.clamp(min=0) * ok))      # store is not changed mid-step)
            bm._gaps["g"].append(torch.where(ex, rec["gap_g"], torch.full_like(rec["gap_g"], float("nan"))))
            bm._gaps["p"].append(torch.where(ex, rec["gap_p"], torch.full_like(rec["gap_p"], float("nan"))))
            plv = rec["plc"] & ex
            bm._mstats += torch.stack([(plv.float() * (-b + pl)).sum(), plv.float().sum(),
                                       ok.float().sum(), (ok & (rec["r2"] > 0)).float().sum(),
                                       (rec["is_shared"] * Gg.clamp(min=0) * ok).sum() + (rec["is_shared"] * Gp.clamp(min=0) * ok).sum(),
                                       (G.clamp(min=0) * ok).sum()])
        # the gate: y on its exploration rows
        gr = sel.get("gate_rows")
        if gr is None or gate.opt is None:
            continue
        xo = gr["row"]
        okg = gr["ok"] & (seenfull[xo] > 0)
        bt = -bfull[xo]                                                        # b~ = -dl/ds: unbiased gain
        bcv = bt + (sfull[xo] - 0.5) * kap * r2full[xo]
        y = bm.y_scale * bcv
        yv = torch.where(okg, y, torch.full_like(y, float("nan")))
        med = _nanmedian(yv)
        mad = _nanmedian((yv - med).abs())
        s_now = 1.4826 * mad
        sr = torch.nan_to_num(s_now, nan=1e-4).clamp(min=1e-4)
        med = torch.nan_to_num(med, nan=0.0)
        y = torch.minimum(torch.maximum(y, med - 5 * sr), med + 5 * sr)
        # (r6 G2: the target in robust units: y / EMA(1.4826 MAD(y)), Huber delta 1; a positive rescaling keeps the
        # sign, so "read iff g_hat > lambda" is unchanged. Device-side, no sync.)
        good = torch.isfinite(s_now) & (s_now > 0)
        ysc = torch.where(gate.y_sc > 0, gate.y_sc.lerp(s_now, 0.05), s_now)
        gate.y_sc.copy_(torch.where(good, ysc, gate.y_sc))
        yn = y / torch.where(gate.y_sc > 0, gate.y_sc, sr)
        delta = torch.ones_like(sr)
        w = okg.float() / gr["prop"].clamp(min=1e-6)
        with torch.enable_grad():
            loss, g = gate.loss(gr, yn.detach(), w, delta)
            (loss / max(1, bm.accum)).backward()
        bm._gate_last = dict(loss=loss.detach(), n=okg.sum())
        _yring_add(bm, torch.stack([g.detach().float(), yn.float(), okg.float()], -1))
        bm._mstats2 += torch.stack([(okg.float() * y).sum(), okg.float().sum()])
        bm._hy += torch.stack([(okg.float() * y).sum(), (okg.float() * y * y).sum(), okg.float().sum(), loss.detach()])
        out.append(loss.detach())
    return out


@torch.no_grad()
def step_end(bm, step):
    """Once per optimizer step: the gate's AdamW step + thresholds + strata edges, kappa, sigma^2, renewals, the
    strength decay. No host sync."""
    gate = bm.gate
    if gate is not None and gate.opt is not None:
        torch.nn.utils.clip_grad_norm_(list(gate.parameters()), bm.gate_clip)
        gate.opt.step()
        gate.opt.zero_grad(set_to_none=True)
        gate.update_lambda()
        if bm._edges_A:
            gate.update_edges(torch.cat(bm._edges_A))
    bm._edges_A = []
    n, sx, sy, sxx, sxy = bm._kstats
    den = n * sxx - sx * sx
    slope = torch.where((n >= 8) & (den > 0), (n * sxy - sx * sy) / den.clamp(min=1e-30), bm.kappa)
    bm.kappa.copy_(torch.where(n >= 8, bm.kappa.lerp(slope.clamp(min=0), 0.01), bm.kappa))
    bm._kstats.zero_()
    if gate is not None and bm._gq:   # the open-rate bias during the warm-up (then fixed; the gate learns)
        q = torch.stack(bm._gq).mean()
        warm = gate.warm_left > 0
        first = gate.warm_left == bm.gate_warm
        new = torch.where(first, -q, gate.open_bias.lerp(-q, 0.2))
        gate.open_bias.copy_(torch.where(warm & torch.isfinite(q), new, gate.open_bias))
        gate.warm_left.copy_((gate.warm_left - 1).clamp(min=0))
    bm._gq = []
    for nm, mbuf in (("g", getattr(bm, "gapm_g", None)), ("p", getattr(bm, "gapm_p", None))):
        if mbuf is not None and bm._mgaps[nm]:   # the Gaussian kernel's running median of d2 - d1 (training reads)
            mq = nan_quantile(torch.cat(bm._mgaps[nm]), 0.5)
            new = torch.where(mbuf > 0, mbuf.lerp(mq, 0.05), mq)
            mbuf.copy_(torch.where(torch.isfinite(mq) & (mq > 0), new, mbuf))
        bm._mgaps[nm] = []
    for nm, h0, s2 in (("g", bm.h0_g, bm.sigma2_g), ("p", bm.h0_p, bm.sigma2_p)):
        if bm._gaps[nm]:
            m = nan_quantile(torch.cat(bm._gaps[nm]), 0.5)
            new = (s2.lerp(m / (2 * math.log(2)), 0.01)).clamp(min=0.05 * float(1.0) * 0 + 0.05 * h0 * h0)
            s2.copy_(torch.where(torch.isfinite(m), new, s2))
        bm._gaps[nm] = []
    from blob_write import renew
    moved = {}
    for nm in ("g", "p"):
        if bm._conf[nm]:
            j, q, g, ok = (torch.cat(t) for t in zip(*bm._conf[nm]))
            moved[nm] = renew(bm, nm, j, q, g, ok, step)
        bm._conf[nm] = []
    f = 2.0 ** (-1.0 / bm.strength_half)
    bm.glob.strength.mul_(f)
    bm.priv.strength.mul_(f)
    for nm in ("g", "p"):   # the step's positive benefit (EMA with the half-life above)
        sp = bm.space(nm)
        for j, c in bm._credit[nm]:
            sp.strength.view(-1).index_add_(0, j, c)
        bm._credit[nm] = []
    return moved
