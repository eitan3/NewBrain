"""Memory system of the looped brain (plan_v2/12_memory_full_plan.md, revision 2). Every part is behind flags.

    ConceptHier    hierarchical concept heads on the workspace (section 1): targets, Gaussian NLL + masked variance
                   hinge, uncertainty u, feedback W_fb, rank-r write-back into the next pass's core input
    Memory         translators + product keys (2), shared pool read by MemRead (3), MemOpt (3.1), private expert
                   memory (4), sub-key balancing and the freeze (2.2), margin loss (2.3), copy-on-split
    MemoryProbes   drift probes, stability guard, repair (5, 6)
    LifeCycle      the memory life cycle (plan_v2/21): infancy (decaying, rewritable rows) -> ramp -> mature (frozen)

Life cycle (plan_v2/21, every part behind its flag): lazy row decay through a global Lambda(t) and a per-row Lambda_r
(applied on read through the MemRead weights and on write in MemOpt), a discounted accumulator through Psi(t) and
Psi_r, an uncapped dead-key restart with a band reset, fixed random addressing (addr_learn False), a coarse
product-key pool next to the fine one (one table, rows at an offset) and neighbour writes (ranks km+1 .. nbr written
in MemRead's backward only).

Tables (V_sh, V_e and the private keys P) are BUFFERS, never nn.Parameters: BrainOpt, clip_grad_norm_,
carry_brain_state, build_opt and model.zero_grad never see them. Their gradients go through MemRead / GatherRows
into ONE preallocated dense buffer per table, and MemOpt updates them once per optimizer step (after BrainOpt).
"Addressing" (everything that turns a thought into row indices) runs in fp32 with autocast and TF32 off.
"""
import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

NRES = 64       # recent query halves kept per sub-key codebook (re-seeding dead sub-keys)
NRES_RESTART = 1024   # --mem-restart: the larger reservoir (distinct slots per restart event)
S_FLOOR = 0.1   # --mem-life: the discounted accumulator never falls below S_FLOOR x S0
P_TOPK = 4      # private memory: rows read per (token, chosen expert)
# speed (plan_v2/14, same math)
TOPK_SYNC_FREE = False    # stable_topk boundary ties: False = those rows are redone by the full stable sort (one host
                          # sync per call), True = fixed by a second top-k over the rows (no host sync, more GPU work)
READ_DW_BMM = True        # MemRead backward: dw by one gather + bmm (False = the per-column loop: same values up to
                          # the summation order; the equivalence tests use it)
READ_BWD_ELEMS = 1 << 26  # MemRead backward: temporaries of at most this many elements per chunk
SELECT_BUDGET = 1 << 26   # select(): score-buffer elements per group of levels (bounded temporaries)
_INT = {8: torch.int64, 4: torch.int32, 2: torch.int16}   # bit views of float tables (row "touched" = any bit set)


# ---------------------------------------------------------------- helpers
@contextlib.contextmanager
def fp32_math(device_type):
    """No autocast and no TF32 (matmul and cuDNN) inside; the global flags are restored afterwards."""
    mm, cd = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast(device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = mm, cd


def _sort_rows(x, k, max_elems):
    """The full stable descending sort of the index-ordered rows of x (R, S), first k (in chunks of rows, so the
    temporaries stay bounded)."""
    step = max(1, max_elems // max(1, x.size(-1)))
    vs, ix = [], []
    for c in range(0, x.size(0), step):
        v, i = torch.sort(x[c:c + step], dim=-1, descending=True, stable=True)
        vs.append(v[:, :k])
        ix.append(i[:, :k])
    return torch.cat(vs), torch.cat(ix)


def _resort(v, i):
    """k (score, index) pairs per row in (score desc, index asc) order: by index first (distinct), then stably by
    score."""
    o = i.argsort(-1)
    v, i = v.gather(-1, o), i.gather(-1, o)
    v, o = torch.sort(v, dim=-1, descending=True, stable=True)
    return v, i.gather(-1, o)


def stable_topk(s, k, max_elems=1 << 24):
    """Top-k along the last dim, ties -> the LOWEST index (= a stable descending sort of the index-ordered scores).
    torch.topk(k + 1) instead of the full sort, then the k winners in (score desc, index asc) order. The k-set is
    ambiguous only when the k-th and the (k+1)-th scores tie: those rows are redone with the full stable sort (one
    host sync per call; TOPK_SYNC_FREE: the lowest-index entries equal to the k-th score are taken from a second
    top-k over the row, no sync). Identical indices and values; finite scores (-0.0 == +0.0, as in torch.sort)."""
    flat = s.reshape(-1, s.size(-1))
    S, shape = flat.size(1), tuple(s.shape[:-1]) + (k,)
    if k >= S:
        v, i = _sort_rows(flat, k, max_elems)
        return v.view(shape), i.view(shape)
    v, i = torch.topk(flat, k + 1, dim=-1, sorted=True)
    if TOPK_SYNC_FREE:
        thr = v[:, k - 1:k]
        gt = (v[:, :k] > thr).sum(1, keepdim=True)                     # the winners strictly above the k-th score
        rev = torch.arange(S - 1, -1, -1, device=flat.device, dtype=torch.int32)
        eq = torch.topk(torch.where(flat == thr, rev, rev.new_tensor(-1)), k, dim=-1).indices   # lowest indices first
        p = torch.arange(k, device=flat.device)
        i = torch.where(p < gt, i[:, :k], eq.gather(1, (p - gt).clamp(min=0)))
        v, i = _resort(flat.gather(1, i), i)
    else:
        eq = v[:, 1:] == v[:, :-1]                                     # equal neighbours; column k - 1: the boundary
        v, i = v[:, :k], i[:, :k]
        if bool(eq.any()):  # (rare on real scores) ties among the winners: re-sort; at the boundary: the full sort
            v, i = _resort(v, i)
            r = eq[:, k - 1].nonzero().squeeze(1)
            if r.numel():
                fv, fi = _sort_rows(flat[r], k, max_elems)
                v, i = v.index_copy(0, r, fv), i.index_copy(0, r, fi)
    return v.reshape(shape), i.reshape(shape)


def pk_topk(v1, i1, v2, i2, S, k):
    """Product-key top-k pairs from the per-half top-m lists (score v, index i; (..., m)). Ties -> the lowest flat
    index r = i * S + j: the candidates are sorted by r first (unique), then stably by score (descending).
    Returns (i, j, score) of the top k pairs, (..., k)."""
    sig = (v1[..., :, None] + v2[..., None, :]).flatten(-2)
    flat = (i1[..., :, None] * S + i2[..., None, :]).flatten(-2)
    o = flat.argsort(-1)
    sig, flat = sig.gather(-1, o), flat.gather(-1, o)
    v, o2 = torch.sort(sig, dim=-1, descending=True, stable=True)
    fl = flat.gather(-1, o2[..., :k])
    return fl // S, fl % S, v[..., :k]


def _anchor(device, need):
    """A scalar leaf that requires grad, so autograd calls the custom backward (which writes the table's grad
    buffer) even when nothing else of the read needs a gradient (e.g. after the freeze, with sg addressing)."""
    return torch.ones((), device=device, requires_grad=bool(need))


class MemRead(torch.autograd.Function):
    """r[b] = sum_j w[b, j] V[rows[b, j]] (embedding_bag, mode sum). Backward: index_add_ into the ONE preallocated
    grad buffer G (no fresh table-sized allocation), and the gradient of the weights w. V gets no .grad.
    Optional (plan_v2/21): gw = the weights of the G write instead of w (--mem-life: w = omega x the pending decay
    factors f, gw = omega, so G holds dL/dV in the rows' current, decayed units); xrows / xw = extra rows written
    with weights xw in backward only (neighbour writes: no forward cost, no gradient to xw)."""

    @staticmethod
    def forward(ctx, anchor, w, V, G, rows, gw=None, xrows=None, xw=None):
        ctx.save_for_backward(rows, w)
        ctx.V, ctx.G, ctx.gw, ctx.x = V, G, gw, (xrows, xw)
        return F.embedding_bag(rows, V, per_sample_weights=w.to(V.dtype), mode="sum")

    @staticmethod
    def backward(ctx, gout):
        rows, w = ctx.saved_tensors
        V, G = ctx.V, ctx.G
        g = gout.to(V.dtype)
        B, k = rows.shape
        if G is not None:  # ONE flattened index_add_ (column-major: the order of adds of one column at a time)
            wg = w if ctx.gw is None else ctx.gw
            kc = max(1, READ_BWD_ELEMS // max(1, g.numel()))                    # (columns per chunk: bounded)
            for c in range(0, k, kc):
                G.index_add_(0, rows[:, c:c + kc].t().reshape(-1),
                             (wg[:, c:c + kc].t().to(G.dtype)[..., None] * g[None]).reshape(-1, g.size(1)))
            xrows, xw = ctx.x
            if xrows is not None:  # neighbour writes: G += xw g on the extra rows (after the read's own columns)
                for c in range(0, xrows.size(1), kc):
                    G.index_add_(0, xrows[:, c:c + kc].t().reshape(-1),
                                 (xw[:, c:c + kc].t().to(G.dtype)[..., None] * g[None]).reshape(-1, g.size(1)))
        dw = None
        if ctx.needs_input_grad[1]:
            if READ_DW_BMM:  # dw[b, j] = <g[b], V[rows[b, j]]>: one gather + bmm (in chunks of bags)
                nb = max(1, READ_BWD_ELEMS // max(1, k * V.size(1)))
                dw = torch.cat([torch.bmm(F.embedding(rows[c:c + nb], V), g[c:c + nb, :, None])[..., 0]
                                for c in range(0, B, nb)]).to(w.dtype)
            else:
                dw = torch.stack([(g * F.embedding(rows[:, j], V)).sum(-1) for j in range(k)], 1).to(w.dtype)
        return None, dw, None, None, None, None, None, None


class GatherRows(torch.autograd.Function):
    """T[rows] whose backward index_add_'s into a preallocated grad buffer G (the private keys P)."""

    @staticmethod
    def forward(ctx, anchor, T, G, rows):
        ctx.save_for_backward(rows)
        ctx.G = G
        return T[rows]

    @staticmethod
    def backward(ctx, g):
        rows, = ctx.saved_tensors
        if ctx.G is not None:
            ctx.G.index_add_(0, rows.reshape(-1), g.reshape(-1, g.size(-1)).to(ctx.G.dtype))
        return None, None, None, None


class _QSg(torch.autograd.Function):
    """The online query halves (q1, q2) once more, as outputs whose gradient reaches W_q ONLY, i.e. as if they were
    computed from sg(a_hat) (the margin loss): no second W_q matmul in forward; backward runs the two normalisations'
    backward by hand (y = x / |x|: dx = (g - y <y, g>) / |x|; the norms here are O(1), far above F.normalize's
    eps) and one einsum for dW_q. Inputs: wq, then a_hat, qr = W_q a_hat, q1, q2, all detached."""

    @staticmethod
    def forward(ctx, wq, a_hat, qr, q1, q2):
        ctx.save_for_backward(a_hat, qr, q1, q2)
        return q1.view_as(q1), q2.view_as(q2)

    @staticmethod
    def backward(ctx, g1, g2):
        a_hat, qr, q1, q2 = ctx.saved_tensors
        nb = lambda g, y, x: (g - y * (y * g).sum(-1, keepdim=True)) / x.norm(dim=-1, keepdim=True)
        g1 = torch.zeros_like(q1) if g1 is None else g1
        g2 = torch.zeros_like(q2) if g2 is None else g2
        q = F.normalize(qr, dim=-1)
        dq = torch.cat([nb(g1, q1, q[..., :64]), nb(g2, q2, q[..., 64:])], -1)
        dwq = torch.einsum("lhno,lnc->lhoc", nb(dq, q, qr), a_hat)
        return dwq, None, None, None, None


def plan_balance(U, busy=4.0, dead=0.05, max_keys=0, gen=None, keep=None, dead_max=0):
    """Sub-key balancing plan for codebooks U (C, S) of usage shares (CPU), all codebooks at once. Busy
    (U > busy x mean) keys split into the least-used keys (busiest first, ties -> the lower index); dead
    (U < dead x mean) keys not used by a split are re-seeded. (The busy keys lead the descending order and the
    non-busy keys the ascending one, so the busy key of rank t pairs with the t-th least-used key.)
    Calmer balancing (--mem-balance-max / --mem-balance-keep): max_keys > 0 = at most that many keys change per
    codebook (a split changes 2; splits first, busiest first, then dead keys, drawn at random with gen when more are
    dead than the budget left); keep (C, S) bool = keys that are never re-seeded (used since the last balancing).
    Returns LongTensors splits (n, 3) = (c, i, j) and deads (n, 2) = (c, i), codebook by codebook."""
    C, S = U.shape
    mean = U.mean(1, keepdim=True)
    isb, isd = U > busy * mean, U < dead * mean
    if keep is not None:
        isd &= ~keep
    desc = U.argsort(dim=1, descending=True, stable=True)
    asc = U.argsort(dim=1, stable=True)
    nb = isb.sum(1)
    npair = torch.minimum(nb, S - nb)
    if max_keys:
        npair = npair.clamp(max=max_keys // 2)
    cc, tt = (torch.arange(S)[None] < npair[:, None]).nonzero(as_tuple=True)
    splits = torch.stack([cc, desc[cc, tt], asc[cc, tt]], 1)
    used = isb.clone()
    used[cc, asc[cc, tt]] = True
    isd &= ~used
    if max_keys or dead_max:  # dead_max > 0: the dead re-seeds get their own cap (--mem-balance-dead-max)
        room = torch.full_like(npair, dead_max) if dead_max else (max_keys - 2 * npair).clamp(min=0)
        r = torch.where(isd, torch.rand(C, S, generator=gen), torch.full((C, S), 2.0))
        isd &= r.argsort(dim=1, stable=True).argsort(1) < room[:, None]    # a random subset of the dead keys
    return splits, isd.nonzero()


# ---------------------------------------------------------------- concepts (section 1)
def concept_targets(z, idx, horizons, eot):
    """z: (B, T, d) = sg(normf(x_K)); idx: the INPUT tokens. For each horizon n: target z*(t) = mean of z over
    t+1 .. e(t), e(t) = min(t + n, T - 1, b(t) - 1), b(t) = the next EOT strictly after t (shifted suffix-min), and
    the valid mask c(t) = e(t) - t >= n / 4 and idx[t] != EOT. Returns [(target, valid)] per horizon."""
    B, T, D = z.shape
    P = torch.cat([z.new_zeros(B, 1, D), z.cumsum(1)], 1)                       # P[:, j] = sum z[:, :j]
    t = torch.arange(T, device=z.device)
    e = torch.where(idx == eot, t[None], torch.full_like(idx, T))
    rc = e.flip(1).cummin(1).values.flip(1)                                    # rc[t] = min_{j >= t} e[j]
    b = torch.cat([rc[:, 1:], torch.full_like(rc[:, :1], T)], 1)               # b(t) = rc[t + 1]; b(T-1) = T
    out = []
    for n in horizons:
        end = torch.minimum(torch.clamp(t + n, max=T - 1)[None], b - 1)       # (B, T)
        c = end - t[None]
        hi = P.gather(1, (end + 1)[..., None].expand(B, T, D))
        lo = P[:, 1:]                                                          # P[:, t + 1]
        tgt = (hi - lo) / c.clamp(min=1)[..., None].to(z.dtype)
        out.append((tgt, (c >= n / 4) & (idx != eot)))
    return out


def masked_std(x, v, nv):
    """Per-dimension std over the valid (b, t): sqrt(mean_v(x^2) - mean_v(x)^2 + 1e-6) (no nonzero(), no sync)."""
    m = (x * v).sum((0, 1)) / nv
    m2 = (x * x * v).sum((0, 1)) / nv
    return (m2 - m * m + 1e-6).clamp(min=1e-12).sqrt()


def concept_loss(mu, nu, tgt, valid):
    """Gaussian NLL (masked, / (d x n_valid)) + variance hinge with masked moments (anti-collapse)."""
    v = valid.to(mu.dtype)[..., None]
    nv = v.sum().clamp(min=1)
    nll = (0.5 * (nu + (tgt - mu) ** 2 * torch.exp(-nu)) * v).sum() / (mu.size(-1) * nv)
    hinge = F.relu(masked_std(tgt, v, nv).detach() - masked_std(mu, v, nv)).mean()
    return nll + hinge


class ConceptHier(nn.Module):
    """Shared trunk h = GELU(W_t p) on the pooled workspace p (B, T, slots x 64); level l predicts [mu_l; nu_l]
    from [h; mu_{l-1}] (mu_{l-1} WITH gradient), nu soft-clamped to (-6, 4)."""

    def __init__(self, d, d_in, L, fb=False, inject=False, rank=64, trunk=256):
        super().__init__()
        self.L, self.d = L, d
        self.trunk = nn.Linear(d_in, trunk)
        self.levels = nn.ModuleList(nn.Linear(trunk + (d if l else 0), 2 * d) for l in range(L))
        self.fb = nn.Linear(2 * L, 64) if fb else None             # W_fb (zero-init): u -> every workspace slot
        self.inj_a = self.inj_b = self.inj = None
        if inject:                                                 # usefulness: sum_l B_l A_l mu_l (B_l = 0)
            if rank:
                self.inj_a = nn.Parameter(torch.randn(L, rank, d) * 0.02)
                self.inj_b = nn.Parameter(torch.zeros(L, d, rank))
            else:
                self.inj = nn.Parameter(torch.zeros(L, d, d))

    @property
    def has_inject(self):
        return self.inj_b is not None or self.inj is not None

    def forward(self, p):
        h = F.gelu(self.trunk(p))
        acc = torch.promote_types(self.trunk.weight.dtype, torch.float32)
        mus, nus, prev = [], [], None
        for lin in self.levels:
            out = lin(h if prev is None else torch.cat([h, prev.to(h.dtype)], -1)).to(acc)
            mu, raw = out.chunk(2, -1)
            mus.append(mu)
            nus.append(-1.0 + 5.0 * torch.tanh(raw / 5.0))
            prev = mu
        return mus, nus

    @staticmethod
    @torch.no_grad()
    def uncertainty(mus, nus, prev):
        """u = concat_l [(mean nu_l + 6) / 10, log1p(|mu_l - mu_l,prev| / |mu_l|)] (2L, detached)."""
        u = []
        for l, (mu, nu) in enumerate(zip(mus, nus)):
            var_u = (nu.mean(-1) + 6.0) / 10.0
            move = ((mu - prev[l]).norm(dim=-1) / mu.norm(dim=-1).clamp(min=1e-6) if prev is not None
                    else torch.zeros_like(var_u))
            u += [var_u, torch.log1p(move)]
        return torch.stack(u, -1)

    def write_back(self, mus):
        if self.inj_b is not None:
            mu = torch.stack(mus).to(self.inj_a.dtype)
            return torch.einsum("lbtr,ldr->btd", torch.einsum("lbtd,lrd->lbtr", mu, self.inj_a), self.inj_b)
        mu = torch.stack(mus).to(self.inj.dtype)
        return torch.einsum("lbtd,lod->bto", mu, self.inj)


# ---------------------------------------------------------------- the memory (sections 2-4)
class Memory(nn.Module):
    """Translators (L levels x H heads) with product keys over a shared pool of L x S^2 rows, and (R > 0) a private
    memory of R rows per expert, preallocated for n_exp experts. See the module doc and plan sections 2-4."""

    def __init__(self, d, ds, L, H=3, S=512, m=8, km=6, tau=0.03, dv=256, R=0, n_exp=0, n_live=0, read0=True,
                 addr_sg=True, margin=(0.01, 0.01), ema=0.999, seed=0, life=False, addr_learn=True, restart=0,
                 coarse=0, coarse_heads=1, nbr=0, nbr_w=0.15):
        super().__init__()
        assert km <= m and km + 1 <= m * m, "product-key top-k is exact only for k <= m"
        assert not S or S >= m + 1, "each half needs at least m + 1 sub-keys (the margin uses the (m+1)-th)"
        assert not R or R >= P_TOPK
        assert not coarse or (S and coarse >= m + 1), "the coarse pool needs the shared pool and >= m + 1 sub-keys"
        assert not nbr or (S and km + 1 <= nbr <= m * m), "neighbour writes: km + 1 <= --mem-nbr <= m^2"
        self.d, self.ds, self.L, self.H, self.S, self.m, self.km = d, ds, L, H, S, m, km
        self.tau, self.dv, self.R, self.Emax = tau, dv, R, n_exp
        self.read0, self.addr_sg, self.ema = bool(read0), bool(addr_sg), ema
        self.lam_m, self.delta = float(margin[0]), float(margin[1])
        # plan_v2/21 (all off by default): life cycle, fixed addressing, restart, coarse pool, neighbour writes
        self.life, self.addr_learn, self.restart = bool(life), bool(addr_learn), int(restart)
        self.Sc, self.Hc, self.nbr, self.nbr_w = int(coarse), int(coarse_heads), int(nbr or 0), float(nbr_w)
        self.nres = NRES_RESTART if self.restart else NRES
        if not self.addr_learn:  # fixed random addressing: no margin loss (the margin only trains W_q / the keys)
            self.lam_m = 0.0
        self.ko_coarse = False     # knock-out probe: no coarse read
        self.restart_zero = False  # --mem-restart-zero: a re-seeded key's band also gets V <- 0 (the trainer sets it)
        self.decay0, self.beta0 = 1e-3, 0.99   # --mem-decay / --mem-acc-beta (the trainer sets them; LifeCycle ramps)
        # host state of the life cycle and the restart (in the extra state when either is on): Lam / Psi = the global
        # decay / discount sums (fp64), the current per-step rates, the restart clock and the switch
        self.life_state = dict(Lam=0.0, Psi=0.0, decay=None, beta=None, restart_every=None, last_restart=0,
                               phase="infancy", t_switch=None, reason=None, stab_run=0, settled=False) \
            if (self.life or self.restart) else {}
        self.ain = d + ds
        # MemOpt and schedule (the trainer sets these)
        self.lr, self.s0_cfg, self.clip, self.balance_every, self.usage_rate = 0.01, "auto", 0.05, 250, 0.01
        self.write_on = True       # trainer: step >= --mem-write-start
        self.ckpt = True           # activation-checkpoint the private read (the LoopBrain block has its own flag)
        self.margin_fused = True   # the margin from the online pass's queries and sub-key gather (False = its own
                                   # pass, as before: the same values, gradients summed in another order; tests)
        self.balance_max = 0.0     # --mem-balance-max: at most this fraction of the keys per codebook per event (0 = all)
        self.balance_dead_max = 0.0  # --mem-balance-dead-max: a separate cap for the dead re-seeds (0 = share balance_max)
        self.balance_keep = False  # --mem-balance-keep: never re-seed a key used since the last balancing event
        self._touch = None         # MemOpt: the rows written this step (write_counts / after_step)
        self.ko_shared, self.ko_private, self.ko_levels = False, False, ()
        self.record = None         # MemoryProbes: dict the forward fills (eval only)
        self.probe_state = {}      # MemoryProbes' freeze-time / last-eval records (saved as extra state)
        self.gen = torch.Generator().manual_seed((seed + 4242) % (2 ** 63))   # dedicated RNG: balancing
        self.gen_margin = torch.Generator().manual_seed((seed + 2718) % (2 ** 63))   # --mem-margin-passes: the pass
        self.stats = {}            # last balancing / freeze numbers (logging)
        self.margin_stats = None
        self.opt_reset = None      # trainer: fn(param name, index) -> fresh optimizer moments for edited sub-key rows
        self._collect = False
        self.keys = nn.Module()    # W_q and the sub-keys: AdamW without weight decay (BrainOpt "mem.keys.")
        self.keys.wq = nn.Parameter(torch.randn(L, H, 128, self.ain) * 0.02)
        self.register_buffer("wq_ema", self.keys.wq.detach().clone())
        if S:
            self.keys.k1 = nn.Parameter(F.normalize(torch.randn(L, H, S, 64), dim=-1))
            self.keys.k2 = nn.Parameter(F.normalize(torch.randn(L, H, S, 64), dim=-1))
            self.register_buffer("k1_ema", self.keys.k1.detach().clone())
            self.register_buffer("k2_ema", self.keys.k2.detach().clone())
        self.register_buffer("norm_m", torch.zeros(L, self.ain))
        self.register_buffer("norm_v", torch.ones(L, self.ain))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.long))
        self.register_buffer("n_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("s0", torch.full((2,), -1.0))          # S0 of the shared / private table (-1 = unset)
        self.register_buffer("ns_sum", torch.zeros(L, self.ain), persistent=False)
        self.register_buffer("ns_sq", torch.zeros(L, self.ain), persistent=False)
        self.register_buffer("ns_cnt", torch.zeros(()), persistent=False)
        self.gates = nn.Module()   # gate weights: AdamW (BrainOpt "mem.gates.")
        if S:
            n = L * S * S
            self.register_buffer("V", torch.randn(n, dv) * 0.02)
            self.register_buffer("S_acc", torch.zeros(n))
            self.register_buffer("wcount", torch.zeros(n, dtype=torch.int32))
            self.register_buffer("G", torch.zeros(n, dv), persistent=False)
            self.register_buffer("U", torch.full((L, H, 2, S), 1.0 / S))
            self.register_buffer("pend", torch.zeros(L, H, 2, S), persistent=False)
            self.register_buffer("qres", torch.zeros(L, H, 2, self.nres, 64), persistent=False)
            self.register_buffer("win", torch.zeros(L, H, 2, S), persistent=False)   # --mem-balance-keep: uses since
                                                                                     # the last balancing event
            self.gates.w = nn.Parameter(torch.randn(L, H, ds + dv) * 0.02)
            self.gates.b = nn.Parameter(torch.full((L, H), -2.0))
            self.w_o = nn.Parameter(torch.zeros(L, d, dv))
            self.w_ms = nn.Linear(d, ds, bias=False)
            nn.init.zeros_(self.w_ms.weight)
        if R:
            E = n_exp
            self.register_buffer("P", F.normalize(torch.randn(E, R, 64), dim=-1))
            self.register_buffer("P_ema", self.P.clone())
            self.register_buffer("V_e", torch.randn(E * R, dv) * 0.02)
            self.register_buffer("S_e", torch.zeros(E * R))
            self.register_buffer("wcount_e", torch.zeros(E * R, dtype=torch.int32))
            self.register_buffer("P_m", torch.zeros(E * R, 64))
            self.register_buffer("P_v", torch.zeros(E * R, 64))
            self.register_buffer("P_n", torch.zeros(E * R))
            self.register_buffer("U_e", torch.full((E, R), 1.0 / R))
            self.register_buffer("e_live", torch.tensor(n_live or E))
            self.register_buffer("G_e", torch.zeros(E * R, dv), persistent=False)
            self.register_buffer("G_P", torch.zeros(E * R, 64), persistent=False)
            self.register_buffer("pend_e", torch.zeros(E, R), persistent=False)
            self.register_buffer("win_e", torch.zeros(E, R), persistent=False)
            self.register_buffer("lastq", torch.zeros(E, 64), persistent=False)
            self.register_buffer("lastq_ok", torch.zeros(E, dtype=torch.bool), persistent=False)
            self.gates.b_e = nn.Parameter(torch.full((E,), -2.0))
            self.w_p = nn.Linear(dv, d, bias=False)
            nn.init.zeros_(self.w_p.weight)
        # ---- plan_v2/21, built after everything else (so the existing parts' initialisation is unchanged)
        # pools of the shared table: (suffix, S, H, row offset); the coarse pool's rows follow the fine pool's
        self.pools = [("", S, H, 0)] if S else []
        if self.Sc:
            Sc, Hc, n0 = self.Sc, self.Hc, L * S * S
            self.pools.append(("_c", Sc, Hc, n0))
            nc = L * Sc * Sc
            self.V = torch.cat([self.V, torch.randn(nc, dv) * 0.02])
            self.S_acc = torch.cat([self.S_acc, self.S_acc.new_zeros(nc)])
            self.wcount = torch.cat([self.wcount, self.wcount.new_zeros(nc)])
            self.G = torch.cat([self.G, self.G.new_zeros(nc, dv)])
            self.keys.wq_c = nn.Parameter(torch.randn(L, Hc, 128, self.ain) * 0.02)
            self.keys.k1_c = nn.Parameter(F.normalize(torch.randn(L, Hc, Sc, 64), dim=-1))
            self.keys.k2_c = nn.Parameter(F.normalize(torch.randn(L, Hc, Sc, 64), dim=-1))
            self.register_buffer("wq_c_ema", self.keys.wq_c.detach().clone())
            self.register_buffer("k1_c_ema", self.keys.k1_c.detach().clone())
            self.register_buffer("k2_c_ema", self.keys.k2_c.detach().clone())
            self.register_buffer("U_c", torch.full((L, Hc, 2, Sc), 1.0 / Sc))
            self.register_buffer("pend_c", torch.zeros(L, Hc, 2, Sc), persistent=False)
            self.register_buffer("qres_c", torch.zeros(L, Hc, 2, self.nres, 64), persistent=False)
            self.register_buffer("win_c", torch.zeros(L, Hc, 2, Sc), persistent=False)
            self.gates.w_c = nn.Parameter(torch.randn(L, Hc, ds + dv) * 0.02)
            self.gates.b_c = nn.Parameter(torch.full((L, Hc), -2.0))
            self.w_o_c = nn.Parameter(torch.zeros(L, d, dv))
        if self.life:  # lazy decay / discount: Lambda_r, Psi_r per row; the device copies of Lambda(t), Psi(t)
            if S:
                self.register_buffer("lam_r", torch.zeros(self.V.size(0)))
                self.register_buffer("psi_r", torch.zeros(self.V.size(0)))
            if R:
                self.register_buffer("lam_r_e", torch.zeros(self.V_e.size(0)))
                self.register_buffer("psi_r_e", torch.zeros(self.V_e.size(0)))
            self.register_buffer("lam_now", torch.zeros(()), persistent=False)
            self.register_buffer("psi_now", torch.zeros(()), persistent=False)
        if not self.addr_learn:  # fixed random W_q and sub-keys: no gradient from the start (EMA == online)
            self._apply_freeze()
        self._sync_host()

    # ------------------------------------------------------------ state
    def _sync_host(self):
        """Host mirrors of the small state buffers (reading a GPU buffer per forward would be a host sync)."""
        self._frozen = bool(self.frozen.item())
        self._step = int(self.n_step.item())
        self._s0 = [float(v) for v in self.s0.tolist()]
        self._e_live = int(self.e_live.item()) if self.R else 0
        if self.life:
            self.lam_now.fill_(self.life_state["Lam"])
            self.psi_now.fill_(self.life_state["Psi"])

    @property
    def is_frozen(self):
        return self._frozen

    @property
    def _decaying(self):
        """--mem-life with the rows not yet settled (the decay factors apply on read and write)."""
        return self.life and not self.life_state["settled"]

    def get_extra_state(self):
        if self.life_state:
            return {"probe_state": self.probe_state, "life": dict(self.life_state)}
        return {"probe_state": self.probe_state}

    def set_extra_state(self, state):
        self.probe_state = dict(state.get("probe_state", {})) if isinstance(state, dict) else {}
        if self.life_state and isinstance(state, dict) and "life" in state:
            self.life_state.update(state["life"])

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._sync_host()
        if self._frozen or not self.addr_learn:  # the freeze is part of the state: re-applied on reload
            self._apply_freeze()

    def host_state(self):
        """What a full resume needs beyond state_dict(): the non-persistent buffers (the grad buffers excepted:
        zero between optimizer steps), the RNGs and the logging stats."""
        bufs = {n: b.clone() for n, b in self.named_buffers(recurse=False)
                if n in self._non_persistent_buffers_set and n not in ("G", "G_e", "G_P")}
        return dict(bufs=bufs, gen=self.gen.get_state(), gen_margin=self.gen_margin.get_state(), stats=dict(self.stats))

    def load_host_state(self, st):
        for n, b in st["bufs"].items():
            getattr(self, n).copy_(b)
        self.gen.set_state(st["gen"].cpu())
        self.gen_margin.set_state(st["gen_margin"].cpu())
        self.stats = dict(st["stats"])
        self._sync_host()

    def _apply_freeze(self):
        for p in self.keys.parameters():
            p.requires_grad_(False)
            p.grad = None

    @torch.no_grad()
    def freeze(self):
        """t_f: theta <- theta_bar, P <- P_bar, then the translators, sub-keys, norm statistics and private keys
        are frozen (no gradient, no EMA, no norm updates, no balancing)."""
        self.keys.wq.copy_(self.wq_ema)
        if self.S:
            self.keys.k1.copy_(self.k1_ema)
            self.keys.k2.copy_(self.k2_ema)
        if self.Sc:
            self.keys.wq_c.copy_(self.wq_c_ema)
            self.keys.k1_c.copy_(self.k1_c_ema)
            self.keys.k2_c.copy_(self.k2_c_ema)
        if self.R:
            self.P.copy_(self.P_ema)
            self.G_P.zero_()  # (the keys get no gradient from now on: after_step no longer clears G_P)
        self.frozen.fill_(1)
        self._frozen = True
        self._apply_freeze()
        if self._decaying:  # mature: no decay from now on, the pending factors are applied once
            self.settle()

    # ------------------------------------------------------------ life cycle (plan_v2/21 section 1)
    def _tables(self, which):
        """The per-row tables of the shared (0) / private (1) memory that move together with a row."""
        t = [self.V, self.wcount, self.S_acc] if which == 0 else [self.V_e, self.wcount_e, self.S_e]
        if self.life:
            t += [self.lam_r, self.psi_r] if which == 0 else [self.lam_r_e, self.psi_r_e]
        return t

    def _factors(self, rows, which=0):
        """The pending decay exp(-(Lambda(t) - Lambda_r)) of the gathered rows (None when no decay is pending)."""
        if not self._decaying:
            return None
        lr = self.lam_r if which == 0 else self.lam_r_e
        return torch.exp(lr[rows] - self.lam_now)

    def life_advance(self, decay, beta):
        """One optimizer step of the global sums: Lambda += -log(1 - lambda), Psi += -log(beta) (fp64 on the host,
        an fp32 device copy)."""
        ls = self.life_state
        ls["Lam"] += -math.log1p(-decay)
        ls["Psi"] += -math.log(beta)
        self.lam_now.fill_(ls["Lam"])
        self.psi_now.fill_(ls["Psi"])

    @torch.no_grad()
    def settle(self, chunk=1 << 16):
        """Apply every row's pending decay and discount once (V <- f V, S <- max(B S, floor), Lambda_r <- Lambda,
        Psi_r <- Psi): the effective values do not change; from then on the plain row-Adagrad applies."""
        for which in (0, 1):
            if (which == 0 and not self.S) or (which == 1 and not self.R):
                continue
            V, _, S_acc, lr, pr = self._tables(which)
            floor = S_FLOOR * self._s0[which] if self._s0[which] > 0 else 0.0
            for c in range(0, V.size(0), chunk):
                sl = slice(c, c + chunk)
                V[sl] *= torch.exp(lr[sl] - self.lam_now)[:, None]
                S_acc[sl] = torch.clamp(S_acc[sl] * torch.exp(pr[sl] - self.psi_now), min=floor)
            pr.copy_(self.psi_now.expand_as(pr))
            lr.copy_(self.lam_now.expand_as(lr))
        self.life_state["settled"] = True

    def _reset_rows(self, which, rows, full=False, zero=False):
        """Rows of a split / re-seeded key: S <- S0 (Psi_r <- Psi). full (plan_v2/21 section 1.3, the band of a
        re-seeded key): also the write count 0 and the pending decay applied (V <- f V, Lambda_r <- Lambda: the
        content keeps its decayed value; a band is shared with the other heads' live keys, so V is not cleared);
        zero (--mem-restart-zero): also V <- 0. rows: distinct."""
        S_acc = self.S_acc if which == 0 else self.S_e
        self._reset_s(S_acc, rows, which)
        if self.life:
            (self.psi_r if which == 0 else self.psi_r_e).index_fill_(0, rows, self.psi_now)
        if full or zero:
            V = self.V if which == 0 else self.V_e
            (self.wcount if which == 0 else self.wcount_e).index_fill_(0, rows, 0)
            if zero:
                V.index_fill_(0, rows, 0.0)
            elif self._decaying:
                f = self._factors(rows, which)
                V.index_copy_(0, rows, V.index_select(0, rows) * f[:, None])
            if self.life:
                (self.lam_r if which == 0 else self.lam_r_e).index_fill_(0, rows, self.lam_now)

    def begin_forward(self, collect):
        """Start of a forward. collect (a training micro-batch): usage counts and, for the EMA norm statistics,
        this micro-batch replaces the previous one (the statistics come from the LAST micro-batch of a step)."""
        self._collect = bool(collect) and not self._frozen
        if self._collect:
            self.ns_sum.zero_()
            self.ns_sq.zero_()
            self.ns_cnt.zero_()

    # ------------------------------------------------------------ addressing (fp32, no autocast, no TF32)
    def prep(self, mus, pbar):
        """a = [mu_l ; p_bar] (L, N, d + 64) and a_hat = (a - m) / sqrt(v + 1e-5) with the EMA statistics."""
        dt = self.keys.wq.dtype
        with fp32_math(pbar.device.type):
            pb = pbar.to(dt).flatten(0, -2)
            a = torch.stack([torch.cat([mu.to(dt).flatten(0, -2), pb], -1) for mu in mus])
            a_hat = (a - self.norm_m[:, None]) / torch.sqrt(self.norm_v[:, None] + 1e-5)
        return a, a_hat

    def _queries(self, a_hat, wq, raw=False):
        """q = normalize(W_q a_hat) per level / head, halves q1 = normalize(q[:64]), q2 = normalize(q[64:]).
        Without a shared pool only level 1 head 1 is needed (the private query): (1, 1, N, 64). raw: also W_q a_hat."""
        if self.S:
            qr = torch.einsum("lnc,lhoc->lhno", a_hat, wq)
        else:
            qr = (a_hat[0] @ wq[0, 0].t())[None, None]
        q = F.normalize(qr, dim=-1)
        q1, q2 = F.normalize(q[..., :64], dim=-1), F.normalize(q[..., 64:], dim=-1)
        return (qr, q1, q2) if raw else (q1, q2)

    @torch.no_grad()
    def norm_collect(self, a):
        """select()'s EMA norm statistics of a (L, N, d + 64), on their own (plan_v2/22: the gate gathers the open
        positions before select(), the statistics stay over ALL positions)."""
        with fp32_math(a.device.type):
            if self._collect:
                self.ns_sum += a.sum(1)
                self.ns_sq += ((a - self.norm_m[:, None]) ** 2).sum(1)
                self.ns_cnt += a.size(1)

    @torch.no_grad()
    def select(self, a, a_hat, collect, stats=True):
        """Row indices from the EMA translator theta_bar (no gradient). Returns i, j (L, H, N, km + 1) top pairs, I1,
        I2 (L, H, N, m + 1) per-half top lists, rows (L, H, N, km) and qb1 (N, 64), the EMA private query; after the
        freeze with sg addressing also q = (q1, q2), the queries (online = EMA then: online() reuses them).
        stats False (the gate): no norm statistics here (norm_collect() takes them over all positions)."""
        L, H, S, m, km = self.L, self.H, self.S, self.m, self.km
        out = {}
        with fp32_math(a_hat.device.type):
            if collect and self._collect and stats:  # EMA norm statistics (applied once per optimizer step)
                self.ns_sum += a.sum(1)
                self.ns_sq += ((a - self.norm_m[:, None]) ** 2).sum(1)
                self.ns_cnt += a.size(1)
            q1, q2 = self._queries(a_hat, self.wq_ema)
            out["qb1"] = q1[0, 0]
            if self._frozen and self.addr_sg:  # theta == theta_bar and the same sg input: the same queries
                out["q"] = (q1, q2)
            if S:
                kk = max(km + 1, self.nbr)     # --mem-nbr: the top-nbr pairs (reads use the top-km)
                out["i"], out["j"], out["I1"], out["I2"] = self._addr(q1, q2, kk=kk)
                off = (torch.arange(L, device=a.device) * S * S).view(L, 1, 1, 1)
                out["rows"] = off + out["i"][..., :km] * S + out["j"][..., :km]
                if self.nbr:  # the neighbour rows: ranks km + 1 .. nbr
                    out["nrows"] = off + out["i"][..., km:self.nbr] * S + out["j"][..., km:self.nbr]
                if collect and self._collect:  # sub-key usage (top-m selections) and recent query halves
                    self._usage(out, q1, q2, a.size(1), "")
            for sfx, Sp, _, off0 in self.pools[1:]:  # the coarse pool: its own fixed translator and sub-keys
                q1c, q2c = self._queries(a_hat, getattr(self, f"wq{sfx}_ema"))
                o = {}
                if self._frozen and self.addr_sg:
                    o["q"] = (q1c, q2c)
                o["i"], o["j"], o["I1"], o["I2"] = self._addr(q1c, q2c, sfx)
                off = off0 + (torch.arange(L, device=a.device) * Sp * Sp).view(L, 1, 1, 1)
                o["rows"] = off + o["i"][..., :km] * Sp + o["j"][..., :km]
                if collect and self._collect:
                    self._usage(o, q1c, q2c, a.size(1), sfx)
                out[sfx[1:]] = o
        return out

    def _usage(self, o, q1, q2, n, sfx):
        """select(): a pool's sub-key usage (top-m selections) and its recent query halves."""
        m = self.m
        pend, qres = getattr(self, "pend" + sfx), getattr(self, "qres" + sfx)
        sel = torch.stack([o["I1"][..., :m], o["I2"][..., :m]], 2).flatten(3)    # (L, H, 2, N m)
        pend.scatter_add_(-1, sel, torch.ones_like(sel, dtype=pend.dtype))
        pos = torch.linspace(0, n - 1, self.nres, device=q1.device).long()
        qres[:, :, 0] = q1[:, :, pos]
        qres[:, :, 1] = q2[:, :, pos]

    def _addr(self, q1, q2, sfx="", kk=None):
        """select()'s product-key addressing, all levels and both halves at once: the score GEMMs stay one call per
        level and half (the same kernels as one level at a time), written into one buffer; then ONE stable_topk over
        every (half, level, head, token) row and ONE pk_topk. Levels are grouped so that a group's buffer stays
        under SELECT_BUDGET elements. Returns i, j (L, H, N, kk = km + 1), I1, I2 (L, H, N, m + 1). sfx: the pool."""
        L, H, N = q1.shape[:3]
        S, m, km = self.Sc if sfx else self.S, self.m, (kk or self.km + 1) - 1
        k1, k2 = F.normalize(getattr(self, f"k1{sfx}_ema"), dim=-1), F.normalize(getattr(self, f"k2{sfx}_ema"), dim=-1)
        per = max(1, min(L, SELECT_BUDGET // max(1, 2 * H * N * S)))
        outs = []
        for l0 in range(0, L, per):
            l1 = min(L, l0 + per)
            sc = q1.new_empty(2, l1 - l0, H, N, S)
            for l in range(l0, l1):
                torch.matmul(q1[l], k1[l].transpose(-1, -2), out=sc[0, l - l0])
                torch.matmul(q2[l], k2[l].transpose(-1, -2), out=sc[1, l - l0])
            v, ix = stable_topk(sc, m + 1)                                     # (2, l, H, N, m + 1)
            del sc
            ii, jj, _ = pk_topk(v[0, ..., :m], ix[0, ..., :m], v[1, ..., :m], ix[1, ..., :m], S, km + 1)
            outs.append((ii, jj, ix[0], ix[1]))
        return outs[0] if len(outs) == 1 else tuple(torch.cat(t) for t in zip(*outs))

    def _key_rows(self, K, idx):
        """The normalised sub-keys K (L, H, S, 64) at idx (L, H, N, k): (L, H, N, k, 64)."""
        L, H, S = K.shape[:3]
        base = torch.arange(L * H, device=idx.device).view(L, H, 1, 1) * S
        return F.embedding(base + idx, F.normalize(K, dim=-1).flatten(0, 2))

    def _key_scores(self, q, K, idx):
        """Cosines of the queries q (L, H, N, 64) with the normalised sub-keys K (L, H, S, 64) at idx (L, H, N, k)."""
        return (q[..., None, :] * self._key_rows(K, idx)).sum(-1)

    def online(self, a_hat, sel, margin=False):
        """Weights from the ONLINE translator on the EMA-chosen pairs: w = softmax(sigma(theta) / tau) (L, H, N, km);
        sigma also for the (km+1)-th pair; q1 = the online private query (N, 64). Gradient reaches W_q, K only here.
        margin (infancy): also the margin loss (as margin_loss) from the SAME queries, as a second use whose
        gradient reaches W_q only (as if computed from sg(a): _QSg), and the boundary sub-keys gathered together
        with the read's. After the freeze with sg addressing, sel's EMA queries are these queries (no second W_q)."""
        m, km = self.m, self.km
        with fp32_math(a_hat.device.type):
            if "q" in sel:
                assert not margin
                q1, q2 = sel["q"]
            else:
                qr, q1, q2 = self._queries(a_hat, self.keys.wq, raw=True)
            out = {"q1": q1[0, 0]}
            if self.S:
                i, j = sel["i"], sel["j"]
                if self.nbr:  # (the extra neighbour pairs are scored below, without gradient)
                    i, j = i[..., :km + 1], j[..., :km + 1]
                if margin:  # + the per-half boundary sub-keys (m-th, (m+1)-th)
                    i, j = torch.cat([i, sel["I1"][..., m - 1:m + 1]], -1), torch.cat([j, sel["I2"][..., m - 1:m + 1]], -1)
                r1, r2 = self._key_rows(self.keys.k1, i), self._key_rows(self.keys.k2, j)
                if margin:
                    sig = ((q1[..., None, :] * r1[..., :km + 1, :]).sum(-1)
                           + (q2[..., None, :] * r2[..., :km + 1, :]).sum(-1))
                    p1, p2 = _QSg.apply(self.keys.wq, a_hat.detach(), qr.detach(), q1.detach(), q2.detach())
                    s1 = (p1[..., None, :] * r1[..., km - 1:, :]).sum(-1)     # pairs km, km+1 and the half
                    s2 = (p2[..., None, :] * r2[..., km - 1:, :]).sum(-1)     # boundaries, as in gaps()
                    gs = s1[..., :2] + s2[..., :2]
                    out["margin"] = self._margin(gs[..., 0] - gs[..., 1], s1[..., 2] - s1[..., 3], s2[..., 2] - s2[..., 3])
                else:
                    sig = (q1[..., None, :] * r1).sum(-1) + (q2[..., None, :] * r2).sum(-1)
                out["sig"] = sig
                out["w"] = torch.softmax(sig[..., :self.km] / self.tau, -1)
                if self.nbr and self.training:  # neighbour-write weights (training only: eval never writes): alpha x
                    # softmax over ranks km + 1 .. nbr of sigma / tau
                    with torch.no_grad():
                        ix, jx = sel["i"][..., km + 1:], sel["j"][..., km + 1:]
                        sx = (self._key_scores(q1.detach(), self.keys.k1, ix)
                              + self._key_scores(q2.detach(), self.keys.k2, jx))
                        sn = torch.cat([sig[..., km:km + 1].detach(), sx], -1)
                        out["wn"] = self.nbr_w * torch.softmax(sn / self.tau, -1)
            for sfx, _, _, _ in self.pools[1:]:  # the coarse pool: its own translator / sub-keys, no margin
                sc = sel[sfx[1:]]
                if "q" in sc:
                    p1, p2 = sc["q"]
                else:
                    p1, p2 = self._queries(a_hat, getattr(self.keys, "wq" + sfx))
                sg = (self._key_scores(p1, getattr(self.keys, "k1" + sfx), sc["i"])
                      + self._key_scores(p2, getattr(self.keys, "k2" + sfx), sc["j"]))
                out[sfx[1:]] = dict(sig=sg, w=torch.softmax(sg[..., :km] / self.tau, -1))
        return out

    def gaps(self, a_hat, sel):
        """Set-boundary gaps with the online scores at the EMA ranks: gamma = sigma_(km) - sigma_(km+1) and per half
        s_(m) - s_(m+1); (L, H, N) each."""
        m, km = self.m, self.km
        with fp32_math(a_hat.device.type):
            q1, q2 = self._queries(a_hat, self.keys.wq)
            s1 = self._key_scores(q1, self.keys.k1, torch.cat([sel["i"][..., km - 1:km + 1], sel["I1"][..., m - 1:m + 1]], -1))
            s2 = self._key_scores(q2, self.keys.k2, torch.cat([sel["j"][..., km - 1:km + 1], sel["I2"][..., m - 1:m + 1]], -1))
            sig = s1[..., :2] + s2[..., :2]
            return sig[..., 0] - sig[..., 1], s1[..., 2] - s1[..., 3], s2[..., 2] - s2[..., 3]

    def margin_loss(self, a_hat_sg, sel):
        """Infancy only, on sg(a): lambda * mean[relu(delta - g) + 1/2 relu(delta - g1) + 1/2 relu(delta - g2)]
        (its own pass; training uses online(margin=True): the same values)."""
        return self._margin(*self.gaps(a_hat_sg.detach(), sel))

    def _margin(self, g, g1, g2):
        dl = self.delta
        loss = self.lam_m * (F.relu(dl - g) + 0.5 * F.relu(dl - g1) + 0.5 * F.relu(dl - g2)).mean()
        with torch.no_grad():
            self.margin_stats = dict(active=(g < dl).float().mean(), active_half=((g1 < dl).float().mean()
                                                                                  + (g2 < dl).float().mean()) / 2,
                                     gap=g.mean(), gap_half=(g1.mean() + g2.mean()) / 2)
        return loss

    # ------------------------------------------------------------ shared read (3)
    def read_shared(self, w, sel, pbar, on=None):
        """One fused MemRead for all L x H heads, gates and W_o. Returns M (N, d) and the gates (N, L, H).
        on: the online() outputs (the neighbour-write weights; the coarse pool's weights, whose read, gates and W_o^c
        are added to M: one more MemRead; its gates are stored in on["g_c"])."""
        L, H, km, dv = self.L, self.H, self.km, self.dv
        N = w.size(2)
        rows = sel["rows"].permute(2, 0, 1, 3).reshape(N * L * H, km)
        ww = w.permute(2, 0, 1, 3).reshape(N * L * H, km)
        write = torch.is_grad_enabled() and self.write_on
        with fp32_math(pbar.device.type):
            if self._decaying or (self.nbr and write):  # (plan_v2/21: the decay factors, the neighbour writes)
                ex = {}
                if self.nbr and write:
                    nx = self.nbr - km
                    ex = dict(xrows=sel["nrows"].permute(2, 0, 1, 3).reshape(N * L * H, nx),
                              xw=on["wn"].permute(2, 0, 1, 3).reshape(N * L * H, nx))
                f = self._factors(rows)
                r = MemRead.apply(_anchor(pbar.device, write), ww if f is None else ww * f, self.V,
                                  self.G if write else None, rows, None if f is None else ww.detach(), *ex.values())
            else:
                r = MemRead.apply(_anchor(pbar.device, write), ww, self.V, self.G if write else None, rows)
        r = r.view(N, L, H, dv)
        gw = self.gates.w
        pb = pbar.flatten(0, -2)
        g = torch.sigmoid(torch.einsum("nc,lhc->nlh", pb.to(gw.dtype), gw[..., :self.ds])
                          + torch.einsum("nlhc,lhc->nlh", r.to(gw.dtype), gw[..., self.ds:]) + self.gates.b)
        if self.ko_levels:  # knock-out probe: the reads of these levels are zero
            keep = torch.ones(L, dtype=g.dtype, device=g.device)
            keep[list(self.ko_levels)] = 0
            g = g * keep[None, :, None]
        u = (g[..., None].to(r.dtype) * r).sum(2)                                    # (N, L, dv)
        M = torch.einsum("nlc,ldc->nd", u.to(self.w_o.dtype), self.w_o)
        for sfx, _, Hp, _ in self.pools[1:]:  # the coarse pool: M += sum_l W_o^c,l g^c r^c
            if self.ko_coarse:
                continue
            sc, oc = sel[sfx[1:]], on[sfx[1:]]
            rows = sc["rows"].permute(2, 0, 1, 3).reshape(N * L * Hp, km)
            ww = oc["w"].permute(2, 0, 1, 3).reshape(N * L * Hp, km)
            with fp32_math(pbar.device.type):
                f = self._factors(rows)
                r = MemRead.apply(_anchor(pbar.device, write), ww if f is None else ww * f, self.V,
                                  self.G if write else None, rows, None if f is None else ww.detach())
            r = r.view(N, L, Hp, dv)
            gw = getattr(self.gates, "w" + sfx)
            gc = torch.sigmoid(torch.einsum("nc,lhc->nlh", pb.to(gw.dtype), gw[..., :self.ds])
                               + torch.einsum("nlhc,lhc->nlh", r.to(gw.dtype), gw[..., self.ds:])
                               + getattr(self.gates, "b" + sfx))
            if self.ko_levels:
                gc = gc * keep[None, :, None]
            u = (gc[..., None].to(r.dtype) * r).sum(2)
            w_o = getattr(self, "w_o" + sfx)
            M = M + torch.einsum("nlc,ldc->nd", u.to(w_o.dtype), w_o)
            oc["g"] = gc
        return M, g

    # ------------------------------------------------------------ private expert memory (4)
    def private(self, q1, qb1, e_k, p_k, t_k, g_k, fpos, N, k, E, C, omask=None):
        """Called inside RoutedMLP with its capacity layout (kept slots: expert e_k, slot p_k, token t_k, gate g_k,
        flat (token, choice) position fpos). Indices from the EMA side (qb1 with P_bar, bmm over the capacity
        layout, top-4, stable tie-break), weights from the online side. Returns W_p(sum_e gate sigma(b_e) rho).
        omask (plan_v2/22, the memory gate): (N,) bool, the open tokens; a closed token's slots get no output (so no
        G_e / G_P rows and no b_e gradient from it), no usage count, and the re-seed query comes from open slots."""
        R, dev = self.R, q1.device
        dt = self.P.dtype
        collect = self._collect and torch.is_grad_enabled()      # a training pass inside the gradient window
        ok = None if omask is None else omask[t_k]
        with torch.no_grad(), fp32_math(dev.type):
            Q = qb1.new_zeros(E, C, 64)
            Q[e_k, p_k] = qb1[t_k]
            sc = torch.bmm(Q, F.normalize(self.P_ema[:E], dim=-1).transpose(1, 2))   # (E, C, R)
            idx = stable_topk(sc[e_k, p_k], P_TOPK)[1]                               # kept slots only: empty
            rows = e_k[:, None] * R + idx                                            # slots are never read
            del sc
            if collect:
                ones = (torch.ones(rows.numel(), dtype=self.pend_e.dtype, device=dev) if ok is None
                        else ok[:, None].expand_as(rows).flatten().to(self.pend_e.dtype))
                self.pend_e.view(-1).index_add_(0, rows.flatten(), ones)
                if ok is None:
                    has = torch.zeros(E, dtype=torch.bool, device=dev)
                    has[e_k] = True
                else:  # (slot 0 open: its query is a real one; no boolean index, no host sync)
                    has = torch.zeros(E, dtype=torch.float32, device=dev).index_add_(
                        0, e_k, (ok & (p_k == 0)).float()) > 0
                self.lastq[:E] = torch.where(has[:, None], Q[:, 0].to(self.lastq.dtype), self.lastq[:E])
                self.lastq_ok[:E] |= has
            if self.record is not None:
                rt = rows.new_full((N * k, P_TOPK), -1)
                rt[fpos] = rows
                self._prows = rt.view(N, k * P_TOPK)[self.record["pos"]].sort(-1).values.cpu()
        learn = torch.is_grad_enabled() and not self._frozen
        write = torch.is_grad_enabled() and self.write_on
        tau = self.tau

        def f(q1, g_k, b_e):
            with fp32_math(dev.type):
                Pf = self.P.view(-1, 64)
                Pk = GatherRows.apply(_anchor(dev, True), Pf, self.G_P, rows) if learn else Pf[rows]
                Pk = F.normalize(Pk, dim=-1)
                w = torch.softmax((q1.to(dt)[t_k][:, None, :] * Pk).sum(-1) / tau, -1)
                fe = self._factors(rows, 1)   # --mem-life: the pending decay of the private rows
                if fe is None:
                    rho = MemRead.apply(_anchor(dev, write), w, self.V_e, self.G_e if write else None, rows)
                else:
                    rho = MemRead.apply(_anchor(dev, write), w * fe, self.V_e, self.G_e if write else None, rows,
                                        w.detach())
                gs = g_k.to(dt) * torch.sigmoid(b_e[e_k].to(dt))
                if ok is not None:  # (the gate: closed tokens' slots contribute nothing, forward or backward)
                    gs = gs * ok.to(dt)
                z = rho.new_zeros(N, self.dv).index_add_(0, t_k, rho * gs[:, None])
            return self.w_p(z.to(self.w_p.weight.dtype))

        if torch.is_grad_enabled() and self.ckpt:
            return torch.utils.checkpoint.checkpoint(f, q1, g_k, self.gates.b_e, use_reentrant=False)
        return f(q1, g_k, self.gates.b_e)

    # ------------------------------------------------------------ growth (copy-on-split)
    @torch.no_grad()
    def on_grow(self, src, E0, split):
        """Expert births: a split twin gets exact copies of its parent's P, P_bar, V_e, S_e, b_e and MemOpt rows
        (and usage); ramp-mode children are never selected at birth and keep their own fresh rows."""
        n = len(src)
        if not self.R:
            if self._frozen:
                self._apply_freeze()
            return
        assert E0 + n <= self.Emax, f"private memory preallocated for {self.Emax} experts, got {E0 + n}"
        if split:
            dev = self.P.device
            src = src.to(dev)
            dst = torch.arange(E0, E0 + n, device=dev)
            R = self.R
            for t in (self.P, self.P_ema, self.U_e, self.gates.b_e.data):
                t[dst] = t[src]
            for t in self._tables(1) + [self.P_m, self.P_v, self.P_n]:
                tv = t.view(self.Emax, R, *t.shape[1:])
                tv[dst] = tv[src]
        self.e_live.fill_(E0 + n)
        self._e_live = E0 + n
        if self._frozen:  # (births should end before the freeze; if not, the keys stay frozen)
            self._apply_freeze()

    # ------------------------------------------------------------ MemOpt (3.1) and the per-step updates
    @torch.no_grad()
    def write_counts(self):
        """The rows this step's MemOpt / private-key Adam will touch: any bit set in their G row (one pass over G,
        the live part of G_e and G_P). Returns their counts as ONE device tensor (or None): the trainer reads it
        together with the loss (one host sync) and passes the list to after_step; the rows themselves are taken
        with nonzero_static (no further sync)."""
        nl = self._e_live * self.R
        self._touch = [(n, G[:rows].view(_INT[G.element_size()]).any(1)) for n, G, rows, on in (
            ("V", getattr(self, "G", None), None, self.S and self.write_on),
            ("V_e", getattr(self, "G_e", None), nl, self.R and self.write_on),
            ("P", getattr(self, "G_P", None), nl, self.R and not self._frozen)) if on]
        return torch.stack([t.sum() for _, t in self._touch]) if self._touch else None

    def _rows(self, name, counts):
        """The touched rows of one table (ascending), from write_counts' masks and the host counts."""
        k = [n for n, _ in self._touch].index(name)
        return torch.nonzero_static(self._touch[k][1], size=int(counts[k])).squeeze(1)

    @torch.no_grad()
    def _row_adagrad(self, V, G, S_acc, wcount, which, idx, chunk=1 << 17):
        """Sparse row-Adagrad with a prior: for rows with G_r != 0: S_r += mean_i G_ri^2 (S starts at S0),
        dV = -lr G_r / sqrt(S_r), clipped to |dV| <= clip; V_r += dV. Other rows stay bit-identical. Only the
        touched rows idx (any bit set in G_r: a superset of G_r != 0) are gathered, updated and written back; G
        is cleared on them (the other rows are zero already).
        --mem-life (plan_v2/21), before the update: V_r <- exp(-(Lambda - Lambda_r)) V_r, Lambda_r <- Lambda, and
        the discounted accumulator S_r <- max(exp(-(Psi - Psi_r)) S_r + mean G_r^2, S_FLOOR x S0), Psi_r <- Psi."""
        if self._s0[which] < 0:  # S0: absolute, or auto = 100 x median over the written rows (first write step)
            ms = []
            for c in range(0, idx.numel(), chunk):  # (per-row means first: no table-sized temporary)
                g = G.index_select(0, idx[c:c + chunk])
                ms.append((g * g).mean(1)[(g != 0).any(1)])
            ms = torch.cat(ms) if ms else G.new_zeros(0)
            if not ms.numel():
                return
            s0 = 100.0 * float(ms.median()) if self.s0_cfg == "auto" else float(self.s0_cfg)
            S_acc.fill_(s0)
            self.s0[which] = s0
            self._s0[which] = s0
            if self.life:  # (S0 is set now: its discount starts now)
                (self.psi_r if which == 0 else self.psi_r_e).copy_(self.psi_now.expand_as(S_acc))
        lr, clip = self.lr, self.clip
        life = self._decaying
        if life:
            lam_r, psi_r = (self.lam_r, self.psi_r) if which == 0 else (self.lam_r_e, self.psi_r_e)
            floor = S_FLOOR * self._s0[which]
        for c in range(0, idx.numel(), chunk):
            ii = idx[c:c + chunk]
            g = G.index_select(0, ii)
            act = (g != 0).any(1)
            s = S_acc.index_select(0, ii)
            v = V.index_select(0, ii)
            if life:  # the pending decay and discount first, then the usual update
                lri, pri = lam_r.index_select(0, ii), psi_r.index_select(0, ii)
                s = torch.where(act, torch.clamp(s * torch.exp(pri - self.psi_now) + (g * g).mean(1), min=floor), s)
                v = torch.where(act[:, None], v * torch.exp(lri - self.lam_now)[:, None], v)
                lam_r.index_copy_(0, ii, torch.where(act, self.lam_now, lri))
                psi_r.index_copy_(0, ii, torch.where(act, self.psi_now, pri))
            else:
                s = torch.where(act, s + (g * g).mean(1), s)
            dv = -lr * g / s.sqrt()[:, None]
            nrm = dv.norm(dim=1, keepdim=True)
            dv = dv * (clip / nrm.clamp(min=1e-30)).clamp(max=1.0)
            V.index_copy_(0, ii, torch.where(act[:, None], v + dv, v))
            S_acc.index_copy_(0, ii, s)
            wcount.index_add_(0, ii, act.to(wcount.dtype))
        G.index_fill_(0, idx, 0)

    @torch.no_grad()
    def _adam_keys(self, lr, idx, b1=0.9, b2=0.95, eps=1e-8):
        """Private keys P: Adam, no weight decay, lazy per row (rows without gradient untouched). Only the touched
        rows idx are gathered and written back (the bias corrections over all rows, as one small (rows, 1) op)."""
        G = self.G_P
        g = G.index_select(0, idx)
        act = (g != 0).any(1)
        self.P_n.index_add_(0, idx, act.to(self.P_n.dtype))
        a = act[:, None]
        m, v = self.P_m.index_select(0, idx), self.P_v.index_select(0, idx)
        m = torch.where(a, b1 * m + (1 - b1) * g, m)
        v = torch.where(a, b2 * v + (1 - b2) * g * g, v)
        n = self.P_n.clamp(min=1)[:, None]
        upd = (m / (1 - b1 ** n).index_select(0, idx)) / ((v / (1 - b2 ** n).index_select(0, idx)).sqrt() + eps)
        P = self.P.view(-1, 64)
        p = P.index_select(0, idx)
        P.index_copy_(0, idx, torch.where(a, p - lr * upd, p))
        self.P_m.index_copy_(0, idx, m)
        self.P_v.index_copy_(0, idx, v)
        G.index_fill_(0, idx, 0)

    @torch.no_grad()
    def after_step(self, lr, counts=None):
        """Once per optimizer step, after BrainOpt. Infancy: EMA norm statistics, EMA translator / private keys,
        private-key Adam (lr = the trainer's current rate), usage EMAs, balancing every balance_every steps.
        Always: the table writes (after --mem-write-start). counts: write_counts() read on the host by the trainer
        (with the loss); None = read here (one host sync)."""
        if counts is None:
            cnt = self.write_counts()
            counts = cnt.tolist() if cnt is not None else []
        self._step += 1
        self.n_step.fill_(self._step)
        ls = self.life_state
        if self._decaying:  # Lambda(t), Psi(t) of this step (the writes below and the next reads use them)
            self.life_advance(self.decay0 if ls["decay"] is None else ls["decay"],
                              self.beta0 if ls["beta"] is None else ls["beta"])
        ema = self.ema
        if not self._frozen:
            cnt = self.ns_cnt.clamp(min=1)
            has = self.ns_cnt > 0
            self.norm_m.copy_(torch.where(has, ema * self.norm_m + (1 - ema) * self.ns_sum / cnt, self.norm_m))
            self.norm_v.copy_(torch.where(has, ema * self.norm_v + (1 - ema) * self.ns_sq / cnt, self.norm_v))
            self.wq_ema.lerp_(self.keys.wq.detach(), 1 - ema)
            for sfx, _, _, _ in self.pools:
                if sfx:
                    getattr(self, f"wq{sfx}_ema").lerp_(getattr(self.keys, "wq" + sfx).detach(), 1 - ema)
                getattr(self, f"k1{sfx}_ema").lerp_(getattr(self.keys, "k1" + sfx).detach(), 1 - ema)
                getattr(self, f"k2{sfx}_ema").lerp_(getattr(self.keys, "k2" + sfx).detach(), 1 - ema)
                pend, U = getattr(self, "pend" + sfx), getattr(self, "U" + sfx)
                tot = pend.sum(-1, keepdim=True)
                U.copy_(torch.where(tot > 0, (1 - self.usage_rate) * U + self.usage_rate * pend
                                    / tot.clamp(min=1), U))
                if self.balance_keep or self.restart:
                    getattr(self, "win" + sfx).add_(pend)
                pend.zero_()
            if self.R:
                self._adam_keys(lr, self._rows("P", counts))
                self.P_ema.lerp_(self.P, 1 - ema)
                tot = self.pend_e.sum(-1, keepdim=True)                          # experts without traffic: kept
                self.U_e.copy_(torch.where(tot > 0, (1 - self.usage_rate) * self.U_e + self.usage_rate * self.pend_e
                                           / tot.clamp(min=1), self.U_e))
                if self.balance_keep:
                    self.win_e += self.pend_e
                self.pend_e.zero_()
            if self.balance_every and self._step % self.balance_every == 0:
                self.balance()
            if self.restart and self.S:  # --mem-restart: re-seed every key unused since the last event
                every = self.restart if ls["restart_every"] is None else ls["restart_every"]
                if every and self._step - ls["last_restart"] >= every:
                    self.restart_keys()
                    ls["last_restart"] = self._step
        else:  # (G_P stays zero after the freeze: cleared by freeze())
            for sfx, _, _, _ in self.pools:
                getattr(self, "pend" + sfx).zero_()
            if self.R:
                self.pend_e.zero_()
        if self.S:
            if self.write_on:
                self._row_adagrad(self.V, self.G, self.S_acc, self.wcount, 0, self._rows("V", counts))
            else:
                self.G.zero_()
        if self.R:
            if self.write_on:
                self._row_adagrad(self.V_e, self.G_e, self.S_e, self.wcount_e, 1, self._rows("V_e", counts))
            else:
                self.G_e.zero_()
        self._touch = None

    # ------------------------------------------------------------ sub-key balancing (2.2)
    def band(self, l, half, i):
        """Rows addressed through sub-key i of half `half` at level l: half 1 = rows i*S .. i*S + S - 1, half 2 =
        the strided column i (a*S + i)."""
        S = self.S
        a = torch.arange(S, device=self.V.device)
        return l * S * S + (i * S + a if half == 0 else a * S + i)

    def _reset_s(self, S_acc, rows, which):
        if self._s0[which] > 0:
            S_acc.index_fill_(0, rows, self._s0[which])

    def _bands(self, l, half, i, S=None, off=0):
        """band() for many keys at once (CPU LongTensors (n,)): (n, S). S, off: another pool's size and row offset."""
        S = S or self.S
        a = torch.arange(S)
        return off + l[:, None] * S * S + torch.where(half[:, None] == 0, i[:, None] * S + a, a * S + i[:, None])

    def _reset_keys(self, l, h, half, i, sfx=""):
        """The trainer's fresh optimizer moments for edited sub-keys, one call per codebook half (the reset value,
        the median of the second moments, does not change when entries are set to it: one call == one per key)."""
        if self.opt_reset is None:
            return
        for hf in (0, 1):
            s = half == hf
            if s.any():
                self.opt_reset(f"mem.keys.k{hf + 1}{sfx}", (l[s], h[s], i[s]))

    @torch.no_grad()
    def restart_keys(self):
        """--mem-restart (plan_v2/21 section 2), per pool: every sub-key unused since the last event (no top-m
        selection: the window win) is re-seeded, uncapped, as normalize(a recent query half + N(0, 0.05)) from the
        reservoir of nres recent query halves of its codebook: distinct slots per event (a random order of the
        recorded slots per codebook, with the dedicated RNG). Its band's MemOpt state is reset (S = S0, write
        count 0, Lambda_r / Psi_r brought current with the pending decay applied; V <- 0 only with
        --mem-restart-zero: the band's rows are shared with the other heads' live keys). No splits. Host syncs: two small transfers (win, qres.any) per pool."""
        gen, st = self.gen, {}
        for sfx, S, H, off in self.pools:
            L, nres, dev = self.L, self.nres, self.V.device
            win, Uv = getattr(self, "win" + sfx), getattr(self, "U" + sfx).view(-1, S)
            qres = getattr(self, "qres" + sfx).view(-1, nres, 64)
            dead = (win.view(-1, S) == 0).cpu()                                 # (C, S)
            okq = qres.any(-1).cpu()                                            # (C, nres)
            C = dead.size(0)
            order = torch.where(okq, torch.rand(C, nres, generator=gen), torch.full((C, nres), 2.0))
            order = order.argsort(dim=1, stable=True)                           # recorded slots first, random order
            rank = dead.long().cumsum(1) - 1                                    # the t-th dead key of its codebook
            c, i = dead.nonzero(as_tuple=True)
            r = rank[c, i]
            ok = r < okq.sum(1)[c]
            c, i, r = c[ok], i[ok], r[ok]
            n = len(c)
            if n:
                slot = order[c, r]
                noise = torch.randn(n, 64, generator=gen) * 0.05
                l, h, half = c // (2 * H), (c // 2) % H, c % 2
                q = qres[c.to(dev), slot.to(dev)]
                new = F.normalize(q + noise.to(q), dim=-1)
                K = (getattr(self.keys, "k1" + sfx).data, getattr(self.keys, "k2" + sfx).data)
                Ke = (getattr(self, f"k1{sfx}_ema"), getattr(self, f"k2{sfx}_ema"))
                for hf in (0, 1):
                    s = half == hf
                    if s.any():
                        ls, hs, ist, sd = l[s].to(dev), h[s].to(dev), i[s].to(dev), s.to(dev)
                        K[hf][ls, hs, ist] = new[sd].to(K[hf])
                        Ke[hf][ls, hs, ist] = new[sd].to(Ke[hf])
                self._reset_rows(0, torch.unique(self._bands(l, half, i, S, off).flatten()).to(dev), full=True,
                                 zero=self.restart_zero)
                self._reset_keys(l, h, half, i, sfx)
                Uv[c.to(dev), i.to(dev)] = 1.0 / S
            win.zero_()
            st["dead" + sfx] = n
        self.stats["restart"] = st
        return st

    @torch.no_grad()
    def balance(self):
        """Every balance_every optimizer steps in infancy, per codebook, with the dedicated RNG: busy sub-keys split
        into the least-used key (the band through j gets a copy of the band through i: V rows and write counts;
        both bands' MemOpt rows reset to S0; the EMA copy gets the same edit); dead keys not used by a split are
        re-seeded from a recent query half (+ N(0, 0.05)) and their band's S is reset to S0. The private keys
        are balanced the same way per expert (the 'band' is one row).
        Batched, with the result of the one-key-at-a-time order: the same random draws in the same order (n draws
        of 64 normals == one (n, 64) draw), no key is edited twice in one event (plan_balance), and the band copies,
        which overlap (bands of both halves and of all heads of a level share rows), are chained on the host: the
        row map M (M[dst] = M[src] split by split) gives every row's final source, then one gather / scatter.
        Host syncs: two small transfers (qres.any, lastq_ok) per event."""
        gen, st = self.gen, dict(splits=0, dead=0, p_splits=0, p_dead=0)
        if self.S and not self.restart:  # (--mem-restart: the shared pools are re-seeded by restart_keys instead)
            L, H, S = self.L, self.H, self.S
            dev = self.V.device
            cap = max(1, int(self.balance_max * S)) if self.balance_max > 0 else 0
            dcap = max(1, int(self.balance_dead_max * S)) if self.balance_dead_max > 0 else 0
            keep = (self.win.view(-1, S) > 0).cpu() if self.balance_keep else None
            splits, deads = plan_balance(self.U.view(-1, S).float().cpu(), max_keys=cap, gen=gen, keep=keep, dead_max=dcap)
            K, Ke = (self.keys.k1.data, self.keys.k2.data), (self.k1_ema, self.k2_ema)
            Uv = self.U.view(-1, S)
            if len(splits):
                c, i, j = splits.unbind(1)
                l, h, half = c // (2 * H), (c // 2) % H, c % 2
                eps = torch.randn(len(c), 64, generator=gen) * 0.02
                for hf in (0, 1):
                    s = half == hf
                    if not s.any():
                        continue
                    ls, hs, ist, jst = (t[s].to(dev) for t in (l, h, i, j))
                    e = eps[s].to(K[hf])
                    for Kt in (K[hf], Ke[hf]):
                        ki = Kt[ls, hs, ist]
                        Kt[ls, hs, jst] = F.normalize(ki + e, dim=-1)
                        Kt[ls, hs, ist] = F.normalize(ki - e, dim=-1)
                src, dst = self._bands(l, half, i).numpy(), self._bands(l, half, j).numpy()
                M = np.arange(L * S * S)
                for t in range(len(c)):  # (host index arithmetic only)
                    M[dst[t]] = M[src[t]]
                rows = np.nonzero(M != np.arange(L * S * S))[0]
                rows, srcs = torch.from_numpy(rows).to(dev), torch.from_numpy(M[rows]).to(dev)
                for T in self._tables(0):
                    T.index_copy_(0, rows, T.index_select(0, srcs))
                self._reset_rows(0, torch.from_numpy(np.concatenate([src, dst]).ravel()).to(dev))
                self._reset_keys(torch.cat([l, l]), torch.cat([h, h]), torch.cat([half, half]), torch.cat([i, j]))
                cd, id_, jd = c.to(dev), i.to(dev), j.to(dev)
                u = Uv[cd, id_] / 2
                Uv[cd, jd] = u
                Uv[cd, id_] = u
            if len(deads):
                okq = self.qres.any(-1).view(-1, NRES).cpu().numpy()
                kept, noise = [], []
                for c, i in deads.tolist():  # randint, then (if that query half was recorded) 64 normals
                    r = int(torch.randint(NRES, (), generator=gen))
                    if okq[c, r]:
                        kept.append((c, i, r))
                        noise.append(torch.randn(64, generator=gen))
                if kept:
                    c, i, r = torch.tensor(kept).unbind(1)
                    l, h, half = c // (2 * H), (c // 2) % H, c % 2
                    q = self.qres.view(-1, NRES, 64)[c.to(dev), r.to(dev)]
                    new = F.normalize(q + (torch.stack(noise) * 0.05).to(q), dim=-1)
                    for hf in (0, 1):
                        s = half == hf
                        if s.any():
                            ls, hs, ist, sd = l[s].to(dev), h[s].to(dev), i[s].to(dev), s.to(dev)
                            K[hf][ls, hs, ist] = new[sd].to(K[hf])
                            Ke[hf][ls, hs, ist] = new[sd].to(Ke[hf])
                    self._reset_rows(0, torch.unique(self._bands(l, half, i).flatten()).to(dev), full=self.life)
                    self._reset_keys(l, h, half, i)
                    Uv[c.to(dev), i.to(dev)] = 1.0 / S
                    st["dead"] = len(kept)
            st["splits"] = len(splits)
            if self.balance_keep:
                self.win.zero_()
        if self.R:
            E, R = self._e_live, self.R
            dev = self.P.device
            cap = max(1, int(self.balance_max * R)) if self.balance_max > 0 else 0
            dcap = max(1, int(self.balance_dead_max * R)) if self.balance_dead_max > 0 else 0
            keep = (self.win_e[:E] > 0).cpu() if self.balance_keep else None
            splits, deads = plan_balance(self.U_e[:E].float().cpu(), max_keys=cap, gen=gen, keep=keep, dead_max=dcap)
            Pv = self.P
            if len(splits):
                e, i, j = (t.to(dev) for t in splits.unbind(1))
                eps = (torch.randn(len(e), 64, generator=gen) * 0.02).to(Pv)
                for Pt in (self.P, self.P_ema):
                    pi = Pt[e, i]
                    Pt[e, j] = F.normalize(pi + eps, dim=-1)
                    Pt[e, i] = F.normalize(pi - eps, dim=-1)
                ri, rj = e * R + i, e * R + j                                  # (every row in at most one split)
                for t in self._tables(1):
                    t.index_copy_(0, rj, t.index_select(0, ri))
                rij = torch.cat([ri, rj])
                self._reset_rows(1, rij)
                for t in (self.P_m, self.P_v, self.P_n):
                    t.index_fill_(0, rij, 0)
                u = self.U_e[e, i] / 2
                self.U_e[e, j] = u
                self.U_e[e, i] = u
            if len(deads):
                deads = deads[self.lastq_ok[:E].cpu()[deads[:, 0]]]
            if len(deads):
                e, i = (t.to(dev) for t in deads.unbind(1))
                new = F.normalize(self.lastq[e] + (torch.randn(len(e), 64, generator=gen) * 0.05).to(Pv), dim=-1)
                self.P[e, i] = new.to(Pv)
                self.P_ema[e, i] = new.to(Pv)
                ri = e * R + i
                self._reset_rows(1, ri, full=self.life, zero=self.life)   # (a private row: its key's alone)
                for t in (self.P_m, self.P_v, self.P_n):
                    t.index_fill_(0, ri, 0)
                self.U_e[e, i] = 1.0 / R
                st["p_dead"] = len(deads)
            st["p_splits"] = len(splits)
            if self.balance_keep:
                self.win_e.zero_()
        self.stats["balance"] = st
        return st

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def usage_stats(self):
        out = {}
        if self.S:
            mean = self.U.mean(-1, keepdim=True)
            out["subkeys_busy"] = (self.U > 4 * mean).float().mean().item()
            out["subkeys_dead"] = (self.U < 0.05 * mean).float().mean().item()
            wc = self.wcount[:self.L * self.S * self.S].view(self.L, -1).float()   # (the fine pool)
            out["rows_unwritten_by_level"] = (wc == 0).float().mean(1).tolist()
            qs = torch.tensor([0.1, 0.5, 0.9], device=wc.device)
            out["wcount_q10_50_90"] = torch.quantile(wc.flatten()[::max(1, wc.numel() // (1 << 22))], qs).tolist()
            n0 = self.L * self.S * self.S
            if self.Sc:
                mean = self.U_c.mean(-1, keepdim=True)
                out["subkeys_busy_c"] = (self.U_c > 4 * mean).float().mean().item()
                out["subkeys_dead_c"] = (self.U_c < 0.05 * mean).float().mean().item()
                out["rows_unwritten_c"] = (self.wcount[n0:] == 0).float().mean().item()
            if self.life:  # mean |V| (with the pending decay) per write-count bucket 1-10, 10-100, 100+
                for sfx, sl in (("", slice(0, n0)), ("_c", slice(n0, None))) if self.Sc else (("", slice(0, n0)),):
                    nrm = self.V[sl].norm(dim=1)
                    f = self._factors(torch.arange(nrm.numel(), device=nrm.device) + sl.start)
                    nrm = nrm if f is None else nrm * f
                    wcs = self.wcount[sl]
                    out["vnorm_by_wcount" + sfx] = [nrm[(wcs >= a) & (wcs < b)].mean().item() if ((wcs >= a) & (wcs < b)).any()
                                                    else float("nan") for a, b in ((1, 10), (10, 100), (100, 2 ** 31 - 1))]
        if self.R:
            E = self._e_live
            Ue = self.U_e[:E]
            mean = Ue.mean(-1, keepdim=True)
            out["p_keys_busy"] = (Ue > 4 * mean).float().mean().item()
            out["p_keys_dead"] = (Ue < 0.05 * mean).float().mean().item()
            out["p_rows_unwritten"] = (self.wcount_e.view(self.Emax, -1)[:E] == 0).float().mean().item()
            out["p_gate"] = torch.sigmoid(self.gates.b_e[:E]).mean().item()
        return out


# ---------------------------------------------------------------- the memory access gate (plan_v2/22)
GATE_FEATS = ("pbar", "nu", "dmu", "dthought", "surp", "trace", "ent", "pass")
GATE_MODES = ("learned", "random", "inverted", "oracle", "open", "off")
GATE_NTHETA = 11      # theta_k for k = 0 .. 9; reads k >= 10 share theta_10
GATE_STAT_RATE = 1e-3  # the EMA rate of the scalar feature statistics


def _orderable(s):
    """float32 (N,) -> int64 keys with the same order (-0.0 == +0.0): the sign-magnitude bits made monotone."""
    b = s.float().contiguous().view(torch.int32).to(torch.int64)
    return torch.where(b >= 0, b, -(b & 0x7FFFFFFF))


class MemGate(nn.Module):
    """The predictor router of the memory gate (plan_v2/22 revision 2): s = w . phi + b on DETACHED features phi
    (pbar standardised by the translator's norm, the concept uncertainty and change, the thought change, the last-token
    surprise and its trace, the next-token entropy, the pass one-hot), trained by BCE towards the detached mean content
    gate g_bar of the open positions (LoopBrain._hier_step). Selection (training): the top n_c positions by s (ties by a
    fixed random key) plus a uniform exploration floor of n_e positions drawn on the device; inference: s > theta_k, an
    EMA of the training top-k cut-off. Its parameters (weight, bias) are trained by their own AdamW (the trainer), never
    by BrainOpt / the global clip. Buffers: theta / theta_ok (the thresholds), fm / fv / fn (the scalar features' EMA
    statistics). The exploration generator lives on the device (host_state)."""

    def __init__(self, L, feats=GATE_FEATS, trace=8, thr_ema=0.02, seed=0):
        super().__init__()
        feats = tuple(f for f in GATE_FEATS if f in feats)
        assert feats, "--mem-gate-feat: no known feature"
        assert "trace" not in feats or "surp" in feats, "--mem-gate-feat: the trace needs surp"
        self.feats, self.L, self.trace, self.thr_ema, self.seed = feats, L, int(trace), float(thr_ema), int(seed)
        sc = []
        if "nu" in feats:
            sc += [f"nu{l}" for l in range(L)]
        if "dmu" in feats:
            sc += [f"dmu{l}" for l in range(L)]
        sc += [f for f in ("dthought", "surp", "trace", "ent") if f in feats]
        self.scalars = sc
        self.dim = 64 * ("pbar" in feats) + len(sc) + ("surp" in feats) + 16 * ("pass" in feats)
        g = torch.Generator().manual_seed((self.seed + 2222) % (2 ** 63))   # dedicated: the global stream is untouched
        self.weight = nn.Parameter(torch.randn(self.dim, generator=g) * 0.02)
        self.bias = nn.Parameter(torch.tensor(-2.0))      # sigma(-2) = 0.12, as the content gates' initial bias
        self.register_buffer("theta", torch.zeros(GATE_NTHETA))
        self.register_buffer("theta_ok", torch.zeros(GATE_NTHETA, dtype=torch.bool))
        self.register_buffer("fm", torch.zeros(len(sc)))
        self.register_buffer("fv", torch.ones(len(sc)))
        self.register_buffer("fn", torch.zeros((), dtype=torch.long))
        self._gen, self._keys = None, {}

    # ---- state
    def gen(self, device):
        """The exploration generator, on the device of the positions (created on first use, seeded)."""
        dev = torch.device(device)
        if self._gen is None or self._gen.device != dev:
            self._gen = torch.Generator(device=dev).manual_seed((self.seed + 3141) % (2 ** 63))
        return self._gen

    def host_state(self):
        return dict(gen=None if self._gen is None else self._gen.get_state(),
                    gen_device=None if self._gen is None else str(self._gen.device))

    def load_host_state(self, st):
        if st and st.get("gen") is not None:
            self.gen(st["gen_device"]).set_state(st["gen"].cpu())

    def tie_key(self, n, device):
        """A fixed random permutation of 0 .. n-1 (by n; deterministic from the seed): the tie-break among equal s."""
        key = (n, str(device))
        if key not in self._keys:
            g = torch.Generator().manual_seed((self.seed + 1618 + n) % (2 ** 63))
            self._keys[key] = torch.randperm(n, generator=g).to(device)
        return self._keys[key]

    # ---- the score
    def score(self, phi):
        """s = w . phi + b (N,), in the router's own precision (fp32; no autocast)."""
        with fp32_math(phi.device.type):
            return phi.to(self.weight.dtype) @ self.weight + self.bias

    def stats_update(self, x):
        """One training read's scalar features x (N, n_s): the EMA statistics (rate GATE_STAT_RATE; the first read sets
        them). Device-side, no host sync."""
        if not x.size(1):
            return
        with torch.no_grad():
            x = x.to(self.fm.dtype)
            m, v = x.mean(0), ((x - self.fm) ** 2).mean(0)
            first = self.fn == 0
            r = GATE_STAT_RATE
            self.fm.copy_(torch.where(first, m, (1 - r) * self.fm + r * m))
            self.fv.copy_(torch.where(first, x.var(0, unbiased=False), (1 - r) * self.fv + r * v))
            self.fn.add_(1)

    def standardize(self, x):
        return (x.to(self.fm.dtype) - self.fm) / torch.sqrt(self.fv + 1e-5)

    def stat_mean(self, name):
        return self.fm[self.scalars.index(name)]

    # ---- selection
    def thr(self, k):
        """theta_k (a 0-dim tensor), -inf before the first training cut-off (c = 1)."""
        i = min(max(k, 0), GATE_NTHETA - 1)
        return torch.where(self.theta_ok[i], self.theta[i], torch.full_like(self.theta[i], float("-inf")))

    @torch.no_grad()
    def select_train(self, s, n_c, n_e, gen=None):
        """Training selection of one read: A = the top n_c of s (ties: the fixed random key), E = n_e uniform draws from
        the rest (device-side), O = sort(A u E). Static sizes, no host sync. Returns O, isA (N,) bool, tau = the n_c-th
        largest s."""
        N, dev = s.numel(), s.device
        key = _orderable(s) * (1 << 32) + self.tie_key(N, dev)
        A = torch.topk(key, n_c, sorted=True).indices
        tau = s.float().index_select(0, A[-1:]).squeeze(0)
        isA = torch.zeros(N, dtype=torch.bool, device=dev).index_fill_(0, A, True)
        if n_e:
            u = torch.rand(N, generator=gen if gen is not None else self.gen(dev), device=dev)
            E = torch.topk(u.index_fill_(0, A, -1.0), n_e).indices
            O = torch.cat([A, E]).sort().values
        else:
            O = A.sort().values
        return O, isA, tau

    @torch.no_grad()
    def update_theta(self, k, tau):
        """theta_k <- (1 - rho) theta_k + rho tau (the first cut-off sets it). Device-side, no host sync."""
        i = min(max(k, 0), GATE_NTHETA - 1)
        t = tau.to(self.theta.dtype)
        self.theta[i:i + 1].copy_(torch.where(self.theta_ok[i], (1 - self.thr_ema) * self.theta[i] + self.thr_ema * t, t))
        self.theta_ok[i:i + 1].fill_(True)


def gate_sizes(c, explore, N, quant=64):
    """(n_c, n_e) of one training read: n_c = q(c N), n_e = q(explore N) once c < 1, n_c + n_e <= N; q = round to a
    multiple of quant (at least one quantum when the rate is > 0)."""
    if c >= 1:
        return N, 0
    q = lambda f: min(N, max(quant, int(round(f * N / quant)) * quant)) if f > 0 else 0
    n_c = max(1, min(N, q(c)))
    n_e = min(q(explore), N - n_c)
    return n_c, n_e


def gate_c(frac, c0=1.0, c1=0.5, s0=0.2, s1=0.5):
    """The open-rate schedule: c0 until s0 of the run, a cosine to c1 at s1, then c1."""
    if frac <= s0:
        return c0
    if frac >= s1:
        return c1
    return c1 + (c0 - c1) * 0.5 * (1 + math.cos(math.pi * (frac - s0) / max(s1 - s0, 1e-12)))


def gate_trace(sig, start, gamma, W):
    """tr_t = max over 0 <= j < W with no document start in (t - j, t] of gamma^j sig_{t-j} (B, T): the windowed form
    of tr_t = max(sig_t, gamma tr_{t-1}), reset at document starts (start (B, T) bool: t = 0 or idx[t-1] = EOT); equal
    to the recursion except for terms below gamma^W sig. One unfold, no loop over T."""
    B, T = sig.shape
    pad = lambda x, v: torch.cat([x.new_full((B, W - 1), v), x], 1)
    win = pad(sig, float("-inf")).unfold(1, W, 1)                        # (B, T, W): positions t - W + 1 .. t
    cs = start.long().cumsum(1)
    cw = pad(cs, -1).unfold(1, W, 1)                                     # cs at t - j (-1 before the row)
    j = torch.arange(W - 1, -1, -1, device=sig.device)                   # window column -> lag j
    ok = (cs[..., None] - cw == 0) | (j == 0)                            # no start in (t - j, t]
    ok = ok & (cw >= 0)
    w = gamma ** j.to(sig.dtype)
    return torch.where(ok, win * w, torch.full_like(win, float("-inf"))).amax(-1)


# ---------------------------------------------------------------- probes, guard, repair (5, 6)
class MemoryProbes:
    """Two disjoint fixed probe sets (A: repair, B: guard), each n_seq validation sequences x n_pos positions, drawn
    once with a dedicated RNG; run in fixed batches, eval mode, K = 6, head-group offset 0."""

    def __init__(self, val, seq, n_seq=32, n_pos=32, seed=0, K=6, batch=32, guard=0.95, repair=0.0,
                 repair_usage=0.1):
        rng = np.random.RandomState((seed + 97531) % (2 ** 32))
        n = len(val) - seq - 1
        starts = rng.choice(n, 2 * n_seq, replace=False) if n >= 2 * n_seq else rng.randint(0, n, 2 * n_seq)
        self.seq, self.K, self.batch = seq, K, batch
        self.guard, self.beta, self.repair_usage = guard, repair, repair_usage
        self.rng = np.random.RandomState((seed + 13579) % (2 ** 32))   # offsets for the overlap metric
        self.x, self.pos = {}, {}
        for s, name in ((0, "A"), (1, "B")):
            st = starts[s * n_seq:(s + 1) * n_seq]
            self.x[name] = torch.from_numpy(np.stack([val[i:i + seq].astype(np.int64) for i in st]))
            self.pos[name] = torch.from_numpy(np.stack([np.sort(rng.choice(seq, min(n_pos, seq), replace=False))
                                                        for _ in st]))

    def run(self, model, name, offset=None):
        """Records of every read (rows, w, gates, a, gaps) and core pass (experts, private rows) at the probes."""
        mem = model.mem
        dev = model.emb.weight.device
        was, K0, off0 = model.training, model.K, model.probe_offset
        model.eval()
        model.K = self.K
        model.probe_offset = offset
        x, pos = self.x[name], self.pos[name]
        recs = []
        try:
            for b0 in range(0, x.size(0), self.batch):
                xb, pb = x[b0:b0 + self.batch].to(dev), pos[b0:b0 + self.batch]
                flat = (torch.arange(pb.size(0))[:, None] * self.seq + pb).flatten().to(dev)
                mem.record = dict(pos=flat, reads=[], calls=[])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    model(xb)
                recs.append(mem.record)
        finally:
            mem.record = None
            model.K, model.probe_offset = K0, off0
            model.train(was)
        out = dict(reads=[], calls=[])
        for key in ("reads", "calls"):
            for i in range(len(recs[0][key])):
                r0 = recs[0][key][i]
                d = {}
                for f, v in r0.items():
                    if v is None or not torch.is_tensor(v):
                        d[f] = v
                    else:  # every record is (P, ...) except a (L, P, d + 64)
                        d[f] = torch.cat([r[key][i][f] for r in recs], 1 if f == "a" else 0)
                out[key].append(d)
        return out

    @staticmethod
    def _overlap(rows, w, ref):
        """Set overlap |R n R0| / k and the w-weighted overlap sum_{r in R n R0} w_r; (P, L, H) each."""
        inter = (rows[..., :, None] == ref[..., None, :]).any(-1)
        return inter.float().mean(-1), (w.float() * inter).sum(-1)

    def compare(self, cur, ref, prefix):
        out = {}
        reads = [(c, r) for c, r in zip(cur["reads"], ref["reads"]) if c["k"] == r["k"]]
        if not reads:
            return out
        st, sw, by_pass = [], [], []
        for c, r in reads:
            s, w = self._overlap(c["rows"], c["w"], r["rows"])
            st.append(s)
            sw.append(w)
            by_pass.append(w.mean().item())
        main = [i for i, (c, _) in enumerate(reads) if c["k"] >= 0] or list(range(len(reads)))
        out[f"{prefix}stab_w"] = torch.stack([sw[i] for i in main]).mean().item()
        out[f"{prefix}stab_set"] = torch.stack([st[i] for i in main]).mean().item()
        out[f"{prefix}stab_w_by_level"] = torch.stack([sw[i] for i in main]).mean((0, 1, 3)).tolist()
        out[f"{prefix}stab_w_by_pass"] = by_pass
        if any(reads[i][0].get("open") is not None for i in main):  # plan_v2/22: over the probes the gate opens
            ow = [sw[i][reads[i][0]["open"]] for i in main if reads[i][0].get("open") is not None]
            ow = torch.cat([w.flatten() for w in ow])
            out[f"{prefix}stab_w_open"] = ow.mean().item() if ow.numel() else float("nan")
            out[f"{prefix}probe_open_rate"] = torch.cat([reads[i][0]["open"].float() for i in main
                                                         if reads[i][0].get("open") is not None]).mean().item()
        if reads[0][0]["k"] < 0:
            out[f"{prefix}stab_w_pass0"] = by_pass[0]
        if reads[0][0].get("rows_c") is not None and reads[0][1].get("rows_c") is not None:  # the coarse pool
            ov = [self._overlap(c["rows_c"], c["w_c"], r["rows_c"]) for i, (c, r) in enumerate(reads) if i in main]
            out[f"{prefix}stab_w_c"] = torch.stack([w for _, w in ov]).mean().item()
            out[f"{prefix}stab_set_c"] = torch.stack([s for s, _ in ov]).mean().item()
            out[f"{prefix}stab_w_by_level_c"] = torch.stack([w for _, w in ov]).mean((0, 1, 3)).tolist()
        drift = [((c["a"] - r["a"]).norm(dim=-1) / r["a"].norm(dim=-1).clamp(min=1e-12)).mean(1)
                 for c, r in reads]                                                       # (L,) per read
        out[f"{prefix}input_drift_by_level"] = torch.stack(drift).mean(0).tolist()
        calls = list(zip(cur["calls"], ref["calls"]))
        if calls:
            out[f"{prefix}routing_same"] = sum((c["experts"] == r["experts"]).all(-1).float().mean().item()
                                               for c, r in calls) / len(calls)
            pr = [(c["prows"], r["prows"]) for c, r in calls if c.get("prows") is not None and r.get("prows") is not None]
            if pr:
                out[f"{prefix}private_stab"] = sum(((a[..., :, None] == b[..., None, :]).any(-1) & (a >= 0))
                                                   .float().sum().item() / max(1, (a >= 0).sum().item())
                                                   for a, b in pr) / len(pr)
        return out

    def measure(self, model):
        """All section-6 metrics; returns (metrics, alarm)."""
        mem = model.mem
        ps = mem.probe_state
        cur = {n: self.run(model, n) for n in ("A", "B")}
        out = dict(mem_frozen=int(mem.is_frozen))
        if mem.S:
            reads = [r for r in cur["B"]["reads"] if r.get("gap") is not None]
            if reads:  # set-boundary gaps (gamma, gamma1, gamma2): mean and 5% quantile
                gg = torch.stack([torch.cat([r[f].flatten() for r in reads]) for f in ("gap", "gap1", "gap2")]).float()
                out["mem_gap_mean"] = gg.mean(1).tolist()
                out["mem_gap_q05"] = torch.quantile(gg, 0.05, dim=1).tolist()
            ro = [(r["k"], r["reuse_ovl"].mean().item()) for r in cur["B"]["reads"] if r.get("reuse_ovl") is not None]
            if ro:  # --mem-readdr-every: per reused read (pass k), the set overlap with fresh addressing
                out["mem_reuse_overlap"] = {str(k): v for k, v in ro}
            out["mem_gate_by_level_pass"] = [r["g"].float().mean((0, 2)).tolist() for r in cur["B"]["reads"]
                                             if r.get("g") is not None]
            if mem.life:  # the gate means per pool (the life-cycle log: do the gates fall as the rows decay?)
                for sfx in ("", "_c"):
                    gs = [r["g" + sfx].float().mean() for r in cur["B"]["reads"] if r.get("g" + sfx) is not None]
                    if gs:
                        out["mem_gate_mean" + sfx] = torch.stack(gs).mean().item()
            coll = []  # collisions: distinct probe positions per addressed row (sets A and B, reads after a pass)
            sets = [(s, r["rows"]) for s, n in enumerate(("A", "B")) for r in cur[n]["reads"] if r["k"] >= 0]
            if sets:
                for l in range(mem.L):
                    rl = torch.cat([r[:, l].flatten() for _, r in sets])
                    pid = torch.cat([(s * 10 ** 6 + torch.arange(r.size(0)))[:, None, None].expand_as(r[:, l]).flatten()
                                     for s, r in sets])
                    pairs = torch.unique(torch.stack([rl, pid], 1), dim=0)
                    coll.append(pairs.size(0) / max(1, torch.unique(pairs[:, 0]).numel()))
                out["mem_collisions_by_level"] = coll
        if mem.R:
            pr = [c["prows"] for c in cur["B"]["calls"] if c.get("prows") is not None]
            if pr:
                wc = mem.wcount_e.cpu()
                hit = [(wc[p.clamp(min=0)] > 0)[p >= 0].float().mean().item() for p in pr]
                out["mem_private_hit"] = sum(hit) / len(hit)
        out.update({f"mem_{k}": v for k, v in mem.usage_stats().items()})
        if "last" in ps:
            out.update(self.compare(cur["B"], ps["last"], "mem_last_"))
        n_groups = model.blocks[0].n_groups
        if n_groups > 1 and mem.S:  # training uses a random head-group offset: the same context can address other rows
            alt = self.run(model, "B", offset=int(self.rng.randint(1, n_groups)))
            ov = [self._overlap(c["rows"], c["w"], r["rows"])[0].mean().item()
                  for c, r in zip(alt["reads"], cur["B"]["reads"])]
            out["mem_offset_overlap"] = sum(ov) / max(1, len(ov))
        alarm = False
        if mem.is_frozen and "ref" in ps:
            out.update(self.compare(cur["B"], ps["ref"]["B"], "mem_"))
            out.update({k.replace("mem_", "mem_A_", 1): v for k, v in self.compare(cur["A"], ps["ref"]["A"], "mem_").items()
                        if k in ("mem_stab_w", "mem_stab_set")})
            if "mem_stab_w" in out and out["mem_stab_w"] < self.guard:
                alarm = True
            if self.beta > 0 and mem.S:
                out["mem_repairs_by_level"] = self.repair(model, cur["A"], ps["ref"]["A"])
        out["mem_alarm"] = int(alarm)
        ps["last"] = cur["B"]
        return out, alarm

    def freeze(self, model):
        """At t_f: probes before, theta <- theta_bar + freeze, probes after (the freeze-time reference R0, never
        reset). Logs how much the copy changed the read weights w: on the FIRST read the rows are the same (theta_bar
        picks them both times, from the same input), so the L1 change of w there is the copy's own effect; later
        reads also move because the changed weights changed the thought (their set overlap is logged)."""
        mem = model.mem
        pre = self.run(model, "B")
        mem.freeze()
        ref = {n: self.run(model, n) for n in ("A", "B")}
        mem.probe_state["ref"] = ref
        info = {}
        pairs = [(a, b) for a, b in zip(pre["reads"], ref["B"]["reads"]) if a.get("rows") is not None]
        if pairs:
            a, b = pairs[0]
            info = dict(mem_freeze_dw_l1=(a["w"] - b["w"]).abs().sum(-1).mean().item(),
                        mem_freeze_rows_same=bool(torch.equal(a["rows"], b["rows"])),
                        mem_freeze_set_overlap=sum(self._overlap(a["rows"], a["w"], b["rows"])[0].mean().item()
                                                   for a, b in pairs) / len(pairs))
        mem.stats["freeze"] = info
        return info

    @torch.no_grad()
    def repair(self, model, cur, ref):
        """--mem-repair beta: for a probe of set A whose address set moved, each NEW row whose write count is below
        the repair-usage quantile gets V_r <- (1 - beta) V_r + beta V_drop (the dropped row with the largest w0).
        Busy rows are never overwritten; R0 is never reset. Returns the number of repaired rows per level."""
        mem = model.mem
        wc = mem.wcount.float()
        thr = float(torch.quantile(wc[::max(1, wc.numel() // (1 << 22))], self.repair_usage))
        pairs = {}
        for c, r in zip(cur["reads"], ref["reads"]):
            if c["k"] != r["k"]:
                continue
            rows, r0, w0 = c["rows"], r["rows"], r["w"].float()
            new = ~(rows[..., :, None] == r0[..., None, :]).any(-1)
            dropped = ~(r0[..., :, None] == rows[..., None, :]).any(-1)
            best = torch.where(dropped, w0, torch.full_like(w0, -1.0)).argmax(-1)          # (P, L, H)
            has = dropped.any(-1)
            rdrop = r0.gather(-1, best[..., None]).squeeze(-1)
            for idx in (new & has[..., None]).nonzero().tolist():
                p, l, h, j = idx
                row = int(rows[p, l, h, j])
                if row not in pairs and wc[row] < thr:
                    pairs[row] = int(rdrop[p, l, h])
        per_level = [0] * mem.L
        if pairs:
            dst = torch.tensor(list(pairs.keys()), device=mem.V.device)
            src = torch.tensor(list(pairs.values()), device=mem.V.device)
            mem.V[dst] = (1 - self.beta) * mem.V[dst] + self.beta * mem.V[src]
            for row in pairs:
                per_level[row // (mem.S * mem.S)] += 1
        return per_level


# ---------------------------------------------------------------- the life cycle (plan_v2/21 section 1.4)
class LifeCycle:
    """--mem-life: infancy -> ramp -> mature, driven by the trainer once per step (begin_step, at the top of the
    step) and after every probe evaluation (observe). Its state lives in mem.life_state (the memory's extra state:
    checkpoints and --resume carry it). The switch fires at the first of (a) the probe stability vs the last probe
    eval >= switch_stab on switch_n consecutive probe evals, (b) the learning-rate decay start, (c) the safety cap,
    and only once no more births can happen. Over the ramp (rho: 0 -> 1) the decay lambda and 1 - beta fall
    linearly to 0 and the restart interval grows from N to 10 N; at its end the trainer runs the freeze path."""

    def __init__(self, mem, steps, decay=1e-3, beta=0.99, switch_stab=0.9, switch_n=3, switch_max=0.9, lr_decay=0.2,
                 ramp=0.05):
        self.mem = mem
        mem.decay0, mem.beta0 = decay, beta
        self.decay, self.beta, self.stab, self.n = decay, beta, switch_stab, switch_n
        self.t_decay = (1 - lr_decay) * steps          # (b) the learning-rate decay start
        self.t_max = switch_max * steps                # (c) the safety cap
        self.ramp = max(1, round(ramp * steps))

    @property
    def state(self):
        return self.mem.life_state

    def observe(self, stab):
        """After a probe evaluation: the w-weighted row stability vs the last probe evaluation (mem_last_stab_w)."""
        st = self.state
        if st["phase"] == "infancy" and stab is not None:
            st["stab_run"] = st["stab_run"] + 1 if stab >= self.stab else 0

    def rho(self, step):
        st = self.state
        if st["phase"] == "infancy":
            return 0.0
        if st["phase"] == "mature":
            return 1.0
        return min(1.0, (step - st["t_switch"]) / self.ramp)

    def begin_step(self, step, births_done):
        """The switch check and this step's rates (lambda, beta, restart interval: used by after_step). Returns True
        when the ramp has ended (and no births are left): the trainer then runs the freeze path and calls mature()."""
        st = self.state
        if st["phase"] == "infancy":
            reason = ("stab" if st["stab_run"] >= self.n else "lr_decay" if step >= self.t_decay
                      else "max" if step >= self.t_max else None)
            if reason is not None and births_done:
                st.update(phase="ramp", t_switch=step, reason=reason)
        rho = self.rho(step)
        N = self.mem.restart
        if st["phase"] == "mature":
            st.update(decay=0.0, beta=1.0, restart_every=0)
        else:
            st.update(decay=(1 - rho) * self.decay, beta=1 - (1 - rho) * (1 - self.beta),
                      restart_every=round(N * (1 + 9 * rho)) if N else 0)
        return st["phase"] == "ramp" and rho >= 1.0 and births_done

    def mature(self):
        self.state.update(phase="mature", decay=0.0, beta=1.0, restart_every=0)

    def log(self):
        st = self.state
        return dict(phase=st["phase"], decay=st["decay"], beta=st["beta"], Lam=st["Lam"], reason=st["reason"],
                    t_switch=st["t_switch"], stab_run=st["stab_run"])
