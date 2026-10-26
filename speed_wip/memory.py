"""Memory system of the looped brain (plan_v2/12_memory_full_plan.md, revision 2). Every part is behind flags.

    ConceptHier    hierarchical concept heads on the workspace (section 1): targets, Gaussian NLL + masked variance
                   hinge, uncertainty u, feedback W_fb, rank-r write-back into the next pass's core input
    Memory         translators + product keys (2), shared pool read by MemRead (3), MemOpt (3.1), private expert
                   memory (4), sub-key balancing and the freeze (2.2), margin loss (2.3), copy-on-split
    MemoryProbes   drift probes, stability guard, repair (5, 6)

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
P_TOPK = 4      # private memory: rows read per (token, chosen expert)


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


TOPK_FAST = True  # stable_topk / pk_topk: top-k + small re-sort (False = the full stable sort of the reference code)


def _sort_topk(x, k):
    """The reference: a stable descending sort of the index-ordered rows of x (R, S), first k."""
    v, i = torch.sort(x, dim=-1, descending=True, stable=True)
    return v[:, :k], i[:, :k]


def _resort(v, i):
    """Order k (score, index) pairs (rows of (R, k)) by score descending, ties -> the lower index: by index first
    (indices are distinct), then stably by score."""
    o = i.argsort(dim=-1, stable=True)
    v, i = v.gather(-1, o), i.gather(-1, o)
    v, o = torch.sort(v, dim=-1, descending=True, stable=True)
    return v, i.gather(-1, o)


def _topk_rows(x, k, sync_free):
    """Exactly _sort_topk(x, k) without sorting the S scores: torch.topk(k + 1) (any tie order), then the k winners
    are re-sorted by (score desc, index asc). The k-set itself is ambiguous only when the k-th and the (k+1)-th
    scores tie (a tie AT the boundary); then the lowest-index equal scores must win:
      sync_free=False: those rows (found with one host sync) are recomputed with the full stable sort;
      sync_free=True:  every row replaces its boundary-valued entries (they sit at the end of the sorted top-k) by
                       the lowest-index entries equal to the k-th score (a second top-k over distinct integer keys),
                       no host sync (GPU).
    Finite scores assumed (NaN is not ordered like torch.sort orders it); -0.0 == +0.0 as in torch.sort."""
    R, S = x.shape
    if k + 1 > S:
        return _sort_topk(x, k)
    v, i = torch.topk(x, k + 1, dim=-1, largest=True, sorted=True)
    tie = v[:, k - 1] == v[:, k]
    v, i = v[:, :k], i[:, :k]
    if sync_free:
        thr = v[:, k - 1:k]
        gt = (v > thr).sum(1, keepdim=True)                       # entries strictly above the k-th score
        rev = torch.arange(S - 1, -1, -1, device=x.device, dtype=torch.int32)
        eq = torch.topk(torch.where(x == thr, rev, rev.new_tensor(-1)), k, dim=-1, sorted=True).indices
        p = torch.arange(k, device=x.device)
        i = torch.where(p < gt, i, eq.gather(1, (p - gt).clamp(min=0)))
        v = x.gather(1, i)
    elif bool(tie.any()):
        r = tie.nonzero().squeeze(1)
        fv, fi = _sort_topk(x[r], k)
        v, i = v.index_copy(0, r, fv), i.index_copy(0, r, fi)
    return _resort(v, i)


def stable_topk(s, k, max_elems=1 << 24, sync_free=None):
    """Top-k along the last dim, ties -> the LOWEST index (= a stable descending sort of the index-ordered scores),
    in chunks of rows so the temporaries stay bounded. TOPK_FAST: torch.topk + a re-sort of the k winners
    (_topk_rows; identical indices and values); sync_free None = on CUDA only."""
    flat = s.reshape(-1, s.size(-1))
    step = max(1, max_elems // max(1, s.size(-1)))
    if sync_free is None:
        sync_free = flat.is_cuda
    vs, ix = [], []
    for c in range(0, flat.size(0), step):
        v, i = _topk_rows(flat[c:c + step], k, sync_free) if TOPK_FAST else _sort_topk(flat[c:c + step], k)
        vs.append(v)
        ix.append(i)
    shape = tuple(s.shape[:-1]) + (k,)
    if len(vs) == 1:
        return vs[0].view(shape), ix[0].view(shape)
    return torch.cat(vs).view(shape), torch.cat(ix).view(shape)


def pk_topk(v1, i1, v2, i2, S, k):
    """Product-key top-k pairs from the per-half top-m lists (score v, index i; (..., m)). Ties -> the lowest flat
    index r = i * S + j. Reference: the candidates are sorted by r first (unique), then stably by score
    (descending). TOPK_FAST: each half's list is sorted by index (m entries), so the outer product is already in r
    order (r is lexicographic in (i, j)), then stable_topk. Returns (i, j, score) of the top k pairs, (..., k)."""
    if TOPK_FAST:
        i1, o = i1.sort(-1)
        v1 = v1.gather(-1, o)
        i2, o = i2.sort(-1)
        v2 = v2.gather(-1, o)
        sig = (v1[..., :, None] + v2[..., None, :]).flatten(-2)
        flat = (i1[..., :, None] * S + i2[..., None, :]).flatten(-2)
        v, o2 = stable_topk(sig, k)
        fl = flat.gather(-1, o2)
        return fl // S, fl % S, v
    sig = (v1[..., :, None] + v2[..., None, :]).flatten(-2)
    flat = (i1[..., :, None] * S + i2[..., None, :]).flatten(-2)
    o = flat.argsort(-1)
    sig, flat = sig.gather(-1, o), flat.gather(-1, o)
    v, o2 = torch.sort(sig, dim=-1, descending=True, stable=True)
    fl = flat.gather(-1, o2[..., :k])
    return fl // S, fl % S, v[..., :k]


def _addr_group(q1, q2, k1, k2, S, m, km):
    """Addressing of a group of levels (Memory._select_batched): q (l, H, N, 64), k (l, H, S, 64) normalised.
    The per-level score GEMMs are the reference's calls (q[l] @ k[l]^T), written in place into one buffer."""
    nl, H, N = q1.shape[:3]
    sc = q1.new_empty(2, nl, H, N, S)
    for l in range(nl):
        torch.matmul(q1[l], k1[l].transpose(-1, -2), out=sc[0, l])
        torch.matmul(q2[l], k2[l].transpose(-1, -2), out=sc[1, l])
    v, ix = stable_topk(sc, m + 1)                                           # (2, l, H, N, m + 1)
    del sc
    ii, jj, _ = pk_topk(v[0, ..., :m], ix[0, ..., :m], v[1, ..., :m], ix[1, ..., :m], S, km + 1)
    return ii, jj, ix[0], ix[1]


def _addr_group_pure(q1, q2, k1, k2, S, m, km):
    """_addr_group without out= (for torch.compile)."""
    sc = torch.stack([torch.stack([q1[l] @ k1[l].transpose(-1, -2) for l in range(q1.size(0))]),
                      torch.stack([q2[l] @ k2[l].transpose(-1, -2) for l in range(q2.size(0))])])
    v, ix = stable_topk(sc, m + 1)
    ii, jj, _ = pk_topk(v[0, ..., :m], ix[0, ..., :m], v[1, ..., :m], ix[1, ..., :m], S, km + 1)
    return ii, jj, ix[0], ix[1]


def _anchor(device, need):
    """A scalar leaf that requires grad, so autograd calls the custom backward (which writes the table's grad
    buffer) even when nothing else of the read needs a gradient (e.g. after the freeze, with sg addressing)."""
    return torch.ones((), device=device, requires_grad=bool(need))


class MemRead(torch.autograd.Function):
    """r[b] = sum_j w[b, j] V[rows[b, j]] (embedding_bag, mode sum). Backward: index_add_ into the ONE preallocated
    grad buffer G (no fresh table-sized allocation), and the gradient of the weights w. V gets no .grad."""

    @staticmethod
    def forward(ctx, anchor, w, V, G, rows):
        ctx.save_for_backward(rows, w)
        ctx.V, ctx.G = V, G
        return F.embedding_bag(rows, V, per_sample_weights=w.to(V.dtype), mode="sum")

    @staticmethod
    def backward(ctx, gout):
        rows, w = ctx.saved_tensors
        V, G = ctx.V, ctx.G
        g = gout.to(V.dtype)
        if G is not None:  # one column of the bag at a time: bounded temporaries
            for j in range(rows.size(1)):
                G.index_add_(0, rows[:, j], g * w[:, j:j + 1].to(G.dtype))
        dw = None
        if ctx.needs_input_grad[1]:
            dw = torch.stack([(g * F.embedding(rows[:, j], V)).sum(-1) for j in range(rows.size(1))], 1).to(w.dtype)
        return None, dw, None, None, None


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


def plan_balance(U, busy=4.0, dead=0.05):
    """Sub-key balancing plan for codebooks U (C, S) of usage shares (CPU). Busy (U > busy x mean) keys split into
    the least-used keys (busiest first); dead (U < dead x mean) keys not used by a split are re-seeded.
    Returns ([(c, i, j)], [(c, i)])."""
    splits, deads = [], []
    mean = U.mean(1)
    for c in range(U.size(0)):
        u = U[c]
        bi = (u > busy * mean[c]).nonzero().flatten()
        di = (u < dead * mean[c]).nonzero().flatten()
        if not len(bi) and not len(di):
            continue
        bi = bi[u[bi].argsort(descending=True, stable=True)].tolist()
        low = u.argsort(stable=True).tolist()
        used, lp = set(bi), 0
        for i in bi:
            while lp < len(low) and low[lp] in used:
                lp += 1
            if lp == len(low):
                break
            used.add(low[lp])
            splits.append((c, i, low[lp]))
        deads += [(c, i) for i in di.tolist() if i not in used]
    return splits, deads


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
                 addr_sg=True, margin=(0.01, 0.01), ema=0.999, seed=0):
        super().__init__()
        assert km <= m and km + 1 <= m * m, "product-key top-k is exact only for k <= m"
        assert not S or S >= m + 1, "each half needs at least m + 1 sub-keys (the margin uses the (m+1)-th)"
        assert not R or R >= P_TOPK
        self.d, self.ds, self.L, self.H, self.S, self.m, self.km = d, ds, L, H, S, m, km
        self.tau, self.dv, self.R, self.Emax = tau, dv, R, n_exp
        self.read0, self.addr_sg, self.ema = bool(read0), bool(addr_sg), ema
        self.lam_m, self.delta = float(margin[0]), float(margin[1])
        self.ain = d + ds
        # MemOpt and schedule (the trainer sets these)
        self.lr, self.s0_cfg, self.clip, self.balance_every, self.usage_rate = 0.01, "auto", 0.05, 250, 0.01
        self.write_on = True       # trainer: step >= --mem-write-start
        self.ckpt = True           # activation-checkpoint the private read (the LoopBrain block has its own flag)
        # speed (same math): batched addressing over levels / halves (score buffer <= select_budget elements per
        # group of levels), optional torch.compile of it (falls back to eager if it cannot compile), the
        # active-rows-only MemOpt update
        self.batch_select, self.select_budget, self.compile_addr = True, 1 << 26, False
        self.fused_adagrad = True
        self.ko_shared, self.ko_private, self.ko_levels = False, False, ()
        self.record = None         # MemoryProbes: dict the forward fills (eval only)
        self.probe_state = {}      # MemoryProbes' freeze-time / last-eval records (saved as extra state)
        self.gen = torch.Generator().manual_seed((seed + 4242) % (2 ** 63))   # dedicated RNG: balancing
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
            self.register_buffer("qres", torch.zeros(L, H, 2, NRES, 64), persistent=False)
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
            self.register_buffer("lastq", torch.zeros(E, 64), persistent=False)
            self.register_buffer("lastq_ok", torch.zeros(E, dtype=torch.bool), persistent=False)
            self.gates.b_e = nn.Parameter(torch.full((E,), -2.0))
            self.w_p = nn.Linear(dv, d, bias=False)
            nn.init.zeros_(self.w_p.weight)
        self._sync_host()

    # ------------------------------------------------------------ state
    def _sync_host(self):
        """Host mirrors of the small state buffers (reading a GPU buffer per forward would be a host sync)."""
        self._frozen = bool(self.frozen.item())
        self._step = int(self.n_step.item())
        self._s0 = [float(v) for v in self.s0.tolist()]
        self._e_live = int(self.e_live.item()) if self.R else 0

    @property
    def is_frozen(self):
        return self._frozen

    def get_extra_state(self):
        return {"probe_state": self.probe_state}

    def set_extra_state(self, state):
        self.probe_state = dict(state.get("probe_state", {})) if isinstance(state, dict) else {}

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._sync_host()
        if self._frozen:  # the freeze is part of the state: re-applied on reload
            self._apply_freeze()

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
        if self.R:
            self.P.copy_(self.P_ema)
        self.frozen.fill_(1)
        self._frozen = True
        self._apply_freeze()

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

    def _queries(self, a_hat, wq):
        """q = normalize(W_q a_hat) per level / head, halves q1 = normalize(q[:64]), q2 = normalize(q[64:]).
        Without a shared pool only level 1 head 1 is needed (the private query): (1, 1, N, 64)."""
        if self.S:
            q = torch.einsum("lnc,lhoc->lhno", a_hat, wq)
        else:
            q = (a_hat[0] @ wq[0, 0].t())[None, None]
        q = F.normalize(q, dim=-1)
        return F.normalize(q[..., :64], dim=-1), F.normalize(q[..., 64:], dim=-1)

    @torch.no_grad()
    def select(self, a, a_hat, collect):
        """Row indices from the EMA translator theta_bar (no gradient). Returns i, j (L, H, N, km + 1) top pairs, I1,
        I2 (L, H, N, m + 1) per-half top lists, rows (L, H, N, km) and qb1 (N, 64), the EMA private query."""
        L, H, S, m, km = self.L, self.H, self.S, self.m, self.km
        out = {}
        with fp32_math(a_hat.device.type):
            if collect and self._collect:  # EMA norm statistics (applied once per optimizer step)
                self.ns_sum += a.sum(1)
                self.ns_sq += ((a - self.norm_m[:, None]) ** 2).sum(1)
                self.ns_cnt += a.size(1)
            q1, q2 = self._queries(a_hat, self.wq_ema)
            out["qb1"] = q1[0, 0]
            if S:
                k1, k2 = F.normalize(self.k1_ema, dim=-1), F.normalize(self.k2_ema, dim=-1)
                if self.batch_select:
                    out["i"], out["j"], out["I1"], out["I2"] = self._select_batched(q1, q2, k1, k2)
                else:  # the reference: one level at a time
                    I, J, I1, I2 = [], [], [], []
                    for l in range(L):
                        v1, i1 = stable_topk(q1[l] @ k1[l].transpose(-1, -2), m + 1)   # (H, N, m + 1)
                        v2, i2 = stable_topk(q2[l] @ k2[l].transpose(-1, -2), m + 1)
                        ii, jj, _ = pk_topk(v1[..., :m], i1[..., :m], v2[..., :m], i2[..., :m], S, km + 1)
                        I.append(ii), J.append(jj), I1.append(i1), I2.append(i2)
                    out["i"], out["j"], out["I1"], out["I2"] = (torch.stack(t) for t in (I, J, I1, I2))
                off = (torch.arange(L, device=a.device) * S * S).view(L, 1, 1, 1)
                out["rows"] = off + out["i"][..., :km] * S + out["j"][..., :km]
                if collect and self._collect:  # sub-key usage (top-m selections) and recent query halves
                    sel = torch.stack((out["I1"][..., :m], out["I2"][..., :m]), 2).flatten(3)   # (L, H, 2, N m)
                    self.pend.scatter_add_(-1, sel, torch.ones_like(sel, dtype=self.pend.dtype))
                    pos = torch.linspace(0, a.size(1) - 1, NRES, device=a.device).long()
                    self.qres[:, :, 0] = q1[:, :, pos]
                    self.qres[:, :, 1] = q2[:, :, pos]
        return out

    def _select_batched(self, q1, q2, k1, k2):
        """select()'s addressing for all levels and both halves at once: the score GEMMs stay one call per level
        and half (the same kernels as the per-level code), written into one buffer; ONE stable_topk over every
        (half, level, head, token) row and ONE pk_topk. Levels are grouped so that a group's score buffer stays
        under select_budget elements (bounded temporaries). Returns i, j (L, H, N, km + 1), I1, I2 (L, H, N, m + 1)."""
        L, H, N = q1.shape[:3]
        S, m, km = self.S, self.m, self.km
        per = max(1, min(L, self.select_budget // max(1, 2 * H * N * S)))
        fn = _addr_group
        if self.compile_addr:
            fn = self._compiled_addr()
        outs = []
        for l0 in range(0, L, per):
            l1 = min(L, l0 + per)
            if fn is not _addr_group:
                try:
                    outs.append(fn(q1[l0:l1], q2[l0:l1], k1[l0:l1], k2[l0:l1], S, m, km))
                    continue
                except Exception as e:  # e.g. no C++ compiler (CPU) / no Triton (GPU): eager from now on
                    print(f"memory: torch.compile of the addressing failed ({type(e).__name__}: "
                          f"{str(e).splitlines()[0][:200]}); using eager", flush=True)
                    self.compile_addr, fn = False, _addr_group
            outs.append(_addr_group(q1[l0:l1], q2[l0:l1], k1[l0:l1], k2[l0:l1], S, m, km))
        if len(outs) == 1:
            return outs[0]
        return tuple(torch.cat(t) for t in zip(*outs))

    def _compiled_addr(self):
        fn = getattr(Memory, "_addr_compiled", None)
        if fn is None:
            fn = Memory._addr_compiled = torch.compile(_addr_group_pure, dynamic=True)
        return fn

    def _key_scores(self, q, K, idx):
        """Cosines of the queries q (L, H, N, 64) with the normalised sub-keys K (L, H, S, 64) at idx (L, H, N, k)."""
        L, H, S = K.shape[:3]
        base = torch.arange(L * H, device=idx.device).view(L, H, 1, 1) * S
        rows = F.embedding(base + idx, F.normalize(K, dim=-1).flatten(0, 2))
        return (q[..., None, :] * rows).sum(-1)

    def online(self, a_hat, sel):
        """Weights from the ONLINE translator on the EMA-chosen pairs: w = softmax(sigma(theta) / tau) (L, H, N, km);
        sigma also for the (km+1)-th pair; q1 = the online private query (N, 64). Gradient reaches W_q, K only here."""
        with fp32_math(a_hat.device.type):
            q1, q2 = self._queries(a_hat, self.keys.wq)
            out = {"q1": q1[0, 0]}
            if self.S:
                sig = self._key_scores(q1, self.keys.k1, sel["i"]) + self._key_scores(q2, self.keys.k2, sel["j"])
                out["sig"] = sig
                out["w"] = torch.softmax(sig[..., :self.km] / self.tau, -1)
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
        """Infancy only, on sg(a): lambda * mean[relu(delta - g) + 1/2 relu(delta - g1) + 1/2 relu(delta - g2)]."""
        g, g1, g2 = self.gaps(a_hat_sg.detach(), sel)
        dl = self.delta
        loss = self.lam_m * (F.relu(dl - g) + 0.5 * F.relu(dl - g1) + 0.5 * F.relu(dl - g2)).mean()
        with torch.no_grad():
            self.margin_stats = dict(active=(g < dl).float().mean(), active_half=((g1 < dl).float().mean()
                                                                                  + (g2 < dl).float().mean()) / 2,
                                     gap=g.mean(), gap_half=(g1.mean() + g2.mean()) / 2)
        return loss

    # ------------------------------------------------------------ shared read (3)
    def read_shared(self, w, sel, pbar):
        """One fused MemRead for all L x H heads, gates and W_o. Returns M (N, d) and the gates (N, L, H)."""
        L, H, km, dv = self.L, self.H, self.km, self.dv
        N = w.size(2)
        rows = sel["rows"].permute(2, 0, 1, 3).reshape(N * L * H, km)
        ww = w.permute(2, 0, 1, 3).reshape(N * L * H, km)
        write = torch.is_grad_enabled() and self.write_on
        with fp32_math(pbar.device.type):
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
        return M, g

    # ------------------------------------------------------------ private expert memory (4)
    def private(self, q1, qb1, e_k, p_k, t_k, g_k, fpos, N, k, E, C):
        """Called inside RoutedMLP with its capacity layout (kept slots: expert e_k, slot p_k, token t_k, gate g_k,
        flat (token, choice) position fpos). Indices from the EMA side (qb1 with P_bar, bmm over the capacity
        layout, top-4, stable tie-break), weights from the online side. Returns W_p(sum_e gate sigma(b_e) rho)."""
        R, dev = self.R, q1.device
        dt = self.P.dtype
        collect = self._collect and torch.is_grad_enabled()      # a training pass inside the gradient window
        with torch.no_grad(), fp32_math(dev.type):
            Q = qb1.new_zeros(E, C, 64)
            Q[e_k, p_k] = qb1[t_k]
            sc = torch.bmm(Q, F.normalize(self.P_ema[:E], dim=-1).transpose(1, 2))   # (E, C, R)
            idx = stable_topk(sc[e_k, p_k], P_TOPK)[1]       # the kept slots only (empty slots are never read)
            rows = e_k[:, None] * R + idx
            del sc
            if collect:
                self.pend_e.view(-1).index_add_(0, rows.flatten(),
                                                torch.ones(rows.numel(), dtype=self.pend_e.dtype, device=dev))
                has = torch.zeros(E, dtype=torch.bool, device=dev)
                has[e_k] = True
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
                rho = MemRead.apply(_anchor(dev, write), w, self.V_e, self.G_e if write else None, rows)
                z = rho.new_zeros(N, self.dv).index_add_(
                    0, t_k, rho * (g_k.to(dt) * torch.sigmoid(b_e[e_k].to(dt)))[:, None])
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
            for t in (self.V_e, self.S_e, self.wcount_e, self.P_m, self.P_v, self.P_n):
                tv = t.view(self.Emax, R, *t.shape[1:])
                tv[dst] = tv[src]
        self.e_live.fill_(E0 + n)
        self._e_live = E0 + n
        if self._frozen:  # (births should end before the freeze; if not, the keys stay frozen)
            self._apply_freeze()

    # ------------------------------------------------------------ MemOpt (3.1) and the per-step updates
    @torch.no_grad()
    def _row_adagrad(self, V, G, S_acc, wcount, which, chunk=1 << 17):
        """Sparse row-Adagrad with a prior: for rows with G_r != 0: S_r += mean_i G_ri^2 (S starts at S0),
        dV = -lr G_r / sqrt(S_r), clipped to |dV| <= clip; V_r += dV. Other rows stay bit-identical."""
        if self._s0[which] < 0:  # S0: absolute, or auto = 100 x median over the written rows (first write step)
            act = (G != 0).any(1)
            if not bool(act.any()):
                return
            if self.s0_cfg == "auto":  # (per-row means first: no table-sized temporary)
                s0 = 100.0 * float((G * G).mean(1)[act].median())
            else:
                s0 = float(self.s0_cfg)
            S_acc.fill_(s0)
            self.s0[which] = s0
            self._s0[which] = s0
        lr, clip = self.lr, self.clip
        if self.fused_adagrad:
            return self._row_adagrad_rows(V, G, S_acc, wcount, chunk)
        for c in range(0, V.size(0), chunk):
            g = G[c:c + chunk]
            act = (g != 0).any(1)
            s = S_acc[c:c + chunk]
            s.copy_(torch.where(act, s + (g * g).mean(1), s))
            dv = -lr * g / s.sqrt()[:, None]
            nrm = dv.norm(dim=1, keepdim=True)
            dv = dv * (clip / nrm.clamp(min=1e-30)).clamp(max=1.0)
            v = V[c:c + chunk]
            v.copy_(torch.where(act[:, None], v + dv, v))
            wcount[c:c + chunk] += act.to(wcount.dtype)
        G.zero_()

    @torch.no_grad()
    def _row_adagrad_rows(self, V, G, S_acc, wcount, chunk):
        """_row_adagrad on the touched rows only, bit-identical: ONE sweep finds the rows of G with any non-zero BIT
        pattern (a superset of the active rows that also catches -0.0 entries, so zeroing only these rows leaves G
        exactly as G.zero_() would); they are gathered in chunks padded to the reference's chunk shape (same row
        length, same number of rows per kernel call, so every per-row reduction runs the same kernel), updated with
        the reference's formulas (where(act, ...) per row) and scattered back. Then G is zeroed on those rows."""
        lr, clip = self.lr, self.clip
        n = V.size(0)
        ib = {8: torch.int64, 4: torch.int32, 2: torch.int16}[G.element_size()]
        idx = (G.view(ib) != 0).any(1).nonzero().squeeze(1)
        nt = idx.numel()                                      # (one host sync, once per optimizer step)
        if not nt:
            return
        rows = min(chunk, n)                                  # the reference's chunk height (all chunks full when
        for c in range(0, nt, rows):                          # the table is a multiple of it, as in the real sizes)
            ii = idx[c:c + rows]
            k = ii.numel()
            g = G.new_zeros(rows, G.size(1))
            g[:k] = G.index_select(0, ii)
            s = S_acc.new_ones(rows)                          # (padding rows: inactive, discarded)
            s[:k] = S_acc.index_select(0, ii)
            act = (g != 0).any(1)
            s = torch.where(act, s + (g * g).mean(1), s)
            dv = -lr * g / s.sqrt()[:, None]
            nrm = dv.norm(dim=1, keepdim=True)
            dv = dv * (clip / nrm.clamp(min=1e-30)).clamp(max=1.0)
            v = V.new_zeros(rows, V.size(1))
            v[:k] = V.index_select(0, ii)
            v = torch.where(act[:, None], v + dv, v)
            V.index_copy_(0, ii, v[:k])
            S_acc.index_copy_(0, ii, s[:k])
            wcount.index_add_(0, ii, act[:k].to(wcount.dtype))
        G.index_fill_(0, idx, 0)

    @torch.no_grad()
    def _adam_keys(self, lr, b1=0.9, b2=0.95, eps=1e-8):
        """Private keys P: Adam, no weight decay, lazy per row (rows without gradient untouched)."""
        G = self.G_P
        act = (G != 0).any(1)
        self.P_n.add_(act.to(self.P_n.dtype))
        a = act[:, None]
        self.P_m.copy_(torch.where(a, b1 * self.P_m + (1 - b1) * G, self.P_m))
        self.P_v.copy_(torch.where(a, b2 * self.P_v + (1 - b2) * G * G, self.P_v))
        n = self.P_n.clamp(min=1)[:, None]
        upd = (self.P_m / (1 - b1 ** n)) / ((self.P_v / (1 - b2 ** n)).sqrt() + eps)
        P = self.P.view(-1, 64)
        P.copy_(torch.where(a, P - lr * upd, P))
        G.zero_()

    @torch.no_grad()
    def after_step(self, lr):
        """Once per optimizer step, after BrainOpt. Infancy: EMA norm statistics, EMA translator / private keys,
        private-key Adam (lr = the trainer's current rate), usage EMAs, balancing every balance_every steps.
        Always: the table writes (after --mem-write-start)."""
        self._step += 1
        self.n_step.fill_(self._step)
        ema = self.ema
        if not self._frozen:
            cnt = self.ns_cnt.clamp(min=1)
            has = self.ns_cnt > 0
            self.norm_m.copy_(torch.where(has, ema * self.norm_m + (1 - ema) * self.ns_sum / cnt, self.norm_m))
            self.norm_v.copy_(torch.where(has, ema * self.norm_v + (1 - ema) * self.ns_sq / cnt, self.norm_v))
            self.wq_ema.lerp_(self.keys.wq.detach(), 1 - ema)
            if self.S:
                self.k1_ema.lerp_(self.keys.k1.detach(), 1 - ema)
                self.k2_ema.lerp_(self.keys.k2.detach(), 1 - ema)
                tot = self.pend.sum(-1, keepdim=True)
                self.U.copy_(torch.where(tot > 0, (1 - self.usage_rate) * self.U + self.usage_rate * self.pend
                                         / tot.clamp(min=1), self.U))
                self.pend.zero_()
            if self.R:
                self._adam_keys(lr)
                self.P_ema.lerp_(self.P, 1 - ema)
                tot = self.pend_e.sum(-1, keepdim=True)                          # experts without traffic: kept
                self.U_e.copy_(torch.where(tot > 0, (1 - self.usage_rate) * self.U_e + self.usage_rate * self.pend_e
                                           / tot.clamp(min=1), self.U_e))
                self.pend_e.zero_()
            if self.balance_every and self._step % self.balance_every == 0:
                self.balance()
        else:
            if self.S:
                self.pend.zero_()
            if self.R:
                self.G_P.zero_()
                self.pend_e.zero_()
        if self.S:
            if self.write_on:
                self._row_adagrad(self.V, self.G, self.S_acc, self.wcount, 0)
            else:
                self.G.zero_()
        if self.R:
            if self.write_on:
                self._row_adagrad(self.V_e, self.G_e, self.S_e, self.wcount_e, 1)
            else:
                self.G_e.zero_()

    # ------------------------------------------------------------ sub-key balancing (2.2)
    def band(self, l, half, i):
        """Rows addressed through sub-key i of half `half` at level l: half 1 = rows i*S .. i*S + S - 1, half 2 =
        the strided column i (a*S + i)."""
        S = self.S
        a = torch.arange(S, device=self.V.device)
        return l * S * S + (i * S + a if half == 0 else a * S + i)

    def _reset_s(self, S_acc, rows, which):
        if self._s0[which] > 0:
            S_acc[rows] = self._s0[which]

    @torch.no_grad()
    def balance(self):
        """Every balance_every optimizer steps in infancy, per codebook, with the dedicated RNG: busy sub-keys split
        into the least-used key (the band through j gets a copy of the band through i: V rows and write counts;
        both bands' MemOpt rows reset to S0; the EMA copy gets the same edit); dead keys not used by a split are
        re-seeded from a recent query half (+ N(0, 0.05)) and their band's S is reset to S0. The private keys
        are balanced the same way per expert (the 'band' is one row)."""
        gen, st = self.gen, dict(splits=0, dead=0, p_splits=0, p_dead=0)
        if self.S:
            L, H, S = self.L, self.H, self.S
            splits, deads = plan_balance(self.U.view(-1, S).float().cpu())
            K, Ke = (self.keys.k1.data, self.keys.k2.data), (self.k1_ema, self.k2_ema)
            for c, i, j in splits:
                l, h, half = c // (2 * H), (c // 2) % H, c % 2
                eps = (torch.randn(64, generator=gen) * 0.02).to(K[half])
                for Kt in (K[half], Ke[half]):
                    ki = Kt[l, h, i].clone()
                    Kt[l, h, j] = F.normalize(ki + eps, dim=-1)
                    Kt[l, h, i] = F.normalize(ki - eps, dim=-1)
                src, dst = self.band(l, half, i), self.band(l, half, j)
                self.V[dst] = self.V[src]
                self.wcount[dst] = self.wcount[src]
                self.S_acc[dst] = self.S_acc[src]
                self._reset_s(self.S_acc, torch.cat([src, dst]), 0)
                if self.opt_reset is not None:  # the split key and its twin start with fresh optimizer moments
                    self.opt_reset(f"mem.keys.k{half + 1}", (l, h, [i, j]))
                u = self.U[l, h, half]
                u[j] = u[i] = u[i] / 2
            for c, i in deads:
                l, h, half = c // (2 * H), (c // 2) % H, c % 2
                q = self.qres[l, h, half, int(torch.randint(NRES, (), generator=gen))]
                if not bool(q.any()):
                    continue
                new = F.normalize(q + (torch.randn(64, generator=gen) * 0.05).to(q), dim=-1)
                K[half][l, h, i] = new.to(K[half])
                Ke[half][l, h, i] = new.to(Ke[half])
                self._reset_s(self.S_acc, self.band(l, half, i), 0)
                if self.opt_reset is not None:
                    self.opt_reset(f"mem.keys.k{half + 1}", (l, h, [i]))
                self.U[l, h, half, i] = 1.0 / S
                st["dead"] += 1
            st["splits"] = len(splits)
        if self.R:
            E, R = self._e_live, self.R
            splits, deads = plan_balance(self.U_e[:E].float().cpu())
            Pv = self.P
            for e, i, j in splits:
                eps = (torch.randn(64, generator=gen) * 0.02).to(Pv)
                for Pt in (self.P, self.P_ema):
                    pi = Pt[e, i].clone()
                    Pt[e, j] = F.normalize(pi + eps, dim=-1)
                    Pt[e, i] = F.normalize(pi - eps, dim=-1)
                ri, rj = e * R + i, e * R + j
                self.V_e[rj] = self.V_e[ri]
                self.wcount_e[rj] = self.wcount_e[ri]
                self.S_e[rj] = self.S_e[ri]
                self._reset_s(self.S_e, torch.tensor([ri, rj], device=Pv.device), 1)
                for t in (self.P_m, self.P_v, self.P_n):
                    t[ri] = 0
                    t[rj] = 0
                self.U_e[e, j] = self.U_e[e, i] = self.U_e[e, i] / 2
            for e, i in deads:
                if not bool(self.lastq_ok[e]):
                    continue
                new = F.normalize(self.lastq[e] + (torch.randn(64, generator=gen) * 0.05).to(Pv), dim=-1)
                self.P[e, i] = new.to(Pv)
                self.P_ema[e, i] = new.to(Pv)
                ri = e * R + i
                self._reset_s(self.S_e, torch.tensor([ri], device=Pv.device), 1)
                for t in (self.P_m, self.P_v, self.P_n):
                    t[ri] = 0
                self.U_e[e, i] = 1.0 / R
                st["p_dead"] += 1
            st["p_splits"] = len(splits)
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
            wc = self.wcount.view(self.L, -1).float()
            out["rows_unwritten_by_level"] = (wc == 0).float().mean(1).tolist()
            qs = torch.tensor([0.1, 0.5, 0.9], device=wc.device)
            out["wcount_q10_50_90"] = torch.quantile(wc.flatten()[::max(1, wc.numel() // (1 << 22))], qs).tolist()
        if self.R:
            E = self._e_live
            Ue = self.U_e[:E]
            mean = Ue.mean(-1, keepdim=True)
            out["p_keys_busy"] = (Ue > 4 * mean).float().mean().item()
            out["p_keys_dead"] = (Ue < 0.05 * mean).float().mean().item()
            out["p_rows_unwritten"] = (self.wcount_e.view(self.Emax, -1)[:E] == 0).float().mean().item()
            out["p_gate"] = torch.sigmoid(self.gates.b_e[:E]).mean().item()
        return out


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
        if reads[0][0]["k"] < 0:
            out[f"{prefix}stab_w_pass0"] = by_pass[0]
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
            out["mem_gate_by_level_pass"] = [r["g"].float().mean((0, 2)).tolist() for r in cur["B"]["reads"]
                                             if r.get("g") is not None]
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
