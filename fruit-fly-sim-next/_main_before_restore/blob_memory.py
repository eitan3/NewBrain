"""Global + per-expert blob memory (plan_v2/25d_global_expert_blob_plan.md + the external review; STABLE parts only).

Built here:
  * storage (all BUFFERS, never parameters): ONE global space (default 262,144 slots) and one PRIVATE space per expert
    (default 512 slots): space e = routed expert e (0 .. E-1), the LAST space = the always-on shared expert
    (preallocated for `max_experts` routed experts + 1; past that the family grows by concatenation and the shared space
    moves to the new last index). Per slot: key (64-d fp32) + bf16 scoring copy + |k|^2 (derived), content at model
    width d (global fp32; private bf16 with stochastic rounding, or fp32), the content optimizer's state (ONE
    second-moment scalar per row + the row's update count), strength (= EMA of the positive benefit; policy later),
    birth step, key time stamp, value-fade / strength clocks, touch step, use count, merge count, maturity, occupied
    flag; global slots also hold their two preferred IVF cells and the assigned cell. The PROBATION store of the write
    path ("keep what comes back": probation, then promotion; flag, default off): a FIFO ring sized rate x life
    (~3,000 steps) of (64-d key, 64-d projected gradient, entry step, hits, last hit, source), with exact / IVF lookup
    over the global cells and accessors for promotion (get / take / mark_hit / drop / age); no policy yet.
  * the fixed address map: x_hat = ((xf - mu) / sigma) @ R; mu / sigma per channel from the fit window (sigma floored
    at 1e-3 x its median), R (d x 64) i.i.d. N(0, 1/d) from a fixed seed on a forked generator. Gathered while
    `collect` is on (the trainer: the --gm-fit-window steps before --gm-fit-step), then fit() sets mu, sigma and the IVF
    cells ONCE (the cells on a mix of query samples and write-candidate samples, observe(xf, kind)). Before the fit
    nothing is searched or read (the map is the identity then: tests / the write stub only).
  * search, no gradient, no host sync; top `cand` (16) candidates per query (for later merge / renewal):
      global "ivf" (default): k-means cells (1,024), each padded to cell_cap (512) slots (a static table; a full
      cell spills to the slot's second-nearest cell; what still does not fit is counted), refit hook refit() (warm
      Lloyd on recent query / write samples + the stored keys, keys re-assigned; the trainer calls it every
      --gm-ivf-refit steps) and set_centroids(); queries probe their `probe` (16) nearest cells, laid out per cell with
      a static capacity q_cap (pairs past it dropped, farthest probes first); one bf16 bmm on RESIDUALS to the cell
      centroid [cells, q_cap, 64] x [cells, 64, cell_cap] into a preallocated workspace; bf16 top-`cell_topk` per
      (token, probe) row (default 32 = 2 cand: with 8, recall@16 fell to 0.60 on clustered keys, whose neighbours
      share a cell; 16 -> 0.92, 32 -> 0.997); the best 2 x cand by bf16 score re-ranked in fp32 -> top-cand.
      "brute": chunked bf16 scores over every slot (top-8 per row minimum) + fp32 re-rank: the reference for tests
      ONLY (at 256K slots it builds a multi-GB score matrix per call).
      private: one bf16 bmm on RoutedMLP's capacity layout [E, C, 64] x [E, 64, P] into a workspace -> bf16 top
      2 x cand -> fp32 re-rank -> top-cand per (token, expert) pair; the shared expert's space for every token.
  * read + inject: the first k_g (8) global / k_p (4) private candidates. Match weights
    w_j = softmax_j(-(d_j - d_1) / ((d_1 + c h0^2) tau)) (bounded as d_1 -> 0), A = exp(-d_1 / (2 h0^2)) (h0 per space
    type: buffers with an update hook, set_h0(); policy later), u = sum_j w_j v_j as ONE F.embedding_bag(mode="sum",
    per_sample_weights=w) (BagRead: nothing of size (n, k, d) is saved for backward); no content fade. Global +
    private: r = max(A_g, A_p) (pi_g u_g + pi_p u_p), pi = A / (A_g + A_p); optionally times a learnable per-channel
    injection scale (flag inj_scale, init 1: the only memory PARAMETER, trained by the brain's optimizer). r_g / u_g
    once per token, the private part per (token, expert) pair; buf[e_k, p_k] += mask * r (out-of-place index_put)
    before w_up; the shared expert gets x + mask_s * r_shared. The read mask is pluggable (mask_fn; default: the
    placeholder modes all / none / random p); an optional static compute buffer (compute_frac, e.g. 0.55 of the pair
    slots) compacts the open pairs before the private read. Gradient: contents into DENSE fp32 grad buffers (+ touched-
    row masks), through w and A into the query xf (match_grad); keys and indices detached.
  * static sizes: every buffer is sized from tensor shapes (N, the pair list's length, E, C of buf); pair_valid (an
    optional mask over the pair list) lets a static N k pair list carry padding rows.
  * recompute safety: with a call id (LoopBrain passes the pass index) the first run's selection (candidates + masks)
    is cached and reused when activation checkpointing recomputes the forward.
  * the content update: lazy row-wise RMSprop on the TOUCHED rows only - v <- b2 v + (1 - b2) mean_c g^2 once per step
    (b2 0.9), u = lr g / (sqrt(v) + eps) (lr 0.03), per-row norm cap, no momentum, no weight decay; untouched rows (value
    AND state) bit-identical. Private bf16 contents rounded stochastically.
  * births: a split child gets a copy of its parent's private space at half strength with a fresh content-optimizer
    state; ramp children start empty. Nothing global changes.
  * per-token running sum / count of xf over one forward's passes (detached, flag track_xf): the future write key.
  * utilities for the later write path: reset_opt_state(space, flat), cap_norm(v, ref_norm, cap); after_step() is the
    step-end hook (content update now; write / merge / evict decisions with at most one host sync later).

Not here (review pending): the gate and its update, write selection / content (write() is a placeholder for tests),
dedupe / merge, spawn / eviction, confirmation / key moves, fades, h0 calibration / tracking.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

READ_MODES = ("all", "none", "random")
SEARCHES = ("ivf", "brute")
INF = float("inf")


def stochastic_bf16(x, gen):
    """fp32 -> bf16 rounded up or down with probability proportional to the remainder (unbiased): random low 16 bits are
    added to the fp32 bit pattern, then truncated. A value that is already a bf16 number is unchanged."""
    x = x.float().contiguous()
    r = torch.randint(0, 1 << 16, x.shape, generator=gen, device=x.device, dtype=torch.int32)
    out = ((x.view(torch.int32) + r) & -65536).view(torch.float32)
    return torch.where(torch.isfinite(x), out, x).to(torch.bfloat16)


def cap_norm(v, ref_norm, cap):
    """Scale each row of v (n, d) down to at most cap x ref_norm (n,) (e.g. 0.3 |xf| at a write, 0.5 |xf| after)."""
    lim = (cap * ref_norm.to(v.dtype))[..., None]
    return v * (lim / v.norm(dim=-1, keepdim=True).clamp(min=1e-30)).clamp(max=1.0)


class BlobSpace(nn.Module):
    """The slots of one family of spaces: lead = (G,) global, (S, P) private. Storage dtypes are fixed (a model-wide
    .double() does not change them, see _apply)."""

    FLOAT = ("key", "opt_v", "strength", "path")
    LONG = ("opt_n", "birth", "t_key", "clock_v", "clock_s", "touch", "uses", "merges", "mature", "t_conf",
            "win_start", "moves_win", "t_mature")

    def __init__(self, lead, d, kd=64, value_dtype=torch.float32, cells=False):
        super().__init__()
        self.d, self.kd, self.value_dtype, self.has_cells = d, kd, value_dtype, cells
        z = lambda *s, dt=torch.float32: torch.zeros(*lead, *s, dtype=dt)
        self.register_buffer("key", z(kd))
        self.register_buffer("value", z(d, dt=value_dtype))
        self.register_buffer("opt_v", z())
        self.register_buffer("strength", z())
        self.register_buffer("path", z())                    # summed key travel (moves + merges)
        for n in self.LONG:
            self.register_buffer(n, z(dt=torch.long))
        self.register_buffer("occ", z(dt=torch.bool))
        if cells:  # IVF: nearest / second-nearest cell, and the assigned cell (-1 = none / did not fit)
            self.register_buffer("cell_pref", torch.full((*lead, 2), -1, dtype=torch.long))
            self.register_buffer("cell", torch.full(tuple(lead), -1, dtype=torch.long))
            self.register_buffer("spawn_key", z(kd))              # (travel logging: the key at spawn)
        self.register_buffer("key_bf", z(kd, dt=torch.bfloat16), persistent=False)
        self.register_buffer("ksq", torch.full(tuple(lead), INF), persistent=False)

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for n in self.FLOAT:
            setattr(self, n, getattr(self, n).float())
        self.value = self.value.to(self.value_dtype)
        if self.has_cells:
            self.spawn_key = self.spawn_key.float()
        self.key_bf = self.key_bf.to(torch.bfloat16)
        self.ksq = self.ksq.float()
        return self

    def persistent_names(self):
        return self.FLOAT + ("value",) + self.LONG + ("occ",) + (("cell_pref", "cell", "spawn_key") if self.has_cells
                                                                 else ())

    @torch.no_grad()
    def refresh(self):
        self.key_bf = self.key.to(torch.bfloat16)
        self.ksq = torch.where(self.occ, self.key.pow(2).sum(-1), torch.full_like(self.ksq, INF))

    @torch.no_grad()
    def grow(self, n_lead0):
        """Grow the first dimension to n_lead0 (private spaces); the new spaces are empty."""
        S = self.occ.size(0)
        if n_lead0 <= S:
            return
        for n in self.persistent_names() + ("key_bf",):
            t = getattr(self, n)
            fill = -1 if n.startswith("cell") else 0
            setattr(self, n, torch.cat([t, t.new_full((n_lead0 - S, *t.shape[1:]), fill)]))
        self.ksq = torch.cat([self.ksq, self.ksq.new_full((n_lead0 - S, *self.ksq.shape[1:]), INF)])

    @torch.no_grad()
    def copy_spaces(self, src, dst):
        for n in self.persistent_names() + ("key_bf", "ksq"):
            t = getattr(self, n)
            t.index_copy_(0, dst, t.index_select(0, src))

    @torch.no_grad()
    def clear_spaces(self, idx):
        for n in self.persistent_names() + ("key_bf",):
            getattr(self, n).index_fill_(0, idx, 0)
        self.ksq.index_fill_(0, idx, INF)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "occ"  # a checkpoint with more private spaces (grown past the preallocation): resize first
        if k in state_dict and state_dict[k].dim() == self.occ.dim() and state_dict[k].size(0) > self.occ.size(0):
            self.grow(state_dict[k].size(0))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.refresh()


def nearest_cells(k, centroids):
    """The nearest and second-nearest cells of keys k (n, kd) -> (n, 2) (the same cell twice with one cell)."""
    c = centroids
    csq = c.pow(2).sum(-1)[None]
    chunk = 65536  # (in row chunks: the probation store has millions of keys, a full (n, cells) matrix would OOM)
    pref = torch.cat([((kk.pow(2).sum(-1)[:, None] - 2 * kk @ c.t()) + csq).topk(
        min(2, c.size(0)), -1, largest=False).indices for kk in k.split(chunk)]) if k.size(0) else \
        torch.zeros(0, min(2, c.size(0)), dtype=torch.long, device=k.device)
    return torch.cat([pref, pref], 1) if pref.size(1) == 1 else pref


def _cell_positions(cid, prio, n_cells):
    """cid (n,) with the sentinel n_cells for 'none' -> (order, sorted cell ids, position within the cell), sorted by
    (cell, prio, index); counts by index_add_ (no sync)."""
    order = (cid * 2 + prio).argsort(stable=True)
    cs = cid[order]
    cnt = torch.zeros(n_cells + 1, dtype=torch.long, device=cid.device).index_add_(0, cs, torch.ones_like(cs))
    pos = torch.arange(cid.size(0), device=cid.device) - (cnt.cumsum(0) - cnt)[cs]
    return order, cs, pos


@torch.no_grad()
def ivf_table(key, occ, pref, centroids, cell_cap):
    """A static IVF table [cells, cell_cap] (entry ids, -1 = padding) over key (n, kd) / occ (n,) with the preferred
    cells pref (n, 2): each occupied entry goes to its nearest cell; one past its cell's capacity spills to its
    second-nearest cell (after that cell's own entries); what still does not fit is left out (counted). Returns dict
    table, kr (bf16 residuals k - c per table entry), ksq (bf16 |k - c|^2, +inf = padding), cell (n,) the assigned cell
    (-1 = none), overflow, spilled (device scalars). No host sync."""
    Cn, Cc, n = centroids.size(0), cell_cap, key.size(0)
    dev = key.device
    none = torch.full((n,), Cn, dtype=torch.long, device=dev)
    c1 = torch.where(occ & (pref[:, 0] >= 0), pref[:, 0], none)
    order, cs, pos = _cell_positions(c1, torch.zeros_like(c1), Cn)
    over = torch.zeros(n, dtype=torch.bool, device=dev).index_copy_(0, order, (cs < Cn) & (pos >= Cc))
    cid = torch.where(over, pref[:, 1].clamp(min=0), c1)
    order, cs, pos = _cell_positions(cid, over.long(), Cn)
    keep = (cs < Cn) & (pos < Cc)
    dest = torch.where(keep, cs * Cc + pos, torch.full_like(pos, Cn * Cc))
    tab = torch.full((Cn * Cc + 1,), -1, dtype=torch.long, device=dev)
    tab.index_copy_(0, dest, torch.where(keep, order, torch.full_like(order, -1)))
    table = tab[:Cn * Cc].view(Cn, Cc)
    cell = torch.full((n,), -1, dtype=torch.long, device=dev).index_copy_(
        0, order, torch.where(keep, cs, torch.full_like(cs, -1)))
    kr = key.float()[table.clamp(min=0)] - centroids[:, None, :]
    ksq = torch.where(table >= 0, kr.pow(2).sum(-1), torch.full_like(kr[..., 0], INF))
    return dict(table=table, kr=kr.to(torch.bfloat16), ksq=ksq.to(torch.bfloat16), cell=cell,
                overflow=((cs < Cn) & (pos >= Cc)).sum(), spilled=(over & (cell >= 0)).sum())


@torch.no_grad()
def ivf_candidates(q, centroids, table, kr, ksq, probe, q_cap, cell_topk, M, workspace=None, pos=None):
    """IVF shortlist: q (N, kd) probes its `probe` nearest cells; the (query, probe) pairs are laid out per cell with a
    static capacity q_cap (farthest probes dropped first); one bf16 bmm on residuals to the cell centroid
    [cells, q_cap, kd] x [cells, kd, cell_cap]; bf16 top-cell_topk per pair; the best M per query by bf16 score.
    Returns (cand (N, M) entry ids, -1 = none; the probe-overflow rate). workspace(name, shape, dtype): optional static
    buffers. pos (N,): order each cell's queries by token position first (causal: a query's capacity never depends
    on a later position; the default orders by probe rank). No host sync."""
    Cn, Cc, kd = table.size(0), table.size(1), q.size(1)
    n_p = min(probe, Cn)
    N, dev = q.size(0), q.device
    ws = workspace or (lambda name, shape, dt: torch.empty(shape, dtype=dt, device=dev))
    c = centroids
    pc = (c.pow(2).sum(-1)[None] - 2 * q @ c.t()).topk(n_p, -1, largest=False).indices   # (N, n_p), nearest first
    cellf = pc.flatten()
    if pos is None:
        order = (cellf * n_p + torch.arange(n_p, device=dev).repeat(N)).argsort(stable=True)  # by (cell, probe rank)
    else:
        pmax = int(N) + 1
        key_ = (cellf * pmax + pos.repeat_interleave(n_p)) * n_p + torch.arange(n_p, device=dev).repeat(N)
        order = key_.argsort(stable=True)                                  # by (cell, position, probe rank)
    cs = cellf[order]
    cnt = torch.zeros(Cn, dtype=torch.long, device=dev).index_add_(0, cs, torch.ones_like(cs))
    pos = torch.arange(N * n_p, device=dev) - (cnt.cumsum(0) - cnt)[cs]
    if q_cap < 0:   # tiled layout (no overflow): each cell's pairs in tiles of |q_cap| rows, static tile count
        return _ivf_tiled(q, c, table, kr, ksq, -q_cap, min(cell_topk, Cc), M, ws, order, cs, cnt, pos, n_p)
    keep = pos < q_cap
    dummy = Cn * q_cap
    dest = torch.where(keep, cs * q_cap + pos, torch.full_like(pos, dummy))
    Qb = ws("ivf_q", (dummy + 1, kd), torch.bfloat16)
    Qb.zero_()
    res = q[order // n_p] - c[cs]                                     # residuals to the probed cell
    Qb.index_copy_(0, dest, res.to(torch.bfloat16))
    S = ws("ivf_s", (Cn, q_cap, Cc), torch.bfloat16)
    torch.bmm(Qb[:dummy].view(Cn, q_cap, kd), kr.transpose(1, 2), out=S)
    S.mul_(-2).add_(ksq[:, None, :])                                  # |k - c|^2 - 2 (q - c).(k - c)
    kk = min(cell_topk, Cc)
    top = S.topk(kk, -1, largest=False)
    slots = table.gather(1, top.indices.flatten(1)).view(-1, kk)
    sc = top.values.float().view(-1, kk)
    rows = torch.cat([slots, slots.new_full((1, kk), -1)]).index_select(0, dest)
    rsc = torch.cat([sc, sc.new_full((1, kk), INF)]).index_select(0, dest) + res.pow(2).sum(-1)[:, None]
    cand = torch.empty_like(rows).index_copy_(0, order, rows).view(N, n_p * kk)
    csc = torch.empty_like(rsc).index_copy_(0, order, rsc).view(N, n_p * kk)
    csc = torch.where(cand >= 0, csc, torch.full_like(csc, INF))
    bt = csc.topk(min(M, cand.size(1)), -1, largest=False)
    best = torch.where(bt.values < INF, cand.gather(1, bt.indices), torch.full_like(bt.indices, -1))
    return best, (~keep).float().mean()


@torch.no_grad()
def _ivf_tiled(q, c, table, kr, ksq, Tq, kk, M, ws, order, cs, cnt, pos, n_p):
    """ivf_candidates with --gm-qcap -Tq: the (query, probe) pairs, sorted by cell, fill ceil(cnt_c / Tq) tiles of Tq
    rows per cell; tiles = ceil(N n_p / Tq) + cells (a static bound); one bf16 bmm [tiles, Tq, kd] x [tiles, kd,
    cell_cap] against each tile's cell. Same scores and shortlist as an unbounded q_cap (no pair is dropped)."""
    Cn, Cc, kd = table.size(0), table.size(1), q.size(1)
    N, dev = q.size(0), q.device
    n_t = -(-(N * n_p) // Tq) + Cn
    nt_c = (cnt + Tq - 1) // Tq                                        # tiles per cell
    t0 = nt_c.cumsum(0) - nt_c                                         # first tile of each cell
    dest = (t0[cs] + pos // Tq) * Tq + pos % Tq
    dummy = n_t * Tq
    tcell = torch.searchsorted(nt_c.cumsum(0), torch.arange(n_t, device=dev), right=True).clamp(max=Cn - 1)
    Qb = ws("ivf_qt", (dummy + 1, kd), torch.bfloat16)
    Qb.zero_()
    res = q[order // n_p] - c[cs]
    Qb.index_copy_(0, dest, res.to(torch.bfloat16))
    S = ws("ivf_st", (n_t, Tq, Cc), torch.bfloat16)
    torch.bmm(Qb[:dummy].view(n_t, Tq, kd), kr.index_select(0, tcell).transpose(1, 2), out=S)
    S.mul_(-2).add_(ksq.index_select(0, tcell)[:, None, :])
    top = S.topk(kk, -1, largest=False)
    slots = table.index_select(0, tcell).gather(1, top.indices.flatten(1)).view(-1, kk)
    sc = top.values.float().view(-1, kk)
    rows = torch.cat([slots, slots.new_full((1, kk), -1)]).index_select(0, dest)
    rsc = torch.cat([sc, sc.new_full((1, kk), INF)]).index_select(0, dest) + res.pow(2).sum(-1)[:, None]
    cand = torch.empty_like(rows).index_copy_(0, order, rows).view(N, n_p * kk)
    csc = torch.empty_like(rsc).index_copy_(0, order, rsc).view(N, n_p * kk)
    csc = torch.where(cand >= 0, csc, torch.full_like(csc, INF))
    bt = csc.topk(min(M, cand.size(1)), -1, largest=False)
    best = torch.where(bt.values < INF, cand.gather(1, bt.indices), torch.full_like(bt.indices, -1))
    return best, torch.zeros((), device=dev)


@torch.no_grad()
def exact_candidates(q, key_bf, ksq, M, chunk=1024, key_chunk=262144):
    """Brute-force shortlist: half-precision scores |k|^2 - 2 q.k over every entry, in query chunks x key blocks (a
    running top-M across the blocks) -> the best M ids (N, M)."""
    n = key_bf.size(0)
    M = min(M, n)
    out = []
    step = chunk if chunk and chunk > 0 else max(q.size(0), 1)
    for a in range(0, q.size(0), step):
        qa = q[a:a + step].to(key_bf.dtype)
        best_s = best_i = None
        for b in range(0, n, key_chunk):
            s = ksq[None, b:b + key_chunk] - 2 * (qa @ key_bf[b:b + key_chunk].t()).float()
            t = s.topk(min(M, s.size(1)), -1, largest=False)
            ti = t.indices + b
            if best_s is None:
                best_s, best_i = t.values, ti
            else:
                cs, ci = torch.cat([best_s, t.values], 1), torch.cat([best_i, ti], 1)
                tt = cs.topk(M, -1, largest=False)
                best_s, best_i = tt.values, ci.gather(1, tt.indices)
        out.append(best_i)
    return torch.cat(out) if out else q.new_zeros(0, M, dtype=torch.long)


@torch.no_grad()
def rerank(q, key, ok, cand, k):
    """Exact fp32 distances of candidates cand (N, M) (-1 = none) in a key table (n, kd) with ok (n,) -> top-k (ids,
    d2 ascending; +inf = none)."""
    safe = cand.clamp(min=0)
    d2 = (q[:, None, :] - key[safe].float()).pow(2).sum(-1)
    d2 = torch.where((cand >= 0) & ok[safe], d2, torch.full_like(d2, INF))
    top = d2.topk(min(k, d2.size(1)), -1, largest=False)
    ids = safe.gather(1, top.indices)
    return torch.where(torch.isfinite(top.values), ids, torch.full_like(ids, -1)), top.values   # (r6 C2: -1 = none)


class ProbationStore(nn.Module):
    """The probation store of the write path ("keep what comes back": probation, then promotion; flag-gated). A FIFO
    ring of n entries (capacity flag; default ceil(rate x life), rate = candidates per optimizer step, life ~ 3,000
    steps; static allocation). Per entry ONLY: a kd-d key (the first kd coordinates of x_hat: 64 default, 32 option),
    a kd-d gradient sketch, the step stamp (int32), the return count (int32) and an occupied flag (+ its preferred IVF
    cells, an index). No dataset / source position, window index or fact id (the owner's no-pointer rule); the entry's
    surprise is kept only in the optional logging-only buffer `log_sig` (log=True; off by default).
    Storage dtype: "fp16" (default) or "int8" with a per-entry fp16 scale (dequantised to fp32 for distances / cosines).
    Lookup: exact (a half-precision shortlist over key blocks + an fp32 re-rank) or IVF over the global space's cells
    (rebuild() after pushes / a refit). Accessors for promotion: get / take (read and free) / mark_hit / drop / age.
    No policy (the thresholds live in blob_write.step_writes). Not built yet: the split-storage option (keys on the GPU,
    sketches in pinned CPU RAM, the cosine check one step later)."""

    def __init__(self, n, kd=64, cells=1024, cell_cap=0, dtype="fp16", log=False, index=True):
        super().__init__()
        assert dtype in ("fp16", "int8")
        self.n, self.kd, self.qdtype = n, kd, dtype
        self.cell_cap = cell_cap or max(16, math.ceil(2 * n / max(cells, 1)))
        if not index:  # (exact lookup only: the IVF table is never read; a 1 x 1 placeholder instead of ~1.4 GB)
            cells, self.cell_cap = 1, 1
        sd = torch.float16 if dtype == "fp16" else torch.int8
        self.register_buffer("key", torch.zeros(n, kd, dtype=sd))
        self.register_buffer("pgrad", torch.zeros(n, kd, dtype=sd))
        if dtype == "int8":
            # (r6 C1: fp32 scales; an fp16 scale of a ~1e-6 row underflowed to 0)
            self.register_buffer("key_scale", torch.zeros(n, dtype=torch.float32))
            self.register_buffer("pgrad_scale", torch.zeros(n, dtype=torch.float32))
        self.register_buffer("step", torch.zeros(n, dtype=torch.int32))
        self.register_buffer("hits", torch.zeros(n, dtype=torch.int32))
        self.register_buffer("occ", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("lost", torch.zeros(n, dtype=torch.int8))      # flag: lost a promotion budget once
        self.register_buffer("pref", torch.full((n, 2), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))       # entries pushed so far (FIFO position)
        self.log_sig = torch.zeros(n, dtype=torch.float16) if log else None  # (logging only; not saved)
        self.register_buffer("ksq", torch.full((n,), INF), persistent=False)
        self.register_buffer("table", torch.full((cells, self.cell_cap), -1, dtype=torch.long), persistent=False)
        self.register_buffer("table_kr", torch.zeros(cells, self.cell_cap, kd, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("table_ksq", torch.full((cells, self.cell_cap), INF, dtype=torch.bfloat16),
                             persistent=False)
        self._ptr = 0                   # host mirror of ptr (static FIFO positions, no sync)
        self.stats = {}

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        sd = torch.float16 if self.qdtype == "fp16" else torch.int8
        self.key, self.pgrad = self.key.to(sd), self.pgrad.to(sd)
        if self.qdtype == "int8":
            self.key_scale, self.pgrad_scale = self.key_scale.float(), self.pgrad_scale.float()
        self.ksq = self.ksq.float()
        self.table_kr, self.table_ksq = self.table_kr.to(torch.bfloat16), self.table_ksq.to(torch.bfloat16)
        if self.log_sig is not None:
            self.log_sig = self.log_sig.to(self.key.device)
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._ptr = int(state_dict.get(prefix + "ptr", self.ptr))
        self.refresh()

    # ---------------------------------------------------------------- (de)quantisation
    def _q(self, x):
        """fp32 rows -> (stored rows, per-row scale or None)."""
        if self.qdtype == "fp16":
            return x.to(torch.float16), None
        s = x.abs().amax(-1).clamp(min=1e-12) / 127.0
        return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8), s.float()

    def keys_f(self, ids=None):
        """Dequantised keys (fp32) of the given entries (all when None)."""
        k = self.key if ids is None else self.key[ids]
        if self.qdtype == "fp16":
            return k.float()
        s = self.key_scale if ids is None else self.key_scale[ids]
        return k.float() * s.float()[..., None]

    def sketch_f(self, ids):
        g = self.pgrad[ids]
        if self.qdtype == "fp16":
            return g.float()
        return g.float() * self.pgrad_scale[ids].float()[..., None]

    @torch.no_grad()
    def refresh(self):
        self.ksq = torch.where(self.occ, self.keys_f().pow(2).sum(-1), torch.full_like(self.ksq, INF))

    @torch.no_grad()
    def push(self, key, pgrad, step, centroids=None, sig=None):
        """Append n candidates FIFO (the oldest entries are overwritten). Returns their entry ids. centroids: assign
        their preferred IVF cells (rebuild() makes them visible to the IVF lookup; the exact lookup sees them at once)."""
        n = key.size(0)
        dev = self.key.device
        slots = (torch.arange(n, device=dev) + self._ptr) % self.n
        self._ptr += n
        self.ptr.fill_(self._ptr)
        kq, ks = self._q(key[:, :self.kd].float())
        gq, gs = self._q(pgrad[:, :self.kd].float())
        self.key.index_copy_(0, slots, kq)
        self.pgrad.index_copy_(0, slots, gq)
        if self.qdtype == "int8":
            self.key_scale.index_copy_(0, slots, ks)
            self.pgrad_scale.index_copy_(0, slots, gs)
        self.step.index_copy_(0, slots, torch.as_tensor(step, device=dev).to(torch.int32).expand(n).contiguous())
        self.hits.index_fill_(0, slots, 0)
        self.lost.index_fill_(0, slots, 0)
        self.occ.index_fill_(0, slots, True)
        if sig is not None and self.log_sig is not None:
            self.log_sig.index_copy_(0, slots, sig.to(torch.float16))
        kf = self.keys_f(slots)
        self.ksq.index_copy_(0, slots, kf.pow(2).sum(-1))
        if centroids is not None:
            self.pref.index_copy_(0, slots, nearest_cells(kf, centroids[:, :self.kd]))
        return slots

    @torch.no_grad()
    def rebuild(self, centroids, reassign=True):
        """The IVF table over the occupied entries (after pushes, or a centroid refit: reassign recomputes every
        entry's preferred cells)."""
        centroids = centroids[:, :self.kd]
        kf = self.keys_f()
        if reassign:
            self.pref.copy_(nearest_cells(kf, centroids))
        t = ivf_table(kf, self.occ, self.pref, centroids, self.cell_cap)
        self.table, self.table_kr, self.table_ksq = t["table"], t["kr"], t["ksq"]
        self.stats = dict(overflow=t["overflow"], spilled=t["spilled"])

    @torch.no_grad()
    def _exact(self, q, M, chunk=1024, key_chunk=262144, group=64, groups=32):
        """The exact shortlist, fast (same ids as _exact_ref up to ties / fp16 near-ties): only the written range
        [0, min(ptr, n)) of the FIFO ring (host mirror, no sync). Stage 1: per key block, one fp16 matmul of the
        augmented [q, 1] x [-2 k, |k|^2] (approximate scores, fp16), the minimum per group of `group` entries; the
        `groups` best groups per query over the whole store. Stage 2: the reference score |k|^2 - 2 fp16(q.k) of
        the entries of those groups only, top-M. No (Q, n) fp32 matrix, no full top-k over the store."""
        n = min(self._ptr, self.n)
        M = min(M, self.n)
        dev = q.device
        if n == 0:
            return torch.zeros((q.size(0), M), dtype=torch.long, device=dev)
        kc = max(group, key_chunk // group * group)
        nfull, last = n // kc, n % kc
        lp = -(-last // group) * group                                         # the last block, padded
        gpb = kc // group                                                      # groups per full block
        ng = nfull * gpb + lp // group
        G = min(groups, ng)
        out = []
        step = chunk if chunk and chunk > 0 else max(q.size(0), 1)
        for a in range(0, q.size(0), step):
            qa = q[a:a + step].to(torch.float16)
            Qn = qa.size(0)
            qx = torch.cat([qa, qa.new_ones(Qn, 1), qa.new_zeros(Qn, 7)], 1)     # [q, 1, 0] (K = 72)
            gmin = torch.empty(Qn, ng, dtype=torch.float16, device=dev)
            for b in range(0, n, kc):
                e = min(n, b + kc)
                kb = self.key[b:e].to(torch.float16)
                if self.qdtype == "int8":
                    kb = (kb.float() * self.key_scale[b:e, None]).to(torch.float16)
                kx = torch.cat([-2 * kb, self.ksq[b:e, None].clamp(max=6e4).to(torch.float16),
                                kb.new_zeros(e - b, 7)], 1)
                s = qx @ kx.t()                                                 # (Qn, e - b) fp16
                pad = (-(e - b)) % group
                if pad:
                    s = torch.cat([s, s.new_full((Qn, pad), float("inf"))], 1)
                # group j of a block = its entries j, j + L/group, ... (strided: the fast reduction direction)
                gmin[:, b // group:b // group + s.size(1) // group] = s.view(Qn, group, -1).amin(1)
            gi = gmin.float().topk(G, -1, largest=False).indices                   # (Qn, G)
            blk = gi // gpb
            j = gi - blk * gpb
            stride = torch.where(blk < nfull, torch.full_like(gi, gpb), torch.full_like(gi, max(lp // group, 1)))
            ids = (blk * kc + j)[:, :, None] + stride[:, :, None] * torch.arange(group, device=dev)
            ids = ids.view(Qn, -1)
            ok = ids < n
            ids = ids.clamp(max=n - 1)
            kb = self.key[ids].to(torch.float16)
            if self.qdtype == "int8":
                kb = (kb.float() * self.key_scale[ids][..., None]).to(torch.float16)
            dot = torch.bmm(kb, qa[:, :, None]).squeeze(-1)                      # fp16 q.k (as the reference)
            sc = torch.where(ok, self.ksq[ids] - 2 * dot.float(), torch.full_like(dot, INF, dtype=torch.float32))
            t = sc.topk(min(M, sc.size(1)), -1, largest=False)
            out.append(ids.gather(1, t.indices))
        return torch.cat(out) if out else q.new_zeros(0, M, dtype=torch.long)

    @torch.no_grad()
    def _exact_ref(self, q, M, chunk=1024, key_chunk=65536):   # (1024 x 65536 fp32 scores = 268 MB transient)
        """The reference: half-precision scores over every key block (dequantised per block), a running top-M."""
        n = self.n
        M = min(M, n)
        out = []
        step = chunk if chunk and chunk > 0 else max(q.size(0), 1)
        for a in range(0, q.size(0), step):
            qa = q[a:a + step].to(torch.float16)
            best_s = best_i = None
            for b in range(0, n, key_chunk):
                kb = self.keys_f(torch.arange(b, min(n, b + key_chunk), device=q.device)).to(torch.float16)
                s = self.ksq[None, b:b + key_chunk] - 2 * (qa @ kb.t()).float()
                t = s.topk(min(M, s.size(1)), -1, largest=False)
                ti = t.indices + b
                if best_s is None:
                    best_s, best_i = t.values, ti
                else:
                    cs, ci = torch.cat([best_s, t.values], 1), torch.cat([best_i, ti], 1)
                    tt = cs.topk(M, -1, largest=False)
                    best_s, best_i = tt.values, ci.gather(1, tt.indices)
            out.append(best_i)
        return torch.cat(out) if out else q.new_zeros(0, M, dtype=torch.long)

    @torch.no_grad()
    def lookup(self, q, k, mode="exact", centroids=None, probe=16, q_cap=192, cell_topk=32, chunk=1024):
        """The k nearest occupied entries of x_hat queries q (N, kd) -> (ids (N, k), d2 (N, k) ascending; +inf =
        none). mode "exact": a half-precision shortlist over every entry; "ivf": over the table of the last rebuild()."""
        q = q.float()[:, :self.kd]
        M = max(2 * k, 8)
        if mode == "ivf":
            cand, ov = ivf_candidates(q, centroids[:, :self.kd], self.table, self.table_kr, self.table_ksq, probe, q_cap,
                                      cell_topk, M)
            self.stats["probe_overflow"] = ov
        else:
            cand = self._exact(q, M, chunk)
        safe = cand.clamp(min=0)
        d2 = (q[:, None, :] - self.keys_f(safe)).pow(2).sum(-1)
        d2 = torch.where((cand >= 0) & self.occ[safe], d2, torch.full_like(d2, INF))
        top = d2.topk(min(k, d2.size(1)), -1, largest=False)
        return safe.gather(1, top.indices), top.values

    @torch.no_grad()
    def mark_hit(self, ids, step=None, ok=None):
        """Candidates came back near entries ids (n,) (ok: an optional mask): the return count += 1."""
        inc = torch.ones_like(ids, dtype=torch.int32) if ok is None else ok.to(torch.int32)
        self.hits.index_add_(0, ids, inc)

    @torch.no_grad()
    def get(self, ids):
        return dict(key=self.keys_f(ids), pgrad=self.sketch_f(ids), step=self.step[ids].long(), hits=self.hits[ids].long(),
                    occ=self.occ[ids])

    @torch.no_grad()
    def take(self, ids):
        """Promotion: read the entries and free them (distinct ids)."""
        out = self.get(ids)
        self.drop(ids)
        return out

    @torch.no_grad()
    def drop(self, ids):
        self.occ.index_fill_(0, ids, False)
        self.ksq.index_fill_(0, ids, INF)

    def age(self, now):
        """Steps since each entry entered (tensor)."""
        # (r6 C7: a Python int enters as a kernel scalar: no host-to-device tensor / sync per step)
        return (now.to(self.step.device).long() if torch.is_tensor(now) else int(now)) - self.step.long()

    def fill(self):
        return self.occ.float().mean()


class BagRead(torch.autograd.Function):
    """u = sum_j w_j V[rows_j] as ONE F.embedding_bag(mode="sum", per_sample_weights=w). Saves only (rows, w); the
    backward computes d/dw_j = <g, V[rows_j]> (a transient gather) and hands (rows, w_j g) to `sink` (the dense fp32 grad
    buffer's index_add_). V is a buffer (not a parameter: no optimizer, no clip, no dense autograd gradient)."""

    @staticmethod
    def forward(ctx, w, V, rows, sink):
        ctx.save_for_backward(w, rows)
        ctx.V, ctx.sink = V, sink
        return _bag(w, V, rows)

    @staticmethod
    def backward(ctx, g):
        w, rows = ctx.saved_tensors
        gw = torch.einsum("nd,nkd->nk", g.float(), ctx.V[rows].float()).to(w.dtype)
        if ctx.sink is not None:
            ctx.sink(rows, w.float()[..., None] * g.float()[:, None, :])
        return gw, None, None, None


def _bag(w, V, rows):
    """The embedding_bag blend in the weights' precision (fp32 training: the fp32 / bf16 table as it is; fp64 tests: an
    fp64 copy of the table)."""
    dt = V.dtype if w.dtype == torch.float32 else w.dtype
    W = V if V.dtype == dt else V.to(dt)
    return F.embedding_bag(rows, W, mode="sum", per_sample_weights=w.to(dt)).to(w.dtype)


def kernel_weights(q, keys, valid, tau, h0, c=1.0, w2=None, gauss_scale=None, use_A=True, d=None):
    """q (n, kd); keys (n, k, kd) detached; valid (n, k). d_j = |q - k_j|^2, d_1 = the nearest valid one,
    w = softmax_j(-(d_j - d_1) / ((d_1 + c h0^2) tau)) over the valid j, A = exp(-d_1 / (2 h0^2)). Returns w (n, k),
    A (n,), d1 (n,); a row without a valid candidate has w = 0 and A = 0 (finite gradients). w2: the width term
    in place of c h0^2 (2 sigma^2, tracked; plan rev 2 section 6.1). gauss_scale (the oracle default): the
    scale-free Gaussian w_j ~ exp(-(d_j - d_1) / gauss_scale) (gauss_scale = c m, m = a running median of d_2 - d_1).
    use_A=False: A = 1 for every row with a candidate (no match-strength factor). d: precomputed squared distances
    (n, k) (e.g. at model width after the re-rank); keys is then unused."""
    if d is None:
        d = (q[:, None, :] - keys).pow(2).sum(-1)
    any_ = valid.any(-1)
    inf = torch.full_like(d, INF)
    d1 = torch.where(valid, d, inf).amin(-1)
    d1 = torch.where(any_, d1, torch.zeros_like(d1))
    dv = torch.where(valid, d, d1[:, None].expand_as(d))
    h2 = h0 * h0
    if gauss_scale is not None:
        logits = torch.where(valid, -(dv - d1[:, None]) / gauss_scale, -inf)
    else:
        wt = c * h2 if w2 is None else w2
        logits = torch.where(valid, -(dv - d1[:, None]) / ((d1[:, None] + wt) * tau), -inf)
    logits = torch.where(any_[:, None], logits, torch.zeros_like(logits))
    w = torch.where(any_[:, None], logits.softmax(-1), torch.zeros_like(logits))
    A = torch.where(any_, torch.exp(-d1 / (2 * h2)) if use_A else torch.ones_like(d1), torch.zeros_like(d1))
    return w, A, d1


def blend(Ag, ug, Ap, up):
    """r = max(A_g, A_p) (pi_g u_g + pi_p u_p), pi = A / (A_g + A_p); 0 when both A are 0."""
    S = Ag + Ap
    pos = S > 0
    Ss = torch.where(pos, S, torch.ones_like(S))
    pg = torch.where(pos, Ag / Ss, torch.zeros_like(S))
    pp = torch.where(pos, Ap / Ss, torch.zeros_like(S))
    return torch.maximum(Ag, Ap)[:, None] * (pg[:, None] * ug + pp[:, None] * up)


def read_reference(q, keys_g, vals_g, valid_g, keys_p, vals_p, valid_p, tau, h0_g, h0_p, c=1.0):
    """Dense reference of one global + private read (gathers (n, k, D)): the same maths as BlobMemory."""
    wg, Ag, _ = kernel_weights(q, keys_g, valid_g, tau, h0_g, c)
    wp, Ap, _ = kernel_weights(q, keys_p, valid_p, tau, h0_p, c)
    return blend(Ag, (wg[..., None] * vals_g).sum(1), Ap, (wp[..., None] * vals_p).sum(1))


class BlobMemory(nn.Module):
    """Global + per-expert blob memory of the loop's routed MLP (module doc). RoutedMLP.forward calls it as
    bmem(xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep) -> (buf, x_shared)."""

    def __init__(self, d, n_exp, max_experts=0, G=262144, P=512, kd=64, cand=16, k_g=8, k_p=4, search="ivf",
                 cells=1024, cell_cap=512, probe=16, q_cap=192, cell_topk=32, brute_chunk=1024, chunk_e=0,
                 read_mode="all", read_p=0.5, compute_frac=1.0, tau=0.1, kernel_c=1.0, h0=4.0, priv_bf16=True,
                 match_grad=True, inj_scale=False, fit_samples=65536, ring=16384, lr=0.03, beta2=0.9, clip=0.05,
                 track_xf=False, probation_rate=0.0, probation_life=3000, seed=1234,
                 gate=False, gate_levels=0, gate_side=0, explore=0.05, buf_frac=0.55, gate_lr=1e-3, gate_clip=1.0,
                 write="off", cand_rate=0.10, write_w=32, private_rate=0.256, promo_gamma=1.0, promo_theta=0.3,
                 prob_lookup="exact", gamma_g=0.25, gamma_p=0.25, theta_m=0.2, protect_age=2000, protect_frac=0.5,
                 strength_half=4000.0, track_sigma=False, rerank_full=False, rerank_dtype="bf16", rerank_top=64,
                 v_init=0.2, v_cap_spawn=0.3, v_cap_after=0.5, conf_w1=0.3, conf_margin=1.1, conf_A=0.1,
                 conf_cos=0.3, max_moves=8, move_window=1000, cap_moves_g=256, cap_moves_p=2, eot=None,
                 content_opt="rmsprop", promo_budget=64, promo_rank="cos", probation_cap=0, probation_dim=64,
                 promo_min_returns=1, probation_dtype="fp16", probation_log=False, kernel="bounded", kernel_cg=0.1,
                 kernel_dist="full", use_A=True, gate_open0=0.23, gate_warm=200):
        super().__init__()
        assert read_mode in READ_MODES and search in SEARCHES
        self.d, self.kd, self.G, self.P = d, kd, G, P
        self.cand, self.k_g, self.k_p = cand, min(k_g, cand), min(k_p, cand)
        self.search, self.n_cells, self.cell_cap, self.probe, self.q_cap = search, cells, cell_cap, probe, q_cap
        self.cell_topk, self.brute_chunk, self.chunk_e = max(cell_topk, 1), brute_chunk, chunk_e
        self.read_mode, self.read_p, self.compute_frac = read_mode, float(read_p), float(compute_frac)
        self.tau, self.kernel_c, self.seed = float(tau), float(kernel_c), seed
        self.priv_bf16, self.match_grad, self.track_xf = priv_bf16, match_grad, track_xf
        self.lr, self.beta2, self.clip = lr, beta2, clip
        # phase 2 (plan 25d rev 2 sections 5-10): the gate, writes, merges, eviction, renewal; all off by default
        assert write in ("off", "probation", "direct")
        self.write_mode, self.cand_rate, self.write_w, self.private_rate = write, cand_rate, write_w, private_rate
        assert content_opt in ("rmsprop", "adagrad")
        self.content_opt, self.promo_budget, self.promo_rank = content_opt, promo_budget, promo_rank
        self.promo_min_returns = promo_min_returns
        assert kernel in ("bounded", "gauss") and kernel_dist in ("full", "key")
        self.kernel, self.kernel_cg, self.kernel_dist, self.use_A = kernel, kernel_cg, kernel_dist, use_A
        self.gate_open0, self.gate_warm = gate_open0, gate_warm
        self._mgaps, self._gq = {"g": [], "p": []}, []
        self._promo_log = []    # sampled promotion-candidate matches (logging only)
        self.promo_gamma, self.promo_theta, self.prob_lookup = promo_gamma, promo_theta, prob_lookup
        self.prob_life = probation_life
        self.gamma_g, self.gamma_p, self.theta_m = gamma_g, gamma_p, theta_m
        self.protect_age, self.protect_frac, self.strength_half = protect_age, protect_frac, strength_half
        self.track_sigma = bool(track_sigma or gate)
        self.v_init, self.v_cap_spawn, self.v_cap_after = v_init, v_cap_spawn, v_cap_after
        self.conf_w1, self.conf_margin, self.conf_A, self.conf_cos = conf_w1, conf_margin, conf_A, conf_cos
        self.max_moves, self.move_window, self.cap_moves_g, self.cap_moves_p = max_moves, move_window, cap_moves_g, cap_moves_p
        self.explore_frac, self.buf_frac, self.gate_clip = explore, buf_frac, gate_clip
        self.rerank_full, self.rerank_top = rerank_full, rerank_top
        self.rerank_rows = 0                    # (r6 A3: query rows per re-rank block; 0 = auto, ~512 MB)
        self.eot = eot
        self.phase2 = bool(gate) or write != "off" or bool(track_sigma) or kernel != "bounded"
        if write != "off":
            track_xf = True                     # the write key is the pass-averaged xf
            self.track_xf = True
        self.ko = False                         # knock-out (eval deltas): nothing is read
        self.train_forward = True               # (LoopBrain.forward: a target was given)
        self.y_scale, self.accum = 1.0, 1       # (the trainer: tokens per loss x accumulation)
        self.Emax = max(int(max_experts or 0), n_exp)
        self.glob = BlobSpace((G,), d, kd, cells=True)
        self.priv = BlobSpace((self.Emax + 1, P), d, kd, value_dtype=torch.bfloat16 if priv_bf16 else torch.float32)
        n_prob = int(probation_cap) or (math.ceil(probation_rate * probation_life) if probation_rate > 0 else 0)
        self.prob = (ProbationStore(n_prob, min(probation_dim, kd), cells, dtype=probation_dtype, log=probation_log,
                                    index=getattr(self, "prob_lookup", "exact") == "ivf")
                     if n_prob else None)                                            # (the write path; flag)
        # the only memory parameter (flag, default off): it is trained by the brain's optimizer like any other gain
        self.inj_scale = nn.Parameter(torch.ones(d)) if inj_scale else None
        gen = torch.Generator().manual_seed(seed)  # (forked: the global RNG is not touched)
        self.register_buffer("proj", torch.randn(d, kd, generator=gen) / math.sqrt(d))
        if self.phase2:  # (drawn after proj: the map is the same with or without phase 2)
            self.register_buffer("gproj", torch.randn(d, kd, generator=gen) / math.sqrt(d))   # gradient sketch
            self.register_buffer("kappa", torch.zeros(()))                                       # curvature EMA
            self.register_buffer("sigma2_g", torch.tensor(kernel_c * float(h0) ** 2 / 2))       # kernel widths
            self.register_buffer("sigma2_p", torch.tensor(kernel_c * float(h0) ** 2 / 2))
            self.register_buffer("gapm_g", torch.zeros(()))     # running median of d2 - d1 (the Gaussian kernel's m)
            self.register_buffer("gapm_p", torch.zeros(()))
            self.register_buffer("xf_norm_med", torch.zeros(()))                                 # EMA median |xf|
            self.register_buffer("credit", torch.zeros(self_emax1 := max(int(max_experts or 0), n_exp) + 1))
            self.register_buffer("ch_mean", torch.zeros(32))                                     # C(H) bins
            self.register_buffer("ch_n", torch.zeros(32, dtype=torch.long))
            self.register_buffer("wstats", torch.zeros(8))                                       # write counters
            self.register_buffer("evict_age", torch.zeros(2, 256))                               # ring of ages
            self.register_buffer("evict_n", torch.zeros(2, dtype=torch.long))
        if rerank_full:
            self.register_buffer("xfull", torch.zeros(G, d, dtype=torch.bfloat16 if rerank_dtype == "bf16" else torch.float32))
        else:
            self.xfull = None
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sigma", torch.ones(d))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))
        self.register_buffer("centroids", torch.zeros(cells, kd))
        self.register_buffer("stat_sum", torch.zeros(d, dtype=torch.float64))
        self.register_buffer("stat_sq", torch.zeros(d, dtype=torch.float64))
        self.register_buffer("stat_n", torch.zeros((), dtype=torch.float64))
        # fit samples (raw xf rows, freed at the fit): queries and write candidates; then the refit rings (x_hat)
        self.register_buffer("fit_x", torch.zeros(fit_samples, d))
        self.register_buffer("fit_w", torch.zeros(fit_samples // 4, d))
        self.register_buffer("fit_n", torch.zeros(2, dtype=torch.long))
        self.register_buffer("ring_q", torch.zeros(ring, kd))
        self.register_buffer("ring_w", torch.zeros(ring // 4, kd))
        self.register_buffer("ring_n", torch.zeros(2, dtype=torch.long))       # rows written so far (q, w)
        self.register_buffer("h0_g", torch.tensor(float(h0)))
        self.register_buffer("h0_p", torch.tensor(float(h0)))
        self.register_buffer("now", torch.zeros((), dtype=torch.long))
        self.register_buffer("n_live", torch.tensor(n_exp, dtype=torch.long))
        # derived (not saved): the IVF cell table, its residual bf16 keys and |k - c|^2, rebuilt by rebuild_cells()
        self.register_buffer("table", torch.full((cells, cell_cap), -1, dtype=torch.long), persistent=False)
        self.register_buffer("table_kr", torch.zeros(cells, cell_cap, kd, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("table_ksq", torch.full((cells, cell_cap), INF, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("cell_overflow", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("cell_spilled", torch.zeros((), dtype=torch.long), persistent=False)
        self.collect = False               # the trainer: gather the fit statistics on this step's training forwards
        self.mask_fn = None                # pluggable read mask (the gate later): fn(info) -> (mask_pairs, mask_shared)
        self.ring_take = 64                # query rows per training search call into the refit ring
        self._fitted, self._fit_n, self._ring_n = False, [0, 0], [0, 0]
        self._cache, self._seen = {}, set()
        self._vg, self._touched = {}, {}   # dense fp32 grad buffers + touched-row masks ("g" / "p")
        self._ws = {}                      # static search workspaces (name -> tensor; replaced only when a size changes)
        self._gens, self._gen_states = {}, {}
        self.xf_sum = self.xf_cnt = None
        self.last, self.stats = None, {}
        # phase-2 per-forward / per-step state (host lists of device tensors; no sync)
        self.prev_mu, self.prev_halt, self._prev, self._prev_none = None, None, None, None
        self._recs, self._conf, self._gaps = {}, {"g": [], "p": []}, {"g": [], "p": []}
        self._credit = {"g": [], "p": []}
        self._edges_A, self._stats_step, self._yring, self._wstats = [], [], None, []
        self._yring_n, self._stats_acc = 0, None   # (r6 G3: the static y ring's host pointer; older stats rows' sum)
        self._kept_pairs, self._cap, self._wsel, self._wqueue = [], {}, None, []
        self._gate_last, self._last_writes, self._evict, self._wcount = None, {}, [], {}
        import random as _random
        self._salt_rng, self._salt = _random.Random(seed + 4), 0
        self._kstats = torch.zeros(5)
        self._mstats = torch.zeros(6)
        self._mstats2 = torch.zeros(2)
        self._hy = torch.zeros(4)               # (health: the gate's learning signal since the last health line)
        self._hcount, self._hevict, self._hsteps = {}, [], 0
        self.guard_violations = 0               # store-changing calls outside training (must stay 0)
        g = None
        if gate:
            from blob_gate import BlobGate
            g = BlobGate(d, gate_side, gate_levels, self.Emax + 1, seed=seed + 11, warm=gate_warm)
            g.lr = gate_lr
        object.__setattr__(self, "gate", g)       # (NOT a submodule: out of model.parameters() / make_opt / clip)
        self.register_load_state_dict_post_hook(lambda mod, keys: mod.post_load())

    # ------------------------------------------------------------------ layout / dtypes / checkpoint
    def _live(self):
        """A training forward of the training step (model.train() AND a target): the only forwards that may touch any
        store, statistic or generator of the memory (eval, val, fact tests, loop reports, probes and target-less
        train-mode forwards read only; LoopBrain.forward sets train_forward)."""
        return self.training and self.train_forward

    @property
    def shared_idx(self):
        return self.priv.occ.size(0) - 1

    def space(self, name):
        return self.glob if name == "g" else self.priv

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for n in ("proj", "mu", "sigma", "centroids", "h0_g", "h0_p", "fit_x", "fit_w", "ring_q", "ring_w"):
            setattr(self, n, getattr(self, n).float())
        for n in ("stat_sum", "stat_sq", "stat_n"):
            setattr(self, n, getattr(self, n).double())
        self.table_kr, self.table_ksq = self.table_kr.to(torch.bfloat16), self.table_ksq.to(torch.bfloat16)
        if self.inj_scale is not None and self.inj_scale.dtype != torch.float32:
            self.inj_scale.data = self.inj_scale.data.float()
        if self.phase2:
            for n in ("gproj", "kappa", "sigma2_g", "sigma2_p", "xf_norm_med", "credit", "ch_mean", "wstats",
                      "evict_age", "gapm_g", "gapm_p"):
                setattr(self, n, getattr(self, n).float())
            dev = self.proj.device
            self._kstats, self._mstats, self._mstats2, self._hy = (t.to(dev) for t in (self._kstats, self._mstats,
                                                                                        self._mstats2, self._hy))
        if self.gate is not None:
            dev = self.proj.device
            self.gate.to(dev).float()
            if self.gate.opt is not None:
                for st in self.gate.opt.state.values():
                    for k, v in st.items():
                        if torch.is_tensor(v) and v.dim() > 0:
                            st[k] = v.to(dev)
        self._vg, self._touched, self._ws = {}, {}, {}
        return self

    def get_extra_state(self):
        st = dict(self._gen_states)
        st.update({n: g.get_state() for n, g in self._gens.items()})
        out = dict(gens=st)
        if self.phase2:
            out["salt_rng"] = self._salt_rng.getstate()
        if self.gate is not None:
            if self.gate.opt is None:
                self.gate.make_opt()
            out["gate"] = {k: v.clone() for k, v in self.gate.state_dict().items()}
            out["gate_opt"] = self.gate.opt.state_dict()
            out["gate_rows"] = self.gate.beta.size(0)
        return out

    def set_extra_state(self, st):
        st = st or {}
        self._gens, self._gen_states = {}, dict(st.get("gens", {}))
        if "salt_rng" in st:
            self._salt_rng.setstate(st["salt_rng"])
        if self.gate is not None and "gate" in st:
            self.gate.grow(st.get("gate_rows", self.gate.beta.size(0)))
            dev = self.proj.device
            sd = {k: v.clone() for k, v in self.gate.state_dict().items()}   # (keys missing from an older
            sd.update({k: v.to(dev) for k, v in st["gate"].items()})          # checkpoint, e.g. y_sc: kept as built)
            self.gate.load_state_dict(sd)
            self.gate.make_opt()
            self.gate.opt.load_state_dict(st["gate_opt"])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for n in ("fit_x", "fit_w"):  # (freed after the fit)
            k = prefix + n
            if k in state_dict and state_dict[k].shape != getattr(self, n).shape:
                setattr(self, n, getattr(self, n).new_zeros(state_dict[k].shape))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._fitted = bool(state_dict.get(prefix + "fitted", self.fitted))
        self._fit_n = [int(v) for v in state_dict.get(prefix + "fit_n", self.fit_n)]
        self._ring_n = [int(v) for v in state_dict.get(prefix + "ring_n", self.ring_n)]
        self._cache, self._seen, self._ws = {}, set(), {}

    def post_load(self):
        """After load_state_dict (the spaces load after this module's own buffers): the derived cell table."""
        self.rebuild_cells()

    def _generator(self, name, device):
        g = self._gens.get(name)
        if g is None or g.device != device:
            g = torch.Generator(device=device)
            if name in self._gen_states:
                g.set_state(self._gen_states.pop(name))
            else:
                g.manual_seed(self.seed + {"mask": 1, "sr": 2, "kmeans": 3, "salt": 4}[name])
            self._gens[name] = g
        return g

    def _workspace(self, name, shape, dtype, device):
        w = self._ws.get(name)
        if w is None or tuple(w.shape) != tuple(shape) or w.dtype != dtype or w.device != device:
            w = self._ws[name] = torch.empty(shape, dtype=dtype, device=device)
        return w

    # ------------------------------------------------------------------ address map, fit, cells
    def address(self, x):
        """xf (..., d) -> x_hat (..., kd), fp32 (or the input's wider float); differentiable in x, the map is fixed."""
        acc = torch.promote_types(x.dtype, torch.float32)
        return ((x.to(acc) - self.mu.to(acc)) / self.sigma.to(acc)) @ self.proj.to(acc)

    @torch.no_grad()
    def observe(self, xf, kind="query"):
        """Fit-window samples (training with `collect` on, before the fit): running sums of the queries + the first
        rows of each kind (query / write candidate), spread evenly over the call. Host-side sizes only."""
        if not self.collect or self._fitted or not self._live():
            return
        x = xf.detach()
        j, buf = (0, self.fit_x) if kind == "query" else (1, self.fit_w)
        if j == 0:
            xd = x.double()
            self.stat_sum += xd.sum(0)
            self.stat_sq += xd.pow(2).sum(0)
            self.stat_n += x.size(0)
        take = min(buf.size(0) - self._fit_n[j], x.size(0))
        if take > 0:
            buf[self._fit_n[j]:self._fit_n[j] + take] = x[::x.size(0) // take][:take].float()
            self._fit_n[j] += take
            self.fit_n[j] = self._fit_n[j]

    @torch.no_grad()
    def _ring_add(self, xh, j):
        """Recent x_hat samples for refit() (a ring; host pointer)."""
        ring = self.ring_q if j == 0 else self.ring_w
        n = min(self.ring_take if j == 0 else xh.size(0), xh.size(0), ring.size(0))
        if n <= 0:
            return
        rows = xh[::max(1, xh.size(0) // n)][:n].float()
        idx = (torch.arange(n, device=ring.device) + self._ring_n[j]) % ring.size(0)
        ring.index_copy_(0, idx, rows)
        self._ring_n[j] += n
        self.ring_n[j] = self._ring_n[j]

    def observe_write(self, xf):
        """Write-candidate samples (the future write path calls this): the fit mix before the fit, the ring after."""
        if not self._fitted:
            self.observe(xf, "write")
        else:
            self._ring_add(self.address(xf.detach().float()), 1)

    @torch.no_grad()
    def fit(self, iters=10):
        """ONCE: mu / sigma per channel (sigma floored at 1e-3 x its median; no statistics -> identity), the IVF cells by
        weighted k-means (k-means++ init, forked generator) on the window's x_hat samples (queries + write candidates;
        none -> the stored global keys; none -> a fixed random draw), the stored keys' cells and the cell table."""
        n = self.stat_n.clamp(min=1)
        mu = self.stat_sum / n
        sd = (self.stat_sq / n - mu.pow(2)).clamp(min=0).sqrt()
        sd = torch.maximum(sd, 1e-3 * sd.median())
        seen = self.stat_n > 0
        self.mu.copy_(torch.where(seen, mu, torch.zeros_like(mu)).float())
        self.sigma.copy_(torch.where(seen & (sd > 0), sd, torch.ones_like(sd)).float())
        gen = self._generator("kmeans", self.centroids.device)
        X = torch.cat([self.address(self.fit_x[:self._fit_n[0]]), self.address(self.fit_w[:self._fit_n[1]])])
        wts = torch.ones(X.size(0), device=X.device)
        if X.size(0) == 0:
            X, wts = self.glob.key, self.glob.occ.float()
        if float(wts.sum()) == 0:  # (one host sync, once)
            X = torch.randn(self.n_cells, self.kd, generator=gen, device=self.centroids.device)
            wts = torch.ones(X.size(0), device=X.device)
        self.centroids.copy_(self.kmeans(X.float(), wts, self.n_cells, iters, gen))
        self.fitted.fill_(True)
        self._fitted = True
        self.fit_x, self.fit_w = self.fit_x.new_zeros(0, self.d), self.fit_w.new_zeros(0, self.d)
        self._fit_n = [0, 0]
        self.fit_n.zero_()
        self.assign_cells()

    @torch.no_grad()
    def refit(self, iters=3):
        """Refit hook: warm Lloyd iterations from the current centroids on the recent query / write-candidate rings and
        the stored global keys, then every key re-assigned (an index operation: keys unchanged)."""
        nq = min(self._ring_n[0], self.ring_q.size(0))
        nw = min(self._ring_n[1], self.ring_w.size(0))
        X = torch.cat([self.ring_q[:nq], self.ring_w[:nw], self.glob.key])
        wts = torch.cat([X.new_ones(nq + nw), self.glob.occ.float()])
        self.centroids.copy_(self.kmeans(X, wts, self.n_cells, iters, None, init=self.centroids))
        self.assign_cells()

    @staticmethod
    @torch.no_grad()
    def kmeans(X, wts, k, iters, gen, init=None):
        """Weighted k-means: k-means++ init (or `init`) + `iters` Lloyd iterations (an empty cell keeps its centroid)."""
        n = X.size(0)
        if init is not None:
            C = init.clone()
        else:
            if n < k:
                rep = torch.arange(k, device=X.device) % n
                X, wts, n = X[rep], wts[rep], k
            C = X.new_empty(k, X.size(1))
            p0 = wts + 1e-12
            C[0] = X[torch.multinomial(p0 / p0.sum(), 1, generator=gen)]
            dmin = (X - C[0]).pow(2).sum(-1)
            for i in range(1, k):
                p = dmin.clamp(min=0) * wts + 1e-12
                C[i] = X[torch.multinomial(p / p.sum(), 1, generator=gen)]
                dmin = torch.minimum(dmin, (X - C[i]).pow(2).sum(-1))
        for _ in range(iters):
            a = ((X.pow(2).sum(-1)[:, None] - 2 * X @ C.t()) + C.pow(2).sum(-1)[None]).argmin(-1)
            s = torch.zeros_like(C).index_add_(0, a, X * wts[:, None])
            c = torch.zeros(k, device=X.device, dtype=X.dtype).index_add_(0, a, wts.to(X.dtype))
            C = torch.where(c[:, None] > 0, s / c.clamp(min=1e-12)[:, None], C)
        return C

    @torch.no_grad()
    def set_centroids(self, C):
        """Online centroid update hook (policy later): new centroids, keys re-assigned, the table rebuilt."""
        self.centroids.copy_(C)
        self.assign_cells()

    @torch.no_grad()
    def set_h0(self, h0_g=None, h0_p=None):
        """Kernel-width update hook (calibration / tracking policy later)."""
        if h0_g is not None:
            self.h0_g.copy_(torch.as_tensor(h0_g))
        if h0_p is not None:
            self.h0_p.copy_(torch.as_tensor(h0_p))

    @torch.no_grad()
    def assign_cells(self, flat=None):
        """Nearest and second-nearest cells of the global slots (all, or the given slots), then the table."""
        sp = self.glob
        if flat is None:
            sp.cell_pref.copy_(nearest_cells(sp.key, self.centroids))
        else:
            sp.cell_pref.index_copy_(0, flat, nearest_cells(sp.key[flat], self.centroids))
        self.rebuild_cells()

    @torch.no_grad()
    def rebuild_cells(self, probation=True):
        """The global space's static cell table (ivf_table: nearest cell, spill to the second-nearest, the rest left
        out and counted - the future spawn / eviction rules keep cells below capacity); the probation store's too."""
        sp = self.glob
        t = ivf_table(sp.key, sp.occ, sp.cell_pref, self.centroids, self.cell_cap)
        self.table, self.table_kr, self.table_ksq = t["table"], t["kr"], t["ksq"]
        sp.cell.copy_(t["cell"])
        self.cell_overflow, self.cell_spilled = t["overflow"], t["spilled"]
        # (the probation store's IVF table only exists for --gm-prob-lookup ivf: the exact lookup never reads it, and
        # building it over millions of entries at every merge / refit costs gigabytes)
        if probation and self.prob is not None and self._fitted and self.prob_lookup == "ivf":
            self.prob.rebuild(self.centroids)

    # ------------------------------------------------------------------ search (no gradient)
    @torch.no_grad()
    def _rerank(self, q, cand, k):
        """Exact fp32 distances of global candidates cand (N, M) (-1 = none) -> top-k (idx, d2 ascending; +inf = none)."""
        return rerank(q, self.glob.key, self.glob.occ, cand, k)

    @torch.no_grad()
    def search_global(self, q, pos=None, qfull=None):
        """q (N, kd) -> the top-`cand` (idx (N, cand) long, d2 (N, cand) fp32 ascending, +inf = none). pos (N,): the
        token positions (IVF: a cell's query capacity goes to earlier positions first: causal). qfull (N, d): with
        rerank_full, the top-rerank_top by the 64-d fp32 distance are re-ranked by the model-width distance (the
        returned d2 stay the 64-d kernel distances)."""
        q = q.float()
        k = min(self.cand, self.G)
        kk = min(max(self.rerank_top, k), self.G) if (self.rerank_full and qfull is not None) else k
        idx, d2 = self._search_brute(q, kk) if self.search == "brute" else self._search_ivf(q, kk, pos)
        if kk == k:
            return idx, d2
        # (r6 A3: the model-width distances in row blocks: no (N, kk, d) fp32 gather of the whole batch; same values)
        safe = idx.clamp(min=0)
        qf32 = qfull.float()
        rb = self.rerank_rows or max(1, (1 << 27) // max(1, kk * self.d))  # rows per block (~512 MB of fp32)
        df = torch.cat([(qf32[a:a + rb, None, :] - self.xfull[safe[a:a + rb]].float()).pow(2).sum(-1)
                        for a in range(0, idx.size(0), rb)]) if idx.size(0) else d2.new_zeros(0, kk)
        df = torch.where(torch.isfinite(d2), df, torch.full_like(df, INF))
        top = df.topk(k, -1, largest=False)
        ids, dk = idx.gather(1, top.indices), d2.gather(1, top.indices)
        if self.kernel == "gauss" and self.kernel_dist == "full":
            # (r6 C4: the read uses these model-width distances: return them, ascending (topk sorts))
            return ids, top.values
        # (r6 C4: the bounded / key-distance read uses the 64-d distances: sort the read set [:k_g] and the rest
        # separately by them, so column 0 is the read's nearest and the read set is unchanged)
        out_i, out_d = [], []
        for a, b in ((0, min(self.k_g, k)), (min(self.k_g, k), k)):
            if b > a:
                o = dk[:, a:b].sort(dim=-1, stable=True)
                out_i.append(ids[:, a:b].gather(1, o.indices))
                out_d.append(o.values)
        return torch.cat(out_i, 1), torch.cat(out_d, 1)

    @torch.no_grad()
    def _search_brute(self, q, k):
        """Reference / tests only: chunked bf16 scores over every slot, the bf16 top-max(2k, 8) re-ranked in fp32."""
        sp = self.glob
        return self._rerank(q, exact_candidates(q, sp.key_bf, sp.ksq, max(2 * k, 8), self.brute_chunk), k)

    @torch.no_grad()
    def _search_ivf(self, q, k, pos=None):
        ws = lambda name, shape, dt: self._workspace(name, shape, dt, q.device)
        cand, ov = ivf_candidates(q, self.centroids, self.table, self.table_kr, self.table_ksq, self.probe, self.q_cap,
                                  self.cell_topk, 2 * k, ws, pos=pos)
        self.stats = dict(probe_overflow=ov, cell_overflow=self.cell_overflow, cell_spilled=self.cell_spilled)
        return self._rerank(q, cand, k)

    # ------------------------------------------------------------------ the probation store (write path; flag)
    @torch.no_grad()
    def probation_push(self, xh, pgrad, src=None, sig=None):
        """Append write candidates (x_hat keys (n, kd), projected gradients (n, kd)) FIFO, stamped with `now`."""
        if self.phase2 and not self.training:
            self.guard_violations += 1
        return self.prob.push(xh, pgrad, self.now, self.centroids if self._fitted else None, sig)

    @torch.no_grad()
    def probation_lookup(self, q, k=None, mode="ivf"):
        """The probation store's k nearest entries of x_hat queries q: exact, or over the global IVF cells (after the
        fit, and over the table of the last rebuild: call prob.rebuild(centroids, reassign=False) after pushes)."""
        return self.prob.lookup(q, k or self.cand, mode if self._fitted else "exact", self.centroids, self.probe,
                                self.q_cap, self.cell_topk, self.brute_chunk)

    @torch.no_grad()
    def _rerank_priv(self, q, spaces, cand, k):
        """Private candidates (n, M) in spaces (n,) -> (flat slot ids (n, k), d2 (n, k)) by the exact fp32 distance."""
        sp = self.priv
        flat = spaces[:, None] * self.P + cand
        d2 = (q[:, None, :] - sp.key.view(-1, self.kd)[flat]).pow(2).sum(-1)
        d2 = torch.where(sp.occ.view(-1)[flat], d2, torch.full_like(d2, INF))
        top = d2.topk(min(k, d2.size(1)), -1, largest=False)
        return flat.gather(1, top.indices), top.values

    @torch.no_grad()
    def search_private(self, Q, E):
        """Q (E, C, kd): the capacity layout -> the bf16 top-min(2 cand, P) slot indices (E, C, M) within each routed
        expert's own space: one batched bmm [E, C, kd] x [E, kd, P] into a workspace (in expert chunks of chunk_e)."""
        sp = self.priv
        M = min(max(2 * self.cand, 8), self.P)
        C = Q.size(1)
        Qb = Q.to(torch.bfloat16)
        S = self._workspace("priv_s", (E, C, self.P), torch.bfloat16, Q.device)
        ksq = sp.ksq[:E].to(torch.bfloat16)
        step = self.chunk_e if self.chunk_e and self.chunk_e > 0 else max(E, 1)
        out = []
        for a in range(0, E, step):
            b = min(E, a + step)
            torch.bmm(Qb[a:b], sp.key_bf[a:b].transpose(1, 2), out=S[a:b])
            out.append(S[a:b].mul_(-2).add_(ksq[a:b, None, :]).topk(M, -1, largest=False).indices)
        return torch.cat(out) if out else Q.new_zeros(0, C, M, dtype=torch.long)

    @torch.no_grad()
    def search_shared(self, q):
        sp, j = self.priv, self.shared_idx
        s = sp.ksq[j][None] - 2 * (q.to(torch.bfloat16) @ sp.key_bf[j].t()).float()
        return s.topk(min(max(2 * self.cand, 8), self.P), -1, largest=False).indices

    # ------------------------------------------------------------------ the read mask (pluggable; placeholder modes)
    @torch.no_grad()
    def read_mask(self, n, device):
        if self.read_mode == "all":
            return torch.ones(n, dtype=torch.bool, device=device)
        if self.read_mode == "none":
            return torch.zeros(n, dtype=torch.bool, device=device)
        return torch.rand(n, generator=self._generator("mask", device), device=device) < self.read_p

    # ------------------------------------------------------------------ the differentiable read
    def _sink(self, name, table, valid=None):
        def sink(rows, g):
            vg = self._vg.get(name)
            if vg is None or vg.shape != table.shape or vg.device != g.device:
                vg = self._vg[name] = torch.zeros(table.shape, dtype=torch.float32, device=g.device)
                self._touched[name] = torch.zeros(table.size(0), dtype=torch.bool, device=g.device)
            r = rows.reshape(-1)
            vg.index_add_(0, r, g.reshape(-1, g.size(-1)).float())
            if valid is None:
                self._touched[name].index_fill_(0, r, True)
            else:   # (r6 C2: a padding / unoccupied candidate (weight 0) does not mark its row as touched)
                hit = torch.zeros(table.size(0), device=g.device).index_add_(0, r, valid.reshape(-1).float())
                self._touched[name] |= hit > 0
        return sink

    def _read(self, q, space, flat, h0, grad=True, qfull=None, stats=True):
        """q (n, kd), flat (n, k) flat slot ids (-1 = none) -> u = sum_j w_j v_j (n, D) in q's dtype, A (n,), d1 (n,).
        qfull (n, d): the standardised xf rows (the Gaussian kernel at model width, global space, kernel_dist full).
        Only real candidates (id >= 0, occupied, finite distance) enter the softmax; a row without one reads 0."""
        name = "g" if space is self.glob else "p"
        safe = flat.clamp(min=0)                                           # (r6 C2: never gather with -1)
        keys = space.key.view(-1, self.kd)[safe].to(q.dtype)              # detached (buffers)
        valid = (flat >= 0) & space.occ.view(-1)[safe]
        w2 = scale = d = None
        qq = q if self.match_grad else q.detach()
        if self.kernel == "gauss":
            if name == "g" and self.kernel_dist == "full" and qfull is not None and self.xfull is not None:
                qf = qfull if self.match_grad else qfull.detach()
                dist = lambda qf_, safe_: (qf_[:, None, :] - self.xfull[safe_].to(qf_.dtype)).pow(2).sum(-1)
                if getattr(self, "ckpt_dist", False) and qf.requires_grad:
                    # (VRAM saver --vram-ckpt-gmdist: the (n, k, d) fp32 difference is recomputed in backward
                    # instead of being kept for it; same maths. xfull only changes at the step end.)
                    from spiking_lm import ck_ready, ck_run
                    d = ck_run(dist, qf, safe) if ck_ready() else dist(qf, safe)
                else:
                    d = dist(qf, safe)                                                 # bf16 keys, fp32 distance
            else:
                d = (qq[:, None, :] - keys).pow(2).sum(-1)
            valid = valid & torch.isfinite(d.detach())
            m = self.gapm_g if name == "g" else self.gapm_p
            scale = self.kernel_cg * torch.where(m > 0, m, h0.to(m.dtype) ** 2).to(q.dtype)
            if stats and self._live():  # (the running scale: training-mode reads only; per-step median, EMA at the end)
                with torch.no_grad():
                    dd = torch.where(valid, d.detach(), torch.full_like(d, INF)).float().sort(-1).values
                    gap = dd[:, 1] - dd[:, 0] if dd.size(1) > 1 else dd[:, 0] * 0
                    self._mgaps[name].append(torch.where(torch.isfinite(gap), gap, torch.full_like(gap, float("nan"))))
        elif self.track_sigma:
            w2 = 2 * (self.sigma2_g if name == "g" else self.sigma2_p).to(q.dtype)
        w, A, d1 = kernel_weights(qq, keys, valid, self.tau, h0.to(q.dtype), self.kernel_c, w2, gauss_scale=scale,
                                  use_A=self.use_A, d=d)
        table = space.value.view(-1, self.d)
        if grad and torch.is_grad_enabled() and w.requires_grad:
            return BagRead.apply(w, table, safe, self._sink(name, table, valid)), A, d1
        return _bag(w, table, safe), A, d1

    def begin_forward(self):
        """A new model forward: drop the cached selections and the per-token xf sums of the previous one."""
        self._cache, self._seen = {}, set()
        self.xf_sum = self.xf_cnt = None
        if self.phase2:
            self.prev_mu, self.prev_halt, self._prev, self._prev_none = None, None, None, None
            if self.gate is not None and self._live() and torch.is_grad_enabled():   # (a host salt per forward)
                self._salt = self._salt_rng.getrandbits(31)   # (pure Python: no device, no sync; saved)
            else:
                self._salt = 0

    def __call__(self, xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=None, pair_valid=None):
        if not self.phase2:
            return self.inject(xf, buf, x, e_k, p_k, t_k, call=call, pair_valid=pair_valid)
        rec_on = self._live() and torch.is_grad_enabled() and call is not None
        first = call is None or call not in self._cache
        if rec_on and first and self._fitted:
            with torch.no_grad():  # the content norm scale (EMA of the median |xf|)
                med = xf.detach().float().norm(dim=-1).median()
                self.xf_norm_med.copy_(torch.where(self.xf_norm_med > 0, self.xf_norm_med.lerp(med, 0.01), med))
        if self.gate is not None and self._fitted:
            from blob_gate import inject_gated
            self.observe(xf) if first else None
            self._track(xf, call)
            bufo, xs = inject_gated(self, xf, buf, x, side, e_k, p_k, t_k, call=call, pair_valid=pair_valid)
        else:
            bufo, xs = self.inject(xf, buf, x, e_k, p_k, t_k, call=call, pair_valid=pair_valid)
        if rec_on and self.write_mode != "off" and self._fitted:
            from blob_write import capture_hooks
            if first:
                self._kept_pairs.append((e_k, t_k))
            capture_hooks(self, call, bufo, xs, e_k, p_k, t_k)
        return bufo, xs

    def _make_rec(self, call, part, sel, rows, tok, q, Ag, Ap, jg, jp, gdr, pdr, s, rg, rp, n_pairs):
        """The record of one Scale read-out (the gated read): filled by Scale.backward, read by after_backward."""
        with torch.no_grad():
            dev = q.device
            n = rows.numel()
            k = sel["k_r"] if part == "r" else sel["k_s"]
            s_raw = sel["s_r"] if part == "r" else sel["s_s"]
            rgd, rpd = rg.detach().float(), rp.detach().float()

            def w1(dd, name):
                d1 = dd[:, :1]
                fin = torch.isfinite(dd)
                wt = 2 * (self.sigma2_g if name == "g" else self.sigma2_p)
                z = torch.where(fin, torch.exp(-(dd - d1) / ((d1 + wt) * self.tau)), torch.zeros_like(dd))
                return torch.where(fin[:, 0], 1.0 / z.sum(-1).clamp(min=1e-30), torch.zeros_like(d1[:, 0]))

            gap = lambda dd: torch.where(torch.isfinite(dd[:, 1]), dd[:, 1] - dd[:, 0], torch.zeros_like(dd[:, 0]))
            ratio = lambda dd: torch.where(torch.isfinite(dd[:, 1]), dd[:, 1] / dd[:, 0].clamp(min=1e-12),
                                           torch.full_like(dd[:, 0], 1e9))
            gen = self._generator("mask", dev)
            jr = torch.randint(0, self.G, (n,), generator=gen, device=dev)
            plc = (torch.rand(n, generator=gen, device=dev) < 0.01) & self.glob.occ[jr]
            vg = jg >= 0                         # (r6 C2: -1 = the token had no global candidate: no credit / renewal)
            jg = jg.clamp(min=0)
            rec = dict(bm=self, bk=torch.zeros(n, 6, device=dev), seen=torch.zeros(1, device=dev), k=k,
                       s=s.detach().float(), ex=(s_raw < 1) & k, r2=(rgd + rpd).pow(2).sum(-1),
                       rg2=rgd.pow(2).sum(-1), rp2=rpd.pow(2).sum(-1), jg=jg, jp=jp, q=q.detach()[tok].float(),
                       Ag=Ag.detach().float(), Ap=Ap.detach().float(), w1g=w1(gdr[:, :self.k_g], "g"),
                       w1p=w1(pdr[:, :self.k_p], "p"), ratio_g=ratio(gdr), ratio_p=ratio(pdr), gap_g=gap(gdr),
                       gap_p=gap(pdr), orig=rows + (0 if part == "r" else n_pairs), jr=jr, plc=plc,
                       is_shared=torch.full((n,), float(part == "s"), device=dev), vg=vg)
        slot = self._recs.setdefault(call, {})
        slot[part] = rec
        slot["sel"] = sel
        slot["n_total"] = n_pairs + (sel["o_s"].numel() and sel["prev"].size(0))
        return rec

    def after_backward(self):
        """After each micro-batch's backward (the trainer): the gate's post-backward update, the confirmation / strength
        / curvature / width statistics, the write candidates' gradients."""
        out = None
        if self.gate is not None:
            from blob_gate import after_backward
            if self.gate.opt is None:
                self.gate.make_opt()
            out = after_backward(self)
        else:
            self._recs = {}
        if self.write_mode != "off":
            from blob_write import collect_writes
            collect_writes(self)
        return out

    def select_writes(self, logits, targets, idx=None, ce_h=None):
        """Between the loss and backward (the trainer): this micro-batch's write candidates. ce_h (--chunk-ce, logits
        None): the final read-out's (ce, raw entropy, V) per token."""
        if self.write_mode == "off" or not self._fitted or not self._live():
            return None
        from blob_write import select_writes
        return select_writes(self, None if logits is None else logits.detach(), targets, idx, ce_h=ce_h)

    @torch.no_grad()
    def search_shared_like(self, keys, spaces):
        """Private candidates of arbitrary (key, space) rows: bf16 scores over each row's own space -> top-2 cand."""
        sp = self.priv
        kb = sp.key_bf[spaces]                                               # (n, P, kd)
        s = sp.ksq[spaces] - 2 * torch.bmm(kb, keys.to(torch.bfloat16)[:, :, None]).squeeze(-1).float()
        return s.topk(min(max(2 * self.cand, 8), self.P), -1, largest=False).indices

    @torch.no_grad()
    def _track(self, xf, call):
        if not self.track_xf or (call is not None and call in self._seen):
            return
        if call is not None:
            self._seen.add(call)
        x = xf.detach().float()
        if self.xf_sum is None or self.xf_sum.shape != x.shape:
            self.xf_sum, self.xf_cnt = torch.zeros_like(x), torch.zeros(x.size(0), device=x.device)
        self.xf_sum += x
        self.xf_cnt += 1

    def inject(self, xf, buf, x, e_k, p_k, t_k, call=None, pair_valid=None):
        """RoutedMLP's hook. xf (N, D) the experts' input; buf (E, C, D) its capacity layout; x (B, T, D) the shared
        expert's input; e_k, p_k, t_k the (expert, capacity slot, token) pair list (pair_valid: an optional mask of its
        real rows). Returns (buf, x_shared) with the reads added (unchanged before the fit and in read mode 'none').
        call: the cache id of this read (the first run's selection is reused by a recompute)."""
        first = call is None or call not in self._cache
        if first:
            self.observe(xf)
        self._track(xf, call)
        if not self._fitted or self.read_mode == "none":
            return buf, x
        N, dev, n_pairs = xf.size(0), xf.device, e_k.size(0)
        E, C = buf.size(0), buf.size(1)
        with torch.autocast(dev.type, enabled=False):
            acc = torch.promote_types(xf.dtype, torch.float32)
            q = self.address(xf.to(acc))                                   # (N, kd), the gradient reaches xf
            sel = None if first else self._cache[call]
            if sel is None:
                with torch.no_grad():
                    qd = q.detach().float()
                    gi, gd = self.search_global(qd)
                    Q = self._workspace("priv_q", (E, C, self.kd), torch.float32, dev)
                    Q.zero_()
                    Q.index_put_((e_k, p_k), qd[t_k])
                    pf, pd = self._rerank_priv(qd[t_k], e_k, self.search_private(Q, E)[e_k, p_k], self.cand)
                    sh = torch.full((N,), self.shared_idx, dtype=torch.long, device=dev)
                    sf, sd = self._rerank_priv(qd, sh, self.search_shared(qd), self.cand)
                    info = dict(gi=gi, gd=gd, pf=pf, pd=pd, sf=sf, sd=sd, e_k=e_k, t_k=t_k)
                    mr, ms = self.mask_fn(info) if self.mask_fn is not None else (
                        self.read_mask(n_pairs, dev), self.read_mask(N, dev))
                    if pair_valid is not None:
                        mr = mr & pair_valid
                    sel = dict(gi=gi, gd=gd, pf=pf, pd=pd, sf=sf, sd=sd, mr=mr, ms=ms)
                    if self.compute_frac < 1:  # static compute buffer: the first M open pairs (M from the list length)
                        M = max(1, min(n_pairs, math.ceil(self.compute_frac * n_pairs)))
                        o = (~mr).to(torch.uint8).argsort(stable=True)[:M]
                        sel.update(o=o, mo=mr[o])
                    if self._live():
                        self._ring_add(qd, 0)
                    self._count(sel, t_k)
                if call is not None and torch.is_grad_enabled():
                    self._cache[call] = sel
            kg, kp = self.k_g, self.k_p
            qfull = ((xf.to(acc) - self.mu.to(acc)) / self.sigma.to(acc)) if self.xfull is not None else None
            ug, Ag, dg = self._read(q, self.glob, sel["gi"][:, :kg], self.h0_g, qfull=qfull)   # once per token
            us, As, ds = self._read(q, self.priv, sel["sf"][:, :kp], self.h0_p)      # the shared expert
            o = sel.get("o")
            tp = t_k if o is None else t_k[o]
            up, Ap, dp = self._read(q[tp], self.priv, (sel["pf"] if o is None else sel["pf"][o])[:, :kp], self.h0_p)
            mp = (sel["mr"] if o is None else sel["mo"])[:, None].to(acc)
            r = blend(Ag[tp], ug[tp], Ap, up) * mp
            r_s = blend(Ag, ug, As, us) * sel["ms"][:, None].to(acc)
            if self.inj_scale is not None:
                r, r_s = r * self.inj_scale.to(acc), r_s * self.inj_scale.to(acc)
            idx = (e_k, p_k) if o is None else (e_k[o], p_k[o])
            buf = buf.index_put(idx, r.to(buf.dtype), accumulate=True)
            x_s = x + r_s.to(x.dtype).view(x.shape)
            self.last = dict(A_g=Ag.detach(), d1_g=dg.detach(), A_p=Ap.detach(), d1_p=dp.detach(), A_s=As.detach(),
                             read_frac=sel["mr"].float().mean())
        return buf, x_s

    @torch.no_grad()
    def _count(self, sel, t_k):
        """Use count + touch step of the nearest blob of every masked read of an occupied slot (no sync). Training
        only: an evaluation forward leaves every store bit-identical."""
        if not self._live():
            return
        now = self.now
        N = sel["ms"].size(0)
        g_on = sel["ms"] | (torch.zeros(N, device=t_k.device).index_add_(0, t_k, sel["mr"].float()) > 0)
        for space, flat, m in ((self.glob, sel["gi"][:, 0], g_on), (self.priv, sel["pf"][:, 0], sel["mr"]),
                               (self.priv, sel["sf"][:, 0], sel["ms"])):
            ok = (flat >= 0) & space.occ.view(-1)[flat.clamp(min=0)] & m      # (r6 C2: -1 = no candidate)
            flat = flat.clamp(min=0)
            space.uses.view(-1).index_add_(0, flat, ok.long())
            space.touch.view(-1).scatter_reduce_(0, flat, torch.where(ok, now, torch.full_like(flat, -1)), "amax")

    # ------------------------------------------------------------------ the content update (storage mechanics)
    @torch.no_grad()
    def apply_value_grads(self, lr=None, beta2=None, clip=None, eps=1e-8):
        """Lazy row-wise RMSprop on the rows touched since the last call (then the buffers are zeroed): for a touched
        row v <- b2 v + (1 - b2) mean_c g^2, u = lr g / (sqrt(v) + eps), |u| capped at clip (0 = no cap), value -= u
        (private bf16: rounded stochastically), opt_n += 1. No momentum, no weight decay; untouched rows (value and
        state) are left bit-identical (torch.where, dense, no sync)."""
        lr = self.lr if lr is None else lr
        b2 = self.beta2 if beta2 is None else beta2
        clip = self.clip if clip is None else clip
        for name in ("g", "p"):
            g = self._vg.get(name)
            if g is None:
                continue
            space, t = self.space(name), self._touched[name]
            v = space.opt_v.view(-1)
            if self.content_opt == "adagrad":   # (plain row-wise Adagrad on the touched rows; no momentum, no decay)
                v_new = torch.where(t, v + g.pow(2).mean(-1), v)
            else:
                v_new = torch.where(t, b2 * v + (1 - b2) * g.pow(2).mean(-1), v)
            upd = g * (lr / (v_new.sqrt() + eps))[:, None]
            if clip and clip > 0:
                upd = upd * (clip / upd.norm(dim=-1, keepdim=True).clamp(min=clip))
            val = space.value.view(-1, self.d)
            new = val.float() - upd
            new = stochastic_bf16(new, self._generator("sr", val.device)) if val.dtype == torch.bfloat16 else new
            if self.phase2:  # (rev 2: |v| <= 0.5 median |xf| after any update; touched rows only)
                lim = self.v_cap_after * self.xf_norm_med
                nn_ = new.float().norm(dim=-1, keepdim=True)
                at_cap = t & (nn_[:, 0] > lim) & (lim > 0)
                new = torch.where(at_cap[:, None], new.float() * (lim / nn_.clamp(min=1e-30)), new.float())
                self.wstats[6 if name == "g" else 7] += at_cap.float().sum()
            val.copy_(torch.where(t[:, None], new.to(val.dtype), val))
            v.copy_(v_new)
            space.opt_n.view(-1).add_(t.long())
            g.zero_()
            t.zero_()

    def zero_value_grads(self, free=False):
        if free:
            self._vg, self._touched = {}, {}
        for g in self._vg.values():
            g.zero_()
        for t in self._touched.values():
            t.zero_()

    @torch.no_grad()
    def reset_opt_state(self, name, flat):
        """Content-optimizer state reset of the given flat slots (write / merge / split)."""
        sp = self.space(name)
        sp.opt_v.view(-1).index_fill_(0, flat, 0.0)
        sp.opt_n.view(-1).index_fill_(0, flat, 0)

    @torch.no_grad()
    def after_step(self, step, refit_every=0):
        if self.phase2 and not self.training:   # (guard: the step end never runs outside training)
            self.guard_violations += 1
            return
        """Step-end hook: the content update, the IVF refit every `refit_every` steps (0 = never); later the write /
        merge / evict decisions (at most one host sync)."""
        if self.gate is not None:
            from blob_gate import step_end
            self._last_writes["moved"] = step_end(self, step)
        self.apply_value_grads()
        if self.phase2 and self._fitted:
            from blob_gate import step_end
            if self.gate is None:
                self._last_writes["moved"] = step_end(self, step)
            if self.write_mode != "off":
                from blob_write import step_writes
                self._last_writes.update(step_writes(self, step))
            self._hsteps += 1
            for k, v in self._last_writes.items():
                if torch.is_tensor(v) and not k.startswith("_"):
                    self._hcount[k] = self._hcount.get(k, 0) + v
            for k, v in self._last_writes.items():
                if k.startswith("_"):
                    continue
                if torch.is_tensor(v):
                    self._wcount[k] = self._wcount.get(k, 0) + v
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        self._wcount[f"{k}_{k2}"] = self._wcount.get(f"{k}_{k2}", 0) + v2
            self._last_writes = {}
        if refit_every and self._fitted and step > 0 and step % refit_every == 0:
            self.refit()

    # ------------------------------------------------------------------ births
    @torch.no_grad()
    def on_grow(self, src, E0, split=True):
        """n = len(src) routed experts were born as E0 .. E0 + n - 1 (parents src). split: each child's private space
        becomes a copy of its parent's at half strength with a fresh content-optimizer state (both twins keep the
        blobs); otherwise (ramp) it starts empty. Grows the family past its preallocation (the shared space moves to
        the new last index). Nothing global changes."""
        n = len(src)
        dev = self.priv.occ.device
        if E0 + n > self.Emax:
            old = torch.tensor([self.shared_idx], device=dev)
            self.priv.grow(E0 + n + 1)
            self.priv.copy_spaces(old, torch.tensor([self.shared_idx], device=dev))
            self.priv.clear_spaces(old)
            self.Emax = E0 + n
            self._vg.pop("p", None)
            self._touched.pop("p", None)
            self._ws.pop("priv_s", None)
        dst = torch.arange(E0, E0 + n, device=dev)
        if self.gate is not None:
            self.gate.grow(self.priv.occ.size(0))
            if split:
                self.gate.copy_rows(torch.as_tensor(src, device=dev).long(), dst)
            else:
                self.gate.reset_rows(dst)
        if self.phase2 and self.credit.numel() < self.priv.occ.size(0):
            c = self.credit
            new = c.new_zeros(self.priv.occ.size(0))
            new[:c.numel() - 1] = c[:-1]
            new[-1] = c[-1]
            self.credit = new
        if self.phase2 and split:
            self.credit[dst] = self.credit[torch.as_tensor(src, device=dev).long()]
        if split and self.phase2:  # (review: BOTH twins at half strength, both with a fresh content-optimizer state)
            srcl = torch.as_tensor(src, device=dev).long()
            half = self.priv.strength.index_select(0, srcl) * 0.5
            self.priv.strength.index_copy_(0, srcl, half)
            self.priv.opt_v.index_fill_(0, srcl, 0.0)
            self.priv.opt_n.index_fill_(0, srcl, 0)
        if split:
            self.priv.copy_spaces(torch.as_tensor(src, device=dev).long(), dst)
            if not self.phase2:  # (phase 2: the parent was halved above, the child copied that)
                self.priv.strength.index_copy_(0, dst, self.priv.strength.index_select(0, dst) * 0.5)
            self.priv.opt_v.index_fill_(0, dst, 0.0)
            self.priv.opt_n.index_fill_(0, dst, 0)
        else:
            self.priv.clear_spaces(dst)
        self.n_live.fill_(E0 + n)

    # ------------------------------------------------------------------ writing: PLACEHOLDER (tests only)
    @torch.no_grad()
    def write(self, x, values, expert_ids=None, slots=None, force=False):
        """PLACEHOLDER (the write path is not built): store blobs with key = address(x), content = values. expert_ids
        None -> global; else (n,) long: -1 = the shared expert, e >= 0 = routed expert e. slots: indices within the
        space; None = (global only) the first free slots by a stable sort of the occupied flags (a full space is
        overwritten from slot 0: no eviction policy). No selection, merge, budget, full-cell or norm rule. Before fit()
        only with force=True."""
        assert self._fitted or force, "BlobMemory.write before fit(): the keys would move at the fit (force=True)"
        k = self.address(x.detach().float()).float()
        n, dev = k.size(0), k.device
        if expert_ids is None:
            space, name = self.glob, "g"
            base = torch.zeros(n, dtype=torch.long, device=dev)
            if slots is None:
                slots = self.glob.occ.to(torch.uint8).argsort(stable=True)[:n]
        else:
            assert slots is not None, "BlobMemory.write: private writes need explicit slots"
            space, name = self.priv, "p"
            e = torch.as_tensor(expert_ids, device=dev).long()
            base = torch.where(e < 0, torch.full_like(e, self.shared_idx), e) * self.P
        flat = base + torch.as_tensor(slots, device=dev).long()
        now = self.now.expand(n).contiguous()
        space.key.view(-1, self.kd).index_copy_(0, flat, k)
        space.value.view(-1, self.d).index_copy_(0, flat, values.detach().to(space.value.dtype))
        space.strength.view(-1).index_fill_(0, flat, 0.0)
        for nm in ("birth", "t_key", "clock_v", "clock_s", "touch"):
            getattr(space, nm).view(-1).index_copy_(0, flat, now)
        for nm in ("uses", "merges", "mature"):
            getattr(space, nm).view(-1).index_fill_(0, flat, 0)
        self.reset_opt_state(name, flat)
        space.occ.view(-1).index_fill_(0, flat, True)
        if space is self.glob and self.xfull is not None:
            self.xfull.index_copy_(0, flat, ((x.detach().float() - self.mu) / self.sigma).to(self.xfull.dtype))
        if space is self.glob and self.phase2:
            space.spawn_key.index_copy_(0, flat, k)
        space.refresh()
        if space is self.glob:
            if self._fitted:
                self.assign_cells(flat)
            else:
                self.rebuild_cells()
        return flat

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def fill(self):
        live = torch.arange(self.priv.occ.size(0), device=self.n_live.device) < self.n_live
        po = self.priv.occ.float().mean(-1)
        return dict(global_fill=self.glob.occ.float().mean(),
                    private_fill=(po * live).sum() / live.sum().clamp(min=1),
                    shared_fill=po[self.shared_idx])

    @torch.no_grad()
    def health(self):
        """The trainer's health line (--health-every): gate open rates per expert (mean / quantiles), the gate's
        learning signal, promotions per step, the waiting-area (probation) fill, blob lifetimes at eviction, the
        eval-write guard counter. Resets its own accumulators (independent of metrics())."""
        out = dict(gm_guard_violations=self.guard_violations)
        if not self.phase2:
            return out
        E = int(self.n_live)
        if self.gate is not None:
            g = self.gate
            r = (g.rate_sum / g.rate_n.clamp(min=1))[:E]
            ok = g.rate_n[:E] > 0
            if bool(ok.any()):
                rr = r[ok]
                q = [float(v) for v in torch.quantile(rr, torch.tensor([0.1, 0.5, 0.9], device=rr.device))]
                out.update(gm_open_rate_mean=float(rr.mean()), gm_open_rate_q10=q[0], gm_open_rate_q50=q[1],
                           gm_open_rate_q90=q[2], gm_open_rate_pinned0=float((rr == 0).float().mean()),
                           gm_open_rate_ge50=float((rr >= 0.5).float().mean()))
            g.rate_sum.zero_()
            g.rate_n.zero_()
            s1, s2, n, ls = (float(v) for v in self._hy)
            if n:
                out.update(gm_gate_y_mean=s1 / n, gm_gate_y_std=max(0.0, s2 / n - (s1 / n) ** 2) ** 0.5,
                           gm_gate_rows=n, gm_lam_pos_frac=float((g.lam[:E] > 0).float().mean()))
            self._hy.zero_()
        st = max(1, self._hsteps)
        for k, v in self._hcount.items():
            out[f"gm_{k}_per_step"] = float(v) / st
        self._hcount, self._hsteps = {}, 0
        if self.prob is not None:
            out["gm_waiting_fill"] = float(self.prob.fill())
        if self._hevict:
            ea = torch.cat(self._hevict)
            ea = ea[ea >= 0]
            if ea.numel():
                out.update(gm_lifetime_med=float(ea.median()), gm_lifetime_n=int(ea.numel()))
        self._hevict = []
        return out

    @torch.no_grad()
    def diag_snapshot(self):
        """--diag-every: device tensors (no sync, no state change, no reset of any accumulator) for the diag line."""
        out = {}
        g, p = self.glob, self.priv
        live = torch.arange(p.occ.size(0), device=self.n_live.device) < self.n_live
        out["fill_global"] = g.occ.float().mean()
        out["fill_private"] = (p.occ.float().mean(-1) * live).sum() / live.sum().clamp(min=1)
        out["fill_shared"] = p.occ[self.shared_idx].float().mean()
        if self.prob is not None:
            out["waiting_size"] = self.prob.occ.float().sum()
        edges = torch.tensor([100.0, 300.0, 1000.0, 3000.0, 10000.0], device=g.birth.device)
        age = (self.now - g.birth).float()
        b = torch.bucketize(age, edges)
        out["age_hist_global"] = torch.zeros(6, device=age.device).index_add_(0, b, g.occ.float())
        if self._hevict:
            ea = torch.cat(self._hevict)
            out["lifetime_hist"] = torch.zeros(7, device=ea.device).index_add_(
                0, torch.where(ea >= 0, torch.bucketize(ea, edges) + 1, torch.zeros_like(ea, dtype=torch.long)),
                torch.ones_like(ea))[1:]
        for k, v in self._wcount.items():
            if torch.is_tensor(v):
                out[f"since_eval_{k}"] = v.float()
        if self.last is not None:
            Ag, Ap = self.last["A_g"], self.last["A_p"]
            out["read_d1_global_mean"] = self.last["d1_g"].float().mean()
            if self.last["d1_p"].numel():
                out["read_d1_private_mean"] = self.last["d1_p"].float().mean()
            out["read_frac"] = self.last["read_frac"].float()
            if Ap.numel():
                out["global_read_share"] = Ag.float().mean() / (Ag.float().mean() + Ap.float().mean()).clamp(min=1e-12)
        if self._stats_step:
            st = torch.stack(self._stats_step)
            out["open_rate_by_call"] = (st[:, 0] + st[:, 2]) / (st[:, 1] + st[:, 3]).clamp(min=1)
            out["explore_share"] = st[:, 5].sum() / (st[:, 1] + st[:, 3]).sum().clamp(min=1)
            out["buffer_overflow"] = st[:, 4].sum()
        if self.gate is not None:
            gt = self.gate
            E = self.n_live
            idx = torch.arange(gt.rate_sum.numel(), device=gt.rate_sum.device)
            grp = (idx * 8 // E.clamp(min=1)).clamp(max=7)
            ok = idx < E
            rs = torch.zeros(8, device=idx.device).index_add_(0, grp, gt.rate_sum * ok)
            rn = torch.zeros(8, device=idx.device).index_add_(0, grp, gt.rate_n * ok)
            out["open_rate_by_expert_group"] = rs / rn.clamp(min=1)
            out["lambda_mean"] = (gt.lam * ok).sum() / ok.sum().clamp(min=1)
            out["lambda_pos_frac"] = ((gt.lam > 0) & ok).float().sum() / ok.sum().clamp(min=1)
            s1, s2, n = self._hy[0], self._hy[1], self._hy[2]
            out["gate_signal_mean"] = s1 / n.clamp(min=1)
            out["gate_signal_var"] = (s2 / n.clamp(min=1) - (s1 / n.clamp(min=1)) ** 2).clamp(min=0)
            out["gate_open_bias"] = gt.open_bias.float()
        if self.phase2:
            out["kernel_m_global"], out["kernel_m_private"] = self.gapm_g.float(), self.gapm_p.float()
            out["kappa"] = self.kappa.float()
        return out

    def metrics(self, step=None):
        """Logging (eval steps): the broken-if metrics the memory can measure, fill / lifetime per space."""
        from blob_write import gm_metrics
        return gm_metrics(self, step)

    def nbytes(self):
        return sum(b.numel() * b.element_size() for b in self.buffers())
