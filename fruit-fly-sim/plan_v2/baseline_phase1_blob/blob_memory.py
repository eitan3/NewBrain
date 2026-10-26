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

    FLOAT = ("key", "opt_v", "strength")
    LONG = ("opt_n", "birth", "t_key", "clock_v", "clock_s", "touch", "uses", "merges", "mature")

    def __init__(self, lead, d, kd=64, value_dtype=torch.float32, cells=False):
        super().__init__()
        self.d, self.kd, self.value_dtype, self.has_cells = d, kd, value_dtype, cells
        z = lambda *s, dt=torch.float32: torch.zeros(*lead, *s, dtype=dt)
        self.register_buffer("key", z(kd))
        self.register_buffer("value", z(d, dt=value_dtype))
        self.register_buffer("opt_v", z())
        self.register_buffer("strength", z())
        for n in self.LONG:
            self.register_buffer(n, z(dt=torch.long))
        self.register_buffer("occ", z(dt=torch.bool))
        if cells:  # IVF: nearest / second-nearest cell, and the assigned cell (-1 = none / did not fit)
            self.register_buffer("cell_pref", torch.full((*lead, 2), -1, dtype=torch.long))
            self.register_buffer("cell", torch.full(tuple(lead), -1, dtype=torch.long))
        self.register_buffer("key_bf", z(kd, dt=torch.bfloat16), persistent=False)
        self.register_buffer("ksq", torch.full(tuple(lead), INF), persistent=False)

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for n in self.FLOAT:
            setattr(self, n, getattr(self, n).float())
        self.value = self.value.to(self.value_dtype)
        self.key_bf = self.key_bf.to(torch.bfloat16)
        self.ksq = self.ksq.float()
        return self

    def persistent_names(self):
        return self.FLOAT + ("value",) + self.LONG + ("occ",) + (("cell_pref", "cell") if self.has_cells else ())

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
            setattr(self, n, torch.cat([t, t.new_full((n_lead0 - S, *t.shape[1:]), -1 if n.startswith("cell") else 0)]))
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
    pref = ((k.pow(2).sum(-1)[:, None] - 2 * k @ c.t()) + c.pow(2).sum(-1)[None]).topk(
        min(2, c.size(0)), -1, largest=False).indices
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
def ivf_candidates(q, centroids, table, kr, ksq, probe, q_cap, cell_topk, M, workspace=None):
    """IVF shortlist: q (N, kd) probes its `probe` nearest cells; the (query, probe) pairs are laid out per cell with a
    static capacity q_cap (farthest probes dropped first); one bf16 bmm on residuals to the cell centroid
    [cells, q_cap, kd] x [cells, kd, cell_cap]; bf16 top-cell_topk per pair; the best M per query by bf16 score.
    Returns (cand (N, M) entry ids, -1 = none; the probe-overflow rate). workspace(name, shape, dtype): optional static
    buffers. No host sync."""
    Cn, Cc, kd = table.size(0), table.size(1), q.size(1)
    n_p = min(probe, Cn)
    N, dev = q.size(0), q.device
    ws = workspace or (lambda name, shape, dt: torch.empty(shape, dtype=dt, device=dev))
    c = centroids
    pc = (c.pow(2).sum(-1)[None] - 2 * q @ c.t()).topk(n_p, -1, largest=False).indices   # (N, n_p), nearest first
    cellf = pc.flatten()
    order = (cellf * n_p + torch.arange(n_p, device=dev).repeat(N)).argsort(stable=True)  # by (cell, probe rank)
    cs = cellf[order]
    cnt = torch.zeros(Cn, dtype=torch.long, device=dev).index_add_(0, cs, torch.ones_like(cs))
    pos = torch.arange(N * n_p, device=dev) - (cnt.cumsum(0) - cnt)[cs]
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
def exact_candidates(q, key_bf, ksq, M, chunk=1024):
    """Brute-force shortlist: bf16 scores |k|^2 - 2 q.k over every entry in query chunks -> the best M ids (N, M)."""
    out = []
    step = chunk if chunk and chunk > 0 else max(q.size(0), 1)
    for a in range(0, q.size(0), step):
        s = ksq[None] - 2 * (q[a:a + step].to(torch.bfloat16) @ key_bf.t()).float()
        out.append(s.topk(min(M, key_bf.size(0)), -1, largest=False).indices)
    return torch.cat(out) if out else q.new_zeros(0, min(M, key_bf.size(0)), dtype=torch.long)


@torch.no_grad()
def rerank(q, key, ok, cand, k):
    """Exact fp32 distances of candidates cand (N, M) (-1 = none) in a key table (n, kd) with ok (n,) -> top-k (ids,
    d2 ascending; +inf = none)."""
    safe = cand.clamp(min=0)
    d2 = (q[:, None, :] - key[safe].float()).pow(2).sum(-1)
    d2 = torch.where((cand >= 0) & ok[safe], d2, torch.full_like(d2, INF))
    top = d2.topk(min(k, d2.size(1)), -1, largest=False)
    return safe.gather(1, top.indices), top.values


class ProbationStore(nn.Module):
    """The probation store of the write path ("keep what comes back": probation, then promotion; flag-gated). A FIFO
    ring of n = ceil(rate x life) entries (rate = candidates per optimizer step, life ~ 3,000 steps): each entry holds a
    64-d key and a 64-d projected gradient (bf16), the step it entered, how often it came back (hits) and when last,
    its source (-1 = global / shared, e >= 0 = a routed expert) and an occupied flag. Lookup: exact (bf16 shortlist +
    fp32 re-rank) or IVF over the global space's cells (rebuild() after pushes / a refit). Accessors for promotion:
    get / take (read and free) / mark_hit / drop / age. No policy (the thresholds come later)."""

    def __init__(self, n, kd=64, cells=1024, cell_cap=0):
        super().__init__()
        self.n, self.kd = n, kd
        self.cell_cap = cell_cap or max(16, math.ceil(4 * n / max(cells, 1)))
        self.register_buffer("key", torch.zeros(n, kd, dtype=torch.bfloat16))
        self.register_buffer("pgrad", torch.zeros(n, kd, dtype=torch.bfloat16))
        self.register_buffer("step", torch.zeros(n, dtype=torch.long))
        self.register_buffer("hits", torch.zeros(n, dtype=torch.long))
        self.register_buffer("last", torch.zeros(n, dtype=torch.long))
        self.register_buffer("src", torch.full((n,), -1, dtype=torch.long))
        self.register_buffer("occ", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("pref", torch.full((n, 2), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))       # entries pushed so far (FIFO position)
        self.register_buffer("ksq", torch.full((n,), INF), persistent=False)
        self.register_buffer("table", torch.full((cells, self.cell_cap), -1, dtype=torch.long), persistent=False)
        self.register_buffer("table_kr", torch.zeros(cells, self.cell_cap, kd, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("table_ksq", torch.full((cells, self.cell_cap), INF, dtype=torch.bfloat16),
                             persistent=False)
        self._ptr = 0                   # host mirror of ptr (static FIFO positions, no sync)
        self.stats = {}

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.key, self.pgrad = self.key.to(torch.bfloat16), self.pgrad.to(torch.bfloat16)
        self.ksq = self.ksq.float()
        self.table_kr, self.table_ksq = self.table_kr.to(torch.bfloat16), self.table_ksq.to(torch.bfloat16)
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._ptr = int(state_dict.get(prefix + "ptr", self.ptr))
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        self.ksq = torch.where(self.occ, self.key.float().pow(2).sum(-1), torch.full_like(self.ksq, INF))

    @torch.no_grad()
    def push(self, key, pgrad, step, src=None, centroids=None):
        """Append n candidates FIFO (the oldest entries are overwritten). Returns their entry ids. centroids: assign
        their preferred IVF cells (rebuild() makes them visible to the IVF lookup; the exact lookup sees them at once)."""
        n = key.size(0)
        dev = self.key.device
        slots = (torch.arange(n, device=dev) + self._ptr) % self.n
        self._ptr += n
        self.ptr.fill_(self._ptr)
        stp = torch.as_tensor(step, device=dev).long().expand(n).contiguous()
        kb = key.to(torch.bfloat16)
        self.key.index_copy_(0, slots, kb)
        self.pgrad.index_copy_(0, slots, pgrad.to(torch.bfloat16))
        self.step.index_copy_(0, slots, stp)
        self.last.index_copy_(0, slots, stp)
        self.hits.index_fill_(0, slots, 0)
        self.src.index_copy_(0, slots, torch.full((n,), -1, dtype=torch.long, device=dev) if src is None
                             else torch.as_tensor(src, device=dev).long())
        self.occ.index_fill_(0, slots, True)
        self.ksq.index_copy_(0, slots, kb.float().pow(2).sum(-1))
        if centroids is not None:
            self.pref.index_copy_(0, slots, nearest_cells(kb.float(), centroids))
        return slots

    @torch.no_grad()
    def rebuild(self, centroids, reassign=True):
        """The IVF table over the occupied entries (after pushes, or a centroid refit: reassign recomputes every
        entry's preferred cells)."""
        if reassign:
            self.pref.copy_(nearest_cells(self.key.float(), centroids))
        t = ivf_table(self.key, self.occ, self.pref, centroids, self.cell_cap)
        self.table, self.table_kr, self.table_ksq = t["table"], t["kr"], t["ksq"]
        self.stats = dict(overflow=t["overflow"], spilled=t["spilled"])

    @torch.no_grad()
    def lookup(self, q, k, mode="exact", centroids=None, probe=16, q_cap=192, cell_topk=32, chunk=1024):
        """The k nearest occupied entries of x_hat queries q (N, kd) -> (ids (N, k), d2 (N, k) ascending; +inf =
        none). mode "exact": a bf16 shortlist over every entry; "ivf": over the table of the last rebuild()."""
        q = q.float()
        M = max(2 * k, 8)
        if mode == "ivf":
            cand, ov = ivf_candidates(q, centroids, self.table, self.table_kr, self.table_ksq, probe, q_cap,
                                      cell_topk, M)
            self.stats["probe_overflow"] = ov
        else:
            cand = exact_candidates(q, self.key, self.ksq, M, chunk)
        return rerank(q, self.key, self.occ, cand, k)

    @torch.no_grad()
    def mark_hit(self, ids, step, ok=None):
        """Candidates came back near entries ids (n,) (ok: an optional mask): hits += 1, last = step."""
        inc = torch.ones_like(ids) if ok is None else ok.long()
        self.hits.index_add_(0, ids, inc)
        now = torch.as_tensor(step, device=ids.device).long().expand(ids.shape)
        self.last.scatter_reduce_(0, ids, torch.where(inc > 0, now, torch.full_like(ids, -1)), "amax")

    @torch.no_grad()
    def get(self, ids):
        return dict(key=self.key[ids].float(), pgrad=self.pgrad[ids].float(), step=self.step[ids], hits=self.hits[ids],
                    last=self.last[ids], src=self.src[ids], occ=self.occ[ids])

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
        return torch.as_tensor(now, device=self.step.device) - self.step


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


def kernel_weights(q, keys, valid, tau, h0, c=1.0):
    """q (n, kd); keys (n, k, kd) detached; valid (n, k). d_j = |q - k_j|^2, d_1 = the nearest valid one,
    w = softmax_j(-(d_j - d_1) / ((d_1 + c h0^2) tau)) over the valid j, A = exp(-d_1 / (2 h0^2)). Returns w (n, k),
    A (n,), d1 (n,); a row without a valid candidate has w = 0 and A = 0 (finite gradients)."""
    d = (q[:, None, :] - keys).pow(2).sum(-1)
    any_ = valid.any(-1)
    inf = torch.full_like(d, INF)
    d1 = torch.where(valid, d, inf).amin(-1)
    d1 = torch.where(any_, d1, torch.zeros_like(d1))
    dv = torch.where(valid, d, d1[:, None].expand_as(d))
    h2 = h0 * h0
    logits = torch.where(valid, -(dv - d1[:, None]) / ((d1[:, None] + c * h2) * tau), -inf)
    logits = torch.where(any_[:, None], logits, torch.zeros_like(logits))
    w = torch.where(any_[:, None], logits.softmax(-1), torch.zeros_like(logits))
    A = torch.where(any_, torch.exp(-d1 / (2 * h2)), torch.zeros_like(d1))
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
                 track_xf=False, probation_rate=0.0, probation_life=3000, seed=1234):
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
        self.Emax = max(int(max_experts or 0), n_exp)
        self.glob = BlobSpace((G,), d, kd, cells=True)
        self.priv = BlobSpace((self.Emax + 1, P), d, kd, value_dtype=torch.bfloat16 if priv_bf16 else torch.float32)
        n_prob = math.ceil(probation_rate * probation_life) if probation_rate > 0 else 0
        self.prob = ProbationStore(n_prob, kd, cells) if n_prob else None    # (the write path's probation; flag)
        # the only memory parameter (flag, default off): it is trained by the brain's optimizer like any other gain
        self.inj_scale = nn.Parameter(torch.ones(d)) if inj_scale else None
        gen = torch.Generator().manual_seed(seed)  # (forked: the global RNG is not touched)
        self.register_buffer("proj", torch.randn(d, kd, generator=gen) / math.sqrt(d))
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
        self.register_load_state_dict_post_hook(lambda mod, keys: mod.post_load())

    # ------------------------------------------------------------------ layout / dtypes / checkpoint
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
        self._vg, self._touched, self._ws = {}, {}, {}
        return self

    def get_extra_state(self):
        st = dict(self._gen_states)
        st.update({n: g.get_state() for n, g in self._gens.items()})
        return dict(gens=st)

    def set_extra_state(self, st):
        self._gens, self._gen_states = {}, dict((st or {}).get("gens", {}))

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
                g.manual_seed(self.seed + {"mask": 1, "sr": 2, "kmeans": 3}[name])
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
        if not self.collect or self._fitted or not self.training:
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
        if probation and self.prob is not None and self._fitted:
            self.prob.rebuild(self.centroids)

    # ------------------------------------------------------------------ search (no gradient)
    @torch.no_grad()
    def _rerank(self, q, cand, k):
        """Exact fp32 distances of global candidates cand (N, M) (-1 = none) -> top-k (idx, d2 ascending; +inf = none)."""
        return rerank(q, self.glob.key, self.glob.occ, cand, k)

    @torch.no_grad()
    def search_global(self, q):
        """q (N, kd) -> the top-`cand` (idx (N, cand) long, d2 (N, cand) fp32 ascending, +inf = none)."""
        q = q.float()
        k = min(self.cand, self.G)
        return self._search_brute(q, k) if self.search == "brute" else self._search_ivf(q, k)

    @torch.no_grad()
    def _search_brute(self, q, k):
        """Reference / tests only: chunked bf16 scores over every slot, the bf16 top-max(2k, 8) re-ranked in fp32."""
        sp = self.glob
        return self._rerank(q, exact_candidates(q, sp.key_bf, sp.ksq, max(2 * k, 8), self.brute_chunk), k)

    @torch.no_grad()
    def _search_ivf(self, q, k):
        ws = lambda name, shape, dt: self._workspace(name, shape, dt, q.device)
        cand, ov = ivf_candidates(q, self.centroids, self.table, self.table_kr, self.table_ksq, self.probe, self.q_cap,
                                  self.cell_topk, 2 * k, ws)
        self.stats = dict(probe_overflow=ov, cell_overflow=self.cell_overflow, cell_spilled=self.cell_spilled)
        return self._rerank(q, cand, k)

    # ------------------------------------------------------------------ the probation store (write path; flag)
    @torch.no_grad()
    def probation_push(self, xh, pgrad, src=None):
        """Append write candidates (x_hat keys (n, kd), projected gradients (n, kd)) FIFO, stamped with `now`."""
        return self.prob.push(xh, pgrad, self.now, src, self.centroids if self._fitted else None)

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
    def _sink(self, name, table):
        def sink(rows, g):
            vg = self._vg.get(name)
            if vg is None or vg.shape != table.shape or vg.device != g.device:
                vg = self._vg[name] = torch.zeros(table.shape, dtype=torch.float32, device=g.device)
                self._touched[name] = torch.zeros(table.size(0), dtype=torch.bool, device=g.device)
            r = rows.reshape(-1)
            vg.index_add_(0, r, g.reshape(-1, g.size(-1)).float())
            self._touched[name].index_fill_(0, r, True)
        return sink

    def _read(self, q, space, flat, h0):
        """q (n, kd), flat (n, k) flat slot ids -> u = sum_j w_j v_j (n, D) in q's dtype, A (n,), d1 (n,)."""
        name = "g" if space is self.glob else "p"
        keys = space.key.view(-1, self.kd)[flat].to(q.dtype)              # detached (buffers)
        valid = space.occ.view(-1)[flat]
        w, A, d1 = kernel_weights(q if self.match_grad else q.detach(), keys, valid, self.tau, h0.to(q.dtype),
                                  self.kernel_c)
        table = space.value.view(-1, self.d)
        if torch.is_grad_enabled() and w.requires_grad:
            return BagRead.apply(w, table, flat, self._sink(name, table)), A, d1
        return _bag(w, table, flat), A, d1

    def begin_forward(self):
        """A new model forward: drop the cached selections and the per-token xf sums of the previous one."""
        self._cache, self._seen = {}, set()
        self.xf_sum = self.xf_cnt = None

    def __call__(self, xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=None, pair_valid=None):
        return self.inject(xf, buf, x, e_k, p_k, t_k, call=call, pair_valid=pair_valid)

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
                    if self.training:
                        self._ring_add(qd, 0)
                    self._count(sel, t_k)
                if call is not None and torch.is_grad_enabled():
                    self._cache[call] = sel
            kg, kp = self.k_g, self.k_p
            ug, Ag, dg = self._read(q, self.glob, sel["gi"][:, :kg], self.h0_g)      # once per token
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
        """Use count + touch step of the nearest blob of every masked read of an occupied slot (no sync)."""
        now = self.now
        N = sel["ms"].size(0)
        g_on = sel["ms"] | (torch.zeros(N, device=t_k.device).index_add_(0, t_k, sel["mr"].float()) > 0)
        for space, flat, m in ((self.glob, sel["gi"][:, 0], g_on), (self.priv, sel["pf"][:, 0], sel["mr"]),
                               (self.priv, sel["sf"][:, 0], sel["ms"])):
            ok = space.occ.view(-1)[flat] & m
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
            v_new = torch.where(t, b2 * v + (1 - b2) * g.pow(2).mean(-1), v)
            upd = g * (lr / (v_new.sqrt() + eps))[:, None]
            if clip and clip > 0:
                upd = upd * (clip / upd.norm(dim=-1, keepdim=True).clamp(min=clip))
            val = space.value.view(-1, self.d)
            new = val.float() - upd
            new = stochastic_bf16(new, self._generator("sr", val.device)) if val.dtype == torch.bfloat16 else new
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
        """Step-end hook: the content update, the IVF refit every `refit_every` steps (0 = never); later the write /
        merge / evict decisions (at most one host sync)."""
        self.apply_value_grads()
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
        if split:
            self.priv.copy_spaces(torch.as_tensor(src, device=dev).long(), dst)
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

    def nbytes(self):
        return sum(b.numel() * b.element_size() for b in self.buffers())
