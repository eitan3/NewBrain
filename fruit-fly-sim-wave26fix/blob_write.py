"""Step-end maintenance of the blob memory (plan_v2/25d rev 2 sections 8-10 + the owner's write decision; phase 2).

All functions take the BlobMemory and run after the optimizer step (BlobMemory.after_step), with at most ONE host sync
per step (the sizes of the step's write lists, read together).

  * writes, "keep what comes back" (DEFAULT, owner): each micro-batch's candidates (the top r% of tokens by
    entropy-normalised surprise, per-sequence quota) are looked up in the probation store (top-1); a candidate whose
    top-1 is a held entry within promo_gamma h0^2 AND whose 64-d gradient sketch agrees with the entry's
    (cos > promo_theta) PROMOTES it: the entry is taken out and a global blob is written (key = the fixed projection of
    the token's pass-averaged xf, plus that xf standardised at model width for the optional re-rank; content
    v = 0.1 |xf| q / |q| with q = -(the summed gradient at the token's expert inputs), capped); the other candidates
    enter the probation FIFO. `--gm-write direct`: the top write_w candidates are written at once (comparison).
  * private writes: a per-expert credit (+private_rate per step, ~0.256) fires a write of that expert's best candidate
    (the highest-surprise candidate token it processed in-window; the shared expert: the best candidate overall) when
    the candidate was promoted (direct mode: always); content from that pair's own gradient.
  * merge (global and private): the new blob's nearest live blob j1 with d1 < gamma h0^2 AND cos(q_new, v_j1) > theta_m
    AND move budget left -> merge into j1 (key step eta = max(1/(n+2), 1/4) capped at min(sigma, 0.5 h0); content
    v += v_new / (n + 2), then the norm cap; n <- min(n + 1, 8)); otherwise spawn. Candidates of one step within
    gamma h0^2 of an earlier (higher-surprise) one are suppressed.
  * eviction: free slots first; else the lowest strength among unprotected slots, never one touched this step; blobs
    younger than protect_age are protected while protected blobs hold <= protect_frac of the space (above it the
    oldest lose protection first). Strength = EMA of the positive benefit, half-life 4,000 steps (a dense decay each
    step + the step's credits).
  * renewal: confirmed reads (G > 0, top 25% of the micro-batch's open rows, clear winner w1 >= omega, d2/d1 >= mu_m,
    A >= A_min, content test cos(-g + kappa r, v_j1) > 0.3) move their blob toward the segment mean of the confirming
    queries: alpha = max(0.3, 1/(n+1)), |step| <= min(sigma, 0.5 h0), <= 1 move per blob per step, <= 8 per 1,000
    steps (merges included), <= 256 moving global slots and <= 2 per private space per step.
"""
import math

import torch
import torch.nn.functional as F

INF = float("inf")


def _segments(keys):
    """keys (n,) long -> (order, segment id per sorted row, first-row flag); no sync."""
    order = keys.argsort(stable=True)
    ks = keys[order]
    first = torch.ones_like(ks, dtype=torch.bool)
    first[1:] = ks[1:] != ks[:-1]
    seg = first.long().cumsum(0) - 1
    return order, ks, seg, first


def cap_rows(v, lim):
    """Rows of v (n, d) scaled down to norm <= lim (scalar tensor or (n,))."""
    lim = torch.as_tensor(lim, device=v.device, dtype=torch.float32)
    lim = lim[..., None] if lim.dim() else lim
    return v * (lim / v.norm(dim=-1, keepdim=True).clamp(min=1e-30)).clamp(max=1.0)


@torch.no_grad()
def victims(bm, space, flat_base, n, step):
    """The weakest slot per space row. space: BlobSpace; flat_base (m,) the first flat slot of each target space (global:
    one space, m = 1 per request); n: slots wanted per row. Returns flat slot ids (m, n) and an ok mask."""
    occ = space.occ.view(-1)
    P = space.occ.size(-1) if space.occ.dim() == 2 else space.occ.size(0)
    rows = flat_base[:, None] + torch.arange(P, device=occ.device)[None]                   # (m, P)
    o = occ[rows]
    age = step - space.birth.view(-1)[rows]
    prot = o & (age < bm.protect_age)
    cap = int(bm.protect_frac * P)
    # above the limit the OLDEST protected lose protection first
    rank = torch.where(prot, age, torch.full_like(age, -1)).argsort(-1, descending=True).argsort(-1)
    n_prot = prot.sum(-1, keepdim=True)
    prot = prot & ~((n_prot > cap) & (rank < (n_prot - cap)))
    score = torch.where(o, space.strength.view(-1)[rows], torch.full_like(rows, -1, dtype=torch.float32) * INF)
    blocked = (prot | (o & (space.touch.view(-1)[rows] >= step)))
    score = torch.where(blocked, torch.full_like(score, INF), score)
    top = score.topk(min(n, P), -1, largest=False)
    return rows.gather(1, top.indices), top.values < INF


@torch.no_grad()
def write_blobs(bm, name, keys, xfull, vals, qdir, spaces, step, active=None):
    """Spawn-or-merge n blobs (all tensors (n, ...); spaces (n,) the private space of each, ignored for the global).
    keys: x_hat (n, kd); xfull: standardised xf (n, d) (global re-rank store); vals: initial contents (n, d); qdir: the
    content direction for the merge test (n, d); active (n,): the rows to write (the others are ignored). Rows are in
    priority order (the earlier wins a suppression / a merge target). Returns counts (tensors, no sync)."""
    sp = bm.space(name)
    n = keys.size(0)
    dev = keys.device
    if not bm.training:                                                     # (guard: never outside training)
        bm.guard_violations += 1
    if n == 0:
        return dict(spawned=torch.zeros((), device=dev), merged=torch.zeros((), device=dev),
                    _stored=torch.zeros(0, dtype=torch.bool, device=dev))
    h2 = (bm.h0_g if name == "g" else bm.h0_p) ** 2
    sig2 = bm.sigma2_g if name == "g" else bm.sigma2_p
    gamma = bm.gamma_g if name == "g" else bm.gamma_p
    # within-step suppression: a later (lower-priority) candidate of the same space within gamma h0^2 of an earlier one
    same = (spaces[:, None] == spaces[None, :]) if name == "p" else torch.ones(n, n, dtype=torch.bool, device=dev)
    dd = torch.cdist(keys, keys).pow(2)
    act = torch.ones(n, dtype=torch.bool, device=dev) if active is None else active
    sup = ((dd < gamma * h2) & same & act[:, None] & act[None, :]).triu(1).any(0) | ~act
    # the nearest live blob
    if name == "g":
        j, d = bm.search_global(keys)
        j1, d1 = j[:, 0].clamp(min=0), d[:, 0]                         # (r6 C2: -1 = none; d1 = inf never merges)
    else:
        cand = bm.search_shared_like(keys, spaces)
        j, d = bm._rerank_priv(keys, spaces, cand, 1)
        j1, d1 = j[:, 0], d[:, 0]
    V = sp.value.view(-1, bm.d)
    cos = F.cosine_similarity(qdir.float(), V[j1].float(), dim=-1)
    win_ok = moves_left(bm, sp, j1, step)
    merge = (~sup) & torch.isfinite(d1) & (d1 < gamma * h2) & (cos > bm.theta_m) & win_ok & sp.occ.view(-1)[j1]
    # one merge per target slot per step: the first (highest-priority) candidate wins, the others spawn
    first = torch.full((sp.occ.numel(),), n, dtype=torch.long, device=dev)
    first.scatter_reduce_(0, j1, torch.where(merge, torch.arange(n, device=dev), torch.full_like(j1, n)), "amin")
    merge = merge & (first[j1] == torch.arange(n, device=dev))
    _merge(bm, sp, name, j1, merge, keys, vals, step, sig2, h2)
    spawn = (~sup) & (~merge)
    # victims: global one space; private one per target space (each private request has its own space)
    if name == "g":
        vic, ok = victims(bm, sp, torch.zeros(1, dtype=torch.long, device=dev), n, step)
        vic, ok = vic[0], ok[0]
        rank = spawn.long().cumsum(0) - 1                                    # the i-th spawn takes the i-th victim
        slot = vic[rank.clamp(min=0)]
        go = spawn & ok[rank.clamp(min=0)]
    else:
        vic, ok = victims(bm, sp, spaces * bm.P, 1, step)
        slot, go = vic[:, 0], spawn & ok[:, 0]
    ev = go & sp.occ.view(-1)[slot]                                          # evictions: their ages (logging)
    bm._evict.append(torch.where(ev, (step - sp.birth.view(-1)[slot]).float(), torch.full_like(slot, -1).float()))
    bm._evict = bm._evict[-32:]
    bm._hevict = (bm._hevict + [bm._evict[-1]])[-32:]
    _spawn(bm, sp, name, slot, go, keys, xfull, vals, step)
    return dict(spawned=go.sum(), merged=merge.sum(), suppressed=(sup & act).sum(), blocked=(spawn & ~go).sum(),
                _stored=go | merge)


@torch.no_grad()
def moves_left(bm, sp, j, step):
    """The per-blob move budget (<= max_moves per window steps, merges included)."""
    start = sp.win_start.view(-1)[j]
    used = torch.where(step - start >= bm.move_window, torch.zeros_like(start), sp.moves_win.view(-1)[j])
    return used < bm.max_moves


@torch.no_grad()
def _bump_moves(bm, sp, j, mask, step):
    ws, mv = sp.win_start.view(-1), sp.moves_win.view(-1)
    stale = (step - ws[j]) >= bm.move_window
    # (j unique where mask is True)
    new_start = torch.where(mask & stale, torch.full_like(j, step), ws[j])
    new_cnt = torch.where(mask, torch.where(stale, torch.zeros_like(j), mv[j]) + 1, mv[j])
    jj = torch.where(mask, j, torch.full_like(j, -1))
    _put(ws, jj, new_start)
    _put(mv, jj, new_cnt)


def _put(t, idx, val):
    """t[idx] = val for the rows with idx >= 0 (unique among them), no sync: the other rows are redirected to the first
    valid row's slot WITH that row's value (identical duplicate writes are deterministic); with no valid row they
    rewrite slot idx[0] with its own value."""
    if idx.numel() == 0:
        return
    ok = idx >= 0
    r0 = ok.long().argmax().view(1)                                       # (1-d indices: a 0-d index would sync)
    i0 = idx.clamp(min=0).index_select(0, r0)
    v = val.to(t.dtype)
    shape = (-1,) + (1,) * (v.dim() - 1)
    v0 = torch.where(ok.index_select(0, r0).view(shape), v.index_select(0, r0), t.index_select(0, i0))
    t.index_copy_(0, torch.where(ok, idx, i0), torch.where(ok.view(shape), v, v0))


@torch.no_grad()
def _merge(bm, sp, name, j1, merge, keys, vals, step, sig2, h2):
    n_j = sp.merges.view(-1)[j1].float()
    eta = torch.maximum(1.0 / (n_j + 2), torch.full_like(n_j, 0.25))
    K = sp.key.view(-1, bm.kd)
    dk = eta[:, None] * (keys - K[j1])
    lim = torch.minimum(sig2.sqrt(), 0.5 * h2.sqrt())
    dk = cap_rows(dk, lim) * merge[:, None]
    V = sp.value.view(-1, bm.d)
    vnew = V[j1].float() + vals.float() / (n_j + 2)[:, None]
    vnew = cap_rows(vnew, bm.v_cap_after * bm.xf_norm_med)
    jj = torch.where(merge, j1, torch.full_like(j1, -1))
    K.index_add_(0, j1, dk)
    _put(V, jj, vnew.to(V.dtype))
    _put(sp.merges.view(-1), jj, (sp.merges.view(-1)[j1] + 1).clamp(max=8))
    _put(sp.touch.view(-1), jj, torch.full_like(j1, step))
    _put(sp.t_key.view(-1), jj, (sp.t_key.view(-1)[j1].float() + eta * (step - sp.t_key.view(-1)[j1].float())).long())
    sp.path.view(-1).index_add_(0, j1, dk.norm(dim=-1))
    _bump_moves(bm, sp, j1, merge, step)
    _put(sp.opt_v.view(-1), jj, torch.zeros_like(n_j))                  # optimizer-state reset on merge
    _put(sp.opt_n.view(-1), jj, torch.zeros_like(j1))
    sp.refresh()
    if name == "g":
        bm.assign_cells(j1)


@torch.no_grad()
def _spawn(bm, sp, name, slot, go, keys, xfull, vals, step):
    jj = torch.where(go, slot, torch.full_like(slot, -1))
    K, V = sp.key.view(-1, bm.kd), sp.value.view(-1, bm.d)
    _put(K, jj, keys.float())
    _put(V, jj, cap_rows(vals.float(), bm.v_cap_spawn * bm.xf_norm_med).to(V.dtype))
    if name == "g" and bm.xfull is not None and xfull is not None:
        _put(bm.xfull, jj, xfull.to(bm.xfull.dtype))
    z = torch.zeros_like(slot)
    stp = torch.full_like(slot, step)
    for nm in ("birth", "t_key", "clock_v", "clock_s", "touch", "win_start"):
        _put(getattr(sp, nm).view(-1), jj, stp)
    for nm in ("uses", "merges", "mature", "moves_win", "opt_n"):
        _put(getattr(sp, nm).view(-1), jj, z)
    for nm in ("strength", "opt_v", "path"):
        _put(getattr(sp, nm).view(-1), jj, z.float())
    _put(sp.t_conf.view(-1), jj, torch.full_like(slot, -10 ** 9))
    _put(sp.occ.view(-1), jj, torch.ones_like(go))
    if name == "g":
        _put(sp.spawn_key.view(-1, bm.kd), jj, keys.float())
    sp.refresh()
    if name == "g":
        bm.assign_cells(slot)


@torch.no_grad()
def renew(bm, name, slots, q, gain, ok, step):
    """Confirmed reads -> key moves. slots (n,) flat slot ids, q (n, kd) the confirming queries, gain (n,) their
    curvature-corrected benefit, ok (n,) the confirmation mask (all tests passed). One move per blob per step (segment
    mean), budgets, per-space caps (global <= cap_g moving slots; private <= cap_p per space), lowest gain dropped."""
    sp = bm.space(name)
    n = slots.size(0)
    if n == 0:
        return torch.zeros((), device=slots.device)
    dev = slots.device
    key = torch.where(ok, slots, torch.full_like(slots, sp.occ.numel()))           # sentinel: not confirmed
    order, ks, seg, first = _segments(key)
    nseg = n
    qs = torch.zeros(nseg, bm.kd, device=dev).index_add_(0, seg, q[order].float() * ok[order, None])
    cnt = torch.zeros(nseg, device=dev).index_add_(0, seg, ok[order].float())
    gs = torch.zeros(nseg, device=dev).index_add_(0, seg, gain[order].float() * ok[order])
    sslot = torch.full((nseg,), -1, dtype=torch.long, device=dev).scatter_(       # (equal values per segment)
        0, seg, torch.where(ks < sp.occ.numel(), ks, torch.full_like(ks, -1)))
    valid = (cnt > 0) & (sslot >= 0)
    qbar = qs / cnt.clamp(min=1)[:, None]
    # per-space caps, by gain
    if name == "g":
        rank = torch.where(valid, gs, torch.full_like(gs, -INF)).argsort(descending=True).argsort()
        valid = valid & (rank < bm.cap_moves_g)
    else:
        sid = torch.where(valid, sslot // bm.P, torch.full_like(sslot, -1))
        oa = torch.where(valid, gs, torch.full_like(gs, -INF)).argsort(descending=True)        # gain desc, then
        o2 = oa[sid[oa].argsort(stable=True)]                                                     # by space (stable)
        s_sorted = sid[o2]
        f2 = torch.ones_like(s_sorted, dtype=torch.bool)
        f2[1:] = s_sorted[1:] != s_sorted[:-1]
        ar = torch.arange(nseg, device=dev)
        start = torch.where(f2, ar, torch.zeros_like(ar)).cummax(0).values
        r_in = torch.empty_like(o2).index_copy_(0, o2, torch.arange(nseg, device=dev) - start)
        valid = valid & (r_in < bm.cap_moves_p)
    j = sslot.clamp(min=0)
    valid = valid & moves_left(bm, sp, j, step)
    nj = sp.merges.view(-1)[j].float()
    alpha = torch.maximum(torch.full_like(nj, 0.3), 1.0 / (nj + 1))
    K = sp.key.view(-1, bm.kd)
    h2 = (bm.h0_g if name == "g" else bm.h0_p) ** 2
    sig2 = bm.sigma2_g if name == "g" else bm.sigma2_p
    dk = cap_rows(alpha[:, None] * (qbar - K[j]), torch.minimum(sig2.sqrt(), 0.5 * h2.sqrt())) * valid[:, None]
    K.index_add_(0, j, dk)
    jj = torch.where(valid, j, torch.full_like(j, -1))
    _put(sp.merges.view(-1), jj, (nj + 1).clamp(max=8).long())
    _put(sp.touch.view(-1), jj, torch.full_like(j, step))
    _put(sp.t_key.view(-1), jj, torch.full_like(j, step))
    _put(sp.t_conf.view(-1), jj, torch.full_like(j, step))
    mat = sp.mature.view(-1)[j]
    up = valid & (step - sp.t_mature.view(-1)[j] >= 64)
    _put(sp.mature.view(-1), torch.where(up, j, torch.full_like(j, -1)), (mat + 1).clamp(max=3))
    _put(sp.t_mature.view(-1), torch.where(up, j, torch.full_like(j, -1)), torch.full_like(j, step))
    sp.path.view(-1).index_add_(0, j, dk.norm(dim=-1))
    _bump_moves(bm, sp, j, valid, step)
    sp.refresh()
    if name == "g":
        bm.assign_cells(j)
    return valid.sum()


# ====================================================================== write candidates (between the loss and backward)
@torch.no_grad()
def select_writes(bm, logits, targets, idx=None, chunk=2048, ce_h=None):
    """Write candidates of this micro-batch: the top `cand_rate` of each sequence's tokens by the entropy-normalised
    surprise sigma = ce - C(H) (C an EMA of the mean CE per entropy bin, 32 bins; the current EMA is used, then
    updated), eligible = not EOT and >= 8 tokens after an EOT in the window. Also each expert's best candidate (the
    highest-surprise candidate it processed in-window; the shared expert: the best overall), for its private write.
    The gradient hooks of the forward read this selection during backward."""
    B, T = targets.shape
    N = B * T
    dev = targets.device
    if ce_h is not None:   # (--chunk-ce: computed by the chunked read-out, same math)
        ce, H, V = ce_h[0].reshape(N).float(), ce_h[1].reshape(N).float(), int(ce_h[2])
    else:
        lg = logits.reshape(N, -1)
        V = lg.size(1)
        ce = torch.empty(N, device=dev)
        H = torch.empty(N, device=dev)
        tf = targets.reshape(-1)
        for a in range(0, N, chunk):
            lp = F.log_softmax(lg[a:a + chunk].float(), -1)
            ce[a:a + chunk] = -lp.gather(1, tf[a:a + chunk, None]).squeeze(1)
            H[a:a + chunk] = -(lp.exp() * lp).sum(-1)
    nb = bm.ch_mean.numel()
    b = (H / math.log(V) * nb).long().clamp(0, nb - 1)
    sig = ce - bm.ch_mean[b]
    s_ = torch.zeros(nb, device=dev).index_add_(0, b, ce)
    c_ = torch.zeros(nb, device=dev).index_add_(0, b, torch.ones_like(ce))
    seen = c_ > 0
    first = bm.ch_n == 0
    bm.ch_mean.copy_(torch.where(seen, torch.where(first, s_ / c_.clamp(min=1),
                                                   bm.ch_mean.lerp(s_ / c_.clamp(min=1), 0.01)), bm.ch_mean))
    bm.ch_n.add_(seen.long())
    elig = torch.ones(B, T, dtype=torch.bool, device=dev)
    if idx is not None and bm.eot is not None:
        pos = torch.arange(T, device=dev).expand(B, T)
        last = torch.where(idx == bm.eot, pos, torch.full_like(pos, -10 ** 6)).cummax(1).values
        elig = (idx != bm.eot) & (pos - last >= 8)
    sc = torch.where(elig, sig.view(B, T), torch.full_like(sig.view(B, T), -INF))
    c = max(1, math.ceil(bm.cand_rate * T))
    top = sc.topk(min(c, T), -1)
    cand_tok = (torch.arange(B, device=dev)[:, None] * T + top.indices).flatten()
    cand_sig = top.values.flatten()
    valid = cand_sig > -INF
    W = cand_tok.numel()
    order = torch.where(valid, cand_sig, torch.full_like(cand_sig, -INF)).argsort(descending=True)
    rank = torch.empty_like(order).index_copy_(0, order, torch.arange(W, device=dev))
    tok2cand = torch.full((N + 1,), -1, dtype=torch.long, device=dev)
    tok2cand.index_copy_(0, torch.where(valid, cand_tok, torch.full_like(cand_tok, N)),
                         torch.where(valid, torch.arange(W, device=dev), torch.full_like(cand_tok, -1)))
    tok2cand = tok2cand[:N]
    S = bm.priv.occ.size(0)
    best = torch.full((S,), W, dtype=torch.long, device=dev)
    for e_k, t_k in bm._kept_pairs:
        cc = tok2cand[t_k]
        best.scatter_reduce_(0, e_k, torch.where(cc >= 0, rank[cc.clamp(min=0)], torch.full_like(cc, W)), "amin")
    best[S - 1] = 0                                                       # the shared expert: the best overall
    has = best < W
    chosen = torch.where(has & valid[order[best.clamp(max=W - 1)]], cand_tok[order[best.clamp(max=W - 1)]],
                         torch.full_like(best, -1))
    bm._wsel = dict(cand_tok=cand_tok, valid=valid, sig=cand_sig, tok2cand=tok2cand, chosen=chosen, N=N)
    bm._wstats.append(torch.stack([ce.mean(), sig.mean(), valid.float().sum()]))
    bm._wstats = bm._wstats[-64:]                                        # (r6: bounded; was never cleared)
    return bm._wsel


def capture_hooks(bm, call, bufo, xs, e_k, p_k, t_k):
    """Gradient capture at the write candidates (registered by the forward of an in-window training pass; reads the
    selection made before backward; each hook overwrites its own (call, part) slot: idempotent)."""
    N, D = xs.numel() // xs.size(-1), xs.size(-1)
    S = bm.priv.occ.size(0)

    def hb(g):
        ws = bm._wsel
        if ws is None:
            return
        with torch.no_grad():
            gp = g[e_k, p_k].float()
            W = ws["cand_tok"].numel()
            c = ws["tok2cand"][t_k]
            bm._cap[(call, "g")] = torch.zeros(W + 1, D, device=g.device).index_add_(
                0, torch.where(c >= 0, c, torch.full_like(c, W)), gp)[:W]
            m = ws["chosen"][e_k] == t_k
            bm._cap[(call, "p")] = torch.zeros(S + 1, D, device=g.device).index_add_(
                0, torch.where(m, e_k, torch.full_like(e_k, S)), gp)[:S]

    def hs(g):
        ws = bm._wsel
        if ws is None:
            return
        with torch.no_grad():
            gs = g.reshape(N, D).float()
            bm._cap[(call, "gs")] = gs[ws["cand_tok"]] * ws["valid"][:, None]
            ch = ws["chosen"][S - 1:S]                                   # (1-d: a 0-d index would sync)
            bm._cap[(call, "ps")] = (gs.index_select(0, ch.clamp(min=0)) * (ch >= 0)[:, None])[0]

    if bufo.requires_grad:
        bufo.register_hook(hb)
    if xs.requires_grad:
        xs.register_hook(hs)


@torch.no_grad()
def collect_writes(bm):
    """After a micro-batch's backward: the candidates' summed gradients (over the in-window passes, all kept experts +
    the shared expert), their pass-averaged keys, the private targets; queued for the step-end writes."""
    ws = bm._wsel
    if ws is None or not bm._cap:
        bm._wsel, bm._cap, bm._kept_pairs = None, {}, []
        return
    W = ws["cand_tok"].numel()
    dev = ws["cand_tok"].device
    D = bm.d
    S = bm.priv.occ.size(0)
    G = torch.zeros(W, D, device=dev)
    P = torch.zeros(S, D, device=dev)
    for (call, part), t in bm._cap.items():
        if part in ("g", "gs"):
            G += t
        elif part == "p":
            P += t
        else:
            P[S - 1] += t
    xs, xc = bm.xf_sum, bm.xf_cnt
    if xs is None:
        bm._wsel, bm._cap, bm._kept_pairs = None, {}, []
        return
    tok = ws["cand_tok"]
    xbar = xs[tok] / xc[tok].clamp(min=1)[:, None]
    ch = ws["chosen"]
    xbp = xs[ch.clamp(min=0)] / xc[ch.clamp(min=0)].clamp(min=1)[:, None]
    bm._wqueue.append(dict(key=bm.address(xbar), xfull=(xbar - bm.mu) / bm.sigma, q=-G, norm=xbar.norm(dim=-1),
                           sketch=F.normalize((-G) @ bm.gproj, dim=-1), valid=ws["valid"] & (xc[tok] > 0), sig=ws["sig"],
                           pkey=bm.address(xbp), pq=-P, pnorm=xbp.norm(dim=-1), pcand=ws["tok2cand"][ch.clamp(min=0)],
                           phas=(ch >= 0) & (xc[ch.clamp(min=0)] > 0) & (P.abs().sum(-1) > 0)))
    bm._wsel, bm._cap, bm._kept_pairs = None, {}, []


def _content(bm, q, norm, gen):
    """v = sigma_xi xi + 0.1 |xf| q / |q| (the epsilon guard: xi only when |q| is tiny)."""
    qn = q.norm(dim=-1, keepdim=True)
    eps = 1e-6 * qn.median().clamp(min=1e-30)
    vbar = bm.v_init * norm[:, None]
    xi = torch.randn(q.shape, generator=gen, device=q.device) * (0.02 * vbar)
    return xi + torch.where(qn >= eps, vbar * q / qn.clamp(min=1e-30), torch.zeros_like(q))


@torch.no_grad()
def step_writes(bm, step):
    """The step's writes from the queued candidates (both micro-batches): probation lookup + promotion (or direct),
    pushes, global spawn / merge, the private budget. ONE host sync (the push / private counts)."""
    if not bm._wqueue:
        return {}
    Q = {k: torch.cat([w[k] for w in bm._wqueue]) for k in ("key", "xfull", "q", "norm", "sketch", "valid", "sig")}
    PQ = bm._wqueue
    offs = [0]
    for w in PQ[:-1]:
        offs.append(offs[-1] + w["key"].size(0))
    bm._wqueue = []
    dev = Q["key"].device
    gen = bm._generator("mask", dev)
    v = _content(bm, Q["q"], Q["norm"], gen)
    order = torch.where(Q["valid"], Q["sig"], torch.full_like(Q["sig"], -INF)).argsort(descending=True)
    Q = {k: t[order] for k, t in Q.items()}
    v = v[order]
    act = Q["valid"]
    pr = bm.prob
    promoted = torch.zeros_like(act)
    if bm.write_mode == "probation" and pr is not None:
        expired = (pr.age(step) > bm.prob_life) & pr.occ                       # (FIFO life; no sync)
        pr.occ.copy_(pr.occ & ~expired)
        pr.ksq.masked_fill_(expired, INF)
        ids, dd = bm.probation_lookup(Q["key"], 2, "exact" if bm.prob_lookup == "exact" else "ivf")
        j, d1 = ids[:, 0], dd[:, 0]
        old = pr.sketch_f(j)
        cos = F.cosine_similarity(Q["sketch"][:, :pr.kd], old, dim=-1)
        hit = act & torch.isfinite(d1) & (d1 < bm.promo_gamma * bm.h0_g ** 2) & pr.occ[j] & (pr.step[j].long() < step)
        promoted = hit & (cos >= bm.promo_theta) & (pr.hits[j] + 1 >= bm.promo_min_returns)
        # keep the best promo_budget per step, ranked by the gradient cosine (option: key cosine x gradient cosine)
        score = cos if bm.promo_rank == "cos" else cos * F.cosine_similarity(Q["key"][:, :pr.kd], pr.keys_f(j), dim=-1)
        rk = torch.where(promoted, score, torch.full_like(score, -INF)).argsort(descending=True).argsort()
        qualified = promoted
        promoted = promoted & (rk < bm.promo_budget)
        # a qualified candidate that lost the per-step budget does NOT consume its held entry (it stays live until its
        # hold expires; the query itself is pushed as a normal candidate below); the entry is flagged for the log
        lost = qualified & ~promoted
        pr.lost.index_add_(0, j, lost.to(pr.lost.dtype))
        # one promotion per held entry: the highest-surprise candidate takes it
        n = j.numel()
        firstc = torch.full((pr.n,), n, dtype=torch.long, device=dev)
        firstc.scatter_reduce_(0, j, torch.where(promoted, torch.arange(n, device=dev), torch.full_like(j, n)), "amin")
        promoted = promoted & (firstc[j] == torch.arange(n, device=dev))
        # logging only: up to 64 sampled candidate matches (margin, distances, surprises, returns, age)
        lo = torch.where(hit, torch.rand(hit.shape, generator=gen, device=dev), torch.full_like(d1, 2.0)).argsort()[:64]
        bm._promo_log.append(torch.stack([
            hit[lo].float(), (dd[lo, 1] / dd[lo, 0].clamp(min=1e-12)).clamp(max=1e6), d1[lo], cos[lo],
            Q["sig"][lo], (pr.log_sig[j[lo]].float() if pr.log_sig is not None else torch.full_like(d1[lo], float("nan"))),
            (pr.hits[j[lo]] + 1).float(), (step - pr.step[j[lo]].long()).float(),
            promoted[lo].float()], -1))
        bm._promo_log = bm._promo_log[-8:]
        pr.mark_hit(j, step, hit)
        gw = promoted
    else:
        gw = act & (torch.cumsum(act.long(), 0) <= bm.write_w)                       # direct: the top write_w
    res = write_blobs(bm, "g", Q["key"], Q["xfull"], v, Q["q"], torch.zeros_like(Q["norm"], dtype=torch.long), step,
                      active=gw)
    if bm.write_mode == "probation" and pr is not None:
        # only the promotions actually stored (spawned or merged) leave probation; count the 2nd+-attempt successes
        stored = promoted & res["_stored"]
        kill = torch.zeros(pr.n, device=dev).index_add_(0, j, stored.float()) > 0
        res["promoted_after_drop"] = (stored & (pr.lost[j] > 0)).sum()
        pr.occ.copy_(pr.occ & ~kill)
        pr.ksq.masked_fill_(kill, INF)
        pr.lost.masked_fill_(kill, 0)
        res["promoted_stored"] = stored.sum()
    # private targets (one per expert per micro-batch; the latest queue wins per expert)
    pk = torch.stack([w["pkey"] for w in PQ])
    pq = torch.stack([w["pq"] for w in PQ])
    pn = torch.stack([w["pnorm"] for w in PQ])
    ph = torch.stack([w["phas"] for w in PQ])
    last = (ph.long() * torch.arange(1, len(PQ) + 1, device=dev)[:, None]).argmax(0)       # (S,)
    S = ph.size(1)
    ar = torch.arange(S, device=dev)
    pk, pq, pn, ph = pk[last, ar], pq[last, ar], pn[last, ar], ph.any(0)
    bm.credit.add_(bm.private_rate).clamp_(max=4.0)
    if bm.write_mode == "probation" and pr is not None:
        # the private writes follow the promotions: an expert writes when its candidate token was promoted
        prom0 = torch.zeros_like(promoted).index_copy_(0, order, promoted)          # (in queue order)
        gidx = torch.stack([w["pcand"] + o for w, o in zip(PQ, offs)])[last, ar]
        ph = ph & prom0[gidx.clamp(min=0)]
    pw = ph & (bm.credit >= 1)
    # ---- the one host sync: how many to push / write privately
    if bm.write_mode == "probation" and pr is not None:
        push = act & ~promoted
        n_push, n_priv = (int(v_) for v_ in torch.stack([push.sum(), pw.sum()]).tolist())
        if n_push:
            po = (~push).to(torch.uint8).argsort(stable=True)[:n_push]
            bm.probation_push(Q["key"][po], Q["sketch"][po], sig=Q["sig"][po])
        if bm.prob_lookup == "ivf":                                         # (the IVF lookup sees the new entries)
            pr.rebuild(bm.centroids, reassign=False)
    else:
        n_priv = int(pw.sum().tolist())
    if n_priv:
        eo = (~pw).to(torch.uint8).argsort(stable=True)[:n_priv]
        pv = _content(bm, pq[eo], pn[eo], gen)
        rp = write_blobs(bm, "p", pk[eo], None, pv, pq[eo], eo, step, active=torch.ones(n_priv, dtype=torch.bool, device=dev))
        bm.credit.index_add_(0, eo, -torch.ones(n_priv, device=dev))
        res.update({f"p_{k}": t for k, t in rp.items() if not k.startswith("_")})
    res["promoted"] = promoted.sum()
    res["candidates"] = act.sum()
    return res


# ====================================================================== logging (eval time; host syncs allowed here)
def _auc(score, is_pos):
    pos, neg = score[is_pos], score[~is_pos]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    r = allv.argsort().argsort().float() + 1
    return float((r[:pos.numel()].sum() - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel()))


@torch.no_grad()
def gm_metrics(bm, step=None):
    """The plan's broken-if metrics (section 16.1) that the memory itself can measure, plus fill / lifetime per space.
    Called on logging steps only (host syncs here are fine). Resets the per-interval accumulators."""
    out = {}
    g, p = bm.glob, bm.priv
    E = int(bm.n_live)
    out["gm_live_g"] = int(g.occ.sum())
    pf = p.occ[:E].float().mean(-1)
    out["gm_private_fill_min"], out["gm_private_fill_med"], out["gm_private_fill_max"] = (
        float(pf.min()), float(pf.median()), float(pf.max())) if E else (0.0, 0.0, 0.0)
    out["gm_private_full_frac"] = float((pf >= 1).float().mean()) if E else 0.0
    out["gm_shared_fill"] = float(p.occ[bm.shared_idx].float().mean())
    if bm.prob is not None:
        out["gm_prob_fill"] = float(bm.prob.fill())
        out["gm_prob_entries"] = bm.prob.n
    if bm.stats:
        out.update({f"gm_{k}": float(v) for k, v in bm.stats.items()})
    cnt = torch.zeros(bm.n_cells + 1, device=g.cell.device).index_add_(
        0, torch.where(g.cell >= 0, g.cell, torch.full_like(g.cell, bm.n_cells)), torch.ones_like(g.cell, dtype=torch.float))
    c = cnt[:-1].sort().values
    n = c.numel()
    out["gm_cell_gini"] = float(((2 * torch.arange(1, n + 1, device=c.device) - n - 1) * c).sum() / (n * c.sum().clamp(min=1)))
    if bm.phase2:
        out["gm_kappa"] = float(bm.kappa)
        out["gm_sigma2_g"], out["gm_sigma2_p"] = float(bm.sigma2_g), float(bm.sigma2_p)
        out["gm_sigma2_at_floor"] = int(float(bm.sigma2_g) <= 0.05 * float(bm.h0_g) ** 2 * 1.0001)
        out["gm_xf_norm_med"] = float(bm.xf_norm_med)
        occ = g.occ
        if bool(occ.any()):
            vn = g.value.float().norm(dim=-1)[occ]
            out["gm_v_norm_rel_med"] = float(vn.median() / bm.xf_norm_med.clamp(min=1e-12))
            out["gm_path_med_g"] = float(g.path[occ].median())
            u = g.uses[occ].float().sort(descending=True).values
            k1 = max(1, u.numel() // 100)
            out["gm_hub_top1pct_share"] = float(u[:k1].sum() / u.sum().clamp(min=1))
            age = (bm.now - g.birth)[occ]
            out["gm_protected_frac_g"] = float((age < bm.protect_age).float().mean())
        out["gm_v_at_cap"] = [float(bm.wstats[6]), float(bm.wstats[7])]
        bm.wstats[6:8] = 0
        if bm._stats_step:
            st = torch.stack(bm._stats_step).sum(0)
            if getattr(bm, "_stats_acc", None) is not None:              # (r6 G3: the folded older rows)
                st = st + bm._stats_acc
            out["gm_open_rate_r"] = float(st[0] / st[1].clamp(min=1))
            out["gm_open_rate_s"] = float(st[2] / st[3].clamp(min=1))
            out["gm_buffer_overflow"] = float(st[4] / (st[0] + st[2]).clamp(min=1))
            out["gm_explore_n"] = float(st[5])
            bm._stats_step = []
        bm._stats_acc = None
        ms = bm._mstats
        out["gm_placebo_mean"] = float(ms[0] / ms[1].clamp(min=1)) if float(ms[1]) else float("nan")
        out["gm_shared_share_of_gain"] = float(ms[4] / ms[5].clamp(min=1e-30)) if float(ms[5]) else float("nan")
        out["gm_y_mean"] = float(bm._mstats2[0] / bm._mstats2[1].clamp(min=1)) if float(bm._mstats2[1]) else float("nan")
        bm._mstats.zero_()
        bm._mstats2.zero_()
        if bm._yring is not None and bm._yring_n:   # (r6 G3: the static ring of this eval window)
            yr = bm._yring[:min(bm._yring_n, bm._yring.size(0))]
            yr = yr[yr[:, 2] > 0]
            if yr.size(0) > (1 << 24) - 1:   # (r6: torch.quantile refuses > 2^24 elements: an even subsample)
                yr = yr[::-(-yr.size(0) // ((1 << 24) - 1))]
            if yr.numel():
                out["gm_gate_auc"] = _auc(yr[:, 0], yr[:, 1] > 0)
                if yr.size(0) > 1:   # (r6 G4: corr(g_hat, y) on the window's exploration rows)
                    out["gm_gate_corr"] = float(torch.corrcoef(yr[:, :2].t())[0, 1])
                q = torch.quantile(yr[:, 0], torch.linspace(0, 1, 11, device=yr.device))
                dec = torch.bucketize(yr[:, 0].contiguous(), q[1:-1])
                mg = torch.stack([yr[dec == i, 0].mean() if (dec == i).any() else yr.new_tensor(float("nan")) for i in range(10)])
                my = torch.stack([yr[dec == i, 1].mean() if (dec == i).any() else yr.new_tensor(float("nan")) for i in range(10)])
                ok = torch.isfinite(mg) & torch.isfinite(my)
                if int(ok.sum()) >= 2:
                    a, b = mg[ok] - mg[ok].mean(), my[ok] - my[ok].mean()
                    out["gm_gate_decile_slope"] = float((a * b).sum() / (a * a).sum().clamp(min=1e-30))
            bm._yring_n = 0
        if bm.gate is not None:
            out["gm_lam_pos_frac"] = float((bm.gate.lam[:E] > 0).float().mean()) if E else 0.0
            if bm._gate_last is not None:
                out["gm_gate_loss"] = float(bm._gate_last["loss"])
        wc = getattr(bm, "_wcount", None)
        if wc:
            out.update({f"gm_w_{k}": float(v) for k, v in wc.items()})
            bm._wcount = {}
        if bm._evict:
            ea = torch.cat(bm._evict)
            ea = ea[ea >= 0]
            out["gm_evict_age_med"] = float(ea.median()) if ea.numel() else float("nan")
            bm._evict = []
        if bm._promo_log:
            pl = torch.cat(bm._promo_log)
            hit = pl[:, 0] > 0
            if bool(hit.any()):
                h = pl[hit]
                out.update(gm_promo_margin_med=float(h[:, 1].median()), gm_promo_d1_med=float(h[:, 2].median()),
                           gm_promo_cos_med=float(h[:, 3].median()), gm_promo_qsig_med=float(h[:, 4].median()),
                           gm_promo_returns_med=float(h[:, 6].median()), gm_promo_age_med=float(h[:, 7].median()),
                           gm_promo_rate_of_hits=float(h[:, 8].mean()))
            bm._promo_log = []
    return out
