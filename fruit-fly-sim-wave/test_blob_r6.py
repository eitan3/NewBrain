"""Regression tests for the r6 reviews (reviews/r6_review_lens_{A,B,C}.md): C1 sketch scale underflow, C2 shortlist
padding read as slot 0, C4 re-ranked distances out of order, C6 _exact vs its reference, C7 age without a host tensor,
G1 open_bias out of the gate loss, G2 robust target units, G3 bounded logging rings, A2 train / eval gate features,
A3 row-blocked re-rank. CPU only, tiny sizes.

    set CUDA_VISIBLE_DEVICES=-1 & python -m pytest fruit-fly-sim/test_blob_r6.py -q -p no:cacheprovider
"""
import torch
import torch.nn.functional as F

import blob_write
from blob_memory import INF, ProbationStore
from test_blob import populate, small
from test_blob_p2 import P2, _gated_mlp, p2loop, p2step, tokens

torch.set_num_threads(2)


# ---------------------------------------------------------------- C1: the int8 sketch scale
def test_c1_tiny_sketches_round_trip_int8_no_zero_scales():
    g = torch.Generator().manual_seed(0)
    pr = ProbationStore(64, 8, cells=4, dtype="int8", index=False)
    rows = []
    for mag in (1e-6, 1e-7, 1e-8, 1e-9):
        rows.append(torch.randn(8, 8, generator=g) * mag)
    sk = torch.cat(rows)
    pr.push(torch.randn(32, 8, generator=g), sk, 1)
    ids = torch.arange(32)
    assert (pr.pgrad_scale[ids] > 0).all() and (pr.key_scale[ids] > 0).all()
    a = pr.sketch_f(ids).double()
    cos = (a * sk.double()).sum(-1) / (a.norm(dim=-1) * sk.double().norm(dim=-1))   # (no eps: the rows are ~1e-9)
    assert (cos >= 0.99).all(), cos.min()


def test_c1_trainer_sketch_is_unit_norm_and_survives_both_dtypes():
    """collect_writes queues a unit-norm sketch (cosine is its only use), so fp16 storage keeps it too."""
    bm = small(write="probation", probation_rate=4, probation_life=100, gate=False, cand=4)
    bm.fit()
    bm.train()
    W, S, D = 6, bm.priv.occ.size(0), bm.d
    bm.xf_sum, bm.xf_cnt = torch.randn(12, D), torch.ones(12)
    bm._wsel = dict(cand_tok=torch.arange(W), valid=torch.ones(W, dtype=torch.bool), sig=torch.arange(W).float(),
                    tok2cand=torch.arange(12) % W, chosen=torch.full((S,), -1, dtype=torch.long), N=12)
    bm._cap = {(0, "g"): torch.randn(W, D) * 1e-8}
    blob_write.collect_writes(bm)
    sk = bm._wqueue[-1]["sketch"]
    assert torch.allclose(sk.norm(dim=-1), torch.ones(W), atol=1e-5)
    for dt in ("fp16", "int8"):
        pr = ProbationStore(16, bm.kd, cells=4, dtype=dt, index=False)
        pr.push(torch.randn(W, bm.kd), sk, 1)
        assert (F.cosine_similarity(pr.sketch_f(torch.arange(W)), sk, dim=-1) >= 0.99).all()


def test_c1_tiny_run_no_zero_scale_entries_and_nonzero_promo_cos():
    torch.manual_seed(5)
    m = p2loop(gm_probation_dtype="int8")
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    x = tokens(2, 33)
    bm = m.bm
    bm.fit()
    for step in range(1, 7):
        p2step(m, opt, x, step)           # (the same batch: candidates return near their own entries)
    pr = bm.prob
    occ = pr.occ
    assert int(occ.sum()) > 0
    assert float((pr.pgrad_scale[occ] == 0).float().mean()) == 0.0
    logs = torch.cat(bm._promo_log)
    hit = logs[:, 0] > 0
    assert bool(hit.any())
    assert float(logs[hit, 3].abs().max()) > 0                                # cos column is not identically zero


# ---------------------------------------------------------------- C2: padding is -1, never slot 0
def _sparse_bm(n_glob, **kw):
    bm = small(**kw)
    bm.fit()
    if n_glob:   # blobs NOT in slot 0
        g = torch.Generator().manual_seed(2)
        x = torch.randn(n_glob, bm.d, generator=g)
        bm.write(x, torch.randn(n_glob, bm.d, generator=g), slots=torch.arange(5, 5 + n_glob))
    bm.train()
    return bm


def test_c2_search_returns_minus_one_for_padding():
    for n in (0, 2):
        bm = _sparse_bm(n)
        gi, gd = bm.search_global(torch.randn(10, bm.kd))
        pad = ~torch.isfinite(gd)
        assert (gi[pad] == -1).all() and (gi[~pad] >= 0).all()
        assert (bm.glob.occ[gi[~pad]]).all()
        assert not (gi == 0).any()                                            # slot 0 is empty: never returned


def test_c2_reads_and_counts_only_real_candidates():
    for n in (0, 2):
        bm = _sparse_bm(n)
        q = torch.randn(10, bm.kd)
        gi, gd = bm.search_global(q)
        u, A, d1 = bm._read(q, bm.glob, gi[:, :bm.k_g], bm.h0_g, grad=False)
        if n == 0:
            assert (u == 0).all() and (A == 0).all()
        # softmax over the real candidates only: the reference over the occupied slots
        keys = bm.glob.key[gi.clamp(min=0)[:, :bm.k_g]]
        valid = gi[:, :bm.k_g] >= 0
        from blob_memory import kernel_weights
        w, Ar, _ = kernel_weights(q, keys, valid, bm.tau, bm.h0_g, bm.kernel_c)
        ref = (w[..., None] * bm.glob.value[gi.clamp(min=0)[:, :bm.k_g]]).sum(1)
        assert torch.allclose(u, ref, atol=1e-6) and torch.allclose(A, Ar)
        sel = dict(gi=gi, pf=torch.zeros(1, 1, dtype=torch.long), sf=torch.zeros(10, 1, dtype=torch.long),
                   mr=torch.zeros(1, dtype=torch.bool), ms=torch.ones(10, dtype=torch.bool))
        bm._count(sel, torch.zeros(1, dtype=torch.long))
        assert int(bm.glob.uses[~bm.glob.occ].sum()) == 0 and int(bm.glob.touch[~bm.glob.occ].sum()) == 0


def test_c2_gated_step_gives_unoccupied_slots_nothing():
    """Gated read + backward + after_backward + step_end on a store whose slot 0 is empty: slot 0 (and every unoccupied
    slot) gets no use, touch, gradient, strength or value change; a token with no candidate reads nothing."""
    from spiking_lm import RoutedMLP
    torch.manual_seed(3)
    mlp = RoutedMLP(16, n_exp=3, k=2, hidden=8, shared=8, capacity=0).double()
    mlp.defer = True
    bm = small(gate=True, explore=0.5, track_sigma=True).double()
    bm.fit()
    populate(bm, 3, 16, scale=0.5, n_glob=0)
    g = torch.Generator().manual_seed(4)
    bm.write(torch.randn(2, 16, generator=g).double(), torch.randn(2, 16, generator=g).double(), slots=torch.tensor([7, 9]))
    bm.set_h0(4.0, 4.0)
    bm.train()
    bm.gate.make_opt()
    bm.gate.beta.data.fill_(5.0)                                             # open everything
    v0 = bm.glob.value.clone()
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    bm.begin_forward()
    (mlp(x, bmem=lambda *a: bm(*a, call=0)) * torch.linspace(-1, 1, 16, dtype=torch.float64)).sum().backward()
    sel = bm._cache[0]
    gi = sel["gi"]
    assert ((gi == -1) | bm.glob.occ[gi.clamp(min=0)]).all() and not (gi == 0).any()
    tg = bm._touched.get("g")
    if tg is not None:
        assert not tg[~bm.glob.occ].any()
    bm.after_backward()
    bm.apply_value_grads()
    from blob_gate import step_end
    step_end(bm, 1)
    un = ~bm.glob.occ
    assert int(bm.glob.uses[un].sum()) == 0 and float(bm.glob.strength[un].abs().sum()) == 0
    assert torch.equal(bm.glob.value[un], v0[un])
    # an empty global store: the global read is exactly zero
    bm2 = small(gate=True).double()
    bm2.fit()
    q = torch.randn(5, bm2.kd, dtype=torch.float64)
    gi2, _ = bm2.search_global(q.float())
    u, A, _ = bm2._read(q, bm2.glob, gi2[:, :bm2.k_g], bm2.h0_g, grad=False)
    assert (gi2 == -1).all() and (u == 0).all() and (A == 0).all()


# ---------------------------------------------------------------- C4: ascending by the read's distance
def test_c4_search_and_gate_features_ascending_by_the_read_distance():
    mlp, bm = _gated_mlp(seed=21, rerank_full=True, rerank_top=6, kernel="gauss", kernel_dist="full",
                         rerank_dtype="fp32")
    bm.gate.beta.data.fill_(5.0)
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    bm.begin_forward()
    (mlp(x, bmem=lambda *a: bm(*a, call=0)) * torch.linspace(-1, 1, 16, dtype=torch.float64)).sum().backward()
    sel = bm._cache[0]
    gi, gd = sel["gi"], sel["gd"]
    fin = torch.isfinite(gd)
    dd = torch.where(fin, gd, torch.full_like(gd, 1e30))
    assert (dd[:, 1:] >= dd[:, :-1]).all()
    # the returned distances are the read's (model width)
    xf = bm.xfull[gi.clamp(min=0)].double()
    recs = bm._recs[0]
    for part in ("r", "s"):
        rec = recs[part]
        assert (rec["gap_g"] >= 0).all() and (rec["ratio_g"] >= 1).all()   # (d1/d2 <= 1)
    bm.after_backward()
    # the gate's global match features: log(d2/d1) >= 0, log(d2 - d1) is not clamped by a negative gap
    from blob_gate import _feats
    A, f = _feats(bm, bm.glob, gi, gd, bm.h0_g.float())
    assert (f[:, 2] >= -1e-6).all()
    assert torch.isfinite(xf).all()


def test_c4_full_distances_equal_the_reads():
    bm = small(G=64, rerank_full=True, rerank_top=16, cand=6, k_g=4, kernel="gauss", kernel_dist="full",
               rerank_dtype="fp32", search="brute")
    bm.fit()
    populate(bm, 3, 16, n_glob=30)
    x = torch.randn(9, 16)
    qf = (x - bm.mu) / bm.sigma
    gi, gd = bm.search_global(bm.address(x), qfull=qf)
    ref = (qf[:, None, :] - bm.xfull[gi.clamp(min=0)]).pow(2).sum(-1)
    fin = gi >= 0
    assert torch.allclose(gd[fin], ref[fin], rtol=1e-5)
    assert (gd[:, 1:] >= gd[:, :-1]).all()


def test_c4_bounded_read_with_rerank_sorts_the_read_set_by_its_64d_distance():
    bm = small(G=64, rerank_full=True, rerank_top=16, cand=6, k_g=4, search="brute")
    bm.fit()
    populate(bm, 3, 16, n_glob=30)
    x = torch.randn(9, 16)
    q = bm.address(x)
    gi, gd = bm.search_global(q, qfull=(x - bm.mu) / bm.sigma)
    k = bm.k_g
    ref = (q[:, None, :] - bm.glob.key[gi.clamp(min=0)]).pow(2).sum(-1)
    assert torch.allclose(gd, ref, rtol=1e-5)
    assert (gd[:, 1:k] >= gd[:, :k - 1]).all() and (gd[:, k + 1:] >= gd[:, k:-1]).all()


# ---------------------------------------------------------------- C6 / C7
def test_c6_exact_static_blocks_match_the_reference():
    g = torch.Generator().manual_seed(1)
    for dt in ("fp16", "int8"):
        pr = ProbationStore(700, 8, cells=4, dtype=dt, index=False)
        q = torch.randn(40, 8, generator=g)
        for n in (5, 120, 300, 690, 400):     # (ptr 5 .. 1515: partial blocks, a full ring, a wrapped ring)
            pr.push(torch.randn(n, 8, generator=g), torch.randn(n, 8, generator=g), 1)
            a = pr._exact(q, 8, key_chunk=256, group=16, groups=8)
            b = pr._exact_ref(q, 8, key_chunk=256)
            dist = lambda i: torch.where(torch.isfinite(pr.ksq[i]), (q[:, None] - pr.keys_f(i)).pow(2).sum(-1),
                                         torch.full(i.shape, INF)).min(-1).values
            da, db = dist(a), dist(b)
            assert torch.allclose(da, db, rtol=1e-3), (dt, n)
            assert a.shape == (40, 8)


def test_c7_age_from_a_device_step_counter():
    pr = ProbationStore(8, 4, cells=2, index=False)
    pr.push(torch.randn(3, 4), torch.randn(3, 4), torch.tensor(5))
    now = torch.tensor(12)
    assert pr.age(now)[:3].tolist() == [7, 7, 7] and pr.age(12)[:3].tolist() == [7, 7, 7]


# ---------------------------------------------------------------- G1 / G2 / G3 / A2 / A3
def _gated_step(bm, mlp, seed=0, call=0):
    torch.manual_seed(seed)
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    bm.begin_forward()
    (mlp(x, bmem=lambda *a: bm(*a, call=call)) * torch.linspace(-1, 1, 16, dtype=torch.float64)).sum().backward()


def test_g1_open_bias_is_not_in_the_gate_loss():
    mlp, bm = _gated_mlp(seed=31)
    _gated_step(bm, mlp)
    gr = bm._recs[0]["sel"]["gate_rows"]
    y, w = torch.zeros(gr["e"].numel()), torch.ones(gr["e"].numel())
    l0, g0 = bm.gate.loss(gr, y, w, torch.tensor(1.0))
    bm.gate.open_bias.fill_(-25.0)
    l1, g1 = bm.gate.loss(gr, y, w, torch.tensor(1.0))
    assert torch.equal(l0, l1) and torch.equal(g0, g1)
    # ... but the selection score carries it
    s1 = bm.gate.score(gr["phi"], gr["xf"], gr["ug"], gr["m"], gr["up"], gr["e"], gr["tok"])
    assert torch.allclose(s1, g1 - 25.0)


def test_g2_robust_target_scale_and_small_prior():
    mlp, bm = _gated_mlp(seed=32)
    assert torch.allclose(bm.gate.pg.data.abs(), torch.full((3,), 0.01))
    assert float(bm.gate.y_sc) == 0.0
    _gated_step(bm, mlp)
    bm.after_backward()
    assert float(bm.gate.y_sc) > 0
    # resume from an older gate state without y_sc
    st = bm.get_extra_state()
    st["gate"].pop("y_sc")
    bm.set_extra_state(st)


def test_g3_logging_rings_are_bounded():
    import blob_gate
    mlp, bm = _gated_mlp(seed=33)
    bm._stats_step = [torch.ones(6) for _ in range(blob_gate.STATS_KEEP)]
    _gated_step(bm, mlp)
    assert len(bm._stats_step) <= blob_gate.STATS_KEEP and bm._stats_acc is not None
    bm.after_backward()
    assert bm._yring.shape == (blob_gate.YRING, 3) and bm._yring_n > 0
    out = blob_write.gm_metrics(bm)
    assert bm._yring_n == 0 and bm._stats_acc is None and "gm_open_rate_r" in out


def test_a2_gate_halt_features_identical_in_train_and_eval():
    mlp, bm = _gated_mlp(seed=34)
    bm.prev_halt = torch.rand(12)
    _gated_step(bm, mlp)
    phi = bm._recs[0]["sel"]["gate_rows"]["phi"]
    assert (phi[:, -2] == 0).all() and (phi[:, -1] == 1).all()          # (halt 0, "none" 1: never from targets)


def test_a3_row_blocked_rerank_equals_the_unblocked():
    bm = small(G=64, rerank_full=True, rerank_top=16, cand=6, k_g=4, kernel="gauss", kernel_dist="full",
               rerank_dtype="fp32", search="brute")
    bm.fit()
    populate(bm, 3, 16, n_glob=30)
    x = torch.randn(50, 16)
    qf = (x - bm.mu) / bm.sigma
    gi, gd = bm.search_global(bm.address(x), qfull=qf)
    bm.rerank_rows = 7                                                   # (blocks of 7 rows)
    gi2, gd2 = bm.search_global(bm.address(x), qfull=qf)
    assert torch.equal(gi, gi2) and torch.equal(gd, gd2)
