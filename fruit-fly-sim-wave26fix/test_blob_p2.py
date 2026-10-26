"""Phase-2 tests of the blob memory (plan_v2/25d rev 2 sections 5-10 + the review checklist): gate, writes ("keep what
comes back"), merges, eviction, renewal, causality, eval guards, resume, no host syncs. CPU only, tiny sizes.

    set CUDA_VISIBLE_DEVICES=-1 & python -m pytest fruit-fly-sim/test_blob_p2.py -q -p no:cacheprovider
"""
import copy
import inspect
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import blob_gate
import blob_write
from blob_gate import BlobGate, Scale, compact_causal, hash_uniform, nan_quantile
from blob_memory import BlobMemory
from blob_write import renew, step_writes, write_blobs
from test_blob import _call, _exact, _routing, loop, populate, small, tokens
from test_memory import _run_main, _tiny_data, current

torch.set_num_threads(2)

P2 = dict(gm_gate=1, gm_write="probation", gm_probation_rate=4, gm_probation_life=50, gm_content_opt="adagrad",
          gm_lr=0.01, gm_track_sigma=1, gm_rerank_full=1, gm_rerank_top=16, gm_kernel="gauss", gm_use_A=0,
          gm_gate_warm=3, concepts=[2, 4], think_w=0.1, think_fb=True, think_use=True)


def p2loop(**kw):
    return loop(gm=1, **{**P2, **kw})


def p2step(m, opt, x, step, writes=True):
    """One training step the way train_lm runs it (select_writes between the loss and backward, after_backward,
    after_step)."""
    bm = m.bm
    bm.now.fill_(step)
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 3, 1, 0
    logits, loss = m(x[:, :-1], x[:, 1:])
    total = loss + m.aux_loss()
    if writes:
        bm.select_writes(logits, x[:, 1:], x[:, :-1])
    opt.zero_grad()
    total.backward()
    bm.after_backward()
    opt.step()
    m.blocks[0].mlp.apply_balance()
    bm.after_step(step)
    return total.detach()


def p2model(fill=True, **kw):
    m = p2loop(**kw)
    m.bm.fit()
    if fill:
        populate(m.bm, m.blocks[0].mlp.E, 32, scale=0.3)
        m.bm.set_h0(5.0, 5.0)
    return m


def store_snapshot(bm):
    """Every tensor of the stores (state dict incl. the gate / probation / RNG extra state) + the derived tables."""
    sd = copy.deepcopy(bm.state_dict())
    der = {n: getattr(bm, n).clone() for n in ("table", "table_kr", "table_ksq")}
    der.update({f"g.{n}": getattr(bm.glob, n).clone() for n in ("key_bf", "ksq")})
    return sd, der


def assert_same(a, b):
    sa, da = a
    sb, db = b
    assert list(sa) == list(sb)
    for k in sa:
        if torch.is_tensor(sa[k]):
            assert torch.equal(sa[k], sb[k]), k
        else:
            assert repr(sa[k]) == repr(sb[k]), k
    for k in da:
        assert torch.equal(da[k], db[k]), k


# ---------------------------------------------------------------- (2) eval / no-target forwards leave every store untouched
def test_p2_02_eval_leaves_every_store_bit_identical():
    m = p2model()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for i in range(3):
        p2step(m, opt, tokens(seed=40 + i), i)
    snap = store_snapshot(m.bm)
    m.eval()
    with torch.no_grad():
        m(tokens(seed=50)[:, :-1])                                         # plain eval forward
        x = tokens(seed=51)
        m(x[:, :-1], x[:, 1:])                                             # eval forward with targets
        m.tick_losses(x[:, :-1], x[:, 1:], 3)                              # the loop report's per-pass losses
    m.train()
    with torch.no_grad():                                                  # a training-mode forward without grad
        m(tokens(seed=52)[:, :-1])
    tl = current().train_lm
    args = SimpleNamespace(k_mean=3, gm_heldout="", eval_batches=2, eval_batch=2, seq=16, device="cpu")
    data = SimpleNamespace(batch=lambda b, split, gen: (lambda w: (w[:, :-1], w[:, 1:]))(
        torch.randint(1, 50, (b, 17), generator=torch.Generator().manual_seed(int(gen.randint(1000))))))
    d = tl.gm_eval_delta(m, data, args)                                    # memory on / off delta (eval)
    assert "gm_val_ko_delta" in d and math.isfinite(d["gm_val_ko_delta"])
    assert not m.bm._recs and not m.bm._wqueue and m.bm._wsel is None
    assert_same(snap, store_snapshot(m.bm))


# ---------------------------------------------------------------- (3) causality
def test_p2_03a_the_store_does_not_change_before_the_step_end():
    m = p2model()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    p2step(m, opt, tokens(seed=60), 0)
    bm = m.bm
    read_fields = lambda: {**{f"g.{n}": getattr(bm.glob, n).clone() for n in ("key", "value", "occ", "strength", "cell")},
                           **{f"p.{n}": getattr(bm.priv, n).clone() for n in ("key", "value", "occ", "strength")},
                           "prob": bm.prob.occ.clone(), "probk": bm.prob.key.clone(), "table": bm.table.clone()}
    before = read_fields()
    bm.now.fill_(1)
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 3, 1, 0
    x = tokens(seed=61)
    logits, loss = m(x[:, :-1], x[:, 1:])
    bm.select_writes(logits, x[:, 1:], x[:, :-1])
    (loss + m.aux_loss()).backward()
    bm.after_backward()
    mid = read_fields()
    for k in before:
        assert torch.equal(before[k], mid[k]), k                          # reads see only earlier steps' blobs
    bm.after_step(1)
    after = read_fields()
    assert any(not torch.equal(mid[k], after[k]) for k in after)          # ... the step end changes it


def test_p2_03b_no_future_token_reaches_a_read():
    """Training mode (gate, exploration, hash draws, the causal buffers, IVF capacity by position) on a dropless
    model: changing the tokens after position t0 changes nothing at positions <= t0."""
    outs = []
    for seed in (70, 71):
        m = p2model(moe=dict(n_exp=8, k=2, hidden=16, shared=16, capacity=0, birth_mode="split"), gm_buf=0.3,
                    gm_explore=0.3)
        m.bm.gate.beta.data.fill_(0.05)                                   # (most pairs open: the buffer binds)
        x = tokens(seed=3).clone()
        x[:, 10:] = tokens(seed=seed)[:, 10:]
        m.train()
        m.K = 3
        torch.manual_seed(0)
        outs.append(m(x[:, :-1], x[:, 1:])[0])                            # (a training forward WITH targets)
        st = m.bm._stats_step
        assert st and float(torch.stack(st)[:, 4].sum()) > 0              # the 30% buffer overflowed (causal drop)
        assert float(torch.stack(st)[:, 5].sum()) > 0                     # exploration happened
    assert torch.allclose(outs[0][:, :10], outs[1][:, :10], atol=1e-10)
    assert not torch.allclose(outs[0][:, 10:], outs[1][:, 10:])


def test_p2_03c_threshold_rule_and_causal_compaction():
    pos = torch.tensor([3, 0, 2, 1, 0, 3, 2])
    open_ = torch.tensor([True, True, True, True, True, False, True])
    rows, kr, keep, ov = compact_causal(open_, pos, 3)
    assert set(rows[kr].tolist()) == {1, 4, 3} and int(ov) == 3            # positions 0, 0, 1 kept; later dropped
    g = BlobGate(8, 0, 0, 4)
    g.lam.copy_(torch.tensor([0.0, 0.1, 0.0, 0.0]))
    g.count(torch.tensor([1, 1, 1, 1]), torch.tensor([0.2, 0.2, 0.2, 0.0]), torch.zeros(4, dtype=torch.bool),
            torch.tensor([0.1] * 4))
    g.update_lambda()
    assert abs(float(g.lam[1]) - (0.1 + 0.01 * (0.75 - 0.5))) < 1e-7 and float(g.lam[0]) == 0.0
    u1 = hash_uniform(torch.tensor([5, 6]), torch.tensor([2, 2]), 123)
    u2 = hash_uniform(torch.tensor([5]), torch.tensor([2]), 123)
    assert float(u1[0]) == float(u2[0]) and 0 <= float(u1.min()) and float(u1.max()) < 1    # a pair's own draw


# ---------------------------------------------------------------- (4) no labels / source tags in the training path
def test_p2_04_no_dataset_labels_or_pointers_in_the_memory_code():
    import io
    import tokenize
    import blob_memory
    bad = {"val_tiers", "tier", "tiers", "doc_id", "fact_id", "source_tag", "answer_mask", "is_answer", "label", "labels",
           "window_index", "dataset", "tags"}          # (".data" is the tensor attribute: allowed)
    for mod in (blob_memory, blob_gate, blob_write):                      # (identifiers in code, not comments)
        names = {t.string.lower() for t in tokenize.generate_tokens(io.StringIO(inspect.getsource(mod)).readline)
                 if t.type == tokenize.NAME}
        assert not (names & bad), (mod.__name__, names & bad)


# ---------------------------------------------------------------- (5) s ~ U(0,1) on exploration only; b = dl/ds; stop-grad
def _gated_mlp(seed=8, **kw):
    from spiking_lm import RoutedMLP
    torch.manual_seed(seed)
    mlp = RoutedMLP(16, n_exp=3, k=2, hidden=8, shared=8, capacity=0).double()
    mlp.defer = True
    bm = small(**{**dict(gate=True, explore=0.5, track_sigma=True), **kw}).double()
    bm.fit()
    populate(bm, 3, 16, scale=0.5)
    bm.set_h0(4.0, 4.0)
    bm.train()
    bm.gate.make_opt()
    return mlp, bm


def test_p2_05_read_scale_exploration_only_b_equals_dl_ds_and_the_gate_is_stop_grad():
    mlp, bm = _gated_mlp()
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    W = torch.linspace(-1, 1, 16, dtype=torch.float64)
    bm.begin_forward()
    y = mlp(x, bmem=lambda *a: bm(*a, call=0))
    loss = (y * W).sum()
    loss.backward()
    rec = bm._recs[0]["r"]
    sel = bm._cache[0]
    k = rec["k"]
    assert (rec["s"][rec["ex"]] < 1).all() and (rec["s"][k & ~rec["ex"]] == 1).all() and rec["ex"].any()
    # b = dl/ds by a central finite difference on one exploration row (the cached selection makes it exact)
    i = int(torch.nonzero(rec["ex"])[0])
    b = float(rec["bk"][i, 0])

    def L(ds):
        s0 = sel["s_r"].clone()
        sel["s_r"][i] += ds
        with torch.no_grad():
            v = float((mlp(x.detach(), bmem=lambda *a: bm(*a, call=0)) * W).sum())
        sel["s_r"].copy_(s0)
        return v
    eps = 1e-5
    assert abs((L(eps) - L(-eps)) / (2 * eps) - b) < 1e-6 * max(1.0, abs(b))
    # the gate's update: recomputed g_hat == the forward's, only gate parameters receive gradient
    model_grads = [None if p.grad is None else p.grad.clone() for p in mlp.parameters()]
    xg = x.grad.clone()
    gr = sel["gate_rows"]
    g_re = bm.gate.score(gr["phi"], gr["xf"], gr["ug"], gr["m"], gr["up"], gr["e"], gr["tok"])
    assert torch.allclose(g_re, gr["ghat"], atol=1e-6)
    bm.after_backward()
    assert all((a is None and p.grad is None) or torch.equal(a, p.grad) for a, p in zip(model_grads, mlp.parameters()))
    assert torch.equal(xg, x.grad)
    assert bm.gate.W1.grad is not None and bm.gate.beta.grad is not None


def test_p2_05b_post_backward_gate_gradient_equals_independent_autograd():
    mlp, bm = _gated_mlp(seed=9)
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    bm.begin_forward()
    (mlp(x, bmem=lambda *a: bm(*a, call=0)) * torch.linspace(-1, 1, 16, dtype=torch.float64)).sum().backward()
    rr = bm._recs[0]
    sel = rr["sel"]
    gr = sel["gate_rows"]
    # an independent target / loss: y from the records, Huber written out, weights 1 / propensity
    n_tot = rr["n_total"]
    b = torch.zeros(n_tot + 1)
    s = torch.ones(n_tot + 1)
    r2 = torch.zeros(n_tot + 1)
    for part in ("r", "s"):
        rec = rr[part]
        dest = torch.where(rec["k"], rec["orig"], torch.full_like(rec["orig"], n_tot))
        b.index_copy_(0, dest, rec["bk"][:, 0])
        s.index_copy_(0, dest, rec["s"])
        r2.index_copy_(0, dest, rec["r2"])
    xo = gr["row"]
    y = bm.y_scale * (-b[xo] + (s[xo] - 0.5) * bm.kappa * r2[xo])
    ok = gr["ok"]
    med = nan_quantile(torch.where(ok, y, torch.full_like(y, float("nan"))), 0.5)
    mad = nan_quantile(torch.where(ok, (y - med).abs(), torch.full_like(y, float("nan"))), 0.5)
    sr = torch.nan_to_num(1.4826 * mad, nan=1e-4).clamp(min=1e-4)
    y = y.clamp(min=float(med - 5 * sr), max=float(med + 5 * sr))
    y = y / (1.4826 * mad)                     # (r6 G2: robust units, the first call's scale; Huber delta 1)
    delta = 1.0
    w = ok.float() / gr["prop"]
    params = list(bm.gate.parameters())
    gh = bm.gate.score(gr["phi"], gr["xf"], gr["ug"], gr["m"], gr["up"], gr["e"], gr["tok"], bias=False)   # (r6 G1)
    e = gh - y
    hub = torch.where(e.abs() <= delta, 0.5 * e * e, delta * (e.abs() - 0.5 * delta))
    ref = torch.autograd.grad((w * hub).sum() / w.sum(), params, allow_unused=True)
    bm.after_backward()
    for p, r in zip(params, ref):
        if r is None:
            assert p.grad is None or p.grad.abs().sum() == 0
        else:
            assert torch.allclose(p.grad, r, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------- (6) probation: candidates, promotion, budget, hold
def _queue(bm, keys, sketch, sig=None, n_exp=None):
    W = keys.size(0)
    S = bm.priv.occ.size(0)
    D = bm.d
    return dict(key=keys, xfull=torch.zeros(W, D), q=sketch.new_ones(W, D), norm=torch.ones(W), sketch=sketch,
                valid=torch.ones(W, dtype=torch.bool), sig=torch.arange(W, 0, -1).float() if sig is None else sig,
                pkey=torch.zeros(S, bm.kd), pq=torch.zeros(S, D), pnorm=torch.ones(S),
                phas=torch.zeros(S, dtype=torch.bool), pcand=torch.full((S,), -1, dtype=torch.long))


def _pbm(**kw):
    bm = small(write="probation", probation_rate=4, probation_life=100, gate=False, promo_theta=0.3, promo_gamma=1.0,
               cand=4, **kw)
    bm.fit()
    bm.xf_norm_med.fill_(1.0)
    return bm


def test_p2_06a_candidates_are_the_top_rate_by_entropy_normalised_surprise():
    bm = _pbm(cand_rate=0.25)
    bm.train()
    bm.xf_sum, bm.xf_cnt = torch.zeros(16, 16), torch.ones(16)
    logits = torch.randn(2, 8, 50)
    tg = torch.randint(0, 50, (2, 8))
    bm.ch_mean.fill_(0.0)
    ws = blob_write.select_writes(bm, logits, tg)
    lp = F.log_softmax(logits.float(), -1)
    ce = -lp.gather(-1, tg[..., None]).squeeze(-1)
    top = ce.topk(2, -1).indices
    assert set(ws["cand_tok"].tolist()) == set((torch.arange(2)[:, None] * 8 + top).flatten().tolist())
    assert int(bm.ch_n.sum()) > 0                                          # C(H) updated after use


def test_p2_06b_promotion_top1_and_cosine_budget_min_returns_and_hold():
    bm = _pbm(promo_budget=2)
    pr = bm.prob
    g = torch.Generator().manual_seed(3)
    xs = torch.randn(4, 16, generator=g) * 3
    keys = bm.address(xs)
    sk = torch.randn(4, 8, generator=g)
    bm.now.fill_(1)
    ids = bm.probation_push(keys, sk)
    assert pr.lookup(keys, 1, "exact")[0][:, 0].tolist() == ids.tolist()
    # step 5: candidates return near entries 0..3; 0, 1, 2 agree (cos 1, 0.9, 0.8), 3 disagrees
    sk_new = torch.stack([sk[0], 0.9 * sk[1] / sk[1].norm() + 0.1 * sk[0] / sk[0].norm(), sk[2], -sk[3]])
    bm._wqueue = [_queue(bm, keys + 1e-3, sk_new)]
    res = step_writes(bm, 5)
    assert int(res["promoted"]) == 2 and int(res["promoted_stored"]) == 2   # the budget: the best 2 by cosine
    assert int(bm.glob.occ.sum()) == 2
    live = pr.occ[ids]
    assert live.tolist().count(False) == 2                                # only stored promotions leave probation
    lost = (pr.lost[ids] > 0) & live
    assert int(lost.sum()) == 1                                           # the budget loser stays live, flagged
    # the loser returns at step 6 and is promoted then (a 2nd-attempt success)
    j = int(torch.nonzero(lost)[0])
    bm._wqueue = [_queue(bm, keys[j:j + 1], sk[j:j + 1])]                 # (exactly on the held entry)
    res = step_writes(bm, 6)
    assert int(res["promoted_stored"]) == 1 and int(res["promoted_after_drop"]) == 1
    # min returns 2: the first return only counts
    bm2 = _pbm(promo_min_returns=2)
    bm2.now.fill_(1)
    bm2.probation_push(keys[:1], sk[:1])
    bm2._wqueue = [_queue(bm2, keys[:1] + 1e-3, sk[:1])]
    assert int(step_writes(bm2, 3)["promoted"]) == 0 and int(bm2.prob.hits[0]) == 1
    bm2._wqueue = [_queue(bm2, keys[:1], sk[:1])]                       # (exactly on the held entry: its 2nd return)
    assert int(step_writes(bm2, 4)["promoted"]) == 1
    # the hold: an entry older than the life expires
    bm3 = _pbm()
    bm3.now.fill_(0)
    bm3.probation_push(keys[:1], sk[:1])
    bm3._wqueue = [_queue(bm3, keys[:1] + 1e-3, sk[:1])]
    assert int(step_writes(bm3, 101)["promoted"]) == 0                    # (expired at 101 > 100)
    assert small(write="probation", probation_rate=1).prob_life == 3000   # the default hold


def test_p2_06c_probation_int8_storage_and_defaults():
    from blob_memory import ProbationStore
    a, b = ProbationStore(40, 8, cells=4), ProbationStore(40, 8, cells=4, dtype="int8")
    k = torch.randn(30, 8)
    a.push(k, k, 1)
    b.push(k, k, 1)
    assert b.key.dtype == torch.int8 and b.key_scale.dtype == torch.float32   # (r6 C1: fp32 scales)
    assert torch.allclose(b.keys_f(torch.arange(30)), k, atol=0.03 * k.abs().max())
    q = k[:5] + 0.01
    assert torch.equal(a.lookup(q, 1)[0], b.lookup(q, 1)[0])
    m = loop(gm=1)
    assert m.bm.prob is None and m.bm.write_mode == "off"                 # off by default
    from loop_brain import LoopBrain
    assert LoopBrain.GM_DEFAULTS["gm_probation_life"] == 3000 and LoopBrain.GM_DEFAULTS["gm_promo_theta"] == 0.3


# ---------------------------------------------------------------- (7) writes: no grad, state resets, lazy row-Adagrad
def test_p2_07a_writes_merges_and_splits_reset_the_row_optimizer_state():
    bm = _pbm()
    bm.glob.opt_v.fill_(5.0)
    bm.glob.opt_n.fill_(3)
    keys = bm.address(torch.randn(3, 16) * 3)
    v = torch.randn(3, 16) * 0.1
    with torch.enable_grad():
        res = write_blobs(bm, "g", keys, torch.zeros(3, 16), v, v, torch.zeros(3, dtype=torch.long), 7)
    j = torch.nonzero(bm.glob.occ).flatten()
    assert int(res["spawned"]) == 3 and (bm.glob.opt_v[j] == 0).all() and (bm.glob.opt_n[j] == 0).all()
    assert not bm.glob.value.requires_grad and (bm.glob.birth[j] == 7).all()
    bm.glob.opt_v[j] = 4.0
    res = write_blobs(bm, "g", keys[:1] + 1e-4, torch.zeros(1, 16), v[:1], v[:1], torch.zeros(1, dtype=torch.long), 8)
    assert int(res["merged"]) == 1 and int((bm.glob.opt_v[j] == 0).sum()) == 1     # the merged row reset
    assert int(bm.glob.merges[j].max()) == 1 and int(bm.glob.moves_win[j].max()) == 1
    p = small(gate=True, track_sigma=True)
    p.fit()
    populate(p, 3, 16)
    p.priv.strength.fill_(0.8)
    p.priv.opt_v.fill_(2.0)
    p.gate.make_opt()
    p.on_grow(torch.tensor([1]), 3, split=True)
    assert torch.allclose(p.priv.strength[1], torch.full((8,), 0.4)) and torch.allclose(p.priv.strength[3], torch.full((8,), 0.4))
    assert (p.priv.opt_v[1] == 0).all() and (p.priv.opt_v[3] == 0).all()                # both twins
    assert torch.equal(p.priv.key[3], p.priv.key[1])


def test_p2_07b_lazy_rowwise_adagrad_matches_reference_untouched_rows_identical():
    torch.manual_seed(10)
    bm = small(content_opt="adagrad", lr=0.01, clip=0.05, priv_bf16=False)
    bm.fit()
    populate(bm, 3, 16)
    rt = _routing()
    v0, s0 = bm.glob.value.clone(), bm.glob.opt_v.clone()
    touched = torch.zeros(64, dtype=torch.bool)
    for step in range(4):
        xf = torch.randn(12, 16, requires_grad=True)
        bm.begin_forward()
        buf, xs = _call(bm, xf, rt)
        (buf.pow(2).sum() + xs.sum()).backward()
        g, t = bm._vg["g"].clone(), bm._touched["g"].clone()
        touched |= t
        val, S = bm.glob.value.clone(), bm.glob.opt_v.clone()
        S = torch.where(t, S + g.pow(2).mean(-1), S)
        u = 0.01 * g / (S.sqrt() + 1e-8)[:, None]
        u = u * (0.05 / u.norm(dim=-1, keepdim=True).clamp(min=0.05))
        ref = torch.where(t[:, None], val - u, val)
        bm.apply_value_grads()
        assert torch.allclose(bm.glob.value, ref, atol=1e-7) and torch.allclose(bm.glob.opt_v, S)
    assert 0 < touched.sum() < 64
    assert torch.equal(bm.glob.value[~touched], v0[~touched]) and torch.equal(bm.glob.opt_v[~touched], s0[~touched])


# ---------------------------------------------------------------- (8) renewal: anti-capture, move <= 1 sigma, budgets
def test_p2_08a_renewal_moves_capped_one_per_blob_budgets_and_caps():
    bm = small(gate=True, track_sigma=True, cap_moves_g=2, max_moves=2)
    bm.fit()
    populate(bm, 3, 16)
    bm.sigma2_g.fill_(0.25)                                             # sigma 0.5 < 0.5 h0
    j = torch.nonzero(bm.glob.occ).flatten()[:4]
    k0 = bm.glob.key.clone()
    q = bm.glob.key[j] + 10.0
    slots = torch.cat([j, j[:1]])
    qq = torch.cat([q, bm.glob.key[j[:1]] - 10.0])                       # two confirmations of j0 (mean ~ k0)
    gain = torch.tensor([1.0, 3.0, 2.0, 0.5, 0.5])
    ok = torch.tensor([True, True, True, False, True])
    n = renew(bm, "g", slots, qq, gain, ok, 5)
    moved = (bm.glob.key - k0).norm(dim=-1)
    assert int(n) == 2 and (moved <= 0.5 + 1e-5).all()                  # cap 2 by gain (j1, j2), each <= sigma
    assert moved[j[1]] > 0.49 and moved[j[2]] > 0.49 and moved[j[0]] == 0 and moved[j[3]] == 0
    for t in (6, 7):
        renew(bm, "g", j[1:2], q[1:2], torch.ones(1), torch.ones(1, dtype=torch.bool), t)
    assert int(bm.glob.moves_win[j[1]]) == 2                            # the window budget (2) stops the 3rd move
    k1 = bm.glob.key[j[1]].clone()
    renew(bm, "g", j[1:2], q[1:2], torch.ones(1), torch.ones(1, dtype=torch.bool), 8)
    assert torch.equal(bm.glob.key[j[1]], k1)


def test_p2_08b_confirmation_needs_gain_geometry_and_content_agreement():
    mlp, bm = _gated_mlp(seed=12, explore=0.0)
    bm.gate.beta.data.fill_(1.0)                                         # open everything
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    bm.begin_forward()
    (mlp(x, bmem=lambda *a: bm(*a, call=0)) * torch.linspace(-1, 1, 16, dtype=torch.float64)).sum().backward()
    recs = {p: bm._recs[0][p] for p in ("r", "s")}
    bm.after_backward()
    for (jg, q, Gg, okg), rec in zip(bm._conf["g"], recs.values()):
        if okg.any():
            assert (Gg[okg] > 0).all() and (rec["bk"][okg, 3] > bm.conf_cos).all()
            assert (rec["ratio_g"][okg] >= bm.conf_margin).all() and (rec["Ag"][okg] >= bm.conf_A).all()
        cos_bad = rec["bk"][:, 3] <= bm.conf_cos
        assert not (okg & cos_bad).any()                                 # anti-capture: content must agree


# ---------------------------------------------------------------- (9) 384-d re-rank
def test_p2_09_full_width_rerank():
    bm = small(G=64, rerank_full=True, rerank_top=8, cand=2, search="brute")
    bm.fit()
    x = torch.randn(1, 16)
    other = x.clone()
    other[0, 0] += 5.0                                                   # same 64-d key? make the keys equal:
    bm.write(torch.cat([x, x]), torch.zeros(2, 16))
    bm.xfull[1] = ((other - bm.mu) / bm.sigma).to(bm.xfull.dtype)        # slot 1's full-width record differs
    qf = ((other - bm.mu) / bm.sigma)
    idx, d = bm.search_global(bm.address(x), qfull=qf)
    assert int(idx[0, 0]) == 1                                           # the 384-d record decides the tie
    idx, d = bm.search_global(bm.address(x), qfull=(x - bm.mu) / bm.sigma)
    assert int(idx[0, 0]) == 0


# ---------------------------------------------------------------- (10) splits through LoopBrain: rows, moments, strength
def test_p2_10_split_copies_private_blobs_gate_rows_and_moments_both_half():
    m = p2model()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for i in range(2):
        p2step(m, opt, tokens(seed=80 + i), i)
    bm = m.bm
    E0 = m.blocks[0].mlp.E
    st = {n: getattr(bm.priv, n).clone() for n in ("key", "strength")}
    gstate = {p: {k: v.clone() for k, v in bm.gate.opt.state[p].items() if torch.is_tensor(v) and v.dim()}
              for p in (bm.gate.c, bm.gate.beta, bm.gate.ge)}
    src = m.grow_experts(2)
    for i, s_ in enumerate(src.tolist()):
        assert torch.equal(bm.priv.key[E0 + i], st["key"][s_])
        assert torch.allclose(bm.priv.strength[E0 + i], st["strength"][s_] * 0.5)
        assert torch.allclose(bm.priv.strength[s_], st["strength"][s_] * 0.5)
        assert torch.equal(bm.gate.beta.data[E0 + i], bm.gate.beta.data[s_])
        for p in (bm.gate.c, bm.gate.beta, bm.gate.ge):
            for k, v in bm.gate.opt.state[p].items():
                if torch.is_tensor(v) and v.dim():
                    assert torch.equal(v[E0 + i], v[s_]), k


# ---------------------------------------------------------------- (11) save / resume
GM2_ARGS = ("--gm 1 --gm-preset 25d --gm-global 256 --gm-private 16 --gm-fit-step 5 --gm-fit-window 3 --gm-cells 8 "
            "--gm-cell-cap 64 --gm-probe 4 --gm-qcap 256 --gm-cand 8 --gm-topk 4,4 --gm-ivf-refit 15 --gm-rerank-top 16 "
            "--gm-probation-cap 4000 --gm-cand-rate 0.05 --gm-eval-every 20 --concepts 2,4 --think-w 0.1")


def test_p2_11a_trainer_preset_smoke(tmp_path):
    data = _tiny_data(tmp_path / "data")
    run, recs, ck = _run_main(current().train_lm, tmp_path / "runs", "p2", data, GM2_ARGS.split())
    log = (run / "train.log").read_text()
    assert "GM ADDRESS FITTED" in log
    last = recs[-1]
    for k in ("gm_open_rate_r", "gm_kappa", "gm_prob_fill", "gm_cell_gini", "gm_val_ko_delta", "gm_live_g"):
        assert k in last, k
    from loop_brain import LoopBrain
    m = LoopBrain(**ck["cfg"])
    m.load_state_dict(ck["state"], strict=True)
    assert m.bm.gate is not None and m.bm.write_mode == "probation" and m.bm.content_opt == "adagrad"
    assert int(m.bm.glob.occ.sum()) + int(m.bm.prob.occ.sum()) > 0


class _Stop(Exception):
    pass


def test_p2_11b_resume_is_bit_identical_incl_stores_probation_gate_and_rng(tmp_path, monkeypatch):
    tl = current().train_lm
    data = _tiny_data(tmp_path / "data")
    extra = (GM2_ARGS + " --ckpt-every 20").split()
    run_a, ra, ca = _run_main(tl, tmp_path / "runs_a", "a", data, extra)
    save = tl.save_checkpoint

    def crash(path, obj):
        save(path, obj)
        raise _Stop

    monkeypatch.setattr(tl, "save_checkpoint", crash)
    with pytest.raises(_Stop):
        _run_main(tl, tmp_path / "runs_b", "b", data, extra)
    monkeypatch.setattr(tl, "save_checkpoint", save)
    last = tmp_path / "runs_b" / "b" / "last.pt"
    run_b, rb, cb = _run_main(tl, tmp_path / "runs_b", "b", data, extra + ["--resume", str(last)])
    assert "RESUMED at step 20" in (run_b / "train.log").read_text()
    assert len(ra) == len(rb)
    for a, b in zip(ra, rb):
        a, b = dict(a), dict(b)
        for k in ("tok_s", "elapsed"):
            a.pop(k), b.pop(k)
        assert a == b
    assert list(ca["state"]) == list(cb["state"])
    for k in ca["state"]:
        if torch.is_tensor(ca["state"][k]):
            assert torch.equal(ca["state"][k], cb["state"][k]), k
        else:
            assert repr(ca["state"][k]) == repr(cb["state"][k]), k
    st = ca["state"]
    assert int(st["bm.prob.occ"].sum()) > 0 and "gate" in st["bm._extra_state"]


# ---------------------------------------------------------------- (12) sync-free, static buffers
def test_p2_12_no_host_sync_in_the_gated_step_one_at_the_step_end(monkeypatch):
    m = p2model()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    p2step(m, opt, tokens(seed=90), 0)
    bm = m.bm
    calls = {"tolist": 0}

    def boom(*a, **k):
        raise AssertionError("host sync")

    orig_tolist = torch.Tensor.tolist

    def counted_tolist(self):
        calls["tolist"] += 1
        return orig_tolist(self)

    orig_getitem = torch.Tensor.__getitem__

    def getitem(self, i):
        items = i if isinstance(i, tuple) else (i,)
        if any(torch.is_tensor(t) and t.dtype == torch.bool for t in items):
            raise AssertionError("boolean-mask indexing")
        if any(torch.is_tensor(t) and t.dim() == 0 for t in items):
            raise AssertionError("0-d tensor index")
        return orig_getitem(self, i)

    x = tokens(seed=91)
    bm.now.fill_(1)
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 3, 1, 0
    # (the model's own existing code has one sync per RoutedMLP forward, counts.tolist(): it runs unpatched; the memory's
    # hooks / after_backward / after_step run patched. torch.optim's CPU Adam reads its step counter with .item(): a
    # CPU-only artefact (on the GPU the step count is a host tensor), allowed from inside torch.optim only.)
    import sys as _sys
    logits, loss = m(x[:, :-1], x[:, 1:])
    for name in ("item", "__bool__", "nonzero", "__int__", "__float__", "numpy", "cpu"):
        orig = getattr(torch.Tensor, name)

        def guarded(*a, _o=orig, **k):
            f = _sys._getframe(1)
            while f is not None:
                if "torch" in f.f_code.co_filename and "optim" in f.f_code.co_filename:
                    return _o(*a, **k)
                f = f.f_back
            raise AssertionError("host sync")
        monkeypatch.setattr(torch.Tensor, name, guarded)
    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
    monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
    for name in ("bincount", "nonzero", "unique", "masked_select", "quantile", "nanquantile"):
        monkeypatch.setattr(torch, name, boom)
    bm.select_writes(logits, x[:, 1:], x[:, :-1])
    (loss + m.aux_loss()).backward()
    bm.after_backward()
    bm.after_step(1)
    monkeypatch.undo()
    assert calls["tolist"] <= 1                                           # the step end's one sync (write counts)


def test_p2_12b_gated_forward_pass_has_no_sync(monkeypatch):
    mlp, bm = _gated_mlp(seed=13)
    from spiking_lm import RoutedMLP
    x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    # run the gated hook directly on a prepared capacity layout (RoutedMLP itself has its existing counts.tolist())
    rt = _routing()
    e_k, p_k, t_k, E, C = rt
    xf = torch.randn(12, 16, dtype=torch.float64, requires_grad=True)
    buf = xf.new_zeros(E, C, 16).index_put((e_k, p_k), xf[t_k])

    def boom(*a, **k):
        raise AssertionError("host sync")

    for name in ("item", "tolist", "__bool__", "nonzero", "__int__", "__float__", "numpy", "cpu"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    for name in ("bincount", "nonzero", "unique", "masked_select", "quantile", "nanquantile"):
        monkeypatch.setattr(torch, name, boom)
    bm.begin_forward()
    b2, xs = bm(xf, buf, xf.view(2, 6, 16), None, e_k, p_k, t_k, None, None, 12, E, C, len(e_k), call=0)
    (b2.sum() + xs.sum()).backward()
    bm.after_backward()
    monkeypatch.undo()
    assert xf.grad.abs().sum() > 0


def test_p2_12c_static_shapes_do_not_depend_on_the_data():
    """The read buffers' sizes come from the pair-list length and N only (not from how many pairs open)."""
    shapes = []
    for beta in (-5.0, 5.0):
        mlp, bm = _gated_mlp(seed=14)
        bm.gate.beta.data.fill_(beta)
        rt = _routing()
        xf = torch.randn(12, 16, dtype=torch.float64, requires_grad=True)
        e_k, p_k, t_k, E, C = rt
        buf = xf.new_zeros(E, C, 16).index_put((e_k, p_k), xf[t_k])
        bm.begin_forward()
        bm(xf, buf, xf.view(2, 6, 16), None, e_k, p_k, t_k, None, None, 12, E, C, len(e_k), call=0)
        sel = bm._cache[0]
        shapes.append((tuple(sel["o_r"].shape), tuple(sel["o_s"].shape), tuple(sel["gate_rows"]["phi"].shape)))
    assert shapes[0] == shapes[1]


# ---------------------------------------------------------------- misc: eviction protection, sigma tracking, metrics
def test_p2_13_eviction_prefers_free_then_weakest_unprotected_never_touched():
    bm = _pbm()
    bm.glob.occ[:] = True
    bm.glob.birth[:] = 0
    bm.glob.strength.copy_(torch.linspace(1, 2, 64))
    bm.glob.strength[5] = -1.0
    bm.glob.touch[5] = 50                                                 # touched this step: never a victim
    bm.glob.birth[3] = 49                                                 # young: protected (< 50% protected)
    bm.glob.strength[3] = -2.0
    vic, ok = blob_write.victims(bm, bm.glob, torch.zeros(1, dtype=torch.long), 2, 50)
    assert vic[0].tolist() == [0, 1] and ok.all()
    bm.glob.occ[10] = False
    vic, ok = blob_write.victims(bm, bm.glob, torch.zeros(1, dtype=torch.long), 1, 50)
    assert vic[0].tolist() == [10]


def test_p2_14_metrics_cover_the_broken_if_list():
    m = p2model()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for i in range(3):
        p2step(m, opt, tokens(seed=95 + i), i)
    out = m.bm.metrics(3)
    for k in ("gm_open_rate_r", "gm_buffer_overflow", "gm_kappa", "gm_sigma2_g", "gm_live_g", "gm_private_fill_med",
              "gm_prob_fill", "gm_cell_gini", "gm_lam_pos_frac", "gm_hub_top1pct_share", "gm_v_norm_rel_med",
              "gm_shared_share_of_gain", "gm_placebo_mean", "gm_y_mean"):
        assert k in out, k


def test_p2_15_gate_parameters_stay_out_of_the_brains_optimizer_and_clip():
    tl = current().train_lm
    m, m0 = p2model(), loop(gm=1)
    names = [n for n, _ in m.named_parameters()]
    assert not any(n.startswith("bm.") for n in names)                     # (no memory parameter registered at all)
    gate_ids = {id(p) for p in m.bm.gate.parameters()}
    assert not gate_ids & {id(p) for p in m.parameters()}
    opt = tl.make_opt(m, 1e-3, 0.1)
    assert not gate_ids & {id(p) for g in opt.param_groups for p in g["params"]}
    assert "gate" in m.bm.state_dict()["_extra_state"]                      # (saved with the model)


# ---------------------------------------------------------------- run control: --stop-at, --nan-abort, --save-at, health
def test_p2_16_stop_at_then_resume_equals_the_uninterrupted_run_and_health_save_at(tmp_path):
    import json
    tl = current().train_lm
    data = _tiny_data(tmp_path / "data")
    extra = (GM2_ARGS + " --save-at 10 --health-every 5").split()
    run_a, ra, ca = _run_main(tl, tmp_path / "runs_a", "a", data, extra)
    out_b = tmp_path / "runs_b" / "b"
    try:
        _run_main(tl, tmp_path / "runs_b", "b", data, extra + ["--stop-at", "20"])
    except FileNotFoundError:                                              # (no final.pt after a stop: expected)
        pass
    log = (out_b / "train.log").read_text()
    assert "STOPPED (--stop-at 20)" in log and not (out_b / "final.pt").exists()
    run_b, rb, cb = _run_main(tl, tmp_path / "runs_b", "b", data, extra + ["--resume", str(out_b / "last.pt")])
    assert len(ra) == len(rb)
    for a, b in zip(ra, rb):
        a, b = dict(a), dict(b)
        for k in ("tok_s", "elapsed"):
            a.pop(k), b.pop(k)
        assert a == b
    for k in ca["state"]:
        if torch.is_tensor(ca["state"][k]):
            assert torch.equal(ca["state"][k], cb["state"][k]), k
        else:
            assert repr(ca["state"][k]) == repr(cb["state"][k]), k
    assert (run_a / "ckpt_10.pt").exists()
    h = [json.loads(l) for l in (run_a / "health.jsonl").read_text().splitlines()]
    assert len(h) == 8 and all(r["gm_guard_violations"] == 0 for r in h)
    assert any("gm_open_rate_mean" in r for r in h) and any("gm_waiting_fill" in r for r in h)


def test_p2_17_nan_abort_saves_the_state_and_exits(tmp_path, monkeypatch):
    tl = current().train_lm
    data = _tiny_data(tmp_path / "data")
    calls = {"n": 0}
    orig = torch.nn.utils.get_total_norm

    def fake(ps, *a, **k):
        calls["n"] += 1
        g = orig(ps, *a, **k)
        return g * float("nan") if calls["n"] == 7 else g
    monkeypatch.setattr(torch.nn.utils, "get_total_norm", fake)
    try:
        _run_main(tl, tmp_path / "runs", "n", data, ["--nan-abort", "1"])
    except FileNotFoundError:
        pass
    run = tmp_path / "runs" / "n"
    log = (run / "train.log").read_text()
    assert "step 6: NAN ABORT" in log and (run / "nan_abort.pt").exists() and not (run / "final.pt").exists()
    ck = torch.load(run / "nan_abort.pt", weights_only=False)
    assert ck["step"] == 6 and "rng" in ck and "opt" in ck


# ---------------------------------------------------------------- the oracle's kernel and the label-free gate init
def test_p2_18_scale_free_gaussian_kernel():
    from blob_memory import kernel_weights
    g = torch.Generator().manual_seed(15)
    q, keys = torch.randn(5, 8, generator=g), torch.randn(5, 4, 8, generator=g)
    valid = torch.ones(5, 4, dtype=torch.bool)
    valid[1, 3] = False
    d = (q[:, None] - keys).pow(2).sum(-1)
    w, A, d1 = kernel_weights(q, keys, valid, 0.1, torch.tensor(1.0), gauss_scale=torch.tensor(0.3), use_A=False)
    ref = torch.where(valid, torch.exp(-(d - d.masked_fill(~valid, 1e9).amin(-1, keepdim=True)) / 0.3), torch.zeros_like(d))
    assert torch.allclose(w, ref / ref.sum(-1, keepdim=True), atol=1e-6) and torch.equal(A, torch.ones(5))
    w2, _, _ = kernel_weights(q, keys, valid, 0.1, torch.tensor(1.0), gauss_scale=torch.tensor(0.3) * 7, use_A=False,
                              d=d * 7)
    assert torch.allclose(w, w2, atol=1e-6)                               # scale-free: d and m scale together
    # the running m: from training reads only, kept on the device, the median of d2 - d1, saved in the state dict
    m = p2model()
    bm = m.bm
    assert float(bm.gapm_g) == 0 and "bm.gapm_g" in m.state_dict()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    p2step(m, opt, tokens(seed=100), 0)
    m0 = float(bm.gapm_g)
    assert m0 > 0
    m.eval()
    with torch.no_grad():
        m(tokens(seed=101)[:, :-1])
        bm.after_step(1)                                                   # (guarded: counted, nothing changes)
    assert float(bm.gapm_g) == m0 and bm.guard_violations == 1 and not bm._mgaps["g"]


def test_p2_19_label_free_gate_init_reaches_the_target_open_rate():
    m = p2model(gm_gate_warm=4)
    g = m.bm.gate
    assert torch.equal(g.pg.data, torch.tensor([-0.01, 0.01, 0.01])) and torch.equal(g.pp.data, g.pg.data)   # (r6 G2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-9)
    # (r6 G1/G2: the gate now learns in the target's units, so within a few steps it moves its own open rate; the
    # label-free start itself is checked with the gate's optimizer frozen)
    g.lr = 0.0
    if g.opt is not None:
        for grp in g.opt.param_groups:
            grp["lr"] = 0.0
    for i in range(5):
        p2step(m, opt, tokens(seed=110 + i), i, writes=False)
    assert int(g.warm_left) == 0 and float(g.open_bias) != 0
    b0 = float(g.open_bias)
    p2step(m, opt, tokens(seed=120), 6, writes=False)
    assert float(g.open_bias) == b0                                        # fixed after the warm-up
    st = torch.stack(m.bm._stats_step)
    rate = float((st[:, 0] + st[:, 2]).sum() / (st[:, 1] + st[:, 3]).sum())
    assert 0.08 < rate < 0.45, rate                                        # ~23% (+ exploration), from no labels


def test_p2_20_in_run_diagnostics_change_nothing(tmp_path):
    import json
    tl = current().train_lm
    data = _tiny_data(tmp_path / "data")
    _, ra, ca = _run_main(tl, tmp_path / "runs", "off", data, GM2_ARGS.split())
    run, rb, cb = _run_main(tl, tmp_path / "runs", "on", data, GM2_ARGS.split() + ["--diag-every", "5"])
    for a, b in zip(ra, rb):
        a, b = dict(a), dict(b)
        for k in ("tok_s", "elapsed"):
            a.pop(k), b.pop(k)
        assert a == b
    for k in ca["state"]:
        if torch.is_tensor(ca["state"][k]):
            assert torch.equal(ca["state"][k], cb["state"][k]), k
    d = [json.loads(l) for l in (run / "diag.jsonl").read_text().splitlines()]
    assert len(d) == 8
    last = d[-1]
    for k in ("experts", "expert_load_entropy", "loop_pass_ce", "routing_overlap_lag1", "concept_level_loss",
              "mem_fill_global", "mem_waiting_size", "mem_age_hist_global", "mem_open_rate_by_call",
              "mem_open_rate_by_expert_group", "mem_gate_signal_mean", "mem_kernel_m_global"):
        assert k in last, k
