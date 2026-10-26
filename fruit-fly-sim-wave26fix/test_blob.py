"""Tests of the global + per-expert blob memory (plan_v2/25d + the review; blob_memory.py). CPU only, tiny sizes.

    set CUDA_VISIBLE_DEVICES=-1 & python -m pytest fruit-fly-sim/test_blob.py -q -p no:cacheprovider
"""
import copy
import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import blob_memory
from blob_memory import (BagRead, BlobMemory, ProbationStore, _bag, blend, cap_norm, kernel_weights, read_reference,
                         stochastic_bf16)
from test_memory import _drive, _run_main, _tiny_data, current, mem_loop, train_step

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
BASE = HERE / "plan_v2" / "baseline_pre_blob"
MODS = ("spiking_lm", "loop_brain", "brain_opt", "neurogenesis", "train_lm", "memory", "concepts", "blob_memory")
_BASELINE = None


def baseline():
    """The code before the blob memory (plan_v2/baseline_pre_blob) as separate module objects (as test_memory does for
    baseline_pre_mem): the current modules leave sys.modules while it imports and come back afterwards."""
    global _BASELINE
    if _BASELINE is None:
        saved = {n: sys.modules.pop(n) for n in MODS if n in sys.modules}
        dont = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(BASE))
        try:
            _BASELINE = SimpleNamespace(**{n: importlib.import_module(n) for n in
                                           ("spiking_lm", "loop_brain", "brain_opt", "neurogenesis", "train_lm")})
        finally:
            sys.path.remove(str(BASE))
            sys.dont_write_bytecode = dont
            for n in MODS:
                sys.modules.pop(n, None)
            sys.modules.update(saved)
        assert Path(_BASELINE.loop_brain.__file__).parent == BASE
    return _BASELINE


def small(**kw):
    cfg = dict(d=16, n_exp=3, max_experts=4, G=64, P=8, kd=8, cand=6, k_g=4, k_p=3, cells=4, cell_cap=32, probe=4,
               q_cap=64, fit_samples=256, ring=64)
    cfg.update(kw)
    return BlobMemory(**cfg)


def populate(bm, n_exp, d, seed=11, n_glob=40, n_priv=4, scale=1.0):
    """Test-only: blobs through the write() stub (keys from random model-width inputs)."""
    g = torch.Generator().manual_seed(seed)
    bm.write(torch.randn(n_glob, d, generator=g), scale * torch.randn(n_glob, d, generator=g), force=True)
    ids = torch.arange(-1, n_exp).repeat_interleave(n_priv)
    slots = torch.arange(n_priv).repeat(n_exp + 1)
    bm.write(torch.randn(len(ids), d, generator=g), scale * torch.randn(len(ids), d, generator=g), expert_ids=ids,
             slots=slots, force=True)


def fitted(**kw):
    bm = small(**kw)
    bm.fit()
    return bm


def loop(gm=0, dtype=torch.float64, **kw):
    from loop_brain import LoopBrain
    cfg = dict(vocab=50, d=32, d_s=8, heads=4, head_dim=8, spike="graded", chunk=16, k_mean=4, k_min=2, k_max=6,
               bptt=3, state="", refractory=1, slots=4, waves=True, ws_direct=True, head_group=2, head_groups=2,
               deep_sup=0.3, halt=True, think_w=0.1, think_fb=True, think_use=True, think_target="latent",
               moe=dict(n_exp=8, k=2, hidden=16, shared=16, capacity=1.5, birth_mode="split", birth_delta=0.1))
    if gm:
        cfg.update(gm=1, gm_global=128, gm_private=8, gm_max_experts=12, gm_cand=8, gm_topk_g=4, gm_topk_p=4,
                   gm_cells=4, gm_cell_cap=64, gm_probe=4, gm_qcap=128, gm_h0=4.0, gm_fit_samples=512, gm_ring=128)
    cfg.update(kw)
    torch.manual_seed(0)
    m = LoopBrain(**cfg).to(dtype)
    for s in m.spikes():
        s.calibrated.fill_(True)
    return m


def tokens(B=2, T=17, seed=3):
    return torch.randint(1, 50, (B, T), generator=torch.Generator().manual_seed(seed))


def _routing(N=12, E=3, k=2, seed=5):
    """A fixed (token, expert) capacity layout like RoutedMLP's."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(N)])
    flat = idx.flatten()
    order = flat.argsort(stable=True)
    e_sorted, tok = flat[order], order // k
    counts = torch.zeros(E, dtype=torch.long).index_add_(0, flat, torch.ones_like(flat))
    pos = torch.arange(N * k) - (counts.cumsum(0) - counts)[e_sorted]
    return e_sorted, pos, tok, E, int(counts.max())


def _call(bm, xf, rt, call=None, **kw):
    e_k, p_k, t_k, E, C = rt
    buf = xf.new_zeros(E, C, xf.size(1)).index_put((e_k, p_k), xf[t_k])
    return bm.inject(xf, buf, xf.view(1, -1, xf.size(1)), e_k, p_k, t_k, call=call, **kw)


# ---------------------------------------------------------------- 1. storage
def test_01_storage_shapes_dtypes_growth_and_births():
    bm = small()
    assert bm.glob.key.shape == (64, 8) and bm.glob.value.shape == (64, 16) and bm.glob.value.dtype == torch.float32
    assert bm.priv.key.shape == (5, 8, 8) and bm.priv.value.dtype == torch.bfloat16 and bm.shared_idx == 4
    for sp in (bm.glob, bm.priv):
        assert sp.key.dtype == sp.opt_v.dtype == sp.strength.dtype == torch.float32 and sp.key_bf.dtype == torch.bfloat16
        for n in sp.LONG:
            assert getattr(sp, n).dtype == torch.long and getattr(sp, n).shape == sp.occ.shape
    assert bm.glob.cell_pref.shape == (64, 2) and not hasattr(bm.priv, "cell")
    sd = bm.state_dict()
    assert "glob.key" in sd and "glob.key_bf" not in sd and "table" not in sd and "proj" in sd
    assert len(list(bm.parameters())) == 0 and small(inj_scale=True).inj_scale.shape == (16,)
    bm.double()                                                                     # storage dtypes are fixed
    assert bm.glob.key.dtype == torch.float32 and bm.priv.value.dtype == torch.bfloat16 and bm.stat_sum.dtype == torch.float64
    bm.fit()
    populate(bm, 3, 16)
    bm.priv.strength.fill_(0.8)
    bm.priv.opt_v.fill_(2.0)
    before = {n: getattr(bm.priv, n).clone() for n in bm.priv.persistent_names()}
    bm.on_grow(torch.tensor([1, 2, 1]), 3, split=True)                # 3 split births: 6 > 4 preallocated
    assert bm.priv.key.shape == (7, 8, 8) and bm.shared_idx == 6 and bm.Emax == 6 and int(bm.n_live) == 6
    for n, t in before.items():
        new = getattr(bm.priv, n)
        assert torch.equal(new[:3], t[:3]), n                                        # parents unchanged
        assert torch.equal(new[6], t[4]), n                                          # the shared space moved
        if n not in ("strength", "opt_v", "opt_n"):
            assert torch.equal(new[3], t[1]) and torch.equal(new[4], t[2]) and torch.equal(new[5], t[1]), n
    assert torch.equal(bm.priv.strength[3:6], torch.full((3, 8), 0.4))               # half strength
    assert (bm.priv.opt_v[3:6] == 0).all() and (bm.priv.opt_v[:3] == 2).all()       # fresh optimizer state
    assert torch.equal(bm.priv.key_bf[4], bm.priv.key_bf[2]) and torch.equal(bm.priv.ksq[6], before["key"][4].pow(2).sum(-1).where(before["occ"][4], torch.tensor(float("inf"))))
    bm.on_grow(torch.tensor([0]), 6, split=False)                     # ramp: the child starts empty
    assert not bm.priv.occ[6].any() and bm.priv.occ[bm.shared_idx].sum() == 4


def test_02_address_map_fit_window_and_no_reads_before_the_fit():
    bm = small()
    x = torch.randn(300, 16) * torch.linspace(0.5, 3, 16) + 2
    assert torch.allclose(bm.address(x), x @ bm.proj)                               # before the fit: identity
    rt = _routing()
    xf = torch.randn(12, 16)
    b, xs = _call(bm, xf, rt)
    assert torch.equal(xs.view(12, 16), xf)                                         # nothing read before the fit
    p0 = bm.proj.clone()
    bm.train()
    bm.observe(x[:100])                                                             # collect off: ignored
    assert bm.stat_n.item() == 0
    bm.collect = True
    for a in range(0, 300, 100):
        bm.observe(x[a:a + 100])
    bm.observe_write(x[:50] + 1)
    assert bm._fit_n == [256, 50]
    bm.fit()
    assert bm._fitted and bool(bm.fitted) and bm.fit_x.numel() == 0
    mu, sd = x.double().mean(0), x.double().std(0, unbiased=False)
    assert torch.allclose(bm.mu.double(), mu, atol=1e-4) and torch.allclose(bm.sigma.double(), sd, atol=1e-4)
    z = bm.address(x)
    assert abs(z.pow(2).mean().item() - 1) < 0.25                                   # ~unit variance per coordinate
    assert torch.equal(bm.proj, p0) and torch.equal(small().proj, p0)               # fixed seed, never learned
    bm.observe(x * 100)
    assert bm.stat_n.item() == 300


# ---------------------------------------------------------------- 2. search
def _exact(q, key, occ, k):
    d = (q[:, None, :].double() - key[None].double()).pow(2).sum(-1)
    d = torch.where(occ[None], d, torch.full_like(d, float("inf")))
    return d.topk(k, -1, largest=False)


def _store(bm, keys, occ=None):
    bm.glob.key.copy_(keys)
    bm.glob.occ.copy_(torch.ones(len(keys), dtype=torch.bool) if occ is None else occ)
    bm.glob.refresh()
    bm.fit() if not bm._fitted else bm.assign_cells()


def test_03_brute_search_matches_exact_fp32():
    torch.manual_seed(1)
    bm = small(G=512, kd=64, d=96, cand=16, search="brute", brute_chunk=37)
    occ = torch.rand(512) < 0.7
    _store(bm, torch.randn(512, 64), occ)
    q = torch.cat([bm.glob.key[:40] + 0.05 * torch.randn(40, 64), torch.randn(40, 64)])
    ref = _exact(q, bm.glob.key, occ, 16)
    idx, d2 = bm.search_global(q)
    hit = (idx[:, :, None] == ref.indices[:, None, :]).any(-1).float().mean()     # recall@16 of bf16 + fp32 re-rank
    assert hit >= 0.97 and occ[idx].all()
    near = torch.arange(40)[occ[:40]]
    assert torch.equal(idx[near, 0], near)                                          # the nearest is always found
    same = idx == ref.indices
    assert torch.allclose(d2[same].double(), ref.values[same], rtol=1e-4)           # exact fp32 distances
    assert (d2[:, 1:] >= d2[:, :-1]).all()


def test_04_ivf_with_every_cell_probed_equals_brute_force():
    torch.manual_seed(2)
    bm = small(G=512, kd=16, cand=16, cells=8, cell_cap=512, probe=8, q_cap=200, cell_topk=16)
    _store(bm, torch.randn(512, 16), torch.rand(512) < 0.8)
    q = torch.randn(60, 16)
    ivf = bm.search_global(q)
    bm.search = "brute"
    br = bm.search_global(q)
    ref = _exact(q, bm.glob.key, bm.glob.occ, 16)
    assert (ivf[0] == ref.indices).all(1).float().mean() >= 0.95
    assert (ivf[0][:, :4] == br[0][:, :4]).all(1).float().mean() >= 0.95


def test_05_ivf_recall_at_16_on_low_rank_data():
    """Recall@16 of IVF (64 cells, n_p 16) vs exact search on synthetic low-rank clustered keys >= 0.9."""
    g = torch.Generator().manual_seed(3)
    kd, G = 64, 8192
    basis = torch.randn(8, kd, generator=g)
    centres = torch.randn(40, 8, generator=g) * 3
    lab = torch.randint(0, 40, (G,), generator=g)
    keys = (centres[lab] + 0.6 * torch.randn(G, 8, generator=g)) @ basis + 0.1 * torch.randn(G, kd, generator=g)
    bm = small(G=G, kd=kd, d=kd, cand=16, cells=64, cell_cap=512, probe=16, q_cap=512)
    bm.proj.copy_(torch.eye(kd))                                            # (keys given directly in x_hat space)
    bm.train()
    bm.collect = True
    bm.observe(keys)
    bm.fit()
    bm.glob.key.copy_(keys)
    bm.glob.occ.fill_(True)
    bm.glob.refresh()
    bm.assign_cells()
    qi = torch.randint(0, G, (300,), generator=g)
    q = keys[qi] + 0.3 * torch.randn(300, kd, generator=g)
    idx, _ = bm.search_global(q)
    ref = _exact(q, keys, bm.glob.occ, 16).indices
    recall = (idx[:, :, None] == ref[:, None, :]).any(-1).float().mean().item()
    assert recall >= 0.9, recall
    assert bm.stats["probe_overflow"].item() < 0.05


def test_06_full_cells_spill_to_the_second_cell_and_refit_reassigns():
    torch.manual_seed(4)
    bm = small(G=64, kd=4, d=4, cells=4, cell_cap=8, probe=4, q_cap=64, cand=8)
    bm.proj.copy_(torch.eye(4))
    bm.fit()
    bm.centroids.copy_(torch.tensor([[0., 0, 0, 0], [5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0]]))
    keys = torch.cat([torch.randn(12, 4) * 0.1, torch.randn(4, 4) * 0.1 + torch.tensor([5., 0, 0, 0])])
    occ = torch.zeros(64, dtype=torch.bool)
    occ[:16] = True
    bm.glob.key[:16] = keys
    bm.glob.occ.copy_(occ)
    bm.glob.refresh()
    bm.assign_cells()
    cell = bm.glob.cell[:16]
    assert (cell == 0).sum() == 8 and (cell >= 0).all()                              # 4 of cell 0's 12 spilled
    assert bm.cell_spilled.item() == 4 and bm.cell_overflow.item() == 0
    assert set(bm.table[0][bm.table[0] >= 0].tolist()) <= set(range(16))
    idx, d2 = bm.search_global(keys[:1])
    assert idx[0, 0] == 0                                                            # still found exactly
    c0 = bm.centroids.clone()
    bm._ring_add(torch.randn(64, 4) * 0.1 + 3, 0)
    bm.refit()
    assert not torch.equal(bm.centroids, c0) and (bm.glob.cell[:16] >= 0).all()


def test_07_private_batched_search_matches_per_expert_loop():
    torch.manual_seed(5)
    E, C, P, kd = 5, 6, 16, 8
    bm = small(n_exp=E, max_experts=E, P=P, kd=kd, cand=4, chunk_e=2)
    bm.priv.key.copy_(torch.randn(E + 1, P, kd))
    bm.priv.occ.copy_(torch.rand(E + 1, P) < 0.7)
    bm.priv.refresh()
    Q = torch.randn(E, C, kd)
    cand = bm.search_private(Q, E)
    e_k = torch.arange(E).repeat_interleave(C)
    p_k = torch.arange(C).repeat(E)
    flat, d2 = bm._rerank_priv(Q.view(-1, kd), e_k, cand[e_k, p_k], 4)
    for e in range(E):
        ref = _exact(Q[e], bm.priv.key[e], bm.priv.occ[e], 4)
        got = d2.view(E, C, 4)[e].double()
        assert torch.allclose(got, ref.values, rtol=1e-5, atol=1e-5)
        assert ((flat.view(E, C, 4)[e] - e * P) == ref.indices).float().mean() >= 0.9
    bm.chunk_e = 0
    assert torch.equal(bm.search_private(Q, E), cand)


# ---------------------------------------------------------------- 3. read maths and gradients
def test_08_kernel_and_blend_gradcheck_and_bounded_near_zero_distance():
    torch.manual_seed(6)
    n, k, kd, D = 5, 4, 3, 2
    dt = torch.float64
    q = torch.randn(n, kd, dtype=dt, requires_grad=True)
    kg, kp = torch.randn(n, k, kd, dtype=dt), torch.randn(n, k, kd, dtype=dt)
    vg = torch.randn(n, k, D, dtype=dt, requires_grad=True)
    vp = torch.randn(n, k, D, dtype=dt, requires_grad=True)
    valid_g = torch.ones(n, k, dtype=torch.bool)
    valid_p = valid_g.clone()
    valid_g[1, 2:] = False
    valid_g[2] = False                                                               # no global blob
    valid_p[2] = False                                                               # nor private: reads 0
    h0g, h0p = torch.tensor(1.5, dtype=dt), torch.tensor(2.0, dtype=dt)
    f = lambda q, a, b: read_reference(q, kg, a, valid_g, kp, b, valid_p, 0.3, h0g, h0p, 1.0)
    assert torch.autograd.gradcheck(f, (q, vg, vp))
    r = f(q, vg, vp)
    assert torch.equal(r[2], torch.zeros(D, dtype=dt))
    # the definition by hand on row 0
    def one(kk, vv, h0):
        d = (q[0] - kk[0]).pow(2).sum(-1)
        w = (-(d - d.min()) / ((d.min() + h0 ** 2) * 0.3)).softmax(0)
        return torch.exp(-d.min() / (2 * h0 ** 2)), (w[:, None] * vv[0]).sum(0)
    Ag, ug = one(kg, vg, h0g)
    Ap, up = one(kp, vp, h0p)
    assert torch.allclose(r[0], torch.maximum(Ag, Ap) * (Ag * ug + Ap * up) / (Ag + Ap))
    # the gradient reaches the query through both w and A
    wg, A, _ = kernel_weights(q, kg, valid_g, 0.3, h0g)
    assert torch.autograd.grad(wg[0, 1], q, retain_graph=True)[0].abs().sum() > 0
    assert torch.autograd.grad(A[0], q)[0].abs().sum() > 0
    # a query sitting on a key: finite, bounded gradient (the (d_1 + c h0^2) denominator)
    q0 = kg[:, 0].clone().requires_grad_()
    w0, _, _ = kernel_weights(q0, kg, torch.ones(n, k, dtype=torch.bool), 0.1, h0g)
    gq = torch.autograd.grad(w0[:, 1].sum(), q0)[0]
    assert torch.isfinite(gq).all() and gq.abs().max() < 1e3


def test_09_embedding_bag_blend_equals_the_reference_gather_blend():
    torch.manual_seed(7)
    table = torch.randn(20, 6)
    rows = torch.tensor([[1, 4, 4], [9, 0, 3], [19, 2, 2]])
    w = torch.rand(3, 3, requires_grad=True)
    got = {}
    out = BagRead.apply(w, table, rows, lambda r, g: got.update(r=r, g=g))
    W = torch.randn(3, 6)
    (out * W).sum().backward()
    leaf = table.clone().requires_grad_()
    w2 = w.detach().clone().requires_grad_()
    ref = (w2[..., None] * leaf[rows]).sum(1)
    (ref * W).sum().backward()
    assert torch.allclose(out, ref, atol=1e-6) and torch.allclose(w.grad, w2.grad, atol=1e-6)
    dense = torch.zeros_like(table).index_add_(0, got["r"].flatten(), got["g"].reshape(-1, 6))
    assert torch.allclose(dense, leaf.grad, atol=1e-6)
    assert torch.allclose(_bag(w.detach().double(), table, rows), ref.detach().double(), atol=1e-6)


def test_10_inject_equals_the_dense_reference_values_and_gradients():
    torch.manual_seed(8)
    bm = fitted().double()
    bm.priv_bf16 = False
    bm.priv.value_dtype = torch.float32
    bm.priv.value = bm.priv.value.float()
    populate(bm, 3, 16, scale=0.5)
    bm.set_h0(3.0, 3.0)
    rt = _routing()
    e_k, p_k, t_k, E, C = rt
    xf = torch.randn(12, 16, dtype=torch.float64, requires_grad=True)
    buf, xs = _call(bm, xf, rt, call=0)
    sel = bm._cache[0]
    Wb, Ws = torch.randn_like(buf), torch.randn_like(xs)
    ((buf * Wb).sum() + (xs * Ws).sum()).backward()
    vg = {n: g.clone() for n, g in bm._vg.items()}
    # the reference: dense gathers from leaf value tables, the same selection
    gv = bm.glob.value.double().clone().requires_grad_()
    pv = bm.priv.value.double().view(-1, 16).clone().requires_grad_()
    x2 = xf.detach().clone().requires_grad_()
    q = bm.address(x2)
    K = lambda sp, flat: sp.key.view(-1, 8)[flat].double()
    Vd = lambda sp, flat: sp.occ.view(-1)[flat]
    gi, pf, sf = sel["gi"][:, :bm.k_g], sel["pf"][:, :bm.k_p], sel["sf"][:, :bm.k_p]
    r_pair = read_reference(q[t_k], K(bm.glob, gi[t_k]), gv[gi[t_k]], Vd(bm.glob, gi[t_k]), K(bm.priv, pf), pv[pf],
                            Vd(bm.priv, pf), bm.tau, bm.h0_g.double(), bm.h0_p.double())
    r_sh = read_reference(q, K(bm.glob, gi), gv[gi], Vd(bm.glob, gi), K(bm.priv, sf), pv[sf], Vd(bm.priv, sf),
                          bm.tau, bm.h0_g.double(), bm.h0_p.double())
    buf_ref = x2.new_zeros(E, C, 16).index_put((e_k, p_k), x2[t_k] + r_pair)
    xs_ref = x2.view(1, -1, 16) + r_sh.view(1, -1, 16)
    assert torch.allclose(buf, buf_ref, atol=1e-6) and torch.allclose(xs, xs_ref, atol=1e-6)
    ((buf_ref * Wb).sum() + (xs_ref * Ws).sum()).backward()
    assert torch.allclose(xf.grad, x2.grad, atol=1e-5)                               # through w and A to the query
    assert torch.allclose(vg["g"].double(), gv.grad, atol=1e-5) and torch.allclose(vg["p"].double(), pv.grad, atol=1e-5)
    assert vg["g"].abs().sum() > 0 and vg["p"].abs().sum() > 0 and vg["g"].dtype == vg["p"].dtype == torch.float32
    assert bm.glob.key.grad is None and not bm.glob.key.requires_grad               # none to the keys
    # match_grad off: the query gets only the direct path
    bm.match_grad = False
    bm.begin_forward()
    x3 = xf.detach().clone().requires_grad_()
    b3, s3 = _call(bm, x3, rt)
    ((b3 * Wb).sum() + (s3 * Ws).sum()).backward()
    direct = torch.zeros_like(x3).index_add_(0, t_k, Wb[e_k, p_k]) + Ws.view(12, 16)
    assert torch.allclose(x3.grad, direct)


def test_11_read_modes_masks_pair_valid_compute_buffer_and_mask_fn():
    bm = fitted()
    populate(bm, 3, 16)
    rt = _routing()
    e_k, p_k, t_k, E, C = rt
    xf = torch.randn(12, 16)
    base = xf.new_zeros(E, C, 16).index_put((e_k, p_k), xf[t_k])
    full, fs = _call(bm, xf, rt)
    assert (full != base).any() and (fs.view(12, 16) != xf).any()
    bm.read_mode = "none"
    b, s = _call(bm, xf, rt)
    assert torch.equal(b, base) and torch.equal(s.view(12, 16), xf)
    bm.read_mode = "random"
    b, s = _call(bm, xf, rt)
    changed = (b != base).any(-1)[e_k, p_k]
    assert 0 < changed.sum() < len(e_k)
    bm.read_mode = "all"
    pv = torch.arange(len(e_k)) % 2 == 0
    b, _ = _call(bm, xf, rt, pair_valid=pv)
    assert torch.equal(b[e_k[~pv], p_k[~pv]], base[e_k[~pv], p_k[~pv]])
    bm.mask_fn = lambda info: (info["e_k"] == 1, torch.zeros_like(info["t_k"][:0]).new_zeros(12, dtype=torch.bool))
    b, s = _call(bm, xf, rt)
    assert torch.equal(s.view(12, 16), xf) and torch.equal(b[0], base[0]) and (b[1] != base[1]).any()
    # the static compute buffer: with room for every open pair it equals the full read
    bm.compute_frac = 0.5
    b2, _ = _call(bm, xf, rt)
    assert torch.allclose(b2, b, atol=1e-6)
    bm.mask_fn = None
    b3, _ = _call(bm, xf, rt)                                  # all open, half computed: the rest unchanged
    assert ((b3 != base).any(-1)[e_k, p_k]).sum() == len(e_k) // 2
    bm.compute_frac = 1.0
    empty = fitted()
    b, s = _call(empty, xf, rt)
    assert torch.equal(b, base) and torch.equal(s.view(12, 16), xf)                 # an empty store reads 0


def test_12_activation_checkpoint_recompute_is_consistent():
    """RoutedMLP with the blob read (random read mask) inside torch.utils.checkpoint == without it: outputs, input /
    router gradients and the content gradients; the cached selection is reused (the mask stream advances once)."""
    from spiking_lm import RoutedMLP
    torch.manual_seed(9)
    mlp = RoutedMLP(16, n_exp=3, k=2, hidden=8, shared=8, capacity=0).double()
    mlp.defer = True                                     # (no balancer update between the forward and its recompute)
    bm = fitted(read_mode="random").double()
    populate(bm, 3, 16, scale=0.5)
    x0 = torch.randn(2, 6, 16, dtype=torch.float64)
    res = []
    for ck in (False, True):
        bm._gens, bm._gen_states = {}, {}
        bm.begin_forward()
        bm.zero_value_grads(free=True)
        mlp.zero_grad()
        x = x0.clone().requires_grad_()
        hook = lambda *a: bm(*a, call=0)
        if ck:
            y = torch.utils.checkpoint.checkpoint(lambda t: mlp(t, bmem=hook), x, use_reentrant=False)
        else:
            y = mlp(x, bmem=hook)
        (y * torch.linspace(-1, 1, 16, dtype=torch.float64)).sum().backward()
        res.append((y.detach(), x.grad, mlp.w_up.grad.clone(), mlp.router.weight.grad.clone(), mlp.shared.up.weight.grad.clone(),
                    {n: g.clone() for n, g in bm._vg.items()}, bm._gens["mask"].get_state()))
    a, b = res
    assert torch.equal(a[0], b[0]) and all(torch.allclose(u, v) for u, v in zip(a[1:5], b[1:5]))
    assert all(torch.allclose(a[5][n], b[5][n]) for n in a[5]) and torch.equal(a[6], b[6])
    assert a[1].abs().sum() > 0 and a[5]["g"].abs().sum() > 0


# ---------------------------------------------------------------- 4. the content update
def _rmsprop_ref(val, v, g, touched, lr, b2, clip, eps=1e-8):
    val, v = val.clone().float(), v.clone()
    for i in range(val.size(0)):
        if not touched[i]:
            continue
        v[i] = b2 * v[i] + (1 - b2) * g[i].pow(2).mean()
        u = lr * g[i] / (v[i].sqrt() + eps)
        n = u.norm()
        if clip and n > clip:
            u = u * clip / n
        val[i] -= u
    return val, v


def test_13_lazy_rowwise_rmsprop_matches_reference_and_untouched_rows_are_bit_identical():
    torch.manual_seed(10)
    bm = fitted(priv_bf16=False)
    populate(bm, 3, 16)
    rt = _routing()
    v0 = bm.glob.value.clone()
    state0 = {n: getattr(bm.glob, n).clone() for n in bm.glob.persistent_names()}
    touched_ever = torch.zeros(64, dtype=torch.bool)
    for step in range(5):
        xf = torch.randn(12, 16, requires_grad=True)
        bm.begin_forward()
        buf, xs = _call(bm, xf, rt)
        (buf.pow(2).sum() + xs.sum()).backward()
        g, t = bm._vg["g"].clone(), bm._touched["g"].clone()
        touched_ever |= t
        ref_val, ref_v = _rmsprop_ref(bm.glob.value, bm.glob.opt_v, g, t, 0.03, 0.9, 0.05)
        bm.apply_value_grads()
        assert torch.allclose(bm.glob.value, ref_val, atol=1e-6) and torch.allclose(bm.glob.opt_v, ref_v, atol=1e-7)
        assert bm._vg["g"].abs().sum() == 0 and not bm._touched["g"].any()
    assert 0 < touched_ever.sum() < 64
    un = ~touched_ever
    assert torch.equal(bm.glob.value[un], v0[un])                                     # bit-identical
    for n in ("opt_v", "opt_n", "key", "strength"):
        assert torch.equal(getattr(bm.glob, n)[un], state0[n][un]), n
    assert (bm.glob.opt_n[touched_ever] > 0).all()


def test_14_stochastic_bf16_rounding_is_unbiased():
    g = torch.Generator().manual_seed(11)
    x = torch.full((4000,), 1.0) + torch.rand(4000, generator=g) * 0.01
    v32, vb = x.clone(), x.to(torch.bfloat16)
    step = torch.full((4000,), 2e-5)                                          # far below the bf16 step at 1.0
    for _ in range(1000):
        v32 = v32 - step
        vb = stochastic_bf16(vb.float() - step, g)
    err = (vb.float() - v32).mean().abs().item()
    assert err < 1e-3, err                                                    # plain rounding would not move at all
    assert (x.to(torch.bfloat16).float() - (x - 0.02)).mean() > 0.015
    b = torch.tensor([1.0, 2.5, -3.0], dtype=torch.bfloat16)
    assert torch.equal(stochastic_bf16(b.float(), g), b)                      # bf16 numbers unchanged


def test_15_private_bf16_contents_learn_with_fp32_grads_untouched_identical():
    torch.manual_seed(12)
    bm = fitted()
    populate(bm, 3, 16)
    rt = _routing()
    p0 = bm.priv.value.clone()
    xf = torch.randn(12, 16, requires_grad=True)
    buf, xs = _call(bm, xf, rt)
    (buf.sum() + xs.sum()).backward()
    assert bm._vg["p"].dtype == torch.float32
    t = bm._touched["p"].clone()
    bm.apply_value_grads()
    pv = bm.priv.value.view(-1, 16)
    assert torch.equal(pv[~t], p0.view(-1, 16)[~t]) and (pv[t] != p0.view(-1, 16)[t]).any()


def test_16_utilities_probation_cap_norm_reset_and_track_xf():
    assert small(probation_rate=2, probation_life=5).prob.key.shape == (10, 8) and small().prob is None
    v = torch.randn(5, 4) * 10
    ref = torch.full((5,), 2.0)
    c = cap_norm(v, ref, 0.3)
    assert torch.allclose(c.norm(dim=-1), torch.full((5,), 0.6)) and torch.equal(cap_norm(v * 1e-3, ref, 0.3), v * 1e-3)
    bm = fitted()
    bm.glob.opt_v.fill_(1.0)
    bm.reset_opt_state("g", torch.tensor([3, 4]))
    assert bm.glob.opt_v[3] == 0 and bm.glob.opt_v[5] == 1
    m = loop(gm=1, gm_track_xf=1)
    m.eval()
    m.K = 3
    seen = []
    h = m.blocks[0].mlp.register_forward_pre_hook(lambda mod, a: seen.append(a[0].detach().reshape(-1, 32)))
    with torch.no_grad():
        m(tokens()[:, :-1])
    h.remove()
    assert len(seen) == 3 and m.bm.xf_cnt is not None and (m.bm.xf_cnt == 3).all()
    assert torch.allclose(m.bm.xf_sum, sum(s.float() for s in seen), atol=1e-5)    # xf = the routed MLP's input


# ---------------------------------------------------------------- 5. LoopBrain: births, save / load, parameters
def test_17_split_births_copy_the_parents_private_blobs_through_loopbrain():
    m = loop(gm=1)
    bm = m.bm
    bm.fit()
    E0 = m.blocks[0].mlp.E
    populate(bm, E0, 32)
    before = bm.priv.key.clone()
    src = m.grow_experts(3)
    assert m.blocks[0].mlp.E == E0 + 3 and int(bm.n_live) == E0 + 3
    for i, s in enumerate(src.tolist()):
        assert torch.equal(bm.priv.key[E0 + i], before[s]) and torch.equal(bm.priv.value[E0 + i], bm.priv.value[s])
    assert torch.equal(bm.priv.key[:E0], before[:E0]) and torch.equal(bm.priv.key[bm.shared_idx], before[-1])
    m.grow_experts(4)                                                     # past the preallocation (12)
    assert bm.priv.occ.size(0) == m.blocks[0].mlp.E + 1 and bm.priv.occ[bm.shared_idx].sum() == 4
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(tokens()[:, :-1])).all()
    r = loop(gm=1, moe=dict(n_exp=8, k=2, hidden=16, shared=16, capacity=1.5, birth_mode="ramp"))
    r.bm.fit()
    populate(r.bm, 8, 32)
    r.grow_experts(2)
    assert not r.bm.priv.occ[8:10].any()                                  # ramp: empty


def test_18_save_load_round_trip_restores_everything():
    from loop_brain import LoopBrain
    m = loop(gm=1, gm_read_mode="random", gm_read_p=0.6)
    m.train()
    m.K = 3
    m.bm.collect = True
    m(tokens()[:, :-1])
    m.bm.collect = False
    m.bm.fit()
    populate(m.bm, m.blocks[0].mlp.E, 32)
    m.grow_experts(3)
    m.grow_experts(3)                                                     # past the preallocation
    m(tokens(seed=4)[:, :-1])
    ck = dict(cfg=copy.deepcopy(m.cfg), state=copy.deepcopy(m.state_dict()))
    m2 = LoopBrain(**ck["cfg"]).double()
    m2.load_state_dict(ck["state"], strict=True)
    s1, s2 = m.state_dict(), m2.state_dict()
    assert list(s1) == list(s2)
    for k in s1:
        if torch.is_tensor(s1[k]):
            assert torch.equal(s1[k], s2[k]), k
    assert m2.bm._fitted and torch.equal(m2.bm.table, m.bm.table) and torch.equal(m2.bm.priv.ksq, m.bm.priv.ksq)
    for mm in (m, m2):
        for s in mm.spikes():
            s.calibrated.fill_(True)
        mm.train()
        mm.K = 3
    torch.manual_seed(9)
    y1 = m(tokens(seed=5)[:, :-1])
    torch.manual_seed(9)
    y2 = m2(tokens(seed=5)[:, :-1])
    assert torch.equal(y1, y2)                                            # incl. the random masks (generator restored)


def test_19_store_tensors_are_buffers_and_the_optional_scale_is_handled_explicitly():
    tl = current().train_lm
    off, on = loop(gm=0), loop(gm=1)
    assert [n for n, _ in off.named_parameters()] == [n for n, _ in on.named_parameters()]
    assert not any(n.startswith("bm.") for n, _ in on.named_parameters())
    assert any(n.startswith("bm.glob.") for n, _ in on.named_buffers())
    sc = loop(gm=1, gm_inj_scale=1)
    extra = [n for n, _ in sc.named_parameters() if n.startswith("bm.")]
    assert extra == ["bm.inj_scale"]
    args = SimpleNamespace(opt="adamw", lr=1e-3, wd=0.1, muon=0, maturity=0.0, lazy=0, tau=10)
    opt = tl.make_opt(sc, 1e-3, 0.1)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(sc.bm.inj_scale) in ids                                     # trained by the brain's optimizer
    assert not any(id(b) in ids for b in sc.bm.buffers())
    sc.bm.fit()
    populate(sc.bm, 8, 32)
    sc.train()
    sc.K = 3
    x = tokens()
    _, loss = sc(x[:, :-1], x[:, 1:])
    loss.backward()
    assert sc.bm.inj_scale.grad is not None and sc.bm.inj_scale.grad.abs().sum() > 0
    torch.nn.utils.clip_grad_norm_(sc.parameters(), 1.0)                  # (part of the global clip)
    from spiking_lm import carry_optimizer, snapshot_optimizer
    opt.step()
    snap = snapshot_optimizer(sc, opt)
    sc.grow_experts(2)
    opt2 = carry_optimizer(sc, snap, tl.make_opt(sc, 1e-3, 0.1))
    assert id(sc.bm.inj_scale) in {id(p) for g in opt2.param_groups for p in g["params"]}


# ---------------------------------------------------------------- 6. flags off / empty memory: bit-identical
def test_20_flags_off_bit_identical_to_the_pre_blob_code_over_40_steps():
    old, new = _drive(baseline()), _drive(current())
    assert old["E"] == new["E"] == 12
    for a, b in zip(old["losses"], new["losses"]):
        assert torch.equal(a, b)
    for a, b in zip(old["outs"], new["outs"]):
        assert torch.equal(a, b)
    assert old["cfg"] == new["cfg"] and list(old["state"]) == list(new["state"])
    for k in old["state"]:
        assert torch.equal(old["state"][k], new["state"][k]), k
    for name, st in old["opt_state"].items():
        for key, buf in st.items():
            assert torch.equal(buf, new["opt_state"][name][key]), (name, key)
    assert torch.equal(old["rng"], new["rng"])


def test_21_flags_off_bit_identical_with_the_old_memory_on():
    """(b) the old memory config (rows + private): 4 training steps, old modules vs new."""
    def run(mods):
        saved = {n: sys.modules.get(n) for n in ("loop_brain", "spiking_lm")}
        torch.manual_seed(0)
        LB = mods.loop_brain.LoopBrain
        cfg = dict(vocab=50, d=32, d_s=8, heads=4, head_dim=8, spike="graded", chunk=16, k_mean=4, k_min=2, k_max=6,
                   bptt=3, state="", refractory=1, slots=4, waves=True, ws_direct=True, head_group=2, head_groups=2,
                   deep_sup=0.3, halt=True, think_w=0.1, think_fb=True, think_use=True, think_target="latent",
                   moe=dict(n_exp=12, k=3, hidden=16, shared=16, capacity=0, birth_mode="split"),
                   concepts=[2, 4, 8, 16], mem_shared=8, mem_private=8, mem_dv=16, mem_max_experts=16, mem_topm=4,
                   mem_topk=3)
        m = LB(**cfg).double()
        for s in m.spikes():
            s.calibrated.fill_(True)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        losses = []
        for i in range(4):
            if i == 2:
                m.grow_experts(2)
                opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            losses.append(train_step(m, opt, tokens(seed=20 + i), offset=i % 2))
        return losses, m.state_dict()
    la, sa = run(baseline())
    lb, sb = run(current())
    assert all(torch.equal(a, b) for a, b in zip(la, lb))
    assert list(sa) == list(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]) if torch.is_tensor(sa[k]) else repr(sa[k]) == repr(sb[k]), k


def test_22_trainer_flags_off_bit_identical_end_to_end(tmp_path):
    data = _tiny_data(tmp_path / "data")
    _, ra, ca = _run_main(baseline().train_lm, tmp_path / "runs_old", "old", data)
    _, rb, cb = _run_main(current().train_lm, tmp_path / "runs_new", "new", data)
    assert len(ra) == len(rb) == 3
    for a, b in zip(ra, rb):
        for k in ("tok_s", "elapsed"):
            a.pop(k), b.pop(k)
        assert a == b
    assert ca["cfg"] == cb["cfg"] and list(ca["state"]) == list(cb["state"])
    assert all(torch.equal(ca["state"][k], cb["state"][k]) for k in ca["state"])


def _train(m, steps=6, seed=0, fit_at=None):
    torch.manual_seed(seed)
    params = [p for n, p in m.named_parameters()]
    opt = torch.optim.Adam(params, lr=3e-3)
    out = []
    for i in range(steps):
        if m.bm is not None and fit_at is not None:
            m.bm.collect = fit_at - 2 <= i < fit_at
            if i == fit_at:
                m.bm.fit()
        m.train()
        m.K, m.aux_k_fixed, m.group_offset = 3 + i % 2, 1, i % 2
        x = tokens(seed=10 + i)
        _, loss = m(x[:, :-1], x[:, 1:])
        total = loss + m.aux_loss()
        opt.zero_grad()
        total.backward()
        out.append((total.detach(), [p.grad.clone() if p.grad is not None else None for p in params],
                    m.blocks[0].mlp.last_idx.clone(), m.blocks[0].mlp.pending.clone()))
        opt.step()
        m.blocks[0].mlp.apply_balance()
        if m.bm is not None:
            m.bm.after_step(i)
        if i == 2:
            m.grow_experts(2)
            params = [p for n, p in m.named_parameters()]
            opt = torch.optim.Adam(params, lr=3e-3)
    return out, {k: v for k, v in m.state_dict().items() if not k.startswith("bm.")}


@pytest.mark.parametrize("mode", ["none", "all", "random"])
def test_23_gm_on_with_an_empty_store_changes_nothing(mode):
    """Before the fit nothing is read, after it an empty store reads exactly 0 (mode 'none': nothing): losses,
    gradients, routing and the model state equal the gm-off model's, through a fit and an expert split."""
    a, sa = _train(loop(gm=0))
    b, sb = _train(loop(gm=1, gm_read_mode=mode), fit_at=2)
    for (la, ga, ia, pa), (lb, gb, ib, pb) in zip(a, b):
        assert torch.equal(la, lb) and torch.equal(ia, ib) and torch.equal(pa, pb)
        assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(ga, gb))
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_24_blobs_change_the_output_not_the_routing_and_the_contents_learn():
    m, m0 = loop(gm=1), loop(gm=0)
    m.bm.fit()
    populate(m.bm, 8, 32, scale=0.5)
    m.bm.set_h0(6.0, 6.0)
    v0 = m.bm.glob.value.clone()
    x = tokens()
    for mm in (m, m0):
        mm.eval()
        mm.K = 3
    with torch.no_grad():
        assert not torch.equal(m(x[:, :-1]), m0(x[:, :-1]))
    mlp, mlp0 = m.blocks[0].mlp, m0.blocks[0].mlp
    xin = torch.randn(2, 5, 32, dtype=torch.float64)
    mlp.train(), mlp0.train()
    p0, p00 = mlp.pending.clone(), mlp0.pending.clone()
    mlp(xin, bmem=m.bm)
    mlp0(xin)
    assert torch.equal(mlp.last_idx, mlp0.last_idx) and torch.equal(mlp.stats[0], mlp0.stats[0])   # routing invariant
    assert torch.equal(mlp.pending - p0, mlp0.pending - p00)
    _train(m, steps=2)
    assert not torch.equal(m.bm.glob.value, v0) and m.bm.glob.uses.sum() > 0 and m.bm.priv.uses.sum() > 0


# ---------------------------------------------------------------- 7. no host syncs
FORWARD_FNS = ("inject", "__call__", "search_global", "_search_ivf", "_rerank", "search_private", "_rerank_priv",
               "search_shared", "read_mask", "_read", "_sink", "_count", "observe", "_ring_add", "_track",
               "apply_value_grads", "after_step", "begin_forward", "address", "rebuild_cells", "assign_cells",
               "_search_brute", "probation_push", "probation_lookup", "on_grow", "reset_opt_state", "fill", "_workspace")


def test_25_no_host_sync_source():
    src = inspect.getsource(blob_memory)
    for bad in ("bincount", "nonzero", "masked_select", "torch.unique", ".tolist("):
        assert bad not in src, bad
    for n in FORWARD_FNS:
        s = inspect.getsource(getattr(BlobMemory, n))
        for bad in (".item(", ".cpu(", ".numpy(", "bool(", "int(self", "float(w", "print("):
            assert bad not in s, (n, bad)
    for f in (kernel_weights, blend, BagRead, stochastic_bf16, cap_norm, blob_memory.ivf_table,
              blob_memory.ivf_candidates, blob_memory.exact_candidates, blob_memory.rerank, ProbationStore):
        assert ".item(" not in inspect.getsource(f)


def test_26_no_host_sync_runtime(monkeypatch):
    """Every host-sync entry point raises while the read, its backward, the content step, a refit and a birth run."""
    bm = fitted(read_mode="random", compute_frac=0.7)
    bm.train()
    populate(bm, 3, 16)
    pb = fitted(probation_rate=4, probation_life=5)
    rt = _routing()
    xf = torch.randn(12, 16, requires_grad=True)

    def boom(*a, **k):
        raise AssertionError("host sync")

    orig = torch.Tensor.__getitem__

    def getitem(self, i):
        items = i if isinstance(i, tuple) else (i,)
        if any(torch.is_tensor(t) and t.dtype == torch.bool for t in items):
            raise AssertionError("boolean-mask indexing")
        return orig(self, i)

    for name in ("item", "tolist", "__bool__", "nonzero", "__int__", "__float__", "numpy", "cpu"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    monkeypatch.setattr(torch.Tensor, "__getitem__", getitem)
    for name in ("bincount", "nonzero", "unique", "masked_select"):
        monkeypatch.setattr(torch, name, boom)
    bm.collect = True
    buf, xs = _call(bm, xf, rt, call=0)
    ((buf * 1.5).sum() + xs.sum()).backward()
    bm.after_step(1)
    bm.on_grow(torch.tensor([0, 2]), 3, split=True)
    bm.rebuild_cells()
    ids = pb.probation_push(torch.randn(5, 8), torch.randn(5, 8), torch.tensor([0, 1, -1, 2, 0]))
    pb.prob.rebuild(pb.centroids, reassign=False)
    for mode in ("ivf", "exact"):
        hit, d2 = pb.probation_lookup(torch.randn(4, 8), 3, mode)
    pb.prob.mark_hit(hit[:, 0], 7, d2[:, 0] < 50)
    pb.prob.take(ids[:2])
    monkeypatch.undo()
    assert xf.grad.abs().sum() > 0


# ---------------------------------------------------------------- 8. the trainer
GM_ARGS = ("--gm 1 --gm-global 256 --gm-private 16 --gm-fit-step 5 --gm-fit-window 3 --gm-read-mode random "
           "--gm-cells 8 --gm-cell-cap 64 --gm-probe 4 --gm-qcap 256 --gm-cand 8 --gm-topk 4,4 --gm-ivf-refit 15")


def test_27_trainer_gm_smoke_flags_and_reload(tmp_path):
    data = _tiny_data(tmp_path / "data")
    run, recs, ck = _run_main(current().train_lm, tmp_path / "runs", "gm", data, GM_ARGS.split())
    log = (run / "train.log").read_text()
    assert "blob memory (25d):" in log and "GM ADDRESS FITTED" in log
    from loop_brain import LoopBrain
    m = LoopBrain(**ck["cfg"])
    m.load_state_dict(ck["state"], strict=True)
    assert m.bm is not None and m.bm._fitted and m.blocks[0].mlp.E == 12
    assert m.bm.priv.occ.size(0) >= 13 and m.bm.G == 256 and m.bm.P == 16 and m.bm.k_g == 4
    assert "gm_global_fill" in recs[-1] and "gm_read_frac" in recs[-1]


class _Stop(Exception):
    pass


def test_28_trainer_resume_with_blobs_is_bit_identical(tmp_path, monkeypatch):
    """40 straight steps == 20 + save + resume + 20 with the memory on, blobs present (written by a test hook at the
    fit), random read masks, content learning, an IVF refit and expert splits before the cut."""
    tl = current().train_lm
    data = _tiny_data(tmp_path / "data")
    extra = (GM_ARGS + " --ckpt-every 20").split()
    fit = BlobMemory.fit

    def fit_and_fill(self, *a, **k):
        fit(self, *a, **k)
        populate(self, self.Emax, self.d, n_priv=3, scale=0.3)

    monkeypatch.setattr(BlobMemory, "fit", fit_and_fill)
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
    assert len(ra) == len(rb) == 3
    for a, b in zip(ra, rb):
        a, b = dict(a), dict(b)
        for k in ("tok_s", "elapsed"):
            a.pop(k), b.pop(k)
        assert a == b
    assert ca["cfg"] == cb["cfg"] and list(ca["state"]) == list(cb["state"])
    for k in ca["state"]:
        if torch.is_tensor(ca["state"][k]):
            assert torch.equal(ca["state"][k], cb["state"][k]), k
    assert ca["state"]["bm.glob.uses"].sum() > 0 and ca["state"]["bm.glob.opt_n"].sum() > 0


def test_29_vram_flags_are_cpu_noops_and_default_off(tmp_path):
    tl = current().train_lm
    calls = []
    orig = tl.free_cache
    data = _tiny_data(tmp_path / "data")
    try:
        tl.free_cache = lambda dev: (calls.append(dev), orig(dev))
        _run_main(tl, tmp_path / "runs", "v", data, ["--empty-cache-eval", "1", "--mem-fraction", "0.5"])
        n_on = len(calls)
        calls.clear()
        _run_main(tl, tmp_path / "runs", "w", data)
        n_off = len(calls)
    finally:
        tl.free_cache = orig
    assert n_on == n_off + 3                                  # one before each of the 3 evaluations


# ---------------------------------------------------------------- 9. the probation store (write path, flag-gated)
def test_30_probation_fifo_accessors_and_lookup():
    torch.manual_seed(13)
    bm = small(probation_rate=4, probation_life=6, cells=4, cell_cap=32, cand=4)       # 24 entries
    pb = bm.prob
    assert pb.n == 24 and pb.key.dtype == torch.float16 and "prob.key" in bm.state_dict()
    assert not any(n in bm.state_dict() for n in ("prob.src", "prob.last", "prob.sig"))     # no pointers in the store
    keys = [torch.randn(10, 8) for _ in range(3)]
    bm.now.fill_(1)
    ids0 = bm.probation_push(keys[0], keys[0] * 0.1)
    assert torch.equal(ids0, torch.arange(10)) and pb._ptr == 10 and (pb.step[:10] == 1).all()
    assert pb.step.dtype == torch.int32
    bm.now.fill_(2)
    bm.probation_push(keys[1], keys[1])
    bm.now.fill_(3)
    ids2 = bm.probation_push(keys[2], keys[2])                                # FIFO: wraps over the oldest
    assert torch.equal(ids2, torch.tensor([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]))
    assert (pb.step[:6] == 3).all() and (pb.step[6:10] == 1).all() and pb.occ.all()
    assert torch.equal(pb.age(5)[:6], torch.full((6,), 2))
    # exact lookup == brute force over the stored (fp16) keys, before the fit
    q = torch.randn(7, 8)
    ids, d2 = bm.probation_lookup(q, 3)
    ref = _exact(q, pb.key.float(), pb.occ, 3)
    assert torch.equal(ids, ref.indices) and torch.allclose(d2.double(), ref.values, rtol=1e-5)
    # after the fit: IVF over the global cells (every cell probed) == exact
    bm.fit()
    pb.rebuild(bm.centroids)
    ivf, d_ivf = bm.probation_lookup(q, 3, "ivf")
    assert torch.equal(ivf, ref.indices)
    # promotion accessors
    assert torch.equal(bm.probation_push(torch.randn(2, 8), torch.randn(2, 8)), torch.tensor([6, 7]))
    pb.mark_hit(torch.tensor([8, 8, 9]), 11, torch.tensor([True, True, False]))
    assert pb.hits[8] == 2 and pb.hits[9] == 0
    got = pb.take(torch.tensor([8, 9]))
    assert got["hits"].tolist() == [2, 0] and not pb.occ[8] and not pb.occ[9]
    assert torch.allclose(got["pgrad"][0], (keys[0][8] * 0.1).to(torch.float16).float())
    ids, d = bm.probation_lookup(q, 24, "exact")
    assert not ((ids == 8) & (d < float("inf"))).any()                        # a taken entry is never found
    sd = copy.deepcopy(bm.state_dict())
    bm2 = small(probation_rate=4, probation_life=6, cells=4, cell_cap=32, cand=4)
    bm2.load_state_dict(sd)
    pb.rebuild(bm.centroids)                                                   # (the load rebuilds the derived table)
    assert bm2.prob._ptr == pb._ptr and torch.equal(bm2.prob.ksq, pb.ksq) and torch.equal(bm2.prob.table, pb.table)


def test_31_probation_ivf_recall_on_low_rank_data():
    g = torch.Generator().manual_seed(14)
    kd, n = 64, 6000
    basis = torch.randn(8, kd, generator=g)
    centres = torch.randn(30, 8, generator=g) * 3
    keys = (centres[torch.randint(0, 30, (n,), generator=g)] + 0.6 * torch.randn(n, 8, generator=g)) @ basis
    bm = small(G=64, kd=kd, d=kd, cand=16, cells=64, cell_cap=64, probe=16, q_cap=512, probation_rate=2,
               probation_life=3000)
    bm.proj.copy_(torch.eye(kd))
    bm.train()
    bm.collect = True
    bm.observe(keys)
    bm.fit()
    bm.probation_push(keys, torch.zeros(n, kd))
    bm.prob.rebuild(bm.centroids)
    q = keys[torch.randint(0, n, (200,), generator=g)] + 0.3 * torch.randn(200, kd, generator=g)
    ids, _ = bm.probation_lookup(q, 16, "ivf")
    ref = _exact(q, bm.prob.key.float(), bm.prob.occ, 16).indices
    recall = (ids[:, :, None] == ref[:, None, :]).any(-1).float().mean().item()
    assert recall >= 0.9, recall


def test_32_probation_flags_build_and_bit_identity_of_the_rest():
    m = loop(gm=1, gm_probation_rate=2, gm_probation_life=10)
    assert m.bm.prob is not None and m.bm.prob.n == 20 and m.cfg["gm_probation_rate"] == 2
    assert loop(gm=1).bm.prob is None
    a, _ = _train(loop(gm=1), steps=3, fit_at=1)
    b, _ = _train(loop(gm=1, gm_probation_rate=2, gm_probation_life=10), steps=3, fit_at=1)
    assert all(torch.equal(x[0], y[0]) for x, y in zip(a, b))
