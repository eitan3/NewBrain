"""Tests of the memory speed package (plan_v2/14_memory_speed_plan.md): the same-math items S1-S9 against the code
before them (plan_v2/baseline_mem_v2), the tie rule, and the mild flags M1-M3. CPU only, tiny configs.

    python -m pytest test_speed.py -q

Exactness (plan_v2/14): bit-identical (E) for S1 (top-k), S2 (private top-4), S3 (balancing), S6 (MemOpt on the
touched rows), S7 (batched select), S8 (frozen online = EMA queries), S9 (probes; training untouched) and the G
part of S5 (column-major flattened index_add_, the same order of adds on the CPU); near-exact (NE: the same values
up to the floating-point summation order) for S4 (the margin inside the online pass: W_q / sub-key gradients) and
the dw part of S5 (gather + bmm). test_01 runs the trainer's step with S4 / S5-dw in their reference mode
(mem.margin_fused = False, memory.READ_DW_BMM = False) and checks everything bit for bit; test_02 runs it with
everything on (float64) and checks tight allclose plus identical addresses.
"""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_memory import _run_main, _tiny_data, mem_loop, tokens

HERE = Path(__file__).resolve().parent
BASE = HERE / "plan_v2" / "baseline_mem_v2"
MODS = ("spiking_lm", "loop_brain", "brain_opt", "neurogenesis", "train_lm", "memory", "concepts")
_BASELINE = None


def baseline():
    """The code before the speed package (plan_v2/baseline_mem_v2) as separate module objects (as test_memory's
    baseline(): the current modules are out of sys.modules while it imports, then put back; no bytecode written)."""
    global _BASELINE
    if _BASELINE is None:
        saved = {n: sys.modules.pop(n) for n in MODS if n in sys.modules}
        dont = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(BASE))
        try:
            _BASELINE = SimpleNamespace(**{n: importlib.import_module(n) for n in MODS[:6]})
        finally:
            sys.path.remove(str(BASE))
            sys.dont_write_bytecode = dont
            for n in MODS:
                sys.modules.pop(n, None)
            sys.modules.update(saved)
        assert Path(_BASELINE.memory.__file__).parent == BASE
    return _BASELINE


def current():
    import brain_opt
    import loop_brain
    import memory
    import neurogenesis
    import spiking_lm
    import train_lm
    return SimpleNamespace(spiking_lm=spiking_lm, loop_brain=loop_brain, brain_opt=brain_opt,
                           neurogenesis=neurogenesis, train_lm=train_lm, memory=memory)


# ---------------------------------------------------------------- the trainer's step with the memory on
def _poke(mem):
    """Busy and dead keys in several codebooks (both halves, several heads of level 1: their bands overlap) and in
    a few experts, so the next balancing event splits and re-seeds."""
    S, R = mem.S, mem.R
    for l, h, half, b in ((0, 0, 0, 1), (0, 0, 1, 2), (0, 1, 0, 3), (0, 2, 1, 1), (0, 1, 1, 6), (1, 0, 0, 7),
                          (3, 2, 1, 4)):
        mem.U[l, h, half] = 0.2 / (S - 1)
        mem.U[l, h, half, b] = 0.8
        mem.U[l, h, half, [(b + 2) % S, (b + 5) % S]] = 1e-6       # the split takes the first, the other is dead
    for e, b in ((0, 1), (3, 0), (5, 6), (2, 3)):
        mem.U_e[e] = 0.2 / (R - 1)
        mem.U_e[e, b] = 0.8
        mem.U_e[e, [(b + 3) % R, (b + 4) % R]] = 1e-6


def _drive(mods, steps=14, accum=2, seed=0, dtype=torch.float32, new=False, freeze_at=10):
    """test_memory._drive's trainer step on a tiny v4d-like brain with the full memory: concepts (4 levels), shared
    + private memory, growth (expert splits + head births), balancing every 4 steps (two events, with forced busy
    and dead keys), writes from step 2, the freeze at step 10, accumulation 2, BrainOpt rebuilt on births with the
    trainer's optimizer-row reset. Returns the losses, gradients, G buffers, eval outputs and records (addresses),
    the balancing stats and the final state."""
    LoopBrain, BrainOpt = mods.loop_brain.LoopBrain, mods.brain_opt.BrainOpt
    carry, Neuro = mods.brain_opt.carry_brain_state, mods.neurogenesis.Neurogenesis
    RoutedMLP = mods.spiking_lm.RoutedMLP
    torch.manual_seed(seed)
    np.random.seed(seed)
    start, final = dict(layers=6, heads=4, mlp=1536, experts=8), dict(layers=6, heads=8, mlp=1536, experts=12)
    moe = dict(n_exp=8, k=2, hidden=16, shared=16, balance="aux", capacity=1.5, u=1e-3, aux_w=0.02, z_w=1e-4,
               birth_mode="split", birth_delta=0.1)
    model = LoopBrain(50, 32, 8, start["heads"], 8, moe=moe, spike="graded", chunk=16, k_mean=4, k_min=2, k_max=6,
                      bptt=3, state="", refractory=1, slots=4, waves=True, pops=1, ws_direct=True, head_group=4,
                      head_groups=2, deep_sup=0.3, halt=True, think_w=0.1, think_n=4, think_fb=True,
                      think_target="latent", think_use=True, concepts=[2, 4, 8, 16], mem_shared=8, mem_private=8,
                      mem_dv=16, mem_max_experts=12, mem_topm=4, mem_topk=3).to(dtype)
    mem = model.mem
    mem.balance_every = 4
    model.set_homeo(0.01, 0.15, "floor")
    mk = lambda: BrainOpt(model, 2e-3, 0.1, muon=True, lazy=True, maturity=0.0, tau=10)
    box = [mk()]
    mem.opt_reset = lambda name, sl: mods.train_lm.reset_mem_key_rows(box[0], model, name, sl)
    neuro = Neuro(model, start, final, steps, every=2, t0=0.05, t1=0.5)
    corpus = torch.randint(0, 50, (5000,), generator=torch.Generator().manual_seed(1))
    rng = np.random.RandomState(0)
    batch = lambda: torch.stack([corpus[i:i + 17] for i in rng.randint(0, len(corpus) - 17, 4)])
    xv = torch.randint(0, 50, (2, 16), generator=torch.Generator().manual_seed(2))
    out = dict(losses=[], grads=[], G=[], outs=[], reads=[], stats=[])
    for step in range(steps + 1):
        if neuro.step(step):
            model.set_homeo(0.01, 0.15, "floor")
            box[0] = carry(box[0], model, mk())
        if step == freeze_at:
            assert neuro.done()
            mem.freeze()
        frac = step / steps
        lr = 2e-3 * min(1.0, step / max(1, 0.02 * steps))
        if frac > 0.8:
            lr *= max(0.0, (1 - frac) / 0.2)
        model.set_lam(min(1.0, max(0.0, (frac - 0.05) / 0.15)))
        for mod in model.modules():
            if isinstance(mod, RoutedMLP):
                mod.noise, mod.eval_noise = 0.3, 0.0
        if step % 3 == 0 or step == steps:
            model.eval()
            model.K = 5
            mem.record = dict(pos=torch.arange(32), reads=[], calls=[])
            with torch.no_grad():
                out["outs"].append(model(xv))
            out["reads"].append(mem.record["reads"])
            mem.record = None
            model.train()
        if step == steps:
            break
        model.K = model.sample_K()
        model.aux_k_fixed = model.sample_aux_k(model.K)
        model.group_offset = model.sample_group_offset()
        mem.write_on = step >= 2
        opt = box[0]
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            xb = batch()
            x, y = xb[:, :-1], xb[:, 1:]
            logits, loss = model(x, y)
            total = loss
            aux = model.aux_loss()
            if aux is not None:
                total = total + aux
            if model.aux_x is not None:
                total = total + 0.2 * F.cross_entropy(model.heads["lm"](model.aux_x).float().flatten(0, 1), y.flatten())
            (total / accum).backward()
            out["losses"].append(torch.stack([loss.detach(), total.detach()]))
        out["grads"].append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        out["G"].append([mem.G.clone(), mem.G_e.clone(), mem.G_P.clone()])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(lr / 2e-3)
        model.blocks[0].mlp.apply_balance()
        if step in (3, 7):
            _poke(mem)
        if new:  # the trainer's path: the touched-row counts read on the host, then after_step
            cnt = mem.write_counts()
            mem.after_step(lr, cnt.tolist() if cnt is not None else None)
        else:
            mem.after_step(lr)
        out["stats"].append(dict(mem.stats.get("balance", {})))
    out.update(state=model.state_dict(), cfg=model.cfg, E=model.blocks[0].mlp.E, heads=model.blocks[0].mix.h,
               opt_state=box[0].state, rng=torch.get_rng_state(), gen=mem.gen.get_state(), frozen=mem.is_frozen)
    return out


def _check(old, new, exact, rtol=1e-9, atol=1e-11):
    same = (lambda a, b: torch.equal(a, b)) if exact else \
        (lambda a, b: a.shape == b.shape and torch.allclose(a, b, rtol=rtol, atol=atol))
    assert old["E"] == new["E"] == 12 and old["heads"] == new["heads"] == 8 and old["frozen"] and new["frozen"]
    assert old["stats"] == new["stats"]
    assert old["stats"][3]["splits"] >= 7 and old["stats"][3]["dead"] >= 1          # the forced balancing events
    assert old["stats"][3]["p_splits"] >= 4 and old["stats"][3]["p_dead"] >= 1 and old["stats"][7]["splits"] >= 7
    assert len(old["losses"]) == len(new["losses"]) == 28
    for i, (a, b) in enumerate(zip(old["losses"], new["losses"])):
        assert same(a, b), (i, a, b)
    for i, (ga, gb) in enumerate(zip(old["grads"], new["grads"])):
        assert ga.keys() == gb.keys(), i
        for n in ga:
            assert same(ga[n], gb[n]), (i, n)
    for i, (ga, gb) in enumerate(zip(old["G"], new["G"])):
        for a, b in zip(ga, gb):
            assert same(a, b), i
    for a, b in zip(old["outs"], new["outs"]):
        assert same(a, b)
    n_reads = 0
    for ra, rb in zip(old["reads"], new["reads"]):  # the addresses: identical in both modes
        assert len(ra) == len(rb) == 5
        for a, b in zip(ra, rb):
            assert a["k"] == b["k"] and torch.equal(a["rows"], b["rows"])
            for f in ("w", "g", "gap", "gap1", "gap2", "a"):
                assert same(a[f], b[f]), f
            n_reads += 1
    assert n_reads == 30
    assert old["cfg"] == new["cfg"] and list(old["state"]) == list(new["state"])
    for k in old["state"]:
        a, b = old["state"][k], new["state"][k]
        if not torch.is_tensor(a):  # (the memory's extra state: the probe records)
            assert a == b, k
            continue
        assert (torch.equal(a, b) if not a.is_floating_point() or "wcount" in k else same(a, b)), k
    assert list(old["opt_state"]) == list(new["opt_state"])
    for name, st in old["opt_state"].items():
        for key, buf in st.items():
            assert same(buf, new["opt_state"][name][key]), (name, key)
    assert torch.equal(old["rng"], new["rng"]) and torch.equal(old["gen"], new["gen"])


def test_01_same_math_bit_identical_with_the_margin_and_dw_in_reference_mode():
    """S1-S3, S5 (G), S6-S8 bit for bit over 14 steps (growth, two balancing events, the freeze, accumulation)."""
    old = _drive(baseline())
    import memory
    rd = memory.READ_DW_BMM
    memory.READ_DW_BMM = False
    orig = memory.Memory.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.margin_fused = False

    memory.Memory.__init__ = init
    try:
        new = _drive(current(), new=True)
    finally:
        memory.READ_DW_BMM = rd
        memory.Memory.__init__ = orig
    _check(old, new, exact=True)


def test_02_everything_on_near_exact_float64():
    """S4 and S5 (dw) on: tight allclose (summation order only) and identical addresses, float64."""
    old = _drive(baseline(), dtype=torch.float64)
    new = _drive(current(), dtype=torch.float64, new=True)
    _check(old, new, exact=False)
    # (and they are not bit-identical: the near-exact items are really on)
    assert any(not torch.equal(a[n], b[n]) for a, b in zip(old["grads"], new["grads"]) for n in a)


# ---------------------------------------------------------------- the parts one by one, against the old code
def test_03_plan_balance_matches_the_per_key_loop():
    old = baseline().memory.plan_balance
    from memory import plan_balance
    g = torch.Generator().manual_seed(0)
    for trial in range(200):
        C, S = int(torch.randint(1, 6, (), generator=g)), int(torch.randint(2, 40, (), generator=g))
        U = torch.randint(0, 6, (C, S), generator=g).float() ** 3          # many ties, busy and dead keys
        U[torch.rand(C, S, generator=g) < 0.2] = 0
        U = U / U.sum(1, keepdim=True).clamp(min=1)
        sp, dd = old(U)
        sp2, dd2 = plan_balance(U)
        assert sp2.tolist() == [list(t) for t in sp] and dd2.tolist() == [list(t) for t in dd], trial


def test_04_memread_backward_G_exact_dw_near_exact():
    import memory
    old = baseline().memory.MemRead
    torch.manual_seed(0)
    for dt in (torch.float32, torch.float64):
        V = torch.randn(300, 16, dtype=dt)
        rows = torch.randint(0, 300, (500, 6))                               # many repeated rows
        w0 = torch.rand(500, 6, dtype=dt)
        gout = torch.randn(500, 16, dtype=dt)
        res = []
        for fn, elems in ((old.apply, None), (memory.MemRead.apply, 1 << 26), (memory.MemRead.apply, 3000)):
            lim = memory.READ_BWD_ELEMS
            memory.READ_BWD_ELEMS = elems or lim                            # (3000: chunked columns and bags)
            try:
                G = torch.zeros_like(V)
                w = w0.clone().requires_grad_()
                r = fn(memory._anchor("cpu", True), w, V, G, rows)
                r.backward(gout)
            finally:
                memory.READ_BWD_ELEMS = lim
            res.append((r.detach(), G, w.grad))
        for r, G, dw in res[1:]:
            assert torch.equal(r, res[0][0]) and torch.equal(G, res[0][1])
            assert torch.allclose(dw, res[0][2], rtol=1e-5 if dt == torch.float32 else 1e-12, atol=1e-6)


def _pair(dtype=torch.float64, **kw):
    """The same memory model built with the old and the new code (the same seed, so the same state)."""
    from loop_brain import LoopBrain
    old_lb = baseline().loop_brain.LoopBrain
    cfg = dict(vocab=50, d=32, d_s=8, heads=4, head_dim=8, spike="graded", chunk=16, k_mean=4, k_min=2, k_max=6,
               bptt=3, state="", refractory=1, slots=4, waves=True, ws_direct=True, head_group=2, head_groups=2,
               deep_sup=0.3, halt=True, think_w=0.1, think_fb=True, think_use=True, think_target="latent",
               moe=dict(n_exp=12, k=3, hidden=16, shared=16, capacity=0, birth_mode="split"),
               concepts=[2, 4, 8, 16], mem_shared=16, mem_private=8, mem_dv=16, mem_max_experts=16, mem_topm=4,
               mem_topk=3)
    cfg.update(kw)
    ms = []
    for cls in (old_lb, LoopBrain):
        torch.manual_seed(0)
        m = cls(**cfg).to(dtype)
        for s in m.spikes():
            s.calibrated.fill_(True)
        ms.append(m)
    return ms


def test_05_after_step_balance_and_private_bit_identical_on_a_real_state():
    """after_step (MemOpt on the touched rows, the key Adam, the EMAs, a balancing event with real U, qres and
    lastq) and the private read, old vs new, over 6 training steps (the margin and dw in their reference mode)."""
    import memory
    from brain_opt import BrainOpt
    mo, mn = _pair(torch.float32)
    mn.mem.margin_fused = False
    x = tokens(2, 25)
    rd = memory.READ_DW_BMM
    memory.READ_DW_BMM = False
    try:
        for m in (mo, mn):
            torch.manual_seed(1)
            m.mem.balance_every = 3
            opt = BrainOpt(m, lr=3e-3)
            for i in range(6):
                m.train()
                m.K, m.aux_k_fixed, m.group_offset = 4, 2, i % 2
                _, loss = m(x[:, :-1], x[:, 1:])
                opt.zero_grad()
                (loss + m.aux_loss()).backward()
                opt.step()
                m.blocks[0].mlp.apply_balance()
                if i == 5:
                    _poke(m.mem)
                m.mem.after_step(3e-3)
    finally:
        memory.READ_DW_BMM = rd
    st = mo.mem.stats["balance"]
    assert st == mn.mem.stats["balance"] and min(st.values()) > 0
    so, sn = mo.state_dict(), mn.state_dict()
    for k in so:
        assert torch.equal(so[k], sn[k]) if torch.is_tensor(so[k]) else so[k] == sn[k], k
    for n in ("G", "G_e", "G_P", "pend", "pend_e", "lastq"):
        assert torch.equal(getattr(mo.mem, n), getattr(mn.mem, n)), n


def _scores_with_ties(g, R, S):
    s = torch.randint(0, 4, (R, S), generator=g).float() / 4
    s[torch.rand(R, S, generator=g) < 0.1] = -0.0                           # -0.0 and +0.0 tie, as in torch.sort
    return s


@pytest.mark.parametrize("sync_free", [False, True])
def test_06_tie_rule_lowest_index_wins(sync_free):
    """S1 / S2: torch.topk + the re-sort keep the lowest-index rule bit for bit, ties inside the winners and at the
    k / k+1 boundary included (both tie paths), on heavily tied scores and through select() and the private read
    with duplicated keys (exact score ties)."""
    import memory
    old_topk = baseline().memory.stable_topk
    mode = memory.TOPK_SYNC_FREE
    memory.TOPK_SYNC_FREE = sync_free
    try:
        g = torch.Generator().manual_seed(0)
        for trial in range(200):
            R, S = int(torch.randint(1, 50, (), generator=g)), int(torch.randint(2, 40, (), generator=g))
            s = _scores_with_ties(g, R, S)
            for k in {1, 2, min(5, S), S - 1, S}:
                v0, i0 = old_topk(s, k)
                v1, i1 = memory.stable_topk(s, k)
                assert torch.equal(i0, i1) and torch.equal(v0, v1) and torch.equal(v0.signbit(), v1.signbit()), \
                    (trial, k)
        s = _scores_with_ties(g, 7 * 3, 30).view(7, 3, 30)                  # batched leading dims, chunked sort
        assert all(torch.equal(a, b) for a, b in zip(old_topk(s, 9, max_elems=64), memory.stable_topk(s, 9, max_elems=64)))
        # select() with duplicated sub-keys and the private read with duplicated private keys: exact score ties
        mo, mn = _pair(torch.float32, mem_topm=3, mem_topk=2)
        for m in (mo, mn):
            mem = m.mem
            with torch.no_grad():
                for t in (mem.k1_ema, mem.k2_ema):
                    t[:, :, 8:] = t[:, :, :8]                                # keys 8..15 = keys 0..7
                mem.P_ema[:, 4:] = mem.P_ema[:, :4]
                mem.w_p.weight.copy_(torch.randn(mem.w_p.weight.shape, generator=torch.Generator().manual_seed(4)))
        torch.manual_seed(1)
        mus = [torch.randn(2, 11, 32) for _ in range(4)]
        pbar = torch.randn(2, 11, 64)
        sels = []
        for m in (mo, mn):
            a, a_hat = m.mem.prep(mus, pbar)
            sels.append(m.mem.select(a, a_hat, False))
        for k in ("i", "j", "I1", "I2", "rows", "qb1"):
            assert torch.equal(sels[0][k], sels[1][k]), k
        assert (sels[1]["I1"][..., 0] < 8).all() and (sels[1]["I1"] >= 8).any()   # duplicates lose the ties
        E, C, N, k = 6, 5, 11, 2
        e_k = torch.tensor([0, 0, 1, 2, 2, 5, 5, 5])
        p_k = torch.tensor([0, 3, 1, 0, 2, 0, 1, 4])
        t_k = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        outs = []
        for m, sel in zip((mo, mn), sels):
            q1 = m.mem.online(m.mem.prep(mus, pbar)[1], sel)["q1"]
            with torch.no_grad():
                outs.append(m.mem.private(q1, sel["qb1"], e_k, p_k, t_k, torch.rand(8, generator=g.manual_seed(3)),
                                          t_k, N, k, E, C))
        assert torch.equal(outs[0], outs[1]) and outs[0].abs().sum() > 0
    finally:
        memory.TOPK_SYNC_FREE = mode


def test_07_frozen_online_reuses_the_ema_queries_bit_identical():
    """S8: after the freeze (sg addressing) online() takes select's queries: the same w, sig and q1 as computing
    them again with the online translator."""
    m = mem_loop(torch.float32)
    mem = m.mem
    mem.freeze()
    torch.manual_seed(2)
    mus = [torch.randn(2, 9, 32) for _ in range(4)]
    a, a_hat = mem.prep(mus, torch.randn(2, 9, 64))
    sel = mem.select(a, a_hat, False)
    assert "q" in sel
    reuse = mem.online(a_hat, sel)
    again = mem.online(a_hat, {k: v for k, v in sel.items() if k != "q"})
    for k in ("w", "sig", "q1"):
        assert torch.equal(reuse[k], again[k]), k


def test_08_fused_margin_has_the_same_value_and_scope():
    """S4: the margin from the online pass equals margin_loss (forward bit for bit), its gradient reaches only the
    translator and the sub-keys (none to the thought) and matches the separate pass's up to the summation order."""
    m = mem_loop()
    mem = m.mem
    torch.manual_seed(3)
    mus = [torch.randn(2, 9, 32, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    pbar = torch.randn(2, 9, 64, dtype=torch.float64)
    a, a_hat = mem.prep(mus, pbar)
    sel = mem.select(a.detach(), a_hat.detach(), False)
    ref = mem.margin_loss(a_hat.detach(), sel)
    st_ref = dict(mem.margin_stats)
    on = mem.online(a_hat, sel, margin=True)
    assert torch.equal(on["margin"], ref) and ref > 0
    assert all(torch.equal(st_ref[k], mem.margin_stats[k]) for k in st_ref)
    assert torch.equal(on["w"], mem.online(a_hat, sel)["w"])
    params = list(mem.keys.parameters())
    g_ref = torch.autograd.grad(ref, params)
    g_new = torch.autograd.grad(on["margin"], params + mus, allow_unused=True)
    for a_, b_ in zip(g_ref, g_new):
        assert torch.allclose(a_, b_, rtol=1e-10, atol=1e-14) and a_.abs().sum() > 0
    assert all(g is None or g.abs().sum() == 0 for g in g_new[len(params):])   # sg(a): nothing reaches the thought


# ---------------------------------------------------------------- mild flags (M1-M3), off by default
def test_09_readdr_every_reuses_indices_on_alternate_reads():
    """M1: with --mem-readdr-every 2 the thought reads 1, 3 reuse the indices of reads 0, 2 (the read before pass 1
    keeps its own and is never reused); select runs only on the fresh reads (so usage is collected only there);
    every pass still reads, with weights from its own thought; a reused read crosses into the gradient window
    and the keys get their gradient there."""
    m = mem_loop()
    mem = m.mem
    calls, onl, reads = [], [], []
    sel_fn, on_fn, rd_fn = mem.select, mem.online, mem.read_shared
    mem.select = lambda *a: (calls.append(1), sel_fn(*a))[1]
    mem.online = lambda a_hat, sel, **kw: (onl.append((a_hat.detach(), sel)), on_fn(a_hat, sel, **kw))[1]
    mem.read_shared = lambda *a: (reads.append(1), rd_fn(*a))[1]
    x = tokens(2, 17)
    m.mem_readdr_start = 0                                                         # (fresh from pass 0 on: 0, 2, 4)
    for every, n_sel in ((1, 6), (2, 4)):
        m.mem_readdr_every = every
        calls.clear(), onl.clear(), reads.clear()
        m.eval()
        m.K = 6
        mem.record = dict(pos=torch.arange(20), reads=[], calls=[])
        with torch.no_grad():
            m(x)
        rec = mem.record["reads"]
        mem.record = None
        assert len(reads) == len(rec) == 6                                         # read0 + passes 0..4 all read
        assert len(calls) == n_sel + (6 - n_sel)                                   # (+ the probe metric's selects)
        calls.clear()
        with torch.no_grad():
            m(x)
        assert len(calls) == n_sel                                                 # without records: fresh reads only
        assert [r["k"] for r in rec] == [-1, 0, 1, 2, 3, 4]
        if every == 2:
            for k in (1, 3):  # rec index k + 1
                assert torch.equal(rec[k + 1]["rows"], rec[k]["rows"])
                assert all(torch.equal(onl[k + 1][1][n], onl[k][1][n]) for n in ("i", "j", "rows", "qb1"))
                assert not torch.equal(onl[k + 1][0], onl[k][0])                   # the thought moved ...
                assert not torch.equal(rec[k + 1]["w"], rec[k]["w"])               # ... and w follows it
            assert not torch.equal(rec[1]["rows"], rec[0]["rows"])                 # pass 0 does not reuse read0
    # training: K = 6, bptt 3 (passes 3..5 with gradient): pass 3 reuses pass 2's (no-grad) indices
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 6, 4, 0
    calls.clear()
    _, loss = m(x[:, :-1], x[:, 1:])
    (loss + m.aux_loss()).backward()
    assert len(calls) == 4 and mem.keys.wq.grad.abs().sum() > 0 and mem.keys.k1.grad.abs().sum() > 0


def test_10_margin_passes_one_random_read_per_forward():
    """M2: --mem-margin-passes 1: the margin on one read per training forward, drawn (dedicated RNG) from the
    reads of the gradient window (fresh reads only with M1); 0 = every such read, as before."""
    m = mem_loop()
    mem = m.mem
    marg, steps = [], []
    mfn, hs = mem._margin, m._hier_step
    mem._margin = lambda *a: (marg.append(1), mfn(*a))[1]
    m._hier_step = lambda s, read, address, k=0, **kw: (steps.append((k, kw.get("margin", True))),
                                                        hs(s, read, address, k, **kw))[1]
    x = tokens(2, 17)
    m.train()
    m.mem_readdr_start = 0
    picked = set()
    for every, n, K, cands in ((1, 0, 3, {-1, 0, 1}), (1, 1, 3, {-1, 0, 1}), (1, 1, 6, {3, 4}), (2, 1, 6, {4}),
                               (1, 2, 6, {3, 4})):
        m.mem_readdr_every, m.mem_margin_passes = every, n
        m.K, m.aux_k_fixed, m.group_offset = K, max(0, K - 2), 0
        for _ in range(4 if n == 1 else 1):
            marg.clear(), steps.clear()
            m(x[:, :-1], x[:, 1:])
            on = {k for k, mg in steps if mg}
            assert len(marg) == (len(cands) if n == 0 else min(n, len(cands)))
            assert on <= cands or (n == 0 and on >= cands)
            if n == 1 and K == 3:
                picked |= on & cands
    assert len(picked) >= 2                                                       # it is drawn, not fixed
    g1 = mem.gen_margin.get_state()
    m.mem_margin_passes = 0
    m(x[:, :-1], x[:, 1:])
    assert torch.equal(g1, mem.gen_margin.get_state())                            # (off: the RNG is not touched)


def test_11_calmer_balancing_caps_and_keep():
    """M3: --mem-balance-max caps the changed keys per codebook (a split = 2 keys; busiest first; dead keys drawn
    with the dedicated RNG, so the same RNG state picks the same keys); --mem-balance-keep never re-seeds a key used
    since the last event, and the window restarts at every event."""
    from memory import plan_balance
    g = torch.Generator().manual_seed(0)
    U = torch.full((3, 64), 1e-6)
    U[:, :3] = 0.3                                                                 # 3 busy keys per codebook
    U[:, 3:20] = 0.01                                                              # the rest: 44 dead keys
    U = U / U.sum(1, keepdim=True)
    sp0, dd0 = plan_balance(U)
    assert len(sp0) == 9 and len(dd0) == 3 * (64 - 20 - 3)
    sp, dd = plan_balance(U, max_keys=6, gen=g)
    assert torch.equal(sp, sp0) and len(dd) == 0                                  # 3 splits = 6 keys: no room left
    sp, dd = plan_balance(U, max_keys=5, gen=g)
    assert len(sp) == 6 and all(int((dd[:, 0] == c).sum()) == 1 for c in range(3))   # 2 splits (busiest) + 1 dead
    assert torch.equal(sp, sp0.view(3, 3, 3)[:, :2].reshape(-1, 3))
    sp, dd = plan_balance(U, max_keys=10, gen=torch.Generator().manual_seed(5))
    _, dd2 = plan_balance(U, max_keys=10, gen=torch.Generator().manual_seed(5))
    _, dd3 = plan_balance(U, max_keys=10, gen=torch.Generator().manual_seed(6))
    assert torch.equal(dd, dd2) and not torch.equal(dd, dd3)                      # the dedicated RNG decides
    assert len(sp) == 9 and all(int((dd[:, 0] == c).sum()) == 4 for c in range(3))
    assert set(map(tuple, dd.tolist())) <= set(map(tuple, dd0.tolist()))
    # --mem-balance-dead-max: the dead re-seeds get their own cap; the splits keep theirs
    sp, dd = plan_balance(U, max_keys=2, gen=torch.Generator().manual_seed(1), dead_max=7)
    assert torch.equal(sp, sp0.view(3, 3, 3)[:, :1].reshape(-1, 3))              # 1 split (2 keys) per codebook
    assert all(int((dd[:, 0] == c).sum()) == 7 for c in range(3))                 # + 7 dead keys, independently
    _, dd_all = plan_balance(U, max_keys=2, dead_max=10 ** 6)                     # every dead key not split into
    assert len(dd_all) == 3 * (64 - 20 - 1) and set(map(tuple, dd.tolist())) <= set(map(tuple, dd_all.tolist()))
    sp, dd = plan_balance(U, gen=torch.Generator().manual_seed(1), dead_max=7)
    assert torch.equal(sp, sp0) and len(dd) == 21                                 # splits uncapped, deads capped
    keep = torch.zeros(3, 64, dtype=torch.bool)
    keep[0, 30] = True
    _, ddk = plan_balance(U, keep=keep)
    assert [0, 30] not in ddk.tolist() and len(ddk) == len(dd0) - 1
    # in the memory: the window counts, the keep rule, the cap, the window reset
    m = mem_loop(torch.float32, mem_shared=16)
    mem = m.mem
    mem.balance_keep, mem.balance_max = True, 0.25                                 # 4 of 16 keys per codebook
    mem.qres.normal_()
    mem.lastq_ok.fill_(True)
    mem.pend[0, 0, 0, 5] = 3.0
    mem.pend_e[1, 2] = 1.0
    mem.balance_every = 0
    mem.after_step(1e-3)
    assert mem.win[0, 0, 0, 5] == 3.0 and mem.win_e[1, 2] == 1.0 and mem.win.sum() == 3.0
    with torch.no_grad():
        mem.U.fill_(1.0 / 16)
        mem.U[0, 0, 0, 5] = mem.U[0, 0, 0, 6] = 1e-7                                # two dead keys, one used
        mem.U[1, 1, 1, :10] = 1e-7                                                   # ten dead keys
        mem.U_e.fill_(1.0 / 8)
        mem.U_e[1, 2] = mem.U_e[1, 3] = 1e-7
    k1, k2 = mem.keys.k1.detach().clone(), mem.keys.k2.detach().clone()
    st = mem.balance()
    assert torch.equal(mem.keys.k1[0, 0, 5], k1[0, 0, 5]) and not torch.equal(mem.keys.k1[0, 0, 6], k1[0, 0, 6])
    assert int((mem.keys.k2[1, 1] != k2[1, 1]).any(-1).sum()) == 4                # capped: 4 of the 10
    assert st["dead"] == 1 + 4 and st["p_dead"] == 1 and st["splits"] == 0
    assert mem.win.sum() == 0 and mem.win_e.sum() == 0                             # a new window
    mem.balance_keep = False
    mem.pend[0, 0, 0, 5] = 3.0
    mem.after_step(1e-3)
    assert mem.win.sum() == 0                                                      # (off: nothing counted)


def test_12_trainer_flags_probe_every_and_mild_flags_end_to_end(tmp_path):
    """S9 and the mild flags through train_lm.main: probes on every 2nd evaluation and the last; M1-M3 on."""
    data = _tiny_data(tmp_path / "data")
    extra = ("--concepts 2,4,8,16 --mem-shared 8 --mem-topm 4 --mem-topk 3 --mem-private 8 --mem-dv 16 "
             "--mem-write-start 5 --mem-freeze 0.75 --mem-probes 2 --eval-every 10 --mem-probe-every 2 "
             "--mem-readdr-every 2 --mem-readdr-start 1 --mem-margin-passes 1 --mem-balance-every 7 --mem-balance-max 0.25 "
             "--mem-balance-keep 1").split()
    run, recs, ck = _run_main(current().train_lm, tmp_path / "runs", "mem", data, extra)
    assert len(recs) == 5
    probed = ["mem_offset_overlap" in r for r in recs]
    assert probed == [True, False, True, False, True]
    assert all("mem_subkeys_dead" in r for r in recs) and "mem_balance" in recs[1]
    assert recs[-1]["mem_frozen"] == 1 and all(np.isfinite(r["val"]) for r in recs)
    assert all(set(r["mem_reuse_overlap"]) == {"2", "4"} for r, p in zip(recs, probed) if p)   # probes: K = 6
    assert "MEMORY FROZEN" in (run / "train.log").read_text()


def test_13_readdr_start_fresh_first_passes_then_every_2nd():
    """M1 with --mem-readdr-start P (default 4): the reads after passes 0 .. P-1 always address anew, from pass P
    on every 2nd (P, P + 2, ...), the others reuse the previous read's indices; flag off (every 1) = every read."""
    m = mem_loop()
    mem = m.mem
    assert m.mem_readdr_every == 1 and m.mem_readdr_start == 4                    # the defaults
    calls = []
    sel_fn = mem.select
    mem.select = lambda *a: (calls.append(1), sel_fn(*a))[1]
    x = tokens(2, 17)
    m.eval()
    m.K = 9                                                                        # reads after passes 0 .. 7
    for every, start, fresh in ((1, 4, range(8)), (2, 4, (0, 1, 2, 3, 4, 6)), (2, 2, (0, 1, 2, 4, 6)),
                                (3, 5, (0, 1, 2, 3, 4, 5))):
        m.mem_readdr_every, m.mem_readdr_start = every, start
        calls.clear()
        with torch.no_grad():
            m(x)
        assert len(calls) == 1 + len(fresh)                                        # read0 + the fresh reads
        mem.record = dict(pos=torch.arange(20), reads=[], calls=[])
        with torch.no_grad():
            m(x)
        rec = mem.record["reads"][1:]                                              # (without read0)
        mem.record = None
        assert len(rec) == 8
        for k in range(1, 8):
            same = torch.equal(rec[k]["rows"], rec[k - 1]["rows"])
            assert same == (k not in fresh), (every, start, k)
            assert (rec[k]["reuse_ovl"] is None) == (k in fresh)


def test_14_reuse_overlap_probe_metric():
    """mem_reuse_overlap: on the probe batch, for every reused read, the set overlap of the reused rows with the
    rows fresh addressing would pick (eval only; training forwards never compute it)."""
    from memory import MemoryProbes
    m = mem_loop()
    m.mem_readdr_every, m.mem_readdr_start = 2, 1
    val = np.random.RandomState(0).randint(1, 50, 600).astype(np.uint16)
    pr = MemoryProbes(val, 16, n_seq=2, n_pos=4, K=6)
    out = pr.run(m, "B")
    reused = [r for r in out["reads"] if r["reuse_ovl"] is not None]
    assert [r["k"] for r in reused] == [2, 4]
    for r in reused:
        ref = (r["rows"][..., :, None] == r["rows_fresh"][..., None, :]).any(-1).float().mean(-1)
        assert torch.equal(r["reuse_ovl"], ref) and ((0 <= ref) & (ref <= 1)).all()
    met, _ = pr.measure(m)
    assert set(met["mem_reuse_overlap"]) == {"2", "4"}
    assert all(0 <= v <= 1 for v in met["mem_reuse_overlap"].values())
    m.mem_readdr_every = 1
    assert "mem_reuse_overlap" not in pr.measure(m)[0]                             # flag off: no metric
    # training forwards: no extra select for the metric (only the fresh reads address)
    calls = []
    sel_fn = m.mem.select
    m.mem.select = lambda *a: (calls.append(1), sel_fn(*a))[1]
    m.mem_readdr_every = 2
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 6, 4, 0
    x = tokens(2, 17)
    m(x[:, :-1], x[:, 1:])
    assert len(calls) == 1 + 3                                                     # read0 + passes 0, 1, 3
