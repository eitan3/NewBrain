"""Tests of the memory life cycle package (plan_v2/21_lifecycle_plan.md section 7): life cycle (lazy decay, discounted
accumulator, switch, ramp), fixed addressing, uncapped restart, coarse pool, neighbour writes, resume and the spill /
free_cache paths. CPU only, tiny configs.

    python -m pytest test_lifecycle.py -q

Flags off == the code before the package (plan_v2/baseline_mem_v3), bit for bit (test 1).
"""
import copy
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_memory import _run_main, _tiny_data, mem_loop, tokens, train_step

HERE = Path(__file__).resolve().parent
BASE = HERE / "plan_v2" / "baseline_mem_v3"
MODS = ("spiking_lm", "loop_brain", "brain_opt", "neurogenesis", "train_lm", "memory", "concepts")
_BASELINE = None


def baseline():
    """The code before the life-cycle package (plan_v2/baseline_mem_v3) as separate module objects (as test_speed's
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


MEM_ARGS = ("--concepts 2,4,8,16 --mem-shared 8 --mem-topm 4 --mem-topk 3 --mem-private 8 --mem-dv 16 "
            "--mem-write-start 5 --mem-probes 2 --mem-repair 0.25")


def _same_recs(ra, rb):
    assert len(ra) == len(rb)
    for a, b in zip(ra, rb):
        a, b = dict(a), dict(b)
        for k in ("tok_s", "elapsed"):  # wall-clock
            a.pop(k), b.pop(k)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), a["step"]


def _same_state(sa, sb):
    assert list(sa) == list(sb)
    for k in sa:
        a, b = sa[k], sb[k]
        if torch.is_tensor(a):
            assert torch.equal(a, b), k
        else:  # (the memory's extra state)
            assert repr(a) == repr(b), k


# ---------------------------------------------------------------- test 1: all flags off == baseline_mem_v3
def test_01a_flags_off_bit_identical_memory_step_with_growth_balancing_freeze():
    """test_speed's trainer step with the full memory (growth, two balancing events, the freeze, accumulation)."""
    from test_speed import _check, _drive
    old = _drive(baseline(), new=True)
    new = _drive(current(), new=True)
    _check(old, new, exact=True)


def test_01b_flags_off_bit_identical_40_steps_without_memory():
    """test_memory's 40-step trainer step (memory off) against baseline_mem_v3."""
    from test_memory import _drive
    old, new = _drive(baseline()), _drive(current())
    assert len(old["losses"]) == len(new["losses"]) == 80
    for a, b in zip(old["losses"], new["losses"]):
        assert torch.equal(a, b)
    for a, b in zip(old["outs"], new["outs"]):
        assert torch.equal(a, b)
    assert old["cfg"] == new["cfg"]
    _same_state(old["state"], new["state"])
    assert torch.equal(old["rng"], new["rng"])


def test_01c_flags_off_bit_identical_end_to_end_with_memory(tmp_path):
    """train_lm.main with the memory on (the new flags at their defaults), old file vs new file: metrics and
    final.pt identical (growth, the freeze, probes, repair, the loop report)."""
    data = _tiny_data(tmp_path / "data")
    extra = (MEM_ARGS + " --mem-freeze 0.5 --eval-every 10 --mem-readdr-every 2 --mem-readdr-start 1").split()
    _, ra, ca = _run_main(baseline().train_lm, tmp_path / "runs_old", "old", data, extra)
    _, rb, cb = _run_main(current().train_lm, tmp_path / "runs_new", "new", data, extra)
    assert len(ra) == 5 and ra[-1]["mem_frozen"] == 1
    _same_recs(ra, rb)
    assert ca["cfg"] == cb["cfg"]
    _same_state(ca["state"], cb["state"])


# ---------------------------------------------------------------- helpers
def small_mem(dtype=torch.float64, **kw):
    from memory import Memory
    torch.manual_seed(0)
    cfg = dict(d=8, ds=64, L=2, H=1, S=8, m=4, km=3, dv=4, life=True)
    cfg.update(kw)
    mem = Memory(**cfg).to(dtype)
    mem.s0_cfg, mem.lr, mem.clip = "1.0", 0.05, 10.0
    return mem


def _touched(G):
    return (G != 0).any(1).nonzero().squeeze(1)


def LoopBrainFromCfg(m):
    from loop_brain import LoopBrain
    m2 = LoopBrain(**copy.deepcopy(m.cfg)).double()
    m2.load_state_dict(m.state_dict())
    return m2


# ---------------------------------------------------------------- test 2: lazy decay == global decay
@pytest.mark.parametrize("beta_sched", [False, True])
def test_02_lazy_decay_equals_eager_global_decay(beta_sched):
    """Random reads / writes over 200 steps under a lambda (and beta) schedule: the lazy reads (MemRead with the
    per-row factors) and the effective values / accumulators equal an eager V <- (1 - lambda) V, S <- beta S
    reference to 1e-6; settle() keeps the effective values."""
    from memory import MemRead, _anchor, S_FLOOR
    mem = small_mem()
    n = mem.V.size(0)
    Ve, Se = mem.V.clone(), None
    gen = torch.Generator().manual_seed(1)
    for t in range(200):
        lam = 0.02 * (1 + math.sin(t / 15))
        beta = 0.9 + 0.09 * math.cos(t / 20) if beta_sched else 1.0
        rows = torch.randint(0, n, (5, 3), generator=gen)
        w = torch.softmax(torch.randn(5, 3, generator=gen, dtype=torch.float64), -1)
        c = torch.randn(5, 4, generator=gen, dtype=torch.float64)
        f = mem._factors(rows)
        r = MemRead.apply(_anchor("cpu", True), w * f, mem.V, mem.G, rows, w)
        (r * c).sum().backward()
        ve = Ve.clone().requires_grad_(True)
        re = F.embedding_bag(rows, ve, per_sample_weights=w, mode="sum")
        g, = torch.autograd.grad((re * c).sum(), ve)
        assert torch.allclose(r, re, atol=1e-6, rtol=0)
        assert torch.allclose(mem.G, g, atol=1e-12, rtol=0)                   # G in current (decayed) units
        # one optimizer step: lazy (Lambda, Psi) vs eager (every row)
        mem.life_advance(lam, beta)
        mem._row_adagrad(mem.V, mem.G, mem.S_acc, mem.wcount, 0, _touched(mem.G))
        act = (g != 0).any(1)
        Ve = Ve * (1 - lam)
        Se = torch.full((n,), 1.0, dtype=torch.float64) if Se is None else Se * beta
        Se = torch.where(act, torch.clamp(Se + (g * g).mean(1), min=S_FLOOR * 1.0), Se)
        dv = -mem.lr * g / Se.sqrt()[:, None]
        dv = dv * (mem.clip / dv.norm(dim=1, keepdim=True).clamp(min=1e-30)).clamp(max=1.0)
        Ve = torch.where(act[:, None], Ve + dv, Ve)
        eff = mem.V * torch.exp(mem.lam_r - mem.lam_now)[:, None]
        assert torch.allclose(eff, Ve, atol=1e-6, rtol=0), t
        seff = mem.S_acc * torch.exp(mem.psi_r - mem.psi_now)
        assert torch.allclose(seff, Se, atol=1e-6, rtol=1e-6), t
        assert mem.G.abs().sum() == 0
    assert mem.life_state["Lam"] > 2.0                                          # (real decay happened)
    eff = mem.V * torch.exp(mem.lam_r - mem.lam_now)[:, None]
    mem.settle()
    assert torch.allclose(mem.V, eff, atol=1e-12) and torch.equal(mem.lam_r, mem.lam_now.expand_as(mem.lam_r))
    assert mem._factors(torch.arange(4)) is None                                # mature: no factors


# ---------------------------------------------------------------- test 3: the discounted accumulator
def test_03_accumulator_beta1_is_adagrad_bit_for_bit_and_the_floor_holds():
    from memory import S_FLOOR
    a, b = small_mem(life=True), small_mem(life=False)
    assert torch.equal(a.V, b.V)
    gen = torch.Generator().manual_seed(2)
    for t in range(30):
        G = torch.zeros_like(a.G)
        rows = torch.randint(0, a.V.size(0), (6,), generator=gen)
        G[rows] = torch.randn(6, a.dv, generator=gen, dtype=G.dtype)
        a.G.copy_(G)
        b.G.copy_(G)
        a.life_advance(0.0, 1.0)                                                 # lambda = 0, beta = 1
        a._row_adagrad(a.V, a.G, a.S_acc, a.wcount, 0, _touched(G))
        b._row_adagrad(b.V, b.G, b.S_acc, b.wcount, 0, _touched(G))
        assert torch.equal(a.V, b.V) and torch.equal(a.S_acc, b.S_acc) and torch.equal(a.wcount, b.wcount)
    # beta < 1: the eager formula; a strong discount hits the floor
    m = small_mem()
    G = torch.zeros_like(m.G)
    G[3] = 0.01
    m.G.copy_(G)
    m.life_advance(0.0, 0.5)
    m._row_adagrad(m.V, m.G, m.S_acc, m.wcount, 0, _touched(G))               # first write: S0 = 1, + g^2
    assert m.S_acc[3].item() == pytest.approx(1.0 + 1e-4)
    for _ in range(3):
        m.life_advance(0.0, 0.5)
    m.G[3] = 0.01
    m._row_adagrad(m.V, m.G, m.S_acc, m.wcount, 0, torch.tensor([3]))
    assert m.S_acc[3].item() == pytest.approx((1.0 + 1e-4) * 0.125 + 1e-4)     # B = beta^3, no floor yet
    for _ in range(10):
        m.life_advance(0.0, 0.5)
    m.G[3] = 0.01
    m._row_adagrad(m.V, m.G, m.S_acc, m.wcount, 0, torch.tensor([3]))
    assert m.S_acc[3].item() == pytest.approx(S_FLOOR * 1.0)                   # the floor


# ---------------------------------------------------------------- test 4: the switch and the ramp
def _life(steps=1000, restart=100, **kw):
    from memory import LifeCycle
    mem = small_mem(restart=restart)
    return LifeCycle(mem, steps, **kw), mem


def test_04_switch_triggers_ramp_and_freeze_condition():
    # (a) stability >= 0.9 on 3 consecutive probe evals (a dip resets the run)
    lc, mem = _life()
    for t, s in ((100, 0.95), (200, 0.95), (300, 0.5), (400, 0.92), (500, 0.93)):
        assert not lc.begin_step(t, True)
        lc.observe(s)
    assert mem.life_state["phase"] == "infancy"
    lc.begin_step(550, True)
    lc.observe(0.97)
    assert mem.life_state["phase"] == "infancy"                                 # (switches at the next step)
    lc.begin_step(551, True)
    st = mem.life_state
    assert st["phase"] == "ramp" and st["reason"] == "stab" and st["t_switch"] == 551
    # the ramp: lambda, 1 - beta linear to 0, the restart interval 100 -> 1000
    for k in range(0, 50):
        assert not lc.begin_step(551 + k, True)
        assert st["decay"] == pytest.approx(1e-3 * (1 - k / 50))
        assert 1 - st["beta"] == pytest.approx(0.01 * (1 - k / 50))
        assert st["restart_every"] == round(100 * (1 + 9 * k / 50))
    assert not lc.begin_step(601, False)                                        # births left: no freeze yet
    assert lc.begin_step(601, True) and st["decay"] == 0.0 and st["beta"] == 1.0
    lc.mature()
    assert st["phase"] == "mature" and st["restart_every"] == 0
    # (b) the learning-rate decay start; births must be done
    lc, mem = _life()
    for t in range(0, 800):
        lc.begin_step(t, t >= 700)
    assert mem.life_state["phase"] == "infancy"
    lc.begin_step(800, False)
    assert mem.life_state["phase"] == "infancy"
    lc.begin_step(805, True)
    assert mem.life_state["reason"] == "lr_decay" and mem.life_state["t_switch"] == 805
    # (c) the safety cap
    lc, mem = _life(lr_decay=0.05)
    for t in range(0, 901):
        lc.begin_step(t, True)
    assert mem.life_state["reason"] == "max" and mem.life_state["t_switch"] == 900
    # the rates reach the memory: after_step advances Lambda by -log(1 - lambda)
    lam0 = mem.life_state["Lam"]
    mem.write_on = False
    mem.after_step(1e-3, [])
    assert mem.life_state["Lam"] - lam0 == pytest.approx(-math.log1p(-mem.life_state["decay"]))


# ---------------------------------------------------------------- test 5: fixed addressing
def test_05_addr_learn_0_no_grad_online_equals_ema_norm_stats_update():
    from brain_opt import BrainOpt
    m = mem_loop(mem_addr_learn=0)
    mem = m.mem
    assert mem.lam_m == 0.0 and all(not p.requires_grad for p in mem.keys.parameters())
    opt = BrainOpt(m, lr=3e-3)
    x = tokens()
    nm0 = mem.norm_m.clone()
    for i in range(3):
        train_step(m, opt, x, offset=i % 2)
        assert all(p.grad is None for p in mem.keys.parameters())
    assert mem.margin_stats is None                                             # no margin loss
    assert torch.equal(mem.keys.wq, mem.wq_ema) and torch.equal(mem.keys.k1, mem.k1_ema)
    assert torch.equal(mem.keys.k2, mem.k2_ema)
    assert not torch.equal(mem.norm_m, nm0)                                     # the norm statistics update
    a_hat = torch.randn(mem.L, 7, mem.ain, dtype=torch.float64)
    sel = mem.select(a_hat, a_hat, False)
    on = mem.online(a_hat, sel)
    q1, q2 = mem._queries(a_hat, mem.wq_ema)
    ref = mem._key_scores(q1, mem.k1_ema, sel["i"]) + mem._key_scores(q2, mem.k2_ema, sel["j"])
    assert torch.equal(on["sig"], ref)                                          # online == EMA addressing
    mem.freeze()
    nm1 = mem.norm_m.clone()
    train_step(m, opt, x)
    assert torch.equal(mem.norm_m, nm1)                                         # ... until the freeze
    m2 = LoopBrainFromCfg(m)
    assert all(not p.requires_grad for p in m2.mem.keys.parameters())


# ---------------------------------------------------------------- test 6: the restart
def _restart_mem():
    from brain_opt import BrainOpt
    m = mem_loop(mem_restart=4, mem_life=1, mem_addr_learn=0, mem_coarse=5)
    opt = BrainOpt(m, lr=3e-3)
    x = tokens()
    for i in range(2):
        train_step(m, opt, x, offset=i % 2)
    return m, opt, x


def test_06_restart_uncapped_distinct_slots_band_reset_deterministic():
    m, opt, x = _restart_mem()
    mem = m.mem
    assert mem.nres == 1024 and mem.qres.shape[-2] == 1024 and mem.stats.get("restart") is None
    S = mem.S
    g = torch.Generator().manual_seed(5)
    mem.qres.copy_(F.normalize(torch.randn(mem.qres.shape, generator=g, dtype=mem.qres.dtype), dim=-1))
    mem.qres_c.copy_(F.normalize(torch.randn(mem.qres_c.shape, generator=g, dtype=mem.qres.dtype), dim=-1))
    mem.win.fill_(1.0)
    mem.win_c.fill_(1.0)
    mem.win[0, 0, 0, 1:] = 0                                                    # 7 of 8 dead: uncapped
    mem.win[2, 1, 1, [2, 5]] = 0
    mem.win_c[3, 0, 0, 4] = 0
    mem.V.add_(1.0)
    mem.life_advance(1e-2, 0.9)
    mem.opt_reset = None
    before = copy.deepcopy(mem)
    twin, zero = copy.deepcopy(mem), copy.deepcopy(mem)
    st = mem.restart_keys()
    assert st == {"dead": 9, "dead_c": 1}
    k1, k1b = mem.keys.k1.detach(), before.keys.k1.detach()
    changed = (k1 != k1b).any(-1)
    assert changed[0, 0, 1:].all() and not changed[0, 0, 0] and changed.sum() == 7
    assert ((mem.keys.k2 != before.keys.k2).any(-1)).sum() == 2
    assert ((mem.keys.k1_c != before.keys.k1_c).any(-1)).sum() == 1
    assert torch.equal(mem.keys.k1, mem.k1_ema) and torch.equal(mem.keys.k1_c, mem.k1_c_ema)
    # distinct reservoir slots: each new key's nearest reservoir entry, all different within the codebook
    res = mem.qres[0, 0, 0]                                                     # (nres, 64)
    near = (F.normalize(mem.keys.k1[0, 0, 1:].detach(), dim=-1) @ res.t()).argmax(-1)
    assert len(set(near.tolist())) == 7
    # the band reset: the dead keys' bands (the coarse one at its offset) and nothing else: their MemOpt state, V
    # kept (with its pending decay applied: the effective values do not change; other heads' live keys share them)
    z = lambda n: torch.zeros(n, dtype=torch.long)
    rows = torch.cat([mem._bands(z(7), z(7), torch.arange(1, 8)).flatten(),
                      mem._bands(torch.tensor([2, 2]), torch.tensor([1, 1]), torch.tensor([2, 5])).flatten(),
                      mem._bands(torch.tensor([3]), torch.tensor([0]), torch.tensor([4]), mem.Sc,
                                 mem.L * S * S).flatten()])
    reset = torch.zeros(mem.V.size(0), dtype=torch.bool)
    reset[rows] = True
    f = torch.exp(before.lam_r - before.lam_now)[:, None]
    assert torch.allclose(mem.V[reset], (before.V * f)[reset], rtol=0, atol=1e-12) and (mem.V[reset] != 0).all()
    assert (mem.wcount[reset] == 0).all()
    assert (mem.S_acc[reset] == mem._s0[0]).all() and mem._s0[0] > 0
    assert (mem.lam_r[reset] == mem.lam_now).all() and (mem.psi_r[reset] == mem.psi_now).all()
    for t in ("V", "wcount", "S_acc", "lam_r", "psi_r"):
        assert torch.equal(getattr(mem, t)[~reset], getattr(before, t)[~reset]), t
    assert (mem.win == 0).all() and (mem.win_c == 0).all()
    # deterministic with mem.gen
    twin.restart_keys()
    assert torch.equal(twin.keys.k1, mem.keys.k1) and torch.equal(twin.keys.k1_c, mem.keys.k1_c)
    assert torch.equal(twin.V, mem.V) and torch.equal(twin.gen.get_state(), mem.gen.get_state())
    # --mem-restart-zero: the same re-seeds, and the bands' V cleared too
    zero.restart_zero = True
    zero.restart_keys()
    assert torch.equal(zero.keys.k1, mem.keys.k1) and (zero.V[reset] == 0).all()
    assert torch.equal(zero.V[~reset], mem.V[~reset]) and torch.equal(zero.S_acc, mem.S_acc)
    assert torch.equal(zero.wcount, mem.wcount) and torch.equal(zero.lam_r, mem.lam_r)


def test_06b_restart_schedule_and_none_when_mature():
    m, opt, x = _restart_mem()
    mem = m.mem
    for i in range(2):                                                          # steps 3, 4: the event at 4
        train_step(m, opt, x)
    assert mem.life_state["last_restart"] == 4 and "restart" in mem.stats
    mem.freeze()
    k1 = mem.keys.k1.detach().clone()
    mem.win.zero_()
    for i in range(5):
        train_step(m, opt, x)
    assert mem.life_state["last_restart"] == 4 and torch.equal(mem.keys.k1, k1)


# ---------------------------------------------------------------- test 7: the coarse pool
def test_07_coarse_pool_offsets_zero_init_and_probes():
    from memory import MemoryProbes
    from brain_opt import BrainOpt
    m0, m = mem_loop(), mem_loop(mem_coarse=5, mem_coarse_heads=2)
    mem = m.mem
    n0, nc = mem.L * mem.S ** 2, mem.L * mem.Sc ** 2
    assert mem.V.size(0) == n0 + nc and torch.equal(mem.V[:n0], m0.mem.V)
    a_hat = torch.randn(mem.L, 30, mem.ain, dtype=torch.float64)
    sel = mem.select(a_hat, a_hat, False)
    assert sel["rows"].max() < n0 and sel["c"]["rows"].min() >= n0 and sel["c"]["rows"].max() < n0 + nc
    assert sel["c"]["rows"].shape == (mem.L, 2, 30, mem.km)
    x = tokens()
    m0.eval(), m.eval()
    with torch.no_grad():
        assert torch.equal(m0(x), m(x))                                         # zero-init W_o^c: no change
    opt = BrainOpt(m, lr=3e-3)
    l0 = train_step(m0, BrainOpt(m0, lr=3e-3), x)
    l1 = train_step(m, opt, x)
    assert torch.equal(l0, l1) and m.mem.w_o_c.abs().sum() > 0                 # then W_o^c learns
    train_step(m, opt, x)
    assert m.mem.wcount[n0:].sum() > 0                                          # the coarse rows are written
    val = np.random.RandomState(0).randint(1, 50, 600).astype(np.uint16)
    pr = MemoryProbes(val, 16, n_seq=2, n_pos=4, K=4)
    pr.measure(m)
    met, _ = pr.measure(m)
    assert 0 <= met["mem_last_stab_w_c"] <= 1 and "mem_last_stab_w" in met and "mem_subkeys_dead_c" in met
    assert m.cfg["mem_coarse"] == 5 and "mem_coarse" not in m0.cfg
    m2 = LoopBrainFromCfg(m)
    m2.eval(), m.eval()
    with torch.no_grad():
        assert torch.equal(m2(x), m(x))


# ---------------------------------------------------------------- test 8: neighbour writes
def test_08_neighbour_writes_backward_only():
    from memory import MemRead, _anchor
    g = torch.Generator().manual_seed(3)
    V = torch.randn(40, 5, generator=g, dtype=torch.float64)
    rows = torch.randint(0, 40, (6, 3), generator=g)
    xrows = torch.randint(0, 40, (6, 2), generator=g)
    w = torch.softmax(torch.randn(6, 3, generator=g, dtype=torch.float64), -1).requires_grad_(True)
    xw = 0.15 * torch.softmax(torch.randn(6, 2, generator=g, dtype=torch.float64), -1)
    c = torch.randn(6, 5, generator=g, dtype=torch.float64)
    G0, G1 = torch.zeros_like(V), torch.zeros_like(V)
    r0 = MemRead.apply(_anchor("cpu", True), w, V, G0, rows)
    dw0, = torch.autograd.grad((r0 * c).sum(), w)
    r1 = MemRead.apply(_anchor("cpu", True), w, V, G1, rows, None, xrows, xw)
    dw1, = torch.autograd.grad((r1 * c).sum(), w)
    assert torch.equal(r0, r1) and torch.equal(dw0, dw1)                       # forward and dw unchanged
    ref = torch.zeros_like(V).index_add_(0, xrows.flatten(), (xw[..., None] * c[:, None]).reshape(-1, 5))
    assert torch.allclose(G1 - G0, ref, atol=1e-14)                            # G += alpha omega^n g
    # in the model: the forward and every parameter gradient are unchanged; more rows are written
    m0, m = mem_loop(), mem_loop(mem_nbr=8)
    x = tokens()
    outs = []
    w_o = torch.randn_like(m0.mem.w_o) * 0.1                                    # (zero-init W_o: no write yet)
    for mm in (m0, m):
        mm.mem.w_o.data.copy_(w_o)
        mm.train()
        mm.K, mm.aux_k_fixed, mm.group_offset = 4, 2, 0
        _, loss = mm(x[:, :-1], x[:, 1:])
        (loss + mm.aux_loss()).backward()
        outs.append(loss.detach())
    assert torch.equal(outs[0], outs[1])
    t0, t1 = m0.mem.G.ne(0).any(1), m.mem.G.ne(0).any(1)
    assert not (t0 & ~t1).any() and t1.sum() > t0.sum()
    for (n, p), (_, q) in zip(m0.named_parameters(), m.named_parameters()):
        assert (p.grad is None) == (q.grad is None) and (p.grad is None or torch.allclose(p.grad, q.grad, atol=1e-12)), n
    a_hat = torch.randn(m.mem.L, 9, m.mem.ain, dtype=torch.float64)
    sel = m.mem.select(a_hat, a_hat, False)
    on = m.mem.online(a_hat, sel)
    km = m.mem.km
    assert sel["nrows"].shape[-1] == 8 - km and on["wn"].shape[-1] == 8 - km
    assert torch.allclose(on["wn"].sum(-1), torch.full_like(on["wn"][..., 0], 0.15))
    assert torch.equal(on["wn"][..., :1] > 0, torch.ones_like(on["wn"][..., :1], dtype=torch.bool))
    assert (on["wn"][..., 0] >= on["wn"][..., -1]).all()                       # rank km + 1 weighs the most


# ---------------------------------------------------------------- test 9: resume
class _Stop(Exception):
    pass


LIFE_ARGS = (MEM_ARGS + " --eval-every 10 --mem-life 1 --mem-addr-learn 0 --mem-restart 3 --mem-coarse 6 --mem-nbr 8 "
             "--mem-switch-max 0.45 --mem-ramp 0.25 --neuro-t 0.05,0.3 --ckpt-every 20")


def test_09_resume_is_bit_identical(tmp_path, monkeypatch):
    """40 straight steps == 20 + save + resume + 20 (train_lm.main, CPU): growth before the cut, the switch at 18
    and the ramp across the cut (freeze at 28), restart events every 3 steps and probe evals on both sides."""
    tl = current().train_lm
    data = _tiny_data(tmp_path / "data")
    extra = LIFE_ARGS.split()
    run_a, ra, ca = _run_main(tl, tmp_path / "runs_a", "a", data, extra)
    save = tl.save_checkpoint

    def crash(path, obj):  # the run "dies" right after writing last.pt at step 20
        save(path, obj)
        raise _Stop

    monkeypatch.setattr(tl, "save_checkpoint", crash)
    with pytest.raises(_Stop):
        _run_main(tl, tmp_path / "runs_b", "b", data, extra)
    monkeypatch.setattr(tl, "save_checkpoint", save)
    last = tmp_path / "runs_b" / "b" / "last.pt"
    assert last.exists() and not Path(str(last) + ".tmp").exists()
    run_b, rb, cb = _run_main(tl, tmp_path / "runs_b", "b", data, extra + ["--resume", str(last)])
    log = (run_b / "train.log").read_text()
    assert "RESUMED at step 20" in log and "MEMORY SWITCH (max)" in log and "MEMORY FROZEN" in log
    assert len(ra) == len(rb) == 5
    assert ra[2]["mem_life_state"]["phase"] == "ramp" and ra[3]["mem_life_state"]["phase"] == "mature"
    assert ra[2]["births"] == ra[4]["births"] > 0 and "mem_last_stab_w" in ra[3]
    assert all("mem_restart_last" in r for r in ra[1:])                        # (events every 3 steps; tiny
    # pools: every key is used, so nothing is re-seeded here, but each event draws from mem.gen; test 6: re-seeds)
    _same_recs(ra, rb)
    assert ca["cfg"] == cb["cfg"]
    _same_state(ca["state"], cb["state"])


# ---------------------------------------------------------------- test 10: spill guard / free_cache / checkpoint helpers
def test_10_free_cache_rng_and_atomic_save_on_cpu(tmp_path):
    import random
    tl = current().train_lm
    tl.free_cache("cpu")                                                        # a no-op on the CPU
    st = tl.get_rng_state("cpu")
    a = (torch.rand(3), np.random.rand(), random.random())
    tl.set_rng_state(st, "cpu")
    b = (torch.rand(3), np.random.rand(), random.random())
    assert torch.equal(a[0], b[0]) and a[1:] == b[1:]
    tl.save_checkpoint(tmp_path / "x.pt", {"a": 1})
    tl.save_checkpoint(tmp_path / "x.pt", {"a": 2})
    assert torch.load(tmp_path / "x.pt")["a"] == 2 and not (tmp_path / "x.pt.tmp").exists()


def test_10b_report_checkpoints_and_life_logging(tmp_path):
    data = _tiny_data(tmp_path / "data")
    extra = (LIFE_ARGS.replace("--ckpt-every 20", "--ckpt-every 0") + " --save-report-ckpts 1").split()
    run, recs, ck = _run_main(current().train_lm, tmp_path / "runs", "r", data, extra)
    names = sorted(p.name for p in run.glob("ckpt_*.pt"))
    assert names == ["ckpt_10.pt", "ckpt_20.pt", "ckpt_30.pt", "ckpt_40.pt"]
    from loop_brain import LoopBrain
    c = torch.load(run / "ckpt_20.pt", weights_only=False)
    m = LoopBrain(**c["cfg"])
    m.load_state_dict(c["state"])
    assert m.mem.life_state["phase"] == "ramp"
    for r in recs:
        assert {"phase", "decay", "beta", "Lam", "reason"} <= set(r["mem_life_state"])
        assert len(r["mem_vnorm_by_wcount"]) == 3 and "mem_gate_mean" in r
    assert recs[-1]["mem_life_state"]["phase"] == "mature" and recs[-1]["mem_frozen"] == 1


# ---------------------------------------------------------------- gpu.lock: a reused pid is not a holder
def test_11_stale_gpu_lock_with_a_reused_non_python_pid(tmp_path, capsys):
    """A lock whose pid now belongs to a non-python process (Windows reuses pids) is stale: removed and replaced;
    a live python train_lm.py process still holds the GPU; a dead pid is stale too."""
    import os
    import subprocess
    tl = current().train_lm
    other = subprocess.Popen(["ping", "-n", "60", "127.0.0.1"] if sys.platform == "win32" else ["sleep", "60"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "train_lm.py"])
    try:
        lock = tmp_path / "gpu.lock"
        lock.write_text(f"{other.pid} old_run\n")
        assert tl.pid_alive(other.pid) and not tl.lock_holder_alive(other.pid)
        tl.claim_gpu(tmp_path, "new_run")
        assert lock.read_text().split() == [str(os.getpid()), "new_run"]
        assert "stale gpu.lock removed" in capsys.readouterr().out
        lock.write_text(f"{holder.pid} busy_run\n")
        assert tl.lock_holder_alive(holder.pid)
        with pytest.raises(SystemExit):
            tl.claim_gpu(tmp_path, "new_run")
        assert lock.read_text().split()[0] == str(holder.pid)
    finally:
        other.kill()
        holder.kill()
        other.wait()
        holder.wait()
    lock.write_text(f"{other.pid} dead_run\n")
    tl.claim_gpu(tmp_path, "new_run")
    assert lock.read_text().split()[0] == str(os.getpid())


# ---------------------------------------------------------------- the eval peak: chunked report forwards
def test_12_chunked_loop_report_forwards_equal_the_full_batch():
    """--mem-probe-batch: think_diag / tick_losses on chunks of the batch give the full batch's numbers (dropless
    routing here; float64), the neighbour weights are not built in eval, and the probes run in small batches."""
    from memory import MemoryProbes
    m = mem_loop(mem_nbr=8)
    m.eval()
    x = tokens(4, 17, seed=7)
    xi, y = x[:, :-1], x[:, 1:]
    full, part = m.think_diag(xi, y, K_max=6, kmax_eval=4), m.think_diag(xi, y, K_max=6, kmax_eval=4, chunk=2)
    for k in full:
        a, b = np.asarray(full[k], dtype=float), np.asarray(part[k], dtype=float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), k
    assert np.allclose(m.tick_losses(xi, y, 5), m.tick_losses(xi, y, 5, chunk=2), rtol=1e-6)   # (fp32 CE)
    a_hat = torch.randn(m.mem.L, 9, m.mem.ain, dtype=torch.float64)
    sel = m.mem.select(a_hat, a_hat, False)
    assert "wn" not in m.mem.online(a_hat, sel)                                 # (eval: no neighbour weights)
    val = np.random.RandomState(0).randint(1, 50, 600).astype(np.uint16)
    p8, p1 = MemoryProbes(val, 16, n_seq=2, n_pos=4, K=4), MemoryProbes(val, 16, n_seq=2, n_pos=4, K=4, batch=1)
    r8, r1 = p8.run(m, "B"), p1.run(m, "B")
    for a, b in zip(r8["reads"], r1["reads"]):
        assert a["rows"].device.type == "cpu" and torch.allclose(a["w"], b["w"], atol=1e-9)
