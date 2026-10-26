"""Tests of the learned memory access gate (plan_v2/22_memory_gate_plan.md revision 2, section 8). CPU only, tiny configs.

    python -m pytest test_gate.py -q

Flags off == the plan-21 code (plan_v2/baseline_mem_v4), bit for bit (test 1). Test 10 (the write variant) is not here:
--mem-wgate is not implemented in v1 (off; the trainer refuses it).
"""
import copy
import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_memory import _run_main, _tiny_data, mem_loop, tokens

HERE = Path(__file__).resolve().parent
BASE = HERE / "plan_v2" / "baseline_mem_v4"
MODS = ("spiking_lm", "loop_brain", "brain_opt", "neurogenesis", "train_lm", "memory", "concepts")
_BASELINE = None


def baseline():
    """The plan-21 code (plan_v2/baseline_mem_v4) as separate module objects (as test_lifecycle's baseline(): the current
    modules are out of sys.modules while it imports, then put back; no bytecode written)."""
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
            "--mem-write-start 5 --mem-probes 2")
LIFE_ARGS = (MEM_ARGS + " --mem-repair 0.25 --eval-every 10 --mem-life 1 --mem-addr-learn 0 --mem-restart 3 --mem-coarse 6 "
             "--mem-nbr 8 --mem-switch-max 0.45 --mem-ramp 0.25 --neuro-t 0.05,0.3")
GATE_ARGS = " --mem-read0 0 --mem-gate 1 --mem-gate-sched 0.25,0.75 --mem-gate-quant 64"


def _same_recs(ra, rb):
    import json
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
        else:
            assert repr(a) == repr(b), k


def gmodel(dtype=torch.float64, quant=8, **kw):
    """A tiny gated memory LoopBrain (test_memory.mem_loop + --mem-gate, read0 off)."""
    kw = dict(dict(mem_gate=1, mem_read0=0), **kw)
    m = mem_loop(dtype, **kw)
    if m.mem_gate is not None:
        m.gate_quant = quant
    return m


def _live(m, seed=11, scale=0.3):
    """Make every memory path matter (zero-init W_o, W_ms, W_p, write-backs otherwise)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mem = m.mem
        for t in [mem.w_o, mem.w_ms.weight] + ([mem.w_p.weight] if mem.R else []) + (
                [mem.w_o_c] if mem.Sc else []):
            t.copy_(torch.randn(t.shape, generator=g, dtype=t.dtype) * scale)
        if m.hier.inj_b is not None:
            m.hier.inj_b.copy_(torch.randn(m.hier.inj_b.shape, generator=g, dtype=m.hier.inj_b.dtype) * scale)


def brain_step(m, opt, x, K=4, offset=0, lr=3e-3, gate_opt=None):
    """test_memory.train_step with the trainer's gate handling: the global clip over the brain only, the router's own clip
    and AdamW. Returns (total, the non-router gradients before the clip, G buffers)."""
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = K, max(0, K - 2), offset
    _, loss = m(x[:, :-1], x[:, 1:])
    total = loss + m.aux_loss()
    opt.zero_grad()
    for p in m.mem_gate.parameters() if m.mem_gate is not None else []:
        p.grad = None
    total.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None and not n.startswith("mem_gate.")}
    Gs = [t.clone() for t in (m.mem.G, getattr(m.mem, "G_e", None), getattr(m.mem, "G_P", None)) if t is not None]
    torch.nn.utils.clip_grad_norm_([p for n, p in m.named_parameters() if not n.startswith("mem_gate.")], 1.0)
    if gate_opt is not None:
        torch.nn.utils.clip_grad_norm_(list(m.mem_gate.parameters()), 1.0)
        gate_opt.step()
    opt.step()
    m.blocks[0].mlp.apply_balance()
    m.mem.after_step(lr)
    return loss.detach(), total.detach(), grads, Gs


# ---------------------------------------------------------------- test 1: flags off == plan 21 (baseline_mem_v4)
def test_01a_flags_off_bit_identical_memory_step_with_growth_balancing_freeze():
    from test_speed import _check, _drive
    old = _drive(baseline(), new=True)
    new = _drive(current(), new=True)
    _check(old, new, exact=True)


def test_01b_flags_off_bit_identical_40_steps_without_memory():
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


def test_01c_flags_off_bit_identical_end_to_end_with_plan21_memory(tmp_path):
    """train_lm.main with every plan-21 flag on (life cycle, restart, coarse, neighbours, probes, repair, growth) and the
    gate flags at their defaults: the old file vs the new file, metrics and final.pt identical."""
    data = _tiny_data(tmp_path / "data")
    extra = (LIFE_ARGS + " --mem-readdr-every 2 --mem-readdr-start 1").split()
    _, ra, ca = _run_main(baseline().train_lm, tmp_path / "runs_old", "old", data, extra)
    _, rb, cb = _run_main(current().train_lm, tmp_path / "runs_new", "new", data, extra)
    assert len(ra) == 5
    _same_recs(ra, rb)
    assert ca["cfg"] == cb["cfg"] and not any(k.startswith("mem_gate") for k in cb["cfg"])
    _same_state(ca["state"], cb["state"])


# ---------------------------------------------------------------- test 2: gate on, c = 1 == plan 21
@pytest.mark.parametrize("ckpt", [True, False])
def test_02_gate_on_c1_bit_identical_forward_gradients_and_G(ckpt):
    """c = 1 (infancy): the forward, every non-router gradient, G / G_e / G_P and the trajectory over 4 optimizer steps
    are bit-identical to the ungated model (incl. the read-out reorder); the router alone gets a gradient (from L_r)."""
    from brain_opt import BrainOpt
    kw = dict(mem_coarse=5, mem_nbr=8, mem_life=1, mem_addr_learn=0, mem_restart=3, mem_read0=0)
    m0, m1 = mem_loop(**kw), gmodel(**kw)
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    assert [k for k in sd1 if k not in sd0] and all(k.startswith("mem_gate.") for k in sd1 if k not in sd0)
    _same_state(sd0, {k: sd1[k] for k in sd0})
    assert {k for k in m1.cfg if k not in m0.cfg} == {"mem_gate"}
    for m in (m0, m1):
        _live(m)
        m.mem_ckpt = m.mem.ckpt = ckpt
    o0, o1 = BrainOpt(m0, lr=3e-3), BrainOpt(m1, lr=3e-3)
    assert not any(n.startswith("mem_gate.") for n in o1.groups) and list(o0.groups) == list(o1.groups)
    go = torch.optim.AdamW(m1.mem_gate.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    x = tokens(2, 33)
    for i, K in enumerate((4, 5, 3, 4)):
        a = brain_step(m0, o0, x, K=K, offset=i % 2)
        b = brain_step(m1, o1, x, K=K, offset=i % 2, gate_opt=go)
        assert torch.equal(a[0], b[0]), i                                      # the LM loss
        assert a[2].keys() == b[2].keys()
        for n in a[2]:
            assert torch.equal(a[2][n], b[2][n]), (i, n)
        for ga, gb in zip(a[3], b[3]):
            assert torch.equal(ga, gb), i
        assert m1.gate_loss is not None and float(m1.gate_loss) > 0
        assert all(st["c"] == 1.0 and "n_c" not in st for st in m1.gate_stats)
    assert m1.mem_gate.weight.grad is not None and m1.mem_gate.weight.grad.abs().sum() > 0
    assert not bool(m1.mem_gate.theta_ok.any())                                   # theta = -inf while c = 1
    sd1 = m1.state_dict()
    _same_state(m0.state_dict(), {k: sd1[k] for k in sd0})
    m0.eval(), m1.eval()
    with torch.no_grad():
        assert torch.equal(m0(x), m1(x))                                          # eval: theta = -inf -> all open


# ---------------------------------------------------------------- test 3: the BCE router learns a planted ranking
def _plant(m, x, marker, beta=4.0):
    """The content gates of every (level, head) (and the coarse pool) made a sharp function of the thought pbar: a
    least-squares direction that separates the marker positions at the first read (so g_bar is high there)."""
    mem = m.mem
    cap = []
    prep = mem.prep
    mem.prep = lambda mus, pbar: (cap.append(pbar.detach().flatten(0, -2)), prep(mus, pbar))[1]
    m.eval()
    with torch.no_grad():
        m.K = 2
        m(x)
    mem.prep = prep
    pb = cap[0].double()
    y = (x.flatten() == marker).double() * 2 - 1
    A = torch.cat([pb, torch.ones(len(pb), 1, dtype=pb.dtype)], 1)
    sol = torch.linalg.solve(A.t() @ A + 1e-3 * torch.eye(A.size(1), dtype=A.dtype), A.t() @ y)
    with torch.no_grad():
        for w, b in [(mem.gates.w, mem.gates.b)] + ([(mem.gates.w_c, mem.gates.b_c)] if mem.Sc else []):
            w[..., :mem.ds] = (beta * sol[:-1]).to(w.dtype)
            w[..., mem.ds:] = 0
            b.fill_(float(beta * sol[-1]))


def _batch(g, B, T, marker, p=0.15):
    x = torch.randint(1, 50, (B, T), generator=g)
    x[torch.rand(B, T, generator=g) < p] = marker
    return x


@pytest.mark.parametrize("c", [1.0, 0.5])
def test_03_bce_router_learns_a_planted_ranking(c):
    """Content gates present (planted on the thought of the first read, then trained normally with W_o; the core is
    held fixed so the planted read's thought does not drift): after training, the router's AUC for the planted (marker)
    positions at the first read is >= 0.9, g_bar there exceeds g_bar elsewhere and s tracks g_bar. With c = 0.5 and the
    floor the router sees only the open sample (Proposition 2): the same."""
    from brain_opt import BrainOpt
    from train_lm import _auc
    torch.manual_seed(0)
    marker = 7
    m = gmodel(torch.float32, quant=8, mem_private=0)
    _live(m, scale=0.1)
    g = torch.Generator().manual_seed(1)
    _plant(m, _batch(g, 32, 32, marker), marker)
    for n, p in m.named_parameters():  # (the thought at the planted read stays put: the core is fixed; the content
        p.requires_grad_(n.startswith(("mem.gates.", "mem.w_o", "mem_gate.")))   # gates and W_o train)
    opt = BrainOpt(m, lr=1e-3)
    go = torch.optim.AdamW(m.mem_gate.parameters(), lr=1e-2, betas=(0.9, 0.99), weight_decay=0.0)
    m.mem_gate_c = c
    for i in range(120):
        xb = _batch(g, 4, 33, marker)
        brain_step(m, opt, xb, K=2, gate_opt=go)
    if c < 1:
        assert all(st["n_c"] + st["n_e"] == st["n_open"] < 128 for st in m.gate_stats)
    xe = _batch(g, 8, 32, marker)
    m.mem_gate.theta_ok.fill_(False)                                # (measured with every position open)
    gb = []
    rs = m.mem.read_shared
    m.mem.read_shared = lambda *a, **k: (lambda out: (gb.append(out[1].detach()), out)[1])(rs(*a, **k))
    m.eval()
    m.K = 2
    with torch.no_grad():
        m(xe)
    m.mem.read_shared = rs
    lab = xe.flatten() == marker
    s = m.gate_open[0]["s"]
    auc = _auc(s, lab)
    gbar = gb[0].float().mean((1, 2))
    assert auc >= 0.9, auc
    assert gbar[lab].mean() > gbar[~lab].mean() + 0.2
    assert torch.corrcoef(torch.stack([s.double(), gbar.double()]))[0, 1] > 0.8     # it predicts g_bar itself


# ---------------------------------------------------------------- test 4: exact capacity, static shapes, no host sync
def test_04_capacity_exact_tie_key_static_and_no_host_sync():
    from memory import MemGate, gate_sizes
    for c, N, q in ((0.5, 8192, 64), (0.3, 1000, 64), (0.97, 512, 8), (0.5, 64, 8)):
        n_c, n_e = gate_sizes(c, 0.1, N, q)
        assert n_c % q == 0 and n_e % q == 0 and n_c + n_e <= N and n_c > 0
    assert gate_sizes(1.0, 0.1, 777, 64) == (777, 0)
    gate = MemGate(4, seed=3)
    g = torch.Generator().manual_seed(0)
    s = torch.randn(1024, generator=g)
    O, isA, tau = gate.select_train(s, 512, 128, gen=torch.Generator().manual_seed(1))
    A = isA.nonzero().squeeze(1)
    assert A.numel() == 512 and O.numel() == 640 and torch.equal(O, O.sort().values) and O.unique().numel() == 640
    assert isA[O].sum() == 512 and torch.equal(s[A].sort().values, s.topk(512).values.sort().values)
    assert tau == s.topk(512).values[-1]
    # a constant router: the tie-key order, not the first positions
    O, isA, _ = gate.select_train(torch.zeros(1024), 256, 0, gen=torch.Generator().manual_seed(1))
    key = gate.tie_key(1024, "cpu")
    assert torch.equal(isA.nonzero().squeeze(1), key.topk(256).indices.sort().values)
    assert not torch.equal(isA.nonzero().squeeze(1), torch.arange(256))
    # in a training forward: |O| identical across reads and batches; the gate's training path never syncs the host
    m = gmodel(quant=8)
    m.mem_gate_c = 0.5
    bad = []

    def boom(name):
        def f(*a, **k):
            if guard[0]:
                bad.append(name)
            return orig[name](*a, **k)
        return f
    names = ("item", "tolist", "__bool__", "__int__", "__float__", "__index__", "nonzero")
    orig = {n: getattr(torch.Tensor, n) for n in names}
    guard = [False]

    def guarded(fn):
        def f(*a, **k):
            guard[0] = True
            try:
                return fn(*a, **k)
            finally:
                guard[0] = False
        return f
    for meth in ("_gate_open", "_gate_after", "_gate_surprise"):
        setattr(m, meth, guarded(getattr(m, meth)))
    try:
        for n in names:
            setattr(torch.Tensor, n, boom(n))
        opens = []
        for seed in (3, 4):
            x = tokens(2, 33, seed=seed)
            m.train()
            m.K, m.aux_k_fixed, m.group_offset = 4, 2, 0
            _, loss = m(x[:, :-1], x[:, 1:])
            (loss + m.aux_loss()).backward()
            opens += [st["n_open"] for st in m.gate_stats]
    finally:
        for n in names:
            setattr(torch.Tensor, n, orig[n])
    assert not bad, bad
    assert len(opens) == 6 and len(set(opens)) == 1 and opens[0] == sum(gate_sizes(0.5, 0.1, 64, 8))


# ---------------------------------------------------------------- test 5: the exploration floor
def test_05_floor_is_uniform_deterministic_and_drawn_once():
    from memory import MemGate
    gate = MemGate(4, seed=0)
    N, n_c, n_e, R = 512, 256, 64, 2000
    s = torch.randn(N, generator=torch.Generator().manual_seed(2))
    gen = torch.Generator().manual_seed(5)
    cnt = torch.zeros(N)
    for _ in range(R):
        O, isA, _ = gate.select_train(s, n_c, n_e, gen=gen)
        cnt[O] += 1
    top = s.topk(n_c).indices
    assert (cnt[top] == R).all()
    rest = torch.ones(N, dtype=torch.bool)
    rest[top] = False
    eps = n_e / (N - n_c)
    se = math.sqrt(eps * (1 - eps) / R)
    f = cnt[rest] / R
    assert ((f - eps).abs() <= 4 * se).all(), (f.min(), f.max(), eps)
    a = gate.select_train(s, n_c, n_e, gen=torch.Generator().manual_seed(9))[0]
    b = gate.select_train(s, n_c, n_e, gen=torch.Generator().manual_seed(9))[0]
    assert torch.equal(a, b)


# ---------------------------------------------------------------- test 6: the EMA updates only on the first run
def test_06_theta_stats_and_generator_change_once_per_training_read_only():
    from memory import MemoryProbes
    m = gmodel(quant=8)
    _live(m)
    gate = m.mem_gate
    m.mem_gate_c = 0.5
    calls = {"theta": 0, "stats": 0}
    ut, su = gate.update_theta, gate.stats_update
    gate.update_theta = lambda *a: (calls.__setitem__("theta", calls["theta"] + 1), ut(*a))[1]
    gate.stats_update = lambda *a: (calls.__setitem__("stats", calls["stats"] + 1), su(*a))[1]
    x = tokens(2, 33)
    snap = lambda: ([t.clone() for t in (gate.theta, gate.theta_ok, gate.fm, gate.fv, gate.fn)],
                    None if gate._gen is None else gate._gen.get_state().clone())
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 4, 2, 0                 # reads after passes 0 (no_grad), 1, 2 (checkpointed)
    _, loss = m(x[:, :-1], x[:, 1:])
    assert calls == {"theta": 3, "stats": 3}
    N = 64
    ref = torch.Generator().manual_seed((gate.seed + 3141) % (2 ** 63))
    for _ in range(3):
        torch.rand(N, generator=ref)
    assert torch.equal(gate._gen.get_state(), ref.get_state())   # one exploration draw per read
    assert bool(gate.theta_ok[:3].all()) and not bool(gate.theta_ok[3:].any()) and int(gate.fn) == 3
    s0 = snap()
    (loss + m.aux_loss()).backward()                             # the checkpoint recompute: no update
    assert calls == {"theta": 3, "stats": 3}

    def same(a, b):
        assert all(torch.equal(p, q) for p, q in zip(a[0], b[0])) and torch.equal(a[1], b[1])
    same(s0, snap())
    m.eval()
    with torch.no_grad():
        m(x)                                                     # eval
        m.mem.ko_shared = True
        m(x)                                                     # a knockout
        m.mem.ko_shared = False
    m.halt_eval(x[:, :-1], x[:, 1:], K_max=4)
    val = np.random.RandomState(0).randint(1, 50, 600).astype(np.uint16)
    MemoryProbes(val, 16, n_seq=2, n_pos=4, K=4).measure(m)      # probes
    same(s0, snap())
    assert calls == {"theta": 3, "stats": 3}


# ---------------------------------------------------------------- test 7: theta consistency and causality
@pytest.mark.parametrize("ar", [0.0, 0.95])
def test_07_threshold_tracks_the_training_cut_off(ar):
    """i.i.d. and AR(1)-correlated scores: theta (the EMA of the top-k cut-off) gives an inference open rate within the
    order-statistic bound of c, and the threshold decision agrees with the top-k decision up to that bound."""
    from memory import MemGate
    gate = MemGate(4, thr_ema=0.02, seed=0)
    N, c = 8192, 0.5
    g = torch.Generator().manual_seed(0)

    rng = np.random.RandomState(0)

    def scores():
        e = rng.standard_normal(N)
        if ar:
            z, a, b = np.empty(N), ar, math.sqrt(1 - ar * ar)
            z[0] = e[0]
            for t in range(1, N):
                z[t] = a * z[t - 1] + b * e[t]
            e = z
        return torch.from_numpy(e).float()
    batches = [scores() for _ in range(40)]
    for i in range(400):
        _, _, tau = gate.select_train(batches[i % 40] + 0.0 * i, int(c * N), 0)
        gate.update_theta(0, tau)
    n_eff = N * (1 - ar) / (1 + ar)
    sd_b = math.sqrt(c * (1 - c) / n_eff)                           # one batch's cut-off, in rate units
    sd_t = math.sqrt(0.02 / (2 - 0.02)) * sd_b                      # the EMA's
    thr = gate.thr(0)
    fresh = [scores() for _ in range(10)]
    rate = torch.stack([(s > thr).double().mean() for s in fresh]).mean().item()
    assert abs(rate - c) <= 4 * math.sqrt(sd_t ** 2 + sd_b ** 2 / 10) + 1e-3, (rate, sd_t, sd_b)
    agree = []
    for s in fresh:
        _, isA, _ = gate.select_train(s, int(c * N), 0)
        agree.append(((s > thr) == isA).double().mean().item())
    assert min(agree) >= 1 - 4 * math.sqrt(sd_t ** 2 + sd_b ** 2) - 1e-3, agree


def test_07b_threshold_mode_is_causal():
    m = gmodel()
    _live(m)
    m.eval()
    m.K = 5
    x = tokens(2, 30)
    with torch.no_grad():
        m(x)
    gate = m.mem_gate
    with torch.no_grad():                                           # a threshold per read at its median score
        for r in m.gate_open:
            gate.theta[r["k"]] = r["s"].median()
        gate.theta_ok.fill_(True)
    x2 = x.clone()
    t = 17
    x2[:, t + 1:] = torch.randint(1, 50, x2[:, t + 1:].shape, generator=torch.Generator().manual_seed(4))
    outs, masks = [], []
    for xx in (x, x2):
        with torch.no_grad():
            outs.append(m(xx))
        masks.append([r["mask"].view(2, 30) for r in m.gate_open])
    assert all(0 < mk.float().mean() < 1 for mk in masks[0])        # (the gate is really closing positions)
    assert torch.allclose(outs[0][:, :t + 1], outs[1][:, :t + 1], atol=1e-9)
    assert not torch.allclose(outs[0][:, t + 1:], outs[1][:, t + 1:])
    for a, b in zip(*masks):
        assert torch.equal(a[:, :t + 1], b[:, :t + 1])


# ---------------------------------------------------------------- test 8: checkpointing with a variable open count
def test_08_checkpointed_gated_block_equals_the_plain_one():
    m1 = gmodel(mem_coarse=5, mem_nbr=8)
    _live(m1)
    m1.mem_gate_c = 0.5
    m2 = copy.deepcopy(m1)
    m2.mem_ckpt = m2.mem.ckpt = False
    from brain_opt import BrainOpt
    o1, o2 = BrainOpt(m1, lr=3e-3), BrainOpt(m2, lr=3e-3)
    x = tokens(2, 33)
    for c in (0.5, 0.3, 0.7):                                       # a different n_open on every step
        m1.mem_gate_c = m2.mem_gate_c = c
        a = brain_step(m1, o1, x, K=5)
        b = brain_step(m2, o2, x, K=5)
        assert torch.equal(a[0], b[0]) and torch.allclose(a[1], b[1], atol=1e-12, rtol=0)
        for n in a[2]:
            assert torch.allclose(a[2][n], b[2][n], atol=1e-12, rtol=1e-9), n
        for ga, gb in zip(a[3], b[3]):
            assert torch.equal(ga.ne(0), gb.ne(0)) and torch.allclose(ga, gb, atol=1e-12, rtol=1e-9)
        assert [st["n_open"] for st in m1.gate_stats] == [st["n_open"] for st in m2.gate_stats]
        assert torch.allclose(m1.mem_gate.weight.grad, m2.mem_gate.weight.grad, atol=1e-12)
    assert len({st["n_open"] for st in m1.gate_stats}) == 1
    for (n, p), q in zip(m1.state_dict().items(), m2.state_dict().values()):
        if torch.is_tensor(p):
            assert torch.allclose(p.double(), q.double(), atol=1e-9) if p.is_floating_point() else torch.equal(p, q), n


# ---------------------------------------------------------------- test 9: the features
def test_09_features_surprise_chunked_readout_and_trace():
    from memory import gate_trace
    m = gmodel()
    _live(m)
    x = tokens(2, 33)
    x[0, 10] = 0                                                    # an EOT inside a row
    # training: sigma comes from the pass-0 read-out's CE (the very tensor), shifted by one
    seen, pairs = [], []
    gs, rp = m._gate_surprise, m._readout_pair
    m._gate_surprise = lambda ce0, idx, ctx: (seen.append((ce0, ctx)), gs(ce0, idx, ctx))[1]
    m._readout_pair = lambda xx, tgt: (lambda o: (pairs.append(o), o)[1])(rp(xx, tgt))
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 4, 2, 0
    with torch.no_grad():
        m.mem_gate.fm.fill_(0.75)                                    # (the EMA mean of sigma at this forward)
    m(x[:, :-1], x[:, 1:])
    ce0, ctx = seen[0]
    assert ce0 is pairs[0][0] and len(seen) == 1
    sig, flag = ctx["surp"], ctx["flag"]
    assert flag[:, 0].all() and flag[0, 11] and flag.sum() == 3
    mean = 0.75
    assert torch.equal(sig[~flag], torch.cat([ce0.new_zeros(2, 1), ce0[:, :-1]], 1).detach().to(sig.dtype)[~flag])
    assert (sig[flag] == mean).all()
    # eval: the chunked read-out == the unchunked one; ce[t] = -log p(idx[t + 1])
    xx = torch.randn(2, 32, 32, dtype=torch.float64)
    idx = x[:, :-1]
    m.gate_chunk = 5
    ent, ce = m._gate_readout(xx, idx, True)
    lp = m.readout_lp(xx)
    ent_ref = -(lp.exp() * lp).sum(-1) / math.log(lp.size(-1))
    ce_ref = -lp[:, :-1].gather(-1, idx[:, 1:, None]).squeeze(-1)
    assert torch.allclose(ent, ent_ref, atol=1e-12) and torch.allclose(ce[:, :-1], ce_ref, atol=1e-12)
    # the windowed trace == the recursion up to gamma^W; resets at document starts
    g = torch.Generator().manual_seed(0)
    s = torch.rand(3, 200, generator=g, dtype=torch.float64) * 5
    start = torch.rand(3, 200, generator=g) < 0.03
    start[:, 0] = True
    gamma, W = math.exp(-1 / 8), 24
    tr = gate_trace(s, start, gamma, W)
    rec = torch.empty_like(s)
    for t in range(200):
        rec[:, t] = torch.where(start[:, t], s[:, t], torch.maximum(s[:, t], gamma * rec[:, t - 1] if t else s[:, t]))
    assert (tr <= rec + 1e-12).all() and ((rec - tr) <= gamma ** W * s.max() + 1e-12).all()
    assert torch.equal(tr[start], s[start])                         # a document start: the trace starts anew
    # the router's input: 93 dims with the default features (L = 4)
    assert m.mem_gate.dim == 64 + 4 + 4 + 1 + 2 + 1 + 1 + 16 == 93


# ---------------------------------------------------------------- test 11: the private memory under the mask
def test_11_private_closed_tokens_have_no_output_no_writes_no_counts():
    from memory import Memory
    torch.manual_seed(0)
    mk = lambda: Memory(d=8, ds=64, L=2, H=1, S=8, m=4, km=3, dv=4, R=8, n_exp=4, n_live=4).double()
    mem = mk()
    mem2 = copy.deepcopy(mem)
    N, k, E = 12, 2, 4
    g = torch.Generator().manual_seed(3)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(N)])
    gate = torch.rand(N, k, generator=g, dtype=torch.float64)

    def layout(tok_ok):
        keep = tok_ok[:, None].expand(N, k).flatten()
        flat = idx.flatten()
        order = flat.argsort(stable=True)
        order = order[keep[order]]
        e_s, tok = flat[order], order // k
        counts = torch.bincount(e_s, minlength=E)
        pos = torch.arange(len(order)) - (counts.cumsum(0) - counts)[e_s]
        return e_s, pos, tok, gate.flatten()[order], order, int(counts.max())
    q1 = F.normalize(torch.randn(N, 64, generator=g, dtype=torch.float64), dim=-1).requires_grad_(True)
    qb1 = F.normalize(torch.randn(N, 64, generator=g, dtype=torch.float64), dim=-1)
    om = torch.rand(N, generator=g) < 0.5
    c = torch.randn(N, 8, generator=g, dtype=torch.float64)
    outs = []
    for mm, tok_ok, kw in ((mem, torch.ones(N, dtype=torch.bool), dict(omask=om)), (mem2, om, {})):
        mm.write_on = True
        mm._collect = True
        with torch.no_grad():
            mm.w_p.weight.normal_(0, 0.5, generator=torch.Generator().manual_seed(1))
        e_k, p_k, t_k, g_k, fpos, C = layout(tok_ok)
        out = mm.private(q1, qb1, e_k, p_k, t_k, g_k, fpos, N, k, E, C, **kw)
        (out * c).sum().backward()
        outs.append(out.detach())
    assert (outs[0][~om] == 0).all() and (outs[0][om] != 0).any()
    assert torch.allclose(outs[0][om], outs[1][om], atol=1e-12)
    for n in ("G_e", "G_P"):
        a, b = getattr(mem, n), getattr(mem2, n)
        bits = lambda t: t.view(torch.int64).ne(0).any(1)              # (write_counts' rule: any bit set)
        assert torch.equal(bits(a), bits(b)), n                         # the same rows touched: the open tokens' only
        assert torch.allclose(a, b, atol=1e-12), n
    assert torch.equal(mem.pend_e, mem2.pend_e) and mem.pend_e.sum() == om.sum() * k * 4
    assert mem.gates.b_e.grad is not None and torch.allclose(mem.gates.b_e.grad, mem2.gates.b_e.grad, atol=1e-12)


# ---------------------------------------------------------------- test 12: the plan-21 couplings
def test_12_couplings_norm_stats_usage_probes_freeze_readdr():
    from memory import MemoryProbes
    m1 = gmodel(quant=8)
    m2 = copy.deepcopy(m1)
    m1.mem_gate_c, m2.mem_gate_c = 0.5, 1.0
    x = tokens(2, 33)
    pend = []
    for m in (m1, m2):                                              # (zero-init W_o: the thoughts do not depend on c)
        m.train()
        m.K, m.aux_k_fixed, m.group_offset = 4, 2, 0
        m.mem.begin_forward(True)
        m(x[:, :-1], x[:, 1:])
        pend.append(m.mem.pend.clone())
    mem1, mem2 = m1.mem, m2.mem
    for n in ("ns_sum", "ns_sq", "ns_cnt"):                         # the norm statistics over ALL positions
        assert torch.equal(getattr(mem1, n), getattr(mem2, n)), n
    # usage (sub-key restarts, the reservoir) from the open positions only: the reads inside the gradient window
    L, H, mm = mem1.L, mem1.H, mem1.m
    n_open = [st["n_open"] for st in m1.gate_stats]
    assert n_open[0] < 64
    assert pend[0].sum() == sum(n_open[1:]) * L * H * 2 * mm and pend[1].sum() == 2 * 64 * L * H * 2 * mm
    # probes: every probe position recorded (ungated), plus the gate's decision; the stability over the open ones
    _live(m1)
    m1.eval()
    with torch.no_grad():
        m1.K = 4
        m1(x[:, :-1])
        for r in m1.gate_open:
            m1.mem_gate.theta[r["k"]] = r["s"].median()
        m1.mem_gate.theta_ok.fill_(True)
    val = np.random.RandomState(0).randint(1, 50, 600).astype(np.uint16)
    pr = MemoryProbes(val, 16, n_seq=2, n_pos=4, K=4)
    rec = pr.run(m1, "B")
    assert all(r["rows"].size(0) == 8 and r["open"].shape == (8,) for r in rec["reads"])
    pr.measure(m1)
    met, _ = pr.measure(m1)
    assert "mem_last_stab_w_open" in met and 0 < met["mem_last_probe_open_rate"] < 1
    # the router is never frozen: trainable after the memory's freeze
    mem1.freeze()
    m1.train()
    m1.mem_gate.weight.grad = None
    _, loss = m1(x[:, :-1], x[:, 1:])
    (loss + m1.aux_loss()).backward()
    assert m1.mem_gate.weight.requires_grad and m1.mem_gate.weight.grad.abs().sum() > 0
    # address reuse is off under the gate: every read addresses anew
    m3 = gmodel(quant=8)
    m3.mem_readdr_every, m3.mem_readdr_start = 2, 0
    n_sel = []
    sel = m3.mem.select
    m3.mem.select = lambda *a, **k: (n_sel.append(1), sel(*a, **k))[1]
    m3.eval()
    m3.K = 6
    with torch.no_grad():
        m3(x)
    assert len(n_sel) == 5


# ---------------------------------------------------------------- test 13: resume
class _Stop(Exception):
    pass


def test_13_resume_is_bit_identical_inside_the_cosine(tmp_path, monkeypatch):
    """40 straight steps == 20 + save + resume + 20 (train_lm.main, CPU) with the gate on: the cut at c ~ 0.75 inside the
    cosine (10 .. 30), theta updates on both sides, restart events every 3 steps, probe evals on both sides, the life
    cycle's switch; the router's AdamW state, theta, the feature statistics and the device generator restored."""
    tl = current().train_lm
    data = _tiny_data(tmp_path / "data")
    extra = (LIFE_ARGS + GATE_ARGS + " --ckpt-every 20").split()
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
    ck = torch.load(last, weights_only=False)
    assert ck["gate"]["opt"]["state"] and ck["gate"]["host"]["gen"] is not None
    run_b, rb, cb = _run_main(tl, tmp_path / "runs_b", "b", data, extra + ["--resume", str(last)])
    assert "RESUMED at step 20" in (run_b / "train.log").read_text()
    assert ra[1]["gate_c"] == 1.0 and 0.5 < ra[2]["gate_c"] < 1 and ra[2]["gate_reads"][0]["n_open"] < 1280
    assert all(r["gate_reads"][0]["above"] == r["gate_reads"][0]["above"] for r in ra[2:])   # (theta valid: c < 1)
    assert "mem_restart_last" in ra[2] and "mem_last_stab_w" in ra[3]
    _same_recs(ra, rb)
    assert ca["cfg"] == cb["cfg"] and ca["cfg"]["mem_gate"] == 1
    _same_state(ca["state"], cb["state"])
    assert bool(ca["state"]["mem_gate.theta_ok"].any())


# ---------------------------------------------------------------- test 14: the eval modes and the report
def test_14_eval_modes_rates_capture_and_report():
    import train_lm
    m = gmodel()
    _live(m)
    x = tokens(4, 33)
    m.eval()
    m.K = 4
    with torch.no_grad():
        m(x[:, :-1])
        for r in m.gate_open:
            m.mem_gate.theta[r["k"]] = r["s"].quantile(0.4)
        m.mem_gate.theta_ok.fill_(True)
    rates = {}
    for mode in ("learned", "random", "inverted", "open", "off"):
        m.mem_gate_eval = mode
        with torch.no_grad():
            out = m(x[:, :-1])
        assert torch.isfinite(out).all()
        rates[mode] = [1.0 if r["mask"] is None else r["mask"].float().mean().item() for r in m.gate_open]
    assert rates["open"] == [] and rates["off"] == []
    assert rates["learned"][0] == rates["inverted"][0] and all(0.3 < v < 0.9 for v in rates["learned"])
    assert all(abs(a - b) < 0.25 for a, b in zip(rates["learned"], rates["random"]))
    m.mem_gate_eval = "oracle"
    m.gate_oracle_b = torch.randn(4, 32)
    with torch.no_grad():
        m(x[:, :-1])
    assert [r["mask"].float().mean().item() for r in m.gate_open][0] == rates["learned"][0]
    m.mem_gate_eval, m.gate_oracle_b = "learned", None

    class _Data:
        def batch(self, B, split, gen):
            t = torch.from_numpy(gen.randint(1, 50, (B, 33))).long()
            return t[:, :-1], t[:, 1:]
    args = SimpleNamespace(k_mean=4, eval_batch=4, mem_probe_batch=2, device="cpu")
    rep = train_lm.gate_report(m, _Data(), args, batches=1)
    for mode in ("learned", "random", "inverted", "oracle", "open", "off"):
        assert math.isfinite(rep[f"gate_loss_{mode}"])
    for mode in ("learned", "random", "inverted", "oracle"):
        assert math.isfinite(rep[f"gate_capture_{mode}"]) and 0 < rep[f"gate_open_rate_{mode}"] < 1
    assert abs(rep["gate_open_rate_inverted"] - rep["gate_open_rate_learned"]) < 0.25
    assert 0 <= rep["gate_top20_share"] <= 1 and len(rep["gate_open_by_pass"]) == 3
    assert math.isfinite(rep["gate_ko_open"]) and math.isfinite(rep["gate_ko_closed"])
    assert m.mem_gate_eval == "learned" and m.training
    assert "gate c 0.500" in train_lm.gate_line(dict(gate_c=0.5, gate_reads=[]))
    # the halting eval logs the open rate by exit pass
    he = m.halt_eval(x[:, :-1], x[:, 1:], K_max=4)
    assert len(he["gate_open_by_exit"]) == 4


def test_15_trainer_router_outside_brainopt_and_global_clip(tmp_path):
    """The router is not in BrainOpt / make_opt; its parameters are excluded from the global clip; cfg round trip with
    theta; a short end-to-end gated run logs the gate's training diagnostics."""
    import train_lm
    m = gmodel()
    o = train_lm.make_opt(m, 1e-3, 0.1)
    ids = {id(p) for g in o.param_groups for p in g["params"]}
    assert all(id(p) not in ids for p in m.mem_gate.parameters())
    assert len(ids) == len(list(m.parameters())) - 2
    with torch.no_grad():
        m.mem_gate.theta.fill_(0.3)
        m.mem_gate.theta_ok.fill_(True)
    from loop_brain import LoopBrain
    m2 = LoopBrain(**copy.deepcopy(m.cfg)).double()
    m2.load_state_dict(m.state_dict(), strict=True)
    assert torch.equal(m2.mem_gate.theta, m.mem_gate.theta) and m2.mem_gate.feats == m.mem_gate.feats
    data = _tiny_data(tmp_path / "data")
    extra = (MEM_ARGS + GATE_ARGS + " --mem-gate-sched 0.1,0.3 --eval-every 20").split()
    run, recs, ck = _run_main(train_lm, tmp_path / "runs", "g", data, extra)
    last = recs[-1]
    assert last["gate_c"] == 0.5 and last["gate_reads"][0]["n_c"] == 640 and "gate_capture_learned" in last
    assert "gate c 0.500" in (run / "train.log").read_text()


def test_14b_eval_world_gate_modes_and_type_rates():
    """eval_world.py: the gate's eval modes join the knockout list (model attributes), and the open rate by position
    type is the mean over reads of each typed position."""
    import eval_world
    m = gmodel()
    _live(m)
    names = [n for n, _, _ in eval_world.knockouts(m)]
    assert {"gate_open", "gate_off", "gate_random", "gate_inverted"} <= set(names)
    assert not any(n.startswith("gate_") for n, _, _ in eval_world.knockouts(m, gate_modes=False))
    with eval_world.KO(m, "model.mem_gate_eval", "off"):
        assert m.mem_gate_eval == "off"
    assert m.mem_gate_eval == "learned"
    m.eval()
    m.K = 4
    with torch.no_grad():
        m(tokens(2, 20)[:, :-1])
        for r in m.gate_open:
            m.mem_gate.theta[r["k"]] = r["s"].median()
        m.mem_gate.theta_ok.fill_(True)
    a = SimpleNamespace(batch=4, device="cpu")
    seqs = [tokens(1, n, seed=n)[0].tolist() for n in (12, 15, 9)]
    typed = [(sq, {len(sq) - 3: "pre_attribute", 1: "other"}) for sq in seqs]
    out = eval_world.gate_type_rates(m, typed, a, eot=0)
    assert set(out) == {"pre_attribute", "other"} and all(v["n"] == 3 and 0 <= v["rate"] <= 1 for v in out.values())
    assert len(out["other"]["by_pass"]) == 3


def test_14c_eval_read_with_every_position_closed():
    """Eval: a read whose threshold closes every position gives no memory output at all (no shared, no private)."""
    m = gmodel()
    _live(m)
    m.eval()
    m.K = 4
    x = tokens(2, 20)[:, :-1]
    with torch.no_grad():
        m.mem_gate.theta.fill_(1e9)
        m.mem_gate.theta_ok.fill_(True)
        a = m(x)
        assert all(r["mask"] is not None and not bool(r["mask"].any()) for r in m.gate_open)
        m.mem_gate_eval = "off"
        b = m(x)
    assert torch.equal(a, b)
