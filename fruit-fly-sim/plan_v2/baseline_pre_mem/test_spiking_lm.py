"""Unit tests for spiking_lm.py (run: python -m pytest fruit-fly-sim/test_spiking_lm.py -q)."""
import copy
import math

import pytest
from pathlib import Path

import torch
import torch.nn.functional as F

from spiking_lm import SpikingLM, carry_optimizer, snapshot_optimizer, ssd_naive, ssd_scan


def rand_inputs(B=2, H=3, T=150, N=16, P=8, decay="mixed", dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, N, generator=g, dtype=dtype)
    k = torch.randn(B, H, T, N, generator=g, dtype=dtype)
    v = torch.randn(B, H, T, P, generator=g, dtype=dtype)
    if decay == "near1":
        la = -torch.rand(B, H, T, generator=g, dtype=dtype) * 1e-4
    elif decay == "near0":
        la = -10 - torch.rand(B, H, T, generator=g, dtype=dtype) * 20
    else:
        la = -torch.exp(torch.randn(B, H, T, generator=g, dtype=dtype) * 2)
    return q, k, v, la


@pytest.mark.parametrize("decay", ["mixed", "near1", "near0"])
@pytest.mark.parametrize("chunk", [16, 64])
def test_chunked_scan_matches_recurrence(decay, chunk):
    q, k, v, la = rand_inputs(decay=decay)
    y_ref, S_ref = ssd_naive(q, k, v, la)
    y, S = ssd_scan(q, k, v, la, chunk)
    assert torch.allclose(y, y_ref, rtol=1e-9, atol=1e-9)
    assert torch.allclose(S, S_ref, rtol=1e-9, atol=1e-9)


def test_chunked_scan_fp32_and_initial_state():
    q, k, v, la = rand_inputs(decay="mixed")
    S0 = torch.randn(2, 3, 16, 8, dtype=torch.float64)
    y_ref, _ = ssd_naive(q, k, v, la, S0)
    y, _ = ssd_scan(*(t.float() for t in (q, k, v, la)), 64, S0.float())
    err = ((y.double() - y_ref).abs().max() / y_ref.abs().max()).item()
    assert err < 1e-4


def test_scan_gradcheck():
    q, k, v, la = (t.requires_grad_() for t in rand_inputs(B=1, H=1, T=20, N=3, P=2))
    assert torch.autograd.gradcheck(lambda *a: ssd_scan(*a, chunk=8)[0], (q, k, v, la))


def tiny(mixer="ssd", spike="analog", **kw):
    torch.manual_seed(0)
    cfg = dict(vocab=50, d=32, layers=2, heads=2, head_dim=8, mlp=48, mixer=mixer, spike=spike, chunk=16)
    cfg.update(kw)
    return SpikingLM(**cfg).double()


@pytest.mark.parametrize("mixer", ["ssd", "attn"])
def test_causal(mixer):
    m = tiny(mixer)
    x = torch.randint(0, 50, (2, 40))
    full = m(x)
    part = m(x[:, :25])
    assert torch.allclose(full[:, :25], part, atol=1e-10)


@pytest.mark.parametrize("mixer", ["ssd", "attn"])
@pytest.mark.parametrize("spike", ["analog", "graded", "ternary", "binary"])
@pytest.mark.parametrize("op", ["heads", "mlp", "layers"])
def test_growth_preserves_function(mixer, spike, op):
    m = tiny(mixer, spike)
    x = torch.randint(0, 50, (2, 40))
    before = m(x)
    getattr(m, f"grow_{op}")(2 if op != "mlp" else 16)
    m.double()
    after = m(x)
    assert (after - before).abs().max().item() < 1e-10


def train_steps(m, opt, x, n=3):
    for _ in range(n):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


@pytest.mark.parametrize("mixer", ["ssd", "attn"])
def test_optimizer_carry_after_growth(mixer):
    m = tiny(mixer, "graded")
    x = torch.randint(0, 50, (4, 33))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    train_steps(m, opt, x)
    snap = snapshot_optimizer(m, opt)
    old_up = copy.deepcopy(snap["blocks.0.mlp.up.weight"])
    m.grow_mlp(16)
    m.grow_heads(1)
    m.grow_layers(1)
    m.double()
    opt2 = carry_optimizer(m, snap, torch.optim.AdamW(m.parameters(), lr=1e-3))
    st = opt2.state[m.blocks[0].mlp.up.weight]
    n_old = old_up["exp_avg"].shape[0]
    # old slice kept, new slice: zero momentum, median second moment
    assert torch.equal(st["exp_avg"][:n_old], old_up["exp_avg"])
    assert st["exp_avg"][n_old:].abs().max() == 0
    assert torch.allclose(st["exp_avg_sq"][n_old:], old_up["exp_avg_sq"].median().expand_as(st["exp_avg_sq"][n_old:]))
    # the first step after growth must not blow up the new weights
    w_new = m.blocks[0].mlp.up.weight[n_old:].detach().clone()
    loss = train_steps(m, opt2, x, n=1)
    step = (m.blocks[0].mlp.up.weight[n_old:].detach() - w_new).abs().max().item()
    assert step <= 3e-3  # about lr, not 3x lr
    assert loss == loss  # no NaN


@pytest.mark.parametrize("mixer", ["ssd", "attn"])
@pytest.mark.parametrize("init", ["fresh", "copy"])
def test_spread_layer_growth_preserves_function_and_optimizer(mixer, init):
    from spiking_lm import remap_snapshot
    m = tiny(mixer, "graded", layers=3)
    x = torch.randint(0, 50, (4, 33))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    train_steps(m, opt, x)
    m.eval()
    before = m(x)
    snap = snapshot_optimizer(m, opt)
    top = snap["blocks.2.mix.wo.weight"]["exp_avg"].clone()
    mapping = m.grow_layers(2, "spread", init)
    m.double()
    assert len(m.blocks) == 5 and mapping[2] == 4
    assert (m(x) - before).abs().max().item() < 1e-10
    opt2 = carry_optimizer(m, remap_snapshot(snap, mapping), torch.optim.AdamW(m.parameters(), lr=1e-3))
    assert torch.equal(opt2.state[m.blocks[4].mix.wo.weight]["exp_avg"], top)
    m.train()
    loss = train_steps(m, opt2, x, 1)
    assert loss == loss  # trains on after growth, no NaN


def test_redo_preserves_function_and_homeostasis_moves_thresholds():
    from spiking_lm import redo_dead
    m = tiny("ssd", "graded")
    m.set_homeo(0.05, 0.15)
    x = torch.randint(0, 50, (4, 33))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sp = m.blocks[0].mlp.spike
    train_steps(m, opt, x, 1)  # calibrates
    a0 = sp.alpha.detach().clone()
    sp.alpha.data[:5] = 1e3  # make 5 units silent
    for _ in range(20):
        train_steps(m, opt, x, 1)
    assert (sp.alpha[:5] < 1e3).all()          # homeostasis lowers the threshold of silent units
    sp.ema_rate[:3] = 0.0                     # pretend 3 units are dead
    m.eval()
    with torch.no_grad():  # make units 0-2 contribute nothing, as a truly silent unit would
        m.blocks[0].mlp.down.weight[:, :3] = 0
    before = m(x)
    assert redo_dead(m, opt) >= 3
    assert (m(x) - before).abs().max().item() < 1e-10
    assert a0.shape == sp.alpha.shape


def test_ngc_scale_invariant_and_leaky():
    from spiking_lm import Spike
    torch.manual_seed(0)
    sp = Spike(16, "ngc", signed=False).double()
    x = torch.randn(64, 16, dtype=torch.float64)
    sp.train()
    sp(x)  # calibrates
    sp.eval()
    q1 = (sp(x) != 0)
    q2 = (sp(x * 10) != 0)
    assert torch.equal(q1, q2)  # firing pattern does not depend on the overall signal scale
    xs = (x * 100).requires_grad_()  # very large inputs saturate at 7 spikes...
    sp(xs).sum().backward()
    assert xs.grad.abs().sum() > 0  # ...but still get a (leaky) gradient


@pytest.mark.parametrize("mixer", ["ssd", "attn"])
@pytest.mark.parametrize("op", ["heads", "mlp", "layers"])
def test_ngc_growth_preserves_function(mixer, op):
    m = tiny(mixer, "ngc")
    x = torch.randint(0, 50, (2, 40))
    before = m(x)
    getattr(m, f"grow_{op}")(2 if op != "mlp" else 16)
    m.double()
    assert (m(x) - before).abs().max().item() < 1e-6  # float rounding of the weighted mean only


MOE = dict(n_exp=8, k=2, hidden=12, shared=16, capacity=100.0)


def test_routed_matches_dense_reference_and_is_causal():
    torch.manual_seed(0)
    m = tiny("ssd", "analog", moe=MOE)
    x = torch.randint(0, 50, (2, 40))
    mlp = m.blocks[0].mlp.eval()
    h = torch.randn(2, 40, 32, dtype=torch.float64)
    assert torch.allclose(mlp(h), mlp.dense_reference(h), atol=1e-10)
    m.eval()
    assert torch.allclose(m(x)[:, :25], m(x[:, :25]), atol=1e-10)  # routing is per token, so still causal


def test_routed_trains_and_balances():
    torch.manual_seed(0)
    m = tiny("ssd", "analog", moe=dict(MOE, capacity=1.25))
    x = torch.randint(0, 50, (8, 65))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(30):
        _, loss = m(x[:, :-1], x[:, 1:])
        loss = loss + m.aux_loss()
        opt.zero_grad()
        loss.backward()
        opt.step()
    st = m.routing_stats()
    assert len(st) == 2 and all(s["entropy"] > 0.8 for s in st), st
    assert m.blocks[0].mlp.bias.abs().sum() > 0  # the balancer moved
    assert m.n_active() < n_params_all(m)


def n_params_all(m):
    from spiking_lm import n_params
    return n_params(m)


@pytest.mark.parametrize("muon", [True, False])
def test_brainopt_trains_lazy_and_carries_state_through_growth(muon):
    from brain_opt import BrainOpt, carry_brain_state
    torch.manual_seed(0)
    m = tiny("ssd", "graded", moe=dict(MOE, capacity=1.25)).float()
    x = torch.randint(0, 50, (4, 33))
    opt = BrainOpt(m, lr=3e-3, muon=muon, lazy=True, maturity=4, tau=5)
    losses = []
    for _ in range(25):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        (loss + m.aux_loss()).backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.3
    # lazy: an expert that gets no gradient is not changed (no weight decay while it sleeps)
    w = m.blocks[0].mlp.w_up
    before = w.detach().clone()
    opt.zero_grad()
    w.grad = torch.zeros_like(w)
    w.grad[0] = 1e-3
    opt.step()
    assert torch.equal(w[1:], before[1:]) and not torch.equal(w[0], before[0])
    # growth: state is padded, new elements are born now
    m2 = tiny("ssd", "graded", moe=dict(MOE, capacity=1.25)).float()
    m2.load_state_dict(m.state_dict())
    m2.grow_heads(1)
    opt2 = carry_brain_state(opt, m2, BrainOpt(m2, lr=3e-3, muon=muon, maturity=4, tau=5))
    st = opt2.state["blocks.0.mix.wq.weight"]
    assert st["born"][-1].min() == opt.t and st["born"][0].max() == 0
    _, loss = m2(x[:, :-1], x[:, 1:])
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    assert math.isfinite(loss.item())


@pytest.mark.parametrize("spike", ["graded", "ngc"])
@pytest.mark.parametrize("where", ["grad", "uniform"])
def test_neurogenesis_births_are_exact_and_reach_target(spike, where):
    from brain_opt import BrainOpt, carry_brain_state
    from neurogenesis import Neurogenesis, reset_opt_rows
    torch.manual_seed(0)
    m = tiny("ssd", spike, heads=1, mlp=16).float()
    final = dict(layers=2, heads=2, mlp=48)
    ng = Neurogenesis(m, dict(layers=2, heads=1, mlp=16), final, steps=400, every=5, where=where,
                      renew_rate=0.01, renew_cap=1, mature=50, ramp=20)
    opt = BrainOpt(m, lr=3e-3, maturity=4, tau=20)
    x = torch.randint(0, 50, (4, 33))
    for step in range(400):
        if step % 5 == 0:
            ng.ramps()
            m.eval()
            before = m(x[:, :-1])
            changed = ng.births_step(step)
            assert (m(x[:, :-1]) - before).abs().max().item() < 1e-5  # births never change the output
            m.train()
            ng.renew(step)
            if changed:
                opt = carry_brain_state(opt, m, BrainOpt(m, lr=3e-3, maturity=4, tau=20))
            for i, idx in ng.pop_renewed():
                reset_opt_rows(opt, m, i, idx)
        else:
            ng.step(step)
        scoring = ng.want_score(step)
        if scoring:
            ng.attach()
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        if scoring:
            ng.detach_and_score()
        opt.step()
    assert [b.mlp.up.out_features for b in m.blocks] == [48, 48]
    assert [b.mix.h for b in m.blocks] == [2, 2]
    assert ng.renewals > 0 and math.isfinite(loss.item())


def test_expert_birth_is_exact_and_spiking_experts_train():
    from brain_opt import BrainOpt, carry_brain_state
    torch.manual_seed(0)
    m = tiny("ssd", "graded", moe=dict(MOE, capacity=1.25)).float()
    m.set_homeo(0.01, 0.15, "floor")
    x = torch.randint(0, 50, (8, 33))
    opt = BrainOpt(m, lr=3e-3)
    for _ in range(10):  # calibrates the expert spikes on routed tokens only
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        (loss + m.aux_loss()).backward()
        opt.step()
    sp = m.blocks[0].mlp.spike_e
    assert sp.calibrated and torch.isfinite(sp.alpha).all()
    m.eval()
    before = m(x[:, :-1])
    for b in m.blocks:
        b.mlp.grow_experts(3)
    assert (m(x[:, :-1]) - before).abs().max().item() < 1e-6  # newborns are not selected at birth
    m.train()
    opt = carry_brain_state(opt, m, BrainOpt(m, lr=3e-3))
    for _ in range(5):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        (loss + m.aux_loss()).backward()
        opt.step()
    assert m.blocks[0].mlp.E == 11 and math.isfinite(loss.item())


def test_neurogenesis_grows_experts_and_heads():
    from neurogenesis import Neurogenesis
    torch.manual_seed(0)
    m = tiny("ssd", "graded", heads=1, moe=dict(MOE, n_exp=4, capacity=1.25)).float()
    ng = Neurogenesis(m, dict(layers=2, heads=1, mlp=48, experts=4), dict(layers=2, heads=2, mlp=48, experts=8),
                      steps=200, every=5)
    for step in range(200):
        ng.step(step)
    assert [b.mlp.E for b in m.blocks] == [8, 8] and [b.mix.h for b in m.blocks] == [2, 2]
    assert m.cfg["moe"]["n_exp"] == 8 and ng.births == 8
    x = torch.randint(0, 50, (2, 33))
    _, loss = m(x[:, :-1], x[:, 1:])
    assert math.isfinite(loss.item())


def test_ramp_birth_newborns_get_adopted_without_bias_cascade():
    torch.manual_seed(0)
    m = tiny("ssd", "analog", moe=dict(MOE, n_exp=8, k=2, capacity=1.25)).float()
    m.blocks[0].mlp.u = 1e-2
    x = torch.randint(0, 50, (8, 33))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    mlp = m.blocks[0].mlp

    def train(n):
        for _ in range(n):
            _, loss = m(x[:, :-1], x[:, 1:])
            opt.zero_grad()
            (loss + m.aux_loss()).backward()
            opt.step()

    train(20)
    for _ in range(4):  # repeated births: must not keep cloning the same (or a newborn) parent
        mlp.grow_experts(1)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        train(30)
    train(1050)  # ramps finish (2.0 / 0.002 = 1000 steps)
    seen = torch.zeros_like(mlp.bias)
    for _ in range(50):
        train(1)
        seen += mlp.stats[0].float()
    assert mlp.debt.max() == 0
    assert (mlp.bias[8:] > mlp.bias[:8].min() - 1.0).all(), mlp.bias  # no cascade of sinking biases
    assert (seen[8:] > 0).all(), seen                                  # every newborn is used

def _tiny_concept_tok(tmp_path, train_steps=150):
    from concepts import ConceptTokenizer
    torch.manual_seed(0)
    tok = ConceptTokenizer(vocab=50, d=32, chunk=8, codebook=32, heads=2)
    opt = torch.optim.AdamW(tok.parameters(), lr=3e-3)
    x = torch.randint(0, 50, (64, 8))
    for _ in range(train_steps):
        ce, vq, _, _ = tok(x, nested_p=0.5)
        opt.zero_grad()
        (ce + vq).backward()
        opt.step()
    path = tmp_path / "concept_tok.pt"
    torch.save({"cfg": dict(vocab=50, d=32, chunk=8, codebook=32, heads=2), "state": tok.state_dict()}, path)
    return tok, x, path


def test_concept_tokenizer_learns_and_levels_refine(tmp_path):
    tok, x, _ = _tiny_concept_tok(tmp_path)
    tok.eval()
    with torch.no_grad():
        z = tok.encode_latent(x)
        accs = [(tok.decode(tok.quantize(z, k)[1]).argmax(-1) == x).float().mean().item() for k in (1, 4)]
    assert accs[1] > 0.6 and accs[1] >= accs[0], accs   # more levels -> better reconstruction
    codes = tok.codes(x, 4)
    assert [c.shape[1] for c in codes] == [1, 2, 4, 8]


def test_concept_brain_is_causal_starts_silent_and_trains(tmp_path):
    _, _, path = _tiny_concept_tok(tmp_path, train_steps=20)
    torch.manual_seed(0)
    m = tiny("ssd", "graded", layers=3, concept=dict(path=str(path), levels=2, split=1)).float()
    plain = tiny("ssd", "graded", layers=3).float()
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if k in plain.state_dict()})
    x = torch.randint(0, 50, (2, 33))
    m.eval()
    plain.eval()
    assert torch.allclose(m(x[:, :-1]), plain(x[:, :-1]), atol=1e-6)        # plan_proj starts at zero
    with torch.no_grad():
        m.plan_proj.weight.normal_(0, 0.5)                                   # make the plan matter
    full = m(x[:, :-1])
    assert torch.allclose(full[:, :20], m(x[:, :20]), atol=1e-5)             # still causal (partial chunk too)
    m.train()
    _, loss = m(x[:, :-1], x[:, 1:])
    assert m.concept_loss is not None and math.isfinite(m.concept_loss.item())
    (loss + m.concept_loss).backward()
    assert m.concept_head.out[0].weight.grad.abs().sum() > 0
    assert m.blocks[0].mix.wq.weight.grad.abs().sum() > 0                     # the plan trains lower layers
    assert all(p.grad is None for p in m.concept_tok.parameters())           # tokenizer stays frozen


def tiny_loop(**kw):
    from loop_brain import LoopBrain
    torch.manual_seed(0)
    cfg = dict(vocab=50, d=32, d_s=8, heads=2, head_dim=8, spike="graded", chunk=16, k_mean=4, k_min=2,
               k_max=6, bptt=3, state="carry", refractory=1,
               moe=dict(n_exp=12, k=3, hidden=16, shared=16, capacity=0))
    cfg.update(kw)
    return LoopBrain(**cfg).double()


def test_loop_brain_causal_refractory_and_gradients():
    m = tiny_loop()
    x = torch.randint(0, 50, (2, 40))
    m.eval()
    assert torch.allclose(m(x)[:, :25], m(x[:, :25]), atol=1e-9)        # causal across tokens
    # refractory: record each pass's choices; consecutive passes never share an expert for a token
    core, picks = m.blocks[0], []
    h = core.mlp.register_forward_hook(lambda mod, a, o: picks.append(mod.last_idx.clone()))
    m.K = 5
    m(x)
    h.remove()
    for a, b in zip(picks, picks[1:]):
        assert not (a[:, :, None] == b[:, None, :]).any()
    assert core.mlp.stats[1].item() == 0                                # dropless
    m.train()
    m.K = 3
    _, loss = m(x[:, :-1], x[:, 1:])
    loss.backward()
    assert core.w_z.weight.grad.abs().sum() > 0                          # memory is trained
    assert core.mlp.router.weight.grad.abs().sum() > 0
    assert core.mlp.pending.sum() > 0
    bias0 = core.mlp.bias.clone()
    core.mlp.apply_balance()
    assert not torch.equal(bias0, core.mlp.bias) and core.mlp.pending.sum() == 0


def test_loop_brain_variants_and_diagnostics():
    x = torch.randint(0, 50, (2, 33))
    for state in ("carry", "reset", ""):
        m = tiny_loop(state=state)
        m.eval()
        out = m(x[:, :-1])
        assert torch.isfinite(out).all()
        tl = m.tick_losses(x[:, :-1], x[:, 1:], 4)
        assert len(tl) == 4 and all(math.isfinite(v) for v in tl)
    assert 0 <= m.diag["overlap_k_k2"] <= 1 and m.diag["distinct_experts"] >= 3
    m = tiny_loop(refractory=0)
    m.eval()
    m(x[:, :-1])


@pytest.mark.parametrize("opts", [dict(waves=True), dict(slots=4), dict(pops=3), dict(waves=True, slots=4, pops=3)])
def test_loop_brain_waves_workspace_pops(opts):
    m = tiny_loop(heads=4, **opts)
    x = torch.randint(0, 50, (2, 40))
    m.eval()
    m.K = 4
    full = m(x)
    assert torch.allclose(full[:, :25], m(x[:, :25]), atol=1e-9)        # still causal
    core = m.blocks[0]
    if opts.get("slots"):
        base = m(x)
        m.knockout = "last"
        ko_last = m(x)
        m.knockout = "all"
        ko_all = m(x)
        m.knockout = None
        assert not torch.allclose(base, ko_last) and not torch.allclose(base, ko_all)
    if opts.get("waves"):
        assert "wave_travel_cells" in m.diag and m.diag["wave_gamma"] > 0
    if opts.get("pops"):
        assert torch.allclose(core.mix.last_load.sum(), torch.tensor(2.0, dtype=core.mix.last_load.dtype))
    m.train()
    m.K = 3
    _, loss = m(x[:, :-1], x[:, 1:])
    (loss + m.aux_loss()).backward()
    if opts.get("waves"):
        assert core.router_conn.grad is not None and core.router_conn.grad.abs().sum() > 0
        assert core.wave_gamma.grad is not None
    if opts.get("slots"):
        assert core.w_wv.weight.grad.abs().sum() > 0 and core.w_ro.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("opts", [dict(waves=True), dict(waves=True, slots=4, pops=3)])
def test_loop_brain_trains_with_brainopt(opts):
    """The real training path: BrainOpt (Muon + lazy AdamW) incl. scalar parameters (wave strengths)."""
    from brain_opt import BrainOpt
    m = tiny_loop(heads=4, **opts).float()
    opt = BrainOpt(m, lr=3e-3)
    x = torch.randint(0, 50, (2, 33))
    g0 = m.blocks[0].wave_gamma.detach().clone()
    for _ in range(3):
        m.K = 3
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        (loss + m.aux_loss()).backward()
        opt.step()
        m.blocks[0].mlp.apply_balance()
    assert math.isfinite(loss.item()) and not torch.equal(g0, m.blocks[0].wave_gamma.detach())


def test_workspace_v2_direct_plus_workspace_trains_with_brainopt():
    from brain_opt import BrainOpt
    m = tiny_loop(heads=4, waves=True, slots=4, ws_direct=True).float()
    core = m.blocks[0]
    assert torch.allclose(torch.sigmoid(core.ws_mix), torch.full_like(core.ws_mix, 0.1))  # starts 90% direct
    opt = BrainOpt(m, lr=3e-3)
    x = torch.randint(0, 50, (2, 33))
    mix0 = core.ws_mix.detach().clone()
    for _ in range(3):
        m.K = 3
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        (loss + m.aux_loss()).backward()
        opt.step()
        core.mlp.apply_balance()
    assert math.isfinite(loss.item()) and not torch.equal(mix0, core.ws_mix.detach())
    m.eval()
    m.K = 4
    base = m(x)
    m.knockout = "all"
    assert not torch.allclose(base, m(x))   # the workspace still matters
    m.knockout = None
    assert torch.allclose(m(x)[:, :20], m(x[:, :20]), atol=1e-5)  # causal


# ---------------------------------------------------------------- v4: accumulation, head groups, growing waves
def wave_loop(**kw):
    """The v4 base: waves + workspace v2, no vector state (double precision)."""
    cfg = dict(state="", waves=True, slots=4, ws_direct=True, heads=2)
    cfg.update(kw)
    return tiny_loop(**cfg)


def calibrate_all(m):
    for s in m.spikes():  # first-use calibration samples a random subset: make it batch-independent
        s.calibrated.fill_(True)


def loop_total(m, x, y):
    """The trainer's per-batch objective: LM + router aux (z) + 0.2 deep supervision."""
    _, loss = m(x, y)
    total = loss + m.aux_loss()
    if m.aux_x is not None:
        total = total + 0.2 * torch.nn.functional.cross_entropy(m.heads["lm"](m.aux_x).flatten(0, 1), y.flatten())
    return total


@pytest.mark.parametrize("groups", [0, 2])
def test_accumulation_equals_full_batch(groups):
    m = wave_loop(heads=6, head_group=2 if groups else 0, head_groups=groups)
    calibrate_all(m)
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 4, 2, 1
    m2 = copy.deepcopy(m)
    x = torch.randint(0, 50, (4, 33))
    loop_total(m, x[:, :-1], x[:, 1:]).backward()
    for half in (x[:2], x[2:]):
        (loop_total(m2, half[:, :-1], half[:, 1:]) / 2).backward()
    for (name, p), p2 in zip(m.named_parameters(), m2.parameters()):
        assert (p.grad is None) == (p2.grad is None), name
        if p.grad is not None:
            assert torch.allclose(p.grad, p2.grad, atol=1e-10, rtol=1e-8), name
    a, b = m.blocks[0].mlp, m2.blocks[0].mlp
    assert torch.equal(a.pending, b.pending) and torch.equal(a.pending_elig, b.pending_elig)
    a.apply_balance()
    b.apply_balance()
    assert torch.allclose(a.bias, b.bias, atol=1e-12) and torch.equal(a.age, b.age)


@pytest.mark.parametrize("groups", [0, 2])
def test_loop_head_births_exact_and_groups(groups):
    m = wave_loop(heads=2, head_group=2 if groups else 0, head_groups=groups)
    x = torch.randint(0, 50, (2, 33))
    mix = m.blocks[0].mix
    m.eval()
    for _ in range(4):
        before = {}
        for K in (1, 3, 5):
            m.K = K
            before[K] = m(x)
        m.grow_heads(1, ramp=True)
        for K in (1, 3, 5):
            m.K = K
            assert (m(x) - before[K]).abs().max().item() < 1e-10   # a birth never changes the output
    assert m.cfg["heads"] == 6 and isinstance(m.cfg["heads"], int)
    assert mix.hgroup.tolist() == ([0, 0, 1, 2, 1, 2] if groups else [0] * 6)  # round-robin
    # a newborn only acts through its ramp: wo columns trained (here: random) but ramp 0 -> no change
    new = slice(5 * mix.p, 6 * mix.p)
    m.K = 3
    base = m(x)
    with torch.no_grad():
        mix.wo.weight[:, new] = torch.randn_like(mix.wo.weight[:, new]) * 0.1
    assert (m(x) - base).abs().max().item() < 1e-10
    mix.ramp[new] = 0.5
    assert (m(x) - base).abs().max().item() > 1e-6                   # output changes once ramp > 0
    if groups:  # a head gets no gradient on passes that do not use its group
        mix.ramp.fill_(1.0)
        with torch.no_grad():
            mix.wo.weight.normal_(0, 0.1)
        m.train()
        m.K, m.group_offset = 1, 0                                       # one pass, group 1
        _, loss = m(x[:, :-1], x[:, 1:])
        loss.backward()
        g = mix.wz.weight.grad.view(mix.h, mix.p, -1).abs().sum((1, 2))
        assert (g[mix.hgroup == 2] == 0).all() and (g[mix.hgroup != 2] > 0).all()
        assert m.diag["head_group_passes"] == [1, 0]
        gn = m.head_group_grad_norms()
        assert gn[2] == 0 and gn[0] > 0 and gn[1] > 0


def test_pass_group_rotation():
    m = wave_loop(heads=7, head_group=2, head_groups=5)
    x = torch.randint(0, 50, (2, 17))
    m.eval()
    m.K, m.group_offset = 6, 3
    assert [m.pass_group(k) for k in range(6)] == [1, 2, 3, 4, 5, 1]   # eval ignores the offset
    m(x)
    assert m.diag["head_group_passes"] == [2, 1, 1, 1, 1]
    m.train()
    assert [m.pass_group(k) for k in range(7)] == [4, 5, 1, 2, 3, 4, 5]  # K > G wraps around
    seen = set()
    for _ in range(40):
        m.group_offset = m.sample_group_offset()
        seen.add(m.pass_group(0))
    assert seen == {1, 2, 3, 4, 5}   # with random offsets every group also gets the first pass


def test_brainopt_leaves_idle_head_groups_untouched():
    """A head group not computed on any pass this step gets zero gradient AND no optimizer change at all
    (no stale momentum, no weight decay), including its wo columns and 1-D per-head tensors."""
    from brain_opt import BrainOpt
    torch.manual_seed(0)
    m = wave_loop(heads=2, head_group=2, head_groups=2).float()
    m.grow_heads(2, ramp=False)   # heads 2, 3 -> groups 1, 2
    opt = BrainOpt(m, lr=3e-3, muon=True, lazy=True)
    x = torch.randint(0, 50, (2, 17))
    m.train()
    m.K = 4                       # first step: both groups used, momentum everywhere
    m.group_offset = 0
    m(x, x)[1].backward()
    opt.step()
    opt.zero_grad()
    mix = m.blocks[0].mix
    idle = mix.hgroup == 2
    snap = {n: p.detach().clone() for n, p in mix.named_parameters()}
    m.K = 1                       # second step: only group 1 -> group 2 idle
    m(x, x)[1].backward()
    opt.step()
    P = mix.p
    ch = idle.repeat_interleave(P)
    for n, p in mix.named_parameters():
        old = snap[n]
        if n == "wo.weight":
            assert torch.equal(p[:, ch], old[:, ch]) and not torch.equal(p[:, ~ch], old[:, ~ch])
        elif p.dim() >= 1 and p.size(0) == mix.h * P:
            assert torch.equal(p[ch], old[ch]), n
        elif p.dim() >= 1 and p.size(0) == mix.h:
            assert torch.equal(p[idle], old[idle]), n


def test_brainopt_fine_lazy_only_touches_head_group_membranes():
    """Fairness with v2: every parameter outside the head-group-sliced membranes gets bit-for-bit the update
    of the original optimizer rule (fine_lazy=False), and a dense (ungrouped) model is unaffected entirely."""
    import copy
    from brain_opt import BrainOpt
    for groups in (2, 0):
        torch.manual_seed(0)
        m = wave_loop(heads=2, head_group=2 if groups else 0, head_groups=groups).float()
        m.grow_heads(2, ramp=False)
        m2 = copy.deepcopy(m)
        o1, o2 = BrainOpt(m, lr=3e-3), BrainOpt(m2, lr=3e-3, fine_lazy=False)
        assert bool(o1.fine) == bool(groups) and not o2.fine
        x = torch.randint(0, 50, (2, 17))
        # step 1 uses every group (identical results); step 2 leaves group 2 idle. Compare that one step
        # from the same state: only the membranes may differ (later steps differ everywhere, via the forward)
        for K in (4, 1):
            for mm, o in ((m, o1), (m2, o2)):
                torch.manual_seed(K)
                mm.train()
                mm.K, mm.group_offset = K, 0
                o.zero_grad()
                mm(x, x)[1].backward()
                o.step()
            if K == 4:
                assert all(torch.equal(p, p2) for p, p2 in zip(m.parameters(), m2.parameters()))
        for (n, p), (_, p2) in zip(m.named_parameters(), m2.named_parameters()):
            if n not in o1.fine:
                assert torch.equal(p, p2), n
        if groups:  # and the membranes do differ: the old rule moved the idle heads
            assert not all(torch.equal(p, p2) for (n, p), (_, p2) in
                           zip(m.named_parameters(), m2.named_parameters()) if n in o1.fine)


def test_v4_head_layout_six_groups_of_five():
    """The v4 run: 6 always-on heads + 30 newborn heads round-robin into 6 routed groups of 5, so the K=6
    evaluation uses every routed group exactly once."""
    m = wave_loop(heads=6, head_group=6, head_groups=6)
    for _ in range(30):  # heads are born one at a time during training
        m.grow_heads(1, ramp=True)
    mix = m.blocks[0].mix
    assert m.cfg["heads"] == 36 and isinstance(m.cfg["heads"], int)
    assert torch.bincount(mix.hgroup).tolist() == [6, 5, 5, 5, 5, 5, 5]
    m.eval()
    m.K = 6
    assert [m.pass_group(k) for k in range(6)] == [1, 2, 3, 4, 5, 6]
    m(torch.randint(0, 50, (2, 17)))
    assert m.diag["head_group_passes"] == [1, 1, 1, 1, 1, 1]
    assert all(len(mix.active(g)[0]) == 11 for g in range(1, 7))  # 6 core + 5 routed heads per pass


@pytest.mark.parametrize("spike", ["analog", "graded", "ngc"])
def test_grouped_mixer_matches_dense(spike):
    from spiking_lm import SSDMixer
    torch.manual_seed(0)
    dense = SSDMixer(16, 6, 4, spike, 8).double()
    x = torch.randn(2, 20, 16, dtype=torch.float64)
    with torch.no_grad():
        dense.ramp[8:12] = 0.5  # a ramping head too
    dense.ramping = True
    for G in (1, 2):
        grp = SSDMixer(16, 6, 4, spike, 8, group0=2, n_groups=G).double()
        sd = dict(dense.state_dict(), hgroup=grp.hgroup.clone())
        grp.load_state_dict(sd)
        assert grp.ramping
        if G == 1:  # group 0 + group 1 = every head: identical to the dense mixer, values and gradients
            xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
            ya, yb = grp(xa, group=1), dense(xb)
            assert torch.allclose(ya, yb, atol=1e-12)
            ya.pow(2).sum().backward()
            yb.pow(2).sum().backward()
            assert torch.allclose(xa.grad, xb.grad, atol=1e-12)
            for (n, p), q in zip(grp.named_parameters(), dense.parameters()):
                assert (p.grad is None and q.grad is None) or torch.allclose(p.grad, q.grad, atol=1e-12), n
            dense.zero_grad()
        elif spike != "ngc":  # (ngc normalises over the computed channels, by design)
            ref = copy.deepcopy(dense)
            idle = (grp.hgroup == 2).nonzero().squeeze(1)
            with torch.no_grad():  # an idle head contributes nothing: dense with its outputs cut
                for h in idle.tolist():
                    ref.wo.weight[:, h * 4:(h + 1) * 4] = 0
            assert torch.allclose(grp(x, group=1), ref(x), atol=1e-12)


@pytest.mark.parametrize("balance", ["aux", "auxfree"])
def test_loop_expert_births_with_waves(balance):
    m = wave_loop(moe=dict(n_exp=32, k=3, hidden=16, shared=16, capacity=0, balance=balance, aux_w=0.01))
    core, mlp = m.blocks[0], m.blocks[0].mlp
    x = torch.randint(0, 50, (4, 33))
    m.eval()
    C0, pos0, adj0 = core.router_conn.detach().clone(), core.pos.clone(), core.adj2.clone()
    rs0 = mlp.router_s.weight.detach().clone()
    from loop_brain import sheet_adj
    assert torch.equal(sheet_adj(pos0, 16, 2), adj0)        # full 16x2 grid: same neighbours as the torus
    before = {}
    for K in (2, 4):
        m.K = K
        before[K] = m(x)
    src = m.grow_experts(3).tolist()
    for K in (2, 4):
        m.K = K
        assert (m(x) - before[K]).abs().max().item() < 1e-10   # newborns are not selected at birth
    C = core.router_conn.detach()
    assert C.shape == (35, 35) and torch.equal(C[:32, :32], C0) and m.cfg["moe"]["n_exp"] == 35
    for i, p in enumerate(src):
        n = 32 + i
        sib = [32 + j for j, q in enumerate(src) if q == p]
        others = [j for j in range(32) if j != p]
        assert torch.equal(C[n, others], C0[p, others])   # outgoing row = parent's
        assert torch.equal(C[others, n], C0[others, p])   # incoming column = parent's
        assert C[n, p] == 0 and C[p, n] == 0 and all(C[n, s] == 0 for s in sib)  # twin entries and diagonal
        for j, q in enumerate(src):
            if q != p:
                assert C[n, 32 + j] == C0[p, q]              # newborn <-> newborn from the parents
        assert torch.equal(mlp.router_s.weight[n], rs0[p])
    assert core.pos.shape == (35, 2) and torch.equal(core.pos[:32], pos0)
    assert (core.pos[32:, 0] >= 0).all() and (core.pos[32:, 0] < 16).all() and (core.pos[32:, 1] < 2).all()
    assert core.adj2.shape == (35, 35) and torch.equal(core.adj2[:32, :32], adj0)
    if balance == "aux":  # bug fix: the newborn's debt is paid back in aux mode too, then it gets selected
        for _ in range(1100):
            mlp.pending.fill_(1.0)
            mlp.pending[32:] = 0                                 # not selected yet
            mlp.pending_elig.fill_(1.0)
            mlp.apply_balance()
        assert mlp.debt.max() == 0 and torch.allclose(mlp.bias, torch.zeros_like(mlp.bias), atol=1e-9)
        picks = []
        h = mlp.register_forward_hook(lambda mod, a, o: picks.append(mod.last_idx.clone()))
        m.K = 4
        m(x)
        h.remove()
        used = torch.cat(picks).flatten()
        assert all((used == 32 + i).any() for i in range(3)), used.bincount(minlength=35)[32:]


FIXTURE = Path(__file__).resolve().parent / "test_fixtures" / "v4_step2000_router.pt"


def _replay_router(lift, steps):
    """The TRAINED router of v4 at step 2000 (aux mode, 119 newborns), replayed on its real inputs (3 passes of
    1024 words) with the balancer running per step; no learning (router weights fixed). Returns the load per
    expert (1 = average) of the last step and the module."""
    from spiking_lm import RoutedMLP
    fx = torch.load(FIXTURE, weights_only=False)
    E, D = fx["router"].shape
    mlp = RoutedMLP(D, n_exp=E, k=fx["k"], hidden=4, shared=0, balance="aux", capacity=0, d_side=64, lift=lift)
    mlp.defer = True
    with torch.no_grad():
        mlp.router.weight.copy_(fx["router"])
        mlp.router_s.weight.copy_(fx["router_s"])
        for n in ("bias", "debt", "age", "load_ema"):
            getattr(mlp, n).copy_(fx[n])
    mlp.train()
    for _ in range(steps):
        cnt = torch.zeros(E)
        with torch.no_grad():
            for p in fx["passes"]:
                wave = None if p["prev_idx"] is None else (p["prev_idx"], p["fatigue"].float(), fx["conn"],
                                                           fx["gamma"], fx["beta"])
                mlp(p["xf"].float()[None], side=p["side"].float()[None], wave=wave)
                cnt += torch.bincount(mlp.last_idx.flatten(), minlength=E)
        mlp.apply_balance()
    return cnt / cnt.mean(), mlp


@pytest.mark.skipif(not FIXTURE.exists(), reason="needs test_fixtures/v4_step2000_router.pt")
def test_aux_lift_brings_stale_newborns_back_on_the_real_router():
    """v4 bug: with --route-aux, newborns whose debt was paid back stayed unpicked (their fresh router rows
    were shrunk by the z-loss while ramping, and the aux loss never pulls an unpicked expert up)."""
    rel0, m0 = _replay_router(0.0, 100)
    rel1, m1 = _replay_router(0.5, 100)
    paid = (m0.debt == 0) & (torch.arange(m0.E) >= 192)       # newborns already fully ramped at step 2000
    assert paid.sum() > 20
    assert rel0[paid].median() == 0                            # bug reproduced: none of them is picked
    assert rel1[paid].median() > 0.2                           # lift: they get traffic within 100 steps
    assert (rel1 < 0.05).float().mean() < (rel0 < 0.05).float().mean() - 0.05  # fewer dead experts overall
    assert m1.bias.max() < 0.05 and m1.bias[:192].min() >= 0   # small lift, never pushes an old expert down


def test_aux_lift_stops_newborn_ramp_at_fair_share():
    """With lift, a newborn whose fresh row scores high stops paying back its debt once it gets its fair share
    (as in auxfree), instead of climbing to bias 0 and flooding."""
    from spiking_lm import RoutedMLP
    res = {}
    for lift in (0.0, 0.5):
        torch.manual_seed(0)
        mlp = RoutedMLP(16, n_exp=8, k=2, hidden=4, shared=0, balance="aux", capacity=0, lift=lift).double()
        mlp.defer = True
        x = torch.randn(1, 256, 16, dtype=torch.float64)
        mlp.grow_experts(1)
        with torch.no_grad():  # the newborn's row scores far above every trained row (like a fresh v4 newborn)
            mlp.router.weight[:8] *= 0.1
            mlp.router.weight[8] = x[0].mean(0) / x[0].mean(0).norm() * 4
            for _ in range(1100):
                mlp(x)
                mlp.apply_balance()
            mlp(x)
        load = torch.bincount(mlp.last_idx.flatten(), minlength=9).double()
        res[lift] = (mlp.bias[8].item(), (load[8] / load.mean()).item(), mlp.debt[8].item())
    assert res[0.0][0] == 0 and res[0.0][1] > 2                 # full payback: the newborn floods
    assert res[0.5][2] == 0 and res[0.5][0] < -0.3 and res[0.5][1] < 1.6  # stopped at about its fair share


def test_aux_lift_relaxes_raised_bias_and_birth_delta():
    from spiking_lm import RoutedMLP
    mlp = RoutedMLP(16, n_exp=4, k=2, hidden=4, shared=0, balance="aux", capacity=0, lift=0.5, birth_delta=0.1)
    with torch.no_grad():
        mlp.bias.copy_(torch.tensor([0.01, 0.0, 0.0, 0.0]))
        mlp._balance(torch.tensor([20.0, 10.0, 6.0, 4.0]), torch.tensor(10.0))
    assert torch.allclose(mlp.bias, torch.tensor([0.009, 0.0, 0.0, 0.001 * 0.2]))  # busy: relaxes toward 0, quiet: lifted
    w = mlp.router.weight.data.clone()
    mlp.age.fill_(10**6)
    src = mlp.grow_experts(2)
    d = (mlp.router.weight.data[4:] - w[src]).norm(dim=1) / w[src].norm(dim=1)
    assert torch.allclose(d, torch.full_like(d, 0.1), atol=1e-5)  # newborn row = parent + 0.1 x its norm


def test_shield_gives_ramping_newborns_no_router_gradient():
    """shield: a newborn still ramping in (debt > 0) gets exactly zero router gradient from the z-loss and the
    aux loss (in v4 the aux P normalisation inflated unpicked ramping rows); everyone else is unchanged."""
    from spiking_lm import RoutedMLP
    res = {}
    for shield in (False, True):
        torch.manual_seed(0)
        mlp = RoutedMLP(16, n_exp=6, k=2, hidden=4, shared=0, balance="aux", capacity=0, shield=shield).double()
        mlp.age.fill_(10**6)
        mlp.grow_experts(2)
        x = torch.randn(2, 30, 16, dtype=torch.float64)
        mlp.train()
        mlp(x).sum().mul(0).add(mlp.aux).backward()           # z-loss + aux loss only
        res[shield] = mlp.router.weight.grad.clone()
    assert res[False][6:].abs().sum() > 0                       # without shield: newborn rows are pushed
    assert res[True][6:].abs().sum() == 0                       # shield: untouched (lazy optimizer: no step, no decay)
    assert res[True][:6].abs().sum() > 0


def test_balance_both_runs_bias_balancer_and_aux_loss():
    from spiking_lm import RoutedMLP
    torch.manual_seed(0)
    mlp = RoutedMLP(16, n_exp=6, k=2, hidden=4, shared=0, balance="both", aux_w=0.01, capacity=0).double()
    x = torch.randn(2, 30, 16, dtype=torch.float64)
    mlp.train()
    with torch.no_grad():
        mlp(x)
    assert mlp.bias.abs().sum() > 0                            # bias balancer moved
    zl = mlp.z_w * torch.logsumexp(mlp.router(x.reshape(-1, 16)), -1).pow(2).mean()
    assert mlp.aux > zl                                         # aux balance term is in the loss


def test_birth_mode_and_flags_reach_the_loop_model():
    m = wave_loop(heads=2, moe=dict(n_exp=8, k=2, hidden=4, shared=8, capacity=0, balance="aux",
                                    lift=0.5, shield=True, birth_mode="split", birth_delta=0.1))
    mlp = m.blocks[0].mlp
    assert (mlp.lift, mlp.shield, mlp.birth_mode, mlp.birth_delta) == (0.5, True, "split", 0.1)
    m2 = type(m)(**m.cfg)                                       # rebuilt from the saved config
    assert (m2.blocks[0].mlp.shield, m2.blocks[0].mlp.birth_mode) == (True, "split")


def test_bias_logit_selects_on_the_logit_scale():
    """bias_logit: selection = topk(logit + bias). With zero bias it picks exactly what the score-based rule picks
    (sigmoid is monotone), so outputs are identical; with a bias it follows the logit-scale ranking."""
    from spiking_lm import RoutedMLP
    torch.manual_seed(0)
    a = RoutedMLP(16, n_exp=8, k=2, hidden=4, shared=0, balance="aux", capacity=0).double()
    b = RoutedMLP(16, n_exp=8, k=2, hidden=4, shared=0, balance="aux", capacity=0, bias_logit=True).double()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 20, 16, dtype=torch.float64)
    a.eval(), b.eval()
    assert torch.allclose(a(x), b(x), atol=1e-12)
    with torch.no_grad():
        b.bias.copy_(torch.linspace(-1, 1, 8, dtype=torch.float64))
        b(x)
        lg = b.router(x.reshape(-1, 16))
        ref = (lg + b.bias).topk(2, -1).indices
    assert torch.equal(b.last_idx.sort(1).values, ref.sort(1).values)


def test_deep_sup_and_halt_off_change_nothing():
    m = wave_loop(heads=2)
    assert m.halt_head is None and m.cfg["deep_sup"] == 0 and not m.cfg["halt"]
    x = torch.randint(0, 50, (2, 17))
    m.train()
    m.K, m.aux_k_fixed = 4, 2
    m(x, x)
    assert m.loop_extra is None and m.aux_x is not None      # the old single deep-supervision pass is kept


def _per_pass_ce(m, x, K):
    """Per-pass CE of every pass (training mode, no targets: same computation as the per-pass read-out)."""
    outs = m.run(x, K, every_tick=True, raw=True)
    return [F.cross_entropy(m.heads["lm"](m.normf(o)).float().flatten(0, 1), x.flatten()) for o in outs]


def test_deep_supervision_is_the_mean_of_the_earlier_backprop_passes():
    m = wave_loop(heads=2, deep_sup=0.3)
    x = torch.randint(0, 50, (2, 17))
    m.train()
    m.K = 5                                                   # bptt 3: passes 2, 3 backprop, pass 4 is the main loss
    torch.manual_seed(1)
    m(x, x)
    got = m.loop_extra
    assert m.aux_x is None                                    # replaces the single sampled pass
    torch.manual_seed(1)
    ce = _per_pass_ce(m, x, 5)
    assert torch.allclose(got, 0.3 * (ce[2] + ce[3]) / 2, atol=1e-10)
    got.backward()
    assert m.heads["lm"].weight.grad.abs().sum() > 0


def test_halt_head_trains_only_itself_and_distribution_is_valid():
    m = wave_loop(heads=2, halt=True, halt_prior=0.3, halt_beta=0.05)
    x = torch.randint(0, 50, (2, 17))
    m.train()
    m.K = 6
    m(x, x)
    m.zero_grad()
    m.loop_extra.backward()                                   # halting loss only (deep_sup is off)
    assert m.halt_head[0].weight.grad.abs().sum() > 0
    others = [p.grad for n, p in m.named_parameters() if not n.startswith("halt_head")]
    assert all(g is None or g.abs().sum() == 0 for g in others)  # never shapes the brain itself
    lams = [torch.rand(2, 3, dtype=torch.float64) for _ in range(5)]
    P = m.halt_distribution(lams)
    assert torch.allclose(P.sum(0), torch.ones(2, 3, dtype=torch.float64))
    assert 1 <= m.halt_stats["expected_passes"] <= 6
    m2 = type(m)(**m.cfg).double()                            # rebuilt from the saved config, strict reload
    m2.load_state_dict(m.state_dict())


def test_halt_head_learns_on_passes_before_the_bptt_window():
    """The stop probabilities of the EARLY passes (outside backprop through the brain) still train the head."""
    m = wave_loop(heads=2, halt=True)                         # bptt 3
    x = torch.randint(0, 50, (2, 17))
    m.train()
    m.K = 6
    m(x, x)
    lam0 = []
    orig = m.halt_lambda
    m.halt_lambda = lambda feat: (lam0.append(orig(feat)), lam0[-1])[1]
    m(x, x)
    m.halt_lambda = orig
    assert len(lam0) == 6 and lam0[0].requires_grad          # pass 0 (outside the window) has a gradient path
    g = torch.autograd.grad(m.loop_extra, lam0[0], retain_graph=True)[0]
    assert g.abs().sum() > 0


def test_halt_eval_reads_each_word_at_its_own_exit():
    m = wave_loop(heads=2, halt=True)
    x = torch.randint(0, 50, (2, 17))
    with torch.no_grad():
        m.halt_head[2].bias.fill_(-20.0)                      # never stops early -> every word runs to the ceiling
    r = m.halt_eval(x, x, K_max=6)
    assert r["halt_passes"] == 6 and abs(r["halt_loss"] - r["fixed_loss"][6]) < 1e-9
    with torch.no_grad():
        m.halt_head[2].bias.fill_(20.0)                       # always stops at once -> pass 1
    r = m.halt_eval(x, x, K_max=6)
    assert r["halt_passes"] == 1 and abs(r["halt_loss"] - r["fixed_loss"][1]) < 1e-9
    m.eval()
    m.K = 3
    assert abs(r["fixed_loss"][3] - F.cross_entropy(m(x).float().flatten(0, 1), x.flatten()).item()) < 1e-5  # fp32 read-out


def test_free_cache_only_on_cuda(monkeypatch):
    import train_lm
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    train_lm.free_cache("cpu")
    train_lm.free_cache("cuda")
    assert calls == [1]


def test_think_head_off_by_default():
    m = wave_loop(heads=2)
    assert not m.think and m.think_head is None and m.think_fb is None
    x = torch.randint(0, 50, (2, 17))
    m.eval()
    m(x)
    assert m.think_u == []


def test_think_loss_is_the_bag_of_next_words_cross_entropy():
    m = wave_loop(heads=2, think_w=0.5, think_n=3)
    B, T, V = 2, 5, 50
    cl = torch.log_softmax(torch.randn(B, T, V, dtype=torch.float64), -1)
    y = torch.randint(0, V, (B, T))
    tot, cnt = 0.0, 0
    for b in range(B):
        for t in range(T):
            for j in range(3):
                if t + j < T:                                  # the bag of position t: y[t], y[t+1], y[t+2]
                    tot -= cl[b, t, y[b, t + j]].item()
                    cnt += 1
    assert abs(m.think_loss(cl, y).item() - tot / cnt) < 1e-10


def test_think_training_loss_and_uncertainty_per_pass():
    m = wave_loop(heads=2, think_w=0.5)
    x = torch.randint(0, 50, (2, 17))
    m.train()
    m.K = 5
    m(x, x)
    assert m.loop_extra is not None and m.think_last > 0 and len(m.think_u) == 5
    assert m.think_u[0].shape == (2, 17, 2) and torch.all(m.think_u[0][..., 1] == 0)  # no change on pass 1
    m.loop_extra.backward()
    assert m.think_head[0].weight.grad.abs().sum() > 0
    assert m.blocks[0].w_wv.weight.grad.abs().sum() > 0        # it trains the workspace to hold the direction


def test_think_feedback_starts_silent_and_learns():
    m = wave_loop(heads=2, think_fb=True)
    x = torch.randint(0, 50, (2, 17))
    m.eval()
    m.K = 4
    with torch.no_grad():
        y_fb = m(x)
        fb, m.think_fb = m.think_fb, None
        y_plain = m(x)
        m.think_fb = fb
    assert torch.equal(y_fb, y_plain)                          # zero-initialised: nothing changes at birth
    m.train()
    m.K = 4
    loss = m(x, x)[1]
    loss.backward()
    assert m.think_fb.weight.grad.abs().sum() > 0              # the brain can learn to use its uncertainty


def test_think_with_halt_and_diagnostics():
    m = wave_loop(heads=2, think_w=0.3, think_fb=True, halt=True)
    assert m.halt_head[0].in_features == 32 + 5                # x + output change/entropy/pass + thought u
    x = torch.randint(0, 50, (2, 17))
    m.train()
    m.K = 4
    m(x, x)
    m.loop_extra.backward()
    r = m.halt_eval(x, x, K_max=6)
    assert 1 <= r["halt_passes"] <= 6
    d = m.think_diag(x, x, K_max=8, kmax_eval=6)
    assert all(0 <= d[k] <= 1 or d[k] != d[k] for k in ("auc_concept_entropy", "auc_concept_change", "auc_output_change"))
    assert len(d["concept_entropy_by_pass"]) == 6
    m2 = type(m)(**m.cfg).double()
    m2.load_state_dict(m.state_dict())


def test_latent_target_is_the_stopgrad_window_mean_of_the_final_representation():
    m = wave_loop(heads=2, think_w=0.1, think_target="latent", think_n=3)
    x = torch.randn(2, 6, 32, dtype=torch.float64, requires_grad=True)
    tgt, mask = m.think_target(x)
    z = m.normf(x).detach()
    for t in range(6):
        w = z[:, t + 1:t + 4]
        if w.size(1):
            assert torch.allclose(tgt[:, t].double(), w.mean(1), atol=1e-6)  # the target is kept in fp32
    assert mask.tolist() == [True] * 5 + [False] and not tgt.requires_grad


def test_latent_loss_nll_mask_and_variance_hinge():
    m = wave_loop(heads=2, think_w=0.1, think_target="latent")
    mu = torch.randn(2, 4, 3, dtype=torch.float64)
    lv = torch.randn(2, 4, 3, dtype=torch.float64).clamp(-6, 4)
    tg = torch.randn(2, 4, 3, dtype=torch.float64)
    mask = torch.tensor([True, True, True, False])
    nll = (0.5 * (lv + (tg - mu) ** 2 / lv.exp()))[:, :3].mean()
    hinge = torch.relu(tg[:, :3].flatten(0, 1).std(0) - mu[:, :3].flatten(0, 1).std(0)).mean()
    assert torch.allclose(m.think_latent_loss(mu, lv, tg, mask), nll + hinge, atol=1e-10)


def test_latent_concept_trains_and_logs_collapse_detectors():
    m = wave_loop(heads=2, think_w=0.1, think_target="latent", think_fb=True, halt=True)
    assert m.halt_head[0].in_features == 32 + 5
    x = torch.randint(0, 50, (2, 17))
    m.train()
    m.K = 5
    m(x, x)
    assert m.think_last is not None and set(m.think_stats) == {"target_std", "pred_std", "cosine"}
    m.loop_extra.backward()
    assert m.think_head[0].weight.grad.abs().sum() > 0
    r = m.halt_eval(x, x, K_max=6)
    assert 1 <= r["halt_passes"] <= 6
    m2 = type(m)(**m.cfg).double()
    m2.load_state_dict(m.state_dict())


def test_think_use_starts_silent_and_reaches_later_words():
    """Usefulness: the concept of word t0, written back into the next pass's core input, changes nothing at
    birth (zero init) and gets gradient from the loss of LATER words only (through the membrane scan)."""
    m = wave_loop(heads=2, think_w=0.1, think_target="latent", think_use=True)
    x = torch.randint(0, 50, (2, 17))
    m.eval()
    m.K = 4
    with torch.no_grad():
        y = m(x)
        m.think_zero = True
        y0 = m(x)
        m.think_zero = False
    assert torch.equal(y, y0)                                 # silent at birth
    with torch.no_grad():
        m.think_in.weight.normal_(0, 0.1)                     # as after some training
    mus = []
    orig = m.think_latent_readout
    m.think_latent_readout = lambda s, prev: (lambda o: (mus.append(o[0]), o)[1])(orig(s, prev))
    m.train()
    m.K = 4
    logits = m(x)
    m.think_latent_readout = orig
    t0 = 5
    later = F.cross_entropy(logits[:, t0 + 1:].float().flatten(0, 1), x[:, t0 + 1:].flatten())
    g = torch.autograd.grad(later, mus[-2])[0]                # concept of the second-to-last pass
    assert g[:, t0].abs().sum() > 0                           # word t0's thought helps the words after it


def _noise_mlp():
    from spiking_lm import RoutedMLP
    torch.manual_seed(0)
    mlp = RoutedMLP(16, n_exp=8, k=2, hidden=8, shared=0, balance="aux", aux_w=0.01, capacity=0).double()
    mlp.defer = True  # no balancer state changes between calls
    return mlp, torch.randn(3, 20, 16, dtype=torch.float64)


def _combine(mlp, x, idx):
    """Dense output of the experts in idx, weighted by the CLEAN router scores (like dense_reference)."""
    xf = x.reshape(-1, x.size(-1))
    score = mlp.route(xf, None, None, None)[0]
    gate = torch.zeros_like(score).scatter(1, idx, score.gather(1, idx))
    gate = gate / gate.sum(1, keepdim=True)
    h = torch.relu(torch.einsum("nd,edh->enh", xf, mlp.w_up))
    return torch.einsum("enh,ehd,ne->nd", h, mlp.w_down, gate).view(x.shape), score


def test_route_noise_off_is_the_old_code_and_eval_is_clean():
    mlp, x = _noise_mlp()
    mlp.train()
    rng = torch.get_rng_state()
    y0 = mlp(x)
    assert torch.equal(torch.get_rng_state(), rng)          # noise 0 draws no random numbers
    assert torch.allclose(y0, mlp.dense_reference(x), atol=1e-10)
    mlp.noise = 1.0
    mlp.eval()                                               # evaluation never uses the noise
    assert torch.equal(mlp(x), mlp(x)) and torch.allclose(mlp(x), mlp.dense_reference(x), atol=1e-10)


def test_route_noise_changes_selection_but_not_gates_or_balance_loss():
    mlp, x = _noise_mlp()
    mlp.train()
    clean_idx = (mlp.route(x.reshape(-1, 16), None, None, None)[0] + mlp.bias).topk(2, -1).indices
    mlp.noise = 1.0
    torch.manual_seed(1)
    y = mlp(x)
    idx = mlp.last_idx
    changed = (idx.sort(1).values != clean_idx.sort(1).values).any(1).float().mean()
    assert 0.05 < changed < 0.95                              # some tokens pick other experts, not all
    ref, score = _combine(mlp, x, idx)
    assert torch.allclose(y, ref, atol=1e-10)                 # gate weights: clean scores of the picked experts
    f = torch.bincount(idx.flatten(), minlength=8).double() / idx.numel()
    P = (score / score.sum(1, keepdim=True)).mean(0)
    zl = mlp.z_w * torch.logsumexp(mlp.router(x.reshape(-1, 16)), -1).pow(2).mean()
    assert torch.allclose(mlp.aux, zl + 0.01 * 8 * (f * P).sum(), atol=1e-12)  # balance loss uses clean P
    y.sum().backward()                                        # gradients flow as usual
    assert mlp.router.weight.grad.abs().sum() > 0


def test_eval_route_noise_is_a_separate_temperature():
    mlp, x = _noise_mlp()
    mlp.eval()
    mlp.noise = 1.0                                           # training temperature: ignored at eval
    clean = mlp(x)
    assert torch.equal(mlp(x), clean)
    mlp.eval_noise = 1.0                                      # eval temperature: picks change, gates stay clean
    torch.manual_seed(3)
    y = mlp(x)
    assert not torch.equal(y, clean)
    assert torch.allclose(y, _combine(mlp, x, mlp.last_idx)[0], atol=1e-10)


def test_route_noise_in_the_loop_eval_unchanged():
    m = wave_loop(heads=2).float()
    x = torch.randint(0, 50, (2, 17))
    m.eval()
    m.K = 4
    ref = m(x)
    m.blocks[0].mlp.noise = 0.7
    assert torch.equal(m(x), ref)                             # eval: noise ignored
    m.train()
    m.K, m.group_offset = 4, 0
    loss = m(x, x)[1]
    loss.backward()
    assert torch.isfinite(loss) and m.train_route[0] >= 2 and 0 <= m.train_route[1] <= 1


def test_routed_aux_mode_newborns_get_selected_after_ramp():
    """Regression (plan bug 1): with --route-aux the debt payback never ran -> newborns 2 below forever."""
    from spiking_lm import RoutedMLP
    torch.manual_seed(0)
    mlp = RoutedMLP(16, n_exp=6, k=2, hidden=8, shared=0, balance="aux", aux_w=0.01, capacity=0).double()
    x = torch.randn(4, 16, 16, dtype=torch.float64)
    mlp.grow_experts(2)
    mlp.train()
    with torch.no_grad():
        for _ in range(1100):
            mlp(x)
    assert mlp.debt.max() == 0 and torch.allclose(mlp.bias, torch.zeros_like(mlp.bias), atol=1e-9)
    mlp(x)
    assert (mlp.stats[0][6:] > 0).all(), mlp.stats[0]


def test_brain_state_carried_through_router_conn_growth():
    from brain_opt import BrainOpt, carry_brain_state
    m = wave_loop(heads=4, head_group=2, head_groups=2).float()
    opt = BrainOpt(m, lr=3e-3, maturity=4, tau=5)
    x = torch.randint(0, 50, (2, 33))

    def step(o):
        m.K = 3
        _, loss = m(x[:, :-1], x[:, 1:])
        o.zero_grad()
        (loss + m.aux_loss()).backward()
        o.step()
        m.blocks[0].mlp.apply_balance()
        return loss.item()

    for _ in range(3):
        step(opt)
    old = {k: v.clone() for k, v in opt.state["blocks.0.router_conn"].items()}
    m.grow_experts(2)
    m.grow_heads(2, ramp=True)
    opt2 = carry_brain_state(opt, m, BrainOpt(m, lr=3e-3, maturity=4, tau=5))
    st = opt2.state["blocks.0.router_conn"]
    assert set(st) == {"born", "m", "v", "n"} and st["m"].shape == (14, 14)
    assert old["n"].shape == (12, 1) and st["n"].shape == (14, 14)  # per-row count; per element once columns grow
    for key, buf in st.items():
        assert torch.equal(buf[:12, :12], old[key].expand(12, 12))
        new = torch.cat([buf[12:].flatten(), buf[:12, 12:].flatten()])
        expect = {"born": float(opt.t), "v": old["v"].median().item()}.get(key, 0.0)
        assert torch.allclose(new, torch.full_like(new, expect))
    assert all(math.isfinite(step(opt2)) for _ in range(2))
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_checkpoint_reload_after_loop_growth_mid_ramp():
    """Regression (plan bugs 2, 3): Neurogenesis set cfg["heads"] to a list (LoopBrain(**cfg) broke) and
    the mixer's `ramping` flag was not restored by load_state_dict (a mid-ramp checkpoint lost its ramp)."""
    from loop_brain import LoopBrain
    from neurogenesis import Neurogenesis
    m = wave_loop(heads=2, head_group=2, head_groups=2)
    ng = Neurogenesis(m, dict(layers=1, heads=2, mlp=0, experts=12), dict(layers=1, heads=4, mlp=0, experts=16),
                      steps=100, every=5, ramp=50)
    for step in range(46):
        ng.step(step)
    mix = m.blocks[0].mix
    assert mix.h == 4 and mix.ramping and ((mix.ramp > 0) & (mix.ramp < 1)).any()
    with torch.no_grad():  # as if the newborn heads had trained a little
        mix.wo.weight[:, 2 * mix.p:] = torch.randn_like(mix.wo.weight[:, 2 * mix.p:]) * 0.1
    cfg = copy.deepcopy(m.cfg)
    assert isinstance(cfg["heads"], int) and cfg["heads"] == 4 and cfg["moe"]["n_exp"] == m.blocks[0].mlp.E
    m2 = LoopBrain(**cfg).double()
    m2.load_state_dict(m.state_dict(), strict=True)
    assert m2.blocks[0].mix.ramping
    m.eval()
    m2.eval()
    x = torch.randint(0, 50, (2, 33))
    for K in (2, 5):
        m.K = m2.K = K
        assert torch.equal(m(x), m2(x))
    m2.blocks[0].mix.ramping = False   # what the old reload did: the ramp was silently dropped
    assert not torch.allclose(m(x), m2(x))


def test_reload_mid_ramp_keeps_ramping():
    from neurogenesis import Neurogenesis
    m = tiny("ssd", "graded", heads=1, mlp=16)
    ng = Neurogenesis(m, dict(layers=2, heads=1, mlp=16), dict(layers=2, heads=2, mlp=32), steps=100, every=5,
                      where="uniform", ramp=50)
    for step in range(30):
        ng.step(step)
    assert m.blocks[0].mlp.ramping
    with torch.no_grad():
        for b in m.blocks:
            b.mlp.down.weight.normal_(0, 0.1)
    m2 = SpikingLM(**copy.deepcopy(m.cfg)).double()
    m2.load_state_dict(m.state_dict(), strict=True)
    for b, b2 in zip(m.blocks, m2.blocks):
        assert b2.mlp.ramping == b.mlp.ramping and getattr(b2.mix, "ramping", None) == getattr(b.mix, "ramping", None)
    m.eval()
    m2.eval()
    x = torch.randint(0, 50, (2, 33))
    assert torch.equal(m(x), m2(x))


def test_neurogenesis_grows_the_loop_with_brainopt():
    from brain_opt import BrainOpt, carry_brain_state
    from neurogenesis import Neurogenesis
    m = wave_loop(heads=2, head_group=2, head_groups=2).float()
    ng = Neurogenesis(m, dict(layers=1, heads=2, mlp=0, experts=12), dict(layers=1, heads=6, mlp=0, experts=20),
                      steps=60, every=5, ramp=20)
    opt = BrainOpt(m, lr=3e-3, maturity=4, tau=10)
    x = torch.randint(0, 50, (2, 33))
    rebuilds = 0
    for step in range(60):
        if step % 5 == 0:  # ramps are an intended change; the births themselves must be exact
            ng.ramps()
            m.eval()
            m.K = 3
            before = m(x[:, :-1])
            changed = ng.births_step(step)
            assert (m(x[:, :-1]) - before).abs().max().item() < 1e-5   # births never change the output
            m.train()
        else:
            changed = ng.step(step)
        if changed:
            opt = carry_brain_state(opt, m, BrainOpt(m, lr=3e-3, maturity=4, tau=10))
            rebuilds += 1
            params = dict(m.named_parameters())
            assert all(p is params[n] for n, (p, _) in opt.groups.items())  # optimizer sees the new tensors
        m.K = 3
        m.group_offset = m.sample_group_offset()
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        (loss + m.aux_loss()).backward()
        opt.step()
        m.blocks[0].mlp.apply_balance()
    core = m.blocks[0]
    assert core.mix.h == 6 and core.mlp.E == 20 and core.router_conn.shape == (20, 20) and core.pos.shape == (20, 2)
    assert m.cfg["heads"] == 6 and m.cfg["moe"]["n_exp"] == 20 and rebuilds >= 4
    assert core.mix.hgroup.tolist() == [0, 0, 1, 2, 1, 2] and math.isfinite(loss.item())
    assert ng.done() and not Neurogenesis(wave_loop(heads=2), dict(layers=1, heads=2, mlp=0, experts=12),
                                          dict(layers=1, heads=6, mlp=0, experts=20), steps=60).done()


def test_embedding_bag_wave_equals_dense():
    torch.manual_seed(0)
    E, N, k = 12, 20, 3
    C = torch.randn(E, E, dtype=torch.float64, requires_grad=True)
    idx = torch.randn(N, E).topk(k, -1).indices
    prev = torch.zeros(N, E, dtype=torch.float64).scatter(1, idx, 1.0)
    w = torch.randn(N, E, dtype=torch.float64)
    a = torch.nn.functional.embedding_bag(idx, C, mode="sum")
    ga, = torch.autograd.grad((a * w).sum(), C)
    b = prev @ C
    gb, = torch.autograd.grad((b * w).sum(), C)
    assert torch.allclose(a, b, atol=1e-12) and torch.allclose(ga, gb, atol=1e-12)
    # and inside the router: the tuple path equals the old dense `extra` path, values and gradients
    m = wave_loop()
    mlp, core = m.blocks[0].mlp, m.blocks[0]
    h = torch.randn(2, 10, 32, dtype=torch.float64)
    idx = torch.randn(20, 12).topk(3, -1).indices
    prev = torch.zeros(20, 12, dtype=torch.float64).scatter(1, idx, 1.0)
    fat = prev * 1.7
    g, bb = torch.nn.functional.softplus(core.wave_gamma), torch.nn.functional.softplus(core.wave_beta)
    mlp.eval()
    y1 = mlp(h, extra=g * (prev @ core.router_conn) - bb * fat)
    g1 = torch.autograd.grad(y1.pow(2).sum(), core.router_conn)[0]
    y2 = mlp(h, wave=(idx, fat, core.router_conn, g, bb))
    g2 = torch.autograd.grad(y2.pow(2).sum(), core.router_conn)[0]
    assert torch.allclose(y1, y2, atol=1e-12) and torch.allclose(g1, g2, atol=1e-12)


@pytest.mark.parametrize("balance", ["aux", "auxfree"])
def test_checkpointed_routing_equals_plain(balance):
    m = wave_loop(heads=4, head_group=2, head_groups=1,
                  moe=dict(n_exp=12, k=3, hidden=16, shared=16, capacity=0, balance=balance, aux_w=0.01))
    calibrate_all(m)
    m.train()
    m.K, m.aux_k_fixed = 4, 2
    m2 = copy.deepcopy(m)
    m2.blocks[0].mlp.ckpt_route = True
    x = torch.randint(0, 50, (2, 33))
    t1, t2 = loop_total(m, x[:, :-1], x[:, 1:]), loop_total(m2, x[:, :-1], x[:, 1:])
    assert torch.allclose(t1, t2, atol=1e-12)
    t1.backward()
    t2.backward()
    for (n, p), q in zip(m.named_parameters(), m2.parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-12), n
    assert m.blocks[0].router_conn.grad.abs().sum() > 0


# ---------------------------------------------------------------- memory / speed: read-out, optimizer state, router
def v4d_loop(**kw):
    """The v4d per-pass features: deep supervision, halting, latent thinking with feedback and write-back."""
    cfg = dict(heads=4, head_group=2, head_groups=2, deep_sup=0.3, halt=True, think_w=0.1, think_target="latent",
               think_n=4, think_fb=True, think_use=True)
    cfg.update(kw)
    m = wave_loop(**cfg)
    calibrate_all(m)
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 5, 2, 1
    return m


def _loop_step(m, x, seed=0):
    torch.manual_seed(seed)
    _, loss = m(x[:, :-1], x[:, 1:])
    total = loss + m.aux_loss()
    total.backward()
    return total.detach(), m.loop_extra.detach()


def test_readout_checkpoint_and_full_fraction_are_the_plain_readout():
    """B: the checkpointed read-out (log-probs recomputed in backward) gives bit-identical values and gradients;
    A: readout_frac 1 draws nothing and changes nothing, and a subset that holds every word is the same maths."""
    x = torch.randint(0, 50, (2, 18))
    ref = v4d_loop()
    ref.ckpt_readout = False                                  # the old path: log-probs kept for backward
    ck, probe, alls = (copy.deepcopy(ref) for _ in range(3))
    ck.ckpt_readout = True
    alls.cfg["readout_frac"] = 0.9999                         # subset path, n = round(0.9999 * 34) = all 34 words
    t0, e0 = _loop_step(ref, x)
    t1, e1 = _loop_step(ck, x)
    assert torch.equal(t0, t1) and torch.equal(e0, e1)
    assert ck.cfg["readout_frac"] == 1.0
    for (n, p), q in zip(ref.named_parameters(), ck.parameters()):
        assert (p.grad is None) == (q.grad is None), n
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), n
    torch.manual_seed(0)
    rng = torch.get_rng_state()
    probe(x[:, :-1], x[:, 1:])
    assert torch.equal(torch.get_rng_state(), rng)            # F = 1 draws no random numbers
    t2, e2 = _loop_step(alls, x)
    assert torch.allclose(e0, e2, atol=1e-12) and torch.allclose(t0, t2, atol=1e-12)
    for (n, p), q in zip(ref.named_parameters(), alls.parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-12), n


def test_readout_fraction_subsamples_words_and_trains():
    x = torch.randint(0, 50, (2, 41))                         # 80 words -> 20 read out per pass
    m = v4d_loop(readout_frac=0.25)
    seen = []
    orig = m.pass_readout_ce
    m.pass_readout_ce = lambda xr, tgt, *a: (seen.append((xr.shape, tgt.shape, a[2])), orig(xr, tgt, *a))[1]
    t, extra = _loop_step(m, x)
    m.pass_readout_ce = orig
    assert len(seen) == m.K and all(s[0] == (20, 32) and s[1] == (20,) for s in seen)
    assert all(s[2] is not None and s[2].shape == (20, 2) for s in seen)  # the thought's uncertainty, same words
    assert torch.isfinite(t) and torch.isfinite(extra)
    assert m.halt_head[0].weight.grad.abs().sum() > 0 and m.heads["lm"].weight.grad.abs().sum() > 0
    assert m.think_head[0].weight.grad.abs().sum() > 0 and m.think_last is not None
    assert 1 <= m.halt_stats["expected_passes"] <= m.K
    picks = []                                                # the subset is drawn once per forward
    m.pass_readout_ce = lambda xr, tgt, *a: (picks.append(tgt.clone()), orig(xr, tgt, *a))[1]
    m(x[:, :-1], x[:, 1:])
    m.pass_readout_ce = orig
    assert all(torch.equal(p, picks[0]) for p in picks)
    r = m.halt_eval(x[:, :-1], x[:, 1:], K_max=4)             # evaluation reads every word
    assert sum(r["passes_hist"]) == 80


def per_element_opt(m, **kw):
    """The previous BrainOpt state layout: a step count per element (birth times are unused without maturity)."""
    from brain_opt import BrainOpt

    class Old(BrainOpt):
        def count_shape(self, name, p):
            return tuple(p.shape)

    return Old(m, **kw)


def test_brainopt_compact_state_is_bit_identical_through_lazy_steps_and_growth():
    from brain_opt import BrainOpt, carry_brain_state
    m = wave_loop(heads=4, head_group=2, head_groups=2).float()
    calibrate_all(m)
    m2 = copy.deepcopy(m)
    new, old = BrainOpt(m, lr=3e-3), per_element_opt(m2, lr=3e-3)
    x = torch.randint(0, 50, (2, 25))

    def step(mod, o, i):
        mod.train()
        mod.K, mod.group_offset = 3, i % 2
        torch.manual_seed(i)
        _, loss = mod(x[:, :-1], x[:, 1:])
        o.zero_grad()
        (loss + mod.aux_loss()).backward()
        if i % 2:  # lazy: some experts, router rows and embedding rows get no gradient
            for p in (mod.blocks[0].mlp.w_up, mod.blocks[0].mlp.w_down, mod.blocks[0].mlp.router.weight,
                      mod.blocks[0].router_conn, mod.emb.weight):
                p.grad[:3] = 0
        o.step(0.7)
        mod.blocks[0].mlp.apply_balance()

    for i in range(8):  # (> 6 steps: CPU pow differs between its scalar / vector paths at 0.95 ** 6)
        step(m, new, i)
        step(m2, old, i)
    st = new.state
    assert all("born" not in s for s in st.values())          # maturity 0: no birth times
    assert st["blocks.0.mlp.router.weight"]["n"].shape == (12, 1) and st["heads.lm.weight"]["n"].shape == (50, 1)
    assert st["normf.weight"]["n"].shape == (1,) and st["blocks.0.mix.A_log"]["n"].shape == (4,)  # fine: per element
    for mod in (m, m2):
        torch.manual_seed(99)                                 # the same newborns in both
        mod.grow_experts(2)
        mod.grow_heads(2, ramp=True)
    new = carry_brain_state(new, m, BrainOpt(m, lr=3e-3))
    old = carry_brain_state(old, m2, per_element_opt(m2, lr=3e-3))
    assert new.state["blocks.0.mlp.router.weight"]["n"].shape == (14, 1)   # rows appended, still per row
    assert new.state["blocks.0.router_conn"]["n"].shape == (14, 14)       # columns grew too: per element
    for i in range(8, 14):
        step(m, new, i)
        step(m2, old, i)
    for (n, p), q in zip(m.named_parameters(), m2.parameters()):
        assert torch.equal(p, q), n
    for name, s in old.state.items():
        for key, buf in s.items():
            assert torch.equal(new.state[name][key].expand_as(buf), buf), (name, key)


@pytest.mark.parametrize("router", [dict(balance="aux", aux_w=0.01, lift=0.5, shield=True, bias_logit=True),
                                    dict(balance="auxfree", bias_logit=False), dict(balance="both", aux_w=0.01)])
def test_checkpointed_routing_is_exact_with_every_router_feature(router):
    """D: --ckpt-route recomputes the router in backward; with noise, the bias on the logits, the shield (ramping
    newborns with debt), the wave tuple and lift it must give bit-identical values, selections and gradients."""
    m = v4d_loop(moe=dict(n_exp=12, k=3, hidden=16, shared=16, capacity=1.5, birth_mode="ramp", **router))
    m.grow_experts(2)                                         # ramping newborns: debt > 0 (the shield's case)
    mlp = m.blocks[0].mlp
    mlp.noise = 0.3
    with torch.no_grad():
        mlp.bias.copy_(torch.linspace(-0.2, 0.2, 14, dtype=mlp.bias.dtype))
    m2 = copy.deepcopy(m)
    m2.blocks[0].mlp.ckpt_route = True
    x = torch.randint(0, 50, (2, 25))
    picks = [[], []]
    for mod, pk in zip((m, m2), picks):
        mod.blocks[0].mlp.register_forward_hook(lambda md, a, o, pk=pk: pk.append(md.last_idx.clone()))
    t1, e1 = _loop_step(m, x)
    t2, e2 = _loop_step(m2, x)
    assert torch.equal(t1, t2) and torch.equal(e1, e2)
    assert all(torch.equal(a, b) for a, b in zip(*picks)) and len(picks[0]) == m.K
    for (n, p), q in zip(m.named_parameters(), m2.parameters()):
        assert (p.grad is None) == (q.grad is None), n
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), n
    assert m.blocks[0].router_conn.grad.abs().sum() > 0
    a, b = m.blocks[0].mlp, m2.blocks[0].mlp
    a.apply_balance()
    b.apply_balance()
    assert torch.equal(a.bias, b.bias) and torch.equal(a.debt, b.debt)
