"""Tests of the memory system (plan_v2/12_memory_full_plan.md section 9, tests 1-17). CPU only, tiny configs.

    python -m pytest fruit-fly-sim/test_memory.py -q
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

HERE = Path(__file__).resolve().parent
BASE = HERE / "plan_v2" / "baseline_pre_mem"
MODS = ("spiking_lm", "loop_brain", "brain_opt", "neurogenesis", "train_lm", "memory", "concepts")
_BASELINE = None


def baseline():
    """The pristine pre-memory code (plan_v2/baseline_pre_mem), imported as separate module objects: the current
    modules are taken out of sys.modules while it imports (so its `from spiking_lm import ...` binds to the baseline
    files) and put back afterwards. No bytecode is written into the plan folder."""
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


def current():
    import brain_opt
    import loop_brain
    import neurogenesis
    import spiking_lm
    import train_lm
    return SimpleNamespace(spiking_lm=spiking_lm, loop_brain=loop_brain, brain_opt=brain_opt,
                           neurogenesis=neurogenesis, train_lm=train_lm)


# ---------------------------------------------------------------- test 1: flags off == the old code, bit for bit
def _drive(mods, steps=40, accum=2, seed=0):
    """The trainer's step (train_lm.main) on a tiny v4d-like brain: growth on (expert splits + head births with
    ramps), graded spikes with the anneal, homeostasis, routing noise, deep supervision, halting, latent thinking
    with feedback and write-back, BrainOpt with rebuilds, accumulation. Returns every micro-batch loss, the eval
    outputs every 10 steps and the final state."""
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
                      think_target="latent", think_use=True)
    model.set_homeo(0.01, 0.15, "floor")
    mk = lambda: BrainOpt(model, 2e-3, 0.1, muon=True, lazy=True, maturity=0.0, tau=10)
    opt = mk()
    neuro = Neuro(model, start, final, steps, every=5, t0=0.05, t1=0.6)
    corpus = torch.randint(0, 50, (5000,), generator=torch.Generator().manual_seed(1))
    rng = np.random.RandomState(0)
    batch = lambda: torch.stack([corpus[i:i + 17] for i in rng.randint(0, len(corpus) - 17, 4)])
    xv = torch.randint(0, 50, (2, 16), generator=torch.Generator().manual_seed(2))
    losses, outs, last_grow = [], [], -10 ** 9
    for step in range(steps + 1):
        if neuro.step(step):
            model.set_homeo(0.01, 0.15, "floor")
            opt = carry(opt, model, mk())
        mem = getattr(model, "mem", None)  # (the new trainer's hooks; None with the flags off)
        frac = step / steps
        lr = 2e-3 * min(1.0, step / max(1, 0.02 * steps))
        if frac > 0.8:
            lr *= max(0.0, (1 - frac) / 0.2)
        lam = min(1.0, max(0.0, (frac - 0.05) / 0.15))
        for mod in model.modules():
            if isinstance(mod, RoutedMLP):
                mod.noise, mod.eval_noise = 0.3, 0.0
        model.set_lam(lam)
        if step % 10 == 0 or step == steps:
            model.eval()
            model.K = 4
            with torch.no_grad():
                outs.append(model(xv))
            model.train()
        if step == steps:
            break
        model.K = model.sample_K()
        model.aux_k_fixed = model.sample_aux_k(model.K)
        model.group_offset = model.sample_group_offset()
        if mem is not None:
            mem.write_on = True
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
            losses.append(torch.stack([loss.detach(), total.detach()]))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(lr / 2e-3)
        model.blocks[0].mlp.apply_balance()
        if mem is not None:
            mem.after_step(lr)
    return dict(losses=losses, outs=outs, state=model.state_dict(), cfg=model.cfg, E=model.blocks[0].mlp.E,
                heads=model.blocks[0].mix.h, opt_state=opt.state, rng=torch.get_rng_state())


def test_01_flags_off_bit_identical_to_the_old_code_over_40_steps():
    old, new = _drive(baseline()), _drive(current())
    assert old["E"] == new["E"] == 12 and old["heads"] == new["heads"] == 8      # growth happened
    assert len(old["losses"]) == len(new["losses"]) == 80
    for i, (a, b) in enumerate(zip(old["losses"], new["losses"])):
        assert torch.equal(a, b), (i, a, b)
    for a, b in zip(old["outs"], new["outs"]):
        assert torch.equal(a, b)
    assert old["cfg"] == new["cfg"]
    assert list(old["state"]) == list(new["state"])
    for k in old["state"]:
        assert torch.equal(old["state"][k], new["state"][k]), k
    assert list(old["opt_state"]) == list(new["opt_state"])
    for name, st in old["opt_state"].items():
        for key, buf in st.items():
            assert torch.equal(buf, new["opt_state"][name][key]), (name, key)
    assert torch.equal(old["rng"], new["rng"])


def _tiny_data(root, n_train=20000, n_val=6000):
    from tokenizers import Tokenizer, models, pre_tokenizers
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(5)
    for name, n in (("train", n_train), ("val", n_val)):
        t = rng.randint(1, 50, n).astype(np.uint16)
        t[rng.rand(n) < 0.02] = 0                                        # documents end with EOT = 0
        t.tofile(root / f"{name}.bin")
    vocab = {"<|endoftext|>": 0, **{f"w{i}": i for i in range(1, 50)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(root / "tokenizer.json"))
    (root / "meta.json").write_text(json.dumps({"eot_id": 0, "dtype": "uint16"}))
    return root


TINY_ARGS = ("--arch loop --spike graded --homeo 0.01 --homeo-mode floor --opt brain --batch 4 --accum 2 --eval-batch 8 "
             "--route-aux 0.02 --capacity 1.5 --moe 12,2,16,16,auxfree --heads 8 --start heads=4,experts=8 --neuro "
             "--neuro-every 5 --neuro-t 0.05,0.6 --head-group 4 --head-groups 2 --state none --waves --slots 4 "
             "--ws-direct --anneal 0.05,0.2 --birth-mode split --birth-delta 0.1 --route-z 1e-4 --route-noise 0.3 "
             "--route-noise-decay 0 --deep-sup 0.3 --halt 1 --think-w 0.1 --think-fb 1 --think-target latent "
             "--think-n 4 --think-use 1 --d 32 --d-state 8 --head-dim 8 --k-mean 4 --k-min 2 --k-max 6 --bptt 3 "
             "--seq 320 --tokens 102400 --eval-every 20 --eval-batches 2 --ckpt-every 0 --device cpu").split()


def _run_main(train_lm, runs, name, data, extra=()):
    argv, out, err = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["train_lm.py", "--run-name", name, "--runs", str(runs), "--data", str(data), *TINY_ARGS, *extra]
    try:
        train_lm.main()
    finally:
        log = sys.stdout.file if hasattr(sys.stdout, "file") else None
        sys.argv, sys.stdout, sys.stderr = argv, out, err
        if log is not None:
            log.close()
    run = Path(runs) / name
    recs = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    return run, recs, torch.load(run / "final.pt", weights_only=False)


def test_01b_trainer_flags_off_bit_identical_end_to_end(tmp_path):
    """train_lm.main itself (old file vs new file, same arguments, CPU): metrics and the final checkpoint match."""
    data = _tiny_data(tmp_path / "data")
    _, ra, ca = _run_main(baseline().train_lm, tmp_path / "runs_old", "old", data)
    _, rb, cb = _run_main(current().train_lm, tmp_path / "runs_new", "new", data)
    assert len(ra) == len(rb) == 3
    for a, b in zip(ra, rb):
        for k in ("tok_s", "elapsed"):  # wall-clock
            a.pop(k), b.pop(k)
        assert a == b
    assert ca["cfg"] == cb["cfg"] and list(ca["state"]) == list(cb["state"])
    assert all(torch.equal(ca["state"][k], cb["state"][k]) for k in ca["state"])


# ---------------------------------------------------------------- helpers for the memory tests
def mem_loop(dtype=torch.float64, **kw):
    from loop_brain import LoopBrain
    cfg = dict(vocab=50, d=32, d_s=8, heads=4, head_dim=8, spike="graded", chunk=16, k_mean=4, k_min=2, k_max=6,
               bptt=3, state="", refractory=1, slots=4, waves=True, ws_direct=True, head_group=2, head_groups=2,
               deep_sup=0.3, halt=True, think_w=0.1, think_fb=True, think_use=True, think_target="latent",
               moe=dict(n_exp=12, k=3, hidden=16, shared=16, capacity=0, birth_mode="split"),
               concepts=[2, 4, 8, 16], mem_shared=8, mem_private=8, mem_dv=16, mem_max_experts=16, mem_topm=4,
               mem_topk=3)
    cfg.update(kw)
    torch.manual_seed(0)
    m = LoopBrain(**cfg).to(dtype)
    for s in m.spikes():
        s.calibrated.fill_(True)
    return m


def train_step(m, opt, x, lr=3e-3, K=4, offset=0):
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = K, max(0, K - 2), offset
    _, loss = m(x[:, :-1], x[:, 1:])
    total = loss + m.aux_loss()
    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    m.blocks[0].mlp.apply_balance()
    if m.mem is not None:
        m.mem.after_step(lr)
    return total.detach()


def tokens(B=2, T=17, seed=3):
    return torch.randint(1, 50, (B, T), generator=torch.Generator().manual_seed(seed))


def mem_params(m):
    return {n: p for n, p in m.named_parameters() if n.startswith(("mem.", "hier."))}


# ---------------------------------------------------------------- test 2: concept targets
def test_02_concept_targets_match_brute_force():
    from memory import concept_targets
    torch.manual_seed(0)
    B, T, D, eot = 3, 40, 5, 0
    z = torch.randn(B, T, D, dtype=torch.float64)
    idx = torch.randint(1, 9, (B, T))
    idx[0, [5, 6, 20]] = eot                      # consecutive EOTs, an EOT in the middle
    idx[1, T - 1] = eot                           # at the end
    idx[2, 0] = eot                               # at the start
    for n in (4, 8, 16):
        (tgt, valid), = concept_targets(z, idx, [n], eot)
        for b in range(B):
            for t in range(T):
                bt = next((j for j in range(t + 1, T) if idx[b, j] == eot), T)   # next EOT strictly after t
                e = min(t + n, T - 1, bt - 1)
                c = e - t
                ref = z[b, t + 1:e + 1].mean(0) if c > 0 else torch.zeros(D, dtype=z.dtype)
                assert torch.allclose(tgt[b, t], ref, atol=1e-12), (n, b, t)
                assert bool(valid[b, t]) == (c >= n / 4 and int(idx[b, t]) != eot), (n, b, t)
    (tgt, valid), = concept_targets(z, idx, [8], eot)
    assert not valid[0, 5] and not valid[0, 6] and not valid[2, 0]       # EOT positions predict nothing
    assert torch.allclose(tgt[0, 2], z[0, 3:5].mean(0)) and valid[0, 2]       # cut at the EOT: 2 >= 8/4 words
    assert torch.allclose(tgt[0, 3], z[0, 4:5].mean(0)) and not valid[0, 3]   # 1 < 8/4 words
    assert torch.allclose(tgt[0, 10], z[0, 11:19].mean(0)) and valid[0, 10]


# ---------------------------------------------------------------- test 3: level conditioning, masked hinge
def test_03_level_conditioning_and_masked_moment_hinge():
    from memory import ConceptHier, concept_loss, masked_std
    torch.manual_seed(0)
    h = ConceptHier(16, 24, 3).double()
    p = torch.randn(2, 9, 24, dtype=torch.float64)
    mus, nus = h(p)
    assert len(mus) == 3 and all(((-6 < n) & (n < 4)).all() for n in nus)
    tgt = torch.randn(2, 9, 16, dtype=torch.float64)
    valid = torch.rand(2, 9) < 0.6
    concept_loss(mus[1], nus[1], tgt, valid).backward()
    assert h.levels[0].weight.grad.abs().sum() > 0                    # level 2's loss trains level 1 (no sg)
    assert h.levels[2].weight.grad is None
    x = torch.randn(2, 9, 16, dtype=torch.float64)
    v = valid.double()[..., None]
    got = masked_std(x, v, v.sum())
    sel = valid.flatten().nonzero().squeeze(1)
    xs = x.flatten(0, 1)[sel]
    ref = (xs.var(0, unbiased=False) + 1e-6).sqrt()                    # the nonzero() version
    assert torch.allclose(got, ref, atol=1e-12)
    mu, nu = mus[0].detach(), nus[0].detach()
    nll = (0.5 * (nu + (tgt - mu) ** 2 * torch.exp(-nu))).flatten(0, 1)[sel].mean()
    hinge = F.relu((tgt.flatten(0, 1)[sel].var(0, unbiased=False) + 1e-6).sqrt()
                   - (mu.flatten(0, 1)[sel].var(0, unbiased=False) + 1e-6).sqrt()).mean()
    assert torch.allclose(concept_loss(mu, nu, tgt, valid), nll + hinge, atol=1e-12)


# ---------------------------------------------------------------- test 4: product keys
def test_04_product_keys_exact_with_lowest_index_tie_break():
    from memory import pk_topk, stable_topk
    g = torch.Generator().manual_seed(0)
    S, m, k = 16, 5, 4
    for trial in range(300):
        # integer scores (exact sums) in a small range: many ties, also across pairs
        s1 = torch.randint(0, 4, (S,), generator=g).double()
        s2 = torch.randint(0, 4, (S,), generator=g).double()
        v1, i1 = stable_topk(s1, m)
        v2, i2 = stable_topk(s2, m)
        ii, jj, vv = pk_topk(v1, i1, v2, i2, S, k)
        allsig = (s1[:, None] + s2[None, :]).flatten()                 # brute force over all S^2 pairs
        bv, bi = torch.sort(allsig, descending=True, stable=True)      # index order = flat order: lowest wins
        assert torch.equal(ii * S + jj, bi[:k]) and torch.equal(vv, bv[:k]), trial
    s = torch.tensor([0.5, 0.9, 0.9, 0.1, 0.9])
    assert stable_topk(s, 2)[1].tolist() == [1, 2]                     # duplicated scores -> lowest index


# ---------------------------------------------------------------- test 5: EMA indices vs online weights
def test_05_ema_picks_indices_online_weights_carry_the_gradient():
    m = mem_loop()
    mem = m.mem
    torch.manual_seed(1)
    mus = [torch.randn(2, 7, 32, dtype=torch.float64) for _ in range(4)]
    pbar = torch.randn(2, 7, 64, dtype=torch.float64)
    a, a_hat = mem.prep(mus, pbar)
    sel = mem.select(a, a_hat, False)
    with torch.no_grad():  # perturb the ONLINE translator a lot: the indices must not move
        for p in mem.keys.parameters():
            p.add_(torch.randn_like(p))
    sel2 = mem.select(a, a_hat, False)
    assert all(torch.equal(sel[k], sel2[k]) for k in ("i", "j", "I1", "I2", "rows", "qb1"))
    with torch.no_grad():
        mem.k1_ema.add_(torch.randn_like(mem.k1_ema))
    assert not torch.equal(mem.select(a, a_hat, False)["rows"], sel["rows"])   # the EMA side decides
    on = mem.online(a_hat, sel)
    (on["w"] * torch.randn_like(on["w"])).sum().backward()
    assert mem.keys.wq.grad.abs().sum() > 0 and mem.keys.k1.grad.abs().sum() > 0 and mem.keys.k2.grad.abs().sum() > 0
    assert torch.allclose(on["w"].sum(-1), torch.ones_like(on["w"][..., 0]))


# ---------------------------------------------------------------- test 6: freeze
def test_06_freeze_keeps_addressing_bit_identical_for_1000_steps_and_under_bf16():
    from brain_opt import BrainOpt
    m = mem_loop(torch.float32, k_mean=2, k_min=2, k_max=2, bptt=2, mem_shared=8, mem_private=4,
                 moe=dict(n_exp=6, k=2, hidden=8, shared=8, capacity=0, birth_mode="split"))
    opt = BrainOpt(m, lr=3e-3)
    x = tokens(1, 9)
    for i in range(3):  # infancy: everything moves
        train_step(m, opt, x, K=2, offset=i % 2)
    mem = m.mem
    mem.freeze()
    frozen = {n: t.detach().clone() for n, t in list(mem.keys.named_parameters()) +
              [(n, getattr(mem, n)) for n in ("wq_ema", "k1_ema", "k2_ema", "norm_m", "norm_v", "P", "P_ema")]}
    V0 = mem.V.clone()
    for i in range(1000):
        train_step(m, opt, x, K=2, offset=i % 2)
    for n, t0 in frozen.items():
        t = getattr(mem.keys, n) if hasattr(mem.keys, n) else getattr(mem, n)
        assert torch.equal(t, t0), n
    assert not torch.equal(mem.V, V0)                                  # the tables keep learning
    torch.manual_seed(2)
    mus = [torch.randn(1, 8, 32) for _ in range(4)]
    pbar = torch.randn(1, 8, 64)
    a, a_hat = mem.prep(mus, pbar)
    r1 = mem.select(a, a_hat, False)
    r2 = mem.select(a, a_hat, False)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a3, ah3 = mem.prep(mus, pbar)
        r3 = mem.select(a3, ah3, False)
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16
    for k in ("rows", "i", "j", "I1", "I2", "qb1"):
        assert torch.equal(r1[k], r2[k]) and torch.equal(r1[k], r3[k]), k
    assert r3["qb1"].dtype == torch.float32


# ---------------------------------------------------------------- test 7: which parts learn at step 0 / step 1
def test_07_step0_and_step1_gradients():
    from brain_opt import BrainOpt
    m = mem_loop(mem_margin=(0.0, 0.01))                                # the margin loss trains keys from step 0
    mem = m.mem
    opt = BrainOpt(m, lr=3e-3)
    x = tokens()
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 4, 2, 0
    _, loss = m(x[:, :-1], x[:, 1:])
    (loss + m.aux_loss()).backward()
    nz = lambda p: p.grad is not None and bool(p.grad.abs().sum() > 0)
    for p in (mem.w_o, mem.w_p.weight, m.hier.inj_b, m.hier.fb.weight):
        assert nz(p)                                                    # step 0: the zero-init outputs learn
    # (W_ms writes W_o's output M into the workspace: M = 0 at step 0, so W_ms's gradient is 0 too)
    for p in (mem.w_ms.weight, m.hier.inj_a, mem.gates.w, mem.gates.b, mem.gates.b_e, *mem.keys.parameters()):
        assert not nz(p)
    assert mem.G.abs().sum() == 0 and mem.G_e.abs().sum() == 0 and mem.G_P.abs().sum() == 0
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    m.blocks[0].mlp.apply_balance()
    mem.after_step(3e-3)
    opt.zero_grad()
    _, loss = m(x[:, :-1], x[:, 1:])
    (loss + m.aux_loss()).backward()
    for p in (mem.w_ms.weight, m.hier.inj_a, mem.gates.w, mem.gates.b, mem.gates.b_e, *mem.keys.parameters()):
        assert nz(p)                                                    # step 1: gates, translators, W_ms
    assert mem.G.abs().sum() > 0 and mem.G_e.abs().sum() > 0 and mem.G_P.abs().sum() > 0   # tables and keys


# ---------------------------------------------------------------- test 8: zero-init exactness
def test_08_zero_init_memory_changes_nothing():
    from loop_brain import LoopBrain
    base_kw = dict(vocab=50, d=32, d_s=8, heads=4, head_dim=8, spike="graded", chunk=16, k_mean=4, bptt=3, state="",
                   slots=4, waves=True, ws_direct=True, head_group=2, head_groups=2, deep_sup=0.3, halt=True,
                   moe=dict(n_exp=12, k=3, hidden=16, shared=16, capacity=0))
    torch.manual_seed(0)
    plain = LoopBrain(**base_kw).double()
    rng_plain = torch.get_rng_state()
    torch.manual_seed(0)
    mm = LoopBrain(**base_kw, think_w=0.1, think_fb=True, think_use=True, concepts=[2, 4, 8, 16], mem_shared=8,
                   mem_private=8, mem_dv=16, mem_max_experts=16, mem_topm=4, mem_topk=3).double()
    assert torch.equal(torch.get_rng_state(), rng_plain)               # the memory used its own random stream
    mp = dict(mm.named_parameters())
    for n, p in plain.named_parameters():                              # the existing modules: same init
        if not n.startswith("halt_head"):
            assert torch.equal(p, mp[n]), n
    for mod in (plain, mm):
        for s in mod.spikes():
            s.calibrated.fill_(True)
    x = tokens(2, 21)
    for K in (1, 3, 6):
        plain.eval(), mm.eval()
        plain.K = mm.K = K
        with torch.no_grad():
            assert torch.equal(plain(x), mm(x))
    plain.train(), mm.train()
    plain.K = mm.K = 4
    plain.aux_k_fixed = mm.aux_k_fixed = 2
    assert torch.equal(plain(x[:, :-1], x[:, 1:])[1], mm(x[:, :-1], x[:, 1:])[1])


# ---------------------------------------------------------------- test 9: sparse write
def test_09_sparse_row_adagrad_with_prior_and_update_clip():
    m = mem_loop()
    mem = m.mem
    mem.lr, mem.clip = 0.05, 0.03
    V0, S0 = mem.V.clone(), mem.S_acc.clone()
    torch.manual_seed(4)
    rows = torch.tensor([3, 17, 100])
    G = torch.randn(3, mem.dv, dtype=torch.float64) * torch.tensor([[1e-3], [1.0], [1e-2]], dtype=torch.float64)
    mem.G[rows] = G
    mem.s0_cfg = "auto"
    mem.after_step(1e-3)
    s0 = 100 * (G ** 2).mean(1).median().item()
    assert mem._s0[0] == pytest.approx(s0) and mem.s0[0].item() == pytest.approx(s0)
    S = s0 + (G ** 2).mean(1)
    dV = -0.05 * G / S.sqrt()[:, None]
    dV = dV * (0.03 / dV.norm(dim=1, keepdim=True)).clamp(max=1)
    assert torch.allclose(mem.V[rows], V0[rows] + dV, atol=1e-14) and torch.allclose(mem.S_acc[rows], S, atol=1e-14)
    assert dV.norm(dim=1).max() <= 0.03 + 1e-12 and dV[1].norm() == pytest.approx(0.03)   # clip on the update
    keep = torch.ones(len(V0), dtype=torch.bool)
    keep[rows] = False
    assert torch.equal(mem.V[keep], V0[keep])                           # unread rows: bit-identical
    assert (mem.S_acc[keep] == s0).all() and mem.wcount[rows].tolist() == [1, 1, 1] and mem.wcount[keep].sum() == 0
    assert mem.G.abs().sum() == 0
    # a real read: rows that no head read get exactly zero gradient
    from brain_opt import BrainOpt
    opt = BrainOpt(m, lr=3e-3)
    x = tokens()
    for _ in range(2):
        train_step(m, opt, x)
    m.train()
    m.K, m.aux_k_fixed = 4, 2
    m.mem.record = None
    _, loss = m(x[:, :-1], x[:, 1:])
    (loss + m.aux_loss()).backward()
    assert 0 < (mem.G != 0).any(1).sum() < len(mem.G)


# ---------------------------------------------------------------- test 10: causality
def test_10_causal_reads_and_outputs():
    m = mem_loop()
    with torch.no_grad():  # make every memory path matter
        m.mem.w_o.normal_(0, 0.3)
        m.mem.w_ms.weight.normal_(0, 0.3)
        m.mem.w_p.weight.normal_(0, 0.3)
        m.hier.inj_b.normal_(0, 0.3)
        m.hier.fb.weight.normal_(0, 0.3)
    m.eval()
    m.K = 5
    x = tokens(2, 30)
    x2 = x.clone()
    t = 17
    x2[:, t + 1:] = torch.randint(1, 50, x2[:, t + 1:].shape)
    recs = []
    outs = []
    for xx in (x, x2):
        m.mem.record = dict(pos=torch.arange(60), reads=[], calls=[])
        with torch.no_grad():
            outs.append(m(xx))
        recs.append(m.mem.record)
        m.mem.record = None
    assert torch.allclose(outs[0][:, :t + 1], outs[1][:, :t + 1], atol=1e-9)
    assert not torch.allclose(outs[0][:, t + 1:], outs[1][:, t + 1:])
    early = torch.tensor([b * 30 + i for b in range(2) for i in range(t + 1)])
    for r0, r1 in zip(recs[0]["reads"], recs[1]["reads"]):
        assert torch.equal(r0["rows"][early], r1["rows"][early])
        assert torch.allclose(r0["w"][early], r1["w"][early], atol=1e-9)
    for c0, c1 in zip(recs[0]["calls"], recs[1]["calls"]):
        assert torch.equal(c0["experts"][early], c1["experts"][early])
        if c0["prows"] is not None:
            assert torch.equal(c0["prows"][early], c1["prows"][early])


# ---------------------------------------------------------------- test 11: copy-on-split
def test_11_copy_on_split_twins_share_the_private_rows():
    from brain_opt import BrainOpt
    m = mem_loop()
    mem = m.mem
    opt = BrainOpt(m, lr=3e-3)
    x = tokens()
    for _ in range(3):
        train_step(m, opt, x)
    with torch.no_grad():  # distinct state per expert
        for t in (mem.P, mem.P_ema, mem.V_e, mem.S_e, mem.P_m, mem.P_v, mem.P_n, mem.U_e, mem.gates.b_e):
            t.add_(torch.rand_like(t))
        mem.wcount_e.random_(0, 100)
    snap = {n: getattr(mem, n).clone() for n in ("P", "P_ema", "V_e", "S_e", "wcount_e", "P_m", "P_v", "P_n", "U_e")}
    b0 = mem.gates.b_e.detach().clone()
    src = m.grow_experts(3)
    E0, R = 12, mem.R
    assert m.blocks[0].mlp.E == 15 and mem._e_live == 15 and int(mem.e_live) == 15
    for i, p in enumerate(src.tolist()):
        c = E0 + i
        for n in ("P", "P_ema", "U_e"):
            assert torch.equal(getattr(mem, n)[c], snap[n][p]), n
        for n in ("V_e", "S_e", "wcount_e", "P_m", "P_v", "P_n"):
            t = getattr(mem, n).view(mem.Emax, R, *getattr(mem, n).shape[1:])
            assert torch.equal(t[c], t[p]) and torch.equal(t[p], snap[n].view_as(t)[p]), n
        assert mem.gates.b_e[c] == b0[p]
    assert torch.equal(mem.P[:E0], snap["P"][:E0])                     # the parents are unchanged
    mem.freeze()                                                       # a birth after the freeze keeps it frozen
    m.grow_experts(1)
    assert all(not p.requires_grad for p in mem.keys.parameters())
    m.eval()
    m.K = 3
    with torch.no_grad():
        assert torch.isfinite(m(x)).all()
    rm = mem_loop(moe=dict(n_exp=12, k=3, hidden=16, shared=16, capacity=0, birth_mode="ramp"))
    rm.blocks[0].mlp.age.fill_(10 ** 6)
    P0 = rm.mem.P.clone()
    rm.grow_experts(2)
    assert torch.equal(rm.mem.P, P0)                                   # ramp children keep their own fresh rows


# ---------------------------------------------------------------- test 12: sub-key split with the band copy
def test_12_subkey_split_band_copy_keeps_the_contexts_values():
    m = mem_loop()
    mem = m.mem
    S = mem.S
    torch.manual_seed(5)
    mus = [torch.randn(3, 20, 32, dtype=torch.float64) for _ in range(4)]
    pbar = torch.randn(3, 20, 64, dtype=torch.float64)
    a, a_hat = mem.prep(mus, pbar)
    sel = mem.select(a, a_hat, False)
    l, h = 0, 0
    i = int(sel["i"][l, h, :, 0].mode().values)                         # a sub-key of half 1 in use
    mem.s0[0] = 0.25
    mem._s0[0] = 0.25
    mem.S_acc.fill_(0.7)
    with torch.no_grad():
        mem.U[l, h, 0] = 0.2 / (S - 1)
        mem.U[l, h, 0, i] = 0.8                                          # busy: > 4 x mean (= 1 / S)
    j = min(c for c in range(S) if c != i)                               # the least used (ties: lowest index)
    V0, K0, Ke0 = mem.V.clone(), mem.keys.k1.detach().clone(), mem.k1_ema.clone()
    assert torch.equal(K0, Ke0)                                          # (no training yet: EMA = online)
    from train_lm import reset_mem_key_rows                               # the trainer's optimizer-row reset

    class FakeOpt:  # BrainOpt-like: state by parameter name
        groups = {}
        state = {n: dict(m=torch.ones_like(p), v=torch.arange(p.numel(), dtype=p.dtype).view_as(p) + 1.0)
                 for n, p in m.named_parameters() if n.startswith("mem.keys.k")}
    opt = FakeOpt()
    mem.opt_reset = lambda name, sl: reset_mem_key_rows(opt, m, name, sl)
    v_med = {n: st["v"].median().item() for n, st in opt.state.items()}
    st = mem.balance()
    assert st["splits"] == 1
    m1, v1 = opt.state["mem.keys.k1"]["m"], opt.state["mem.keys.k1"]["v"]
    reset = torch.zeros_like(m1, dtype=torch.bool)
    reset[l, h, [i, j]] = True
    assert (m1[reset] == 0).all() and (m1[~reset] == 1).all()            # fresh momentum on the split pair only
    assert (v1[reset] == v_med["mem.keys.k1"]).all() and (v1[~reset] != v_med["mem.keys.k1"]).sum() >= (~reset).sum() - 1
    assert (opt.state["mem.keys.k2"]["m"] == 1).all()                   # the other codebook untouched
    band_i, band_j = mem.band(l, 0, i), mem.band(l, 0, j)
    assert band_i.tolist() == list(range(i * S, i * S + S))              # half 1: rows i*S .. i*S + S - 1
    assert torch.equal(mem.V[band_j], V0[band_i]) and torch.equal(mem.V[band_i], V0[band_i])   # the band copy
    assert (mem.S_acc[band_i] == 0.25).all() and (mem.S_acc[band_j] == 0.25).all()          # MemOpt rows reset
    assert (mem.S_acc == 0.25).sum() == 2 * S
    k1 = mem.keys.k1.detach()
    assert torch.equal(k1, mem.k1_ema)                                   # the EMA copy got the same edit
    changed = (k1[l, h] != K0[l, h]).any(-1).nonzero().flatten().tolist()
    assert changed == sorted([i, j]) and torch.equal(k1[:, 1:], K0[:, 1:])
    for c in (i, j):  # K_j = normalize(K_i + eps), K_i = normalize(K_i - eps), eps ~ N(0, 0.02)
        assert abs(k1[l, h, c].norm().item() - 1) < 1e-12 and F.cosine_similarity(k1[l, h, c], K0[l, h, i], 0) > 0.95
    assert mem.U[l, h, 0, i] == mem.U[l, h, 0, j] == 0.4
    # contexts that read through i and now read through j find the same values
    sel2 = mem.select(a, a_hat, False)
    km = mem.km
    moved = ((sel["i"][l, h] == i) & (sel2["i"][l, h] == j) & (sel["j"][l, h] == sel2["j"][l, h]))[..., :km]
    assert torch.equal(mem.V[sel2["rows"][l, h][moved]], V0[sel["rows"][l, h][moved]])
    swap = sel["rows"][l, h].clone()                                     # every context of i, sent to j instead
    on_i = sel["i"][l, h][..., :km] == i
    swap[on_i] = swap[on_i] + (j - i) * S
    assert on_i.any() and torch.equal(mem.V[swap[on_i]], V0[sel["rows"][l, h][on_i]])
    # dead re-seed: the band's S is reset to S0
    mem.qres.normal_()
    mem.S_acc.fill_(0.7)
    with torch.no_grad():
        mem.U.fill_(1.0 / S)
        mem.U[1, 2, 1, 3] = 1e-6
    k_old = mem.keys.k2[1, 2, 3].detach().clone()
    st = mem.balance()
    assert st["dead"] == 1 and not torch.equal(mem.keys.k2[1, 2, 3].detach(), k_old)
    m2 = opt.state["mem.keys.k2"]["m"]
    assert (m2[1, 2, 3] == 0).all() and (m2 == 0).sum() == m2.size(-1)  # re-seeded key: fresh momentum
    assert torch.equal(mem.keys.k2[1, 2, 3].detach(), mem.k2_ema[1, 2, 3])
    assert (mem.S_acc[mem.band(1, 1, 3)] == 0.25).all() and (mem.S_acc == 0.25).sum() == S


# ---------------------------------------------------------------- test 13: repair
def test_13_repair_only_low_usage_new_rows():
    from memory import MemoryProbes
    m = mem_loop()
    mem = m.mem
    pr = MemoryProbes(np.random.RandomState(0).randint(1, 50, 400).astype(np.uint16), 16, n_seq=2, n_pos=4,
                      repair=0.25, repair_usage=0.5)
    P, L, H, km = 4, mem.L, mem.H, mem.km
    g = torch.Generator().manual_seed(0)
    ref_rows = torch.randint(0, mem.S * mem.S, (P, L, H, km), generator=g) + (torch.arange(L) * mem.S ** 2).view(1, L, 1, 1)
    ref_rows = torch.sort(ref_rows, -1).values
    ref_rows[..., 1] = ref_rows[..., 0] + 1                              # distinct rows
    ref_rows[..., 2] = ref_rows[..., 0] + 2
    cur_rows = ref_rows.clone()
    cur_rows[0, 0, 0, 2] = ref_rows[0, 0, 0, 0] + 7                      # probe 0 moved: new row, dropped row
    cur_rows[1, 1, 2, 0] = ref_rows[1, 1, 2, 0] + 9                      # probe 1 moved
    w0 = torch.softmax(torch.randn(P, L, H, km, generator=g), -1)
    ref = dict(reads=[dict(k=0, rows=ref_rows, w=w0)])
    cur = dict(reads=[dict(k=0, rows=cur_rows, w=w0)])
    new_a, new_b = int(cur_rows[0, 0, 0, 2]), int(cur_rows[1, 1, 2, 0])
    drop_a, drop_b = int(ref_rows[0, 0, 0, 2]), int(ref_rows[1, 1, 2, 0])
    mem.wcount.fill_(10)
    mem.wcount[new_a] = 0                                                 # low usage: may be repaired
    mem.wcount[new_b] = 50                                               # busy: never overwritten
    V0 = mem.V.clone()
    ref0 = copy.deepcopy(ref)
    per_level = pr.repair(m, cur, ref)
    assert per_level == [1, 0, 0, 0]
    assert torch.allclose(mem.V[new_a], 0.75 * V0[new_a] + 0.25 * V0[drop_a])
    changed = (mem.V != V0).any(1).nonzero().flatten().tolist()
    assert changed == [new_a] and torch.equal(mem.V[new_b], V0[new_b])
    assert all(torch.equal(a, b) for a, b in ((ref["reads"][0]["rows"], ref0["reads"][0]["rows"]),
                                              (ref["reads"][0]["w"], ref0["reads"][0]["w"])))   # R0 unchanged
    assert drop_b != new_b


# ---------------------------------------------------------------- test 14: margin scope
def test_14_margin_loss_only_trains_the_translators():
    m = mem_loop()
    m.train()
    m._targets = torch.zeros(1)                                          # (a training forward: collect on)
    s = m.blocks[0].init_state(m.emb(tokens(2, 9))).requires_grad_()
    (mus, nus, M, q1, margin), sel = m._hier_step(s, read=True, address=True)
    m._targets = None
    assert margin is not None and margin > 0
    params = [p for p in m.parameters() if p.requires_grad]
    grads = torch.autograd.grad(margin, params + [s], allow_unused=True)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    for n, g in zip(names, grads):
        if n.startswith("mem.keys."):
            assert g is not None and g.abs().sum() > 0, n
        else:
            assert g is None or g.abs().sum() == 0, n
    assert grads[-1] is None or grads[-1].abs().sum() == 0               # sg(a): nothing reaches the thought
    assert set(m.mem.margin_stats) == {"active", "active_half", "gap", "gap_half"}


# ---------------------------------------------------------------- test 15: exclusions
def test_15_tables_are_outside_brainopt_clip_carry_and_build_opt():
    from brain_opt import BrainOpt, carry_brain_state
    m = mem_loop()
    mem = m.mem
    tables = {id(getattr(mem, n)) for n in ("V", "G", "S_acc", "V_e", "G_e", "S_e", "P", "P_ema", "G_P", "P_m", "P_v")}
    assert not tables & {id(p) for p in m.parameters()}
    opt = BrainOpt(m, lr=3e-3, wd=0.1)
    assert not tables & {id(p) for p, _ in opt.groups.values()}
    assert all(not n.startswith(("mem.V", "mem.P", "mem.G", "mem.S")) for n in opt.groups)
    kinds = {n: k for n, (_, k) in opt.groups.items()}
    assert all(kinds[n] == "adam" for n in kinds if n.startswith(("mem.keys.", "mem.gates.")))
    assert kinds["mem.w_o"] == "muon"
    x = tokens()
    train_step(m, opt, x)
    ptr = {n: getattr(mem, n).data_ptr() for n in ("V", "G", "V_e", "G_e", "P", "G_P")}
    m.grow_experts(2)                                                   # a birth: the optimizer is rebuilt
    opt = carry_brain_state(opt, m, BrainOpt(m, lr=3e-3, wd=0.1))
    assert {n: getattr(mem, n).data_ptr() for n in ptr} == ptr          # the tables are never rebuilt
    assert not tables & {id(p) for p, _ in opt.groups.values()}
    train_step(m, opt, x)
    # keys: AdamW without weight decay (the same step with wd 0.1 and wd 0)
    m.loop_extra = m.loop_aux = m.blocks[0].mlp.aux = None              # (graph tensors cannot be deep-copied)
    m2 = copy.deepcopy(m)
    o1, o2 = BrainOpt(m, lr=3e-3, wd=0.1), BrainOpt(m2, lr=3e-3, wd=0.0)
    for mod, o in ((m, o1), (m2, o2)):
        for p in mod.mem.keys.parameters():
            p.grad = torch.ones_like(p)
        o.step()
    for (n, p), q in zip(m.mem.keys.named_parameters(), m2.mem.keys.parameters()):
        assert torch.equal(p, q), n
    sd = m.state_dict()
    assert "mem.G" not in sd and "mem.G_e" not in sd and "mem.V" in sd and "mem.P" in sd


# ---------------------------------------------------------------- test 16: checkpoint reload
def test_16_checkpoint_reload_restores_everything():
    from brain_opt import BrainOpt
    from loop_brain import LoopBrain
    from memory import MemoryProbes
    m = mem_loop()
    opt = BrainOpt(m, lr=3e-3)
    x = tokens()
    for i in range(3):
        train_step(m, opt, x, offset=i % 2)
    val = np.random.RandomState(0).randint(1, 50, 600).astype(np.uint16)
    pr = MemoryProbes(val, 16, n_seq=2, n_pos=4, K=4)
    pr.freeze(m)
    for i in range(2):
        train_step(m, opt, x, offset=i % 2)
    met, _ = pr.measure(m)
    assert met["mem_frozen"] == 1 and "mem_stab_w" in met
    m2 = LoopBrain(**copy.deepcopy(m.cfg)).double()
    m2.load_state_dict(m.state_dict(), strict=True)
    mem, mem2 = m.mem, m2.mem
    assert mem2.is_frozen and all(not p.requires_grad for p in mem2.keys.parameters())
    assert mem2._s0 == mem._s0 and mem2._step == mem._step == 5 and mem2._e_live == mem._e_live
    for n, b in mem.named_buffers():
        if n in dict(mem2.named_buffers()) and n not in ("G", "G_e", "G_P", "pend", "pend_e", "qres", "lastq",
                                                         "lastq_ok", "ns_sum", "ns_sq", "ns_cnt"):
            assert torch.equal(b, dict(mem2.named_buffers())[n]), n
    assert mem2.probe_state.keys() == mem.probe_state.keys()
    assert torch.equal(mem2.probe_state["ref"]["B"]["reads"][0]["rows"], mem.probe_state["ref"]["B"]["reads"][0]["rows"])
    for s in m2.spikes():
        s.calibrated.fill_(True)
    m.eval(), m2.eval()
    for K in (2, 5):
        m.K = m2.K = K
        mem.record = dict(pos=torch.arange(34), reads=[], calls=[])
        mem2.record = dict(pos=torch.arange(34), reads=[], calls=[])
        with torch.no_grad():
            assert torch.equal(m(x), m2(x))
        for r1, r2 in zip(mem.record["reads"], mem2.record["reads"]):
            assert torch.equal(r1["rows"], r2["rows"])
        mem.record = mem2.record = None
    met2, _ = MemoryProbes(val, 16, n_seq=2, n_pos=4, K=4).measure(m2)
    assert met2["mem_stab_w"] == met["mem_stab_w"]


# ---------------------------------------------------------------- test 17: memory estimate and an end-to-end smoke
def _saved_bytes(m, x, ckpt):
    total = [0]

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t

    m.mem_ckpt = ckpt
    m.mem.ckpt = ckpt
    m.train()
    m.K, m.aux_k_fixed, m.group_offset = 4, 2, 0
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _, loss = m(x[:, :-1], x[:, 1:])
    return total[0], loss + m.aux_loss()


def test_17a_checkpointed_block_is_exact_and_saves_activation_memory():
    x = tokens(2, 33)
    ma = mem_loop(mem_shared=16, mem_topm=8, mem_topk=6)
    mb = copy.deepcopy(ma)
    plain, la = _saved_bytes(ma, x, False)
    ck, lb = _saved_bytes(mb, x, True)
    assert torch.equal(la, lb)
    la.backward()
    lb.backward()
    for (n, p), q in zip(ma.named_parameters(), mb.parameters()):
        assert (p.grad is None) == (q.grad is None), n
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, atol=1e-12, rtol=1e-9), n
    assert torch.equal(ma.mem.G, mb.mem.G) and torch.equal(ma.mem.G_e, mb.mem.G_e) and torch.equal(ma.mem.G_P, mb.mem.G_P)
    print(f"\nsaved-for-backward bytes (tiny model, B=2, T=32, K=4): plain {plain / 1e6:.2f} MB, checkpointed "
          f"{ck / 1e6:.2f} MB")
    assert ck < 0.85 * plain


def test_17b_end_to_end_cpu_smoke_with_growth_freeze_probes_guard_and_report(tmp_path):
    data = _tiny_data(tmp_path / "data")
    extra = ("--concepts 2,4,8,16 --mem-shared 8 --mem-topm 4 --mem-topk 3 --mem-private 8 --mem-dv 16 "
             "--mem-write-start 5 --mem-freeze 0.5 --mem-probes 2 --mem-repair 0.25 --mem-guard 0.999 --eval-every 40").split()
    run, recs, ck = _run_main(current().train_lm, tmp_path / "runs", "mem", data, extra)
    log = (run / "train.log").read_text()
    assert "MEMORY FROZEN" in log and "(rows the same: True)" in log and "memory knocked out" in log
    last = recs[-1]
    assert last["mem_frozen"] == 1 and "mem_stab_w" in last and "mem_input_drift_by_level" in last
    assert "mem_offset_overlap" in last and len(last["mem_collisions_by_level"]) == 4
    assert "loss_mem_ko_shared" in last and "loss_mem_ko_private" in last and "loss_mem_ko_level4" in last
    assert len(recs) == 2 and last["mem_freeze_step"] >= 20 and all(math.isfinite(r["val"]) for r in recs)
    assert ("MEMORY DRIFT" in log) == bool(last["mem_alarm"]) or any(r.get("mem_alarm") for r in recs)
    from loop_brain import LoopBrain
    m2 = LoopBrain(**ck["cfg"])
    m2.load_state_dict(ck["state"], strict=True)
    assert m2.mem.is_frozen and m2.blocks[0].mlp.E == 12 and m2.mem._e_live == 12
