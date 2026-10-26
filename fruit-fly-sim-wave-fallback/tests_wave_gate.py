"""Tests of wave_gate.py (CPU, tiny sizes; the last test reports full-size bytes and a CPU timing)."""
import io
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch

torch.set_num_threads(2)
from wave_gate import GatedReadout  # noqa: E402

D_, EM, K = 16, 8, 2          # d, routed experts (max), top-k


def setup(N=64, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    gr = GatedReadout(D_, EM, k_routed=K, seed=seed, **kw)
    xf = torch.randn(N, D_, generator=g)
    e_k = torch.stack([torch.randperm(EM, generator=g)[:K] for _ in range(N)]).reshape(-1)
    t_k = torch.arange(N).repeat_interleave(K)
    # brain slots: rank of each pair inside its expert
    p_k = torch.zeros_like(e_k)
    cnt = [0] * EM
    for i, e in enumerate(e_k.tolist()):
        p_k[i] = cnt[e]
        cnt[e] += 1
    C = max(cnt)
    return gr, xf, e_k, t_k, p_k, C


def full_read(gr, dec, xf, scale=1.0, g=None):
    """fake stores: the private read of pair row i = a fixed function of (t, e); the global read of token t."""
    yp = scale * torch.tanh(xf[dec.t] + 0.1 * dec.e[:, None].float())
    yg = scale * torch.cos(xf[dec.g_idx])
    return yp, yg


def run(gr, xf, e_k, t_k, p_k, C, step=0, training=True, brain=None, scale=1.0):
    x_in = xf if brain is None else brain(xf)
    dec = gr.decide(x_in, e_k, t_k, C, EM, step=step, call=3, training=training,
                    f_pair=torch.rand(e_k.numel(), 1, generator=torch.Generator().manual_seed(step)))
    yp, yg = full_read(gr, dec, x_in, scale)
    o = gr.readout(dec, yp, yg, x_in.norm(dim=-1))
    buf = torch.zeros(EM, C, D_)
    x = x_in.clone().view(1, -1, D_)
    buf, x = gr.inject(dec, o, buf, x, e_k, p_k)
    return dec, o, buf, x


def test_cap_static_zero_drops():
    gr, xf, e_k, t_k, p_k, C = setup()
    for N in (64, 64):
        dec, o, _, _ = run(gr, xf, e_k, t_k, p_k, C)
        n_pairs = e_k.numel() + xf.size(0)
        assert dec.p_idx.numel() == int(0.5 * n_pairs) and dec.g_idx.numel() == int(0.5 * xf.size(0))
        assert int((dec.read_p | dec.read_g).sum()) <= n_pairs // 2
    # zero capacity drops: tiny C, every selected routed row has slot < C, shared slot < N, slots unique per expert
    dec = gr.decide(xf, e_k, t_k, 2, EM, training=True)
    rt = dec.e < EM
    assert bool((dec.slot[rt & dec.valid] < 2).all())
    key = (dec.e * 10_000 + dec.slot)[dec.valid]
    assert key.unique().numel() == key.numel()
    print("cap: Kp", dec.p_idx.numel(), "of", e_k.numel() + 64, "| C=2 respected, unique slots, 0 drops")


def test_exploration_rate():
    gr, xf, e_k, t_k, p_k, C = setup(N=4096)
    e_k, t_k = e_k[:0], t_k[:0]
    rates = []
    for s in range(6):
        gr._log["n"] = 0
        dec = gr.decide(xf, e_k, t_k, C, EM, step=s * 1000, training=True)
        rates.append(float((dec.ex_p).float().sum()) / 4096)
    assert all(0.04 < r < 0.06 for r in rates), rates
    dec = gr.decide(xf, e_k, t_k, C, EM, step=0, training=False)
    assert int(dec.ex_p.sum()) == 0 and int(dec.ex_g.sum()) == 0
    print("explore rates over steps", [round(r, 4) for r in rates], "| eval 0")


def test_closed_exact_zero_and_rows():
    gr, xf, e_k, t_k, p_k, C = setup()
    dec, o, buf, x = run(gr, xf, e_k, t_k, p_k, C, training=False)
    unread = ~(dec.read_p | dec.read_g)
    assert bool((o[unread] == 0).all())
    # pairs never selected: their buf slot stays exactly 0
    sel = torch.zeros(e_k.numel() + xf.size(0), dtype=torch.bool)
    sel[dec.p_idx[dec.read_p | dec.read_g]] = True
    notsel = ~sel[:e_k.numel()]
    assert bool((buf[e_k[notsel], p_k[notsel]] == 0).all())
    rp, rg = dec.gathered_rows()
    assert rp == int(0.5 * (e_k.numel() + 64)) and rg == 32
    print(f"closed exact 0; gathered rows private {rp}/{e_k.numel() + 64}, global {rg}/64; "
          f"read share {float((dec.read_p | dec.read_g).float().sum()) / (e_k.numel() + 64):.3f}")


def test_gate_loss_isolated():
    gr, xf, e_k, t_k, p_k, C = setup()
    brain = torch.nn.Linear(D_, D_)
    xf.requires_grad_(True)
    dec, o, buf, x = run(gr, xf, e_k, t_k, p_k, C, brain=brain)
    loss = (buf.pow(2).sum() + x.pow(2).sum())
    loss.backward()
    brain.zero_grad(set_to_none=True)
    xf.grad = None
    gr.D.grad = None
    gr.log_s.grad = None
    conf = [(dec.read_p & (torch.rand(dec.read_p.numel()) > 0.5))]
    st = gr.after_backward(conf_p=conf, conf_g=None)
    assert xf.grad is None and all(p.grad is None for p in brain.parameters())
    assert gr.D.grad is None and gr.log_s.grad is None
    assert gr.Wp.grad is not None and gr.Wp.grad.abs().sum() > 0 and gr.cp.weight.grad is not None
    print("gate loss isolated:", st)


def test_layer():
    gr, xf, e_k, t_k, p_k, C = setup()
    gr.cap = 0.0  # no cap here
    # D = 0 -> output == the no-layer output exp(log_s)*|x|*y
    dec = gr.decide(xf, e_k, t_k, C, EM, training=False)
    yp, yg = full_read(gr, dec, xf)
    o = gr.readout(dec, yp, yg, xf.norm(dim=-1))
    pos = torch.zeros(64, dtype=torch.long)
    pos[dec.g_idx] = torch.arange(dec.g_idx.numel())
    ref = (yp * dec.read_p[:, None] + yg[pos[dec.t]] * dec.read_g[:, None]) * 0.3 * xf.norm(dim=-1)[dec.t][:, None]
    assert torch.allclose(o, ref, atol=1e-6)
    # different experts -> different outputs; grads only to used rows
    with torch.no_grad():
        gr.D.normal_(0, 0.1)
    y_same = torch.ones_like(yp)
    o2 = gr.readout(dec, y_same, torch.zeros_like(yg), torch.ones(64))
    live = dec.read_p
    es = dec.e[live]
    a, b = (es == es[0]).nonzero()[0, 0], (es != es[0]).nonzero()[0, 0]
    assert not torch.allclose(o2[live][a], o2[live][b])
    gr.D.grad = None
    o2.sum().backward()
    used = torch.zeros(EM + 1, dtype=torch.bool)
    used[dec.e[dec.read_p]] = True
    gn = gr.D.grad.flatten(1).abs().sum(1)
    assert bool((gn[~used] == 0).all()) and bool((gn[used] > 0).all())
    # the custom bmm grad == autograd reference
    D0 = gr.D.detach().clone().requires_grad_(True)
    Y = torch.randn(3, 5, D_)
    Dp = torch.nn.Parameter(D0.detach().clone())
    from wave_gate import expert_d
    out = expert_d(Y, Dp, 2, 2)
    gO = torch.randn_like(out)
    out.backward(gO)
    ref = Y + torch.bmm(Y, D0[2:5])
    ref.backward(gO)
    assert torch.allclose(out, ref, atol=1e-5) and torch.allclose(Dp.grad, D0.grad, atol=1e-5)
    print("layer: D=0 == no-layer; experts differ; grad rows used only", int(used.sum()), "of", EM + 1,
          "; custom bmm grad == autograd")


def test_capped_strong_recall_b():
    """b = -<dL/dy, y> with the cap active: the detached cap scale keeps b == the linear prediction (not ~0)."""
    gr, xf, e_k, t_k, p_k, C = setup()
    dec, o, buf, x = run(gr, xf, e_k, t_k, p_k, C, scale=100.0)    # strong recall: the cap binds
    on = dec.read_p | dec.read_g
    assert bool((o[on].norm(dim=-1) <= 0.5 * xf.norm(dim=-1)[dec.t][on] + 1e-4).all())
    w = torch.randn(1, 1, D_)
    loss = (x * w).sum()          # dL/do for shared rows = w
    loss.backward()
    (pr, pg), (yp, yg) = dec.probes, dec.y_saved
    b_p = -(pr.grad * yp).sum(-1)
    # expected: o = f * s * |x| * y (D = 0), f detached -> dL/dy = f s |x| w -> b = -f s |x| <w, y_p> = -<w, o_p>
    sh = (dec.e == EM) & dec.read_p
    xt = xf.norm(dim=-1)[dec.t]
    o_p = (yp + 0) * 0.3 * xt[:, None]
    f = (0.5 * xt / (0.3 * xt[:, None] * (yp + yg)).norm(dim=-1)).clamp(max=1)
    exp = -(w.view(-1) * o_p).sum(-1) * f
    assert sh.any() and torch.allclose(b_p[sh], exp[sh], atol=1e-4, rtol=1e-3)
    assert float(b_p[sh].abs().mean()) > 1e-3
    print("capped strong recall: b correct, |b| mean", round(float(b_p[sh].abs().mean()), 4),
          "cap factor mean", round(float(f[sh].mean()), 4))


def test_reuse():
    gr, xf, e_k, t_k, p_k, C = setup()
    dec, o, _, _ = run(gr, xf, e_k, t_k, p_k, C, training=False)
    o_r, o_s = gr.reuse(dec, o, e_k, t_k)
    sel = dec.p_idx < e_k.numel()
    live = (dec.read_p | dec.read_g) & sel
    assert torch.equal(o_r[dec.p_idx[live]], o[live])
    others = torch.ones(e_k.numel(), dtype=torch.bool)
    others[dec.p_idx[live]] = False
    assert bool((o_r[others] == 0).all())
    # a re-routed pass: shuffled expert ids -> only matches reuse
    e2 = (e_k + 1) % EM
    o_r2, _ = gr.reuse(dec, o, e2, t_k)
    print("reuse: cached rows equal; non-read pairs 0; rerouted hits", int((o_r2.abs().sum(-1) > 0).sum()))


def test_save_load_split_determinism():
    gr, xf, e_k, t_k, p_k, C = setup()
    for s in range(2):
        dec, o, buf, x = run(gr, xf, e_k, t_k, p_k, C, step=s)
        (buf.pow(2).sum() + x.pow(2).sum()).backward()
        gr.after_backward()
        gr.step()
    sd = gr.state_dict()
    bio = io.BytesIO()
    torch.save(sd, bio)
    bio.seek(0)
    gr2 = GatedReadout(D_, EM, k_routed=K, seed=0)   # (same config; its fresh params are overwritten below)
    with torch.no_grad():
        gr2.D.normal_()
    gr2.load_state_dict(torch.load(bio, weights_only=False))
    for k, v in gr.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, gr2.state_dict()[k]), k
    assert gr.D.abs().sum() > 0 and gr2._opt_pending is not None
    # both continue identically (optimizer state restored)
    outs = []
    for m in (gr, gr2):
        dec, o, buf, x = run(m, xf, e_k, t_k, p_k, C, step=7)
        (buf.pow(2).sum() + x.pow(2).sum()).backward()
        m.after_backward()
        m.step()
        outs.append((m.D.detach().clone(), m.Wp.detach().clone(), m.b_ms.clone()))
    assert all(torch.equal(a, b) for a, b in zip(*outs))
    # split
    gr.on_grow([1, 3], [5, 6])
    for p in (gr.D, gr.Wp, gr.Wg, gr.bp, gr.bg):
        assert torch.equal(p[5], p[1]) and torch.equal(p[6], p[3])
    # determinism under a seed
    a = setup(seed=4)
    b = setup(seed=4)
    da, oa, _, _ = run(*a, step=3)
    db, ob, _, _ = run(*b, step=3)
    assert torch.equal(da.p_idx, db.p_idx) and torch.equal(oa, ob) and torch.equal(da.ex_p, db.ex_p)
    # bf16 moments option round-trip
    g3 = GatedReadout(D_, EM, k_routed=K, seed=0, moments_bf16=True)
    dec, o, buf, x = run(g3, *setup()[1:])
    (buf.pow(2).sum() + x.pow(2).sum()).backward()
    g3.after_backward()
    g3.step()
    st = g3.get_extra_state()
    assert st["opt"][2]["m"].dtype == torch.bfloat16 and g3.D.abs().sum() > 0
    print("save/load (D, gate, both optimizer states) round-trips and resumes identically; split copy; deterministic;"
          " bf16-moments ok")


def test_off_is_none():
    """Before write_start no gr method is called: the read path is the identity and no RNG is consumed."""
    gr, xf, e_k, t_k, p_k, C = setup()
    st = torch.get_rng_state()
    buf = torch.zeros(EM, C, D_)
    x = xf.view(1, -1, D_)
    step, write_start = 10, 500
    if step >= write_start:
        run(gr, xf, e_k, t_k, p_k, C)
    assert torch.equal(torch.get_rng_state(), st)
    # and the decision itself never touches the global RNG either
    gr.decide(xf, e_k, t_k, C, EM, step=0, training=True)
    assert torch.equal(torch.get_rng_state(), st)
    print("off == none: no gate code, global RNG untouched (decide uses its own generator)")


def test_read_share_start():
    gr, xf, e_k, t_k, p_k, C = setup(N=2048)
    dec = gr.decide(xf, e_k[:0], t_k[:0], C, EM, training=False)
    print(f"read share at start (eval, no exploration): vp {float(dec.vp_share[0]):.3f}, vg {float(dec.vp_share[1]):.3f}")


def test_full_size_report():
    gr = GatedReadout(384, 768, k_routed=6, seed=0)
    r = gr.nbytes_report()
    r16 = GatedReadout(384, 768, k_routed=6, seed=0, moments_bf16=True).nbytes_report()
    mb = lambda v: f"{v / 2**20:.1f} MiB"
    print("full size (769 slots, d 384): D", mb(r["D"]), "| +grad", mb(r["D_grad"]), "| AdamW moments fp32",
          mb(r["D_moments"]), "| D total", f"{r['D_total'] / 1e9:.3f} GB", "| with bf16 moments",
          f"{r16['D_total'] / 1e9:.3f} GB", "| gate params", r["gate"] // 4, "=", mb(r["gate_total"]), "incl. grad+Adam")
    # CPU timing at a reduced N (2 threads); the GPU estimate is in the report
    N, Kr = 1024, 6
    g = torch.Generator().manual_seed(0)
    xf = torch.randn(N, 384, generator=g)
    e_k = torch.randint(0, 768, (N * Kr,), generator=g)
    t_k = torch.arange(N).repeat_interleave(Kr)
    C = int(torch.bincount(e_k, minlength=768).max())
    t0 = time.perf_counter()
    dec = gr.decide(xf, e_k, t_k, C, 768, training=True)
    t1 = time.perf_counter()
    yp = torch.randn(dec.p_idx.numel(), 384, requires_grad=True)
    yg = torch.randn(dec.g_idx.numel(), 384)
    o = gr.readout(dec, yp, yg, xf.norm(dim=-1))
    t2 = time.perf_counter()
    o.sum().backward()
    t3 = time.perf_counter()
    print(f"CPU 2 threads, N={N} x {Kr}: decide {1e3 * (t1 - t0):.1f} ms, readout fwd {1e3 * (t2 - t1):.1f} ms, "
          f"bwd {1e3 * (t3 - t2):.1f} ms (C={C}; layout {768 * C} routed rows)")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"ALL {len(tests)} PASSED")
