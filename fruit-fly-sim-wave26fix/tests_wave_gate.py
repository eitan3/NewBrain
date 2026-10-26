"""wave_gate.py v2.1 tests (CPU, 2 threads, tiny preset: d 16, 8 routed + shared, B2xT64): U21 U23 U24 U26 U39,
split copy, save/load, full-size bytes."""
import io
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_num_threads(2)
from wave_gate import SH, GatedReadout, Translate, tkey  # noqa: E402

D_, EL, N = 16, 8, 128
BAD = ("_local_scalar_dense", "nonzero", "masked_select", "unique")


class SyncGuard(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.hits = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if any(b in name for b in BAD):
            self.hits.append(name)
        return func(*args, **(kwargs or {}))


def make(seed=0, **kw):
    gr = GatedReadout(D_, seed=seed, **kw)
    g = torch.Generator().manual_seed(seed + 1)
    xf = torch.randn(N, D_, generator=g)
    e_p = torch.stack([torch.randperm(EL, generator=g)[:6] for _ in range(N)])
    row_of = torch.full((SH + 1,), -1, dtype=torch.long)
    row_of[:EL] = torch.arange(EL)
    row_of[SH] = SH
    f_pair = torch.rand(N, 6, generator=g)
    return gr, xf, e_p, row_of, f_pair


def reads(dec, xf, scale=1.0):
    y_g = scale * torch.cos(xf[dec.GT])
    y_p = scale * torch.tanh(xf[dec.tok[dec.PR]] + 0.1 * dec.rows.view(-1)[dec.S[dec.PR]][:, None].float())
    return y_g, y_p


def test_U21():
    gr, xf, e_p, row_of, fp = make()
    with torch.no_grad():
        gr.bp_b.add_(0.02)   # more wanting pairs, so caps / budgets bind
        gr.bg_b.add_(0.02)
    guard = SyncGuard()
    with guard:
        dec = gr.decide(xf, e_p, row_of, EL, step=3, micro=1, training=True, f_pair=fp)
        y_g, y_p = reads(dec, xf)
        gr.readout(dec, y_g, y_p, xf.norm(dim=-1))
    assert not guard.hits, guard.hits
    assert dec.S.numel() == dec.K == 7 * N // 2
    assert int(dec.xp.sum()) == dec.K_xp and int(dec.xg.sum()) == dec.K_xg
    # all exploration kept: xp pairs in S with vp; xg tokens gused with all wanted pairs kept
    inS = torch.zeros(7 * N, dtype=torch.bool)
    inS[dec.S[dec.valid]] = True
    assert bool(inS[dec.xp.view(-1)].all()) and bool(dec.vp_d.view(-1)[dec.xp.view(-1)].all())
    assert bool(dec.gused[dec.xg].all())
    # per-group count <= cap, valid slots unique, shared pair never private
    g = dec.grp_S[dec.valid]
    cnt = torch.bincount(g, minlength=EL + 1)
    assert bool((cnt[:EL] <= dec.C).all()) and int(cnt[EL]) <= dec.C_sh
    sl = dec.slot[dec.valid]
    assert sl.unique().numel() == sl.numel()
    assert not bool(dec.vp_d[:, 6].any())
    assert int(dec.vp.sum()) <= dec.K_pr
    # gused subset of gsel; every vg token in GT
    assert not bool((dec.gused & ~dec.gsel).any())
    inGT = torch.zeros(N, dtype=torch.bool)
    inGT[dec.GT[dec.gvalid]] = True
    assert bool(inGT[dec.tok[dec.vg]].all())
    # same seed -> same S; eval -> no exploration
    dec2 = gr.decide(xf, e_p, row_of, EL, step=3, micro=1, training=True, f_pair=fp)
    assert torch.equal(dec.S, dec2.S) and torch.equal(dec.PR, dec2.PR)
    de = gr.decide(xf, e_p, row_of, EL, training=False, f_pair=fp)
    assert de.K_xp == 0 and int(de.xp.sum()) == 0 and int(de.xg.sum()) == 0
    # ties -> the lower index
    x = torch.tensor([1.0, 3.0, 3.0, 2.0, 3.0, -0.0, 0.0])
    assert torch.topk(tkey(x), 2).indices.tolist() == [1, 2]
    assert torch.argsort(tkey(torch.tensor([-1.0, -2.0, float("-inf"), float("inf")])), descending=True).tolist() == [3, 0, 1, 2]
    print(f"U21 ok: |S|={dec.K}, K_xp={dec.K_xp}, K_xg={dec.K_xg}, C={dec.C}, C_sh={dec.C_sh}, private reads "
          f"{int(dec.vp.sum())}/{dec.K_pr}, global tokens {int(dec.gvalid.sum())}/{dec.K_g}, excluded {int(dec.n_excluded)},"
          f" demoted {int(dec.n_demoted)}; no sync ops")


def test_U23():
    gr, xf, e_p, row_of, fp = make()
    brain = torch.nn.Linear(D_, D_)
    xin = xf.clone().requires_grad_(True)
    h = brain(xin)

    def fwd():
        dec = gr.decide(h, e_p, row_of, EL, step=1, training=True, f_pair=fp)
        y_g, y_p = reads(dec, h.detach())
        rp, rs = gr.readout(dec, y_g, y_p, h.norm(dim=-1))
        return dec, rp, rs

    dec, rp, rs = fwd()
    tgt = torch.randn_like(rp.float())
    loss = ((rp.float() - tgt).square().sum() + rs.float().square().sum()) + h.square().sum()
    g_brain = torch.autograd.grad(loss, list(brain.parameters()) + [gr.log_s], retain_graph=True)
    assert all(p.grad is None for p in gr.gate_parameters())
    # d(sum out)/d(xf) = 0
    gx = torch.autograd.grad((rp.float().sum() + rs.float().sum()), xin, retain_graph=True, allow_unused=True)[0]
    assert gx is None or float(gx.abs().sum()) == 0
    gr.D_grad.zero_()
    loss.backward()
    assert g_brain[2].abs() > 0 and gr.log_s.grad is not None
    read_rows = torch.zeros(SH + 1, dtype=torch.bool)
    read_rows[dec.grp_S[dec.vp | dec.vg].clamp(max=EL)] = True
    rowmap = torch.cat([torch.arange(EL), torch.tensor([SH])])
    rn = gr.D_grad[rowmap].flatten(1).abs().sum(1)
    used = torch.zeros(EL + 1, dtype=torch.bool)
    used[dec.grp_S[dec.vp | dec.vg]] = True
    assert bool((rn[used] > 0).all()) and bool((rn[~used] == 0).all())
    assert gr.D.grad is None and not gr.D.requires_grad
    bg = [p.grad.clone() for p in brain.parameters()]
    st = gr.gate_loss(dec, conf_p=dec.vp & (torch.arange(dec.K) % 2 == 0), conf_g=dec.gused)
    assert all(torch.equal(a, p.grad) for a, p in zip(bg, brain.parameters()))
    assert xin.grad is not None  # (from the main loss only) ...
    # ... and the gate loss alone has no path to xf / the brain
    dec2, rp2, rs2 = fwd()
    ((rp2.float() - tgt).square().sum() + rs2.float().square().sum()).backward(retain_graph=True)
    xin.grad = None
    for p in brain.parameters():
        p.grad = None
    for p in gr.gate_parameters():
        p.grad = None
    gr.gate_loss(dec2)
    assert xin.grad is None and all(p.grad is None for p in brain.parameters())
    assert gr.Wp.grad is not None and float(gr.bg_b.grad.abs().sum()) > 0
    print("U23 ok: gate grads only from the gate loss; brain grads unchanged; D_grad rows only for read experts; "
          f"no D.grad; d out/d xf = 0; stats read_share {float(st['read_share']):.3f}")


def test_U24():
    for s, sc, tag in ((1e-5, 1.0, "inactive"), (1.0, 3.0, "active")):
        gr, xf, e_p, row_of, fp = make(log_s0=float(torch.log(torch.tensor(s))))
        with torch.no_grad():
            gr.D.normal_(0, 0.05)
        dec = gr.decide(xf, e_p, row_of, EL, step=2, training=True, f_pair=fp)
        y_g, y_p = reads(dec, xf, sc)
        tgt = torch.randn(N, 7, D_)
        rp, rs = gr.readout(dec, y_g, y_p, xf.norm(dim=-1), out_dtype=torch.float32)
        o = dec.r3
        o.retain_grad()
        L = 0.5 * (torch.cat([rp, rs[:, None]], 1) - tgt).square().sum()
        L.backward()
        gr.gate_loss(dec)
        lhs = (dec.b_p + dec.b_g).double().sum()
        rhs = -(o.grad.double() * o.double()).sum()
        assert abs(float(lhs - rhs)) <= 1e-6 * abs(float(rhs)) + 1e-9, (float(lhs), float(rhs))
        rel = None
        if tag == "inactive":   # first order: b ~ L(0) - L(y)
            ov = o.detach()[:-1].double().view(N, 7, D_)
            dL = float((ov * tgt.double() - 0.5 * ov.square()).sum())     # L(0) - L(y), exact in fp64
            rel = abs(dL - float(lhs)) / abs(dL)
            assert rel < 0.1
        cf = float((o[:-1].norm(dim=-1) / (0.5 * xf.norm(dim=-1).repeat_interleave(7))).max())
        print(f"U24 ok ({tag}): sum b = {float(lhs):.6g}, -<dL/do,o> = {float(rhs):.6g}, max |o|/cap {cf:.3f}"
              + (f", vs L(0)-L(y) rel {rel:.4f}" if rel is not None else ""))


def test_U26():
    torch.manual_seed(0)
    E, C, Cs = 5, 4, 6
    D = torch.randn(SH + 1, D_, D_) * 0.1
    Xr = torch.randn(E, C, D_, requires_grad=True)
    Xs = torch.randn(Cs, D_, requires_grad=True)
    gb = torch.zeros_like(D)
    outr, outs = Translate.apply(Xr, Xs, D, E, gb)
    gr_, gs_ = torch.randn_like(outr), torch.randn_like(outs)
    (outr * gr_).sum().backward(retain_graph=True)
    (outs * gs_).sum().backward()
    # naive per pair (I + D_e), fp64
    Dd = D.double().requires_grad_(True)
    Xr64, Xs64 = Xr.detach().double().requires_grad_(True), Xs.detach().double().requires_grad_(True)
    eye = torch.eye(D_, dtype=torch.float64)
    rr = torch.stack([torch.stack([Xr64[e, c] @ (eye + Dd[e]) for c in range(C)]) for e in range(E)])
    rs = torch.stack([Xs64[i] @ (eye + Dd[SH]) for i in range(Cs)])
    ((rr * gr_.double()).sum() + (rs * gs_.double()).sum()).backward()
    rel = lambda a, b: float((a.double() - b).norm() / b.norm().clamp_min(1e-30))
    assert rel(outr, rr) <= 1e-5 and rel(outs, rs) <= 1e-5
    assert rel(gb[:E], Dd.grad[:E]) <= 1e-5 and rel(gb[SH], Dd.grad[SH]) <= 1e-5
    assert float(gb[E:SH].abs().sum()) == 0
    assert rel(Xr.grad, Xr64.grad) <= 1e-6 and rel(Xs.grad, Xs64.grad) <= 1e-6
    Xr2 = Xr.detach().requires_grad_(True)
    o2, _ = Translate.apply(Xr2, Xs.detach(), D, E, None)
    o2.sum().backward()
    assert Xr2.grad is not None and D.grad is None
    print("U26 ok: Translate == naive (I+D_e) fwd/grad D/grad y; grad_buf=None works; D never gets .grad")


def test_U39():
    gr = GatedReadout(D_, seed=0)
    E = 6
    torch.manual_seed(1)
    D0 = gr.D.clone()          # (D = 0: the reference compares the update itself at fp32 resolution)
    G = torch.zeros_like(gr.D)
    G[[0, 2, 5, SH]] = torch.randn(4, D_, D_)
    for it in range(3):
        gr.D_grad.copy_(G * (it + 1))
        guard = SyncGuard()
        with guard:
            gr.d_step(E, chunk=4)
        assert not guard.hits, guard.hits
        assert float(gr.D_grad.abs().sum()) == 0
    for r in (1, 3, 4, 6, 7, 100):
        assert torch.equal(gr.D[r], D0[r])
    # fp64 reference
    b2, eps, lr, d = 0.99, 1e-30, 1e-4, D_
    Dr, R, Cv = D0.double().clone(), torch.zeros(SH + 1, d, dtype=torch.float64), torch.zeros(SH + 1, d, dtype=torch.float64)
    for it in range(3):
        for r in (0, 2, 5, SH):
            g = (G[r] * (it + 1)).double()
            R[r] = b2 * R[r] + (1 - b2) * g.norm(dim=-1).square() / d
            Cv[r] = b2 * Cv[r] + (1 - b2) * g.norm(dim=-2).square() / d
            bc = 1 - b2 ** (it + 1)
            rh, ch = R[r] / bc, Cv[r] / bc
            V = rh[:, None] * ch[None, :] / max(float(rh.mean()), eps) + eps
            U = g / V.sqrt()
            rms = float(U.norm() / d)
            Dr[r] -= lr * U / max(rms / 1.0, 1.0)
    err = float((gr.D.double() - Dr).abs().max() / (Dr - D0.double()).abs().max())
    assert err < 1e-6, err
    # independence: another row's grad doesn't change row 0's update
    g2 = GatedReadout(D_, seed=0)
    g2.D.copy_(D0)
    G2 = G.clone()
    G2[2] = G2[2] * 7.0
    G2[3] = torch.randn(D_, D_)
    for it in range(3):
        g2.D_grad.copy_(G2 * (it + 1))
        g2.d_step(E, chunk=4)
    assert torch.equal(g2.D[0], gr.D[0]) and torch.equal(g2.D[5], gr.D[5])
    print(f"U39 ok: zero-grad rows unchanged; rows independent; vs fp64 ref max rel err {err:.2e}; no syncs")


def test_split_saveload():
    gr, xf, e_p, row_of, fp = make()
    opt = torch.optim.AdamW(gr.param_groups(), betas=(0.9, 0.99), weight_decay=0.0)
    dec = gr.decide(xf, e_p, row_of, EL, step=1, training=True, f_pair=fp)
    y_g, y_p = reads(dec, xf)
    rp, rs = gr.readout(dec, y_g, y_p, xf.norm(dim=-1))
    (rp.float().square().sum() + rs.float().square().sum()).backward()
    gr.gate_loss(dec)
    opt.step()
    gr.d_step(EL)
    # split: parent rows 1, 3 -> children 8, 3 (a child reusing its parent row)
    gr.on_grow([1, 3], [8, 3], opt)
    for name in ("D", "R", "Cv", "n", "Wp", "Wg", "bp_b", "bg_b"):
        t = getattr(gr, name)
        assert torch.equal(t[8], t[1]), name
    assert torch.equal(opt.state[gr.Wp]["exp_avg"][8], opt.state[gr.Wp]["exp_avg"][1])
    assert float(gr.R[1].abs().sum()) > 0
    # save / load
    bio = io.BytesIO()
    torch.save(dict(gr=gr.state_dict(), opt=opt.state_dict()), bio)
    bio.seek(0)
    st = torch.load(bio, weights_only=False)
    assert "D_grad" not in st["gr"] and all(k in st["gr"] for k in ("D", "R", "Cv", "n", "sigma_b"))
    g2 = GatedReadout(D_, seed=5)
    g2.load_state_dict(st["gr"])
    for k, v in gr.state_dict().items():
        assert torch.equal(v, g2.state_dict()[k]), k
    print("split copy (D/R/Cv/n + gate rows + AdamW moments, child reusing parent) ok; save/load round-trips "
          "(D_grad not saved)")


def test_bytes():
    S, d = SH + 1, 384
    D = S * d * d * 4
    small = 2 * S * d * 4 + S * 8
    print(f"full size: D {D / 2**20:.1f} MiB + D_grad {D / 2**20:.1f} MiB + R/Cv/n {small / 2**10:.0f} KiB = "
          f"{(2 * D + small) / 1e9:.3f} GB (memory file: D + R/Cv/n = {(D + small) / 1e9:.3f} GB)")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"ALL {len(tests)} PASSED")
