"""CPU microbenchmark of the memory speed package (plan_v2/14), not a test: the code before it
(plan_v2/baseline_mem_v2) vs now, time per training step on a mid-size CPU config and the isolated ops at the real
sizes (select, private top-4, MemRead backward, MemOpt on a 1M x 256 table, private-key Adam, balancing at the
step-500 counts). Every new result is checked against the old one (bit-identical, or allclose for dw).

    CUDA_VISIBLE_DEVICES=-1 python -B bench_speed.py [--quick]

CPU only: the ratios say little about the GPU (sorts, gathers and launches scale differently there).
"""
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from test_speed import baseline, current

QUICK = "--quick" in sys.argv


def tm(fn, reps=3, setup=None):
    """Median wall time of fn() over reps runs (after one warm-up); setup() runs untimed before each."""
    ts = []
    for r in range(reps + 1):
        if setup is not None:
            setup()
        t = time.perf_counter()
        fn()
        if r:
            ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def line(name, old, new, note=""):
    print(f"{name:58s} old {old * 1e3:9.1f} ms   new {new * 1e3:9.1f} ms   x{old / max(new, 1e-9):5.2f}  {note}",
          flush=True)


# ---------------------------------------------------------------- one training step, mid-size CPU config
def train_step_bench(mods, new, steps, mild=False):
    LoopBrain, BrainOpt = mods.loop_brain.LoopBrain, mods.brain_opt.BrainOpt
    torch.manual_seed(0)
    m = LoopBrain(512, 128, 32, 4, 32, moe=dict(n_exp=16, k=2, hidden=64, shared=64, capacity=1.5,
                                                birth_mode="split"),
                  spike="graded", chunk=64, k_mean=4, k_min=4, k_max=4, bptt=3, state="", refractory=1, slots=4,
                  waves=True, ws_direct=True, head_group=2, head_groups=2, deep_sup=0.3, halt=True, think_w=0.1,
                  think_fb=True, think_use=True, think_target="latent", concepts=[16, 32, 64, 128], mem_shared=128,
                  mem_private=64, mem_dv=64, mem_topm=8, mem_topk=6, mem_max_experts=16)
    for s in m.spikes():
        s.calibrated.fill_(True)
    opt = BrainOpt(m, lr=1e-3)
    x = torch.randint(0, 512, (4, 257), generator=torch.Generator().manual_seed(1))
    mem = m.mem
    if mild:  # M1 + M2 (M3 acts only on balancing steps)
        m.mem_readdr_every, m.mem_margin_passes = 2, 1
    ts = []
    for i in range(steps + 2):
        t = time.perf_counter()
        m.train()
        m.K, m.aux_k_fixed, m.group_offset = 4, 2, i % 2
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        (loss + m.aux_loss()).backward()
        opt.step()
        m.blocks[0].mlp.apply_balance()
        if new:
            cnt = mem.write_counts()
            li = torch.cat([loss.detach().double().view(1), cnt.double()]).tolist()
            mem.after_step(1e-3, [int(v) for v in li[1:]])
        else:
            mem.after_step(1e-3)
            loss.item()
        if i >= 2:
            ts.append(time.perf_counter() - t)
    return float(np.median(ts)), loss.detach()


def main():
    torch.manual_seed(0)
    old, new = baseline(), current()
    OM, NM = old.memory, new.memory
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, CPU only\n", flush=True)

    # ---- training step (N = 4 x 256 tokens, K = 4, S = 128, R = 64, 16 experts; writes on, infancy)
    n = 2 if QUICK else 4
    to, lo = train_step_bench(old, False, n)
    tn, ln = train_step_bench(new, True, n)
    line("training step (mid-size: N=1024, K=4, S=128, R=64)", to, tn, f"loss old {lo.item():.4f} new {ln.item():.4f}")
    tm_, _ = train_step_bench(new, True, n, mild=True)
    line("training step, new + --mem-readdr-every 2 --mem-margin-passes 1", to, tm_, "(mild: changes the model)")

    # ---- select (N = 8192, L = 4, H = 3, S = 512, m = 8, km = 6)
    N = 2048 if QUICK else 8192
    mo = OM.Memory(384, 64, 4, H=3, S=512, m=8, km=6, dv=1)
    mn = NM.Memory(384, 64, 4, H=3, S=512, m=8, km=6, dv=1)
    mn.load_state_dict(mo.state_dict())
    mus = [torch.randn(1, N, 384) for _ in range(4)]
    pbar = torch.randn(1, N, 64)
    a, a_hat = mo.prep(mus, pbar)
    so, sn = mo.select(a, a_hat, False), mn.select(a, a_hat, False)
    assert all(torch.equal(so[k], sn[k]) for k in so)
    line(f"select (N={N}, L=4, H=3, S=512, m=8, km=6)", tm(lambda: mo.select(a, a_hat, False)),
         tm(lambda: mn.select(a, a_hat, False)), "identical")
    del mo, mn

    # ---- private top-4 over the capacity layout (E x C = 192 x 384 slots, 2/3 kept, R = 512)
    E, C, R = 192, 384, 512
    sc = torch.randn(E, C, R)
    keep = torch.randperm(E * C, generator=torch.Generator().manual_seed(2))[:E * C * 2 // 3].sort().values
    e_k, p_k = keep // C, keep % C
    io = OM.stable_topk(sc, 4)[1][e_k, p_k]
    assert torch.equal(io, NM.stable_topk(sc[e_k, p_k], 4)[1])
    line("private top-4 (E=192, C=384, R=512; 49K kept slots)", tm(lambda: OM.stable_topk(sc, 4)[1][e_k, p_k]),
         tm(lambda: NM.stable_topk(sc[e_k, p_k], 4)[1]), "identical")
    del sc

    # ---- MemRead backward (98,304 bags x 6 rows of a 1M x 256 table)
    nrow, B = (1 << 18, 24576) if QUICK else (1 << 20, 98304)
    V = torch.randn(nrow, 256) * 0.02
    rows = torch.randint(0, nrow, (B, 6))
    w0 = torch.softmax(torch.randn(B, 6), -1)
    gout = torch.randn(B, 256)
    res = {}
    for name, M in (("old", OM), ("new", NM)):
        G = torch.zeros_like(V)

        def run():
            w = w0.clone().requires_grad_()
            M.MemRead.apply(M._anchor("cpu", True), w, V, G, rows).backward(gout)
            res[name] = w.grad
        res[name + "_t"] = tm(run, setup=G.zero_)
        res[name + "_G"] = G
        res[name + "_f"] = tm(lambda: M.MemRead.apply(M._anchor("cpu", False), w0, V, None, rows))
    assert torch.equal(res["old_G"], res["new_G"]) and torch.allclose(res["old"], res["new"], rtol=1e-4, atol=1e-5)
    line(f"MemRead fwd+bwd ({B} bags x 6 of {nrow} x 256)", res["old_t"], res["new_t"],
         f"(forward alone {res['old_f'] * 1e3:.0f} / {res['new_f'] * 1e3:.0f} ms); G identical, dw allclose")
    del res, G

    # ---- MemOpt row-Adagrad on the 1M x 256 shared table, 10% / 30% of the rows touched
    for f in (0.1, 0.3):
        G0 = torch.zeros(nrow, 256)
        touched = torch.randperm(nrow, generator=torch.Generator().manual_seed(3))[:int(f * nrow)]
        G0[touched] = torch.randn(len(touched), 256) * 1e-3
        st = {}
        for name in ("old", "new"):
            ns = SimpleNamespace(lr=0.01, clip=0.05, s0_cfg="auto", _s0=[1e-4, 1e-4], s0=torch.full((2,), 1e-4))
            T = SimpleNamespace(V=V.clone(), G=torch.zeros_like(G0), S=torch.full((nrow,), 1e-4),
                                wc=torch.zeros(nrow, dtype=torch.int32))
            if name == "old":
                fn = lambda: OM.Memory._row_adagrad(ns, T.V, T.G, T.S, T.wc, 0)
            else:
                def fn():  # write_counts + the host count + nonzero_static + the update on the touched rows
                    mask = T.G.view(torch.int32).any(1)
                    idx = torch.nonzero_static(mask, size=int(mask.sum())).squeeze(1)
                    NM.Memory._row_adagrad(ns, T.V, T.G, T.S, T.wc, 0, idx)
            st[name] = tm(fn, setup=lambda: T.G.copy_(G0))
            st[name + "_T"] = T
        a, b = st["old_T"], st["new_T"]
        assert all(torch.equal(getattr(a, k), getattr(b, k)) for k in ("V", "G", "S", "wc"))
        line(f"MemOpt row-Adagrad ({nrow} x 256, {int(f * 100)}% rows touched)", st["old"], st["new"], "identical")
        del st, a, b, G0

    # ---- private: row-Adagrad on V_e and the key Adam (768 experts preallocated, 192 live, live rows touched)
    Emax, El, Rp = 768, 192, 512
    ne, nl = Emax * Rp, El * Rp
    G0 = torch.zeros(ne, 256)
    G0[:nl] = torch.randn(nl, 256) * 1e-3
    GP0 = torch.zeros(ne, 64)
    GP0[:nl] = torch.randn(nl, 64) * 1e-3
    st = {}
    for name in ("old", "new"):
        ns = SimpleNamespace(lr=0.01, clip=0.05, s0_cfg="auto", _s0=[1e-4, 1e-4], s0=torch.full((2,), 1e-4))
        T = SimpleNamespace(V=torch.randn(ne, 256, generator=torch.Generator().manual_seed(7)) * 0.02,
                            G=torch.zeros(ne, 256), S=torch.full((ne,), 1e-4),
                            wc=torch.zeros(ne, dtype=torch.int32))
        K = SimpleNamespace(G_P=torch.zeros(ne, 64), P=F.normalize(torch.randn(Emax, Rp, 64,
                                                                           generator=torch.Generator().manual_seed(4)), dim=-1),
                            P_m=torch.zeros(ne, 64), P_v=torch.zeros(ne, 64), P_n=torch.zeros(ne))
        if name == "old":
            fa = lambda: OM.Memory._row_adagrad(ns, T.V, T.G, T.S, T.wc, 1)
            fk = lambda: OM.Memory._adam_keys(K, 1e-3)
        else:
            def fa():
                mask = T.G[:nl].view(torch.int32).any(1)
                NM.Memory._row_adagrad(ns, T.V, T.G, T.S, T.wc, 1, torch.nonzero_static(mask, size=int(mask.sum())).squeeze(1))

            def fk():
                mask = K.G_P[:nl].view(torch.int32).any(1)
                NM.Memory._adam_keys(K, 1e-3, torch.nonzero_static(mask, size=int(mask.sum())).squeeze(1))
        st[name + "_a"] = tm(fa, setup=lambda: T.G.copy_(G0))
        st[name + "_k"] = tm(fk, setup=lambda: K.G_P.copy_(GP0))
        st[name] = (T, K)
    (ta, ka), (tb, kb) = st["old"], st["new"]
    assert all(torch.equal(getattr(ta, k), getattr(tb, k)) for k in ("V", "G", "S", "wc"))
    assert all(torch.equal(getattr(ka, k), getattr(kb, k)) for k in ("P", "P_m", "P_v", "P_n", "G_P"))
    line("MemOpt row-Adagrad V_e (768 x 512 rows, 192 experts live)", st["old_a"], st["new_a"], "identical")
    line("private-key Adam P (768 x 512 x 64, 192 experts live)", st["old_k"], st["new_k"], "identical")
    del st, ta, tb, ka, kb, G0, GP0

    # ---- balancing at the step-500 counts (shared 864 splits / 3,408 dead; private 5,760 / 33,408 at 192 experts)
    mems = []
    for M in (OM, NM):
        torch.manual_seed(5)
        mem = M.Memory(384, 64, 4, H=3, S=512, m=8, km=6, dv=256 if not QUICK else 16, R=512, n_exp=768, n_live=192)
        mem.s0.fill_(1e-4)
        mem._sync_host()
        U = torch.full((24, 512), 1.0)
        U[:, :36] = 30.0                                                    # 36 busy keys per codebook
        U[:, 36:36 + 178] = 1e-6                                            # 178 tiny: 36 split twins + 142 dead
        mem.U.copy_((U / U.sum(1, keepdim=True)).view(4, 3, 2, 512))
        Ue = torch.full((192, 512), 1.0)
        Ue[:, :30] = 30.0
        Ue[:, 30:30 + 204] = 1e-6                                           # 30 split twins + 174 dead
        mem.U_e[:192] = Ue / Ue.sum(1, keepdim=True)
        mem.qres.normal_()
        mem.lastq.normal_()
        mem.lastq_ok.fill_(True)
        state = {n: dict(m=torch.ones_like(p), v=torch.rand(p.shape, generator=torch.Generator().manual_seed(6)))
                 for n, p in (("mem.keys.k1", mem.keys.k1), ("mem.keys.k2", mem.keys.k2))}

        def reset(name, sl, state=state):  # train_lm.reset_mem_key_rows on a BrainOpt-like state
            state[name]["m"][sl] = 0.0
            state[name]["v"][sl] = state[name]["v"].median()
        mem.opt_reset = reset
        t = time.perf_counter()
        st = mem.balance()
        mems.append((time.perf_counter() - t, st, mem, state))
    (t0, s0, m0, o0), (t1, s1, m1, o1) = mems
    assert s0 == s1 and s0["splits"] == 864 and s0["dead"] == 3408 and s0["p_splits"] == 5760 and s0["p_dead"] == 33408
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0 if torch.is_tensor(sd0[k]))
    assert all(torch.equal(o0[n][k], o1[n][k]) for n in o0 for k in ("m", "v"))
    line("balance (step-500 counts: 864 / 3,408 shared, 5,760 / 33,408 private)", t0, t1, "identical")


if __name__ == "__main__":
    main()
