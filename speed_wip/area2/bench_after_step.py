"""Area 2 CPU micro-benchmarks (CUDA_VISIBLE_DEVICES=""): MemOpt row-Adagrad variants, private-key Adam, balancing.
Imports memory.py read-only (run with python -B so no __pycache__ is written into fruit-fly-sim)."""
import os, sys, time, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # on Windows an empty value deletes the variable; -1 hides every GPU
sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
import torch
from torch.profiler import profile, ProfilerActivity
import memory as M

torch.manual_seed(0)
assert not torch.cuda.is_available()


def bench(fn, reps=3):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)


def count_ops(fn):
    with profile(activities=[ProfilerActivity.CPU]) as p:
        fn()
    # top-level aten ops ~ kernel launches on GPU (each elementwise/reduction aten op = 1 kernel)
    ev = [e for e in p.events() if e.name.startswith("aten::") and e.cpu_parent is None]
    return len(ev)


class Fake:  # just the attributes _row_adagrad / _adam_keys need
    pass


def make(nrows, dv, frac, seed=0):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(nrows, dv, generator=g) * 0.02
    G = torch.zeros(nrows, dv)
    nact = int(frac * nrows)
    rows = torch.randperm(nrows, generator=g)[:nact]
    G[rows] = torch.randn(nact, dv, generator=g) * 1e-3
    S = torch.full((nrows,), 1e-4)
    wc = torch.zeros(nrows, dtype=torch.int32)
    return V, G, S, wc


def mem_stub():
    m = Fake()
    m._s0 = [1e-4, 1e-4]; m.s0 = torch.tensor([1e-4, 1e-4]); m.s0_cfg = "auto"; m.lr, m.clip = 0.01, 0.05
    return m


def cur(m, V, G, S, wc):
    M.Memory._row_adagrad.__wrapped__(m, V, G, S, wc, 0) if hasattr(M.Memory._row_adagrad, "__wrapped__") else \
        M.Memory._row_adagrad(m, V, G, S, wc, 0)


@torch.no_grad()
def sparse(m, V, G, S, wc):
    """touched rows only: one nonzero (a host sync on GPU), gather -> update -> scatter, zero only those rows."""
    idx = (G != 0).any(1).nonzero().squeeze(1)
    g = G.index_select(0, idx)
    s = S.index_select(0, idx) + (g * g).mean(1)
    dv = -m.lr * g / s.sqrt()[:, None]
    nrm = dv.norm(dim=1, keepdim=True)
    dv = dv * (m.clip / nrm.clamp(min=1e-30)).clamp(max=1.0)
    V.index_copy_(0, idx, V.index_select(0, idx) + dv)
    S.index_copy_(0, idx, s)
    wc.index_add_(0, idx, torch.ones_like(idx, dtype=wc.dtype))
    G.index_fill_(0, idx, 0.0)


@torch.no_grad()
def dense_lean(m, V, G, S, wc, chunk=1 << 17):
    """same math, dense, fewer full-table passes: no torch.where on V (dv is exactly 0 on inactive rows:
    g = 0 -> dv = -lr*0/sqrt(s) = -0.0; v + (-0.0) == v bit-exact for every finite v incl. +0.0 (+0 + -0 = +0))."""
    for c in range(0, V.size(0), chunk):
        g = G[c:c + chunk]
        act = (g != 0).any(1)
        s = S[c:c + chunk]
        s.add_((g * g).mean(1))        # inactive rows add exactly +0.0
        dv = -m.lr * g / s.sqrt()[:, None]
        nrm = dv.norm(dim=1, keepdim=True)
        dv = dv * (m.clip / nrm.clamp(min=1e-30)).clamp(max=1.0)
        V[c:c + chunk].add_(dv)        # inactive rows add -0.0: bit-identical
        wc[c:c + chunk] += act.to(wc.dtype)
    G.zero_()


@torch.no_grad()
def adam_cur(m):
    M.Memory._adam_keys(m, 1e-3)


def main():
    out = {}
    # ---- shared table at full size (1,048,576 x 256) and private at Emax (393,216 x 256), several touched fractions
    for name, nrows in (("V_sh", 1 << 20), ("V_e", 768 * 512)):
        for frac in (1.0, 0.3, 0.05):
            V, G, S, wc = make(nrows, 256, frac)
            m = mem_stub()
            ref = [t.clone() for t in (V, G, S, wc)]
            t_cur = bench(lambda: (G.copy_(ref[1]), cur(m, V, G, S, wc)), reps=2)
            V2, G2, S2, wc2 = (t.clone() for t in ref)
            V, G, S, wc = (t.clone() for t in ref)
            cur(m, V, G, S, wc)
            sparse(m, V2, G2, S2, wc2)
            ex_sparse = all(torch.equal(a, b) for a, b in ((V, V2), (S, S2), (wc, wc2), (G, G2)))
            maxd_sparse = (V - V2).abs().max().item()
            V3, G3, S3, wc3 = (t.clone() for t in ref)
            dense_lean(m, V3, G3, S3, wc3)
            maxd_lean = (V - V3).abs().max().item()
            ex_lean = torch.equal(V, V3) and torch.equal(S, S3) and torch.equal(wc, wc3)
            V2, G2, S2, wc2 = (t.clone() for t in ref)
            t_sp = bench(lambda: (G2.copy_(ref[1]), sparse(m, V2, G2, S2, wc2)), reps=2)
            t_ln = bench(lambda: (G3.copy_(ref[1]), dense_lean(m, V3, G3, S3, wc3)), reps=2)
            Gc = ref[1].clone()
            t_copy = bench(lambda: Gc.copy_(ref[1]), reps=2)
            out[(name, frac)] = dict(cur=t_cur - t_copy, sparse=t_sp - t_copy, lean=t_ln - t_copy,
                                     exact_sparse=ex_sparse, maxdiff_sparse=maxd_sparse,
                                     exact_lean=ex_lean, maxdiff_lean=maxd_lean)
            print(name, frac, {k: (round(v * 1e3, 1) if isinstance(v, float) and k in ("cur", "sparse", "lean") else v)
                               for k, v in out[(name, frac)].items()}, flush=True)
            del V, G, S, wc, V2, G2, S2, wc2, V3, G3, S3, wc3, ref

    # ---- op (kernel) counts per call
    V, G, S, wc = make(1 << 20, 256, 0.3)
    m = mem_stub()
    print("ops _row_adagrad V_sh:", count_ops(lambda: cur(m, V, G, S, wc)))
    V, G, S, wc = make(1 << 20, 256, 0.3)
    print("ops sparse V_sh:", count_ops(lambda: sparse(m, V, G, S, wc)))
    V, G, S, wc = make(1 << 20, 256, 0.3)
    print("ops lean V_sh:", count_ops(lambda: dense_lean(m, V, G, S, wc)))
    del V, G, S, wc

    # ---- private-key Adam over full Emax vs e_live (192 of 768 experts)
    for E in (768, 192):
        a = Fake()
        n = E * 512
        a.G_P = torch.zeros(n, 64); a.G_P[torch.randperm(n)[: int(0.9 * n)]] = torch.randn(int(0.9 * n), 64) * 1e-3
        a.P_n = torch.zeros(n); a.P_m = torch.zeros(n, 64); a.P_v = torch.zeros(n, 64)
        a.P = torch.randn(E, 512, 64)
        g0 = a.G_P.clone()
        t = bench(lambda: (a.G_P.copy_(g0), adam_cur(a)), reps=2)
        print(f"_adam_keys E={E}: {t * 1e3:.1f} ms CPU, ops {count_ops(lambda: adam_cur(a))}")

    # ---- balancing plan (CPU part) on the real shapes
    U = torch.rand(4 * 3 * 2, 512) ** 3; U = U / U.sum(1, keepdim=True)
    t = bench(lambda: M.plan_balance(U), reps=3)
    sp, de = M.plan_balance(U)
    print(f"plan_balance shared (24x512): {t * 1e3:.1f} ms, splits {len(sp)} deads {len(de)}")
    for E in (192, 768):
        Ue = torch.rand(E, 512) ** 3; Ue = Ue / Ue.sum(1, keepdim=True)
        t = bench(lambda: M.plan_balance(Ue), reps=3)
        sp, de = M.plan_balance(Ue)
        print(f"plan_balance private ({E}x512): {t * 1e3:.1f} ms, splits {len(sp)} deads {len(de)}")


if __name__ == "__main__":
    main()
