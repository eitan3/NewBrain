"""FastMem packs (--fastmem-pack P), CPU:  set CUDA_VISIBLE_DEVICES=-1 & python -B tests_fastmem_pack.py

(p1) P = 1 == the previous code (fruit-fly-sim-fastmem-b/fastmem.py): outputs, committed states and every gradient,
     over two carried micro-batches
(p2) a pack of P sequences == the same P sequences as P consecutive micro-batches (P = 1, carried state): the scan-pass
     reads and the committed end state agree (the pack only adds gradient, not different values)
(p3) gradient: a loss on sequence n of a pack reaches the WRITES of sequence n - 1 (xf, write grad full; and the
     write projections Wk / Wv); with P = 1 it does not; zero into the other stream's pack; causal (no gradient
     into the later sequences of the pack)
(p4) the trainer's pack data layout (train_lm.Data): streams per micro-batch / step, sequences, starts, wrap
(VRAM: measured on the GPU with vram_guard.py --worst-route 1 (whole 2,048-token packs to the same rows); the CPU
 working-set estimates tried here did not register the scan's allocations, so they were removed)
off == none (t5) and the rest of the module's tests: tests_fastmem.py
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tests_fastmem import call, check, randomize, small_mem, OK  # noqa: E402

E_, D_ = 6, 32


def old_fastmem_class():
    spec = importlib.util.spec_from_file_location("fastmem_b", HERE.parent / "fruit-fly-sim-fastmem-b" / "fastmem.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FastMem


def make(cls=None, **kw):
    c = dict(fastmem_heads=2, fastmem_dk=16, fastmem_dv=16, fastmem_global=4, fastmem_global_topk=2,
             fastmem_dtype="fp32", fastmem_explore=0.0)
    c.update(kw)
    m = (cls or __import__("fastmem").FastMem)(D_, E_, **c)
    randomize(m)
    m.train()
    m.train_forward = True
    return m


def fwd(m, xf, B, streams, starts, pack=None, seed=3):
    if pack is None:
        m.set_streams(streams, starts)
    else:
        m.set_streams(streams, starts, pack=pack)
    m.begin_forward()
    (buf, x), (e_k, p_k, t_k) = call(m, xf, B, n_passes=1, seed=seed)
    return buf, x, (e_k, p_k, t_k)


def p1():
    Old = old_fastmem_class()
    B, T = 2, 12
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(B * T, D_, generator=g) for _ in range(2)]
    res = []
    for cls, pk in ((Old, None), (None, 1)):
        m = make(cls)
        outs, grads = [], []
        for i, x0 in enumerate(xs):
            xf = x0.clone().requires_grad_()
            buf, x, _ = fwd(m, xf, B, [0, 1], [i == 0] * 2, pack=pk, seed=3 + i)
            (buf.square().sum() + x.square().sum()).backward()
            m.after_backward()
            outs += [buf.detach(), x.detach()]
            grads.append(xf.grad)
        grads += [p.grad.clone() if p.grad is not None else None for _, p in sorted(m.named_parameters())]
        res.append((outs, grads, m.M.clone(), m.Mg.clone()))
    (oa, ga, Ma, Mga), (ob, gb, Mb, Mgb) = res
    same_o = all(torch.equal(a, b) for a, b in zip(oa, ob))
    same_g = all((a is None and b is None) or torch.equal(a, b) for a, b in zip(ga, gb))
    check("p1 P = 1 == fastmem-b: outputs bit-identical over 2 carried micro-batches", same_o)
    check("p1 P = 1 == fastmem-b: every gradient (xf + all parameters) bit-identical", same_g and len(ga) == len(gb))
    check("p1 P = 1 == fastmem-b: committed M / Mg bit-identical", torch.equal(Ma, Mb) and torch.equal(Mga, Mgb))


def routing_rows(N, B, gen, k=2):
    """A per-token routing that does not depend on the batch layout (the same tokens -> the same experts)."""
    return torch.rand(N, E_, generator=gen).topk(k, -1).indices


def call_fixed(m, xf, B, idx, cap=10.0):
    """call() with a given top-k index (so a pack and its P single micro-batches route identically); capacity large
    enough that nothing is dropped."""
    N, d = xf.shape
    k = idx.size(1)
    flat = idx.flatten()
    order = flat.argsort(stable=True)
    e_sorted, tok = flat[order], order // k
    counts = torch.bincount(flat, minlength=E_)
    C = int(counts.max())
    pos = torch.arange(N * k) - (counts.cumsum(0) - counts)[e_sorted]
    e_k, p_k, t_k, slot_k = e_sorted, pos, tok, order % k
    buf = xf.new_zeros(E_, C, d).index_put((e_k, p_k), xf[t_k])
    x = xf.view(B, N // B, d)
    m.n_passes = 1
    return m(xf, buf, x, None, e_k, p_k, t_k, slot_k, counts, N, E_, C, N * k, call=0), (e_k, p_k, t_k)


def p2():
    P, S, T = 4, 2, 10                   # 2 streams, packs of 4 sequences
    g = torch.Generator().manual_seed(2)
    seqs = torch.randn(S, P, T, D_, generator=g)
    idx_all = routing_rows(S * P * T, S * P, g).view(S, P, T, 2)
    # (a) one micro-batch of 2 packs: rows = stream-major (stream s, sequence t) = row s * P + t
    ma = make()
    m0 = ma.M.clone()
    xf = seqs.reshape(S * P * T, D_)
    ma.set_streams([0, 1], [False, False], pack=P)
    ma.begin_forward()
    (bufa, xa), (eka, pka, tka) = call_fixed(ma, xf, S * P, idx_all.reshape(-1, 2))
    ra_x = xa.detach().view(S, P, T, D_)
    ra_r = torch.zeros(S * P * T, E_, D_)
    ra_r[tka, eka] = bufa.detach()[eka, pka]
    ra_r = ra_r.view(S, P, T, E_, D_)
    ma.after_backward()
    # (b) the same sequences as P micro-batches of the 2 streams (P = 1), state carried
    mb = make()
    with torch.no_grad():
        mb.M.copy_(m0)
    rb_x = torch.zeros(S, P, T, D_)
    rb_r = torch.zeros(S, P, T, E_, D_)
    for t in range(P):
        mb.set_streams([0, 1], [False, False], pack=1)
        mb.begin_forward()
        xf_t = seqs[:, t].reshape(S * T, D_)
        (bufb, xb), (ekb, pkb, tkb) = call_fixed(mb, xf_t, S, idx_all[:, t].reshape(-1, 2))
        rb_x[:, t] = xb.detach().view(S, T, D_)
        r = torch.zeros(S * T, E_, D_)
        r[tkb, ekb] = bufb.detach()[ekb, pkb]
        rb_r[:, t] = r.view(S, T, E_, D_)
        mb.after_backward()
    ex = float((ra_x - rb_x).abs().max() / rb_x.abs().max())
    er = float((ra_r - rb_r).abs().max() / (rb_r.abs().max() + 1e-12))
    eM = float((ma.M - mb.M).abs().max() / mb.M.abs().max())
    eMg = float((ma.Mg - mb.Mg).abs().max() / mb.Mg.abs().max())
    check("p2 pack of 4 == 4 carried micro-batches: shared + global reads", ex < 1e-4, f"(rel {ex:.1e})")
    check("p2 pack of 4 == 4 carried micro-batches: routed reads", er < 1e-4, f"(rel {er:.1e})")
    check("p2 pack of 4 == 4 carried micro-batches: committed end states M / Mg", max(eM, eMg) < 1e-4,
          f"(rel {eM:.1e} / {eMg:.1e})")


def p3():
    P, S, T = 4, 2, 10
    g = torch.Generator().manual_seed(4)
    seqs = torch.randn(S * P * T, D_, generator=g)
    idx = routing_rows(S * P * T, S * P, g)
    out = {}
    for pk in (P, 1):
        m = make()
        xf = seqs.clone().requires_grad_()
        if pk == P:
            m.set_streams([0, 1], [False, False], pack=P)
        else:
            m.set_streams(list(range(S * P)) if False else [0, 0, 0, 0, 1, 1, 1, 1][:S * P], [False] * (S * P), pack=1)
        m.begin_forward()
        (buf, x), (e_k, p_k, t_k) = call_fixed(m, xf, S * P, idx)
        row = 2                                    # stream 0, sequence n = 2 of the pack
        sel = (t_k // T) == row
        loss = x.view(S * P, T, D_)[row].square().sum() + buf[e_k[sel], p_k[sel]].square().sum()
        loss.backward()
        gx = xf.grad.view(S * P, T, D_).abs().sum((1, 2))
        out[pk] = (gx, m.Wk.grad.abs().sum().item() if m.Wk.grad is not None else 0.0)
    gx, _ = out[P]
    check("p3 pack: loss in seq n reaches the writes of seq n - 1 (xf grad)", gx[1] > 1e-6, f"({gx[1]:.2e})")
    check("p3 pack: ... and of seq n - 2", gx[0] > 1e-6, f"({gx[0]:.2e})")
    check("p3 pack: no gradient into the later sequence n + 1 (causal)", gx[3] == 0, f"({gx[3]:.2e})")
    check("p3 pack: no gradient into the other stream's pack", gx[4:].abs().max() == 0, f"({gx[4:].abs().max():.2e})")
    g1, _ = out[1]
    check("p3 P = 1 control: NO gradient into seq n - 1 (separate segments)", g1[1] == 0 and g1[0] == 0,
          f"({g1[1]:.2e})")
    # the path is the WRITE side: with --fastmem-write-grad detached, seq n - 1 (which seq n only sees through its
    # writes) gets no gradient; and the write projections Wk / Wv of seq n - 1's experts get gradient that changes
    # when only seq n - 1's inputs change (the loss is on seq n alone)
    m = make(fastmem_write_grad="detached")
    xf = seqs.clone().requires_grad_()
    m.set_streams([0, 1], [False, False], pack=P)
    m.begin_forward()
    (buf, x), _ = call_fixed(m, xf, S * P, idx)
    x.view(S * P, T, D_)[2].square().sum().backward()
    gd = xf.grad.view(S * P, T, D_).abs().sum((1, 2))
    check("p3 write grad detached: no gradient into seq n - 1 (so the pack gradient is the write side)",
          gd[1] == 0 and gd[2] > 0, f"({gd[1]:.2e}, own seq {gd[2]:.2e})")
    wks = []
    for shift in (0.0, 1.0):
        m = make()
        xin = seqs.clone().view(S * P, T, D_)
        xin[1] += shift                            # only seq n - 1 changes
        m.set_streams([0, 1], [False, False], pack=P)
        m.begin_forward()
        (buf, x), _ = call_fixed(m, xin.reshape(-1, D_), S * P, idx)
        x.view(S * P, T, D_)[2].square().sum().backward()
        wks.append(torch.cat([m.Wk.grad.flatten(), m.Wv.grad.flatten()]))
    dwk = float((wks[0] - wks[1]).abs().max())
    check("p3 pack: seq n's loss -> Wk / Wv gradient depends on seq n - 1's writes", dwk > 1e-6, f"({dwk:.2e})")


def p4():
    import train_lm
    data_dir = Path("X:/Datasets/world_change_mix")
    if not (data_dir / "meta.json").exists():
        check("p4 data layout (X:/Datasets/world_change_mix missing)", False)
        return
    P, A, B = 4, 2, 16
    d = train_lm.Data(data_dir, 512, "cpu", mode="stream", pack=P, accum=A)
    seen = {}
    for step in range(4):
        d.ctr = step * A
        for i in range(A):
            x, y = d.batch(B)
            streams, starts, pk = d.last_streams
            assert pk == P and len(streams) == B // P
            for li, s in enumerate(streams):
                k = step // 2
                j0 = k * P
                for t in range(P):
                    j = j0 + t
                    ref = np.asarray(d.train[s * d.L + j * 512: s * d.L + j * 512 + 513]).astype(np.int64)
                    assert torch.equal(x[li * P + t], torch.from_numpy(ref[:-1])), (step, i, s, t)
                assert starts[li] == (k == 0), (step, s, starts)
                seen.setdefault(s, []).append(j0)
    groups_ok = sorted(seen) == list(range(16)) and all(len(v) == 2 for v in seen.values())
    adv_ok = all(v[1] - v[0] == P for v in seen.values())
    check("p4 data: 16 rows = 4 streams x 4 consecutive seqs; 8 streams per step, groups alternate by step",
          groups_ok, f"({ {s: v for s, v in list(seen.items())[:3]} })")
    check("p4 data: a stream advances one pack (4 seqs) per turn, packs aligned (no wrap); starts on its first pack",
          adv_ok)
    d.ctr = (d.n_seq // P) * 2 * A                  # (the first pack after a full pass: aligned again, reset)
    d.batch(B)
    st2, sts2, _ = d.last_streams
    check("p4 data: after a full pass the stream restarts at sequence 0 with a reset", all(sts2), f"({sts2})")
    tok = A * B * 512
    check("p4 data: 16,384 tokens per step", tok == 16384)


if __name__ == "__main__":
    p1()
    p2()
    p3()
    p4()
    print(f"{sum(OK)}/{len(OK)} passed", flush=True)
    sys.exit(0 if all(OK) else 1)
