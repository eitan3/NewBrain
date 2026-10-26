"""FastMem tests (fastmem.py, SPEC_fastmem.md), CPU:  set CUDA_VISIBLE_DEVICES=-1 & python -B tests_fastmem.py [work]

(t1) the closed-form scan == the token-by-token recurrence (reads and the merged end state)
(t2) write k -> v, then reading k returns v (in-step, and from the persistent M after the merge)
(t3) overwrite: k -> v1, k -> v2, read k = v2
(t4) causality: a token's read does not change when later tokens change (scan and module level)
(t5) off == none: the brain is bit-identical (build, forward, backward) with the module switched off
(t6) gradients: Wk, Wv, beta / omega / lam, the gates, D and the router get gradient through the read; write grad
     full: the writes reach xf; detached: no gradient into xf from the write side
(t7) bytes fixed (steps, writes, growth)
(t8) save / load of one expert's memory (+ memory.pt round trip)
(t9) a tiny train_lm run: --memory fastmem --data-mode stream on X:/Datasets/world_change_mix (+ detached)
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fastmem import FastMem, merge_into, scan_read  # noqa: E402
from memory_api import load_memory_file, memory_off, save_memory_file  # noqa: E402

PY = r"C:\Eitan\Python\NewBrain\.venv\Scripts\python.exe"
OK = []


def check(name, cond, info=""):
    print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    OK.append(bool(cond))


def recurrence(q, u, w, lam, M, segb, valid, B):
    """Token by token (the spec's update), per row and sequence: read S_{i-1}, then S <- (I - u u^T) e^lam S + u w^T."""
    R, C, H, dk = u.shape
    O = torch.zeros(R, C, H, w.size(-1), dtype=u.dtype)
    dM = torch.zeros_like(M)
    for r in range(R):
        for b in range(B):
            St = M[r].clone()
            for c in range(C):
                if not valid[r, c] or segb[r, c] != b:
                    continue
                O[r, c] = torch.einsum("hd,hde->he", q[r, c], St)
                a = torch.exp(lam[r, c])[:, None, None]
                uu = u[r, c]
                St = a * St - uu[:, :, None] * torch.einsum("hd,hde->he", uu, a * St)[:, None, :] \
                    + uu[:, :, None] * w[r, c][:, None, :]
            dM[r] += St - M[r]
    return O, dM


def rand_case(R=3, C=12, H=2, dk=8, dv=6, B=2, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    n = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    l2 = lambda x: x / x.norm(dim=-1, keepdim=True)
    q, k = l2(n(R, C, H, dk)), l2(n(R, C, H, dk))
    beta = torch.sigmoid(n(R, C, H, dk))
    u, w = beta.sqrt() * k, n(R, C, H, dv)
    lam = -torch.rand(R, C, H, generator=g, dtype=dtype) * 0.3
    M = n(R, H, dk, dv) * 0.3
    valid = torch.rand(R, C, generator=g) < 0.8
    segb = torch.randint(0, B, (R, C), generator=g).sort(-1).values         # (slots in token order: sequences ascend)
    segp = torch.zeros(R, C, dtype=torch.long)
    for r in range(R):
        cnt = {}
        for c in range(C):
            if valid[r, c]:
                segp[r, c] = cnt.get(int(segb[r, c]), 0)
                cnt[int(segb[r, c])] = segp[r, c].item() + 1
    Cs = int(segp.max()) + 1
    return q, u, w, lam, M, segb, segp, valid, B, Cs


def t1():
    q, u, w, lam, M, segb, segp, valid, B, Cs = rand_case()
    O, parts = scan_read(q, u, w, lam, M, segb, segp, valid, B, Cs, merge=True)
    dM = torch.zeros_like(M)
    merge_into(dM, M, parts)
    Or, dMr = recurrence(q, u, w, lam, M, segb, valid, B)
    vm = valid[..., None, None]
    e1 = ((O - Or) * vm).abs().max().item()
    e2 = (dM - dMr).abs().max().item()
    check("t1 closed form == recurrence", e1 < 1e-10 and e2 < 1e-10, f"(read err {e1:.1e}, merge err {e2:.1e})")


def one_seg(ks, vs, qs, M=None, H=1, dk=16, dv=16):
    """One row, one sequence: tokens write (k, v) with beta = omega = 1, lam = 0; then query tokens."""
    n = len(ks) + len(qs)
    q = torch.zeros(1, n, H, dk, dtype=torch.float64)
    u = torch.zeros(1, n, H, dk, dtype=torch.float64)
    w = torch.zeros(1, n, H, dv, dtype=torch.float64)
    for i, (k, v) in enumerate(zip(ks, vs)):
        u[0, i, 0], w[0, i, 0] = k, v
    for j, qq in enumerate(qs):
        q[0, len(ks) + j, 0] = qq
    lam = torch.zeros(1, n, H, dtype=torch.float64)
    M = torch.zeros(1, H, dk, dv, dtype=torch.float64) if M is None else M
    valid = torch.ones(1, n, dtype=torch.bool)
    O, parts = scan_read(q, u, w, lam, M, torch.zeros(1, n, dtype=torch.long), torch.arange(n)[None], valid, 1, n,
                         merge=True)
    dM = torch.zeros_like(M)
    merge_into(dM, M, parts)
    return O, M + dM


def t2_t3():
    g = torch.Generator().manual_seed(1)
    l2 = lambda x: x / x.norm()
    ks = [l2(torch.randn(16, generator=g, dtype=torch.float64)) for _ in range(4)]
    vs = [torch.randn(16, generator=g, dtype=torch.float64) for _ in range(4)]
    O, dM = one_seg(ks[:1], vs[:1], ks[:1])
    check("t2 write k->v, read k == v (in-step)", torch.allclose(O[0, 1, 0], vs[0], atol=1e-10))
    O2, _ = one_seg([], [], ks[:1], M=dM)                   # (the merged persistent M, B = 1: M_end)
    check("t2 ... and from the persistent M after the merge", torch.allclose(O2[0, 0, 0], vs[0], atol=1e-10))
    O, _ = one_seg([ks[0], ks[0]], [vs[0], vs[1]], [ks[0]])
    check("t3 overwrite k->v1, k->v2: read k == v2", torch.allclose(O[0, 2, 0], vs[1], atol=1e-10))
    # several orthogonal keys coexist
    Q, _ = torch.linalg.qr(torch.randn(16, 4, generator=g, dtype=torch.float64))
    kk = [Q[:, i] for i in range(4)]
    O, _ = one_seg(kk, vs, kk)
    err = max((O[0, 4 + i, 0] - vs[i]).abs().max().item() for i in range(4))
    check("t2 four orthogonal keys recalled", err < 1e-10, f"(err {err:.1e})")


def routing(N, E, k, gen, cap=1.5):
    """RoutedMLP's capacity layout for a random top-k routing."""
    sc = torch.rand(N, E, generator=gen)
    idx = sc.topk(k, -1).indices
    flat = idx.flatten()
    order = flat.argsort(stable=True)
    e_sorted, tok = flat[order], order // k
    counts = torch.bincount(flat, minlength=E)
    cl = counts.tolist()
    used = max(1, sum(c > 0 for c in cl))
    C = max(1, min(max(cl), math.ceil(cap * N * k / used)))
    pos = torch.arange(N * k) - (counts.cumsum(0) - counts)[e_sorted]
    keep = pos < C
    kidx = keep.nonzero().squeeze(-1)
    return e_sorted[kidx], pos[kidx], tok[kidx], (order % k)[kidx], counts, C, int(keep.sum())


def small_mem(E=6, d=32, **kw):
    c = dict(fastmem_heads=2, fastmem_dk=16, fastmem_dv=16, fastmem_global=4, fastmem_global_topk=2)
    c.update(kw)
    return FastMem(d, E, **c)


def call(m, xf, B, E=6, k=2, seed=3, n_passes=1, call_k=0):
    gen = torch.Generator().manual_seed(seed)
    N, d = xf.shape
    e_k, p_k, t_k, slot_k, counts, C, n_keep = routing(N, E, k, gen)
    buf = xf.new_zeros(E, C, d).index_put((e_k, p_k), xf[t_k])
    x = xf.view(B, N // B, d)
    m.n_passes = n_passes
    return m(xf, buf, x, None, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=call_k), (e_k, p_k, t_k)


def randomize(m, seed=5):
    """Non-zero D / M so reads are visible (the module starts with D = 0, M = 0)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for t in (m.D, m.D_g):
            t.copy_(torch.randn(t.shape, generator=g) * 0.05)
        for t in (m.M, m.Mg):
            t.copy_(torch.randn(t.shape, generator=g) * 0.1)
        m.bp_b.fill_(5.0)                       # (gate open everywhere)


def t4():
    q, u, w, lam, M, segb, segp, valid, B, Cs = rand_case(seed=7)
    O, _ = scan_read(q, u, w, lam, M, segb, segp, valid, B, Cs)
    q2, u2, w2 = q.clone(), u.clone(), w.clone()
    late = segp >= 2                            # (change every token from position 2 on)
    u2[late], w2[late], q2[late] = u2[late] * 0.5 + 0.1, w2[late] + 1.0, q2[late] * -1
    O2, _ = scan_read(q2, u2, w2, lam, M, segb, segp, valid, B, Cs)
    early = (segp <= 1) & valid
    e = (O - O2)[early].abs().max().item()
    check("t4 causality (scan): reads before the change unchanged", e < 1e-12, f"(err {e:.1e})")
    m = small_mem()
    randomize(m)
    m.eval()
    B, T, d = 2, 10, 32
    xf = torch.randn(B * T, d, generator=torch.Generator().manual_seed(2))
    (buf1, x1), (e_k, p_k, t_k) = call(m, xf, B)
    xf2 = xf.clone().view(B, T, d)
    xf2[:, 6:] = torch.randn(B, T - 6, d)
    (buf2, x2), _ = call(m, xf2.reshape(-1, d), B)
    pre = (t_k % T) < 6
    eb = (buf1 - buf2)[e_k[pre], p_k[pre]].abs().max().item()
    es = (x1 - x2)[:, :6].abs().max().item()
    late_changed = (x1 - x2)[:, 7:].abs().max().item() > 0
    check("t4 causality (module): tokens < t unchanged by later tokens (fp32 shapes: tol 1e-5)",
          eb < 1e-5 and es < 1e-5 and late_changed,
          f"(routed {eb:.1e}, shared+global {es:.1e})")


def t6():
    for mode in ("full", "detached"):
        m = small_mem(fastmem_write_grad=mode)
        randomize(m)
        m.train()
        m.train_forward = True
        m.begin_forward()
        B, T, d = 2, 10, 32
        xf = torch.randn(B * T, d, generator=torch.Generator().manual_seed(4), requires_grad=True)
        (buf, x), _ = call(m, xf, B)
        loss = buf.pow(2).sum() + x.pow(2).sum()
        loss.backward()
        names = ["Wq", "Wk", "Wv", "Wb", "Ww", "Wl", "s", "Wg", "bg", "D", "Wp", "bp_b", "Wk_g", "Wv_g", "Wr", "D_g"]
        nz = {n: getattr(m, n).grad is not None and getattr(m, n).grad.abs().sum().item() > 0 for n in names}
        check(f"t6 [{mode}] grads reach the memory's parameters", all(nz.values()),
              str([n for n, v in nz.items() if not v]))
        g_xf = xf.grad.clone()
        # the write side's own contribution to xf: the same graph with the write inputs a separate leaf
        X = xf.detach().clone().requires_grad_(True)
        Xw = xf.detach().clone().requires_grad_(True)
        old = m.write_full
        import fastmem as fm_
        y = None
        H, dk, dv = m.H, m.dk, m.dv
        E = m.n_live
        W = (m.Wq[:1], m.Wk[:1], m.Wv[:1], m.Wb, m.bb, m.Ww, m.bw, m.Wl[:1], m.bl[:1], m.s[:1], m.Wg[:1], m.bg[:1],
             m.g_norm)
        ar = torch.arange(B * T)
        y, _ = m._family(X[None], (Xw if mode == "full" else Xw.detach())[None], torch.ones(1, B * T, dtype=torch.bool),
                         (ar // T)[None], (ar % T)[None], B, T, m.M[:B, :1].transpose(0, 1), W, True, False)
        y.pow(2).sum().backward()
        has_w = Xw.grad is not None and Xw.grad.abs().sum().item() > 0
        check(f"t6 [{mode}] gradient into xf through the writes: {'yes' if mode == 'full' else 'none'}",
              has_w == (mode == "full"))
        if mode == "full":
            g_full = g_xf
        else:
            check("t6 full vs detached: different xf gradients (the write path)", (g_full - g_xf).abs().max() > 0)


def tiny_brain(memory, **kw):
    from loop_brain import LoopBrain
    torch.manual_seed(0)
    mk = dict(memory=memory) if memory != "none" else dict(memory="none")
    if memory == "fastmem":
        mk.update({**dict(fastmem_heads=2, fastmem_dk=16, fastmem_dv=16, fastmem_global=4, fastmem_global_topk=2), **kw})
    return LoopBrain(64, 32, 0, 4, 8, moe=dict(n_exp=6, k=2, hidden=32, shared=32, balance="auxfree", capacity=1.5), k_mean=3, k_min=3, k_max=3,
                     bptt=2, state="", slots=0, **mk)


def t5():
    a, b = tiny_brain("none"), tiny_brain("fastmem")
    sa = a.state_dict()
    sb = {k: v for k, v in b.state_dict().items() if not k.startswith("bm.")}
    same = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    check("t5 build: brain state bit-identical (none vs fastmem)", same)
    idx = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(1))
    a.train(), b.train()
    torch.manual_seed(5)
    _, la = a(idx, idx)
    la.backward()
    randomize(b.bm)
    torch.manual_seed(5)
    with memory_off(b):
        _, lb = b(idx, idx)
        lb.backward()
    ga = {n: p.grad for n, p in a.named_parameters()}
    gb = {n: p.grad for n, p in b.named_parameters() if not n.startswith("bm.")}
    eq = torch.equal(la, lb) and all((ga[n] is None and gb[n] is None) or torch.equal(ga[n], gb[n]) for n in ga)
    check("t5 off == none: loss and every brain gradient bit-identical", eq, f"({la.item():.6f} vs {lb.item():.6f})")
    torch.manual_seed(5)
    _, lc = b(idx, idx)
    check("t5 (on, with a non-zero memory: the loss differs)", not torch.equal(la, lc))


def t7_t8(work, tag="lite, per stream", **kw):
    b = tiny_brain("fastmem", **kw)
    m = b.bm
    nb0 = m.nbytes()
    idx = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(1))
    b.train()
    for i in range(3):
        m.set_streams([0, 1], [i == 0, i == 0])
        _, l = b(idx, idx)
        l.backward()
        m.after_backward()
        m.after_step(i)
    moved = m.M.abs().sum().item() > 0
    tag = f"[{tag}]"
    check(f"t7 {tag} the writes reach M", moved)
    check(f"t7 {tag} bytes fixed after steps", m.nbytes() == nb0, f"({nb0} B)")
    st = m.expert_state(2)
    m2 = m.M.clone()
    with torch.no_grad():
        m.M[m._mrow(2)] += 1.0
        m.Wk[2].add_(1.0)
    m.load_expert_state(2, st)
    check(f"t8 {tag} save / load one expert: restored exactly", torch.equal(m.M, m2) and torch.equal(m.Wk[2], st["Wk"]))
    m.reset_expert(3)
    check(f"t8 {tag} reset one expert: its M zero, the others kept",
          m.M[m._mrow(3)].abs().sum() == 0 and torch.equal(m.M[m._mrow(2)], m2[m._mrow(2)]))
    p = Path(work) / "fm.memory.pt"
    save_memory_file(p, b, 3)
    b.eval()
    with torch.no_grad():
        _, l1 = b(idx, idx)
    sd = {k: v.clone() for k, v in m.state_dict().items() if torch.is_tensor(v)}
    with torch.no_grad():
        m.M.zero_()
        m.D.zero_()
    load_memory_file(p, b)
    with torch.no_grad():
        _, l2 = b(idx, idx)
    check(f"t8 {tag} memory.pt round trip: identical outputs", torch.equal(l1, l2))
    m.on_grow([1, 2], 4)
    check(f"t8 {tag} growth: children copy M and the per-row parameters",
          torch.equal(m.M[m._mrow(4)], m.M[m._mrow(1)]) and torch.equal(m.Wq[5], m.Wq[2]) and m.nbytes() == nb0)
    if m.stream:
        s0 = m.stream_state(0)
        Mold = m.M.clone()
        with torch.no_grad():
            m.M[0] += 1.0
            m.Mg[0] += 1.0
        m.load_stream_state(0, s0)
        check(f"t8 {tag} save / load one stream (user): restored exactly", torch.equal(m.M, Mold))


def t10():
    """Per stream: isolation (stream 0's writes never change stream 1's reads) and the carry across micro-batches."""
    B, T, d = 2, 10, 32
    g = torch.Generator().manual_seed(11)
    xf1, xf2 = torch.randn(B * T, d, generator=g), torch.randn(B * T, d, generator=g)
    xf1b = xf1.clone().view(B, T, d)
    xf1b[0] = torch.randn(T, d, generator=g)                 # (stream 0's first micro-batch changed)

    def two(x1, carry=True):
        m = small_mem(fastmem_dtype="fp32")
        randomize(m)
        m.train()
        m.train_forward = True
        m.set_streams([0, 1], [True, True])
        m.begin_forward()
        call(m, x1, B, n_passes=1)
        m.after_backward()
        m.set_streams([0, 1] if carry else None, [False, False])
        m.begin_forward()
        (b2, x2), (e_k, p_k, t_k) = call(m, xf2, B, n_passes=1)
        return b2, x2, (e_k, p_k, t_k), m
    ba, xa, (e_k, p_k, t_k), ma = two(xf1)
    bb, xb, _, mb = two(xf1b.reshape(-1, d))
    bc, xc, _, _ = two(xf1, carry=False)
    s1 = (t_k // T) == 1
    d1 = max((ba - bb)[e_k[s1], p_k[s1]].abs().max().item(), (xa - xb)[1].abs().max().item())
    d0 = max((ba - bb)[e_k[~s1], p_k[~s1]].abs().max().item(), (xa - xb)[0].abs().max().item())
    check("t10 isolation: stream 0's writes leave stream 1's reads unchanged", d1 < 1e-6 and d0 > 1e-4,
          f"(stream 1 {d1:.1e}, stream 0 {d0:.1e})")
    check("t10 isolation: stream 1's state identical", torch.allclose(ma.M[1], mb.M[1], atol=1e-6)
          and not torch.allclose(ma.M[0], mb.M[0]))
    dc = (xa - xc).abs().max().item()
    check("t10 carry: micro-batch 2 reads micro-batch 1's writes (vs no carried memory)", dc > 1e-4, f"({dc:.1e})")


def t11():
    """Distance-1 recall across micro-batches (the stream path's 5-d start state): micro-batch 1 of stream 0 writes
    k -> v (stream 1 writes k2 -> v2); micro-batch 2 reads k at its first token: stream 0 gets v, stream 1 does not."""
    g = torch.Generator().manual_seed(12)
    l2 = lambda x: x / x.norm()
    k, k2 = l2(torch.randn(16, generator=g, dtype=torch.float64)), l2(torch.randn(16, generator=g, dtype=torch.float64))
    v, v2 = torch.randn(16, generator=g, dtype=torch.float64), torch.randn(16, generator=g, dtype=torch.float64)
    z = lambda *s: torch.zeros(*s, dtype=torch.float64)
    q, u, w, lam = z(1, 2, 1, 16), z(1, 2, 1, 16), z(1, 2, 1, 16), z(1, 2, 1)
    u[0, 0, 0], w[0, 0, 0], u[0, 1, 0], w[0, 1, 0] = k, v, k2, v2
    segb, segp, valid = torch.tensor([[0, 1]]), torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool)
    _, Mend = scan_read(q, u, w, lam, z(1, 2, 1, 16, 16), segb, segp, valid, 2, 1, merge=True)
    q[0, 0, 0], q[0, 1, 0] = k, k
    O, _ = scan_read(q, u * 0, w * 0, lam, Mend, segb, segp, valid, 2, 1)
    check("t11 distance-1 recall across micro-batches (stream 0 reads v)", torch.allclose(O[0, 0, 0], v, atol=1e-10))
    check("t11 ... stream 1 does not (its own memory only)",
          torch.allclose(O[0, 1, 0], (k @ k2) * v2, atol=1e-10))


def t9(work):
    data = Path(r"X:\Datasets\world_change_mix")
    if not data.exists():
        check("t9 tiny train_lm run", False, "(no data)")
        return
    for mode in ("full", "detached"):
        name = f"fm_tiny_{mode}"
        cmd = [PY, "-B", str(HERE / "train_lm.py"), "--run-name", name, "--runs", str(work), "--arch", "loop",
               "--device", "cpu", "--data", str(data), "--data-mode", "stream", "--seq", "512", "--batch", "2",
               "--accum", "2", "--tokens", str(6 * 2 * 2 * 512), "--d", "64", "--heads", "4", "--head-dim", "16",
               "--moe", "8,2,64,64,auxfree", "--k-mean", "3", "--k-min", "3", "--k-max", "3", "--bptt", "2", "--state", "none",
               "--opt", "brain", "--eval-every", "3", "--eval-batches", "1", "--ckpt-every", "0",
               "--memory", "fastmem", "--fastmem-write-grad", mode, "--memory-save", "1"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        log = (r.stdout + r.stderr).strip().splitlines()
        ok = r.returncode == 0 and any("step      6/" in ln or "step 6" in ln or "final" in ln.lower() for ln in log)
        tail = [ln for ln in log if ln.startswith("step") or "fastmem" in ln or "Error" in ln][-4:]
        check(f"t9 tiny train_lm --memory fastmem --data-mode stream [{mode}]", r.returncode == 0,
              "\n    " + "\n    ".join(tail if ok else log[-15:]))


if __name__ == "__main__":
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="fm_tests_"))
    work.mkdir(parents=True, exist_ok=True)
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    full = dict(fastmem_size="full", fastmem_stream=0, fastmem_heads=2, fastmem_dk=16, fastmem_dv=16)
    for nm, fn in (("t1", t1), ("t2", t2_t3), ("t4", t4), ("t5", t5), ("t6", t6), ("t7", lambda: t7_t8(work)),
                   ("t7f", lambda: t7_t8(work, "full, one memory", **full)), ("t10", t10), ("t11", t11),
                   ("t9", lambda: t9(work))):
        if only is None or nm in only:
            fn()
    print(f"{sum(OK)}/{len(OK)} passed", flush=True)
    sys.exit(0 if all(OK) else 1)
