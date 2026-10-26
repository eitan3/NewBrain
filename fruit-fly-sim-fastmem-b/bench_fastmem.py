"""FastMem speed / VRAM microbenchmark at 192 experts (lite + per stream by default; --size full --stream 0: the first
build).

CPU (the estimate; no GPU touched):
  set CUDA_VISIBLE_DEVICES=-1 & python -B bench_fastmem.py --device cpu --n-seq 1 --passes 6
GPU (when the GPU is free; one job at a time; a hard per-process cap on an explicit device; 2 threads):
  python -B bench_fastmem.py --device cuda:0 --n-seq 16 --passes 6 --cap-gb 16

One training micro-batch of n_seq x 512 tokens (6 routed of 192 + the shared expert + the global sheet), `passes`
passes, all with gradient (bptt = K): lite reads the start state at passes < 3, scans + writes at pass 3, reuses that
read at the later passes; then backward (--write-grad full: through the writes too), after_backward (the per-stream
carry) and after_step (AdamW, D's factored update). A warm-up micro-batch runs first (so the streams carry state and
the optimizer's moments exist).
CPU: times, the FLOPs of the dense ops counted analytically, and the memory from an allocation tracker
(TorchDispatchMode: every tensor an op creates, by storage, live until freed): the forward's peak and what stays live
for backward (the activations), the backward's peak, the optimizer's. The GPU estimate scales FLOPs and activations to
16 sequences (both linear in the tokens: the scan's segments are per sequence).
GPU: cuda events and torch.cuda.max_memory_allocated.
"""
import argparse
import contextlib
import math
import os
import sys
import time
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastmem import FastMem  # noqa: E402


def routing(N, E, k, gen, cap=1.5):
    """RoutedMLP's capacity layout for a random top-k routing (as tests_fastmem.routing; not imported: that module
    hides the GPU)."""
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
    kidx = (pos < C).nonzero().squeeze(-1)
    return e_sorted[kidx], pos[kidx], tok[kidx], (order % k)[kidx], counts, C, int(kidx.numel())


class MemTrack(TorchDispatchMode):
    """Live bytes of the tensors created by ops inside the mode (by storage; views count once; views of the known
    pre-existing storages are free), and the peak."""

    def __init__(self, known=()):
        super().__init__()
        self.live, self.peak, self.refs = 0, 0, {}
        self.known = {t.untyped_storage().data_ptr() for t in known}

    def reset_peak(self):
        self.peak = self.live

    def _drop(self, key):
        n, nb = self.refs[key]
        if n <= 1:
            del self.refs[key]
            self.live -= nb
        else:
            self.refs[key] = (n - 1, nb)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor) and t.device.type == "cpu":
                st = t.untyped_storage()
                key = st.data_ptr()
                if key in self.known:
                    continue
                if key in self.refs:
                    n, nb = self.refs[key]
                    self.refs[key] = (n + 1, nb)
                else:
                    self.refs[key] = (1, st.nbytes())
                    self.live += st.nbytes()
                    self.peak = max(self.peak, self.live)
                weakref.finalize(t, self._drop, key)
        return out


def flops(m, n_pairs, N, Cs, Csg, T, scan):
    """Forward FLOPs of one pass's dense ops: scan (projections, the M terms, the segment terms, D) or read-only (q,
    q M, D)."""
    H, dk, dv, d, gk = m.H, m.dk, m.dv, m.d, m.gk
    f = 0
    for P, h, cs in ((n_pairs, H, Cs), (N, H, T), (N * gk, 1, Csg)):          # routed / shared / global
        if scan:
            f += P * d * h * (2 * dk + dv + dk + dv)      # q, k, v, beta, omega
            f += 2 * P * h * dk * dv                      # q M, u M
            f += P * h * cs * (2 * dk + 2 * dv)           # u u^T, q u^T, the solve, P Delta (per segment: cs^2)
            f += P * h * dk * dv                          # the end state
        else:
            f += P * d * h * dk + P * h * dk * dv         # q, q M
        f += P * h * dv * d                               # D
    return 2 * f


def measure(a, dev, n_seq):
    E, k, d, T = 192, 6, 384, 512
    N = n_seq * T
    m = FastMem(d, E, fastmem_write_grad=a.write_grad, fastmem_ckpt=a.ckpt, fastmem_proj=a.proj,
                fastmem_size=a.size, fastmem_stream=a.stream).to(dev)
    with torch.no_grad():
        m.D.normal_(0, 0.01)
        m.D_g.normal_(0, 0.01)
    gen = torch.Generator().manual_seed(1)
    e_k, p_k, t_k, slot_k, counts, C, n_keep = routing(N, E, k, gen)
    e_k, p_k, t_k, slot_k, counts = (t.to(dev) for t in (e_k, p_k, t_k, slot_k, counts))
    xf = torch.randn(N, d, generator=gen).to(dev).requires_grad_(True)
    x = xf.detach().view(n_seq, T, d)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)

    def forward(train, passes):
        m.train(train)
        m.train_forward = train
        m.n_passes = passes
        if m.stream:
            m.set_streams(list(range(n_seq)) if train else None, [False] * n_seq)
        m.begin_forward()
        outs = []
        for kk in range(passes):
            buf = xf.detach().new_zeros(E, C, d).index_put((e_k, p_k), xf.detach()[t_k])
            outs.append(m(xf, buf, x, None, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=kk))
        return outs

    def clock(fn):
        sync()
        t0 = time.perf_counter()
        r = fn()
        sync()
        return r, time.perf_counter() - t0

    def step():
        outs = forward(True, a.passes)
        sum(b.pow(2).sum() + xs.pow(2).sum() for b, xs in outs).backward()
        m.after_backward()
        m.after_step(0)

    step()                                                   # (warm-up: carried state, AdamW moments)
    xf.grad = None
    with torch.no_grad():
        _, t_read = clock(lambda: forward(False, 1))         # (an eval pass at p' = 0: one scan, no write)
    known = list(m.parameters()) + list(m.buffers()) + [xf, x] + [
        t for st in (m._opt.state.values() if m._opt is not None else []) for t in st.values() if torch.is_tensor(t)]
    trk = MemTrack(known) if dev.type == "cpu" else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
    with trk if trk is not None else contextlib.nullcontext():
        outs, t_fw = clock(lambda: forward(True, a.passes))
        act = trk.live if trk else torch.cuda.memory_allocated(dev) - base
        pk_fw = trk.peak if trk else torch.cuda.max_memory_allocated(dev) - base
        loss = sum(b.pow(2).sum() + xs.pow(2).sum() for b, xs in outs)
        if trk:
            trk.reset_peak()
        _, t_bw = clock(lambda: loss.backward())
        pk_bw = trk.peak if trk else torch.cuda.max_memory_allocated(dev) - base
        gb = sum(p.grad.numel() * p.grad.element_size() for p in m.parameters() if p.grad is not None)
        del outs, loss
        if trk:
            trk.reset_peak()
        live0 = trk.live if trk else 0
        _, t_st = clock(lambda: (m.after_backward(), m.after_step(0)))
        pk_st = (trk.peak - live0) if trk else torch.cuda.max_memory_allocated(dev) - base
    one = torch.ones((), dtype=torch.bool, device=dev)
    ve = torch.zeros(E, C, dtype=torch.bool, device=dev).index_put((e_k, p_k), one)
    tk = torch.zeros(E, C, dtype=torch.long, device=dev).index_put((e_k, p_k), t_k)
    Cs = m._segments(tk, ve, T, n_seq)[2]
    Csg = max(8, 2 * T * m.gk // m.G)
    rp = a.passes if m.read_pass < 0 else min(m.read_pass, a.passes - 1)
    n_scan = a.passes if m.read_pass < 0 else 1
    n_read = 0 if m.read_pass < 0 else rp
    f_scan, f_read = flops(m, n_keep, N, Cs, Csg, T, True), flops(m, n_keep, N, Cs, Csg, T, False)
    f_fw = n_scan * f_scan + n_read * f_read
    G = 1e9
    n_par = sum(p.numel() for p in m.parameters())
    n_adam = n_par - m.D.numel() - m.D_g.numel()
    print(f"{dev}: fastmem {a.size}, {'per stream' if m.stream else 'one memory'}, {E} experts, {n_seq} x {T} tokens, "
          f"{a.passes} passes with gradient (scan at pass {rp if m.read_pass >= 0 else 'every'}; {n_keep} routed pairs, "
          f"longest segment {Cs}); write grad {a.write_grad}, ckpt {a.ckpt}, proj {a.proj}")
    print(f"  micro-batch: forward {t_fw * 1e3:.0f} ms, backward {t_bw * 1e3:.0f} ms, after_backward + after_step "
          f"{t_st * 1e3:.0f} ms (an eval scan pass: {t_read * 1e3:.0f} ms)")
    print(f"  memory: forward peak {pk_fw / G:.2f} GB, kept for backward {act / G:.2f} GB; backward peak "
          f"{pk_bw / G:.2f} GB (incl. the parameter grads {gb / G:.2f}); after_step peak {pk_st / G:.2f} GB")
    print(f"  state: {m.nbytes() / G:.3f} GB (M + Mg {(m.M.numel() + m.Mg.numel()) * 4 / G:.3f}, params "
          f"{n_par * 4 / G:.3f} = {n_par / 1e6:.1f}M); + grads {n_par * 4 / G:.3f}, AdamW moments {n_adam * 8 / G:.3f}")
    res = dict(act=act, fw=pk_fw - act, bw=pk_bw - gb - act, f_fw=f_fw, t_fw=t_fw, state=m.nbytes() + n_par * 4 + n_adam * 8,
               n_par=n_par, M=m.M.numel())
    if dev.type != "cpu":
        return res
    print(f"  forward {f_fw / G:.1f} GFLOP per micro-batch ({n_scan} scan pass {f_scan / G:.1f}, {n_read} read-only "
          f"passes {f_read / G:.1f} each); CPU achieved {f_fw / t_fw / G:.1f} GFLOP/s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-seq", default="1", help="sequences per micro-batch; CPU: two values (1,2) fit act = a + b n")
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--cap-gb", type=float, default=16.0)
    ap.add_argument("--gpu-tflops", type=float, default=50.0, help="the CPU estimate's effective rate (bf16 / fp32)")
    ap.add_argument("--write-grad", default="full")
    ap.add_argument("--size", default="lite")
    ap.add_argument("--stream", type=int, default=1)
    ap.add_argument("--ckpt", type=int, default=0, help="--fastmem-ckpt: the memory graph recomputed in backward")
    ap.add_argument("--proj", default="expert", help="--fastmem-proj expert | shared")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        tot = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.cap_gb * 1e9 / tot), dev)
    ns = [int(v) for v in str(a.n_seq).split(",")]
    rs = [measure(a, dev, n) for n in ns]
    if dev.type != "cpu":
        return
    G, r = 1e9, rs[-1]
    fit = lambda key: (rs[0][key] + (rs[-1][key] - rs[0][key]) / (ns[-1] - ns[0]) * (16 - ns[0])) if len(ns) > 1         else rs[0][key] * 16 / ns[0]
    accum = 2
    t_mb = 3 * r["f_fw"] * 16 / ns[-1] / (a.gpu_tflops * 1e12)
    print(f"GPU estimate (micro-batch 16 x 512, K = bptt = {a.passes}, accum {accum}, {a.gpu_tflops:.0f} TFLOP/s "
          f"effective; memory {'fitted a + b n over n = ' + str(ns) if len(ns) > 1 else 'scaled linearly'}):")
    print(f"  per micro-batch {t_mb * 1e3:.0f} ms (forward + 2x backward), per optimizer step ~{accum * t_mb * 1e3:.0f} ms"
          f" + after_step ~{(r['n_par'] * 16 + r['M'] * 8) / 0.9e12 * 1e3:.1f} ms (memory bound, ~0.9 TB/s)")
    print(f"  VRAM: activations kept for backward {fit('act') / G:.2f} GB per micro-batch ({a.passes} passes); backward "
          f"transient ~{fit('bw') / G:.2f} GB; forward transient ~{fit('fw') / G:.2f} GB; state + grads + moments "
          f"{r['state'] / G:.2f} GB")


if __name__ == "__main__":
    main()
