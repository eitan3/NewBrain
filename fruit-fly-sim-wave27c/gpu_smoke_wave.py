"""GPU smoke of the wave memory at FULL size (not run by the tests; run it when the GPU is free):

    python -B gpu_smoke_wave.py [--steps 5] [--warm 2] [--passes 6] [--accum 2] [--vram-cap-gb 16]

(GPU rule: torch threads 2 and torch.cuda.set_per_process_memory_fraction at --vram-cap-gb, so it OOMs instead of
spilling into shared RAM; log the job to the coordinator before starting it.)

One CodedWaveMemory (the defaults: d 384, 769 experts, the v2 column store at 1,024 places, global sheet only) on CUDA,
driven exactly as the trainer drives it, with the reference run's shapes (world_blob25d_12k: batch 16 x seq 512 =
8,192 tokens per micro-batch, accum 2, top-6 of 768 routed experts, K = 6 passes, all in the gradient window):
per micro-batch begin_forward, one RoutedMLP read per pass (the module's __call__), a stand-in loss on the injected
inputs, select_writes, backward, after_backward (writes + confirms + the address update); after_step per step
(forgetting, the gate's open bias). Thoughts: the W-bench capture's real xf (pass 3 fit rows, tiled, plus noise) if
available, else Gaussian at the same scale. The first --warm steps are excluded from the timing (they warm the
address and the caches). Reports per optimizer step: read ms, backward-side memory ms (after_backward), after_step ms,
the memory's total ms, peak VRAM (allocated / reserved), the state's bytes; compare with the blob memory's ~1.27 s
per step and ~1.9 GB. --tiny --device cpu: a small CPU dry run of this script itself (set CUDA_VISIBLE_DEVICES=-1).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from memory_api import budget_line, byte_budget  # noqa: E402
from wave_memory import CodedWaveMemory  # noqa: E402

CAP = Path(r"C:\Eitan\Python\NewBrain\speed_wip\mem26\cap\w1c_6k.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--experts", type=int, default=768)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--read-chunk", type=int, default=512)
    ap.add_argument("--vram-cap-gb", type=float, default=16.0, help="per-process CUDA memory cap (OOM, never spill)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if dev.type == "cuda":     # (the laptop rule: OOM instead of spilling into shared RAM)
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_cap_gb * 1e9 / total), dev)
        print(f"CUDA memory capped at {min(total, a.vram_cap_gb * 1e9) / 1e9:.1f} GB of {total / 1e9:.1f} GB", flush=True)
    if a.tiny:
        a.tokens, a.experts, a.passes = 512, 12, 5
    torch.manual_seed(0)
    kw = dict(wave_warm=2 * a.accum, wave_read_chunk=a.read_chunk)
    if a.tiny:
        kw.update(wave_side=4, wave_slices=32, wave_codes=16)                    # (the column store's address shape)
    D = 384 if not a.tiny else 32
    m = CodedWaveMemory(D, a.experts + 1, **kw).to(dev).train()
    print(m.describe(), flush=True)
    print(budget_line(byte_budget(m, 1.9e9)), flush=True)
    if CAP.exists() and not a.tiny:
        Z = np.load(CAP)
        base = torch.from_numpy(Z["fit"][3, :, :384].astype(np.float32))
    else:
        base = torch.randn(4096, D) * 4
    g = torch.Generator().manual_seed(1)
    N, E, k = a.tokens, a.experts, a.topk
    T = 512 if N % 512 == 0 else N
    B = N // T
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    cuda = dev.type == "cuda"
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    rec = []
    for step in range(a.warm + a.steps):
        t_read = t_back = 0.0
        for _ in range(a.accum):
            m.train_forward = True
            m.begin_forward()
            loss = 0.0
            for kk in range(a.passes):
                xf = (base[torch.randint(0, len(base), (N,), generator=g)] + 0.3 * torch.randn(N, D, generator=g)).to(dev)
                xf.requires_grad_(True)
                idx = torch.rand(N, E, generator=g).topk(k, -1).indices.to(dev)
                flat = idx.flatten()
                order = flat.argsort(stable=True)
                e_k, t_k = flat[order], order // k
                slot_k = order % k
                counts = torch.bincount(flat, minlength=E)
                p_k = torch.arange(N * k, device=dev) - (counts.cumsum(0) - counts)[e_k]
                C = int(counts.max())
                buf = xf.new_zeros(E, C, D)
                buf = buf.index_put((e_k, p_k), xf[t_k])
                x = xf.view(B, T, D)
                sync()
                t0 = time.time()
                bo, xs = m(xf, buf, x, None, e_k, p_k, t_k, slot_k, counts, N, E, C, N * k, call=kk)
                sync()
                t_read += time.time() - t0
                loss = loss + (bo.float() * 1e-3).sum() + (xs.float() * 1e-3).sum()
            ce = (torch.rand(N, generator=g) * 6).to(dev)
            H = (torch.rand(N, generator=g) * 6).to(dev)
            m.select_writes(None, torch.zeros(B, T, dtype=torch.long, device=dev), None, ce_h=(ce, H, 32768))
            loss.backward()
            sync()
            t0 = time.time()
            m.after_backward()
            sync()
            t_back += time.time() - t0
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        t0 = time.time()
        m.after_step(step)
        sync()
        t_after = time.time() - t0
        line = (f"step {step}: read {1e3 * t_read:.0f} ms ({a.accum} x {a.passes} reads of {N} tokens), after_backward "
                f"{1e3 * t_back:.0f} ms, after_step {1e3 * t_after:.0f} ms, memory total "
                f"{1e3 * (t_read + t_back + t_after):.0f} ms; ready {m.ready}")
        if step >= a.warm:
            rec.append((t_read, t_back, t_after))
        print(line + ("  (warm-up, not timed)" if step < a.warm else ""), flush=True)
    r = np.array(rec)
    if len(r):
        print(f"MEAN over {len(r)} steps: read {1e3 * r[:, 0].mean():.0f} ms, write side {1e3 * r[:, 1].mean():.0f} ms, "
              f"after_step {1e3 * r[:, 2].mean():.0f} ms, total {1e3 * r.sum(1).mean():.0f} ms per step "
              f"(blob memory today: ~1270 ms)", flush=True)
    if cuda:
        print(f"peak VRAM allocated {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, reserved "
              f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB (the memory alone + this stand-in's activations)",
              flush=True)
    print(f"store {m.store.fill()}; log {dict((k_, v) for k_, v in m.log_record().items() if 'gate' in k_ or 'writes' in k_ or 'recalls' in k_ or 'fwd' in k_)}")


if __name__ == "__main__":
    main()
