"""Profile of the wave memory module alone at FULL size (the GPU smoke's setup; run it when the GPU is free, log the job
to the coordinator first):

    python -B prof_wave.py [--device cuda:0] [--vram-cap-gb 16] [--warm 3] [--steps 2] [--accum 2] [--passes 6]
    python -B prof_wave.py --tiny --device cpu          (a CPU dry run of this script itself; CUDA_VISIBLE_DEVICES=-1)

GPU rule: torch threads 2, torch.cuda.set_per_process_memory_fraction(cap / total, <explicit device index>) before
anything is allocated, so it OOMs instead of spilling into shared RAM.

The module (CodedWaveMemory, the defaults: the column store at 1,024 places, the pre-read gate + per-expert output
layer, the private parts) is driven as the trainer drives it (gpu_smoke_wave.py: per micro-batch begin_forward, one
__call__ per pass, a stand-in loss, select_writes, backward, after_backward; after_step per optimizer step), with the
reference shapes (8,192 tokens, top-6 of 768 experts, K = 6, accum 2). The first --warm steps are not profiled; then
torch.profiler records --steps optimizer steps. Prints:
  * recall calls per micro-batch forward (must be 1: pass 3 recalls, passes 4-5 reuse)
  * the named regions (wave.address, wave.decide, wave.recall_global, wave.cols.key_gather_lut, wave.cols.chain,
    wave.recall_private, wave.readout, wave.null, wave.write, wave.sweep): total ms per step (CUDA and CPU)
  * the top 25 ops by CUDA time and by CPU time; kernel launches per step
  * host syncs: one step under torch.cuda.set_sync_debug_mode("warn"), the warnings counted by call site
"""
import argparse
import collections
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

REGIONS = ("wave.address", "wave.decide", "wave.recall_global", "wave.cols.key_gather_lut", "wave.cols.chain",
           "wave.recall_private", "wave.readout", "wave.null", "wave.write", "wave.sweep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vram-cap-gb", type=float, default=16.0)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--experts", type=int, default=768)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default="", help="also write the profiler's chrome trace here")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    if cuda:                                   # (the laptop rule, with an explicit device index)
        idx = dev.index if dev.index is not None else 0
        dev = torch.device("cuda", idx)
        total = torch.cuda.get_device_properties(idx).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_cap_gb * 1e9 / total), idx)
        print(f"CUDA:{idx} memory capped at {min(total, a.vram_cap_gb * 1e9) / 1e9:.1f} GB of {total / 1e9:.1f} GB",
              flush=True)
    from memory_api import budget_line, byte_budget
    from wave_memory import CodedWaveMemory
    if a.tiny:
        a.tokens, a.experts, a.passes, a.warm = 512, 12, 6, 3
    kw = dict(wave_warm=2 * a.accum, wave_k_routed=a.topk)
    if a.tiny:
        kw.update(wave_side=4, wave_slices=32, wave_codes=16)
    D = 32 if a.tiny else 384
    torch.manual_seed(0)
    m = CodedWaveMemory(D, a.experts + 1, **kw).to(dev).train()
    print(m.describe(), flush=True)
    print(budget_line(byte_budget(m, 1.9e9)), flush=True)
    g = torch.Generator().manual_seed(1)
    base = torch.randn(4096, D, generator=g) * 4
    N, E, k = a.tokens, a.experts, a.topk
    T = 512 if N % 512 == 0 else N
    B = N // T
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    def micro():
        m.train_forward = True
        m.begin_forward()
        r0 = m.cnt["recalls"]
        loss = 0.0
        for kk in range(a.passes):
            xf = (base[torch.randint(0, len(base), (N,), generator=g)] + 0.3 * torch.randn(N, D, generator=g)).to(dev)
            xf.requires_grad_(True)
            idx_ = torch.rand(N, E, generator=g).topk(k, -1).indices.to(dev)
            flat = idx_.flatten()
            order = flat.argsort(stable=True)
            e_k, t_k, slot_k = flat[order], order // k, order % k
            counts = torch.bincount(flat, minlength=E)
            p_k = torch.arange(N * k, device=dev) - (counts.cumsum(0) - counts)[e_k]
            C = int(counts.max())
            buf = xf.new_zeros(E, C, D).index_put((e_k, p_k), xf[t_k])
            bo, xs = m(xf, buf, xf.view(B, T, D), None, e_k, p_k, t_k, slot_k, counts, N, E, C, N * k, call=kk)
            loss = loss + (bo.float() * 1e-3).sum() + (xs.float() * 1e-3).sum()
        ce = (torch.rand(N, generator=g) * 6).to(dev)
        H = (torch.rand(N, generator=g) * 6).to(dev)
        m.select_writes(None, torch.zeros(B, T, dtype=torch.long, device=dev), None, ce_h=(ce, H, 32768))
        loss.backward()
        m.after_backward()
        return m.cnt["recalls"] - r0

    def step(i):
        rec = [micro() for _ in range(a.accum)]
        m.after_step(i)
        sync()
        return rec

    for i in range(a.warm):
        t0 = time.time()
        rc = step(i)
        print(f"warm step {i}: {1e3 * (time.time() - t0):.0f} ms; ready {m.ready}; recalls per micro-batch {rc}",
              flush=True)
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
    t0 = time.time()
    with torch.profiler.profile(activities=acts, record_shapes=False) as prof:
        rcs = [step(a.warm + i) for i in range(a.steps)]
    wall = (time.time() - t0) / a.steps
    print(f"\nPROFILED {a.steps} steps: {1e3 * wall:.0f} ms per step (wall, incl. profiler overhead); recalls per "
          f"micro-batch {rcs} (must be all 1)", flush=True)
    ev = prof.key_averages()
    by = {e.key: e for e in ev}
    attr = "device_time_total" if hasattr(ev[0], "device_time_total") else "cuda_time_total"
    print("\nREGIONS (ms per step): name | CUDA | CPU")
    for r in REGIONS:
        e = by.get(r)
        if e is not None:
            print(f"  {r:28s} {getattr(e, attr, 0) / 1e3 / a.steps:9.1f} {e.cpu_time_total / 1e3 / a.steps:9.1f}")
    if cuda:
        print("\nTOP 25 by CUDA time:")
        print(ev.table(sort_by="cuda_time_total" if attr == "cuda_time_total" else "device_time_total", row_limit=25))
        kern = [e for e in prof.events() if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA]
        print(f"kernel launches per step: {len(kern) / a.steps:.0f}")
    print("\nTOP 25 by CPU time:")
    print(ev.table(sort_by="self_cpu_time_total", row_limit=25))
    if a.out:
        prof.export_chrome_trace(a.out)
        print(f"trace: {a.out}")
    if cuda:                                    # host syncs: one step in warn mode
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            step(a.warm + a.steps)
        torch.cuda.set_sync_debug_mode(0)
        cnt = collections.Counter(f"{x.filename.split(chr(92))[-1]}:{x.lineno}" for x in w
                                  if "synchroniz" in str(x.message))
        print(f"\nHOST SYNCS in one step: {sum(cnt.values())}")
        for site, n in cnt.most_common(30):
            print(f"  {n:5d}  {site}")
        print(f"peak VRAM allocated {torch.cuda.max_memory_allocated(dev) / 1e9:.2f} GB, reserved "
              f"{torch.cuda.max_memory_reserved(dev) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
