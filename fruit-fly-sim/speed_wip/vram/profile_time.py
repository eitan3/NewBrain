"""Where a full-size K=6 training step spends its time (torch.profiler): GPU kernel time vs wall time (is the step
GPU-bound or host/sync-bound?) and the top CUDA ops.

    python profile_time.py [--variant ...] [--batch 16 --accum 2]
"""
import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

import vram_common as vc

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="")
ap.add_argument("--batch", type=int, default=0)
ap.add_argument("--accum", type=int, default=0)
ap.add_argument("--K", type=int, default=6)
cli = ap.parse_args()
vc.gpu_guard()
torch.manual_seed(0)
S = vc.load(full=True)
if cli.variant:
    import variants
    variants.apply(S, cli.variant.split(","))
kw = dict(batch=cli.batch or None, accum=cli.accum or None)
vc.train_step(S, cli.K, **kw)
vc.train_step(S, cli.K, **kw)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t0 = time.time()
    vc.train_step(S, cli.K, **kw)
    torch.cuda.synchronize()
    wall = time.time() - t0
ka = prof.key_averages()
gpu = sum(e.self_device_time_total for e in ka) / 1e6
print(f"wall {wall:.3f} s, summed GPU kernel time {gpu:.3f} s ({100 * gpu / wall:.0f}% of wall)")
print(ka.table(sort_by="self_device_time_total", row_limit=30, max_name_column_width=60))
sync = [e for e in ka if "Synchronize" in e.key or "item" in e.key or "_local_scalar_dense" in e.key or "tolist" in e.key]
for e in sync:
    print(f"{e.key:50s} count {e.count:6d} cpu total {e.cpu_time_total / 1e3:9.1f} ms")
print("touched rows this step (V, V_e, P):", getattr(S, "last_counts", None), "of", S.mem.V.size(0), S.mem.V_e.size(0))
