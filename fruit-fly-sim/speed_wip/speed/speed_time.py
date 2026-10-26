"""Task 2: expected tok/s of a variant over the sampled-K distribution (median of --reps steps at K = 2, 6, 10, 16),
the number of host syncs per K=6 step, and the loss of the first (K=6) step from the same seed and data (math check).

    python speed_time.py --batch 8 --accum 4 --variant topk_nosync,route_nosync0 --tag nosync0
"""
import argparse
import json
import time
import warnings

import numpy as np
import torch

import sp_common as sc
from sp_common import vc
import step_lib

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--variant", default="")
ap.add_argument("--ks", default="2,6,10,16")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--tag", required=True)
cli = ap.parse_args()

sc.guard()
torch.manual_seed(0)
S = vc.load(full=True)
vn = [v for v in cli.variant.split(",") if v]
import variants  # noqa: E402
import speed_variants  # noqa: E402
for v in vn:
    if v in ("chunk_ce", "ckpt_pass", "bf16_mom", "nohomeo"):
        variants.apply(S, [v])
    else:
        speed_variants.apply(S, [v])
torch.manual_seed(1000)
np.random.seed(1234)
kw = dict(batch=cli.batch, accum=cli.accum)
first_loss = step_lib.train_step(S, 6, step=7000, **kw)

n_sync = [0]


def show(message, *a, **k):
    if "synchroniz" in str(message):
        n_sync[0] += 1


old = warnings.showwarning
warnings.showwarning = show
warnings.simplefilter("always")
torch.cuda.set_sync_debug_mode("warn")
step_lib.train_step(S, 6, step=7001, sync=False, **kw)
torch.cuda.set_sync_debug_mode("default")
warnings.showwarning = old

times = {}
for K in [int(k) for k in cli.ks.split(",")]:
    step_lib.train_step(S, K, **kw)
    ts = []
    for _ in range(cli.reps):
        torch.cuda.synchronize()
        t0 = time.time()
        step_lib.train_step(S, K, **kw)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    times[K] = sorted(ts)[len(ts) // 2]
    print(f"K={K}: {times[K]:.3f} s", flush=True)
et = vc.expected_time(times)
res = dict(tag=cli.tag, variant=cli.variant, batch=cli.batch, accum=cli.accum, first_loss=first_loss,
           syncs_K6=n_sync[0], times={str(k): v for k, v in times.items()}, expected_step_s=et,
           tok_s=cli.batch * cli.accum * 512 / et, peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
           peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
(sc.HERE / f"speed_{cli.tag}.json").write_text(json.dumps(res, indent=1))
print(f"{cli.tag}: {res['tok_s'] / 1e3:.2f}k tok/s, first loss {first_loss:.6f}, syncs/K6-step {n_sync[0]}, "
      f"peak {res['peak_alloc_gb']:.2f} GB", flush=True)
