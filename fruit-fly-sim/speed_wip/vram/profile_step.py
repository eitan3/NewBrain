"""Profile one full-size training step (and optionally record a memory snapshot of it).

    python profile_step.py --ks 6,10,16 --snapshot 10 --out step_profile.json
    python profile_step.py --time 2,6,10,16 --reps 3          (speed only)
    python profile_step.py --batch 8 --accum 4 ...            (micro-batch variant)
    python profile_step.py --variant ckpt_pass,chunk_ce ...   (monkeypatched what-ifs, see variants.py)
"""
import argparse
import json
import pickle
import time

import torch

import vram_common as vc

ap = argparse.ArgumentParser()
ap.add_argument("--ks", default="10")
ap.add_argument("--snapshot", type=int, default=0, help="record a memory-history snapshot of the step at this K")
ap.add_argument("--time", default="", help="time steps at these K (no probes)")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--batch", type=int, default=0)
ap.add_argument("--accum", type=int, default=0)
ap.add_argument("--variant", default="")
ap.add_argument("--empty-cache-steps", type=int, default=0, help="empty_cache before each measured step")
ap.add_argument("--out", default="step_profile.json")
ap.add_argument("--no-full", action="store_true")
cli = ap.parse_args()

vc.gpu_guard()
torch.manual_seed(0)
S = vc.load(full=not cli.no_full)
if cli.variant:
    import variants
    variants.apply(S, cli.variant.split(","))
res = dict(variant=cli.variant, batch=cli.batch or S.args.batch, accum=cli.accum or S.args.accum)
torch.cuda.synchronize()
res["after_load"] = vc.mstat("after_load")
pb = vc.persistent_breakdown(S)
res["persistent"] = pb
print("after load", res["after_load"], flush=True)
print({k: (round(v / 1e9, 3) if isinstance(v, (int, float)) else {kk: round(vv / 1e9, 3) for kk, vv in v.items()})
       for k, v in pb.items()}, flush=True)

kw = dict(batch=cli.batch or None, accum=cli.accum or None)
# warm-up: the newborn parameters' optimizer state and the allocator's first segments
t0 = time.time()
vc.train_step(S, 6, **kw)
res["warmup_s"] = time.time() - t0
res["after_warmup"] = vc.mstat("after_warmup")
print("after warmup", res["after_warmup"], flush=True)

phases = {}
for K in [int(k) for k in cli.ks.split(",") if k]:
    if cli.empty_cache_steps:
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rows = []

    def probe(tag):
        torch.cuda.synchronize()
        st = vc.mstat(tag)
        rows.append(st)
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    vc.train_step(S, K, probe=probe, **kw)
    dt = time.time() - t0
    top = max(r["peak_alloc"] for r in rows)
    topr = max(r["peak_reserved"] for r in rows)
    phases[K] = dict(rows=rows, peak_alloc=top, peak_reserved=topr, seconds=dt,
                     after=vc.mstat("after"), stats=torch.cuda.memory_stats())
    print(f"K={K}: peak alloc {top:.2f} GB, peak reserved {topr:.2f} GB, {dt:.2f} s; by phase "
          + " ".join(f"{r['tag']}:{r['peak_alloc']:.2f}/{r['reserved']:.2f}" for r in rows), flush=True)
res["phases"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "stats"} for k, v in phases.items()}
res["alloc_stats"] = {str(k): {n: v["stats"].get(n) for n in ("num_alloc_retries", "num_ooms", "num_device_alloc",
                                                              "num_device_free", "inactive_split_bytes.all.peak",
                                                              "reserved_bytes.all.peak", "allocated_bytes.all.peak")}
                      for k, v in phases.items()}

if cli.time:
    times = {}
    for K in [int(k) for k in cli.time.split(",")]:
        vc.train_step(S, K, **kw)  # (one untimed: shapes of this K)
        ts = []
        for _ in range(cli.reps):
            torch.cuda.synchronize()
            t0 = time.time()
            vc.train_step(S, K, **kw)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        times[K] = sorted(ts)[len(ts) // 2]
        print(f"time K={K}: {times[K]:.3f} s (median of {cli.reps}) peak reserved {torch.cuda.max_memory_reserved() / 1e9:.2f}",
              flush=True)
    et = vc.expected_time(times)
    res["times"] = {str(k): v for k, v in times.items()}
    res["expected_step_s"] = et
    res["tok_s"] = S.args.batch * S.args.accum * S.args.seq / et
    print(f"expected step {et:.3f} s -> {res['tok_s'] / 1e3:.1f}k tok/s", flush=True)

if cli.snapshot:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(max_entries=3_000_000, stacks="python")
    vc.train_step(S, cli.snapshot, **kw)
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    path = vc.HERE / f"snapshot_K{cli.snapshot}{'_' + cli.variant.replace(',', '+') if cli.variant else ''}.pickle"
    with open(path, "wb") as f:
        pickle.dump(snap, f)
    res["snapshot"] = dict(path=str(path), base_alloc=base / 1e9, peak_alloc=torch.cuda.max_memory_allocated() / 1e9,
                           peak_reserved=torch.cuda.max_memory_reserved() / 1e9)
    print("snapshot", res["snapshot"], flush=True)

(vc.HERE / cli.out).write_text(json.dumps(res, indent=1, default=str))
print("wrote", vc.HERE / cli.out, flush=True)
