"""GPU profile of world_wave27b_small_12k's training step (the pause): the run's own checkpoint, its own flags, 5 steps,
NOTHING saved (the run folder is only read).

    python -B prof_wave27.py --code-dir <code> [--run C:\\Eitan\\Python\\NewBrain\\runs\\world_wave27b_small_12k]
                             [--device cuda:0] [--vram-cap-gb 16] [--warm 1] [--timed 2] [--sync 1] [--prof 1]
        --code-dir: the code to profile: the run's code_snapshot (before) or fruit-fly-sim-wave27c (after)

GPU rule: torch threads 2; torch.cuda.set_per_process_memory_fraction(cap / total, <explicit device index>) before
anything is allocated (an OOM instead of a spill into shared RAM). Loads <run>\\last.pt (the brain + the embedded memory,
train_lm.load_brain_state) and <run>\\last.memory.pt (memory_api.load_memory_file), read-only; the trainer's args are
the run's launch_args.txt parsed by the code's own train_lm parser (nothing of train_lm.main runs past the parse).
Steps (warm + timed + sync + prof, default 5): warm-up (Triton / allocator), wall-timed steps (cuda-synchronized), one
step under torch.cuda.set_sync_debug_mode("warn") (host syncs counted by call site), the profiled step(s). Prints:
  * s/step (the timed steps) and the peak VRAM
  * host syncs per step, by file:line
  * the ms split per step (CUDA and CPU) of the named regions: address, decide, recall (global / private; the key
    gather LUT, the chain), decode, readout, null, confirm, gate loss, writes (the write path; delta recall = _entry0,
    write_delta), learn (fix A / the private address), address update, AdamW, D update, sweep, empty_cache, plus the
    brain's forward / backward / optimizer
  * the top 25 ops by CUDA time and by CPU time
The regions are added here by wrapping the methods (so the old code, which lacks some record_function names, splits the
same way); no maths changes.
"""
import argparse
import collections
import functools
import importlib
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True

REGIONS = ("prof.forward", "prof.backward", "prof.brain_opt", "wave.call", "wave.address", "wave.decide",
           "wave.recall_global", "wave.recall_private", "wave.cols.key_gather_lut", "wave.cols.chain", "wave.decode",
           "wave.readout", "wave.select_writes", "wave.after_backward", "wave.null", "wave.addr_update",
           "wave.gate_loss", "wave.confirm", "wave.writes", "wave.delta_recall", "wave.write_delta", "wave.encode",
           "wave.learn_writes", "wave.learn_private", "wave.after_step", "wave.adamw", "wave.d_update", "wave.sweep",
           "empty_cache")


class _Args(Exception):
    pass


def trainer_args(train_lm, argv):
    """The run's args through the code's own parser (main() stops at parse_args)."""
    import argparse as ap_
    orig = ap_.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        raise _Args(orig(self, args, namespace))
    ap_.ArgumentParser.parse_args = grab
    old = sys.argv
    sys.argv = ["train_lm.py"] + argv
    try:
        train_lm.main()
    except _Args as e:
        return e.args[0]
    finally:
        ap_.ArgumentParser.parse_args = orig
        sys.argv = old
    raise RuntimeError("train_lm.main() did not parse its args")


def wrap(cls, name, region):
    """cls.name runs inside record_function(region) (the profiler's split; same maths)."""
    import torch
    fn = getattr(cls, name, None)
    if fn is None:
        return False
    @functools.wraps(fn)
    def w(*a, **k):
        with torch.profiler.record_function(region):
            return fn(*a, **k)
    setattr(cls, name, w)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--run", default=r"C:\Eitan\Python\NewBrain\runs\world_wave27b_small_12k")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vram-cap-gb", type=float, default=16.0)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--timed", type=int, default=2)
    ap.add_argument("--sync", type=int, default=1)
    ap.add_argument("--prof", type=int, default=1)
    ap.add_argument("--no-memory-file", action="store_true", help="skip last.memory.pt (last.pt embeds the memory)")
    ap.add_argument("--trace", default="", help="also write the profiler's chrome trace here (outside the run folder)")
    a = ap.parse_args()
    code = Path(a.code_dir).resolve()
    run = Path(a.run)
    sys.path.insert(0, str(code))
    import torch
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    idx = dev.index if dev.index is not None else 0
    dev = torch.device("cuda", idx)
    total = torch.cuda.get_device_properties(idx).total_memory
    frac = min(1.0, a.vram_cap_gb * 1e9 / total)
    torch.cuda.set_per_process_memory_fraction(frac, idx)
    torch.cuda.set_device(idx)
    print(f"code {code}; CUDA:{idx} capped at {frac * total / 1e9:.1f} GB of {total / 1e9:.1f} GB; 2 threads", flush=True)

    train_lm = importlib.import_module("train_lm")
    from memory_api import load_memory_file
    import wave_columns
    import wave_v21
    import wave_address
    argv = (run / "launch_args.txt").read_text().split() + ["--resume", str(run / "last.pt"), "--run-name", run.name]
    args = trainer_args(train_lm, argv)
    args.device = "cuda"
    for cls, name, region in ((wave_v21.WaveV21Memory, "__call__", "wave.call"),
                              (wave_v21.WaveV21Memory, "select_writes", "wave.select_writes"),
                              (wave_v21.WaveV21Memory, "after_backward", "wave.after_backward"),
                              (wave_v21.WaveV21Memory, "_gate_loss", "wave.gate_loss"),
                              (wave_v21.WaveV21Memory, "_writes", "wave.writes"),
                              (wave_v21.WaveV21Memory, "_entry0", "wave.delta_recall"),
                              (wave_v21.WaveV21Memory, "_learn_private_slow", "wave.learn_private"),
                              (wave_v21.WaveV21Memory, "after_step", "wave.after_step"),
                              (wave_v21.WaveV21Memory, "_adamw", "wave.adamw"),
                              (wave_v21.WaveV21Memory, "_d_update", "wave.d_update"),
                              (wave_v21.WaveV21Memory, "_null", "wave.null"),
                              (wave_columns.WaveColumns, "decode", "wave.decode"),
                              (wave_columns.WaveColumns, "confirm", "wave.confirm"),
                              (wave_columns.WaveColumns, "write_delta", "wave.write_delta"),
                              (wave_columns.WaveColumns, "encode", "wave.encode"),
                              (wave_address.WaveAddress, "learn_writes", "wave.learn_writes"),
                              (wave_address.WaveAddress, "update", "wave.addr_update")):
        wrap(cls, name, region)
    _ec = torch.cuda.empty_cache

    def empty_cache():
        with torch.profiler.record_function("empty_cache"):
            _ec()
    torch.cuda.empty_cache = empty_cache

    t0 = time.time()
    rs = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    from loop_brain import LoopBrain
    model = LoopBrain(**rs["cfg"]).to(dev)
    train_lm.load_brain_state(model, rs["state"], args)
    side = run / "last.memory.pt"
    if side.exists() and not a.no_memory_file:
        info = load_memory_file(side, model)
        print(f"memory: loaded {side} (saved at step {info.get('step')})", flush=True)
    opt = train_lm.build_opt(model, args)
    if args.opt == "brain":
        opt.t = rs["opt"]["t"]
        opt.state = {n: {k: v.to(dev) for k, v in st.items()} for n, st in rs["opt"]["state"].items()}
    else:
        opt.load_state_dict(rs["opt"]["state"])
    tr = rs["trainer"]
    if tr.get("group_passes") is not None:
        model.group_passes = list(tr["group_passes"])
    step0 = rs["step"]
    del rs
    bm = model.bm
    train_lm.apply_vram_savers(model, args)
    if args.chunk_ce:
        model.chunk_ce = args.chunk_ce
    bm.trainer_setup(args)
    data = train_lm.Data(Path(args.data), args.seq, "cuda")
    model.train()
    print(f"loaded step {step0} in {time.time() - t0:.0f} s; {bm.describe(args)}", flush=True)
    F_ = torch.nn.functional
    frag = args.frag_gb * 1e9

    def near_cap():
        return args.frag_gb > 0 and torch.cuda.memory_reserved(idx) > frac * total - frag

    def step(i):
        if args.arch == "loop":
            model.K = model.sample_K() if args.k_min != args.k_max else args.k_mean
            if args.accum > 1:
                model.aux_k_fixed = model.sample_aux_k(model.K)
            model.group_offset = model.sample_group_offset()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            x, y = data.batch(args.batch)
            if near_cap():
                torch.cuda.empty_cache()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.profiler.record_function("prof.forward"):
                    logits, loss = model(x, y)
                    bm.select_writes(logits, y, x, ce_h=model.last_readout if logits is None else None)
                    total_ = loss
                    aux = model.aux_loss()
                    if aux is not None:
                        total_ = total_ + aux
                    if model.concept_loss is not None:
                        total_ = total_ + args.concept_w * model.concept_loss
                    if model.aux_x is not None:
                        if args.chunk_ce:
                            total_ = total_ + 0.2 * train_lm.chunked_readout(model.heads["lm"], model.aux_x, y,
                                                                             args.chunk_ce)[0].mean()
                        else:
                            total_ = total_ + 0.2 * F_.cross_entropy(model.heads["lm"](model.aux_x).float().flatten(0, 1),
                                                                     y.flatten())
            with torch.profiler.record_function("prof.backward"):
                (total_ / args.accum if args.accum > 1 else total_).backward()
            bm.after_backward()
            del logits, total_
        with torch.profiler.record_function("prof.brain_opt"):
            torch.nn.utils.clip_grad_norm_([p for _, p in train_lm.brain_params(model)], 1.0)
            opt.step() if args.opt == "adamw" else opt.step(1.0)
            model.blocks[0].mlp.apply_balance()
        if near_cap():
            torch.cuda.empty_cache()
        bm.after_step(step0 + i, refit_every=args.gm_ivf_refit)
        if near_cap():
            torch.cuda.empty_cache()

    i = 0
    for _ in range(a.warm):
        step(i)
        i += 1
    torch.cuda.synchronize()
    ts = []
    for _ in range(a.timed):
        t = time.time()
        step(i)
        torch.cuda.synchronize()
        ts.append(time.time() - t)
        i += 1
    if ts:
        print(f"\ntimed steps: {', '.join(f'{t:.3f}' for t in ts)} s/step (mean {sum(ts) / len(ts):.3f}); peak "
              f"reserved {torch.cuda.max_memory_reserved(idx) / 1e9:.2f} GB, allocated "
              f"{torch.cuda.max_memory_allocated(idx) / 1e9:.2f} GB", flush=True)
    sites = collections.Counter()
    for _ in range(a.sync):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                step(i)
            finally:
                torch.cuda.set_sync_debug_mode(0)
        i += 1
        for w in ws:
            if "synchroniz" in str(w.message).lower():
                sites[f"{Path(w.filename).name}:{w.lineno}"] += 1
    if a.sync:
        print(f"\nhost syncs per step: {sum(sites.values()) / a.sync:.0f} (by call site, per step)", flush=True)
        for s_, n_ in sites.most_common(40):
            print(f"  {n_ / a.sync:8.1f}  {s_}")
    if a.prof:
        from torch.profiler import ProfilerActivity, profile
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.prof):
                step(i)
                i += 1
            torch.cuda.synchronize()
        ev = prof.key_averages()
        n = a.prof
        print(f"\nregions, ms per step (CUDA time = device time inside the region; CPU = host wall in the region; "
              f"regions nest: wave.call > address / decide / recall / readout > decode; after_backward > null / "
              f"gate_loss / confirm / writes > delta_recall / write_delta / learn_*; after_step > adamw / d_update / "
              f"sweep):")
        byname = {e.key: e for e in ev}
        for r in REGIONS:
            e = byname.get(r)
            if e is None:
                continue
            cu = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            print(f"  {r:28s} CUDA {cu / 1e3 / n:9.1f}  CPU {e.cpu_time_total / 1e3 / n:9.1f}  calls {e.count // n}")
        key = "device_time_total" if hasattr(ev[0], "device_time_total") else "cuda_time_total"
        print(f"\ntop 25 ops by CUDA time:\n{ev.table(sort_by=key.replace('_total', '_total'), row_limit=25)}")
        print(f"\ntop 25 ops by CPU time:\n{ev.table(sort_by='self_cpu_time_total', row_limit=25)}")
        if a.trace:
            prof.export_chrome_trace(a.trace)
    print("done; nothing saved", flush=True)


if __name__ == "__main__":
    main()
