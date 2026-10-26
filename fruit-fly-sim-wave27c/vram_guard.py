"""Launch-time VRAM guard (GPU; run it BEFORE a long run, never next to one): ~20 training steps of the run's own
train_lm flags at the FINAL size (--start = the final heads / experts), with the blob memory fitted early, one IVF
refit, one evaluate() and the worst-case pass count (K = 16 on every step), under the run's --mem-fraction cap.
Refuses (exit code 1) when the peak RESERVED memory exceeds cap - margin (default 2 GB) or the steps OOM.

    python fruit-fly-sim/vram_guard.py --config runs/<name>/config.json [--margin-gb 2] [--steps 20] [--K 16]
                                       [--extra "--vram-savers 1"]
    python fruit-fly-sim/vram_guard.py --argv "<the train_lm.py arguments>" ...

What changes vs the real run (memory-neutral or worst-case on purpose): the start size = the final size (no births),
the fit at step 5 (window 3; the stores are emptier, the shapes are the same static shapes), the IVF refit at step 15,
eval at step 10 (2 batches), lam = 1 (spikes on, as after the anneal), K = 16, its own run folder under a temp dir.
"""
import argparse
import json
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GB = 1e9
DROP = {"--run-name", "--runs", "--start", "--stop-at", "--resume", "--tokens", "--save-at", "--keep-ckpts",
        "--save-report-ckpts", "--diag-every", "--gm-fit-step", "--gm-fit-window", "--gm-ivf-refit", "--eval-every",
        "--eval-batches", "--ckpt-every", "--health-every"}   # (each takes one value)


def strip(argv):
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in DROP:
            i += 2
            continue
        out.append(a)
        i += 1
    return out


# VRAM saved by each saver at the final size of world_blob25d (768 experts, 36 heads, batch 16 x 2, K 10), from the
# CPU allocation tracker (speed_wip / the r7 VRAM study; calibrated to 0.1 GB of the GPU at step 2000); speed cost
# estimated (recompute share of the step). Cheapest first.
SAVERS = (("--vram-readout-lean 1", 3.2, "0%"), ("--vram-ckpt-concept 1", 0.9, "~0.5%"),
          ("--vram-ckpt-gmdist 1", 0.6, "~0.5%"), ("--vram-ckpt-norm 1", 1.5, "~2%"),
          ("--vram-ckpt-experts 1", 2.4, "~3-4%"), ("--vram-ckpt-mixer 1", 2.0, "~3-4%"))


def suggest(argv, excess_gb):
    """The cheapest savers not already on (train_lm defaults: readout-lean on, the rest off) that cover excess_gb."""
    def is_on(flag):
        name, _ = flag.split()
        if "--vram-savers" in argv and argv[argv.index("--vram-savers") + 1] != "0":
            return True
        if name in argv:
            return argv[argv.index(name) + 1] != "0"
        return name == "--vram-readout-lean"          # (default on)
    pick, got = [], 0.0
    for flag, gb, cost in SAVERS:
        if got >= excess_gb:
            break
        if not is_on(flag):
            pick.append(f"{flag} (-{gb} GB, {cost} time)")
            got += gb
    if got < excess_gb:
        return (f"over by {excess_gb:.1f} GB; even all savers ({got:.1f} GB est.) may not fit: also consider "
                f"--batch 8 --accum 4 (about -7 GB, -4.5% speed, not bit-identical) " + "; ".join(pick))
    return f"over by {excess_gb:.1f} GB; add: " + "; ".join(pick) + f" (est. -{got:.1f} GB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="a run's config.json (its argv is used)")
    ap.add_argument("--argv", default="", help="train_lm.py arguments (instead of --config)")
    ap.add_argument("--argv-file", default="", help="a file holding the train_lm.py arguments (one line)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--margin-gb", type=float, default=2.0)
    ap.add_argument("--extra", default="", help="extra train_lm arguments, e.g. '--vram-savers 1'")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--runs", default="", help="the runs folder whose gpu.lock must be free (default: the config's)")
    h = ap.parse_args()
    if h.config:
        cfg = json.loads(Path(h.config).read_text())
        argv = list(cfg["argv"][1:])
    elif h.argv_file:
        argv = shlex.split(Path(h.argv_file).read_text(encoding="utf-8").strip(), posix=False)
    else:
        argv = shlex.split(h.argv, posix=False)
    argv = [a[1:-1] if len(a) > 1 and a[0] == a[-1] == '"' else a for a in argv]
    import torch
    assert torch.cuda.is_available(), "vram_guard needs the GPU"
    import train_lm as tl
    # never next to a live run: its runs folder's gpu.lock (the guard's own run folder is a temp dir)
    runs = Path(h.runs or (cfg.get("runs", "") if h.config else "") or
                (argv[argv.index("--runs") + 1] if "--runs" in argv else "") or HERE.parent / "runs")
    lock = runs / "gpu.lock"
    if lock.exists():
        try:
            owner, holder = lock.read_text().split()[:2]
            if tl.lock_holder_alive(int(owner)):
                sys.exit(f"vram_guard: the GPU is in use by run {holder} (pid {owner}); refusing to run next to it")
        except ValueError:
            pass
    from loop_brain import LoopBrain

    final = {}
    for flag, key in (("--heads", "heads"), ("--moe", "experts")):
        if flag in argv:
            v = argv[argv.index(flag) + 1]
            final[key] = int(v.split(",")[0])
    start = ",".join(f"{k}={v}" for k, v in final.items())
    work = Path(h.workdir or tempfile.mkdtemp(prefix="vram_guard_"))
    seq = int(argv[argv.index("--seq") + 1]) if "--seq" in argv else 512
    batch = int(argv[argv.index("--batch") + 1]) if "--batch" in argv else 16
    accum = int(argv[argv.index("--accum") + 1]) if "--accum" in argv else 1
    steps = h.steps
    run_argv = (strip(argv) + ["--run-name", "guard", "--runs", str(work), "--tokens", str((steps + 1) * batch * accum * seq),
                               "--stop-at", str(steps), "--gm-fit-step", "5", "--gm-fit-window", "3",
                               "--gm-ivf-refit", str(min(15, steps - 1)), "--eval-every", "10", "--eval-batches", "2",
                               "--ckpt-every", "0", "--keep-ckpts", "0", "--save-report-ckpts", "0"]
                + (["--start", start] if start else []) + shlex.split(h.extra))
    # worst case on purpose: K on every step, spikes fully on
    LoopBrain.sample_K = lambda self: h.K
    set_lam0 = LoopBrain.set_lam
    LoopBrain.set_lam = lambda self, lam: set_lam0(self, 1.0)
    print("vram_guard: train_lm " + " ".join(run_argv), flush=True)
    sys_argv = sys.argv
    sys.argv = ["train_lm.py", *run_argv]
    t0, err = time.time(), None
    try:
        tl.main()
    except torch.cuda.OutOfMemoryError as e:  # (refuse below)
        err = f"OOM: {str(e).splitlines()[0]}"
    finally:
        sys.argv = sys_argv
        out = sys.stdout.file if hasattr(sys.stdout, "file") else None
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        if out is not None:
            out.close()
    total = torch.cuda.get_device_properties(0).total_memory
    frac = float(argv[argv.index("--mem-fraction") + 1]) if "--mem-fraction" in argv else 1.0
    cap = frac * total
    peak_a, peak_r = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    rep = dict(peak_allocated_gb=peak_a / GB, peak_reserved_gb=peak_r / GB, cap_gb=cap / GB, margin_gb=h.margin_gb,
               K=h.K, steps=steps, minutes=(time.time() - t0) / 60, error=err, workdir=str(work))
    ok = err is None and peak_r <= cap - h.margin_gb * GB
    rep["verdict"] = "OK" if ok else "REFUSE"
    if not ok:
        rep["suggest"] = suggest(argv + shlex.split(h.extra), peak_r / GB - (cap / GB - h.margin_gb))
    print(json.dumps(rep, indent=1), flush=True)
    if not ok:
        print(f"vram_guard: REFUSE: peak reserved {peak_r / GB:.2f} GB > cap {cap / GB:.2f} - margin {h.margin_gb} GB"
              + (f" ({err})" if err else "") + f". {rep['suggest']}", flush=True)
    else:
        print(f"vram_guard: OK: peak reserved {peak_r / GB:.2f} GB <= cap {cap / GB:.2f} - margin {h.margin_gb} GB",
              flush=True)
    (work / "vram_guard.json").write_text(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
