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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="a run's config.json (its argv is used)")
    ap.add_argument("--argv", default="", help="train_lm.py arguments (instead of --config)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--margin-gb", type=float, default=2.0)
    ap.add_argument("--extra", default="", help="extra train_lm arguments, e.g. '--vram-savers 1'")
    ap.add_argument("--workdir", default="")
    h = ap.parse_args()
    if h.config:
        cfg = json.loads(Path(h.config).read_text())
        argv = list(cfg["argv"][1:])
    else:
        argv = shlex.split(h.argv)
    import torch
    assert torch.cuda.is_available(), "vram_guard needs the GPU"
    import train_lm as tl
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
    print(json.dumps(rep, indent=1), flush=True)
    (work / "vram_guard.json").write_text(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
