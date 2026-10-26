"""GPU guard for the world_fastmem_change_12k resume (GPU; run it ONLY when announced, never next to a run).

Resumes the run's last.pt (step 2,000) READ-ONLY into a temp run folder and runs a few steps with the run's own flags
(+ any --extra), under a per-process cap (default 16 GB, device index explicit), 2 CPU threads:
  step 2000          the deterministic step-2,000 batch at 78 experts (the batch that OOMed)
  step 2001          forced growth to the final 192 experts (Neurogenesis target = final), then the step-2,000 batch again
  step 2002          the step-2,000 batch again (192 experts)
  steps 2003..2005   worst-case routing (vram_guard.install_worst_route): every sequence's tokens to the same k experts
Prints the peak allocated / reserved per step (reset each step) and the overall peaks; exit 1 on OOM or a peak
reserved above cap - margin.

    python guard_resume_2000.py [--cap-gb 16] [--device 0] [--margin-gb 0.5] [--extra "--fastmem-scan-budget 4194304"]
"""
import argparse
import json
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUN = Path(r"C:\Eitan\Python\NewBrain\runs\world_fastmem_change_12k")
GB = 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gb", type=float, default=16.0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--margin-gb", type=float, default=0.5)
    ap.add_argument("--extra", default="")
    ap.add_argument("--workdir", default="")
    h = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), "needs the GPU"
    torch.cuda.set_device(h.device)
    total = torch.cuda.get_device_properties(h.device).total_memory
    frac = h.cap_gb * GB / total
    import train_lm as tl
    import neurogenesis
    import vram_guard
    from fastmem import FastMem
    lock = RUN.parent / "gpu.lock"
    if lock.exists():
        try:
            owner, holder = lock.read_text().split()[:2]
            if tl.lock_holder_alive(int(owner)):
                sys.exit(f"guard: the GPU is in use by run {holder} (pid {owner}); refusing")
        except ValueError:
            pass
    argv = shlex.split((RUN / "launch_args.txt").read_text(encoding="utf-8").strip(), posix=False)
    argv = vram_guard.strip(argv)
    while "--mem-fraction" in argv:
        i = argv.index("--mem-fraction")
        del argv[i:i + 2]
    work = Path(h.workdir or tempfile.mkdtemp(prefix="guard_resume_"))
    seq = 512
    run_argv = argv + ["--run-name", "guard2000", "--runs", str(work), "--tokens", "200e6",
                       "--resume", str(RUN / "last.pt"), "--stop-at", "2006", "--mem-fraction", f"{frac:.4f}",
                       "--keep-ckpts", "0", "--save-report-ckpts", "0", "--ckpt-every", "0"] + shlex.split(h.extra)
    # the step plan
    state = {"step": None}
    births0 = neurogenesis.Neurogenesis.births_step
    step0 = neurogenesis.Neurogenesis.step

    def nstep(self, step):
        state["step"] = step
        if step == 2001:                                   # force the final size now (every layer -> 192 experts)
            self.ramps()
            self.t1 = self.t0 + 1
            return births0(self, step)
        return step0(self, step)
    neurogenesis.Neurogenesis.step = nstep
    batch0 = tl.Data.batch

    def batch(self, B, split="train", gen=None):
        s = state["step"]
        if split == "train" and s is not None and 2001 <= s <= 2002:
            self.ctr = 2000 * 2 + self.ctr % 2             # (--accum 2: the step-2,000 micro-batches again)
        if split == "train" and s is not None and s >= 2003:
            vram_guard.install_worst_route(seq)
        return batch0(self, B, split, gen)
    tl.Data.batch = batch
    peaks = []
    after0 = FastMem.after_step

    def after_step(self, step, *a, **k):
        r = after0(self, step, *a, **k)
        pa, pr = torch.cuda.max_memory_allocated(h.device) / GB, torch.cuda.max_memory_reserved(h.device) / GB
        n_exp = getattr(self, "n_live", None)
        peaks.append(dict(step=state["step"], experts=n_exp, peak_alloc_gb=round(pa, 3), peak_reserved_gb=round(pr, 3)))
        print(f"guard2000: step {state['step']} experts {n_exp} peak alloc {pa:.2f} GB reserved {pr:.2f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats(h.device)
        return r
    FastMem.after_step = after_step
    # (the step counter for steps without a neuro call: the train loop's own `step` is not exposed; the first step)
    state["step"] = 2000
    sys_argv, sys.argv = sys.argv, ["train_lm.py", *run_argv]
    print("guard2000: train_lm " + " ".join(run_argv), flush=True)
    t0, err = time.time(), None
    try:
        tl.main()
    except torch.cuda.OutOfMemoryError as e:
        err = f"OOM at step {state['step']}: {str(e).splitlines()[0]}"
    finally:
        sys.argv = sys_argv
        vram_guard.install_worst_route(seq, on=False)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    worst_a = max([p["peak_alloc_gb"] for p in peaks] or [0])
    worst_r = max([p["peak_reserved_gb"] for p in peaks] or [0])
    ok = err is None and worst_r <= h.cap_gb - h.margin_gb
    rep = dict(per_step=peaks, peak_alloc_gb=worst_a, peak_reserved_gb=worst_r, cap_gb=h.cap_gb, error=err,
               minutes=(time.time() - t0) / 60, workdir=str(work), verdict="OK" if ok else "REFUSE")
    print(json.dumps(rep, indent=1), flush=True)
    (work / "guard2000.json").write_text(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
