"""The --vram-* savers (activation checkpointing) are same-maths and state-compatible (CPU only, tiny sizes, the live
run's switches incl. the blob memory preset 25d).

  1. identity: a run with every saver on (--vram-savers 1, re-rank in 3-row blocks) == the plain run, bit for bit:
     every per-step train loss and the full checkpoint at step N (model incl. the blob stores / gate / generators,
     BrainOpt state, neurogenesis, trainer counters, every RNG) and the blob memory's derived tables.
  2. resume: a checkpoint written WITHOUT the savers, resumed WITH them == the plain uninterrupted run.
  3. each saver alone (one step pair) == plain.
  4. gm 0 (no blob memory): savers on == plain.

    set CUDA_VISIBLE_DEVICES=-1 & python fruit-fly-sim/test_vram_savers.py [tmp_dir]
    set CUDA_VISIBLE_DEVICES=-1 & python -m pytest fruit-fly-sim/test_vram_savers.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import shutil
import sys
from pathlib import Path

import torch

import test_resume_gm as R
from test_memory import _tiny_data

torch.set_num_threads(2)
_derived0 = R._derived
R._derived = lambda m: {} if getattr(m, "bm", None) is None else _derived0(m)   # (gm 0: no blob memory)

ALL = ["--vram-savers", "1", "--vram-rerank-rows", "3"]
PLAIN = ["--vram-readout-lean", "0"]   # (lean is on by default in this folder: the plain path has it off)
SINGLE = (["--vram-ckpt-norm", "1"], ["--vram-ckpt-mixer", "1"], ["--vram-ckpt-experts", "1"],
          ["--vram-ckpt-concept", "1"], ["--vram-ckpt-gmdist", "1"], ["--vram-rerank-rows", "2"],
          ["--vram-readout-lean", "1"])


def _bad(rows):
    # (host/bm.rerank_rows is the saver's own setting, not a result)
    return [f"{n}: {d}" for n, ok, d in rows if not ok and n != "host/bm.rerank_rows"]


def _cmp_runs(ra, rb):
    """compare() of test_resume_gm, but every step's loss (no cut)."""
    rows = R.compare(ra, rb)
    la, lb = dict(ra[0]), dict(rb[0])
    rows.append(("all train losses", la == lb, max((abs(la[s] - lb.get(s, float("nan"))) for s in la), default=0.0)))
    return rows


def _gm0(extra):
    """GM_ARGS without the blob memory (gm 0) for the identity of the non-gm savers."""
    keep = R.GM_ARGS
    R.GM_ARGS = ["--chunk-ce", "512", "--ckpt-route", "1", "--concepts", "2,4", "--think-w", "0.1",
                 "--nan-abort", "1", "--diag-every", "5", "--keep-ckpts", "0"]
    try:
        return extra()
    finally:
        R.GM_ARGS = keep


def run_all(root, singles=True):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    import train_lm as tl
    data = _tiny_data(root / "data")
    out = {}
    ra = R._run(tl, root / "a", "a", data, ["--stop-at", str(R.N), *PLAIN])
    rs = R._run(tl, root / "s", "s", data, ["--stop-at", str(R.N), *ALL])
    out["1 identity (all savers vs plain)"] = _cmp_runs(ra, rs)
    lb1 = R._run(tl, root / "b", "b", data, ["--stop-at", str(R.K), *PLAIN])
    last = root / "b" / "b" / "last.pt"
    lb2 = R._run(tl, root / "b", "b", data, ["--stop-at", str(R.N), "--resume", str(last), *ALL])
    out["2 resume (plain ckpt -> savers on)"] = R.compare(ra, (lb1[0] + lb2[0], lb2[1]))
    if singles:
        for i, fl in enumerate(SINGLE):
            r1 = R._run(tl, root / f"x{i}", "x", data, ["--stop-at", str(R.N), *PLAIN, *fl])
            out[f"3 {' '.join(fl)} alone"] = _cmp_runs(ra, r1)
    g0 = _gm0(lambda: (R._run(tl, root / "g0a", "a", data, ["--stop-at", str(R.N), *PLAIN]),
                       R._run(tl, root / "g0s", "s", data, ["--stop-at", str(R.N), *ALL])))
    out["4 gm 0 (all savers vs plain)"] = _cmp_runs(*g0)
    return out


LAUNCH_CODE = Path(__file__).resolve().parent.parent / "fruit-fly-sim-orig-reconstructed"


def flags_off_vs_launch(tmp, extra=("--vram-readout-lean", "0")):
    """5. this code with every --vram-* flag off == the code the live run was launched with (its 4 changed files,
    fruit-fly-sim-orig-reconstructed; the other modules are shared), each in its own process (test_vram_flags_off)."""
    import subprocess
    here = Path(__file__).resolve().parent
    r = subprocess.run([sys.executable, str(here / "test_vram_flags_off.py"), str(LAUNCH_CODE), str(tmp), "--", *extra],
                       capture_output=True, text=True, cwd=str(here))
    print(r.stdout[-2000:], r.stderr[-2000:] if r.returncode else "")
    return r.returncode == 0


def test_flags_off_equals_launch_code(tmp_path):
    assert flags_off_vs_launch(tmp_path / "fo")
    assert flags_off_vs_launch(tmp_path / "fo_default", ())   # (this folder's defaults: readout-lean on)


def test_vram_savers_same_maths(tmp_path):
    res = run_all(tmp_path / "vs", singles=False)
    bad = {k: _bad(v) for k, v in res.items() if _bad(v)}
    assert not bad, "\n".join(f"{k}:\n  " + "\n  ".join(v) for k, v in bad.items())


if __name__ == "__main__":
    tmp0 = Path(sys.argv[1] if len(sys.argv) > 1 else Path.cwd() / "_vram_savers_tmp")
    fo = flags_off_vs_launch(tmp0.parent / (tmp0.name + "_flags_off"))
    print(f"{'PASS' if fo else 'FAIL'}  5 flags off == launch code ({LAUNCH_CODE})")
    fo2 = flags_off_vs_launch(tmp0.parent / (tmp0.name + "_defaults"), ())
    print(f"{'PASS' if fo2 else 'FAIL'}  5b this folder's defaults (readout-lean on) == launch code")
    fo = fo and fo2
    res = run_all(tmp0)
    n_bad = 0
    for k, rows in res.items():
        b = _bad(rows)
        n_bad += len(b)
        print(f"{'PASS' if not b else 'FAIL'}  {k}  ({len(rows)} items compared)")
        for line in b[:12]:
            print("      ", line)
    n_bad += 0 if fo else 1
    print("ALL PASS" if not n_bad else f"{n_bad} FAIL")
