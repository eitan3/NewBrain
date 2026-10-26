"""The recall confidence z of the wave memory (the null control, spec PART 2) on the W-bench: is a confident read a
right one? CPU, 2 threads.

    python -B tests_wave_conf.py [--null-bins fill|load] [--holdout 5] [--kw wave_map_beta=0.25]

The default module (the v2 column store, place balance A, 4 read entries; gate bypassed) writes the facts of 4 in 5
fact labels (label % --holdout != 0; the rest are NO-MATCH facts: never written) through bench_write with the step
end once per 32 writes; then the null control observes 400 x 256 null cues (own entry place, another thought's keys)
on the capture's fit thoughts (as _null_update does on each training micro-batch). Every held / cloze cue is read
(the deployed read: winner entry by chibar, z = (chibar - mu_b) / sd_b). Reports per z bin (< 0, 0-1, 1-2, 2-4, 4-6, >= 6)
the cues of written facts (n, precision = top-1 right) and the no-match cues (n), and the threshold trade-off: for
z0, recall = P(right and z >= z0) over the written-fact cues, confident-false = P(wrong and z >= z0) over all cues
(written + no-match), and over the no-match cues alone.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
MEM26 = Path(r"C:\Eitan\Python\NewBrain\speed_wip\mem26")
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(MEM26))
import wbench as W  # noqa: E402
import wbench_module as WM  # noqa: E402
import wave_bench as WB  # noqa: E402
from wave_memory import WAVE_DEFAULTS  # noqa: E402

BINS = ((-np.inf, 0), (0, 1), (1, 2), (2, 4), (4, 6), (6, np.inf))
Z0 = (-1, 0, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-bins", default="fill", choices=["fill", "load"])
    ap.add_argument("--holdout", type=int, default=5, help="labels with label %% holdout == 0 are never written")
    ap.add_argument("--kw", default="")
    a = ap.parse_args()
    k = 3
    b = W.Bench(passes=(k,))
    h = WM.Harness(b)
    t0 = time.time()
    kw = dict(wave_priv_levels=0, wave_null_bins=a.null_bins)
    for p in filter(None, a.kw.split(",")):
        key, v = p.split("=")
        kw[key] = type(WAVE_DEFAULTS[key])(v)
    m = WB.factory(b, int(h.cue_e.max()) + 1, **kw)()
    keep = (b.store_lab % a.holdout) != 0
    xs = torch.from_numpy(b.feats("store", k, ["xf"]))[torch.from_numpy(keep)]
    lab = b.store_lab[keep]
    tg = torch.from_numpy(h.content[[h.lab_pos[int(l)] for l in lab]])
    n_w = 0
    for i in range(0, len(xs), 4096):
        m.bench_write(xs[i:i + 4096], tg[i:i + 4096], call=k)
        n_w += len(xs[i:i + 4096])
        while n_w >= 32:
            m.after_step(0)
            n_w -= 32
    fit = torch.from_numpy(b.feats("fit", k, ["xf"]))
    g = torch.Generator().manual_seed(0)
    for _ in range(400):
        m.bench_null(fit[torch.randint(0, len(fit), (2048,), generator=g)])
    written = set(int(l) for l in lab)
    print(f"[{time.time() - t0:.0f}s] wrote {len(xs)} sentences of {len(written)} facts; null bins {a.null_bins}: "
          f"mu {np.round(m.null_mu.numpy(), 3).tolist()} sd {np.round(m.null_sd.numpy(), 3).tolist()} "
          f"n {m.null_n.tolist()}", flush=True)
    for q, idx, qlab in (("held", b.held_idx, b.held_lab), ("cloze", None, b.cz_lab)):
        X = torch.from_numpy(b.feats(q, k, ["xf"]))
        with torch.no_grad():
            rec = m._recall_of(X)
        z = rec["z"].numpy()
        r = h._read_vecs(m, b.feats(q, k, ["xf"]), idx, k)
        rn = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)
        S = rn @ (h.content / np.linalg.norm(h.content, axis=1, keepdims=True)).T
        want = np.array([h.lab_pos[int(l)] for l in qlab])
        right = S.argmax(1) == want
        match = np.array([int(l) in written for l in qlab])
        rows = []
        for lo, hi in BINS:
            s_ = (z >= lo) & (z < hi)
            nm, nn = int((s_ & match).sum()), int((s_ & ~match).sum())
            prec = float(right[s_ & match].mean()) if nm else float("nan")
            rows.append(f"z [{lo:g}, {hi:g}): written {nm} (precision {prec:.3f}), no-match {nn}")
        print(f"{q}: {int(match.sum())} written-fact cues (top-1 {right[match].mean():.3f}), {int((~match).sum())} "
              f"no-match cues | " + "; ".join(rows), flush=True)
        tr = []
        for z0 in Z0:
            on = z >= z0
            rec_ = float((right & on & match).sum() / max(1, match.sum()))
            cf = float(((~right | ~match) & on).mean())
            cfn = float((on & ~match).sum() / max(1, (~match).sum()))
            tr.append(f"z0 {z0}: recall {rec_:.3f}, confident-false {cf:.3f} (no-match {cfn:.3f})")
        print(f"{q} trade-off: " + "; ".join(tr), flush=True)


if __name__ == "__main__":
    main()
