"""Resume identity of the blob memory with the live run's gm flags (CPU only, tiny sizes).

Run A: uninterrupted to step N (--stop-at N on an N+1-step schedule, so the full resumable state at N is saved).
Run B: --stop-at K (past the fit), then --resume last.pt --stop-at N.
Compared at N: every per-step train loss, the full checkpoint (model state incl. the blob stores, probation, gate,
generators; BrainOpt state; neurogenesis; trainer counters; every RNG) and the live objects' derived (non-persistent)
tables and host attributes of the blob memory.

    set CUDA_VISIBLE_DEVICES=-1 & python fruit-fly-sim/test_resume_gm.py [runs_dir]
    set CUDA_VISIBLE_DEVICES=-1 & python -m pytest fruit-fly-sim/test_resume_gm.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import copy
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from test_memory import TINY_ARGS, _tiny_data

torch.set_num_threads(2)

N, K, FIT = 20, 12, 5
TOK_PER_STEP = 4 * 2 * 320                                   # batch x accum x seq of TINY_ARGS
GM_ARGS = ("--gm 1 --gm-preset 25d --gm-probation-dtype int8 --gm-qcap -64 --chunk-ce 512 --ckpt-route 1 "
           "--gm-global 256 --gm-private 16 --gm-fit-step 5 --gm-fit-window 3 --gm-cells 8 --gm-cell-cap 64 "
           "--gm-probe 4 --gm-cand 8 --gm-topk 4,4 --gm-ivf-refit 15 --gm-rerank-top 16 --gm-probation-life 8 "
           "--gm-cand-rate 0.05 --gm-gate-warm 15 --gm-eval-every 20 --concepts 2,4 --think-w 0.1 "
           "--nan-abort 1 --diag-every 5 --save-report-ckpts 1 --keep-ckpts 0").split()


def _run(tl, runs, name, data, extra):
    """train_lm.main in-process; returns the per-step losses and (model, opt, ckpt) captured at the final --stop-at
    save."""
    losses, cap = [], {}
    real_math, real_save = tl.math, tl.save_checkpoint

    class _M:                                                  # math proxy: records the loss checked by isfinite
        def __getattr__(self, n):
            return getattr(real_math, n)

        def isfinite(self, x):
            f = sys._getframe(1)
            if f.f_code.co_name == "main" and "li" in f.f_locals and f.f_locals.get("li") is x:
                losses.append((int(f.f_locals["step"]), float(x)))
            return real_math.isfinite(x)

    def save(path, obj):
        real_save(path, obj)
        f = sys._getframe(1)
        cap.update(ckpt=copy.deepcopy(obj), step=obj["step"], model=f.f_locals["model"], opt=f.f_locals["opt"])

    tl.math, tl.save_checkpoint = _M(), save
    argv, out, err = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["train_lm.py", "--run-name", name, "--runs", str(runs), "--data", str(data), *TINY_ARGS,
                "--tokens", str((N + 1) * TOK_PER_STEP), *GM_ARGS, *extra]
    try:
        tl.main()
    finally:
        log = sys.stdout.file if hasattr(sys.stdout, "file") else None
        sys.argv, sys.stdout, sys.stderr = argv, out, err
        if log is not None:
            log.close()
        tl.math, tl.save_checkpoint = real_math, real_save
    return losses, cap


# ---------------------------------------------------------------- comparison
def _flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, f"{p}/{k}" if p else str(k))
    elif isinstance(x, (list, tuple)) and any(torch.is_tensor(v) or isinstance(v, (dict, list, tuple, np.ndarray))
                                              for v in x):
        for i, v in enumerate(x):
            yield from _flat(v, f"{p}[{i}]")
    else:
        yield p, x


def _diff(a, b):
    """(equal, max abs diff or note)"""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return False, "type"
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, f"shape/dtype {tuple(a.shape)}{a.dtype} vs {tuple(b.shape)}{b.dtype}"
        if a.numel() == 0:
            return True, 0.0
        if a.dtype == torch.bool:
            a, b = a.int(), b.int()
        if not (a.is_floating_point() or a.is_complex()):
            d = (a.long() - b.long()).abs().max().item()
            return d == 0, float(d)
        a, b = a.double(), b.double()
        same = (a == b) | (torch.isnan(a) & torch.isnan(b))
        if bool(same.all()):
            return True, 0.0
        fin = torch.isfinite(a) & torch.isfinite(b)
        d = (a - b)[fin & ~same].abs().max().item() if bool((fin & ~same).any()) else float("inf")
        return False, d
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        ok = isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b)
        return ok, 0.0 if ok else "ndarray differs"
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return True, 0.0
        return False, abs(a - b)
    ok = repr(a) == repr(b)
    return ok, 0.0 if ok else f"{repr(a)[:60]} vs {repr(b)[:60]}"


def _component(path):
    """Group a flattened checkpoint path into a reported component."""
    if path.startswith("state/"):
        k = path[6:]
        if ".bm." not in "." + k and not k.startswith("bm."):
            return "model weights/buffers (non-gm)"
        k = k[k.index("bm.") + 3:]
        if k.startswith("glob."):
            return "gm global store: " + k[5:]
        if k.startswith("priv."):
            return "gm private store: " + k[5:]
        if k.startswith("prob."):
            return "gm probation: " + k[5:]
        if k.startswith("_extra_state"):
            r = k[len("_extra_state/"):]
            top = r.split("/")[0]
            if top == "gate":
                return "gm gate (extra): " + r.split("/")[1]
            if top == "gate_opt":
                return "gm gate optimizer state"
            if top == "gens":
                return "gm generators: " + r.split("/")[1]
            return "gm extra: " + top
        if k.startswith("gate."):
            return "gm gate buffer/param: " + k[5:]
        return "gm buffer: " + k
    top = path.split("/")[0]
    if top == "opt":
        return "BrainOpt state (t + per-param)"
    if top == "rng":
        return "RNG: " + path.split("/")[1]
    return "ckpt: " + top


def _derived(m):
    """The blob memory's non-persistent tensors and plain host attributes (not in the checkpoint)."""
    bm, out = m.bm, {}
    for mod_name, mod in bm.named_modules():
        for n, b in mod.named_buffers(recurse=False):
            if n in mod._non_persistent_buffers_set:
                out[f"nonpersistent/{mod_name or 'bm'}.{n}"] = b.detach().clone()
    for n, v in vars(bm).items():
        if n.startswith("_") and n not in ("_salt_rng", "_fitted", "_gen_states"):
            continue
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[f"host/bm.{n}"] = v
    out["host/bm._fitted"] = getattr(bm, "_fitted", None)
    if getattr(bm, "gate", None) is not None:
        for n, v in vars(bm.gate).items():
            if not n.startswith("_") and (isinstance(v, (int, float, bool, str)) or v is None):
                out[f"host/gate.{n}"] = v
    return out


def compare(ra, rb):
    (la, ca), (lb, cb) = ra, rb
    rows = []
    # per-step losses K..N-1 (the train steps after the cut; train step s is logged with step = s)
    sa, sb = dict(la), dict(lb)
    steps = [s for s in range(K, N)]
    miss = [s for s in steps if s not in sa or s not in sb]
    ld = max((abs(sa[s] - sb[s]) for s in steps if s in sa and s in sb), default=0.0)
    ok = not miss and all(sa[s] == sb[s] for s in steps)
    rows.append((f"train loss steps {K + 1}..{N} ({len(steps)} lines)", ok, ld if not miss else f"missing {miss}"))
    pre = [s for s in range(0, K) if sa.get(s) != sb.get(s)]
    rows.append((f"train loss steps 1..{K} (before the cut)", not pre, 0.0 if not pre else f"differ at {pre}"))
    groups = {}
    fa, fb = dict(_flat(ca["ckpt"])), dict(_flat(cb["ckpt"]))
    fa.pop("trainer/elapsed", None), fb.pop("trainer/elapsed", None)
    for k in sorted(set(fa) | set(fb)):
        c = _component(k)
        if k not in fa or k not in fb:
            ok, d = False, f"missing in {'A' if k not in fa else 'B'}: {k}"
        else:
            ok, d = _diff(fa[k], fb[k])
            if not ok:
                d = f"{k}: {d}"
        g = groups.setdefault(c, [True, 0.0, 0])
        g[2] += 1
        if not ok:
            g[0] = False
            g[1] = d if not isinstance(g[1], str) or True else g[1]
    for c, (ok, d, n) in groups.items():
        rows.append((f"{c} [{n}]", ok, d))
    da, db = _derived(ca["model"]), _derived(cb["model"])
    for k in sorted(set(da) | set(db)):
        if k not in da or k not in db:
            rows.append((k, False, "missing"))
            continue
        ok, d = _diff(da[k], db[k])
        rows.append((k, ok, d))
    return rows


def run_all(root):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    import train_lm as tl
    data = _tiny_data(root / "data")
    ra = _run(tl, root / "runs_a", "a", data, ["--stop-at", str(N)])
    assert ra[1]["step"] == N
    lb1 = _run(tl, root / "runs_b", "b", data, ["--stop-at", str(K)])
    assert lb1[1]["step"] == K
    last = root / "runs_b" / "b" / "last.pt"
    lb2 = _run(tl, root / "runs_b", "b", data, ["--stop-at", str(N), "--resume", str(last)])
    assert lb2[1]["step"] == N
    rb = (lb1[0] + lb2[0], lb2[1])
    logs = {n: (root / f"runs_{n}" / n / "train.log").read_text() for n in "ab"}
    assert "GM ADDRESS FITTED" in logs["a"] and f"RESUMED at step {K}" in logs["b"]
    return compare(ra, rb), ra[0], rb[0]


def test_resume_gm_identity(tmp_path):
    rows, _, _ = run_all(tmp_path / "rt")
    bad = [r for r in rows if not r[1]]
    assert not bad, "\n".join(f"{n}: {d}" for n, _, d in bad)


if __name__ == "__main__":
    rows, la, lb = run_all(sys.argv[1] if len(sys.argv) > 1 else Path.cwd() / "_resume_gm_tmp")
    print("\nlosses A:", [(s + 1, round(v, 6)) for s, v in la])
    print("losses B:", [(s + 1, round(v, 6)) for s, v in lb])
    for n, ok, d in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {n}  max|diff| {d}")
    print("ALL PASS" if all(r[1] for r in rows) else f"{sum(not r[1] for r in rows)} FAIL")
