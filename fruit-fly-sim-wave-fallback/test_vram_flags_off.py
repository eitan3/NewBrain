"""Flags-off identity across code versions: the same tiny run (test_resume_gm's live-run switches incl. the blob memory
preset 25d) with an older code folder (e.g. the code the live run was launched with) and with this folder (every
--vram-* flag at its default = off), each in its own process; every per-step loss and the full checkpoint at step N
must be bit-identical.

    set CUDA_VISIBLE_DEVICES=-1 & python test_vram_flags_off.py <old_code_dir> [tmp_dir] [-- extra train_lm args]
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def child(code_dir, root, out, extra):
    """(in a subprocess) test_resume_gm._run of run A with code_dir's modules; torch.save of (losses, ckpt)."""
    sys.path.insert(0, str(code_dir))
    sys.path.insert(1, str(HERE))          # (the test helpers, if the old folder lacks them)
    import random
    random.seed(0)   # (train_lm never seeds Python's `random`: a fresh process would differ there regardless of code)
    import torch
    import test_resume_gm as R
    from test_memory import _tiny_data
    import train_lm as tl
    assert Path(tl.__file__).resolve().parent == Path(code_dir).resolve(), tl.__file__
    data = _tiny_data(Path(root) / "data")
    losses, cap = R._run(tl, Path(root) / "runs", "a", data, ["--stop-at", str(R.N), *extra])
    torch.save(dict(losses=losses, ckpt=cap["ckpt"]), out)


def main():
    old = Path(sys.argv[1]).resolve()
    rest = sys.argv[2:]
    extra = rest[rest.index("--") + 1:] if "--" in rest else []
    rest = rest[:rest.index("--")] if "--" in rest else rest
    tmp = Path(rest[0] if rest else HERE / "_flags_off_tmp").resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    res = {}
    for tag, code in (("old", old), ("new", HERE)):
        out = tmp / f"{tag}.pt"
        ex = extra if tag == "new" else []
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(code), str(tmp / tag), str(out),
                            *ex], cwd=str(tmp), env=env, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit(f"{tag} run failed")
        res[tag] = out
    import torch
    sys.path.insert(0, str(HERE))
    import test_resume_gm as R
    a, b = (torch.load(res[t], weights_only=False) for t in ("old", "new"))
    la, lb = dict(a["losses"]), dict(b["losses"])
    bad = []
    if la != lb:
        bad.append(f"losses differ: {[(s, la.get(s), lb.get(s)) for s in sorted(set(la) | set(lb)) if la.get(s) != lb.get(s)][:5]}")
    fa, fb = dict(R._flat(a["ckpt"])), dict(R._flat(b["ckpt"]))
    for k in ("trainer/elapsed",):
        fa.pop(k, None), fb.pop(k, None)
    n = 0
    for k in sorted(set(fa) | set(fb)):
        n += 1
        if k not in fa or k not in fb:
            bad.append(f"missing {k}")
            continue
        ok, d = R._diff(fa[k], fb[k])
        if not ok:
            bad.append(f"{k}: {d}")
    print(f"compared {len(la)} step losses and {n} checkpoint entries (old: {old}, new: {HERE}, new extra args {extra})")
    print("ALL PASS" if not bad else f"{len(bad)} FAIL\n  " + "\n  ".join(bad[:20]))
    return not bad


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        sys.exit(0 if main() else 1)
