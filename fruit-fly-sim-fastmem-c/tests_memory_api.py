"""Tests of the memory-module interface (memory_api.py, README_memory_api.md). CPU only, tiny configs.

    set CUDA_VISIBLE_DEVICES=-1 & python tests_memory_api.py [work_dir]
    set CUDA_VISIBLE_DEVICES=-1 & python -m pytest tests_memory_api.py -q -p no:cacheprovider

(a) the default flags reproduce fruit-fly-sim-next bit for bit: per-step train losses and the full checkpoint (model
    incl. the blob stores, optimizer, RNG, trainer counters), with the blob memory (--gm-preset 25d) and with --gm 0;
    the reference runs import fruit-fly-sim-next's own modules in a subprocess (no .pyc written there)
(b) --memory none == --gm 0 (same flags otherwise); and --memory wave --wave-mode stub (reads zeros) == --memory none
    (the wave memory itself: tests_wave.py)
(c) the memory alone saved (memory.pt), reset, loaded -> identical outputs; and a run stopped with brain-only
    checkpoints (--memory-embed 0: last.pt + last.memory.pt) resumes bit-identically
(d) the runtime memory-off switch at eval: output == the brain without a memory module, the stores untouched, the
    memory back on == before
(e) --memory-train-only experts=1,shared: after the steps every brain parameter and buffer, the global space and the
    other experts' private spaces are bit-identical; the brain's parameters are requires_grad False and out of the
    optimizer
(f) hook-based: the memory-side losses (the blob gate's loss, backpropagated in after_backward) reach no brain
    parameter and no thought (xf) tensor
(g) the byte budget of the memory state vs a configured cap

Every train_lm run is a subprocess of this file in driver mode (--driver <code dir> <out.json> <train args...>).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEXT = HERE.parent / "fruit-fly-sim-next"
N, K = 20, 12
TOK_PER_STEP = 4 * 2 * 320                                   # batch x accum x seq of TINY_ARGS
GM_25D = ("--gm 1 --gm-preset 25d --gm-probation-dtype int8 --gm-qcap -64 --chunk-ce 512 --ckpt-route 1 "
          "--gm-global 256 --gm-private 16 --gm-fit-step 5 --gm-fit-window 3 --gm-cells 8 --gm-cell-cap 64 "
          "--gm-probe 4 --gm-cand 8 --gm-topk 4,4 --gm-ivf-refit 15 --gm-rerank-top 16 --gm-probation-life 8 "
          "--gm-cand-rate 0.05 --gm-gate-warm 15 --gm-eval-every 20 --concepts 2,4 --think-w 0.1 "
          "--diag-every 5 --keep-ckpts 0").split()
GM_0 = "--concepts 2,4 --think-w 0.1 --chunk-ce 512 --ckpt-route 1 --diag-every 5 --keep-ckpts 0".split()


# ====================================================================== driver (subprocess) side
def driver(code_dir, out_json, train_args, hook=False):
    """train_lm.main of `code_dir` in this process; the per-step train losses (and, hook=True, the (f) check) to
    out_json."""
    code_dir = str(Path(code_dir).resolve())
    sys.path[:] = [code_dir] + [p for p in sys.path[1:] if Path(p).resolve() != HERE]
    import torch
    torch.set_num_threads(2)
    import train_lm as tl
    assert Path(tl.__file__).resolve().parent == Path(code_dir), (tl.__file__, code_dir)
    losses = []
    real_math = tl.math

    class _M:                                                  # math proxy: records the loss checked by isfinite
        def __getattr__(self, n):
            return getattr(real_math, n)

        def isfinite(self, x):
            f = sys._getframe(1)
            if f.f_code.co_name == "main" and "li" in f.f_locals and f.f_locals.get("li") is x:
                losses.append((int(f.f_locals["step"]), float(x)))
            return real_math.isfinite(x)

    tl.math = _M()
    hook_rec = _install_grad_hooks() if hook else None
    argv, out, err = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["train_lm.py", *train_args]
    try:
        tl.main()
    finally:
        log = sys.stdout.file if hasattr(sys.stdout, "file") else None
        sys.argv, sys.stdout, sys.stderr = argv, out, err
        if log is not None:
            log.close()
    res = dict(losses=losses, code=code_dir)
    if hook_rec is not None:
        res["hook"] = hook_rec
    Path(out_json).write_text(json.dumps(res))


def _install_grad_hooks():
    """(f): around every BlobMemory.after_backward (where the memory-side losses are backpropagated): hooks on every
    brain parameter and on every thought tensor xf the memory read in this micro-batch; snapshots of the brain's .grad.
    Records the hook firings / grad changes inside the memory-side backward, and whether the gate got a gradient."""
    import torch
    import blob_memory as bmm
    rec = dict(calls=0, gate_backward=0, brain_hook_fired=0, xf_hook_fired=0, brain_grad_changed=0, xf_tensors=0,
               brain_params=0)
    state = dict(phase="main", xfs=[])
    real_call, real_ab = bmm.BlobMemory.__call__, bmm.BlobMemory.after_backward

    def call(self, xf, *a, **k):
        if xf.requires_grad and torch.is_grad_enabled():
            def h(g):
                if state["phase"] == "memory":
                    rec["xf_hook_fired"] += 1
            xf.register_hook(h)
            rec["xf_tensors"] += 1
        return real_call(self, xf, *a, **k)

    def after_backward(self):
        model = state.get("model")
        if model is None:
            import gc
            from loop_brain import LoopBrain
            model = state["model"] = next(o for o in gc.get_objects() if isinstance(o, LoopBrain))
        brain = [(n, p) for n, p in model.named_parameters() if not n.startswith("bm.") and p.requires_grad]
        rec["brain_params"] = len(brain)
        before = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in brain}
        handles = []
        for n, p in brain:
            def h(g, _n=n):
                if state["phase"] == "memory":
                    rec["brain_hook_fired"] += 1
            handles.append(p.register_hook(h))
        gate = self.gate
        gb = None if gate is None else {n: (None if p.grad is None else p.grad.detach().clone())
                                        for n, p in gate.named_parameters()}
        state["phase"] = "memory"
        try:
            out = real_ab(self)
        finally:
            state["phase"] = "main"
            for hd in handles:
                hd.remove()
        rec["calls"] += 1
        for n, p in brain:
            a, b = before[n], p.grad
            if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                rec["brain_grad_changed"] += 1
        if gate is not None:
            for n, p in gate.named_parameters():
                a, b = gb[n], p.grad
                if b is not None and (a is None or not torch.equal(a, b)):
                    rec["gate_backward"] += 1
                    break
        return out

    bmm.BlobMemory.__call__, bmm.BlobMemory.after_backward = call, after_backward
    return rec


# ====================================================================== parent side
def _env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def run(root, name, code_dir, args, hook=False, stop=N):
    """One train_lm run in a subprocess. Returns (losses dict step -> loss, run dir, the driver record)."""
    runs, out = root / "runs", root / f"{name}.json"
    data = root / "data"
    if os.environ.get("MEMAPI_REUSE") and out.exists():      # (debugging: reuse a finished run of this work dir)
        res = json.loads(out.read_text())
        return {s: v for s, v in res["losses"]}, runs / name, res
    from test_memory import TINY_ARGS
    argv = ["--run-name", name, "--runs", str(runs), "--data", str(data), *TINY_ARGS,
            "--tokens", str((N + 1) * TOK_PER_STEP), "--stop-at", str(stop), *args]
    cmd = [sys.executable, str(HERE / "tests_memory_api.py"), "--driver", str(code_dir), str(out),
           *(["--hook"] if hook else []), "--", *argv]
    p = subprocess.run(cmd, env=_env(), capture_output=True, text=True, cwd=str(root))
    if p.returncode != 0:
        raise RuntimeError(f"run {name} failed:\n{p.stdout[-3000:]}\n{p.stderr[-6000:]}")
    res = json.loads(out.read_text())
    return {s: v for s, v in res["losses"]}, runs / name, res


def _load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def ckpt_equal(ca, cb, skip=("trainer/elapsed",)):
    """Flattened full-checkpoint comparison (test_resume_gm's _flat / _diff): [] = bit-identical."""
    from test_resume_gm import _diff, _flat
    fa, fb = dict(_flat(ca)), dict(_flat(cb))
    bad = []
    for k in sorted(set(fa) | set(fb)):
        if k in skip or k.startswith("rng/python"):
            # (Python's random module: never seeded and never drawn from by the trainer, only saved / restored; its
            # state comes from the OS entropy of each process, so it differs between ANY two runs, next vs next too)
            continue
        if k not in fa or k not in fb:
            bad.append((k, "missing in " + ("A" if k not in fa else "B")))
            continue
        ok, d = _diff(fa[k], fb[k])
        if not ok:
            bad.append((k, d))
    return bad


def losses_equal(la, lb, steps):
    miss = [s for s in steps if s not in la or s not in lb]
    diff = [s for s in steps if s in la and s in lb and la[s] != lb[s]]
    md = max((abs(la[s] - lb[s]) for s in steps if s in la and s in lb), default=0.0)
    return not miss and not diff, dict(missing=miss, differ=diff, max_abs=md)


def _build(ck):
    """A LoopBrain from a checkpoint (eval mode, CPU)."""
    import torch
    from loop_brain import LoopBrain
    torch.manual_seed(0)
    m = LoopBrain(**ck["cfg"])
    m.load_state_dict(ck["state"], strict=False)
    return m.eval()


def _batch(seed=3, B=2, T=64):
    import torch
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(1, 50, (B, T + 1), generator=g)
    return x[:, :-1], x[:, 1:]


def _out(m, x, y):
    import torch
    with torch.no_grad():
        m.K = m.cfg["k_mean"]
        h = m.features(x)
        return h.detach().clone(), float(m(x, y)[1])


def _mem_buffers(m):
    return {n: b.detach().clone() for n, b in m.bm.state_dict().items() if hasattr(b, "detach")}


def run_all(root=None):
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(HERE))
    from test_memory import _tiny_data
    import memory_api as api
    root = Path(root or tempfile.mkdtemp(prefix="memapi_")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _tiny_data(root / "data")
    R, out = [], {}

    def check(name, ok, detail=""):
        R.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    # ---------------------------------------------------------------- (a) default == fruit-fly-sim-next
    la_n, d_n, _ = run(root, "next_25d", NEXT, GM_25D)
    la_w, d_w, _ = run(root, "wave_25d", HERE, GM_25D + ["--memory-save", "1"])
    ok, info = losses_equal(la_n, la_w, range(N))
    check("(a) blob 25d: per-step train losses == fruit-fly-sim-next", ok, f"{N} steps, {info}")
    bad = ckpt_equal(_load(d_n / "last.pt"), _load(d_w / "last.pt"))
    check("(a) blob 25d: full checkpoint at step N == fruit-fly-sim-next", not bad, f"{len(bad)} differing: {bad[:4]}")
    lg_n, g_n, _ = run(root, "next_gm0", NEXT, GM_0)
    lg_w, g_w, _ = run(root, "wave_gm0", HERE, GM_0)
    ok, info = losses_equal(lg_n, lg_w, range(N))
    check("(a) --gm 0: per-step train losses == fruit-fly-sim-next", ok, f"{info}")
    bad = ckpt_equal(_load(g_n / "last.pt"), _load(g_w / "last.pt"))
    check("(a) --gm 0: full checkpoint == fruit-fly-sim-next", not bad, f"{len(bad)} differing: {bad[:4]}")

    # ---------------------------------------------------------------- (b) --memory none == --gm 0 ; wave stub == none
    ln, n_d, _ = run(root, "wave_none", HERE, GM_25D + ["--memory", "none"])
    ok, info = losses_equal(lg_w, ln, range(N))
    check("(b) --memory none (with the --gm 1 flags) == --gm 0: losses", ok, f"{info}")
    bad = ckpt_equal(_load(g_w / "last.pt"), _load(n_d / "last.pt"))
    check("(b) --memory none == --gm 0: full checkpoint", not bad, f"{len(bad)} differing: {bad[:4]}")
    lv, v_d, _ = run(root, "wave_stub", HERE, GM_0 + ["--memory", "wave", "--wave-mode", "stub"])
    ok, info = losses_equal(lg_w, lv, range(N))
    check("(b) --memory wave (stub, reads zeros) == --memory none: losses", ok, f"{info}")
    ckv = _load(v_d / "last.pt")
    bad = [b for b in ckpt_equal(_load(g_w / "last.pt"), ckv) if not b[0].startswith("cfg/")]
    check("(b) --memory wave == --memory none: checkpoint (except cfg memory=wave)", not bad,
          f"{len(bad)} differing: {bad[:4]}; cfg memory={ckv['cfg'].get('memory')}")

    # ---------------------------------------------------------------- (c) memory.pt save / reset / load
    ck = _load(d_w / "last.pt")
    m = _build(ck)
    x, y = _batch()
    h0, l0 = _out(m, x, y)
    fill = {k: float(v) for k, v in m.bm.fill().items()}
    side = api.memory_path(d_w / "last.pt")
    check("(c) last.memory.pt written next to last.pt (--memory-save 1)", side.exists(), str(side))
    m.reset_memory()
    m.eval()
    h1, l1 = _out(m, x, y)
    check("(c) reset: the memory starts fresh (output changes; empty stores)",
          not torch.equal(h0, h1) and float(m.bm.glob.occ.sum()) == 0,
          f"loss with memory {l0:.6f}, after reset {l1:.6f}; fill before {fill}")
    api.load_memory_file(side, m)
    h2, l2 = _out(m, x, y)
    check("(c) load memory.pt after the reset: outputs bit-identical", torch.equal(h0, h2) and l0 == l2,
          f"max |dh| {float((h0 - h2).abs().max()):.3e}")
    tmp = root / "m_resaved.memory.pt"
    api.save_memory_file(tmp, m, step=N)
    m2 = _build(ck)
    m2.reset_memory()
    m2.eval()
    api.load_memory_file(tmp, m2)
    h3, l3 = _out(m2, x, y)
    check("(c) save -> fresh model -> reset -> load: outputs bit-identical", torch.equal(h0, h3) and l0 == l3)
    # brain-only checkpoints: stop at K with --memory-embed 0, resume, == the uninterrupted run
    lb1, b_d, hk = run(root, "split_b", HERE, GM_25D + ["--memory-embed", "0"], hook=True, stop=K)
    ckb = _load(b_d / "last.pt")
    check("(c) --memory-embed 0: last.pt holds no memory key, last.memory.pt exists",
          not any(k.startswith("bm.") for k in ckb["state"]) and api.memory_path(b_d / "last.pt").exists())
    lb2, b2_d, _ = run(root, "split_b2", HERE, GM_25D + ["--memory-embed", "0", "--resume", str(b_d / "last.pt")])
    lb = {**lb1, **lb2}
    ok, info = losses_equal(la_w, lb, range(N))
    check("(c) stop at K (brain-only ckpt + memory.pt) and resume == uninterrupted: losses", ok, f"{info}")
    full_b = _load(b2_d / "last.pt")
    ref = _load(d_w / "last.pt")
    ref["state"] = {k: v for k, v in ref["state"].items() if not k.startswith("bm.")}
    bad = ckpt_equal(ref, full_b)
    check("(c) ... brain checkpoint at N == uninterrupted", not bad, f"{len(bad)} differing: {bad[:4]}")
    mb = _load(api.memory_path(b2_d / "last.pt"))["state"]
    mr = {k[3:]: v for k, v in _load(d_w / "last.pt")["state"].items() if k.startswith("bm.")}
    bad = ckpt_equal(mr, mb)
    check("(c) ... memory.pt at N == the uninterrupted run's memory", not bad, f"{len(bad)} differing: {bad[:4]}")

    # ---------------------------------------------------------------- (d) memory off at eval
    m = _build(ck)
    s0 = _mem_buffers(m)
    h_on, l_on = _out(m, x, y)
    with api.memory_off(m):
        h_off, l_off = _out(m, x, y)
    h_on2, l_on2 = _out(m, x, y)
    s1 = _mem_buffers(m)
    m_nomem = _build(ck)
    m_nomem.bm = None
    h_nm, l_nm = _out(m_nomem, x, y)
    check("(d) memory off == the brain without a memory module (bit-identical)", torch.equal(h_off, h_nm) and l_off == l_nm,
          f"loss on {l_on:.6f}, off {l_off:.6f}, no module {l_nm:.6f}")
    check("(d) memory off changes the output (the memory is in use)", not torch.equal(h_on, h_off))
    check("(d) back on == before; the eval forwards left every memory buffer bit-identical",
          torch.equal(h_on, h_on2) and all(torch.equal(s0[k], s1[k]) for k in s0))

    # ---------------------------------------------------------------- (e) train only experts 1 + shared
    sel = [1, -1]
    le, e_d, _ = run(root, "train_only", HERE, GM_25D + ["--resume", str(b_d / "last.pt"), "--memory-train-only",
                                                         "experts=1,shared", "--memory-eval-off", "1"])
    a_ck = _load(b_d / "last.pt")                 # (brain-only; its memory in last.memory.pt)
    a_mem = _load(api.memory_path(b_d / "last.pt"))["state"]
    e_ck = _load(e_d / "last.pt")
    brain_a = {k: v for k, v in a_ck["state"].items()}
    brain_e = {k: v for k, v in e_ck["state"].items() if not k.startswith("bm.")}
    bad = [k for k in brain_a if not torch.equal(brain_a[k], brain_e[k])]
    check("(e) every brain parameter and buffer bit-identical after the train-only steps", not bad and
          set(brain_a) == set(brain_e), f"{len(brain_a)} tensors, {N - K} steps; differing {bad[:5]}")
    mem_e = {k[3:]: v for k, v in e_ck["state"].items() if k.startswith("bm.")}
    S = mem_e["priv.value"].size(0)
    spaces = sorted({S - 1 if e < 0 else e for e in sel})
    keep = torch.zeros(S, dtype=torch.bool)
    keep[spaces] = True
    diff_other, changed = [], {}
    for k, v in mem_e.items():
        if not torch.is_tensor(v):
            continue
        a = a_mem[k]
        if k in ("priv.value", "priv.opt_v", "priv.opt_n"):
            if not torch.equal(a[~keep], v[~keep]):
                diff_other.append(k)
            changed[k] = int((a[keep] != v[keep]).reshape(len(spaces), a.size(1), -1).any(-1).sum())
        elif k == "now":
            continue   # (the step stamp: set by step_begin at every step, bookkeeping, not learning)
        elif not torch.equal(a, v):
            diff_other.append(k)
    ex_a, ex_e = a_mem.get("_extra_state", {}), mem_e.get("_extra_state", {})
    gate_same = all(torch.equal(ex_a["gate"][k], ex_e["gate"][k]) for k in ex_a.get("gate", {}))
    check("(e) the global space, the other experts' private spaces, keys / stats / probation, the gate: unchanged",
          not diff_other and gate_same, f"differing: {diff_other[:6]}; gate same {gate_same}")
    check("(e) the selected experts' memory did learn (contents of private spaces 1 + shared)",
          changed.get("priv.value", 0) > 0, f"rows changed: {changed}")
    log = (e_d / "train.log").read_text()
    check("(e) brain out of the optimizer and requires_grad False (the trainer's line)",
          "MEMORY TRAIN-ONLY: experts [1, -1]; trainable parameters none" in log)
    opt_names = set(e_ck["opt"]["state"])
    check("(e) the saved optimizer state holds no brain parameter", not opt_names, f"{sorted(opt_names)[:5]}")

    # ---------------------------------------------------------------- (f) no gradient from memory-side losses
    hr = hk["hook"]
    check("(f) memory-side backward ran (the gate's loss reached the gate)", hr["calls"] > 0 and hr["gate_backward"] > 0,
          f"{hr}")
    check("(f) ... and reached no brain parameter (hooks + .grad unchanged) and no thought xf",
          hr["brain_hook_fired"] == 0 and hr["brain_grad_changed"] == 0 and hr["xf_hook_fired"] == 0
          and hr["brain_params"] > 0 and hr["xf_tensors"] > 0, f"{hr}")

    # ---------------------------------------------------------------- (g) byte budget
    m = _build(ck)
    rep = api.byte_budget(m, cap_bytes=10e6)
    print("  " + api.budget_line(rep))
    saved = sum(v.numel() * v.element_size() for v in torch.load(side, weights_only=False)["state"].values()
                if torch.is_tensor(v))
    check("(g) the budget's persistent bytes == the tensors in memory.pt", rep["persistent"] + rep["params"] == saved,
          f"budget {rep['persistent'] + rep['params']} B, memory.pt tensors {saved} B, extra state {rep['extra_state']} B")
    small = api.byte_budget(m, cap_bytes=rep["total"] // 2)
    check("(g) the cap check: under 10 MB OK, under half its size OVER", rep["ok"] and not small["ok"],
          api.budget_line(small))
    w = _build(ckv)
    wr = api.byte_budget(w, cap_bytes=1)
    check("(g) the wave stub: 0 bytes of state, no parameters", wr["total"] == 0 and wr["ok"] and
          not list(w.bm.parameters()), api.budget_line(wr))
    return R, root


def test_memory_api(tmp_path):
    R, _ = run_all(tmp_path / "memapi")
    bad = [r for r in R if not r[1]]
    assert not bad, "\n".join(f"{n}: {d}" for n, _, d in bad)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--driver":
        a = sys.argv[2:]
        hook = "--hook" in a[:3]
        i = a.index("--")
        driver(a[0], a[1], a[i + 1:], hook=hook)
        sys.exit(0)
    R, root = run_all(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nwork dir {root}")
    print("ALL PASS" if all(r[1] for r in R) else f"{sum(not r[1] for r in R)} FAIL of {len(R)}")
