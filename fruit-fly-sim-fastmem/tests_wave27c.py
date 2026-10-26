"""Run-critical CPU tests of the wave27c patch (the pause of world_wave27b_small_12k): the sync-free write / confirm /
sweep path, the gathered decode, the learned cut-off c = sigmoid(a_e (z - b_e)) (conf_a / conf_b) and its resume.

    set CUDA_VISIBLE_DEVICES=-1 & python -B tests_wave27c.py --old <dir of the run's code snapshot> [--ckpt last.pt]

  (A) a fixed tiny run (tests_wave_v21's tiny preset, 14 steps: 6 before write_start, 8 with reads / writes / confirms /
      sweeps) in the OLD code (a subprocess in --old) and the NEW code with the cut-off frozen at init: the thought
      gradients (the reads' effect on the loss), the memory's state and the counters; bit-identical with the dense
      decode, the gathered decode within float rounding; the new code with the cut-off TRAINED: its first step identical
  (B) resume: the OLD module's state dict (no conf_a / conf_b) loads strictly into the NEW module, conf_a / conf_b at
      2 / 3, the memory's AdamW moments kept; --ckpt: the run's own last.pt through train_lm.load_brain_state
  (C) gradients reach conf_a / conf_b (global entry and private rows); z detached
  (D) the sync guard (a TorchDispatchMode: _local_scalar_dense, nonzero, unique, masked_select, bool-mask index) over
      after_backward + after_step: the write / confirm / sweep path is sync-free
"""
import argparse
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
STEPS = 14


def tiny_run(out, frozen_conf, dense):
    """(runs in old or new code: sys.path[0] is that code's folder) -> torch.save of the run's trace."""
    import torch
    torch.set_num_threads(2)
    import tests_wave as T
    import tests_wave_v21 as V
    import wave_columns
    if dense and hasattr(wave_columns, "DECODE_GATHER"):
        wave_columns.DECODE_GATHER = False
    if dense == 2:                                     # (the reference path: topk, the per-place sweep)
        import wave_v21
        wave_columns.TOP2_MAX = False
        wave_v21.SWEEP_BATCHED = False
    m = V.tiny(wave_read_unit="full")              # (the read unit as run; old code: the key is ignored)
    if frozen_conf and hasattr(m, "conf_parameters"):
        m.conf_parameters = lambda: []                 # (the cut-off stays at its init: compare the rest exactly)
    trace = dict(grads=[], c=[])
    for s in range(STEPS):
        xs = [T._clusters(V.B * V.TT, 32, 100 + 10 * s + j) for j in range(6)]
        leaves = T._micro(m, xs, V.B, V.TT)
        trace["grads"].append(torch.stack([x.grad for x in leaves]))
        rec = m._cache
        if rec is not None and "c_g" in rec:
            trace["c"].append(torch.cat([rec["c_g"], rec["c_p"]]))
    st = m.store
    trace["state"] = {k: v.clone() for k, v in m.state_dict().items() if torch.is_tensor(v)}
    trace["store"] = dict(key=st.key_syn.clone(), chain=st.chain_bits.clone(), load=st.load.clone())
    trace["cnt"] = {k: int(v) for k, v in m.cnt.items()}
    trace["stats"] = {k: int(v) for k, v in st.stats_acc.items()}
    trace["sd_old"] = m.state_dict()
    torch.save(trace, out)


def sub(code_dir, out, frozen, dense):
    cmd = [sys.executable, "-B", "-c",
           f"import sys; sys.path.insert(0, {str(code_dir)!r}); sys.path.insert(1, {str(HERE)!r}); "
           f"import tests_wave27c as X; X.tiny_run({str(out)!r}, {frozen}, {dense})"]
    r = subprocess.run(cmd, cwd=str(code_dir), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-4000:])


def run(check, old, ckpt=""):
    import torch
    torch.set_num_threads(2)
    tmp = Path(tempfile.mkdtemp(prefix="w27c_"))
    sub(Path(old), tmp / "old.pt", False, True)
    sub(HERE, tmp / "new_dense.pt", True, 2)
    sub(HERE, tmp / "new_top2.pt", True, True)
    sub(HERE, tmp / "new_gather.pt", True, False)
    sub(HERE, tmp / "new_train.pt", False, False)
    A, Bd, Bg, Bt = (torch.load(tmp / f, weights_only=False) for f in ("old.pt", "new_dense.pt", "new_gather.pt",
                                                                        "new_train.pt"))

    def same(a, b):                                    # (the store's bytes and the counters exactly; the float state
        sd = max(float((v.float() - b["state"][k].float()).abs().max() / v.float().abs().max().clamp(min=1e-30))
                 for k, v in a["state"].items() if v.is_floating_point() and v.numel())
        ok_ = (all(torch.equal(a["store"][k], b["store"][k]) for k in a["store"]) and a["cnt"] == b["cnt"]
               and all(torch.equal(v, b["state"][k]) for k, v in a["state"].items() if not v.is_floating_point())
               and sd < 1e-5 and gdiff(a, b) < 1e-5)
        return ok_, sd

    def gdiff(a, b):
        return max(float((x - y).abs().max() / y.abs().max().clamp(min=1e-30)) for x, y in zip(a["grads"], b["grads"]))
    nw = A["cnt"]["writes_g"] + A["cnt"]["writes_p"]
    ok1, sd1 = same(A, Bd)
    check("(A1) tiny run, old code vs new (dense decode, cut-off frozen at 2 / 3, read unit 'full'): the store's key / "
          "chain bytes, the loads, every integer state and the counters bit-identical; the thought grads and the float "
          "state within float rounding (the masked means' summation order)", ok1 and nw > 0,
          f"{STEPS} steps; writes g/p {A['cnt']['writes_g']}/{A['cnt']['writes_p']}, confirms "
          f"{A['cnt']['confirms_g']}/{A['cnt']['confirms_p']}, erased {A['stats'].get('erased', 0)}; "
          f"max rel grad diff {gdiff(A, Bd):.2e}, float state {sd1:.2e}")
    kd = sum(int((Bg['store'][k] != A['store'][k]).sum()) for k in ("key", "chain"))
    Bm = torch.load(tmp / "new_top2.pt", weights_only=False)
    km = sum(int((Bm['store'][k] != Bd['store'][k]).sum()) for k in ("key", "chain"))
    check("(A1b) top-2 by two max reductions (+ the batched sweep) vs topk: same maths, ties broken low-index-first "
          "(topk's tie order is unspecified): report only", True,
          f"max rel grad diff {gdiff(Bm, Bd):.2e}; store bytes differing {km}; cnt equal {Bm['cnt'] == Bd['cnt']}")
    kd = sum(int((Bg['store'][k] != Bm['store'][k]).sum()) for k in ("key", "chain"))
    check("(A2) the gathered decode (bmm over the top-2 rows) vs the dense einsum: thought grads within float rounding",
          gdiff(Bm, Bg) < 1e-4, f"max rel grad diff {gdiff(Bm, Bg):.2e}; store bytes differing {kd}; "
          f"cnt equal {Bm['cnt'] == Bg['cnt']}")
    i0 = next((i for i, c in enumerate(A["c"]) if c.numel()), None)
    first = i0 is not None and torch.equal(Bg["c"][i0], Bt["c"][i0]) and \
        all(torch.equal(x, y) for x, y in zip(Bg["grads"][:i0 + 1], Bt["grads"][:i0 + 1]))
    ca, cb = Bt["state"]["conf_a"], Bt["state"]["conf_b"]
    check("(A3) cut-off trained vs frozen (new code): identical at init (the reads' c and thought grads up to the first active step); then "
          "conf_a / conf_b move", first and not torch.all(ca == 2.0),
          f"first active step {i0}; conf_a [{float(ca.min()):.5f}, {float(ca.max()):.5f}], conf_b "
          f"[{float(cb.min()):.5f}, {float(cb.max()):.5f}]")

    # (B) resume: the old module's state dict into the new module, strict
    sys.path.insert(0, str(HERE))
    import tests_wave as T
    import tests_wave_v21 as V
    sd_old = A["sd_old"]
    m = V.tiny(wave_read_unit="full")
    miss = [k for k in m.state_dict() if k not in sd_old]
    r = m.load_state_dict(sd_old, strict=True)
    ok_vals = all(torch.equal(m.state_dict()[k], v) for k, v in sd_old.items() if torch.is_tensor(v))
    m.train()
    T._micro(m, [T._clusters(V.B * V.TT, 32, 999 + j) for j in range(6)], V.B, V.TT)
    o = m._opt
    st_g = o.state.get(m.g_read, {})
    old_opt = sd_old["_extra_state"]["opt"]
    kept = old_opt is not None and len(old_opt["state"]) > 0 and "exp_avg" in st_g and int(st_g["step"]) > 1
    check("(B1) resume: the old checkpoint (no conf_a / conf_b) loads strictly into the new module; conf_a / conf_b = 2 "
          "/ 3; every other tensor as saved; the memory's AdamW moments kept (not reset)",
          miss == ["conf_a", "conf_b"] and not r.missing_keys and ok_vals and kept
          and torch.all(m.conf_a.detach()[:m.n_live] != 0),
          f"missing in the old state {miss}; g_read AdamW step after one more step {int(st_g.get('step', 0))}")
    if ckpt:
        import types
        import train_lm
        from loop_brain import LoopBrain
        rs = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = dict(rs["cfg"])
        model = LoopBrain(**cfg)
        has = any(k.endswith("bm.conf_a") for k in rs["state"])
        train_lm.load_brain_state(model, rs["state"], types.SimpleNamespace(resume=ckpt, memory_load=""))
        bm = model.bm
        eq = all(torch.equal(model.state_dict()[k], v) for k, v in list(rs["state"].items()) if torch.is_tensor(v)
                 and k.startswith("bm.") and k in model.state_dict() and v.dtype != torch.uint8)
        eqb = torch.equal(model.state_dict()["bm.store.key_syn"], rs["state"]["bm.store.key_syn"])
        check("(B2) the run's own checkpoint (saved by the current code) through train_lm.load_brain_state (strict): "
              "loads; conf_a / conf_b at 2 / 3; the memory's tensors as saved",
              not has and eq and eqb and torch.all(bm.conf_a == 2.0) and torch.all(bm.conf_b == 3.0),
              f"{ckpt}: step {rs['step']}, n_live {bm.n_live}, conf_a {tuple(bm.conf_a.shape)}")
        del rs, model

    # (C) gradients reach conf_a / conf_b through the read only
    m = V.tiny()
    V.warm(m)
    m.conf_parameters_ = m.conf_parameters
    xs = [T._clusters(V.B * V.TT, 32, 500 + j) for j in range(6)]
    T._micro(m, xs, V.B, V.TT, step=False)
    ga, gb = m.conf_a.grad, m.conf_b.grad
    rec = m._cache
    zg = rec["z_g"].requires_grad or rec["z_p"].requires_grad
    check("(C) gradients reach conf_a / conf_b (global entry S and private rows) from the main loss; z detached",
          ga is not None and gb is not None and float(ga[m.S].abs()) > 0 and float(ga[:m.S].abs().sum()) > 0
          and float(gb.abs().sum()) > 0 and not zg,
          f"|grad a| global {float(ga[m.S].abs()):.3e}, private rows {int((ga[:m.S] != 0).sum())} nonzero; "
          f"|grad b| {float(gb.abs().sum()):.3e}")
    for p in m.parameters():
        p.grad = None

    # (D) the sync guard over after_backward (confirms, gate loss, writes) + after_step (AdamW, D update, sweep)
    from torch.utils._python_dispatch import TorchDispatchMode
    import traceback

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.hits = []

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            name = str(func.overloadpacket.__name__)
            bad = name in ("_local_scalar_dense", "nonzero", "unique", "_unique", "_unique2", "unique_dim",
                           "unique_consecutive", "masked_select", "is_nonzero", "equal")
            if name in ("index", "index_put", "index_put_", "_index_put_impl_") and not bad:
                idx = args[1] if len(args) > 1 else None
                bad = any(torch.is_tensor(t) and t.dtype == torch.bool for t in (idx or []) if t is not None)
                name += "[bool]" if bad else ""
            if bad and any("torch" in f.filename and "optim" in f.filename for f in traceback.extract_stack()):
                bad = False               # (AdamW's step counter: a CPU tensor on GPU runs too (not capturable))
            if bad:
                fr = [f for f in traceback.extract_stack() if Path(f.filename).parent == HERE
                      and Path(f.filename).name.startswith("wave")]
                self.hits.append((name, f"{Path(fr[-1].filename).name}:{fr[-1].lineno} {fr[-1].name}" if fr else "?"))
            return func(*args, **(kwargs or {}))

    m = V.tiny()
    V.warm(m)
    n_hits = {}
    orig_ab, orig_as = m.after_backward, m.after_step
    for s in range(4):
        g = Guard()

        def ab():
            with g:
                return orig_ab()

        def as_(step, refit_every=0):
            with g:
                return orig_as(step, refit_every)
        m.after_backward, m.after_step = ab, as_
        T._micro(m, [T._clusters(V.B * V.TT, 32, 700 + 10 * s + j) for j in range(6)], V.B, V.TT)
        for h in g.hits:
            n_hits[h] = n_hits.get(h, 0) + 1
    wpath = {k: v for k, v in n_hits.items() if any(f in k[1] for f in (
        "_writes", "write_delta", "_key_update", "_chain_set", "confirm", "_forget", "_sweep", "learn_writes",
        "_learn_private_slow", "_move", "encode", "_topmask", "_gate_loss", "after_backward", "_entry0", "_zc",
        "_d_update", "_adamw", "_uniq"))}
    other = {k: v for k, v in n_hits.items() if k not in wpath}
    check("(D) sync guard (CPU TorchDispatchMode: _local_scalar_dense / nonzero / unique / masked_select / bool-mask "
          "index): after_backward + after_step, 4 active steps: no host sync in the write / confirm / sweep / D path",
          not wpath and m.cnt["writes_g"] + m.cnt["writes_p"] > 0,
          f"write path {wpath or 'none'}; elsewhere (not the write path) {other or 'none'}")


def run2(check):
    """(E) the batched sweep == the per-place loop; (F) the read magnitude after the writes start (read unit slice)."""
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(HERE))
    import copy
    import tests_wave as T
    import tests_wave_v21 as V
    import wave_v21
    m = V.tiny()
    V.warm(m, steps=10)
    m._cache, m._samp, m._rows = None, None, []
    m.store.fill_target, m.store.fill_ch = 0.01, 0.002             # (force the sweep to decrement)
    a, b = copy.deepcopy(m), copy.deepcopy(m)
    for k in range(3):
        wave_v21.SWEEP_BATCHED = True
        a._sweep()
        wave_v21.SWEEP_BATCHED = False
        b._sweep()
    wave_v21.SWEEP_BATCHED = True
    sa, sb = a.store, b.store
    ok = all(torch.equal(x, y) for x, y in ((sa.key_syn, sb.key_syn), (sa.chain_bits, sb.chain_bits), (sa.load, sb.load),
                                           (sa.fill_est, sb.fill_est), (sa.fill_ch_est, sb.fill_ch_est)))
    ch = int((sa.key_syn != m.store.key_syn).sum())
    check("(E) the batched sweep == the per-place forget loop, bit for bit (key / chain bytes, loads, fill estimates, "
          "n_step)", ok and sa.n_step == sb.n_step and ch > 0 and a.ptr_g == b.ptr_g,
          f"3 sweeps x {min(a.sweep_g, a.Pg) + min(a.sweep_p, a.n_live * a.J)} places; {ch} bytes changed")
    out = {}
    from wave_v21 import WaveV21Memory
    for unit in ("full", "slice"):
        torch.manual_seed(0)
        m = WaveV21Memory(384, 8, wave_side=4, wave_warm=2, wave_anneal=10, wave_write_start=4, wave_upd_rows=512,
                          wave_null_every=1, wave_drop=0.0, wave_read_unit=unit)
        rs, fs, conf = [], [], []
        for s_ in range(STEPS + 6):
            T._micro(m, [T._clusters(V.B * V.TT, 384, 300 + 10 * s_ + j) for j in range(6)], V.B, V.TT)
            if m._ratio is not None and m.active:
                rs.append(float(m._ratio))
                fs.append(float(m._fire))
                rec = m._cache
                o = rec["r3"].detach()[rec["dec"]["S"]].norm(dim=-1) / rec["xn"][rec["dec"]["tok"]].clamp(min=1e-6)
                cg = torch.zeros(rec["xn"].numel()).index_put((rec["dec"]["GT"],), rec["c_g"] * rec["kap_g"])
                ck = cg[rec["dec"]["tok"]] * rec["dec"]["vg"].float()
                hi = ck > 0.5
                lo = ck > 0.05
                conf.append((float(o[hi].mean()) if bool(hi.any()) else float("nan"),
                             float((o / ck.clamp(min=1e-6))[lo].median()) if bool(lo.any()) else float("nan")))
        out[unit] = (rs, fs, conf)
    rs, fs, conf = out["slice"]
    per = [c[1] for c in conf if c[1] == c[1]]
    perf = [c[1] for c in out["full"][2] if c[1] == c[1]]
    check("(F) read magnitude (slice): a confident read (c kappa_bar = 1, s at init) arrives at ~0.1-0.2 |xf|",
          bool(per) and 0.08 <= sum(per) / len(per) <= 0.3,
          f"d 384, dl 2048, 64 columns (tiny sheet) after write_start: wmem_read_ratio slice {[round(r, 4) for r in rs[-4:]]} vs full "
          f"{[round(r, 4) for r in out['full'][0][-4:]]}; fire_rate {[round(f, 3) for f in fs[-4:]]}; |o| / |xf| per "
          f"unit c kappa_bar (median, global reads): slice {sum(per) / max(1, len(per)):.3f}, full "
          f"{sum(perf) / max(1, len(perf)):.3f}; at c kappa_bar > 0.5: {[round(c[0], 3) for c in conf[-4:]]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="the run's code snapshot (or a copy): the reference code")
    ap.add_argument("--ckpt", default="", help="also resume this checkpoint (the run's last.pt) on CPU (read-only)")
    a = ap.parse_args()
    fails = []

    def check(name, ok, info=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)
        if not ok:
            fails.append(name)
    run(check, a.old, a.ckpt)
    run2(check)
    print("ALL PASS" if not fails else f"{len(fails)} FAIL", flush=True)
    sys.exit(1 if fails else 0)
