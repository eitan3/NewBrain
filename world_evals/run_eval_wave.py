"""eval_world.py for the wave-memory run (world_wave26_12k: --memory wave --wave-mode v21), with wave knockouts.

    CPU:  python -B run_eval_wave.py --ckpt <run>/final.pt [eval_world args]          (CUDA hidden, 2 threads)
    GPU:  python -B run_eval_wave.py --ckpt <run>/final.pt --device cuda [args]       (16 GB cap, 2 threads)
    checks: python -B run_eval_wave.py --ckpt <run>/ckpt_3000.pt --verify 1 [--limit 2 ...]   (CPU self-test)

The model code is imported from <run>/code_snapshot (never written: no bytecode, --out always outside the run).
Loading: eval_world.load_model (the checkpoint, its embedded bm.* state) on the CPU, then <ckpt>.memory.pt (the memory
file the trainer saved next to it) with memory_api.load_memory_file, strict; both checksums are printed. Then .to(device).
Knockouts (each one a mode of the SAME loaded memory; nothing is written during eval: model.eval() -> the module's
_live is False -> no writes / counters / address learning / exploration / sequence drop):
    mem_on        the training-time read path in eval mode (exploration 0, keep 1)
    ko_wave       memory_api.memory_off: bm.enabled = False -> LoopBrain runs the no-memory code path (module not called)
    private_only  the global reads zeroed (vg, gvalid, c_g = 0 inside WaveV21Memory._hint; selection unchanged)
    global_only   the private reads zeroed (vp, c_p = 0 inside _hint; selection unchanged)
WAVE_KO=mem_on,ko_wave (env) restricts the list. The patch lives only in this file.
"""
import os
import sys
from pathlib import Path

argv = sys.argv


def _opt(name, default=None):
    if name in argv[:-1]:
        return argv[argv.index(name) + 1]
    for x in argv:
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return default


DEV = _opt("--device", "cpu")
VERIFY = int(_opt("--verify", "0"))
if "--verify" in argv:
    i = argv.index("--verify")
    del argv[i:i + 2]
sys.dont_write_bytecode = True                      # (never write __pycache__ into the run's code_snapshot)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if DEV != "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
if _opt("--threads") is None:
    argv += ["--threads", "2"]
CKPT = Path(_opt("--ckpt", r"C:\Eitan\Python\NewBrain\runs\world_wave26_12k\final.pt"))
SNAP = CKPT.parent / "code_snapshot"
if not SNAP.exists():
    SNAP = Path(r"C:\Eitan\Python\NewBrain\runs\world_wave26_12k\code_snapshot")
OUT_DIR = Path(r"C:\Eitan\Python\NewBrain\world_evals")
if _opt("--out") is None:   # (eval_world's default is <snapshot>/../world_evals = inside the run: never)
    argv += ["--out", str(OUT_DIR / f"{CKPT.parent.name}_{CKPT.stem}_wave{'_verify' if VERIFY else ''}.json")]
sys.path.insert(0, str(SNAP))

import torch                                        # noqa: E402
torch.set_num_threads(2)
if DEV == "cuda":                                   # (before any CUDA allocation)
    _total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 16e9 / _total), torch.device("cuda", 0))
import hashlib                                      # noqa: E402

import eval_world as ew                             # noqa: E402  (the snapshot's copy: identical to fruit-fly-sim's)
import memory_api                                   # noqa: E402
import wave_v21                                     # noqa: E402
from wave_v21 import WaveV21Memory                  # noqa: E402

MODES = [("mem_on", None), ("ko_wave", "off"), ("private_only", "private_only"), ("global_only", "global_only")]

# ---------------------------------------------------------------- the read masks (private_only / global_only)
_orig_hint = WaveV21Memory._hint


def _hint(self, rec, keep):
    m = getattr(self, "_ko_mask", None)
    if m is None:
        return _orig_hint(self, rec, keep)
    assert not self._live, "read masks are an eval-only knockout"
    dec = dict(rec["dec"])
    r2 = dict(rec, dec=dec)
    if m == "private_only":          # global reads zeroed: no token's global read reaches any pair (incl. the shared)
        dec["vg"] = torch.zeros_like(dec["vg"])
        dec["gvalid"] = torch.zeros_like(dec["gvalid"])
        r2["c_g"] = torch.zeros_like(rec["c_g"])
    elif m == "global_only":         # private reads zeroed
        dec["vp"] = torch.zeros_like(dec["vp"])
        r2["c_p"] = torch.zeros_like(rec["c_p"])
    else:
        raise ValueError(m)
    out = _orig_hint(self, r2, keep)
    for k in ("y_raw", "y_g_tok"):
        if k in r2:
            rec[k] = r2[k]
    return out


WaveV21Memory._hint = _hint


def knockouts(model, gate_modes=True):
    out = [(n, "wave", v) for n, v in MODES]
    only = os.environ.get("WAVE_KO")
    if only:
        out = [o for o in out if o[0] in only.split(",")]
    return out


class KO:
    def __init__(self, model, attr, value):
        self.bm, self.value = model.bm, value

    def __enter__(self):
        self.old = (self.bm.enabled, getattr(self.bm, "_ko_mask", None))
        if self.value == "off":
            self.bm.enabled = False
        elif self.value is not None:
            self.bm._ko_mask = self.value

    def __exit__(self, *exc):
        self.bm.enabled, self.bm._ko_mask = self.old


ew.knockouts, ew.KO = knockouts, KO


# ---------------------------------------------------------------- loading + checksums
def _h_obj(h, v):
    if torch.is_tensor(v):
        t = v.detach().cpu().contiguous()
        h.update(f"{t.dtype}{tuple(t.shape)}".encode())
        if t.numel():
            h.update(t.reshape(-1).view(torch.uint8).numpy())
    elif isinstance(v, dict):
        for k in sorted(v, key=str):
            h.update(str(k).encode())
            _h_obj(h, v[k])
    elif isinstance(v, (list, tuple)):
        for x in v:
            _h_obj(h, x)
    else:
        h.update(repr(v).encode())


def mem_checksum(bm, with_python=True):
    """Every parameter and buffer of the memory (persistent or not), its extra state (address / store host state,
    counters, optimizer state) and the python counters -> blake2b hex (16)."""
    h = hashlib.blake2b(digest_size=8)
    for n, p in sorted(bm.named_parameters()):
        h.update(n.encode())
        _h_obj(h, p)
    for n, b in sorted(bm.named_buffers()):
        h.update(n.encode())
        _h_obj(h, b)
    _h_obj(h, bm.get_extra_state())
    if with_python:
        _h_obj(h, dict(cnt=bm.cnt, n_fwd=bm.n_fwd, n_opt=bm.n_opt, ptr=(bm.ptr_g, bm.ptr_p), kappa=bm.kappa,
                       n_upd=getattr(bm.address, "n_upd", None), n_step=getattr(bm.store, "n_step", None)))
    return h.hexdigest()


_orig_load = ew.load_model
_cache_model = {}


def load_model(a):
    if "out" in _cache_model:
        return _cache_model["out"]
    dev = a.device
    a.device = "cpu"                                # (load + memory file on the CPU, then one move to the device)
    try:
        model, step, run_cfg, missing, unexpected = _orig_load(a)
    finally:
        a.device = dev
    bm = model.bm
    assert type(bm) is WaveV21Memory, f"expected the v21 wave memory, got {type(bm).__name__}"
    miss_bm = [k for k in missing if k.startswith("bm.")]
    h_emb = mem_checksum(bm, with_python=False)
    mp = memory_api.memory_path(a.ckpt)
    if mp.exists():
        obj = memory_api.load_memory_file(mp, model, strict=True)
        h_file = mem_checksum(bm, with_python=False)
        info = (f"memory file {mp.name} (step {obj.get('step')}, kind {obj['kind']}, n_experts {obj.get('n_experts')}) "
                f"loaded strict; checksum embedded {h_emb} -> file {h_file} "
                f"({'IDENTICAL' if h_emb == h_file else 'DIFFERENT: the file wins'})")
        del obj
    else:
        assert not miss_bm, f"no {mp.name} and the checkpoint lacks {miss_bm[:4]}"
        info = f"no memory file; embedded memory state ({h_emb})"
    model.to(dev).eval()
    print(f"[wave wrapper] {info}; memory {type(bm).__name__} active {bm.active} n_opt {bm.n_opt} n_live {bm.n_live} "
          f"read_pass {bm.read_pass} explore {bm.explore} training {bm.training}; missing bm keys {len(miss_bm)}; "
          f"threads {torch.get_num_threads()}, cuda {torch.cuda.is_available()}, snapshot {SNAP}", flush=True)
    if DEV == "cuda":
        print(f"[wave wrapper] CUDA cap {torch.cuda.get_per_process_memory_fraction(0) if hasattr(torch.cuda, 'get_per_process_memory_fraction') else '16e9/total'}; "
              f"allocated after load {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    out = (model, step, run_cfg, missing, unexpected)
    _cache_model["out"] = out
    return out


ew.load_model = load_model


# ---------------------------------------------------------------- the CPU self-test (--verify 1)
def verify():
    import types
    import json
    import numpy as np
    import torch.nn.functional as F
    t0 = __import__("time").time()
    a = types.SimpleNamespace(ckpt=str(CKPT), config=_opt("--config", ""), capacity="dropless", device="cpu",
                              k=int(_opt("--k", "6")))
    model = load_model(a)[0]
    bm = model.bm
    data = Path(_opt("--data", "X:/Datasets/wiki_world_mix"))
    meta = json.loads((data / "meta.json").read_text())
    vw = np.memmap(data / "val_world.bin", dtype=np.dtype(meta.get("dtype", "uint16")), mode="r")
    B, T = 4, 128
    g = np.random.RandomState(7)
    xy = torch.from_numpy(np.stack([vw[i:i + T + 1].astype(np.int64) for i in g.randint(0, len(vw) - T - 1, B)]))
    x, y = xy[:, :-1], xy[:, 1:]

    rec = dict(calls=[], X=[], hint=[], cap_in=None)
    orig_call = WaveV21Memory.__call__
    zero_inject = {"on": False}

    def rec_call(self, xf, buf, xx, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=None):
        if int(call or 0) == self.read_pass and rec["cap_in"] is None:
            rec["cap_in"] = (xf.clone(), buf.clone(), xx.clone(), side, e_k.clone(), p_k.clone(), t_k.clone(),
                             slot_k.clone(), counts, N, E, C, n_keep, call)
        b2, x2 = orig_call(self, xf, buf, xx, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=call)
        rec["calls"].append((int(call or 0), float((b2.float() - buf.float()).norm()), float((x2.float() - xx.float()).norm())))
        if zero_inject["on"]:
            return buf, xx
        return b2, x2
    WaveV21Memory.__call__ = rec_call

    class TRec:
        @staticmethod
        def apply(Xr, Xs, D, E, sh, gb):
            rec["X"].append((Xr.detach().clone(), Xs.detach().clone()))
            return wave_v21.Translate.__orig__.apply(Xr, Xs, D, E, sh, gb)
    TRec.__orig__ = wave_v21.Translate
    wave_v21.Translate = TRec
    TRec_orig = TRec.__orig__

    oh = WaveV21Memory._hint

    def hint_rec(self, r, keep):
        out = oh(self, r, keep)
        rec["hint"].append(dict(dec={k: (v.clone() if torch.is_tensor(v) else v) for k, v in r["dec"].items()},
                                y_g_tok=r["y_g_tok"].clone(), y_raw=tuple(t.clone() for t in r["y_raw"]),
                                c_g=r["c_g"].clone(), c_p=r["c_p"].clone(), r3=out.detach().clone()))
        return out
    WaveV21Memory._hint = hint_rec

    def run(name, ctx):
        rec["calls"], rec["X"], rec["hint"], rec["cap_in"] = [], [], [], None
        with torch.no_grad(), ctx:
            lg = model(x).float()
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1)).item()
        inj_b = sum(c[1] for c in rec["calls"])
        inj_x = sum(c[2] for c in rec["calls"])
        return dict(name=name, logits=lg, loss=loss, calls=list(rec["calls"]), inj_pairs=inj_b, inj_shared=inj_x,
                    X=list(rec["X"]), hint=list(rec["hint"]), cap_in=rec["cap_in"])

    import contextlib
    null = contextlib.nullcontext()
    h0 = mem_checksum(bm)
    R = {}
    R["on"] = run("mem_on", KO(model, "wave", None))
    R["off"] = run("ko_wave", KO(model, "wave", "off"))
    R["off_api"] = run("memory_off", memory_api.memory_off(model))
    saved, model.bm = model.bm, None
    R["removed"] = run("bm=None", null)
    model.bm = saved
    zero_inject["on"] = True
    R["zero"] = run("called, inject dropped", null)
    zero_inject["on"] = False
    R["priv"] = run("private_only", KO(model, "wave", "private_only"))
    R["glob"] = run("global_only", KO(model, "wave", "global_only"))
    R["on2"] = run("mem_on again", null)
    h1 = mem_checksum(bm)
    WaveV21Memory._hint = oh
    pr = lambda *s: print("[verify]", *s, flush=True)
    pr(f"fixed batch: {B} x {T} val_world tokens; K={model.K}; read pass {bm.read_pass}; memory active {bm.active}")
    on = R["on"]
    for k, r in R.items():
        d = (r["logits"] - on["logits"]).abs().max().item()
        pr(f"{r['name']:24s} loss {r['loss']:.5f}  logits[0,-1,:3] {r['logits'][0, -1, :3].tolist()}  "
           f"max|logits - on| {d:.4e}  memory calls {len(r['calls'])} (passes {sorted({c[0] for c in r['calls']})}) "
           f"inject |pairs| {r['inj_pairs']:.4e} |shared| {r['inj_shared']:.4e}")
    # (a) OFF
    eq = lambda u, v: bool(torch.equal(R[u]["logits"], R[v]["logits"]))
    pr(f"(a) OFF: ko_wave memory calls {len(R['off']['calls'])}, inject sum {R['off']['inj_pairs'] + R['off']['inj_shared']:.1f}; "
       f"logits ko_wave == memory_off(model): {eq('off', 'off_api')}; == bm removed (model.bm=None): {eq('off', 'removed')}; "
       f"== memory called with its inject dropped: {eq('off', 'zero')}; ko_wave != mem_on: {not eq('off', 'on')}")
    # (b) ON: nonzero inject, = the training-time read path (exploration off, keep 1) on the captured pass-3 inputs
    st = on["hint"][0]["dec"]["stats"].tolist()
    pr(f"(b) ON: inject |pairs| {on['inj_pairs']:.4f} |shared| {on['inj_shared']:.4f} over passes "
       f"{sorted({c[0] for c in on['calls'] if c[1] + c[2] > 0})}; reads: private pairs {int(st[3])}, global tokens "
       f"{int(st[4])} of N={B * T}; exploration picks (private, global) = ({int(st[5])}, {int(st[6])}); "
       f"mem_on deterministic (rerun identical): {eq('on', 'on2')}")
    cap = on["cap_in"]
    keep_state = dict(kappa=bm.kappa.clone(), cnt=dict(bm.cnt), n_fwd=bm.n_fwd, stat=getattr(bm, "_stat_acc", None),
                      explore=bm.explore, drop=bm.drop)
    outs = {}
    WaveV21Memory.__call__ = orig_call
    for mode in ("eval", "train"):
        bm._cache, bm._rows, bm._samp = None, [], None
        if mode == "train":
            bm.train()
            bm.explore, bm.drop, bm._live = 0.0, 0.0, True
        else:
            bm._live = False
        with torch.set_grad_enabled(mode == "train"):
            b2, x2 = bm(*cap[:-1], call=cap[-1])
        outs[mode] = (b2.detach().float(), x2.detach().float(), bm._cache["dec"]["stats"].tolist())
        bm.eval()
        bm._live = False
    bm.kappa.copy_(keep_state["kappa"])
    bm.cnt, bm.n_fwd, bm._stat_acc = keep_state["cnt"], keep_state["n_fwd"], keep_state["stat"]
    bm.explore, bm.drop = keep_state["explore"], keep_state["drop"]
    bm._cache, bm._rows, bm._samp = None, [], None
    for p in bm.parameters():
        p.grad = None
    db = (outs["eval"][0] - outs["train"][0]).abs().max().item()
    dx = (outs["eval"][1] - outs["train"][1]).abs().max().item()
    pr(f"(b) pass-3 read, eval path vs training path (_live, explore 0, drop 0; restored after): max|d buf| {db:.3e} "
       f"max|d shared x| {dx:.3e}; selection stats eval {outs['eval'][2]} train {outs['train'][2]}; "
       f"inject |pairs| {(outs['eval'][0] - cap[1].float()).norm():.4f}")
    h2 = mem_checksum(bm)
    # (d) masks
    hon, hpv, hgl = on["hint"][0], R["priv"]["hint"][0], R["glob"]["hint"][0]
    dec = hon["dec"]
    same_sel = all(torch.equal(hon["dec"][k], h["dec"][k]) for h in (hpv, hgl) for k in ("S", "slot", "PR", "GT", "tok"))
    vg, vp = dec["vg"], dec["vp"]
    Xon = torch.cat([on["X"][0][0].reshape(-1, bm.d), on["X"][0][1]])
    Xpv = torch.cat([R["priv"]["X"][0][0].reshape(-1, bm.d), R["priv"]["X"][0][1]])
    Xgl = torch.cat([R["glob"]["X"][0][0].reshape(-1, bm.d), R["glob"]["X"][0][1]])
    # per selected pair (row of the Translate input): its slot; components of mem_on
    K = dec["K"]
    slot = dec["slot"]
    n_slot = Xon.size(0)
    okslot = slot < n_slot
    pair_of_slot = lambda X: X[slot.clamp(max=n_slot - 1)] * okslot[:, None].float()
    Pon, Ppv, Pgl = pair_of_slot(Xon), pair_of_slot(Xpv), pair_of_slot(Xgl)
    yg_comp = vg[:, None].float() * hon["y_g_tok"][dec["tok"]]
    yp_comp = hon["y_raw"][1]
    only_g = vg & ~vp
    only_p = vp & ~vg
    shared = (dec["S"] % 7) == 6
    nz = lambda t: int((t.abs().sum(-1) > 0).sum())
    pr(f"(d) selection identical in on / private_only / global_only: {same_sel}; selected pairs {K}: global-read "
       f"{int(vg.sum())}, private-read {int(vp.sum())}, both {int((vg & vp).sum())}, shared-expert pairs {int(shared.sum())}")
    pr(f"(d) private_only: global part zero everywhere: token global reads |.| {hpv['y_g_tok'].abs().sum():.1f} "
       f"(on {hon['y_g_tok'].abs().sum():.3f}), pair global part |.| {hpv['y_raw'][0].abs().sum():.1f}; "
       f"pairs reading only global -> nonzero {nz(Ppv[only_g])}/{int(only_g.sum())} (on: {nz(Pon[only_g])}); shared pairs "
       f"nonzero {nz(Ppv[shared])} (on {nz(Pon[shared])}); input == on's private part exactly: "
       f"{bool(torch.equal(Ppv, yp_comp * okslot[:, None].float()))}")
    pr(f"(d) global_only: private part zero: pair private part |.| {hgl['y_raw'][1].abs().sum():.1f} "
       f"(on {hon['y_raw'][1].abs().sum():.3f}); "
       f"pairs reading only private -> nonzero {nz(Pgl[only_p])}/{int(only_p.sum())} (on: {nz(Pon[only_p])}); "
       f"input == on's global part exactly: {bool(torch.equal(Pgl, yg_comp * okslot[:, None].float()))}; "
       f"on == private part + global part: {bool(torch.equal(Pon, (yg_comp + yp_comp) * okslot[:, None].float()))} "
       f"(max diff {(Pon - (yg_comp + yp_comp) * okslot[:, None].float()).abs().max():.2e})")
    pr(f"(c) memory checksum before {h0}, after the 8 verification forwards {h1}, after the eval/train read test {h2}: "
       f"{'IDENTICAL' if h0 == h1 == h2 else 'CHANGED'}")
    wave_v21.Translate = TRec_orig
    WaveV21Memory._hint = _hint
    WaveV21Memory.__call__ = orig_call
    del R, outs, cap
    # the tiny eval_world run through the same wrapper, checksum around it
    pr(f"--- tiny eval_world.main (checks took {__import__('time').time() - t0:.0f}s)")
    hb = mem_checksum(bm)
    ew.main()
    ha = mem_checksum(bm)
    pr(f"(c) memory checksum before eval_world.main {hb}, after {ha}: {'IDENTICAL' if hb == ha else 'CHANGED'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if VERIFY:
        assert DEV == "cpu", "--verify runs on the CPU"
        verify()
    else:
        ew.main()
