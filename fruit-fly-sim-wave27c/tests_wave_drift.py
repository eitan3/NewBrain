"""Address drift of the wave memory (plan 26; coordinator: two-speed map + reconsolidation). CPU, 2 threads.

    set CUDA_VISIBLE_DEVICES=-1 & python -B tests_wave_drift.py [--pass 3] [--updates 2000] [--skip-map 1]
        [--write-cap cap/w1c_9250.npz --read-cap cap/w1c_12207.npz] [--kw wave_anneal=2000]

Captures (read-only): speed_wip/mem26/cap/w1c_6k.npz (blob run ckpt_6000) and this folder's cap/w1c_9250.npz (the
same documents through ckpt_9250: capture_xf.py, a copy of mem26's w1c_capture.py). The same facts, the same order.

1. MAP drift (the address alone): the address learns in-step style on the 6K fit thoughts (the warm-up), then keeps
   learning for `--updates` more updates (4096 thoughts each; 2000 = ~1000 steps at accum 2) on the 9250 fit
   thoughts. Agreement of the stored facts' codes (their 6K thoughts, before vs after): the top-1 entry location and
   the 96 top-1 level-slice codes. fast map (one speed) vs two-speed map (fast in the warm-up, then slow).
   Target: two-speed >= 0.8.
2. BRAIN drift (the whole memory, the default store (the v2 column store), global store, gate bypassed, wbench_module
   scoring: cosine of the
   read with every fact's content): write every stored fact with its 6K thought, then read the held-out / cloze cues
   with their 9250 thoughts. Variants: map frozen after the warm-up / fast map / two-speed map, each without and with
   reconsolidation (a rehearsal: every stored fact recurs once with its 9250 training-template thought; a recall that
   matches the fact's content is confirmed: +1 along the recalled path under the CURRENT codes; nothing new is
   written). Reference: write and read at 6K. Forgetting does not run (no training steps here).
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
import copy
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
from wave_memory import WAVE_DEFAULTS, CodedWaveMemory  # noqa: E402

CAP9 = HERE / "cap" / "w1c_9250.npz"


def warm_address(m, xs, seed, epochs=8, batch=256):
    g = np.random.default_rng(seed)
    X = torch.from_numpy(xs)
    for _ in range(epochs):
        p = torch.from_numpy(g.permutation(len(X)))
        for i in range(0, len(X), batch):
            m.address.update(X[p[i:i + batch]])


def continue_address(addr, xs, updates, seed, rows=4096):
    g = torch.Generator().manual_seed(seed)
    X = torch.from_numpy(xs)
    for _ in range(updates):
        addr.update(X[torch.randint(0, len(X), (rows,), generator=g)])


def agreement(a0, a1, X):
    c0, c1 = a0.encode(X), a1.encode(X)
    loc = float((c0.loc[:, 0] == c1.loc[:, 0]).float().mean())
    code = float((c0.code[:, :, 0] == c1.code[:, :, 0]).float().mean())
    return loc, code, (loc + 96 * code) / 97


def address_only(k, n_warm, fast):
    """The memory's address alone, as CodedWaveMemory builds it (the map-drift part needs no store)."""
    from wave_address import WaveAddress
    d = WAVE_DEFAULTS
    a = WaveAddress(384, side=d["wave_side"], levels=d["wave_levels"], slices=d["wave_slices"], codes=d["wave_codes"],
                    topk=d["wave_topk"], entries=d["wave_entries"], seed=k, warm=1, anneal=int(0.8 * n_warm),
                    fast=fast, eta_slow=d["wave_eta_slow"])
    return type("A", (), dict(address=a))()


def build(k, n_warm, fast, **kw):
    c = dict(WAVE_DEFAULTS, wave_drop=0.0, wave_explore=0.0, wave_warm=1, wave_seed=k, wave_priv_levels=0,
             wave_fast=fast, wave_anneal=int(0.8 * n_warm))         # (the parity protocol's anneal)
    c.update(kw)                                                    # (e.g. wave_anneal=2000: wave_bench.factory's)
    return CodedWaveMemory(384, 769, **c).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="k", type=int, default=3)
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--skip-recall", type=int, default=0)
    ap.add_argument("--skip-map", type=int, default=0, help="1 = part 2 only (the map-drift part takes ~1 h)")
    ap.add_argument("--kw", default="", help="part 2: extra cfg, e.g. wave_map_beta=0.25,wave_entries=4")
    ap.add_argument("--write-cap", default="", help="the capture written with (default: mem26 w1c_6k.npz)")
    ap.add_argument("--read-cap", default="", help="the capture read with (default: cap/w1c_9250.npz)")
    a = ap.parse_args()
    k = a.k
    b6 = W.Bench(path=a.write_cap, passes=(k,)) if a.write_cap else W.Bench(passes=(k,))
    b9 = W.Bench(path=a.read_cap or CAP9, passes=(k,))
    print(f"write capture {a.write_cap or 'mem26 w1c_6k'}, read capture {a.read_cap or CAP9}", flush=True)
    same = np.array_equal(b6.rec, b9.rec) and np.array_equal(b6.crec, b9.crec)
    print(f"{'PASS' if same else 'FAIL'}  the 6K and 9250 captures hold the same facts in the same order", flush=True)
    h = WM.Harness(b6)
    fit6, fit9 = b6.feats("fit", k, ["xf"]), b9.feats("fit", k, ["xf"])
    st6 = torch.from_numpy(b6.feats("store", k, ["xf"]))
    st9 = torch.from_numpy(b9.feats("store", k, ["xf"]))
    n_warm = 8 * ((len(fit6) + 255) // 256)
    xcos = float(torch.nn.functional.cosine_similarity(st6, st9, dim=-1).mean())
    print(f"pass {k}: cos(xf at 6K, xf at 9250) of the same stored sentence: {xcos:.3f}", flush=True)

    # ---------------------------------------------------------------- 1. map drift
    t0 = time.time()
    res = {}
    for name, fast in ((("fast map (one speed)", 0), ("two-speed map", n_warm)) if not a.skip_map else ()):
        m = address_only(k, n_warm, fast)
        warm_address(m, fit6, k)
        a0 = copy.deepcopy(m.address)
        continue_address(m.address, fit9, a.updates, k)
        res[name] = agreement(a0, m.address, st6)
        # pure map noise: the same 6K stream continued
        m2 = address_only(k, n_warm, fast)
        warm_address(m2, fit6, k)
        b0 = copy.deepcopy(m2.address)
        continue_address(m2.address, fit6, a.updates, k + 100)
        same6 = agreement(b0, m2.address, st6)
        print(f"{name:22s}: codes of the stored facts after {a.updates} more updates on the 9250 thoughts: "
              f"location {res[name][0]:.3f}, codes {res[name][1]:.3f}, all {res[name][2]:.3f} | "
              f"(same 6K stream continued: all {same6[2]:.3f}) ({time.time() - t0:.0f}s)", flush=True)
    if res:
        ok = res["two-speed map"][2] >= 0.8
        print(f"{'PASS' if ok else 'FAIL'}  two-speed map: stored codes keep >= 0.8 agreement over ~1000 steps "
              f"({res['two-speed map'][2]:.3f}; one speed {res['fast map (one speed)'][2]:.3f})", flush=True)
    if a.skip_recall:
        return

    # ---------------------------------------------------------------- 2. brain drift, the whole memory
    tg = torch.from_numpy(h.content[[h.lab_pos[int(l)] for l in b6.store_lab]])

    def score(m, b, label):
        out = []
        for q, qlab in (("held", b.held_lab), ("cloze", b.cz_lab)):
            xq = torch.from_numpy(b.feats(q, k, ["xf"]))
            _, rs = m.read(xq, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), call=k)
            r = np.zeros((len(xq), 384), np.float32) if rs is None else rs.float().numpy()
            out.append(h.score(r, qlab)[0])
        print(f"  {label:58s} held top-1 {out[0]:.3f}  cloze top-1 {out[1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        return out

    base = {}
    for name, fast in (("frozen", None), ("fast", 0), ("two-speed", n_warm)):
        kw = {p_.split("=")[0]: type(WAVE_DEFAULTS[p_.split("=")[0]])(p_.split("=")[1]) for p_ in a.kw.split(",") if p_}
        m = build(k, n_warm, fast if fast is not None else WAVE_DEFAULTS["wave_fast"], **kw)   # (frozen: as T3)
        warm_address(m, fit6, k)
        m.bench_open = True
        if name == "frozen":
            for i in range(0, len(st6), 4096):
                m.bench_write(st6[i:i + 4096], tg[i:i + 4096])
            print(f"memory: {api_bytes(m) / 1e6:.0f} MB; store {m.store.fill()}", flush=True)
            score(m, b6, "reference: write 6K, read 6K")
            written = (copy.deepcopy(m.store.state_dict()), m.store.host_state(),
                       copy.deepcopy(m.address.state_dict()), m.address.host_state())   # (the map as written)
        else:
            m.store.load_state_dict(written[0])
            m.store.load_host_state(written[1])
            m.address.load_state_dict(written[2])
            m.address.load_host_state(written[3])
        if fast is not None:
            continue_address(m.address, fit9, a.updates, k)
        base[name] = score(m, b9, f"write 6K, read 9250, map {name}, no reconsolidation")
        conf = []
        for i in range(0, len(st9), 4096):
            conf.append(m.bench_confirm(st9[i:i + 4096], tg[i:i + 4096]))
        score(m, b9, f"... + reconsolidation (rehearsal; {np.mean(conf):.2f} of recurrences confirmed)")


def api_bytes(m):
    from memory_api import byte_budget
    return byte_budget(m)["total"]


if __name__ == "__main__":
    main()
