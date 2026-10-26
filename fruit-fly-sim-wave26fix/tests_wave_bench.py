"""W-bench parity of the wave ADDRESS (wave_address.py) and the full-module bench of --memory wave. CPU, 2 threads.

    set CUDA_VISIBLE_DEVICES=-1 & python -B tests_wave_bench.py [--passes 1,2,3,4,5] [--module 1] [--check-scorer 1]

(1) parity: the module learns in-step style (online, batched updates on the capture's fit thoughts, a few epochs =
    the first training steps' xf), then its codes are scored with the W-bench's idealised code-match scoring of
    variant "(c+d) SOM256 + 3 residual levels x 32 slices, top-2" (speed_wip/mem26/wbench.py m_entry_code score:
    w_loc 4 x [any of the query's 2 locations == any of the stored 2] + sum over the 96 (level, slice) of [stored
    top-1 code within the query's top-2], 1e-3 random tie noise from default_rng(1)). The scoring below is that
    function computed with one-hot matmuls (identical numbers: --check-scorer 1 compares it with wbench's own on a
    subset). wbench reference, mean over passes 1-5: held top-1 0.684, cloze top-1 0.524.
(2) --module 1: the whole WaveMemory through wbench_module.Harness (its real write / read path, the placeholder
    storage) via wave_bench.py.

wbench / wbench_module are imported read-only from speed_wip/mem26 (no bytecode written there).
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
from wave_address import WaveAddress  # noqa: E402

REF = dict(held=0.684, cloze=0.524)


def train_address(xs, seed, epochs=8, batch=256, **kw):
    """In-step style learning: the fit thoughts as a stream of micro-batches (epochs x shuffled), one update each."""
    n = len(xs)
    steps = epochs * ((n + batch - 1) // batch)
    a = WaveAddress(xs.shape[1], seed=seed, anneal=kw.pop("anneal", int(0.8 * steps)), warm=1, **kw)
    g = np.random.default_rng(seed)
    X = torch.from_numpy(xs)
    for _ in range(epochs):
        p = torch.from_numpy(g.permutation(n))
        for i in range(0, n, batch):
            a.update(X[p[i:i + batch]])
    return a


def fast_score(w_loc=4.0, L=256, n_lev_sl=96, M=16):
    """wbench m_entry_code's score(q, s) as matmuls (exact same numbers)."""
    def score(q, s):
        ql, qu = q
        sl, su = s
        oh = lambda idx, K: np.eye(K, dtype=np.float32)[idx].reshape(idx.shape[0], -1, K).max(1)  # multi-hot
        locm = (oh(ql, L) @ oh(sl, L).T > 0).astype(np.float32)
        nq, J = qu.shape[0], qu.shape[1]
        qh = np.zeros((nq, J * M), np.float32)
        for t in range(qu.shape[2]):
            qh[np.arange(nq)[:, None], np.arange(J)[None] * M + qu[:, :, t]] = 1
        sh = np.zeros((su.shape[0], J * M), np.float32)
        sh[np.arange(su.shape[0])[:, None], np.arange(J)[None] * M + su[:, :, 0]] = 1
        um = qh @ sh.T
        return w_loc * locm + um + 1e-3 * np.random.default_rng(1).random(locm.shape, dtype=np.float32)
    return score


def make_module_codes(epochs=8, batch=256, **kw):
    def make(b, k):
        xs = b.feats("fit", k, ["xf"])
        a = train_address(xs, seed=k, epochs=epochs, batch=batch, **kw)

        def enc(which, k_):
            c = a.encode(torch.from_numpy(b.feats(which, k_, ["xf"])))
            return c.loc.numpy(), c.code.numpy()
        return enc, fast_score(L=a.L, M=a.M)
    return make


def check_scorer(b, k=3, nq=200):
    """fast_score == wbench's own score on wbench's own (c+d) codes (a query subset)."""
    enc, score = W.m_entry_code("som", 256, 32, 16, r=2, coarse_fine=3)(b, k)
    cs = enc("store", k)
    cq = enc("held", k)
    cq = (cq[0][:nq], cq[1][:nq])
    S0 = score(cq, cs)
    S1 = fast_score()(cq, cs)
    return bool(np.array_equal(S0, S1)), float(np.abs(S0 - S1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", default="1,2,3,4,5")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--check-scorer", type=int, default=1)
    ap.add_argument("--module", type=int, default=1)
    ap.add_argument("--wbench-ref", type=int, default=0, help="1 = also run wbench's own (c+d) with the fast scorer")
    a = ap.parse_args()
    b = W.Bench(passes=tuple(int(x) for x in a.passes.split(",")))
    if a.check_scorer:
        ok, md = check_scorer(b)
        print(f"{'PASS' if ok else 'FAIL'}  fast scorer == wbench m_entry_code score (200 held queries, pass 3), "
              f"max |diff| {md:.2e}", flush=True)
    if a.wbench_ref:
        def ref(b_, k):
            enc, _ = W.m_entry_code("som", 256, 32, 16, r=2, coarse_fine=3)(b_, k)
            return enc, fast_score()
        b.run("wbench (c+d) itself, fast scorer", ref, 49)
    t0 = time.time()
    res = b.run(f"wave_address (in-step learning, {a.epochs} ep x {a.batch})", make_module_codes(a.epochs, a.batch), 49)
    m = lambda q: float(np.mean([res[(q, k)][0] for k in b.passes]))
    held, cloze = m("held"), m("cloze")
    full = tuple(b.passes) == (1, 2, 3, 4, 5)
    tol = 0.03
    ok = abs(held - REF["held"]) <= tol and abs(cloze - REF["cloze"]) <= tol
    print(f"{'PASS' if ok or not full else 'FAIL'}  parity: held top-1 {held:.3f} (wbench {REF['held']}), cloze top-1 "
          f"{cloze:.3f} (wbench {REF['cloze']}), tolerance {tol}" + ("" if full else " (not all passes: info only)")
          + f" ({time.time() - t0:.0f}s)", flush=True)
    if a.module:
        import wave_bench
        wave_bench.run_harness(b)


if __name__ == "__main__":
    main()
