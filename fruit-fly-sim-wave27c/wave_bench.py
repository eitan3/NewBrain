"""The W-bench adapter of --memory wave: scores the whole CodedWaveMemory (its real address, write path and store) with
speed_wip/mem26/wbench_module.Harness, offline on the ckpt_6000 capture. wbench / wbench_module are imported
read-only (no bytecode written there).

    set CUDA_VISIBLE_DEVICES=-1 & python -B wave_bench.py [--passes 3]

Per pass k, the factory builds a fresh module and lets its address learn in-step style on the capture's fit thoughts
(the first training steps' xf; tests_wave_bench.train_address's protocol), then the Harness writes every stored fact
through bench_write (the real write path: a read first, then the policy's plan - count 1 after a confirmed recall, 2 and
the second entry on surprise -, confirm, write; content = the fact's fixed random vector; private parts: the
sentence's first captured expert + the shared part) and reads every query through read() with the gate bypassed
(bench_open: g = 1, gains 1; the learned gate needs the brain's loss). Variants (WAVE_BENCH_VARIANTS=0,1,2,3): the
v2 column store (wave_columns.py, the default) global only, + the private stores, the earlier hebb placeholder and the
pre-v2 wave sheet (wave_sheet.py) for comparison.
Forgetting does not run here (no training steps): the bench measures capacity / addressing, not survival. The
sheet's content codebooks are first warmed on random vectors at the content scale (never the facts' contents) and
put on their slow rate, as after a training run's warm-up; otherwise they move between the bench's write batches.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MEM26 = Path(r"C:\Eitan\Python\NewBrain\speed_wip\mem26")
sys.path.insert(0, str(HERE))
if str(MEM26) not in sys.path:
    sys.path.insert(1, str(MEM26))
from wave_memory import WAVE_DEFAULTS, CodedWaveMemory  # noqa: E402


def factory(bench, n_exp, epochs=8, batch=256, learn=False, **kw):
    """-> a no-argument factory for Harness.run (it builds one module per pass, in bench.passes order)."""
    passes = iter(bench.passes)

    def make():
        k = next(passes)
        c = dict(WAVE_DEFAULTS, wave_drop=0.0, wave_explore=0.0, wave_warm=1, wave_seed=k, **kw)
        xs = torch.from_numpy(bench.feats("fit", k, ["xf"]))
        n = len(xs)
        c.setdefault("wave_anneal", int(0.8 * epochs * ((n + batch - 1) // batch)))
        m = CodedWaveMemory(384, n_exp, **c).eval()
        g = np.random.default_rng(k)
        for _ in range(epochs):
            p = torch.from_numpy(g.permutation(n))
            for i in range(0, n, batch):
                m.address.update(xs[p[i:i + batch]])
        m.bench_open = True
        if hasattr(m.store, "update_books"):       # (the content codebooks as after the warm-up of a training run:
            g2 = torch.Generator().manual_seed(1000 + k)                 # random vectors at the content scale,
            scale = 0.18 * float(np.median(np.linalg.norm(bench.feats("fit", k, ["xf"]), axis=1)))  # not the facts)
            for _ in range(40):
                m.store.update_books(torch.nn.functional.normalize(torch.randn(2048, 384, generator=g2), dim=-1) * scale)
            m.store.n_cw = max(m.store.n_cw, m.store.fast)              # (then the slow rate, as in training)
        if learn:                                  # (the address keeps learning on the written thoughts)
            real = m.bench_write
            m.bench_write = lambda xf, target, e_k=None, t_k=None, call=None: real(xf, target, e_k, t_k, call, learn=True)
        return m
    return make


def run_harness(bench=None, passes=None):
    import wbench as W
    import wbench_module as WM
    b = bench or W.Bench(passes=passes or (3,))
    h = WM.Harness(b)
    n_exp = int(h.cue_e.max()) + 1
    out = {}
    variants = [("v2 columns (the default, global only)", dict(wave_priv_levels=0)),
                ("v2 columns + private hebb stores", dict(wave_priv_levels=1)),
                ("hebb placeholder (global only)", dict(wave_store="hebb", wave_priv_levels=0)),
                ("pre-v2 wave sheet (global only)", dict(wave_store="sheet", wave_priv_levels=0))]
    if os.environ.get("WAVE_BENCH_KW"):            # (one custom variant: "wave_side=8,wave_entries=3")
        kv = dict(p.split("=") for p in os.environ["WAVE_BENCH_KW"].split(","))
        variants = [(f"wave {os.environ['WAVE_BENCH_KW']}",
                     dict(wave_priv_levels=0, **{k: type(WAVE_DEFAULTS[k])(v) for k, v in kv.items()}))]
    elif os.environ.get("WAVE_BENCH_VARIANTS"):
        keep = {int(v) for v in os.environ["WAVE_BENCH_VARIANTS"].split(",")}
        variants = [v for i, v in enumerate(variants) if i in keep]
    for name, kw in variants:
        learn = kw.pop("learn", False)
        out[name] = h.run(name, factory(b, n_exp, learn=learn, **kw))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", default="3")
    a = ap.parse_args()
    run_harness(passes=tuple(int(x) for x in a.passes.split(",")))
