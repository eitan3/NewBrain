"""Capacity estimate from val curves (CPU):  python analysis/capacity_fit.py

Fits L(tokens) = a + b * tokens^-c to the PRE-DECAY part of each run's val curve (WSD: the learning rate is flat
until the last `decay` fraction, then the loss drops extra). a is the loss the run's setup would approach with
unlimited data at a flat learning rate (its capacity floor on this data); the decay bonus (final val minus the fitted
pre-decay curve at the final token count) is measured on each finished run and added back when extrapolating.
Uncertainty: refits with different start points (and the 3 best c values) give a range.
"""
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"
TARGETS = (400e6, 600e6, 1e9)
RUN_NAMES = {"v4b_split": "v4b_split (loop v4b)", "org2b_s0": "v2 grown", "org2b_full_s0": "v2 full size",
             "s0_attn": "transformer", "loop_carry": "loop_carry (first loop)"}


def load(run):
    recs = [json.loads(line) for line in (RUNS / run / "metrics.jsonl").open() if line.strip()]
    cfg = json.loads((RUNS / run / "config.json").read_text())
    pts = np.array([(r["tokens"], r["val"]) for r in recs if "val" in r and r["tokens"] > 0])
    return pts, cfg


def fit(t, v, cs=np.linspace(0.05, 1.5, 146)):
    best = []
    for c in cs:
        X = np.stack([np.ones_like(t), t ** -c], 1)
        coef, res, *_ = np.linalg.lstsq(X, v, rcond=None)
        err = ((X @ coef - v) ** 2).mean()
        if coef[1] > 0:
            best.append((err, c, coef[0], coef[1]))
    best.sort()
    return best[:3]


out = {}
for run, label in RUN_NAMES.items():
    if not (RUNS / run / "metrics.jsonl").exists():
        continue
    pts, cfg = load(run)
    total = float(cfg.get("tokens", pts[-1, 0]))
    decay = float(cfg.get("decay", 0.2))
    pre = pts[pts[:, 0] <= (1 - decay) * total]
    finished = pts[-1, 0] >= 0.99 * total
    preds, floors, bonus = [], [], []
    for start in (10e6, 20e6, 40e6):
        sel = pre[pre[:, 0] >= start]
        if len(sel) < 5:
            continue
        for err, c, a, b in fit(sel[:, 0] / 1e6, sel[:, 1]):
            curve = lambda tok: a + b * (tok / 1e6) ** -c
            bon = (pts[-1, 1] - curve(pts[-1, 0])) if finished else None
            floors.append(a)
            bonus.append(bon)
            preds.append([curve(tok) for tok in TARGETS])
    preds = np.array(preds)
    bon = [b for b in bonus if b is not None]
    out[run] = dict(label=label, final=float(pts[-1, 1]), final_tokens=float(pts[-1, 0]), finished=finished,
                    floor=(min(floors), max(floors)), decay_bonus=(min(bon), max(bon)) if bon else None,
                    pre_decay=[(preds[:, i].min(), preds[:, i].max()) for i in range(len(TARGETS))])

bon_all = [x for r in out.values() if r["decay_bonus"] for x in r["decay_bonus"]]
typ_bonus = float(np.median(bon_all)) if bon_all else 0.0
print(f"fit: L = a + b * (tokens/1M)^-c on the pre-decay val curve; decay bonus (final - fitted curve) median "
      f"{typ_bonus:+.3f} over finished runs\n")
print(f"{'run':26s} {'final':>14s} {'floor a':>15s} {'decay bonus':>15s}   " +
      "   ".join(f"{t/1e6:.0f}M (+decay)" for t in TARGETS))
for run, r in out.items():
    b = r["decay_bonus"]
    bo = b if b else (typ_bonus, typ_bonus)
    ext = "   ".join(f"{lo + bo[1]:.3f}-{hi + bo[0]:.3f}" for lo, hi in r["pre_decay"])
    print(f"{r['label']:26s} {r['final']:.4f} @{r['final_tokens']/1e6:4.0f}M {r['floor'][0]:6.3f}-{r['floor'][1]:6.3f} "
          f"{(f'{b[0]:+.3f}..{b[1]:+.3f}' if b else 'n/a (median)'):>15s}   {ext}")
Path(__file__).with_name("capacity_fit_results.json").write_text(json.dumps(out, indent=1, default=float))
