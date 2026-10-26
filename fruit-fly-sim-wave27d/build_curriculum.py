"""Split tokenised TinyStories into difficulty buckets for a simple-to-hard curriculum.

Difficulty of a story = z(log length in tokens) + z(mean token rarity), rarity = -log(token frequency).
Writes curr_0.bin (easiest third) ... curr_2.bin (hardest third) next to train.bin, and
val_curr_0.bin ... from val.bin using the SAME statistics and cut points, so validation can be
reported per difficulty tier.

    python build_curriculum.py --data ../data/ts4k --buckets 3
    python build_curriculum.py --val-only      (only the validation tiers; leaves curr_*.bin alone)
"""
import argparse
from pathlib import Path

import numpy as np


def story_stats(tok, rarity):
    ends = np.flatnonzero(tok == 0)                       # <|endoftext|> closes every story
    starts = np.concatenate([[0], ends[:-1] + 1])
    lengths = ends - starts + 1
    mean_rarity = np.add.reduceat(rarity[tok], starts) / lengths
    return starts, ends, lengths, mean_rarity


def write_buckets(tok, starts, ends, assign, n, prefix, path, lengths, mean_rarity):
    for b in range(n):
        idx = np.flatnonzero(assign == b)  # corpus order inside a bucket
        out = np.concatenate([tok[s:e + 1] for s, e in zip(starts[idx], ends[idx])])
        out.tofile(path / f"{prefix}{b}.bin")
        print(f"{prefix}{b}: {len(idx):,} stories, {len(out)/1e6:.1f}M tokens, "
              f"median length {np.median(lengths[idx]):.0f}, mean rarity {mean_rarity[idx].mean():.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data" / "ts4k"))
    ap.add_argument("--buckets", type=int, default=3)
    ap.add_argument("--val-only", action="store_true")
    a = ap.parse_args()
    path = Path(a.data)
    tok = np.fromfile(path / "train.bin", dtype=np.uint16)
    freq = np.bincount(tok, minlength=4096).astype(np.float64)
    rarity = (-np.log((freq + 1) / freq.sum())).astype(np.float32)
    starts, ends, lengths, mean_rarity = story_stats(tok, rarity)
    ll = np.log(lengths)
    mu = (ll.mean(), ll.std(), mean_rarity.mean(), mean_rarity.std())
    score_of = lambda L, R: (np.log(L) - mu[0]) / mu[1] + (R - mu[2]) / mu[3]
    score = score_of(lengths, mean_rarity)
    cuts = np.quantile(score, np.linspace(0, 1, a.buckets + 1)[1:-1])
    if not a.val_only:
        write_buckets(tok, starts, ends, np.searchsorted(cuts, score), a.buckets, "curr_", path, lengths, mean_rarity)
    vt = np.fromfile(path / "val.bin", dtype=np.uint16)
    vs, ve, vl, vr = story_stats(vt, rarity)
    write_buckets(vt, vs, ve, np.searchsorted(cuts, score_of(vl, vr)), a.buckets, "val_curr_", path, vl, vr)


if __name__ == "__main__":
    main()
