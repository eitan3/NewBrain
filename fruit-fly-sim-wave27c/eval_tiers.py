"""Validation loss of saved runs, overall and per difficulty tier (easy / medium / hard).

    python eval_tiers.py s0_ssd s0g2_grow_curr ...      (run names under ../runs)
    python eval_tiers.py --all --device cpu
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from spiking_lm import SpikingLM
from train_lm import Data, tier_losses

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--runs-dir", default=str(HERE.parent / "runs"))
    ap.add_argument("--data", default=str(HERE.parent / "data" / "ts4k"))
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    root = Path(a.runs_dir)
    names = sorted(p.parent.name for p in root.glob("*/final.pt")) if a.all else a.runs
    data = Data(Path(a.data), 512, a.device)
    print(f"{'run':24s} {'all':>7s} {'easy':>7s} {'medium':>7s} {'hard':>7s}", flush=True)
    for name in names:
        ck = torch.load(root / name / "final.pt", map_location=a.device)
        m = SpikingLM(**ck["cfg"]).to(a.device)
        missing, unexpected = m.load_state_dict(ck["state"], strict=False)
        assert not unexpected and all(k.endswith(("ema_rate", "norm_w", "ramp", "util", "debt", "load_ema", "age")) for k in missing), (missing, unexpected)
        m.eval()
        gen = np.random.RandomState(1234)
        with torch.no_grad():
            overall = np.mean([m(*data.batch(a.batch, "val", gen))[1].item() for _ in range(a.batches)])
        t = tier_losses(m, data, a.batches, a.batch, a.device)
        print(f"{name:24s} {overall:7.4f} " + " ".join(f"{t[f'val_t{k}']:7.4f}" for k in range(len(t))), flush=True)


if __name__ == "__main__":
    main()
