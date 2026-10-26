"""Is the brain really using its next-concept plan?   python ablate_concepts.py ../runs/org3_ncm_s0

Validation loss (overall and easy/medium/hard) of a concept-planning model in three modes:
  own plan  - normal: the brain's own predicted concept of the next chunk (what training used)
  no plan   - the top-down plan switched off
  oracle    - the plan built from the TRUE next-chunk concept codes (cheating; an upper bound on how
              much a perfect plan could help)
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from spiking_lm import SpikingLM
from train_lm import Data

run = Path(sys.argv[1])
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(run / "final.pt", map_location=dev)
m = SpikingLM(**ck["cfg"]).to(dev)
m.load_state_dict(ck["state"], strict=False)
m.eval()
data = Data(Path(__file__).resolve().parent.parent / "data" / "ts4k", 512, dev)
orig_plan = m._plan


def oracle_plan(x, idx):
    C, (B, T, D) = m.concept_tok.chunk, x.shape
    n = T // C
    with torch.no_grad():
        codes = m.concept_tok.codes(idx.reshape(-1, C), m.concept_head.K)
        rec = m.concept_tok.recon_from_codes(codes).view(B, n, C, -1)
    plan = m.plan_proj(rec.to(x.dtype))                    # chunk j gets the true concept of chunk j
    return plan.reshape(B, T, D)


def no_plan(x, idx):
    return torch.zeros_like(x)


def score(batches=10):
    out = {}
    for name, split in (("all", None), ("easy", 0), ("medium", 1), ("hard", 2)):
        gen = np.random.RandomState(99)
        tot = 0.0
        for _ in range(batches):
            x, y = data.batch(32, "val", gen) if split is None else data.tier_batch(split, 32, gen)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                tot += m(x, y)[1].item()
        out[name] = tot / batches
    return out


for label, fn in (("own plan", orig_plan), ("no plan", no_plan), ("oracle plan", oracle_plan)):
    m._plan = fn
    s = score()
    print(f"{label:12s} " + "  ".join(f"{k} {v:.4f}" for k, v in s.items()), flush=True)
