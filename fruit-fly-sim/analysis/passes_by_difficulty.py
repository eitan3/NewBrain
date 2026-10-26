"""Do harder words / stories gain from more passes?   python analysis/passes_by_difficulty.py [run] [batches]  (CPU)

One K=16 evaluation run per batch of the easy / medium / hard validation tiers; the decoder reads every pass.
Reports the loss after each pass per tier, for the hardest 10% of words (by their loss at pass 6), the share of
words whose best pass is after pass 6, and their mean gain from pass 6 to their best pass.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from loop_brain import LoopBrain  # noqa: E402
from train_lm import Data  # noqa: E402

torch.set_num_threads(6)
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "runs" / "v4c_all"
BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 2
B, KMAX = 4, 16
ck = torch.load(RUN / "final.pt", map_location="cpu", weights_only=False)
m = LoopBrain(**ck["cfg"])
m.load_state_dict(ck["state"])
m.eval()
data = Data(HERE.parent.parent / "data" / "ts4k", 512, "cpu")

per_tier = {}
with torch.no_grad():
    for tier, name in enumerate(("easy", "medium", "hard")):
        gen = np.random.RandomState(77)
        ces = []
        for _ in range(BATCHES):
            x, y = data.tier_batch(tier, B, gen)
            outs = m.run(x, KMAX, every_tick=True)
            ces.append(torch.stack([F.cross_entropy(m.heads["lm"](o).float().flatten(0, 1), y.flatten(),
                                                    reduction="none") for o in outs], 1))   # (words, K)
        per_tier[name] = torch.cat(ces, 0)
        print(f"{name} done", flush=True)

allw = torch.cat(list(per_tier.values()), 0)
k6 = allw[:, 5]
hard10 = allw[k6 >= k6.quantile(0.9)]
groups = dict(per_tier, all=allw, hardest_10pct_words=hard10)
res = {}
for name, ce in groups.items():
    curve = ce.mean(0)
    best = ce.argmin(1) + 1
    late = best > 6
    gain = (ce[:, 5] - ce.min(1).values)[late].mean().item() if late.any() else 0.0
    res[name] = dict(words=int(ce.size(0)), curve=[round(v, 4) for v in curve.tolist()],
                     best_K=int(curve.argmin()) + 1, gain_6_to_16=round((curve[5] - curve[15]).item(), 4),
                     gain_6_to_best_curve=round((curve[5] - curve.min()).item(), 4),
                     share_best_pass_after_6=round(late.float().mean().item(), 3), mean_gain_of_those=round(gain, 4))
out = RUN / "passes_by_difficulty.json"
out.write_text(json.dumps(res, indent=1))
for name, r in res.items():
    c = r["curve"]
    print(f"{name:20s} n={r['words']:6d}  K1 {c[0]:.3f} K2 {c[1]:.3f} K4 {c[3]:.3f} K6 {c[5]:.3f} K8 {c[7]:.3f} "
          f"K12 {c[11]:.3f} K16 {c[15]:.3f} | best K {r['best_K']:2d} (K6->best {r['gain_6_to_best_curve']:+.4f}) | "
          f"best pass >6: {r['share_best_pass_after_6']:.2f} of words, their gain {r['mean_gain_of_those']:.3f}")
