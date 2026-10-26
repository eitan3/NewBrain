"""Who gets traffic in a routed brain? Per layer: load of the original experts vs newborns (by birth
order), their balancer biases and remaining ramp debt.     python diag_experts.py ../runs/org2_s0"""
import sys
from pathlib import Path

import numpy as np
import torch

from spiking_lm import RoutedMLP, SpikingLM
from train_lm import Data

run = Path(sys.argv[1])
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(run / "final.pt" if (run / "final.pt").exists() else run / "last.pt", map_location=dev)
print("checkpoint step", ck.get("step", "final"))
m = SpikingLM(**ck["cfg"]).to(dev)
print("missing:", m.load_state_dict(ck["state"], strict=False).missing_keys[:5])
m.eval()
data = Data(Path(__file__).resolve().parent.parent / "data" / "ts4k", 512, dev)
gen = np.random.RandomState(0)
loads = None
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
    for _ in range(8):
        x, y = data.batch(32, "val", gen)
        m(x, y)
        c = torch.stack([b.mlp.stats[0].float() for b in m.blocks if isinstance(b.mlp, RoutedMLP)])
        loads = c if loads is None else loads + c
E0 = 32
for i, b in enumerate(m.blocks):
    mlp = b.mlp
    load = loads[i] / loads[i].mean()
    bias, debt = mlp.bias.float(), mlp.debt.float()
    old, new = load[:E0], load[E0:]
    print(f"layer {i}: E={mlp.E}  dead(<0.1x mean): original {(old < 0.1).sum().item()}/{len(old)}, "
          f"newborn {(new < 0.1).sum().item()}/{len(new)}")
    print(f"   load  original mean {old.mean():.2f} | newborn by birth order (first->last): "
          + " ".join(f"{v:.2f}" for v in new.tolist()))
    print(f"   bias  original mean {bias[:E0].mean():+.2f} (min {bias[:E0].min():+.2f}, max {bias[:E0].max():+.2f}) | "
          f"newborn mean {bias[E0:].mean():+.2f} (min {bias[E0:].min():+.2f}); debt left {debt.sum():.2f}")
    # how similar are newborn router rows to their nearest original row?
    W = torch.nn.functional.normalize(mlp.router.weight.float(), dim=1)
    sim = (W[E0:] @ W[:E0].T).max(1).values
    print(f"   newborn router-row cosine to nearest original: mean {sim.mean():.3f}")
