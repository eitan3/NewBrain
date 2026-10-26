"""Why do newborn experts die in v4 (aux balance mode)?   python analysis/dead_experts_probe.py [ckpt]   (CPU)

1. Per newborn age bin: router row norm, raw router score, bias / debt, load_ema.
2. Which loss terms send gradient to which router rows: LM (+ deep supervision), router z-loss, aux balance
   loss - for old experts, newborns still ramping (debt > 0) and newborns that finished ramping.
   BrainOpt is lazy per ROW: a row with any nonzero gradient gets an AdamW step AND weight decay; Adam
   normalises the step, so even a tiny but consistent gradient moves a row at almost the full learning rate.
3. Logit scale: are all router logits pushed far below 0 (sigmoid scores ~0.005)?
"""
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
CK = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "runs" / "v4_grow_fair" / "step5000.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
m = LoopBrain(**ck["cfg"])
m.load_state_dict(ck["state"])
core, mlp = m.blocks[0], m.blocks[0].mlp
E0 = 192
E = mlp.E
age, debt, bias, le = mlp.age.float(), mlp.debt, mlp.bias, mlp.load_ema
W = mlp.router.weight.data
data = Data(HERE.parent.parent / "data" / "ts4k", 512, "cpu")
x, y = data.batch(2, "val", np.random.RandomState(3))
print(f"checkpoint step {ck['step']}: {E} experts ({E - E0} born), balance={mlp.balance} z_w={mlp.z_w} aux_w={mlp.aux_w}")

# ---------------------------------------------------------------- 1. newborns by age
scores = []
orig = mlp.route


def grab(xf, side, extra, wave):
    out = orig(xf, side, extra, wave)
    scores.append(out[0].detach())
    return out


mlp.route = grab
m.eval()
m.K = 6
with torch.no_grad():
    m(x, y)
mlp.route = orig
S = torch.stack(scores).mean(0)                     # (tokens, E) raw sigmoid score, averaged over passes
thr = torch.stack([(s + bias).topk(mlp.k, -1).values[:, -1] for s in scores]).median()
print(f"\n6th-pick threshold (score + bias), median: {thr:.4f}")
print(f"old experts: row norm {W[:E0].norm(dim=1).median():.3f}  raw score mean {S[:, :E0].mean():.4f}  "
      f"mean logit {torch.logit(S[:, :E0].clamp(1e-6, 1 - 1e-6)).mean():.2f}  load_ema median {le[:E0].median():.2f}")
print("newborns by age (steps since birth):")
for lo, hi in ((0, 100), (100, 250), (250, 500), (500, 750), (750, 1000), (1000, 1500), (1500, 2500), (2500, 5000)):
    s = (age >= lo) & (age < hi)
    s[:E0] = False
    if s.sum() == 0:
        continue
    print(f"  age {lo:4d}-{hi:4d} n={int(s.sum()):3d}: row norm {W[s].norm(dim=1).median():.3f}  raw score mean "
          f"{S[:, s].mean():.4f} (max/expert median {S[:, s].max(0).values.median():.3f})  bias {bias[s].median():+.2f}  "
          f"debt {debt[s].median():.2f}  load_ema {le[s].median():.3f}  dead {(le[s] < 0.05).float().mean():.2f}")

# ---------------------------------------------------------------- 2. gradient sources on router rows
m.train()
m.K, m.aux_k_fixed, m.group_offset = 6, 4, 0
groups = {"old": torch.arange(E) < E0, "ramping": (torch.arange(E) >= E0) & (debt > 0),
          "paid": (torch.arange(E) >= E0) & (debt == 0)}


def router_grad(kind):
    z_w, aux_w = mlp.z_w, mlp.aux_w
    if kind == "z":
        mlp.aux_w = 0.0
    if kind == "aux":
        mlp.z_w = 0.0
    m.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    _, loss = m(x, y)
    if kind == "lm":
        tot = loss + 0.2 * F.cross_entropy(m.heads["lm"](m.aux_x).float().flatten(0, 1), y.flatten())
    else:
        tot = m.aux_loss()
    tot.backward()
    mlp.z_w, mlp.aux_w = z_w, aux_w
    return mlp.router.weight.grad.clone()


G = {k: router_grad(k) for k in ("lm", "z", "aux")}
print("\nrouter-row gradient norm (median per row) by source; 'nonzero' = share of rows with any gradient")
for gname, sel in groups.items():
    print(f"  {gname:8s} (n={int(sel.sum()):3d}): " + "  ".join(
        f"{k} {G[k][sel].norm(dim=1).median():.2e} nonzero {(G[k][sel].abs().sum(1) > 0).float().mean():.2f}"
        for k in G))
# direction: does the z-loss push the newborn's logits down (gradient aligned with the mean router input)?
xin = []
orig2 = mlp.route
mlp.route = lambda xf, side, extra, wave: (xin.append(xf.detach()), orig2(xf, side, extra, wave))[1]
with torch.no_grad():
    m(x, y)
mlp.route = orig2
xm = torch.cat(xin).mean(0)
for gname, sel in groups.items():
    g = G["z"][sel]
    # descent step = -g; its effect on the logit of the average input: -g . mean_x  (negative = pushed down)
    print(f"  z-loss descent step on {gname} rows changes the mean logit by {(-(g @ xm)).median():+.2e} per unit lr "
          f"(sign-consistent across rows: {((-(g @ xm)) < 0).float().mean():.2f} pushed down)")
mlp.route = orig
