"""Does the looped brain know when it is unsure, and would "loop until confident" help?  (CPU, no training)

    python analysis/uncertainty_probe.py [run_dir] [batches] [batch_size]

Part 1 - signals. One K=16 evaluation run; after every pass k the decoder reads x (normf + lm head, as deep
supervision does). Per token and pass:
  entropy   H(p_k) of the next-token prediction
  kl        KL(p_k || p_{k-1}): how much the prediction changed on this pass (pass 1: vs the uniform prior)
  gap       6th minus 7th best router selection score (small = undecided which expert to run)
  rent      entropy of the normalised router scores (/ log E)
  loss      cross-entropy of the true next token
How well does each signal at pass k predict the gain from more passes?
  gain16 = loss_k - loss_16,  gainmax = loss_k - min over later passes;  label: gainmax > 0.05 nats.
(Readouts of a K=16 run equal fixed-K runs: passes are causal in k and `last` only matters for vector state.)

Part 2 - exit rules, no retraining. A word stops after the first pass k >= kmin whose signal is below tau;
a stopped word keeps its x (and workspace) from then on, while the others keep looping (up to 16); the
frozen words still take part in the membrane scan with their frozen inputs. Mean loss vs mean passes per
word, against the fixed-K curve. Also the cheap "post-hoc" version (loss of pass k_exit from Part 1).
"""
import json
import math
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
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "runs" / "v3c_waves_ws2"
BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
KMAX = 16
OUT = HERE / f"uncertainty_{RUN.name}"
OUT.mkdir(exist_ok=True)

ck = torch.load(RUN / "final.pt", map_location="cpu", weights_only=False)
m = LoopBrain(**ck["cfg"])
m.load_state_dict(ck["state"], strict=False)
m.eval()
core, mlp = m.blocks[0], m.blocks[0].mlp
data = Data(HERE.parent.parent / "data" / "ts4k", 512, "cpu")
gen = np.random.RandomState(1234)
batches = [data.batch(BS, "val", gen) for _ in range(BATCHES)]


def readout(x):
    return F.log_softmax(m.heads["lm"](m.normf(x)).float(), -1)


def rank(a):
    r = np.empty(len(a))
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def auc(score, label):  # P(score of a positive > score of a negative), Mann-Whitney
    pos, n = label.sum(), len(label)
    if pos == 0 or pos == n:
        return float("nan")
    return float((rank(score)[label].sum() - pos * (pos - 1) / 2) / (pos * (n - pos)))


# ------------------------------------------------------------------ Part 1: signals per token and pass
route_log = []
_route = mlp.route


def route_hook(xf, side, extra, wave):
    out = _route(xf, side, extra, wave)
    score = out[0].float()
    top = (score + mlp.bias).topk(mlp.k + 1, -1).values
    p = score / score.sum(1, keepdim=True)
    route_log.append(((top[:, mlp.k - 1] - top[:, mlp.k]).numpy(),
                      (-(p * p.clamp_min(1e-12).log()).sum(1) / math.log(mlp.E)).numpy()))
    return out


mlp.route = route_hook
sig = {k: [] for k in ("entropy", "kl", "gap", "rent", "loss")}
with torch.no_grad():
    for x, y in batches:
        route_log.clear()
        outs = m.run(x, KMAX, every_tick=True)
        prev = torch.full((x.numel(), m.heads["lm"].out_features), -math.log(m.heads["lm"].out_features))
        rows = {k: [] for k in sig}
        for k, xo in enumerate(outs):  # run(every_tick) already returns normf(x): only the lm head here
            lp = F.log_softmax(m.heads["lm"](xo).float(), -1).flatten(0, 1)
            pk = lp.exp()
            rows["entropy"].append((-(pk * lp).sum(1)).numpy())
            rows["kl"].append((pk * (lp - prev)).sum(1).numpy())
            rows["loss"].append(F.nll_loss(lp, y.flatten(), reduction="none").numpy())
            rows["gap"].append(route_log[k][0])
            rows["rent"].append(route_log[k][1])
            prev = lp
        for k in sig:
            sig[k].append(np.stack(rows[k], 1))          # (tokens, passes)
mlp.route = _route
sig = {k: np.concatenate(v, 0) for k, v in sig.items()}
L = sig["loss"]
N = L.shape[0]
fixed = L.mean(0)                                         # fixed-K curve, K = 1..16

# direction: larger value = "more unsure"
unsure = {"entropy": sig["entropy"], "kl": sig["kl"], "neg_gap": -sig["gap"], "rent": sig["rent"]}
pred = {}
for k in range(KMAX - 1):
    gain16 = L[:, k] - L[:, -1]
    gainmax = L[:, k] - L[:, k + 1:].min(1)
    lab = gainmax > 0.05
    pred[k + 1] = {"helped_share": float(lab.mean()), "mean_gain16": float(gain16.mean())}
    for name, v in unsure.items():
        pred[k + 1][name] = dict(rho_gain16=spearman(v[:, k], gain16), rho_gainmax=spearman(v[:, k], gainmax),
                                 auc=auc(v[:, k], lab))

# ------------------------------------------------------------------ Part 2: exit rules
def posthoc(signal, tau, kmin=2):
    """Exit after the first pass k >= kmin with signal < tau; loss = that pass's loss from the K=16 run."""
    below = signal < tau
    below[:, :kmin - 1] = False
    below[:, -1] = True
    k_exit = below.argmax(1)
    return float(L[np.arange(N), k_exit].mean()), float((k_exit + 1).mean())


def frozen(rule, tau, kmin=2):
    """True simulation: stopped words keep their x / workspace; the rest keep looping with them in the scan."""
    _fwd = core.forward
    tot, passes, n = 0.0, 0.0, 0
    for x, y in batches:
        st = dict(done=None, k=0, prev=None, used=None)

        def fwd(xin, s, *a, **kw):
            xn, sn = _fwd(xin, s, *a, **kw)
            st["k"] += 1
            if st["done"] is not None:
                d = st["done"][..., None]
                xn = torch.where(d, xin, xn)
                if sn is not None:
                    sn = torch.where(d.view(*d.shape[:2], *([1] * (sn.dim() - 2))), s, sn)
            lp = readout(xn)
            if st["prev"] is None:
                st["done"] = torch.zeros(lp.shape[:2], dtype=torch.bool)
                st["used"] = torch.ones(lp.shape[:2])
            else:
                st["used"] = st["used"] + (~st["done"]).float()  # this pass was computed for the words still looping
                v = (lp.exp() * (lp - st["prev"])).sum(-1) if rule == "kl" else -(lp.exp() * lp).sum(-1)
                if st["k"] >= kmin:
                    st["done"] = st["done"] | (v < tau)
            st["prev"] = lp
            return xn, sn

        core.forward = fwd
        with torch.no_grad():
            xo = m.run(x, KMAX)
        core.forward = _fwd
        tot += F.cross_entropy(m.heads["lm"](xo).float().flatten(0, 1), y.flatten(), reduction="sum").item()
        passes += st["used"].sum().item()
        n += y.numel()
    return tot / n, passes / n


# sanity: never exiting must reproduce fixed K=16, always exiting at pass 2 must reproduce fixed K=2
sanity = dict(never=frozen("kl", -1.0), always_at_2=frozen("kl", float("inf")),
              fixed_K16=float(fixed[-1]), fixed_K2=float(fixed[1]))
print("sanity:", sanity, flush=True)
exits = {"posthoc": {}, "frozen": {}}
for rule, taus in (("kl", (0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002)), ("entropy", (2.5, 2.0, 1.5, 1.0, 0.5, 0.25))):
    s = sig["kl"] if rule == "kl" else sig["entropy"]
    exits["posthoc"][rule] = [dict(tau=t, loss=l, passes=p) for t in taus for l, p in [posthoc(s, t)]]
    exits["frozen"][rule] = [dict(tau=t, loss=l, passes=p) for t in taus for l, p in [frozen(rule, t)]]
    print(f"done {rule}", flush=True)


def fixed_at(p):  # fixed-K loss interpolated at a (fractional) number of passes
    return float(np.interp(p, np.arange(1, KMAX + 1), fixed))


for kind in exits.values():
    for rows in kind.values():
        for r in rows:
            r["fixed_at_same_passes"] = fixed_at(r["passes"])
            r["better_than_fixed_by"] = r["fixed_at_same_passes"] - r["loss"]

res = dict(run=str(RUN), tokens=int(N), fixed_K=[float(v) for v in fixed], signals_by_pass=pred, exits=exits,
           sanity=sanity,
           best_fixed=dict(K=int(fixed.argmin() + 1), loss=float(fixed.min())))
(OUT / "results.json").write_text(json.dumps(res, indent=1))

# ------------------------------------------------------------------ text summary
lines = [f"run {RUN.name}, {N} tokens, K up to {KMAX}",
         "fixed K loss: " + " ".join(f"K{k + 1}:{v:.3f}" for k, v in enumerate(fixed)),
         "", "how well 'unsure at pass k' predicts gain from more passes (Spearman rho vs loss_k - loss_16; "
             "AUC for gain > 0.05 nats)"]
for k in (1, 2, 3, 4, 6, 8):
    p = pred[k]
    lines.append(f" pass {k:2d} (helped {p['helped_share']:.2f}, mean gain to K16 {p['mean_gain16']:+.3f}): " + "  ".join(
        f"{n} rho {p[n]['rho_gain16']:+.2f} auc {p[n]['auc']:.2f}" for n in unsure))
for kind, d in exits.items():
    lines.append(f"\nexit rule ({kind}): mean loss / mean passes per word / fixed-K loss at the same passes")
    for rule, rows in d.items():
        for r in rows:
            lines.append(f"  {rule:7s} tau {r['tau']:<6} loss {r['loss']:.4f}  passes {r['passes']:.2f}  "
                         f"fixed {r['fixed_at_same_passes']:.4f}  ({r['better_than_fixed_by']:+.4f} vs fixed)")
(OUT / "summary.txt").write_text("\n".join(lines))
print("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, KMAX + 1), fixed, "k-o", ms=3, label="fixed K")
    for kind, ls in (("frozen", "-"), ("posthoc", ":")):
        for rule, c in (("kl", "C0"), ("entropy", "C1")):
            rows = exits[kind][rule]
            ax.plot([r["passes"] for r in rows], [r["loss"] for r in rows], ls, color=c, marker="s", ms=3,
                    label=f"exit on {rule} ({kind})")
    ax.set_xlabel("mean passes per word")
    ax.set_ylabel("val loss")
    ax.set_title(f"Loop until confident vs fixed passes ({RUN.name})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "exit_vs_fixed.png", dpi=130)
except ImportError:
    pass
