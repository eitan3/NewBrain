"""Table of the dead-expert ablations:  python analysis/ablation_table.py [prefix=abl_]"""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"
prefix = sys.argv[1] if len(sys.argv) > 1 else "abl_"
rows = []
for run in sorted(RUNS.glob(prefix + "*")):
    f = run / "metrics.jsonl"
    if not f.exists():
        continue
    recs = [json.loads(line) for line in f.open() if line.strip()]
    if not recs:
        continue
    last = recs[-1]
    traj = [(r["step"], r["routing"][0]["dead"], r["routing"][0]["drop"]) for r in recs if "routing" in r]
    worst_dead = max((d for _, d, _ in traj), default=float("nan"))
    worst_drop = max((d for _, _, d in traj), default=float("nan"))
    vals = [(r["step"], r["tokens"], r["val"], r.get("train")) for r in recs if "val" in r]
    steps_total = vals[-1][0]
    tail = [v for v in vals if v[0] >= 0.75 * steps_total]
    slope = None
    if len(tail) >= 2:  # least-squares val slope over the last 25% (nats per 10M tokens; includes the LR decay)
        t0 = sum(v[1] for v in tail) / len(tail)
        v0 = sum(v[2] for v in tail) / len(tail)
        slope = sum((v[1] - t0) * (v[2] - v0) for v in tail) / max(1e-9, sum((v[1] - t0) ** 2 for v in tail)) * 1e7
    live = None
    ck = run / "final.pt"
    if ck.exists():
        import torch
        st = torch.load(ck, map_location="cpu", weights_only=False)["state"]
        le = st.get("blocks.0.mlp.load_ema")
        live = None if le is None else int((le > 0.2).sum())
    rt = last.get("routing", [{}])[0]
    nb = last.get("newborn", {})
    paid = [v for k, v in nb.items() if k in ("1k-2k", "2k+")]
    rows.append(dict(run=run.name[len(prefix):], step=last["step"], val=last["val"], experts=last.get("moe", {}).get("n_exp"),
                     dead=rt.get("dead"), drop=rt.get("drop"), maxload=rt.get("maxload"), worst_dead=worst_dead,
                     worst_drop=worst_drop,
                     nb_young=nb.get("0-500", {}).get("load"), nb_mid=nb.get("500-1k", {}).get("load"),
                     nb_paid_load=(sum(v["load"] * v["n"] for v in paid) / max(1, sum(v["n"] for v in paid))) if paid else None,
                     nb_paid_dead=(sum(v["dead"] * v["n"] for v in paid) / max(1, sum(v["n"] for v in paid))) if paid else None,
                     K=last.get("loss_at_K"), vals=vals, slope=slope, live=live,
                     gen_gap=(last["val"] - last["train"]) if last.get("train") is not None else None))


ctl = next((r for r in rows if r["run"].startswith("ctl")), None)
TOTAL = ctl["vals"][-1][0] if ctl else max(r["vals"][-1][0] for r in rows)


def val_at(r, frac):  # val interpolated at the same absolute step (frac of the full schedule); None if not reached
    target = frac * TOTAL
    pts = r["vals"]
    for (s0, _, v0, _), (s1, _, v1, _) in zip(pts, pts[1:]):
        if s0 <= target <= s1:
            return v0 + (v1 - v0) * (target - s0) / max(1, s1 - s0)
    return pts[-1][2] if pts[-1][0] == target else None


def fmt(v, p=3):
    return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


hdr = ["run", "step", "experts", "val", "dead", "drop", "max/mean", "worst dead", "worst drop", "newborn load: <500",
       "500-1k", ">=1k", "dead >=1k"]
print(" | ".join(hdr))
for r in rows:
    print(" | ".join([r["run"], str(r["step"]), fmt(r["experts"]), fmt(r["val"], 4), fmt(r["dead"], 2), fmt(r["drop"], 3),
                      fmt(r["maxload"], 1), fmt(r["worst_dead"], 2), fmt(r["worst_drop"], 3), fmt(r["nb_young"], 2),
                      fmt(r["nb_mid"], 2), fmt(r["nb_paid_load"], 2), fmt(r["nb_paid_dead"], 2)]))
for r in rows:
    if r["K"]:
        print(f"{r['run']}: loss by passes " + " ".join(f"K{k}:{v:.3f}" for k, v in r["K"].items()))

print("\nlonger-term view: val slope over the last 25% (nats / 10M tokens, includes LR decay), gap to the control "
      "at 50/75/100% (+ = worse), val - train, live experts (load > 0.2x average)")
for r in rows:
    gaps = [None if (ctl is None or val_at(r, f) is None) else val_at(r, f) - val_at(ctl, f) for f in (0.5, 0.75, 1.0)]
    closing = None not in gaps and gaps[0] > 0 and gaps[2] < gaps[0] - 0.005
    print(f"  {r['run']:10s} slope {fmt(r['slope'], 4)}  gap50 {fmt(gaps[0], 4)}  gap75 {fmt(gaps[1], 4)}  "
          f"gap100 {fmt(gaps[2], 4)}  val-train {fmt(r['gen_gap'], 4)}  live {fmt(r['live'])}/{fmt(r['experts'])}"
          + ("  <- behind but closing" if closing else ""))
