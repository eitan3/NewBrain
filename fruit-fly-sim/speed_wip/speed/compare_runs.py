"""Task 1: compare accum_run.py runs against a reference (A = 16x2) and the run-to-run floor (A2 = 16x2 again).

    python compare_runs.py A A2 B C D
"""
import json
import math
import sys

import numpy as np
import torch

import sp_common as sc

tags = sys.argv[1:] or ["A", "A2", "B", "C", "D"]
ref = tags[0]
J = {t: json.loads((sc.HERE / f"t1_{t}.json").read_text()) for t in tags}
F = {t: torch.load(sc.SCRATCH / f"t1_final_{t}.pt") for t in tags}
G = {t: torch.load(sc.SCRATCH / f"t1_grads_{t}.pt") for t in tags}
init_path = sc.SCRATCH / "t1_init_params.pt"
P0 = torch.load(init_path) if init_path.exists() else None
out = {}


def flat(d, keys):
    return torch.cat([d[k].flatten() for k in keys])


def js(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a.clamp(min=1e-12)) - torch.log(b.clamp(min=1e-12)))).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


pkeys = [k for k in F[ref] if not k.startswith(("mlp.", "mem.", "loads")) and not k.endswith("ema_rate")]
for t in tags:
    r = dict(batch=J[t]["batch"], accum=J[t]["accum"], fix=J[t]["fix"])
    s, s0 = J[t]["steps"], J[ref]["steps"]
    r["tok_same"] = all(a["tok_chk"] == b["tok_chk"] for a, b in zip(s, s0))
    r["K_same"] = all(a["K"] == b["K"] for a, b in zip(s, s0))
    r["loss_mean"] = float(np.mean([x["loss"] for x in s]))
    r["loss_last10"] = float(np.mean([x["loss"] for x in s[-10:]]))
    r["dloss_mean_abs"] = float(np.mean([abs(a["loss"] - b["loss"]) for a, b in zip(s, s0)]))
    r["dloss_step0"] = s[0]["loss"] - s0[0]["loss"]
    r["gn_mean"] = float(np.mean([x["grad_norm"] for x in s]))
    r["dgn_rel_mean_abs"] = float(np.mean([abs(a["grad_norm"] - b["grad_norm"]) / b["grad_norm"] for a, b in zip(s, s0)]))
    r["gn_step0"] = s[0]["grad_norm"]
    r["drop_mean"] = float(np.mean([x["drop"] for x in s]))
    # step-1 gradients (identical starting state): cosine and relative difference vs the reference
    gk = [k for k in G[ref] if k in G[t]]
    ga, gb = flat(G[ref], gk), flat(G[t], gk)
    r["grad1_cos"] = float(torch.dot(ga.double(), gb.double()) / (ga.double().norm() * gb.double().norm()))
    r["grad1_rel_diff"] = float((ga - gb).norm() / ga.norm())
    per = {}
    for grp, pre in (("experts", "blocks.0.mlp.w_"), ("router", "blocks.0.mlp.router"), ("mixer", "blocks.0.mix."),
                     ("memory", "mem."), ("lm_head", "heads."), ("alpha", None)):
        ks = [k for k in gk if (k.startswith(pre) if pre else k.endswith("alpha"))]
        if ks:
            a_, b_ = flat(G[ref], ks), flat(G[t], ks)
            per[grp] = round(float((a_ - b_).norm() / a_.norm().clamp(min=1e-30)), 4)
    r["grad1_rel_diff_by_group"] = per
    # final parameters after the run: distance to the reference, relative to the reference's own update
    pa, pb = flat(F[ref], pkeys), flat(F[t], pkeys)
    r["param_dist"] = float((pa - pb).norm())
    if P0 is not None:
        p0 = flat(P0, pkeys)
        r["param_update_norm"] = float((pb - p0).norm())
        r["param_dist_over_update"] = float((pa - pb).norm() / (pa - p0).norm())
    r["bias_l1"] = float((F[ref]["mlp.bias"] - F[t]["mlp.bias"]).abs().mean())
    r["load_ema_l1"] = float((F[ref]["mlp.load_ema"] - F[t]["mlp.load_ema"]).abs().mean())
    r["norm_m_rel"] = float((F[ref]["mem.norm_m"] - F[t]["mem.norm_m"]).norm() / F[ref]["mem.norm_m"].norm())
    r["V_rel"] = float((F[ref]["mem.V_sample"] - F[t]["mem.V_sample"]).norm() / F[ref]["mem.V_sample"].norm())
    ek = [k for k in F[t] if k.endswith("ema_rate")]
    r["ema_rate_l1"] = {k: float((F[ref][k] - F[t][k]).abs().mean()) for k in ek}
    La, Lb = F[ref]["loads"].double(), F[t]["loads"].double()
    r["load_js_mean"] = float(np.mean([js(La[i], Lb[i]) for i in range(La.size(0))]))
    r["load_js_step0"] = js(La[0], Lb[0])
    p = Lb / Lb.sum(1, keepdim=True)
    ent = (-(p * p.clamp(min=1e-12).log()).sum(1) / math.log(p.size(1))).mean()
    r["load_entropy"] = float(ent)
    r["load_max_over_mean"] = float((Lb.max(1).values / Lb.mean(1)).mean())
    r["load_dead_frac"] = float((Lb < 0.1 * Lb.mean(1, keepdim=True)).double().mean())
    out[t] = r

(sc.HERE / "t1_compare.json").write_text(json.dumps(out, indent=1))
cols = ["loss_mean", "loss_last10", "dloss_mean_abs", "dloss_step0", "gn_step0", "gn_mean", "dgn_rel_mean_abs",
        "drop_mean", "grad1_cos", "grad1_rel_diff", "param_dist_over_update", "bias_l1", "load_js_step0",
        "load_js_mean", "load_entropy", "load_max_over_mean", "norm_m_rel", "V_rel"]
print("tag  " + " ".join(f"{c[:14]:>14s}" for c in cols))
for t, r in out.items():
    print(f"{t:4s} " + " ".join(f"{r.get(c, float('nan')):14.5g}" for c in cols))
for t, r in out.items():
    print(t, "tok_same", r["tok_same"], "K_same", r["K_same"], "grad1 by group", r["grad1_rel_diff_by_group"])
