import json, sys, numpy as np
R = json.load(open(sys.argv[1]))
cols = ["dist", "dist7", "top1", "dead", "score", "xdoc", "lag1", "lag16", "lag256", "stab_n03", "stab_n10", "stab_drift"]
rows = []
for k, v in R.items():
    p, norm, hv, kv = k.split("|")
    rows.append((norm, hv, kv, {c: float(np.mean([x[c] for x in v])) for c in cols}))
print("| norm | halves | keys | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 3) + "|")
for norm, hv, kv, d in rows:
    print(f"| {norm} | {hv} | {kv} | " + " | ".join(f"{d[c]:.0f}" if c.startswith("dist") else f"{d[c]:.3f}" for c in cols) + " |")
