import json, sys, numpy as np
cols = ["dist", "dist7", "top1", "dead", "score", "xdoc", "lag1", "lag16", "lag256", "stab_n03", "stab_n10", "stab_drift"]
rows = []
for f in sys.argv[1:]:
    R = json.load(open(f))
    for k, v in R.items():
        p, a, b, kv = k.split("|")
        rows.append((f.split(".")[0], a, b, kv, {c: float(np.mean([x[c] for x in v])) for c in cols}, sorted({x["level"] for x in v})))
print("| src | input | query/halves | keys | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 4) + "|")
for src, a, b, kv, d, lv in rows:
    print(f"| {src} L{','.join(map(str, lv))} | {a} | {b} | {kv} | " + " | ".join(f"{d[c]:.0f}" if c.startswith("dist") else f"{d[c]:.3f}" for c in cols) + " |")
