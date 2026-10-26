import json, numpy as np
cols = ["dist", "dist7", "top1", "dead", "xdoc", "lag1", "lag16", "lag256", "stab_n03", "stab_n10", "stab_drift"]
rows = {}
def add(f, tag, levels):
    for k, v in json.load(open(f)).items():
        _, a, b, kv = k.split("|")
        w = [x for x in v if x["level"] in levels]
        name = f"{a} / {b} / {kv}"
        rows[name + tag] = {c: float(np.mean([x[c] for x in w])) for c in cols}
add("sim2_p3.json", "", (0, 3))
s1 = json.load(open("sim_p3.json"))
for k in s1:
    _, a, b, kv = k.split("|")
    if b != "plain" or kv in ("trained", "random", "fP_opt", "attract_only") and a in ("ema", "zca", "shrink0.1"):
        if not any(x in k for x in ("|plain|trained", "|plain|random")):
            pass
add_keys = [k for k in s1 if ("|whiten_q|" in k or "|rot_q|" in k or "|pow0.5|" in k or "|shrink0.3|" in k or "kpp" in k)]
for k in add_keys:
    _, a, b, kv = k.split("|")
    w = [x for x in s1[k] if x["level"] in (0, 3)]
    rows[f"{a} / {b} / {kv} (from trained)"] = {c: float(np.mean([x[c] for x in w])) for c in cols}
ok = lambda d: d["stab_n03"] >= 0.85 and d["stab_drift"] >= 0.10
items = sorted(rows.items(), key=lambda kv: (not ok(kv[1]), -kv[1]["dist"]))
print("| rank | input / query or halves / keys | stable? | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 3) + "|")
for i, (n, d) in enumerate(items, 1):
    print(f"| {i} | {n} | {'yes' if ok(d) else 'no'} | " + " | ".join(f"{d[c]:.0f}" if c.startswith("dist") else f"{d[c]:.3f}" for c in cols) + " |")
