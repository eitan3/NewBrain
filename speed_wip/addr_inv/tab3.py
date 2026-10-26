import json, glob, numpy as np
R = {}
diag = []
for f in sorted(glob.glob("sim3_L*_h*.json")):
    for k, v in json.load(open(f)).items():
        if k == "diag": diag += v; continue
        R.setdefault(k, []).extend(v)
print("drift diag:", {l: {k: round(np.mean([d[k] for d in diag if d["level"] == l]), 3) for k in ("cos_ahat", "rel_change", "knn10_preserved", "q_cos_same_pos")} for l in (0, 3)})
cols = ["dist", "dead", "xdoc", "lag1", "row_stab", "key_stab", "max_key_cos", "dup95", "dup99"]
mig = ["mig_carried_key_map", "mig_carried_row_map", "mig_unmapped_rows_eval", "mig_key_conflict_share", "mig_row_conflict_share", "mig_old_rows_copied_to_2plus", "mig_new_rows_mapped"]
rows = []
for k, v in R.items():
    d = {c: float(np.mean([x[c] for x in v])) for c in cols}
    d["L0"] = float(np.mean([x["row_stab"] for x in v if x["level"] == 0])); d["L3"] = float(np.mean([x["row_stab"] for x in v if x["level"] == 3]))
    for c in mig:
        if c in v[0]: d[c] = float(np.mean([x[c] for x in v]))
    rows.append((k, d))
rows.sort(key=lambda r: -r[1]["row_stab"])
print("| rank | setting | distinct /1k (S512 ideal 5932; S64 max 4096) | dead | xdoc | lag1 | 6K→7K row stab (L16 / L128) | key-ID stab | max key cos | near-dup keys >0.95 / >0.99 (per 2 codebooks) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for i, (k, d) in enumerate(rows, 1):
    print(f"| {i} | {k} | {d['dist']:.0f} | {d['dead']:.3f} | {d['xdoc']:.3f} | {d['lag1']:.3f} | **{d['row_stab']:.3f}** ({d['L0']:.3f} / {d['L3']:.3f}) | {d['key_stab']:.3f} | {d['max_key_cos']:.3f} | {d['dup95']:.1f} / {d['dup99']:.1f} |")
print()
for k, d in rows:
    if "mig_carried_key_map" in d:
        print(k, {c: round(d[c], 3) for c in mig})
