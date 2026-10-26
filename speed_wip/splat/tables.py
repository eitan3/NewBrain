"""Markdown tables for plan_v2/23 section 9 from the res_*.json files."""
import json
R = {d: json.load(open(f"res_main_{d}.json")) for d in ("rot", "warp")}
names = [("pk_rows_S256_it3", "(a) PK rows, 65K rows, gradient (3 steps / batch)"),
         ("knn_white32 k1", "(b) kNN-1, whitened 32"),
         ("knn_white32 k8", "(b) kNN soft top-8, whitened 32"),
         ("knn_raw448 k1", "(b) kNN-1, raw a (448)"),
         ("splatgrad_d16_65K", "(c') gradient splats d16, 65K fixed centres")]
for d_m in (8, 16, 32):
    for V in (1, 3):
        for mode, lab in (("point_h1", "point"), ("point_h2", "point, 2 hops"), ("ray-proj_h1", "ray"),
                          ("ray-proj_h2", "ray, 2 hops"), ("ray-oracle_h1", "ray, ORACLE direction")):
            names.append((f"splat_d{d_m}_v{V}_{mode}", f"(c) splat d{d_m}, {V} view{'s' if V > 1 else ''}, {lab}"))
print("| memory | rot: lvl 0 / 0.5 / 1 / 2 (recall@1) | rot lvl 1 cos | rot lvl 1 early / late | warp: lvl 0 / 0.5 / 1 / 2 | warp lvl 1 cos | warp lvl 1 early / late | CPU µs / read |")
print("|---|---|---|---|---|---|---|---|")
for k, lab in names:
    cells = []
    for d in ("rot", "warp"):
        r = R[d][k]
        cells.append(" / ".join(f"{r[l]['r1']:.2f}" for l in ("0", "0.5", "1", "2")))
        cells.append(f"{r['1']['cos']:.2f}")
        cells.append(f"{r['1']['r1_early']:.2f} / {r['1']['r1_late']:.2f}")
    b = "**" if k in ("knn_white32 k1", "splat_d32_v1_point_h1", "pk_rows_S256_it3") else ""
    print(f"| {b}{lab}{b} | " + " | ".join(cells) + f" | {R['rot'][k]['1']['us_per_read']:.0f} |")
print()
for d in ("rot", "warp"):
    W = json.load(open(f"res_width_{d}.json"))
    print(f"width {d}")
    for d_m in (8, 16):
        for ws in (0.25, 0.5, 1, 2, 4, 8):
            r = W[f"d{d_m}_w{ws}"]
            print(f"| {d_m} | {ws} | {r['sigma_over_drift']:.2f} | " +
                  " / ".join(f"{r[l]['r1']:.2f}" for l in ("0", "0.25", "0.5", "1", "2")) + " |")
