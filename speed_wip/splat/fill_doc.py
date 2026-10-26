"""Fill the result sections of plan_v2/23 from tables_out.md (run tables.py first)."""
import io
import os

DOC = r"C:\Eitan\Python\NewBrain\fruit-fly-sim\plan_v2\23_splat_memory_design.md"
HERE = os.path.dirname(os.path.abspath(__file__))
t = io.open(os.path.join(HERE, "tables_out.md"), encoding="utf-8", errors="replace").read().replace("\ufffd", "µ")
main, width = t.split("\nwidth rot\n")
w_rot, w_warp = width.split("width warp\n")
wr = [l for l in w_rot.strip().splitlines() if l.startswith("|")]
ww = [l for l in w_warp.strip().splitlines() if l.startswith("|")]
wtab = ["| d_m | width × | σ / drift (rot) | rot: lvl 0 / 0.25 / 0.5 / 1 / 2 | σ / drift (warp) | warp: lvl 0 / 0.25 / 0.5 / 1 / 2 |",
        "|---|---|---|---|---|---|"]
for a, b in zip(wr, ww):
    ca = [c.strip() for c in a.strip("|").split("|")]
    cb = [c.strip() for c in b.strip("|").split("|")]
    wtab.append(f"| {ca[0]} | {ca[1]} | {ca[2]} | {ca[3]} | {cb[2]} | {cb[3]} |")
wtab = "\n".join(wtab)

s = io.open(DOC, encoding="utf-8").read()
s = s.replace("""| d_m | width × | σ / drift | rot | warp |
|---|---|---|---|---|
| (filled from `res_width_*.json`, §9.3) | | | | |""", wtab + """

  The best width is the smallest one tested (σ ≈ 0.2-0.4 × the drift length, ≈ 0.25 × the birth spacing). At
  σ = the 3-NN spacing (×1), recall is already 0.14-0.18 **at zero drift**: neighbours blend. Wide splats never help
  under drift. The narrow-kernel limit is kNN-1.""")
s = s.replace("they never beat 1 view of 32;", "they never beat 1 view of 32 (§9.2: d16 1 view 0.42, 3 views 0.60, d32 1 view 0.94 at 1× warp);")
s = s.replace("(0.42 → 0.62 in v1)", "(0.42 → 0.60)")

sec92 = """### 9.2 Main table (16K facts; recall@1 at drift levels 0 / 0.5 / 1 (= 6K→7K) / 2, value cosine and early (first 10%) vs late (last 10%) facts at level 1)
Full rows: `log_main_rot.txt`, `log_main_warp.txt`. CPU µs per read: 4 threads, brute-force candidate search, so it
compares the maths, not the GPU design (§7).

""" + main.strip() + """

What the table says:
1. **Rows vs everything that stores the write-time key.**
   - PK rows (the best fixed addressing of plan 18, gradient-written) are fine at drift 0 (0.99) and lose most of it at
     0.5× (0.75-0.82) and 1× (0.22-0.24).
   - That matches plans 16-20: their cells are re-entered by different contexts after the drift.
   - The kNN-1 and narrow-splat memories keep 0.94-1.00 at 1×.
2. **kNN-1 in pzca32 is the best and the simplest** (1.00 / 0.99 at 1×, 0.69-0.99 at 2×).
   - A soft top-8 blend hurts badly (0.39-0.58). **Do not average neighbours: take the nearest.**
   - Raw 448-d kNN is fine under rot (0.98) but loses to whitening under warp (0.65): the minor directions carry the
     warp.
3. **The splat field ≈ kNN when it is narrow and d_m = 32** (0.94-0.95 at 1×, 1 view, point read). The small gap is
   the kernel blend (value cos 0.85-0.87 vs 0.99).
   - d_m = 16 / 8 lose 0.05-0.6, the dimension effect of §2.
4. **Features**:
   - **2 hops (mean shift)** cost 0.03-0.6 everywhere except d16 3-views under warp (+0.05). The location estimate is
     pulled to the local density centre, not to the drifted fact.
   - **The ray with an unlearned direction** is neutral to −0.07.
   - **The ray with the ORACLE direction** (pointing back to the write-time location) gains +0.14-0.20 under warp at
     d16 and +0.02-0.03 at d32. It is a ceiling for a learned ray.
   - **3 views in one shared field**: +0.18 at d16 under warp; −0.13 / −0.04 at d8; +0.01-0.03 at d32. Each view
     costs one splat per fact (×3 slots).
   - **Gradient splats** (fixed centres, values by the delta rule) behave like rows (0.19-0.37 at 1×): without a key
     at the write-time location, there is no drift robustness.
5. **Early vs late**:
   - The splat field's early facts are 0.05-0.4 worse than its late facts at 1×. The conflict shrink makes early
     splats narrower than the later ones around them: an order dependence of the densify rules.
   - kNN has none (0.99 / 0.99). Rows: 0.16 / 0.26 (overwriting).
"""
s = s.replace("""### 9.2 Main table (16K facts; recall@1 / value cos at drift level 0, 0.5, 1 (= 6K→7K), 2)
(filled from `log_main_rot.txt`, `log_main_warp.txt` — see below)""", sec92)

s = s.replace("""### 9.3 Width sweep
(filled from `log_width_*.txt`)""", """### 9.3 Width sweep
See §3.1 (the table is there). d16, 1 view, no shrink, σ = width × (the 3-NN spacing at birth, κ = 1).""")

sec94 = """### 9.4 Capacity and interference
**Capacity** (recall@1 at drift 0 / 1×, early facts in brackets; rot, warp in `log_cap_warp.txt`):

| facts N | PK rows 16K | PK rows 65K | kNN-1, 16K slots | kNN-1, 65K slots | splat d16 1 view, 16K slots | splat d16 1 view, 65K slots | splat d16 3 views, 65K slots |
|---|---|---|---|---|---|---|---|
| 4,096 | 0.99 / 0.32 (0.26) | 1.00 / 0.43 (0.38) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.95 (0.91) | 1.00 / 0.95 | 1.00 / 0.97 |
| 16,384 | 0.73 / 0.08 (0.03) | 0.99 / 0.22 (0.20) | 1.00 / 0.99 | 1.00 / 0.99 | 1.00 / 0.90 (0.87) | 1.00 / 0.90 | 1.00 / 0.92 |
| 65,536 | 0.18 / 0.01 (0.00) | 0.71 / 0.05 (0.01) | 0.25 / 0.25 (0.00) | 1.00 / 0.98 (0.99) | 0.25 / 0.23 (0.00) | 1.00 / 0.83 (0.75) | 0.34 / 0.30 (0.00: over budget) |

- Rows degrade by interference before they are "full": 0.73 at 1 fact per row, 0.18 at 4.
- kNN / splats are exact up to the slot budget, then evict the oldest (the surviving 25%, early 0).
- Splats need one slot per fact per view.
- With more facts, drift robustness decreases slowly for kNN (0.99 → 0.98), faster for the d16 splat
  (0.95 → 0.83 rot; 0.54 → 0.31 warp). That is the N^{1/d} effect at d = 16.

**Interference** (`log_interf_rot.txt`; groups of 8 facts within ε (relative) of a centre, different values; recall@1
at drift 0 / 1×):

| ε | PK rows 65K | kNN-1 | splat (no shrink) | splat + conflict shrink |
|---|---|---|---|---|
| 0.05 | 0.14 / 0.05 | 1.00 / 0.22 | 0.39 / 0.11 | 0.91 / 0.11 |
| 0.1 | 0.16 / 0.06 | 1.00 / 0.34 | 0.60 / 0.12 | 1.00 / 0.12 |
| 0.2 | 0.42 / 0.09 | 1.00 / 0.61 | 0.92 / 0.15 | 1.00 / 0.34 |
| 0.4 | 0.85 / 0.17 | 1.00 / 0.92 | 1.00 / 0.53 | 1.00 / 0.68 |

- The conflict shrink does its job at drift 0 (it separates near-duplicates the plain kernel blends).
- Under drift, near-duplicates are lost by every memory. kNN loses least, because it does not blend.
"""
s = s.replace("""### 9.4 Capacity and interference
(filled from `log_cap_*.txt`, `log_interf_rot.txt`)""", sec94)

sec95 = """### 9.5 Forgetting, consolidation, rehearsal (`log_fade_rot.txt`; d16, 1 view, 16K facts, fading λ per write batch of 512, 32 batches)

| fade λ | rehearsal | splats | lvl 1: all / early / late |
|---|---|---|---|
| 0 | no | 16,384 | 0.909 / 0.858 / 0.983 |
| 0 | yes | 16,497 | 0.916 / 0.921 / 0.981 |
| 0.01 | no | 16,384 | 0.883 / 0.786 / 0.985 |
| 0.01 | yes | 16,579 | 0.901 / 0.913 / 0.988 |
| 0.03 | no | 16,384 | 0.797 / 0.539 / 0.993 |
| 0.03 | yes | 16,772 | 0.840 / 0.898 / 0.990 |
| kNN-1 (no fade) | yes (a second key) | - | 0.984 / 1.000 / - |

- "Rehearsal" re-writes the first 25% of the facts once, half-way through, at their 0.5×-drifted thought.
- Most rehearsals land on the old splat and reinforce it (α + 1, consolidated, no further fading). Only 113-388
  spawn a second key.
- Fading forgets the old facts in proportion to λ (early 0.86 → 0.54). Rehearsal + consolidation protects them
  (0.90-0.92), so the plan-21 "fade unless reinforced" works as designed.
- Note: at drift 0 every variant is 1.00. The fading here never pushes α below the prune threshold within 32 batches.
"""
s = s.replace("""### 9.5 Forgetting, consolidation, rehearsal
(filled from `log_fade_rot.txt`)""", sec95)

sec10 = """## 10. Honest assessment and recommendation

**Does the splat field beat rows on drift-robust recall?**
- Yes, by a wide margin: 0.94-0.97 vs 0.22-0.24 at the 6K→7K drift, and 0.55-0.79 vs 0.04-0.12 at 2×.
- That is under the conditions:
  - d_m ≈ 32 truncated-whitened;
  - narrow splats;
  - the key stored where the thought was at write time;
  - a direct write.

**Does it beat kNN?** No.
- At its best it equals kNN-1, because at its best it is kNN-1 with a soft kernel.
- None of the splat-specific mechanisms (wide / anisotropic kernels, rays, multi-hop compositing, several views in one
  field) improved drift robustness in the toy. Hops and wide kernels hurt.
- The ray could help only with a learned "where was this written" direction (the oracle ceiling: +0.14-0.20 at d16,
  +0.02-0.03 at d32). Nothing in the toy says how to learn that, and it doubles the read cost.

**At what cost?**
- At the GPU scale, the splat / kNN read must go through an index.
- With one cell table it loses most of its edge (0.27-0.35 at 1×). With **T = 4 rotated tables** it keeps 0.74-0.95,
  depending on candidates (33-965) and N.
- In plain PyTorch:
  - 87 ops per batched read;
  - ≈ 0.2-0.34 s/step for today's read pattern (12 queries per token × 5 passes), vs today's ≈ 0.9 s;
  - ≈ 0.05 s/step for the lean pattern (4 queries × 2 passes).
- Both meet ≥ 10k tok/s. The lean one is at about 20k if the views do not need ConceptHier.

**What is genuinely useful from the splat idea**:
- the **opacity** (a confidence / "is anything here" signal for the gate, and the handle for fading);
- **α fading with consolidation** (= plan 21, now with a natural per-key form);
- **densify / prune as slot management** (spawn on error + low opacity, bucket replacement, conflict shrink with a
  floor);
- **multiple views — in the index, not in the kernel.**

**Recommendation**
0. **Gate: measure the real self-recall first** (CPU, offline, about 1 h).
   1. Capture a = [μ; p̄] for 16-32 val sequences at two checkpoints 1,000 steps apart in infancy (the next run; the
      old captures are in the Recycle Bin).
   2. Fit pzca32 on the earlier one.
   3. Measure kNN-1 self-recall of each position's own key (the stored set = the earlier capture, plus other
      positions as distractors), at 16K and 128K stored keys.
   - The toy predicts ≥ 0.9; plan 20's 10-NN survival (0.4) and frozen-row overlap (0.21) are both consistent with
     that.
   - **If the real self-recall is ≲ 0.5** (dense document manifolds would do that), stop: no key-based memory will
     beat the rows, and plan 21's life cycle is the answer.
1. **If it passes, build "splat-lite" as a flagged second pool** (`--mem-splat`), not a replacement at first:
   - keys = frozen pzca32 coordinates of the write-time thought (d_m = 32, one key per level per written position);
   - narrow isotropic splats (σ = 0.25 × the birth spacing), opacity with plan-21 fading / consolidation, values
     d_v = 256 by row-Adagrad + delta-target spawning (§4.2, option A);
   - a T = 4 rotated PQ cell index (C = 512 per half, cap 8, kc 4, about 128 candidates), a top-4 point read, 1 hop,
     no rays;
   - the opacity as an extra gate feature;
   - a coarse config first (config B: 4 queries × 2 passes), so that it costs ≤ 0.05 s/step.
2. **Benchmark on the GPU when it is free** (`bench_gpu.py --device cuda --config all`). Confirm ≤ 50 ms/step for B and
   ≤ 350 ms for A, and the VRAM (chunk / bf16 the candidate temp).
3. **A 12K run on the world-facts mix** with eval_world (plan 21 §8 B): splat-lite pool on vs off, with the row pool
   and the life cycle as today. The question is whether early facts survive the drift better than in rows (the
   knockout deltas of early vs late facts).
4. **Not now**: rays, multi-hop, wide or anisotropic kernels, gradient on μ / Σ, learned views. Each is ×2 cost
   without a measured gain. Revisit only if (3) shows splat-lite helps and a learned direction can be trained against
   the retrieval error.
"""
s = s.replace("""## 10. Honest assessment and recommendation
(filled after §9)""", sec10)
io.open(DOC, "w", encoding="utf-8").write(s)
print("ok", s.count("(filled"))
