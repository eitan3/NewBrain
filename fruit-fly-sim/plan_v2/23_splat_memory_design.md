# 23. Gaussian-splat memory field: design + offline toy study (CPU only, no trainer integration) — 2026-10-24

Scope: the owner's idea of a continuous low-dimensional memory filled with Gaussian "splats" (position μ, covariance
Σ, opacity α, latent value v), read through several views of the thought, by rays with front-to-back compositing
and / or multi-hop, in one batched op for all tokens / views / experts. This doc gives the maths, the analysis, the
write rules, the speed budget (the owner's hard requirement: ≥ 10k tok/s, target 20k), the plug-in and the risks, and
an offline toy that compares it with the current product-key rows and a kNN memory under the measured thought drift.
Scripts, logs and JSON: `speed_wip/splat/` (list at the end). CPU, 4 threads; the GPU was not touched.

Structure (for the merge with the parallel study): §1 maths · §2-3 geometry (dimension, views, width vs drift) ·
§4 writes · §5 capacity · §6 lookup · §7 speed · §8 plug-in · §9 toy · §10 assessment + recommendation ·
§11 related work · §12 risks.

## TL;DR
1. **What makes a memory drift-robust is not the splat machinery but "store the write-time key, read the nearest
   one, in a ~32-d truncated-whitened space".**
   - In the toy, the drift is calibrated to the measured 72% relative change / 40% 10-NN survival per 1,000 steps. It
     is slightly harsher than the real drift on addressing (frozen PK rows keep 0.16 here vs 0.21 measured).
   - Under that drift:
     - product-key rows (gradient-written) keep recall@1 **0.22-0.24** (from 0.99);
     - kNN-1 in the whitened-32 space keeps **0.99-1.00**;
     - the splat field keeps **0.94-0.97** with narrow splats in d_m = 32.
2. **A splat field with narrow splats IS a soft kNN memory. Every "splat-specific" feature tested was neutral or
   harmful for drift-robust recall:**
   - wide splats: recall collapses once σ ≳ the neighbour spacing, already at zero drift;
   - rays with an unlearned direction: neutral;
   - mean-shift / multi-hop reads: −0.03 to −0.6;
   - several kernel views in one shared field: interference at small d_m, a small gain only when one view is too small.
   - A ray helps only with an oracle direction (pointing back to where the fact was written): +0.14-0.20 recall under
     the warp drift at d_m 16 (+0.02 at 32). That is a ceiling, not something we know how to learn.
3. **Dimension: the memory space must be "low" (~16-32 dims), not 3D-like.**
   - At d ≤ 8, the drift is 1-15× the nearest-neighbour spacing (recall 0.01-0.3).
   - In 448-d, distances concentrate: 32,765 of 1M points lie within 1.1× the nearest distance.
   - Measured best: truncated whitening to r = 32-64.
4. **The real cost is the index, not the kernel.**
   - A single product-quantised cell lookup (the GPU-friendly analogue of product keys) throws the robustness away
     again (0.27-0.35 instead of 0.97-1.00 with 9 candidates) because the drift crosses cell borders.
   - **T = 4 independently rotated cell tables** (multi-table, LSH-style) recover most of it (65K facts, rot / warp):
     0.74 / 0.77 at 33 candidates per query, 0.86 / 0.89 at 64, 0.92 / 0.95 at 125. This is where "multiple views" actually pay: in the index, not in
     the kernel.
5. **Speed: it fits the budget, if it stays a point read.**
   - Plain PyTorch (no Triton), 87 ops per batched read (fwd + bwd, all views at once).
   - Config A (12 queries per token, every pass, 4 tables): ≈ 0.2-0.34 s/step → ≈ 15k tok/s.
   - Config B (4 queries per token, 2 passes): ≈ 0.05 s/step → ≈ 20k tok/s.
   - Today's memory costs ≈ 0.9 s/step.
   - Rays (×2.2 ops, ×1.7 bytes) and hops (×2) do not fit B's budget.
6. **Recommendation:**
   - Build a **"splat-lite" prototype**: narrow-Gaussian keys at the frozen pzca32 coordinate of the write-time thought,
     a learned value, opacity with plan-21 fading / consolidation, a T = 4 multi-table PQ cell index, a top-4 point read,
     1 hop, no rays.
   - **But first measure one number on real thoughts**: the kNN self-recall of a position's own key in pzca32 across a
     1,000-step checkpoint pair. The toy says 0.99. If the real value is ≲ 0.5, the whole approach has no edge over
     rows, and it should not be built.

---

## 1. The maths

### 1.1 Splats and the read kernel
- A splat i has:
  - a position μ_i ∈ R^{d_m};
  - a covariance Σ_i, either diagonal Σ_i = diag(s_i²) or diagonal + low rank, Σ_i = diag(s_i²) + U_i U_iᵀ with
    U_i ∈ R^{d_m×r}, r = 1-2;
  - an opacity α_i > 0 (stored as log α_i);
  - a value v_i ∈ R^{d_v} (d_v = 256 as today's rows).
- The density of splat i at a point p:

      ρ_i(p) = α_i exp(−½ m_i(p)),   m_i(p) = (p − μ_i)ᵀ Σ_i⁻¹ (p − μ_i).

  - Diagonal: m_i = Σ_d (p_d − μ_id)² / s_id².
  - Low rank (Woodbury), with D = diag(s²):
    Σ⁻¹ = D⁻¹ − D⁻¹U (I_r + UᵀD⁻¹U)⁻¹ UᵀD⁻¹, so m_i = ‖p − μ‖²_{D⁻¹} − ‖(I + UᵀD⁻¹U)^{-½} UᵀD⁻¹(p − μ)‖².
- **GPU form.** One splat's diagonal kernel is a dot product against 2d_m + 1 packed coefficients:

      −2 log ρ_i(p) = [p², p, 1] · [1/s_i², −2 μ_i/s_i², Σ_d μ_id²/s_id² − 2 log α_i].

  So a query against K candidates is one batched GEMV (one bmm for all queries; §6-7).

### 1.2 Point read (opacity compositing at one point)
- Total density R(p) = Σ_i ρ_i(p); opacity A(p) = 1 − e^{−R}; colour c(p) = Σ_i ρ_i v_i / R.
- The read:

      r(p) = A(p) · c(p) = Σ_i w_i v_i,   w_i = A ρ_i / R.

  - A ∈ [0, 1] is the memory's own confidence ("is anything stored here?"). It is a natural extra gate feature.
  - Computed in the log domain: c = softmax_i(log ρ_i) · v and log R = logsumexp_i(log ρ_i). The toy needed this:
    narrow splats underflow in fp32.
- Several views v = 1..V give r_v and log R_v. They are combined by r = Σ_v softmax_v(log R_v) c_v (= opacity
  weighting when R ≪ 1), or kept as separate heads with their own gates (today's 12-head layout).

### 1.3 Ray read (front-to-back compositing with transmittance)
- Ray p(t) = o + t d, t ∈ [0, L], with S samples t_s, spacing Δ_s.
  - o = the view of the thought;
  - d = a learned direction (another projection of the thought, or route- / pass-dependent);
  - L = the reach.
- Per sample: density σ_s = R(p(t_s)), alpha a_s = 1 − e^{−σ_s Δ_s}, transmittance T_s = Π_{s'<s} (1 − a_{s'}) =
  exp(−Σ_{s'<s} σ_{s'} Δ_{s'}).
- The read and the per-splat weights (all weights are computed in the log domain: log T_s = −Σ σΔ, and
  log a_s = log(−expm1(−σ_s Δ_s))):

      r = Σ_s T_s a_s c(p(t_s)),   A_ray = Σ_s T_s a_s = 1 − T_{S+1},
      w_i = Σ_s T_s a_s ρ_i(p(t_s)) / σ_s.

- **Sample-free "splatting" form** (as 3DGS does it, exact for Gaussians and cheaper: K terms instead of S·K). Along a
  line, splat i has:
  - a closest-approach depth t_i* = −(o − μ_i)ᵀΣ_i⁻¹d / (dᵀΣ_i⁻¹d);
  - a perpendicular Mahalanobis distance m_i⊥ = m_i(o) − ((o − μ_i)ᵀΣ_i⁻¹d)² / (dᵀΣ_i⁻¹d);
  - an integrated density τ_i = α_i √(2π / dᵀΣ_i⁻¹d) · exp(−½ m_i⊥) (restricted to [0, L] by the Gaussian CDF if
    needed).

  Sort the K candidates by t_i*, then a_i = 1 − e^{−τ_i}, T_i = Π_{j before i}(1 − a_j), r = Σ_i T_i a_i v_i. It costs
  one more sort of K (a topk / sort op).

### 1.4 Multi-hop read
- Hop h + 1's query is updated from hop h's result. Either:
  - a learned update p^{(h+1)} = p^{(h)} + W_h r^{(h)}; or
  - the location estimate μ̂^{(h)} = Σ_i w_i μ_i / Σ_i w_i, i.e. **one mean-shift / modern-Hopfield step**, usually
    with a coarse-to-fine width schedule s^{(1)} = κ_1 s (κ_1 = 2), s^{(2)} = s.
- The output is the last hop's r, or a gated sum of the hops' r.

### 1.5 Gradients (point read, diagonal Σ; g = ∂L/∂r)
- Values: ∂L/∂v_i = w_i g (an index_add_ into one grad buffer, exactly today's MemRead backward).
- Through the weights (with c = r/A, R = Σρ):

      ∂L/∂ρ_j = e^{−R} (g·c) + (A/R) (g·v_j − g·c).

  The first term says "more mass here would help"; the second is the softmax-like "this splat vs the others".
- Per splat:
  - ∂ρ_j/∂ log α_j = ρ_j;
  - ∂ρ_j/∂μ_jd = ρ_j (p_d − μ_jd)/s_jd²;
  - ∂ρ_j/∂ log s_jd = ρ_j (p_d − μ_jd)²/s_jd².
- Query: ∂L/∂p = −Σ_j (∂L/∂ρ_j) ρ_j Σ_j⁻¹(p − μ_j). The view p = P_v a gives ∂L/∂P_v = (∂L/∂p) aᵀ and
  ∂L/∂a = P_vᵀ ∂L/∂p (zero with today's sg(a) on the addressing input).
- Ray:
  - ∂r/∂a_s = T_s c_s − (Σ_{s'>s} T_{s'} a_{s'} c_{s'}) / (1 − a_s) (the NeRF backward, one reverse cumsum);
  - ∂a_s/∂σ_s = Δ_s (1 − a_s);
  - then the per-sample ρ derivatives at p_s = o + t_s d, with ∂p_s/∂o = I and ∂p_s/∂d = t_s I.
- Multi-hop: autograd through μ̂ (a ratio of two weighted sums).
- **In the recommended configuration, μ and Σ get no gradient**: they are set by the write rules (§4). Only v (and
  optionally log α and the views) learn by gradient. This keeps the backward as cheap as today's (§7), and keeps a
  written key where it was written, which is the whole point (§3).

---

## 2. How low-dimensional must the memory space be? (`dimtheory.py`, `run_toy.py dim`)

The owner's intuition is half right. The space must be low enough that distances still discriminate, but high
enough that the drift is smaller than the spacing between stored keys.

**(a) Distance concentration in high D** (iid N(0, I_d), 1,048,576 points, 64 queries):

| d | 1-NN / median distance | points within 1.1 × the 1-NN distance |
|---|---|---|
| 2 | 0.001 | 1 |
| 4 | 0.031 | 1 |
| 8 | 0.149 | 1 |
| 16 | 0.340 | 3 |
| 32 | 0.505 | 11 |
| 64 | 0.637 | 33 |
| 128 | 0.746 | 355 |
| 448 | 0.860 | 32,765 |

At 448-d, a Gaussian kernel cannot single out a point: thousands sit at nearly the nearest distance. Grids and rays
(§6) are only meaningful at d ≲ 6.

**(b) But too low is fatal under drift.** The ratio of the 1× drift to the nearest-key distance, on the synthetic
thoughts (PR 5.3), median over queries:

| memory space | rot drift: 16K / 1M keys | warp drift: 16K / 1M keys |
|---|---|---|
| whitened r = 4 | 2.65 / 7.38 | 5.88 / 15.4 |
| whitened r = 8 | 1.03 / 1.68 | 1.81 / 2.71 |
| whitened r = 16 | 0.62 / 0.81 | 1.00 / 1.37 |
| **whitened r = 32** | **0.60 / 0.71** | **0.71 / 0.82** |
| whitened r = 64 | 0.67 / 0.74 | 0.57 / 0.62 |
| raw a (448) | 1.01 / 1.12 | 0.97 / 1.04 |

**(c) What that does to recall** (kNN-1, 16K facts; `res_dim_*.json`):

| space | rot: lvl 0.5 / 1 / 2 | warp: lvl 0.5 / 1 / 2 |
|---|---|---|
| whitened r = 2 | 0.005 / 0.001 / 0.000 | 0.006 / 0.003 / 0.002 |
| whitened r = 4 | 0.191 / 0.021 / 0.003 | 0.065 / 0.012 / 0.015 |
| whitened r = 8 | 0.964 / 0.315 / 0.036 | 0.705 / 0.164 / 0.074 |
| whitened r = 16 | 1.000 / 0.952 / 0.328 | 0.999 / 0.662 / 0.444 |
| **whitened r = 32** | 1.000 / **0.989** / 0.685 | 1.000 / **1.000** / 0.987 |
| whitened r = 64 | 1.000 / 0.997 / 0.861 | 1.000 / 1.000 / 1.000 |
| whitened r = 128 | 1.000 / 0.985 / 0.766 | 1.000 / 1.000 / 1.000 |
| raw a (448) | 1.000 / 0.977 / 0.652 | 1.000 / 0.653 / 0.590 |
| PCA r = 32, NOT whitened | 1.000 / 0.857 / 0.322 | 0.955 / 0.353 / 0.296 |

**The stability law.** For a narrow kernel (h → 0), recall ≈ P(no other key nearer than the drift) ≈
exp(−(δ/r_NN)^{d_eff}), with δ/r_NN the ratio in table (b). It fits the toy:
- rot r = 8: ratio 1.03 → e^{−1} ≈ 0.37, measured 0.32;
- rot r = 16: ratio 0.62 → ≈ 1, measured 0.95;
- warp r = 4: ratio 5.9 → 0, measured 0.01.

So the knobs are:
- **(i) the ratio**: whitening matters, because it puts the slow content directions on an equal footing with the
  others (unwhitened PCA-32 loses 0.13-0.65);
- **(ii) d_eff ≈ 16-64**;
- **(iii) the number of stored keys**: the ratio grows only as N^{1/d}, i.e. slowly at d = 32 (0.60 → 0.71 from 16K
  to 1M).

Recommendation: **d_m = 32** (truncated whitening r = 32, plan 18's best point), not 3-8.

---

## 3. Multi-view redundancy and width vs drift

### 3.1 Kernel width vs drift (`run_toy.py width`; analytic in `dimtheory.py`)
- Isotropic case: d dims, drift δ ~ N(0, s² I), kernel width h, N keys of unit spread.
  - The target's expected density is (1 + s²/h²)^{−d/2}.
  - The competitors' mass is ≈ N (h²/(1 + h²))^{d/2} e^{−d/(2(1+h²))}.
- Their ratio (target share T/(T+C)) is **highest for h ≲ s and collapses for h > s**. Example, d = 16, N = 1M,
  s = 0.5, at h/s = 0.25 / 0.5 / 1 / 2: 0.99 / 0.97 / 0.47 / 0.002.
- **Wider splats do not buy drift tolerance**: the competitor mass grows like h^d, faster than the target mass.
- Measured (16K facts, 1 view, point read; σ = ws × κ × the 3-NN spacing at birth), recall@1 at drift lvl 0 / 0.5 / 1 / 2:

| d_m | width × | σ / drift (rot) | rot: lvl 0 / 0.25 / 0.5 / 1 / 2 | σ / drift (warp) | warp: lvl 0 / 0.25 / 0.5 / 1 / 2 |
|---|---|---|---|---|---|
| 8 | 0.25 | 0.31 | 1.00 / 1.00 / 0.93 / 0.34 / 0.05 | 0.19 | 1.00 / 0.98 / 0.55 / 0.12 / 0.05 |
| 8 | 0.5 | 0.61 | 1.00 / 1.00 / 0.91 / 0.30 / 0.04 | 0.37 | 1.00 / 0.95 / 0.52 / 0.11 / 0.05 |
| 8 | 1 | 1.23 | 0.18 / 0.14 / 0.09 / 0.03 / 0.01 | 0.74 | 0.18 / 0.15 / 0.10 / 0.04 / 0.03 |
| 8 | 2 | 2.46 | 0.01 / 0.01 / 0.01 / 0.00 / 0.00 | 1.49 | 0.01 / 0.01 / 0.01 / 0.01 / 0.00 |
| 8 | 4 | 4.91 | 0.01 / 0.01 / 0.01 / 0.00 / 0.00 | 2.97 | 0.01 / 0.01 / 0.01 / 0.00 / 0.00 |
| 8 | 8 | 9.82 | 0.01 / 0.01 / 0.01 / 0.00 / 0.00 | 5.95 | 0.01 / 0.01 / 0.01 / 0.00 / 0.00 |
| 16 | 0.25 | 0.43 | 1.00 / 1.00 / 1.00 / 0.95 / 0.42 | 0.26 | 1.00 / 1.00 / 0.99 / 0.56 / 0.38 |
| 16 | 0.5 | 0.87 | 1.00 / 1.00 / 1.00 / 0.84 / 0.26 | 0.52 | 1.00 / 1.00 / 0.97 / 0.50 / 0.34 |
| 16 | 1 | 1.73 | 0.14 / 0.12 / 0.09 / 0.04 / 0.02 | 1.04 | 0.14 / 0.14 / 0.12 / 0.06 / 0.05 |
| 16 | 2 | 3.47 | 0.01 / 0.01 / 0.01 / 0.01 / 0.01 | 2.07 | 0.01 / 0.01 / 0.01 / 0.01 / 0.01 |
| 16 | 4 | 6.93 | 0.00 / 0.01 / 0.01 / 0.00 / 0.01 | 4.14 | 0.00 / 0.01 / 0.01 / 0.00 / 0.01 |
| 16 | 8 | 13.87 | 0.00 / 0.00 / 0.01 / 0.00 / 0.00 | 8.28 | 0.00 / 0.01 / 0.01 / 0.00 / 0.00 |

  The best width is the smallest one tested (σ ≈ 0.2-0.4 × the drift length, ≈ 0.25 × the birth spacing). At
  σ = the 3-NN spacing (×1), recall is already 0.14-0.18 **at zero drift**: neighbours blend. Wide splats never help
  under drift. The narrow-kernel limit is kNN-1.

### 3.2 Multiple views
- A view is p_v = P_v a. All views see the same drifted thought, so their drifts are correlated through the shared
  directions of a.
  - If the P_v span disjoint blocks of the whitened space, the view drifts are independent.
  - Then the **product** of the view kernels is just one kernel in the stacked V·d_m space (no new information).
  - The **sum** (OR) is more tolerant (one view may fail) but also sums the competitor mass.
- So the most V views can do is what one view of V·d_m dims does. That is, views are a way to buy dimensions, not
  robustness.
- Measured (§9.2):
  - 3 views of d_m = 16 (in one shared field) beat 1 view of 16 under the warp drift (0.42 → 0.60);
  - they never beat 1 view of 32 (§9.2: d16 1 view 0.42, 3 views 0.60, d32 1 view 0.94 at 1× warp);
  - at d_m = 8 they hurt, because of cross-view interference in a shared field.
- **Where views do pay: in the index (§6).** T independently rotated cell tables over the same key space change
  where the cell borders are. A drifted query that crosses a border in one table usually doesn't in another. This
  is the multi-table LSH idea and it is the measured way to recover the brute-force robustness at GPU cost.

### 3.3 Retrieval stability, summary
- Expected self-recall (narrow kernel, exact search) is ≈ exp(−(δ/r_NN)^{d_eff}).
- With an index of T tables, each probing kc cells per query, multiply by P(the own cell is probed in at least one
  table). Measured (§6): with T = 1, kc = 8 that is 0.27-0.51; with T = 4 it is 0.65-0.95.
- Rays / hops / width do not change the first factor in the toy (§9).

---

## 4. Write rules

### 4.1 (i) Gradient
- v_i by the read gradient (MemRead's index_add_ into one buffer; MemOpt's row-Adagrad with S₀, unchanged).
- log α_i by its gradient (cheap: one scalar per candidate winner).
- μ_i, Σ_i: no gradient by default (§1.5).
- The views P_v: frozen after the fit (§8). A learned view moves the read location away from keys written earlier,
  which is the stale-key problem again.

### 4.2 (ii) Direct write = densification
- For a chosen position: the view location p = P_v a and a value target v*.
  1. Read first.
  2. **If A > 0.5 and cos(c, v*) > 0.9: reinforce.** The top splat gets α ← min(α + 1, α_max) and is marked as a
     consolidation candidate. Nothing is spawned.
  3. **Otherwise spawn:** μ = p, s = κ · (mean distance to its 3 nearest splats) (the 3DGS init), clamped
     [s_min, s_max]; α = α₀ (α₀ = 3 → A ≈ 0.95 at the centre); v = v* (or v = v* − r, the delta-rule form, so that
     the read becomes v*).
  4. **Conflict:** an existing splat j with m_j(p) < 2.5² and cos(v_j, v*) < 0.5 is shrunk so that p falls outside
     its 2.5-σ ball (s_j ← s_j · √m_j / 2.5), with a floor of 0.3 × its birth width. Without the floor, the shrinks
     compound (the v1 toy had about 2.3 shrinks per splat and underflowed).
- **What v\* is in the brain is the crux** (the toy is handed the values). Options:
  - **(A, recommended)** the delta target v* = sg(r − η_w ∂L/∂r) at that position. It is available from MemRead's
    backward (gout). A direct write is then "the gradient write, but into a fresh slot instead of shared rows", so it
    has no interference.
  - (B) the concept target z*_l mapped into value space.
  - (C) the final thought sg(x_K).
- Which positions to write: the top-surprise / gate-open positions of the last micro-batch (plan 22's gate exists),
  about 1-2% of the positions per step (§5 for the slot budget).

### 4.3 (iii) Adaptive density control (3DGS-style), in the maintenance slot (every 250 steps, as the balancing)
- **Clone** (under-reconstruction): a small splat (s below the median) with a high accumulated query gradient
  ‖∂L/∂p‖ while being the top contributor gets a copy displaced along that gradient.
- **Split** (over-reconstruction): a large splat (s above the 90th percentile) with high accumulated value error
  (Σ w ‖v* − c‖ over its reads) or high usage with mixed targets is replaced by 2 children. They are sampled from
  N(μ, Σ), with s/1.6 (the 3DGS factor) and the parent's value. Their values then separate by gradient.
- **Prune**: α_eff < α_min (0.05), or a near-duplicate (m < 0.5 and cos(v) > 0.95: merge, α summed).
- **Opacity reset** (3DGS resets α every 3,000 iterations to purge floaters): our equivalent is the fading below,
  which removes what is not reinforced.
- **Bucket replacement = prune at write time**: in the GPU layout (§6), a new splat overwrites the weakest (faded)
  pointer of its cell's bucket, so the store never overflows.

### 4.4 Forgetting and consolidation (plan 21 life cycle)
- **α fading = forgetting**: log α_eff = log α − λ(t)·(t − t_i), applied lazily with plan 21's global Λ(t) and per-splat
  Λ_i (one float per splat, as Λ_r for rows).
  - Infancy: λ ≈ 1e-3 per step (half-life 693 steps), matched to the address half-life. Under fading, splats that
    are not reinforced disappear.
- **Consolidation**: a splat is reinforced n_c times (about 3), or the plan-21 switch fires (stab ≥ 0.9 or the LR
  decay). It is then marked consolidated: no fading, μ / Σ frozen, its value moves with the decaying-to-Adagrad
  accumulator.
- **The ramp** follows plan 21: λ → 0 over the ramp, the maintenance interval grows from 250 to 1,000 steps, and the
  write rate drops to the surprising positions only.
- Measured (§9.5):
  - fading forgets early facts in proportion to λ (early recall at 1× drift: 0.86 → 0.54 at λ = 0.03 per write-batch);
  - one rehearsal at a half-drifted thought brings them back to 0.90-0.92, better than never fading. The re-write
    mostly reinforces and consolidates the old splat; a few spawn a second key along the drift path. Re-meeting a fact
    is the strongest drift-robustness lever we found besides the space itself.

---

## 5. Capacity and interference
- **Rows (today)**:
  - superposition: every row is a sum of the writes of all contexts that reach it;
  - capacity is set by the interference noise: the toy at 16K facts in 16K rows holds 0.73 at drift 0, with the
    early facts mostly overwritten (0.34), and 0.18 at 65K facts in 16K rows;
  - the 6-row × 3-head spread costs capacity.
- **Splats / kNN (direct writes)**:
  - one slot per written key, exact until the budget is reached;
  - then eviction (FIFO / weakest-α) forgets the oldest: 0.25 recall at 65K facts in 16K slots, i.e. exactly the
    surviving quarter, with the early facts at 0.
  - **Capacity is a slot budget, not a noise floor.** That makes forgetting explicit and controllable (fading /
    consolidation), which matches the plan-21 life cycle.
- **Slot budget at our scale**:
  - 1M splats × (65 coefficients + log α + birth + 256 values) × 4 B ≈ 1.3 GB, plus a 1.07 GB value-gradient buffer,
    plus 4 tables × 2 M pointers × 8 B ≈ 0.07-0.13 GB. That is the same order as today's V_sh + G_sh (2.14 GB).
  - At 2% of 8,192 positions × 4 keys (one per concept level) ≈ 650 new splats per step, 1M slots turn over in about
    1,600 steps. That fits the infancy half-life (≈ 700 steps). After consolidation, the write rate must drop to about
    10-50 per step (surprising positions only), which lasts 2·10⁴-10⁵ steps.
- **Interference** (`run_toy.py interf`, groups of 8 facts within ε of a centre, different values; §9.4):
  - tight groups (ε ≤ 0.1 relative) are separable at drift 0 by kNN and by splats with the conflict shrink;
  - nothing separates them under 1× drift, because the drift is larger than the group's internal spacing (recall
    0.2-0.34 for kNN, about 0.05-0.06 for rows).
  - This is the stability law again: near-duplicate thoughts with different answers are not recoverable after the
    thought moves.

---

## 6. Batched neighbour lookup (one call for all tokens / views / experts)
- **Brute force** (one GEMM against all splats, then top-k): exact, but at 98K queries per read × 1M splats × 32 dims it
  is 6.3 TFLOP per read, and the top-k over 1M columns per query cannot be fused in plain PyTorch. Ruled out.
- **Uniform grid / spatial hashing** (Teschner et al. 2003; Instant-NGP's hash grid):
  - each query visits 3^d neighbour cells: 6,561 at d = 8, 4·10⁷ at d = 16;
  - usable only at d ≤ 5-6, which §2 rules out for drift.
- **Product-quantised cells (IVF-PQ style, = product keys over cells)**: two codebooks of C centroids over the halves
  of d_m; a query scores 2C centroids, takes the top-m per half, then the top-kc pairs; each cell owns a bucket of
  `cap` splat pointers.
  - It is the existing pk_topk machinery with buckets instead of rows, and it is one bmm + three topk for all queries.
  - **Its weakness is the border problem**: `ivf_toy.py`, 65K facts, d = 32, 1× drift:

| index (d = 32, 65K facts) | candidates / query | recall@1 at lvl 1 (rot) | own cell probed |
|---|---|---|---|
| brute force | 65,536 | 0.974 | - |
| 1 table, C = 256², m 4, kc 8 | 9 | 0.347 | 0.35 |
| 1 table, C = 64², m 8, kc 32 | 514 | 0.756 | 0.76 |
| **4 tables (rotated), C = 256², kc 8 each** | **33** | **0.742** | 0.74 |
| 4 tables, keys filed in their top-2 cells | 64 | 0.856 | 0.86 |
| 4 tables, top-4 cells | 125 | 0.917 | 0.92 |
| 4 tables, C = 64², top-2 cells | 965 | 0.952 | 0.97 |

  - Scale check (`ivf2_toy.py rot 262144 512`, 262K facts at the same occupancy, C = 512²): 1 table 0.272;
    4 tables 0.651 / 0.777 / 0.858 at 33 / 65 / 128 candidates. It degrades slowly with N (65K → 262K: −0.06 to −0.09).
  - Warp drift (`log_ivf2_warp_65k.txt`, 65K facts, C = 256²): 1 table 0.342; 4 tables 0.773 / 0.893 / 0.946 at
    33 / 64 / 124 candidates.
- **GPU recipe (plain PyTorch, one call for all queries; `bench_gpu.py`)**:
  1. queries p = x W_view: one GEMM for all views;
  2. rotate into the T tables: one einsum;
  3. all centroid scores: one bmm (2T, Q, C);
  4. topk m: one op; pair scores + topk kc: three ops;
  5. pointer gather: one index op;
  6. de-duplication across tables: sort + compare + masked_fill;
  7. coefficient gather: one index_select of (Q, T·kc·cap, 2d+1);
  8. all Mahalanobis distances: one bmm;
  9. topk k under no_grad;
  10. re-score the k winners with grad: one gather + one bmm;
  11. weights: about 6 elementwise ops;
  12. embedding_bag for the values.

  - Experts / routes / passes are just more query rows (Q = N · V · …). A route- or pass-dependent view is a different
    P_v (a per-expert offset or rotation applied before step 2).
  - The buckets are rebuilt only at the write / maintenance time (a new splat replaces the weakest pointer of its cell
    in each table).

---

## 7. Performance (hard requirement: ≥ 10k tok/s, target 20k) — `perf_model.py`, `bench_gpu.py`

### 7.1 The budget
- Step = 2 micro-batches × 8,192 tokens = 16,384 tokens.
  - 10k tok/s ⇒ ≤ 1.64 s/step;
  - 20k tok/s ⇒ ≤ 0.82 s/step.
- The no-memory control runs at 20-23k tok/s ⇒ 0.71-0.82 s/step.
- **Memory budget per step**:
  - **≤ 0.8-0.9 s for 10k**, which is about today's whole memory overhead (plan 13: 0.94 s/step, ≈ 22.5K extra ops
    at ≈ 40 µs/op, mostly launch-bound);
  - **≤ 0.05-0.1 s for 20k**.
- The 20k budget leaves no room for the ConceptHier level heads + translators (75-105 ms of fp32 einsums + about
  85 ms of recompute in plan 13's table). **For 20k the views must come from the pass's thought directly** (one
  448 → V·d_m GEMM from a = [x_k-derived μ; p̄] or from x_k), not from the concept heads. If the concept heads stay for
  their own sake, they are not memory cost, but they eat the 20k budget.

### 7.2 Measured op counts (TorchDispatchMode on the real implementation, `bench_cpu.json`; size-independent)

| config | ops per read (fwd + bwd, all views) | ops per write (once per step) |
|---|---|---|
| splat point read, 4 tables (A / B) | 87 | 71 |
| splat, 1 table (A_T1) | 81 | 71 |
| splat ray, S = 8 (A_ray8) | 195 | 71 |
| splat 2 hops (A_hop2) | 176 | 71 |
| splat ray, S = 4 (B_ray4) | 171 | 71 |
| today's PK row read pattern (reference, without gates / W_o / margin / EMA path) | 58 | - |

- The gates + W_o per read add about 10 fwd / 15 bwd ops, as today.
- Per step = ops/read × reads per forward × 2 micro-batches × 1.33 (checkpoint recompute) + the write.

### 7.3 Per-step cost at B = 16, T = 512, 1M splats, d_v = 256 (analytic, `perf_model.py`)
Assumptions: 896 GB/s × 70% for gathers; 35 µs per op; fp32 coefficients.

| config | queries / read | cand. | GB / read (fwd + recomp + bwd) | GFLOP / read | ops / step | work ms / step | launch ms / step | **est. ms / step** | largest temp GB | saved for bwd GB | tables GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** (V = 12, every pass ×5, T = 4, 128 cand, k = 4, d_m 32) | 98,304 | 128 | 20.3 | 23.7 | 1,228 | 323 | 43 | **336** | 3.27 | 0.61 | 2.50 |
| **B** (V = 4, 2 reads, T = 4, 128 cand, k = 4, d_m 32) | 32,768 | 128 | 6.8 | 7.9 | 534 | 43 | 19 | **49** | 1.09 | 0.08 | 2.50 |
| A_T1 (1 table, 128 cand) | 98,304 | 128 | 17.1 | 14.1 | 1,148 | 272 | 40 | 284 | 3.27 | 0.61 | 2.45 |
| A_ray8 | 98,304 | 128 | 33.3 | 36.3 | 2,664 | 531 | 93 | 559 | 3.27 | 0.61 | 2.50 |
| A_hop2 | 98,304 | 128 | 38.7 | 38.4 | 2,412 | 617 | 84 | 643 | 3.27 | 0.61 | 2.50 |
| A_d16 | 98,304 | 128 | 13.5 | 12.2 | 1,228 | 215 | 43 | 228 | 1.66 | 0.32 | 2.36 |
| B_ray4 | 32,768 | 128 | 11.1 | 9.7 | 981 | 71 | 34 | 81 | 1.09 | 0.08 | 2.50 |

Reading it:
- **The point read is bandwidth-bound on the candidate-coefficient gather**: Q × candidates × (2d_m + 1) × 4 B,
  written to a temp and read back by the bmm.
  - bf16 coefficients for the no-grad scoring halve it (A ≈ 0.2 s/step); the k winners are re-scored in fp32.
  - Chunking the queries in 2-4 pieces caps the 3.27 GB temp at ≤ 1 GB, for +2-3 launches per chunk.
- **Launches are not the problem any more**: about 1,200 ops/step (A) or 530 (B), vs today's memory at about
  22,500 extra ops/step.
- **Today's memory path vs config A**: about 0.9 s → about 0.2-0.34 s. The whole step goes from 1.6-1.8 s
  (9-10k tok/s) to about 1.0-1.1 s (≈ 15-16k tok/s). That is if ConceptHier remains (≈ 0.15-0.2 s of it), otherwise
  ≈ 18k.
- **Config B**: ≈ 0.05 s/step ⇒ ≈ 0.80 s/step ⇒ ≈ 20k tok/s, if the views come from the thought directly (§7.1).

### 7.4 Which options fit
| option | cost multiplier (bytes / ops) | fits 10k (A-class) | fits 20k (B-class) | toy benefit |
|---|---|---|---|---|
| point read, 1 hop | 1 / 1 | yes | yes | (the baseline) |
| 2 hops | ×1.9 / ×2.0 | yes (0.64 s, tight with ConceptHier) | no (≈ 0.1 s) | negative |
| ray, S = 8 | ×1.6 / ×2.2 | yes (0.56 s) | no | neutral (unlearned direction) |
| ray, S = 4 | ×1.6 / ×2.0 | yes | borderline (81 ms) | neutral |
| V = 12 vs 4 queries per token | ×3 | yes | no | more heads, not more robustness |
| reads on every pass (5) vs 2 | ×2.5 | yes | no | untested (today's readdr-every 2 exists) |
| d_m 32 vs 16 | ×1.5 | yes | yes | d 32: +0.03-0.5 recall at 1× |
| T = 4 tables vs 1 | ×1.2 bytes, +6 ops | yes | yes | **+0.3-0.4 recall at 1×** |
| candidates 64 → 33 | ×0.6 | yes | yes | −0.1 recall (T = 4: 0.86 → 0.74) |
| k winners 4 → 8 | +3% | yes | yes | (k = 1 is the most drift-robust) |
| low-rank Σ (r = 1) | +d_m + 1 coefficients per slot, +2 ops | yes | yes | untested |
| μ / Σ gradients | +1 scatter of Q·k·(2d+1) per read | yes | no | not needed (rule-based writes) |

### 7.5 What fused kernels would buy (not available: no Triton / torch.compile on this machine)
1. **lookup + gather + kernel + topk** in one kernel: the candidate coefficients are streamed from the table once, with
   no temp. ×3 fewer bytes (A ≈ 0.11 s/step) and about 30 fewer launches per read.
2. **A sorted segment-reduce for the value backward** instead of index_add_ atomics. Plan 13 measured 120-180 ms for
   today's 6-column index_add_ backward, from hot-row atomics. The splat read has one index_add_ of Q·k rows; if hot
   splats contend, sort by slot + a segment sum (torch.sort + index_add_ on unique ids, +3 ops) is the plain-PyTorch
   fallback.
3. Without them, everything above is plain ops. The counts in §7.2 are what will run.

### 7.6 Configurations
- **≥ 10k tok/s (A)**:
  - 12 queries per token (4 levels × 3 heads, as today), d_m = 32, 1M splats;
  - T = 4 tables, C = 512 per half, cap 8, kc 4 → 128 candidates, k = 4;
  - a point read after every pass, 1 hop, no rays;
  - bf16 scoring, 2 query chunks.
  - Estimate: memory ≈ 0.2-0.34 s/step ⇒ ≈ 13-16k tok/s with ConceptHier kept.
- **≈ 20k tok/s (B)**:
  - 4 queries per token (one per level, views from the pass's thought), the same index;
  - reads after 2 passes only (e.g. passes 2 and 4, reusing the read for the next pass as `--mem-readdr-every 2`
    does);
  - writes 1% of positions; no ConceptHier on the memory path.
  - Estimate: ≈ 0.05 s/step ⇒ ≈ 20k tok/s.
- Both are estimates. **`bench_gpu.py` measures them** (and a PK-row reference) when the GPU is free:

      CUDA_VISIBLE_DEVICES=0 python -B speed_wip\splat\bench_gpu.py --device cuda --config all

  It prints per config: ops, ms per read (fwd + bwd), ms per write, peak VRAM, and the per-step extrapolation.

---

## 8. How it plugs into the brain (in place of the shared row pool; no code)
- **Where the views come from**:
  - the addressing input a = [μ_l; p̄] (448-d) per level, as today, or from x_k for the 20k config;
  - its EMA standardisation is replaced by **the frozen truncated whitening W_32 = Λ_32^{-½} V_32ᵀ** (plan 18), fitted
    once from about 200K-1M thoughts (refit only at the plan-21 switch);
  - the view of head h: p = R_{l,h} W_32 â_l, with R_{l,h} a fixed rotation (the heads differ only in their index
    tables), or a learned rotation frozen at the switch;
  - sg(a) on the addressing path as today (`--mem-addr-sg`): the thought is never constrained.
- **The read**: replaces the shared-pool part of `Memory.read_shared`. The gather of `rows` / `ω` becomes the splat
  lookup (§6), returning the slot ids and weights w (Q, k) that go into the same MemRead (embedding_bag + index_add_
  backward).
  - Gates g_lh (now with the opacity A as an extra feature), W_o, and the destinations (inject_{k+1}, slot 15) are
    unchanged.
  - The private expert memory stays as is (a later step could give experts their own view offsets into the same
    field).
- **The write**:
  - MemOpt.after_step keeps the row-Adagrad on the touched values (now splat slots).
  - New: (a) densify on the chosen positions of the last micro-batch (§4.2, value target from MemRead's gout);
    (b) α fading via plan 21's Λ;
  - every 250 steps: clone / split / prune (§4.3).
- **The life cycle (plan 21)**:
  - infancy: λ = 1e-3, all positions with high surprise may spawn;
  - switch / ramp: λ → 0, consolidation of reinforced splats, fewer spawns;
  - mature: frozen positions, values by Adagrad.
- **Probes (plan 12 §6)**: the stability measure becomes "self-recall of the probe contexts' own splats", the
  quantity that the stability law (§3.3) predicts.
- **Flags** (proposal): `--mem-splat 1` (instead of `--mem-shared`), `--splat-dm 32`, `--splat-tables 4`,
  `--splat-cand 4,8` (kc, cap), `--splat-k 4`, `--splat-spawn 0.02`, `--splat-fade 1e-3`, `--splat-reads 1,3`
  (passes).

---

## 9. The offline toy (`speed_wip/splat/`)

### 9.1 Setup
- **Thoughts**: a ∈ R^448 with spectrum λ_i ∝ i^{−1.5}. PR 5.3; top-1/5/20 variance shares 0.40 / 0.70 / 0.86, close to
  plan 16 T4 for â.
- **Facts**: N key → value pairs; a fact's thought is a random draw; its value is a random unit vector (d_v = 64).
  Written in order, in batches of 512.
- **Drift, calibrated to the measured 6K → 7K numbers** (8,192 positions, 10-NN in a-space; `calib.py`, `calib.json`):

| model | parameters | median ‖Δa‖/‖a‖ | 10-NN kept | centred cos | 10-NN after a LINEAR re-alignment | frozen PK row overlap S = 512 / 64 (`chk_rows.py`) |
|---|---|---|---|---|---|---|
| measured (plans 18 / 20) | - | 0.72-0.74 | 0.36-0.42 | 0.73-0.75 | no gain (0.199 vs 0.210 frozen) | 0.21-0.22 / 0.42 |
| rot: a' = R_θ a + σ noise | θ 0.662, σ 0.322 | 0.720 | 0.400 | 0.748 | 0.735 (strong gain) | 0.159 / 0.302 |
| warp: a' = a + A·F(a) + σ noise | A 0.525, ℓ 1.21, σ 0.1 | 0.721 | 0.400 | 0.942 | 0.512 (some gain) | 0.156 / 0.296 |

  - noise = √0.75 · (data-shaped) + √0.25 · (isotropic);
  - F(a) = a smooth random displacement field (random Fourier features, length scale ℓ), locally coherent and
    globally non-linear;
  - level t ∈ [0, 1] interpolates a + t(a' − a) (as plan 20 streamed the real drift); "2×" composes two independent
    1× drifts.
  - Neither model matches all three secondary numbers. rot gets the centred cos but is linearly undoable; warp is
    less linear but too coherent. Both are **slightly harsher than reality on the addressing** (PK row overlap 0.16 vs
    0.21), so the toy's drift is conservative. All main results are given for both.
- **Memories** (`memories.py`):
  - (a) **PK rows** as memory.py, with addr-learn 0: plan 18's best fixed addressing (pzca32 → random projection to
    128 → halves → spherical k-means S per half), 3 heads into one S² table, top-8 / top-6, τ 0.03. Gradient-written:
    1-3 steps per batch with the per-fact normalised (delta-rule) step, the best case of a gradient write.
  - (b) **kNN**: keys = the write-time thought in pzca32 (or raw a); top-1 or a soft top-8.
  - (c) **splat field**: d_m 8 / 16 / 32 (view 0 = the top d_m whitened PCs; views 1-2 = random d_m-dim subspaces of
    the whitened 32), 1 or 3 views in one shared field. Direct writes + reinforce + conflict shrink (floored) + prune;
    κ = 0.5. Reads: point / ray (S = 8, L = the median 1× drift length; direction from a fixed projection, or an
    ORACLE direction toward the write-time location), 1 or 2 hops (mean shift, hop 1 at 2× width).
  - (c') **gradient splats**: 65K fixed Gaussian centres (sampled thoughts) with values written by the delta rule
    (soft rows).
- **Metrics**: recall@1 = the decoded nearest stored value is the right fact; value cosine; early (first 10%) vs late
  (last 10%) facts; CPU µs per read (4 threads, 4,096 queries, brute-force candidate search: the CPU times compare the
  maths, not the GPU design).

### 9.2 Main table (16K facts; recall@1 at drift levels 0 / 0.5 / 1 (= 6K→7K) / 2, value cosine and early (first 10%) vs late (last 10%) facts at level 1)
Full rows: `log_main_rot.txt`, `log_main_warp.txt`. CPU µs per read: 4 threads, brute-force candidate search, so it
compares the maths, not the GPU design (§7).

| memory | rot: lvl 0 / 0.5 / 1 / 2 (recall@1) | rot lvl 1 cos | rot lvl 1 early / late | warp: lvl 0 / 0.5 / 1 / 2 | warp lvl 1 cos | warp lvl 1 early / late | CPU µs / read |
|---|---|---|---|---|---|---|---|
| **(a) PK rows, 65K rows, gradient (3 steps / batch)** | 0.99 / 0.82 / 0.24 / 0.04 | 0.29 | 0.16 / 0.26 | 0.99 / 0.75 / 0.22 / 0.12 | 0.28 | 0.19 / 0.31 | 10 |
| **(b) kNN-1, whitened 32** | 1.00 / 1.00 / 0.99 / 0.69 | 0.99 | 0.99 / 0.99 | 1.00 / 1.00 / 1.00 / 0.99 | 1.00 | 1.00 / 1.00 | 46 |
| (b) kNN soft top-8, whitened 32 | 0.76 / 0.61 / 0.39 / 0.16 | 0.46 | 0.39 / 0.38 | 0.76 / 0.71 / 0.58 / 0.49 | 0.51 | 0.56 / 0.57 | 52 |
| (b) kNN-1, raw a (448) | 1.00 / 1.00 / 0.98 / 0.65 | 0.98 | 0.97 / 0.98 | 1.00 / 1.00 / 0.65 / 0.59 | 0.65 | 0.66 / 0.63 | 93 |
| (c') gradient splats d16, 65K fixed centres | 1.00 / 0.87 / 0.37 / 0.08 | 0.39 | 0.22 / 0.52 | 1.00 / 0.66 / 0.19 / 0.11 | 0.24 | 0.13 / 0.23 | 243 |
| (c) splat d8, 1 view, point | 1.00 / 0.91 / 0.35 / 0.05 | 0.41 | 0.25 / 0.56 | 1.00 / 0.56 / 0.11 / 0.05 | 0.15 | 0.06 / 0.22 | 48 |
| (c) splat d8, 1 view, point, 2 hops | 0.97 / 0.74 / 0.21 / 0.03 | 0.29 | 0.09 / 0.42 | 0.97 / 0.53 / 0.12 / 0.06 | 0.16 | 0.06 / 0.25 | 100 |
| (c) splat d8, 1 view, ray | 1.00 / 0.91 / 0.35 / 0.05 | 0.41 | 0.24 / 0.55 | 1.00 / 0.57 / 0.11 / 0.05 | 0.15 | 0.06 / 0.22 | 379 |
| (c) splat d8, 1 view, ray, 2 hops | 0.93 / 0.65 / 0.18 / 0.02 | 0.25 | 0.08 / 0.37 | 0.92 / 0.50 / 0.11 / 0.05 | 0.16 | 0.07 / 0.23 | 383 |
| (c) splat d8, 1 view, ray, ORACLE direction | 1.00 / 0.92 / 0.39 / 0.07 | 0.45 | 0.28 / 0.58 | 1.00 / 0.64 / 0.21 / 0.13 | 0.27 | 0.15 / 0.32 | 327 |
| (c) splat d8, 3 views, point | 1.00 / 0.95 / 0.22 / 0.02 | 0.28 | 0.13 / 0.41 | 1.00 / 0.66 / 0.07 / 0.04 | 0.12 | 0.02 / 0.14 | 444 |
| (c) splat d8, 3 views, point, 2 hops | 1.00 / 0.84 / 0.13 / 0.01 | 0.20 | 0.05 / 0.31 | 1.00 / 0.67 / 0.11 / 0.06 | 0.15 | 0.04 / 0.25 | 893 |
| (c) splat d8, 3 views, ray | 1.00 / 0.96 / 0.24 / 0.02 | 0.28 | 0.15 / 0.42 | 1.00 / 0.69 / 0.10 / 0.05 | 0.13 | 0.04 / 0.19 | 3020 |
| (c) splat d8, 3 views, ray, 2 hops | 1.00 / 0.77 / 0.10 / 0.01 | 0.17 | 0.04 / 0.23 | 1.00 / 0.63 / 0.10 / 0.06 | 0.15 | 0.04 / 0.21 | 3452 |
| (c) splat d8, 3 views, ray, ORACLE direction | 1.00 / 0.97 / 0.29 / 0.03 | 0.34 | 0.20 / 0.50 | 1.00 / 0.82 / 0.25 / 0.17 | 0.31 | 0.19 / 0.37 | 3016 |
| (c) splat d16, 1 view, point | 1.00 / 1.00 / 0.91 / 0.37 | 0.82 | 0.86 / 0.98 | 1.00 / 0.98 / 0.42 / 0.24 | 0.48 | 0.29 / 0.67 | 51 |
| (c) splat d16, 1 view, point, 2 hops | 0.97 / 0.87 / 0.48 / 0.10 | 0.47 | 0.30 / 0.82 | 0.97 / 0.85 / 0.39 / 0.23 | 0.42 | 0.24 / 0.68 | 94 |
| (c) splat d16, 1 view, ray | 1.00 / 1.00 / 0.90 / 0.35 | 0.80 | 0.83 / 0.98 | 1.00 / 0.98 / 0.39 / 0.22 | 0.45 | 0.27 / 0.65 | 324 |
| (c) splat d16, 1 view, ray, 2 hops | 0.91 / 0.74 / 0.35 / 0.07 | 0.37 | 0.21 / 0.67 | 0.90 / 0.74 / 0.30 / 0.18 | 0.34 | 0.16 / 0.57 | 375 |
| (c) splat d16, 1 view, ray, ORACLE direction | 1.00 / 1.00 / 0.90 / 0.37 | 0.79 | 0.83 / 0.98 | 1.00 / 0.98 / 0.56 / 0.41 | 0.60 | 0.46 / 0.75 | 307 |
| (c) splat d16, 3 views, point | 1.00 / 1.00 / 0.92 / 0.30 | 0.81 | 0.89 / 0.99 | 1.00 / 1.00 / 0.60 / 0.33 | 0.60 | 0.42 / 0.85 | 510 |
| (c) splat d16, 3 views, point, 2 hops | 1.00 / 1.00 / 0.59 / 0.08 | 0.51 | 0.46 / 0.82 | 1.00 / 1.00 / 0.65 / 0.41 | 0.57 | 0.49 / 0.90 | 900 |
| (c) splat d16, 3 views, ray | 1.00 / 1.00 / 0.90 / 0.28 | 0.78 | 0.86 / 0.98 | 1.00 / 1.00 / 0.62 / 0.36 | 0.60 | 0.45 / 0.87 | 2643 |
| (c) splat d16, 3 views, ray, 2 hops | 1.00 / 0.97 / 0.44 / 0.05 | 0.42 | 0.30 / 0.72 | 1.00 / 0.99 / 0.56 / 0.34 | 0.50 | 0.38 / 0.81 | 3063 |
| (c) splat d16, 3 views, ray, ORACLE direction | 1.00 / 1.00 / 0.91 / 0.33 | 0.79 | 0.87 / 0.99 | 1.00 / 1.00 / 0.80 / 0.65 | 0.75 | 0.72 / 0.95 | 2627 |
| **(c) splat d32, 1 view, point** | 1.00 / 1.00 / 0.95 / 0.55 | 0.85 | 0.94 / 0.99 | 1.00 / 1.00 / 0.94 / 0.75 | 0.87 | 0.92 / 0.99 | 52 |
| (c) splat d32, 1 view, point, 2 hops | 0.88 / 0.70 / 0.34 / 0.08 | 0.38 | 0.19 / 0.67 | 0.88 / 0.80 / 0.53 / 0.41 | 0.49 | 0.34 / 0.87 | 105 |
| (c) splat d32, 1 view, ray | 1.00 / 1.00 / 0.94 / 0.49 | 0.81 | 0.94 / 0.98 | 1.00 / 1.00 / 0.93 / 0.72 | 0.84 | 0.89 / 0.99 | 314 |
| (c) splat d32, 1 view, ray, 2 hops | 0.69 / 0.46 / 0.19 / 0.04 | 0.27 | 0.11 / 0.47 | 0.69 / 0.58 / 0.34 / 0.24 | 0.38 | 0.20 / 0.67 | 363 |
| (c) splat d32, 1 view, ray, ORACLE direction | 1.00 / 1.00 / 0.95 / 0.51 | 0.82 | 0.93 / 0.98 | 1.00 / 1.00 / 0.96 / 0.85 | 0.89 | 0.95 / 0.99 | 291 |
| (c) splat d32, 3 views, point | 1.00 / 1.00 / 0.97 / 0.61 | 0.88 | 0.96 / 0.99 | 1.00 / 1.00 / 0.97 / 0.79 | 0.89 | 0.94 / 1.00 | 482 |
| (c) splat d32, 3 views, point, 2 hops | 0.99 / 0.90 / 0.46 / 0.08 | 0.45 | 0.31 / 0.74 | 0.99 / 0.97 / 0.81 / 0.62 | 0.63 | 0.71 / 0.95 | 907 |
| (c) splat d32, 3 views, ray | 1.00 / 1.00 / 0.96 / 0.56 | 0.85 | 0.94 / 0.99 | 1.00 / 1.00 / 0.96 / 0.77 | 0.87 | 0.92 / 1.00 | 2900 |
| (c) splat d32, 3 views, ray, 2 hops | 0.93 / 0.67 / 0.23 / 0.03 | 0.32 | 0.15 / 0.51 | 0.94 / 0.87 / 0.62 / 0.44 | 0.52 | 0.45 / 0.86 | 3084 |
| (c) splat d32, 3 views, ray, ORACLE direction | 1.00 / 1.00 / 0.97 / 0.60 | 0.86 | 0.96 / 0.99 | 1.00 / 1.00 / 0.98 / 0.90 | 0.92 | 0.97 / 1.00 | 3286 |

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


### 9.3 Width sweep
See §3.1 (the table is there). d16, 1 view, no shrink, σ = width × (the 3-NN spacing at birth, κ = 1).

### 9.4 Capacity and interference
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


### 9.5 Forgetting, consolidation, rehearsal (`log_fade_rot.txt`; d16, 1 view, 16K facts, fading λ per write batch of 512, 32 batches)

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


---

## 10. Honest assessment and recommendation

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


---

## 11. Related work
- **Kanerva's Sparse Distributed Memory** (Kanerva 1988, *Sparse Distributed Memory*, MIT Press).
  - Fixed random "hard locations" in a high-dim binary space; a write / read activates every location within a
    Hamming radius.
  - Splats are soft hard-locations with learned radii and opacity.
  - SDM's critical-distance analysis is the binary version of §3's stability law (a cue must lie within the basin of
    its own item).
- **Modern Hopfield networks / dense associative memory** (Krotov & Hopfield 2016,
  [arXiv:1606.01164](https://arxiv.org/abs/1606.01164); Ramsauer et al. 2020,
  [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)).
  - Softmax retrieval over stored patterns; one update step = attention.
  - Our multi-hop mean shift is a Hopfield iteration on splat centres. It converges to the dense centre of the basin,
    not to the drifted fact, which is why it hurt here.
- **Kernel / RBF memories**:
  - RBF networks (Broomhead & Lowe 1988; Moody & Darken 1989);
  - Nadaraya-Watson kernel regression;
  - kernel memory networks (Iatropoulos, Brea & Gerstner, NeurIPS 2022,
    [arXiv:2208.09416](https://arxiv.org/abs/2208.09416)), which unify SDM / Hopfield as kernel machines;
  - mean shift (Comaniciu & Meer 2002, TPAMI).
  - The point read is a Nadaraya-Watson estimate with per-centre bandwidth and an opacity prior. Gradient splats
    (c') are an RBF network.
- **Gaussian-mixture / volumetric rendering**:
  - 3D Gaussian Splatting with adaptive density control (Kerbl et al., SIGGRAPH 2023,
    [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)): densify every 100 iterations on the view-space positional
    gradient, clone small / split large (scale /1.6), prune α < 0.005, opacity reset every 3,000;
  - EWA splatting (Zwicker et al. 2001);
  - NeRF compositing (Mildenhall et al. 2020, [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)).
  - §1.3's sample-free form is the 3DGS ray-Gaussian integral; §4.3 is its ADC with our error / usage signals.
- **Retrieval memories**:
  - kNN-LM (Khandelwal et al. 2020, [arXiv:1911.00172](https://arxiv.org/abs/1911.00172)) and Memorizing Transformers
    (Wu et al. 2022, [arXiv:2203.08913](https://arxiv.org/abs/2203.08913)): a non-parametric kNN over stored
    (context, value) pairs. Memorizing Transformers note key staleness as the network trains, our drift problem;
  - RETRO (Borgeaud et al. 2021, [arXiv:2112.04426](https://arxiv.org/abs/2112.04426)).
  - **The toy says the splat field's drift-robustness is inherited from this family.**
- **Product-key memory** (Lample et al. 2019, [arXiv:1907.05242](https://arxiv.org/abs/1907.05242)); Memory Layers at
  Scale (Berges et al. 2024, [arXiv:2412.09764](https://arxiv.org/abs/2412.09764)). Today's rows. The cell index of §6
  reuses its lookup.
- **Indexing**:
  - FAISS IVF / PQ (Johnson, Douze & Jégou 2017, [arXiv:1702.08734](https://arxiv.org/abs/1702.08734); Jégou et al.
    2011, TPAMI);
  - multi-table / multi-probe LSH (Indyk & Motwani 1998; Lv et al. 2007);
  - spatial hashing (Teschner et al. 2003); Instant-NGP hash grids (Müller et al. 2022,
    [arXiv:2201.05989](https://arxiv.org/abs/2201.05989)).
- **Distance concentration**: Beyer et al. 1999, "When is nearest neighbor meaningful?" (ICDT); Aggarwal, Hinneburg &
  Keim 2001 (ICDT).

---

## 12. Risks
1. **The toy is iid-Gaussian facts.**
   - Real contexts are dense on a document manifold: neighbouring tokens sit much closer than random facts, so the
     effective δ/r_NN is larger.
   - The calibration fixes the drift against 8K-point neighbourhoods, but the self-recall of a position's own key on
     REAL thoughts was never measured, and the captures are in the Recycle Bin.
   - **This is the go / no-go number** (§10). It needs only two checkpoints 1,000 steps apart and a capture of a.
2. **The value target v\*.** Direct writes need a target. Option A (the delta target from gout) is plausible but
   untested. Gradient-only splats (c') behave like rows (they inherit the PK fragility: 0.19-0.37 at 1×).
3. **The index border problem.** The multi-table index recovers about 0.65-0.9 of brute-force robustness, not all
   of it, and degrades slowly with N (0.74 → 0.65 from 65K to 262K facts at T = 4, 33 candidates).
4. **Views must be frozen** after the whitening fit. A learned view moves the reads away from the keys written
   earlier (the stale-key effect on the read side). A learned linear re-alignment can track a linear drift (the rot
   model), but plan 20 found the real drift has no useful linear part.
5. **Slot turnover.** About 650 spawns per step fill 1M slots in about 1,600 steps. Too slow a fade or too many spawns
   evict useful splats; too fast a fade forgets (§9.5). The plan-21 ramp has to set the write rate as well as λ.
6. **Hot-slot atomics** in the value backward (plan 13's 120-180 ms for rows). Mitigation: a sorted segment sum.
7. **VRAM**: the 3.27 GB candidate temp of config A must be chunked or put into bf16. Tables are 2.5 GB, like today.
8. **The splat-specific features (rays, hops, wide kernels, shared multi-view fields) did not help in the toy.**
   Building them first would add cost (×2 per read) without a measured benefit. They become interesting only with a
   learned direction / hop that the toy cannot test (e.g. a ray toward "where this context was written 1,000 steps
   ago", trained by the retrieval error).
9. **The toy's numbers are recall of exact facts**, not LM loss. The memory's value to the LM is unproven either way
   (plans 16 / 18 caveats).

---

## Files (`C:\Eitan\Python\NewBrain\speed_wip\splat\`)
- `common.py` — thoughts, drift models, whitening, metrics
- `calib.py` → `calib.json`
- `chk_rows.py` → `chk_rows.json`
- `memories.py` — PKRows, KNN, SplatField, SplatGrad
- `run_toy.py <main|width|dim|cap|interf|fade> <rot|warp>` → `res_*.json`, `log_*.txt`
  (`v1/` = the first run, before the log-domain kernel and the shrink floor)
- `dimtheory.py` → `dimtheory.json`
- `ivf_toy.py` / `ivf2_toy.py` → `res_ivf*_*.json`, `log_ivf2_rot_262k.txt` (the index tests)
- `bench_gpu.py` — the plain-PyTorch GPU read / write; CPU smoke → `bench_cpu.json`; run it on the GPU when it is free
- `perf_model.py` → `perf_model.json`

---

## 13. Addendum: the life cycle for this memory — see plan_v2/24_consolidation_review.md
- Plan 21's global switch + freeze adds nothing for a key memory under drift (toy: 0.78 vs 0.77 for constant fading).
- What keeps early facts is renewing the KEYS:
  - re-encode on a confirmed recall: 0.92;
  - replay re-registration from the stored source context, through one replay window per micro-batch: 1.00, and the
    only mechanism that survives the harsh drift.
- Per-splat maturity is kept for value plasticity and for re-registration eligibility only.
- Cost: < 3 ms per step for re-encoding; ≈ 2-6% throughput for the replay row. This replaces §4.4's
  "consolidated = frozen μ" rule: a consolidated splat keeps its value protected but its key is re-registered.
