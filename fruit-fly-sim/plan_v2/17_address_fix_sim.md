# 17. Address fixes, simulated offline on the captured thoughts (no code, no training changes) — 2026-10-24

Follows plan_v2/16. Scripts and raw JSON: `speed_wip/addr_inv/` (`sim.py`, `sim2.py`, `rank.py`, `sim_p3.json`,
`sim_p1.json`, `sim2_p3.json`). CPU, 4 threads, about 35 min in total. `runs/wiki_mem_12k/final.pt` did not appear while
I worked (the run was at step 7000), so the data are the step-6000 and step-7000 captures of the fixed val batch.


> **CORRECTION (later on 2026-10-24, see plan 20): the dynamic key results below are invalid.**
>
> The read-pull proxy (`attract`) pulls all 8 selected keys toward the same queries, so the keys converge into
> near-duplicates. That affects f·P, B, C, D and F as simulated here, which all build on it:
> - 330-510 of 512 keys have a twin at cosine > 0.99, leaving only 43-175 effective keys per codebook;
> - today's REAL trained keys have about 0 such twins (507 effective keys), so real training does not clump like the proxy.
>
> The "distinct rows" therefore counted duplicate rows. Re-scored with keys of cos > 0.95 merged (`check17b.py`; ema input,
> plain query, p3, levels 16 + 128, 3 heads):
>
> | keys (ema input, plain, p3; L16+L128, 3 heads) | n_eff_keys | dist | dist_dedup | stab_n03 | stab_n03_dedup | drift | drift_dedup |
> |---|---|---|---|---|---|---|---|
> | trained_static | 507 | 1840 | 1676 | 0.943 | 0.948 | 0.334 | 0.347 |
> | random_static | 512 | 2910 | 2910 | 0.924 | 0.924 | 0.205 | 0.205 |
> | kmeans_top1+restart | 495 | 3114 | 3025 | 0.928 | 0.930 | 0.212 | 0.221 |
> | attract | 175 | 3275 | 555 | 0.908 | 0.974 | 0.166 | 0.334 |
> | fP | 49 | 3556 | 405 | 0.900 | 0.978 | 0.137 | 0.359 |
> | C_restart | 82 | 3455 | 612 | 0.906 | 0.973 | 0.155 | 0.318 |
> | D_sinkhorn | 47 | 3509 | 413 | 0.903 | 0.979 | 0.142 | 0.367 |
> | F_gumbel | 43 | 3517 | 366 | 0.901 | 0.978 | 0.145 | 0.351 |
>
> **Corrected reading:**
> - Distinct keys (spherical k-means with top-1 assignment + dead restart, or even random static keys) give ×1.7-1.8 the real
>   spread of today's keys (de-duplicated 3,025 / 2,910 vs 1,676 per 1K) at 3% noise stability 0.93 vs 0.95. The cost is
>   6K→7K drift stability 0.21-0.22 vs 0.35, the finer-cell trade-off.
> - The ranking of f·P / C / D / F could not be established with this proxy.
> - The recommendation changes from "f·P or C" to "keep the keys distinct and restart the dead ones".
>   - The restart itself is fine; it only re-seeds unused keys.
>   - f·P and Sinkhorn must be re-tested with a key-update model that does not collapse the selected keys into one (e.g.
>     top-1 or softmax-weighted pulls), or in a short real run.
> - The whitening, query-BN and half-rotation conclusions (input-side, static keys) are unaffected.

## Setup
- **Thoughts**: the captured addressing inputs a (the read after pass 3 = "p3"; pass 1 as a check) of the 6K and 7K
  checkpoints, the same 16,384 positions. The 6K W_q (EMA) is used throughout.
- **Out-of-sample**: every statistic (whitening, BN), re-seed and key-training step is fit on **sequences 0-15 of the 6K
  thoughts**. Everything is scored on **sequences 16-31** (other documents):
  - on the 6K thoughts;
  - on the **7K thoughts with the 6K-fitted addressing** (`dist7`);
  - on the 6K thoughts plus Gaussian noise at 3% / 10% of ‖a‖.
- **Dynamic key methods** (attract, fP, B, C, D, F in the extension):
  - start from **random unit keys** (a fresh run);
  - stream 2,000 batches of 512 fit queries per codebook (about 125 passes over the fit set);
  - use Adam lr 3e-3.
  - The read's gradient on the keys is approximated by "pull the selected keys toward their queries" (maximise the
    matched scores), the method called `attract`, which is today's dynamics.
  - Scoring always uses **plain top-m selection** (the post-infancy / post-freeze addressing).
  - The first simulation (`sim.py`) started the key fits from the trained 6K keys (100 steps of 4,096); its rows are marked
    "(from trained)".
- **Scorecard columns**
  - `dist`: distinct rows per 1,000 positions per head (6,000 draws; ideal 5,932).
  - `top1`: the share of the reads on the top 1% of rows (uniform: about 0.1 for this sample).
  - `dead`: sub-keys never in a top-8 on the scored half.
  - `xdoc`: row-set overlap (shared rows / 6) of positions in different documents.
  - `lag1/16/256`: overlap of same-document neighbours.
  - `stab_n03` / `stab_n10`: top-6 row overlap under 3% / 10% noise.
  - `stab_drift`: top-6 row overlap of the same positions between the 6K and the 7K thoughts, with the same fixed addressing.
- **Averaging**: the scorecard averages levels 16 and 128 (L0, L3) and the 3 heads. The first simulation's all-level tables
  are in the appendix.
- **Methods**
  - Input norm:
    - `ema` = today's per-coordinate EMA norm;
    - `zca` = full whitening;
    - `pow0.5` = eigenvalue power ½;
    - `shrinkλ` = Σ_λ = (1−λ)Σ + λ·tr(Σ)/d·I, then whitened.
  - Query: `BN` = BatchNorm of the raw 128-d query (running stats from the fit half) before the split (A).
  - Halves:
    - `whiten_q` = whiten the 128-d q, then split;
    - `rot_q` = a fixed random orthogonal rotation, then split.
  - Keys:
    - `fP` = read pull + 0.1·S·Σ f_i P_i (τ_u = 0.1); λ_u was tuned (0.1 / 1 / 3), and 0.1 gave the most spread;
    - `B` = EMA k-means (decay 0.99, no gradient);
    - `C` = uncapped restart of keys unused for 100 steps to a random batch query + N(0, 0.05);
    - `D` = indices from a Sinkhorn-balanced score matrix (ε = 0.05, 3 iterations);
    - `F` = Gumbel noise (T = 0.05) on the selection scores;
    - `kpp_reseed` = k-means++-style re-seeding of the dead keys.
  - (The LSH / FSQ addressing is in plan 18, not here.)

## The main result: a spread vs. stability frontier, and keys are the only free lunch
1. **Restoring the keys is the large, cheap gain, and it is nearly method-independent.** Every method that keeps dead keys
   at about 0 lands at 3,400-3,600 distinct rows per 1K (×1.85-1.95 today's 1,857). Dead keys go from 31% to about 0.2%,
   and the top-1% read share from 0.95 to 0.67-0.69. Document separation stays perfect (xdoc 0.001).
   - Today's dynamics from scratch (`attract`) reach 3,293 but keep 15% dead keys.
   - B (EMA k-means) is the same: 3,218, 19% dead. It moves keys to the centroid of whoever already selects them, which is
     rich-get-richer again.
   - f·P, C, D and F are within ±3% of each other.
2. **Whitening the input buys at most about 10% more spread at a catastrophic stability cost.**
   - ZCA + any keys: dist 3,650-3,900, but noise-3% stability 0.47-0.50 (vs 0.90), noise-10% 0.09-0.12, drift 0.016-0.02.
     Almost every row changes between 6K and 7K, and even 3% input noise moves half the rows.
   - It also loses out-of-sample spread: dist7 falls 3,900 → 3,150.
   - Partial whitening (pow0.5, shrink 0.1 / 0.3) sits between the two: with restored keys, −2…+3% spread vs `ema`, with
     noise-10% stability −0.05…−0.08 and drift −0.01…−0.02. So it does not pay.
   - Whitening amplifies the weak directions of the thought, which are mostly noise and drift.
3. **BatchNorm on the query (A) gives nothing**: ±0-5%, slightly worse with ema. **Decorrelating the halves** gives nothing
   once the keys are refit: a random rotation is worse (−10%), and whitening q costs stability like input whitening (ema +
   whiten_q + fP: 3,590 with drift 0.117 vs 0.137 for plain).
4. **The price of resolution is unavoidable.** Within the stable family, more spread means finer cells and so less
   persistence:
   - today's trained keys: drift 0.33, noise-10% 0.83, lag1 0.27;
   - restored keys: drift 0.14-0.16, noise-10% 0.71-0.72, lag1 0.10-0.11.

   This is the same trade-off for every restoring method. The stable family keeps n03 ≥ 0.90, which is the regime where a
   context reliably reads back what it wrote. Note that the real 1,000-step thought drift already changes about 2/3 of the
   rows even with today's addressing (drift 0.33), so thoughts moving under a frozen addressing (after the freeze) is a
   separate, existing issue.
5. The pass-1 run confirms this. There are more dead keys on p1 today (46%). ema/plain + fP_opt: 1,499 → 3,610, dead
   0.46 → 0.014, n03 0.947 → 0.916, drift 0.35 → 0.15. ZCA again collapses stability (n03 0.47, drift 0.015).

Per level, for the recommended setting vs today (p3; extension simulation):

| setting | level | dist | dist7 | top1 | dead | xdoc | lag1 | lag16 | lag256 | n03 | n10 | drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| today: ema / plain / trained keys | 16 | 1827 | 1771 | 0.947 | 0.281 | 0.005 | 0.324 | 0.225 | 0.118 | 0.955 | 0.867 | 0.368 |
| today | 128 | 1888 | 1715 | 0.946 | 0.337 | 0.005 | 0.222 | 0.168 | 0.089 | 0.931 | 0.792 | 0.300 |
| ema / plain / f·P | 16 | 3411 | 3332 | 0.705 | 0.003 | 0.001 | 0.126 | 0.063 | 0.029 | 0.920 | 0.759 | 0.160 |
| ema / plain / f·P | 128 | 3786 | 3536 | 0.640 | 0.002 | 0.001 | 0.077 | 0.044 | 0.018 | 0.881 | 0.656 | 0.113 |
| ema / plain / C restart | 16 | 3331 | 3249 | 0.716 | 0.001 | 0.001 | 0.142 | 0.077 | 0.030 | 0.925 | 0.774 | 0.176 |
| ema / plain / C restart | 128 | 3634 | 3368 | 0.665 | 0.003 | 0.001 | 0.085 | 0.054 | 0.025 | 0.886 | 0.673 | 0.135 |

## Recommendation for a fresh 12K run
Keep the **per-coordinate EMA input norm, the plain query and the plain split**. Fix the **keys**:

**1st: C, the uncapped dead-key restart** (CVQ-VAE style), on the existing balancing machinery.
- **Score**: dist 3,482 (×1.88), dead 0.2%, top-1% 0.69, xdoc 0.001; n03 0.906, n10 0.724, drift 0.155. Among the
  zero-dead methods it is the most stable, and within 3% of the most spread.
- **Cost**: about 0 s/step. One balancing event every 100 steps instead of 1,000. The re-seeds only write keys, their EMA
  copies, Adam moments and S_acc bands; the splits stay capped. Under 0.001 s/step amortised.
- **Implementation**: mostly flags (`--mem-balance-every 100`, `--mem-balance-dead-max 1.0` (uncapped dead), keep
  `--mem-balance-keep 1`, which is exactly "unused since the last event"). One small change is needed: NRES = 64 recent
  query halves per codebook is too few to seed hundreds of dead keys, so they would be clones of 64 points. Raise it to
  about 1,024, or draw from the current batch. That is about 10 lines.
- **Compatibility**: full. balance() already edits K and K_EMA together and resets the optimizer moments; the online
  weights are untouched; it stops at the freeze.

**2nd (or together with C): f·P, the usage-balance loss of plan 15 (a), λ_u ≈ 0.1 relative to the read pull.**
- **Score**: the best spread of the stable family. dist 3,598 (×1.94), dead 0.2%, top-1% 0.67; n03 0.900, n10 0.708,
  drift 0.137.
- **Cost**: plan 15's estimate, +0.01-0.02 s/step (one N×S GEMM with backward per level / head / half, on the margin pass
  and 1,024 tokens). That is about 1% of the ≈1.8 s step.
- **Implementation**: about 60-80 lines plus tests (plan 15 lists them).
- **Compatibility**: full. f comes from the EMA indices (the `pend` counts), P from the online scores; the gradient goes to
  W_q / keys only; it is off at the freeze.
- C and f·P are complementary: f·P shapes keys continuously, and C catches whatever still dies. Their joint effect was not
  simulated.

**Also good, but not preferred:**
- **D (Sinkhorn indices in infancy)**: 3,531, dead 0.1%, n03 0.903, drift 0.142.
  - Cost: about +0.01-0.03 s/step, because 3 normalisations over every N×S score buffer of every fresh read.
  - Implementation: about 30 lines in `_addr`.
  - Compatibility: it is index-side only, so it fits the EMA-index / online-weight split. But each token's index depends
    on the batch (worse for the probes and reuse diagnostics), and the indices change distribution at the freeze switch
    (plain top-k afterwards). The simulation scores exactly that post-switch state and finds it fine.
- **F (Gumbel selection noise, T = 0.05)**: 3,545, n03 0.901, drift 0.145.
  - Cost: about 0 s/step, about 5 lines.
  - Compatibility: index-side.
  - Downside: it writes rows at noisy addresses during infancy, which the offline simulation cannot score.

**Not recommended:**
- **B (EMA k-means keys)**: no restoring force, so it keeps 19% dead keys. It also replaces the key gradient, which breaks
  the online / EMA key split and the margin loss.
- **A (query BN)**: no gain.
- **Whitening (W)** and the W + A / B / C / D combinations: stability collapse (n03 ≈ 0.49, drift ≈ 0.02).
- **Partial whitening**: −2…+3% spread at a stability cost.
- **Rotation or whitening of q before the split**: no gain, or a stability loss.
- **k-means++ re-seeding of the trained keys only**: ×1.35, and 22% of the keys stay dead.

**Caveats**
- The simulation freezes the query distribution. In a real run the thought and W_q co-adapt with the keys; the 16
  analysis showed that the drift of the query distribution is what stranded today's keys, and C and f·P are precisely
  the mechanisms that track a moving distribution.
- The read pull is a proxy for the LM gradient.
- Today's EMA-key lag (indices from K_EMA with decay 0.999) is not modelled. It slows all the dynamic methods equally.
- The "ideal" 5,932 distinct rows is not reachable while keeping content: positions in the same document legitimately
  share rows. The stable methods plateau at about 3,400-3,600 on this batch.

## Scorecard: every combination, ranked (levels 16 + 128, 3 heads, pass p3)
"Stable" = n03 ≥ 0.85 and drift ≥ 0.10; within each group the rows are sorted by `dist`. "(from trained)" = the first
simulation, whose key fits start from the trained 6K keys. Other rows start from random keys, or use the trained or random
keys without fitting (`_static`).

| rank | input / query or halves / keys | stable? | dist | dist7 | top1 | dead | xdoc | lag1 | lag16 | lag256 | stab_n03 | stab_n10 | stab_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ema / plain / fP | yes | 3598 | 3434 | 0.673 | 0.002 | 0.001 | 0.102 | 0.054 | 0.024 | 0.900 | 0.708 | 0.137 |
| 2 | ema / whiten_q / fP_opt (from trained) | yes | 3590 | 3382 | 0.641 | 0.003 | 0.002 | 0.155 | 0.108 | 0.075 | 0.871 | 0.627 | 0.117 |
| 3 | shrink0.1 / plain / F_gumbel | yes | 3566 | 3344 | 0.650 | 0.001 | 0.002 | 0.124 | 0.090 | 0.055 | 0.869 | 0.630 | 0.120 |
| 4 | shrink0.1 / BN / F_gumbel | yes | 3549 | 3304 | 0.653 | 0.002 | 0.002 | 0.129 | 0.094 | 0.056 | 0.871 | 0.635 | 0.127 |
| 5 | ema / plain / F_gumbel | yes | 3545 | 3358 | 0.682 | 0.001 | 0.001 | 0.106 | 0.060 | 0.024 | 0.901 | 0.710 | 0.145 |
| 6 | shrink0.1 / plain / fP | yes | 3534 | 3267 | 0.658 | 0.002 | 0.002 | 0.128 | 0.090 | 0.054 | 0.878 | 0.651 | 0.125 |
| 7 | ema / plain / D_sinkhorn | yes | 3531 | 3409 | 0.683 | 0.001 | 0.001 | 0.102 | 0.059 | 0.025 | 0.903 | 0.716 | 0.142 |
| 8 | shrink0.1 / BN / D_sinkhorn | yes | 3516 | 3256 | 0.661 | 0.001 | 0.002 | 0.128 | 0.089 | 0.050 | 0.881 | 0.655 | 0.124 |
| 9 | shrink0.1 / BN / fP | yes | 3513 | 3258 | 0.661 | 0.002 | 0.002 | 0.131 | 0.092 | 0.055 | 0.879 | 0.654 | 0.128 |
| 10 | shrink0.1 / plain / D_sinkhorn | yes | 3509 | 3262 | 0.661 | 0.002 | 0.002 | 0.134 | 0.095 | 0.057 | 0.880 | 0.656 | 0.127 |
| 11 | shrink0.1 / plain / C_restart | yes | 3495 | 3288 | 0.664 | 0.002 | 0.002 | 0.136 | 0.097 | 0.061 | 0.879 | 0.656 | 0.130 |
| 12 | shrink0.1 / plain / attract | yes | 3489 | 3269 | 0.665 | 0.014 | 0.002 | 0.138 | 0.099 | 0.065 | 0.880 | 0.657 | 0.133 |
| 13 | ema / plain / C_restart | yes | 3482 | 3308 | 0.690 | 0.002 | 0.001 | 0.114 | 0.065 | 0.027 | 0.906 | 0.724 | 0.155 |
| 14 | shrink0.1 / plain / B_ema_kmeans | yes | 3480 | 3225 | 0.669 | 0.024 | 0.002 | 0.138 | 0.101 | 0.061 | 0.882 | 0.663 | 0.137 |
| 15 | shrink0.1 / BN / B_ema_kmeans | yes | 3474 | 3235 | 0.667 | 0.021 | 0.002 | 0.141 | 0.100 | 0.061 | 0.884 | 0.664 | 0.141 |
| 16 | shrink0.1 / BN / C_restart | yes | 3474 | 3241 | 0.666 | 0.003 | 0.003 | 0.143 | 0.103 | 0.066 | 0.881 | 0.660 | 0.139 |
| 17 | shrink0.1 / BN / attract | yes | 3468 | 3235 | 0.667 | 0.012 | 0.002 | 0.143 | 0.102 | 0.065 | 0.882 | 0.660 | 0.138 |
| 18 | shrink0.1 / whiten_q / fP_opt (from trained) | yes | 3450 | 3094 | 0.665 | 0.003 | 0.002 | 0.175 | 0.128 | 0.089 | 0.854 | 0.581 | 0.106 |
| 19 | ema / BN / F_gumbel | yes | 3445 | 3320 | 0.698 | 0.001 | 0.001 | 0.108 | 0.062 | 0.027 | 0.900 | 0.707 | 0.147 |
| 20 | shrink0.1 / rot_q / trained (from trained) | yes | 3443 | 2860 | 0.682 | 0.007 | 0.002 | 0.131 | 0.091 | 0.052 | 0.888 | 0.676 | 0.130 |
| 21 | shrink0.3 / plain / random (from trained) | yes | 3442 | 3038 | 0.689 | 0.014 | 0.001 | 0.122 | 0.079 | 0.040 | 0.897 | 0.703 | 0.137 |
| 22 | shrink0.1 / rot_q / kpp_reseed (from trained) | yes | 3435 | 2857 | 0.683 | 0.007 | 0.002 | 0.131 | 0.091 | 0.052 | 0.888 | 0.676 | 0.130 |
| 23 | shrink0.1 / BN / random_static | yes | 3430 | 2964 | 0.685 | 0.010 | 0.001 | 0.129 | 0.086 | 0.046 | 0.886 | 0.673 | 0.125 |
| 24 | shrink0.1 / rot_q / random (from trained) | yes | 3426 | 2918 | 0.689 | 0.006 | 0.002 | 0.128 | 0.089 | 0.051 | 0.887 | 0.675 | 0.120 |
| 25 | ema / BN / fP | yes | 3413 | 3303 | 0.705 | 0.002 | 0.001 | 0.114 | 0.064 | 0.029 | 0.903 | 0.717 | 0.148 |
| 26 | shrink0.3 / whiten_q / fP_opt (from trained) | yes | 3411 | 3150 | 0.672 | 0.005 | 0.002 | 0.181 | 0.133 | 0.095 | 0.867 | 0.618 | 0.125 |
| 27 | shrink0.1 / plain / random_static | yes | 3400 | 2944 | 0.690 | 0.012 | 0.001 | 0.130 | 0.088 | 0.048 | 0.887 | 0.675 | 0.124 |
| 28 | ema / BN / D_sinkhorn | yes | 3387 | 3320 | 0.708 | 0.001 | 0.001 | 0.114 | 0.063 | 0.030 | 0.906 | 0.720 | 0.151 |
| 29 | shrink0.3 / plain / fP_opt (from trained) | yes | 3386 | 3243 | 0.690 | 0.001 | 0.002 | 0.131 | 0.089 | 0.050 | 0.889 | 0.683 | 0.145 |
| 30 | shrink0.3 / rot_q / trained (from trained) | yes | 3347 | 2899 | 0.702 | 0.010 | 0.002 | 0.135 | 0.090 | 0.051 | 0.899 | 0.707 | 0.155 |
| 31 | ema / BN / C_restart | yes | 3334 | 3228 | 0.715 | 0.002 | 0.002 | 0.123 | 0.071 | 0.033 | 0.907 | 0.726 | 0.164 |
| 32 | shrink0.3 / rot_q / random (from trained) | yes | 3332 | 2940 | 0.709 | 0.009 | 0.002 | 0.131 | 0.090 | 0.050 | 0.899 | 0.707 | 0.145 |
| 33 | pow0.5 / plain / fP_opt (from trained) | yes | 3305 | 3128 | 0.710 | 0.004 | 0.002 | 0.131 | 0.087 | 0.046 | 0.867 | 0.619 | 0.109 |
| 34 | ema / plain / attract | yes | 3293 | 3132 | 0.728 | 0.150 | 0.001 | 0.124 | 0.075 | 0.034 | 0.908 | 0.733 | 0.166 |
| 35 | pow0.5 / rot_q / trained (from trained) | yes | 3251 | 2676 | 0.722 | 0.011 | 0.002 | 0.141 | 0.093 | 0.051 | 0.872 | 0.631 | 0.108 |
| 36 | shrink0.3 / rot_q / kpp_reseed (from trained) | yes | 3235 | 2853 | 0.720 | 0.009 | 0.002 | 0.140 | 0.095 | 0.055 | 0.900 | 0.710 | 0.159 |
| 37 | ema / plain / B_ema_kmeans | yes | 3218 | 3084 | 0.741 | 0.190 | 0.001 | 0.129 | 0.075 | 0.032 | 0.911 | 0.736 | 0.171 |
| 38 | ema / BN / attract | yes | 3213 | 3122 | 0.740 | 0.134 | 0.002 | 0.128 | 0.078 | 0.036 | 0.909 | 0.730 | 0.172 |
| 39 | pow0.5 / rot_q / kpp_reseed (from trained) | yes | 3152 | 2584 | 0.739 | 0.011 | 0.002 | 0.146 | 0.097 | 0.052 | 0.874 | 0.636 | 0.115 |
| 40 | ema / BN / B_ema_kmeans | yes | 3143 | 3029 | 0.754 | 0.175 | 0.002 | 0.139 | 0.081 | 0.041 | 0.912 | 0.738 | 0.181 |
| 41 | ema / rot_q / fP_opt (from trained) | yes | 3107 | 3030 | 0.765 | 0.004 | 0.002 | 0.132 | 0.076 | 0.035 | 0.914 | 0.745 | 0.176 |
| 42 | shrink0.1 / rot_q / fP_opt (from trained) | yes | 3001 | 2813 | 0.754 | 0.004 | 0.004 | 0.177 | 0.132 | 0.089 | 0.892 | 0.692 | 0.172 |
| 43 | ema / rot_q / trained (from trained) | yes | 2970 | 2705 | 0.783 | 0.037 | 0.001 | 0.151 | 0.095 | 0.040 | 0.923 | 0.776 | 0.208 |
| 44 | ema / plain / random_static | yes | 2936 | 2728 | 0.789 | 0.050 | 0.001 | 0.154 | 0.097 | 0.043 | 0.924 | 0.775 | 0.205 |
| 45 | ema / BN / random_static | yes | 2912 | 2770 | 0.787 | 0.042 | 0.002 | 0.159 | 0.101 | 0.046 | 0.924 | 0.773 | 0.210 |
| 46 | shrink0.3 / rot_q / fP_opt (from trained) | yes | 2905 | 2828 | 0.778 | 0.005 | 0.004 | 0.172 | 0.122 | 0.076 | 0.902 | 0.713 | 0.183 |
| 47 | pow0.5 / rot_q / fP_opt (from trained) | yes | 2892 | 2766 | 0.785 | 0.005 | 0.003 | 0.171 | 0.117 | 0.070 | 0.878 | 0.650 | 0.144 |
| 48 | ema / rot_q / random (from trained) | yes | 2859 | 2656 | 0.801 | 0.038 | 0.002 | 0.159 | 0.103 | 0.045 | 0.925 | 0.780 | 0.211 |
| 49 | ema / plain / kpp_reseed (from trained) | yes | 2524 | 2386 | 0.854 | 0.222 | 0.002 | 0.211 | 0.138 | 0.070 | 0.936 | 0.811 | 0.273 |
| 50 | shrink0.1 / plain / kpp_reseed (from trained) | yes | 2335 | 2083 | 0.869 | 0.145 | 0.005 | 0.235 | 0.175 | 0.107 | 0.912 | 0.744 | 0.238 |
| 51 | shrink0.3 / plain / kpp_reseed (from trained) | yes | 2328 | 2124 | 0.874 | 0.172 | 0.005 | 0.239 | 0.174 | 0.108 | 0.922 | 0.768 | 0.262 |
| 52 | pow0.5 / plain / kpp_reseed (from trained) | yes | 2302 | 2053 | 0.880 | 0.198 | 0.005 | 0.242 | 0.175 | 0.104 | 0.906 | 0.721 | 0.217 |
| 53 | shrink0.1 / BN / trained_static | yes | 2227 | 1976 | 0.884 | 0.140 | 0.006 | 0.250 | 0.187 | 0.118 | 0.915 | 0.750 | 0.253 |
| 54 | shrink0.1 / plain / trained_static | yes | 2206 | 1965 | 0.887 | 0.149 | 0.006 | 0.249 | 0.186 | 0.118 | 0.915 | 0.749 | 0.252 |
| 55 | shrink0.3 / plain / trained (from trained) | yes | 2114 | 1950 | 0.902 | 0.172 | 0.006 | 0.258 | 0.190 | 0.119 | 0.924 | 0.776 | 0.281 |
| 56 | pow0.5 / plain / trained (from trained) | yes | 2023 | 1820 | 0.917 | 0.201 | 0.006 | 0.271 | 0.200 | 0.122 | 0.910 | 0.733 | 0.245 |
| 57 | ema / BN / trained_static | yes | 1865 | 1816 | 0.939 | 0.236 | 0.006 | 0.285 | 0.204 | 0.116 | 0.943 | 0.831 | 0.336 |
| 58 | ema / plain / trained_static | yes | 1857 | 1743 | 0.947 | 0.309 | 0.005 | 0.273 | 0.196 | 0.104 | 0.943 | 0.829 | 0.334 |
| 59 | ema / rot_q / kpp_reseed (from trained) | yes | 1654 | 1552 | 0.952 | 0.067 | 0.008 | 0.285 | 0.211 | 0.116 | 0.945 | 0.836 | 0.350 |
| 60 | ema / whiten_q / random (from trained) | no | 4984 | 3988 | 0.372 | 0.000 | 0.000 | 0.034 | 0.020 | 0.010 | 0.802 | 0.457 | 0.024 |
| 61 | zca / whiten_q / random (from trained) | no | 4898 | 2862 | 0.386 | 0.000 | 0.000 | 0.046 | 0.030 | 0.017 | 0.188 | 0.012 | 0.000 |
| 62 | pow0.5 / whiten_q / random (from trained) | no | 4898 | 3036 | 0.388 | 0.000 | 0.000 | 0.042 | 0.025 | 0.013 | 0.618 | 0.175 | 0.003 |
| 63 | shrink0.3 / whiten_q / random (from trained) | no | 4880 | 3780 | 0.392 | 0.000 | 0.000 | 0.040 | 0.024 | 0.012 | 0.793 | 0.438 | 0.020 |
| 64 | shrink0.1 / whiten_q / random (from trained) | no | 4857 | 3593 | 0.397 | 0.000 | 0.000 | 0.044 | 0.026 | 0.013 | 0.774 | 0.399 | 0.014 |
| 65 | ema / whiten_q / trained (from trained) | no | 4388 | 3718 | 0.485 | 0.000 | 0.001 | 0.085 | 0.061 | 0.040 | 0.832 | 0.525 | 0.060 |
| 66 | ema / whiten_q / kpp_reseed (from trained) | no | 4388 | 3718 | 0.485 | 0.000 | 0.001 | 0.085 | 0.061 | 0.040 | 0.832 | 0.525 | 0.060 |
| 67 | zca / whiten_q / trained (from trained) | no | 4285 | 2766 | 0.497 | 0.000 | 0.001 | 0.105 | 0.079 | 0.053 | 0.268 | 0.029 | 0.002 |
| 68 | zca / whiten_q / kpp_reseed (from trained) | no | 4285 | 2766 | 0.497 | 0.000 | 0.001 | 0.105 | 0.079 | 0.053 | 0.268 | 0.029 | 0.002 |
| 69 | pow0.5 / whiten_q / trained (from trained) | no | 4221 | 2853 | 0.512 | 0.000 | 0.001 | 0.104 | 0.077 | 0.053 | 0.685 | 0.257 | 0.017 |
| 70 | pow0.5 / whiten_q / kpp_reseed (from trained) | no | 4221 | 2853 | 0.512 | 0.000 | 0.001 | 0.104 | 0.077 | 0.053 | 0.685 | 0.257 | 0.017 |
| 71 | shrink0.3 / whiten_q / trained (from trained) | no | 4155 | 3449 | 0.524 | 0.000 | 0.001 | 0.109 | 0.080 | 0.054 | 0.830 | 0.523 | 0.069 |
| 72 | shrink0.3 / whiten_q / kpp_reseed (from trained) | no | 4155 | 3449 | 0.524 | 0.000 | 0.001 | 0.109 | 0.080 | 0.054 | 0.830 | 0.523 | 0.069 |
| 73 | shrink0.1 / whiten_q / trained (from trained) | no | 4121 | 3280 | 0.529 | 0.000 | 0.001 | 0.115 | 0.085 | 0.058 | 0.815 | 0.491 | 0.064 |
| 74 | shrink0.1 / whiten_q / kpp_reseed (from trained) | no | 4121 | 3280 | 0.529 | 0.000 | 0.001 | 0.115 | 0.085 | 0.058 | 0.815 | 0.491 | 0.064 |
| 75 | zca / BN / F_gumbel | no | 3946 | 3214 | 0.575 | 0.000 | 0.002 | 0.108 | 0.078 | 0.054 | 0.465 | 0.110 | 0.016 |
| 76 | zca / plain / F_gumbel | no | 3944 | 3185 | 0.573 | 0.000 | 0.001 | 0.108 | 0.077 | 0.055 | 0.463 | 0.109 | 0.016 |
| 77 | zca / BN / D_sinkhorn | no | 3912 | 3107 | 0.583 | 0.000 | 0.001 | 0.112 | 0.080 | 0.056 | 0.490 | 0.116 | 0.018 |
| 78 | zca / plain / fP | no | 3901 | 3146 | 0.585 | 0.001 | 0.002 | 0.117 | 0.086 | 0.058 | 0.489 | 0.118 | 0.018 |
| 79 | zca / BN / C_restart | no | 3892 | 3112 | 0.586 | 0.002 | 0.002 | 0.116 | 0.083 | 0.059 | 0.491 | 0.118 | 0.016 |
| 80 | zca / BN / fP | no | 3889 | 3083 | 0.586 | 0.001 | 0.001 | 0.115 | 0.084 | 0.057 | 0.490 | 0.116 | 0.017 |
| 81 | zca / plain / C_restart | no | 3887 | 3097 | 0.587 | 0.000 | 0.001 | 0.115 | 0.083 | 0.055 | 0.490 | 0.116 | 0.017 |
| 82 | zca / plain / attract | no | 3884 | 3087 | 0.587 | 0.006 | 0.001 | 0.117 | 0.085 | 0.057 | 0.490 | 0.116 | 0.017 |
| 83 | zca / BN / attract | no | 3882 | 3097 | 0.588 | 0.006 | 0.002 | 0.116 | 0.083 | 0.057 | 0.493 | 0.119 | 0.017 |
| 84 | zca / plain / D_sinkhorn | no | 3880 | 3107 | 0.586 | 0.000 | 0.001 | 0.115 | 0.084 | 0.056 | 0.489 | 0.118 | 0.018 |
| 85 | zca / BN / B_ema_kmeans | no | 3873 | 3044 | 0.589 | 0.008 | 0.001 | 0.116 | 0.082 | 0.053 | 0.498 | 0.118 | 0.018 |
| 86 | zca / plain / B_ema_kmeans | no | 3861 | 3045 | 0.589 | 0.010 | 0.002 | 0.111 | 0.081 | 0.052 | 0.494 | 0.119 | 0.021 |
| 87 | zca / rot_q / random (from trained) | no | 3744 | 2557 | 0.627 | 0.003 | 0.001 | 0.106 | 0.074 | 0.042 | 0.477 | 0.094 | 0.009 |
| 88 | zca / rot_q / trained (from trained) | no | 3736 | 2577 | 0.626 | 0.002 | 0.002 | 0.111 | 0.078 | 0.047 | 0.480 | 0.101 | 0.012 |
| 89 | zca / rot_q / kpp_reseed (from trained) | no | 3736 | 2577 | 0.626 | 0.002 | 0.002 | 0.111 | 0.078 | 0.047 | 0.480 | 0.101 | 0.012 |
| 90 | zca / BN / random_static | no | 3734 | 2646 | 0.625 | 0.004 | 0.001 | 0.109 | 0.077 | 0.048 | 0.478 | 0.093 | 0.013 |
| 91 | zca / whiten_q / fP_opt (from trained) | no | 3705 | 2819 | 0.616 | 0.002 | 0.002 | 0.146 | 0.102 | 0.071 | 0.344 | 0.057 | 0.005 |
| 92 | zca / plain / random_static | no | 3701 | 2609 | 0.632 | 0.004 | 0.001 | 0.111 | 0.077 | 0.048 | 0.481 | 0.095 | 0.014 |
| 93 | pow0.5 / whiten_q / fP_opt (from trained) | no | 3513 | 2808 | 0.653 | 0.003 | 0.002 | 0.168 | 0.118 | 0.084 | 0.743 | 0.353 | 0.045 |
| 94 | zca / rot_q / fP_opt (from trained) | no | 3391 | 2755 | 0.681 | 0.001 | 0.003 | 0.144 | 0.105 | 0.069 | 0.533 | 0.150 | 0.033 |
| 95 | pow0.5 / plain / random (from trained) | no | 3338 | 2797 | 0.710 | 0.018 | 0.001 | 0.129 | 0.083 | 0.042 | 0.869 | 0.626 | 0.090 |
| 96 | pow0.5 / rot_q / random (from trained) | no | 3225 | 2725 | 0.731 | 0.011 | 0.002 | 0.138 | 0.093 | 0.050 | 0.872 | 0.631 | 0.098 |
| 97 | zca / plain / kpp_reseed (from trained) | no | 2580 | 2013 | 0.829 | 0.099 | 0.005 | 0.209 | 0.155 | 0.099 | 0.595 | 0.206 | 0.064 |
| 98 | zca / BN / trained_static | no | 2511 | 1985 | 0.838 | 0.088 | 0.006 | 0.218 | 0.163 | 0.104 | 0.600 | 0.210 | 0.068 |
| 99 | zca / plain / trained_static | no | 2484 | 1961 | 0.842 | 0.098 | 0.006 | 0.218 | 0.163 | 0.104 | 0.601 | 0.212 | 0.068 |

## Appendix A. First simulation, pass p3, ALL levels (16/32/64/128) and 3 heads; key fits start from the trained 6K keys (100 Adam steps on 4,096 fit queries, λ_u = 0.1). "score" = the mean top-8 cosine on the scored half

| norm | halves | keys | dist | dist7 | top1 | dead | score | xdoc | lag1 | lag16 | lag256 | stab_n03 | stab_n10 | stab_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ema | plain | trained | 1867 | 1772 | 0.946 | 0.312 | 0.483 | 0.004 | 0.275 | 0.200 | 0.106 | 0.944 | 0.834 | 0.324 |
| ema | plain | random | 2927 | 2703 | 0.791 | 0.053 | 0.306 | 0.001 | 0.155 | 0.101 | 0.044 | 0.926 | 0.781 | 0.202 |
| ema | plain | kpp_reseed | 2520 | 2385 | 0.855 | 0.225 | 0.578 | 0.002 | 0.211 | 0.141 | 0.070 | 0.937 | 0.815 | 0.263 |
| ema | plain | fP_opt | 3427 | 3271 | 0.704 | 0.002 | 0.647 | 0.001 | 0.112 | 0.066 | 0.030 | 0.911 | 0.735 | 0.151 |
| ema | plain | attract_only | 3062 | 2921 | 0.772 | 0.255 | 0.638 | 0.001 | 0.135 | 0.085 | 0.037 | 0.917 | 0.751 | 0.178 |
| ema | whiten_q | trained | 4396 | 3701 | 0.485 | 0.000 | 0.316 | 0.001 | 0.085 | 0.061 | 0.040 | 0.833 | 0.527 | 0.056 |
| ema | whiten_q | random | 4967 | 3941 | 0.375 | 0.000 | 0.307 | 0.000 | 0.035 | 0.022 | 0.011 | 0.804 | 0.460 | 0.023 |
| ema | whiten_q | kpp_reseed | 4396 | 3701 | 0.485 | 0.000 | 0.316 | 0.001 | 0.085 | 0.061 | 0.040 | 0.833 | 0.527 | 0.056 |
| ema | whiten_q | fP_opt | 3648 | 3392 | 0.635 | 0.003 | 0.346 | 0.001 | 0.140 | 0.096 | 0.063 | 0.869 | 0.620 | 0.099 |
| ema | rot_q | trained | 2939 | 2688 | 0.788 | 0.039 | 0.301 | 0.001 | 0.153 | 0.099 | 0.042 | 0.925 | 0.781 | 0.201 |
| ema | rot_q | random | 2895 | 2686 | 0.795 | 0.038 | 0.307 | 0.002 | 0.156 | 0.102 | 0.047 | 0.926 | 0.782 | 0.203 |
| ema | rot_q | kpp_reseed | 1635 | 1559 | 0.955 | 0.062 | 0.359 | 0.008 | 0.293 | 0.223 | 0.123 | 0.947 | 0.842 | 0.349 |
| ema | rot_q | fP_opt | 3070 | 2982 | 0.769 | 0.005 | 0.625 | 0.001 | 0.132 | 0.078 | 0.037 | 0.917 | 0.752 | 0.174 |
| zca | plain | trained | 2543 | 1995 | 0.835 | 0.104 | 0.453 | 0.005 | 0.209 | 0.160 | 0.106 | 0.603 | 0.212 | 0.066 |
| zca | plain | random | 3741 | 2595 | 0.625 | 0.006 | 0.306 | 0.001 | 0.106 | 0.075 | 0.044 | 0.489 | 0.101 | 0.011 |
| zca | plain | kpp_reseed | 2647 | 2053 | 0.819 | 0.104 | 0.462 | 0.005 | 0.201 | 0.152 | 0.101 | 0.597 | 0.206 | 0.062 |
| zca | plain | fP_opt | 3930 | 3205 | 0.579 | 0.000 | 0.621 | 0.001 | 0.106 | 0.078 | 0.051 | 0.494 | 0.117 | 0.016 |
| zca | whiten_q | trained | 4312 | 2752 | 0.492 | 0.000 | 0.320 | 0.001 | 0.102 | 0.076 | 0.054 | 0.280 | 0.033 | 0.001 |
| zca | whiten_q | random | 4884 | 2947 | 0.388 | 0.000 | 0.306 | 0.000 | 0.046 | 0.029 | 0.016 | 0.202 | 0.013 | 0.000 |
| zca | whiten_q | kpp_reseed | 4312 | 2752 | 0.492 | 0.000 | 0.320 | 0.001 | 0.102 | 0.076 | 0.054 | 0.280 | 0.033 | 0.001 |
| zca | whiten_q | fP_opt | 3730 | 2890 | 0.611 | 0.001 | 0.365 | 0.002 | 0.136 | 0.093 | 0.063 | 0.353 | 0.059 | 0.003 |
| zca | rot_q | trained | 3739 | 2643 | 0.625 | 0.002 | 0.302 | 0.002 | 0.110 | 0.077 | 0.049 | 0.490 | 0.106 | 0.011 |
| zca | rot_q | random | 3726 | 2612 | 0.629 | 0.003 | 0.307 | 0.002 | 0.106 | 0.075 | 0.044 | 0.489 | 0.100 | 0.010 |
| zca | rot_q | kpp_reseed | 3736 | 2643 | 0.626 | 0.002 | 0.302 | 0.002 | 0.110 | 0.077 | 0.049 | 0.490 | 0.106 | 0.011 |
| zca | rot_q | fP_opt | 3430 | 2895 | 0.673 | 0.001 | 0.589 | 0.003 | 0.140 | 0.105 | 0.071 | 0.535 | 0.151 | 0.028 |
| pow0.5 | plain | trained | 2040 | 1832 | 0.916 | 0.209 | 0.478 | 0.006 | 0.265 | 0.198 | 0.120 | 0.910 | 0.734 | 0.229 |
| pow0.5 | plain | random | 3249 | 2747 | 0.725 | 0.021 | 0.306 | 0.002 | 0.137 | 0.091 | 0.046 | 0.873 | 0.635 | 0.092 |
| pow0.5 | plain | kpp_reseed | 2323 | 2065 | 0.877 | 0.200 | 0.513 | 0.004 | 0.239 | 0.174 | 0.104 | 0.906 | 0.722 | 0.203 |
| pow0.5 | plain | fP_opt | 3324 | 3144 | 0.710 | 0.003 | 0.622 | 0.002 | 0.130 | 0.088 | 0.049 | 0.869 | 0.623 | 0.101 |
| pow0.5 | whiten_q | trained | 4251 | 2900 | 0.508 | 0.000 | 0.319 | 0.001 | 0.101 | 0.074 | 0.051 | 0.687 | 0.259 | 0.013 |
| pow0.5 | whiten_q | random | 4888 | 3113 | 0.390 | 0.000 | 0.307 | 0.000 | 0.042 | 0.026 | 0.013 | 0.622 | 0.178 | 0.003 |
| pow0.5 | whiten_q | kpp_reseed | 4251 | 2900 | 0.508 | 0.000 | 0.319 | 0.001 | 0.101 | 0.074 | 0.051 | 0.687 | 0.259 | 0.013 |
| pow0.5 | whiten_q | fP_opt | 3559 | 2884 | 0.648 | 0.002 | 0.360 | 0.002 | 0.157 | 0.109 | 0.074 | 0.741 | 0.349 | 0.032 |
| pow0.5 | rot_q | trained | 3223 | 2704 | 0.728 | 0.012 | 0.302 | 0.002 | 0.140 | 0.095 | 0.051 | 0.874 | 0.636 | 0.101 |
| pow0.5 | rot_q | random | 3196 | 2699 | 0.735 | 0.012 | 0.307 | 0.002 | 0.139 | 0.094 | 0.051 | 0.874 | 0.637 | 0.098 |
| pow0.5 | rot_q | kpp_reseed | 3112 | 2618 | 0.746 | 0.012 | 0.304 | 0.002 | 0.147 | 0.100 | 0.054 | 0.876 | 0.641 | 0.108 |
| pow0.5 | rot_q | fP_opt | 2889 | 2788 | 0.786 | 0.004 | 0.591 | 0.004 | 0.168 | 0.118 | 0.073 | 0.879 | 0.653 | 0.137 |
| shrink0.1 | plain | trained | 2251 | 1952 | 0.882 | 0.161 | 0.466 | 0.006 | 0.242 | 0.183 | 0.117 | 0.915 | 0.750 | 0.237 |
| shrink0.1 | plain | random | 3448 | 2899 | 0.683 | 0.011 | 0.306 | 0.002 | 0.125 | 0.085 | 0.046 | 0.889 | 0.677 | 0.113 |
| shrink0.1 | plain | kpp_reseed | 2406 | 2094 | 0.860 | 0.154 | 0.482 | 0.005 | 0.229 | 0.172 | 0.109 | 0.912 | 0.743 | 0.223 |
| shrink0.1 | plain | fP_opt | 3582 | 3256 | 0.654 | 0.002 | 0.627 | 0.002 | 0.127 | 0.091 | 0.058 | 0.880 | 0.656 | 0.117 |
| shrink0.1 | whiten_q | trained | 4157 | 3304 | 0.525 | 0.000 | 0.323 | 0.001 | 0.112 | 0.083 | 0.057 | 0.815 | 0.490 | 0.060 |
| shrink0.1 | whiten_q | random | 4848 | 3582 | 0.398 | 0.000 | 0.307 | 0.000 | 0.043 | 0.026 | 0.013 | 0.776 | 0.402 | 0.014 |
| shrink0.1 | whiten_q | kpp_reseed | 4157 | 3304 | 0.525 | 0.000 | 0.323 | 0.001 | 0.112 | 0.083 | 0.057 | 0.815 | 0.490 | 0.060 |
| shrink0.1 | whiten_q | fP_opt | 3475 | 3130 | 0.660 | 0.002 | 0.371 | 0.002 | 0.167 | 0.119 | 0.080 | 0.851 | 0.577 | 0.090 |
| shrink0.1 | rot_q | trained | 3428 | 2838 | 0.685 | 0.007 | 0.302 | 0.002 | 0.130 | 0.091 | 0.053 | 0.888 | 0.678 | 0.122 |
| shrink0.1 | rot_q | random | 3404 | 2866 | 0.692 | 0.007 | 0.307 | 0.002 | 0.128 | 0.089 | 0.051 | 0.889 | 0.679 | 0.119 |
| shrink0.1 | rot_q | kpp_reseed | 3416 | 2828 | 0.688 | 0.007 | 0.302 | 0.002 | 0.130 | 0.091 | 0.053 | 0.888 | 0.678 | 0.122 |
| shrink0.1 | rot_q | fP_opt | 3046 | 2842 | 0.748 | 0.006 | 0.595 | 0.004 | 0.167 | 0.125 | 0.081 | 0.892 | 0.689 | 0.156 |
| shrink0.3 | plain | trained | 2154 | 1948 | 0.898 | 0.183 | 0.471 | 0.006 | 0.249 | 0.186 | 0.116 | 0.924 | 0.777 | 0.266 |
| shrink0.3 | plain | random | 3363 | 2955 | 0.703 | 0.016 | 0.306 | 0.002 | 0.127 | 0.085 | 0.045 | 0.900 | 0.710 | 0.139 |
| shrink0.3 | plain | kpp_reseed | 2366 | 2128 | 0.869 | 0.178 | 0.493 | 0.005 | 0.232 | 0.171 | 0.107 | 0.922 | 0.769 | 0.249 |
| shrink0.3 | plain | fP_opt | 3456 | 3275 | 0.682 | 0.002 | 0.624 | 0.002 | 0.124 | 0.084 | 0.049 | 0.890 | 0.683 | 0.134 |
| shrink0.3 | whiten_q | trained | 4187 | 3465 | 0.521 | 0.000 | 0.322 | 0.001 | 0.107 | 0.078 | 0.053 | 0.831 | 0.524 | 0.065 |
| shrink0.3 | whiten_q | random | 4872 | 3753 | 0.393 | 0.000 | 0.307 | 0.000 | 0.041 | 0.025 | 0.012 | 0.794 | 0.441 | 0.019 |
| shrink0.3 | whiten_q | kpp_reseed | 4187 | 3465 | 0.521 | 0.000 | 0.322 | 0.001 | 0.107 | 0.078 | 0.053 | 0.831 | 0.524 | 0.065 |
| shrink0.3 | whiten_q | fP_opt | 3471 | 3242 | 0.663 | 0.004 | 0.368 | 0.002 | 0.168 | 0.121 | 0.085 | 0.865 | 0.612 | 0.107 |
| shrink0.3 | rot_q | trained | 3326 | 2875 | 0.706 | 0.010 | 0.302 | 0.002 | 0.133 | 0.090 | 0.050 | 0.900 | 0.711 | 0.148 |
| shrink0.3 | rot_q | random | 3303 | 2906 | 0.713 | 0.010 | 0.307 | 0.002 | 0.131 | 0.090 | 0.050 | 0.901 | 0.712 | 0.144 |
| shrink0.3 | rot_q | kpp_reseed | 3243 | 2840 | 0.720 | 0.010 | 0.303 | 0.002 | 0.138 | 0.095 | 0.053 | 0.902 | 0.713 | 0.151 |
| shrink0.3 | rot_q | fP_opt | 2948 | 2851 | 0.772 | 0.006 | 0.591 | 0.004 | 0.166 | 0.120 | 0.075 | 0.902 | 0.712 | 0.173 |

## Appendix B. The same, pass p1

| norm | halves | keys | dist | dist7 | top1 | dead | score | xdoc | lag1 | lag16 | lag256 | stab_n03 | stab_n10 | stab_drift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ema | plain | trained | 1499 | 1366 | 0.974 | 0.458 | 0.488 | 0.008 | 0.245 | 0.189 | 0.106 | 0.947 | 0.841 | 0.352 |
| ema | plain | random | 2232 | 1992 | 0.892 | 0.185 | 0.306 | 0.004 | 0.179 | 0.134 | 0.067 | 0.936 | 0.811 | 0.271 |
| ema | plain | kpp_reseed | 2796 | 2587 | 0.816 | 0.281 | 0.659 | 0.001 | 0.151 | 0.106 | 0.048 | 0.936 | 0.808 | 0.245 |
| ema | plain | fP_opt | 3610 | 3399 | 0.673 | 0.014 | 0.733 | 0.001 | 0.080 | 0.052 | 0.021 | 0.916 | 0.749 | 0.151 |
| ema | plain | attract_only | 2934 | 2751 | 0.801 | 0.437 | 0.716 | 0.002 | 0.108 | 0.074 | 0.034 | 0.922 | 0.766 | 0.191 |
| ema | whiten_q | trained | 4538 | 3890 | 0.455 | 0.000 | 0.316 | 0.001 | 0.070 | 0.052 | 0.035 | 0.830 | 0.520 | 0.053 |
| ema | whiten_q | random | 5107 | 4133 | 0.344 | 0.000 | 0.307 | 0.000 | 0.026 | 0.018 | 0.009 | 0.801 | 0.450 | 0.020 |
| ema | whiten_q | kpp_reseed | 4538 | 3890 | 0.455 | 0.000 | 0.316 | 0.001 | 0.070 | 0.052 | 0.035 | 0.830 | 0.520 | 0.053 |
| ema | whiten_q | fP_opt | 4028 | 3677 | 0.568 | 0.001 | 0.344 | 0.001 | 0.091 | 0.063 | 0.040 | 0.859 | 0.595 | 0.075 |
| ema | rot_q | trained | 2260 | 2012 | 0.889 | 0.170 | 0.301 | 0.005 | 0.173 | 0.127 | 0.061 | 0.936 | 0.807 | 0.261 |
| ema | rot_q | random | 2218 | 1998 | 0.893 | 0.176 | 0.308 | 0.005 | 0.180 | 0.133 | 0.068 | 0.936 | 0.811 | 0.270 |
| ema | rot_q | kpp_reseed | 1506 | 1381 | 0.983 | 0.484 | 0.563 | 0.007 | 0.251 | 0.193 | 0.105 | 0.952 | 0.855 | 0.363 |
| ema | rot_q | fP_opt | 3220 | 3053 | 0.749 | 0.022 | 0.718 | 0.001 | 0.093 | 0.061 | 0.027 | 0.919 | 0.758 | 0.176 |
| zca | plain | trained | 2614 | 2134 | 0.824 | 0.101 | 0.453 | 0.005 | 0.181 | 0.145 | 0.097 | 0.583 | 0.190 | 0.058 |
| zca | plain | random | 3895 | 2790 | 0.597 | 0.004 | 0.306 | 0.001 | 0.085 | 0.065 | 0.036 | 0.465 | 0.085 | 0.009 |
| zca | plain | kpp_reseed | 2718 | 2184 | 0.809 | 0.098 | 0.462 | 0.005 | 0.174 | 0.138 | 0.092 | 0.578 | 0.183 | 0.054 |
| zca | plain | fP_opt | 4119 | 3435 | 0.543 | 0.000 | 0.622 | 0.001 | 0.086 | 0.066 | 0.046 | 0.470 | 0.100 | 0.015 |
| zca | whiten_q | trained | 4481 | 2939 | 0.461 | 0.000 | 0.320 | 0.001 | 0.082 | 0.064 | 0.046 | 0.253 | 0.022 | 0.001 |
| zca | whiten_q | random | 5075 | 3144 | 0.348 | 0.000 | 0.306 | 0.000 | 0.032 | 0.023 | 0.012 | 0.178 | 0.009 | 0.000 |
| zca | whiten_q | kpp_reseed | 4481 | 2939 | 0.461 | 0.000 | 0.320 | 0.001 | 0.082 | 0.064 | 0.046 | 0.253 | 0.022 | 0.001 |
| zca | whiten_q | fP_opt | 3968 | 3043 | 0.568 | 0.000 | 0.364 | 0.002 | 0.107 | 0.079 | 0.050 | 0.325 | 0.047 | 0.002 |
| zca | rot_q | trained | 3866 | 2810 | 0.601 | 0.002 | 0.302 | 0.001 | 0.091 | 0.068 | 0.042 | 0.469 | 0.090 | 0.009 |
| zca | rot_q | random | 3866 | 2787 | 0.603 | 0.002 | 0.307 | 0.001 | 0.086 | 0.064 | 0.038 | 0.467 | 0.085 | 0.008 |
| zca | rot_q | kpp_reseed | 3866 | 2810 | 0.601 | 0.002 | 0.302 | 0.001 | 0.091 | 0.068 | 0.042 | 0.469 | 0.090 | 0.009 |
| zca | rot_q | fP_opt | 3622 | 3084 | 0.643 | 0.001 | 0.590 | 0.003 | 0.117 | 0.094 | 0.066 | 0.508 | 0.133 | 0.026 |
| pow0.5 | plain | trained | 2129 | 1942 | 0.903 | 0.202 | 0.477 | 0.006 | 0.232 | 0.179 | 0.108 | 0.907 | 0.726 | 0.221 |
| pow0.5 | plain | random | 3397 | 2920 | 0.698 | 0.017 | 0.306 | 0.002 | 0.112 | 0.080 | 0.040 | 0.871 | 0.627 | 0.089 |
| pow0.5 | plain | kpp_reseed | 2458 | 2205 | 0.858 | 0.192 | 0.510 | 0.004 | 0.201 | 0.152 | 0.090 | 0.902 | 0.713 | 0.191 |
| pow0.5 | plain | fP_opt | 3559 | 3284 | 0.669 | 0.002 | 0.623 | 0.001 | 0.094 | 0.065 | 0.034 | 0.864 | 0.610 | 0.089 |
| pow0.5 | whiten_q | trained | 4395 | 3089 | 0.480 | 0.000 | 0.320 | 0.001 | 0.083 | 0.064 | 0.043 | 0.680 | 0.246 | 0.011 |
| pow0.5 | whiten_q | random | 5055 | 3296 | 0.356 | 0.000 | 0.306 | 0.000 | 0.030 | 0.020 | 0.010 | 0.613 | 0.168 | 0.003 |
| pow0.5 | whiten_q | kpp_reseed | 4395 | 3089 | 0.480 | 0.000 | 0.320 | 0.001 | 0.083 | 0.064 | 0.043 | 0.680 | 0.246 | 0.011 |
| pow0.5 | whiten_q | fP_opt | 3834 | 3048 | 0.598 | 0.002 | 0.358 | 0.002 | 0.121 | 0.089 | 0.061 | 0.731 | 0.325 | 0.025 |
| pow0.5 | rot_q | trained | 3351 | 2895 | 0.704 | 0.012 | 0.302 | 0.002 | 0.118 | 0.085 | 0.043 | 0.872 | 0.631 | 0.096 |
| pow0.5 | rot_q | random | 3346 | 2869 | 0.708 | 0.012 | 0.307 | 0.002 | 0.117 | 0.082 | 0.043 | 0.872 | 0.630 | 0.094 |
| pow0.5 | rot_q | kpp_reseed | 3206 | 2786 | 0.727 | 0.011 | 0.304 | 0.002 | 0.127 | 0.092 | 0.047 | 0.875 | 0.639 | 0.103 |
| pow0.5 | rot_q | fP_opt | 3084 | 2941 | 0.755 | 0.003 | 0.592 | 0.003 | 0.129 | 0.096 | 0.061 | 0.875 | 0.642 | 0.131 |
| shrink0.1 | plain | trained | 2324 | 2046 | 0.872 | 0.152 | 0.466 | 0.006 | 0.215 | 0.168 | 0.108 | 0.912 | 0.740 | 0.227 |
| shrink0.1 | plain | random | 3594 | 3066 | 0.657 | 0.010 | 0.306 | 0.001 | 0.104 | 0.074 | 0.041 | 0.886 | 0.668 | 0.108 |
| shrink0.1 | plain | kpp_reseed | 2497 | 2189 | 0.847 | 0.149 | 0.482 | 0.006 | 0.202 | 0.156 | 0.101 | 0.909 | 0.734 | 0.214 |
| shrink0.1 | plain | fP_opt | 3755 | 3407 | 0.620 | 0.002 | 0.629 | 0.001 | 0.104 | 0.077 | 0.047 | 0.875 | 0.643 | 0.111 |
| shrink0.1 | whiten_q | trained | 4309 | 3513 | 0.496 | 0.000 | 0.323 | 0.001 | 0.092 | 0.071 | 0.049 | 0.810 | 0.477 | 0.053 |
| shrink0.1 | whiten_q | random | 5026 | 3771 | 0.363 | 0.000 | 0.307 | 0.000 | 0.031 | 0.021 | 0.010 | 0.769 | 0.389 | 0.012 |
| shrink0.1 | whiten_q | kpp_reseed | 4309 | 3513 | 0.496 | 0.000 | 0.323 | 0.001 | 0.092 | 0.071 | 0.049 | 0.810 | 0.477 | 0.053 |
| shrink0.1 | whiten_q | fP_opt | 3765 | 3296 | 0.608 | 0.002 | 0.367 | 0.003 | 0.130 | 0.095 | 0.066 | 0.844 | 0.557 | 0.077 |
| shrink0.1 | rot_q | trained | 3559 | 3003 | 0.661 | 0.005 | 0.302 | 0.002 | 0.111 | 0.082 | 0.047 | 0.885 | 0.670 | 0.114 |
| shrink0.1 | rot_q | random | 3536 | 3028 | 0.667 | 0.005 | 0.307 | 0.002 | 0.110 | 0.080 | 0.045 | 0.886 | 0.669 | 0.112 |
| shrink0.1 | rot_q | kpp_reseed | 3533 | 2987 | 0.666 | 0.005 | 0.302 | 0.002 | 0.111 | 0.082 | 0.047 | 0.886 | 0.671 | 0.115 |
| shrink0.1 | rot_q | fP_opt | 3237 | 2991 | 0.717 | 0.004 | 0.597 | 0.003 | 0.144 | 0.112 | 0.079 | 0.886 | 0.675 | 0.151 |
| shrink0.3 | plain | trained | 2230 | 2039 | 0.887 | 0.176 | 0.470 | 0.006 | 0.222 | 0.172 | 0.107 | 0.922 | 0.771 | 0.259 |
| shrink0.3 | plain | random | 3493 | 3109 | 0.678 | 0.015 | 0.306 | 0.001 | 0.107 | 0.076 | 0.040 | 0.898 | 0.703 | 0.135 |
| shrink0.3 | plain | kpp_reseed | 2444 | 2231 | 0.856 | 0.173 | 0.491 | 0.005 | 0.205 | 0.158 | 0.098 | 0.919 | 0.763 | 0.243 |
| shrink0.3 | plain | fP_opt | 3610 | 3359 | 0.652 | 0.003 | 0.626 | 0.001 | 0.106 | 0.077 | 0.042 | 0.889 | 0.679 | 0.133 |
| shrink0.3 | whiten_q | trained | 4325 | 3654 | 0.493 | 0.000 | 0.322 | 0.001 | 0.088 | 0.068 | 0.046 | 0.827 | 0.514 | 0.059 |
| shrink0.3 | whiten_q | random | 5037 | 3937 | 0.360 | 0.000 | 0.307 | 0.000 | 0.029 | 0.020 | 0.010 | 0.789 | 0.429 | 0.016 |
| shrink0.3 | whiten_q | kpp_reseed | 4325 | 3654 | 0.493 | 0.000 | 0.322 | 0.001 | 0.088 | 0.068 | 0.046 | 0.827 | 0.514 | 0.059 |
| shrink0.3 | whiten_q | fP_opt | 3800 | 3395 | 0.604 | 0.002 | 0.364 | 0.002 | 0.126 | 0.093 | 0.065 | 0.857 | 0.589 | 0.088 |
| shrink0.3 | rot_q | trained | 3454 | 3026 | 0.682 | 0.008 | 0.302 | 0.002 | 0.115 | 0.083 | 0.043 | 0.899 | 0.705 | 0.142 |
| shrink0.3 | rot_q | random | 3438 | 3052 | 0.688 | 0.007 | 0.307 | 0.002 | 0.114 | 0.081 | 0.045 | 0.899 | 0.705 | 0.140 |
| shrink0.3 | rot_q | kpp_reseed | 3408 | 2999 | 0.689 | 0.008 | 0.303 | 0.002 | 0.118 | 0.085 | 0.045 | 0.899 | 0.707 | 0.145 |
| shrink0.3 | rot_q | fP_opt | 3098 | 2939 | 0.746 | 0.004 | 0.594 | 0.004 | 0.144 | 0.110 | 0.072 | 0.898 | 0.705 | 0.174 |
