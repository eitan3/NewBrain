# 17. Address fixes, simulated offline on the captured thoughts (no code, no training changes) — 2026-10-24

Follows plan_v2/16. Scripts and raw JSON: `speed_wip/addr_inv/` (`sim.py`, `sim2.py`, `rank.py`, `sim_p3.json`,
`sim_p1.json`, `sim2_p3.json`). CPU, 4 threads, about 35 min in total. `runs/wiki_mem_12k/final.pt` did not appear while
I worked (the run was at step 7000), so the data are the step-6000 and step-7000 captures of the fixed val batch.

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

