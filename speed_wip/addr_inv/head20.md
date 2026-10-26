# 20. Tracking keys, migration and the memory life cycle (offline simulation + plan; no code) — 2026-10-24

Follows plans 16 and 17. Scripts: `speed_wip/addr_inv/sim3.py` (part 1), `check17b.py` (the plan-17 correction),
`nbr.py` (neighbour writes); results in `sim3_L*_h*.json`, `c17_*.json`, `nbr.json` and `tab3.md`. CPU, 4 threads per
process (6 parallel processes once the run had stopped), about 15 min in total. The thoughts are never constrained in any
option below: everything acts on the keys, the rows, or the table optimiser.

## Part 1. Address stability across the REAL thought drift (6K → 7K)

### Setup
- **Data**: the captured addressing inputs a of the 6K and 7K checkpoints, the read after pass 3, levels 16 and 128, 3
  heads. Both are addressed with the **6K EMA norm and the 6K W_q**, so only the thought moves.
- **Fit / score split**: everything is fit on sequences 0-15. Scoring uses the SAME eval positions (sequences 16-31) at 6K
  vs 7K.
- **Stability measure**: the top-6 overlap of row IDs ("row stab") and of the per-half top-8 sub-key IDs ("key stab").
- **Keys**: the "fitted" keys are **spherical k-means with top-1 assignment + dead restart** (rate 0.1, 2,000 batches of 512
  from random), which keeps the keys distinct.
  - The first attempt used the read-pull proxy of plan 17. It collapsed the keys into near-duplicates; see the correction.
  - Today's trained keys are the reference (a0).
- **Streaming the drift**: the 6K→7K drift is streamed as **40 interpolation stages**
  (a_t = (1−t)·a_6K + t·a_7K over 2,000 batches of 512 fit positions).

### The real drift is a local reshuffle, not a global shift

| level | centred cos(â_6K, â_7K), same position | median ‖Δâ‖/‖â‖ | cos(q_6K, q_7K) | 10-NN preserved (q space) |
|---|---|---|---|---|
| 16 | 0.750 | 0.72 | 0.856 | 0.415 |
| 128 | 0.731 | 0.74 | 0.848 | 0.360 |

Over 1,000 steps about 60% of every position's neighbourhood changes. No key motion can follow that, because keys move
with *groups* of queries. The training log's own probe agrees: `mem_last_stab_w` (row overlap between probes 500 steps
apart) is flat at 0.39-0.47 from step 2,500 to 7,000, and 0.54 at 7,500.

### Results (one table, ranked by 6K→7K row stability)

**Main table** (each 6K→7K row-stab cell is L16 / L128):

| rank | keys | S per half | distinct rows /1k pos | dead | cross-doc ovl | lag-1 ovl | **6K→7K row stab** | key-ID stab |
|---|---|---|---|---|---|---|---|---|
| 1 | (a) frozen fitted keys | 64 | 986 (of 4,096) | 0.3% | 0.009 | 0.391 | **0.420** (0.433 / 0.407) | 0.704 |
| 2 | (b) frozen + dead restart | 64 | 1,011 | 0.1% | 0.009 | 0.387 | **0.414** | 0.693 |
| 3 | (c) tracking r = 0.003 | 64 | 977 | 0.1% | 0.009 | 0.402 | **0.369** | 0.667 |
| 4 | (c) tracking r = 0.01 | 64 | 974 | 0.4% | 0.009 | 0.404 | **0.341** | 0.648 |
| 5 | (a0) today's trained keys, frozen | 512 | 1,738 | 34% | 0.005 | 0.308 | **0.334** (0.368 / 0.300) | 0.587 |
| 6 | (c) tracking r = 0.1 | 64 | 984 | 0.5% | 0.008 | 0.398 | **0.295** | 0.623 |
| 7 | (a) frozen fitted keys | 512 | 2,933 | 1.4% | 0.001 | 0.184 | **0.210** (0.233 / 0.187) | 0.505 |
| 8 | (b) frozen + dead restart | 512 | 2,943 | 0.8% | 0.001 | 0.183 | **0.209** | 0.502 |
| 9 | (c) tracking r = 0.001 | 512 | 2,929 | 1.4% | 0.001 | 0.185 | **0.203** | 0.497 |
| 10 | linear re-alignment of â_7K onto â_6K + frozen | 512 | 3,150 | 1.4% | 0.001 | 0.151 | **0.199** | 0.495 |
| 11 | (c) tracking r = 0.003 | 512 | 2,955 | 1.4% | 0.001 | 0.182 | **0.170** | 0.466 |
| 12 | (c) tracking r = 0.01 | 512 | 3,004 | 1.3% | 0.001 | 0.178 | **0.125** | 0.417 |
| 13 | (c) tracking r = 0.03 | 512 | 3,016 | 1.2% | 0.001 | 0.178 | **0.100** | 0.382 |
| 14 | (c) tracking r = 0.1 | 512 | 2,973 | 1.3% | 0.001 | 0.183 | **0.073** | 0.344 |
| 15 | keep training with the read pull (collapses keys) | 512 / 64 | 3,307 / 1,253 | 0% | 0.001 | 0.135 | **0.013 / 0.029** | 0.25 / 0.39 |
| 16 | (d) re-fit at 7K, no migration | 512 / 64 | 2,941 / 938 | 1% | 0.001 | 0.181 | **0.000 / 0.002** | 0.02 / 0.13 |

**(d) with migration** (content-carried stability, see below), not ranked in the main table:

| keys | S per half | distinct rows /1k pos | dead | cross-doc ovl | lag-1 ovl | content-carried stability | key-ID stab |
|---|---|---|---|---|---|---|---|
| (d) re-fit at 7K + migration via key votes | 512 | 2,941 | 1% | 0.001 | 0.181 | 0.06 | - |
| (d) re-fit at 7K + migration via key votes | 64 | 938 | 0.4% | 0.009 | 0.401 | 0.25 (0.30 via row votes) | - |

Notes:
- **(c) with restart** is within ±0.015 of (c) without restart, for every rate.
- **Key merging**:
  - tracking raises the keys with a near-twin above cos 0.95 from 49 to 75-115 (of 1,024, two codebooks); none are above
    0.99, and the max key cosine is 0.98, so keys don't merge;
  - S = 64: 0-1 near-twins.
- **Doc separation** is intact in every setting: cross-doc overlap 0.001 at S = 512, 0.009 at S = 64.

**Which tracking rate keeps the overlap ≥ 0.8? None.** Stability falls *monotonically* with the tracking rate (0.210 frozen →
0.203 / 0.170 / 0.125 / 0.100 / 0.073 for r = 0.001 / 0.003 / 0.01 / 0.03 / 0.1). Because the drift is a local reshuffle,
moving a key toward the new mean of its assigned queries mostly re-partitions the space and adds churn. The same holds at
S = 64. An addressing-side linear re-alignment of the 7K thoughts onto the 6K ones does not help either (0.199), so the
drift is not a linear map.

**(d) Migration** (re-fit at 7K by k-means, map each new sub-key to the old sub-key most of its fit contexts used, rows as
the product map, copy the content):
- **Content carried**:
  - S = 512: only **6%** of a position's new rows hold content from its own old rows (key-vote map; 6.4% with a row-vote
    map), worse than simply freezing (21%);
  - S = 64: 25% via the key votes, 30% via the row votes, vs 42% frozen.
- **Conflicts**:
  - 58% of the new S = 512 sub-keys (18% at S = 64) have no majority old key;
  - with the row-vote map, 53% of the eval rows at S = 512 get no mapping at all (no fit context reached them);
  - 1,566 old rows (S = 512) are copied into 2 or more new rows.
- Migration is not worth it for this kind of drift.

**(e) Coarse vs fine**: S = 64 doubles the drift stability (0.42 vs 0.21) and the neighbour persistence (lag-1 0.39 vs 0.18).
It costs ×3 fewer distinct rows per 1K positions (986 of 4,096 possible vs 2,933) and ×9 more cross-doc sharing (0.009).
Even S = 64 is far from 0.8.

**Neighbour writes**: the share of the 7K top-6 rows that were among the 6K top-K′ pairs, i.e. what writing a small share to
the 7th…K′-th candidates would have pre-filled.

| frozen keys | top-6 (today) | top-8 | top-12 | top-16 | top-32 |
|---|---|---|---|---|---|
| today's trained, S = 512 | 0.334 | 0.388 | 0.459 | 0.505 | 0.592 |
| fitted, S = 512 | 0.210 | 0.248 | 0.306 | 0.348 | 0.440 |
| fitted, S = 64 | 0.420 | 0.490 | 0.584 | 0.645 | 0.755 |

### What would reach 0.8
Stability is set by the thought drift, not by the keys. For reference (plans 16 / 17), fixed keys keep 0.93-0.95 of the
rows under 3% input noise and 0.75-0.87 under 10% noise. The real 1,000-step drift is a 72% relative change. **Overlap ≥ 0.8
per interval therefore needs the thoughts to move about 7× less per interval (≲ 10% relative change)**, or cells far coarser
than S = 64. This is exactly what the owner's "decay early, stop once stable" plan needs as its trigger: the switch must wait
for the drift to slow. Constraining the thought is not needed, only waiting for it.

### Ranking (what to do with the keys)
1. **Frozen distinct keys + dead restart (b).** Best stability per unit spread at S = 512, restart costs nothing in
   stability (−0.001), and dead keys go from 1.4% to 0.8%. Spread is ×1.7 today's at a drift stability of 0.21 vs 0.33.
2. **Coarse level (S = 64) alongside the fine one.** ×2 the drift stability, about ⅓ of the spread, cheap (Part 3).
3. **Neighbour writes to the top-8 / top-12.** +0.04 / +0.10 carried content at S = 512 for ×1.33 / ×2 write traffic.
4. Not recommended:
   - tracking keys, at any rate;
   - re-fit + migration;
   - addressing-side re-alignment.

All of these are worse than frozen under this drift.

### Correction to plan 17
- The dynamic key methods there (read-pull "attract", f·P, B, C, D, F) all collapse into near-duplicate keys under the
  read-pull proxy: only 43-175 effective keys per codebook, against 507 for today's real keys. De-duplicated, their spread
  is 366-612 distinct rows per 1K, *worse* than today's 1,676.
- Distinct keys (k-means + restart: 3,025; random: 2,910) are the valid ×1.7-1.8 result.
- A correction note has been added at the top of plan 17. The whitening / query-BN / rotation conclusions stand.

## Part 2. Design: the memory life cycle (plan only)

The current MemOpt, per touched row r (G_r ≠ 0), written once per optimizer step:
- the accumulator: S_r ← S_r + mean_i G_ri², with S_r starting at S₀;
- the update: ΔV_r = −η·G_r/√S_r, clipped to ‖ΔV_r‖ ≤ c_Δ.

### Infancy (from the write start until the switch)
1. **Row value decay, lazy per row**:
   - store t_r (the step of the row's last touch; int32, 4 MB for 1 M rows);
   - at a touch: V_r ← (1−λ)^(t − t_r)·V_r, then the update, then t_r ← t;
   - at a read, scale the gathered row by the same factor (one gather of t_r, one pow).

   This is exact global decay V ← (1−λ)V at no table-wide cost.
   - **Choice of λ**: match the address turnover. Rows survive with s ≈ 0.21-0.33 per 1,000 steps (the measured `stab_w` is
     0.4-0.45 per 500), i.e. an address half-life of h_addr = 1000·ln 2/(−ln s) ≈ 430-600 steps.
   - Take a value half-life about 1-2 × h_addr: **λ ≈ 1e-3 per step** (h = 690 steps). A row written once per about 60-120
     steps (the median of today's counts) then holds about 6-12 writes' worth of content; the popular rows hold far more.
2. **A decaying accumulator** (RMSprop-like, per touch, with a floor):
   - S_r ← β_S^(n)·S_r + (1−β_S)·mean_i G_ri² at each write, with n = 1 per write; S_r ≥ κ·S₀ (κ ≈ 0.1) keeps a rarely
     written row's first step bounded;
   - **β_S ≈ 0.99** remembers about the last 100 writes;
   - the step size stays at about η/RMS(g) instead of shrinking as 1/√(#writes). Hot rows stay plastic, so they can follow
     their changing contexts.
3. **Dead-key restart**: uncapped, every 100 steps (plan 17 / part 1 (b); keep `--mem-balance-keep`).
   - Seed from a reservoir of at least 1,024 recent query halves per codebook (today's NRES = 64 would clone seeds).
   - **New**: when a key is re-seeded, the 512 rows of its band are set to V ← 0 (or ×(1−λ)^big) and S ← S₀. Today only S is
     reset, so the band keeps content written by contexts that no longer reach it.
4. **The keys stay distinct**: k-means-style updates use top-1 assignment, or the real gradient. Never pull all top-m keys
   toward one query (the plan-17 failure mode).

### The adaptive switch
- **Signal**: `mem_last_stab_w` (already logged: the probe row overlap between evals 500 steps apart; Δ = 500 steps).
- **Trigger** (whichever comes first):
  - **stab_w ≥ 0.9 for N = 3 consecutive probes**. From the noise curve, this means the thoughts move ≲ 3-5% per 500
    steps.
  - **The start of the LR decay** (80% of the run). The thought drift scales with the LR, so the decay is when drift
    naturally slows.
  - In wiki_mem_12k stab_w was flat at 0.39-0.47 through steps 2,500-7,000, so the first trigger would not have fired before
    the LR decay.
- **Ramp** over R steps (about 5% of the run), with ρ: 0 → 1:
  - λ_t = (1−ρ)·λ₀;
  - 1 − β_S,t = (1−ρ)(1−β₀), so β_S → 1 recovers Adagrad's accumulation from the current S_r (no reset);
  - the restart interval grows 100 → 1,000 steps.
- **Then**: freeze the keys / W_q / norm stats (the existing freeze; θ ← θ̄) at ρ = 1, *instead of* the fixed 60% freeze.
  - The fixed freeze at step 7,325 happened while stab_w was still about 0.45, so about half of the rows written before it
    were orphaned within 500 steps.
  - Keep-tracking after the switch is not recommended: part 1 shows tracking only adds churn. Once the drift is slow, frozen
    keys are both stable and enough.

### Expected behaviour and risks
- **Forgetting vs plasticity**: λ = 1e-3 forgets the content of an address in about 700 steps.
  - Too high a λ, and rare rows never accumulate (they hold < 3 writes' worth); too low, and orphaned infancy content
    persists.
  - Diagnostic: the mean ‖V_r‖ by write-count bucket and the gate mean. If the gates fall as λ rises, the content is too
    thin.
- **Interaction with the freeze**: decay must reach 0 before or at the freeze. Otherwise post-freeze rows (now the long-term
  memory) keep leaking.
  - The guard / repair probes (MemoryProbes) compare against freeze-time records, so the switch should define the freeze
    time.
- **Rows written in infancy**:
  - decayed by about (1−λ)^(steps since the last write). A row last written 3,000 steps before the switch keeps about 5%.
  - Rows still in use at the switch keep their recent content, and the Adagrad accumulation then slows their learning
    (stability).
  - Re-seeded bands start empty.
  - Net effect: post-switch memory = recent, consistent content. Long-lived early content is not preserved by design. That
    is acceptable because its addresses did not survive anyway (stab ≈ 0.2-0.33 per 1,000 steps).
- **The gate's view**: the decayed rows have a smaller norm, so the gates may learn "small = stale". That is fine, but it
  shifts at the switch when decay stops. The ramp avoids a step change.
- **Non-stationary accumulator**: with β_S < 1 the row step size no longer anneals. The update clip c_Δ remains the safety
  bound.

## Part 3. Feasibility and cost (estimates for the RTX 5090, 8,192 tokens per micro-batch, 4 levels × 3 heads × about 5 reads)

1. **Row-carried keys** (each row has its own 64-d key k_r, refreshed on use: k_r ← normalize((1−ρ)·k_r + ρ·q)):
   - **Retrieval**: an exact kNN over 1 M rows is about 3·10¹³ FLOP per step, which is infeasible. As a re-ranker of the PK
     candidates it is cheap: re-score the 64 (8×8) candidate pairs by k_r·q, which is about 4 K FLOP per (token, head),
     negligible.
   - **Memory**: 1 M × 64 keys: 128 MB in bf16, 256 MB in fp32; no optimizer state (EMA refresh).
   - **Benefit**: bounded by the candidate set. The PK key-ID stability is about 0.5 per half, so about 25% of a position's
     64 candidates survive 1,000 steps. Row keys would let a row keep "its" contexts *within* that set, but cannot recover
     rows that left it.
   - **Implementation**: about 60-100 lines plus a probe.
   - **Verdict**: plausible as a second stage after the switch; not a fix for drift.
2. **Coarse + fine levels** (an S = 64 per-half codebook next to S = 512):
   - **Size**: 4,096 rows × 256 × 4 B = 4 MB per level (plus grad and accumulator, about 12 MB); the scoring is 1/8 of
     today's.
   - **Effect**: ×2 the drift stability (0.42 vs 0.21) and lag-1 0.39, at about ⅓ of the spread.
   - **Cost**: about +0.01 s/step, plus 3 more gates (or reuse the heads).
   - **Implementation**: about 100 lines (a second, small Memory table per level, or one extra head per level with its own
     S).
   - **Verdict**: the cheapest stability gain; recommended to try.
3. **Neighbour writes to the 7th / 8th candidates**:
   - select() already computes km + 1 pairs (and the per-half top-8). Write pairs 7..8 with a small fixed weight ε (e.g.
     0.1 × the 6th weight) in MemRead's backward only; the forward is unchanged.
   - **Cost**: +33% rows in the MemRead backward scatter and in MemOpt for the top-8 (×2 for the top-12), roughly +2-5% of
     the step.
   - **Effect**: +0.04 carried content (top-8) and +0.10 (top-12) at S = 512; +0.07 / +0.16 at S = 64.
   - **Implementation**: about 30 lines.
   - **Verdict**: cheap and modest. It pairs well with the coarse level.
4. **Online migration** (re-fit + vote map + copy, every K steps):
   - **Cost per event**:
     - k-means of the 24 codebooks on a 64 K-query reservoir, 10 iterations: about 5·10¹¹ FLOP, about 20-50 ms;
     - the vote map from the reservoir: ms;
     - the 1.07 GB V gather / copy: about 5 ms of bandwidth, but it needs a second 1.07 GB buffer (or a chunked in-place
       permutation).
   - Every 1,000 steps that is under 0.0001 s/step amortised.
   - **Verdict: not worth it**, because the content carried is only 6% at S = 512 (25-30% at S = 64), below freezing.

## Bottom line
- The owner's life-cycle idea is consistent with the data: early on the addresses turn over at about 70-80% per 1,000 steps
  whatever the keys do, so early memories *should* decay. The keys should stay fixed-but-distinct with a dead restart, and
  should not chase the thoughts.
- Make the switch, and the freeze, wait for the measured stability (stab_w ≥ 0.9) or the LR decay, instead of a fixed 60%.
- For more stability before that, add a coarse S = 64 level and top-8 / top-12 neighbour writes.
- Tracking keys and migration lose to simple freezing under the measured drift.
