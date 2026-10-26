# 25b. Per-pass memory: the row memory kept, plus a per-pass thought-keyed bank store, a per-pass read gate, rank-based confirmation and windowed drift transport (PLAN ONLY) — 2026-10-24

Revises plan 25 (`25_recall_reencode_memory_plan.md`, kept unchanged). Builds on plans 21 (life cycle, lazy fading),
22 + `22_gate_code.diff` (the MemGate router, already in `memory.py` / `loop_brain.py`), 23 (store the write-time key,
read the nearest one), 24 (key renewal is first-order) and the real-data results
`speed_wip/splat/realcheck/salvage/results.md` (run `world_mem_life_12k`, checkpoints 6K → 7K).

Nothing was trained, no code was edited, the GPU was not used (a VRAM profiling job owns it). Only this file was
written.

**The owner's hard constraints** (each is checked in §17):
1. Addresses come only from the model's thoughts: no token keys, and no address part is learned by gradient.
2. The thoughts are never constrained for the memory's sake (the address path is stop-gradient).
3. No dataset pointers, and no recompute loop outside the training step.
4. The memory is filled during training.
5. Speed ≥ 10k tok/s (aim 20k) on one 24 GB GPU; peak **reserved** VRAM ≤ 20 GB.
6. The brain decides itself, at every pass, whether to read.
7. **Each pass has its own memory.** Pass k writes and reads its own key bank, keyed by its own thought taken
   BEFORE that pass's memory inject.
8. Self-reinforcement (memory → thought → key → memory) is measured, not forbidden.

Structure:
- §0 what the evidence implies (including two new findings: the data loader, and the saturated content gate);
- §1 the architecture in one picture;
- §2 the bank store; §3 the view and the read;
- §4 the per-pass read gate; §5 the benefit signals;
- §6 writes; §7 re-encode on confirmed recall and the renewal probability;
- §8 fading and clocks; §9 drift transport;
- §10 the after_step order and the no-host-sync implementation;
- §11 coexistence with the row memory and ConceptHier;
- §12 self-reinforcement (what is measured);
- §13 training data; §14 metrics and eval; §VRAM; §15 speed budget;
- §16 test plan with go / no-go gates; §17 resolution of the plan-25 review issues 1-21;
- §18 flags, defaults and implementation size; §19 risks and open questions.

---------------------------------------------------------------------------------------------------------------------

## TL;DR

1. **Hybrid, not replacement.** The fixed-random-address shared row memory (plan 21: fine 4 levels × 3 heads × 512²
   rows + coarse 64²) STAYS. It is the only thing that has worked on real data: late cloze exact 0.280 vs 0.182 with
   it knocked out, and vs 0.260 for the no-memory control at the END of training. A new **bank store** is added next
   to it, for the facts the rows do not keep: rare facts (~10 exposures), where the rows give 0.016 vs 0.013 for the
   control.
2. **Per-pass banks.**
   - One slot store, partitioned into 9 banks b = 0..8. The read after core pass k (k = 0..K−2) uses bank
     b(k) = min(k, 8).
   - Its key and its query are the pass-k **pre-inject view**: the mean of the slot-normalised workspace slots 0-14
     after pass k, in a frame fixed per bank. Slot 15 is excluded because it carries the memory's own output. This
     view is taken before read k's output is written into the workspace and into pass k+1's inject.
   - The bank capacities are sized from how often each pass reads and writes (§2.2): 48K slots for banks 0-4, down to
     16K for bank 8, ≈ 360K slots in all, ≈ 0.5 GB.
   - Salvage b5: same-pass recall across the 1K gap is 0.77 (pass 1) → 0.71 (pass 6), R@16 0.95-0.98. A pass-1 bank
     queried from pass 6 reaches only 0.40. So per-pass banks are both what the owner asked for and what the data
     prefers.
3. **A per-pass read gate (plan 22's MemGate, extended).**
   - It has two logits per read: one for the rows, one for the bank. Each bank has its own weights (≈ 1.7K
     parameters).
   - The features are plan 22's 93 detached ones. Selection is by top-c plus a 10% uniform exploration floor; static
     shapes; c = 1 until 20% of the run, then a cosine to c_f = 0.25 (bank) / 0.30 (rows) by 50%. At inference, the
     EMA threshold θ̂_k.
   - **New: the target.** The row content gate is saturated (probe mean 0.999 at step 7K), so plan 22's target ḡ is
     useless as a ranking target. The target becomes the per-position benefit c = −⟨∂L/∂M, M⟩, **centred and
     rank-normalised per micro-batch and per pass**. That fixes the zero-sum problem (§5.1). Only in-window reads have
     it, so only they train the router.
   - This is the direct answer to "the reads help the answer tokens (−0.29 nats) but hurt the other 83% of world text
     (+0.13 nats)".
4. **Writes.**
   - Each in-gradient-window pass writes into its own bank: 16 positions per micro-batch per bank. They are the
     eligible positions with the highest final-pass CE, where the bank has no confident read.
   - Key = the pre-inject view. Value = the normalised gradient write (as plan 25), then trained by compact
     row-Adagrad.
   - **Dedupe at write**: d₁ < γ h₀² merges into the nearest slot, and duplicates within one step are greedily
     suppressed. Eviction never takes a slot touched this step.
5. **Confirmation and re-encode.**
   - A key moves toward the current view only on a confident read: geometry thresholds calibrated on data (§16 T0e),
     whose benefit is in the top quartile of its bank's in-window reads in the micro-batch.
   - There is **no sign-based disconfirmation and no spawn-on-hurt** (c is ≈ zero-sum).
   - A forward-only alternative (the memory-only read-out test CE_mem < CE_pass − m) is specified and must be
     calibrated first. It would also confirm reads outside the gradient window.
   - **The renewal probability is a product** p_ren = p_open · p_win · p_ret · p_conf (§7.3). For recurring facts it
     is far too small to carry them alone: one more reason why the rows stay.
6. **h₀ comes from all reads**, not only the confirmed ones (which shrink it to zero; §3.3). It is floored by the
   bank's key nearest-neighbour spacing.
7. **Transport** (optional stage 2, `--bm-transport`).
   - Windows are aligned to the neurogenesis cadence, so that no window straddles a birth; births get their own
     before / after pairs.
   - Both window ends use the SAME probe routine: eval routing, dropless, head-group offset 0, noise off, a frozen RNG.
   - Each bank has ≈ 1K anchors with a per-window estimate shrunk toward the accumulated per-anchor history mean (not
     toward zero), with a support threshold. Keys get a pro-rata share by their key time stamp.
   - **The math says that transport extends the recall horizon only modestly** (×1.3-1.75 at the measured R² ≈ 0.3-0.43).
     Keeping a key valid through a 3.3× silent gap needs R² ≥ 0.98 (directed drift) or ≥ 0.94 (diffusive). So
     transport is a lever for keys aged 0.5-2K steps, not a retention mechanism.
8. **New finding about the data**: the trainer samples uniform random 512-token windows from the whole 400M-token
   `train.bin` (`Data.batch`, no curriculum).
   - So the "early" people are NOT silent after step 2,441. Their 19 mentions sit in the first 10% of the FILE, and
     each mention is seen ≈ 0.5 times in expectation, at a uniformly random step.
   - The early group is therefore a **rare-fact group**: ≈ 9.5 exposures, mean spacing ≈ 1.3K steps, spread over the
     whole run. That is exactly the bank store's niche.
   - `exposures.npz` / eval_world's "steps since the last mention" buckets assume a sequential read and are wrong
     under this sampler.
   - The v2 data plan (§13) adds a sequential (time-ordered) loader and a scheduled rare-fact stream.
9. **Speed / VRAM.** The current run does 8-9k tok/s (below the 10k floor), with peak allocated 20.4-20.6 GB and
   reserved 17-23 GB (above the 20 GB cap).
   - The hybrid must therefore put the row memory on a diet, each step knockout-checked:
     - private memory off (ko_private ≈ 0; −1.3 GB);
     - neighbour writes off;
     - rows gated to ≈ 40% open;
     - optionally a compact row gradient (−1 GB and −0.1-0.18 s).
   - Estimate: **≈ 10.5-12.5k tok/s**, ≈ 18-19 GB allocated. 20k is not reachable while the rows AND ConceptHier both
     stay; that is stated, not hidden.
   - §VRAM is a placeholder until `speed_wip/vram/vram_study.md` exists (it does not yet).
10. **Order of work**:
    1. CPU now: T0a cross-document capture analysis, T0c directed-vs-diffusive, T0e calibration.
    2. GPU after the VRAM job: T0b (the benefit by token type in train mode), T2 bench.
    3. Stage A run: rows + the per-pass gate only. It must keep late cloze ≥ 0.27 and remove the +0.13-nat harm.
    4. Stage B: + the bank store.
    5. Stage C: + transport.
    - About 2,900 lines in all.

---------------------------------------------------------------------------------------------------------------------

## 0. What the evidence implies

### 0.1 The salvage results (6K → 7K), read as design constraints

| finding | number | design consequence |
|---|---|---|
| rows work for recurring facts | late exact 0.280; ko_shared 0.182; control final 0.260 | keep the rows (the baseline to beat); every change is knockout-checked against 0.280 |
| rare facts | early exact 0.016 vs control 0.013 | the bank store's job |
| reads hurt non-answer tokens | answer −0.29 / −0.20 nats; other world +0.13 on 83% of world tokens; wiki ±0.01 | a per-position, per-pass read gate (§4) |
| the content gate is saturated | probe "gate 0.999 gate_c 0.908" (train.log, step 7000) | its σ′ ≈ 0.001: it can no longer learn to close. It is not a usable router target (§4.3); the bank's content gate gets a logit penalty (§3.4) |
| rows are template-keyed | same template other person J 0.24; same fact other template 0.28; same person other relation 0.045 ≈ baseline | no entity slot; the bank keys may be template-dominated too, so the cross-template retrieval (T0a) is the first gate |
| self-recall 1K gap, pass 1 | R@1 0.774 / R@16 0.983 (4K bank); 0.568 / 0.882 (32K) | recall falls with bank density, so the banks are kept small (§2.2) |
| world text recalls worse | world 0.525 vs wiki 0.902 (4K) | exactly the text the memory is for; transport lifts world to 0.729 |
| rotation / noise | o ≠ o′ 0.995-0.998 within a checkpoint; across the gap, random offsets 0.795 vs pinned 0.793 | **no pinning of a head group needed** (review issue 2 closed by data) |
| transport | per-point field 0.774 → 0.884 (R² 0.43); 4K anchors num/den 0.854 (R² 0.31); 1K anchors 0.839; move the keys, not the queries (LS on queries 0.725) | per-anchor accumulated estimator, keys moved (§9) |
| per-pass banks | same-pass 0.774 / 0.753 / 0.747 / 0.734 / 0.723 / 0.707 (passes 1-6); pass-1 bank from pass 6: 0.401 | per-pass banks (§2) |
| pre-inject views | slot 15 excluded / memory off: +0.00…+0.01 same-pass | use slots 0-14 (cheap; also removes the memory's own slot from its key) |
| drift scale | median drift / 1-NN gap 1.31 (pass 1) … 1.27 (pass 6); rel drift 0.69-0.76 | h₀ must be of the order of the drift, not of the confirmed-read distance (§3.3) |

### 0.2 New finding 1: the training sampler makes "early" facts rare, not silent
- `train_lm.py: Data.batch` draws `gen.randint(0, len(train) − 513, B)`: uniform random windows over the whole
  400,000,000-token file. The run's config has `"curriculum": ""`, so the bucket path is off.
- A 12.2K-step run reads 200M tokens, i.e. every file position is covered **0.5 times in expectation**, at a uniform
  random step.
- The early people's 19 mentions per fact sit at file positions < 40M (`meta.json`: early mean 19.0 at 40M). So each
  early fact is seen ≈ 19 × 0.5 = **9.5 times, spread uniformly over the run, mean spacing ≈ 12,207 / 9.5 ≈ 1,290
  steps**. It is NOT "19 times before step 2,441, then never".
- The late facts: ≈ 233 × 0.5 ≈ 116 exposures, spacing ≈ 105 steps (plan 25's number is right for them by
  coincidence).
- Consequences:
  1. The "early retention" metric measures **recall of a rare fact (~10 exposures, ~1.3K spacing)**, the hardest
     regime for the weights and the rows, and the natural niche for a thought-keyed store.
  2. At the final checkpoint, the age of a rare fact's most recent exposure is roughly exponential with mean ≈ 1.3K
     steps. So the relevant key ages are 0.3-3K steps, exactly the range of the salvage gap test.
  3. Plan 25's "9.8K silent steps ⇒ transport R² ≥ 0.94" scenario does not exist in this data. Transport matters for
     ages of 0.5-2K steps (§9.6).
  4. eval_world's exposure buckets (`--exposure-at 200000000` reads `exposures.npz` keyed by FILE position) are
     mis-specified under random sampling. §13 fixes it with a sequential loader and an exposure log by step.

### 0.3 New finding 2: the row content gates are saturated open
- The probe statistic `mem_gate_mean` = 0.999 (fine) / 0.908 (coarse) at step 7,000.
- The gate is σ(a) with b initialised at −2. At a mean of 0.999 the typical logit is ≈ 7, and σ′(a) ≈ 0.001. The
  gate's gradient ∂L/∂a = σ′(a)·⟨h, W_o r⟩ is then ~1000× smaller than at init. **The content gate can no longer
  learn to close**, which is consistent with the reads injecting into every position.
- Consequences:
  - (i) plan 22's router target ḡ (the mean content gate) is ≈ constant, so it is no ranking signal: the router needs
    the benefit target of §5.
  - (ii) The bank's own content gate gets a small logit L2 penalty, so it stays in its sensitive range.
  - (iii) The rows' gates get the same penalty under a flag (`--mem-gate-l2`, Stage A ablation).

### 0.4 The current speed and VRAM (train.log, steps 6,750-7,750)
- 8-9k tok/s (≈ 1.82-2.05 s/step for 16,384 tokens).
- Peak allocated 20.2-20.6 GB, reserved 16.2-23.1 GB. The spill guard is at 21.5 GB; at step 7,500 reserved was
  23.1 GB.
- **Both owner limits are violated today** (≥ 10k tok/s, reserved ≤ 20 GB). Any addition must be paid for by cuts
  (§11.2, §15).

---------------------------------------------------------------------------------------------------------------------

## 1. Architecture in one picture (read k = after core pass k, feeding pass k+1; k = 0..K−2)

```
 core pass k ──► x_k, s_k
   │
   ├─ (training) read-out ce_k, lp_k          (already computed: deep supervision; moved before the read as in plan 22)
   ├─ ConceptHier(n_s(s_k)) → μ_l, ν_l        (unchanged; the rows' addresses)
   ├─ view  v_k = mean_{i<15} n_s(s_k)_i      (64-d, pre-inject: slot 15 = memory output excluded) ── sg ──┐
   ├─ router φ_k (93-d, detached) → s^row_k, s^bank_k                                                     │
   │      training: top-c_k + 10% floor per store │ eval: s > θ̂_k                                         │
   ├─ ROWS  on O^row_k : plan-21 read (fine + coarse) ─► M^row_k                                          │
   ├─ BANK b = min(k, 8) on O^bank_k : x̂ = (v_k − μ_b)/σ_b ─► index lookup (no grad) ─► top-4 (J, d) ◄──┘
   │      w, A ─► r = A Σ w_j f_j v_j ─► content gate g ─► M^bank_k = W_o^(b) (g r)
   └─ M_k = M^row_k + M^bank_k ─► inject_{k+1} += M_k ; slot 15 += W_ms M_k   (today's destinations)

 after the step (after opt.step): row MemOpt (as today) ; bank: compact Adagrad → re-encode → transport → evict/spawn
```

- Gradient reaches: the bank values (compact buffer), W_o^(b), the content-gate weights, W_ms, the router (BCE only),
  and p̄ through the content gate's feature input (as in today's `read_shared`: a normal downstream use, not a
  constraint).
- No gradient reaches: the keys, the frame, the index, w, A, h₀, the strengths and clocks, and the view on the
  address path.

---------------------------------------------------------------------------------------------------------------------

## 2. The bank store

### 2.1 Layout (one store, banks as contiguous slot ranges)
- Slots i = 0..N_s−1; bank b owns the range [o_b, o_b + N_b). Per slot:

| field | dtype | set by | meaning |
|---|---|---|---|
| k_i (64) | fp32 | rules (§6, §7, §9) | key in the bank's fixed frame |
| ‖k_i‖² | fp32 | with k_i | distance GEMV |
| v_i (256) | fp32 | write init + compact row-Adagrad | value |
| S_i | fp32 | Adagrad | accumulator (importance) |
| log α_i | fp32 | rules | strength (eviction, confidence) |
| Λ^α_i | fp32 | rules | strength clock (reset by confirm / merge / spawn) |
| Λ^v_i, Ψ_i | fp32 | Adagrad touch | value fade clock and accumulator discount clock (lazy, plan 21) — **separate from Λ^α** (issue 12) |
| τ_i | int32 | rules | key time stamp: the step whose weights produced the key's view (for the transport share, §9.4) |
| m_i | int8 | rules | maturity 0..3 (spaced confirmations) |
| n_i | int8 | rules | merge count (≤ 8; for the dedupe merge rule) |
| t^touch_i | int32 | rules | the last step that touched the slot (eviction protection, issue 13) |
| u_i, b̄_i | int32, fp32 | rules | use count; running rank-benefit (diagnostic) |
| cell_{i,τ} (T = 4) | int32 | index | back-pointers |

### 2.2 Bank count and capacities (derived from how often each pass reads and writes)
- K is drawn as U{2..10} with probability 0.95, and U{12..16} with probability 0.05 (`sample_K`).
- Read k exists iff K ≥ k+2. Its gradient window (bptt 6) is K−6 ≤ k ≤ K−2.

      f_k   = P(K ≥ k+2)               = 0.95·(9−k)/9 + 0.05            (k ≤ 8)
      P_win(k) = P(k+2 ≤ K ≤ k+6)      = 0.95·|[k+2, k+6] ∩ [2,10]|/9 + 0.01·|[k+2, k+6] ∩ [12,16]|

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 (+ k ≥ 9) |
|---|---|---|---|---|---|---|---|---|---|
| f_k (read happens) | 1.00 | 0.89 | 0.79 | 0.68 | 0.58 | 0.47 | 0.37 | 0.26 | 0.16 (+0.05) |
| P_win(k) (in gradient window) | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 | 0.42 | 0.33 | 0.23 | 0.14 |

- **Review issue 1 in numbers**: read 0 is outside the window iff K ≥ 7, i.e. with P = 0.95·4/9 + 0.05 = **0.47**.
  So every per-pass mechanism that needs a gradient (value training, the c-based confirmation, the router's target,
  the value init of a write) runs only on in-window reads, with P_win(k) above. The others still read (no_grad) and
  still count for the strength / use clocks.
- **Why 9 banks (k = 0..8), with k ≥ 9 sharing bank 8**:
  - reads beyond k = 8 occur only on the 5% long-thinking steps;
  - within a checkpoint, passes 4-6 are interchangeable (p6 → q4 0.994);
  - the gap cost of sharing is p6-bank / q4-query 0.702 vs the own bank 0.707.
- Capacities: an unconfirmed slot must survive ≥ 3,000 steps (the largest rare-fact spacing, §13), at
  W = 16 writes per micro-batch per in-window bank:

      N_b ≥ 3,000 · 2 · 16 · P_win(b)   →   48K (b = 0..4), 40K (5), 32K (6), 24K (7), 16K (8);  N_s ≈ 360K

- Smaller is better for recall. The density matters: salvage 4K 0.774 vs 32K 0.568. A store's keys are sparse
  surprise positions from many documents, not every position of 32 windows, so the effective competition is smaller;
  T0d measures it at the real density.
- Flags: `--bm-banks 9`, `--bm-cap 48K,48K,48K,48K,48K,40K,32K,24K,16K`.
  - `--bm-banks 1` = a single pass-1 bank (the ablation);
  - `--bm-banks 6` = the cheaper variant (k ≥ 5 shared).

### 2.3 The frame (fixed per bank, no gradient)
- x̂ = (v − μ_b)/σ_b, with μ_b ∈ R^64 and σ_b a scalar RMS, fitted once on 64K positions of pass b at
  `--bm-fit-step 300`. Raw p̄ beat whitening at the full 3K gap, and the gap test used this frame.
- The frame is never refitted: keys and queries share it forever. The drift is handled by re-encoding and
  transport. A global shift of the thoughts is just the constant part of the transport field.

### 2.4 The index (per bank), with the low-rank skew fix (review issue 17)
- **The problem**: p̄ is low-rank (plan 16: participation ratio 3-4 late in training). A product quantiser over two
  32-d halves of a randomly rotated view has strongly correlated halves. Only a thin "diagonal" of the 128 × 128
  product grid is populated: hot buckets overflow while most cells are empty, and the candidate recall drops.
- **The fix (OPQ-style eigenvalue allocation; Ge et al. 2013)**. The index uses its own coordinate z = Vᵀ(x̂ − m)
  (the PCA of the bank's keys, refitted with the codebooks). The principal components are dealt alternately to the
  two halves (1, 4, 5, 8, … → half A; 2, 3, 6, 7, … → half B: a balanced-variance allocation).
  - PCA components are uncorrelated, so the halves decorrelate and the product grid fills.
  - The T = 4 tables use 4 different allocations (permutations within equal-variance groups) and a small random
    rotation within each half.
  - **Distances for ranking stay in the raw frame x̂**; z is only for cell assignment.
- Per bank, per table: codebooks 128 per half (k-means, 5 iterations), 128² = 16,384 cells, cap 8, kc = 4 probed
  cells, so 4 × 4 × 8 = 128 candidates. Buckets: 9 banks × 4 tables × 16,384 × 8 × 4 B ≈ 19 MB.
- Banks ≤ 16K slots (bank 8) may instead use brute force (`--bm-brute 16384`): it is read on ≤ 21% of steps.
- Maintenance: the codebook / PCA refit every 1,000 steps on 32K keys per bank sampled by strength, **re-filing in
  chunks of 32K slots** (issue 14: the temp is 4 tables × 2 halves × 32K × 128 × 4 B = 134 MB per chunk), and the
  buckets rebuilt by a sort that keeps the 8 strongest per cell. It changes only the index, never a key.
- Monitored per bank: occupancy (max / mean bucket load, the fraction of empty cells), the pointer coverage (live slots
  with ≥ 2 table pointers; target ≥ 0.95), and the candidate recall vs brute force on 1K sampled queries every 250
  steps.

---------------------------------------------------------------------------------------------------------------------

## 3. The view and the read

### 3.1 The view (pre-inject)
- v_{k,t} = (1/15) Σ_{i=0}^{14} n_s(s_k)_{t,i}. It is computed inside `_hier_step.block()`, next to today's
  `pbar = ns.mean(-2)` (one more mean over a slice). The rows keep their 16-slot p̄ (unchanged addresses).
- "Before that pass's memory inject": s_k is the workspace after core pass k, BEFORE read k's `_slot_write` and
  before read k's M enters inject_{k+1}. Earlier reads (k′ < k) still influence s_k through pass k's input. That is
  the feedback measured in §12, not removed.
- The salvage measured this exact view: slot 15 excluded gives +0.003…+0.012 same-pass recall over the 16-slot mean.

### 3.2 The lookup and the read kernel (per bank, only for the open positions O^bank_k; Q_o = |O| static)
- The index lookup is as in plan 25 §2.1, in the bank's tables: centroid scores → top-4 cells per table → bucket
  gather → 128 candidates → exact distances in x̂ → top-4 winners (j₁..j₄, d₁ ≤ … ≤ d₄). About 36 ops; no_grad; cached
  for the checkpoint recompute.
- The read kernel (unchanged form):

      ã_j = log α_j − λ_α ρ_j (t − Λ^α_j)                       (lazy strength; ρ_j = 4^{−m_j})
      w_j = softmax_j( ã_j − (d_j/d₁ − 1)/τ_w ),   τ_w = 0.1
      A   = exp(−d₁/(2 h₀,b²)) · σ(ã_{j₁})
      f_j = exp(−λ_v ρ_j (t − Λ^v_j))                           (lazy value fade, its own clock)
      r   = A Σ_j w_j f_j v_j

### 3.3 h₀ from all reads, floored by the key spacing (review issue 6)
- **Why plan 25's h₀ collapses.** Plan 25 set h₀² = EMA of the median d₁ over the CONFIRMED reads. Confirmation
  requires A ≥ A_min, i.e. d₁ ≤ c₀ h₀² with c₀ = 2 ln(1/A_min).
  - Let F be the cdf of d₁ near 0. Its density behaves as d^{d_eff/2 − 1} (the volume growth of a d_eff-dimensional
    neighbourhood).
  - The median of the truncated variable is m(h²) = F⁻¹(F(c₀h²)/2). With F(x) ∝ x^{d_eff/2}, m = 2^{−2/d_eff} c₀ h².
  - The EMA update h²_{n+1} = m(h²_n) = c₀ 2^{−2/d_eff} h²_n contracts to the fixed point h² = 0 whenever
    c₀ < 2^{2/d_eff}.
    - A includes σ(ã): A ≥ A_min ⟺ d₁ ≤ 2h² ln(σ(ã)/A_min). So c₀ = 2 ln(σ(ã)/A_min).
    - A fresh slot (log α = 0, σ(ã) = 0.5) gives c₀ = 2 ln(0.5/0.3) = 1.02 < 2^{2/d_eff} for every d_eff: **a
      geometric collapse at rate ≈ 1.02·2^{−2/d_eff} per EMA time constant** (×0.65 at d_eff = 3).
    - Only strong slots (σ(ã) → 1, c₀ = 2.41) escape, and only for d_eff ≳ 1.6. There, what sets h₀ is the cdf's
      tail, not the drift.
    - The extra margin / w₁ conditions select close reads further (they lower the effective c₀).
  - Either way, h₀ is set by a selection effect, not by the drift. **The feedback loop: confirmed reads are close →
    h₀ shrinks → fewer reads pass A_min → those left are closer.**
- **Rule.**

      h₀,b² ← (1 − 0.01) h₀,b² + 0.01 · q₅₀( d₁ over ALL open reads of bank b in the step )
      h₀,b² ← max(h₀,b², ½ · q₅₀(NN distance² among 4,096 sampled keys of bank b))      (refreshed every 250 steps, brute force)

  - init: q₅₀(d₁) of the first read step after `--bm-write-start`.
  - The population is no longer selected by A, so there is no contraction.
  - The floor ties h₀ to the key spacing, which sets the scale at which "nearest" is meaningful.
  - The median drift is 1.31 × the 1-NN gap (salvage), so a typical drifted owner read has A ≈ e^{−0.86} ≈ 0.42 at
    h₀² = NN²: still above A_min = 0.3.

### 3.4 The content gate and the output (per bank)

      φ^c = [p̄_k (64); r (256); A; log(d₂/d₁); onehot(m_{j₁}) (4); log(1 + age_{j₁})]     (327)
      a   = w_cᵀ φ^c + b_c,   g = σ(a),   b_c = −2;      L_gl = λ_gl · mean(a²),  λ_gl = 1e-3   (keeps |a| ≲ 3, σ′ ≥ 0.045)
      u   = g r,   M^bank = W_o^(b) u   (W_o^(b) ∈ R^{384×256}, zero init, one per bank),   then added to M_k

- **Per-bank W_o** (9 × 98K = 0.9M parameters) instead of a shared W_o with a pass embedding.
  - The owner's "one pass may need history, another physics, via different routes": each bank's values live in their
    own output space.
  - A shared W_o would force all banks' values into one code.
  - `--bm-wo shared` (+ a 16-d pass embedding into the gate) is the ablation.
- `Probe(u)`: an identity autograd function whose backward records c = −⟨∂L/∂u, u⟩ per position (2 ops), used in §5.

---------------------------------------------------------------------------------------------------------------------

## 4. The per-pass read gate (MemGate extended; plan 22 reused)

### 4.1 What is reused unchanged (already in the code: `memory.py: MemGate, gate_sizes, gate_c, gate_trace`;
`loop_brain.py: _gate_phi, _gate_open, _gate_after, _gate_readout, _gate_surprise`)
- The 93 detached features: p̄ (standardised), ν̄_l, Δμ_l, the thought change, the last-token surprise + flag, the
  windowed surprise trace, the next-token entropy, the pass one-hot.
- The read-out reorder: training computes the pass-k read-out before the read.
- The fp32 router in its own AdamW group and its own clip, excluded from the global clip.
- The selection: the top-n_c by score + n_ε uniform outside it, multiples of 64, a device-side generator, a fixed tie
  key, cached for the recompute.
- θ̂_k updated only on the first training run of the block.
- Eval modes learned / random / inverted / oracle / open / off.
- Readdr off under the gate. The norm statistics over ALL positions.

### 4.2 What changes
1. **Two stores, per-bank weights.** s^{(m)}_{k,t} = W^{(m)}_{b(k)} φ_{k,t} + β^{(m)}_{b(k)}, with m ∈ {row, bank}:
   W ∈ R^{2×9×93}, ≈ 1.7K parameters.
   - Each pass has its own decision rule, as the owner asked, not just a pass bias.
   - Two open sets per read, O^row_k and O^bank_k (two independent top-c selections); each has its own θ̂^{(m)}_k.
2. **The schedule per store**: c = 1 until 0.2·S, then a cosine to c_f = 0.30 (rows) and 0.25 (bank) at 0.5·S. The
   floor is 10% once c < 1. So the open shares are ≈ 0.40 / 0.35, on every pass. The budget is static per pass and per
   store (host integers, so no sync).
   - Why so low? The answer tokens are ≈ 17% of world tokens; world is 20% of the data, so ≈ 3.4% of all tokens. A
     35-40% budget is generous for them and closes most of the 83% where the reads hurt.
3. **The target** is the benefit of §5, not the content gate (§0.3).
4. **The inference threshold**: θ̂^{(m)}_k is the EMA of the training cut-off, as in plan 22 §2.6.
   - Its consistency (plan 22 Proposition 4) is unchanged.
   - Its sd ≈ √(c(1−c)/N_eff)/f(q) · √(ρ/(2−ρ)), with N_eff ≈ 256: ≈ 0.03/f(q) per batch, ≈ 0.003/f(q) for the
     EMA.

### 4.3 The BCE target and why it is consistent
- For the in-window open positions t ∈ O^{(m)}_k of read k:

      y^{(m)}_{k,t} = Φ( (c^{(m)}_{k,t} − q₅₀) / (1.4826 · MAD) )            (median / MAD over O^{(m)}_k in this micro-batch)
      L_r = λ_r · mean_{k in window, t ∈ O} BCE( σ(s^{(m)}_{k,t}), y^{(m)}_{k,t} ),   λ_r = 0.1

  - c^{row} comes from a Probe on u^row (summed over the levels / pools); c^{bank} from the bank's Probe.
  - Robust centring: a few huge |c| do not decide the scale.
- **Consistency (plan 22 Proposition 2 carries over).**
  - (i) The selection depends on φ and the exploration draw only.
  - (ii) y_t depends on position t's own read through c_t and on the micro-batch's median / MAD. The latter is a
    batch-level constant: it shifts all targets alike and does not depend on the selection of t.
  - (iii) Positivity: the floor.
  - So y ⊥ 1[open] ∣ φ (missing at random), and the BCE minimiser on the open sample is σ(s) = E[y ∣ φ] on the whole
    support.
  - The centring makes the target a within-batch rank. That is what a top-c selection needs: only the order matters
    for training-time selection, and θ̂ inherits the rate.
- **Out-of-window reads**: no c and no target. They use the router's score (no_grad) but do not train it. The share of
  reads that train the router is P_win(k)/f_k ≈ 0.53 for k = 0 … 0.87 for k = 8.
- **What it fixes**: on non-answer world tokens the salvage measured +0.13 nats (the read hurts). If the train-mode c
  shows the same sign by token type (test T0b), y < 0.5 there, so the router learns to close them.
  - If T0b shows c ≥ 0 on those tokens in TRAINING mode while the eval harm persists, the harm is a train / eval
    mismatch, and a gate trained on c cannot see it (§19, risk 2).

### 4.4 Optional: the brain sets its own budget (`--gate-adapt 1`, default off in v1)
- Every 250 steps (host reads only at logging steps), per pass and per store:
  - compare the mean UNCENTRED c of the explored floor positions whose score lies just below the cut-off (the band
    [q_{c−0.05}, q_c]) with that of the lowest decile of the top-c;
  - c_k ← clip(c_k ± 0.05, 0.10, 0.60) toward the side with the higher marginal benefit.
- Caveat: Σ c ≈ 0 overall (§5.1), so only the marginal difference of the two bands is used, never the sign of a mean.

---------------------------------------------------------------------------------------------------------------------

## 5. The benefit signals

### 5.1 c = −⟨∂L/∂M, M⟩ is ≈ zero-sum (review issue 5), so only ranks / centred values are used
- With M_t = W_o u_t: ∂L/∂W_o = Σ_t g_t u_tᵀ (g_t = ∂L/∂M_t), so

      Σ_t c_t = −Σ_t ⟨g_t, W_o u_t⟩ = −⟨∂L/∂W_o, W_o⟩_F

- Near a stationary point of W_o under AdamW with weight decay ω, ∂L/∂W_o ≈ −ω W_o, so Σ_t c_t ≈ ω ‖W_o‖_F² ≥ 0 and
  small. Without decay, it is ≈ 0 up to gradient noise.
- So the batch mean of c is pinned near 0 whatever the memory's value. About half of all reads look harmful by sign
  even when the memory is very useful.
- Hence:
  - (i) no sign-based disconfirmation and no spawn-on-hurt (both in plan 25);
  - (ii) the router target and the confirmation use the per-micro-batch centred / ranked c;
  - (iii) the absolute usefulness of the memory is measured by knockouts at eval (§14), never by mean c.
- Note: what is pinned is the AGGREGATE. Per position, c is informative: answer tokens vs others is exactly the
  contrast the salvage found.

### 5.2 The forward-only memory test (an alternative; must be calibrated before use)
- For the confident bank reads only (the top E_b = 128 per bank per micro-batch by A):

      CE_mem,t  = −log softmax( W_lm normf(M^bank_{k,t}) )[y_t]       (the memory output decoded alone)
      CE_pass,t = ce_k[t]                                               (pass k's own read-out, already computed in training)
      memory-confirmed ⟺ CE_mem,t < CE_pass,t − m_ce                    (m_ce from T0e)

- It is a real (not first-order) comparison and not zero-sum. It needs no gradient, so it **also confirms reads
  outside the bptt window** (p_win = 1 in §7.3).
- Cost: 128 rows × 384 × 16,384 ≈ 0.8 GFLOP per read; ≈ 6 ops per read.
- **Caveat**: M enters the next pass through n_u(x + … + inject), not the read-out, so M is not trained to be
  decodable by the LM head. Whether it is (on fact positions) is a measurement (T0b): top-1 answer accuracy of the
  decode, and the precision of the test against the true knockout benefit.
- Variant `--bm-confirm skip`: CE(W_lm normf(x_k + M)) instead of CE_mem, a skip-path proxy, calibrated the same
  way.
- Default: `--bm-confirm rank` (§7.1). The CE test is enabled only if T0b passes (precision ≥ 0.85 at recall ≥ 0.3
  on fact positions).

---------------------------------------------------------------------------------------------------------------------

## 6. Writes

### 6.1 Which pass writes
- **Each in-window read k writes into its own bank b(k).** Out-of-window passes do not write: their value init needs
  ∂L/∂inject_{k+1}, and there is none.
- `--bm-write-nograd 1` (ablation): out-of-window spawns get the value σ_ξ ξ only, and learn it on their first
  in-window read.
- So bank b receives writes on P_win(b) of the steps (§2.2).

### 6.2 The surprise and the eligibility
- σ_t = the final-pass next-token CE (detached; `ces[-1]`). It is computed with the rows ACTIVE, so "the rows already
  know it" ⇒ low σ ⇒ no write. This is the division of labour of the hybrid, with no extra rule.
- Eligible_k(t) ⟺
  - t ≥ 8 tokens after a document start, and t is not an EOT;
  - AND the bank has no confident read at (k, t): either t ∉ O^bank_k (closed; dedupe catches what the bank already
    has), or A_{k,t} < A_w = 0.3.
- Budget: per micro-batch per in-window bank, W = 16 positions = the top-16 eligible by σ.
  - At most 2 per sequence; at least 8 tokens apart (a max-pool suppression).
  - topk on (B, T) with static shapes: 16 × 2 micro-batches × ≤ 5 in-window banks ≈ ≤ 160 per step.
- Candidate stash (from the micro-batch's forward and backward): the index, x̂_{k,t} (64), σ_t, and
  ĝ_{k,t} = −∂L/∂inject_{k+1}[t] (384).
  - ĝ is captured by a hook on the summed inject at the W candidate positions (the indices are known at the end of the
    forward; the hook gathers 16 × 384 floats).
  - Total ≈ 25 KB per bank per micro-batch.

### 6.3 Key and value
- Key: k = x̂_{k,t}, τ = the current step.
- Value: v = σ_ξ ξ + ν v̄_b · W_o^(b)ᵀ ĝ / ‖W_o^(b)ᵀ ĝ‖, with σ_ξ = 0.02, ν = 1, v̄_b = the EMA of the median ‖v‖ of
  bank b's slots with m ≥ 1 (init 0.05).
- This is the steepest-descent direction of r through W_o^(b) (plan 25 §5.2). ξ breaks the W_o = 0 deadlock at the
  start.

### 6.4 Dedupe at write and the merge rule (review issue 7)
- **Against the store.** In after_step, the ≤ 2 × 16 candidates per bank get a small lookup in their bank (one
  batched call for all banks: ≤ 160 queries). Let d₁, j₁ be the nearest live key.

      d₁ < γ h₀,b²   (γ = 0.25, i.e. a read there would have A ≥ e^{−γ/2} · σ(ã) ≈ 0.88 σ(ã))   ⟹  MERGE into j₁:
         k_{j₁} ← k_{j₁} + (x̂ − k_{j₁}) / (n_{j₁} + 2)          (a running mean with the old key counted as n + 1 ≥ 1 samples)
         v_{j₁} ← v_{j₁} + (v_new − σ_ξ ξ) / (n_{j₁} + 2)        (the gradient-write increment, damped by the count)
         τ_{j₁} ← τ_{j₁} + (t − τ_{j₁}) / (n_{j₁} + 2);  n_{j₁} ← min(n_{j₁} + 1, 8);  log α_{j₁} += 0.5;  Λ^α_{j₁} ← t
      otherwise ⟹ SPAWN a new slot (§6.5)

  - Why γ = 0.25: two writes at d₁ < γh₀² would read each other with A ≈ σ(ã) at d₁ = 0 down to ≈ 0.88 σ(ã) at the
    threshold. They are the same memory to every reader, and two slots there split the reads and the value gradient
    (each gets ≈ half), the interference that plan 25 §12.8 left open.
- **Within the step** (two candidates of the same fact in one batch): the D = (32 × 32) distance matrix per bank
  among the candidates, ordered by σ. Candidate i is suppressed if some j earlier in the order (higher σ) has
  D_ij < γh₀² (an upper-triangular mask and an any(): 4 ops, static).
  - A suppressed candidate's value increment is added to the kept one's (an index_add_ by the "kept representative"
    index).
- Merges do not consume eviction capacity.

### 6.5 Spawn and eviction
- A spawned slot gets: k = x̂, v as in §6.3, log α = 0, Λ^α = Λ^v = Ψ = τ = t, m = n = 0, S = S₀,b (auto = 100 ×
  the median of the first step's g², per bank).
- Victims: the free list first, then the weakest by lazy strength ã_i in the bank, **excluding slots with
  t^touch_i = t** (read, confirmed, merged, transported or Adagrad-updated this step; issue 13). It is one masked topk
  over the bank's strengths (≤ 48K): 3 ops.
- Turnover: an unconfirmed bank-0 slot lives ≈ N_0 / (32 · 0.53) ≈ 2,830 steps on average; the strength fade makes
  older unconfirmed slots the victims first. Mature slots (ρ = 1/64) are practically never the weakest.

---------------------------------------------------------------------------------------------------------------------

## 7. Re-encode on confirmed recall

### 7.1 The confirmation rule (default `--bm-confirm rank`; review issues 5 and 18)
- Read (k, t) of bank b is **confirmed** iff all of:
  1. **In window** (c exists).
  2. **Rank**: c_{k,t} is in the top q_c = 25% of the bank's in-window open reads of this micro-batch that also pass
     condition 3 (static: a topk on the masked score).
  3. **Geometry**: w₁ ≥ ω₁, d₂/d₁ ≥ μ_m, A ≥ A_min. The defaults (0.7, 1.15, 0.3) are placeholders. **They are
     calibrated in T0e** at the operating point where a top-1 that passes them is the same fact with precision
     ≥ 0.95 on the T0a capture.
  4. The budget: at most E_b = 64 per bank per micro-batch (the top by rank).
- With `--bm-confirm ce` (if T0b passes): conditions 1-2 are replaced by §5.2's test, for all reads, in or out of
  window.
- Removed vs plan 25:
  - the sign test c > κσ̂ (zero-sum);
  - the disconfirmation and the 16-step lock;
  - spawn-on-hurt;
  - the b̄ ≥ 0 condition (b̄ is now a rank average, logged only).

### 7.2 The update (after_step; order in §10)
- As plan 25 §4.2 (per-slot averaging over the step's confirmed readers; η = 0.5, η = 1 for u < 2; the move capped
  at h₀; the minimum distance to the competitor γ_d h₀²). Then:

      τ_j ← t;  Λ^α_j ← t;  log α_j ← min(ã_j + 1, log 8);  t^touch_j ← t;
      m_j ← min(m_j + 1, 3) if t − t^m_j ≥ 64;  re-file if a cell changed

  - The value clock Λ^v is NOT reset by a re-encode (issue 12): the value is renewed only by its own Adagrad touches.
- **The separation bound** (plan 25 §4.4) holds unchanged: ‖k₁′ − k₂‖ ≥ (√μ_m − (1 − η))√d₁ = 0.57√d₁ at μ_m = 1.15,
  η = 0.5. The calibrated μ_m changes the constant; T0e reports it.
- **The fixed point (wording, issue 21)**: plan 25 called k* = Σ ρ_s p_s x_s / Σ ρ_s p_s "the benefit-weighted
  centroid".
  - That is the mean-ODE limit for small η with STATIONARY readers.
  - With η = 0.5 and drifting readers, the key is a lagged EMA of its recent confirmed readers. Its lag is ≈ the
    drift over (1 − η)/(η · rate of confirmations) steps. The bias bound β R_basin applies to the stationary part
    only.

### 7.3 The renewal probability (review issue 8): a product, and what it implies
- A recurrence of fact F at step t renews F's key only if all four events happen:

      p_ren = p_open · p_win · p_ret(a) · p_conf
        p_open  = P(the bank gate opens at the cue position)         (≥ the floor 0.10; target ≥ 0.8 at answer cues)
        p_win   = P_win(k)                                           (0.53 for k ≤ 4; 1 with --bm-confirm ce)
        p_ret(a)= P(top-1 = F's key, passing the geometry | age a)  (T0a cross-document; × the self-recall drift curve)
        p_conf  = P(rank-confirmed | correct retrieval)              (≈ the top-quartile rate among correct reads; T1)

- **A survival model.** Treat the key as lost once its age since the last renewal exceeds the horizon H, where p_ret
  collapses (H ≈ the age at which self-recall R@1 drops below 0.5). Recurrences come at rate 1/Δ.
  - The probability that a renewal window of length H passes without a renewal is e^{−p_ren H/Δ}.
  - Over T steps there are ≈ T/H such windows, so the survival is

        S(T) ≈ exp( −(T/H) · exp(−p_ren H/Δ) )

  - **Late facts** (Δ ≈ 105, T = 12.2K, H ≈ 1,500). S ≥ 0.9 needs (T/H) e^{−p_ren H/Δ} ≤ 0.105, i.e.
    p_ren ≥ (Δ/H) ln(10 T/H) = 0.07 · 4.4 = **0.31**. Example: 0.6 × 0.53 × 0.8 × 0.5 = 0.13 < 0.31. **Re-encode
    alone cannot carry the recurring facts in a bank**, unless p_win → 1 (the CE test) and p_open → 1 at cues.
    - The rows do not need renewal at all (they are rewritten by gradient at every exposure). That is the quantitative
      reason for the hybrid.
  - **Rare facts** (Δ ≈ 1,290 with ~9.5 exposures). Renewal is nearly irrelevant: p_ren H/Δ ≈ 0.15. What matters is
    that each exposure **re-writes** the fact (a surprise write, a merge, or a re-encode), so that at the query the
    newest key is ≈ Exp(1.3K) steps old. The expected recall is then

        P(recall) ≈ p_write · p_open(eval) · E_{a ~ Exp(Δ)}[ p_ret(a) ] · p_val

    - with p_write = P(the last exposure produced or merged a key) (σ high: the weights do not know rare facts, and
      the exposure is in-window with probability ≈ 0.53 at bank 0, more banks raise it);
    - p_val = P(the value decodes the answer).
    - With the untransported curve (R@1 1.0 / 0.77 / ≈ 0.45 / ≈ 0.15 at ages 0 / 1K / 2K / 3K, 4K bank) and
      Exp(1.3K) ages, E[p_ret] ≈ 0.6 × the T0a cross-document factor. Transport moves the 1K point to ≈ 0.85
      (§9.6).
    - This is the number T1 computes, and the reason the T0a bar matters more than the transport.
- **T0a bar derived from it**: early / rare exact ≥ 0.10 (6× today's 0.016) needs
  p_write · p_open · E[p_ret] · p_val ≥ 0.10. With p_write 0.6, p_open 0.8 and p_val 0.6, that is
  E[p_ret] ≥ 0.35, i.e. **a cross-document same-fact R@1 ≥ 0.35 / 0.6 ≈ 0.6 at age 0** (on the same checkpoint). The
  T0a gate (§16) uses ≥ 0.5 as GO and 0.35-0.5 as a weak GO.

---------------------------------------------------------------------------------------------------------------------

## 8. Fading, maturity, clocks (lazy, plan 21 style; no switch, no freeze)
- **Strength** (clock Λ^α): ã_i = log α_i − λ_α ρ_i (t − Λ^α_i), with λ_α = 1e-3 and ρ_i = 4^{−m_i}. It is renewed by
  a spawn, merge or confirmation.
- **Value** (clock Λ^v, separate; issue 12): f_i = exp(−λ_v ρ_i (t − Λ^v_i)), with λ_v = 1e-4. It is folded into v_i
  at every Adagrad touch (v ← f v, Λ^v ← t), exactly plan 21 §1.1. MemRead receives w·f.
  - A key re-encode or transport does not touch Λ^v: moving a key does not make its content fresher.
- **The accumulator** (clock Ψ): the discount 1 − β_i = (1 − β₀) ρ_i, with β₀ = 0.99 and a floor 0.1 S₀
  (plan 25 §6). Immature values stay plastic; mature ones are protected (EWC-like).
- **Keys are never frozen.**

---------------------------------------------------------------------------------------------------------------------

## 9. Drift transport (stage C, `--bm-transport 1`; optional)

### 9.1 What it is for (the honest scope)
- In this data, a bank key's useful life is set by the gaps between a rare fact's exposures (§0.2, §7.3), i.e. ages
  of 0.3-3K steps.
- The salvage: untransported R@1 at 1K 0.774 → 0.854 with the per-anchor estimator. **Transport is a recall lift at
  ages ~1K, not a retention mechanism for silent facts** (§9.6 derives why).

### 9.2 The windows: aligned with neurogenesis, both ends probed the same way (review issues 3, 10, 19)
- **The probe routine** P(θ, seqs) is the same at both ends of every window:
  - eval routing (head-group offset 0, route noise 0), MoE capacity dropless (as eval_world `--capacity dropless`);
  - the memory read in inference mode (θ̂ thresholds), with no writes and no statistics updates;
  - no router θ̂ update, no neuro scoring hooks;
  - a forked RNG (`torch.random.fork_rng` plus the memory generators untouched), so resume is bit-exact;
  - forward only, depth D = 6 passes (banks 0-5), and D = 9 on every 4th event (banks 6-8).
  - It returns the pre-inject views v_k (k < D) of every position: no CE, no read-out.
- **The windows.** Let the neuro events be at steps e_n (every 25 steps, between 5% and 60% of the run).
  - A regular window runs from the step right after e_n to the step right before e_{n+1} (24 optimizer updates, no
    birth inside).
  - Event "after e_n": P on [S_{n−1} ∪ S_n], where S_n = 8 sequences taken from the current batch (token ids: 8 KB).
    S_n's views are the window's "before". S_{n−1}'s views are the post-birth end of the birth pair below.
  - Event "before e_{n+1}": P on S_n gives the window's "after".
  - **Birth pairs**: if e_n had births, S_{n−1}'s views just before e_n (the previous window's "after") and just after
    e_n form a 0-update window. Its displacement is the birth jump alone, measured cleanly, with nothing else
    changed.
  - If e_n had no births, the two probes collapse into one forward on 16 sequences.
  - After the last birth (60% of the run): windows of M = 100 steps, one probe forward on [S_{e−1} ∪ S_e] per window.
- **Both ends are after opt.step** (issue 10). A window from the probe at step t_a to the probe at step t_b spans
  exactly M = t_b − t_a optimizer updates: weights θ^(t_a+1) → θ^(t_b+1). There is no ÷M vs M+1 bias.
- The stash ring holds ≤ 3 sets × 8 sequences: 24 KB of ids + 3 × 8 × 512 × 9 × 64 × 4 B = 28 MB of views.

### 9.3 The estimator (per bank: anchors, a support threshold, shrinkage toward the history mean; issue 3)
- **Anchors**: A_b = 1,024 per bank (k-means on the bank's keys, refitted with the codebooks every 1,000 steps). Each
  pair i (b_i, a_i) of the window, Δ_i = a_i − b_i, is hard-assigned to its anchor j(i) by the midpoint
  (a_i + b_i)/2.
- Per window w, per anchor j (index_add_; static):

      n_j^w = Σ_{i→j} ω_i,    D̂_j^w = Σ_{i→j} ω_i Δ_i / max(n_j^w, ε),    ω_i = min(1, 3·med‖Δ‖/‖Δ_i‖)   (robust)

- **The history** (an EMA over windows of the per-anchor displacement RATE, i.e. per update):

      m_j ← ρ_m m_j + (1 − ρ_m) D̂_j^w / M_w        (only if n_j^w ≥ n_min = 4),   ρ_m = 0.9
      s²_ε,j ← EMA of ‖D̂_j^w/M_w − m_j‖²            (the per-window variability)
      σ²_η,j = EMA of the within-anchor pair variance / n_j^w     (sampling noise of D̂_j^w)

- **The applied displacement** for window w, the shrinkage form (James-Stein toward the history, NOT toward zero):

      λ_j = s²_ε,j / (s²_ε,j + σ²_η,j/M_w²)          (∈ [0, 1]: how much to trust this window over the history)
      Δ̃_j^w = M_w · [ λ_j · D̂_j^w/M_w + (1 − λ_j) · m_j ]      if n_j^w ≥ n_min
      Δ̃_j^w = M_w · m_j                                          if n_j^w < n_min and the anchor's total support ≥ 32 pairs
      Δ̃_j^w = M_w · m̄_parent                                    otherwise (the parent = 64 coarse anchors; else 0)

- **Why this, not plan 25's per-step NW.** Plan 25's Δk = (1/M) num/(w₀ + den) is shrunk toward ZERO by
  w₀/(w₀ + den). With den ≈ a few pairs per key per step, the shrinkage is a systematic under-transport of
  ≈ w₀/(w₀ + den) per step, and systematic errors accumulate linearly (plan 25 §7.5 itself).
  - The new estimator is shrunk toward the anchor's own accumulated mean, which is unbiased for the persistent
    (directed) part.
  - The weight λ_j is the Bayes-optimal trust in the current window when D_j^w = μ_j + ε_j^w (persistent mean +
    per-window part) is observed with noise η: the posterior mean of D_j^w is λ_j D̂ + (1 − λ_j) μ_j.
  - Both ends of the jitter-vs-drift trade-off are explicit: σ²_η (the jitter) and s²_ε (the non-persistent drift).
- **Apply** (after re-encode, before spawn; §10): every live key i of bank b with anchor j(i):

      share_i = clip( (t_b − max(τ_i, t_a)) / M_w, 0, 1 )      (a key made inside the window moves only for its remaining part)
      k_i ← k_i + share_i · Δ̃_{j(i)}^w · min(1, ρ_T h₀,b / ‖Δ̃‖)     (cap ρ_T = 0.5 per window);  τ_i ← t_b;  t^touch_i ← t

- **Birth pairs**: the same per-anchor average, applied with share 1 to every key with τ_i ≤ e_n (the keys older than
  the birth). Birth jumps do not enter the history m_j (they are not drift).
- **Recall pairs** (plan 25 §7.1 b) are dropped from the field (issue 11: they would double-count).
  - Their displacement k_j → x is the RESIDUAL after transport, plus the phrasing offset; it is not a drift sample.
  - Plan 25's velocity (p̄* − k)/(t − Λ_j) also used the wrong time base: Λ_j is the strength clock, while the key
    time is τ_j, which transport advances.
  - They are kept as the online diagnostic "transport residual": ‖x − k_j‖ / h₀ on confirmed reads, by key age
    t − τ_birth (§14.3).

### 9.4 Cost
- While births happen: ≤ 2 probe forwards per 25 steps. Depth 6, or 9 on every 4th event (6.75 passes on average),
  16 sequences.
  - At ≈ 25 ms per forward-only Core pass (review issue 16): ≈ 2 × 6.75 × 25 / 25 ≈ **13.5 ms/step**.
  - After the births: 1 forward per 100 steps ≈ 1.7 ms/step.
- The estimator and apply: ≈ 60 ops per event, on up to 360K keys (a gather of anchors + an add: 92 MB), ≈ 3 ms per
  event, ≈ 0.1 ms/step.
- **Not a dataset pointer**: the sequences come from the current batch and live ≤ 50 steps.
  **Not a loop outside the step**: one forward-only probe inside the step function, at fixed steps.

### 9.5 Directed vs diffusive, and the required R² (review issue 4)
- **The test: the cosine of successive same-period displacements.** For a position i and three equally spaced
  checkpoints θ₀, θ₁, θ₂:

      C = E_i [ ⟨Δ_i^{01}, Δ_i^{12}⟩ / (‖Δ_i^{01}‖ ‖Δ_i^{12}‖) ]

  - Model Δ_i^{e} = μ_i + ε_i^{e} (a persistent per-point velocity plus independent per-window parts). Then
    E⟨Δ^{01}, Δ^{12}⟩ = ‖μ‖², and E‖Δ‖² = ‖μ‖² + s_ε². So C ≈ ‖μ‖²/(‖μ‖² + s_ε²) =: r.
  - r = 1: directed (total displacement ∝ T). r = 0: diffusive (∝ √T).
  - The norm-ratio test of plan 25 (‖Δ_{3K}‖/‖Δ_{1K}‖ ≈ 3 vs √3) needs equal-length periods and is confounded by
    LR / phase changes; the cosine is not.
- **The required accuracy for a silent gap of G × the reference gap** (the gap at which the untransported recall
  falls to the target; G = 3.3 in plan 25).
  - Write the per-point persistent velocity as μ_i = μ̂_i + e_i, where μ̂ is the anchor field's prediction and
    ‖e‖² = (1 − R²_μ) ‖μ‖². R²_μ is the spatial R² of the persistent part.
  - Over E windows, the transport error is

        err = E·e_i  +  Σ_e (estimation noise_e)  +  Σ_e (unpredicted per-window part_e)
            ≈ E·√(1−R²_μ)‖μ‖  +  O(√E)

  - The silent drift is E‖μ‖ (directed). So the relative residual is √(1 − R²_μ): **the idiosyncratic part of a
    directed drift is itself directed and accumulates linearly; it does not average out.** That corrects plan 25's
    "R² ≈ 0.5 suffices under directed drift".
  - For the residual to be ≤ the reference gap: G·√(1 − R²_μ) ≤ 1, i.e. R²_μ ≥ 1 − 1/G². With the stricter
    interpolation target t_eff ≤ 0.45 at G = 3.3: √(1 − R²) ≤ 0.136, so **R² ≥ 0.98**.
  - Diffusive (r = 0): the silent drift is √E s_ε, the error √E · √(1 − R²_ε) s_ε. So the relative residual is
    √(1 − R²_ε), with total displacement √G × the reference. The same target gives √G · √(1 − R²) ≤ 0.45/1 at
    √G = 1.81: **R² ≥ 0.94** (plan 25's number, which is right for this case).

### 9.6 What transport can deliver at the measured R²
- Measured: R² 0.43 (per-point field, 64K pairs) / 0.31 (4K anchors) / ≈ 0.28 (1K anchors, train routing) over 1K
  steps.
- Horizon extension: the untransported recall crosses a level at drift D*. Transport replaces the drift by its
  residual √(1 − R²) D.
  - Directed: the horizon scales ×1/√(1 − R²) = ×1.20 (R² 0.31) … ×1.32 (0.43).
  - Diffusive: drift ∝ √T, so the horizon scales ×1/(1 − R²) = ×1.45 … ×1.75.
- In recall terms (salvage, 4K bank): 1K-age R@1 0.774 → 0.84-0.88; 32K: 0.568 → 0.66-0.70.
- For rare facts with ages ~Exp(1.3K), E[p_ret] rises by ≈ +0.05-0.08 (T1 computes it).
- Ships only if T1 + T0c say so (§16).

---------------------------------------------------------------------------------------------------------------------

## 10. The after_step order and the implementation without host syncs

### 10.1 Order (once per optimizer step, after opt.step and the row MemOpt; review issue 13)
1. **Bank Adagrad** on the values. This step's gradients came from reads of the current keys, so the values update
   first: compact buffer → sort + segment sum → the lazy fold f, Λ^v ← t → discounted S → clipped step → index_copy.
   t^touch ← t.
2. **Re-encode** the confirmed slots (§7.2). τ ← t, t^touch ← t.
3. **Transport** (only at a window end, §9). Keys re-encoded at step t get share 0 (τ = t = t_b), so they are not
   moved twice.
4. **Evict / merge / spawn** (§6.4-6.5). Merges first, then spawns into the free list / the weakest slots with
   t^touch < t. New keys are stamped τ = t, so they are transported pro-rata from the next window on.
5. **Re-file** every slot whose key changed in 2-4 (one batched filing call).
6. EMAs: h₀,b, v̄_b, the router statistics (device scalars).

### 10.2 No host syncs (review issue 15)
- Every budget is a host integer from the step's schedule: W, E_b, n_c, n_ε, the probe sizes; K is drawn on the host.
  All topks have static k.
- Variable-size sets are masks over static buffers:
  - the confirmed reads: topk(E_b) of a masked score (−∞ = not eligible); invalid winners go to a dummy slot N_s
    (one extra row, ignored);
  - the merges / spawns: static ≤ 160 candidates per step; a suppressed or merged candidate writes into the dummy
    slot.
- **Per-slot averaging without torch.unique** (which syncs to size its output):
  1. sort the (slot, item) pairs by slot (static length);
  2. first-occurrence flags by comparing neighbours;
  3. segment ids = cumsum of the flags;
  4. index_add_ into a static buffer of the max length;
  5. gather back.
  - About 6 ops. The same routine serves the compact value gradient (§8.3 of plan 25) and the re-encode averaging.
- The only host read per step stays the one that exists today (`mem.write_counts()` + the loss, ONE sync). The bank
  adds none.
- Logging reads device tensors only on logging steps.

---------------------------------------------------------------------------------------------------------------------

## 11. Coexistence with the row memory and ConceptHier

### 11.1 What stays
- The rows: plan 21 as run (`--mem-life 1 --mem-addr-learn 0 --mem-restart 100 --mem-coarse 64`), addressed by
  [μ_l; p̄] through ConceptHier, gradient-written, with lazy fading.
- ConceptHier (its losses, fb, inject, uncertainty features; the router uses ν and Δμ).
- M = M^row + M^bank into the same destinations (inject_{k+1}, slot 15).

### 11.2 The row-memory diet (each step knockout-checked; Stage A)

| change | evidence | saves (est.) | check |
|---|---|---|---|
| `--mem-private 0` | ko_private: late exact −0.004, loss ±0.02 | ≈ 1.3 GB (V_e, G_e, P, P_ema, P_m, P_v, G_P at E = 768, R = 512) + ≈ 0.08-0.15 s/step | late cloze ≥ 0.27 |
| `--mem-nbr 0` | an add-on (plan 21 §4); no knockout measured | ≈ 2-5% step | late cloze ≥ 0.27 |
| the per-pass row gate, c_f 0.30 + floor | plan 22 §3: memory −15…−30% at 60% open; here ≈ 40% open, readdr off | ≈ 0.15-0.3 s/step | late cloze ≥ 0.27; world_other loss ≤ control + 0.02 |
| row content-gate logit L2 (`--mem-gate-l2 1e-3`) | gate 0.999 saturated | 0 | gate mean < 0.95 at the fact-free positions |
| (optional) compact row gradient instead of the dense G | plan 13: 120-180 ms of hot-row index_add_ atomics; G = 1.07 GB | ≈ 1 GB, ≈ 0.1-0.18 s | bit-equal G sums on CPU (test) |
| coarse pool | ko_coarse −0.001; 16 MB | keep (cheap) | — |

- Readdr (`--mem-readdr-every 2`) is off under the gate (plan 22 §2.7). Fresh addressing on ≈ 40% of positions costs
  ≈ 0.4-0.8× today's reuse schedule.

### 11.3 Bit-identity
- `--bm 0` and `--mem-gate 0`: today's code, bit-identical (the new modules are built after all existing ones, with
  a forked RNG).
- `--bm 0 --mem-gate 1 --gate-target content`: plan 22 exactly.

---------------------------------------------------------------------------------------------------------------------

## 12. Self-reinforcement: measured, not forbidden

**The loops that exist:**
1. Bank read k′ < k → s_k → v_k (the key and the query of bank k) → what bank k reads.
2. Rows → thought → bank keys. Reads → the router target c → what opens → what trains → c.
3. Re-encode: the slots that are read get renewed, and the renewed slots are read more ("rich get richer").

**The metrics** (logged every eval; tensors only on logging steps):

| metric | definition | what it detects |
|---|---|---|
| read-induced view shift | on 8 probe sequences: ‖v_k(memory on) − v_k(all memory reads off)‖ / h₀,b per bank (one extra probe forward per eval) | how much the memory moves its own keys (salvage: the nomem view changes the same-pass recall by ≤ 0.01) |
| echo rate | P(the top-1 slot of read k was written by the SAME document) among confident reads | the memory reading its own recent writes (within-document recall is fine; a high echo at low benefit is a loop) |
| concentration | Gini of the reads per slot per bank; the share of reads on the top 1% of slots; and their trend | attractor slots |
| re-encode feedback | Spearman(the number of re-encodes of a slot so far, its rank-benefit b̄) | rich-get-richer without benefit |
| open-set persistence | the overlap of O^bank_k between consecutive reads (plan 22's overlap_prev) and across steps for the probe set | a gate locked open / closed |
| closed-region starvation | ḡ and c on the explored floor positions vs the top-c | the content loop of plan 22 §2.5 |
| knockout of earlier-pass reads | eval: bank reads at k′ < k off, measure bank k's recall / benefit | the dependence of later passes on earlier ones (wanted: "different routes") |

**Thresholds that trigger a review (not an automatic block)**: echo > 0.5 with the rank-benefit below the median;
top-1% share > 20%; the read-induced shift > 1 h₀.

---------------------------------------------------------------------------------------------------------------------

## 13. Training data (a v2 mix and a loader that makes time meaningful)

**1. A sequential loader** (`--data-order seq`, new; the default stays random for comparability).
- Step s reads the token range [s · 16,384, (s+1) · 16,384) of a time-ordered stream, cut into 32 windows of 512.
  The stream is shuffled at document granularity by the generator; the cut offsets are random within one window.
- Then the file position = the training step, `exposures.npz` is valid, and "steps since the last mention" is exact.
- Both the control and the memory arms use it.

**2. A rare group** (`--rare-people 2000`).
- Each person's 5 facts are mentioned k ∈ {1, 2, 4, 8} times (stratified: 500 people per k).
- The spacing between consecutive mentions is drawn per person from {10, 100, 300, 1,000, 3,000} steps (in stream
  tokens × 16,384).
- Each mention is a different document and template (training templates 0-9).
- Cloze queries use the held-out templates 10-11, as today. The query age = the final step − the step of the last
  mention, bucketed {≤ 300, 300-1K, 1-3K, > 3K}.

**3. Introduce-then-query documents.**
- Fact F appears in document A (template i); Δ ∈ {10, 100, 1,000} steps later, document B gives a different template
  j of F, with the value at the sentence end.
- B's value token is surprising for the weights (seen once) and predictable for a memory: the cleanest per-token
  knockout signal (bank on vs `--ko-bank`).

**4. Keep the existing late group** (recurring, ≈ 105-step spacing) and the existing early group. The early group is
renamed "rare-random" in the eval output. Under random windows it is the ~9.5-exposure group; under the sequential
loader it becomes a true early-then-silent group.

**5. Exposure log.** The loader writes (step, fact id) for every fact mention in each batch into `exposures_run.npz`.
It is cheap: a host-side lookup of the window's document ids from a precomputed index; no GPU, no sync. It is
**data bookkeeping for the eval only: the memory never sees it.**

**Sizes.** 2,000 rare people × 5 facts × (k̄ = 3.75 mentions) ≈ 37.5K fact sentences ≈ 0.4M tokens: negligible in
the 400M stream. The world share stays 20%.

---------------------------------------------------------------------------------------------------------------------

## 14. Metrics and evaluation

### 14.1 Primary (both decide go / no-go)
1. **Fact recall**: cloze exact / first-token accuracy / lp per token, by group:
   - late (recurring), rare-random (today's "early"), and the rare group by k and by query-age bucket;
   - × fact type: city, friend, job, pet, year, geo.
   - The baseline to beat: late 0.280 (mem_life, 7K), rare-random 0.016.
2. **Loss by target-token type** (salvage `extra_split.py` integrated into eval_world): world_answer_first,
   world_answer_rest, world_other, wiki. Plus the knockout deltas for ko_rows, ko_bank, ko_all, ko_bank_k (one bank)
   and ko_gate (forced open).
   - Targets: world_other(hybrid) ≤ world_other(control) + 0.02 (the read harm removed), and answer tokens ≤ the rows
     only.

### 14.2 Automatic eval at 25 / 50 / 75 / 100% of the run (`--auto-eval 1`)
- At each report checkpoint (`--save-report-ckpts 1`), the trainer calls eval_world in-process, between steps, after
  `free_cache()`.
- Settings: `--rank-limit 200 --questions 1 --knockouts rows,bank,all --gate-modes learned,open,off --capacity dropless`.
- It writes `world_evals/<run>_step<N>.{json,txt}`. Expected ≈ 10-15 min each, not counted in tok/s. One GPU job at
  a time is preserved (it is the same process).
- Ranking, Q:A, compare and two-hop are kept as secondary metrics.

### 14.3 Online logs (every eval)
- **Bank, per bank**: live slots by maturity; spawns / merges / confirms / evictions per step; occupancy, coverage,
  candidate recall; h₀; the median d₁, margin, A; the confirmed fraction; p_open at cue positions (eval tags).
- **Gate, per pass and store**: the open rate (train / eval), θ̂, the BCE, the calibration, the overlap with the
  previous read, the score quantiles; the open rates by token type (cue, answer, other world, wiki).
- **Transport**: windows, birth pairs; the online R² (half the window's pairs predicted from the anchors fitted on the
  other half); the per-window cosine with the previous window (the online directedness r); the residual ‖x − k‖/h₀
  of confirmed reads by key age; the slots moved.
- **Renewal factors** (§7.3), measured per bank: p_open at cues, p_win, p_ret (by age bucket, on the probe set's
  facts), p_conf.
- **The probe-set self-recall** (plan 25 §11.4), per bank, with / without transport.
- §12's self-reinforcement table.

---------------------------------------------------------------------------------------------------------------------

## §VRAM — to be filled from speed_wip/vram/vram_study.md

**PLACEHOLDER.** `speed_wip/vram/vram_study.md` does not exist yet (the VRAM profiling job is still running). The
numbers below are planning estimates and the measured totals from train.log. They must be replaced by the study's
breakdown, in particular: the activation peak per pass and micro-batch, the checkpoint recompute peaks, the
allocator fragmentation (reserved − allocated) and whether the 23.1 GB reserved at step 7,500 came from the probes /
eval or from the training step.

| item | today (mem_life run) | hybrid (estimate) |
|---|---|---|
| measured peak allocated / reserved | 20.2-20.6 / 16.2-23.1 GB | target ≤ 19 / ≤ 20 GB |
| shared rows V + G (1,048,576 × 256 fp32 each) | 1.07 + 1.07 GB | same (or V + compact G ≈ 1.1 GB with the optional compact gradient) |
| coarse pool V + G | 0.03 GB | same |
| private memory (E = 768, R = 512): V_e, G_e, P, P_ema, P_m, P_v, G_P | ≈ 1.31 GB | 0 (`--mem-private 0`) |
| row-read temporaries (gathers, the recompute) | (to be measured) | × ≈ 0.4 (≈ 40% open) |
| bank store: keys 92 MB, values 369 MB, meta 12 MB, buckets 19 MB, back-pointers 7 MB | — | ≈ 0.50 GB |
| bank compact value gradients (≈ 2,880 open × 4 winners × 256 × 4 B per in-window read × ≈ 9 reads per step) | — | ≈ 0.11 GB |
| bank lookup temps (128 candidates × 64 fp32 per open query; per read, freed) | — | ≈ 0.10 GB transient |
| transport ring + anchors (9 × 1,024 × 64 × 3 EMAs) | — | ≈ 0.03 GB |
| W_o^(b) + Adam moments (9 × 98K × 3 × 4 B) | — | ≈ 0.01 GB |
| **net vs today** | | **≈ −1.0 to −1.5 GB allocated** (−2.0 to −2.5 with the compact row gradient) |

- Reserved ≤ 20 GB is the binding limit. Static shapes (all budgets fixed per step) keep the allocator's block sizes
  stable. The study must say whether `free_cache()` after the eval / probes suffices or whether the reserved peak
  comes from the K-dependent activation sizes (K up to 16).

---------------------------------------------------------------------------------------------------------------------

## 15. Speed budget (step = 16,384 tokens: 10k tok/s ⇒ ≤ 1.64 s; 20k ⇒ ≤ 0.82 s)

**Assumptions** (all to be replaced by T2 measurements): ≈ 35 µs per launch; a forward-only Core pass ≈ 20-30 ms
(review issue 16); the no-memory base with ConceptHier ≈ 0.95 s/step (plans 13 / 23 / 25: ≈ 0.82 s without
ConceptHier + ≈ 0.16-0.19 s for it); the current memory ≈ 0.9-1.0 s (measured total 1.82-2.05 s/step at 8-9k).

| component | today | Stage A (rows diet + gate) | Stage B (+ bank) | Stage C (+ transport) |
|---|---|---|---|---|
| brain + ConceptHier + halting + deep supervision | 0.95 | 0.95 | 0.95 | 0.95 |
| rows: fine + coarse reads, fwd / recompute / bwd | ≈ 0.65 | ≈ 0.35-0.45 (40% open, readdr off) | same | same |
| private memory | ≈ 0.10-0.15 | 0 | 0 | 0 |
| neighbour writes | ≈ 0.03-0.05 | 0 | 0 | 0 |
| MemOpt (rows), probes amortised | ≈ 0.15 | ≈ 0.12 (≈ 0.02 with the compact gradient) | same | same |
| router: features, 2 topk + floor, BCE (≈ 12 ops per read) | — | ≈ 0.01 | ≈ 0.015 | ≈ 0.015 |
| bank reads: lookup 36 + read 14 + recompute 14 + bwd 18 ops per read × ≈ 5 reads × 2 micro-batches | — | — | ≈ 0.030-0.040 | same |
| bank after_step (Adagrad 22, confirm / re-encode 30, dedupe / merge / spawn 40, re-file 22, EMAs 9) | — | — | ≈ 0.005 | ≈ 0.005 |
| transport probes (births phase / later) | — | — | — | ≈ 0.0135 / 0.002 |
| **total s/step** | **≈ 1.85-2.0** | **≈ 1.43-1.55** (1.33-1.45) | **≈ 1.47-1.60** (1.37-1.50) | **≈ 1.48-1.61** (1.38-1.51) |
| **tok/s** | **8-9k (measured)** | **≈ 10.6-11.5k** (11.3-12.3k) | **≈ 10.2-11.1k** (10.9-12.0k) | **≈ 10.2-11.1k** (10.8-11.9k) |

(Brackets: with the optional compact row gradient.)

- The 10k floor is met with a margin of ≈ 2-15%; it is tight without the compact row gradient. **T2 must confirm it
  before Stage B.**
- **20k is not reachable** while the rows (≈ 0.4 s) and ConceptHier (≈ 0.17 s) both stay: 0.82 s leaves no room for
  either. The bank itself costs ≈ 3% of a 20k step.
- The levers beyond this plan: rows at 1-2 heads per level (×0.33-0.67 of the row reads), a row read only on passes
  0-2 (the gate's budget 0 elsewhere), or ConceptHier cheaper (plan 23 §7.1).

---------------------------------------------------------------------------------------------------------------------

## 16. Test plan with go / no-go gates

### 16.1 T0 — real-thought checks (CPU, ≤ 2 threads, CUDA_VISIBLE_DEVICES=-1 unless marked GPU)

| test | what | GO / NO-GO |
|---|---|---|
| **T0a cross-document same-fact retrieval** (the first gate; issue 9) | **A dedicated capture**: val world docs with every probed fact in **3 documents** (3 different training templates) + 1 held-out-template cloze prompt. 300 late + 300 rare-random people × 5 relations. Capture at ckpt 6K and 7K (GPU when free; CPU possible for ≈ 2K docs). **Train-mode routing** (random offset, noise 0.3, capacity 1.5) and eval routing; the pre-inject view per pass 1-6. Gallery = the cue positions of the 3 docs + 96K distractor positions sampled by high CE (the real write density). Report: same-fact R@1 / R@16 (doc → other doc, doc → cloze), same-relation-only hits; **pass rates of the geometry thresholds (w₁, d₂/d₁, A) for correct vs wrong top-1**; the same across the 6K → 7K gap. | **GO**: cloze-from-doc same-fact R@1 ≥ 0.5 at the best pass (same checkpoint) AND, at the calibrated thresholds, a wrong top-1 passes ≤ 5%. **Weak GO** 0.35-0.5 (Stage B only on the v2 data). **NO-GO** < 0.35: the bank keys only serve same-template reuse; stop Stage B and keep Stage A. |
| T0b benefit by token type in TRAIN mode (GPU, after the VRAM job) | ckpt 7K, 16 world + 16 wiki windows, in-window forward + backward, Probe on M^row: c by token type (answer first / rest / other world / wiki) per pass; also §5.2's CE_mem / skip test vs the true per-position knockout effect at the final output | the gate's target can see the harm iff the median c on world_other < the median c on the answer tokens by ≥ 1 MAD. Else: a train / eval mismatch, so investigate before Stage A. The CE test is enabled iff precision ≥ 0.85 at recall ≥ 0.3 |
| **T0c directed vs diffusive** | the cosine C (§9.5) per position on the same B windows at 3K / 6K / 7K (if the 3K capture of those windows exists in `realcheck/`; else at the next run's 25 / 50 / 75% checkpoints) + the norm ratio; the leave-document-out R² of the per-anchor field split into the persistent / per-window parts | reports r, R²_μ, R²_ε. Transport stage C goes ahead iff T1 (with them) shows ≥ +0.05 on the rare-fact recall |
| T0d index on real density | 48K-key bank built from the T0a capture's high-CE positions; the OPQ-allocated T = 4 index vs the plain random rotation: occupancy (max / mean, the empty fraction), coverage, candidate recall vs brute force | candidate recall ≥ 0.85 at 128 candidates AND coverage ≥ 0.95; else `--bm-brute` for all banks ≤ 48K (cost check in T2) |
| **T0e threshold calibration** (issue 18) | on T0a: the distributions of w₁, d₂/d₁, A for correct vs wrong top-1; choose (ω₁, μ_m, A_min) at precision ≥ 0.95, the dedupe γ at the same-fact pair distance q₉₀, and m_ce for the CE test | the thresholds are written into the defaults; the separation constant reported |
| T0f birth jump | a ckpt + one manual split (CPU, 8 sequences, 6 passes): ‖Δ_birth‖ / ‖Δ_25 steps‖ and its spatial R² | if the birth jumps ≤ 0.2 × a 25-step window: merge the birth pairs into the regular windows (simpler) |
| T0g pass coverage | from the run's K log: the realised f_k, P_win(k) | the capacities of §2.2 within ±10% |

### 16.2 T1 — the toy (CPU; extend `speed_wip/splat/consol_toy.py`)
**Setup.**
- Banks with the §2.2 frequencies.
- The drift calibrated to the salvage: the per-pass R@1 curves of b5 at a 1K gap; the density dependence 4K → 32K;
  r and R² from T0c.
- Facts: late (Δ 105), rare-random (Poisson, 9.5 exposures), the rare group (k × spacing).
- The phrasing jitter from T0a.
- Renewal per §7.3: an explicit p_open (the gate's rate at cues: 0.5 / 0.8 / 1.0), p_win (P_win(k) or 1), p_ret from
  the geometry, p_conf = the top-quartile rate under a c-SNR ∈ {0.3, 1, 3}.

**Policies.**
1. spawn only;
2. + dedupe / merge;
3. + re-encode (rank);
4. + re-encode (CE test, p_win = 1);
5. + transport (the shrinkage estimator; the plan-25 NW as a reference);
6. one pass-1 bank vs 9 banks.

**Thresholds (pass before Stage B / C):**

| metric | threshold |
|---|---|
| rare-random recall, policy 3, at the T0a-measured p_ret | ≥ 0.10 (6× today's 0.016) |
| rare group, k ≥ 2, spacing ≤ 300 | ≥ 0.5 |
| wrong re-filing (re-encodes toward another fact) | ≤ 1% of re-encodes at SNR 1, ≤ 3% at 0.3 |
| duplicate keys per fact after dedupe | ≤ 1.2 |
| transport gain (policy 5 − 3) on the rare-random recall | ≥ +0.05, else stage C off |
| 9 banks vs 1 bank | ≥ +0.03 recall (else `--bm-banks 1` or 6 by default; the owner decides) |

### 16.3 T2 — the GPU bench (after the VRAM job; extend `speed_wip/splat/bench_gpu.py`)
- Stage A / B / C configs at B = 16, T = 512, K ∈ {2, 6, 10, 16}: the ops (TorchDispatchMode), ms/step of each
  §15 row, the forward-only Core pass time, peak allocated / reserved.
- **GO**: Stage A ≥ 10.5k tok/s and reserved ≤ 19.5 GB at K = 10. Stage B adds ≤ 60 ms and ≤ 0.8 GB. Stage C's probe
  ≤ 15 ms/step amortised. The bank adds ≤ 450 launches per step.

### 16.4 T3 — runs (12K steps; the owner's go for each; one GPU job at a time)

| stage | arms | GO to the next stage |
|---|---|---|
| **A: rows diet + per-pass gate** (current data, random loader) | the mem_life config + `--mem-private 0 --mem-nbr 0 --mem-gate 1 --gate-target crank --mem-gate-l2 1e-3` vs the mem_life run (to 7K: res_6k_7k) | late cloze ≥ 0.27 at the matching checkpoint; world_other CE ≤ the ko_shared value + 0.01 (the harm gone); tok/s ≥ 10k; reserved ≤ 20 GB |
| **B: + the bank** (the v2 data, sequential loader) | control (no memory) · rows-only (Stage A) · hybrid | hybrid late ≥ rows-only − 0.01; rare-random and rare-group (k ≥ 2, age ≤ 1K) exact ≥ rows-only + 0.05; ko_bank costs ≥ 0.03 on the rare buckets; world_other ≤ control + 0.02; tok/s ≥ 10k; reserved ≤ 20 GB |
| **C: + transport** | hybrid vs hybrid + transport | rare buckets, age 0.5-2K: +0.03; no loss elsewhere; tok/s ≥ 10k |

- Each run uses the auto-eval at 25 / 50 / 75 / 100%. A stage that fails its GO stops there. Its knockouts are
  reported.

---------------------------------------------------------------------------------------------------------------------

## 17. Resolution of the plan-25 review (1-21) and the owner's constraints

| # | issue | resolution in 25b |
|---|---|---|
| 1 | bptt 6 cuts the early-pass reads on ≈ 47% of steps | per-pass reads and banks; P_win(k) derived (§2.2); confirmation / value grads / the router target / the write values only for in-window reads; out-of-window reads still read; the CE test (§5.2) as the forward-only route if calibrated |
| 2 | pinning a head group for the key | resolved by data: random offsets 0.795 vs pinned 0.793 across the gap; within a checkpoint o ≠ o′ 0.995-0.998. No pinning (§0.1) |
| 3 | per-step NW transport shrinks toward zero | an accumulated per-anchor history EMA with a support threshold; the window estimate shrunk toward the history (James-Stein), not zero; the jitter σ²_η vs the per-window drift s²_ε explicit (§9.3) |
| 4 | the directed-drift R² requirement is ≥ 0.98, not 0.5 | derived: the idiosyncratic persistent part accumulates linearly ⇒ R² ≥ 1 − 1/G², ≥ 0.98 for the t_eff target; diffusive ≥ 0.94; the test = the cosine of successive same-period displacements (§9.5) |
| 5 | c is ≈ zero-sum | Σc = −⟨∂L/∂W_o, W_o⟩ ≈ ω‖W_o‖² derived; centred / ranked per micro-batch for the router target and the confirmation; sign-based disconfirm and spawn-on-hurt dropped; the memory-only read-out test as the alternative (§5) |
| 6 | the h₀ feedback shrink | h₀ from ALL open reads' median d₁, floored by the key NN spacing; the contraction derived (§3.3) |
| 7 | duplicate writes | dedupe at write (d₁ < γh₀² ⇒ merge with a count-damped running mean), the within-step suppression (§6.4) |
| 8 | the renewal probability | p_ren = p_open · p_win · p_ret · p_conf; the survival model; the T0a bar derived from it; in the toy (§7.3, T1) |
| 9 | T0a needs a dedicated capture | 3 docs per fact + a cloze prompt, train-mode routing, the margin / A / w₁ pass rates for correct vs wrong, the real write density (§16.1) |
| 10 | the probe bias; the same probe both ends | both ends after opt.step (exactly M updates) with the same routine (dropless, offset 0, noise off, forked RNG); the pro-rata share by the key time stamp (§9.2-9.3) |
| 11 | recall-pair units | recall pairs dropped from the field (they are the residual after transport, with the wrong time base); kept as the residual diagnostic (§9.3) |
| 12 | a separate value fade clock | Λ^v (Adagrad touches) separate from Λ^α (confirm / merge / spawn) (§8) |
| 13 | after_step order, protect touched slots | Adagrad → re-encode → transport → merge / spawn / evict → re-file; eviction excludes t^touch = t (§10.1) |
| 14 | refit chunks | 32K-slot chunks (134 MB temp) (§2.4) |
| 15 | no host syncs | static budgets, masks + a dummy slot, sort + segment cumsum instead of unique; no new per-step sync (§10.2) |
| 16 | realistic speed | a forward-only Core pass = 20-30 ms; ConceptHier kept; the honest table: ≈ 10.2-12.3k, no 20k (§15) |
| 17 | index skew on low-rank keys; T0e on real density | OPQ-style eigenvalue allocation for the cell coordinates (ranking stays in the raw frame); brute force for small banks; T0d at the real density (§2.4) |
| 18 | calibrate the margin / w₁ thresholds | T0e: the precision ≥ 0.95 operating point on the T0a capture (§16.1) |
| 19 | transport windows straddling births | windows aligned with the neuro cadence (no birth inside); births get their own 0-update pairs (§9.2) |
| 20 | wording | plan 25 called c "the read's total first-order value". It is the first-order term only, and it contains the later passes' losses only through the gradient window (out-of-window reads have no c at all). Reworded in §5.1 / §7.1 |
| 21 | wording | "the fixed point is the benefit-weighted centroid": true only for small η and stationary readers; with η = 0.5 and drift it is a lagged EMA (§7.2) |

(The review's own texts for items 20-21 were not among the files I read. The two corrections above are the wording
problems I found in plan 25 and should be checked against the review.)

**The owner's constraints:**
- (1) the keys are views of the thought; the frame / index / anchors are fitted by statistics, never by gradient;
  w / A no-grad;
- (2) the address path is sg; the content gate's p̄ input is a downstream use (as today);
- (3) no pointers; the probes run inside the step on sequences from the current batch;
- (4) all writes happen in training steps;
- (5) §15 / §VRAM;
- (6) the per-pass gate, two stores, per-bank weights;
- (7) the per-pass banks, the pre-inject view;
- (8) §12.

---------------------------------------------------------------------------------------------------------------------

## 18. Flags, defaults, implementation size

**Stage A** (rows diet + gate):

    --mem-life 1 --mem-addr-learn 0 --mem-restart 100 --mem-coarse 64 --mem-read0 0 --mem-private 0 --mem-nbr 0
    --mem-gate 1 --mem-gate-stores rows --gate-target crank --mem-gate-c 1.0,0.30 --mem-gate-sched 0.2,0.5
    --mem-gate-explore 0.10 --mem-gate-perbank 1 --mem-gate-l2 1e-3 --auto-eval 1 --save-report-ckpts 1

**Stage B** (+ the bank):

    --bm 1 --bm-banks 9 --bm-cap 48K,48K,48K,48K,48K,40K,32K,24K,16K --bm-dv 256 --bm-view pre15 --bm-fit-step 300
    --bm-write-start 300 --bm-write 16 --bm-confirm rank --bm-conf <T0e> --bm-qc 0.25 --bm-dedupe 0.25
    --bm-eta 0.5 --bm-fade 1e-3,1e-4 --bm-wo perbank --bm-index opq --bm-brute 16384
    --mem-gate-stores rows,bank --mem-gate-c-bank 1.0,0.25 --data-order seq (the v2 mix)

**Stage C**: `--bm-transport 1 --bm-tr-anchors 1024 --bm-tr-seqs 8 --bm-tr-depth 6,9 --bm-tr-rho 0.9 --bm-tr-cap 0.5`.

**Ablations**: `--bm-banks 1|6`, `--bm-wo shared`, `--bm-confirm ce|skip`, `--bm-write-nograd 1`, `--gate-target content`
(plan 22), `--gate-adapt 1`, `--ko-bank`, `--ko-bank-pass k`.

**Implementation size (estimate)**

| part | lines |
|---|---|
| `bank_memory.py` (store, banks, frames, OPQ index + maintenance, lookup, read / Probe, compact Adagrad, dedupe / merge / spawn / evict, re-encode, clocks, transport estimator + apply, state / resume, stats) | ≈ 1,050 |
| MemGate extension (two stores, per-bank weights, the crank target, the second θ̂, the logit L2, the adaptive budget) | ≈ 150 |
| `loop_brain.py` wiring (the pre-inject view, the bank read in `_hier_step`, the inject hook for the write gradient, the probe forward helper, `ko_bank`) | ≈ 220 |
| `train_lm.py` (flags, after_step order, the transport events aligned with neuro, param groups, auto-eval, logging, resume) | ≈ 230 |
| `eval_world.py` (the token-type split, the rare buckets by k / age, the new knockouts, the open rates by token type) | ≈ 160 |
| `generate_world.py` v2 (the rare group, introduce-then-query, the stream order) + the sequential loader + the exposure log | ≈ 260 |
| tests (`test_bank_memory.py`: flags-off identity; lookup = brute force; index invariants; dedupe / merge; lazy clocks = eager; compact Adagrad = dense; Probe = finite difference; the crank target; capacity exactness; no host sync; the transport (zero / constant field, pro-rata shares, the birth pairs); resume) | ≈ 650 |
| realcheck T0a-T0g scripts + the toy T1 + the bench T2 | ≈ 450 |
| **total** | **≈ 3,150** (Stage A alone ≈ 600) |

---------------------------------------------------------------------------------------------------------------------

## 19. Risks and open questions
1. **Cross-template retrieval (T0a).**
   - The rows turned out template-keyed. If the pre-inject view at the cue is dominated by the template too, a bank
     key from a training document is not found by the held-out-template cloze prompt.
   - Then the bank serves only same-template reuse (still useful for introduce-then-query with the same template, but
     not for the cloze metric). This is the single largest risk; T0a decides before any bank code.
2. **The harm may be a train / eval mismatch (T0b).**
   - If the training-mode c on the non-answer world tokens is ≥ 0 while eval shows +0.13 nats, a gate trained on c
     will not close them.
   - Then: compare the train vs eval routing / capacity / K on the knockout effect, and consider training the router
     on the eval-mode benefit of a small no-grad subsample (one extra forward of 1-2 sequences per 25 steps with the
     memory off: a counterfactual, like the transport probe).
3. **The c SNR.** Ranks need the per-position c to carry signal beyond noise. The salvage's answer-vs-other contrast
   says the mean differs. The per-position SNR is unknown (T0b gives it; T1 sweeps it).
4. **Capacity vs density.** Smaller banks recall better (4K 0.77 vs 32K 0.57). The 48K banks are sized for survival,
   not recall. T0d / T1 may push toward fewer writes and smaller banks (or a stricter surprise threshold).
5. **The 10k floor is tight** (§15). If T2 shows Stage A below 10.5k: the compact row gradient, rows at 2 heads, or
   row reads only on passes 0-2.
6. **20k is out of reach** with the rows + ConceptHier. That is an owner decision, not something this plan can fix.
7. **Value decodability** (only for `--bm-confirm ce`): M is not trained to be read by the LM head. An optional tiny
   decode loss on M (into the memory parameters only, never the thought) would make it so; it is not proposed for v1.
8. **The birth jumps** may be large. Then the birth pairs matter more than the windows, and the bank keys older than a
   birth event may be best re-validated by the next exposure rather than transported (T0f).
9. **The sequential loader changes the training distribution** (correlated adjacent windows within a step). The
   control must use it too. The current-data Stage A stays on the random loader for comparability with the mem_life
   run.
10. **Open**:
    - 9 vs 6 vs 1 banks (T1 / Stage B ablation);
    - a shared vs per-bank W_o;
    - whether the out-of-window writes (`--bm-write-nograd`) help the deep banks;
    - whether `--gate-adapt` finds budgets far from 0.25-0.30;
    - whether the late facts should also have bank keys (they are written only when σ is high, i.e. before the rows /
      weights know them).
