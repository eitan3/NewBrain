# 25b. Per-pass memory: the shared row memory kept and gated per pass, plus a per-pass thought-keyed bank store (PLAN ONLY) — 2026-10-24, revision 2

Revises plan 25 (`25_recall_reencode_memory_plan.md`, kept unchanged). Revision 1 of this plan is kept unchanged as
`25b_per_pass_memory_plan_rev1.md`. Builds on plans 21 (life cycle, lazy fading), 22 + `22_gate_code.diff` (the
MemGate router, already in `memory.py` / `loop_brain.py`), 23 (store the write-time key, read the nearest one), 24 (key
renewal is first-order), the real-data results `C:\Eitan\Python\NewBrain\speed_wip\splat\realcheck\salvage\results.md`
(run `world_mem_life_12k`, checkpoints 6K → 7K; note: under `NewBrain\speed_wip`, NOT under `fruit-fly-sim\speed_wip`)
and the VRAM study `fruit-fly-sim\speed_wip\vram\vram_study.md`.

Nothing was trained, no code was edited, the GPU was not used. Only this file (and the rev1 copy) was written.

---------------------------------------------------------------------------------------------------------------------

## Changelog: revision 1 → revision 2 (from the second review and the coordinator's owner-decision defaults)

Items marked **[pending Eitan]** still need the owner's sign-off (§OD); only one remains (option A vs B). **Owner answers received during this revision**
(his words: "1. per pass from the beginning 2. The brain 3. Yes, leave the thoughts alone. The thought can change in
the next pass (T+1)"), applied throughout:
- **(a) per-pass memory from the start**: no memory shared across passes, already in Stage A. The memory form
  [pending Eitan; the coordinator asked him directly]: **default = option B, per-pass row tables + per-pass banks**:
  the same ≈ 1M fine row slots split over the 9 pass tables (same VRAM and speed), the banks added in Stage B; the
  rows remain the baseline the banks must beat, and are dropped later only if banks alone beat them (Stage C).
  Alternative **option A, bank-only** (no row table). Private per-expert memory off. Coarse level per pass too (§11.4).
- **(b) the brain decides the read rate**: learned gate, **no imposed budget schedule, no fixed-budget fallback** by
  default (the fixed budget is a flag only). Two safety pieces: a hard speed cap (≤ 50% of positions per pass) and 5%
  uniform exploration.
- **(c) confirmed**: no gate loss pushes the thoughts (sg on every gate's p̄ input, rows' content gate included); the
  memory output flows into pass T+1 as normal; the pass-T address is taken from pass T's thought before pass T's own
  read is injected (owner: "question 3: yeah keep that (what you recommended)"). **Decided**: the main LM loss flows
  back through the read into the thought via BOTH the value read and the MATCH weights (which memories matched: w, A,
  the rows' soft weights) (§3.5).
- **Decided (coordinator)**: the two safety pieces of the learned gate, the hard 50% speed cap and the 5% uniform
  exploration. **The only item still pending Eitan is option A vs B** (default B).
- **Stages restructured**: Stage A = per-pass rows + the brain-decided gate; Stage B = + per-pass banks; Stage C = a
  comparison, banks alone (option A) vs rows (§16.4).

The table below records the review-driven changes; where an owner answer superseded a coordinator default, the row
says so.

| # | review point | change in rev 2 | where |
|---|---|---|---|
| 1 | the router's benefit target c = −⟨∂L/∂u, u⟩ exists only after backward, but rev1 put the router loss in the forward | the router is trained by a **post-backward analytic update** (a linear router has the closed-form gradient: Huber ψ(s − z)φᵀ in the default learned-budget mode, (σ(s) − y)φᵀ under the rank-BCE flag); φ / s / the open indices of the in-window reads are kept until after backward (≈ 55 MB per micro-batch); the Probe writes c into a preallocated slot (idempotent), so the checkpoint recompute cannot double it; a test checks firing count = 1 and the analytic gradient = autograd | §4.3, §4.6, §16.5 |
| 2 | tok/s cannot be a memory-stage GO criterion (the baseline itself is 8.4k; no memory saving buys speed) | tok/s removed from the memory GO criteria; speed is a **separate track** (fusion / `torch.compile`, host-sync removal; study running → `fruit-fly-sim\speed_wip\speed_study.md`, not yet written). Stage A / B are judged on **Δtok/s vs the same-harness baseline ≥ −3%**. §VRAM filled from the study (training peak 20.1 GB allocated / ≈ 21 reserved at 16×2; the 23.1 GB was `evaluate()` on a training-sized cache). New run defaults: `empty_cache()` before every `evaluate()` and loop report; `set_per_process_memory_fraction(≈0.78)` ≈ 20 GB guard; micro-batch 8 × accum 4 (13.2 GB, −4.5%) **conditional on the same-maths audit** (per-forward side effects run 4× instead of 2× per step). The compact row-gradient claim is removed (54% of rows are touched per step: dense is right). §10.2's "one host sync" corrected: the baseline has 48 syncs (≈ 0.2 s) | §0.4, §VRAM, §10.2, §11.2, §15, §16.3-16.4 |
| 3 | owner constraints were overstated as met | stated honestly: the shared rows are read / written by ALL passes (not per pass); bank 8 is shared by every k ≥ 8; in rev1 the read RATE was imposed by a budget schedule. **Owner answers**: (a) per-pass memory from the start (default option B: per-pass rows + per-pass banks; option A bank-only [pending Eitan]); (b) a **learned budget** (read where the predicted gain beats one read cost c_read in nats; no schedule; the fixed budget only as a flag), with a hard 50% cap and 5% exploration; (c) sg on p̄ wherever a gate reads it (confirmed). Table / bank 8 is still shared by k ≥ 8 (5% long-thinking steps; `--bm-banks 15` gives every k its own, §2.2) | §4, §11.4, §17.2, §OD |
| 4 | p_write unknown; the write top-k by raw CE is likely dominated by Wikipedia / generally hard tokens | new **T0h**: the fraction of rare-fact cue positions that land in the write top-k on training windows; candidate criteria using ONLY the brain's own signals (entropy-normalised surprise, CE improvement across passes); **no dataset labels or source tags** in any fix | §6.2, §16.1 |
| 5 | Stage B / A GO criteria | Stage B GO only on the v2 rare group (k × spacing × age buckets) and introduce-then-query; Stage A late cloze at 7K (≥ 0.27 vs 0.280) AND at the end (≥ control final 0.260 + 0.02). The early group's exposure count at 7K is ≈ 5.5, not ≈ 9.5 (that is at 12.2K); eval_world's `--exposure-at` buckets mislabel the early group | §0.2, §16.4 |
| 6 | `--mem-gate-l2` | dropped (the gate mean stays a diagnostic). "Stuck at 0.999" is a symptom, not the cause: the content gate is trained by an Adam-type optimiser, which is invariant to the gradient scale, so σ′ ≈ 0.001 does not stop it learning; it stays open because the loss prefers open on average. The router does the closing. The bank's content-gate logit L2 is dropped for the same reason | §0.3, §3.4, §11.2 |
| 7 | strength decay, the §3.3 example, h₀, wrong merges | λ_α = 2.5e-4 for immature slots (rev1's 1e-3 killed a fresh slot's read weight before the next rare exposure) and, by default, **strength used only for eviction** (A has no σ(ã) factor); the §3.3 example corrected (rev1 forgot σ(ã) and the ½ floor: a median-drifted owner read was at A ≈ 0.09-0.21, below A_min); h₀ calibrated from **same-fact** distances (T0e) and then tracked; T0e bounds the wrong-merge rate | §3.2-3.3, §6.4, §8, §16.1 |
| 8 | transport needs probe forwards outside the training computation | the transport stage removed from this plan (the new "Stage C" is the banks-alone comparison, unrelated); the transport design moved to **Appendix A (future; needs an owner check against the no-recompute constraint)** | App. A |
| 9 | the survival model | corrected to lost-gap events at rate λ e^{−λH}, λ = p_ren/Δ: S(T) = exp(−λT e^{−λH}); late facts need p_ren ≥ ≈ 0.43 (rev1: 0.31). Also: under the gate the rows get gradient only at OPEN cue positions, so "the rows rewrite at every exposure" needs p_open^row at cues | §7.3 |
| 10 | the target's centre | centred on the **uniform exploration sample** (5%, uniform over ALL positions), not on the selected open set | §4.3 |
| 11 | sizes and units | bank 8 ≈ 32K (it takes the writes of every k ≥ 8: Σ P_win = 0.33); all budgets per OPTIMIZER step (independent of 16×2 vs 8×4); value init with an ε guard / ξ-only fallback while W_o ≈ 0 | §2.2, §6.3 |
| 12 | Stage A confounds the diet and the gate | superseded by owner answer (a): Stage A = per-pass rows + the learned gate, judged against the control and the old shared-row run at the same steps. `--mem-nbr` is kept on (fewer confounds; it is a write-side add-on); private memory off (ko_private ≈ 0) | §16.4 |
| 13 | the sequential loader | 32 independent cursors (stream i starts at i·N/32), one window each per step | §13 |
| 15 | expectations | T0a is likely to FAIL (the rows are template-keyed; the views may be too); T1 waits on T0h (p_write) | §16.1-16.2 |
| — | simpler alternatives adopted | Stage B v1 = **spawn + dedupe only** (confirmation / re-encode = Stage B2, only if T1 shows a gain); the index = **brute-force bf16 scoring + fp32 re-rank of the top-32** for banks ≤ 48K keys (OPQ moved to Appendix B) | §2.4, §7, App. B |
| — | order of work | fix plan → owner sign-off (§OD: option A vs B) → T0b on GPU → implement Stage A; T0a + T0h gate Stage B (and C) | TL;DR 10, §16.6 |

(Review point 14 had no change request for this file.)

---------------------------------------------------------------------------------------------------------------------

**The owner's hard constraints** (each is checked in §17, honestly):
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
- §0 what the evidence implies; §1 the architecture;
- §2 the bank store; §3 the view and the read;
- §4 the per-pass read gate (learned budget); §5 the benefit signals;
- §6 writes; §7 confirmation / re-encode (Stage B2, later) and the survival model;
- §8 fading and clocks; §10 the after_step order and host syncs (§9 left empty: transport is Appendix A);
- §11 coexistence with the rows and ConceptHier; §12 self-reinforcement;
- §13 training data; §14 metrics; §VRAM; §15 cost vs the same-harness baseline;
- §16 tests and go / no-go; §17 the review and the constraints; §OD owner decisions;
- §18 flags and size; §19 risks; Appendix A transport; Appendix B the OPQ index.

---------------------------------------------------------------------------------------------------------------------

## TL;DR

1. **Per-pass memory from the start (owner).** No memory is shared across passes, already in Stage A (§11.4):
   - **Default, option B [pending Eitan]: per-pass row tables + per-pass banks.** The fixed-random-address row memory
     (plan 21: 4 levels × S² product-key rows, 3 heads per level, + a coarse pool) is split into 9 pass tables with
     the same ≈ 1M fine slots in total (S = 512 → S = 170 per pass: 9 × 4 × 170² = 1,040,400 rows): same VRAM, same
     speed. Each pass also gets its own coarse pool (64², +0.28 GB). The rows stay the baseline the banks must beat.
     Risk: ≈ 9× less row capacity per pass, so the late cloze (0.280 shared at 7K; 0.182 knocked out; control 0.260
     at the end) may drop; deep passes read rarely, so their tables fill slowly.
   - **Option A: bank-only** (no row table): the per-pass banks are the memory. Risk: it gives up the only memory that
     has worked on real data until the banks prove they carry recurring facts (the survival model, §7.3, says a bank
     needs p_ren ≥ 0.43 for them), and it depends on T0a / T0h, which may fail. Tested as Stage C, not as the start.
   - The banks are for the facts the rows do not keep: rare facts (rows 0.016 vs control 0.013).
2. **Per-pass banks** (Stage B).
   - One slot store, 9 banks b = 0..8. Read k (after core pass k, k = 0..K−2) uses bank b(k) = min(k, 8): **bank 8 is
     shared by all k ≥ 8** (5% long-thinking steps).
   - Key and query = the pass-k **pre-inject view**: the mean of the slot-normalised workspace slots 0-14 after pass
     k, in a frame fixed per bank (slot 15 carries the memory's own output).
   - Capacities from the per-optimizer-step write rate (§2.2): 48K (banks 0-4), 40K, 32K, 24K, **32K (bank 8)**,
     ≈ 368K slots, ≈ 0.5 GB.
   - Salvage b5: same-pass recall across the 1K gap 0.77 (pass 1) → 0.71 (pass 6); a pass-1 bank queried from pass 6
     only 0.40.
   - Lookup: brute-force bf16 scores + fp32 re-rank of the top-32 (banks ≤ 48K keys; no OPQ index in v1).
3. **A per-pass read gate with a LEARNED budget: the brain decides (owner).**
   - Per pass and store, a linear router on plan 22's 93 detached features predicts the **centred gain** ĝ (nats) of
     reading at a position. A position reads iff ĝ > c_read (one constant scalar read cost in nats; default 0, no
     schedule). Two safety pieces (decided): a hard speed cap of 50% of positions per pass, and 5% uniform
     exploration.
   - The target: z = N_tok · (c − median of c over the exploration sample), with c = −⟨∂L/∂u, u⟩ from a Probe
     (only in-window reads have it). The router is updated **after backward**, analytically.
   - No imposed budget schedule and no automatic fallback. The fixed top-c budget of rev1 is a debug flag only; the
     router's AUC is logged and alerts when < 0.6.
   - This answers "the reads help the answer tokens (−0.29 nats) but hurt the other 83% of world text (+0.13 nats)".
4. **Bank writes (Stage B; bank v1 = spawn + dedupe only).**
   - Each in-gradient-window read k writes into bank b(k): W = 32 positions per OPTIMIZER step per bank, chosen by a
     surprise criterion that uses only the brain's own signals (T0h decides between raw final CE, entropy-normalised
     surprise and the CE improvement across passes).
   - Key = the pre-inject view. Value = the normalised gradient write (ε-guarded; ξ-only while W_o ≈ 0), then
     row-Adagrad.
   - **Dedupe at write**: d₁ < γ h₀² merges into the nearest slot (γ from T0e with a wrong-merge bound); within-step
     duplicates suppressed; eviction never takes a slot touched this step.
5. **Confirmation / re-encode / renewal = Stage B2**, only if T1 shows a gain (§7). The corrected survival model
   (§7.3) says late facts need p_ren ≥ ≈ 0.43, far above a plausible ≈ 0.13: the rows must carry the recurring facts.
6. **h₀** is calibrated from same-fact distances (T0e), then tracked by the all-reads median at the calibrated ratio,
   floored by the key spacing. Strength is used for eviction only (default), with λ_α = 2.5e-4 for immature slots.
7. **Transport is out of this plan** (Appendix A: future, needs the owner's check against "no recompute loop outside the
   training step").
8. **Data**: the trainer samples uniform random windows from the whole 400M-token file, so the "early" people are a
   RARE group (≈ 5.5 exposures by 7K, ≈ 9.5 by 12.2K, spread over the run). The v2 data (§13) adds a 32-cursor
   sequential loader, a scheduled rare group and introduce-then-query documents; Stage B is judged ONLY on those.
9. **Speed / VRAM.**
   - Baseline (same harness): 8.4k tok/s; training peak 20.1 GB allocated, ≈ 20.5-21.1 GB reserved at 16×2; the
     23.1 GB reserved was `evaluate()` on the training cache.
   - Memory stages are judged on Δtok/s ≥ −3% vs the same-harness baseline, and reserved ≤ 20 GB. The 10k / 20k
     targets are the **speed track** (fusion / compile / host syncs), not this plan.
   - Run defaults: `empty_cache()` before `evaluate()` and the loop report; a 0.78 memory-fraction guard; micro-batch
     8 × accum 4 if the same-maths audit passes.
10. **Order of work** (§16.6):
    1. this plan fixed (done here);
    2. owner sign-off on §OD (only option A vs B remains);
    3. T0b on the GPU (when free): c by token type in train mode, router AUC;
    4. implement Stage A = per-pass rows + the brain-decided gate (+ the VRAM defaults) and run it;
    5. T0a + T0h (p_write) gate Stage B (+ per-pass banks) and Stage C (banks alone vs rows); T0e / T1 follow T0h.
    - About 2,830 lines in all (Stage A ≈ 850).

---------------------------------------------------------------------------------------------------------------------

## 0. What the evidence implies

### 0.1 The salvage results (6K → 7K), read as design constraints

| finding | number | design consequence |
|---|---|---|
| rows work for recurring facts | late exact 0.280; ko_shared 0.182; control final 0.260 | keep the rows (the baseline to beat); every change is knockout-checked against 0.280 at 7K AND the control's final 0.260 |
| rare facts | early exact 0.016 vs control 0.013 (≈ 5.5 exposures by 7K) | the bank store's job, judged on the v2 rare group |
| reads hurt non-answer tokens | answer −0.29 / −0.20 nats; other world +0.13 on 83% of world tokens; wiki ±0.01 | a per-position, per-pass read gate (§4) |
| the content gate mean is 0.999 | probe "gate 0.999 gate_c 0.908" (train.log, step 7000) | a SYMPTOM: the loss prefers the gate open on average (Adam is scale-invariant, so σ′ is not what holds it). Plan 22's target ḡ is ≈ constant, so it is no ranking target (§4.3) |
| rows are template-keyed | same template other person J 0.24; same fact other template 0.28; same person other relation 0.045 ≈ baseline | the bank keys may be template-dominated too: T0a is the first gate and is expected to be hard |
| self-recall 1K gap, pass 1 | R@1 0.774 / R@16 0.983 (4K bank); 0.568 / 0.882 (32K) | recall falls with bank density, so the banks are kept small (§2.2) |
| world text recalls worse | world 0.525 vs wiki 0.902 (4K) | exactly the text the memory is for |
| rotation / noise | o ≠ o′ 0.995-0.998 within a checkpoint; across the gap 0.795 vs pinned 0.793 | no pinning of a head group needed |
| transport | per-point field 0.774 → 0.884 (R² 0.43); 4K anchors 0.854; 1K anchors 0.839 | Appendix A (not in this plan) |
| per-pass banks | same-pass 0.774 / 0.753 / 0.747 / 0.734 / 0.723 / 0.707 (passes 1-6); pass-1 bank from pass 6: 0.401 | per-pass banks (§2) |
| pre-inject views | slot 15 excluded / memory off: +0.00…+0.01 same-pass | use slots 0-14 |
| drift scale | median drift / 1-NN gap 1.31 (pass 1) … 1.27 (pass 6) | h₀ must be of the order of the drift; calibrated on same-fact pairs (§3.3) |

### 0.2 The training sampler makes "early" facts rare, not silent
- `train_lm.py: Data.batch` draws `gen.randint(0, len(d) − seq − 1, B)`: uniform random windows over the whole
  400,000,000-token file. The run's config has `"curriculum": ""`.
- A 12.2K-step run reads 200M tokens, i.e. every file position is covered **0.5 times in expectation**.
- The early people's 19 mentions per fact sit at file positions < 40M. So each early fact is seen ≈ 19 × 0.5 =
  **9.5 times by the end (12.2K steps)**, but only **≈ 19 × (7,000 × 16,384 / 400M) ≈ 5.5 times by step 7K**, spread
  uniformly over the steps, mean spacing ≈ 1.3K steps.
  - So the 7K salvage number (0.016) and any end-of-run number are at different exposure counts. Comparisons of the
    early group across checkpoints must be made at equal steps.
- Late facts: ≈ 116 exposures by the end, spacing ≈ 105 steps.
- Consequences:
  1. "Early retention" measures **recall of a rare fact** (≈ 5.5-9.5 exposures, ≈ 1.3K spacing).
  2. eval_world's exposure buckets (`--exposure-at 200000000` reads `exposures.npz` keyed by FILE position) assume a
     sequential read; under random sampling they **mislabel** the early group (its "steps since the last mention" is
     not what the bucket says). §13 fixes it (sequential loader + an exposure log by step).
  3. Stage B is therefore judged on the v2 rare group and on introduce-then-query, whose schedule is known (§16.4);
     the rare-random group is reported only.

### 0.3 The row content gates' mean is 0.999: a symptom, not a cause
- `mem_gate_mean` = 0.999 (fine) / 0.908 (coarse) at step 7,000; the gate is σ(a), b initialised at −2.
- Rev1 said σ′ ≈ 0.001 makes the gate unable to close. That is wrong for this optimiser: the content gates are
  trained by an Adam-type update (BrainOpt's Adam group), whose step size is invariant to the gradient's scale. A
  consistent "close" gradient, however small, would move a. The gate stays open because the loss gradient, summed
  over all positions, says open (the reads help the answer tokens a lot; the harm on other tokens is spread thin).
- Consequences:
  - (i) plan 22's router target ḡ is ≈ constant: no ranking signal. The router needs the benefit target (§4.3);
  - (ii) no logit L2 on the content gates (rev1's `--mem-gate-l2` dropped; the bank's L_gl dropped). The gate mean
    stays a logged diagnostic;
  - (iii) the closing is done by the router, which sees per-position benefit, not by the content gate.

### 0.4 Speed and VRAM (from `fruit-fly-sim\speed_wip\vram\vram_study.md`)
- Same harness: 8.4k tok/s (the run logged 8-9k). The step is GPU-bound on elementwise kernels (> 50%), GEMMs ≈ 20%,
  **48 host syncs ≈ 0.2 s**.
- Training peak **20.1 GB allocated** (≈ 2.0 GB per backpropagated pass × bptt 6) at any K ≥ 6; reserved 20.5-21.1 GB.
  The **23.1 GB reserved came from `evaluate()`** (batch-32 logits on a cache sized for training), not from the step.
- No memory saving buys speed (micro-batch 8×4 vs 16×2 vs 32×1 within 10%). So **speed is a separate track** (§15),
  and this plan's stages are judged on not slowing the step (Δ ≥ −3%) and on the 20 GB reserved cap.

---------------------------------------------------------------------------------------------------------------------

## 1. Architecture in one picture (read k = after core pass k, feeding pass k+1; k = 0..K−2)

```
 core pass k ──► x_k, s_k
   │
   ├─ (training) read-out ce_k, lp_k          (already computed: deep supervision; before the read as in plan 22)
   ├─ ConceptHier(n_s(s_k)) → μ_l, ν_l        (unchanged; the rows' addresses)
   ├─ view  v_k = mean_{i<15} n_s(s_k)_i      (64-d, pre-inject) ── sg ──────────────────────────────┐
   ├─ router φ_k (93-d, detached) → ĝ^row_k, ĝ^bank_k  (per pass, per store)                         │
   │      open O^m_k = {ĝ > c_read, ≤ cap} ∪ E_k (5% uniform exploration)                             │
   ├─ ROWS table b(k) (per pass) on O^row_k : plan-21 read ─► u^row ─Probe─► M^row_k                   │
   ├─ BANK b = min(k, 8) on O^bank_k (Stage B): x̂ = (v_k − μ_b)/σ_b ─► bf16 scores, top-32 ─► fp32  ◄─┘
   │      re-rank ─► top-4 (J, d) ─► w, A ─► r = A Σ w_j f_j v_j ─► content gate g (p̄ input sg) ─► u ─Probe─►
   │      M^bank_k = W_o^(b) u
   └─ M_k = M^row_k + M^bank_k ─► inject_{k+1} += M_k ; slot 15 += W_ms M_k   (today's destinations)

 after backward (each micro-batch): Probe c → targets z → analytic router gradient (accumulated)
 after the step (after opt.step): router AdamW ; row MemOpt (as today) ; bank: Adagrad → evict / merge / spawn
```

- Gradient reaches: the bank values, W_o^(b), the content-gate weights, W_ms, and (by the post-backward analytic
  update) the router.
- No gradient reaches: the keys, the frame, h₀, strengths and clocks, the slot / row SELECTION (indices), and **the
  thought through any gate's p̄ input** (sg, owner-confirmed; the rows' content gate included). The read VALUE flows
  into pass k+1 as normal.
- Decided (§3.5): the main LM loss also flows through the MATCH weights (bank w, A; the rows' soft weights) into the
  query, i.e. into the thought.

---------------------------------------------------------------------------------------------------------------------

## 2. The bank store (Stage B)

### 2.1 Layout (one store, banks as contiguous slot ranges)
- Slots i = 0..N_s−1; bank b owns [o_b, o_b + N_b), plus one dummy slot N_s (ignored writes). Per slot:

| field | dtype | set by | meaning |
|---|---|---|---|
| k_i (64) | fp32 (+ a bf16 copy for scoring) | rules (§6) | key in the bank's fixed frame |
| ‖k_i‖² | fp32 | with k_i | distance |
| v_i (256) | fp32 | write init + row-Adagrad | value |
| S_i | fp32 | Adagrad | accumulator |
| log α_i | fp32 | rules | strength (eviction order; in A only under `--bm-strength-read 1`) |
| Λ^α_i | fp32 | rules | strength clock (spawn / merge; + confirmation in B2) |
| Λ^v_i, Ψ_i | fp32 | Adagrad touch | value fade clock and accumulator discount clock (separate from Λ^α) |
| τ_i | int32 | rules | key time stamp |
| m_i, n_i | int8 | rules | maturity 0..3 (B2) / merge count ≤ 8 |
| t^touch_i | int32 | rules | last step that touched the slot (eviction protection) |
| u_i, b̄_i | int32, fp32 | rules | use count; running benefit rank (diagnostic) |

### 2.2 Bank count and capacities (per optimizer step)
- K ~ U{2..10} w.p. 0.95, U{12..16} w.p. 0.05. Read k exists iff K ≥ k+2; its gradient window (bptt 6) is
  K−6 ≤ k ≤ K−2.

      f_k   = P(K ≥ k+2)
      P_win(k) = P(k+2 ≤ K ≤ k+6) = 0.95·|[k+2, k+6] ∩ [2,10]|/9 + 0.01·|[k+2, k+6] ∩ [12,16]|

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f_k | 1.00 | 0.89 | 0.79 | 0.68 | 0.58 | 0.47 | 0.37 | 0.26 | 0.16 | 0.05 | 0.05 | 0.04 | 0.03 | 0.02 | 0.01 |
| P_win(k) | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 | 0.42 | 0.33 | 0.23 | 0.14 | 0.04 | 0.05 | 0.04 | 0.03 | 0.02 | 0.01 |

- Read 0 is outside the window iff K ≥ 7, P = 0.47. Every mechanism that needs a gradient (value training, the
  router's target, a write's value init; in B2 the rank confirmation) runs only on in-window reads.
- **Bank 8 takes every k ≥ 8**: its in-window write rate is Σ_{k≥8} P_win(k) = 0.14 + 0.04 + 0.05 + 0.04 + 0.03 +
  0.02 + 0.01 = **0.33** per step (rev1 used 0.14 and sized it 16K).
- **All budgets are per optimizer step** (32 windows of 512 tokens), so they do not change between 16×2 and 8×4:
  W = 32 writes per step per in-window read, split evenly over the micro-batches (16 / micro-batch at 16×2, 8 at 8×4).
- Capacities: an unconfirmed slot must survive ≥ 3,000 steps (the largest rare-fact spacing, §13):

      N_b ≥ 3,000 · W · Σ_{k: b(k)=b} P_win(k)  →  51K → 48K (b 0-4, ≈ 2.8K-step life), 40K (5), 32K (6), 24K (7), 32K (8);  N_s ≈ 368K

- Density matters (salvage 4K 0.774 vs 32K 0.568). The store's keys are sparse surprise positions, so the effective
  competition may be smaller; T0d measures recall at the real density.
- Flags: `--bm-banks 9` (default), `--bm-banks 1` (a single pass-1 bank, ablation), `--bm-banks 6` (k ≥ 5 shared).

### 2.3 The frame (fixed per bank, no gradient)
- x̂ = (v − μ_b)/σ_b, μ_b ∈ R^64, σ_b a scalar RMS, fitted once on 64K positions of pass b at `--bm-fit-step 300`.
  Never refitted.

### 2.4 The lookup: brute-force bf16 scoring + fp32 re-rank (no index in v1)
- For a bank of N_b ≤ 48K keys and the open queries Q of read k:
  1. scores s = 2 x̂·k_i − ‖k_i‖² in **bf16** (a GEMM on the bf16 key copy), in query chunks of 1,024 (a 1,024 × 48K
     bf16 score block = 98 MB, freed per chunk);
  2. topk 32 per query on the bf16 scores;
  3. exact fp32 distances on the 32 candidates (gather 32 × 64 fp32 keys), top-4 winners (j₁..j₄, d₁ ≤ … ≤ d₄).
- no_grad; the winners are cached for the checkpoint recompute.
- Cost per step: ≈ 5 reads × ≤ 8,192 open queries × 48K × 128 flops ≈ 0.25 TFLOP of bf16 GEMM plus the topk over 48K
  columns (the topk dominates); estimate 15-40 ms per step (T2 measures).
- Why not the OPQ index of rev1: at ≤ 48K keys the brute force is exact up to the bf16 re-rank margin, has no
  maintenance (no refit, no re-filing, no back-pointers, no bucket overflow on low-rank keys), and is cheap enough. The
  OPQ design is kept in Appendix B for banks ≫ 48K.
- Monitored: the fraction of queries whose fp32 top-4 contains a key outside the bf16 top-32 (estimated on 1K sampled
  queries every 250 steps against a full fp32 scan): target ≤ 0.5%.

---------------------------------------------------------------------------------------------------------------------

## 3. The view and the read

### 3.1 The view (pre-inject)
- v_{k,t} = (1/15) Σ_{i=0}^{14} n_s(s_k)_{t,i}, computed in `_hier_step.block()` next to today's `pbar`. The rows keep
  their 16-slot p̄.
- s_k is the workspace after core pass k, before read k's slot write and inject. Earlier reads (k′ < k) still
  influence s_k: the feedback measured in §12.

### 3.2 The read kernel (per bank, open positions O^bank_k)

      ã_j = log α_j − λ_α ρ_j (t − Λ^α_j)            (lazy strength; ρ_j = 4^{−m_j}; λ_α = 2.5e-4)
      w_j = softmax_j( ã_j·[strength-read] − (d_j/d₁ − 1)/τ_w ),   τ_w = 0.1
      A   = exp(−d₁/(2 h₀,b²)) · (σ(ã_{j₁}) if --bm-strength-read 1 else 1)        (default 0: strength only for eviction)
      f_j = exp(−λ_v ρ_j (t − Λ^v_j))
      r   = A Σ_j w_j f_j v_j

- Why strength only for eviction (default): with rev1's λ_α = 1e-3, a fresh slot (m = 0) had ã = −1.3 after the
  mean rare-fact gap of 1.3K steps, σ(ã) = 0.21, i.e. its own owner's read was scaled down ×0.4 relative to a fresh
  write, exactly when the bank is needed. Without σ(ã) in A, the strength only orders eviction; λ_α = 2.5e-4 (−0.33
  per 1.3K steps) keeps unconfirmed slots near-equal over the rare-fact horizon so eviction is close to FIFO among
  them.

### 3.3 h₀: calibrated on same-fact distances, then tracked (review points 6 of rev1 and 7 of rev2)
- **Why an EMA over confirmed reads collapses** (rev1 derivation, kept): with h² ← median d₁ of reads passing
  d₁ ≤ c₀h², and F(x) ∝ x^{d_eff/2} near 0, h² contracts to 0 whenever c₀ < 2^{2/d_eff}.
- **Rev1's example was wrong.** It claimed a median-drifted owner read has A ≈ e^{−0.86} ≈ 0.42 at h₀² = NN². The
  median drift is 1.31 × the 1-NN gap, so d₁ ≈ 1.72 NN²:
  - at the rev1 floor h₀² = ½ NN²: exp(−1.72) = 0.18;
  - at h₀² = NN²: 0.42, but rev1's A also had the factor σ(ã) ≤ 0.5 for a fresh slot: 0.21;
  - so in both cases below A_min = 0.3. The rule was not tied to the distance that matters: the distance between a
    fact's write-time key and a later cue of the SAME fact.
- **Rule (rev2).**
  1. **Calibration (T0e)**: for each pass b, the distribution of d_same = the squared distance between a fact's key
     in one document and its nearest same-fact cue in another document (the T0a capture, across the 6K → 7K gap and
     within a checkpoint). Set

         h₀,b² = q₅₀(d_same,b) / (2 ln 2)          (a median same-fact read has geometry factor A = 0.5)

     and record the ratio κ_b = h₀,b² / q₅₀(d₁ over all open reads of pass b) on the same capture.
  2. **Tracking (online)**: h₀,b² ← (1 − 0.01) h₀,b² + 0.01 · κ_b · q₅₀(d₁ over ALL open reads of bank b in the
     step). The population is not selected by A, so there is no contraction; κ_b carries the same-fact scale.
  3. **Floor**: h₀,b² ≥ ½ q₅₀(NN distance² among 4,096 sampled keys of bank b) (every 250 steps).
  - If T0e cannot separate same-fact from other-fact distances (see T0a), h₀ is moot: Stage B does not start.

### 3.4 The content gate and the output (per bank)

      φ^c = [sg(p̄_k) (64); r (256); A; log(d₂/d₁); onehot(m_{j₁}) (4); log(1 + age_{j₁})]     (327)
      a   = w_cᵀ φ^c + b_c,   g = σ(a),   b_c = −2           (no logit L2: dropped, §0.3)
      u   = g r,   M^bank = W_o^(b) Probe(u)   (W_o^(b) ∈ R^{384×256}, zero init, one per bank),   then added to M_k

- **p̄ enters the content gate under sg** (§OD-c): no gradient from the gate into the thought. The read value r (and
  so M) flows into the brain as normal.
- Per-bank W_o (9 × 98K = 0.9M parameters); `--bm-wo shared` (+ a 16-d pass embedding) is the ablation.
- `Probe(u)`: an identity autograd function; its backward computes c_t = −⟨∂L/∂u_t, u_t⟩ per position and writes it
  into a preallocated buffer slot (§4.6).

### 3.5 Gradient through the match weights (DECIDED: yes)
- **The question**: the owner confirmed that no GATE loss pushes the thought. Separately: may the main LM loss flow
  back through **which memories matched** (the soft match weights) into the thought, not only through the value read?
  **Decided: yes** (both paths).
- **The design**: in the bank, w_j = softmax(−(d_j/d₁ − 1)/τ_w) and A = exp(−d₁/(2h₀²)) are differentiable w.r.t.
  the query x̂ (hence v_k, the pass-k thought); the slot indices j₁..j₄ stay discrete (no gradient), and the keys
  receive none. In the rows the same holds for the soft weights over the top-m product-key rows (today they come from
  the online side; with `--mem-addr-learn 0` the translators are fixed, so only the query side would get gradient).
  - Effect: the LM loss can move the thought toward the key whose value helped ("think in a way that finds the
    useful memory"). This IS a memory → thought pressure through the address, which constraint 2 as written
    ("the address path is stop-gradient") excluded; it is the main LM loss, not a gate loss, and the owner side has
    decided to allow it. Constraint 2 now reads: no GATE loss reaches the thought; the LM loss may, through the read.
  - Risk: the self-reinforcement loop (§12) gets a gradient path (the thought drifts toward stored keys, which then
    match better); monitored by the read-induced view shift and the echo rate.
- Ablation flag `--mem-match-grad 0`: w and A computed from sg(x̂) (rev1's behaviour); the gradient reaches the
  thought only through the value read.
- Implementation note: the bank lookup stays no_grad (it only picks the indices); w and A are recomputed with grad
  from x̂ and the gathered (detached) keys inside the checkpointed block: a few elementwise ops on (Q, 4).

---------------------------------------------------------------------------------------------------------------------

## 4. The per-pass read gate (MemGate extended; learned budget: the brain decides, owner answer b)

### 4.1 What is reused unchanged (`memory.py: MemGate, gate_sizes, gate_c, gate_trace`;
`loop_brain.py: _gate_phi, _gate_open, _gate_after, _gate_readout, _gate_surprise`)
- The 93 detached features (p̄ standardised, ν̄_l, Δμ_l, the thought change, the last-token surprise + flag, the
  surprise trace, the next-token entropy, the pass one-hot).
- The read-out reorder (training computes the pass-k read-out before the read).
- The fp32 router in its own AdamW group and clip, excluded from the global clip.
- The device-side generator, the fixed tie key, the selection cached for the recompute; eval modes learned / random /
  inverted / oracle / open / off; readdr off under the gate.

### 4.2 What changes
1. **Per pass, per store.** ĝ^{(m)}_{k,t} = W^{(m)}_{b(k)} φ_{k,t} + β^{(m)}_{b(k)}, m ∈ {row, bank}: W ∈ R^{2×9×93},
   ≈ 1.7K parameters. Each pass has its own decision rule. (Stage A has only m = row; option A only m = bank.)
2. **What the router predicts**: the centred gain ĝ in nats (§4.3), not a probability.
3. **Selection (default, learned budget)**:

       O^{(m)}_k = TopUpTo_{n_k}( { t : ĝ_{k,t} > c_read } )  ∪  E_k

   - E_k: a 5% **uniform sample over ALL positions** (device generator), always read; it is the exploration and the
     centring sample (§4.3). Overlap with the selected set is counted once.
   - n_k = the static buffer size of read k: the realised count of read k at the PREVIOUS step (a one-step-lagged
     non-blocking device→host copy, read at the next step: no stall), plus a 10% margin, rounded up to 64, clipped to
     [64, ⌊0.5 N⌋]. Positions with ĝ > c_read beyond n_k are dropped lowest-ĝ first and logged as overflow.
   - The cap ⌊0.5 N⌋ per pass is the speed cap.
4. **The read cost c_read** (one constant scalar, nats; **no schedule**, owner answer b):
   - Default c_read = 0: a position reads iff its predicted gain beats that of the median position. The open rate per
     pass and over training is then whatever the router predicts: the brain decides how many positions read.
   - Option `--gate-cost time`: c_read = κ · Δt_read, with Δt_read the measured seconds per opened position (T2) and
     κ an owner-chosen exchange rate (nats per second of step time). Constant over the run.
   - Option `--gate-cost <nats>`: any fixed value.
   - Rev1's c-schedule (all open until 20%, cosine to 0.25-0.30 by 50%) and the coordinator's annealed c_read are
     dropped: both imposed the rate.
   - Early training: the router starts near ĝ ≈ 0 (β = 0, small random W), so ≈ half the positions are open plus E,
     enough to fill the tables; plan 22's "all open until 20%" warm-up is not needed for filling, but is available as
     `--gate-warm 0.2` (a flag, off).
5. **Why it does not collapse**:
   - all-open: impossible beyond the 50% cap; and since z is centred at the population median, only about half the
     positions can have ĝ > 0 ≥ … ≤ c_read once c_read ≥ 0;
   - all-closed: E_k (5%) always reads, so the target is observed on a uniform sample of the closed region and ĝ
     cannot drift down unobserved; because z is centred, its upper half is > 0, so a calibrated ĝ keeps ≈ the share
     of positions with z > c_read open;
   - bounds: the realised open rate is clipped to [5%, 50%] per pass (the two safety pieces);
   - watchdog (§4.5): alerts only (no automatic fallback, owner answer b).
   - What centring gives up: the gate cannot decide "reads are useless everywhere" (the batch mean of c is pinned near
     0, §5.1). The absolute value of reading is the knockouts' job (§14), not the gate's.
6. **Inference**: the same rule ĝ > c_read (no top-c EMA threshold is needed in the learned mode), the cap applied per
   sequence batch; no exploration at eval. θ̂ (plan 22) is kept for the fixed-budget mode.

### 4.3 The target and why it is consistent
- For the in-window reads, from the Probes (after backward):

      c_{k,t}   = −⟨∂L/∂u_{k,t}, u_{k,t}⟩                       (u^row summed over levels / pools; u^bank)
      z_{k,t}   = N_tok · ( c_{k,t} − q₅₀^{E}(c_k) )            (q₅₀ over the exploration sample E_k of this micro-batch)
      L_r       = λ_r · mean_{in-window k, t ∈ O_k} Huber_δ( ĝ_{k,t} − z_{k,t} ),   δ = N_tok · 1.4826 · MAD^{E}(c_k),  λ_r = 0.1

  - N_tok = the number of tokens the micro-batch loss averages over (×accum), so N_tok·c is the first-order change of
    the summed loss in nats when the read at t is removed.
  - **Centred on E** (review point 10): rev1 centred on the open set O, which the router itself selects; its median
    moves with the selection, so the target's zero was not the population's. E is uniform over all positions and
    independent of φ, so q₅₀^E estimates the population median. With 5% of 8,192 (16×2) = 410 positions (205 at 8×4)
    the median's sd is ≈ 0.06 MAD (≈ 0.09 at 8×4).
- **Consistency (plan 22 Proposition 2 carries over)**: the selection depends on φ and the exploration draw only; z_t
  depends on position t's own read and on a batch-level constant (q₅₀^E), which shifts all targets alike. E gives
  positivity everywhere. So the regression minimiser on the open set is E[z | φ] on the whole support (missing at
  random given φ).
- **Out-of-window reads**: no c, no target. They use ĝ (no_grad) but do not train it. Share of reads that train it:
  P_win(k)/f_k ≈ 0.53 (k = 0) … 0.87 (k = 8).
- **Fixed-budget mode** (`--gate-budget fixed`, a flag for debugging / ablation only): top-c by ĝ + 5% E, with rev1's schedule
  (c = 1 until 0.2·S, cosine to 0.30 rows / 0.25 bank by 0.5·S) and θ̂ for inference. The router keeps the same
  regression target (ranking by ĝ). Rev1's rank-BCE target (y = Φ((c − q₅₀^E)/(1.4826 MAD^E)), BCE) is
  `--gate-target crank`.
- **What it fixes**: if the train-mode c on non-answer world tokens is below that on answer tokens (T0b), ĝ ranks them
  low and they close. If T0b shows otherwise, the harm is a train / eval mismatch the gate cannot see (§19, risk 2).

### 4.4 Optional: per-pass budget adaptation in the fixed mode (`--gate-adapt 1`)
- In the fixed mode only, every 250 steps (host reads at logging steps): compare the mean centred z of the explored
  positions just below the cut-off with the lowest decile of the top-c; c_k ← clip(c_k ± 0.05, 0.10, 0.50). (The
  learned mode subsumes this.)

### 4.5 Watchdog (alerts only)
- Logged each eval, per pass and store: the AUC of ĝ vs 1[z > 0] on the exploration sample of a held-out micro-batch
  (plan 22 phase 0 reached only 0.61 with ḡ); the open rate; the overflow rate; the calibration slope of z on ĝ.
- **Alert** (a loud log line, no behaviour change; owner answer b: no automatic fallback): after 0.3·S, the AUC < 0.60
  at two consecutive evals, or the realised open rate of passes 0-5 pinned at the 5% floor or the 50% cap for 500
  steps. The owner / operator then decides (e.g. a rerun with `--gate-budget fixed` for diagnosis).

### 4.6 The post-backward router update (review point 1)
- **The problem in rev1**: the router's loss was formed in the forward (as today's `_gate_after`, whose target ḡ is
  available in the forward). The target c exists only after backward.
- **The fix**: for a linear router the gradient is closed-form, so no autograd graph is needed.
  1. Forward (each micro-batch, each in-window read k, each store): keep φ_{k,O} (fp32, |O| × 93), ĝ_{k,O}, the open
     indices O and the E mask, detached, until after backward. Size: ≤ 8,192 × 93 × 4 B ≈ 3 MB per read per store;
     ≤ 9 reads × 2 stores → **≈ 55 MB per micro-batch** worst case (early in training, everything open), freed after
     the update.
  2. Backward: each Probe writes c_{k,O} into its preallocated buffer `cbuf[mb, k, m]` (an index_copy_, idempotent)
     and increments a device counter `cnt[mb, k, m]`.
  3. After `backward()` of the micro-batch (before the next micro-batch's forward): compute z (§4.3) and

         ∂L_r/∂W^{(m)}_{b} = (λ_r/|O|) Σ_{t∈O} ψ_δ(ĝ_t − z_t) φ_tᵀ,    ∂L_r/∂β^{(m)}_{b} = (λ_r/|O|) Σ_t ψ_δ(ĝ_t − z_t)
         (crank flag: replace ψ_δ(ĝ − z) by σ(s) − y)

     added (÷ accum) into `.grad` of the router parameters. ψ_δ is the Huber derivative (clip to ±δ).
  4. The router's own AdamW steps once per optimizer step, as today. W is unchanged within the step, so ĝ computed in
     the forward used the same W: the update is the exact gradient.
- **Checkpoint recompute**: `_hier_step` is ONE activation-checkpointed block (non-reentrant). The forward runs twice
  (original + recompute during backward), but only the recomputed graph's backward runs, once. The Probe therefore
  does nothing in its forward (no hook registration, no append) and writes only in backward, into a fixed slot. A
  stray double backward would overwrite the same values (idempotent) and show as cnt = 2.
- **Test** (`test_gate_postbwd`, CPU): a tiny LoopBrain with the checkpointed block; asserts (a) cnt == 1 for every
  in-window (k, store) and 0 outside; (b) c equals −⟨∂L/∂u, u⟩ computed with checkpointing off; (c) the analytic router
  gradient equals autograd of L_r with z held fixed (rtol 1e-5); (d) with accum 2 and 4 the accumulated gradient
  equals the full-batch one.
- Today's diagnostic hook (`gate_diag`, `M.register_hook` on the block output, outside the checkpoint) is replaced by
  the Probe buffer.

---------------------------------------------------------------------------------------------------------------------

## 5. The benefit signals

### 5.1 c = −⟨∂L/∂M, M⟩ is ≈ zero-sum, so only centred values are used
- With M_t = W_o u_t: Σ_t c_t = −⟨∂L/∂W_o, W_o⟩_F ≈ ω ‖W_o‖_F² near a stationary point under weight decay ω: small.
- So the batch mean is pinned near 0 whatever the memory's value. Hence: no sign-based disconfirmation, no
  spawn-on-hurt; the targets are centred on E; the absolute usefulness is measured by knockouts (§14).
- Per position, c is informative (answer tokens vs others is exactly the salvage's contrast).

### 5.2 The forward-only memory test (alternative for B2; must be calibrated first)
- For confident bank reads (top E_b = 128 per bank per optimizer step by A): CE_mem,t = −log softmax(W_lm
  normf(M^bank_{k,t}))[y_t]; confirmed iff CE_mem,t < ce_k[t] − m_ce. No gradient needed, so it also works outside
  the window. M is not trained to be decodable: T0b measures it. Enabled only if precision ≥ 0.85 at recall ≥ 0.3.

---------------------------------------------------------------------------------------------------------------------

## 6. Writes (Stage B v1: spawn + dedupe)

### 6.1 Which pass writes
- Each in-window read k writes into bank b(k). Out-of-window passes do not write (no ∂L/∂inject_{k+1} for the value
  init). `--bm-write-nograd 1` (ablation): ξ-only values for out-of-window spawns.

### 6.2 The surprise criterion and eligibility (review point 4; T0h decides)
- **The concern**: the write top-k by raw final-pass CE is likely dominated by generally hard tokens (Wikipedia names,
  numbers, rare words) rather than rare-fact cues. p_write (the fraction of rare-fact cues written) enters the recall
  bound linearly (§7.3) and was assumed 0.6 in rev1 without measurement.
- **Candidate criteria — the brain's own signals only** (no dataset labels, no source tags, no document-type
  features; constraint 1 and the coordinator's rule):
  - σ^raw_t = ce_{K−1}[t] (the final-pass CE; rev1);
  - σ^ent_t = ce_{K−1}[t] − Ĉ(H_{K−1,t}): the CE above what the model's own predictive entropy H predicts (Ĉ = an
    EMA binned regression of CE on H, 32 bins). A token that is hard AND that the model was confident about is a
    knowledge gap; an inherently unpredictable token (high H) is not;
  - σ^imp_t = ce_{K−1}[t] · 1[ce_0[t] − ce_{K−1}[t] < δ_imp]: high CE that thinking did NOT reduce (a gap thinking
    cannot close: what memory is for), or the continuous form ce_{K−1} − β·(ce_0 − ce_{K−1});
  - all are computed from tensors the training step already has (ces per pass, the entropy feature of the gate).
- **T0h** (§16.1) measures p_write for each criterion at the same budget; the best one is the default; if none
  reaches p_write ≥ 0.3, Stage B does not start.
- Eligible_k(t): t ≥ 8 tokens after a document start, not EOT, AND no confident bank read at (k, t) (t ∉ O^bank_k, or
  A_{k,t} < A_w = 0.3).
- Budget: W = 32 per optimizer step per in-window read (§2.2), at most 2 per sequence, ≥ 8 tokens apart; static topk.
- Candidate stash: index, x̂_{k,t} (64), σ_t, ĝ_{k,t} = −∂L/∂inject_{k+1}[t] (384) by a hook on the summed inject at
  the W candidate positions (registered on the block output, outside the checkpoint: fires once). ≈ 50 KB per bank
  per step.

### 6.3 Key and value
- Key: x̂_{k,t}, τ = the current step.
- Value: q = W_o^(b)ᵀ ĝ;  v = σ_ξ ξ + ν v̄_b · q / max(‖q‖, ε_q) · 1[‖q‖ ≥ ε_q],  σ_ξ = 0.02, ν = 1,
  ε_q = 1e-6 · ‖ĝ‖ (relative). v̄_b = the EMA of the median ‖v‖ of bank b's slots with ≥ 1 Adagrad touch (init 0.05).
  - **At the start W_o^(b) = 0** (zero init), so q = 0: the value is ξ-only (no division by zero). W_o grows from the
    ξ-values' reads; the gradient write takes over once ‖q‖ ≥ ε_q.

### 6.4 Dedupe at write and the merge rule (with a wrong-merge bound)
- **Against the store** (in after_step; one batched brute-force lookup for ≤ 160 candidates per step): d₁, j₁ = the
  nearest live key in the bank.

      d₁ < γ h₀,b²  ⟹  MERGE into j₁:
         k_{j₁} ← k_{j₁} + (x̂ − k_{j₁}) / (n_{j₁} + 2);   v_{j₁} ← v_{j₁} + (v_new − σ_ξ ξ) / (n_{j₁} + 2)
         τ_{j₁} ← τ_{j₁} + (t − τ_{j₁}) / (n_{j₁} + 2);  n_{j₁} ← min(n_{j₁} + 1, 8);  log α_{j₁} += 0.5;  Λ^α_{j₁} ← t
      otherwise  ⟹  SPAWN (§6.5)

- **γ from T0e**: the largest γ (≤ 0.5) at which the **wrong-merge rate** — P(j₁ belongs to a different fact | d₁ <
  γh₀²), measured on the T0a capture with the real distractor density — is ≤ 2%; the same-fact merge rate at that γ
  is reported. Placeholder γ = 0.25.
- **Within the step**: per bank, the candidate distance matrix ordered by σ; a candidate is suppressed if an earlier
  one is within γh₀²; its value increment goes to the kept one (index_add_). Merges do not consume eviction capacity.

### 6.5 Spawn and eviction
- A spawned slot: k = x̂, v as §6.3, log α = 0, Λ^α = Λ^v = Ψ = τ = t, m = n = 0, S = S₀,b.
- Victims: the free list, then the weakest lazy strength ã in the bank, excluding t^touch = t (one masked topk).
- Turnover: an unconfirmed bank-0 slot lives ≈ N_0 / (32 · 0.53) ≈ 2,830 steps.

---------------------------------------------------------------------------------------------------------------------

## 7. Confirmation and re-encode (Stage B2, later) and the survival model

### 7.1-7.2 Stage B2 (not in v1)
- Kept from rev1 for later, enabled only if T1 shows ≥ +0.03 rare-group recall from re-encode over spawn + dedupe:
  - rank confirmation: in window, the top 25% of the bank's in-window open reads by centred c that pass the T0e
    geometry thresholds (w₁ ≥ ω₁, d₂/d₁ ≥ μ_m, A ≥ A_min), ≤ E_b = 128 per bank per optimizer step; or the CE test
    (§5.2) if T0b passes;
  - the update: per-slot averaging of the confirmed readers, η = 0.5 (1 for u < 2), the move capped at h₀; τ, Λ^α,
    t^touch ← t; log α ← min(ã + 1, log 8); maturity m ← min(m + 1, 3) if ≥ 64 steps since the last; Λ^v untouched;
  - the separation bound ‖k₁′ − k₂‖ ≥ (√μ_m − (1 − η))√d₁; the "fixed point" is a lagged EMA of recent confirmed
    readers, not a benefit-weighted centroid, under drift.

### 7.3 The renewal probability and the survival model (corrected; review point 9)
- A recurrence renews F's key only if p_ren = p_open · p_win · p_ret(a) · p_conf (B2) — or, in v1, if it is
  **re-written** (a merge): p_ren^{v1} = p_write · p_merge.
- **Survival (corrected).** Renewals arrive as a Poisson process with rate λ = p_ren / Δ (Δ = the recurrence spacing).
  The key is lost when a gap between renewals exceeds the horizon H (where p_ret collapses). Lost-gap events occur at
  rate λ · e^{−λH} (each renewal starts a gap that exceeds H with probability e^{−λH}). Over T steps:

      S(T) = exp( −λ T e^{−λH} )

  - Rev1 used (T/H) · e^{−p_ren H/Δ}, which counts windows instead of renewal-started gaps.
  - **Late facts** (Δ ≈ 105, T = 12.2K, H ≈ 1,500): with x = λH = 14.3 p_ren, λT = 8.13 x. S ≥ 0.9 needs
    8.13 x e^{−x} ≤ 0.105, i.e. x ≥ 6.17: **p_ren ≥ ≈ 0.43** (rev1: 0.31). A plausible B2 value is
    0.6 × 0.53 × 0.8 × 0.5 = 0.13. A bank cannot carry the recurring facts.
  - **The rows under the gate** (correction to rev1's "the rows are rewritten by gradient at every exposure"): with the
    row gate, a row gets gradient only where it is read, i.e. only at OPEN positions. A late fact's rows are renewed at
    an exposure with probability p_open^row(cue) × P(the cue's read is in window). So the Stage A gate must keep the
    answer cues open: logged as p_open^row at cues (§14.3), and the reason the late-cloze criterion is checked at 7K
    AND at the end (§16.4). With per-pass tables (option B) each pass's table is renewed only by its own reads: per
    pass, p_open^row(cue) · P_win(k) ≈ 0.8 × 0.53 = 0.42 ≈ the 0.43 bar for passes 0-4, below it for the deep passes.
    The fact survives if ANY pass's table keeps it (the reads are summed into each pass's inject chain), so the union
    over ≥ 3 passes, 1 − (1 − 0.42)³ ≈ 0.81, is the relevant renewal if the cues stay open; with the shared table of
    the old run every open in-window read renewed the same rows.
  - **Rare facts** (Δ ≈ 1,290, ≈ 5.5-9.5 exposures): λH ≈ 1.16 p_ren, renewal is nearly irrelevant. What matters is
    that the last exposure **writes** the fact:

        P(recall) ≈ p_write · p_open(eval) · E_{a ~ Exp(Δ)}[ p_ret(a) ] · p_val

    - p_write from T0h (rev1 assumed 0.6); p_val = P(the value decodes the answer);
    - with the untransported curve (R@1 1.0 / 0.77 / ≈ 0.45 / ≈ 0.15 at ages 0 / 1K / 2K / 3K, 4K bank),
      E[p_ret] ≈ 0.6 × the T0a cross-document factor.
- **T0a bar** (unchanged logic): rare exact ≥ 0.10 needs p_write · p_open · E[p_ret] · p_val ≥ 0.10; at p_write 0.6,
  p_open 0.8, p_val 0.6 this is a cross-document same-fact R@1 ≥ ≈ 0.6 at age 0; at p_write 0.3 it is unreachable
  (≥ 1.2). So **T0h is as much a gate as T0a**.

---------------------------------------------------------------------------------------------------------------------

## 8. Fading, maturity, clocks (lazy, plan 21 style)
- **Strength** (Λ^α): ã_i = log α_i − λ_α ρ_i (t − Λ^α_i), **λ_α = 2.5e-4** (rev1: 1e-3), ρ_i = 4^{−m_i}; renewed by
  spawn / merge (B2: confirmation). Default: used for eviction only (§3.2).
- **Value** (Λ^v, separate): f_i = exp(−λ_v ρ_i (t − Λ^v_i)), λ_v = 1e-4, folded at every Adagrad touch.
- **Accumulator** (Ψ): discount 1 − β_i = (1 − β₀) ρ_i, β₀ = 0.99, floor 0.1 S₀.
- Keys are never frozen.

## 9. (Transport: removed from this plan; see Appendix A)

---------------------------------------------------------------------------------------------------------------------

## 10. The after_step order and host syncs

### 10.1 Order
- **After each micro-batch's backward**: the router's analytic gradient (§4.6); the bank's compact value gradient is
  accumulated (sort + segment sum into a static buffer).
- **After opt.step** (once per optimizer step, after the row MemOpt):
  1. router AdamW (own group);
  2. bank Adagrad on the values (lazy fold f, Λ^v ← t, discounted S, clipped step, index_copy); t^touch ← t;
  3. (B2: re-encode the confirmed slots);
  4. evict / merge / spawn (merges first; spawns into the free list / weakest with t^touch < t);
  5. refresh the bf16 key copy of the changed slots;
  6. EMAs: h₀,b, v̄_b, Ĉ (§6.2), the router statistics (device scalars).

### 10.2 Host syncs (corrected)
- **Rev1 was wrong** to say "the only host read per step stays the one that exists today": the baseline step has
  **48 host syncs (≈ 0.2 s, ≈ 10% of the step)** (VRAM study, `log_ptime.txt`; e.g. the routing `.tolist()`, the
  loss `.item()`, `write_counts`). Removing them belongs to the speed track.
- This plan adds **no blocking sync**:
  - every budget is a host integer (W, E_b, the exploration size, the chunk sizes; K is drawn on the host);
  - the learned budget's buffer size n_k uses the previous step's count via a non-blocking copy read one step later;
  - variable-size sets are masks over static buffers with a dummy slot; per-slot averaging uses sort + neighbour
    flags + cumsum + index_add_ (no `torch.unique`);
  - logging reads device tensors only on logging steps.

---------------------------------------------------------------------------------------------------------------------

## 11. Coexistence with the row memory and ConceptHier

### 11.1 What stays
- The row mechanics: plan 21 as run (`--mem-life 1 --mem-addr-learn 0 --mem-restart 100`), addressed by [μ_l; p̄]
  through ConceptHier, gradient-written, lazy fading — but **per pass** (option B, §11.4), not shared.
- ConceptHier unchanged. M = M^row + M^bank into the same destinations.

### 11.2 The row-memory changes in Stage A (knockout-checked)

| change | evidence | saves (VRAM study / est.) | check |
|---|---|---|---|
| `--mem-private 0` | ko_private: late exact −0.004 | ≈ 1.3 GB (V_e 0.40 + G_e 0.40 + P, P_ema, P_m, P_v, G_P 0.10 each) | Stage A criteria (§16.4) |
| `--mem-nbr` | an add-on (plan 21 §4); no knockout measured | — | **kept on** (as run) to limit confounds |
| per-pass row tables (§11.4) | owner answer (a) | 0 (same total rows) + 0.28 GB for per-pass coarse pools | Stage A vs the shared-row run at the same steps |
| the per-pass row gate (learned, cap 50%) | plan 22 §3; readdr off under the gate | row-read time ∝ the open share; T2 measures | world_other ≤ ko_shared + 0.01 |

- Removed from rev1: the content-gate logit L2 (§0.3) and the compact row gradient (54% of the shared rows and 36% of
  the private rows are touched per step: a compact buffer + a row→slot map over ≈ 12M row-writes saves ≤ 0.5 GB and
  costs time; dense is right at this density).

### 11.3 Bit-identity
- `--bm 0 --mem-gate 0 --mem-perpass 0`: today's code, bit-identical (new modules built last, forked RNG).
- `--bm 0 --mem-perpass 0 --mem-gate 1 --gate-target content --gate-budget fixed`: plan 22 exactly.

### 11.4 Per-pass memory: option B (default) and option A [the choice pending Eitan]
**Option B: per-pass row tables + per-pass banks (default).**
- **Fine rows**: today `--mem-shared 512` gives 4 levels × 512² = 1,048,576 rows in one V (+ the dense G). Option B
  keeps ONE V / G tensor of about the same size and gives each table b = 0..8 (read k uses table min(k, 8), like the
  banks) its own range: 4 levels × S_p² with **S_p = 170** (9 × 4 × 28,900 = 1,040,400 rows). Row index = o_b + the
  local product-key index. Same VRAM, same read / MemOpt cost (the work scales with the reads, not with the table).
- **Per-pass addressing**: each table has its own fixed random sub-key sets (S_p per half per level per head) and its
  own content gates and W_o (4 × 256 × 384 = 0.39M parameters per pass, 3.5M in all); ConceptHier and its μ_l are
  shared (they are the thought, not the memory).
- **Coarse pool per pass too**: the coarse level is 64² per level (16K rows, 16 MB V). Splitting it 9 ways would leave
  21² per level (too small to mean anything), so each pass gets a full 64² coarse pool: 9 × 16,384 rows ×
  256 × 4 B × 2 (V + G) ≈ **+0.28 GB**. Nothing breaks; its knockout effect was small (−0.001), so dropping it per
  pass (`--mem-coarse 0`) is the fallback if VRAM is tight.
- **What changes in the dynamics**: ≈ 9× fewer rows per pass (≈ 116K), so more collisions per table; the deep tables
  (k ≥ 5) are read on ≤ 47% of steps and fill slowly; `--mem-restart` / the life-cycle constants are per table.
  Optional `--mem-perpass-split freq`: table sizes ∝ f_k instead of equal (more rows for the early passes).
- **Rows stay the baseline** the banks must beat (Stage B); they are dropped only if banks alone beat them (Stage C).

**Option A: bank-only (alternative).**
- No row table: the per-pass banks (§2) are the whole memory, with the learned gate on the bank store only. Frees the
  rows' ≈ 2.2 GB (V + G) and their read time; the banks could then be larger.
- Risk: it drops the only memory proven on real data (shared rows: late cloze 0.280 vs 0.182 knocked out) before the
  banks have shown they carry recurring facts (the survival model needs p_ren ≥ 0.43, §7.3), and before T0a (likely
  to fail) and T0h. It is therefore run as **Stage C** (a comparison) rather than as the starting point.

---------------------------------------------------------------------------------------------------------------------

## 12. Self-reinforcement: measured, not forbidden

**The loops**: (1) bank read k′ < k → s_k → the key / query of bank k; (2) rows → thought → bank keys; reads → c → ĝ
→ what opens → what trains → c; (3) (B2) re-encode: read slots get renewed and read more; (4) with the match-weight
gradient (§3.5, decided): LM loss → match weights → thought → the next query. The sg on every gate's p̄
input (owner-confirmed) removes the gate → thought gradient path; the value path remains.

| metric | definition | what it detects |
|---|---|---|
| read-induced view shift | 8 probe sequences: ‖v_k(memory on) − v_k(all reads off)‖ / h₀,b per bank (one extra probe forward per EVAL, not per step) | the memory moving its own keys |
| echo rate | P(top-1 slot written by the SAME document) among confident reads | reading its own recent writes |
| concentration | Gini of reads per slot per bank; share of reads on the top 1% of slots; trend | attractor slots |
| re-encode feedback (B2) | Spearman(re-encodes so far, rank-benefit b̄) | rich-get-richer without benefit |
| open-set persistence | overlap of O_k between consecutive reads and across steps on the probe set | a gate locked open / closed |
| closed-region starvation | z on E ∩ closed vs on the open set | the content loop of plan 22 §2.5 |
| knockout of earlier-pass reads | eval: reads at k′ < k off, bank k's recall / benefit | dependence of later passes on earlier ones |

**Review thresholds**: echo > 0.5 with the benefit rank below the median; top-1% share > 20%; view shift > 1 h₀.

---------------------------------------------------------------------------------------------------------------------

## 13. Training data (v2 mix and a loader that makes time meaningful)

**1. A sequential loader with 32 cursors** (`--data-order seq`; default stays random for comparability).
- The stream (N tokens, time-ordered, shuffled at document granularity by the generator) is split into 32 segments.
  Cursor i starts at i · N/32 (plus a random offset < 512) and, at step s, yields the window
  [i·N/32 + s·512, +513). One window per cursor per optimizer step = 32 windows = the effective batch, whatever the
  micro-batch split (16×2 or 8×4: cursors 0-15 / 16-31, or 4 groups of 8).
- Adjacent windows within a step are from different segments, so they are not correlated (rev1's "one contiguous
  16,384-token range per step" made them so). Each cursor advances 512 tokens per step: 12.2K steps × 512 = 6.25M
  tokens per segment (of 12.5M at N = 400M).
- "Time" = the step: a mention scheduled at step s is placed by the generator in one segment (chosen at random) at
  offset s·512. `exposures.npz` becomes valid and "steps since the last mention" exact.
- Both the control and the memory arms use it.

**2. A rare group** (`--rare-people 2000`): each person's 5 facts mentioned k ∈ {1, 2, 4, 8} times (500 people per
k); per-person spacing ∈ {10, 100, 300, 1,000, 3,000} steps; each mention a different document and template (0-9).
Cloze queries on held-out templates 10-11. Query age = final step − last mention, buckets {≤ 300, 300-1K, 1-3K, > 3K}.

**3. Introduce-then-query**: fact F in document A (template i); Δ ∈ {10, 100, 1,000} steps later document B with a
different template j and the value at the sentence end. Also a same-template variant (i = j), which isolates the
template-keying risk (T0a).

**4. Keep the late group and the early group** (renamed "rare-random"; under the sequential loader it becomes a true
early-then-silent group). Reported, not a Stage B GO criterion.

**5. Exposure log**: (step, fact id) for each fact mention per batch into `exposures_run.npz` (host-side; eval
bookkeeping only: **the memory and the write criterion never see it**).

**Sizes**: ≈ 37.5K fact sentences ≈ 0.4M tokens; the world share stays 20%.

---------------------------------------------------------------------------------------------------------------------

## 14. Metrics and evaluation

### 14.1 Primary
1. **Fact recall**: cloze exact / first-token / lp per token, by group: late, rare-random, the rare group by k ×
   spacing × query age, introduce-then-query by Δ and same / different template; × fact type.
   - Baselines: late 0.280 (mem_life at 7K), control final 0.260; rare-random 0.016 at 7K (≈ 5.5 exposures; compare
     only at equal steps).
2. **Loss by target-token type** (salvage `extra_split.py` into eval_world): world_answer_first, world_answer_rest,
   world_other, wiki; knockout deltas ko_rows, ko_bank, ko_all, ko_bank_k, ko_gate (forced open).

### 14.2 Automatic eval at 25 / 50 / 75 / 100% (`--auto-eval 1`)
- In-process, between steps, **after `torch.cuda.empty_cache()`** (§VRAM), eval_batch 16; settings `--rank-limit 200
  --questions 1 --knockouts rows,bank,all --gate-modes learned,open,off --capacity dropless`. Not counted in tok/s.

### 14.3 Online logs (every eval)
- **Gate, per pass and store**: open rate (train / eval), c_read, the cap / floor hits, overflow, the AUC and
  calibration slope (§4.5), the open rates by token type (cue, answer, other world, wiki: eval tags only), **p_open^row
  at answer cues** (§7.3).
- **Bank, per bank**: live slots; spawns / merges / evictions per step; h₀ and κ_b; median d₁, margin, A; the bf16
  re-rank miss rate (§2.4); p_write per criterion on tagged eval windows (eval bookkeeping only).
- §12's table.

---------------------------------------------------------------------------------------------------------------------

## §VRAM (from `fruit-fly-sim\speed_wip\vram\vram_study.md`)

### Measured today (world_mem_life_12k at end-of-growth size, 16 × 2, seq 512, bptt 6)

| point | allocated (GB) | reserved (GB) |
|---|---|---|
| persistent state after load | 4.70 (+0.63 grads / opt state when stepping) | 5.6 |
| training step, K = 6 / 10 / 16 | **20.12 / 20.14 / 20.25** (K = 2: 12.07, K = 4: 16.16) | 20.5-21.1 |
| `evaluate()` right after training (eval_batch 32) | 8.71 | **23.05** (the 23.1 in train.log) |
| loop report / probes (after free_cache) | 6.9 / 6.2 | 8.2-8.5 |

- Peak consumers at K = 10: activations of the 6 backprop passes ≈ 12.1 GB (≈ 2.0 per pass: SSD 0.63, experts 0.50,
  RMSNorm fp32 copies 0.28, workspace 0.17, concept heads 0.15, router 0.08); logits / CE ≈ 2.7 transient; memory
  tables 3.54 (V 1.09 + G 1.09, V_e 0.40 + G_e 0.40, P-family 0.5); concept loss 0.9; BrainOpt 0.60; params 0.55;
  micro-batch-1 grads 0.55. Fragmentation 0.4-1.0 GB in training, +2.9 after `evaluate()`.
- `expandable_segments` is not supported on Windows.

### Run defaults (apply to Stage A and later)
1. `torch.cuda.empty_cache()` before every `evaluate()` and before the loop report (today only around the report and
   probes): the eval spike 23.05 → ≈ 9 GB reserved.
2. A guard `torch.cuda.set_per_process_memory_fraction(20e9 / total)` ≈ **0.78** (≈ 20 GB). A guard only: near the cap
   the allocator retries and eval ran 2.4× slower.
3. **micro-batch 8 × accum 4**: 13.16 GB allocated, 13.5-13.8 GB reserved, −4.5% speed (measured). **Conditional on
   the same-maths audit (running)**: per-forward side effects run 4× instead of 2× per optimizer step (homeostasis's
   in-place `alpha` edits, balancer `pending` / `pend_e` / `lastq`, the EMA norm statistics, the MoE capacity C per
   micro-batch). If the audit finds a material difference, stay at 16 × 2 (reserved ≈ 21 GB: then the eval fix + the
   cap + optionally the chunked fused CE, 18.9 GB / −2%).
4. Optional (only if the bank needs room): chunked fused CE + a checkpointed concept loss (−1-2 GB, ≈ −2%).
5. In reserve: per-pass activation checkpointing (10.4 GB at 16×2, −12%; needs side-effect guards).

### This plan's deltas (vs today)

| item | Stage A | Stage B |
|---|---|---|
| private memory (V_e, G_e, P-family) | −1.31 GB | −1.31 GB |
| per-pass fine row tables (same total rows) / per-pass coarse pools (9 × 64², V + G) | 0 / +0.28 GB | same |
| row-read temporaries | ∝ open share (≤ 50%; measured in T2) | same |
| router: kept φ / ĝ / indices until after backward | +≤ 0.055 GB per micro-batch | +≤ 0.055 |
| bank store (368K slots: keys 94 MB + bf16 copy 47 MB, values 377 MB, meta ≈ 15 MB) | — | ≈ +0.53 GB |
| bank value gradients (static, ≤ 8,192 open × 4 winners × 256 × 4 B per in-window read, accumulated compactly) | — | ≈ +0.1 GB |
| lookup temps (1,024 × 48K bf16 score chunk + topk) | — | ≈ +0.1 GB transient |
| W_o^(b) + Adam moments | — | ≈ +0.01 GB |
| **expected peak allocated** | ≈ 19.1 GB at 16×2 / ≈ 12.2 GB at 8×4 | ≈ 19.8 GB at 16×2 / ≈ 12.9 GB at 8×4 |

- At 16×2, both stages are close to the 20 GB reserved cap (fragmentation 0.4-1.0 GB): 8×4 or the chunked CE is
  needed (or drop the per-pass coarse pools, −0.28 GB). Option A (bank-only) frees ≈ 2.2 GB (rows V + G).

---------------------------------------------------------------------------------------------------------------------

## 15. Cost vs the same-harness baseline (speed is a separate track)

- **The speed track** (not this plan): the step is GPU-bound on elementwise kernels; reaching 10-20k tok/s needs
  `torch.compile` / fusion of `Core.forward`, `ssd_scan`, RMSNorm + spike, fewer fp32 round trips, and removal of the
  48 host syncs. A speed study is running; its report goes to `fruit-fly-sim\speed_wip\speed_study.md` (not yet
  written). Memory stages are designed so that they do not block compile (static shapes, no new syncs).
- **This plan's criterion**: Δtok/s ≥ −3% vs the same harness without the change (repeat runs vary ≈ ±3%, so T2 uses
  the median of ≥ 3 runs per config).

| component (s/step, est.; T2 replaces) | Stage A vs baseline | Stage B vs Stage A |
|---|---|---|
| private memory off | −0.08…−0.15 | — |
| per-pass tables (same row count; per-pass coarse reads replace the shared coarse read) | ≈ 0 | — |
| rows read only on the open share (≤ 50%, ≈ 30-40% expected) | −0.1…−0.3 | — |
| router features, selection, the kept tensors, the post-backward update | +0.01…+0.02 | +0.005 |
| bank lookups (bf16 GEMM + topk 32 + fp32 re-rank), reads, backward | — | +0.015…+0.045 |
| bank after_step (Adagrad, dedupe / merge / spawn, key copy, EMAs) | — | +0.005 |
| micro-batch 8 × 4 (if adopted) | +4.5% | — |
| **net** | expected faster than the baseline, worst case ≈ −3% at 8×4 | ≈ −1…−2.5% |

- The 10k floor and the 20k aim are not claimed by this plan. At the measured 8.4k, Stage A may reach ≈ 9-10k by
  itself; the rest is the speed track's.

---------------------------------------------------------------------------------------------------------------------

## 16. Test plan with go / no-go gates

### 16.1 T0 — real-thought checks (CPU, ≤ 2 threads, CUDA_VISIBLE_DEVICES=-1 unless marked GPU)

| test | what | GO / NO-GO |
|---|---|---|
| **T0a cross-document same-fact retrieval** (gates Stage B; **expected to be hard / likely to FAIL**: the rows are template-keyed, and the views may be too) | a dedicated capture: val world docs with every probed fact in 3 documents (3 training templates) + 1 held-out-template cloze prompt; 300 late + 300 rare-random people × 5 relations; ckpt 6K and 7K (GPU when free; CPU possible for ≈ 2K docs); train-mode and eval routing; pre-inject views per pass 1-6. Gallery = the cue positions of the 3 docs + 96K distractors sampled by the T0h write criterion (the real write density). Report: same-fact R@1 / R@16 (doc → doc, doc → cloze), same-template-other-fact hits, the geometry pass rates for correct vs wrong top-1, across the 6K → 7K gap | **GO**: cloze-from-doc same-fact R@1 ≥ 0.5 at the best pass AND ≤ 5% of wrong top-1 pass the calibrated thresholds. **Weak GO** 0.35-0.5: Stage B on the v2 data only, judged on introduce-then-query. **NO-GO** < 0.35: the bank serves same-template reuse only (still measurable with the same-template introduce-then-query variant); keep Stage A |
| T0b benefit by token type in TRAIN mode (**GPU**, first GPU slot) | ckpt 7K (shared rows: the only trained memory), 16 world + 16 wiki windows, in-window forward + backward, Probe on u^row: c by token type per pass; the AUC of a linear router fitted on φ (held-out windows) for 1[z > 0]; the open rate the learned rule would give at c_read = 0; §5.2's CE test vs the true knockout | the gate can see the harm iff median c(world_other) < median c(answer) by ≥ 1 MAD. Router AUC reported (plan 22: 0.61); if < 0.6, the owner is told before Stage A (the learned gate would be near-random). CE test iff precision ≥ 0.85 at recall ≥ 0.3 |
| T0d lookup at real density | 48K-key bank from the T0a capture's write positions: the bf16 top-32 + fp32 re-rank vs the full fp32 scan; recall at density 4K / 16K / 48K | miss rate ≤ 0.5% (else top-64); recall-vs-density curve into T1 |
| **T0e calibration** | on T0a: d_same per pass → h₀,b² and κ_b (§3.3); A_w, (ω₁, μ_m, A_min) at precision ≥ 0.95 (B2); the dedupe γ at a **wrong-merge rate ≤ 2%** (§6.4), with the same-fact merge rate reported; m_ce | defaults written; if same-fact and other-fact d₁ distributions overlap so that no γ ≥ 0.05 meets 2%, dedupe is off |
| **T0h p_write** (new; gates Stage B and T1) | on training windows (train-mode routing, ckpt 7K; final-pass CE, per-pass CE and entropy are needed: GPU, or CPU on ≈ 64 windows): for each criterion σ^raw / σ^ent / σ^imp (§6.2), the fraction of rare-fact answer-cue positions (eval tags: bookkeeping only) that land in the per-step write top-W; also the share of the top-W that is Wikipedia vs world (to quantify the dominance), at W = 32 per step | **GO** p_write ≥ 0.5 for the best criterion; **weak GO** 0.3-0.5 (Stage B with the T1 bound recomputed); **NO-GO** < 0.3. The chosen criterion uses no labels / tags |
| T0g pass coverage | from the run's K log: realised f_k, P_win(k) | §2.2 within ±10% |

(T0c directed vs diffusive and T0f birth jumps belong to transport: Appendix A.)

### 16.2 T1 — the toy (CPU; extend `C:\Eitan\Python\NewBrain\speed_wip\splat\consol_toy.py`) — **waits on T0h**
- Banks with the §2.2 frequencies; drift from the salvage per-pass R@1 curves and density dependence; facts: late,
  rare-random, the rare group (k × spacing); phrasing jitter from T0a; **p_write from T0h** (not assumed); p_open ∈
  {0.5, 0.8, 1.0}.
- Policies: (1) spawn only; (2) + dedupe / merge (Stage B v1); (3) + re-encode rank (B2); (4) + re-encode CE test;
  (5) one pass-1 bank vs 9 banks.

| metric | threshold |
|---|---|
| rare group, k ≥ 2, spacing ≤ 300, policy 2 | ≥ 0.5 |
| rare-random recall, policy 2 | ≥ 0.10 |
| wrong merges | ≤ 2% of merges (T0e operating point) |
| duplicate keys per fact after dedupe | ≤ 1.2 |
| B2 (policy 3 or 4 − 2) | ≥ +0.03, else B2 stays off |
| 9 banks vs 1 bank | ≥ +0.03, else `--bm-banks 1` or 6 (owner) |

### 16.3 T2 — the GPU bench (after the VRAM / speed jobs; extend `C:\Eitan\Python\NewBrain\speed_wip\splat\bench_gpu.py`, or the VRAM study's harness `fruit-fly-sim\speed_wip\vram\vram_common.py`)
- Baseline (shared rows) / Stage A (per-pass rows + gate) / Stage B / option A (bank-only) at 16×2 and 8×4, K ∈ {2, 6, 10, 16}: ms per §15 row, peak allocated /
  reserved, launches, host syncs (must equal the baseline's 48).
- **GO**: Stage A Δtok/s ≥ −3% vs the baseline in the same harness; Stage B ≥ −3% vs Stage A; reserved ≤ 20 GB
  (cap 0.78 on) at K = 10 and 16, including an `evaluate()`.

### 16.4 T3 — runs (12K steps; owner's go for each; one GPU job at a time)

| stage | arms | GO to the next stage |
|---|---|---|
| **A: per-pass rows + the brain-decided gate** (current data, random loader, VRAM defaults; option B tables, private off, nbr on) | one run: mem_life config + `--mem-perpass 1 --mem-private 0 --mem-gate 1 --gate-budget learned`. References at the **same steps**: the no-memory control (world_v4d_12k) and the old shared-row run (world_mem_life_12k, to 7K; beyond 7K only the control exists unless the shared-row run is resumed) | vs the shared-row run and the control at the same steps: **cloze late** exact ≥ 0.27 at 7K (shared 0.280) AND ≥ 0.28 at the end (control final 0.260 + 0.02); **cloze early** (rare-random) ≥ the shared run's at equal steps (0.016 at 7K; compare at equal exposure, §0.2); **fact-token loss** (world_answer_first / rest) ≤ the shared run's + 0.02; **other-text harm**: world_other CE ≤ the ko_shared value + 0.01 and wiki ≤ control + 0.01; p_open^row at answer cues ≥ 0.8 (logged); **≤ 3% slower** than the same-harness baseline; reserved ≤ 20 GB |
| **B: + per-pass banks, v1 = spawn + dedupe** (v2 data, 32-cursor loader; needs T0a ≥ weak GO and T0h ≥ weak GO) | control (no memory) · Stage A config · Stage A + banks | **only on the v2 rare group and introduce-then-query**: rare group k ≥ 2, age ≤ 1K, exact ≥ Stage A + 0.05; introduce-then-query (Δ ≤ 1,000, different template; same-template variant reported) ≥ Stage A + 0.05; ko_bank costs ≥ 0.03 on those buckets; no-harm: late ≥ Stage A − 0.01, world_other ≤ control + 0.02; Δtok/s ≥ −3% vs Stage A; reserved ≤ 20 GB. Rare-random reported only |
| B2: + confirmation / re-encode | only if T1 says so | rare group age 1-3K +0.03 over B v1 |
| **C: banks alone vs rows** (option A; the comparison that decides whether the rows are dropped) | Stage B's data and loader: banks-only (option A, the gate on the bank store) vs Stage A (rows only) vs Stage B (both) | the rows are dropped only if banks-only ≥ Stage A on late cloze (−0.01) AND on the rare group / introduce-then-query, with world_other no worse; otherwise option B stays. The owner decides |

- Each run: auto-eval at 25 / 50 / 75 / 100%. A stage that fails stops; its knockouts are reported.

### 16.5 Unit tests (CPU)
- flags-off identity; lookup (bf16 + re-rank) = fp32 brute force on the top-4 up to ties; dedupe / merge; lazy clocks
  = eager; compact Adagrad = dense; **Probe once under checkpoint + analytic router gradient = autograd + accum
  invariance** (§4.6); the learned-budget selection (static sizes, lagged counts, cap / floor, E uniform); the ξ-only
  value init at W_o = 0; capacity exactness; no new host sync (a CUDA-sync counter on a tiny GPU-less mock / the
  TorchDispatchMode op log); resume.

### 16.6 Order to Stage A (and to Stage B)
1. **Fix the plan** (this revision).
2. **Owner sign-off on §OD** (only option A vs B remains; default B).
3. **T0b on the GPU** when the VRAM / speed jobs release it (the router target's validity, AUC).
4. **Implement Stage A** (per-pass rows + the learned gate) + the VRAM defaults (8×4 only after the same-maths
   audit); T2 for Stage A; run Stage A.
5. In parallel on CPU / the next GPU slots: **T0a + T0h gate Stage B** (and C); then T0e, T0d; T1 after T0h.
6. Stage B v1 only with T0a ≥ weak GO, T0h ≥ weak GO, T1 thresholds met, and the owner's go; Stage C after B.

---------------------------------------------------------------------------------------------------------------------

## 17. Resolution of the reviews and the owner's constraints

### 17.1 Plan-25 review (1-21): as in rev1, with these changes
- 3, 4, 10, 11, 19 (transport): moved to Appendix A with the transport itself.
- 6 (h₀): the contraction derivation kept; the rule now calibrates on same-fact distances (§3.3).
- 8 (renewal): the survival model corrected (§7.3).
- 15 (host syncs): corrected (§10.2).
- 16 (speed): the speed table replaced by a cost-vs-baseline table; speed is a separate track (§15).
- 17 (index skew): moot in v1 (brute force); OPQ kept in Appendix B.
- 1, 2, 5, 7, 9, 12, 13, 14, 18, 20, 21: unchanged in substance (see rev1 §17).

### 17.2 The owner's constraints — honest status

| # | constraint | status in rev 2 |
|---|---|---|
| 1 | addresses only from thoughts, not learned by gradient | met: keys = views; frame / h₀ fitted by statistics; no address PART is learned (keys and sub-keys get no gradient). The query side does get LM-loss gradient through the match weights (decided, §3.5). The rows' addresses are ConceptHier's μ_l + p̄, as today |
| 2 | thoughts never constrained for the memory | **every gate's p̄ input sg** (owner-confirmed: router already; bank content gate; rows' content gate, which today is NOT sg). The read VALUE enters pass k+1 as normal. **Decided**: the main LM loss also flows through the match weights into the query (§3.5) — that relaxes the original "the address path is stop-gradient" to "no gate loss reaches the thought" |
| 3 | no pointers, no recompute loop outside the step | met: the transport stage (probe forwards) removed; the router update reuses the step's own backward; the eval-time view-shift probe is an eval measurement |
| 4 | filled during training | met |
| 5 | ≥ 10k tok/s, reserved ≤ 20 GB | **not met by this plan alone**: the baseline is 8.4k; speed is the separate track. Reserved ≤ 20 GB: met with the §VRAM defaults (the eval spike was the violation) |
| 6 | the brain decides at every pass whether to read | **met, with two bounds**. Rev1 imposed the read RATE by a schedule (the brain chose only WHICH positions). Rev 2: read iff ĝ > c_read (a constant), no schedule, so the count per pass is the brain's; bounded by the hard 50% speed cap and the 5% uniform exploration (both decided) |
| 7 | each pass has its own memory | **met for k ≤ 7, from Stage A on**: per-pass row tables (option B) and per-pass banks; **table / bank 8 is shared by all k ≥ 8** (reads there happen only on the 5% long-thinking steps; `--bm-banks 15` / `--mem-perpass-tables 15` would give each its own, at 9/15 of the rows per table). Rev1's shared rows (read and written by all passes) are gone except as the reference run |
| 8 | self-reinforcement measured | §12 |

---------------------------------------------------------------------------------------------------------------------

## §OD. Owner decisions

**Decided** (owner: "1. per pass from the beginning 2. The brain 3. Yes, leave the thoughts alone. The thought can
change in the next pass (T+1)"; "question 3: yeah keep that (what you recommended)"; the rest relayed by the
coordinator):
- per-pass memory from the start (no shared-across-passes memory in Stage A);
- the brain decides the read rate: learned gate (read iff ĝ > c_read, c_read = 0 constant), no imposed budget
  schedule, the fixed budget only as a flag; safety pieces: a hard speed cap ≤ 50% of positions per pass and 5%
  uniform exploration (also the target's centring sample);
- no gate loss pushes the thought (sg on every gate's p̄ input); the memory output flows into pass T+1; the pass-T
  address = pass T's thought before its own read is injected;
- the main LM loss flows back through the read into the thought via both the value read AND the match weights (§3.5).

**Still pending Eitan** (one item):

| | decision | default | alternative |
|---|---|---|---|
| a | the per-pass memory form | **option B: per-pass row tables (the same ≈ 1M rows split, S = 170 per pass) + per-pass banks (Stage B); per-pass coarse pools (+0.28 GB); private memory off; rows dropped only if Stage C shows banks alone beat them** | option A: bank-only from the start (risks losing the rows' 0.28 late cloze until proven; depends on T0a / T0h) |

**Defaults set by the plan / coordinator (no owner decision needed unless he objects)**:

| | item | default | alternative |
|---|---|---|---|
| b | write-criterion fix (finding 4) | measured first (T0h); only the brain's own signals (entropy-normalised surprise, CE improvement across passes); no dataset labels / source tags | — |
| c | VRAM run defaults | empty_cache before evaluate() / loop report; 0.78 fraction cap; 8 × 4 if the same-maths audit passes | 16 × 2 + chunked CE |
| d | the 10k / 20k speed constraint | owned by the speed track; memory stages judged on Δ ≥ −3% | — |
| e | transport | out of this plan (Appendix A) | revisit after Stage B with an owner check against constraint 3 |
| f | table / bank 8 shared by k ≥ 8 | shared (5% of steps) | 15 tables / banks (each then 9/15 of the size) |
| g | c_read | 0 (read where the predicted gain beats the median position's) | `--gate-cost time` (κ chosen by the owner) or a fixed value |

---------------------------------------------------------------------------------------------------------------------

## 18. Flags, defaults, implementation size

**Run defaults (all stages)**: `--empty-cache-eval 1 --mem-fraction 0.78` (new), `--batch 8 --accum 4` (after the
audit; else `--batch 16 --accum 2`), eval_batch 16.

**Stage A** (per-pass rows + the brain-decided gate; option B):

    --mem-life 1 --mem-addr-learn 0 --mem-restart 100 --mem-read0 0 --mem-private 0 --mem-nbr <as run>
    --mem-perpass 1 --mem-perpass-tables 9 --mem-shared 170 --mem-coarse 64 (per pass)
    --mem-gate 1 --mem-gate-stores rows --gate-budget learned --gate-cost 0 --mem-gate-explore 0.05
    --mem-gate-cap 0.5 --mem-gate-perbank 1 --gate-target gain --mem-cgate-pbar-sg 1 --mem-match-grad 1
    --auto-eval 1 --save-report-ckpts 1

**Stage B v1** (+ the per-pass banks):

    --bm 1 --bm-banks 9 --bm-cap 48K,48K,48K,48K,48K,40K,32K,24K,32K --bm-dv 256 --bm-view pre15 --bm-fit-step 300
    --bm-write-start 300 --bm-write 32 (per optimizer step) --bm-surprise <T0h: raw|ent|imp> --bm-dedupe <T0e γ>
    --bm-h0 <T0e> --bm-fade 2.5e-4,1e-4 --bm-strength-read 0 --bm-wo perbank --bm-lookup brute --bm-rerank 32
    --bm-confirm off --mem-gate-stores rows,bank --data-order seq --data-cursors 32 (the v2 mix)

**Stage C** (option A arm): Stage B's flags with `--mem-shared 0 --mem-coarse 0 --mem-gate-stores bank` (bank-only).

**Stage B2**: `--bm-confirm rank|ce --bm-conf <T0e> --bm-qc 0.25 --bm-eta 0.5`.

**Ablations / debug**: `--mem-perpass 0` (the old shared rows), `--mem-perpass-split freq`, `--bm-banks 1|6|15`,
`--bm-wo shared`, `--bm-write-nograd 1`, `--gate-budget fixed`, `--gate-warm 0.2`, `--gate-target crank|content`,
`--gate-adapt 1`, `--gate-cost time`, `--mem-match-grad 0`, `--ko-bank`, `--ko-bank-pass k`.

**Implementation size (estimate)**

| part | lines |
|---|---|
| per-pass row tables in `memory.py` (table offsets into V / G, per-pass sub-keys, content gates and W_o, per-pass coarse pools, per-table life-cycle state, resume) | ≈ 180 |
| `bank_memory.py` (store, banks, frames, brute bf16 lookup + re-rank, read / Probe, match weights with grad, compact Adagrad, dedupe / merge / spawn / evict, clocks, state / resume, stats) | ≈ 650 |
| MemGate extension (two stores, per-bank weights, the gain target, learned budget + lagged sizes + cap / exploration, the post-backward analytic update, watchdog alerts, the fixed / crank flags) | ≈ 250 |
| `loop_brain.py` wiring (pre-inject view, Probe buffers, kept φ, the table / bank index per read, the bank read in `_hier_step`, the inject hook for the write gradient, sg on content-gate p̄, `ko_bank`) | ≈ 220 |
| `train_lm.py` (flags, the post-backward router step, after_step order, empty_cache / fraction cap, param groups, auto-eval, logging, resume) | ≈ 200 |
| `eval_world.py` (token-type split, rare buckets by k / spacing / age, introduce-then-query, knockouts incl. per-pass tables, open rates by token type, p_open at cues) | ≈ 190 |
| `generate_world.py` v2 + the 32-cursor loader + the exposure log | ≈ 260 |
| tests (§16.5 + per-pass table isolation: a read of pass k touches only table b(k)) | ≈ 480 |
| realcheck T0a / T0b / T0d / T0e / T0h scripts + the toy T1 + the bench T2 | ≈ 400 |
| **total** | **≈ 2,830** (Stage A alone ≈ 850) |

---------------------------------------------------------------------------------------------------------------------

## 19. Risks and open questions
1. **Per-pass row capacity (Stage A).** Splitting the ≈ 1M rows 9 ways leaves ≈ 116K per pass; the deep tables fill
   slowly (read on ≤ 47% of steps). Late cloze may fall below the shared run's 0.280; the Stage A criteria compare at
   the same steps. Mitigations: `--mem-perpass-split freq` (sizes ∝ f_k), fewer tables (e.g. 6, k ≥ 5 shared).
2. **Cross-template retrieval (T0a) is likely to fail.** The rows are template-keyed; if the pre-inject view at the
   cue is template-dominated too, a bank key from a training document is not found by a held-out-template cloze. The
   bank then serves same-template reuse only. T0a decides before any bank code.
3. **The harm may be a train / eval mismatch (T0b).** If train-mode c on non-answer world tokens is ≥ that on the
   answer tokens while eval shows +0.13 nats, the gate will not close them. Then compare train vs eval routing /
   capacity / K, and consider a counterfactual target on a no-grad subsample (owner check vs constraint 3).
4. **The router may not separate positions** (plan 22 phase 0: AUC 0.61). The learned gate then opens near-randomly
   (≈ up to the cap): a speed cost, not a harm fix. There is no automatic fallback (owner decision); the watchdog
   alerts and T0b warns before Stage A.
5. **The match-weight gradient** (decided) gives the self-reinforcement loop a gradient path: thoughts may drift
   toward stored keys. Monitored (§12: view shift, echo, concentration); `--mem-match-grad 0` is the ablation.
6. **p_write may be low (T0h)**: if no own-signal criterion puts rare-fact cues in the write top-k, the bank cannot
   help rare facts, whatever the retrieval.
7. **Capacity vs density**: smaller banks recall better; the 48K banks are sized for survival. T0d / T1 may push to
   fewer writes and smaller banks.
8. **The rows under the gate**: closing cue positions stops their gradient renewal, and with per-pass tables each
   table is renewed only by its own pass (§7.3); the late-cloze end-of-run check guards it.
9. **The 8 × 4 micro-batch** changes per-forward side effects (the audit) and the per-micro-batch quantiles of the gate
   (E has 205 positions per micro-batch: noisier centring).
10. **The sequential loader** changes the training distribution; the control must use it too; Stage A stays on the
    random loader for comparability with the shared-row run and the control.
11. **Open**: 9 vs 6 vs 1 banks / tables; shared vs per-bank W_o; out-of-window writes; whether late facts should also
    get bank keys; whether c_read should differ per store.


## Appendix A. Drift transport (FUTURE; not part of this plan; needs an owner check vs the no-recompute constraint)

**Why it is out**: it needs probe forwards of stashed sequences at window ends (≤ 2 forward-only passes of 16
sequences per 25 steps), in addition to the training computation. Whether that is a "recompute loop outside the
training step" (constraint 3) is the owner's call. Its expected gain is modest: at the measured R² ≈ 0.3-0.43 the
recall horizon extends ×1.2-1.32 (directed) or ×1.45-1.75 (diffusive); keeping a key valid through a 3.3× silent gap
needs R² ≥ 0.98 (directed) or ≥ 0.94 (diffusive). For rare facts with ages ~Exp(1.3K), E[p_ret] would rise by
≈ +0.05-0.08.

**The rev1 design, kept for reference** (rev1 §9, `25b_per_pass_memory_plan_rev1.md` lines 607-729 have the full text):
- **Windows** aligned with the neurogenesis cadence (every 25 steps, 5-60% of the run), so no window straddles a
  birth; births get their own 0-update before / after pairs; after the last birth, 100-step windows.
- **Probe routine**, identical at both window ends, both after opt.step: eval routing (offset 0, noise 0), dropless
  capacity, memory read in inference mode (no writes / stats), forked RNG, forward-only depth 6 (9 every 4th event),
  returning the pre-inject views; 8 sequences from the current batch per window (lives ≤ 50 steps).
- **Estimator** per bank: 1,024 k-means anchors; per window, per anchor, the robust mean displacement; a history EMA of
  the per-update rate m_j (ρ_m 0.9, support ≥ 4 pairs); James-Stein shrinkage of the window estimate toward the
  history (λ_j = s²_ε / (s²_ε + σ²_η/M²)), not toward zero; a parent fallback (64 coarse anchors).
- **Apply**: k_i += share_i · Δ̃_{j(i)} capped at 0.5 h₀ per window, share by the key time stamp τ_i; birth pairs
  applied with share 1 to keys older than the birth and kept out of the history; recall pairs only as the residual
  diagnostic.
- **Directed vs diffusive test** (T0c): the cosine of successive same-period displacements, r = ‖μ‖²/(‖μ‖² + s²_ε);
  the idiosyncratic part of a directed drift accumulates linearly, hence R² ≥ 1 − 1/G².
- **Birth jumps** (T0f): ‖Δ_birth‖ / ‖Δ_25 steps‖ and their spatial R².
- Cost: ≈ 13.5 ms/step during births, ≈ 1.7 ms after; ≈ 0.03 GB.

## Appendix B. The OPQ-style index (only for banks ≫ 48K keys)
- Rev1 §2.4: the index coordinate z = Vᵀ(x̂ − m) (PCA of the bank's keys), principal components dealt alternately to
  two 32-d halves (balanced-variance allocation, Ge et al. 2013) so that the product grid fills despite low-rank keys;
  T = 4 tables with different allocations; 128 codebooks per half, cap 8 per cell, 4 probed cells → 128 candidates;
  ranking in the raw frame; refit every 1,000 steps in 32K-slot chunks; occupancy / coverage / candidate-recall
  monitors. Not needed while every bank is ≤ 48K keys (brute force, §2.4).
