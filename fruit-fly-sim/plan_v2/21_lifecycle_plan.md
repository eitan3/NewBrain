# Memory life cycle + key restart + coarse level + neighbour writes + infrastructure: code plan (2026-10-24)
Builds on the evidence in plans 16 (collapse), 17 (corrected), 18 (alternatives), 19 (resume / spill) and 20 (drift).
No constraint on the thoughts anywhere (Eitan). Every new behaviour is behind a flag; with all new flags off the code
behaves exactly as now (bit-identical tests). No training run until the coordinator's review and Eitan's typed go.

## 0. Why (one paragraph)
While the brain learns at full learning rate, its thoughts reorganise locally:
- 72% relative change per 1,000 steps;
- only 36-42% of the nearest neighbours survive;
- so no addressing scheme keeps "same context → same rows" (best: coarse frozen keys 0.42, fine 0.21).
Today's design also makes rows written early rigid (Adagrad) while their addresses move, and it lets the keys collapse
(34% dead).

The fix is a life cycle:
- infancy: the memory is a fading, rewritable scratchpad, with keys that cannot die;
- consolidation: once the thoughts have calmed (measured, or at the learning-rate decay), it switches to permanent
  storage at frozen addresses;
- a coarse level for drift-robust addresses;
- small neighbour writes as an add-on.

## 1. Life cycle (main fix) — flag --mem-life 1
Notation: t = optimizer step; λ(t) = row decay per step; β(t) = accumulator discount per step.

### 1.1 Lazy row decay (no table-wide pass)
- Global scalar: Λ(t) = Σ_{s ≤ t} −log(1 − λ(s)) (fp64 on the host, plus an fp32 device copy per step).
- Per-row buffer: Λ_r = Λ at the row's last touch. One float per row: 4 MB shared, 1.5 MB private.
- The decayed value is V_r(t) = exp(−(Λ(t) − Λ_r)) · V_r^stored.
- Read: MemRead receives the per-bag factors f_rj = exp(−(Λ(t) − Λ_{rows[j]})) and uses the weights ω_j·f_rj. The
  forward is exact; the backward to ω is unchanged in form. The gradient written into G is
  ∂L/∂V^stored = f·(ω g), so G is already in stored units.
- Write (MemOpt, touched rows only, before the Adagrad update):
  V_r^stored ← exp(−(Λ(t) − Λ_r)) · V_r^stored; Λ_r ← Λ(t); then the usual update.
- Untouched rows are never visited. Their decay is applied on their next read or write. Mathematically this equals a
  global V ← (1 − λ)V every step.
- λ_0 = --mem-decay 1e-3 (half-life ≈ 693 steps, matched to the measured address half-life of 430-600 steps).

### 1.2 Discounted accumulator (rows stay rewritable)
- Replace S_r ← S_r + mean g² by S_r ← B_r · S_r + mean g², with B_r = exp(−(Ψ(t) − Ψ_r)).
  - Ψ(t) = Σ −log β(s) is the same lazy trick: buffer Ψ_r per row.
  - Floor: S_r ≥ 0.1·S₀.
- β_0 = --mem-acc-beta 0.99. At steady state S ≈ g²/(1 − β) = 100 g², so a write is about η/10 of a normalised step.
  It never shrinks further, so rows stay plastic.
- β = 1 gives exactly today's Adagrad (continuous at the switch).
- Prior S₀ as now (auto = 100 × median), so the first writes are the same size as today's.

### 1.3 Dead-key restart, the row part (with §2)
When a key is re-seeded, its band of rows is reset:
- V = 0, wcount = 0, S = S₀, Λ_r = Λ(t), Ψ_r = Ψ(t).
The old content belonged to contexts that no longer use that key.

### 1.4 The adaptive switch
- Signal: mem_last_stab_w, the w-weighted row stability between two consecutive probe evals (already logged; with
  --mem-probe-every 2 it is measured every 500 steps).
- Trigger at the first of:
  - (a) stab_last ≥ --mem-switch-stab 0.9 for --mem-switch-n 3 consecutive probe evals;
  - (b) step ≥ (1 − --decay) · steps, the learning-rate decay start (80%);
  - (c) a safety cap --mem-switch-max 0.9 · steps.
  - It also requires neuro.done() (no more births), as the current freeze does.
- Ramp over --mem-ramp 0.05 · steps: λ(t) and 1 − β(t) go linearly to 0; the restart interval goes from 100 to
  1,000 steps.
- At the end of the ramp: the existing freeze path (θ ← θ̄, freeze keys / W_q / norm statistics, probes record R⁰).
  From then on: no decay, Adagrad, no restarts, frozen addresses. This replaces the fixed --mem-freeze 0.6 when
  --mem-life 1.
- Logged at every eval: phase (infancy / ramp / mature), λ, β, Λ(t), switch reason, mean ‖V‖ per write-count bucket
  (1-10, 10-100, 100+), and the gate means.

## 2. Keys: no gradient, uncapped restart (main fix) — flags --mem-addr-learn 0, --mem-restart
- --mem-addr-learn 0:
  - W_q and the sub-keys get no gradient: requires_grad False from the start, and no margin loss;
  - the EMA copies equal the online ones, so the online and EMA paths are identical;
  - the norm statistics still update in infancy.
  - Evidence: plan 16 (half of the collapse came from the learned W_q and keys) and plan 17 corrected (static random
    keys 2,910 vs k-means 3,025 distinct rows per 1K, both far above today's 1,676).
  - The restart below makes the keys data-fitted over time without the rich-get-richer gradient.
  - --mem-addr-learn 1 = today's behaviour.
- --mem-restart 100: every 100 steps in infancy (1,000 during the ramp, none when mature), every key unused since the
  last event is re-seeded.
  - Uncapped: replaces --mem-balance-max / --mem-balance-dead-max when on.
  - New key = normalize(a recent query half + N(0, 0.05)), drawn from a reservoir of NRES = 1,024 recent query halves
    per codebook (today 64): L·H·2·1024·64 fp32 = 6.3 MB.
  - Distinct draws per event: no two re-seeds from the same reservoir slot.
  - Row-band reset as in §1.3.
  - No splits of busy keys (plan 20: no gain).
- Use = the pend counts since the last event (as --mem-balance-keep does now).

## 3. Coarse level (main fix) — flags --mem-coarse 64, --mem-coarse-heads 1
- A second product-key pool per concept level, with S_c = 64 per half: 4,096 rows per head per level;
  4 levels × 1 head × 4,096 × 256 fp32 = 16 MB (+ G 16 MB).
- Its own fixed translator W_q^c (random, no gradient under --mem-addr-learn 0), its own sub-keys (restart as in §2),
  and the same life cycle (decay / accumulator / reset).
- Read: top-6 of 64², weights ω^c as now. Gate g^c per level, its own W_o^c (zero-init). M adds Σ_l W_o^c,l g^c r^c.
- Probes: stability is logged per pool (fine / coarse).
- Implementation: generalise Memory's shared pool into a list of pools [(S, H, offset)]; select / online / read run per
  pool, one MemRead per pool.

## 4. Neighbour writes (add-on, not the main fix) — flag --mem-nbr 12, --mem-nbr-w 0.15
- select returns the top-12 pairs (pk_topk k = 12 from the 64 candidates; m = 8 already gives 64).
- Reads stay on the top-6, weights ω.
- Writes: the extra rows ranks 7..12 receive G += α · ω^n_j · g, with ω^n = softmax over ranks 7..12 of σ/τ and
  α = 0.15. The extra mass is α relative to a normal write.
- They go through MemRead's backward only (no forward cost): about +2-5% step time.
- They count as writes for wcount and the accumulator. Not on the coarse pool by default.

## 5. Infrastructure (plan 19)
- free_cache() right after probes.freeze() and after any probe eval when reserved > 0.85 × total.
- Spill guard: act on the FIRST eval with reserved > --spill-gb AND tok/s down > --spill-drop.
- Resume: --resume <last.pt>; a full atomic checkpoint every --ckpt-every steps:
  - model (incl. the memory buffers, Λ / Ψ, the life-cycle phase, probe_state);
  - BrainOpt state + t;
  - Neurogenesis and the grown cfg;
  - trainer counters;
  - all RNGs (python, numpy sampler, torch CPU / CUDA, mem.gen / gen_margin);
  - the spill-guard state.
- Report-point checkpoints ckpt_<step>.pt at 25 / 50 / 75 / 100% (for eval_world).

## 6. Flags (defaults = today's behaviour)
- --mem-life 0
- --mem-decay 1e-3
- --mem-acc-beta 0.99
- --mem-switch-stab 0.9
- --mem-switch-n 3
- --mem-switch-max 0.9
- --mem-ramp 0.05
- --mem-addr-learn 1
- --mem-restart 0
- --mem-coarse 0
- --mem-coarse-heads 1
- --mem-nbr 0
- --mem-nbr-w 0.15
- --resume ""
- --save-report-ckpts 0
Run settings: --mem-life 1 --mem-addr-learn 0 --mem-restart 100 --mem-coarse 64 --mem-nbr 12 (+ the existing
--mem-read0 0 --mem-readdr-every 2 --mem-readdr-start 3 --mem-probe-every 2; the margin is irrelevant with
addr-learn 0).

## 7. Tests (CPU)
1. All flags off: bit-identical to the current code over 40 steps (test_01 style) and end to end (test_01b).
2. Lazy decay = global decay: a table with random reads / writes over 200 steps under a λ schedule; the lazy reads /
   values equal an eager V ← (1 − λ)V reference to 1e-6.
3. The discounted accumulator: β = 1 equals Adagrad bit for bit; β < 1 matches the eager formula; the floor holds.
4. The switch: synthetic stab sequences trigger (a) / (b) / (c) at the right step; the ramp is linear; the freeze
   happens at the ramp end only if neuro.done().
5. addr-learn 0: W_q / keys have no grad; online == EMA addressing; the norm stats still update until the freeze.
6. Restart: the dead set is re-seeded uncapped from distinct reservoir slots; the band reset (V, wcount, S, Λ, Ψ);
   deterministic with mem.gen; none when mature.
7. Coarse pool: rows / offsets do not overlap the fine pool; the zero-init W_o^c gives output = without coarse;
   probes per pool.
8. Neighbour writes: the forward is unchanged; G on ranks 7-12 = α·softmax-weights·g; the ranks 1-6 gradients are
   unchanged.
9. Resume: 40 straight steps = 20 + save + resume + 20 (bit-identical on CPU, with growth, the switch, a restart event
   and a probe eval crossing the cut).
10. Spill guard / free_cache paths callable on CPU (no-op).
11. The 158 existing tests pass.

## 8. Runs (after review + Eitan's go)
- A. Wikipedia 12K: the package above, vs the existing control wiki_v4d_12k.
- B. World-facts mix 12K (X:\Datasets\wiki_world_mix): the control (v4d) and the memory package, with
  --save-report-ckpts 1. eval_world.py on ckpt_25/50/75/100 → cloze / ranking accuracy and the memory-knockout deltas.
  The key question: do facts learned before the switch survive the switch (early-retention group) and do post-switch
  facts stick?
- Order: B first (it answers "does the memory store facts"), then A, or as Eitan prefers. One GPU job at a time.
