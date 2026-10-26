# 25c. Blob-only per-pass memory: per-pass thought-keyed banks + a brain-decided read gate, no row table (PLAN ONLY) — 2026-10-24, revision 3

Revision 3 of plan 25b. Revision 2 (`25b_per_pass_memory_plan.md`) and revision 1 (`25b_per_pass_memory_plan_rev1.md`)
are kept unchanged; where this file says "rev2 §x" it points there for derivations that did not change. Inputs: rev2
(changelog, review-fix table, §2-§8, §VRAM, §16, Appendices A/B), the salvage results
`C:\Eitan\Python\NewBrain\speed_wip\splat\realcheck\salvage\results.md` (run `world_mem_life_12k`, checkpoints 6K → 7K),
the VRAM study `fruit-fly-sim\speed_wip\vram\vram_study.md`, and the code `memory.py`, `loop_brain.py`, `train_lm.py`.

Nothing was trained, no code was edited, the GPU was not used. Only this file was written (plus a scratch calculator
for the §7 / §2 numbers, CPU only, outside the project).

---------------------------------------------------------------------------------------------------------------------

## Changelog: revision 2 → revision 3 (owner decision: option A, bank-only from the first run)

**The owner chose option A.** The memory of the new brain is ONLY per-pass splat / blob banks, from the first run.
There is no row or hash table anywhere: no shared rows, no per-pass rows, no coarse pool, no private expert memory.
ConceptHier stays (it is not memory: it produces the concept heads, the concept loss and two of the gate's features).
The old run `world_mem_life_12k` (late cloze 0.280 at 7K, 0.182 with the shared memory knocked out; early 0.016) and
the no-memory control `world_v4d_12k` (late 0.260 final; early 0.023 final) are only scores to beat.

| # | change | why | where |
|---|---|---|---|
| 1 | **Stage A = per-pass banks + the brain-decided gate.** Rev2's Stage A (per-pass rows), Stage B (banks added) and Stage C (banks vs rows) collapse into one stage. Option B, the per-pass row tables, the coarse pools and `--mem-private` are gone | owner decision (option A) | §1, §10, §16 |
| 2 | **Renewal is in Stage A, not B2.** Confirmation by the centred benefit (in window) and a key move on confirmed recall; plus an out-of-window **geometric** renewal (strict thresholds from T0e) because the in-window path alone cannot reach the survival bar even at perfect recall (§7.4) | the banks must now carry the recurring (late) facts too; rev2 §7.3 said a bank needs p_ren ≥ 0.43 for them | §7 |
| 3 | **Survival model extended**: rev2's S(T) = exp(−λT e^{−λH}) (kept, now per pass and for the TRAINED slot), plus re-acquisition by a re-write after a loss (availability μ/(μ+ν)) and the fill time of 20.4K late facts per bank under the write budget | the "lost" state is not absorbing for a key, but it is for a trained value | §7.4-7.6 |
| 4 | **Value-gradient training on every read**: analysed (§7.2). It keeps a found slot's value trained (as the rows' values were) but never moves a key, so it does not replace renewal; the rows' "automatic renewal" came from fixed random addresses, which the bank does not have | requested | §7.2 |
| 5 | **15 banks, one per read k = 0..14** (rev2 shared bank 8 between all k ≥ 8). The deep banks are small (2-20K), cost nothing measurable, and make constraint 7 hold for every pass | constraint 7; the rows' VRAM is freed | §2.2 |
| 6 | **Bank sizes re-derived** from (unconfirmed life ≈ the drift horizon) + (a recurring-fact reserve ≈ 26K for banks 0-5) under a 64K brute-force ceiling: 64K × 5, 56K, 40K, 32K, 20K, 4K × 4, 2K × 2 = **488K slots, 0.73 GB**. VRAM is not the binding limit any more; recall-vs-density and lookup time are | the ≈ 3.5 GB of row tables + grads are freed | §2.2-2.3, §VRAM |
| 7 | **Lookup = bf16 brute force with a block-max pre-selection** (blocks of 16 keys; top-8 blocks by block max; fp32 re-rank of the 128 candidates; exact top-4 up to bf16 ties). Removes rev2's topk-over-48K bottleneck. Appendix B (OPQ) only if a bank must exceed 64K keys or the lookup exceeds 2% of the step | cost at the new sizes | §2.4 |
| 8 | **Match features in the router** (A, d₁/h₀², log d₂/d₁, the winner's maturity and age): the no-grad lookup runs on ALL positions, the read (value gather, inject, backward) only on the open ones. Flag `--gate-match-feats 0` restores lookup-on-open-only | "is there a matching memory here?" is the strongest predictor of a read's gain; the lookup is cheap | §4.2 |
| 9 | **The gate no longer depends on `Memory`**: today `_gate_phi` takes the p̄ feature from the row translator (`a_hat[0, :, mem.d:]`) and `--mem-gate` asserts `--mem-shared > 0`; rev3 standardises the view by the bank frame (EMA statistics before the frame fit) | no `Memory` object in option A | §4.2, §10 |
| 10 | **Zombie guard**: a slot with no confirmation for 3,000 steps loses its maturity protection at eviction time | with renewal in Stage A, mature slots fade at ρ = 4^{−m} λ_α and would otherwise never be evicted after their fact moved to a new slot | §6.6 |
| 11 | **Merge key weight floored** at ¼ (max(1/(n+2), ¼)) so a merged key keeps tracking drift; the value keeps the 1/(n+2) average | rev2's 1/(n+2) with n ≤ 8 made a key lag by up to 10 writes | §6.4 |
| 12 | **Micro-batch 16 × 2 by default** (the VRAM-study default 8 × 4 becomes the fallback): projected peak ≈ 17.4 GB allocated / ≈ 17.8-18.4 GB reserved at 16 × 2 once the row tables are gone, and 8 × 4 alone costs 4.5% speed, more than the 3% speed criterion vs the old run (which ran 16 × 2) | §VRAM arithmetic | §VRAM |
| 13 | **GO criteria restated vs the old run at the same steps** (7K) and vs the control at the end, with statistical bases (early: ≥ 0.035 at 7K, two-proportion z ≈ 2.7 on n = 1,000; paired McNemar on the same items) | requested | §16.2 |
| 14 | **Tests before any run**: T0a (paraphrase: the same fact in other documents / wording finds the same blob; dedicated capture, ≥ 3 docs per fact, train-mode routing), T0b (gate separability on the old run's rows, train-mode c by token type, router AUC), T0h (do rare- AND late-fact cues reach the write top-W), T0e (h₀ calibration and wrong-merge rate), T0d (lookup at real density); CPU / GPU marked | requested | §15 |
| 15 | **Data v2** (32-cursor sequential loader, scheduled rare group, introduce-then-query) is an **option** (Stage A2), not part of Stage A: it changes the training distribution, so the control must be rerun on it, and the old run is then no longer a same-data reference | comparability | §12 |
| 16 | Drift transport stays in Appendix A (future; owner check vs constraint 3) | unchanged | App. A |

**Carried over from rev2 unchanged in substance** (each re-stated where it applies to the blob part): the write
selection with only the brain's own signals and per-step budgets (rev2 finding 4, §6.2); dedupe / merge with a
wrong-merge bound ≤ 2% (§6.4); eviction never takes a slot touched this step (§6.5); strength fade λ_α = 2.5e-4,
strength used for eviction only (§3.2, §8); h₀ from same-fact distances + tracking + floor (§3.3); VRAM spike avoidance
(empty_cache before `evaluate()` and the loop report, the 0.78 fraction cap, eval_batch 16, chunked scoring; §VRAM);
the post-backward analytic router update with Probe idempotence under checkpointing and its test (§4.5-4.6); centring
on the uniform exploration sample (§4.3); the value-init ε guard / ξ-only fallback while W_o ≈ 0 (§6.3); sg on every
gate's p̄ input; the main LM loss flows through the read (value AND match weights) (§3.5); no automatic fallback
(alerts only); the fixed budget as a debug flag only.

**Dropped from rev2** (row-specific): per-pass row tables (rev2 §11.4), per-pass coarse pools, `--mem-private 0` as a
change (there is no `Memory` at all), `--mem-nbr`, the rows' content gate sg, p_open^row at cues, the
bit-identity flag sets that involve rows, rev2's Stage B / C structure, Stage B2 (merged into Stage A).

---------------------------------------------------------------------------------------------------------------------

**The owner's hard constraints** (status in §17):
1. Addresses come only from the model's thoughts: no token keys, and no address part is learned by gradient.
2. The thoughts are never constrained for the memory's sake: no GATE loss reaches the thought (decided: the main LM
   loss may flow through the read, value and match weights).
3. No dataset pointers, and no recompute loop outside the training step.
4. The memory is filled during training.
5. Speed ≥ 10k tok/s (aim 20k) on one 24 GB GPU; peak **reserved** VRAM ≤ 20 GB.
6. The brain decides itself, at every pass, whether to read.
7. Each pass has its own memory, keyed by its own thought taken BEFORE that pass's memory inject.
8. Self-reinforcement (memory → thought → key → memory) is measured, not forbidden.

Structure: §0 evidence; §1 architecture; §2 the bank store (sizes, density, lookup); §3 the view and the read;
§4 the gate; §5 benefit signals; §6 writes; §7 renewal and survival; §8 fading and clocks; §9 after_step and host
syncs; §10 what leaves the brain (code map); §11 self-reinforcement; §12 data; §13 metrics; §VRAM; §14 cost;
§15 tests; §16 the run and GO criteria; §17 constraints; §OD owner decisions; §18 flags and size; §19 risks;
Appendix A transport; Appendix B the OPQ index.

---------------------------------------------------------------------------------------------------------------------

## TL;DR

1. **The memory is 15 per-pass banks** (read k = after core pass k, k = 0..14, bank b = k). Key and query = the pass-k
   **pre-inject view** v_k = mean of the slot-normalised workspace slots 0-14 (slot 15 carries the memory's output),
   in a frame fixed per bank. No row table anywhere; ConceptHier stays.
2. **Sizes**: 64K slots for banks 0-4, then 56K, 40K, 32K, 20K, 4K, 4K, 4K, 4K, 2K, 2K: **488K slots, 0.73 GB**. Each
   is (an unconfirmed-slot life ≈ 2.3K steps ≈ the drift horizon at W = 32 writes per step) + (≈ 26K reserve for the
   recurring facts in banks 0-5). Salvage: R@1 0.77 at 4K keys vs 0.57 at 32K (1K gap); extrapolated ≈ 0.50 at 64K
   (world text ≈ 0.18): density, not VRAM, is the limit. T0d / T1 may move banks 0-4 to 48K or W to 16.
3. **Lookup**: bf16 scores in 1,024-query chunks, a block max over blocks of 16 keys, top-8 blocks, fp32 re-rank of
   128 candidates → exact top-4 (up to bf16 ties). ≈ 30-45 ms/step with the lookup on all positions (≈ 1.5-2.3% of a
   1.95 s step). Appendix B only above 64K keys per bank or above 2% of the step.
4. **Gate (brain-decided)**: per pass, a linear router on 99 detached features (rev2's 93 with the view standardised
   by the bank frame, + 5 match features) predicts the centred gain ĝ (nats); read iff ĝ > c_read = 0; hard cap 50%
   of positions per pass; 5% uniform exploration (also the centring sample); no budget schedule; the fixed budget is a
   debug flag. The target z = N_tok (c − q₅₀^E c), c = −⟨∂L/∂u, u⟩ from an idempotent Probe; the router is updated
   analytically after backward. Gate losses never reach the thought (sg); the LM loss flows through the read value
   and the match weights w, A into the query; indices, keys, h₀ get no gradient.
5. **Writes**: per in-window read, W = 32 per optimizer step per bank, chosen by a surprise criterion from the brain's
   own signals only (T0h picks raw / entropy-normalised / not-reduced-by-thinking), eligible only where no confident
   read happened; value = normalised gradient write (ε-guarded, ξ-only while W_o ≈ 0) then row-Adagrad; dedupe-merge at
   d₁ < γh₀² (γ from T0e at ≤ 2% wrong merges); eviction by lazy strength, never a slot touched this step, zombie
   guard after 3,000 steps without confirmation.
6. **Renewal in Stage A**: in-window confirmation by the centred benefit (top 25% of the bank's open reads, z > 0,
   geometry thresholds) and out-of-window geometric confirmation (strict, precision ≥ 0.95 from T0e) move the key
   toward the confirming queries (η = 0.5, capped at h₀) and raise strength / maturity. Survival of a late fact's
   TRAINED slot (S = exp(−λT e^{−λH}), λ = p_ren/Δ) needs p_ren ≥ 0.43 per pass: the best pass reaches ≈ 0.44 at a
   cross-document hit rate 0.8 with geometric renewal (0.23 without it); union over passes 0-4 ≈ 0.9-0.999. At a hit
   rate of 0.5 (a weak T0a) the best pass falls to S ≈ 0.69. Value-gradient training on every read keeps values
   trained but never moves keys.
7. **GO (Stage A, current data, random loader, 16 × 2)**, same steps as the old run: late cloze ≥ 0.27 at 7K and
   ≥ 0.28 at the end (control final 0.260 + 0.02); early cloze ≥ 0.035 at 7K (old 0.016; z ≈ 2.7, n = 1,000) and
   ≥ 0.045 at the end (control 0.023); harm on other world tokens ≤ +0.04 nats (old +0.133) and the memory net
   non-harmful on world val loss; wiki |Δ| ≤ 0.01; speed ≥ 0.97 × the old run in the same harness; reserved ≤ 20 GB
   including `evaluate()`.
8. **Before any run**: T0a (cross-document same-fact R@1 at real density; likely hard), T0h (p_write for late and rare
   cues), T0e (h₀, γ, renewal thresholds), T0d (lookup) on CPU from captures; T0b and T2 (+ a 1K-step pilot) on the GPU
   when the speed study releases it.
9. **Biggest risk**: there is no fallback memory. If the banks cannot carry the recurring facts (T0a weak, renewal
   short of the bar, slow fill), the late cloze falls toward the no-memory curve (control ≈ 0.11 at 7K by
   interpolation of 0.079 at 6K and 0.182 at 9K), well below the 0.27 bar.
10. **Size**: ≈ 2,500 lines for Stage A (memory code ≈ 1,430, tests and checks ≈ 920, eval ≈ 150; + ≈ 260 for the optional data v2).

---------------------------------------------------------------------------------------------------------------------

## 0. What the evidence implies (bank-only reading of rev2 §0)

| finding (salvage, 7K) | number | consequence for a bank-only memory |
|---|---|---|
| the rows carried the recurring facts | late exact 0.280; ko_shared 0.182; control final 0.260 (0.079 at 6K, 0.182 at 9K) | the banks must now carry them: renewal is first-order (§7), and the fill rate of ≈ 20.4K late facts per bank under the write budget matters (§7.6) |
| rare facts | early 0.016 vs control 0.013 (≈ 5.5 exposures by 7K, mean spacing ≈ 1.3K steps) | what a write-at-surprise store is best at; the GO asks for a clear gain (§16.2) |
| reads hurt non-answer tokens | answer −0.29 / −0.20 nats; other world +0.133 on 83% of world tokens; wiki ±0.01 | the per-position gate (§4); the bank's distance factor A also mutes reads with no near key |
| rows are template-keyed | same template other person 0.24; same fact other template 0.28; same person other relation 0.045 ≈ baseline 0.039 | the bank's view may be template-dominated too: T0a is the first gate, and a template-keyed view also multiplies the keys per fact (capacity, §2.2) |
| self-recall over a 1K gap | R@1 / R@16 0.774 / 0.983 at 4K; 0.568 / 0.882 at 32K; world 0.525 (4K), 0.350 (16K); wiki 0.902 / 0.790 | density costs ≈ 0.07 R@1 per doubling (world ≈ 0.09): keep banks ≤ 64K and renew keys well inside the horizon (§2.3) |
| per-pass banks | same-pass 0.777 → 0.719 (passes 1-6, pre-inject view); pass-1 bank queried from pass 6: 0.401 | a bank per pass (§2) |
| pre-inject views | slot 15 excluded: +0.00…+0.01 | key = slots 0-14 mean |
| routing rotation / noise | 0.3% effect | no pinning |
| drift | median drift / 1-NN gap 1.22-1.31 | h₀ of the order of the drift, calibrated on same-fact pairs (§3.3) |
| transport | 0.774 → 0.884 (per point), 0.854 (4K anchors) | Appendix A (future) |

Other rev2 findings that still hold: the trainer samples uniform random windows (`train_lm.py: Data.batch`,
`gen.randint(0, len(d) − seq − 1, B)`), so "early" facts are a RARE group (≈ 5.5 exposures by 7K, ≈ 9.5 by 12.2K) and
eval_world's exposure buckets mislabel them (rev2 §0.2); c = −⟨∂L/∂M, M⟩ is ≈ zero-sum, so only centred values are
used (rev2 §5.1); the content gate's 0.999 mean is a symptom (Adam is scale-invariant), so no logit L2 (rev2 §0.3);
the baseline step has 48 host syncs (≈ 0.2 s) and is GPU-bound on elementwise kernels (rev2 §0.4).

---------------------------------------------------------------------------------------------------------------------

## 1. Architecture (read k = after core pass k, feeding pass k+1; k = 0..K−2)

```
 core pass k ──► x_k, s_k
   │
   ├─ (training) read-out ce_k, lp_k               (already computed before the read under the gate: plan 22)
   ├─ ConceptHier(n_s(s_k)) → μ_l, ν_l             (unchanged: concept loss, gate features nu / dmu; NOT an address)
   ├─ view v_k = mean_{i<15} n_s(s_k)_i  ── x̂ = (v_k − μ_k)/σ_k (bank frame) ── sg for the lookup ──┐
   ├─ LOOKUP bank k on ALL positions (no_grad): bf16 scores → block max → top-8 blocks ──────────────┤
   │      → (open positions) fp32 re-rank → top-4 (j, d);  (all positions) bf16 match features       │
   ├─ router φ_k (99-d, detached: rev2 features + match features) → ĝ_k                           │
   │      open O_k = TopUpTo_{n_k}{ĝ > c_read} ∪ E_k (5% uniform), cap ⌊0.5 N⌋                        │
   ├─ READ on O_k (inside the checkpointed block): w, A from x̂ WITH grad and the detached keys      ◄┘
   │      r = A Σ_j w_j f_j v_j ─► content gate g (p̄ input sg) ─► u = g r ─Probe─► M_k = W_o^(k) u
   └─ M_k (scattered into zeros at closed positions) ─► inject_{k+1} += M_k ; slot 15 += W_ms M_k

 after each micro-batch's backward: Probe c → z → analytic router gradient; bank value grads accumulated (compact)
 after opt.step: router AdamW; bank Adagrad; confirmation → key move; merge / spawn / evict; key copies; EMAs
```

- **Gradient reaches**: the bank values (row-Adagrad), W_o^(k), W_ms, the content-gate weights, the query x̂ (hence
  the thought) through w and A (decided, rev2 §3.5), and the router (post-backward analytic update).
- **No gradient reaches**: the keys, the frames, h₀, strengths, clocks, the slot selection (indices), and the thought
  through any gate's p̄ input (sg). Keys move only by rules (spawn, merge, confirmed recall).

---------------------------------------------------------------------------------------------------------------------

## 2. The bank store

### 2.1 Layout (one store, 15 banks as contiguous slot ranges; + one dummy slot for ignored writes)

| field | dtype | bytes | set by | meaning |
|---|---|---|---|---|
| k_i (64) | fp32 | 256 | rules (§6, §7) | key in the bank's fixed frame |
| k̃_i (64) | bf16 | 128 | refreshed when k_i changes | scoring copy, laid out in blocks of 16 |
| ‖k_i‖² | fp32 | 4 | with k_i | distance |
| v_i (256) | fp32 | 1,024 | write init + row-Adagrad | value |
| S_i | fp32 | 4 | Adagrad | accumulator |
| log α_i, Λ^α_i | fp32 | 8 | rules | strength (eviction order) and its clock |
| Λ^v_i, Ψ_i | fp32 | 8 | Adagrad touch | value-fade clock, accumulator discount clock |
| τ_i | int32 | 4 | rules | key time stamp (age) |
| t^conf_i | int32 | 4 | confirmation | last confirmation step (zombie guard, §6.6) |
| t^touch_i | int32 | 4 | any change / Adagrad touch | eviction protection |
| m_i, n_i | int8 | 2 | rules | maturity 0..3 / merge count ≤ 8 |
| u_i, b̄_i, d_i | int32, fp32, int32 | 12 | rules | use count, running benefit rank, writing-document hash (echo metric only) |
| **total** | | **≈ 1,460 B / slot** | | |

### 2.2 Bank count, write rates and capacities (all budgets per OPTIMIZER step)
- K ~ U{2..10} w.p. 0.95, U{12..16} w.p. 0.05 (as today). Read k exists iff K ≥ k+2 (f_k); it is in the bptt-6
  gradient window iff K−6 ≤ k ≤ K−2 (P_win(k)); values as rev2 §2.2:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f_k | 1.00 | 0.89 | 0.79 | 0.68 | 0.58 | 0.47 | 0.37 | 0.26 | 0.16 | 0.05 | 0.05 | 0.04 | 0.03 | 0.02 | 0.01 |
| P_win(k) | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 | 0.42 | 0.33 | 0.23 | 0.14 | 0.04 | 0.05 | 0.04 | 0.03 | 0.02 | 0.01 |
| reserve R_k (recurring) | 26K | 26K | 26K | 26K | 26K | 26K | 16K | 16K | 8K | 0 | 0 | 0 | 0 | 0 | 0 |
| **N_k** | **64K** | **64K** | **64K** | **64K** | **64K** | **56K** | **40K** | **32K** | **20K** | **4K** | **4K** | **4K** | **4K** | **2K** | **2K** |
| unconfirmed life (steps) | 2.29K | 2.29K | 2.29K | 2.29K | 2.29K | 2.29K | 2.33K | 2.23K | 2.74K | 3.2K | 2.6K | 3.2K | 4.3K | 3.2K | 6.4K |

- **15 banks** (`--bm-banks 15`, default): every read k has its own bank, so constraint 7 holds for all passes (rev2's
  bank 8 shared k ≥ 8). Banks 9-14 are written only on the 5% long-thinking steps; they are tiny and cost nothing
  measurable. `--bm-banks 9` (rev2's layout) is an ablation.
- **Writes**: W = 32 per optimizer step per in-window read (16 per micro-batch at 16 × 2), i.e. W · P_win(k) ≈ 17
  per step for banks 0-4.
- **The reserve R_k** is not a partition; it is the room the eviction order (§6.5) leaves to confirmed slots. Its size:
  the world has 4,000 late people × 5 facts + ≈ 400 geo facts ≈ **20.4K recurring facts**; at a dedupe factor of
  ≈ 1.25 keys per fact (T0e / T1 measure it) that is ≈ 26K per bank where renewal can keep them (passes 0-5); the
  deep banks cannot keep them (P_win ≤ 0.33, §7.4) and get a smaller reserve.
  - If the view is template-keyed (T0a fails), a fact needs up to one key per training template (10): ≈ 200K keys per
    bank, which does not fit at any density that still recalls. This is the capacity face of the T0a risk.
- **Unconfirmed life** = (N_k − R_k) / (W · P_win(k)) ≈ 2.3K steps ≈ the drift horizon H (R@1 at 1K ≈ 0.77, ≈ 0.45 at
  2K, ≈ 0.15 at 3K, rev2 §7.3): a longer life buys little recall for an unrenewed key and costs density.

### 2.3 Recall vs density: why 64K and not more (the rows' freed VRAM is not the limit)
Salvage (pass 1, 1K gap, untransported): R@1 0.774 at 4K keys, 0.568 at 32K (≈ −0.069 per doubling); world text 0.525
at 4K, 0.350 at 16K (≈ −0.0875 per doubling). Extrapolated, and the unconfirmed life at R = 26K:

| N (bank 0) | R@1 1K gap, all text / world (extrapolated) | unconfirmed life, W = 32 | W = 16 | VRAM (15 banks scaled alike) |
|---|---|---|---|---|
| 32K | 0.57 / 0.26 | 0.36K (too short) | 0.72K | ≈ 0.4 GB |
| 48K | 0.53 / 0.21 | 1.33K | 2.66K | ≈ 0.55 GB |
| **64K (default)** | **0.50 / 0.18** | **2.29K** | 4.6K | **0.73 GB** |
| 96K | 0.46 / 0.12 | 4.2K | 8.5K | ≈ 1.1 GB (above the brute-force ceiling) |

- The freed ≈ 3.5 GB could hold ≈ 2.4M slots, but recall would collapse: VRAM is not the binding constraint.
- The extrapolation is pessimistic in one way and optimistic in another: the salvage gallery is ALL val positions (a
  key's neighbours in the same document are near-duplicates; the bank holds sparse surprise positions, so fewer
  near competitors), but the world part is templated and self-similar. T0d measures recall at the real write density.
- **Renewed keys are young**: a late fact renewed every ≈ Δ/p_ren ≈ 250-500 steps is queried at ages ≪ 1K, where
  the drift is a fraction of the 1K drift; density hurts mostly the rare facts (age ~ Exp(1.3K)).
- Decision rule (T0d + T1): if recall at 64K and age 1K is < 0.35 on world text, move banks 0-4 to 48K and W to 24
  (life ≈ 2.6K at R = 26K).

### 2.4 The lookup: bf16 brute force with a block-max pre-selection (exact top-4 up to bf16 ties)
For bank k (N_k keys in blocks of 16 contiguous slots) and query chunk Q (1,024 positions):
1. scores s = 2 x̂·k̃_i − ‖k_i‖² in bf16 (one GEMM, 1,024 × N_k: 134 MB at 64K, freed per chunk);
2. block max over each 16-key block → (1,024, N_k/16); topk 8 blocks (a topk over 4K columns, not 64K);
3. open positions: gather the 8 × 16 = 128 candidate keys (fp32), exact d, top-4 (j₁..j₄, d₁ ≤ … ≤ d₄);
   all positions: the match features from the bf16 values (best and second-best key of the top blocks).
- **Exactness**: a block whose max is ≥ the 4th-best score must contain one of the top-4 keys, so at most 4 blocks
  have that property and they rank among the top-4 blocks; 8 blocks leave a margin for bf16 rounding. Monitored: on
  1K sampled queries every 250 steps, the fraction whose fp32 top-4 differs from a full fp32 scan (target ≤ 0.5%;
  else top-12 blocks).
- **Cost** (per optimizer step, 16 × 2): Σ_k f_k N_k = 305.5K keys per position-read; with the lookup on all 8,192
  positions of 2 micro-batches: 5.0 × 10⁹ scores. GEMM 0.64 TFLOP (≈ 5-10 ms); materialising and block-reducing the bf16
  scores ≈ 20 GB of traffic (≈ 20-30 ms); block topk and the fp32 re-rank of ≤ 50K open queries (1.6 GB gathered,
  ≈ 3 ms). **≈ 30-45 ms per step, ≈ 1.5-2.3% of the 1.95 s step**; ≈ 15-25 ms with `--gate-match-feats 0` (lookup on
  open positions only). A fused score + block-max kernel (Triton) would remove the 20 GB of traffic: the speed track's
  job.
- The lookup is no_grad; its winners are cached for the checkpoint recompute (as the gate's open set is today).
- **Appendix B (OPQ index) only if** a bank must exceed 64K keys (e.g. T0a shows a template-keyed view and the owner
  still wants more keys per fact) or T2 measures the lookup above 2% of the step.

---------------------------------------------------------------------------------------------------------------------

## 3. The view and the read

### 3.1 The view (pre-inject; the address of pass k)
- v_{k,t} = (1/15) Σ_{i=0}^{14} n_s(s_k)_{t,i}, computed in the checkpointed block next to today's
  `pbar = ns.mean(-2)` (`loop_brain.py: _hier_step`). s_k is the workspace after core pass k, before read k's slot
  write and inject; earlier reads (k′ < k) still influence it (§11).
- **Frame** (fixed per bank, no gradient): x̂ = (v − μ_k)/σ_k, μ_k ∈ R^64, σ_k a scalar RMS, fitted once on 64K
  positions of pass k at `--bm-fit-step 300` (bank k ≥ 9: on the first 64K positions seen after step 300). Never
  refitted. Writes start at step 300; before that the banks are empty and the router's match features are zero with a
  "bank empty" flag.

### 3.2 The read kernel (bank k, open positions O_k)

      w_j = softmax_j( −(d_j/d₁ − 1)/τ_w ),  τ_w = 0.1          (+ ã_j only under --bm-strength-read 1)
      A   = exp(−d₁ / (2 h₀,k²))                                 (the distance factor: no near key → no read)
      f_j = exp(−λ_v ρ_j (t − Λ^v_j))                            (lazy value fade, §8)
      r   = A Σ_j w_j f_j v_j

- Strength is used for eviction only (default, rev2 §3.2): with it in the kernel, a fresh rare-fact slot would be
  muted ×0.4 exactly at its next exposure.
- **Gradient through the match weights (decided)**: w and A are recomputed WITH grad from x̂ and the gathered,
  detached keys inside the checkpointed block (a few elementwise ops on (|O|, 4)); indices stay discrete. Ablation
  `--mem-match-grad 0` (w, A from sg(x̂)).

### 3.3 h₀: calibrated on same-fact distances, tracked, floored (no width collapse; rev2 §3.3 unchanged)
1. **Calibration (T0e)**: d_same,k = squared distance between a fact's key in one document and its nearest same-fact
   cue in another document (T0a capture, within a checkpoint and across 6K → 7K). h₀,k² = q₅₀(d_same,k) / (2 ln 2)
   (a median same-fact read has A = 0.5); κ_k = h₀,k² / q₅₀(d₁ over all open reads of pass k) on the same capture.
2. **Tracking**: h₀,k² ← 0.99 h₀,k² + 0.01 κ_k q₅₀(d₁ over ALL open reads of bank k in the step). The population is
   not selected by A, so the EMA does not contract (rev1's derivation: an EMA over reads selected by d₁ ≤ c₀h²
   contracts to 0 whenever c₀ < 2^{2/d_eff}).
3. **Floor**: h₀,k² ≥ ½ q₅₀(NN distance² among 4,096 sampled keys of bank k), every 250 steps.
- Before T0e: h₀² = the salvage drift scale (median drift² ≈ 1.72 × NN² → h₀² = 1.72 NN² / (2 ln 2) ≈ 1.24 NN²).

### 3.4 Content gate and output (per bank)

      φ^c = [sg(p̄_k) (64); r (256); A; log(d₂/d₁); onehot(m_{j₁}) (4); log(1 + age_{j₁})]     (327)
      g = σ(w_cᵀ φ^c + b_c),  b_c = −2;   u = g r;   M_k = W_o^(k) Probe(u)     (W_o^(k) ∈ R^{384×256}, zero init)

- p̄ enters under sg (no gate → thought gradient); no logit L2 (rev2 §0.3). Per-bank W_o: 15 × 98K = 1.47M
  parameters; `--bm-wo shared` (+ a 16-d pass embedding) is the ablation.
- M_k goes where the row output goes today: `inject` of pass k+1 and slot 15 through W_ms (today `Memory.w_ms`,
  `nn.Linear(384, 64)`, zero init; moved into the bank module, same init).
- `Probe(u)`: identity forward; its backward writes c_t = −⟨∂L/∂u_t, u_t⟩ into a preallocated slot (§4.6).

---------------------------------------------------------------------------------------------------------------------

## 4. The per-pass read gate (brain-decided; rev2 §4 for one store)

### 4.1 Reused from plan 22 (`memory.py: MemGate, gate_sizes, gate_trace`; `loop_brain.py: _gate_phi, _gate_open,
`_gate_after`, `_gate_readout`, `_gate_surprise`)
- The detached features (nu, dmu from `ConceptHier.uncertainty`, the thought change, last-token surprise + flag, its
  trace, the next-token entropy, the pass one-hot); the read-out moved before the read in training; the fp32 router
  in its own AdamW group and clip (excluded from the global clip); the device-side generator and fixed tie key; the
  open set cached for the recompute; eval modes learned / random / inverted / oracle / open / off.

### 4.2 What changes
1. **No `Memory` dependency.** Today's "pbar" feature is `a_hat[0, :, mem.d:]` (p̄ standardised by the row
   translator) and `--mem-gate` asserts `--mem-shared > 0` (`train_lm.py` line ≈ 972, `loop_brain.py` line ≈ 428).
   Rev3: the "view" feature = x̂_k (the bank frame; before the fit, the view standardised by EMA mean / RMS); the
   asserts move to `--bm`.
2. **Match features** (`--gate-match-feats 1`, default): [A (bf16), d₁/h₀,k², log(d₂/d₁), m_{j₁}/3, log(1 + age_{j₁}),
   bank-empty flag]. φ ≈ 64 + 4 + 4 + 1 + 1 + 1 + 1 + 1 + 16 + 6 = **99** features. These need the lookup on all
   positions (§2.4); the cost is in §14.
3. **Per pass**: ĝ_{k,t} = W_k φ_{k,t} + β_k, W ∈ R^{15 × 99} (≈ 1.5K parameters).
4. **What it predicts**: the centred gain in nats (§4.3), not a probability.
5. **Selection** (default, learned):

       O_k = TopUpTo_{n_k}( { t : ĝ_{k,t} > c_read } ) ∪ E_k,     c_read = 0 (constant; no schedule)

   - E_k: 5% uniform over ALL positions, always read (exploration and the centring sample).
   - n_k: static buffer size = the previous step's realised count of read k (non-blocking copy read one step later),
     +10%, rounded up to 64, clipped to [64, ⌊0.5 N⌋]; overflow beyond n_k dropped lowest-ĝ first and logged.
   - ⌊0.5 N⌋ is the hard speed cap per pass.
   - Options: `--gate-cost time` (c_read = κ·Δt_read, κ chosen by the owner), `--gate-cost <nats>`; `--gate-warm 0.2`
     (off); `--gate-budget fixed` (debug only: rev1's top-c schedule + θ̂); `--gate-target crank` (rank-BCE).
6. **Why it does not collapse** (rev2 §4.2.5): all-open is impossible beyond the cap and, with z centred at the
   population median, only ≈ the upper half has ĝ > 0; all-closed is impossible because E always reads and keeps the
   closed region observed. What centring gives up: the gate cannot conclude "reads are useless everywhere"; that is
   the knockouts' job (§13).
7. **Inference**: read iff ĝ > c_read, the cap per batch, no exploration.

### 4.3 The target (centred on the exploration sample)

      c_{k,t} = −⟨∂L/∂u_{k,t}, u_{k,t}⟩                       (in-window reads only)
      z_{k,t} = N_tok · ( c_{k,t} − q₅₀^{E_k}(c_k) )          (median over this micro-batch's exploration sample)
      L_r     = λ_r · mean_{in-window k, t ∈ O_k} Huber_δ(ĝ_{k,t} − z_{k,t}),  δ = N_tok · 1.4826 · MAD^{E}(c_k),  λ_r = 0.1

- E is uniform and independent of φ, so q₅₀^E estimates the population median (410 positions per micro-batch at
  16 × 2: sd of the median ≈ 0.06 MAD). Centring on the open set (rev1) would let the router move its own zero.
- Consistency: selection depends on φ and the exploration draw only; E gives positivity everywhere, so the open-set
  regression minimiser is E[z | φ] (missing at random given φ).
- Out-of-window reads use ĝ but do not train it.

### 4.4 Watchdog (alerts only; no automatic fallback)
Per pass, each eval: AUC of ĝ vs 1[z > 0] on a held-out micro-batch's exploration sample (plan 22: 0.61), calibration
slope, open rate (train / eval), cap / floor hits, overflow. Alert after 0.3·S if AUC < 0.60 at two consecutive evals
or the open rate of passes 0-5 is pinned at 5% or 50% for 500 steps.

### 4.5-4.6 The post-backward analytic router update and Probe idempotence (rev2 review point 1, unchanged)
1. Forward (per micro-batch, per in-window read k): keep φ_{k,O}, ĝ_{k,O}, O and the E mask, detached, until after
   backward (≤ 8,192 × 99 × 4 B ≈ 3.2 MB per read; ≤ 6 in-window reads → ≤ 20 MB).
2. Backward: the Probe writes c into `cbuf[mb, k]` with `index_copy_` (idempotent) and increments `cnt[mb, k]`.
3. After `backward()` of the micro-batch: z (§4.3) and ∂L_r/∂W_k = (λ_r/|O|) Σ_{t∈O} ψ_δ(ĝ_t − z_t) φ_tᵀ,
   ∂L_r/∂β_k = (λ_r/|O|) Σ_t ψ_δ(ĝ_t − z_t), added (÷ accum) into `.grad`; the router's AdamW steps once per
   optimizer step, so ĝ in the forward used the same W (exact gradient).
- **Checkpointing**: `_hier_step` is one non-reentrant checkpointed block: the forward runs twice, its backward once.
  The Probe does nothing in forward (no hook registration, no append) and writes only in backward into a fixed slot,
  so the recompute cannot double c; a stray double backward would overwrite the same values and show cnt = 2.
- **Test `test_gate_postbwd` (CPU)**: a tiny LoopBrain with the checkpointed block and a 2-bank store: (a) cnt == 1
  for every in-window read, 0 outside; (b) c equals −⟨∂L/∂u, u⟩ computed with checkpointing off; (c) the analytic
  router gradient equals autograd of L_r with z fixed (rtol 1e-5); (d) accum 2 and 4 give the full-batch gradient;
  (e) the bank value gradient and the query gradient through w, A equal the non-checkpointed ones.
- Today's `gate_diag` hook (`M.register_hook` outside the checkpoint) is replaced by the Probe buffer.

---------------------------------------------------------------------------------------------------------------------

## 5. Benefit signals
- c is ≈ zero-sum over a batch (Σ_t c_t = −⟨∂L/∂W_o, W_o⟩_F ≈ ω‖W_o‖² under weight decay): only centred values (z) are
  used, for the router and for confirmation; no sign-based disconfirmation, no spawn-on-hurt.
- The forward-only CE test (rev2 §5.2: CE of the read-out of M alone vs ce_k − m_ce) stays an option for
  out-of-window confirmation, enabled only if T0b-style calibration shows precision ≥ 0.85 at recall ≥ 0.3 (it needs
  a trained bank: measured on the pilot, §15.3).

---------------------------------------------------------------------------------------------------------------------

## 6. Writes

### 6.1 Which pass writes
Each in-window read k writes into bank k (out-of-window reads have no ∂L/∂inject_{k+1} for the value init).
`--bm-write-nograd 1` (ablation): ξ-only values for out-of-window spawns.

### 6.2 The surprise criterion: only the brain's own signals (T0h decides)
- Candidates: σ^raw_t = ce_{K−1}[t]; σ^ent_t = ce_{K−1}[t] − Ĉ(H_{K−1,t}) (CE above what the model's own entropy predicts;
  Ĉ an EMA 32-bin regression); σ^imp_t = ce_{K−1}[t]·1[ce_0[t] − ce_{K−1}[t] < δ_imp] (a gap thinking did not close).
  All from tensors the step already has. **No dataset labels, source tags or document-type features** (constraint 1
  and rev2 finding 4).
- **Eligible_k(t)**: ≥ 8 tokens after a document start, not EOT, and no confident bank read at (k, t)
  (t ∉ O_k, or A_{k,t} < A_w = 0.3). A missed recurring fact is therefore re-writable (re-acquisition, §7.5).
- **Budget**: W = 32 per optimizer step per in-window read, ≤ 2 per sequence, ≥ 8 tokens apart; static topk.
- Stash: index, x̂_{k,t}, σ_t, ĝ_{k,t} = −∂L/∂inject_{k+1}[t] (384) by a hook on the summed inject at the W candidate
  positions (on the block output, outside the checkpoint: fires once); ≈ 50 KB per bank per step.

### 6.3 Key and value (ε guard)
- Key: x̂_{k,t}; τ = t.
- Value: q = W_o^(k)ᵀ ĝ;  v = σ_ξ ξ + ν v̄_k · q / max(‖q‖, ε_q) · 1[‖q‖ ≥ ε_q], σ_ξ = 0.02, ν = 1, ε_q = 1e-6 ‖ĝ‖;
  v̄_k = EMA of the median ‖v‖ of bank k's Adagrad-touched slots (init 0.05). At the start W_o = 0 → q = 0 → ξ-only
  (no division by zero); the gradient write takes over once ‖q‖ ≥ ε_q.

### 6.4 Dedupe at write (wrong-merge bound)
- Against the store (after_step; one batched block-max lookup for ≤ 32 × 15 candidates): d₁, j₁ = the nearest live key.

      d₁ < γ h₀,k²  ⟹ MERGE into j₁:
         η_k = max(1/(n_{j₁}+2), 1/4);   k_{j₁} ← k_{j₁} + η_k (x̂ − k_{j₁})          (key keeps tracking: rev3)
         v_{j₁} ← v_{j₁} + (v_new − σ_ξ ξ)/(n_{j₁}+2);   τ_{j₁} ← τ_{j₁} + η_k (t − τ_{j₁})
         n_{j₁} ← min(n_{j₁}+1, 8);  log α_{j₁} += 0.5;  Λ^α_{j₁} ← t;  t^touch ← t
      otherwise ⟹ SPAWN (§6.5)

- **γ from T0e**: the largest γ ≤ 0.5 at which the wrong-merge rate P(j₁ is another fact | d₁ < γh₀²), measured on the
  T0a capture at the real distractor density, is ≤ 2%; the same-fact merge rate at that γ is reported. Placeholder
  0.25. If no γ ≥ 0.05 meets 2%, dedupe is off.
- Within the step: per bank, candidates ordered by σ; one within γh₀² of an earlier one is suppressed and its value
  increment goes to the kept one (`index_add_`). Merges consume no eviction capacity.

### 6.5 Spawn and eviction
- Spawn: k = x̂, v as §6.3, log α = 0, Λ^α = Λ^v = Ψ = τ = t^touch = t, t^conf = −∞, m = n = 0, S = S₀,k.
- Victims: the free list, then the lowest lazy strength ã (§8) in the bank, **excluding t^touch = t** (one masked topk).
- Confirmed slots (m ≥ 1, log α up to log 8) outrank unconfirmed ones, so the recurring-fact reserve forms by itself.

### 6.6 Zombie guard (new)
- A mature slot fades at ρ = 4^{−m} λ_α; at m = 3 its strength would take ≈ 500K steps to fall from log 8 to 0, so a
  slot whose fact has moved to a new slot (lost key, re-acquired elsewhere) would hold its place for ever.
- Rule: at eviction time, if t − t^conf_i > 3,000 steps, ã_i is computed with ρ = 1 (lazy, no state change). A late fact
  is confirmed every ≈ Δ/p_ren ≈ 250-500 steps, far inside 3,000; rare-fact slots are never mature anyway.

---------------------------------------------------------------------------------------------------------------------

## 7. Renewal, confirmation and survival (in Stage A)

### 7.1 Why renewal is first-order now
The bank keeps a write-time key. The thought drifts (median drift 1.22-1.31 × the 1-NN gap per 1K steps), so an
unrenewed key is found with R@1 ≈ 0.77 after 1K steps, ≈ 0.45 after 2K, ≈ 0.15 after 3K (4K bank; lower at 64K). A late
fact recurs every Δ ≈ 105 steps for the whole run; its slot must be renewed faster than it drifts away.

### 7.2 Does value-gradient training on every read suffice (as for the rows)?
- **What it gives**: every in-window open read trains its top-4 slots' values by row-Adagrad (the LM gradient through r).
  A late fact's slot that is found at ≈ 116 × p_open × P_win × p_hit ≈ 35 of its exposures gets ≈ 35 value updates:
  enough for p_val (the value decodes the answer) to approach what the rows reached. Rare facts get 1-3.
- **What it does not give**: it never moves a key. The rows did not need key moves because their addresses were FIXED
  random functions of the current query: wherever a drifted thought landed, those rows were read and trained, so every
  open in-window exposure was a renewal (p_ren^rows = p_open × P_win, rev2 §7.3). A bank's address is the stored
  key: once the query drifts by ≳ h₀, A → 0, the gradient into the slot vanishes, and the nearest slot may be another
  fact's (whose value the gradient then bends toward this fact: interference, damped only by A).
- **Conclusion**: value training is necessary (it makes p_val grow with exposures) but not sufficient; key renewal is
  required. `--bm-track all` (move the winner's key toward EVERY confident read, rows-like, no confirmation) is an
  ablation; it maximises renewal but lets a wrong winner walk toward another fact's queries (attractor risk, §11).

### 7.3 Confirmation and the key move (default)
- **In window (benefit)**: a read (k, t) confirms its winner j₁ iff t ∈ O_k, z_{k,t} > 0 and z_{k,t} is in the top 25%
  of the bank's in-window open reads in this micro-batch, AND the geometry passes: w₁ ≥ ω₁, d₂/d₁ ≥ μ_m, A ≥ A_min
  (T0e, at precision ≥ 0.95 that j₁ is the same fact). ≤ E_b = 128 confirmations per bank per optimizer step.
  - E_b check: late-fact cue positions arrive at ≈ 20.4K × 116 / 12.2K ≈ 194 per step (all 32 windows); at p_open 0.8
    and P_win 0.53 ≈ 82 reach an in-window open read of bank 0; E_b = 128 does not bind.
- **Out of window (geometry, `--bm-renew-geo 1`, default; §OD-1)**: a confident read (t ∈ O_k, out of window) confirms
  j₁ iff the stricter geometry passes (thresholds at precision ≥ 0.95 from T0e, with A_min,geo ≥ A_min). No
  benefit signal exists there (no backward), so the confirmation rests on the match alone.
- **The update** (after_step, per slot, sort + segment mean of its confirming queries q̄):
  k ← k + η (q̄ − k), η = 0.5 (1 if u < 2), the move capped at h₀,k per step; τ, Λ^α, t^conf, t^touch ← t;
  log α ← min(ã + 1, log 8); m ← min(m + 1, 3) if ≥ 64 steps since its last increment; Λ^v untouched; refresh k̃.
- Separation: after a move ‖k₁′ − k₂‖ ≥ (√μ_m − (1 − η))√d₁ from the next key (rev2 §7.2); the "fixed point" is a
  lagged EMA of recent confirming queries (it follows the drift), not a benefit-weighted centroid.

### 7.4 Survival of a recurring fact's TRAINED slot (rev2 §7.3, per pass)
- Renewals of fact F's slot in bank k arrive as a Poisson process with rate λ = p_ren/Δ; the slot is lost when a gap
  exceeds the horizon H ≈ 1,500 steps (p_ret collapses); lost-gap events occur at rate λ e^{−λH}:

      S(T) = exp(−λ T e^{−λH}),   late facts: Δ ≈ 105, T = 12.2K, H ≈ 1,500  ⟹  S ≥ 0.9 needs p_ren ≥ 0.43

- Per pass, with in-window benefit confirmation, out-of-window geometric confirmation and re-write on a miss:

      p_ren(k) = p_open · p_hit · [ P_win(k) p_conf + (f_k − P_win(k)) p_geo ] + p_open · P_win(k) · (1 − p_hit) · p_w

  (p_hit = P(the cue's query finds F's slot as j₁ while it is alive) = the T0a cross-document same-fact R@1 at short
  age; p_conf, p_geo = the pass rates of the confirmation rules on correct hits; p_w = P(a missed cue is written).)

| scenario (p_open 0.8, p_conf 0.6, p_geo 0.7, p_w 0.3) | p_ren passes 0..4 | S(12.2K) passes 0..4 | union, independent / fully correlated |
|---|---|---|---|
| p_hit 0.8, geometric renewal ON (default) | 0.44, 0.39, 0.35, 0.30, 0.25 | 0.91, 0.84, 0.75, 0.61, 0.45 | 0.999 / 0.91 |
| p_hit 0.8, in-window only | 0.23 (each) | 0.36 (each) | 0.90 / 0.36 |
| p_hit 0.5, geometric ON | 0.32, 0.29, 0.26, 0.23, 0.21 | 0.69, 0.59, 0.49, 0.38, 0.28 | 0.97 / 0.69 |
| p_hit 0.5, in-window only | 0.19 (each) | 0.23 (each) | 0.74 / 0.23 |

- **What it implies**:
  1. In-window confirmation alone cannot reach the bar in any single pass: even at p_hit = p_conf = 1 it gives
     p_open · P_win = 0.42 < 0.43. Hence the out-of-window geometric renewal is in the default (§OD-1).
  2. The truth lies between the independent and the fully correlated union (the same exposure's open / window
     events are shared across passes); the eval reads all passes and sums their injects, so one surviving pass can
     carry a fact, diluted by the others' wrong reads (A mutes them).
  3. p_hit is the cross-document (paraphrase) hit rate, which T0a measures: **T0a decides whether late facts survive**,
     not only whether rare facts are found.
  4. The deep banks (k ≥ 5) cannot keep recurring facts (P_win ≤ 0.42, f_k ≤ 0.47); their reserves are small (§2.2).

### 7.5 Re-acquisition (a lost key is not lost for ever; a lost trained value is)
- After a loss, the next exposure misses, the cue is eligible, and it is written with p_acq = p_open · P_win · p_w per
  exposure (≈ 0.127 at p_w 0.3): a two-state chain with loss rate ν = λ e^{−λH} and recovery μ = p_acq/Δ; availability
  of SOME key = μ/(μ + ν) = 0.84 / 0.94 / 0.99 at p_ren = 0.10 / 0.23 / 0.44.
- But a re-acquired slot starts with a one-shot value (ξ + one gradient direction), not the ≈ 35 updates of a kept
  slot: p_val drops after each loss. So S(T) of §7.4 (the trained slot) is the relevant number for late-fact recall,
  and availability is its floor.
- Option (open question, not default): `--bm-rescue`: a spawn whose value direction q matches an existing slot of the
  bank (cos ≥ 0.9) within 4γh₀² moves that slot's key instead of spawning, keeping its trained value.

### 7.6 Fill time: can 20.4K late facts enter each bank by 7K?
- Unacquired late cues compete with every other eligible surprising position for W · P_win ≈ 17 writes per step in
  banks 0-4. With s_late = the share of those writes that go to unacquired late cues (T0h measures it), the
  acquisition rate is ≈ 17 s_late facts per step: 20.4K facts take ≈ 1,200 / s_late steps (s_late = 0.25 → 4.8K steps;
  0.5 → 2.4K).
- The rows had no such limit (every open read trained the rows it hit). This is the main reason the late-cloze bar at
  7K is at risk. Mitigations, in order: (i) σ^ent / σ^imp if T0h shows they raise s_late; (ii) W = 48 for banks 0-4
  until 3K steps (a per-step budget schedule on WRITES, not on reads; life then ≈ 1.5K while it lasts); (iii) a larger
  dedupe γ is NOT a mitigation (it raises wrong merges).
- Per exposure, a fact is first written with probability ≈ p_open P_win p_w = 0.127 (p_w 0.3): ≈ 8 exposures ≈ 830
  steps once the budget is not binding.

### 7.7 Rare facts (unchanged logic, rev2 §7.3)
P(recall) ≈ p_write · p_open(eval) · E_{a ~ Exp(Δ)}[p_ret(a)] · p_val, with E[p_ret] ≈ 0.6 × the T0a cross-document factor
at 4K density (lower at 64K). Rare exact ≥ 0.10 needs a cross-document R@1 ≥ ≈ 0.6 at p_write 0.6; at p_write 0.3 it is
unreachable. The Stage A early bar (0.035, §16.2) is set from what the old run and control show, not from this bound.

---------------------------------------------------------------------------------------------------------------------

## 8. Fading, maturity, clocks (lazy, plan 21 style)
- **Strength** (Λ^α): ã_i = log α_i − λ_α ρ_i (t − Λ^α_i), λ_α = **2.5e-4**, ρ_i = 4^{−m_i} (ρ = 1 under the zombie
  guard, §6.6); renewed by spawn, merge, confirmation. Used for eviction only (default).
- **Value** (Λ^v): f_i = exp(−λ_v ρ_i (t − Λ^v_i)), λ_v = 1e-4, folded at every Adagrad touch.
- **Accumulator** (Ψ): discount 1 − β_i = (1 − β₀) ρ_i, β₀ = 0.99, floor 0.1 S₀.
- Keys are never frozen.

---------------------------------------------------------------------------------------------------------------------

## 9. after_step order and host syncs
- **After each micro-batch's backward**: the router's analytic gradient (§4.6); the bank's value gradient accumulated
  compactly (sort + segment sum into a static buffer of ≤ 4 × the step's open in-window reads); the confirming
  (slot, query) pairs appended to a static buffer (masks + a dummy slot).
- **After opt.step** (once per optimizer step):
  1. router AdamW (own group);
  2. bank Adagrad on the values (lazy fold f, Λ^v ← t, discounted S, clipped step, index_copy); t^touch ← t;
  3. confirmations → key moves (§7.3);
  4. merges, then spawns into the free list / weakest with t^touch < t (§6.4-6.6);
  5. refresh k̃ (bf16 blocks) of the changed slots;
  6. EMAs: h₀,k, v̄_k, Ĉ, the router statistics (device scalars).
- **Host syncs**: none added. Every budget is a host integer (W, E_b, the exploration size, chunk sizes; K is drawn on
  the host); n_k uses the previous step's count via a non-blocking copy; variable-size sets are masks over static
  buffers; per-slot averaging uses sort + neighbour flags + cumsum + index_add_ (no `torch.unique`). Removing the
  rows also removes their `write_counts()` sync (`train_lm.py` ≈ line 1397: "ONE host sync").

---------------------------------------------------------------------------------------------------------------------

## 10. What leaves the brain, what stays (code map)
- **Gone** (`--mem-shared 0 --mem-private 0 --mem-coarse 0`): the `Memory` object (`self.mem = None`), its V / G /
  V_e / G_e / P-family tables, product keys, translators, MemOpt, LifeCycle, MemoryProbes on rows, `balance`,
  `restart_keys`, read0, address reuse (`--mem-readdr-every`), the margin loss, `_private_fn`.
- **Stays**: ConceptHier (`self.hier`, built in `_build_memory` regardless of the memory; concept loss, nu / dmu,
  think features); `_hier_step` stays the one checkpointed block (with address=False today it returns only μ, ν; rev3
  adds the bank lookup / read inside it); MemGate (one store, §4.2).
- **Moves**: `w_ms` and `_slot_write` from `Memory` to the bank module; the gate's view standardisation from the row
  translator to the bank frame.
- **Bit-identity**: `--bm 0` with no row memory = the no-memory control's code path, bit-identical (new modules built
  last on a forked RNG, as plan 22 did).

---------------------------------------------------------------------------------------------------------------------

## 11. Self-reinforcement: measured, not forbidden
Loops: (1) bank read k′ < k → s_k → bank k's query; (2) reads → c → ĝ → what opens → what trains → c; (3) confirmation:
read slots get renewed and read more; (4) the match-weight gradient: LM loss → w, A → thought → the next query; (5)
the key move itself pulls keys toward the thought (with renewal now in Stage A, loops 3-5 are active from the start).

| metric | definition | review threshold |
|---|---|---|
| read-induced view shift | 8 probe sequences per eval: ‖v_k(memory on) − v_k(all reads off)‖ / h₀,k | > 1 h₀ |
| echo rate | P(top-1 slot written by the SAME document) among confident reads | > 0.5 with benefit rank < median |
| concentration | Gini of reads per slot; share of reads on the top 1% of slots | top-1% share > 20% |
| confirmation feedback | Spearman(confirmations so far, rank-benefit b̄) | ≈ 1 with b̄ falling |
| key-move drift vs thought drift | median key move per 1K steps / median thought drift per 1K steps | > 1.5 (keys running ahead of the thought) |
| open-set persistence; closed-region starvation | rev2 §12 | rev2 §12 |
| knockout of earlier-pass reads | eval: reads k′ < k off, bank k's recall / benefit | reported |

---------------------------------------------------------------------------------------------------------------------

## 12. Training data
- **Stage A uses the current data and the random-window loader**, like the old run and the control, so the same-step
  comparisons in §16 are valid. "Early" = a rare group (≈ 5.5 exposures by 7K, ≈ 9.5 by 12.2K); compare it only at
  equal steps.
- **Option: data v2 (Stage A2)**, as rev2 §13: a 32-cursor sequential loader (cursor i starts at i·N/32, one 512-token
  window per cursor per optimizer step), a scheduled rare group (k ∈ {1, 2, 4, 8} mentions × spacing {10 … 3,000}
  steps, held-out cloze templates 10-11, age buckets), introduce-then-query (Δ ∈ {10, 100, 1,000}, different and same
  template), an exposure log by step (eval bookkeeping only: the memory never sees it). **It changes the training
  distribution, so the no-memory control must be rerun on it, and the old run stops being a same-data reference.**
  Worth it if Stage A passes and the owner wants rare-fact behaviour by spacing / age; ≈ 260 lines + one control run.

---------------------------------------------------------------------------------------------------------------------

## 13. Metrics and evaluation
- **Primary**: cloze exact / first-token / lp per token: late (n = 5,000), early (n = 1,000; 200 per relation), geo;
  by relation. Loss by target-token type (salvage `extra_split.py` into eval_world): world_answer_first / rest,
  world_other, wiki; knockouts ko_bank (all banks), ko_bank_k (one pass), ko_gate (forced open), gate modes learned /
  open / off.
- **Automatic eval** at 25 / 50 / 75 / 100% AND at step 7,000 (the old run's last checkpoint), in-process, after
  `torch.cuda.empty_cache()`, eval_batch 16, `--rank-limit 200 --questions 1 --knockouts bank,bank_k --gate-modes
  learned,open,off --capacity dropless`; checkpoints saved there. Not counted in tok/s.
- **Online logs** (every eval): per pass, the gate's open rate (train / eval), cap / floor hits, overflow, AUC,
  calibration slope, open rates by token type (eval tags only), **p_open at answer cues**; per bank, live slots,
  spawns / merges / evictions / confirmations (benefit / geometric) per step, mature-slot count (the reserve), zombie
  count, h₀ and κ, median d₁, A, the block-max miss rate, p_hit at late / early cues (eval tags only), s_late (§7.6,
  eval tags only), §11's table.

---------------------------------------------------------------------------------------------------------------------

## §VRAM

### Measured (VRAM study, old run at end-of-growth size, 16 × 2, seq 512, bptt 6)
Training peak 20.12-20.25 GB allocated (K ≥ 6), 20.5-21.1 reserved; `evaluate()` 23.05 reserved (eval on the training
cache); memory tables **3.54 GB** (V 1.09 + G 1.09, V_e 0.40 + G_e 0.40, P-family 0.5); memory queries / write-back
activations ≈ 0.4; fragmentation 0.4-1.0 GB in training. `expandable_segments` is unsupported on Windows.

### Stage A projection (16 × 2)

| item | Δ GB |
|---|---|
| old-run training peak | 20.14 |
| row tables and their grads (V, G, V_e, G_e, P-family) | −3.54 |
| row read activations (queries, write-back; part of the 0.4) | −0.3 … −0.4 |
| bank store, 488K slots × 1,460 B | +0.73 |
| bank value gradients, compact static buffer (≤ 4 × ≈ 50K open in-window reads × 256 × 4 B) | +0.2 |
| lookup temporaries (1,024 × 64K bf16 score chunk + block maxima), transient | +0.15 |
| bank read inside the checkpointed block (gathered values (|O|, 4, 256), recomputed in backward) | +0.05 |
| router kept tensors, W_o^(k) × 15 with Adam moments | +0.05 |
| **projected peak allocated** | **≈ 17.4** (reserved ≈ 17.8-18.4) |
| `evaluate()` after `empty_cache()` | ≈ 6.2 reserved |

### Run defaults
1. `torch.cuda.empty_cache()` before every `evaluate()` and before the loop report (`--empty-cache-eval 1`).
2. Guard `set_per_process_memory_fraction(20e9 / total)` ≈ **0.78** (`--mem-fraction 0.78`); a guard only (near the cap
   the allocator retries; eval ran 2.4× slower).
3. eval_batch 16 (matches the training block sizes).
4. **Micro-batch 16 × 2 (default, as the old run)**. 8 × 4 (13.2 GB allocated in the study; ≈ 10.8 projected here)
   costs 4.5% speed, more than the 3% speed criterion, and needs the same-maths audit (per-forward side effects run 4×
   instead of 2× per step: homeostasis `alpha`, balancer `pending` / `pend_e` / `lastq`, EMA norm statistics, MoE
   capacity per micro-batch). Use it only if T2 measures reserved > 19.5 GB at 16 × 2 AND the audit passed; next
   lever chunked fused CE (−1.2 GB, −2%); per-pass activation checkpointing stays in reserve.
- VRAM spike avoidance in the bank itself: scores are never built beyond one 1,024-query chunk; the dedupe lookup uses
  the same chunked path; no (B, T, N) tensor exists anywhere.

---------------------------------------------------------------------------------------------------------------------

## 14. Cost vs the old run in the same harness (speed is a separate track)

| component (s / step, est.; T2 replaces) | Stage A vs the old run |
|---|---|
| rows removed: product-key select + online + shared / private read + backward, MemOpt over ≈ 570K + 143K touched rows, dense G zeroing, `write_counts` sync | **−0.15 … −0.40** (not measured separately; the old profile's index_add 80 ms, gather 64 ms and part of the 118 ms topk are largely the rows'; T2 measures `--mem-shared 0 --mem-private 0` in the same harness) |
| bank lookup on all positions (bf16 GEMM, block max, fp32 re-rank) | +0.030 … +0.045 |
| bank read (gather, kernel, content gate, W_o), its backward and value-gradient accumulation | +0.005 … +0.015 |
| router features, match features, selection, kept tensors, post-backward update | +0.010 … +0.020 |
| after_step (Adagrad, confirmation moves, dedupe / spawn / evict, key copies, EMAs) | +0.005 … +0.010 |
| **net** | expected faster than the old run (≈ −5 … −15%); worst case ≈ +2% if the rows were cheaper than estimated |

- Criterion: tok/s ≥ 0.97 × the old run's config in the same harness (median of ≥ 3 runs per config; repeat runs vary
  ≈ ±3%). The 10k / 20k targets belong to the speed track (fusion / `torch.compile`, host-sync removal); the bank keeps
  static shapes and adds no host syncs so it does not block compile.

---------------------------------------------------------------------------------------------------------------------

## 15. Tests before any run

### 15.1 Real-thought checks (CPU = `CUDA_VISIBLE_DEVICES=-1`, ≤ 2 threads)

| test | what | where | GO / NO-GO |
|---|---|---|---|
| **T0a paraphrase / cross-document same-fact retrieval** (likely hard) | A dedicated capture: 300 late + 300 early people × 5 relations = 3,000 facts, each in **≥ 3 documents** with 3 different training templates (0-9) and different surrounding facts, + 1 held-out-template cloze prompt (10-11). Checkpoints 6K and 7K; **train-mode routing** (random head-group offset, route noise 0.3, capacity 1.5) and eval routing; pre-inject views (slots 0-14 mean) per pass 0-5 at the cue (the token before the answer). Gallery per pass = doc-1 cue keys + distractors sampled by the T0h write criterion from ordinary training windows at the real bank density (16K, 32K, 64K). Queries: doc-2 / doc-3 cues (other document, other wording) and the cloze cue. Report R@1 / R@16, the same-template-other-fact top-1 rate (template keying), keys-per-fact after dedupe, within a checkpoint (age 0, the paraphrase factor alone) and 6K → 7K (paraphrase + drift). Diagnostic variants (no design change without the owner): ConceptHier μ_l as the view; the best single slot | capture: GPU when free, or CPU if a 4-window timing shows ≤ 1 min per window (≈ 2-3K short docs); analysis CPU | **GO**: within-checkpoint doc → doc same-fact R@1 ≥ 0.6 (this is p_hit for late renewal) AND doc → cloze R@1 ≥ 0.5 at the best pass at 64K density AND 6K → 7K doc → cloze ≥ 0.35. **Weak GO**: doc → cloze 0.35-0.5 or doc → doc 0.45-0.6: Stage A may run; the owner is told the expected late-fact survival from §7.4 at the measured p_hit. **NO-GO**: doc → cloze < 0.35: the banks serve same-template reuse only; back to the owner (option A was his choice; no automatic fallback) |
| **T0b gate separability** | Old run ckpt 7K (the rows are the only trained memory: a proxy), 16 world + 16 wiki windows, train-mode forward + backward, Probe on u^row: c by token type per pass (answer first / rest, world other, wiki); a linear router on φ (held-out windows) for 1[z > 0]: AUC; the open rate at c_read = 0 | GPU (CPU is feasible only for ≈ 4 + 4 windows: too few for the AUC) | the gate can see the harm iff median c(world_other) < median c(answer) by ≥ 1 MAD; AUC reported (plan 22: 0.61); AUC < 0.6 → the owner is told before the run (the gate would open near-randomly up to the cap). Repeated on the pilot's bank (§15.3) |
| **T0h p_write** | Training windows, ckpt 7K, train-mode routing: per-pass CE, entropy, final CE; for σ^raw / σ^ent / σ^imp at W = 32 per step per bank: the fraction of **rare (early) AND late** answer-cue positions that land in the write top-W (eval tags = bookkeeping only), and s_late for unacquired late cues (§7.6); the Wikipedia vs world share of the top-W | CPU on ≈ 64 windows (forward only), GPU faster | **GO** best criterion p_write ≥ 0.5 on rare cues and s_late ≥ 0.25; **weak GO** 0.3-0.5 / s_late 0.1-0.25 (the write-budget mitigation of §7.6 is enabled); **NO-GO** p_write < 0.3: the bank cannot help rare facts, whatever the retrieval. The chosen criterion uses no labels or tags |
| **T0e calibration** | On the T0a capture: d_same per pass → h₀,k², κ_k; A_w; the in-window geometry (ω₁, μ_m, A_min) and the stricter out-of-window thresholds at precision ≥ 0.95; dedupe γ at wrong-merge ≤ 2% (with the same-fact merge rate); p_conf and p_geo on correct hits (inputs of §7.4) | CPU | defaults written; if same-fact and other-fact d₁ overlap so that no γ ≥ 0.05 meets 2%, dedupe off; if no geometric threshold reaches precision 0.95 at a pass rate ≥ 0.3, `--bm-renew-geo 0` and the owner is told (§7.4: in-window only does not reach the bar) |
| T0d lookup at real density | Block-max top-8 + fp32 re-rank vs a full fp32 scan on T0a keys + synthetic banks of 16K / 32K / 64K; recall vs density into T1 | CPU (timing on the GPU in T2) | miss ≤ 0.5% (else top-12 blocks); the §2.3 decision rule |
| T0g pass coverage | Realised f_k, P_win(k) from the old run's K log | CPU | §2.2 within ±10% |

### 15.2 T1 — the toy (CPU; extend `C:\Eitan\Python\NewBrain\speed_wip\splat\consol_toy.py`), after T0a / T0h / T0e
15 banks with §2.2's frequencies; drift from the salvage per-pass curves with density dependence; late facts (Δ 105,
116 exposures), rare-random (≈ 9.5 exposures, spacing 1.3K); the paraphrase factor from T0a; p_write, s_late from T0h;
p_conf, p_geo, γ from T0e. Policies: spawn + dedupe; + benefit confirmation; + geometric renewal (default); `--bm-track
all`; `--bm-rescue`; W 32 vs 48-until-3K; 64K vs 48K banks. Thresholds: late-fact trained-slot survival at the best pass
≥ 0.8 at 12.2K; rare-random recall ≥ 0.10; wrong merges ≤ 2%; keys per fact ≤ 1.3; fill of 20.4K late facts ≥ 80% by
5K steps. A miss on the late-fact rows goes to the owner before the run.

### 15.3 T2 — GPU bench and pilot (after the speed study releases the GPU)
- Harness `fruit-fly-sim\speed_wip\vram\vram_common.py`: the old run's config, the old config with `--mem-shared 0
  --mem-private 0` (the rows' own cost), and Stage A, at 16 × 2 (and 8 × 4), K ∈ {2, 6, 10, 16}: ms per §14 row, peak
  allocated / reserved incl. one `evaluate()`, host syncs (must not exceed the old run's 48), the block-max miss rate.
- **Pilot**: Stage A for 1,000 steps (a throwaway run): fill rates, open rates, the router's AUC on the bank (T0b
  repeated on a real bank), confirmation counts, reserved VRAM, tok/s. A pilot that shows a fault stops the plan
  before the 12K run.
- **GO**: tok/s ≥ 0.97 × the old run's config; reserved ≤ 20 GB at K = 10 and 16 incl. `evaluate()`.

### 15.4 Unit tests (CPU)
Flags-off identity (`--bm 0` = the control path); block-max lookup = fp32 brute force on the top-4 up to ties; dedupe /
merge (η floor, within-step suppression); confirmation key move (cap at h₀, per-slot averaging = a loop reference);
zombie guard; lazy clocks = eager; compact Adagrad = dense; **Probe once under checkpoint + analytic router gradient =
autograd + accum invariance + query gradient through w, A** (§4.6); learned-budget selection (static sizes, lagged
counts, cap / floor, E uniform); ξ-only value init at W_o = 0; capacity exactness; per-bank isolation (a read of pass k
touches only bank k); no new host sync (a sync counter / TorchDispatchMode op log); resume (store, clocks, frames,
router, generator).

---------------------------------------------------------------------------------------------------------------------

## 16. The run and GO criteria

### 16.1 Stage A run (12.2K steps; the owner's go)
- The old run's config with `--mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-nbr 0` + the Stage A flags (§18);
  current data, random loader, 16 × 2, the VRAM defaults; auto-eval at 25 / 50 / 75 / 100% and at 7,000.
- References at the same steps: the old run `world_mem_life_12k` (to 7K) and the control `world_v4d_12k` (6K, 9K, final).
- **Stop rule at 7K**: late < 0.24 or early ≤ 0.016 → stop, report the knockouts and logs, owner decides.

### 16.2 GO criteria

| criterion | bar | statistical basis |
|---|---|---|
| **late facts** (cloze exact, n = 5,000) | ≥ **0.27 at 7K** (old 0.280) AND ≥ **0.28 at the end** (control final 0.260 + 0.02) | sd of a difference of two n = 5,000 proportions at 0.28 ≈ 0.009: −0.01 is a ≈ 1.1 sd non-inferiority margin; +0.02 over the control ≈ 2.2 sd. Paired by item (same prompts) → also McNemar |
| **early / rare facts** (cloze exact, n = 1,000) | ≥ **0.035 at 7K** (old 0.016, control 0.013 at 6K / 9K) AND ≥ **0.045 at the end** (control final 0.023); at least 3 of the 5 relations improve | two-proportion z (pooled): 0.016 → 0.035 gives z = 2.70 (one-sided p ≈ 0.004); 0.023 → 0.045 gives z = 2.71. The same 1,000 items are scored in both runs, so the primary test is the exact McNemar test on the discordant pairs at p < 0.01; the per-relation condition guards against one relation (pet: 0.050 of the old 0.016) carrying the result. lp/token (old −3.420) and rank top-1 (old 0.035) reported |
| **harm on other world text** | world_other CE (memory on − ko_bank) ≤ **+0.04 nats** (old +0.133) AND world_all(on) ≤ world_all(ko_bank) (the memory is not net-harmful on world val loss; old: +0.069 net harm) AND |Δ wiki| ≤ 0.01 | 54,104 world_other tokens: the sd of a mean-CE difference on the same tokens is ≈ 0.005 nats, so +0.04 is far from noise; answer-token gain (old −0.29 / −0.20) reported, alert if < 0.10 |
| **speed** | tok/s ≥ **0.97 ×** the old run's config in the same harness | T2, median of ≥ 3 runs (± 3% run-to-run) |
| **VRAM** | peak reserved ≤ **20 GB** incl. `evaluate()` (cap 0.78 on) | T2 + the run log |

- Knockouts reported with every eval: ko_bank, ko_bank_k, gate modes learned / open / off.
- Stage A2 (data v2, optional) has its own criteria (rev2 §16.4 Stage B rows: rare group k ≥ 2, age ≤ 1K ≥ Stage A
  + 0.05 on the rerun control's data; introduce-then-query Δ ≤ 1,000, different template).

### 16.3 Order of work
1. This plan (rev3) → owner review of §OD.
2. CPU, in parallel with the speed study: unit-test scaffolding, T0g, T0h (64 windows), T0a capture if the CPU timing
   allows (else it waits for the GPU), T0e and T0d on the capture, then T1.
3. GPU when free: T0b; the T0a capture if not done on CPU.
4. Implement Stage A (+ the VRAM defaults); T2 bench; 1K-step pilot.
5. The 12.2K run with the owner's go; GO at 7K and the end.
6. Then (owner): Stage A2 (data v2 + a control rerun) and / or Appendix A transport.

---------------------------------------------------------------------------------------------------------------------

## 17. The owner's constraints — status in rev3

| # | constraint | status |
|---|---|---|
| 1 | addresses only from thoughts, not learned by gradient | met: keys = pre-inject views; frames / h₀ / thresholds are statistics; keys move only by rules toward thoughts (spawn, merge, confirmation). The query side gets LM-loss gradient through w, A (decided) |
| 2 | no gate loss reaches the thought | met: the router's φ is detached; the content gate's p̄ is sg; the router update is post-backward and touches only router weights. The LM loss reaches the thought through the read value and w, A (decided) |
| 3 | no pointers, no recompute loop outside the step | met: no transport; the router update reuses the step's backward; the out-of-window renewal uses only the step's own reads; the view-shift probe is an eval measurement |
| 4 | filled during training | met |
| 5 | ≥ 10k tok/s, reserved ≤ 20 GB | speed: the separate track (the baseline is 8.4k); Stage A is expected to be faster than the old run. Reserved: ≈ 17.8-18.4 GB projected at 16 × 2 with the defaults |
| 6 | the brain decides at every pass whether to read | met with two decided bounds (50% cap, 5% exploration); no schedule; the fixed budget is a debug flag |
| 7 | each pass has its own memory | **met for every pass** (15 banks, rev2 shared bank 8) |
| 8 | self-reinforcement measured | §11 (with renewal now in Stage A, all loops are live from the first run) |

---------------------------------------------------------------------------------------------------------------------

## §OD. Owner decisions

**Decided**: option A (bank-only from the first run; no rows anywhere; ConceptHier stays); per-pass memory; the brain
decides the read rate (read iff ĝ > c_read = 0; 50% cap; 5% exploration; fixed budget only as a flag); no gate loss on
the thought; the LM loss flows through the value and the match weights; the pass-T address = pass T's pre-inject view.

**New, defaults set by this plan (the owner may object):**

| | item | default | alternative |
|---|---|---|---|
| 1 | out-of-window geometric renewal | **on** (strict thresholds, precision ≥ 0.95 from T0e): without it no single pass reaches the survival bar (§7.4) | off: renewal only from in-window benefit confirmation |
| 2 | number of banks | **15** (each read its own) | 9 (rev2: k ≥ 8 shared) |
| 3 | match features in the router (lookup on all positions) | **on** (+15-20 ms/step) | lookup on open positions only |
| 4 | micro-batch | **16 × 2** (as the old run; 8 × 4 costs 4.5%) | 8 × 4 after the same-maths audit |
| 5 | write budget early | W = 32 throughout; **48 until 3K steps only if T0h shows s_late < 0.25** | fixed 32 |
| 6 | what to do on a T0a NO-GO | stop and ask | run anyway as a measurement; change the view (owner) |
| 7 | data v2 | not in Stage A; Stage A2 with a control rerun | in Stage A (loses the same-data comparison with the old run) |
| 8 | c_read | 0 | `--gate-cost time` (κ by the owner) or a fixed value |

---------------------------------------------------------------------------------------------------------------------

## 18. Flags and implementation size

**Run defaults**: `--empty-cache-eval 1 --mem-fraction 0.78 --eval-batch 16 --batch 16 --accum 2`.

**Stage A**:

    --mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-nbr 0                          (no row memory)
    --bm 1 --bm-banks 15 --bm-cap 64K,64K,64K,64K,64K,56K,40K,32K,20K,4K,4K,4K,4K,2K,2K --bm-dv 256
    --bm-view pre15 --bm-fit-step 300 --bm-write-start 300 --bm-write 32 --bm-surprise <T0h: raw|ent|imp>
    --bm-dedupe <T0e γ> --bm-merge-eta-min 0.25 --bm-h0 <T0e> --bm-fade 2.5e-4,1e-4 --bm-strength-read 0
    --bm-confirm rank --bm-qc 0.25 --bm-conf <T0e> --bm-eta 0.5 --bm-econf 128 --bm-renew-geo 1 --bm-zombie 3000
    --bm-wo perbank --bm-lookup blockmax --bm-block 16 --bm-topblocks 8 --bm-chunk 1024
    --mem-gate 1 --gate-budget learned --gate-cost 0 --mem-gate-explore 0.05 --mem-gate-cap 0.5
    --gate-target gain --gate-match-feats 1 --mem-cgate-pbar-sg 1 --mem-match-grad 1
    --auto-eval 1 --eval-at 7000 --save-report-ckpts 1

**Ablations / debug**: `--bm-banks 9|1`, `--bm-wo shared`, `--bm-write-nograd 1`, `--bm-track all`, `--bm-rescue 1`,
`--bm-renew-geo 0`, `--bm-confirm off|ce`, `--gate-budget fixed`, `--gate-warm 0.2`, `--gate-target crank|content`,
`--gate-cost time`, `--gate-match-feats 0`, `--mem-match-grad 0`, `--ko-bank`, `--ko-bank-pass k`, `--bm-lookup brute`.

**Implementation size (estimate)**

| part | lines |
|---|---|
| `bank_memory.py`: store, 15 banks, frames, block-max lookup + fp32 re-rank, read kernel with match-weight grad, Probe, compact Adagrad, dedupe / merge / spawn / evict, zombie guard, confirmation + key move (benefit and geometric), clocks, h₀ tracking, state / resume, stats | ≈ 800 |
| MemGate changes: one store without `Memory`, per-pass weights, view standardisation by the frame, match features, the gain target, learned budget + lagged sizes + cap / exploration, post-backward analytic update, watchdog, fixed / crank flags | ≈ 220 |
| `loop_brain.py`: the bank path in `_hier_step` (view, lookup, read, scatter), `w_ms` / `_slot_write` moved, Probe buffers and kept φ, the inject hook for the write gradient, confirmation buffers, `ko_bank`, removal of the row-only branches behind flags | ≈ 230 |
| `train_lm.py`: flags, asserts, the post-backward router step, after_step order, empty_cache / fraction cap, param groups (W_o, W_ms, content gates, router), auto-eval at 7,000, logging, resume | ≈ 180 |
| `eval_world.py`: token-type split, ko_bank / ko_bank_k, gate modes, open rates by token type, p_open / p_hit at cues | ≈ 150 |
| tests (§15.4) | ≈ 500 |
| T0a / T0b / T0d / T0e / T0g / T0h scripts, T1 toy extension, T2 bench + pilot wrapper | ≈ 420 |
| **total Stage A** | **≈ 2,500** (memory code proper ≈ 1,430; tests and checks ≈ 920; eval ≈ 150) |
| optional data v2 (generator, 32-cursor loader, exposure log) | ≈ 260 |

---------------------------------------------------------------------------------------------------------------------

## 19. Risks and open questions
1. **No fallback memory (the biggest risk).** If the banks fail on recurring facts, nothing else holds them: late cloze
   falls toward the no-memory curve (control ≈ 0.11 at 7K by linear interpolation), far below 0.27. Three independent ways to
   fail: a low cross-document hit rate (T0a), renewal short of p_ren ≈ 0.43 (§7.4), a slow fill under the write budget
   (§7.6). The pilot and the 7K stop rule limit the cost.
2. **T0a is likely hard.** The rows were template-keyed (same template other person 0.24 vs same person other relation
   0.045); the view may be too. Then cloze prompts in held-out templates do not find the training-document keys, and a
   fact needs one key per template (capacity, §2.2).
3. **Density.** 64K banks extrapolate to R@1 ≈ 0.50 (world ≈ 0.18) at a 1K gap; rare facts suffer most. Renewed late
   facts are queried young; T0d measures at the real (sparse) write density.
4. **Write budget vs late facts** (§7.6): 20.4K late facts per bank enter at ≈ 17 s_late per step.
5. **Self-reinforcement** is live from the first run (confirmation, key moves, match-weight gradient): keys may chase
   the thought and the thought the keys (§11 thresholds; `--bm-renew-geo 0`, `--mem-match-grad 0` ablations).
6. **Wrong confirmations** move a slot toward another fact's queries: the geometric thresholds are set at precision
   ≥ 0.95 (T0e) and the in-window path also requires z in the top 25%.
7. **The gate may not separate positions** (plan 22: AUC 0.61) or the harm may be a train / eval mismatch (T0b): the
   gate then opens near-randomly up to the cap. The match features should help (no near key → low gain); the pilot
   measures AUC on a real bank.
8. **Zombie slots and the reserve**: the zombie guard returns stale mature slots after 3,000 steps without
   confirmation; if a fact's confirmation interval exceeds 3,000 steps it is not recurring in the sense of §7.
9. **Lookup cost** is an estimate (30-45 ms); above 2% of the step: `--gate-match-feats 0`, a fused kernel (speed track),
   or Appendix B.
10. **Comparability**: the old run read every position with address reuse; Stage A reads gated positions; both on the
    same data and loader. The old run ends at 7K, so the end-of-run comparison is with the control only.
11. **Open**: geometric renewal precision on real data; `--bm-rescue`; whether banks 9-14 are worth having vs a shared
    deep bank; the CE test for out-of-window confirmation; per-pass c_read; whether the reserve should be explicit.

---------------------------------------------------------------------------------------------------------------------

## Appendix A. Drift transport (FUTURE; not in this plan; needs an owner check vs constraint 3)
Unchanged from rev2 Appendix A (full design in `25b_per_pass_memory_plan_rev1.md` §9): windows aligned with the
neurogenesis cadence; identical probe routines at both window ends (forward-only, stashed sequences); per-bank
1,024-anchor estimator with James-Stein shrinkage toward the history; keys moved by share of their time stamp, capped
at 0.5 h₀ per window. Expected gain at the measured R² 0.3-0.43: the recall horizon ×1.2-1.75; rare facts' E[p_ret]
+0.05-0.08. Out because it needs probe forwards outside the training computation. With renewal in Stage A, transport
matters mainly for rare facts; revisit after Stage A.

## Appendix B. The OPQ-style index (only for banks > 64K keys, or a lookup > 2% of the step)
Unchanged from rev2 Appendix B: index coordinate z = Vᵀ(x̂ − m) (PCA of the bank's keys), components dealt alternately
to two 32-d halves (balanced variance), T = 4 tables, 128 codebooks per half, cap 8 per cell, 4 probed cells → 128
candidates, ranking in the raw frame, refit every 1,000 steps in 32K-slot chunks, occupancy / coverage / candidate-recall
monitors. Not needed at ≤ 64K keys with the block-max lookup (§2.4).
