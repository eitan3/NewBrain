# 25d. Global + per-expert blob memory, the expert decides (PLAN ONLY) — 2026-10-24, revision 2

## Changelog of this revision (rev 1 → rev 2, 2026-10-24)

The pre-revision text is kept unchanged as `25d_global_expert_blob_plan_rev1.md`. This revision applies the owner's
decisions, the independent reviewer's fixes, the coordinator's defaults for the owner options and the accepted
rebuttals. Owner decisions of the outline are unchanged. Writing only: no code, no GPU, no installs, nothing deleted.

**Decided (recorded as decided, no longer open)**

| # | decision | where |
|---|---|---|
| D1 | speed bar = tok/s ≥ 0.97 × the old run (same harness); ALSO reported: cost vs a no-memory build and kernel launches per step (OQ-1 closed) | §17, §19.2 |
| D2 | private writes = a per-expert, per-step write budget (device credit counter) filled with that expert's most surprising kept tokens, sized for a blob life ≈ 2,000 steps (0.256 writes / expert / step); lifetime logged per bank (OQ-3 closed) | §3.2, §8.4 |
| D3 | the shared expert has the same private budget as an average routed expert (OQ-4 closed) | §3.2, §8.4 |
| D4 | one gate decision per (token, expert) covers global + private (OQ-5 closed) | §5, §6.1 |
| D5 | the global / private blend is weighted by match quality (normalised blend, π ∝ A) | §6.1 |

**Reviewer fixes, HIGH (all applied)**

| # | fix | where |
|---|---|---|
| H1 | gate / confirmation target: on the 5% exploration pairs a leaf read scale s ~ U(0,1); b̃ = −∂l/∂s is unbiased for l(0) − l(1); control variate (s − ½)κ̂‖r‖²; the gate is trained ONLY on exploration pairs; no median centring (zero = no gain); confirmation on opened pairs uses the curvature-corrected −⟨g, r⟩ + ½κ̂‖r‖²; the placebo −⟨g, r − r_rand⟩ is logged only | §5.3, §7, §9.2 |
| H2 | causal threshold gate: read iff ĝ > λ_e, λ_e ← [λ_e + 0.01 (rate_e − 0.5)]₊ per expert (pooled option); static 55% buffer, overflow dropped in token order (the batch rank cap is gone: it was not causal) | §5.4, §14 |
| H3 | IVF: centroids fitted on a mix of write-candidate and query x̂; a full cell spills to the 2nd-nearest cell, then the space's weakest is evicted; n_probe 16; refit every 1,000 steps (index only); alerts on cell Gini / overflow (OQ-7 closed) | §4.1, §8.6 |
| H4 | fp32 content-gradient buffers everywhere (private too); fp32 scores, or bf16 top-8 then fp32 re-rank; bf16 private contents with stochastic rounding stay | §3.1, §4, §6.3 |
| H5 | write budget: ≤ 3-4 writes per sequence (default 4), decoupled from W (the old ≤ 2 × 16 sequences = W made the cap the budget) | §8.1 |
| H6 | VRAM: static preallocated workspaces; `chunk_ce` + checkpointed concept loss become defaults if T2 needs them; reserved VRAM judged over ≥ 200 steps incl. `evaluate()` and a birth | §VRAM, §18.5 |
| H7 | merge only if d₁ < γh₀² AND cos(−g_new, v_old) > θ (θ from T0e); merges count in the per-blob move budget and path length | §8.5, §9.3 |
| H8 | launch-bound cost: realistic 70-100 ms / step (≈ 2,100 launches / step); no `bincount` (`index_add_`), no syncs, static buffer sizes from N·k and E·C | §13, §17 |

**Reviewer fixes, MEDIUM (all applied)**: λ_v = 0 (no content fade); h₀ tracked on exploration pairs, floors on MAD and
δ; bounded match kernel (d_j − d₁)/(d₁ + c·h₀²) with c·h₀² ≡ 2σ², σ² = EMA of median(d₂ − d₁)/(2 ln 2) (squared
distances); write key = the fixed projection of xf averaged over the token's in-window passes (T0c compares uniform /
averaged / maxgrad), content summed over the in-window passes; ‖v‖ capped (0.3 median‖xf‖ at spawn, 0.5 after); init
scale 0.1 × median‖xf‖ (normalised; the oracle test also scores 0.2 for the record); slightly negative gate β init;
store tensors are buffers, memory parameters kept out of `make_opt` / `carry_optimizer` / the global clip; the shared
expert's private space has fp32 grads and a share-of-gain metric (option: shared expert reads global only);
normalised blend r = max(A_g, A_p)(π_g u_g + π_p u_p), π ∝ A; a per-channel injection scale (init 1); the pass-0 flag
renamed "no previous read"; gradient hooks guarded on out-of-window passes; renewal α_j = max(0.3, 1/(n_j + 1)), each
move ≤ 1σ, anti-capture with cos(−g_t + κ̂r, c_j) > 0.3 plus the margin; blobs < 2,000 steps old protected (≤ 50% of
a space's capacity), else evict the lowest strength (EMA of positive benefit, half-life 4,000 steps); split children
inherit at half strength; content optimizer: row-Adagrad (default) vs lazy row-RMSprop (β₂ 0.9 per touch, η 0.03) as a
T1 sweep; top-16 search, blend 8 global + 4 private; match-stratified exploration inside the 5% with known propensities.

**Owner options: coordinator defaults (owner may override; §OD-P)**: (a) exploration warm-up 20% → 5% OFF; match-
stratified exploration on; log the steps until the gate's decile slope reaches 0.7. (b) extra gate inputs ON, both
detached: the previous pass's halting confidence and the content-fit score σ(n(xf) · n(W_cf c)). (c) direct writes
(W = 64 / step, ≤ 4 per sequence) by default; a probation / candidate stage (+0.26 GB) and private-by-consolidation
behind flags, compared in T1; T0h decides.

**Accepted rebuttals (kept as they were)**: global life ≈ 4,096 steps at W = 64; all 6 routed experts of a token are
private-write candidates at each write pass (no subsetting); writes start at the fit step (R@16 is measured, not a
gate); no fixed top-25% read fallback (alert only); no dead-bank recycling; init 0.1 × median‖xf‖, normalised;
row-Adagrad without momentum or decay (RMSprop only as a T1 sweep).

**Rejected (with reasons; §OD-R)**: holding writes until R@16 ≥ 0.9; a fixed top-25% fallback; dead-bank recycling;
random K.

**New evidence: T0b on the old rows** (`speed_wip\splat\realcheck\t0b\t0b_results.md`): reads help every token class
on training windows; the −0.07 … −0.12-nat harm on non-answer world tokens appears only on held-out templates /
documents (val_world): a generalisation failure, not a gating one. Sign-of-benefit AUC 0.52 train / 0.61 val / 0.49
train → val; features predict magnitude (0.68-0.74) and "answer next" (0.987), not sign; match quality adds nothing;
the first-order gradient target correlates 0.22 with knockout. Applied: the gate target is the unbiased s ~ U(0,1)
estimator only; the plan states that a training-time gate cannot remove held-out harm and does not count on the gate
for the harm bar (the gate is kept); the remedy is read robustness across wording, with **T0a as the decisive GO
gate**; a held-out-template harm metric (B22: val_world non-answer delta, memory on − off) joins the broken-if list;
the oracle ceiling also reports held-out harm; **T0b is repeated on the blob pilot** (§0, §5.0, §5.3, §16.1, §18.0,
§18.2, §19.2, §22).

**New sections**: §5.0 what the gate can and cannot do; §16.1 broken-if metrics; §18.0 pre-run GO gates (T0a, T0c incl. the averaged key and the ORACLE
CEILING gate the gate / write code; then T0d on write-distributed keys, drift 3K → 7K, gradient agreement, private
reach, hubness, template collisions 64-d vs 384-d re-rank); Triton 3.6 (installed) for fused IVF score + top-k, the
private search and the gate feature / selection fusion behind speed flags (§17).

Sections not listed above keep the rev-1 text (the 25c → 25d table below is history and keeps rev-1 numbers);
§OD-D / §OD / §OD-P / §OD-A / §OD-R hold the decision record.

---------------------------------------------------------------------------------------------------------------------

Full plan for the outline `25d_outline.md` (the owner-approved spec; its §2b fixes what the address and the gate see).
It replaces 25c (`25c_blob_only_per_pass_plan.md`, per-pass banks, rejected, kept as a record). The blob mechanics,
reviewer fixes and statistics of 25c are reused; where this file says "25c §x" the derivation there still holds and
is not repeated, only adapted.

Nothing was trained, no code was edited, the GPU was not used. Only this file was written (plus a scratch calculator,
CPU only, in the session scratchpad, for the numbers of §3, §9, §VRAM and §17).

Inputs read: `25d_outline.md`, `25c_blob_only_per_pass_plan.md`, `spiking_lm.py` (RoutedMLP, lines 527-822),
`loop_brain.py` (Core 98-253, `run()` 997-1196), `memory.py` (ConceptHier 281-330, `Memory.private` / `on_grow`
870-955), `train_lm.py` (train step 1345-1409), `neurogenesis.py`, and the old run's config / log
(`C:\Eitan\Python\NewBrain\runs\world_mem_life_12k\config.json`, `train.log`). Rev 2 also read
`C:\Eitan\Python\NewBrain\speed_wip\splat\realcheck\t0b\t0b_results.md`, checked that the old run has `halt 1`
(`loop_brain.py` `halt_head`) and that 3K checkpoints exist for both runs; its numbers were computed by hand (no
code run).

Facts taken from the code and the old run (they size everything below):
- MoE `768,6,192,384,auxfree`: up to 768 routed experts, k = 6 per token, hidden 192, one always-on shared expert
  (hidden 384); start 192 experts, `--neuro` split births between 5% and 60% of 12,207 steps (steps ≈ 610 → 7,320;
  548 births by 7K); capacity 1.5 (drop 0.024-0.045 in the log, ≈ 3-4%).
- d = 384, 16 workspace slots × 64, side(s) = n_s(mean of slots) (64-d), ConceptHier with L = 4 levels
  (`16,32,64,128`), each μ_l ∈ R^384 (`ConceptHier.levels`: Linear(…, 2d) → [μ; ν]).
- K ~ U{2..10} w.p. 0.95, U{12..16} w.p. 0.05: E[K] = 6.4 passes; bptt 6 → E[in-window passes] = 4.94.
- 16 × 2 micro-batches of 512 tokens: N = 8,192 tokens per micro-batch, 16,384 per step; ≈ 8.4k tok/s →
  ≈ 1.95 s per optimizer step; 12.8 core-pass forwards per step, ≈ 9.9 of them with backward.
- `RoutedMLP.forward` builds the capacity layout (e_k, p_k, t_k), `buf[e_k, p_k] = xf[t_k]`, one host sync
  (`counts.tolist()`); the `pmem` hook adds a private-memory OUTPUT after the experts (line 678). The core pass is
  not activation-checkpointed (only `route` under `ckpt_route`, and `_hier_step`).
- Births happen at the top of a step (`neuro.step`, before the micro-batches); `Core.grow_experts` already calls a
  `mem_hook(src, E0)` after `RoutedMLP.grow_experts` (copy-on-split of the old private rows).
- No expert is ever removed: there is no expert-death code (`neurogenesis.renew` renews dense-MLP neurons only).

---------------------------------------------------------------------------------------------------------------------

## Changelog: 25c → 25d (owner decisions in the outline)

| # | change | 25c | 25d | where |
|---|---|---|---|---|
| 1 | structure | 15 per-pass banks (488K slots) | **one global space (256K slots, fixed) + one private space per expert (512 slots × 769)**; nothing per pass | §1, §3 |
| 2 | address | pre-inject workspace view (slots 0-14 mean), per-bank frame | **xf = core.n_m(h), the RoutedMLP input**, before any memory; fixed centring / scaling + fixed random projection to 64-d, fit once at step 300, never learned; same map for global and private | §2 |
| 3 | where the read goes | M = W_o^(k) u into the next pass's inject and slot 15 | **into the expert's input**: added to its buffer row before w_up (shared expert: to its input), in the SAME pass; contents stored at d = 384, no W_o, no per-expert output matrix, no slot-15 write | §6 |
| 4 | who decides | one router per pass | **each chosen expert's gate, per (token, expert)**, before the expert runs: a shared trunk + a per-expert head (≈ 90K parameters for 769 experts) | §5 |
| 5 | gate input | 99 features | **xf + side(s) + the previous pass's μ_l (4 levels) + match quality (global and private; previous and current read)**, all detached | §5.1 |
| 6 | search | block-max brute force per bank | **global**: once per token per pass, an inverted-file (IVF) search over 1,024 fixed cells with a static capacity layout (brute force 256K would cost ≈ 110-160 ms/step); **private**: one bmm over RoutedMLP's capacity layout [E, C, 64] × [E, 64, 512] | §4 |
| 7 | writes | per in-window read, into bank k | **one global blob + one private blob per processing expert, same key** (xf at one in-window pass), content = the brain's own gradient at the expert input; W = 64 positions per optimizer step | §8 |
| 8 | renewal | in-window benefit + out-of-window geometric (needed: P_win ≤ 0.53 per bank) | every exposure has ≈ 4.9 in-window passes reading the SAME space, so benefit confirmation alone reaches the survival bar at p_hit ≥ 0.5 with ≥ 2 effective passes; geometric renewal becomes an option (off) | §9 |
| 9 | anti-capture | margin, cap at h₀, E_b | margin, **move cap 0.5 h₀**, ≤ 1 move per blob per step and ≤ 8 per 1,000 steps, per-space caps (global 256 / private 2 per expert per step), path length per slot and net travel on 4,096 tracked slots per space type | §9.3 |
| 10 | experts | — | split births copy the parent's private blobs and gate head; dropped (token, expert) pairs get no private read / write; shared expert = space 768 with a write rate matched to the routed experts | §11 |
| 11 | VRAM | 0.73 GB store | 0.99 GB store (private values bf16 with stochastic rounding) + 0.70 GB value-gradient buffers; projected peak ≈ 18.6 GB allocated, 19.0-19.6 reserved | §VRAM |
| 12 | T0 | T0a/b/d/e/g/h | the same with the NEW address, plus **cross-pass findability** and **expert-reuse findability** and the xf drift 6K → 7K | §18 |

**Carried over from 25c unchanged in substance** (each re-stated where it applies): writes from the brain's own surprise
signals only, with per-step budgets (25c §6.2); dedupe / merge with the wrong-merge bound ≤ 2% and the key-weight floor
¼ (§6.4); eviction never takes a slot touched this step, plus the zombie guard (§6.5-6.6); slow strength fade
λ_α = 2.5e-4, strength for eviction only (§8); h₀ calibration on same-fact distances + tracking over the unselected
population + floor (§3.3); confirmation by the centred benefit + key move (§7.3); the survival model
S(T) = exp(−λT e^{−λH}) and re-acquisition (§7.4-7.5); centring on the 5% uniform exploration sample (§4.3); the
post-backward gate update with an idempotent Probe under checkpointing (§4.5-4.6); the value-init ε guard (§6.3);
fill-rate logging (§7.6); no automatic fallback (alerts only); empty_cache before eval and the loop report, the 0.78
fraction cap, 16 × 2 (§VRAM); GO criteria and the 7K stop rule (§16). **Rev 2 replaced several of these**: the
λ_α strength fade (now a benefit EMA), the centred benefit and the centring sample (now the unbiased s ~ U(0,1)
estimator, no centring), the Probe (now the Scale read-out), the zombie guard (log only), the discounted accumulator
(plain Adagrad); see the changelog at the top.

**Dropped from 25c**: per-pass banks and their sizes table, P_win(k)/f_k per bank, W_o^(k), W_ms / the slot-15 write,
the per-pass router one-hot, out-of-window geometric renewal as a default, the 15-bank flags.

---------------------------------------------------------------------------------------------------------------------

**The owner's hard constraints** (status in §20):
1. Addresses come only from the model's thoughts: no token keys; no address part is learned by gradient.
2. No gate loss reaches the thought (the main LM loss may flow through the read: value and match weights).
3. No dataset pointers, no recompute loop outside the training step.
4. The memory is filled during training.
5. ≥ 10k tok/s target (speed track); peak reserved VRAM ≤ 20 GB.
6. The brain decides, at every pass, whether to read — now per (token, chosen expert).
7. **(25d)** Nothing tied to the pass number: every pass uses the same spaces (6 or 40 passes work the same).
8. Self-reinforcement is measured, not forbidden.

---------------------------------------------------------------------------------------------------------------------

## TL;DR

1. **Memory = blobs only.** One **global space** of 256K slots readable by every expert, and one **private space** of
   512 slots for each of the 769 experts (768 routed + the shared expert), preallocated from step 0. A blob = key
   (64-d), content (384-d), strength, age, use / merge counts, clocks. No row table, nothing per pass.
2. **Address** = x̂ = R · (xf − μ)/σ, xf = core.n_m(h) (the RoutedMLP input, the same for all of a token's experts),
   μ, σ ∈ R^384 measured on 64K positions at step 300, R ∈ R^{64×384} a fixed Gaussian matrix. Never learned. The
   router is untouched (it sees xf, not the read).
3. **Search, every pass, no gradient**: global once per token (IVF: 1,024 k-means cells fitted at step 300 on a mix of
   write-candidate and query x̂, refitted every 1,000 steps as an index operation, 16 probed cells, a static
   [1024, 384, 64] × [1024, 64, 512] bmm, bf16 top-8 per probe row then fp32 re-rank → top-16); private for every kept
   (token, expert) pair in one bmm over the capacity layout [E, C, 64] × [E, 64, 512] (C ≈ 96 at 768 experts),
   plus [N, 64] × [64, 512] for the shared expert → bf16 top-8 → fp32 re-rank → top-4.
4. **Gate, per (token, chosen expert), before the expert runs**: ĝ = (a₀ + a_e)ᵀ GELU(W₁φ_t + b₁) + β_e +
   (γ₀ + γ_e)ᵀ m_{t,e}; φ_t (detached) = [xf, side(s), μ_1..4 of the previous pass, global match now, match of the
   previous read, "no previous read" flag, the previous pass's halting confidence, the content-fit score], m_{t,e} =
   this expert's private match. **Read iff ĝ > λ_e** (a causal per-expert threshold that keeps the open rate ≤ 50%;
   λ_e ≥ 0), a static 55% buffer (overflow dropped in token order), plus 5% match-stratified exploration with known
   propensities. ≈ 0.24M parameters (shared trunk 64K, content-fit map 147K, per-expert 39 × 769). Trained after
   backward **only on exploration pairs**, on b̃ = −∂l/∂s for a random read scale s ~ U(0,1) (unbiased for the gain
   l(0) − l(1), control variate (s − ½)κ̂‖r‖²; zero = no gain, no centring); never pushes the thought. **Not counted
   on to remove held-out harm** (T0b: that harm is a generalisation failure invisible on training text, §5.0).
5. **Read**: u_g(t) = Σ_{j ≤ 8} w_j v_j over the 8 best global winners (per token), u_p(t,e) over the expert's 4
   private winners; **r = max(A_g, A_p)(π_g u_g + π_p u_p), π ∝ A** (blend weighted by match), scaled per channel by
   a learned injection scale (init 1) and **added to the expert's buffer row** (and to the shared expert's input) when
   the gate is open. w = softmax(−(d_j − d₁)/(d₁ + 2σ²)) (bounded), A = exp(−d₁/2h₀²). No content fade (λ_v = 0). The
   LM loss trains the contents (row-Adagrad, fp32 grad buffers) and flows through w, A into xf; keys, indices, h₀ get
   no gradient.
6. **Writes after backward**: **global**: W = 64 surprising positions per optimizer step (32 per micro-batch, ≤ 4 per
   sequence, decoupled from W; brain signals only; eligible only without a confident read), one blob each.
   **Private**: each expert (768 routed + shared) has a write budget of 0.256 blobs per step (device credit counter,
   life ≈ 2,000 steps) filled with that expert's most surprising kept tokens. Key = x̂ averaged over the token's
   in-window passes (T0c compares uniform / averaged / maxgrad); content = −(the loss gradient at the expert input)
   summed over those passes (summed over the token's experts for the global blob), normalised to 0.1 × median‖xf‖,
   ‖v‖ capped at 0.3 (spawn) / 0.5 (after) × median‖xf‖. Merge iff d₁ < γh₀² AND cos(−g_new, v_old) > θ; eviction:
   blobs < 2,000 steps old protected (≤ 50% of capacity), else the lowest strength (EMA of positive benefit).
7. **Keeping blobs findable**: a read whose curvature-corrected gain −⟨g, r⟩ + ½κ̂‖r‖² is positive and in the top
   quarter, with a clear winner and cos(−g_t + κ̂r, v_j) > 0.3, confirms its winner: the key moves toward the query
   (α_j = max(0.3, 1/(n_j + 1)), each move ≤ 1σ and ≤ 0.5 h₀; ≤ 1 move per blob per step, ≤ 8 per 1,000 steps incl.
   merges; per-space caps). Late-fact survival S(12.2K) ≈ 0.99 at a cross-document hit rate 0.8 and 2 effective
   passes, 0.89 at 0.5 (§9.4).
8. **Sizes / VRAM**: store 0.99 GB, value-gradient buffers 1.01 GB (all fp32); static workspaces; projected peak
   ≈ 19.0 GB allocated, ≈ 19.3-19.7 GB reserved at 16 × 2 (old run 20.1 / 20.5-21.1; the rows' 3.5 GB are gone);
   `chunk_ce` + checkpointed concept loss as defaults if T2 (≥ 200 steps incl. `evaluate()` and a birth) needs them.
9. **Cost**: realistic ≈ 70-100 ms per step for the whole memory, launch-bound (≈ 2,100 kernel launches / step), ≈
   3.6-5.1% vs a no-memory build (reported); vs the old run net ≈ 2.5-17% FASTER (the rows cost ≈ 150-400 ms): the
   decided bar (≥ 0.97 × the old run's tok/s) is met with margin. No host sync added. Triton 3.6 (installed) fused
   kernels behind speed flags.
10. **Before any run**: GO before writing gate / write code = T0a paraphrase, T0c pass matrix (incl. the averaged key)
    and the ORACLE CEILING; then T0d IVF recall on write-distributed keys, drift 3K → 7K, gradient agreement, private
    reach, hubness, template collisions (64-d vs 384-d re-rank), T0h, T0e; T0b + T2 + a 1K pilot on the GPU when the
    speed study releases it. GO vs the old run (late 0.280 @7K, rare 0.016, harm +0.13) as in §19.
11. **Size**: ≈ 3,000 lines (memory ≈ 1,500, tests and checks ≈ 1,200, eval ≈ 150, trainer ≈ 150).

---------------------------------------------------------------------------------------------------------------------

## 0. Evidence that shapes the design (25c §0, re-read for 25d)

| finding (salvage, `world_mem_life_12k` 6K → 7K) | number | consequence for 25d |
|---|---|---|
| rows carried the recurring facts | late 0.280 @7K, 0.182 knocked out; control final 0.260 | the blobs must carry them (no backup): renewal (§9) and fill (§9.6) are first-order |
| rare facts | early 0.016 @7K, control 0.013 | the write-at-surprise store's home ground; GO asks ≥ 0.035 |
| reads hurt other tokens | other world +0.133 nats on 83% of world tokens | per-(token, expert) gate (§5) + A mutes far matches |
| template keying of the rows | same template other person 0.24 vs same person other relation 0.045 | T0a with the new address is the first gate |
| self-recall 1K gap | R@1 0.77 at 4K keys, 0.57 at 32K; world 0.525 (4K), 0.350 (16K) | density costs ≈ 0.07 R@1 per doubling: the 256K global space relies on renewal (young keys) and on A muting wrong winners; T0d measures at real density |
| per-pass views drift between passes | same-pass 0.777, pass-1 bank from pass 6: 0.401 (workspace view) | **the cross-pass risk of 25d** (one space for all passes): T0c measures it with xf |
| routing noise | 0.3% effect | no pinning; but expert reuse across exposures is a new question (T0r) |
| drift | median drift / 1-NN gap 1.22-1.31 per 1K steps | h₀ of the order of the drift; key moves on confirmed recall |
| **T0b (rev 2; old rows as proxy, ckpt 7K; `speed_wip\splat\realcheck\t0b\t0b_results.md`)** | on training windows the reads help every token class (world_other_all +0.020 / +0.029 nats); the harm (−0.07 … −0.12 on non-answer world) appears only on held-out templates / documents (val_world); gate AUC for the sign of benefit 0.52 train / 0.61 val / 0.49 train → val; features predict magnitude (0.68-0.74) and "answer next" (0.987), not sign; match-quality features add nothing; the first-order gradient target c correlates 0.22 with knockout benefit and misses the val answer gain | **the harm is a generalisation failure, not a gating failure**: a gate trained on training-time benefit cannot remove it. The remedy is read robustness across wording (T0a is the decisive GO gate); the gate stays (owner design) but is not counted on for the harm bar; the gate target is the unbiased s ~ U(0,1) estimator only; T0b is repeated on the blob pilot (§5.0, §18.2) |

What is new with xf as the address (outline §2b): xf already contains the token, all earlier passes' thought, the
workspace read (concept feedback, earlier write-backs), the previous pass's concept write-back (rank 64) and the
context (SSD membranes). In 25d there is no separate memory inject any more: the previous pass's memory reads reach xf
through the experts' outputs (o → workspace write and x). Not in xf: this pass's concepts / workspace update (they come
after the MoE), the full μ_l, the per-level uncertainty, match quality, side(s): the gate gets μ_l, side(s) and match
quality explicitly (§5.1).

---------------------------------------------------------------------------------------------------------------------

## 1. Architecture: one core pass k (every pass the same)

```
 run(): pass k  ──  ctx = {k, in_window = k ≥ K−bptt, prev_mu, prev_halt (pass k−1, detached), prev_match, train}
 Core.forward:  r = read(s, x);  u = n_u(x + n_e(e) + W_ro r + inject);  h = u + mix(u);  xf = n_m(h)
 RoutedMLP.forward(xf, side = side(s), bmem = blobmem):
   route(xf) → idx, gate weights → capacity layout (e_k, p_k, t_k), C        [unchanged; ONE existing host sync]
   buf[e_k, p_k] = xf[t_k]
   ┌─ blobmem (no_grad part; all workspaces static, preallocated from N·k, E·C) ───────────────────────────┐
   │ x̂ = R (xf − μ)/σ  (sg for the search)                                                                 │
   │ GLOBAL search, per token: centroid scores → 16 cells → static cell layout bmm → bf16 top-8 per probe   │
   │        row → fp32 re-rank → top-16 (j, d)                                                              │
   │ PRIVATE search, per kept pair: Q[E, C, 64] bmm P̃[E, 64, 512] → bf16 top-8 → fp32 → top-4; shared same  │
   │ match features (global per token, private per pair), content-fit scores                               │
   │ GATE: z_t = GELU(W₁ φ_t + b₁);  ĝ_{t,e} = (a₀+a_e)ᵀ z_t + β_e + (γ₀+γ_e)ᵀ m_{t,e}                     │
   │       open = (ĝ > λ_e) ∪ Explore(5%, match-stratified);  overflow of the 55% buffer dropped in token    │
   │       order;  exploration pairs get s ~ U(0,1), all other open pairs s = 1                             │
   └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   READ (with grad): w, A from x̂ (grad) and the detached winner keys; u_g(t), u_p(open pairs) by GatherRead;
                     r = ζ ⊙ max(A_g, A_p)(π_g u_g + π_p u_p)   (ζ ∈ R^384 the injection scale, init 1)
   SCALE (leaf s per open pair, no grad to anything else; bwd stores b = ∂L/∂s = ⟨∂L/∂buf, r⟩ and the parts
          ⟨∂L/∂buf, r^g⟩, ⟨∂L/∂buf, r^p⟩ per open pair)
   buf[e,p] += s · r(t,e)  for open pairs;   x_sh = xf + open_sh · s · r_sh
   h_e = spike(buf @ w_up) … out = h_e @ w_down;  y = Σ gate·out + shared(x_sh)
 Core: s = write(s, m + o); x update      →  run(): ConceptHier on s → μ_l (next pass's prev_mu), think feedback…

 after each micro-batch's backward: gate update from the EXPLORATION pairs' (φ, m, s, b̃) only; content grads
                                    accumulated (fp32); write candidates' gradients summed over their in-window
                                    passes by hooks on buf / x_sh (hooks exist only on in-window passes)
 after opt.step: gate AdamW; content row-Adagrad (+ ‖v‖ cap); confirmations → key moves; merges / spawns /
                 evictions; global cell index rebuilt (centroids refitted every 1,000 steps); EMAs (h₀, σ², κ̂, Ĉ)
```

- **Gradient reaches**: blob contents (row-Adagrad on dense fp32 grad buffers), xf (hence the thought) through w and
  A, the injection scale ζ, the experts (as usual, now also through their read-augmented input), the gate
  (post-backward, its own loss only, exploration pairs only).
- **No gradient reaches**: keys, the map (μ, σ, R), the IVF centroids, h₀, σ², κ̂, strengths, clocks, indices, the
  thought through the gate (φ and m are detached; the gate's loss is computed on stored detached copies after
  backward). The read scale s is a leaf whose gradient is only read out (it updates nothing).
- **Parameter hygiene**: keys, contents, strengths, clocks and the IVF index are registered **buffers**, not
  parameters. The memory's few parameters (ζ; the gate) live in their own optimizer groups and are kept out of
  `make_opt`, `carry_optimizer` (births) and the brain's global grad clip; a unit test asserts that `brain_params`
  contains no memory tensor.

---------------------------------------------------------------------------------------------------------------------

## 2. The address

### 2.1 What it is
- xf = core.n_m(h) at pass k, the tensor `RoutedMLP.forward` receives (`loop_brain.py` line 235), taken BEFORE any
  memory is added (the read goes into `buf` and the shared expert's input, never into xf itself). All of a token's
  routed experts receive the same xf row, so the global search runs once per token per pass.
- **Fixed map**: x̂ = R · ((xf − μ) ⊘ σ), μ, σ ∈ R^384 the per-channel mean and standard deviation of xf over 64K
  positions pooled over all passes of the training forwards of steps 280-300 (a running sum, no host sync; σ floored
  at 1e-3 × its median), R ∈ R^{64×384} with i.i.d. N(0, 1/384) entries from a fixed seed on a forked generator.
  Then E‖x̂‖² ≈ 64 (unit variance per coordinate) and a pairwise squared distance in the standardised space is
  preserved up to a factor distributed as χ²₆₄/64 (sd 0.18): within ≈ ±35% for 95% of pairs, i.e. distances within
  ≈ ±17% (Johnson-Lindenstrauss at m = 64). Never refitted.
- The same map for the global space and every private space.
- Before step 300: no search, no read, no write; the gate's match features are 0 with an "empty" flag.

### 2.2 The n_m gain (a known drift source, measured)
xf includes the learned RMSNorm gain of n_m. A gain change moves every key and query together (a diagonal rescale
before the fixed standardisation). It is part of "drift", handled by confirmation moves and h₀ tracking. T0f splits
the 6K → 7K drift of x̂ into the gain part and the rest; if the gain part is > 30% of the median drift, the owner is
told and `--gm-addr pregain` (x̂ from xf ⊘ g_m, the input to the gain) is offered. Default: xf as specified.

### 2.3 Why a fixed random projection is enough
JL keeps neighbourhoods at 64-d for the distances the kernel cares about (d ≲ 4h₀²); the per-channel standardisation
stops a few high-variance channels from dominating. A learned map is excluded by constraint 1; a PCA map was not
chosen by the owner (it would also be "fixed", an option for T0a diagnostics only: `--gm-map pca`).

---------------------------------------------------------------------------------------------------------------------

## 3. The store

### 3.1 Slot layout

| field | global (per slot) | private (per slot) | notes |
|---|---|---|---|
| key k_i (64) | fp32, 256 B | fp32, 256 B | moved only by rules |
| scoring copy k̃_i (64) | bf16, 128 B | bf16, 128 B | refreshed when k_i changes |
| ‖k_i‖² | 4 B | 4 B | |
| content v_i (384) | **fp32**, 1,536 B | **bf16, 768 B** (stochastic rounding on update) | model width, no output matrix |
| Adagrad accumulator S_i | fp32, 4 B | fp32, 4 B | row-wise |
| log α, Λ^α, Λ^v | 12 B | 12 B | strength (eviction), value-fade clock |
| τ (key time), t^conf, t^touch | 12 B | 12 B | age, zombie guard, eviction protection |
| m (maturity 0..3), n (merges ≤ 8) | 2 B | 2 B | |
| u (uses), b̄ (benefit rank EMA), path length, moves-in-window | 16 B | 16 B | anti-capture / logs |
| cell id (global only) | 4 B | — | IVF |
| **total** | **≈ 1,970 B** | **≈ 1,200 B** | |

- **Global**: 262,144 slots × 1,970 B = **0.52 GB**.
- **Private**: 769 spaces × 512 = 393,728 slots × 1,200 B = **0.47 GB** (preallocated for 768 routed + the shared
  expert at index 768, as the old private memory preallocated `Emax`; fp32 contents would be 0.78 GB).
- **Value-gradient buffers** (dense, so backward is one `index_add_` with no compaction and no host sync), **fp32
  everywhere** (rev 2, H4; bf16 accumulation of many small contributions loses them, the shared expert's row in
  particular): global 256K × 384 × 4 B = 0.40 GB; private 393,728 × 384 × 4 B = 0.61 GB; + a touched-row bitmask.
  Zeroed only on touched rows after the Adagrad step. Private CONTENTS stay bf16 with stochastic rounding.
- **IVF index** (global): 1,024 centroids (64-d fp32, 256 KB) + a cell table [1024, 512] int32 (2 MB), rebuilt every
  step; centroids refitted every 1,000 steps (§4.1).
- Bookkeeping, free lists, the dummy slot for masked writes, the per-expert write credit counters: < 10 MB.
- **Total store + grads ≈ 2.00 GB.** Option (pending, off): a probation store for candidate blobs, +0.26 GB (§8.7).

### 3.2 Capacities, write rates and lives (all per OPTIMIZER step)
- Write positions W = 64 per step (32 per micro-batch).
- **Global**: 64 writes per step at most (merges reduce spawns). Life of an unconfirmed blob = 262,144 / 64 =
  **4,096 steps** without merges; full by ≈ step 300 + 4.1K ≈ 4.4K. 25c chose life ≈ the drift horizon (≈ 2.3K)
  for density; 25d's 256K is the owner's fixed choice, so life is ≈ 2 horizons: unrenewed keys older than ≈ 2.5K
  steps are rarely found again (R@1 ≈ 0.15 at 3K, 25c §7.1) and are the first to be evicted by strength anyway.
  (The reviewer questioned the 4,096-step life; the rebuttal was accepted: life ≈ 4,096 steps at W = 64 stays.)
- The recurring-fact **reserve** forms by itself (confirmed slots outrank unconfirmed ones at eviction): 20.4K late
  facts × ≈ 1.25 keys/fact ≈ 26K slots, up to ≈ 100K if a fact keeps one key per pass region (T0c); both fit
  in 256K, leaving (256K − reserve)/64 ≈ 2.5-3.6K steps of unconfirmed life.
- **Private (every expert, routed and shared; DECIDED, D2 / D3)**: a fixed per-expert write budget
  b = 512 / 2,000 = **0.256 blobs per step**, so an unmerged, unconfirmed private blob lives ≈ **2,000 steps at any
  E_live** (rev 1's "every processing expert" gave 260-1,050 steps, below the drift horizon). The budget is a device
  credit counter per expert (§8.4): credit_e += b/2 per micro-batch; when credit_e ≥ 1 the expert writes ONE blob,
  its most surprising kept token of that micro-batch, and credit_e −= 1. Total ≈ 769 × 0.256 ≈ 197 private writes per
  step (static buffer of 769 per micro-batch, masked). The shared expert (space 768) has exactly the same budget as an
  average routed expert. Merges lengthen lives; confirmation keeps found blobs. A newly born child expert starts with
  credit 0.
- **Lifetime logged per bank** (every 250 steps): per private space, the mean age at eviction, the live / protected /
  confirmed counts and the realised write rate; summarised as min / median / max over experts and the fraction of
  spaces whose median eviction age is < 1,000 or > 4,000 steps (broken-if, §16.1).

### 3.3 Recall vs density (why A and renewal matter at 256K)
The salvage self-recall extrapolates to R@1 ≈ 0.40 / world ≈ 0.10 at 256K for an unrenewed 1K-old key (−0.07 /
−0.09 per doubling from 32K / 16K). Two things change the picture in 25d: (i) renewed late facts are queried young
(every ≈ 105 steps), where drift is a fraction of the 1K drift; (ii) a wrong winner far from the query gets A ≈ 0
and adds little. Rare facts (age ~ Exp(1.3K)) suffer the density; private spaces (512 keys, ≈ 4K-key density
behaviour) give them a second, low-density chance when the cue routes to the same expert (T0r). T0d measures recall at
the real write density with the real IVF.

---------------------------------------------------------------------------------------------------------------------

## 4. Search (no gradient; every pass; host-sync free)

### 4.1 Global: IVF over 1,024 fixed cells with a static capacity layout
Brute force is the reference, not the default: 8,192 queries × 262,144 keys × 12.8 pass-forwards = 27.5 × 10⁹
scores per step; the bf16 GEMM is only 3.5 TFLOP (≈ 25-45 ms) but materialising and reducing the scores moves
≈ 110 GB (≈ 110-160 ms, 5.6-8% of the step). Triton 3.6 is installed (rev 2): the IVF stays the default; a fused
Triton IVF score + top-k kernel (and the private search, §4.2) is a speed-track item behind `--gm-triton 1`, checked
bit-for-bit on the top-16 against the torch path. The inverted file:

1. **Cells** (rev 2, H3): 1,024 centroids c_1..c_1024 = k-means (10 Lloyd iterations, k-means++ init on the forked
   generator) fitted at step 300 on a **50/50 mix of write-candidate x̂ (the positions the write criterion selects,
   with their averaged keys, §8.2) and query x̂ (all positions, all passes)**, 64K samples: the cells follow where
   keys actually are, not only where queries are. **Refitted every 1,000 steps** from a rolling reservoir of the same
   mix (an index operation: centroids and cell ids change, keys do not; allowed by constraint 1). They are an index,
   not the address: keys and queries stay x̂.
2. **Cell table**: each live global slot has cell(i) = argmin_c ‖k_i − c‖² (recomputed when its key moves, and for all
   slots at a refit). Once per optimizer step the table [1024, Cc = 512] is rebuilt: argsort of the 262,144 cell ids,
   offsets by `index_add_` of ones + cumsum (no `bincount`: it syncs to size its output), position = rank − offset.
   **A full cell spills**: a slot whose nearest cell already holds 512 slots is assigned to its 2nd-nearest cell
   (stored as cell id + a "spilled" bit; logged); if both are full, the spawn evicts the space's weakest (§8.6).
   Padding = −1 (score −∞). ≈ 1-2 ms.
3. **Query layout** (per pass): centroid scores x̂ · cᵀ (8,192 × 1,024, 1.1 GFLOP) → top-**n_p = 16** cells per token
   → 131,072 (token, probe) pairs, sorted by (cell, probe rank) → position within the cell; static capacity Cq = 384
   per cell (3 × the mean 128); pairs beyond Cq are dropped, farthest probes first (logged as "probe overflow").
4. **Scores**: one bmm [1024, 384, 64] × [1024, 64, 512] in bf16 (201M scores, 25.8 GFLOP, 402 MB), −‖k‖² added →
   **bf16 top-8 per (token, probe) row** → scattered back to (N, 16 × 8) → top-32 over the 128 → **fp32 re-rank** of
   these 32 candidates (gathered fp32 keys, exact d) → (j₁..j₁₆, d₁ ≤ … ≤ d₁₆) per token. The read blends the best 8;
   all 16 feed the match features, hubness and the confirmation margin.
- **Recall argument**: a key within distance √d of the query is missed only if its cell is not among the query's 8
  nearest centroids. For matches that matter (d ≲ 2h₀², i.e. A ≳ 0.37) and cells of radius ≫ h₀ the key's cell is
  the query's nearest or a neighbour; misses concentrate at far matches, which A mutes anyway. Measured in T0d against
  brute force; monitored online (1K sampled queries every 250 steps vs a chunked brute-force scan; target miss of the
  fp32 top-1 ≤ 2% for d₁ < 2h₀², else n_p = 32 or a refit-interval of 500).
- **Drift of the cell balance**: between refits thoughts drift, so cells may unbalance (more probe overflow, spills,
  hot cells churning). Logged: cell occupancy Gini, max cell fill, spill rate, probe overflow rate. **Alert** if
  overflow > 1% of probes, spills > 5% of live slots, or Gini > 0.6 (alert only; the 1,000-step refit is the
  built-in remedy, `--gm-ivf-refit 1000` default, 0 = never).
- `--gm-search brute` (chunked bf16 + block max, 25c §2.4) stays for tests, T0d and small stores.

### 4.2 Private: one bmm over RoutedMLP's capacity layout
- Q = zeros(E, C, 64) (a static preallocated workspace); Q[e_k, p_k] = x̂[t_k] (the same layout as `buf`); scores =
  Q @ P̃[:E]ᵀ (bf16 [E, C, 512]) − ‖P‖² → bf16 top-8 over 512 at the kept slots only → fp32 re-rank of the 8
  (gathered fp32 keys) → top-4. Triton option (speed track, `--gm-triton 1`): a fused score + top-8 per row.
- C = min(max load, ⌈1.5 N k / E_used⌉): 96 at 768 experts, 384 at 192 — E·C ≈ 73.7K rows either way, so the cost is
  constant through growth: 37.7M scores (4.8 GFLOP, 75 MB) per pass. The shared expert: [8,192, 64] × [64, 512]
  (4.2M scores).
- Eval uses dropless capacity (C = the max load, ≈ 2-3 × the mean): the bmm grows by that factor; still < 2 ms.

### 4.3 Match features (from the search, detached)
- Global, per token (7): A_g = exp(−d₁/2h₀,g²), d₁/h₀,g², log(d₂/d₁), m_{j₁}/3, log(1 + age_{j₁}), empty flag, the
  content-fit score f_g = sigmoid(n(xf) · n(W_cf u_g)) (rev 2, coordinator default b).
- Private, per pair (7): the same for the expert's own space (h₀,p), with f_p = sigmoid(n(xf) · n(W_cf u_p)).
  W_cf ∈ R^{384×384} is a gate parameter (trained by the gate loss only; xf and u are detached); n(·) = L2
  normalisation.
- Previous read, per token (6): pass k−1's A_g, log(d₂/d₁)_g, d₁/h₀,g², the mean and max A_p over the token's pass-(k−1)
  pairs, the fraction of its pass-(k−1) pairs that read; zeros + the **"no previous read"** flag when there was none
  (pass 0, or no pair of the token opened at k−1; rev 1 called it "pass 0": renamed because it is not about the pass
  number).

---------------------------------------------------------------------------------------------------------------------

## 5. The gate (per token, per chosen expert; decides before the expert runs)

### 5.0 What the gate can and cannot do (rev 2, from T0b)
T0b (old rows as the proxy read, ckpt 7K) found that on **training** windows the reads help every token class, and that
the +0.13-nat harm on non-answer world tokens appears only on **held-out** templates and documents (val_world). A gate
fitted on training-time benefit scores AUC 0.52 for the sign of the benefit on training text, 0.49 when applied to
held-out text, and would leave ≈ 74% of the harmful held-out reads open (it removes about a third of the harm at
best). The features predict how much a read matters (0.68-0.74) and that an answer comes next (0.987), but not
whether it helps; the match-quality features added nothing.

**Stated plainly: a gate trained at training time cannot remove harm that exists only on held-out text**, because on
the text it is trained on there is no harm to learn. The remedy is a read that is robust across wording: keys that find
the same fact under new templates and do not fire on other facts of the same template (T0a, the decisive GO gate,
§18.0), contents that generalise (the oracle ceiling G3 measures held-out harm with the gate forced open), A muting far
matches, and the held-out harm metric B22 (§16.1). The gate stays (the owner's design: the expert decides, and it bounds
cost at 50%), its target is the unbiased s ~ U(0,1) estimator only (§5.3), and the plan **does not count on it** to
meet the harm bar (≤ +0.04 nats, §19.2). Match quality stays a gate input (outline §2b, owner decision); T0b on the
blob pilot re-measures whether it adds anything with the new address. T0b is repeated on the blob pilot's real store
(§18.2) before the 12.2K run.

### 5.1 Inputs (all detached)
φ_t (token-level, 1,999): xf (384) ‖ side(s) (64, the router's workspace summary, `Core.side`) ‖ μ_1..μ_4 of the
previous pass (4 × 384 = 1,536, as produced by ConceptHier after pass k−1; zeros when there is none) ‖ global match
now (7, incl. f_g) ‖ previous read's match (6) ‖ "no previous read" flag (1) ‖ **the previous pass's halting
confidence** (1: sigmoid of the existing `halt_head` output after pass k−1, detached; the old run has `halt 1`; 0 +
the flag when there is none). Each group is RMS-normalised per token (xf already is). m_{t,e} (pair-level, 7): the
expert's private match now, incl. f_p.
- The two extra inputs (halting confidence, content-fit score) are the reviewer's proposal; the coordinator's default
  is **ON** (owner may override: `--gm-gate-feats spec` drops both). Both are detached, so constraint 2 holds.
- Options (off): `--gm-gate-feats +cu` (the 2L per-level uncertainty, outline §2b lists it as "not in xf" but not as a
  gate input) and `+pass` (a pass embedding: against constraint 7's spirit, ablation only).

### 5.2 Parameterisation for 769 experts (shared trunk + per-expert head)

      z_t      = GELU(W₁ φ_t + b₁),                          W₁ ∈ R^{32 × 1999}         (shared trunk, 64K)
      ĝ_{t,e}  = (a₀ + a_e)ᵀ z_t + β_e + (γ₀ + γ_e)ᵀ m_{t,e},  a_e ∈ R^32, β_e, γ_e ∈ R^7  (per expert, 40 × 769 = 31K)
      + W_cf ∈ R^{384×384} for the content-fit scores (147K)

- Shared parts carry what is common ("is there a near blob", "is this an answer-like position"); a_e, γ_e, β_e let
  an expert weigh them differently (an expert for numbers may benefit from memory where a syntax expert does not).
- **Init** (rev 2): W₁ ~ N(0, 1/1999), a₀ = a_e = γ = 0, W_cf = I, **β_e = β₀ slightly negative** (β₀ = −0.02 in
  target units, i.e. −0.02 nats per token): ĝ < λ_e = 0 → nothing reads except the exploration sample until the gate
  has seen a positive gain (a 0-init gate would sit exactly on the threshold and open on noise). a₀ and β get
  gradient first, then the trunk.
- **Shrinkage**: AdamW weight decay 0.01 on a_e, γ_e (toward the shared head); none on β_e. An expert with few tokens
  stays close to the shared gate.
- **Cost per pass**: trunk 2 × 1,997 × 32 × 8,192 = 1.05 GFLOP; the per-pair head is a 32-d dot + 6-d dot on ≈ 49K
  pairs (+ 8,192 for the shared expert). Building φ (33 MB bf16) dominates. ≈ 0.6-1.0 ms per pass.
- Alternatives (flags): `--gm-gate lowrank` (ĝ = z_tᵀ U V_e with rank 8 per expert), `--gm-gate shared` (no
  per-expert head; ablation measuring what per-expert heads add).

### 5.3 Target: an unbiased estimate of the read's gain, on exploration pairs only (rev 2, H1)
Rev 1 regressed ĝ on the median-centred first-order probe −⟨g, r⟩ over all open pairs. The reviewer showed three faults:
the first-order probe at s = 1 is biased by the curvature (it measures the slope at the read, not the gain of the read);
median centring makes "gain 0" mean "median gain", so half of all pairs always look useful; and training on gated pairs
mixes the gate's own selection into its target. Fix:

- **Random read scale on exploration pairs.** Every exploration pair (§5.4) injects s · r with a leaf scalar
  s ~ U(0, 1) drawn per pair (device generator, cached for recompute). Let l(s) be the loss as a function of that
  pair's scale. Then

      b̃ = −∂l/∂s |_s = −⟨∂L/∂buf[e,p], r⟩      and      E_s[b̃] = −∫₀¹ l′(s) ds = l(0) − l(1)

  exactly (no Taylor approximation): b̃ is an **unbiased estimate of the gain of reading** (positive = the read
  lowered the loss).
- **Control variate**: for a locally quadratic l, l′(s) = l′(0) + s κ‖r‖², so b̃ varies with s by −(s − ½)κ‖r‖² around
  its mean. b̃_cv = b̃ + (s − ½) κ̂ ‖r‖² keeps the mean (E[s − ½] = 0, s independent of everything else) and removes
  that variance. κ̂ = per-space-type EMA (0.99) of the least-squares slope of −b̃ on s‖r‖² over the step's exploration
  pairs (logged; a non-positive κ̂ is clipped at 0 and alerts).
- Target: y_{t,e} = N_tok · b̃_cv (nats per token). **No centring: y = 0 means no gain.** The sign of ĝ is meaningful.
  **This is the gate's only target** (rev 2, T0b): no first-order probe at s = 1 and no centred c enter the gate loss.
  T0b found that the first-order probe c correlates only 0.22 with the knockout benefit and misses the answer-cue gain
  (≈ 0 vs +0.29 nats on val_world); b̃ integrates the effect from s = 0 to 1 exactly in expectation. The curvature-
  corrected estimate of §7 is used only for confirmation and strength, and the pilot's T0b repeat checks its agreement
  with knockout (§18.2).
- **The gate trains ONLY on the exploration pairs** (≈ 2,460 routed + 410 shared per micro-batch-pass): their
  selection is independent of ĝ (known propensities, §5.4), so the regression is unbiased for E[y | φ, m] without
  importance weighting beyond the stratum weights.

      L_r = λ_r · Σ_{explore pairs} ω_{t,e} Huber_δ(ĝ − y) / Σ ω,    ω = 1/π(stratum),   λ_r = 0.1
      δ = max(N_tok · 1.4826 · MAD(b̃_cv), δ_min),   δ_min = 1e-3 nats;   MAD floored at 1e-4 nats

- Out-of-window reads use ĝ but do not train it (no backward there).
- **Placebo (logged only, never a target)**: for 1% of exploration pairs a random live blob's content v_rand (same
  space, rescaled to ‖r‖) gives −⟨g, r − r_rand⟩: the gain of the actual content over an arbitrary one of the same
  size. Its mean should be > 0; ≤ 0 means the gain is a generic perturbation, not memory (broken-if, §16.1).

### 5.4 Selection (training): a causal threshold (rev 2, H2)
Rev 1 capped each expert at 50% by ranking ĝ over the expert's tokens in the batch. A rank over the batch lets a later
position's ĝ change whether an earlier position reads (a leak from the future inside the sequence, and a train / eval
mismatch). Rev 2 uses a per-expert threshold that depends only on past steps:

      open_{t,e} = ( ĝ_{t,e} > λ_e ) ∨ explore_{t,e}
      after each optimizer step:  λ_e ← [ λ_e + 0.01 · (rate_e − 0.5) ]₊     (λ_e starts at 0; units of ĝ)

- rate_e = the fraction of expert e's kept in-window non-exploration pairs with ĝ > λ_e in that step (device EMA-free
  update, no sync). λ_e = 0 while the expert opens < 50%, so "read where the predicted gain > 0" holds exactly; it
  rises only when the expert would open more than half its tokens (the decided speed safety). `--gm-lambda pooled`:
  one λ for all experts (option). The shared expert has its own λ_sh.
- **Static 55% buffer**: the compact read buffer has n_max = ⌈0.55 n_keep⌉ routed rows (+ ⌈0.55 N⌉ shared), sized on
  the host from n_keep (known). If a step's open pairs exceed it (λ lags by one step), the overflow is **dropped in
  token order** (latest positions first, via a cumsum over a position-sorted mask), which is causal: an earlier
  position's read never depends on a later one. Overflow rate is logged (broken-if if > 1% of open pairs).
- **Exploration (match-stratified, known propensities; rev 2)**: exactly ⌈0.05 n_keep⌉ routed pairs and ⌈0.05 N⌉
  shared pairs per pass, drawn without replacement inside 4 strata of the global match A_g (quartile edges from the
  previous step's EMA): the near-match strata are over-sampled (target shares 0.15 / 0.20 / 0.30 / 0.35 from far to
  near) so the gate sees enough pairs where memory can matter; each pair's propensity π = (its stratum's draw count) /
  (stratum size) is stored and the gate loss weights by 1/π (§5.3). Static sizes: per stratum a top-k of uniform
  randoms with a host-known k. Exploration pairs read regardless of ĝ, with s ~ U(0,1).
- **Why it does not collapse**: all-open is impossible (λ rises above a 50% rate); all-closed is impossible (the
  exploration always reads and trains the gate); there is no centring that could move the zero.
- **Eval / inference**: read iff ĝ > λ_e (the trained thresholds, frozen), s = 1; no exploration; no target, no
  writes. Causal by construction.
- Options: `--gate-cost <nats>` / `--gate-cost time` (an added constant inside the threshold), `--gm-gate-budget
  fixed` (debug only).
- **Exploration warm-up** (20% → 5% over steps 300-1,300): reviewer's proposal; coordinator default **OFF** (owner may
  override, `--gm-explore-warm 0.2,300,1300`). Instead the stratification above, and a log of how many steps the
  gate's **decile slope** (the slope of mean y over ĝ deciles on held-out exploration pairs, normalised so 1 =
  calibrated) takes to reach 0.7.

### 5.5 Post-backward update and the idempotent Scale read-out (25c §4.5-4.6, adapted)
1. Forward (each in-window pass): keep φ, m, s and π of the **exploration pairs only** (≈ 2.9K pairs × 2,000 bf16 ≈
   12 MB per pass, ≤ 0.07 GB for 6 in-window passes), and for all open pairs only what confirmation needs (slot ids,
   ‖r‖², the part norms), until after the micro-batch's backward.
2. Backward: the Scale function (§6.3) writes b = ⟨∂L/∂buf, r⟩ and the parts ⟨∂L/∂buf, r^g⟩, ⟨∂L/∂buf, r^p⟩ into
   `bbuf[mb, slot(k), :n_max]` with `index_copy_` (idempotent) and bumps `cnt[mb, slot(k)]`.
3. After `backward()` of the micro-batch: y (§5.3), then a SEPARATE tiny autograd graph: ĝ recomputed from the kept
   φ, m with the gate parameters, L_r (y fixed) → `backward()` into the gate parameters only (÷ accum). The gate's
   AdamW steps once per optimizer step, after both micro-batches, so the forward ĝ used the same parameters: the
   gradient is exact. (25c's closed-form gradient was for a linear router; with a trunk the recompute is simpler and
   costs ≈ 1-2 ms per micro-batch.)
4. The gate is in its own AdamW group (lr 1e-3) with its own clip (1.0), excluded from the brain's global clip
   (`train_lm.py` lines 1388-1391 already do this for `gate`).
- **Checkpointing**: the core pass is not checkpointed today, but the per-pass checkpoint is a VRAM lever in reserve.
  Under it the RoutedMLP forward would run twice: the search results, the exploration draw and the open masks are
  cached per (micro-batch, pass) on the first run and reused on the recompute (the generator is not advanced again);
  the Scale function does nothing extra in forward and writes only in backward into a fixed slot, so a recompute
  cannot double b. The s draws are cached with the masks.
- **Test `test_gm_gate_postbwd`** (CPU, §18.4): (a) cnt == 1 for every in-window pass, 0 outside; (b) b equals
  ∂L/∂s by autograd with checkpointing off; (c) on a toy quadratic loss, the mean of b̃ over many s draws equals
  l(0) − l(1) (unbiasedness) and the control variate lowers its variance; (d) the recomputed gate gradient equals
  autograd of L_r on the main graph with y fixed (rtol 1e-5); (e) accum 2 and 4 give the full-batch gradient;
  (f) value and query gradients equal the non-checkpointed ones when the core pass is wrapped in `checkpoint`;
  (g) non-exploration pairs contribute nothing to the gate gradient.

### 5.6 Watchdog (alerts only; no automatic fallback)
Every eval: AUC of ĝ vs 1[y > 0] on a held-out micro-batch's exploration pairs (pooled, and for the 32 busiest experts),
the decile slope (§5.4) and the steps until it first reached 0.7, open rate (train / eval) by pass and by token type
(eval tags), λ_e distribution (the fraction of experts with λ_e > 0), buffer overflow, the fraction of experts whose
open rate is pinned at 0 or at 50%, κ̂, the placebo mean. Alert after 0.3·S if the pooled AUC < 0.60 at two
consecutive evals or the pooled open rate is pinned at 5% or 50% for 500 steps. No automatic fallback (a fixed
top-25% read fallback was proposed and rejected: alert only, §OD-R).

---------------------------------------------------------------------------------------------------------------------

## 6. The read and the inject

### 6.1 Kernel (per space; 25c §3.2)

      w_j = softmax_j( −(d_j − d₁) / (d₁ + c·h₀²) ),   c·h₀² ≡ 2σ²     (bounded match kernel; d = squared distance)
      σ²  = EMA_0.99 of  median_{exploration reads}(d₂ − d₁) / (2 ln 2),   floored at 0.05 h₀²   (per space type)
      A   = exp(−d₁ / (2 h₀²))                          (h₀,g for global, h₀,p for private)
      u   = Σ_j w_j v_j        global: j = 1..8 of the top-16;   private: j = 1..4

- **Why this kernel** (rev 2, reviewer): rev 1's (d_j/d₁ − 1)/τ_w divides by d₁, so a query sitting on a blob
  (d₁ → 0) gives a one-hot w and an unbounded match-weight gradient ∝ 1/d₁. The denominator d₁ + 2σ² ≥ 2σ² bounds both;
  σ² is set so that, for a query on a blob, the median runner-up gets half the winner's weight
  (exp(−median gap / 2σ²) = ½); for far queries the weights flatten in proportion to d₁ (relative distances).
- **No content fade** (rev 2: λ_v = 0, f ≡ 1). Contents change only by gradient and merges; strength fades for eviction.
- u_g(t) is computed once per token (for all N tokens: an 8-row gather, ≈ 100 MB, cheaper than compacting); u_p(t,e)
  for the open pairs only (compact layout, ≤ n_max rows).
- Strength is not in the kernel (eviction only), as in 25c.
- **Blend of global and private (DECIDED, D5: weighted by match)**:

      r_{t,e} = ζ ⊙ max(A_g, A_p) · (π_g u_g(t) + π_p u_p(t,e)),     π_g = A_g / (A_g + A_p + ε),  π_p = 1 − π_g

  The better-matching space dominates; the total scale is that of the best match (rev 1 summed r_g + r_p, which doubled
  the read when both spaces held the same fact and let a poor match in one space add noise to a good one). ζ ∈ R^384 is
  a per-channel injection scale (init 1, trained by the LM loss in the memory's own group, lr 1e-3, no weight decay).
  For the confirmation credit, r^g = ζ ⊙ max(A) π_g u_g and r^p = ζ ⊙ max(A) π_p u_p (r = r^g + r^p).
- **One gate decision per (token, expert) covers global + private (DECIDED, D4)**; the gate sees both match qualities.
  (`--gm-gate2` is removed.)
- **The shared expert** reads global + its own private space like any expert. Its share of the total measured gain
  (Σ y over its exploration pairs / Σ y over all) is logged as "share-of-gain"; option `--gm-shared-read global`
  (the shared expert reads the global space only) if its private space contributes nothing (T1 / pilot).

### 6.2 Inject (the expert's input)
- Routed: `buf[e_k_o, p_k_o] += s · r_{t,e}` for the open pairs (an out-of-place `index_put` with accumulate on the
  compact list; s = 1 except on exploration pairs), BEFORE `torch.bmm(buf, w_up)`. The expert then runs on xf + r:
  "together with everything else it receives".
- Shared expert: `self.shared(xf + scatter(open_sh · s · r_sh))`.
- The router never sees r (routing is computed from xf before the read): the read cannot change which experts run,
  so capacity drops and balancer counts are identical with the memory on or off (a test).
- Scale: contents are initialised normalised to v̄₀ = 0.1 × median‖xf‖ (≈ 10% of the input norm, measured at the fit
  step; accepted rebuttal; the oracle ceiling, §18.0, also scores 0.2 for the record), so a first read is a moderate
  perturbation; Adagrad then sets the size. **Norm cap**: ‖v‖ ≤ 0.3 × median‖xf‖ at spawn and ≤ 0.5 × median‖xf‖
  after any update (a row-wise rescale after each Adagrad step and merge; the share of touched rows at the cap is
  logged, broken-if > 30%).

### 6.3 Gradients
- **GatherRead** (custom autograd Function, like `MemRead`): forward Σ_j a_j V[rows_j] with a_j = max(A) π w_j (no fade); it saves
  only rows and a; backward gathers V[rows] again for ∂/∂a_j = ⟨g, V[rows_j]⟩ and `index_add_`s a_j g into the dense
  **fp32** grad buffer (global and private; rev 2, H4). Nothing (|O|, 8, 384)-sized is kept for backward.
- w and A are recomputed WITH grad from x̂ = R((xf − μ)/σ) and the gathered detached winner keys: the LM loss reaches
  xf through the match weights (decided). `--gm-match-grad 0`: w, A from sg(x̂) (ablation).
- **Scale** (custom Function on (s, r^g, r^p) per open pair; replaces rev 1's Probe): forward returns s · (r^g + r^p);
  backward passes s · g to r and stores b = ⟨g, r⟩ (= ∂L/∂s), ⟨g, r^g⟩, ⟨g, r^p⟩ and ‖r‖² per open pair (§5.5). s is
  not a parameter; nothing is updated from ∂L/∂s except the gate target and the confirmation credit.
- Contents: **row-Adagrad, no momentum, no decay** (accepted rebuttal; rev 1's discounted accumulator is dropped) after
  the optimizer step (lr 0.01, per-row clip 0.05, as the old `--mem-lr`, `--mem-clip`), then the norm cap (§6.2).
  T1 sweep (option `--gm-opt rmsprop`): lazy row-RMSprop, β₂ = 0.9 applied per touch (not per step), η = 0.03.
  Private contents: bf16 storage updated with stochastic rounding (the fp32 update is computed, then rounded up or
  down with probability proportional to the remainder: unbiased; a unit test compares 1,000 small steps to fp32).

### 6.4 How much a found blob is trained
Every in-window open read trains its 4 winners. A late fact recurs ≈ 116 times by the end; each exposure has ≈ 4.9
in-window passes reading the SAME space: ≈ 116 × 4.9 × p_open × p_hit ≈ 116 × 4.9 × 0.8 × 0.8 ≈ **360 value
updates** (25c: ≈ 35 per bank). Rare facts get ≈ 5-30. This is the main structural gain of "one space for all passes".

---------------------------------------------------------------------------------------------------------------------

## 7. Benefit signals
Rev 2 (H1): no centring anywhere; zero means no gain.
- **Gate target** (exploration pairs): y = N_tok · b̃_cv, unbiased for l(0) − l(1) (§5.3).
- **Confirmation credit** (opened pairs with s = 1, where l′(0) is not observed): the curvature-corrected estimate

      Ĝ = −⟨g, r⟩ + ½ κ̂ ‖r‖²       (≈ l(0) − l(1) for a locally quadratic l: l(0) = l(1) − l′(1) + ½κ‖r‖²)

  and per part Ĝ^g = −⟨g, r^g⟩ + ½κ̂‖r^g‖², Ĝ^p likewise (exploration pairs use their b̃_cv split in the same ratio).
- **Separate credit**: the global blob's benefit for a token is Ĝ^g_t = Σ_{e open} Ĝ^g_{t,e} (it enters every open
  expert of the token); the private blob's benefit is Ĝ^p_{t,e}.
- **Placebo** −⟨g, r − r_rand⟩: logged only (§5.3), never used for confirmation, strength or the gate.
- No sign-based disconfirmation, no spawn-on-hurt. The positive part of Ĝ feeds the blob strength (§10).

---------------------------------------------------------------------------------------------------------------------

## 8. Writes (after backward; inside the training step; brain signals only)

### 8.1 Which positions (25c §6.2)
- Surprise σ_t from tensors the step already has (choice by T0h): σ^raw = ce_{K−1}[t]; σ^ent = ce_{K−1} − Ĉ(H_{K−1})
  (CE above what the model's own entropy predicts; Ĉ an EMA 32-bin regression); σ^imp = ce_{K−1} · 1[ce_0 − ce_{K−1}
  < δ_imp]. No labels, source tags, document-type features or dataset pointers.
- Eligible: ≥ 8 tokens after a document start, not EOT, token kept by ≥ 1 routed expert at the write pass.
- **Global budget**: W = 32 per micro-batch (64 per optimizer step), **≤ 4 per sequence** (rev 2, H5; range 3-4, T0h
  may pick 3), ≥ 8 tokens apart: a static top-k on masked σ with a per-sequence counter. Rev 1's ≤ 2 per sequence × 16
  sequences = 32 = W made the per-sequence cap the budget (every sequence wrote exactly 2, surprising or not); with 4 the
  cap only stops one sequence from taking the budget.
- **Private budgets** (D2 / D3): per expert, the credit counter of §3.2; when it fires, the expert's write is its
  kept token (at any in-window pass, all 6 routed experts of a token are candidates: accepted rebuttal) with the
  highest σ among the eligible ones, independent of the global W. Static: one candidate per expert per micro-batch
  (a segment max over the capacity layout, masked by credit ≥ 1).
- **Writes start at the fit step** (300): accepted rebuttal. The reviewer's "hold writes until R@16 ≥ 0.9" is
  rejected (§OD-R); R@16 is measured online and in T0d.
- **Direct writes are the default** (coordinator default c; owner may override): a selected position spawns / merges
  in the main store after the step. The probation stage is a flag (§8.7).
- The selection happens after the forward (σ needs the last pass's CE) and **before** `backward()` (train_lm calls
  `model.bm.select_writes(...)` between the loss and `backward`), so the gradient hooks know what to capture.

### 8.2 Which passes and which key (rev 2)
- **Default: the averaged key.** Key = x̂ of xf averaged over the token's in-window passes (for a private write: over
  the in-window passes at which (t, e) was a kept pair) = the mean of the per-pass x̂ (the map is affine). Rationale:
  the memory is pass-agnostic and queries come from every pass; the average sits near the centre of the token's pass
  trajectory, so every pass's query is about as close as it can be (it minimises the summed squared distance to the
  per-pass queries). Stashed in forward as a running sum at the selected positions (static: 32 × 64 global; ≤ 769 × 64
  private).
- **T0c decides between** `--gm-write-key avg` (default), `uniform` (rev 1: x̂ at one uniformly drawn in-window pass)
  and `maxgrad` (the in-window pass with the largest ‖∂L/∂xf_t‖: still the brain's own signal); `last` stays a
  debug option. If T0c shows the averaged key is found worse than a single-pass key (e.g. the trajectory is bimodal),
  the best single-pass rule replaces it before the gate / write code is written (§18.0).

### 8.3 Content: what the gradient says would have helped (ε guard; 25c §6.3)
- Hooks on `buf` and on the shared expert's input of each **in-window** pass (registered in forward, reading the
  selection tensor that is filled before backward) gather, at the selected positions, g_{t,e,k} = ∂L/∂buf[e, p] for
  the token's kept routed experts and g_{t,sh,k} for the shared expert, and **sum them over the in-window passes** k
  (static accumulators: 32 × 7 × 384 global-side, 769 × 384 private-side, fp32, per micro-batch).
- **Hook guard** (rev 2): hooks are registered only when `ctx.in_window` and grad is enabled; an out-of-window pass
  (no_grad) registers nothing, and a unit test asserts that the accumulators are untouched by out-of-window passes and
  by a checkpoint recompute (each hook adds into a slot indexed by (pass, position) with `index_copy_`, then the sum is
  taken after backward, so a recompute cannot double it).
- Private content for expert e: q = −Σ_k g_{t,e,k}; global content: q = −Σ_k (Σ_{e kept at k} g_{t,e,k} + g_{t,sh,k})
  — the derivative of the loss w.r.t. one vector added to all of the token's expert inputs at all in-window passes,
  which is what a global blob read at every pass is.
- v = σ_ξ ξ + v̄₀ · q / max(‖q‖, ε_q) · 1[‖q‖ ≥ ε_q], σ_ξ = 0.02 v̄₀, ε_q = 1e-6 × the running median ‖q‖; v̄₀ =
  0.1 × median‖xf‖ (normalised init, accepted rebuttal); then the spawn cap ‖v‖ ≤ 0.3 median‖xf‖ (§6.2). ξ-only when
  the gradient is absent or below ε_q (a dropped expert has no slot and gets no private write at all). There is no
  W_o, so the 25c "W_o = 0 → q = 0" start problem is gone; the ε guard remains for vanishing gradients.

### 8.4 Where they go
- One **global** blob per position, unless the position had a confident global read at the write pass
  (A_g ≥ A_w = 0.3 on an open pair): then only the private writes happen (a found fact is renewed by confirmation, not
  copied).
- **Private** blobs come from the per-expert budgets (§3.2, §8.1), not from the global positions: expert e (routed or
  shared) writes when its credit fires, its most surprising eligible kept token, unless that pair read its own space
  confidently (A_p ≥ A_w at any in-window pass; then the next most surprising). A routed expert's candidates are the
  tokens it processed (all 6 of a token's experts are candidates; dropped pairs are not). `--pm-rate` is removed (the
  budget is decided).

### 8.5 Dedupe at write (wrong-merge bound; 25c §6.4)
- Against the store, after the optimizer step: the global candidates through the same IVF search (64 queries), the
  private ones by a [n, 64] × [64, 512] score per space.

      d₁ < γ h₀²  AND  cos(−g_new, v_{j₁}) > θ  AND  j₁ has move budget left (§9.3)  ⟹  MERGE into j₁:
          η = max(1/(n_{j₁}+2), 1/4);   Δk = clip_{min(σ, 0.5h₀)}( η (x̂ − k_{j₁}) );   k_{j₁} ← k_{j₁} + Δk
          τ_{j₁} ← τ_{j₁} + η (t − τ_{j₁});   v_{j₁} ← v_{j₁} + (v_new − σ_ξ ξ)/(n_{j₁}+2), then the norm cap
          n ← min(n+1, 8);   Λ^α, t^touch ← t;   path length += ‖Δk‖;   moves-in-window += 1
      otherwise  ⟹  SPAWN

- **Content agreement** (rev 2, H7): a key-near candidate is merged only if its gradient direction agrees with the
  stored content, cos(−g_new, v_old) > θ: two facts with the same template but different answers have near keys and
  disagreeing gradients, so the distance test alone merges them. θ per space type from T0e (the smallest θ at which
  the wrong-merge rate ≤ 2% at the chosen γ; placeholder 0.2). A key-near but disagreeing candidate spawns.
- **Merges are moves**: they count in the per-blob move budget (≤ 1 per step, ≤ 8 per 1,000 steps, §9.3) and in the
  path length; a blob out of budget is not merged into (the candidate spawns). A merge no longer adds strength by
  itself (strength = benefit, §10).
- γ (one per space type) from T0e: the largest γ ≤ 0.5 whose wrong-merge rate P(j₁ is another fact | d₁ < γh₀², cos
  > θ), measured on the T0a capture at the real distractor density (256K global; 512 private), is ≤ 2%. Placeholder
  0.25. If no γ ≥ 0.05 meets 2%, dedupe is off for that space type.
- Within a step: candidates of the same space within γh₀² of an earlier (higher σ) one are suppressed and their content
  increment goes to the kept one (`index_add_`). A merged global key may change cell: its cell id is recomputed.

### 8.6 Spawn and eviction
- Spawn: k = x̂ (averaged key), v as §8.3, strength a = 0, all clocks = t, t^conf = −∞, m = n = 0, S = S₀.
- **Young-blob protection** (rev 2): a blob younger than 2,000 steps (t − t^spawn < 2,000) is not evicted while the
  protected blobs hold ≤ 50% of the space's capacity; above 50% the oldest-protected lose protection first. Rationale:
  a fresh blob has had no chance to be re-read (late facts recur every ≈ 105 steps, rare ones every ≈ 1.3K), so its
  benefit-strength is 0 by construction; without protection the newest writes would be the first victims.
- **Victims**: the free list; else the **lowest strength** among unprotected slots, always **excluding t^touch = t**
  (read with A ≥ 0.05, written, merged or moved this step). Global cell rule: the new blob goes to its nearest cell,
  spills to the 2nd-nearest if full (§4.1); if both are full the victim is the space's weakest (it frees a slot in some
  other cell; the new blob then takes its nearest non-full cell). One masked top-k (global: over 256K, ≈ 1 ms; private:
  per space over 512, batched).
- **Strength** = an EMA of the positive benefit Ĝ⁺ of the blob's confirmed and unconfirmed reads (§7, §10), half-life
  4,000 steps: blobs that keep helping stay, the reserve of recurring facts forms by itself.
- **Zombie guard** (25c §6.6): kept as a log only (a slot read but not confirmed for > 3,000 steps); the benefit
  strength already decays without positive reads.

### 8.7 Options (pending; default OFF)
- **Probation / candidate stage** (`--gm-probation 1`, +0.26 GB): new blobs first enter a small probation store
  (+0.26 GB is the reviewer's size estimate; the exact layout is fixed only if the flag is enabled) and are admitted to
  the main store only after one confirmed read; unread candidates expire. Reviewer's proposal; coordinator default: direct writes (§8.1),
  probation behind the flag, T0h (how many selected positions are ever re-cued) and T1 decide.
- **Private-by-consolidation** (`--gm-private consolidate`): instead of the per-expert budget writing directly, a
  private blob is created by copying a global blob that this expert's reads repeatedly benefit from (Ĝ^g > 0 on
  ≥ 3 confirmed reads through expert e). Compared with direct private writes in T1.

---------------------------------------------------------------------------------------------------------------------

## 9. Keeping blobs findable: confirmation, key moves, anti-capture, survival

### 9.1 Why renewal is still first-order
Keys are write-time thoughts; thoughts drift (median drift 1.22-1.31 × the 1-NN gap per 1K steps). An unrenewed key is
found with R@1 ≈ 0.77 after 1K steps, ≈ 0.45 after 2K, ≈ 0.15 after 3K (4K density; lower at 256K). Late facts recur
every Δ ≈ 105 steps; their blobs must be renewed faster than they drift away. Value training (§6.4) never moves a key
(25c §7.2).

### 9.2 Confirmation (default; in-window benefit)
- **Global winner j₁(t)** is confirmed by token t at pass k iff: the token has ≥ 1 open pair, the curvature-corrected
  gain Ĝ^g_t > 0 (§7; rev 2, no centring) and is in the top 25% of this micro-batch's in-window tokens with an open
  pair, AND the geometry passes: w₁ ≥ ω₁, d₂/d₁ ≥ μ_m (**clear winner**), A_g ≥ A_min (thresholds from T0e at
  precision ≥ 0.95 that j₁ is the same fact), AND the **content agrees** with what the step wanted:
  cos(−g_t + κ̂ r, v_{j₁}) > 0.3 (−g_t + κ̂r ≈ the loss-decreasing direction at s = 0, i.e. what the blob should have
  supplied; rev 2 anti-capture).
- **Private winner j₁(t,e)** likewise with Ĝ^p_{t,e} (top 25% among the open pairs, pooled over experts), the private
  thresholds and the same content test.
- Out-of-window geometric renewal (25c §7.3) is NOT needed by default: every exposure has ≥ 2 in-window passes
  (K ≥ 2), ≈ 4.9 on average, reading the same space. `--gm-renew-geo 1` keeps it as an option for passes 0..K−7 of
  K ≥ 7 steps (≈ 1.5 passes per forward), with the stricter thresholds.

### 9.3 The key move and the anti-capture guard
After the optimizer step, per confirmed slot (sort + segment mean of its confirming queries q̄ over both micro-batches
and all passes):

      k_j ← k_j + clip_{min(σ, 0.5 h₀)}( α_j (q̄ − k_j) ),   α_j = max(0.3, 1/(n_j + 1));   τ, t^conf, t^touch ← t;
      m ← min(m + 1, 3) if ≥ 64 steps since its last increment;  refresh k̃, ‖k‖², cell;  path length += ‖Δk‖

  (n_j = the blob's count of earlier confirmations + merges: α = 1 on the first (still capped), 0.5 on the second,
  0.3 from the fourth on, so a well-established blob still follows the drift but one read cannot drag it.) Strength is updated by the benefit
  EMA (§10), not by the move.

Guards against a blob being pulled away by unrelated thoughts:
1. **Clear-winner margin** (d₂/d₁ ≥ μ_m), A ≥ A_min and the **content test** cos(−g_t + κ̂r, v_j) > 0.3 on every
   confirming read: a blob between two clusters, far from the query, or read by a context that wanted something else,
   cannot be confirmed.
2. **Capped move**: each move ≤ min(1σ, 0.5 h₀) (σ = the kernel width, §6.1; rev 2), so one step never moves a key by
   more than the typical gap to its runner-up.
3. **Per-blob counts**: ≤ 1 move per step (the segment mean, not one move per read) and ≤ 8 moves per sliding 1,000
   steps (a 4-bit counter + the window start), **merges included** (rev 2, H7); beyond that a confirmation renews
   strength / maturity but does not move the key.
4. **Per-space caps**: global ≤ 256 moving slots per step, private ≤ 2 per expert per step (≤ 512 in total), lowest
   Ĝ first dropped. Check for the global cap: late-fact cues arrive at ≈ 194 per step (25c §7.3); with p_open 0.8,
   p_hit 0.8, p_conf 0.6 ≈ 75 confirmations per step: the cap does not bind for the expected traffic.
5. **Travel logging**: per slot, the path length Σ‖Δk‖ (4 B); for 4,096 tracked slots per space type, also the spawn
   key, so the net travel ‖k − k_spawn‖ / h₀ is logged. Alert: median net travel per 1K steps > 1.5 × the median thought
   drift per 1K steps (keys running ahead of the thought), or any tracked slot with net travel > 4 h₀ whose benefit
   rank b̄ is falling.
- Separation after a move (25c §7.3): ‖k₁′ − k₂‖ ≥ (√μ_m − (1 − η))√d₁; the fixed point is a lagged EMA of recent
  confirming queries (it follows the drift).
- `--gm-track all` (move toward every confident read, no benefit test) and `--gm-confirm off` are ablations.

### 9.4 Survival of a recurring fact's trained global blob (25c §7.4, one space for all passes)
Renewals of fact F's blob form a Poisson process with rate λ = p_ren/Δ; the blob is lost when a gap exceeds the drift
horizon H ≈ 1,500 steps; lost-gap events occur at rate λ e^{−λH}:

      S(T) = exp(−λ T e^{−λH}),   Δ ≈ 105, T = 12.2K, H ≈ 1,500   ⟹   S ≥ 0.9 needs p_ren ≥ 0.44

Per exposure, each in-window pass gives a chance s = p_open · p_hit · p_conf (p_hit = the cross-document hit rate from
that pass's query, T0a × T0c); the passes are correlated (same token, similar xf), so

      p_ren = 1 − (1 − s)^{n_eff},   1 ≤ n_eff ≤ E[in-window passes] = 4.94

| p_open 0.8, p_conf 0.6 | n_eff = 1 | n_eff = 2 | n_eff = 3 |
|---|---|---|---|
| p_hit 0.8 | p_ren 0.38, S 0.83 | 0.62, **0.99** | 0.77, 0.998 |
| p_hit 0.5 | 0.24, 0.41 | 0.42, **0.89** | 0.56, 0.98 |
| p_hit 0.3 | 0.14, 0.12 | 0.27, 0.51 | 0.37, 0.81 |

- **What it implies**: (i) with ≥ 2 effective passes the single global space reaches the bar at p_hit ≥ 0.5 — 25c needed
  geometric renewal and a union over 5 banks for the same; (ii) n_eff is set by cross-pass findability (T0c): if a
  blob written at one pass is not found from the others, n_eff → 1 and only p_hit ≥ 0.85 reaches the bar; (iii) p_hit is
  the cross-document hit rate (T0a): **T0a and T0c together decide whether late facts survive**.
- Private copies are a second carrier: a cue at a later exposure reads its own experts' spaces; the fact is there iff
  one of its ≈ 6 processing experts (≈ 4.9 per pass, overlapping) wrote it. With expert overlap ρ_e (T0r: the
  fraction of the write-time experts among the later cue's experts) the private hit rate is ≈ p_hit,p · (1 −
  (1 − ρ_e)^6). Private lives are now ≈ 2,000 steps at any E_live (D2, §3.2; rev 1: 260-1,050), above H, so private
  copies of late facts survive between confirmations; they matter most for rare facts' first re-exposure at low
  density. With the budget, a fact enters only the experts whose credit fired on it, so ρ_e is measured as "private
  reach" (§18.0: the fraction of a fact's later-cue experts that hold a private copy).

### 9.5 Re-acquisition (25c §7.5)
After a loss the next exposure misses (no confident read), the cue is eligible and is written with p_w per exposure (T0h:
the probability that a missed cue is in the top-W): a two-state chain with loss rate ν = λe^{−λH}, recovery μ = p_w/Δ;
availability of SOME key = μ/(μ + ν). A re-acquired blob restarts with a one-shot content, so S(T) of §9.4 (the trained
blob) is the number that matters; availability is its floor. `--gm-rescue` (a spawn whose content direction matches an
existing blob, cos ≥ 0.9, within 4γh₀² moves that blob's key instead) stays an option.

### 9.6 Fill (the main risk without a backup memory)
- 20.4K late facts enter the global space at ≈ W · s_late per step (s_late = the share of writes going to unacquired
  late cues, T0h): **≈ 640 / s_late steps** — 1.3K steps at s_late = 0.25, 3.2K at 0.1 (25c: 2.4-4.8K per bank).
- A fact is first written with probability ≈ p_w per exposure: ≈ 1/p_w exposures ≈ 350 steps at p_w 0.3 once the
  budget does not bind.
- **Logged per space over time** (every 250 steps): live slots (global; private min / median / max over experts and
  the fraction of full spaces), spawns / merges / evictions per step, mean age at eviction, mature count (the reserve),
  zombie count, and the fill of the 20.4K late facts (eval tags only: the fraction of late facts whose cue finds a
  blob with A ≥ 0.37). Alert if the global live count < 60% of 64 × (t − 300) × (1 − merge rate) or the late-fact fill
  < 50% at 3K steps.
- Mitigations if T0h / the pilot show slow fill: σ^ent / σ^imp; W = 96 until 3K steps (a WRITE budget schedule, not a
  read budget); never a larger γ (wrong merges).

### 9.7 Rare facts (25c §7.7)
P(recall) ≈ p_write · p_open(eval) · E_{a ~ Exp(Δ_rare)}[p_ret(a)] · p_val. One space for all passes raises p_val (more
reads per exposure train the content) and p_open (≈ 4.9 chances); the global density (256K) lowers p_ret, which the
private spaces partly recover if the cue routes to a writing expert.

---------------------------------------------------------------------------------------------------------------------

## 10. Strength, maturity and clocks (lazy; rev 2 replaces 25c §8's fade)
- **Strength** (eviction only) = a lazy EMA of the blob's positive benefit, half-life 4,000 steps:
  a_i ← a_i · 2^{−(t − Λ_i)/4000} + Σ_{reads this step} Ĝ⁺ (folded at each touch; Λ_i ← t). Spawn: a = 0 (young-blob
  protection covers the first 2,000 steps, §8.6). Split children inherit **half** the parent's strength (§11.2).
  Rev 1's log α with λ_α fade, maturity-scaled ρ, +0.5 per merge and +1 per confirmation is dropped: strength now
  measures help, not activity. Maturity m (0..3) is kept for logs and the T1 survival model only.
- **Content fade: none** (λ_v = 0; reviewer). **Accumulator: plain Adagrad** (no discount; accepted rebuttal).
- h₀ per space type (25c §3.3): calibration h₀² = q₅₀(d_same)/(2 ln 2) from T0e; tracking h₀² ← 0.99 h₀² + 0.01 κ
  q₅₀(d₁ over the **exploration** reads of that space type in the step) (rev 2: the exploration pairs are selected
  independently of ĝ, so the EMA neither contracts nor follows the gate's choices); floor h₀² ≥ ½ q₅₀(NN distance²
  among 4,096 sampled keys) every 250 steps (private: NN within the same space, pooled over 64 sampled spaces).
  Before T0e: h₀² = 1.24 × q₅₀(NN²) (the salvage drift scale). One h₀,p for all private spaces (per-space values would
  be too noisy at 512 keys).
- **σ²** (the kernel width, §6.1): EMA 0.99 of median(d₂ − d₁)/(2 ln 2) over the exploration reads, floor 0.05 h₀².
- **Floors on the gate's robust scales**: MAD(b̃) ≥ 1e-4 nats, δ ≥ 1e-3 nats (§5.3), so an early quiet phase cannot
  make the Huber loss a pure L1 on noise.

---------------------------------------------------------------------------------------------------------------------

## 11. Experts: capacity drops, births, deaths

### 11.1 Capacity 1.5 and dropped tokens
- Routing is computed from xf before the memory, so the kept / dropped set is identical with memory on or off.
- A dropped (token, expert) pair (≈ 3-4% in training) has no buffer row: no expert compute (as today), **no private
  search result used, no private read, no Scale value, no private write, no confirmation** for that expert. Its
  token's global read still enters every KEPT pair that opens, and the shared expert (which never drops).
- A token with all 6 routed experts dropped (rare) can read only through the shared expert's gate.
- The global content's write gradient sums over the kept experts only (§8.3).
- Eval runs dropless (`--capacity dropless`), so eval reads see every pair.

### 11.2 Split births (`--birth-mode split`, the run's mode)
At the top of a step, `Core.grow_experts` → `RoutedMLP.grow_experts(n)` returns src (parents); the memory hook
(`Core.mem_hook`, already called there) copies, for dst = E0..E0+n−1:
- the parent's whole private space: keys, bf16 copies, ‖k‖², contents, Adagrad S, clocks, counts, maturity
  (t^touch unchanged), with the child's copies at **half the parent's strength** (the parent keeps its own; rev 2:
  unread inherited blobs are evicted before the child's own new ones); travel-tracking entries
  are not copied (the child starts untracked); the child's write credit starts at 0;
- the parent's gate head (a_e, β_e, γ_e), λ_e and its AdamW moments (preallocated 769 rows: no parameter resize, only
  row copies, so the gate's optimizer is not rebuilt; these are memory tensors, outside `carry_optimizer`);
- nothing in the global space changes.
Both twins then hold identical spaces; each receives about half of the parent's tokens (router rows r ± δ), so blobs on
the other side of δ stop being read and fade to eviction. No cross-space dedupe. Births stop at ≈ 7,320 (60%).
- `ramp` birth mode (not used by this run): the child starts with an empty private space and the shared gate head
  (it is not selected at birth).

### 11.3 Expert death
There is none in the current code, so there are no dead banks to recycle (the reviewer's dead-bank recycling is
rejected, §OD-R). If one is added (e.g. resetting a dead expert), the hook `on_expert_reset(e)` frees
its private space (all slots to the free list, strength −∞, counts 0) and resets its gate head and moments to 0
(shared-head behaviour). Neuron renewal inside an expert (dense-MLP renewal) changes what an expert does with its input,
not the input space: its blobs stay valid as addresses; their contents adapt by gradient.

---------------------------------------------------------------------------------------------------------------------

## 12. Integration points (code map)

### 12.1 `spiking_lm.py: RoutedMLP.forward` (one new optional argument; `None` = today's code path)

```python
def forward(self, x, forbid=None, side=None, extra=None, wave=None, pmem=None, bmem=None):
    ...                                        # route, idx, gate, order, counts, cl (host), C, pos, keep, kidx: unchanged
    e_k, p_k, t_k = kept(e_sorted), kept(pos), kept(tok)
    buf = xf.new_zeros(E, C, D)
    buf[e_k, p_k] = xf[t_k]
    valid = ...                                # unchanged
    xs = x
    if bmem is not None:                       # 25d: search, gate, read, inject (input side), probe, write hooks
        buf, xs = bmem(xf, buf, x, side, e_k, p_k, t_k, kept(order) % k, counts, N, E, C, n_keep)
    h = self.spike_e(torch.bmm(buf, self.w_up.to(buf.dtype)), valid, ...)
    ...                                        # out, y.index_add_, pmem (old, off): unchanged
    return y + self.shared(xs) if self.shared is not None else y     # shared expert on its read-augmented input
```
- `n_keep`, C, `cl` are host values RoutedMLP already has (the existing sync), passed so every memory buffer size is
  static on the host (N·k for the pair buffers, E·C for the private layout). `counts` (device) gives n_e for the
  per-expert rate of the λ update.
- The old `pmem` path (output side) stays, off.
- The memory's workspaces (Q, score buffers, candidate lists, compact read buffers, accumulators) are **preallocated
  once** at their maximum static sizes (from N, k, E_max·C_max, n_max) and reused by every pass (rev 2, H6): no
  per-pass allocation, so the allocator does not fragment and CUDA graphs (speed track) can capture them.

### 12.2 `loop_brain.py`
- `Core.forward(…, bmem=None)`: passes `bmem` to `self.mlp(...)` (line 235). Nothing else changes in Core.
- `LoopBrain.__init__` / `_build_memory`: `self.bm = BlobMemory(...)` when `--gm 1`, built LAST on a forked RNG (as
  plan 22 did), so `--gm 0` consumes no random numbers. `Core.mem_hook` chains the old `Memory.on_grow` (if any) and
  `bm.on_grow(src, E0, split)`.
- `run()` (lines 997-1196):
  - before the pass loop: `bm.begin_forward(train=self.training and tgt is not None, K=K, bptt=bptt)` (clears the
    per-forward context, the kept-tensor lists, sets the in-window passes);
  - per pass, before `core(...)` (line 1075): `bm.ctx.update(k=k, in_window=k >= K - bptt, prev_mu=prev_mu,
    prev_halt=…)` and `bmem=self.bm if bm active else None` into `core(...)`; prev_match is kept inside `bm` from the
    previous call; prev_halt = sigmoid of the existing halting head's output after pass k−1 (`halt_head`, detached;
    `halt 1` in the old run's config), None at pass 0;
  - `prev_mu` already exists (line 1141: `[mu.detach() for mu in mus]` in the `hier` branch); with `--mem-shared 0` the
    `_hier_step(s, read=False, address=False)` path still produces μ_l, so the gate's μ input is available without
    the old memory;
  - after the loop: per-token CE of every pass is already in `ces` (training, readout_frac 1) for σ^imp; the final
    CE comes from `forward`.
- `select_writes(ce_last, ce_first, ent_last)`: called by the trainer between the loss and `backward()`; fills the
  device selection tensors the gradient hooks read.
- `ko_gm` (eval knockouts): global off / private off / both off; gate modes learned / open-capped / random / off.

### 12.3 `train_lm.py` (train step, lines 1345-1409)
- Flags (§21); `--gm 1` asserts `--mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-gate 0` (the old memory and the
  new one are exclusive in this plan; the old code stays, off).
- `bm.write_on = step >= args.gm_write_start`; at `--gm-fit-step` the map, the IVF cells, v̄₀ and h₀ init are fitted
  from the accumulated statistics (device ops, no sync).
- In the micro-batch loop: after `model(x, y)` → `bm.select_writes(...)`; after `backward()` → `bm.after_backward()`
  (y, κ̂, the gate's separate graph and its backward, value-grad bookkeeping, the write stash with its captured
  gradients, confirmation candidates).
- The gate's parameters (and ζ) in their own AdamW groups and clip (as `gate_opt` today), excluded from
  `brain_params`, from `make_opt` and from `carry_optimizer`; all store tensors are buffers (§1).
- VRAM defaults if T2 needs them (§VRAM): `--chunk-ce 1` and `--ckpt-concept 1` (checkpointed concept loss).
- After `opt.step()` and `apply_balance()`: `bm.after_step(step)` (§13).
- `torch.cuda.empty_cache()` before every `evaluate()` and before the loop report; `set_per_process_memory_fraction`
  (0.78); eval batch 16.
- Resume: the store, cell table, centroids and the refit reservoir, map, h₀ / σ² / κ̂ / Ĉ EMAs, λ_e, write credits,
  gate parameters and moments, ζ, the generator.

### 12.4 `eval_world.py`
Token-type split (world_answer_first / rest, world_other, wiki; salvage `extra_split.py`), knockouts ko_gm_all,
ko_gm_global, ko_gm_private, gate modes, open rates by token type, p_open / p_hit (global and private) at answer cues,
the late-fact fill.

---------------------------------------------------------------------------------------------------------------------

## 13. Order of operations and host syncs

- **Forward (each pass)**: map → global search → private search → match features (+ content-fit) → gate (φ, ĝ,
  threshold λ_e, stratified exploration, s draws, token-order overflow) → GatherRead + Scale + inject → experts.
  Out-of-window passes run the same under `no_grad` (they read; they do not train the gate, confirm, write, or
  register hooks).
- **Between forward and backward**: `select_writes` (global top-W with the per-sequence cap; per-expert credit picks).
- **After each micro-batch's backward**: gate loss on the kept exploration tensors (separate small graph, backward
  into gate params); κ̂ slope; confirmation candidates (slot, query, Ĝ, content test) appended to static buffers (masks
  + a dummy slot); the write stash (positions, averaged keys, pass-summed gradients) appended.
- **After opt.step (once per step)**:
  1. gate AdamW (own group), ζ AdamW; λ_e update;
  2. content row-Adagrad on touched rows (fp32 grads; private contents with stochastic rounding), per-row clip, norm
     cap; t^touch ← t for touched rows; zero the touched grad rows;
  3. strength EMA fold for read slots; confirmations → key moves under the guards (§9.3); cell ids of moved keys;
  4. writes: dedupe with the content test (§8.5; merges spend move budget), then spawns into free / unprotected
     weakest-untouched slots with the cell spill rule (§8.6); credits decremented;
  5. refresh k̃, ‖k‖² of changed slots; rebuild the global cell table; every 1,000 steps refit the centroids from the
     reservoir and reassign all cell ids;
  6. EMAs (h₀,g, h₀,p, σ², κ̂, Ĉ), floors every 250 steps, the online IVF-miss / R@16 check every 250 steps.
- **Host syncs added: none** (rev 2, H8). Every size is a host integer known before the GPU work (W, the exploration
  counts per stratum from n_keep, n_max = ⌈0.55 n_keep⌉, Cq, Cc, the 769 credit slots, the confirmation / move caps);
  every buffer is sized from N·k or E·C; variable-size sets are masks over static buffers; counts use `index_add_` of
  ones, **never `bincount`** (it reads the max to size its output: a sync) and never `torch.unique` / `nonzero`;
  per-slot averaging uses sort + neighbour flags + cumsum + `index_add_`; the λ, credit and EMA updates stay on the
  device; RoutedMLP's existing `counts.tolist()` provides n_keep and C. Removing the rows also removes their
  `write_counts()` sync (`train_lm.py` line 1397). The unit test counts syncs (§18.4).
- **Launches**: ≈ 110 kernels per pass forward, ≈ 30 per in-window backward, ≈ 60 per micro-batch after backward,
  ≈ 250 after the step → **≈ 2,100 launches per step** (12.8 × 110 + 9.9 × 30 + 2 × 60 + 250; range 1,800-2,600),
  logged by T2 with the profiler (§17).

---------------------------------------------------------------------------------------------------------------------

## 14. Causality (no future leakage)
1. **Within a sequence**: the read at position t uses x̂_t, which depends on tokens ≤ t only (causal SSD mixer,
   per-position workspace and concept heads), and the store as it was before this optimizer step. The gate's inputs
   (xf_t, side(s)_t, μ_l of pass k−1 at t, the halting confidence of pass k−1 at t, match and content-fit features at
   t and at pass k−1) are all position-t quantities. **The read decision is causal** (rev 2, H2): ĝ_t > λ_e with λ_e
   from earlier steps only; rev 1's per-batch rank cap let later positions' ĝ decide an earlier position's read.
   Buffer overflow is dropped latest-position-first, so an earlier read never depends on a later position.
2. **Within a step**: every store change (contents, keys, spawns, strengths, h₀, the cell table) happens after
   `opt.step()`, so micro-batch 2 reads the same store as micro-batch 1, and nothing written from a window can be read
   by that same window in that step. (A content gradient at t contains information about t+1 and later tokens of the
   window — that is the point of a write — but it can only be read in later steps, on other windows: memory of the
   training data, like the weights.)
3. **Write selection** uses the CE at t (target t+1) only to choose WHICH positions write; it never enters a read in
   the same step.
4. **Batch statistics** (MAD, κ̂, σ², the h₀ EMA, the stratum edges, λ_e) only shape the gate's loss and later steps,
   never the current forward's outputs (the stratum edges and λ_e used in a forward come from the previous step).
5. **Eval**: `bm.write_on = False`, no confirmations, no strength / clock / EMA changes, no exploration; the store is
   read-only, so eval scores cannot depend on the order of eval batches. A unit test: shuffling the eval batches gives
   identical per-token losses.

---------------------------------------------------------------------------------------------------------------------

## 15. Self-reinforcement: measured, not forbidden (25c §11)
Loops: reads → expert outputs → later passes' xf → the next query; reads → c → ĝ → what opens → what trains → c;
confirmation → read slots renewed and read more; the match-weight gradient: LM loss → w, A → xf; key moves toward the
thought. All live from the first run.

| metric | definition | review threshold |
|---|---|---|
| read-induced address shift | 8 probe sequences per eval: ‖x̂_k(memory on) − x̂_k(reads off)‖ / h₀,g at pass k ≥ 1 | > 1 h₀ |
| echo rate | P(the global winner was written by the SAME document) among confident reads | > 0.5 with benefit rank < median |
| concentration | Gini of reads per slot; share of reads on the top 1% of global slots | top-1% share > 20% |
| confirmation feedback | Spearman(confirmations so far, benefit rank b̄) | ≈ 1 with b̄ falling |
| key-move vs thought drift | median net key travel per 1K steps / median thought drift per 1K steps | > 1.5 |
| expert capture | per expert, the share of its private reads on its top-8 slots | > 0.5 for > 10% of experts |
| knockouts | ko_gm_global, ko_gm_private, reads at passes < k off | reported |

---------------------------------------------------------------------------------------------------------------------

## 16. Metrics and evaluation
- **Primary**: cloze exact / first-token / lp per token: late (n = 5,000), early (n = 1,000; 200 per relation), geo; by
  relation; loss by token type; knockouts ko_gm_all / _global / _private; gate modes learned / open (capped) / off.
- **Automatic eval** at 25 / 50 / 75 / 100% and at step 7,000, in-process, after `empty_cache()`, eval_batch 16,
  `--rank-limit 200 --questions 1 --knockouts gm,gm_global,gm_private --gate-modes learned,open,off --capacity
  dropless`; checkpoints saved there; not counted in tok/s.
- **Online logs** (every 250 steps): gate open rate (train / eval; by pass; by token type from eval tags only), cap hits,
  AUC, calibration; per space type: fill (§9.6), spawns / merges / evictions, mature / zombie counts, h₀, κ, median d₁,
  A; IVF probe overflow, cell Gini, online miss rate; confirmations and moves (counts, cap hits), path length / net
  travel; §15's table; p_hit at late / early cues (eval tags only); λ_e distribution, buffer overflow, κ̂, σ², the
  placebo mean, the decile slope; per-bank private lifetimes (§3.2); shared-expert share-of-gain; spill rate; R@16 on
  the online brute-force sample; hubness; ‖v‖-at-cap share; kernel launches and memory ms per step (T2 / profiler
  samples).

### 16.1 Broken-if metrics (rev 2; each is an alert to the owner, never an automatic fallback)
Checked at every 250-step log from step 1,000 on (the gate items from 0.3·S, as §5.6), and in the 1K pilot.

| # | area | broken if | likely meaning |
|---|---|---|---|
| B1 | gain | mean y on exploration pairs ≤ 0 over 1,000 steps (pooled) | reads hurt on average: contents or kernel wrong |
| B2 | placebo | mean −⟨g, r − r_rand⟩ ≤ 0 over 1,000 steps | the gain is a generic perturbation, not memory |
| B3 | gate | pooled AUC(ĝ, 1[y > 0]) < 0.60 at two consecutive evals, or decile slope < 0.3 at 3K steps | the gate cannot tell where reading helps |
| B4 | gate | open rate pinned at 0 (only exploration) or at 50% (λ binding) for > 500 steps; > 30% of experts with λ_e > 0 | gate collapsed / saturated |
| B5 | buffer | overflow of the 55% buffer > 1% of open pairs | λ lags; the cap is binding by drop |
| B6 | curvature | κ̂ ≤ 0 for > 500 steps, or κ̂‖r‖² > 2 |⟨g, r⟩| median | the control variate / confirmation correction is unreliable |
| B7 | fill | global live < 60% of 64 (t − 300)(1 − merge rate); late-fact fill < 50% at 3K | blobs not written before they are needed |
| B8 | private life | median eviction age < 1,000 or > 4,000 steps in > 20% of banks | budget mis-sized (D2 assumes ≈ 2,000) |
| B9 | young protection | protected blobs at the 50% limit in > 50% of spaces | write rate too high for the capacity |
| B10 | IVF | online fp32 top-1 miss > 2% for d₁ < 2h₀²; R@16 < 0.9 on the sample; probe overflow > 1%; spills > 5%; cell Gini > 0.6 | index stale or unbalanced |
| B11 | merges | wrong-merge rate (eval tags) > 2%; merges > 50% of write candidates | γ / θ too loose |
| B12 | moves | median net travel per 1K steps > 1.5 × thought drift; > 1% of tracked blobs with net travel > 4 h₀ | capture / runaway keys |
| B13 | hubness | top-1% global slots receive > 20% of confident reads; any slot top-1 for > 1% of queries | hub blobs answer everything |
| B14 | echo | P(winner written by the same document) > 0.5 among confident reads with benefit below median | self-echo, not recall |
| B15 | address | read-induced address shift > 1 h₀ | reads move the thought out of its own address |
| B16 | contents | > 30% of touched rows at the ‖v‖ cap; median ‖v‖ < 0.02 median‖xf‖ | contents saturating / vanishing |
| B17 | width | σ² at its floor for > 1,000 steps; h₀ at its floor | kernel collapsed |
| B18 | shared expert | share-of-gain > 50% (one expert carries the memory) or its private space never confirmed | switch to `--gm-shared-read global` (owner) |
| B19 | harm | world_other CE (on − ko_gm_all) > +0.04 nats at any eval | the GO harm bar is failing |
| B20 | cost | memory > 100 ms / step or launches > 2,600 / step; any added host sync | over the realistic budget (§17) |
| B21 | VRAM | reserved > 19.5 GB in any 200-step window incl. `evaluate()` and a birth | enable `chunk_ce` + checkpointed concept loss |
| B22 | **held-out harm** (rev 2, T0b) | on a fixed val_world subset (held-out templates 10-11, fresh documents; 32 windows, eval routing, every 1,000 steps and at every auto-eval): the non-answer world CE delta, memory on − off (ko_gm_all), > +0.04 nats; or its trend rising over two checks | reads keyed by training wording misfire on new wording: a generalisation failure the gate cannot fix (§5.0) |
| B23 | gate vs harm | on the same subset, the open rate on non-answer world tokens ≥ 0.5 × the open rate on answer cues while B22 > +0.02 | the gate is not separating the held-out harm (expected from T0b; reported, not a fault) |

---------------------------------------------------------------------------------------------------------------------

## §VRAM

### Measured (VRAM study, old run at end-of-growth size, 16 × 2, seq 512, bptt 6)
Training peak 20.12-20.25 GB allocated (K ≥ 6), 20.5-21.1 reserved; `evaluate()` 23.05 reserved on the training cache;
row tables + grads 3.54 GB; row read activations ≈ 0.4 GB; fragmentation 0.4-1.0 GB.

### 25d projection (16 × 2; rev 2)

| item | Δ GB | derivation |
|---|---|---|
| old-run training peak | 20.14 | measured |
| row tables and grads | −3.54 | measured |
| row read activations | −0.35 | 25c |
| global store | +0.52 | 262,144 × 1,970 B |
| private store (bf16 contents) | +0.47 | 393,728 × 1,200 B |
| value-gradient buffers, **fp32 everywhere** (H4) | +1.01 | global 0.40 + private 0.61 |
| read activations kept for backward (rows, a, w, A for 8 + 4 winners; Scale's saved r in bf16) | +0.25 | ≈ 40 MB per in-window pass × 6 |
| gate kept tensors, **exploration pairs only**, one micro-batch | +0.07 | ≈ 2.9K pairs × 2,000 × 2 B × 6 passes |
| **static search / gate / read workspaces** (IVF scores 402 MB at n_p 16, private scores 75 MB, top-k and re-rank buffers), preallocated once | +0.50 | persistent, reused by every pass (H6) |
| gate parameters (incl. W_cf) + moments, ζ, IVF index + refit reservoir, credits, bookkeeping | +0.02 | |
| **projected peak allocated** | **≈ 19.1** | reserved ≈ **19.3-19.7** (+0.25-0.6 fragmentation; static workspaces fragment less than rev 1's per-pass transients) |
| `evaluate()` after `empty_cache()` | ≈ 8.7 reserved | 25c's 6.2 + store 0.99 + grad buffers 1.01 + workspaces 0.50 (persistent) |

- **How it is judged** (rev 2, H6): peak reserved over a window of **≥ 200 consecutive training steps that includes at
  least one `evaluate()` and one expert birth** (a birth copies private spaces and gate rows), at K = 10 and 16; a
  single-step peak is not accepted as evidence.
- Margin to 20 GB reserved: 0.3-0.7 GB. **If T2 measures reserved > 19.5 GB, the defaults become** `chunk_ce` (chunked
  fused CE, −1.2 GB, ≈ −2% speed; VRAM study) **+ the checkpointed concept loss** (ConceptHier's loss recomputed in
  backward); both are maths-identical (the unit fixtures check it). Further levers, in order: (3) the gate trunk on a
  fixed random feature map (−0.05 GB now that only exploration φ is kept; changes the gate, owner); (4) the private
  grad buffer compacted per step (−0.5 GB, + sort cost); (5) 8 × 4 after the same-maths audit (−4.5% speed; 25c §VRAM).
- Run defaults: `empty_cache()` before every `evaluate()` and the loop report; `set_per_process_memory_fraction(0.78)`
  (≈ 20 GB, a guard); eval batch 16; micro-batch 16 × 2.
- Spike avoidance in the memory: no (N, 262,144) tensor exists (IVF layout; the brute-force flag uses 1,024-query
  chunks); the private bmm is [E, C, 512]; the dedupe search uses the same paths.

---------------------------------------------------------------------------------------------------------------------

## 17. Cost (estimates; T2 measures)

Per optimizer step: 12.8 pass-forwards (N = 8,192 each), ≈ 9.9 with backward; step ≈ 1.95 s (old run, 8.4k tok/s);
3% = 58 ms.

Rev 2 (H8): rev 1 summed compute and bandwidth (58-88 ms) and under-counted launch overhead. The memory is ≈ 2,100
small kernels per step (§13); at a realistic 12-25 µs of dispatch + launch each in eager PyTorch, that is ≈ 25-50 ms
by itself, overlapping only partly with the GPU work. The table gives GPU time; the realistic total adds the
non-overlapped dispatch.

| component | per pass | per step (GPU time) |
|---|---|---|
| map R(xf − μ)/σ, centroid scores (1.1 GFLOP), top-16 cells, probe layout (sort 131K) | 0.4-0.6 ms | 5-8 ms |
| global IVF bmm (25.8 GFLOP, 402 MB), per-probe bf16 top-8, merge, fp32 re-rank of 32 → top-16 | 2.0-3.2 ms | 26-41 ms |
| private bmm (5.4 GFLOP, 75 MB) + bf16 top-8 + fp32 re-rank, shared expert | 0.8-1.3 ms | 10-17 ms |
| gate: φ build, content-fit (2 × 384² × (N + pairs)), trunk (1.05 GFLOP), per-pair head, threshold, stratified exploration, compaction | 0.7-1.1 ms | 9-14 ms |
| read (gather of 8 global rows per token + ≤ 4 private per open pair) + blend + inject; backward (re-gather, index_add, Scale dots) | 0.5 + 0.7 ms | 6 + 7 ms |
| gate post-backward (exploration pairs only), κ̂, per micro-batch | — | 2-3 ms |
| after_step: Adagrad on touched rows (fp32 grads, ≈ 1.5-2.5 GB traffic), strength fold, confirmations / moves, dedupe (content test) / spawn / evict, cell-table rebuild; amortised 1,000-step refit (≈ 0.1 ms / step) | — | 9-14 ms |
| **GPU time** | | **≈ 55-80 ms** |
| non-overlapped dispatch of ≈ 2,100 launches | | + 15-25 ms |
| **memory total (realistic)** | | **≈ 70-100 ms ≈ 3.6-5.1% of 1.95 s vs a no-memory build** |
| rows removed (product-key select, shared / private / coarse read + backward, MemOpt, G zeroing, `write_counts` sync) | | −150 … −400 ms |
| **net vs the old run** | | **≈ 2.5-17% faster** (worst case: −150 + 100 = −50 ms) |

- **The bar (DECIDED, D1)**: tok/s ≥ **0.97 × the old run's config** in the same harness (median of ≥ 3 runs; ±3%
  run-to-run): met even in the worst case above. **Also reported** (not bars): the memory's cost vs the no-memory build
  (target ≤ 3%, expected 3.6-5.1% in eager mode) and the **kernel launches per step** (profiler count; expected
  ≈ 2,100, broken-if > 2,600).
- **Rules that keep it launch- and sync-bound only, not worse**: no `bincount` (`index_add_` of ones), no `.item()`,
  `nonzero`, `unique` or boolean-mask indexing; every buffer static from N·k, E·C, n_max; workspaces preallocated
  (§12.1) so CUDA graphs (speed track) can later capture the memory's forward per pass shape.
- **Triton 3.6** (installed; speed track, behind `--gm-triton 1`, off for the first run unless T2 shows it ready and
  bit-checked): (1) a fused IVF score + top-k kernel (no 402 MB score tensor: −15-25 ms, −0.4 GB); (2) a fused private
  search (score + top-8 + re-rank); (3) a fused gate feature build + trunk + threshold / selection. Each has a torch
  reference and a top-k equality test.
- **Brute-force global search** instead of IVF: +85-135 ms (27.5 × 10⁹ bf16 scores, ≈ 110 GB of score traffic): over
  budget without a fused kernel; with the Triton kernel it becomes a T2 comparison, not the default.
- Gate cost alone: ≈ 9-14 ms forward + 2-3 ms update ≈ 0.6-0.9% of the step.

---------------------------------------------------------------------------------------------------------------------

## 18. Tests before any run

All CPU work: `CUDA_VISIBLE_DEVICES=-1`, `torch.set_num_threads(2)` (≤ 2 threads), no installs, nothing deleted. The GPU
belongs to the speed study until it releases it.

### 18.0 Pre-run GO gates (rev 2)

**Tier 1 — GO before any gate or write code is written** (memory search / store scaffolding and the fixtures may be
written meanwhile):

| gate | what | GO / NO-GO |
|---|---|---|
| **G1 = T0a paraphrase — the decisive gate** (rev 2, T0b) | §18.1 row T0a. Since the gate cannot remove held-out harm (§5.0), robustness of the read across wording carries the harm bar: besides doc → doc / doc → cloze recall, T0a reports the **same-template-other-fact top-1 rate on held-out templates** and the A of those wrong winners | as there; in addition, a held-out-template wrong-winner rate with A ≥ 0.37 above 10% is reported to the owner as a harm risk even at GO |
| **G2 = T0c pass matrix incl. the averaged key** | §18.1 row T0c: the 6 × 6 pass matrix for single-pass keys AND the averaged key (one row "avg" queried from every pass), plus `maxgrad` | as there; the write-key rule (avg / uniform / maxgrad) is fixed here |
| **G3 = ORACLE CEILING** | Old run ckpt 6K: forward + backward over documents holding **mention n** of each test fact (300 late + 300 rare/early facts × 5 relations, eval tags only for bookkeeping); write **one-shot** blobs exactly as §8 would (averaged key, content −Σ_k g summed over in-window passes, normalised to 0.1 × median‖xf‖, and a second store at 0.2 for the record), global + the private copies the D2 budget would make, into a store padded with T0h-criterion distractors from ordinary windows to 256K global / 512 per expert. Ckpt 7K (`last.pt`), its rows knocked out (the old memory off, so only the blobs act): score the cloze of **mention n + 1** with the 25d read injected into the expert inputs (a patched RoutedMLP in the test script only), **gate forced open** on every kept pair (s = 1), **exact (brute-force) search**, the §6.1 kernel and blend. Baseline: the same 7K forward without blobs. GPU when free; a 50-fact CPU smoke first | **NO-GO** if the late-fact gain (cloze exact, blobs − no blobs) **< +0.045**, or the rare-fact cloze with blobs **< 0.025**. Reported also: the gain at 0.2 init, global-only vs private-only, the share of gain at pass ≥ 3, and **the held-out harm**: the non-answer world CE delta (blobs − no blobs) on val_world windows (held-out templates, fresh documents) with the gate forced open — an upper bound of the harm the gate would have to remove; > +0.04 nats goes to the owner with the GO / NO-GO |

The oracle is a ceiling for the read path (perfect gate, perfect search, 1K-step drift, untrained one-shot contents
on a brain that never saw such reads): if even this cannot lift late facts by +0.045 (≈ 5 sd at n = 5,000) or reach a
rare-fact cloze of 0.025, the gated, trained system will not reach the GO bars of §19 and the plan goes back to the
owner before the gate / write code exists.

**Tier 2 — pre-run GO gates** (before the 12.2K run; CPU from captures unless noted):

| gate | what | GO / NO-GO |
|---|---|---|
| **G4 = T0d IVF recall on write-distributed keys** | §18.1 row T0d, with the gallery drawn from write-candidate positions (σ-selected, averaged keys) instead of uniform positions, cells fitted on the write / query mix, n_p 8 / 16 / 32, Cq 384, the spill rule | miss ≤ 2% for d₁ < 2h₀² at n_p 16; R@16 ≥ 0.9 of brute force; overflow ≤ 1%; spills ≤ 5% |
| **G5 = drift 3K → 7K** | §18.1 row T0f extended: old run ckpt 3K → 6K → 7K (and control 3K → 6K): median drift / 1-NN gap per 1K steps, self-recall R@1 / R@16 at 256K over 1K, 3K and 4K gaps (the global life is ≈ 4,096 steps), the n_m gain part, the staleness of a map and cells fitted 1K / 4K earlier | reported; R@16 over a 2K gap < 0.5 on world cues → the owner (young-blob protection and renewal must carry everything); gain part > 30% → `--gm-addr pregain` offered (OQ-6) |
| **G6 = gradient agreement** | On the T0a / oracle captures: cos(−g at mention n, −g at mention n + 1) at the cue, same fact vs same-template-other-fact vs random; by pass and for the summed-over-passes content | GO: median same-fact cos ≥ 0.2 and ≥ 0.1 above same-template-other-fact; sets θ for the merge test (T0e) and checks the confirmation content test (cos > 0.3: pass rate on same-fact hits reported) |
| **G7 = private reach** | §18.1 row T0r under the D2 budget: the fraction of mention-(n + 1) cue experts (each pass) that hold a private copy of the fact written at mention n by the credit rule; P(≥ 1 reached expert finds it, R@1 at 512); the split-twin effect | private spaces count as a carrier in §9.4 only if P(≥ 1 reached expert finds it) ≥ 0.3; else reported (the global space carries late and rare facts) |
| **G8 = hubness** | At 256K with write-distributed keys: the k-occurrence N₁₆ distribution of keys over 64K queries (skewness), the share of queries whose top-1 is one of the top-1% hubs, hubness after the 64 → 384-d re-rank | GO: top-1% hubs take ≤ 10% of top-1s (online broken-if is 20%, B13) |
| **G9 = template collisions, 64-d vs 384-d re-rank** | On T0a: the same-template-other-fact top-1 rate with the 64-d search alone vs the 64-d top-16 re-ranked by the standardised 384-d distance ‖(xf − μ)/σ − ·‖² (keys stored at 384-d for the test only) | reported; if the 384-d re-rank halves the collisions AND lifts doc → cloze R@1 by ≥ 0.05, the owner is told (a 384-d re-rank would add ≈ 0.4 GB of keys; not in this plan) |
| T0h, T0e, T0g | §18.1 | as there |
| T0b (GPU), T2 bench + 1K pilot (GPU) | §18.2, §18.5 | as there |

### 18.1 Real-thought checks with the NEW address (xf), from captures

Captures (CPU if a 4-window timing shows ≤ 1 min per window, else GPU when free): checkpoints old run 3K, 6K and 7K
(`runs\world_mem_life_12k\ckpt_3000.pt`, `ckpt_6000.pt`, `last.pt` at 7K) and the no-memory control 3K / 6K / 9K
(`runs\world_v4d_12k\ckpt_3000.pt`, `ckpt_6000.pt`, `ckpt_9000.pt`; the closer proxy for a brain without rows). Train-mode routing
(head-group offset, route noise 0.3, capacity 1.5) and eval routing. At every pass 0..9 and at the cue (the token before
the answer): xf, the routed experts and their capacity slots, side(s), μ_l of the previous pass. The fixed map is fitted
on 64K positions of the same checkpoint (no step-300 checkpoint of a 25d run exists; T0f measures how much a
map fitted at 6K costs at 7K).

| test | what | GO / NO-GO |
|---|---|---|
| **T0a paraphrase** (cross-document same fact) | 300 late + 300 early people × 5 relations = 3,000 facts, each in ≥ 3 documents with 3 training templates (0-9) and other surrounding facts, + 1 held-out-template cloze prompt (10-11). Global gallery = doc-1 cue keys (averaged over the in-window passes, as §8.2; single-pass keys in T0c) + distractors written by the T0h criterion from ordinary windows at 64K / 128K / 256K; queries = doc-2/3 cues and the cloze cue at every pass. R@1 / R@16 with brute force AND the IVF; same-template-other-fact top-1 rate (template keying); keys per fact after dedupe. Within a checkpoint and 6K → 7K | **GO**: doc → doc R@1 ≥ 0.6 and doc → cloze ≥ 0.5 at 256K (best query pass) and 6K → 7K doc → cloze ≥ 0.35. **Weak GO**: doc → cloze 0.35-0.5 or doc → doc 0.45-0.6: the owner gets the §9.4 survival at the measured p_hit. **NO-GO**: doc → cloze < 0.35 → back to the owner |
| **T0c cross-pass findability (pass matrix)** | On the T0a capture: a blob keyed at pass i, queried from pass j (i, j ∈ 0..5), same document (age 0) and other document. R@1 / R@16 matrix; **plus the rows "avg" (the key averaged over the in-window passes), "uniform" (one random pass) and "maxgrad"**, each queried from every pass; the distance ‖x̂_i − x̂_j‖² of the same token vs h₀² and the 1-NN gap; the estimated n_eff (the number of passes whose query finds the key) per key rule | **GO**: for the chosen key rule, R@1 from every pass ≥ 0.7 × the best single cell, and n_eff ≥ 2. **Weak GO**: n_eff 1.3-2: tell the owner (§9.4 row n_eff = 1-2). **NO-GO**: n_eff < 1.3 for every key rule: one key per pass region is needed (write at 2-3 passes per position, owner, OQ-2). The best rule becomes `--gm-write-key` |
| **T0r expert-reuse / private reach** | For the same fact: overlap ρ_e between the write-time cue's 6 experts and the later cue's experts (each pass), other document; P(≥ 1 shared expert); the private hit rate inside the shared experts at 512 density (gallery = that expert's own tokens' keys from ordinary windows); the D2 credit rule's reach (G7); the effect of a split twin (child holding a copy at half strength) | G7 |
| **T0h rare-fact writes** | Training windows, ckpt 7K and control 6K, train-mode routing, ≈ 64 windows forward-only: σ^raw / σ^ent / σ^imp at W = 32 per micro-batch with ≤ 3 and ≤ 4 per sequence (the cap is fixed here); the per-expert credit picks (D2); how many selected positions are ever re-cued (decides whether the probation flag is worth T1); the fraction of rare (early) AND late answer-cue positions in the top-W (eval tags = bookkeeping only); s_late for unacquired late cues (§9.6); wiki vs world share | **GO**: best criterion p_write ≥ 0.5 on rare cues and s_late ≥ 0.25; **weak GO** 0.3-0.5 / 0.1-0.25 (W = 96 until 3K enabled); **NO-GO** < 0.3 |
| **T0e calibration** | On the T0a capture: d_same → h₀,g², h₀,p², κ; the initial σ² (median(d₂ − d₁)/(2 ln 2)); A_w; ω₁, μ_m, A_min at precision ≥ 0.95 that j₁ is the same fact (global and private); γ **and θ** (the content-agreement threshold, from G6's gradients) at wrong-merge ≤ 2% at 256K and at 512 (with the same-fact merge rate); p_conf on correct hits incl. the content test | defaults written; no γ ≥ 0.05 → dedupe off for that space type; no threshold at precision 0.95 with pass rate ≥ 0.3 → the owner is told (§9.4 with lower p_conf) |
| **T0d IVF recall** (= G4) | IVF (1,024 cells fitted on the write / query mix, n_p 8 / 16 / 32, Cq 384, spill rule) vs brute force on **write-distributed** keys (T0a keys + σ-selected averaged distractor keys) at 64K / 128K / 256K: the fp32 top-1 miss rate for d₁ < 2h₀² and overall, R@16, probe overflow, spills, cell Gini; cells fitted at 6K, used at 7K, and refitted | G4 |
| **T0f drift 3K → 7K of x̂** (= G5) | Same tokens / windows at 3K, 6K and 7K: median drift / 1-NN gap per 1K steps, self-recall R@1 / R@16 at 256K density over 1K / 3K / 4K gaps, the part due to the n_m gain (§2.2), the map / cell staleness (μ, σ, centroids fitted 1K or 4K earlier) | G5 |
| **T0g pass coverage** | E[K], E[in-window passes] from the old run's K log | 6.4 / 4.94 within ±10% |

### 18.2 T0b gate separability
**Done on the proxy (rev 2)**: `speed_wip\splat\realcheck\t0b\t0b_results.md` (old rows as the read, ckpt 7K, 128
train.bin + 128 val_world windows): verdict NO — sign-of-benefit AUC 0.52 train / 0.61 val / 0.49 train → val;
magnitude 0.68-0.74; match features add nothing; harm only on held-out text; c vs knockout corr 0.22. Consequences are
in §5.0 (the gate is kept, not counted on for harm; target = b̃ only).

**Required repeat on the blob pilot** (GPU, after the 1K pilot, before the 12.2K run): the same harness on the pilot's
real store and real gate, train.bin and val_world windows, train and eval routing: (a) knockout benefit by token class
(on vs ko_gm_all / _global / _private); (b) b̃ (s ~ U(0,1)) and the curvature-corrected Ĝ vs knockout benefit per
position (correlation; must beat the old proxy's 0.22 to keep Ĝ for confirmation, else the owner is told);
(c) out-of-fold AUC for the sign and the magnitude of b̃ with the §5.1 features, with and without match quality and
the two extra inputs; train → val transfer; (d) what the trained gate's open set gives on val_world vs always-open
(Σ nats). **Reported to the owner with the pilot**; it is not a NO-GO for the run by itself (the gate is not relied on),
but a held-out harm (B22) > +0.04 on the pilot with T0b showing the gate cannot separate it is.

### 18.3 T1 toy (CPU; extend `C:\Eitan\Python\NewBrain\speed_wip\splat\consol_toy.py`)
One global space (256K) + 768 private spaces (512) with routing (6 of E_live, overlap ρ_e from T0r, births 192 → 768
with copies), K ~ the run's distribution, cross-pass key spread from T0c, drift from T0f, paraphrase factor from T0a,
p_write and s_late from T0h, γ, p_conf from T0e. Late facts (Δ 105, 116 exposures), rare (≈ 9.5 exposures, spacing
1.3K). Policies: default (D2 budget, averaged key, causal λ gate, benefit strength, young protection);
`--gm-renew-geo 1`; `--gm-track all`; W 64 vs 96-until-3K; n_p 16 vs 32; **content optimizer sweep**: row-Adagrad
(default) vs lazy row-RMSprop (β₂ 0.9 per touch, η 0.03); **owner options** (pending): probation store on / off;
private-by-consolidation vs direct private writes; exploration warm-up 20% → 5% vs off. Thresholds: late-fact trained-blob survival ≥ 0.8 at 12.2K; rare-fact recall ≥ 0.10; wrong merges ≤ 2%; keys per fact
≤ 1.5; late fill ≥ 80% by 5K steps; anti-capture: < 1% of tracked blobs with net travel > 4 h₀. A miss on late facts goes
to the owner before the run.

### 18.4 Unit tests (CPU, ≤ 2 threads)
0. **Before any code edit**: record reference fixtures with the CURRENT code (tiny LoopBrain configs, fixed seeds, 3
   steps forward + backward + optimizer step, CPU float64): (a) the no-memory control config, (b) the old memory config
   (rows + private + gate on), (c) the old run's MoE / loop flags at tiny size. Saved under `test_fixtures\25d_ref_*`.
1. **Flags-off bit-identity**: `--gm 0` reproduces (a), (b), (c) bit-for-bit (losses, parameters, buffers, RNG state)
   after the edits; `bmem=None` takes exactly today's RoutedMLP path.
2. Routing invariance: with `--gm 1`, the routed indices, capacity drops, balancer counts equal `--gm 0` on the same
   inputs (the read never reaches the router).
3. IVF search = brute force on the top-16 when n_p = all cells; spill rule; refit reassigns every live slot; recall
   measurement code; static overflow handling; no `bincount` in the op log.
4. Private search = a per-expert loop reference (bf16 top-8 → fp32 top-4 = exact fp32 top-4 up to bf16 ties beyond 8);
   dropped pairs get no read / write / Scale value.
5. Gate: causal threshold (open iff ĝ > λ_e; λ update rule; a perturbation of positions > t never changes open at ≤ t,
   incl. buffer overflow dropped latest-first); exact stratified exploration counts and propensities; static
   compaction; eval without exploration; β init negative → only exploration reads at step 300.
6. **Scale once under checkpoint + b̃ unbiased + gate gradient = autograd on exploration pairs only + accum invariance
   + query gradient through w, A** (`test_gm_gate_postbwd`, §5.5).
7. GatherRead gradients vs a dense reference (contents, w, A, x̂, ζ) incl. the bounded kernel near d₁ = 0 (finite
   gradient) and the normalised blend; fp32 grad buffers; stochastic rounding is unbiased (1,000 steps).
8. Writes: selection static sizes, ≤ 4 per sequence, ≥ 8 apart; credit counters (exact 0.256 / step long-run rate,
   one write per expert per micro-batch max); averaged key = the mean of per-pass x̂; captured gradient = Σ over the
   in-window passes of autograd ∂L/∂buf; out-of-window passes and checkpoint recomputes leave the accumulators
   unchanged; global content = the sum over kept experts + shared; norm caps; ε guard (zero gradient → ξ-only, no NaN).
9. Dedupe / merge (content test, move budget spent, η floor, within-step suppression, cell change), spawn / eviction
   (touched excluded, young-blob protection ≤ 50%, benefit strength, spill rule).
10. Confirmation / move (α_j, cap min(σ, 0.5 h₀), content test, ≤ 1 move per step via segment mean = a loop reference,
    ≤ 8 per 1K incl. merges, per-space caps, path length / net travel).
11. Births: split copies a parent's private space (strength halved), gate rows, λ_e exactly (and its AdamW moments);
    ramp leaves the child empty; function unchanged at birth; no memory tensor in `brain_params`, `make_opt` or
    `carry_optimizer`.
12. Causality: a perturbation of tokens > t does not change any read, gate value or injected vector at ≤ t (finite
    difference on a tiny model); micro-batch 2 sees the same store as micro-batch 1; eval is read-only and order
    independent.
13. Lazy strength EMA = eager; h₀ / σ² tracking on exploration reads does not contract on a synthetic population;
    floors (h₀, σ², MAD, δ).
14. No new host sync (a sync counter / TorchDispatchMode op log on a CPU mock); resume (store, cells, map, gate,
    generator) gives identical next steps.

### 18.5 T2 GPU bench and pilot (after the speed study releases the GPU)
- Harness `fruit-fly-sim\speed_wip\vram\vram_common.py`: the old config, the old config with `--mem-shared 0
  --mem-private 0 --mem-coarse 0` (the no-memory build), and 25d, at 16 × 2, K ∈ {2, 6, 10, 16}: ms per §17 row, peak
  allocated / reserved incl. one `evaluate()`, host syncs (must equal the no-memory build's), IVF miss / overflow.
- **Pilot**: 1,000 steps of 25d (throwaway): fill, open rates, the T0b repeat on the real store (§18.2), the held-out
  harm B22, the broken-if table (§16.1), confirmation and move counts, travel, reserved VRAM over ≥ 200 steps incl.
  `evaluate()` and a birth (births start at step ≈ 610, inside the pilot), launches per step,
  tok/s. A fault stops the plan before the 12K run.
- GO: tok/s ≥ 0.97 × the old run's config (D1); reserved ≤ 20 GB at K = 10 and 16 incl. `evaluate()` and a birth over
  ≥ 200 steps; memory overhead vs the no-memory build and launches per step reported.

---------------------------------------------------------------------------------------------------------------------

## 19. The run and GO criteria (outline §10; statistics from 25c §16.2)

### 19.1 The run (12.2K steps; the owner's go)
The old run's config with `--mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-nbr 0 --mem-gate 0` + the 25d flags
(§21); current data, random-window loader, 16 × 2, the VRAM defaults; auto-eval at 25 / 50 / 75 / 100% and at 7,000.
References at the same steps: the old run `world_mem_life_12k` (to 7K) and the control `world_v4d_12k` (6K, 9K, final).
**Early stop at 7K**: late < 0.24 or rare ≤ 0.016 → stop, report knockouts and logs; the owner decides.

### 19.2 GO criteria

| criterion | bar | statistical basis |
|---|---|---|
| late facts (cloze exact, n = 5,000) | ≥ **0.27 at 7K** (old 0.280) AND ≥ **0.28 at the end** (control 0.260 + 0.02) | sd of a difference of two n = 5,000 proportions at 0.28 ≈ 0.009: −0.01 ≈ 1.1 sd non-inferiority; +0.02 ≈ 2.2 sd; paired McNemar on the same prompts |
| rare / early facts (n = 1,000) | ≥ **0.035 at 7K** (old 0.016) AND ≥ **0.045 at the end** (control 0.023); ≥ 3 of 5 relations improve | two-proportion z: 0.016 → 0.035 gives z = 2.70; 0.023 → 0.045 gives 2.71; primary test exact McNemar at p < 0.01 |
| harm on other world text | world_other CE (memory on − ko_gm_all) ≤ **+0.04 nats** (old +0.13) AND world_all(on) ≤ world_all(ko) AND |Δ wiki| ≤ 0.01 | 54,104 world_other tokens: sd of a paired mean-CE difference ≈ 0.005 nats |
| speed (D1) | tok/s ≥ **0.97 ×** the old run's config, same harness; ALSO reported: the memory's cost vs the no-memory build (expected 3.6-5.1%) and kernel launches per step (expected ≈ 2,100) | T2, median of ≥ 3 runs |
| VRAM | peak reserved ≤ **20 GB** incl. `evaluate()` and a birth, judged over ≥ 200 steps (cap 0.78 on) | T2 + run log |

Reported with every eval: ko_gm_all / _global / _private, gate modes, the fill logs, §15's table, §16.1's table.
**The harm bar is met by read robustness, not by the gate** (§5.0): the eval world_other set is held-out text (val_world,
templates 10-11), exactly where T0b found the harm; the gate is not expected to remove it. B22 tracks it during the run.

### 19.3 Order of work
1. This plan → owner review of §OD-P (the pending options) and the open questions.
2. CPU, alongside the speed study: fixtures from the current code (§18.4-0), unit-test scaffolding, T0g; the capture
   (if the CPU timing allows); **Tier 1: T0a, T0c (incl. the averaged key)**; the ORACLE CEILING smoke on CPU.
3. GPU when free: the oracle ceiling (G3); the capture if not done on CPU. **Tier 1 GO → gate and write code may be
   written.** Store / search scaffolding may be written before.
4. Tier 2 (G4-G9), T0h, T0e, then T1 (incl. the optimizer sweep and the pending options' comparisons).
5. Implement 25d (+ the VRAM defaults if needed); unit tests; T2 bench; 1K pilot with the T0b repeat and B22.
6. The 12.2K run with the owner's go; GO at 7K and at the end.

---------------------------------------------------------------------------------------------------------------------

## 20. The owner's constraints — status

| # | constraint | status |
|---|---|---|
| 1 | addresses from thoughts only, not learned | met: key = query = fixed map of xf; μ, σ, R, the IVF cells, h₀ and thresholds are statistics; keys move only by rules (spawn, merge, confirmation) |
| 2 | no gate loss on the thought | met: φ, m detached; the gate's loss runs on stored detached copies in a separate graph after backward. The LM loss reaches xf through w, A (decided) |
| 3 | no pointers, no recompute loop | met: writes, confirmations and the gate update use only the step's own forward / backward |
| 4 | filled during training | met (from step 300) |
| 5 | speed / ≤ 20 GB | ≈ 19.3-19.7 GB reserved projected (judged over ≥ 200 steps incl. eval and a birth); net 2.5-17% faster than the old run (the decided bar); memory 3.6-5.1% vs no memory, ≈ 2,100 launches / step (reported) |
| 6 | the brain decides at every pass | met, per (token, chosen expert), causally (ĝ > λ_e), with two decided bounds (≤ 50% per expert via λ_e, 5% exploration). T0b: the decision cannot remove held-out harm (§5.0) |
| 7 | nothing per pass | met: one global space and per-expert spaces shared by all passes; no pass feature in the gate; K up to 16 (or 40) changes nothing in the memory |
| 8 | self-reinforcement measured | §15 |

---------------------------------------------------------------------------------------------------------------------

## §OD. Defaults set by this plan (the owner may object) and open questions

### §OD-D. Decided (rev 2; recorded, not open)

| | item | decision |
|---|---|---|
| D1 (was OQ-1) | speed bar | tok/s ≥ 0.97 × the old run; cost vs no memory and launches / step reported |
| D2 (was OQ-3) | private writes | per-expert per-step budget 0.256 (credit counter), filled with the expert's most surprising kept tokens; life ≈ 2,000 steps; lifetime logged per bank |
| D3 (was OQ-4) | shared expert's private writes | the same budget as an average routed expert |
| D4 (was OQ-5) | gate decisions | one per (token, expert) for global + private |
| D5 | blend | weighted by match: r = max(A_g, A_p)(π_g u_g + π_p u_p), π ∝ A |
| (was OQ-7) | IVF refit | every 1,000 steps (index only), reviewer fix H3 |

### §OD. Defaults set by this plan (the owner may object)

| | item | default | alternative |
|---|---|---|---|
| 1 | global search | IVF, 1,024 cells (write / query mix, refit 1,000), n_p 16, spill to 2nd cell | brute force (+85-135 ms/step; Triton-fused as a T2 comparison) |
| 2 | write budget | W = 64 per step (32 per micro-batch), ≤ 4 per sequence | ≤ 3 per sequence (T0h) |
| 5 | write key | averaged over the in-window passes; content summed over them | `uniform`, `maxgrad` (T0c decides) |
| 6 | private contents | bf16 + stochastic rounding; all grad buffers fp32 | fp32 contents (+0.3 GB) |
| 7 | geometric renewal | off (in-window confirmation reaches the bar at n_eff ≥ 2) | on |
| 8 | gate | shared trunk (32) + per-expert head, causal threshold λ_e per expert | λ pooled; low-rank per expert; shared only |
| 9 | move rule | α_j = max(0.3, 1/(n_j + 1)), each move ≤ min(1σ, 0.5 h₀), ≤ 8 moves (incl. merges) per 1K steps | — |
| 10 | content optimizer | row-Adagrad, no momentum, no decay | lazy row-RMSprop (T1 sweep) |
| 11 | shared expert reads | global + private | global only (`--gm-shared-read global`) |

### §OD-P. Owner options (pending Eitan; coordinator defaults, owner may override)

| | option | proposed by | default (owner may override) | decided by |
|---|---|---|---|---|
| P1 | exploration warm-up 20% → 5% over steps 300-1,300 | reviewer | **OFF**; instead match-stratified exploration within the 5% with known propensities, and a log of the steps until the gate's decile slope reaches 0.7 | owner; T1 compares |
| P2 | extra gate inputs: previous pass's halting confidence; content-fit score σ(n(xf) · n(W_cf c)) | reviewer | **ON**, both detached | owner; T0b repeat on the pilot reports what they add |
| P3 | probation / candidate store (+0.26 GB) | reviewer | **OFF**: direct writes (W = 64 / step, ≤ 3-4 per sequence); the probation stage behind `--gm-probation 1` | T0h (re-cue rate), T1 |
| P4 | private-by-consolidation | reviewer | **OFF**: direct budgeted private writes; `--gm-private consolidate` | T1 comparison |

### §OD-A. Accepted rebuttals (the reviewer's point answered; the plan keeps its rule)
- Global life ≈ 4,096 steps at W = 64 (§3.2).
- All 6 routed experts of a token are private-write candidates at each write pass (§8.1).
- Writes start at the fit step; R@16 is measured, not a gate on writing (§8.1, T0d).
- No fixed top-25% read fallback: alert only (§5.6).
- No dead-bank recycling (§11.3).
- Content init 0.1 × median‖xf‖, normalised (§8.3).
- Row-Adagrad with no momentum and no decay; RMSprop only as a T1 sweep (§6.3).

### §OD-R. Rejected (with reasons)
- **Holding writes until R@16 ≥ 0.9**: without a backup memory, fill is the first-order risk (§9.6); delaying writes
  delays every fact's first blob, and R@16 at a near-empty store says little about R@16 at 256K. R@16 is measured
  (online sample, T0d / G4) and alerts (B10), it does not gate writing.
- **A fixed top-25% read fallback** when the gate looks weak: the outline says no imposed budget and the expert decides;
  a fixed fraction would override the gate exactly when it is being measured. Alert only (§5.6). (T0b also shows a
  fallback by ĝ rank would not remove held-out harm.)
- **Dead-bank recycling**: the code has no expert death (§11.3), so no private bank ever goes dead; the young-blob
  protection and benefit strength already recycle unused slots inside each bank.
- **Random K**: K (the pass count) keeps the old run's distribution (U{2..10} w.p. 0.95, U{12..16} w.p. 0.05) so the
  run stays comparable to the old run and the control; the memory is pass-agnostic by constraint 7, so a different K
  schedule tests nothing the cross-pass checks (T0c) do not already measure.

**Open questions (still open)**
- **OQ-2 (cross-pass)**: if T0c gives n_eff < 1.3 for every key rule (incl. the averaged key), should a position write
  keys from 2-3 passes (more keys per fact, capacity cost), or should the owner reconsider the pass-agnostic design?
- **OQ-6**: the n_m gain in the address (§2.2): xf as specified, or the pre-gain normalised input if G5 shows the gain
  dominates the drift?
- **OQ-8 (rev 2, T0b)**: if the pilot's T0b repeat shows held-out harm (B22) > +0.04 that the gate cannot separate, the
  remedy has to be on the read side (key robustness across wording, contents); the owner decides whether to proceed to
  the 12.2K run.

---------------------------------------------------------------------------------------------------------------------

## 21. Flags and implementation size

**Run defaults**: `--empty-cache-eval 1 --mem-fraction 0.78 --eval-batch 16 --batch 16 --accum 2`.

**25d**:

    --mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-nbr 0 --mem-gate 0                (old memory code: kept, off)
    --gm 1 --gm-global 262144 --gm-private 512 --gm-dv 384 --gm-key 64 --gm-fit-step 300 --gm-write-start 300
    --gm-map std-randproj --gm-addr xf
    --gm-search ivf --gm-cells 1024 --gm-cell-cap 512 --gm-cell-spill 1 --gm-probe 16 --gm-qcap 384
    --gm-topk 16 --gm-blend 8,4 --gm-ivf-fit mix --gm-ivf-refit 1000 --gm-triton 0
    --gm-write 64 --gm-seq-cap 4 --gm-write-key avg --gm-surprise <T0h: raw|ent|imp>
    --gm-priv-budget 0.256 --gm-shared-budget avg
    --gm-dedupe <T0e γ_g>,<γ_p> --gm-merge-cos <T0e θ> --gm-merge-eta-min 0.25 --gm-h0 <T0e>
    --gm-vfade 0 --gm-strength benefit,4000 --gm-protect 2000,0.5 --gm-child-strength 0.5
    --gm-init 0.1 --gm-vcap 0.3,0.5 --gm-opt adagrad --gm-lr 0.01 --gm-clip 0.05 --gm-priv-bf16 1 --gm-grad-fp32 1
    --gm-kernel bounded --gm-blend-norm 1 --gm-inject-scale 1
    --gm-confirm curv --gm-qc 0.25 --gm-conf <T0e> --gm-conf-cos 0.3 --gm-alpha-min 0.3 --gm-move-cap 1sigma,0.5
    --gm-move-max 8,1000 --gm-move-space 256,2 --gm-renew-geo 0 --gm-track-travel 4096
    --gm-gate trunk --gm-gate-rank 32 --gm-gate-feats spec+halt+fit --gm-gate-target scale --gm-beta0 -0.02
    --gm-lambda expert --gm-lambda-lr 0.01 --gm-cap 0.5 --gm-buffer 0.55 --gate-cost 0
    --gm-explore 0.05 --gm-explore-strat 4 --gm-explore-warm 0 --gm-placebo 0.01
    --gm-gate-lr 1e-3 --gm-gate-wd 0.01 --gm-gate-clip 1.0 --gm-match-grad 1 --gm-shared-read all
    --gm-probation 0 --gm-private direct
    --gm-heldout-check 1000,32                                   (B22: val_world subset every 1,000 steps)
    --auto-eval 1 --eval-at 7000 --save-report-ckpts 1
    (if T2 needs them: --chunk-ce 1 --ckpt-concept 1)

**Ablations / debug**: `--gm-search brute`, `--gm-probe 8|32`, `--gm-ivf-refit 0`, `--gm-write-key uniform|maxgrad|last`,
`--gm-renew-geo 1`, `--gm-track all`, `--gm-confirm off`, `--gm-rescue 1`, `--gm-gate lowrank|shared`,
`--gm-gate-feats spec|+cu|+pass`, `--gm-lambda pooled`, `--gm-gate-budget fixed`, `--gate-cost time`,
`--gm-match-grad 0`, `--gm-priv-bf16 0`, `--gm-opt rmsprop` (β₂ 0.9 per touch, η 0.03), `--gm-shared-read global`,
`--gm-addr pregain`, `--gm-map pca`, `--gm-triton 1`, `--ko-gm all|global|private`. **Pending owner options**:
`--gm-explore-warm 0.2,300,1300`, `--gm-probation 1`, `--gm-private consolidate`. Removed: `--pm-rate`, `--gm-gate2`,
`--gm-fade`, `--gm-zombie` (log only).

**Implementation size (estimate)**

| part | lines |
|---|---|
| `blob_memory.py`: two space types, map fit, IVF (mixed fit, refit reservoir, table rebuild with spill, query layout, bmm, merge, re-rank), private bmm search, match features, bounded kernel + normalised blend + ζ, GatherRead + Scale, static workspaces, fp32 grad buffers, row-Adagrad / RMSprop (+ stochastic rounding, norm caps), write stash / credit budgets / averaged keys / pass-summed content, dedupe (content test) / merge / spawn / evict (spill, young protection, benefit strength), confirmation (curvature-corrected, content test) + guarded moves + travel, h₀ / σ² / κ̂ / Ĉ EMAs, births / reset hooks, state / resume, stats + broken-if checks | ≈ 1,100 |
| gate (`BlobGate`): features (+ halting confidence, content-fit), trunk + per-expert heads, causal λ selection + static buffer + token-order overflow, stratified exploration + s draws, kept exploration tensors, post-backward graph (b̃ + control variate), watchdog, eval modes | ≈ 300 |
| `spiking_lm.py`: the `bmem` argument and the shared-expert input | ≈ 20 |
| `loop_brain.py`: build / RNG fork, ctx per pass, `bmem` through Core, prev μ / match, `select_writes`, hooks, knockouts | ≈ 130 |
| `train_lm.py`: flags, asserts, fit step, write selection / after_backward / after_step calls, gate optimizer group, empty_cache / fraction cap, auto-eval at 7,000, logging, resume | ≈ 150 |
| `eval_world.py`: token-type split, knockouts, gate modes, p_open / p_hit, late-fact fill | ≈ 150 |
| unit tests (§18.4, incl. the reference fixtures) | ≈ 600 |
| T0a / T0c / T0d / T0e / T0f / T0g / T0h / T0r scripts, the oracle ceiling (G3), G6-G9, the T0b repeat on the pilot, T1 toy extension, T2 bench + pilot wrapper | ≈ 600 |
| **total** | **≈ 3,000** (memory code ≈ 1,550; tests and checks ≈ 1,200; eval ≈ 150; trainer ≈ 150) |

---------------------------------------------------------------------------------------------------------------------

## 22. Risks

1. **No backup memory (the biggest risk).** If blobs are not written before they are needed (slow fill), not found
   again (T0a paraphrase, T0c cross-pass, T0f drift) or not renewed (§9.4), late cloze falls toward the no-memory
   curve (control ≈ 0.11 at 7K by interpolation of 0.079 at 6K and 0.182 at 9K), far below 0.27. The pilot and the 7K
   stop rule bound the cost.
2. **Cross-pass findability** (outline risk 2): one space for all passes is the core bet of 25d. If a token's xf moves a
   lot between passes, a blob written at pass i is not found from pass j; n_eff → 1 and survival needs p_hit ≥ 0.85
   (§9.4). T0c measures it before any build.
3. **Template keying**: xf contains the token and the context; same-template other-fact neighbours may win (the rows
   showed 0.24 vs 0.045). The clear-winner margin and A protect reads and moves; T0a measures it.
4. **Density at 256K**: the extrapolated unrenewed 1K-gap R@1 is ≈ 0.40 (world ≈ 0.10); rare facts suffer most. Renewal
   keeps late facts young; private spaces give a low-density second chance if routing repeats (T0r).
5. **Private spaces are thin** (outline risk 3): an expert sees ≈ 0.8% of tokens; with the D2 budget a fact enters only
   the experts whose credit fired on it (≈ 197 private writes per step in all), lives ≈ 2,000 steps; twins after a
   split split the tokens. The global space carries rare facts; private spaces are a bonus that G7 (private reach)
   quantifies.
6. **Gate separability (T0b, measured on the proxy)**: sign-of-benefit AUC 0.52 on training text, 0.49 train → held-out;
   match features added nothing. The gate will likely learn magnitude ("reads matter here") better than sign. It stays
   as the owner's design and as the 50% cost bound; nothing in the GO case depends on it. The pilot's T0b repeat
   re-measures it with the blob store and the unbiased target.
7. **Harm on held-out text (the old +0.13)**: T0b shows it is a generalisation failure (reads keyed by training wording
   misfire on new wording), absent on training text, so **a training-time gate cannot remove it**. Guards: read
   robustness (T0a decisive; G9 template collisions; the oracle's held-out harm), A muting far matches, the bounded
   kernel, the normalised blend (no doubled reads), the norm cap, the content test on merges / confirmations, and B22
   during the run. If B22 stays above +0.04, the harm bar fails regardless of the gate (OQ-8).
8. **IVF cells**: thought drift can unbalance cells between the 1,000-step refits (overflow, spills, misses). Logged and
   alerted (B10).
9. **Speed**: realistic 70-100 ms (3.6-5.1%) vs no memory, launch-bound (≈ 2,100 launches / step); the decided bar
   (0.97 × the old run) has margin; static shapes allow CUDA graphs and Triton fusion later.
10. **VRAM margin** 0.3-0.7 GB below 20 GB reserved (fp32 grads cost +0.3 GB); `chunk_ce` + the checkpointed concept loss
    are ready as defaults (§VRAM).
11. **Self-reinforcement** (reads → thought → key → read; key moves; match-weight gradient) is live from step 300;
    measured (§15), with `--gm-match-grad 0` and `--gm-confirm off` as ablations.
12. **Comparability**: the old run read every position (rows, address reuse); 25d reads gated (token, expert) pairs;
    same data and loader; the old run ends at 7K, so the end-of-run comparison is with the control only.
13. **Estimator variance**: b̃ is unbiased but noisy (one s per pair, ≈ 2.9K exploration pairs per micro-batch-pass);
    the control variate and the Huber loss bound it; the decile slope (§5.4) shows how fast the gate learns.
