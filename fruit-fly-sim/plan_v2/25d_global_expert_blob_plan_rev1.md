# 25d. Global + per-expert blob memory, the expert decides (PLAN ONLY) — 2026-10-24

Full plan for the outline `25d_outline.md` (the owner-approved spec; its §2b fixes what the address and the gate see).
It replaces 25c (`25c_blob_only_per_pass_plan.md`, per-pass banks, rejected, kept as a record). The blob mechanics,
reviewer fixes and statistics of 25c are reused; where this file says "25c §x" the derivation there still holds and
is not repeated, only adapted.

Nothing was trained, no code was edited, the GPU was not used. Only this file was written (plus a scratch calculator,
CPU only, in the session scratchpad, for the numbers of §3, §9, §VRAM and §17).

Inputs read: `25d_outline.md`, `25c_blob_only_per_pass_plan.md`, `spiking_lm.py` (RoutedMLP, lines 527-822),
`loop_brain.py` (Core 98-253, `run()` 997-1196), `memory.py` (ConceptHier 281-330, `Memory.private` / `on_grow`
870-955), `train_lm.py` (train step 1345-1409), `neurogenesis.py`, and the old run's config / log
(`C:\Eitan\Python\NewBrain\runs\world_mem_life_12k\config.json`, `train.log`).

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
fraction cap, 16 × 2 (§VRAM); GO criteria and the 7K stop rule (§16).

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
3. **Search, every pass, no gradient**: global once per token (IVF: 1,024 fixed k-means cells fit at step 300, 8
   probed cells, a static [1024, 192, 64] × [1024, 64, 512] bmm, fp32 re-rank → top-4); private for every kept
   (token, expert) pair in one bmm over the capacity layout [E, C, 64] × [E, 64, 512] (C ≈ 96 at 768 experts),
   plus [N, 64] × [64, 512] for the shared expert → top-4.
4. **Gate, per (token, chosen expert), before the expert runs**: ĝ = (a₀ + a_e)ᵀ GELU(W₁φ_t + b₁) + β_e +
   (γ₀ + γ_e)ᵀ m_{t,e}; φ_t (1,997-d, detached) = [xf, side(s), μ_1..4 of the previous pass, global match now,
   match of the previous read, pass-0 flag], m_{t,e} = this expert's private match (6). Read iff ĝ > 0, at most 50%
   of each expert's tokens (ranked by ĝ), plus 5% uniform exploration (the centring sample). ≈ 90K parameters
   (shared trunk 64K, per-expert 39 × 769). Trained after backward from the measured centred gain; never pushes the
   thought.
5. **Read**: r_g(t) = A_g Σ_j w_j f_j v_j over the 4 global winners (per token), r_p(t,e) the same over the
   expert's 4 private winners; r = r_g + r_p is **added to the expert's buffer row** (and to the shared expert's
   input) when the gate is open. w = softmax of relative distance, A = exp(−d₁/2h₀²). The LM loss trains the
   contents (row-Adagrad) and flows through w, A into xf; keys, indices, h₀ get no gradient.
6. **Writes after backward**: W = 64 surprising positions per optimizer step (32 per micro-batch, ≤ 2 per sequence,
   brain signals only, eligible only without a confident read). Each writes ONE global blob and one private blob into
   each routed expert that processed it (the shared expert at the average routed rate), the same key = x̂ at one
   uniformly drawn in-window pass; content = −(the loss gradient at the expert input) at that pass (summed over the
   token's experts for the global blob), ε-guarded. Dedupe-merge at d₁ < γh₀²; eviction of the weakest untouched.
7. **Keeping blobs findable**: a read whose separate global / private benefit is positive and in the top quarter, with
   a clear winner, confirms its winner: the key moves toward the query (η 0.5, capped at 0.5 h₀; ≤ 1 move per blob per
   step, ≤ 8 per 1,000 steps; per-space caps), strength and maturity rise. Late-fact survival S(12.2K) ≈ 0.99 at a
   cross-document hit rate 0.8 and 2 effective passes, 0.89 at 0.5 (§9.4).
8. **Sizes / VRAM**: store 0.99 GB, value-gradient buffers 0.70 GB; projected peak ≈ 18.6 GB allocated,
   ≈ 19.0-19.6 GB reserved at 16 × 2 (old run 20.1 / 20.5-21.1; the rows' 3.5 GB are gone).
9. **Cost**: ≈ 60-90 ms per step for the whole memory (≈ 3.0-4.5% of the 1.95 s step) vs a build without memory;
   vs the old run net ≈ 10-17% FASTER (the rows cost ≈ 150-400 ms). The "memory ≤ 3% of the step" target is at risk
   (§17; owner question OQ-1). No host sync added.
10. **Before any run**: T0a paraphrase, T0c cross-pass and T0r expert-reuse findability, T0h rare-fact writes, T0e
    calibration, T0d IVF recall, T0f xf drift 6K → 7K on CPU from captures; T0b gate separability + T2 + a 1K pilot on
    the GPU when the speed study releases it. GO vs the old run (late 0.280 @7K, rare 0.016, harm +0.13) as in §19.
11. **Size**: ≈ 2,750 lines (memory ≈ 1,400, tests and checks ≈ 1,050, eval ≈ 150, trainer ≈ 150).

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

What is new with xf as the address (outline §2b): xf already contains the token, all earlier passes' thought, the
workspace read (concept feedback, earlier write-backs), the previous pass's concept write-back (rank 64) and the
context (SSD membranes). In 25d there is no separate memory inject any more: the previous pass's memory reads reach xf
through the experts' outputs (o → workspace write and x). Not in xf: this pass's concepts / workspace update (they come
after the MoE), the full μ_l, the per-level uncertainty, match quality, side(s): the gate gets μ_l, side(s) and match
quality explicitly (§5.1).

---------------------------------------------------------------------------------------------------------------------

## 1. Architecture: one core pass k (every pass the same)

```
 run(): pass k  ──  ctx = {k, in_window = k ≥ K−bptt, prev_mu (pass k−1, detached), prev_match (pass k−1), train}
 Core.forward:  r = read(s, x);  u = n_u(x + n_e(e) + W_ro r + inject);  h = u + mix(u);  xf = n_m(h)
 RoutedMLP.forward(xf, side = side(s), bmem = blobmem):
   route(xf) → idx, gate weights → capacity layout (e_k, p_k, t_k), C        [unchanged; ONE existing host sync]
   buf[e_k, p_k] = xf[t_k]
   ┌─ blobmem (no_grad part) ──────────────────────────────────────────────────────────────────────────────┐
   │ x̂ = R (xf − μ)/σ  (sg for the search)                                                                 │
   │ GLOBAL search, per token: centroid scores → 8 cells → static cell layout bmm → top-4 (j, d), fp32      │
   │ PRIVATE search, per kept pair: Q[E, C, 64] bmm P̃[E, 64, 512] → top-4;  shared: [N,64]×[64,512] → top-4  │
   │ match features (global per token, private per pair)                                                    │
   │ GATE: z_t = GELU(W₁ φ_t + b₁);  ĝ_{t,e} = (a₀+a_e)ᵀ z_t + β_e + (γ₀+γ_e)ᵀ m_{t,e}                     │
   │       open = (ĝ > 0 ∧ rank_e(ĝ) < ⌊0.5 n_e⌋) ∪ Explore(5%)                                             │
   └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
   READ (with grad): w, A from x̂ (grad) and the detached winner keys; r_g(t), r_p(open pairs) by GatherRead
   PROBE (identity fwd; bwd stores c^g = −⟨∂L/∂buf, r_g⟩, c^p = −⟨∂L/∂buf, r_p⟩ per open pair)
   buf[e,p] += r_g(t) + r_p(t,e)  for open pairs;   x_sh = xf + open_sh · (r_g + r_p^sh)
   h_e = spike(buf @ w_up) … out = h_e @ w_down;  y = Σ gate·out + shared(x_sh)
 Core: s = write(s, m + o); x update      →  run(): ConceptHier on s → μ_l (next pass's prev_mu), think feedback…

 after each micro-batch's backward: gate update from (φ, m, ĝ, c) kept; value grads accumulated; write candidates'
                                    gradients captured by hooks on buf / x_sh at their write pass
 after opt.step: gate AdamW; value row-Adagrad; confirmations → key moves; merges / spawns / evictions;
                 global cell index rebuilt; EMAs (h₀, v̄, Ĉ)
```

- **Gradient reaches**: blob contents (row-Adagrad on dense grad buffers), xf (hence the thought) through w and A, the
  experts (as usual, now also through their read-augmented input), the gate (post-backward, its own loss only).
- **No gradient reaches**: keys, the map (μ, σ, R), the IVF centroids, h₀, strengths, clocks, indices, the thought
  through the gate (φ and m are detached; the gate's loss is computed on stored detached copies after backward).

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
- **Value-gradient buffers** (dense, so backward is one `index_add_` with no compaction and no host sync): global fp32
  256K × 384 × 4 B = 0.40 GB; private bf16 393,728 × 384 × 2 B = 0.30 GB (a private row receives only a few
  contributions per step — one expert's open reads — so bf16 accumulation loses little; T-unit checks it against
  fp32) + a touched-row bitmask. Zeroed only on touched rows after the Adagrad step.
- **IVF index** (global): 1,024 centroids (64-d fp32, 256 KB) + a cell table [1024, 512] int32 (2 MB), rebuilt every
  step (§4.1).
- Bookkeeping, free lists, the dummy slot for masked writes: < 10 MB.
- **Total store + grads ≈ 1.69 GB.**

### 3.2 Capacities, write rates and lives (all per OPTIMIZER step)
- Write positions W = 64 per step (32 per micro-batch).
- **Global**: 64 writes per step at most (merges reduce spawns). Life of an unconfirmed blob = 262,144 / 64 =
  **4,096 steps** without merges; full by ≈ step 300 + 4.1K ≈ 4.4K. 25c chose life ≈ the drift horizon (≈ 2.3K)
  for density; 25d's 256K is the owner's fixed choice, so life is ≈ 2 horizons: unrenewed keys older than ≈ 2.5K
  steps are rarely found again (R@1 ≈ 0.15 at 3K, 25c §7.1) and are the first to be evicted by strength anyway.
- The recurring-fact **reserve** forms by itself (confirmed slots outrank unconfirmed ones at eviction): 20.4K late
  facts × ≈ 1.25 keys/fact ≈ 26K slots, up to ≈ 100K if a fact keeps one key per pass region (T0c); both fit
  in 256K, leaving (256K − reserve)/64 ≈ 2.5-3.6K steps of unconfirmed life.
- **Private (routed expert e)**: written at rate W · k · (1 − drop) / E_live per step:
  - E_live = 192 (step ≤ 610): 64 × 6 × 0.97 / 192 ≈ 1.9 per step → life ≈ 260 steps;
  - E_live = 384: ≈ 0.97 → life ≈ 530 steps;
  - E_live = 768 (step ≥ 7,320): ≈ 0.49 → life ≈ 1,050 steps.
  Merges lengthen this. The spec says "one private blob into each processing expert"; if T1 shows lives below the
  drift horizon hurt, `--pm-rate 0.25` subsamples private writes per expert to a fixed 0.25 per step (life 2,048 at
  any E_live: p_priv = 0.25 E_live / (W k) = 0.125 / 0.25 / 0.5 at 192 / 384 / 768). OQ-3.
- **Private (shared expert, space 768)**: it processes every token, so "every processing expert" would write 64 blobs
  per step into 512 slots (life 8 steps: useless). Its private space is written at the AVERAGE routed expert's rate:
  each write position also writes there with probability k / E_live (≈ 0.49 per step at 768). [default, owner check:
  OQ-4]

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
≈ 110 GB (≈ 110-160 ms, 5.6-8% of the step). Triton (a fused score + top-k kernel) is not installed (13_memory_speed
_review: blocked without an install). Hence an inverted file built from fixed parts:

1. **Cells** (fit once at step 300, never refitted by default): 1,024 centroids c_1..c_1024 = k-means (10 Lloyd
   iterations, k-means++ init on the forked generator) on the 64K x̂ samples of §2.1. They are an index, not the
   address: keys and queries stay x̂.
2. **Cell table**: each live global slot has cell(i) = argmin_c ‖k_i − c‖² (recomputed when its key moves). Once per
   optimizer step the table [1024, Cc = 512] is rebuilt: argsort of the 262,144 cell ids, offsets by bincount +
   cumsum, position = rank − offset; positions ≥ 512 are impossible because spawning into a full cell evicts that
   cell's weakest (§8.6). Padding = −1 (score −∞). ≈ 1 ms.
3. **Query layout** (per pass): centroid scores x̂ · cᵀ (8,192 × 1,024, 1.1 GFLOP) → top-n_p = 8 cells per token →
   65,536 (token, probe) pairs, sorted by (cell, probe rank) → position within the cell; static capacity Cq = 192
   per cell (3 × the mean 64); pairs beyond Cq are dropped, farthest probes first (logged as "probe overflow").
4. **Scores**: one bmm [1024, 192, 64] × [1024, 64, 512] in bf16 (100.7M scores, 12.9 GFLOP, 201 MB), −‖k‖² added →
   top-4 per (token, probe) row → scattered back to (N, 8 × 4) → top-4 over the 32 → fp32 re-rank of these 32
   candidates (gathered fp32 keys, exact d) → (j₁..j₄, d₁ ≤ … ≤ d₄) per token.
- **Recall argument**: a key within distance √d of the query is missed only if its cell is not among the query's 8
  nearest centroids. For matches that matter (d ≲ 2h₀², i.e. A ≳ 0.37) and cells of radius ≫ h₀ the key's cell is
  the query's nearest or a neighbour; misses concentrate at far matches, which A mutes anyway. Measured in T0d against
  brute force; monitored online (1K sampled queries every 250 steps vs a chunked brute-force scan; target miss of the
  fp32 top-1 ≤ 2% for d₁ < 2h₀², else n_p = 16).
- **Drift of the cell balance**: centroids are fixed while thoughts drift, so cells may unbalance (more probe overflow,
  hot cells churning). Logged: cell occupancy Gini, max cell fill, probe overflow rate. Alert if overflow > 1% of
  probes or Gini > 0.6; then `--gm-ivf-refit N` (re-fit the centroids and re-assign the cell ids every N steps: an
  index operation, keys unchanged, allowed by constraint 1) is the owner's switch.
- `--gm-search brute` (chunked bf16 + block max, 25c §2.4) stays for tests, T0d and small stores.

### 4.2 Private: one bmm over RoutedMLP's capacity layout
- Q = zeros(E, C, 64); Q[e_k, p_k] = x̂[t_k] (the same layout as `buf`); scores = Q @ P̃[:E]ᵀ (bf16 [E, C, 512]) − ‖P‖²
  → top-4 over 512 at the kept slots only → fp32 re-rank of the 4 (exact over 512 up to bf16 ties of the 4th: the
  re-rank is over the bf16 top-8, then cut to 4).
- C = min(max load, ⌈1.5 N k / E_used⌉): 96 at 768 experts, 384 at 192 — E·C ≈ 73.7K rows either way, so the cost is
  constant through growth: 37.7M scores (4.8 GFLOP, 75 MB) per pass. The shared expert: [8,192, 64] × [64, 512]
  (4.2M scores).
- Eval uses dropless capacity (C = the max load, ≈ 2-3 × the mean): the bmm grows by that factor; still < 2 ms.

### 4.3 Match features (from the search, detached)
- Global, per token (6): A_g = exp(−d₁/2h₀,g²), d₁/h₀,g², log(d₂/d₁), m_{j₁}/3, log(1 + age_{j₁}), empty flag.
- Private, per pair (6): the same for the expert's own space (h₀,p).
- Previous read, per token (6): pass k−1's A_g, log(d₂/d₁)_g, d₁/h₀,g², the mean and max A_p over the token's pass-(k−1)
  pairs, the fraction of its pass-(k−1) pairs that read; zeros + a "pass 0" flag at k = 0.

---------------------------------------------------------------------------------------------------------------------

## 5. The gate (per token, per chosen expert; decides before the expert runs)

### 5.1 Inputs (all detached)
φ_t (token-level, 1,997): xf (384) ‖ side(s) (64, the router's workspace summary, `Core.side`) ‖ μ_1..μ_4 of the
previous pass (4 × 384 = 1,536, as produced by ConceptHier after pass k−1; zeros at pass 0) ‖ global match now (6)
‖ previous read's match (6) ‖ pass-0 flag (1). Each group is RMS-normalised per token (xf already is). m_{t,e}
(pair-level, 6): the expert's private match now. Options (off): `--gm-gate-feats +cu` (the 2L per-level uncertainty,
outline §2b lists it as "not in xf" but not as a gate input) and `+pass` (a pass embedding: against constraint 7's
spirit, ablation only).

### 5.2 Parameterisation for 769 experts (shared trunk + per-expert head)

      z_t      = GELU(W₁ φ_t + b₁),                          W₁ ∈ R^{32 × 1997}         (shared trunk, 64K)
      ĝ_{t,e}  = (a₀ + a_e)ᵀ z_t + β_e + (γ₀ + γ_e)ᵀ m_{t,e},  a_e ∈ R^32, β_e, γ_e ∈ R^6  (per expert, 39 × 769 = 30K)

- Shared parts carry what is common ("is there a near blob", "is this an answer-like position"); a_e, γ_e, β_e let
  an expert weigh them differently (an expert for numbers may benefit from memory where a syntax expert does not).
- **Init**: W₁ ~ N(0, 1/1997), a₀ = a_e = γ = β = 0 → ĝ ≡ 0 → nothing reads except the exploration sample until the
  gate has learned (safe start, no forced warm-up). a₀ gets gradient first, then the trunk.
- **Shrinkage**: AdamW weight decay 0.01 on a_e, γ_e (toward the shared head); none on β_e. An expert with few tokens
  stays close to the shared gate.
- **Cost per pass**: trunk 2 × 1,997 × 32 × 8,192 = 1.05 GFLOP; the per-pair head is a 32-d dot + 6-d dot on ≈ 49K
  pairs (+ 8,192 for the shared expert). Building φ (33 MB bf16) dominates. ≈ 0.6-1.0 ms per pass.
- Alternatives (flags): `--gm-gate lowrank` (ĝ = z_tᵀ U V_e with rank 8 per expert), `--gm-gate shared` (no
  per-expert head; ablation measuring what per-expert heads add).

### 5.3 Target: the centred measured gain (25c §4.3, per pair)
- Probe on the injected read (§6.3): c_{t,e} = c^g_{t,e} + c^p_{t,e}, c^· = −⟨∂L/∂buf[e,p], r^·⟩ (first-order loss
  change from removing that part of the read).
- z_{t,e} = N_tok · (c_{t,e} − q₅₀^{E}(c)), q₅₀ over this micro-batch's exploration pairs of the same pass
  (≈ 2,460 routed + 410 shared pairs: sd of the median ≈ 0.03 MAD). Pooled over experts: the gate learns which
  experts gain more (β_e), and the population zero cannot be moved by the gate itself.
- L_r = λ_r · mean_{in-window open pairs} Huber_δ(ĝ − z), δ = N_tok · 1.4826 · MAD^E(c), λ_r = 0.1.
- Consistency (25c §4.3): selection depends on φ, m and the exploration draw only, and E gives positivity everywhere,
  so the regression on open pairs estimates E[z | φ, m].
- Out-of-window reads use ĝ but do not train it (no backward there).

### 5.4 Selection (training)
For pass k, expert e with n_e = min(count_e, C) kept tokens:

      open_{t,e} = ( ĝ_{t,e} > c_read ∧ rank_e(ĝ_{t,e}) < ⌊0.5 n_e⌋ ) ∨ explore_{t,e},     c_read = 0

- rank_e: a descending sort of ĝ along each [E, C] row (invalid slots −∞): one sort of 73.7K values, n_e is a device
  tensor (no host sync). The shared expert: rank over N with ⌊0.5 N⌋.
- **Exploration**: exactly ⌈0.05 n_keep⌉ routed pairs and ⌈0.05 N⌉ shared pairs, uniform without replacement
  (top-k of uniform randoms from the device generator; n_keep is a host int already known in RoutedMLP), read
  regardless of ĝ and the cap. They are the centring sample.
- Hard bound: open ≤ 0.55 n_keep (+ 0.55 N shared) → the compact read buffer has the static size
  n_max = ⌊0.5 n_keep⌋ + E_used + ⌈0.05 n_keep⌉ (the + E_used covers the per-expert floors).
- **Why it does not collapse** (25c §4.2.6): all-open is impossible beyond the cap and, centred at the exploration
  median, only ≈ the upper half has ĝ > 0; all-closed is impossible because the exploration always reads.
- **Eval / inference**: read iff ĝ > 0 and within the per-expert 50% cap; no exploration; no target, no writes.
- Options: `--gate-cost <nats>` / `--gate-cost time` (c_read = κ Δt_read), `--gm-gate-budget fixed` (debug: a fixed
  open fraction by ĝ rank).

### 5.5 Post-backward update and the idempotent Probe (25c §4.5-4.6, adapted)
1. Forward (each in-window pass): keep φ_t (bf16, all tokens: 33 MB), m and the open / explore masks of the open pairs
   in the compact layout, until after the micro-batch's backward (≤ 6 in-window passes → ≤ 0.2 GB).
2. Backward: the Probe writes c^g, c^p into `cbuf[mb, slot(k), :n_max]` with `index_copy_` (idempotent) and bumps
   `cnt[mb, slot(k)]`.
3. After `backward()` of the micro-batch: z (§5.3), then a SEPARATE tiny autograd graph: ĝ recomputed from the kept
   φ, m with the gate parameters, L_r (z fixed) → `backward()` into the gate parameters only (÷ accum). The gate's
   AdamW steps once per optimizer step, after both micro-batches, so the forward ĝ used the same parameters: the
   gradient is exact. (25c's closed-form gradient was for a linear router; with a trunk the recompute is simpler and
   costs ≈ 1-2 ms per micro-batch.)
4. The gate is in its own AdamW group (lr 1e-3) with its own clip (1.0), excluded from the brain's global clip
   (`train_lm.py` lines 1388-1391 already do this for `gate`).
- **Checkpointing**: the core pass is not checkpointed today, but the per-pass checkpoint is a VRAM lever in reserve.
  Under it the RoutedMLP forward would run twice: the search results, the exploration draw and the open masks are
  cached per (micro-batch, pass) on the first run and reused on the recompute (the generator is not advanced again);
  the Probe does nothing in forward and writes only in backward into a fixed slot, so a recompute cannot double c.
- **Test `test_gm_gate_postbwd`** (CPU, §18.4): (a) cnt == 1 for every in-window pass, 0 outside; (b) c equals
  −⟨∂L/∂r, r⟩ computed with checkpointing off; (c) the recomputed gate gradient equals autograd of L_r on the main
  graph with z fixed (rtol 1e-5); (d) accum 2 and 4 give the full-batch gradient; (e) value and query gradients equal
  the non-checkpointed ones when the core pass is wrapped in `checkpoint` for the test.

### 5.6 Watchdog (alerts only; no automatic fallback)
Every eval: AUC of ĝ vs 1[z > 0] on a held-out micro-batch's exploration pairs (pooled, and for the 32 busiest experts),
calibration slope, open rate (train / eval) by pass and by token type (eval tags), cap hits, the fraction of experts
whose open rate is pinned at 0 or at the cap. Alert after 0.3·S if the pooled AUC < 0.60 at two consecutive evals or
the pooled open rate is pinned at 5% or 55% for 500 steps.

---------------------------------------------------------------------------------------------------------------------

## 6. The read and the inject

### 6.1 Kernel (per space; 25c §3.2)

      w_j = softmax_j( −(d_j/d₁ − 1)/τ_w ),  τ_w = 0.1
      A   = exp(−d₁ / (2 h₀²))                          (h₀,g for global, h₀,p for private)
      f_j = exp(−λ_v ρ_j (t − Λ^v_j))                   (lazy value fade)
      r   = A Σ_{j=1..4} w_j f_j v_j

- r_g(t) is computed once per token (for all N tokens: a 50 MB gather, cheaper than compacting); r_p(t,e) for the
  open pairs only (compact layout, ≤ n_max rows).
- Strength is not in the kernel (eviction only), as in 25c.
- **Blend of global and private**: r_{t,e} = r_g(t) + r_p(t,e). Each A mutes its own space when it has no near blob;
  one gate decision covers both (the gate sees both match qualities). Option `--gm-gate2 1`: two outputs per pair
  (global, private), each with its own target c^g / c^p (OQ-5).

### 6.2 Inject (the expert's input)
- Routed: `buf[e_k_o, p_k_o] += r_{t,e}` for the open pairs (an out-of-place `index_put` with accumulate on the
  compact list), BEFORE `torch.bmm(buf, w_up)`. The expert then runs on xf + r: "together with everything else it
  receives".
- Shared expert: `self.shared(xf + scatter(open_sh · (r_g + r_p^sh)))`.
- The router never sees r (routing is computed from xf before the read): the read cannot change which experts run,
  so capacity drops and balancer counts are identical with the memory on or off (a test).
- Scale: contents are initialised at v̄ = 0.1 × median‖xf‖ (≈ 10% of the input norm, measured at the fit step), so a
  first read is a moderate perturbation; Adagrad then sets the size.

### 6.3 Gradients
- **GatherRead** (custom autograd Function, like `MemRead`): forward Σ_j a_j V[rows_j] with a_j = A w_j f_j; it saves
  only rows and a; backward gathers V[rows] again for ∂/∂a_j = ⟨g, V[rows_j]⟩ and `index_add_`s a_j g into the dense
  grad buffer (global fp32 / private bf16). Nothing (|O|, 4, 384)-sized is kept for backward.
- w and A are recomputed WITH grad from x̂ = R((xf − μ)/σ) and the gathered detached winner keys: the LM loss reaches
  xf through the match weights (decided). `--gm-match-grad 0`: w, A from sg(x̂) (ablation).
- **Probe** (custom Function on (r_g-per-pair, r_p)): forward returns r_g + r_p; backward passes the gradient through
  unchanged and stores c^g = −⟨g, r_g⟩, c^p = −⟨g, r_p⟩ per open pair (§5.5).
- Contents: row-Adagrad after the optimizer step (lr 0.01, per-row clip 0.05, as the old `--mem-lr`, `--mem-clip`),
  discounted accumulator (25c §8). Private contents: bf16 storage updated with stochastic rounding (the fp32 update is
  computed, then rounded up or down with probability proportional to the remainder: unbiased; a unit test compares
  1,000 small steps to fp32).

### 6.4 How much a found blob is trained
Every in-window open read trains its 4 winners. A late fact recurs ≈ 116 times by the end; each exposure has ≈ 4.9
in-window passes reading the SAME space: ≈ 116 × 4.9 × p_open × p_hit ≈ 116 × 4.9 × 0.8 × 0.8 ≈ **360 value
updates** (25c: ≈ 35 per bank). Rare facts get ≈ 5-30. This is the main structural gain of "one space for all passes".

---------------------------------------------------------------------------------------------------------------------

## 7. Benefit signals
- c is ≈ zero-sum over a batch under weight decay-like equilibria (25c §5): only centred values are used (z for the
  gate; z^g, z^p for confirmation). No sign-based disconfirmation, no spawn-on-hurt.
- **Separate credit**: c^g_{t,e} and c^p_{t,e} come from the same gradient, so the global blob's benefit for a token is
  C^g_t = Σ_{e open} c^g_{t,e} (it is added to every open expert of the token), centred on the exploration tokens;
  the private blob's benefit is c^p_{t,e}, centred on the exploration pairs.

---------------------------------------------------------------------------------------------------------------------

## 8. Writes (after backward; inside the training step; brain signals only)

### 8.1 Which positions (25c §6.2)
- Surprise σ_t from tensors the step already has (choice by T0h): σ^raw = ce_{K−1}[t]; σ^ent = ce_{K−1} − Ĉ(H_{K−1})
  (CE above what the model's own entropy predicts; Ĉ an EMA 32-bin regression); σ^imp = ce_{K−1} · 1[ce_0 − ce_{K−1}
  < δ_imp]. No labels, source tags, document-type features or dataset pointers.
- Eligible: ≥ 8 tokens after a document start, not EOT, token kept by ≥ 1 routed expert at the write pass.
- **Budget**: W = 32 per micro-batch (64 per optimizer step), ≤ 2 per sequence, ≥ 8 tokens apart: a static top-k on
  masked σ.
- The selection happens after the forward (σ needs the last pass's CE) and **before** `backward()` (train_lm calls
  `model.bm.select_writes(...)` between the loss and `backward`), so the gradient hooks know what to capture.

### 8.2 Which pass and which key
- Each selected position t draws its write pass k_w(t) uniformly from this forward's in-window passes (device
  generator). Rationale: memory is pass-agnostic, so keys should cover the passes the queries come from; all reads of
  later exposures (≈ 4.9 in-window passes) then find a key written from some pass. Option `--gm-write-pass maxgrad`
  (the in-window pass with the largest ‖∂L/∂xf_t‖: still the brain's own signal) and `last`.
- Key = x̂_t at pass k_w (stashed in forward for all in-window passes at the ≤ 32 positions: 32 × 6 × 64 floats).

### 8.3 Content: what the gradient says would have helped (ε guard; 25c §6.3)
- Hooks on `buf` and on the shared expert's input of each in-window pass (registered in forward, reading the
  selection tensor that is filled before backward) gather, at the selected (t, k_w) only, g_{t,e} = ∂L/∂buf[e, p] for
  the token's kept routed experts and g_{t,sh} for the shared expert (≤ 32 × 7 × 384 floats per micro-batch).
- Private content for expert e: q = −g_{t,e}; global content: q = −(Σ_{e kept} g_{t,e} + g_{t,sh}) — the derivative
  of the loss w.r.t. one vector added to all of the token's expert inputs, which is what a global blob is.
- v = σ_ξ ξ + ν v̄ · q / max(‖q‖, ε_q) · 1[‖q‖ ≥ ε_q], σ_ξ = 0.02 v̄, ν = 1, ε_q = 1e-6 × the running median ‖q‖;
  ξ-only when the gradient is absent or below ε_q (a dropped expert has no slot and gets no private write at all).
  v̄ = EMA of the median ‖v‖ of the space type's Adagrad-touched slots (init 0.1 × median‖xf‖). There is no W_o, so
  the 25c "W_o = 0 → q = 0" start problem is gone; the ε guard remains for vanishing gradients.

### 8.4 Where they go
- One **global** blob per position, unless the position had a confident global read at the write pass
  (A_g ≥ A_w = 0.3 on an open pair): then only the private writes happen (a found fact is renewed by confirmation, not
  copied).
- One **private** blob into each routed expert that processed the token at the write pass (kept slots only), unless
  that pair read its own space confidently (A_p ≥ A_w); the shared expert's space with probability k / E_live (§3.2).
- With `--pm-rate r`: each routed expert's private write is kept with probability p_priv = min(1, r E_live / (W k)).

### 8.5 Dedupe at write (wrong-merge bound; 25c §6.4)
- Against the store, after the optimizer step: the global candidates through the same IVF search (64 queries), the
  private ones by a [n, 64] × [64, 512] score per space.

      d₁ < γ h₀²  ⟹  MERGE into j₁:
          η = max(1/(n_{j₁}+2), 1/4);   k_{j₁} ← k_{j₁} + η (x̂ − k_{j₁});   τ_{j₁} ← τ_{j₁} + η (t − τ_{j₁})
          v_{j₁} ← v_{j₁} + (v_new − σ_ξ ξ)/(n_{j₁}+2);   n ← min(n+1, 8);   log α += 0.5;   Λ^α, t^touch ← t
      otherwise  ⟹  SPAWN

- γ (one per space type) from T0e: the largest γ ≤ 0.5 whose wrong-merge rate P(j₁ is another fact | d₁ < γh₀²),
  measured on the T0a capture at the real distractor density (256K global; 512 private), is ≤ 2%. Placeholder 0.25. If
  no γ ≥ 0.05 meets 2%, dedupe is off for that space type.
- Within a step: candidates of the same space within γh₀² of an earlier (higher σ) one are suppressed and their content
  increment goes to the kept one (`index_add_`). A merged global key may change cell: its cell id is recomputed.

### 8.6 Spawn and eviction
- Spawn: k = x̂, v as §8.3, log α = 0, all clocks = t, t^conf = −∞, m = n = 0, S = S₀.
- Victims (global): the free list, then — if the target cell already holds 512 slots — that cell's weakest lazy strength
  ã, else the space's weakest; always **excluding t^touch = t** (read with A ≥ 0.05, written, merged or moved this
  step). One masked top-k (global: over 256K, ≈ 1 ms; private: per space over 512, batched).
- Confirmed slots (m ≥ 1, log α up to log 8) outrank unconfirmed ones, so the reserve forms by itself.
- **Zombie guard** (25c §6.6): at eviction time a slot with t − t^conf > 3,000 steps has its strength computed with
  ρ = 1 (no maturity protection).

---------------------------------------------------------------------------------------------------------------------

## 9. Keeping blobs findable: confirmation, key moves, anti-capture, survival

### 9.1 Why renewal is still first-order
Keys are write-time thoughts; thoughts drift (median drift 1.22-1.31 × the 1-NN gap per 1K steps). An unrenewed key is
found with R@1 ≈ 0.77 after 1K steps, ≈ 0.45 after 2K, ≈ 0.15 after 3K (4K density; lower at 256K). Late facts recur
every Δ ≈ 105 steps; their blobs must be renewed faster than they drift away. Value training (§6.4) never moves a key
(25c §7.2).

### 9.2 Confirmation (default; in-window benefit)
- **Global winner j₁(t)** is confirmed by token t at pass k iff: the token has ≥ 1 open pair, z^g_t > 0 and z^g_t is in
  the top 25% of this micro-batch's in-window tokens with an open pair, AND the geometry passes: w₁ ≥ ω₁, d₂/d₁ ≥ μ_m
  (**clear winner**), A_g ≥ A_min (thresholds from T0e at precision ≥ 0.95 that j₁ is the same fact).
- **Private winner j₁(t,e)** likewise with z^p_{t,e} (top 25% among the open pairs, pooled over experts) and the private
  thresholds.
- Out-of-window geometric renewal (25c §7.3) is NOT needed by default: every exposure has ≥ 2 in-window passes
  (K ≥ 2), ≈ 4.9 on average, reading the same space. `--gm-renew-geo 1` keeps it as an option for passes 0..K−7 of
  K ≥ 7 steps (≈ 1.5 passes per forward), with the stricter thresholds.

### 9.3 The key move and the anti-capture guard
After the optimizer step, per confirmed slot (sort + segment mean of its confirming queries q̄ over both micro-batches
and all passes):

      k ← k + clip_{0.5 h₀}( η (q̄ − k) ),   η = 0.5 (1 if u < 2);   τ, Λ^α, t^conf, t^touch ← t;
      log α ← min(ã + 1, log 8);   m ← min(m + 1, 3) if ≥ 64 steps since its last increment;  refresh k̃, ‖k‖², cell

Guards against a blob being pulled away by unrelated thoughts:
1. **Clear-winner margin** (d₂/d₁ ≥ μ_m) and A ≥ A_min on every confirming read: a blob between two clusters, or far
   from the query, cannot be confirmed.
2. **Capped move**: ≤ 0.5 h₀ per step (25c capped at h₀; the global space is read from many contexts, so half).
3. **Per-blob counts**: ≤ 1 move per step (the segment mean, not one move per read) and ≤ 8 moves per sliding 1,000
   steps (a 4-bit counter + the window start); beyond that a confirmation renews strength / maturity but does not move
   the key.
4. **Per-space caps**: global ≤ 256 moving slots per step, private ≤ 2 per expert per step (≤ 512 in total), lowest
   z first dropped. Check for the global cap: late-fact cues arrive at ≈ 194 per step (25c §7.3); with p_open 0.8,
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
  (1 − ρ_e)^6). Private lives (260-1,050 steps, §3.2) are shorter than H at E_live ≤ 768, so private copies of late
  facts survive only through frequent confirmation; they matter more for rare facts' first re-exposure at low density.

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

## 10. Fading, maturity and clocks (lazy; 25c §8)
- Strength: ã_i = log α_i − λ_α ρ_i (t − Λ^α_i), λ_α = 2.5e-4, ρ_i = 4^{−m_i} (ρ = 1 under the zombie guard); renewed
  by spawn, merge, confirmation. **Eviction only.**
- Content fade: f_i = exp(−λ_v ρ_i (t − Λ^v_i)), λ_v = 1e-4, folded at every Adagrad touch.
- Accumulator discount: 1 − β_i = (1 − β₀) ρ_i, β₀ = 0.99, floor 0.1 S₀.
- h₀ per space type (25c §3.3): calibration h₀² = q₅₀(d_same)/(2 ln 2) from T0e; tracking h₀² ← 0.99 h₀² + 0.01 κ
  q₅₀(d₁ over ALL open reads of that space type in the step) (unselected population, so the EMA does not contract);
  floor h₀² ≥ ½ q₅₀(NN distance² among 4,096 sampled keys) every 250 steps (private: NN within the same space,
  pooled over 64 sampled spaces). Before T0e: h₀² = 1.24 × q₅₀(NN²) (the salvage drift scale). One h₀,p for all
  private spaces (per-space values would be too noisy at 512 keys).

---------------------------------------------------------------------------------------------------------------------

## 11. Experts: capacity drops, births, deaths

### 11.1 Capacity 1.5 and dropped tokens
- Routing is computed from xf before the memory, so the kept / dropped set is identical with memory on or off.
- A dropped (token, expert) pair (≈ 3-4% in training) has no buffer row: no expert compute (as today), **no private
  search result used, no private read, no Probe value, no private write, no confirmation** for that expert. Its
  token's global read still enters every KEPT pair that opens, and the shared expert (which never drops).
- A token with all 6 routed experts dropped (rare) can read only through the shared expert's gate.
- The global content's write gradient sums over the kept experts only (§8.3).
- Eval runs dropless (`--capacity dropless`), so eval reads see every pair.

### 11.2 Split births (`--birth-mode split`, the run's mode)
At the top of a step, `Core.grow_experts` → `RoutedMLP.grow_experts(n)` returns src (parents); the memory hook
(`Core.mem_hook`, already called there) copies, for dst = E0..E0+n−1:
- the parent's whole private space: keys, bf16 copies, ‖k‖², contents, Adagrad S, strengths, clocks, counts, maturity
  (t^touch unchanged); travel-tracking entries are not copied (the child starts untracked);
- the parent's gate head (a_e, β_e, γ_e) and its AdamW moments (preallocated 769 rows: no parameter resize, only row
  copies, so the gate's optimizer is not rebuilt);
- nothing in the global space changes.
Both twins then hold identical spaces; each receives about half of the parent's tokens (router rows r ± δ), so blobs on
the other side of δ stop being read and fade to eviction. No cross-space dedupe. Births stop at ≈ 7,320 (60%).
- `ramp` birth mode (not used by this run): the child starts with an empty private space and the shared gate head
  (it is not selected at birth).

### 11.3 Expert death
There is none in the current code. If one is added (e.g. resetting a dead expert), the hook `on_expert_reset(e)` frees
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
  static on the host. `counts` (device) gives n_e = min(count_e, C) for the per-expert cap.
- The old `pmem` path (output side) stays, off.

### 12.2 `loop_brain.py`
- `Core.forward(…, bmem=None)`: passes `bmem` to `self.mlp(...)` (line 235). Nothing else changes in Core.
- `LoopBrain.__init__` / `_build_memory`: `self.bm = BlobMemory(...)` when `--gm 1`, built LAST on a forked RNG (as
  plan 22 did), so `--gm 0` consumes no random numbers. `Core.mem_hook` chains the old `Memory.on_grow` (if any) and
  `bm.on_grow(src, E0, split)`.
- `run()` (lines 997-1196):
  - before the pass loop: `bm.begin_forward(train=self.training and tgt is not None, K=K, bptt=bptt)` (clears the
    per-forward context, the kept-tensor lists, sets the in-window passes);
  - per pass, before `core(...)` (line 1075): `bm.ctx.update(k=k, in_window=k >= K - bptt, prev_mu=prev_mu)` and
    `bmem=self.bm if bm active else None` into `core(...)`; prev_match is kept inside `bm` from the previous call;
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
  (z, the gate's separate graph and its backward, value-grad bookkeeping, the write stash with its captured gradients,
  confirmation candidates).
- The gate's parameters in their own AdamW group and clip (as `gate_opt` today), excluded from `brain_params`.
- After `opt.step()` and `apply_balance()`: `bm.after_step(step)` (§13).
- `torch.cuda.empty_cache()` before every `evaluate()` and before the loop report; `set_per_process_memory_fraction`
  (0.78); eval batch 16.
- Resume: the store, cell table, centroids, map, h₀ / v̄ / Ĉ EMAs, gate parameters and moments, the generator.

### 12.4 `eval_world.py`
Token-type split (world_answer_first / rest, world_other, wiki; salvage `extra_split.py`), knockouts ko_gm_all,
ko_gm_global, ko_gm_private, gate modes, open rates by token type, p_open / p_hit (global and private) at answer cues,
the late-fact fill.

---------------------------------------------------------------------------------------------------------------------

## 13. Order of operations and host syncs

- **Forward (each pass)**: map → global search → private search → match features → gate (φ, ĝ, cap, exploration) →
  GatherRead + Probe + inject → experts. Out-of-window passes run the same under `no_grad` (they read; they do not
  train the gate, confirm, or write).
- **Between forward and backward**: `select_writes`.
- **After each micro-batch's backward**: gate loss on the kept tensors (separate small graph, backward into gate
  params); confirmation candidates (slot, query, z) appended to static buffers (masks + a dummy slot); the write stash
  (positions, k_w, keys, captured gradients) appended.
- **After opt.step (once per step)**:
  1. gate AdamW (own group);
  2. content row-Adagrad on touched rows (global fp32; private with stochastic rounding), lazy fade fold, discounted
     S, per-row clip; t^touch ← t for touched rows; zero the touched grad rows;
  3. confirmations → key moves under the guards (§9.3); cell ids of moved global keys;
  4. writes: dedupe (§8.5), merges, then spawns into free / weakest-untouched slots (§8.6), zombie guard;
  5. refresh k̃, ‖k‖² of changed slots; rebuild the global cell table;
  6. EMAs (h₀,g, h₀,p, v̄, Ĉ), floors every 250 steps, the online IVF-miss check every 250 steps.
- **Host syncs added: none.** Every size is a host integer known before the GPU work (W, the exploration counts from
  n_keep, n_max, Cq, Cc, the confirmation / move caps); variable-size sets are masks over static buffers; per-slot
  averaging uses sort + neighbour flags + cumsum + `index_add_` (no `torch.unique`); RoutedMLP's existing
  `counts.tolist()` provides n_keep and C. Removing the rows also removes their `write_counts()` sync
  (`train_lm.py` line 1397). The unit test counts syncs (§18.4).

---------------------------------------------------------------------------------------------------------------------

## 14. Causality (no future leakage)
1. **Within a sequence**: the read at position t uses x̂_t, which depends on tokens ≤ t only (causal SSD mixer,
   per-position workspace and concept heads), and the store as it was before this optimizer step. The gate's inputs
   (xf_t, side(s)_t, μ_l of pass k−1 at t, match features at t and at pass k−1) are all position-t quantities.
2. **Within a step**: every store change (contents, keys, spawns, strengths, h₀, the cell table) happens after
   `opt.step()`, so micro-batch 2 reads the same store as micro-batch 1, and nothing written from a window can be read
   by that same window in that step. (A content gradient at t contains information about t+1 and later tokens of the
   window — that is the point of a write — but it can only be read in later steps, on other windows: memory of the
   training data, like the weights.)
3. **Write selection** uses the CE at t (target t+1) only to choose WHICH positions write; it never enters a read in
   the same step.
4. **Batch statistics** (the exploration median, MAD, the h₀ EMA) only shape the gate's loss and later steps, never the
   current forward's outputs.
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
  travel; §15's table; p_hit at late / early cues (eval tags only).

---------------------------------------------------------------------------------------------------------------------

## §VRAM

### Measured (VRAM study, old run at end-of-growth size, 16 × 2, seq 512, bptt 6)
Training peak 20.12-20.25 GB allocated (K ≥ 6), 20.5-21.1 reserved; `evaluate()` 23.05 reserved on the training cache;
row tables + grads 3.54 GB; row read activations ≈ 0.4 GB; fragmentation 0.4-1.0 GB.

### 25d projection (16 × 2)

| item | Δ GB | derivation |
|---|---|---|
| old-run training peak | 20.14 | measured |
| row tables and grads | −3.54 | measured |
| row read activations | −0.35 | 25c |
| global store | +0.52 | 262,144 × 1,970 B |
| private store (bf16 contents) | +0.47 | 393,728 × 1,200 B |
| value-gradient buffers | +0.70 | global fp32 0.40 + private bf16 0.30 |
| read activations kept for backward (rows, a, w, A, Probe's saved r in bf16) | +0.20 | ≈ 33 MB per in-window pass × 6 |
| gate kept tensors (φ bf16, m, masks), one micro-batch | +0.20 | 8,192 × 1,997 × 2 B × 6 passes |
| search transients (IVF scores 201 MB, private scores 75 MB, re-rank), one pass at a time | +0.30 | in forward, coexisting with stored activations |
| gate parameters + moments, IVF index, bookkeeping | +0.01 | |
| **projected peak allocated** | **≈ 18.65** | reserved ≈ **19.0-19.6** (+0.4-1.0 fragmentation) |
| `evaluate()` after `empty_cache()` | ≈ 7.9 reserved | 25c's 6.2 + store 0.99 + grad buffers 0.70 (persistent) |

- Margin to 20 GB reserved: 0.4-1.0 GB. Levers in order if T2 measures reserved > 19.5 GB: (1) chunked fused CE
  (−1.2 GB, ≈ −2% speed; VRAM study); (2) the gate trunk on a fixed random feature map (store 256-d features instead
  of φ: −0.17 GB; changes the gate, owner); (3) private grad buffer compacted per step (−0.25 GB, + sort cost);
  (4) 8 × 4 after the same-maths audit (−4.5% speed; 25c §VRAM).
- Run defaults: `empty_cache()` before every `evaluate()` and the loop report; `set_per_process_memory_fraction(0.78)`
  (≈ 20 GB, a guard); eval batch 16; micro-batch 16 × 2.
- Spike avoidance in the memory: no (N, 262,144) tensor exists (IVF layout; the brute-force flag uses 1,024-query
  chunks); the private bmm is [E, C, 512]; the dedupe search uses the same paths.

---------------------------------------------------------------------------------------------------------------------

## 17. Cost (estimates; T2 measures)

Per optimizer step: 12.8 pass-forwards (N = 8,192 each), ≈ 9.9 with backward; step ≈ 1.95 s (old run, 8.4k tok/s);
3% = 58 ms.

| component | per pass | per step |
|---|---|---|
| map R(xf − μ)/σ, centroid scores (1.1 GFLOP), top-8 cells, probe layout (sort 65K) | 0.3-0.5 ms | 4-6 ms |
| global IVF bmm (12.9 GFLOP, 201 MB), per-probe top-4, merge, fp32 re-rank | 1.2-2.0 ms | 15-26 ms |
| private bmm (5.4 GFLOP, 75 MB) + top-4 + re-rank, shared expert | 0.7-1.2 ms | 9-15 ms |
| gate: φ build (33 MB), trunk (1.05 GFLOP), per-pair head, per-expert sort, exploration, compaction | 0.6-1.0 ms | 8-13 ms |
| read (gather 50 MB global + ≤ 41 MB private) + inject; backward (re-gather, index_add, Probe dots) | 0.4 + 0.6 ms | 5 + 6 ms |
| gate post-backward (recompute trunk on kept φ + backward), per micro-batch | — | 3-5 ms |
| after_step: Adagrad on touched rows (≈ 1-2 GB traffic), confirmations / moves, dedupe / spawn / evict, cell-table rebuild, EMAs | — | 8-12 ms |
| **memory total** | | **≈ 58-88 ms ≈ 3.0-4.5% of 1.95 s** |
| rows removed (product-key select, shared / private / coarse read + backward, MemOpt, G zeroing, `write_counts` sync) | | −150 … −400 ms |
| **net vs the old run** | | **≈ 10-17% faster** |

- **Brute-force global search** instead of IVF: +85-135 ms (27.5 × 10⁹ bf16 scores, ≈ 110 GB of score traffic): over
  budget without a fused kernel.
- Most memory kernels are small and launch-bound (≈ 40-60 launches per pass); all shapes are static, so CUDA graphs
  (speed track) apply. Levers for the 3% target: n_p 4 instead of 8 (−6-10 ms, T0d decides the recall cost), a fused
  private search + top-4 (speed track), gate φ in fp8 / a smaller μ input (owner).
- Gate cost alone: ≈ 8-13 ms forward + 3-5 ms update ≈ 0.6-0.9% of the step.
- **Criterion**: tok/s ≥ 0.97 × the old run's config in the same harness (median of ≥ 3 runs; ±3% run-to-run): met by
  a wide margin if the row costs are as estimated. The stricter reading "the memory adds ≤ 3% to a no-memory build" is
  at risk (3.0-4.5%): OQ-1.

---------------------------------------------------------------------------------------------------------------------

## 18. Tests before any run

All CPU work: `CUDA_VISIBLE_DEVICES=-1`, `torch.set_num_threads(2)` (≤ 2 threads), no installs, nothing deleted. The GPU
belongs to the speed study until it releases it.

### 18.1 Real-thought checks with the NEW address (xf), from captures

Captures (CPU if a 4-window timing shows ≤ 1 min per window, else GPU when free): checkpoints old run 6K and 7K
(`runs\world_mem_life_12k\ckpt_6000.pt`, `last.pt` at 7K) and the no-memory control 6K / 9K
(`runs\world_v4d_12k\ckpt_6000.pt`, `ckpt_9000.pt`; the closer proxy for a brain without rows). Train-mode routing
(head-group offset, route noise 0.3, capacity 1.5) and eval routing. At every pass 0..9 and at the cue (the token before
the answer): xf, the routed experts and their capacity slots, side(s), μ_l of the previous pass. The fixed map is fitted
on 64K positions of the same checkpoint (no step-300 checkpoint of a 25d run exists; T0f measures how much a
map fitted at 6K costs at 7K).

| test | what | GO / NO-GO |
|---|---|---|
| **T0a paraphrase** (cross-document same fact) | 300 late + 300 early people × 5 relations = 3,000 facts, each in ≥ 3 documents with 3 training templates (0-9) and other surrounding facts, + 1 held-out-template cloze prompt (10-11). Global gallery = doc-1 cue keys (at a uniformly drawn pass, as §8.2) + distractors written by the T0h criterion from ordinary windows at 64K / 128K / 256K; queries = doc-2/3 cues and the cloze cue at every pass. R@1 / R@16 with brute force AND the IVF; same-template-other-fact top-1 rate (template keying); keys per fact after dedupe. Within a checkpoint and 6K → 7K | **GO**: doc → doc R@1 ≥ 0.6 and doc → cloze ≥ 0.5 at 256K (best query pass) and 6K → 7K doc → cloze ≥ 0.35. **Weak GO**: doc → cloze 0.35-0.5 or doc → doc 0.45-0.6: the owner gets the §9.4 survival at the measured p_hit. **NO-GO**: doc → cloze < 0.35 → back to the owner |
| **T0c cross-pass findability** | On the T0a capture: a blob keyed at pass i, queried from pass j (i, j ∈ 0..5), same document (age 0) and other document. R@1 matrix; the distance ‖x̂_i − x̂_j‖² of the same token vs h₀² and the 1-NN gap; the estimated n_eff (the number of passes whose query finds a key written at a random pass) | **GO**: off-diagonal R@1 ≥ 0.7 × the diagonal for |i − j| ≤ 2 and n_eff ≥ 2. **Weak GO**: n_eff 1.3-2: tell the owner (§9.4 row n_eff = 1-2). **NO-GO**: n_eff < 1.3: one key per pass region is needed (write at 2-3 passes per position, owner) |
| **T0r expert-reuse findability** | For the same fact: overlap ρ_e between the write-time cue's 6 experts (at the write pass) and the later cue's experts (each pass), other document; P(≥ 1 shared expert); the private hit rate inside the shared experts at 512 density (gallery = that expert's own tokens' keys from ordinary windows); the effect of a split twin (child holding a copy) | reported; private spaces are counted as a carrier in §9.4 only if P(≥ 1 shared expert) ≥ 0.5 and the private R@1 ≥ 0.6 |
| **T0h rare-fact writes** | Training windows, ckpt 7K and control 6K, train-mode routing, ≈ 64 windows forward-only: σ^raw / σ^ent / σ^imp at W = 32 per micro-batch; the fraction of rare (early) AND late answer-cue positions in the top-W (eval tags = bookkeeping only); s_late for unacquired late cues (§9.6); wiki vs world share | **GO**: best criterion p_write ≥ 0.5 on rare cues and s_late ≥ 0.25; **weak GO** 0.3-0.5 / 0.1-0.25 (W = 96 until 3K enabled); **NO-GO** < 0.3 |
| **T0e calibration** | On the T0a capture: d_same → h₀,g², h₀,p², κ; A_w; ω₁, μ_m, A_min at precision ≥ 0.95 that j₁ is the same fact (global and private); γ at wrong-merge ≤ 2% at 256K and at 512 (with the same-fact merge rate); p_conf on correct hits | defaults written; no γ ≥ 0.05 → dedupe off for that space type; no threshold at precision 0.95 with pass rate ≥ 0.3 → the owner is told (§9.4 with lower p_conf) |
| **T0d IVF recall** | IVF (1,024 cells, n_p 4 / 8 / 16, Cq 192) vs brute force on T0a keys + distractors at 64K / 128K / 256K: the fp32 top-1 miss rate for d₁ < 2h₀² and overall, probe overflow, cell Gini; cells fitted at 6K, used at 7K (drift) | miss ≤ 2% at n_p = 8 (else 16, cost +6-10 ms); overflow ≤ 1% |
| **T0f drift 6K → 7K of x̂** | Same tokens / windows at both checkpoints: median drift / 1-NN gap, self-recall R@1 at 256K density over the 1K gap, the part due to the n_m gain (§2.2), the map / cell staleness (μ, σ, centroids fit at 6K, applied at 7K) | reported; the gain part > 30% of the drift → `--gm-addr pregain` offered; self-recall < 0.35 on world text → the owner (renewal must carry everything) |
| **T0g pass coverage** | E[K], E[in-window passes] from the old run's K log | 6.4 / 4.94 within ±10% |

### 18.2 T0b gate separability (GPU; CPU only for a smoke)
Old run ckpt 7K, 16 world + 16 wiki windows, train-mode forward + backward. (a) Proxy targets: the Probe on the rows'
output (as 25c T0b) — c by token type and pass. (b) An offline 25d store built from the T0a / T0h captures (keys = x̂,
contents = the captured expert-input gradients), read per (token, expert) through the §6 kernel with a frozen store,
Probe c per pair. A gate fitted on held-out windows with the §5.1 features: pooled AUC for 1[z > 0]; what the match
features add (AUC with vs without); what per-expert heads add (shared vs per-expert, cross-validated); the open rate at
c_read = 0. **GO**: AUC ≥ 0.6 and the match features add ≥ 0.03; else the owner is told before the run (the gate would
open near-randomly up to the cap). Repeated on the pilot's real store.

### 18.3 T1 toy (CPU; extend `C:\Eitan\Python\NewBrain\speed_wip\splat\consol_toy.py`)
One global space (256K) + 768 private spaces (512) with routing (6 of E_live, overlap ρ_e from T0r, births 192 → 768
with copies), K ~ the run's distribution, cross-pass key spread from T0c, drift from T0f, paraphrase factor from T0a,
p_write and s_late from T0h, γ, p_conf from T0e. Late facts (Δ 105, 116 exposures), rare (≈ 9.5 exposures, spacing
1.3K). Policies: default; `--pm-rate 0.25`; `--gm-renew-geo 1`; `--gm-track all`; W 64 vs 96-until-3K; n_p 8 vs 16.
Thresholds: late-fact trained-blob survival ≥ 0.8 at 12.2K; rare-fact recall ≥ 0.10; wrong merges ≤ 2%; keys per fact
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
3. IVF search = brute force on the top-4 when n_p = all cells; recall measurement code; static overflow handling.
4. Private search = a per-expert loop reference; dropped pairs get no read / write / Probe value.
5. Gate: selection (per-expert 50% cap by rank, exact exploration count, static compaction, eval without exploration);
   zero-init ĝ ≡ 0.
6. **Probe once under checkpoint + gate gradient = autograd + accum invariance + query gradient through w, A**
   (`test_gm_gate_postbwd`, §5.5).
7. GatherRead gradients vs a dense reference (contents, w, A, x̂); bf16 private grad accumulation vs fp32 within
   tolerance; stochastic rounding is unbiased (1,000 steps).
8. Writes: selection static sizes, ≤ 2 per sequence, ≥ 8 apart; captured gradient = autograd ∂L/∂buf at the chosen
   (t, k_w); global content = the sum over kept experts + shared; ε guard (zero gradient → ξ-only, no NaN).
9. Dedupe / merge (η floor, within-step suppression, cell change), spawn / eviction (touched excluded, full-cell rule,
   zombie guard).
10. Confirmation / move (cap 0.5 h₀, ≤ 1 move per step via segment mean = a loop reference, ≤ 8 per 1K, per-space caps,
    path length / net travel).
11. Births: split copies a parent's private space and gate rows exactly (and its AdamW moments); ramp leaves the child
    empty; function unchanged at birth.
12. Causality: a perturbation of tokens > t does not change any read, gate value or injected vector at ≤ t (finite
    difference on a tiny model); micro-batch 2 sees the same store as micro-batch 1; eval is read-only and order
    independent.
13. Lazy clocks = eager; h₀ tracking does not contract on a synthetic population; floors.
14. No new host sync (a sync counter / TorchDispatchMode op log on a CPU mock); resume (store, cells, map, gate,
    generator) gives identical next steps.

### 18.5 T2 GPU bench and pilot (after the speed study releases the GPU)
- Harness `fruit-fly-sim\speed_wip\vram\vram_common.py`: the old config, the old config with `--mem-shared 0
  --mem-private 0 --mem-coarse 0` (the no-memory build), and 25d, at 16 × 2, K ∈ {2, 6, 10, 16}: ms per §17 row, peak
  allocated / reserved incl. one `evaluate()`, host syncs (must equal the no-memory build's), IVF miss / overflow.
- **Pilot**: 1,000 steps of 25d (throwaway): fill, open rates, gate AUC on the real store (T0b repeated), confirmation
  and move counts, travel, reserved VRAM, tok/s. A fault stops the plan before the 12K run.
- GO: tok/s ≥ 0.97 × the old run's config; reserved ≤ 20 GB at K = 10 and 16 incl. `evaluate()`; memory overhead vs
  the no-memory build reported against 3% (OQ-1).

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
| speed | tok/s ≥ **0.97 ×** the old run's config, same harness; memory overhead vs the no-memory build reported (target ≤ 3%, OQ-1) | T2, median of ≥ 3 runs |
| VRAM | peak reserved ≤ **20 GB** incl. `evaluate()` (cap 0.78 on) | T2 + run log |

Reported with every eval: ko_gm_all / _global / _private, gate modes, the fill logs, §15's table.

### 19.3 Order of work
1. This plan → owner review of §OD and the open questions.
2. CPU, alongside the speed study: fixtures from the current code (§18.4-0), unit-test scaffolding, T0g, T0h, the
   capture (if the CPU timing allows), T0a / T0c / T0r / T0e / T0d / T0f on it, then T1.
3. GPU when free: T0b; the capture if not done on CPU.
4. Implement 25d (+ the VRAM defaults); unit tests; T2 bench; 1K pilot.
5. The 12.2K run with the owner's go; GO at 7K and at the end.

---------------------------------------------------------------------------------------------------------------------

## 20. The owner's constraints — status

| # | constraint | status |
|---|---|---|
| 1 | addresses from thoughts only, not learned | met: key = query = fixed map of xf; μ, σ, R, the IVF cells, h₀ and thresholds are statistics; keys move only by rules (spawn, merge, confirmation) |
| 2 | no gate loss on the thought | met: φ, m detached; the gate's loss runs on stored detached copies in a separate graph after backward. The LM loss reaches xf through w, A (decided) |
| 3 | no pointers, no recompute loop | met: writes, confirmations and the gate update use only the step's own forward / backward |
| 4 | filled during training | met (from step 300) |
| 5 | speed / ≤ 20 GB | ≈ 19.0-19.6 GB reserved projected; net faster than the old run; memory overhead 3.0-4.5% vs no memory (OQ-1) |
| 6 | the brain decides at every pass | met, per (token, chosen expert), with two decided bounds (50% cap per expert, 5% exploration) |
| 7 | nothing per pass | met: one global space and per-expert spaces shared by all passes; no pass feature in the gate; K up to 16 (or 40) changes nothing in the memory |
| 8 | self-reinforcement measured | §15 |

---------------------------------------------------------------------------------------------------------------------

## §OD. Defaults set by this plan (the owner may object) and open questions

| | item | default | alternative |
|---|---|---|---|
| 1 | global search | IVF, 1,024 fixed cells, n_p 8 | brute force (+85-135 ms/step) |
| 2 | write budget | W = 64 per step (32 per micro-batch) | 32 (global life 8K, slower fill) |
| 3 | private writes | every kept routed expert (spec); life 260-1,050 steps | `--pm-rate 0.25` (life 2,048 at any size) |
| 4 | shared expert's private writes | at the average routed rate (k / E_live per position) | every position (life 8 steps) |
| 5 | write pass | uniform over the in-window passes | `maxgrad`, `last` |
| 6 | private contents | bf16 + stochastic rounding (saves 0.3 GB) | fp32 (+0.3 GB VRAM) |
| 7 | geometric renewal | off (in-window confirmation reaches the bar at n_eff ≥ 2) | on |
| 8 | gate | shared trunk (32) + per-expert head (39 / expert), one decision for global + private | low-rank per expert; shared only; two decisions |
| 9 | move cap | 0.5 h₀ per step, ≤ 8 moves per 1K steps | h₀ (25c) |
| 10 | gate features | exactly the spec (no per-level uncertainty, no pass feature) | `+cu` |

**Open questions**
- **OQ-1 (speed)**: the memory is estimated at 3.0-4.5% of the step vs a build without memory (it is 10-17% faster than
  the old run). Is the 3% bar relative to the old run (met) or to no memory (at risk)? If the latter: n_p 4, CUDA
  graphs from the speed track, or accept ≈ 4%.
- **OQ-2 (cross-pass)**: if T0c gives n_eff < 1.3, should a position write keys from 2-3 passes (more keys per fact,
  capacity cost), or should the owner reconsider the pass-agnostic design?
- **OQ-3 (private life)**: the spec's "each processing expert" gives private lives of 260-1,050 steps (below the drift
  horizon). Keep, or `--pm-rate 0.25`?
- **OQ-4**: the shared expert's private write rate (§3.2).
- **OQ-5**: one gate decision for global + private (default) or two?
- **OQ-6**: the n_m gain in the address (§2.2): xf as specified, or the pre-gain normalised input if T0f shows the gain
  dominates the drift?
- **OQ-7**: IVF centroids are fixed (an index, not the address); refit them if the cell balance drifts
  (`--gm-ivf-refit`)?

---------------------------------------------------------------------------------------------------------------------

## 21. Flags and implementation size

**Run defaults**: `--empty-cache-eval 1 --mem-fraction 0.78 --eval-batch 16 --batch 16 --accum 2`.

**25d**:

    --mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-nbr 0 --mem-gate 0                (old memory code: kept, off)
    --gm 1 --gm-global 262144 --gm-private 512 --gm-dv 384 --gm-key 64 --gm-fit-step 300 --gm-write-start 300
    --gm-map std-randproj --gm-addr xf
    --gm-search ivf --gm-cells 1024 --gm-cell-cap 512 --gm-probe 8 --gm-qcap 192 --gm-topk 4 --gm-ivf-refit 0
    --gm-write 64 --gm-write-pass uniform --gm-surprise <T0h: raw|ent|imp> --pm-rate 0 --gm-shared-rate avg
    --gm-dedupe <T0e γ_g>,<γ_p> --gm-merge-eta-min 0.25 --gm-h0 <T0e> --gm-fade 2.5e-4,1e-4 --gm-zombie 3000
    --gm-lr 0.01 --gm-clip 0.05 --gm-priv-bf16 1
    --gm-confirm rank --gm-qc 0.25 --gm-conf <T0e> --gm-eta 0.5 --gm-move-cap 0.5 --gm-move-max 8,1000
    --gm-move-space 256,2 --gm-renew-geo 0 --gm-track-travel 4096
    --gm-gate trunk --gm-gate-rank 32 --gm-gate-feats spec --gate-cost 0 --gm-explore 0.05 --gm-cap 0.5
    --gm-gate-lr 1e-3 --gm-gate-wd 0.01 --gm-gate-clip 1.0 --gm-match-grad 1
    --auto-eval 1 --eval-at 7000 --save-report-ckpts 1

**Ablations / debug**: `--gm-search brute`, `--gm-probe 4|16`, `--pm-rate 0.25`, `--gm-write-pass maxgrad|last`,
`--gm-renew-geo 1`, `--gm-track all`, `--gm-confirm off`, `--gm-rescue 1`, `--gm-gate lowrank|shared`, `--gm-gate2 1`,
`--gm-gate-feats +cu|+pass`, `--gm-gate-budget fixed`, `--gate-cost time`, `--gm-match-grad 0`, `--gm-priv-bf16 0`,
`--gm-addr pregain`, `--gm-map pca`, `--ko-gm all|global|private`.

**Implementation size (estimate)**

| part | lines |
|---|---|
| `blob_memory.py`: two space types, map fit, IVF (cells, table rebuild, query layout, bmm, merge, re-rank), private bmm search, match features, GatherRead + Probe, dense grad buffers, row-Adagrad (+ stochastic rounding), write stash / selection / content, dedupe / merge / spawn / evict (full-cell rule, zombie), confirmation + guarded moves + travel, clocks, h₀ / v̄ / Ĉ EMAs, births / reset hooks, state / resume, stats | ≈ 1,000 |
| gate (`BlobGate`): features, trunk + per-expert heads, selection (cap by rank, exploration, compaction), kept tensors, post-backward graph, watchdog, eval modes | ≈ 250 |
| `spiking_lm.py`: the `bmem` argument and the shared-expert input | ≈ 20 |
| `loop_brain.py`: build / RNG fork, ctx per pass, `bmem` through Core, prev μ / match, `select_writes`, hooks, knockouts | ≈ 130 |
| `train_lm.py`: flags, asserts, fit step, write selection / after_backward / after_step calls, gate optimizer group, empty_cache / fraction cap, auto-eval at 7,000, logging, resume | ≈ 150 |
| `eval_world.py`: token-type split, knockouts, gate modes, p_open / p_hit, late-fact fill | ≈ 150 |
| unit tests (§18.4, incl. the reference fixtures) | ≈ 550 |
| T0a / T0b / T0c / T0d / T0e / T0f / T0g / T0h / T0r scripts, T1 toy extension, T2 bench + pilot wrapper | ≈ 500 |
| **total** | **≈ 2,750** (memory code ≈ 1,400; tests and checks ≈ 1,050; eval ≈ 150; trainer ≈ 150) |

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
5. **Private spaces are thin** (outline risk 3): an expert sees ≈ 0.8% of tokens; private lives are 260-1,050 steps;
   twins after a split split the tokens. The global space carries rare facts; private spaces are a bonus that T0r
   quantifies.
6. **Gate separability**: plan 22's AUC was 0.61; with 769 experts the per-expert heads may overfit or stay at the shared
   head. Exploration keeps it measurable; T0b and the pilot decide; the match features should help most.
7. **Harm on other text**: reads go straight into expert inputs; a wrong global read reaches up to 7 experts of a token.
   Zero-init gate (reads start from exploration only), A muting, and the harm criterion (+0.04 nats) guard it.
8. **IVF on fixed cells**: thought drift can unbalance cells (overflow, hot cells churning, misses). Logged; refit is
   an owner switch (OQ-7).
9. **Speed** (OQ-1): 3.0-4.5% for the memory vs no memory, launch-bound; static shapes allow CUDA graphs.
10. **VRAM margin** 0.4-1.0 GB below 20 GB reserved; levers listed (§VRAM).
11. **Self-reinforcement** (reads → thought → key → read; key moves; match-weight gradient) is live from step 300;
    measured (§15), with `--gm-match-grad 0` and `--gm-confirm off` as ablations.
12. **Comparability**: the old run read every position (rows, address reuse); 25d reads gated (token, expert) pairs;
    same data and loader; the old run ends at 7K, so the end-of-run comparison is with the control only.
