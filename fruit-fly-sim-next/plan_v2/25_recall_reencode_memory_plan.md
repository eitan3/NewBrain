# 25. Recall memory: a kNN key-value store with re-encoding on confirmed recall, surprise writes and drift transport (PLAN ONLY) — 2026-10-24

Builds on plans 16 / 20 (the drift), 21 (lazy fading), 22 (gates), 23 (splat / kNN toys: "store the write-time key,
read the nearest one"), 24 (key renewal is first-order; switch / freeze is not), and the real-thought check
`speed_wip/splat/realcheck/{tables,interp}_3k_6k.md`. No code was integrated and nothing was trained. No GPU was used.
Nothing outside this file was edited.

The owner's constraints:
- addresses come from the thought only (no token keys);
- the thoughts are not constrained (the address path is sg);
- no dataset / source pointers;
- no replay loop outside the normal step: every key update happens inside the normal training step;
- the memory is filled during training;
- ≥ 10k tok/s, target ≈ 20k (laptop RTX 5090 24 GB, Windows, plain PyTorch: no Triton, no torch.compile).

Structure:
- §0 what the real measurements imply;
- §1 the store; §2 the lookup; §3 read placement;
- §4 re-encode on confirmed recall; §5 surprise writes; §6 fading / maturity;
- **§7 drift transport (keeping UNUSED keys valid)**;
- §8 full maths, shapes, budget; §9 training signal; §10 toy / real test plan with pass / fail;
- §11 integration, flags, tests; §12 risks; §13 the default config and the implementation size.

---------------------------------------------------------------------------------------------------------------------

## TL;DR
1. **Store.**
   - 2^20 slots (1M; 2M in the quality config).
   - Each slot holds:
     - a key k ∈ R^64: the pass-1 slot mean p̄ of the write-time thought, in a fixed affine frame (centred, one global
       scale; no learned projection, no full whitening);
     - a value v ∈ R^256, trained by gradient (compact row-Adagrad);
     - a strength log α; a fade clock Λ_i; a maturity m ∈ {0..3}; a use count; a running benefit.
   - About 1.35 GB per 1M slots, less than today's row pool with its gradient buffer (≈ 2.1 GB).
2. **Read.**
   - One batched lookup after pass 1 for all 8,192 positions: T = 4 randomly rotated product-quantised cell tables
     (512² cells each, cap 8, 4 cells probed per table), so 128 candidates.
   - Exact distances, then the top 4 with a relative-distance softmax and an absolute confidence A.
   - The read r = A Σ w_j v_j → a content gate → W_o → the next pass's inject + workspace slot 15 (today's
     destinations).
   - The lookup is no-grad. Only the values, the gate and W_o learn.
3. **Re-encode on a confirmed recall, inside the step.**
   - "Confirmed" uses the read's first-order loss reduction c_t = −⟨∂L/∂u_t, u_t⟩ (u = g·r). It is recorded in the
     backward that already runs.
   - The test: c_t > κ_c σ̂_c, plus a top-1 weight ≥ 0.7, a distance margin d₂/d₁ ≥ 1.15, A ≥ 0.3, and a non-negative
     running benefit of the slot.
   - Then the key moves: k ← k + η (p_t − k), with η = 0.5, a capped move, a minimum distance to the competitor, the
     same-step confirmations of a slot averaged, and the fade clock reset.
   - The fixed point is the benefit-weighted centroid of the slot's confirmed readers (§4.4). The bias toward wrong
     readers is bounded by β = ρ_f p_f / (ρ_o p_o + ρ_f p_f).
4. **Surprise writes.** About 128 per micro-batch (1.6% of the positions):
   - where: the final-pass CE is high, and either no confident memory was found or the memory hurt;
   - key = the current view;
   - value = a small random vector + ν·(−∂L/∂r_t) normalised (the gradient write, but into a fresh slot);
   - then trained by gradient;
   - eviction of the globally weakest slot (lazy strength).
5. **Drift transport** (Semantic Drift Compensation, Yu et al. 2020, run inside the step).
   - Each step, 2 sequences of the current micro-batch are stashed with their pass-1 views.
   - M = 8 steps later, ONE pass of those sequences is re-run forward-only. The step gets about 1,024 exact
     (before, after) pairs.
   - Each key near those pairs moves by a kernel-averaged (Nadaraya-Watson, zero-prior) displacement, ÷ M per step,
     with a cap. Confirmed-recall pairs add free (noisier) samples.
   - Cost ≈ 1-2% (2 sequences) to ≈ 4% (the whole micro-batch).
   - **Gate**: the separate 9K→10K test (the neighbours' displacement predicts each point's displacement).
     - §7.5 derives the accuracy needed: with diffusive drift, the early facts (≈ 9.8K steps with no re-encounter)
       need R² ≳ 0.94;
     - with directed drift, per-step errors average out and R² ≈ 0.5 is enough.
6. **What the real numbers say** (§0).
   - Real facts recur about every 105 steps in the world mix. That is t ≈ 0.035 (if the drift is linear) to t ≈ 0.19
     (if it is a random walk) of the 3K move, where the interpolated self-recall is ≥ 0.98.
   - **So re-encoding alone should keep the recurring facts. Transport is needed only for facts that stop recurring**
     (the early people after the cutoff), and it is the risky part.
   - The single largest unknown is **cross-document similarity**: does a fact's cue in a new document (new template)
     land near the stored key? It is test T0a, and it must pass first.
7. **Speed.**
   - Default config F: ≈ 340 extra launches per step + a 1-pass probe forward. That is an estimated 22-30 ms/step
     (≈ 3% of 0.82 s).
   - The memory is no longer the bottleneck. The remaining gap to 20k is ConceptHier (≈ 0.16-0.19 s/step,
     plan 13 / 23), which stays by the owner's decision. The honest estimate is ≈ 16-18k tok/s with the hierarchy,
     ≈ 20k without it.
   - Quality config Q (2 read passes, 256 candidates, 2M + 1M slots): ≈ 60-80 ms/step, ≥ 13k.
8. **Implementation**: ≈ 2,100 lines (a new `recall_memory.py` ≈ 950, wiring ≈ 300, tests ≈ 450, toys / realcheck
   ≈ 400). Behind `--rm 1`; with it off, today's code is bit-identical.

---------------------------------------------------------------------------------------------------------------------

## 0. What the real-thought check implies (3K → 6K; the 9K → 10K pair pending)

**Measured** (32,768 positions; gallery = all positions of checkpoint A; R@1 of a position's own A-key for its B-query):
- over the full 3K gap, everything fails: whitened-32 levels 0.030, p̄ 0.037; whitened-64 levels 0.042, p̄ 0.056;
  raw-448 levels 0.059; **raw p̄ 0.087 (the best)**; full whitening 0.000-0.001 (ruled out).
- The 10-NN survival over 3K steps is 0.15-0.29. The median drift / NN spacing is 1.75-2.55.
- Straight-line interpolation B_t = A + t(B − A) (whitened-32, R@1, ranges over passes 1 / 3 / 5):

| t | 0.1 | 0.2 | 0.33 | 0.45 | 0.58 | 0.66 | 0.74 | 0.85 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| levels 1-4 | 0.99 | 0.97-0.99 | 0.86-0.90 | 0.61-0.65 | 0.31-0.37 | 0.20-0.23 | 0.12-0.15 | 0.06-0.08 | 0.02-0.04 |
| p̄ | 1.00 | 0.99 | 0.96-0.98 | 0.90 | 0.63-0.67 | 0.47 | 0.29 | 0.13 | 0.03-0.05 |

**Mapping t to training steps.** A 1K gap inside the 3K gap is t = 1/3 if the drift is directed (linear in time) and
t = √(1/3) = 0.58 if it is diffusive (a random walk). The 9K → 10K pair decides which. Steps until R@1 falls to
0.9 / 0.5:

| view | linear: R@1 0.9 / 0.5 | random walk: R@1 0.9 / 0.5 |
|---|---|---|
| levels | ≈ 950 / ≈ 1,550 steps | ≈ 300 / ≈ 800 steps |
| p̄ | ≈ 1,350 / ≈ 1,900 steps | ≈ 600 / ≈ 1,200 steps |

**Consequences for the design:**
1. **View = p̄** (64-d, it exists already, the most drift-robust). The levels lose 0.3 at t = 0.58 vs p̄.
   - **Frame: raw centred p̄ with one global scale.** Raw beat whitened at the full gap: 0.087 vs 0.056 / 0.037.
   - Whitened-32 / partial whitening are T0 alternatives. The interpolation was measured only in whitened-32, so the
     raw-p̄ interpolation curve is T0b.
2. **Pass**: p̄ robustness is nearly pass-independent (pass 1 / 3 / 5: 0.668 / 0.664 / 0.634 at t = 0.58).
   - **Read after pass 1 (k = 0).** It is valid for every halting depth K ≥ 2 (k_min = 2). It gives the read 5 passes
     to act. The transport probe then needs only ONE pass.
3. **Renewal intervals.** In the world mix, a late fact is re-mentioned about every 105 steps (233 exposures over
   400M tokens, 16,384 tokens per step; §9).
   - That is t ≈ 0.035 (linear) to 0.19 (random walk). Both are in the R@1 ≥ 0.98 region.
   - **Re-encode-on-recall is sufficient for recurring facts by a wide margin**, if a recurrence lands near the key at
     all (T0a).
4. **Non-recurring keys** (early people after the 40M-token cutoff ≈ step 2,441; evaluated at step 12.2K, so
   ≈ 9.8K steps later) are ≈ 3.3 × the 3K gap away: hopeless without transport (R@1 ≈ 0). §7.5 says how good the
   transport must be.
5. **Gallery density.** The check's gallery is dense: every position of 32 windows, so neighbouring tokens of the same
   document are competitors. The store's keys are sparser (1.6% of positions) but more numerous (1M).
   - The stability law (plan 23 §2) says recall falls slowly with N (∝ N^{1/d_eff}).
   - T0e re-measures it with a 1M-key gallery built from the captures.

---------------------------------------------------------------------------------------------------------------------

## 1. The store

N_s slots (default 2^20). Per slot i:

| field | shape / dtype | set by | meaning |
|---|---|---|---|
| k_i | (64) fp32 | rules (§4, §5, §7) | key: the view in the fixed frame |
| ‖k_i‖² | fp32 | with k_i | for the distance GEMV |
| v_i | (256) fp32 | gradient (compact row-Adagrad) + write init | value |
| S_i | fp32 | Adagrad | accumulator (importance; EWC-like protection, §6) |
| log α_i | fp32 | rules | strength / opacity: eviction, confidence |
| Λ_i, Ψ_i | fp32 | rules | lazy fade / discount clocks (plan 21 style), reset at renewal |
| m_i | int8 (stored in fp32 meta) | rules | maturity 0..3 (spaced confirmed recalls) |
| ρ_i = 4^{−m_i} | (derived) | — | fade-rate multiplier |
| u_i | int32 | rules | use count (confident reads) |
| b̄_i | fp32 | rules | running benefit EMA of c over its confident reads |
| t^m_i, t^w_i | int32 | rules | step of the last maturity increment / the birth |
| cell_{i,τ}, pos_{i,τ} | (4) int32, (4) int8 | index | back-pointers into the 4 tables |

**The key frame** (fixed, no gradient, no refit):
- p̂ = (p̄ − μ₀)/σ₀ with μ₀ ∈ R^64 and σ₀ a scalar (the RMS norm), both fitted once on 64K positions at `--rm-fit-step`
  (default 300, just before writes start).
- Keys and queries use the same frame forever, so the frame's own staleness does not matter. The drift is handled by
  re-encoding and transport, not by refitting.
- Variant `--rm-frame pw:α`: p̂ = Λ^{−α/2}Vᵀ(p̄ − μ₀) (partial whitening, α ∈ {0.5, 1}, top-32 or 64), chosen by T0b.

**The index**: 4 tables τ. Each has:
- a fixed random orthogonal rotation R_τ ∈ R^{64×64};
- two codebooks C_{τ,1}, C_{τ,2} ∈ R^{512×32} (k-means on the rotated halves; refit every 1,000 steps, §2.3);
- buckets B_τ ∈ int32^{512²×8} (−1 = empty).
- A slot is filed in its top-1 cell of each table. When a bucket is full, the weakest pointer is overwritten (§2.3).

**VRAM per 1M slots**:
- keys 256 MB;
- values 1,024 MB;
- meta (8 fp32) 32 MB;
- back-pointers 20 MB;
- buckets 4 × 262,144 × 8 × 4 B = 34 MB.
- **≈ 1.37 GB.**
- **No table-sized gradient buffer**: MemRead's backward writes a compact (Q·k, 256) buffer of (slot, w·g) pairs that
  after_step segment-sums (§8.3). This saves 1 GB per 1M slots vs today's G.

---------------------------------------------------------------------------------------------------------------------

## 2. Lookup (one batched call for all positions of a read)

### 2.1 The steps and the planned op count (per read, Q = 8,192 queries, no_grad)

| # | operation | shape | ops |
|---|---|---|---|
| 1 | p̂ = (p̄ − μ₀)/σ₀ (one addcmul) | (Q, 64) | 1 |
| 2 | rotate into the 4 tables: one matmul with [R₁ … R₄] | (Q, 256) → view (4, 2, Q, 32) | 2 (matmul + permute-copy) |
| 3 | centroid distances, all tables and halves: baddbmm(‖c‖², z, Cᵀ, α = −2) | (8, Q, 512) = 134 MB | 1 |
| 4 | top-m = 4 per half | (8, Q, 4) | 1 |
| 5 | pair scores (4 × 4 per table), top-kc = 4, cell id = i·512 + j | (4, Q, 16) → (4, Q, 4) | 5 |
| 6 | bucket gather + table offset | (4, Q, 4, 8) | 2 |
| 7 | → candidates (Q, 128), permute-copy | (Q, 128) int32 | 1 |
| 8 | invalid (−1) and duplicate masks: sort, compare neighbours, combine | (Q, 128) | 4 |
| 9 | gather keys (clamped ids) + their norms | (Q, 128, 64) = 268 MB fp32 (2 query chunks → 134 MB) | 3 |
| 10 | distances d = ‖k‖² − 2⟨k, p̂⟩ + ‖p̂‖² (bmm + 2 adds), masked to +∞ | (Q, 128) | 4 |
| 11 | top-k = 4 (smallest) + gather the slot ids | (Q, 4) | 2 |
| 12 | gather meta [log α, Λ_i, ρ_i, m_i] for the 4 winners; effective strengths | (Q, 4, 4) | 3 |
| 13 | weights w, confidence A, margin, features (§2.2) | (Q, 4), (Q) | 6 |
| | **total per read** | | **≈ 36** |

- It is one call for all positions. A config with several views per token (levels) just adds query rows (Q = N·V).
- The candidate distances are exact (fp32 keys). Optionally bf16 for steps 9-10 with an fp32 re-score of the 4
  winners (+2 ops, halves the 268 MB temp).
- Under activation checkpointing, the lookup runs once (cached in the block's cache dict as today's `sel`); the
  recompute reuses it.

### 2.2 The read kernel
- For the query p̂_t with winners j₁..j₄ (d₁ ≤ … ≤ d₄) and effective strengths
  ã_j = log α_j − λ_α ρ_j (Λ(t) − Λ_j):

      w_tj = softmax_j( ã_j − (d_j/d₁ − 1)/τ_w ),     τ_w = 0.1                  (relative: scale-free)
      A_t   = exp(−d₁ / (2 h₀²)) · σ(ã_{j₁})                                      (absolute confidence ∈ (0, 1))
      r_t   = A_t Σ_j w_tj v_{j} · f_j,   f_j = exp(−λ_v ρ_j (Λ(t) − Λ_j))       (lazy value fade, §6)

- h₀² is an EMA (rate 0.01) of the median d₁ over the confirmed reads of the step. A typical confirmed read then has
  A ≈ e^{−½}.
- Plan 23 found that blending neighbours hurts (soft top-8: 0.39-0.58 vs kNN-1 0.99). τ_w = 0.1 makes a neighbour
  10% farther weigh e^{−1}: nearly kNN-1, but still smooth near ties, which the margin rule then refuses to re-encode
  (§4).
- Multiplying by A means a far, irrelevant reader both contributes little and writes little gradient into the slot
  (∂L/∂v_j ∝ A w_j).

**The gate and the output (one head):**

      φ_t = [p̄_t (64); r_t (256); A_t; log(d₂/d₁); onehot(m_{j₁}) (4); log(1 + age_{j₁})]   (327)
      g_t = σ(w_gᵀ φ_t + b_g),   b_g = −2 (as today)
      u_t = g_t r_t,   M_t = W_o u_t (W_o ∈ R^{384×256}, zero init),   inject_{k+1} += M,   slot 15 += W_ms M

- p̄ enters the gate with gradient, exactly as in today's `read_shared`. That is a normal downstream use of the
  thought, not a constraint. The ADDRESS (p̂ → candidates → w, A) is fully no-grad.

### 2.3 Index maintenance
- **Filing a slot** (spawn / re-encode / transport, batched per step): its top-1 cell per table, from steps 1-5 with
  m = 1, kc = 1 (≈ 6 ops for all moved slots).
- Collisions within a step:
  1. sort the inserts by (τ, cell); the rank within a group is a cumsum trick (3 ops);
  2. gather the bucket's 8 pointer strengths and sort them ascending (2 ops);
  3. the r-th insert into a bucket takes the r-th weakest position (r ≥ 8: dropped in this table only);
  4. clear the evictee's back-pointer;
  5. clear the mover's old pointer.
  - About 16 ops per batch of moved slots.
  - A slot with 0 pointers in all tables is dead and goes to the free list (it is evicted first).
- **Codebook refit every 1,000 steps** (it drifts with the keys):
  1. k-means (5 iterations) on 64K keys sampled by strength;
  2. re-file all slots (chunks of 256K: 4 × ≈ 10 ops);
  3. rebuild the buckets by a sort (keeping the strongest 8 per cell).
  - ≈ 60 ops, ≈ 15-25 ms, once per 1,000 steps (≈ 0.02 ms/step amortised).
  - It changes only the index, never a key, so it is not a replay.
- Monitored: the pointer coverage (the fraction of live slots with ≥ 2 table pointers; target ≥ 0.95), the bucket
  overflow rate, and the candidate-recall vs brute force on 1K sampled queries every 250 steps.

---------------------------------------------------------------------------------------------------------------------

## 3. Read placement (decided against the speed budget)

**Default (config F): ONE lookup after pass 1 (k = 0).**
- The output goes into pass 2's inject and into workspace slot 15. The slot carries it through all later passes
  (`_slot_write` is cumulative).
- Why not every pass?
  - The ops are cheap: ≈ 84 per read with backward (§8.4), so 5 reads ≈ +15 ms.
  - But the views differ per pass: each read pass needs its own key family (×passes slots or writes) and its own
    transport probe depth (×passes probe cost).
  - The real check shows no drift-robustness gain from later passes.
  - With halting, K can be 2, and then only k = 0 exists.
- The real check showed no gain from later passes for p̄ (0.668 / 0.664 / 0.634 at t = 0.58).

**Optional**:
- `--rm-reinject 2`: re-inject the same gated read into pass 3 (reusing r; the gate is recomputed from p̄ at k = 2;
  ≈ 8 ops). Off by default: the workspace slot already carries M.
- **Quality config Q**: a second key family at k = 2 (after pass 3) with its own 1M slots, index and writes, if T0g
  shows that pass-3 views add information (cross-pass recall of a pass-1 key from a pass-3 query < 0.8 AND a better
  same-fact retrieval at pass 3).

**What the view is**: p̄ = the mean over the 16 workspace slots of n_s(s) after the pass. It is computed inside
`_hier_step.block()` today. ConceptHier's μ_l are NOT on the memory path (`--rm-view level:l` is the ablation).

**Gates**:
- the content gate g_t of §2.2;
- no access router (plan 22 off): one view per token is cheap enough to read everywhere;
- the read can be closed per position by the content gate only.

---------------------------------------------------------------------------------------------------------------------

## 4. Re-encode on CONFIRMED recall (inside the step)

### 4.1 The confirmation signal (available in the same step)
- u_t = g_t r_t passes through an identity autograd function `Probe` whose backward records

      c_t = −⟨∂L/∂u_t, u_t⟩          (Q,)  — 2 ops in the backward that already runs

  - c_t is the first-order loss reduction from the read: L(u_t) − L(0) ≈ ⟨∂L/∂u_t, u_t⟩.
  - Because M_t = W_o u_t, it equals −⟨∂L/∂M_t, M_t⟩ (the per-position "benefit" of plan 22 §2.4).
  - Under checkpointing the backward runs once per micro-batch, so c is recorded once.
  - The loss L includes deep supervision (every later pass's CE) and the concept losses: c is the read's total
    first-order value.
- σ̂_c = RMS of c over the positions with A_t ≥ A_min in the micro-batch (device scalar; scale-free across LR / loss).
- **Confirmed(t)** ⟺ all of:
  1. **useful**: c_t > κ_c σ̂_c (κ_c = 1.0);
  2. **confident**: w_{t,1} ≥ 0.7 and A_t ≥ A_min = 0.3;
  3. **no competitor**: d₂/d₁ ≥ μ_m = 1.15;
  4. **slot-level consistency**: b̄_{j₁} ≥ 0 after this read's update
     (b̄ ← 0.9 b̄ + 0.1 · c_t/σ̂_c on every confident read of the slot);
  5. **not hurt recently**: no disconfirmation of the slot in the last 16 steps
     (a disconfirmation = a confident read with c_t < −κ_c σ̂_c).
- The budget: the top E_b = 256 confirmed positions per micro-batch by c_t/σ̂_c (static shape: topk on the masked
  score). An excess is ignored. Typical counts are logged.

### 4.2 The update (in after_step, once per optimizer step; both micro-batches' records are stashed)
- For slot j with confirmed reads T_j in this step (averaged, so the result does not depend on the order):

      p̄_j*  = mean_{t ∈ T_j} p̂_t                                       (index_add_ of p̂ and counts)
      Δ_j   = η_k (p̄_j* − k_j),   η_k = 0.5
      Δ_j   ← Δ_j · min(1, h₀ / ‖Δ_j‖)                                   (max move per step = h₀)
      k_j'  = k_j + Δ_j
      skip if ‖k_j' − k_{j₂}‖² < γ_d h₀²  (γ_d = 0.25; k_{j₂} = the competitor gathered at the read)
      on success: Λ_j ← Λ(t), log α_j ← min(log α_j + 1, log 8), u_j += |T_j|,
                  m_j ← min(m_j + 1, 3) if t − t^m_j ≥ 64 (spaced), then t^m_j ← t;  re-file if a cell changed

- ≈ 18 ops + the shared re-filing.
- A disconfirmation does log α_j −= 1 and m_j ← max(m_j − 1, 0).
- **η_k = 0.5, not 1**:
  - under diffusive drift the steady-state key error is δ/√(η(2 − η)) = 1.15 δ (η = 1: δ);
  - under directed drift it is δ/η = 2δ, with δ the drift between renewals (≈ 105 steps: t ≤ 0.19, i.e. R@1 ≥ 0.98
    either way);
  - in exchange, a false confirmation moves the key only half-way, and the key averages over phrasings (§4.4).
  - With `--rm-eta-young 1`, slots with u_j < 2 use a hard move (η = 1).

### 4.3 Guards
| risk | guard |
|---|---|
| confirmation bias (a key drifts toward any reader) | the usefulness test c is independent of distance; only the sign / size of the gradient benefit triggers a move; b̄ ≥ 0 and the 16-step disconfirmation lock |
| wrong re-filing (a key hijacked by another context) | margin d₂/d₁ ≥ 1.15, w₁ ≥ 0.7, A ≥ 0.3; a move of at most η = 0.5 of the way and ≤ h₀ |
| two slots competing | no re-encode in the margin dead zone; the disconfirm-demote makes one of them fade |
| key collapse | the minimum distance to the competitor after the move (γ_d); the separation bound of §4.4; monitored |
| order dependence | per-slot averaging of the step's confirmations |
| runaway (one hot slot reads everything) | its readers must each pass the margin test; the use count and b̄ are logged; a slot whose b̄ < 0 cannot move |

### 4.4 Why it is unbiased enough (equations)
- **Stochastic-approximation view.** For slot j, let the contexts that read it confidently be indexed by s, with read
  rates ρ_s, current views x_s(t) and confirmation probabilities p_s = P(c > κ_c σ̂_c ∣ s).
- With small η, k_j follows the mean ODE (Robbins-Monro):

      k̇ = η Σ_s ρ_s p_s (x_s − k)   ⇒   k* = Σ_s ρ_s p_s x_s / Σ_s ρ_s p_s

  So the key tracks the **benefit-weighted centroid of its confirmed readers**: online k-means with a usefulness
  weight. That is the right target for a fact read through several phrasings.
- **Bias toward non-owners.** Split the readers into the owner set o (the value helps: E[c] = μ_o > 0) and the others
  f (E[c] = 0).
  - With c ≈ Gaussian with scale σ̂_c: p_o = Φ(μ_o/σ − κ_c) and p_f ≤ Φ(−κ_c) = 0.16 at κ_c = 1.
  - The bias weight is

        β = ρ_f p_f / (ρ_o p_o + ρ_f p_f),     ‖k* − x̄_o‖ ≤ β · max_f ‖x_f − x̄_o‖ ≤ β R_basin

    where R_basin is bounded by the margin rule: an f-reader only reaches the update if k_j is its nearest key with
    d₂/d₁ ≥ 1.15.
  - Example: ρ_f = ρ_o, p_o = 0.6 → β = 0.21. With b̄ ≥ 0 and the disconfirmation lock, p_f falls further for slots
    whose net benefit is not positive.
  - The key stays in its owner's basin while β R_basin < r_NN/2. That is checked by the toy at the measured
    false-positive rates (T1, the wrong re-filing rate).
- **Separation (no collapse) after a move.** The move is allowed only if ‖p − k₂‖² ≥ 1.15 ‖p − k₁‖² = 1.15 d₁, and
  k₁' = k₁ + η(p − k₁) lies on the segment. Then

      ‖k₁' − k₂‖ ≥ ‖p − k₂‖ − ‖p − k₁'‖ ≥ (√1.15 − (1 − η)) √d₁ = 0.57 √d₁   (η = 0.5)

  So a re-encode never brings the two nearest keys of the reading context closer than 0.57 of the read distance. γ_d
  adds an absolute floor.
- **Unconfirmed alternatives fail** (plan 24: confidence-only re-encoding: 2-4× more moves, 0.2% hijacks, smaller
  gain). The gradient-benefit test is what separates "near" from "useful".
- **Noise caveat.** c_t is a single-sample gradient inner product; its per-position SNR is unknown.
  - The toy sweeps SNR ∈ {0.3, 1, 3} (T1).
  - The run logs P(c > 0 ∣ A-bucket) and P(confirmed ∣ the value later decoded right) on the probe set (§11.4).

---------------------------------------------------------------------------------------------------------------------

## 5. Surprise-based writing

### 5.1 Where
- σ_t = the final-pass next-token CE at t (detached; already computed by the deep supervision / read-out).
- Eligible(t) ⟺
  - t ≥ 8 tokens after the document start, and t is not an EOT (plan 16: window-start thoughts are generic);
  - AND no confident useful memory: A_t < A_w = 0.3, OR c_t < −κ_c σ̂_c (the memory hurt; then also the
    disconfirm-demote of §4.2).
- Budget: per sequence, the top n_w = W_b/B = 8 eligible positions by σ_t (topk on (B, T); static shapes; at least
  8 tokens apart by a max-pool suppression, 2 ops). **W_b = 128 per micro-batch, 256 per step.**
- The candidates (index, p̂_t, ∂L/∂r_t) are stashed in each micro-batch's backward, then written in after_step.

### 5.2 Key and value
- **Key**: k = p̂_t (the current view, the same frame as the reads).
- **Value**: the target is the change of r that would most reduce this position's loss.
  - With M = W_o g r, the steepest-descent change of r is −∂L/∂r_t = −g_t W_oᵀ ∂L/∂M_t − (the gate path). So

        v_new = σ_ξ ξ + ν v̄ · (−∂L/∂r_t) / ‖∂L/∂r_t‖,     ξ ~ N(0, I), σ_ξ = 0.02, ν = 1, v̄ = EMA of median ‖v‖ of slots with m ≥ 1 (init 0.05)

  - This is "the gradient write, but into a fresh slot". The ξ term breaks the W_o = 0 deadlock at the start: then
    ∂L/∂r = 0, so values start random, as today's rows do.
  - Rejected / ablation alternatives:
    - (b) `--rm-vinit tok`: v = P E[y_{t+1}], a fixed random projection of the target token's embedding (a latent
      kNN-LM). Independent of W_o, but it stores only the next token, not the concept.
    - (c) a learned write head: it has no clean training signal, since the stored values are parameters, not
      functions of the head.
- **Then**: the values train by gradient (§8.3) and by nothing else.

### 5.3 Eviction
- The new slots go to the free list first, then to the W_b globally weakest slots by ã_i (lazy strength, §6).
  - One topk(largest = False) over the N_s strengths: 4 MB, 1 op + 3 ops for the lazy strengths.
  - The evictees' pointers are cleared through the back-pointers (2 ops).
- A new slot starts at log α = 0, Λ_i = Λ(t), m = 0, S_i = S₀ (auto as today), u = 0, b̄ = 0.
- Turnover: with 256 writes per step, an unconfirmed slot survives ≈ N_s/256 ≈ 4,100 steps at 1M; mature slots
  (ρ = 1/64) are practically never the weakest.
- World facts need ≈ 25.6K facts × 1-2 keys ≈ 50K slots: the store is dominated by Wikipedia surprises, which is
  fine.

---------------------------------------------------------------------------------------------------------------------

## 6. Fading and maturity (lazy; no global switch, no freeze)
- **One global clock** Λ(t) = t (the step). Per slot, the effective quantities since the last renewal are:

      ã_i(t) = log α_i − λ_α ρ_i (t − Λ_i),     λ_α = 1e-3        (strength: half-life 693 steps at m = 0, 44K at m = 3)
      f_i(t) = exp(−λ_v ρ_i (t − Λ_i)),        λ_v = 1e-4        (value magnitude: half-life 6.9K / 444K steps)
      ρ_i = 4^{−m_i}

- **Renewal** (a re-encode or a confirmed read) folds the pending fade in and resets the clock:
  log α_i ← ã_i + 1, v_i ← f_i v_i, Λ_i ← t.
- The value fade is folded into the stored v exactly as plan 21 §1.1 does; MemRead receives w·f.
- **Maturity protects the value (EWC-like, through Adagrad)**: the accumulator is discounted per slot,

      S_i ← max( exp(−(1 − β_i)(t − Ψ_i)) S_i + mean(g_i²), 0.1 S₀ ),   1 − β_i = (1 − β₀) ρ_i,   β₀ = 0.99,   Ψ_i ← t

  - immature: S ≈ 100 g² at steady state, so the value stays plastic;
  - m = 3: a near-accumulating S (1 − β = 1.6e-4, horizon 6K steps), i.e. 1/√S protection of a value that has been
    confirmed useful repeatedly.
  - The keys are NEVER frozen by maturity (plan 24: frozen stale keys are the failure); maturity only slows fading and
    protects the value.
- There is no phase, no switch, no ramp. The LR decay enters only through the drift itself. Optionally λ_α ×0.5 after
  the LR-decay start (`--rm-fade-decay`), because the drift slows.

---------------------------------------------------------------------------------------------------------------------

## 7. Drift transport: keeping UNUSED keys valid (no dataset pointers, inside the step)

Re-encoding renews only the keys that are read. A key whose context stops recurring (the early people) is still lost
after ≈ 1-2K steps (§0). Transport moves every key by an estimate of the drift of the thoughts around it. This is
Semantic Drift Compensation (Yu et al., CVPR 2020, "Semantic Drift Compensation for Class-Incremental Learning":
class prototypes are moved by the kernel-weighted mean displacement of the current task's embeddings between the old
and new model), here applied continually inside training, to memory keys.

### 7.1 Drift samples
- **(a) Exact pairs: "brain sampling"** (the owner's (2) + the coordinator's SDC form).
  1. Each step, the first n_p = 2 sequences of the current micro-batch (1,024 tokens: the token ids, 2 KB) are
     stashed, with their pass-1 views b_i = p̂_i from the training forward (free: computed for the read anyway).
     Ring of M = 8 entries, ≈ 2.2 MB.
  2. At step t (after the optimizer step), the entry from t − M is re-run **forward-only, ONE pass**, with the current
     weights, under no_grad. p̄ after pass 1 needs only the embedding, init_state, one Core call and n_s. There is no
     memory input before pass 1.
  3. That gives a_i. So each step has n_p·T = 1,024 exact pairs (b_i at θ_{t−M}, a_i at θ_t):

         Δ_i = a_i − b_i,   location c_i = (a_i + b_i)/2   (the midpoint: the keys are ≈ M/2 steps behind θ_t)

  - **This is not a dataset pointer**: the sequences come from the current batch and live only M steps on the GPU.
  - **It is not a replay loop**: one pass, no gradient, inside the step.
  - Routing noise (0.3 in training) is kept ON in the probe, so b and a are draws from the same noisy view
    distribution (zero-mean jitter in Δ; T0d measures its size).
- **(b) Free pairs from confirmed recalls** (the owner's (1)).
  - Every re-encode of §4 gives a pair (k_j, p̄_j*) spanning t − Λ_j steps.
  - It is used as a velocity sample Δ = (p̄_j* − k_j)/(t − Λ_j) per step, at location k_j, with weight λ_rc = 0.3,
    only if ‖p̄_j* − k_j‖ ≤ 3 × the median read distance.
  - These are noisier (several phrasings, false confirmations), so they get a lower weight.
- **(c) Variant: noise / perturbed probes** (the owner's idea; `--rm-probe-noise`). The stashed sequences re-run
  twice, the second time with p = 15% of the tokens replaced by random tokens. That gives pairs in regions near, but
  off, the data.
  - Pure random-token sequences are likely off-manifold (thoughts of gibberish) and less representative.
  - Tested only as T0f before being enabled.

### 7.2 The field estimate (Nadaraya-Watson with a zero prior)
- The work runs from the probes' side, through the SAME 4-table index:
  1. Each location c_i is looked up (§2.1, 128 candidates).
  2. Its top m_t = 32 slots by distance d_ij are kept.
  3. Kernel K_ij = exp(−d_ij / (2h²)); h² = κ_h · (the median over the probes of the distance to their nearest key),
     κ_h ∈ {1, 2, 4} fitted by T0c (the best-R² kernel).
  4. With robust per-pair weights ω_i = min(1, 3 med‖Δ‖/‖Δ_i‖) (outliers from routing jitter / births):

         num_j += Σ_i ω_i K_ij Δ_i,    den_j += Σ_i ω_i K_ij               (index_add_, 2 ops)
         Δk_j = (1/M) · num_j / (w₀ + den_j),   w₀ = 1                     (zero prior: little support → little move)
         Δk_j ← Δk_j · min(1, ρ_T h₀ / ‖Δk_j‖),   ρ_T = 0.25               (cap per step)

- The 1/M turns the M-step displacement of the window [t − M, t] into a per-step rate. Consecutive windows overlap, so
  each step applies 1/M of a trailing-window average. Smooth, incremental, and no double counting in expectation.
- **Affine fallback** (`--rm-tr-affine 1`, the coordinator's option): where 0.1 < den_j < 1 (thin support), use a
  local weighted least-squares fit Δ(x) ≈ Δ̄ + J(x − c̄) over the probes in the slot's candidate set (a ridge-regularised
  64×64 J per probe-cluster). Default off.
  - Plan 20: a GLOBAL linear re-alignment hurt addressing (0.199 vs 0.210 frozen).
  - A LOCAL affine fit is the only form worth testing (T0c).
- The keys that re-encoded this step skip transport (their pair is already exact). Moved keys are re-filed (shared
  with §4 and §5).
- Coverage: ≤ 1,024 × 32 = 32K slot updates per step; regions visited by current data are transported often. Regions
  no current thought visits get no estimate, and no move (w₀). That is the honest limit: no information there.

### 7.3 Guards
| risk | guard |
|---|---|
| a wrong re-filing pollutes the field | recall pairs at weight 0.3, the 3× median gate, the robust ω; one outlier moves a key by ≤ λ_rc ω K/(w₀ + den) |
| routing-noise jitter | midpoints + the kernel averaging over m_t = 32; the jitter is measured (T0d); n_p can be raised |
| over-transport (wrong direction, accumulating) | the zero prior (it shrinks toward no move), the cap ρ_T h₀ per step, and the 1/M rate |
| the field is not local / smooth | the go / no-go below; `--rm-transport 0` falls back to re-encode only |
| a transport that contracts keys together | not prevented (a real contraction of the thoughts is followed); collisions are monitored every 250 steps (the NN distance of 4K sampled keys) |
| keys far from all data | no move (w₀); they fade unless mature |

### 7.4 Cost vs the speed rule
| variant | extra work per step | estimate |
|---|---|---|
| 2 stashed sequences, 1 pass, no grad (default) | 1,024 tokens through one Core call (≈ 1/96 of a step's forward-equivalent FLOPs) + about 150-300 launches, plus the probe lookup / kernel / apply (≈ 54 ops) | **≈ 8-15 ms ≈ 1-2%** (launch-bound; to be counted with TorchDispatchMode) |
| 16 sequences (the whole micro-batch), 1 pass | ×8 tokens, same launches | ≈ 25-35 ms ≈ 3-4% (≈ the coordinator's +4%) |
| config Q (probe to pass 3 for the second family) | ×3 passes | ≈ +2-4% more |
| (rejected) a full-K forward of 1/8 of the batch every step | 6 passes | ≈ 6-10% |

- Both default variants fit ≥ 10k with a wide margin.
- At the 20k target, the 2-sequence variant costs ≈ 1-2%. n_p is the knob, and T0c's R²-vs-probe-density curve
  decides it.

### 7.5 How accurate must the transport be? (this sets the gate)
- Write the per-step drift of a key's context as δ_s, the estimate as δ̂_s = δ_s + ε_s, with relative error
  ‖ε‖ ≈ √(1 − R²) ‖δ‖. R² is the spatial predictability of a point's displacement from its neighbours' displacements.
- After T steps without a re-encounter, the key error is ‖Σ ε_s‖:
  - **independent errors**: √T ε δ.
    - **Directed drift** (total drift T δ): the relative error is √(1 − R²)/√T. It averages out; even R² ≈ 0.5 keeps
      the error tiny.
    - **Diffusive drift** (total drift √T δ): the relative error is √(1 − R²), i.e. an effective interpolation
      t_eff = √(1 − R²) · t_raw.
  - **Systematic errors** (the shrinkage w₀/(w₀ + den), the kernel bias h²∇²Δ) accumulate linearly in both cases.
    Hence the small w₀ where support exists, and the measured bias term below.
- The early people: no re-encounter from step ≈ 2,441 to the eval at ≈ 12.2K, i.e. T ≈ 9.8K ≈ 3.3 × the 3K gap.
  - **Diffusive**: t_raw = √(9.8/3) = 1.81. For R@1 ≥ 0.9 (p̄: t_eff ≤ 0.45) we need √(1 − R²) ≤ 0.25, i.e.
    **R² ≥ 0.94**. With R² = 0.8: t_eff 0.81 → R@1 ≈ 0.2.
  - **Directed**: the independent part gives t_eff ≈ 3.3 · √(1 − R²)/√T_upd, with T_upd ≈ 9.8K/M ≈ 1,200 independent
    window updates, so ≈ 0.07 · √(1 − R²): practically 0. What remains is the systematic part:
    t_eff ≈ 3.3 · |b|, with b the relative bias of the field.
    - **R² ≥ 0.5 suffices if |b| ≤ 0.1** (t_eff ≤ 0.33).
  - The last 2.4K steps are under LR decay (a slower drift), which relaxes both.
- **Go / no-go (the separate 9K → 10K test is the gate; reference only here).** On the 9K/10K captures, per variant
  (p̄ pass 1 in the chosen frame):
  1. Leave-document-out kNN field: for each position i, Δ̂_i = the kernel average of the Δ_j of its m nearest A-points
     from OTHER documents (the probes in training come from other sequences). Report:
     - R² = 1 − Σ‖Δ_i − Δ̂_i‖²/Σ‖Δ_i − Δ̄‖²;
     - the bias b = ⟨Δ̂ − Δ, Δ⟩/‖Δ‖²;
     - over κ_h ∈ {1, 2, 4}, m ∈ {8, 32}, a probe density of {1K, 4K, 16K} points, and with / without the local
       affine fit.
  2. Recall after transport: R@1 of B_i against the gallery {A_j + Δ̂_j} (all positions transported by the field from
     the probe subset).
  3. Time structure: compare the 1K displacement with the 3K one (3K → 6K) in the same frame:
     - ‖Δ_3K‖/‖Δ_1K‖ ≈ 3 → directed;
     - ≈ √3 → diffusive.
     This picks the row of the table in §0 and the formula above.
  - **PASS** (build transport): R@1 after transport ≥ 0.7 at the 1K gap AND ≥ 2× the untransported R@1, AND the
    predicted early-fact recall from the formula above (with the measured R², bias and time structure) ≥ 0.5.
  - **FAIL**: `--rm-transport 0` by default. The memory still ships with re-encoding (recurring facts) and an honest
    "non-recurring facts are not kept" statement.

---------------------------------------------------------------------------------------------------------------------

## 8. Full maths, shapes, parameters, budget

### 8.1 Shapes (B = 16, T = 512, N = B·T = 8,192 positions per micro-batch, accumulation 2, K = 6, d = 384, d_s = 64, d_v = 256, N_s = 2^20)
| tensor | shape | notes |
|---|---|---|
| p̄ (pass 1) | (16, 512, 64) → (8192, 64) | n_s(s).mean(−2), already computed |
| p̂ | (8192, 64) fp32 | frame (μ₀, σ₀) |
| rotated halves | (4, 2, 8192, 32) | |
| centroid scores | (8, 8192, 512) | 134 MB, freed |
| candidates | (8192, 128) int32 | |
| candidate keys | (8192, 128, 64) | 268 MB fp32 (2 chunks: 134 MB) |
| winners J, d | (8192, 4) | |
| w, A, f | (8192, 4), (8192), (8192, 4) | no grad |
| r, u | (8192, 256) | grad to V only |
| g | (8192) | gate (327 → 1) |
| M | (8192, 384) → (16, 512, 384) | inject + slot 15 (W_ms: 384 → 64) |
| compact value grad | (32,768, 256) + ids | 33 MB per micro-batch |
| stash per micro-batch | p̂ (8192, 64), c, σ, A, J, d, ∂L/∂r of the W_b write candidates (128, 256) | ≈ 3 MB |
| transport ring | 8 × (1,024 ids + (1,024, 64) views) | 2.2 MB |

### 8.2 Gradients (only these parameters learn by gradient)
- With h_t = ∂L/∂u_t (the backward input at the Probe):
  - **W_o**: ∂L/∂W_o = Σ_t (∂L/∂M_t) u_tᵀ.
  - **W_ms**: as today.
  - **gate** (w_g, b_g): ∂L/∂a_t = σ'(a_t) ⟨h_t, r_t⟩, so ∂L/∂w_g = Σ_t ∂L/∂a_t φ_t.
  - Through φ the gradient also reaches r (the gate's r input) and p̄ (the gate's p̄ input, as today).
  - **values**: ∂L/∂r_t = g_t h_t + (∂L/∂a_t) w_g^{(r)}, and

        ∂L/∂v_j = Σ_{t: j ∈ J_t} A_t w_tj f_j ∂L/∂r_t        (the compact buffer, segment-summed per slot in after_step)

- **No gradient**: the keys, the kernel weights w, A, h₀, the frame (μ₀, σ₀, rotations, codebooks), strengths, clocks
  and maturity. All of them are set by the rules of §2.3 / §4 / §5 / §6 / §7.
  - So ∂L/∂p̂ = 0 on the address path: the thought is not constrained, and there is no stale-gradient path into the
    keys (plan 23 §1.5).
- **Parameters**: W_o 98K + gate 328 + W_ms 25K (+ the private memory if kept). The slot store is buffers.

### 8.3 Value update (compact row-Adagrad, once per step)
1. The ids of both micro-batches: torch.unique(return_inverse) (≤ 65K ids).
2. index_add_ into (n_u, 256).
3. The lazy fold: v ← f v.
4. The discounted S of §6, then dV = −lr g/√S, clipped to 0.05 (today's MemOpt values: lr 0.01, S₀ auto =
   100 × median).
5. index_copy back.

≈ 22 ops. It replaces today's G-table path, and it also removes the hot-row atomics into a 1M-row table (plan 13:
120-180 ms).

### 8.4 Launch budget per step (config F; planned counts, to be verified with TorchDispatchMode as in plan 23 §7.2)
| part | ops per micro-batch | per step |
|---|---|---|
| lookup (§2.1) | 36 | 72 |
| read fwd: MemRead (embedding_bag), A·f weights, gate features / sigmoid, u, W_o, inject add, slot write, Probe | 14 | 28 |
| checkpoint recompute of the read part (the lookup cached) | 14 | 28 |
| backward: compact value grad (3), Probe record (2), gate / W_o / W_ms backward (≈ 10), stashes (3) | 18 | 36 |
| **training-pass total** | 82 | **164** |
| after_step: confirmation select (12) + re-encode (18) | | 30 |
| after_step: surprise select (6) + spawn / evict / init (24) | | 30 |
| after_step: re-filing of all moved slots (cells 6, collisions 16) | | 22 |
| after_step: compact Adagrad (§8.3) | | 22 |
| after_step: lazy strengths for eviction (3) + h₀ / σ̂ / v̄ EMAs (6) | | 9 |
| transport: probe lookup (36) + kernel / robust weights / index_add (12) + scale / cap / apply (8) | | 56 |
| **memory ops per step** | | **≈ 333** |
| + the transport probe forward: 1 Core pass of 2 sequences, no grad | | ≈ 150-300 (count it) |

- At 35 µs per launch (plan 23's assumption): ≈ 12 ms + the probe ≈ 5-10 ms.
- The work:
  - the key gathers, 2 × 268 MB × (write + read) ≈ 1.1 GB;
  - the value bags 66 MB;
  - Adagrad ≈ 70 MB;
  - the centroid GEMMs ≈ 4.3 GFLOP;
  - W_o fwd + bwd + recompute ≈ 2 × 6.4 GFLOP.
  - At 70% of 896 GB/s and ≈ 50 TFLOP/s effective: ≈ 2-3 ms + ≈ 0.5 ms.
- **Total ≈ 22-30 ms per step**, vs today's memory ≈ 900 ms (plan 13).

### 8.5 Configurations vs the speed rule (step = 16,384 tokens; 10k ⇒ ≤ 1.64 s; 20k ⇒ ≤ 0.82 s)
| config | contents | memory ms / step (est.) | VRAM of the store | expected tok/s |
|---|---|---|---|---|
| **F (default)** | 1 lookup at pass 1, p̄ 64-d, T = 4, 512², cap 8, kc 4 (128 cand.), k = 4, 1M slots, W_b 128 / mb, transport 2 seqs, M = 8, no rows, no private | 22-30 | 1.37 GB + 0.5 GB temps | ≈ 20k without ConceptHier; **≈ 16-18k with it** (ConceptHier + recompute ≈ 160-190 ms, plans 13 / 23) |
| F-lite | F without transport | 12-18 | same | +1-2% |
| **Q (≥ 10k)** | F + a second family at pass 3 (1M, d_v 192), kc 8 (256 cand.), family 1 at 2M slots, transport 4 seqs to pass 3 | 60-80 | 2.7 + 1.1 GB + 0.9 GB temps | ≈ 14-16k with ConceptHier |
| (today) | PK rows 4 × 3 heads + coarse + nbr + private, read every 2 passes | ≈ 900 | ≈ 2.6 GB | 8-10k |

- VRAM: today's peak of 20.4 GB includes the row pool + G (≈ 2.2 GB) and the private store.
  - F: ≈ −0.5 to −1 GB net.
  - Q: ≈ +1.5-2 GB (≈ 22.4 GB: tight; Q uses d_v 192 for its second family for that reason).

---------------------------------------------------------------------------------------------------------------------

## 9. A training signal that makes memory necessary (generate_world.py)
**What the current mix does** (`X:\Datasets\wiki_world_mix`, meta.json):
- 20% world text; 5,000 people × 5 relations + 600 geo facts (25,600 facts).
- Documents of 5-15 facts or mini-stories; 10 training templates per relation, 2 held-out templates only in val
  (the eval is a cross-paraphrase recall).
- Late people: ≈ 233 exposures per fact over 400M tokens (≈ 114 by the 12.2K-step run's 200M), i.e. **one re-mention
  per ≈ 105 steps**, always in a different document and usually another template.
- Early people (1,000): ≈ 19 exposures in the first 40M tokens (one per ≈ 128 steps, until step ≈ 2,441), then never
  again.

**Is memory necessary on it?** Mostly not.
- 114 spaced exposures are enough for parametric memorisation by the weights.
- The early group tests retention after 9.8K steps, where a key memory needs transport (§7.5) and the weights need
  no forgetting.
- The memory's real niche (few exposures, far apart, beyond the context window) is absent.

**Suggested additions** (a v2 mix, generator flags; the existing groups stay as they are for comparability):
1. **Rare group** (`--rare-people 2000`): each person's facts appear k ∈ {1, 2, 4, 8} times. The spacing between
   mentions is drawn per person from {10, 100, 1,000, 3,000} steps (in tokens: × 16,384).
   - Cloze items per (k, spacing) bucket.
   - This gives the recall curve vs the drift interval directly, in the memory's niche: too few exposures for the
     weights, recurring within or beyond the re-encode horizon.
2. **Introduce-then-query** documents: a fact first appears in document A. A different template of the same fact
   reappears Δ ∈ {10, 100, 1,000} steps later, with the value at the end of a sentence (the cue position precedes it).
   - The second mention is surprising for the weights (seen once) and predictable for the memory.
   - It is the cleanest per-token knockout signal (memory on vs `--ko-recall`).
3. **Keep the early cutoff** (the non-recurring retention test = the transport test).
4. Log the exposure schedule (exposures.npz already) with the step of each mention, so that eval_world can bucket the
   recall by "steps since the last mention".

The spaced repeats are needed: the 105-step spacing of the current mix tests only the easy regime of §0. The 1K-3K
spacings are where re-encoding alone breaks and the curve says whether transport is real.

---------------------------------------------------------------------------------------------------------------------

## 10. Test plan, calibrated to the REAL drift (all before implementation, CPU unless stated)

### 10.1 T0: real-thought checks (extend `speed_wip/splat/realcheck`; captures at 3K / 6K now, 9K / 10K when available)
| test | what | pass / fail |
|---|---|---|
| **T0a cross-document same-fact retrieval** (THE first gate) | Label the world cue positions with (person, relation) (cue_positions() + the name before the cue). Gallery = all positions of the same checkpoint. Query a cue; the hit = its nearest gallery cue from ANOTHER document has the same (person, relation); also a same-relation-only hit; per pass 1 / 3, p̄ raw / pw / whitened-32. | **PASS: same-fact R@1 ≥ 0.4 (R@16 ≥ 0.7) at pass 1, p̄.** FAIL → the keys only work for verbatim repeats. Stop and redesign the view (e.g. a learned contrastive view, which is a different plan). |
| T0b frame | the interpolation table of §0 for raw-64 and pw-0.5-64 p̄ (pass 1) | choose the frame with the best R@1 at t = 0.33 / 0.58; prefer raw if within 0.02 |
| **T0c transport field** (the 9K → 10K gate, run separately) | §7.5 items 1-3 | §7.5 PASS / FAIL |
| T0d routing-noise jitter | the same checkpoint, two forwards with route noise 0.3: self-recall and ‖Δ_noise‖ / ‖Δ_1K‖ | self-recall ≥ 0.95; jitter ≤ 0.3 × the 1K drift (else: probes and reads run with eval routing) |
| T0e index on real keys | 1M-key gallery = the captured views ×(30 nearby jittered copies) or several checkpoints' captures; the T = 4 index candidate-recall vs brute force; pointer coverage under the real occupancy skew | candidate-recall ≥ 0.85 at 128 candidates; coverage ≥ 0.95 |
| T0f probe variants | the field R² from real other-document probes vs perturbed-token probes vs random-token probes | perturbed probes enabled only if they raise R² in thin-support regions by ≥ 0.05 without lowering it elsewhere |
| T0g cross-pass | a pass-1 key vs a pass-3 query self-recall; same-fact retrieval at pass 3 vs pass 1 | for config Q only: a second family is worth it if pass 3 improves same-fact R@1 by ≥ 0.05 |

### 10.2 T1: the synthetic toy (extend `consol_toy.py`, CPU)
**World** (as the data):
- a late group re-mentioned every ≈ 105 steps;
- an early group ×19 every ≈ 128 steps, then silent for 9.8K steps;
- a rare group (k ∈ {1, 2, 4, 8} × spacing {10, 100, 1K, 3K});
- each mention has a phrasing jitter in view space, calibrated to T0a: the same-fact view spread vs the NN spacing.

**Drift.** A warp process (smooth random fields) calibrated jointly to:
- (i) the real interpolation curves of §0 (p̄ and levels at t = 0.2 / 0.33 / 0.58);
- (ii) the field length scale ℓ, so that the toy's leave-document-out kNN R² equals T0c's measured R²;
- (iii) both time mappings (directed / diffusive) until 9K → 10K picks one; the LR decay as drift ×0.3 for the last
  20%.

**Confirmation signal**: c = μ·1[the read decodes the slot's own fact] + N(0, 1), with SNR μ ∈ {0.3, 1, 3}.

**Policies**:
1. no renewal;
2. re-encode (§4);
3. re-encode + recall-pair transport (§7.1 b);
4. + probe transport (§7.1 a);
5. + noise probes;
6. re-encode with η = 1;
7. confidence-only re-encode (plan 24's b_un, as the hijack reference).

**Metrics**:
- recall of the early / late / rare (by spacing) facts at the end and over time;
- one-shot recall vs age;
- **wrong re-filing rate** (re-encodes that moved a key toward a context of a different fact);
- confident-wrong reads;
- key collisions (pairs within γ_d h₀²);
- keys per fact, capacity use by maturity, slots touched by transport per step;
- the transport error ‖k − x_owner‖ over time.

**Thresholds (pass before implementation)**:

| metric | threshold |
|---|---|
| late facts (re-mentioned every ≈ 105 steps), re-encode only, SNR 1, both mappings | recall ≥ 0.90 |
| rare facts at spacing 1,000, re-encode only | ≥ 0.5 (directed) / reported (diffusive) |
| wrong re-filing at SNR 1 / SNR 0.3 | ≤ 1% / ≤ 3% of re-encodes |
| confident-wrong reads | ≤ 1% |
| key collisions | ≤ 0.5% of live keys |
| early facts after 9.8K silent steps, with transport | ≥ 0.5 AND ≥ +0.3 over re-encode only (else transport stays off) |
| re-encode η = 0.5 vs η = 1 | within 0.03 on the late facts (else default η = 1) |

### 10.3 T2: GPU bench (only when the GPU is free; extend `speed_wip/splat/bench_gpu.py`)
- Config F and Q, B = 16, T = 512, 1M / 2M slots: the measured ops per step (TorchDispatchMode), ms per step of the
  memory path and of the 1-pass probe forward, and the peak VRAM.
- **PASS**: F memory ≤ 40 ms/step, probe ≤ 20 ms/step, ≤ 400 ops/step (excl. the probe), store + temps ≤ 2 GB;
  Q ≤ 100 ms/step.

### 10.4 T3: the 12K run (after T0a, T1, T2 pass; owner's go)
- Arms: the control with no memory, F with transport, F without transport (`--rm-transport 0`); optionally today's
  rows.
- Metrics:
  - eval_world early_retention / single-fact cloze;
  - the rare-group curve (if the v2 mix exists);
  - the knockout delta (memory on vs `--ko-recall`) by "steps since the last mention";
  - tok/s.
- **Success**:
  - tok/s ≥ 15k with ConceptHier;
  - the knockout delta > 0 on late facts;
  - early-fact recall F > F-no-transport by ≥ 0.1 (if T0c passed).

---------------------------------------------------------------------------------------------------------------------

## 11. Integration with the existing brain

### 11.1 What changes
- **ConceptHier stays** (its losses, its fb / inject, its uncertainty features). It is just no longer the source of
  the memory's view.
- **The row memory** (shared PK pool, coarse pool, neighbour writes, the plan-21 life cycle for rows, probes) is
  disabled under `--rm 1` and kept intact as the ablation (`--rm 0` = today, bit-identical).
  - Running both (`--rm 1 --mem-shared 512`) is allowed for a comparison: their M's are summed.
- **The private expert memory**: off in F (`--mem-private 0`); it can be kept as an ablation (its cost is separate).
- **Plan 22's access gate**: off (`--mem-gate 0`). The content gate of §2.2 decides use.

### 11.2 Code placement
- **`recall_memory.py`** (new):
  - `RecallMemory(nn.Module)`: buffers, the frame fit, `lookup()`, `read()` (MemRead, reused from memory.py with the
    compact-gradient option), `Probe` (the autograd identity that records c);
  - `stash_*()`, `after_step()` (confirm / re-encode / spawn / evict / Adagrad / fade), `transport()`, `refit_index()`;
  - extra state (host clocks, EMAs, the ring), `usage_stats()`.
- **`loop_brain.py`**:
  - in `_hier_step.block()`, when `self.rmem` is set and k == rm_pass: p̄ (already computed) → `rmem.lookup`
    (cached in `cache`) → `rmem.read` → M. The destinations (inject + `_slot_write`) and the return signature stay as
    today.
  - in `run()`: the write candidates' σ_t (the final-pass CE, detached) are handed to `rmem` after the read-outs, and
    the stash of the first n_p sequences' ids.
- **The trainer**:
  - flags;
  - `rmem.after_step()` after BrainOpt (where MemOpt's after_step is now);
  - `rmem.transport(model)` right after it: the 1-pass no-grad forward helper `model.view_pass1(ids)` = embedding,
    init_state, one Core call, n_s, mean;
  - the parameter groups (W_o, gate, W_ms: AdamW as today's "mem.gates.");
  - logging;
  - resume (all buffers + host state).

### 11.3 Flags (defaults = config F)
- **On / off**: `--rm 0|1` (default 0 until T3).
- **Store and view**:
  - `--rm-slots 1048576`, `--rm-dv 256`;
  - `--rm-view pbar|level:l`, `--rm-pass 0`, `--rm-frame raw|pw:0.5|white:32`, `--rm-fit-step 300`.
- **Index and read**:
  - `--rm-tables 4`, `--rm-cells 512`, `--rm-cap 8`, `--rm-probe 4,4` (m, kc);
  - `--rm-k 4`, `--rm-tau 0.1`, `--rm-refit 1000`.
- **Writes and fading**:
  - `--rm-write 128` (per micro-batch), `--rm-write-start 300`, `--rm-vinit grad|tok`;
  - `--rm-fade 1e-3,1e-4` (λ_α, λ_v), `--rm-mature 3,64`, `--rm-beta 0.99`, `--rm-lr 0.01`, `--rm-clip 0.05`.
- **Re-encode**:
  - `--rm-reenc 256` (per micro-batch, 0 = off), `--rm-eta 0.5`, `--rm-eta-young 0`;
  - `--rm-conf 1.0,0.7,1.15,0.3` (κ_c, w₁, margin, A_min), `--rm-mindist 0.25`.
- **Transport**:
  - `--rm-transport 2,8` (n_p sequences, M; 0 = off), `--rm-tr-k 32`, `--rm-tr-h 2`, `--rm-tr-w0 1`;
  - `--rm-tr-cap 0.25`, `--rm-tr-recall 0.3`, `--rm-tr-affine 0`, `--rm-probe-noise 0`.
- **Misc**: `--rm-reinject 0`, `--rm-fade-decay 0`, `--ko-recall` (the knockout probe).

### 11.4 Logging (every eval)
- **Store**: live slots by maturity; writes / re-encodes / disconfirms / evictions per step; pointer coverage;
  bucket overflow; candidate-recall vs brute force.
- **Reads**: h₀; the median d₁ and margin; the A histogram; P(c > 0 ∣ A bucket); the confirmed fraction.
- **Transport**: slots transported per step; the median ‖Δk‖ / h₀; the probe-pair R² (held-out half of the probes
  predicted from the other half: an online version of T0c).
- **Probe-set self-recall**: fixed val windows; their keys written once at a reference step; recall of their own
  slots every 500 steps, with and without transport, i.e. the online stability law.

### 11.5 Tests (CPU, `test_recall_memory.py`)
1. `--rm 0`: bit-identical outputs and gradients vs today (the existing test_memory / test_gate pass unchanged).
2. Lookup = brute force when the candidates cover all slots (small N_s, cap large); the top-4 and weights are equal.
3. Index invariants after random spawn / evict / move / refit sequences: every pointer ↔ back-pointer is consistent;
   no slot has duplicate pointers in one table; the collision ranks are correct when several inserts hit one bucket.
4. The lazy fade / discount equals the eager per-step version (fp64 reference) over 1,000 random steps.
5. The compact Adagrad equals today's dense `_row_adagrad` on the same gradients.
6. Probe: c_t = −⟨∂L/∂u, u⟩ matches a finite difference L(u) − L(0) to first order on a tiny model; it is recorded
   exactly once under checkpointing.
7. Re-encode: order independence (permuted confirmations → identical keys); the margin / min-dist / cap rules; the
   separation bound of §4.4 holds on random cases.
8. Surprise writes: the static budget; the 8-token suppression; the value init direction = −∂L/∂r.
9. Transport: a zero field → no move; a constant field c → every supported key moves by c/M (within the w₀
   shrinkage); the cap; the robust weights drop the injected outliers.
10. Checkpoint recompute: the lookup and the write candidates cached on the first run; the same outputs as without
    checkpointing.
11. Resume: save / load mid-run → identical next steps (buffers, host clocks, EMAs, transport ring).
12. No host sync per step in the training path (CUDA-sync counter in a GPU test; counts via static budgets).

---------------------------------------------------------------------------------------------------------------------

## 12. Risks and open questions
1. **Cross-document similarity (T0a).** Re-encoding and the eval both assume that a new phrasing of the same fact
   lands near the stored key.
   - The real check measured only self-recall (the same position, the same text).
   - If T0a fails, no rule-based key memory on p̄ will generalise across documents. Only verbatim / same-document
     reuse remains.
2. **The SNR of c.** A per-position gradient inner product may be too noisy, so that p_o ≈ p_f.
   - The bias bound (§4.4) then degrades toward "confidence-only" re-encoding (plan 24: hijacks).
   - Mitigations: a raised κ_c, or confirmations required on 2 reads before a move (a counter per slot).
   - Measured online (§11.4).
3. **Transport accuracy (§7.5).** Under diffusive drift it needs R² ≳ 0.94 for the early facts. That is a demanding
   number for a 64-d field estimated from ≈ 1K probes per step.
   - Expect partial success at best, unless the 9K → 10K test shows directed drift or a high R².
   - Transport is the experimental part; re-encoding is the core.
4. **Routing noise and births.** Noise 0.3 jitters the views; expert births change the thoughts discontinuously (the
   drift is bursty).
   - T0d measures the jitter.
   - Births could trigger an extra probe step (`--rm-tr-on-birth`, open).
5. **Early training.** The thoughts reorganise most before about step 1K (plan 16: PR 87 → 3-4). Writes start at 300
   with the frame fitted at 300. If the early keys are useless, a later write start (500-1,000) is the knob; the
   toy cannot say.
6. **The index under the anisotropy** (p̄ is low-rank): hot cells overflow, and the pointer coverage drops. The
   rotations + k-means codebooks + refit mitigate; T0e checks.
7. **The value target at the start** (W_o = 0 ⇒ ∂L/∂r = 0): random init + the gradient write. If the gates never
   open, the memory is never used ("dead memory"); the gate's bias init −2 is the same as today's rows, which did
   open.
8. **Interference within a slot.** Readers of different facts that pass the margin test train the same value by
   gradient (row-like superposition, but local). A small A for far readers limits it, and the disconfirm-spawn path
   creates a competitor. A slot split rule (plan 23 §4.3 split) is left for later.
9. **Speed claims are estimates.** The op counts are planned, not measured; the probe-forward launch count is
   unknown. The 20k target also needs ConceptHier to be cheaper (outside this plan).
10. **The data** (§9). On the current mix, the memory has little to do beyond what the weights learn from 114
    exposures. A null result there would not be evidence against the memory; the v2 mix is needed for a decisive
    test.
11. **Open**:
    - a per-slot "key EMA" vs the hard η (T1);
    - whether recall pairs or probe pairs carry the field (T1 / T0c);
    - whether a second read family (Q) earns its slots (T0g);
    - whether the value fade λ_v helps at all (T3 ablation).

---------------------------------------------------------------------------------------------------------------------

## 13. Recommended default config and implementation size

**Order of work**:
1. T0a and T0b now, on the 3K / 6K captures (the capture script exists; the fact labels are new);
2. T0c on 9K → 10K (the separate test = the transport gate);
3. T1 toy;
4. T2 bench when the GPU is free;
5. then implement;
6. T3 run only with the owner's go.

**Default (config F)**:

    --rm 1 --mem-shared 0 --mem-private 0 --mem-gate 0
    --rm-slots 1048576 --rm-dv 256 --rm-view pbar --rm-pass 0 --rm-frame raw --rm-fit-step 300
    --rm-tables 4 --rm-cells 512 --rm-cap 8 --rm-probe 4,4 --rm-k 4 --rm-tau 0.1 --rm-refit 1000
    --rm-write 128 --rm-write-start 300 --rm-vinit grad --rm-lr 0.01 --rm-clip 0.05 --rm-beta 0.99
    --rm-fade 1e-3,1e-4 --rm-mature 3,64
    --rm-reenc 256 --rm-eta 0.5 --rm-conf 1.0,0.7,1.15,0.3 --rm-mindist 0.25
    --rm-transport 2,8 --rm-tr-k 32 --rm-tr-h 2 --rm-tr-w0 1 --rm-tr-cap 0.25 --rm-tr-recall 0.3   (0 if T0c fails)

- Expected cost: ≈ 22-30 ms/step memory + probe (≈ 3%); 1.4 GB store + ≈ 0.5 GB temps.
- ≈ 20k tok/s without ConceptHier, ≈ 16-18k with it. ≥ 10k is met with a large margin.

**Implementation size (estimate)**:

| part | lines |
|---|---|
| `recall_memory.py` (store, frame, index + maintenance, lookup, read / Probe, after_step rules, compact Adagrad, transport, state, stats) | ≈ 950 |
| `loop_brain.py` wiring (_hier_step view / read, run() stashes, view_pass1) | ≈ 120 |
| trainer (flags, after_step / transport calls, param groups, logging, resume, ko-recall) | ≈ 180 |
| `test_recall_memory.py` (§11.5) | ≈ 450 |
| realcheck extensions (T0a fact labels, T0b / T0d / T0e / T0g, the T0c field metrics) | ≈ 200 |
| toy extension (T1) + bench (T2) | ≈ 250 |
| generate_world v2 options (§9, optional) | ≈ 120 |
| **total** | **≈ 2,100-2,300** |
