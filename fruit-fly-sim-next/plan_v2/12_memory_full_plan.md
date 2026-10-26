# Memory system: full plan with equations (hierarchical concepts + shared product-key pool + private expert memory)
2026-10-23, revision 2 (after three independent reviews A/B/C + coordinator notes; the change log is section 11).
Builds on 11_memory_plan.md and the code base: loop_brain.py (LoopBrain / Core), spiking_lm.py (RoutedMLP, Spike),
brain_opt.py (BrainOpt), neurogenesis.py, train_lm.py. Plan only; every new part is behind flags, default off.

## 0. Notation (current v4d config)
- B: micro-batch (16). T: sequence length (512). d: width (384). V: vocab (16,384 for wiki16k).
- K: passes (training 2-10, 5% 12-16; eval 6). bptt = 6 (gradient through the last 6 passes).
- Workspace s_k ∈ R^{B×T×16×64} after pass k. n_s = RMSNorm(64) per slot.
- Pooled thought p_k = flatten(n_s(s_k)) ∈ R^{B×T×1024}; slot mean p̄_k = mean_slots(n_s(s_k)) ∈ R^{B×T×64}.
- x_k ∈ R^{B×T×d}: the thought after pass k. normf: the final RMSNorm. Experts E ≤ 768, top-k_e = 6, capacity C.
- Pass k's core input: u_k = n_u(x_{k-1} + n_e(e) + w_ro(read(s_{k-1})) + inject_k) (Core.forward(..., inject=...)).
- sg(·) = stop-gradient. idx = the INPUT tokens (not the targets).
- "Addressing" = everything that turns a thought into row indices (EMA norm, W_q, sub-keys, top-m, top-k). It always
  runs in fp32 under `torch.autocast(enabled=False)` with TF32 off for those matmuls, so bf16/TF32 can't flip ties.

## 1. Hierarchical concepts (flag --concepts 16,32,64,128, replaces --think-target latent)
Levels l = 1..L (L = 4), horizons n_l = 16, 32, 64, 128, loss weights w_l = 1, 1, 0.5, 0.5.

### 1.1 Targets (no gradient, computed once per forward after the last pass)
- z = sg(normf(x_K)) ∈ R^{B×T×d} (fp32).
- Prefix sums: P[:, 0] = 0, P[:, j+1] = P[:, j] + z[:, j]  (B×(T+1)×d).
- Next document boundary: e_j = j if idx[j] = EOT else T; rc = reverse cumulative min of e; b(t) = rc[t+1]
  (the SHIFTED suffix-min, i.e. the smallest j > t with idx[j] = EOT; b(T-1) = T).
- Window end: e_l(t) = min(t + n_l, T-1, b(t) - 1). Count c_l(t) = e_l(t) - t.
- Target: z*_l(t) = (P[:, e_l(t)+1] - P[:, t+1]) / max(c_l(t), 1).
- Valid mask: v_l(t) = [c_l(t) ≥ n_l / 4] AND idx[t] ≠ EOT (an EOT position's future is the next document).
- Reported before the first run: mean of v_l per level on the training data (Wikipedia and TinyStories).
- Checked by reviewer A against a brute-force loop (max error 8e-16).

### 1.2 Level heads (every pass k, inside the grad window when k ≥ K - bptt)
- Shared trunk: h_k = GELU(W_t p_k + b_t), W_t ∈ R^{256×1024}.
- Level l: [μ_l ; ν̃_l] = W_l [h_k ; μ_{l-1}] + b_l, W_1 ∈ R^{2d×256}, W_l ∈ R^{2d×(256+d)} for l > 1.
  The lower level's μ is passed WITH gradient (no sg), so the hierarchy is trained jointly.
- Soft clamp of the log-variance: ν_l = -1 + 5 · tanh(ν̃_l / 5) ∈ (-6, 4) (no dead gradient at the bounds).
- The level heads + translators of a pass are one activation-checkpointed block (section 7).

### 1.3 Loss (per pass k in the grad window, then averaged over those passes)
- Gaussian NLL: N_l = Σ_{b,t} v_l(b,t) Σ_i ½ (ν_l + (z*_l - μ_l)² e^{-ν_l}) / (d · Σ_{b,t} v_l(b,t)).
- Variance hinge (anti-collapse) with MASKED moments (no nonzero(), no host sync):
  mean_v(x) = Σ v·x / Σ v, std_v(x) = sqrt(mean_v(x²) - mean_v(x)² + 1e-6), over (b, t) per dimension i;
  H_l = mean_i relu(sg(std_v(z*_l,i)) - std_v(μ_l,i)).
- L_concept = think_w · Σ_l w_l (N_l + H_l), think_w = 0.1.

### 1.4 Uncertainty, feedback, usefulness
- u_l,k = [ (mean_i ν_l + 6) / 10 , log1p( ||μ_l,k - μ_l,k-1|| / max(||μ_l,k||, 1e-6) ) ] (detached).
  u_k = concat_l u_l,k ∈ R^{2L}.
- Feedback into the workspace (zero-init): s_k ← s_k + (W_fb u_k)[..., None, :], W_fb ∈ R^{64×2L}.
- Usefulness (zero-init, rank 64 by default): inject_{k+1} += Σ_l B_l A_l μ_l,k, A_l ∈ R^{64×d}, B_l ∈ R^{d×64}
  (B_l = 0 at init). Flag --concept-inject-rank (0 = full d×d).
- Stop head: its features gain u_k (2L numbers). The halt fallback input is 2L wide as well.
- All new modules are constructed AFTER the existing ones (so the old RNG stream and the flags-off model are untouched).

## 2. Translators (one per level and head; flag --mem-shared)
H = 3 heads per level. Head (l, h):
- Input a = [μ_l,k ; p̄_k] ∈ R^{448}.
- Normalisation with EMA statistics (infancy only, updated ONCE per optimizer step from the last micro-batch's
  grad-window passes, not once per pass): m ← 0.999 m + 0.001 mean(sg(a)), v ← 0.999 v + 0.001 mean((sg(a) - m)²).
  â = (a - m) / sqrt(v + 1e-5).
- Key: q = normalize(W_q â), W_q ∈ R^{128×448}. Halves q¹ = normalize(q[:64]), q² = normalize(q[64:]).
- Sub-keys: K¹, K² ∈ R^{S×64}, S = 512, rows normalised in the forward.
- Optional (--mem-addr-sg, default ON after the freeze): the addressing input is sg(a) after the freeze, so the
  memory's ω-gradient no longer pulls the thought towards/away from addresses. The READ VALUE still sends gradient
  into the thought through the gate and W_o. Thoughts are not constrained (no commitment loss), per Eitan.

### 2.1 Product-key scoring (Lample et al. 2019)
- s¹ = q¹ K¹ᵀ ∈ R^S, s² = q² K²ᵀ ∈ R^S (cosines in [-1, 1]).
- Top-m (m = 8) per half (I¹, I²) → 64 candidate pairs (i, j), σ_ij = s¹_i + s²_j.
- Top-k_m (k_m = 6) pairs. Exact vs brute force over all S² pairs whenever k_m ≤ m (reviewer A: 300/300 trials).
- Deterministic tie-break (lowest flat index wins): sort the 64 candidates by r_ij = i·S + j ascending, then a
  STABLE descending sort by σ. Same composite rule inside each half's top-m.
- Address r_ij = i·S + j ∈ [0, S²). Row = off_l + r_ij, off_l = (l - 1) · S².
  The 3 heads of a level share that level's region: they are a multi-hash into one table (not separate rows).
- EMA copy θ̄ = (W̄_q, K̄¹, K̄²) (infancy): θ̄ ← 0.999 θ̄ + 0.001 θ after each optimizer step.
  - INDICES (I¹, I², the top-k_m pairs) are chosen with θ̄, under no_grad, outside the checkpointed block.
  - WEIGHTS on those same pairs use the ONLINE θ: ω_ij = softmax_{top}( σ_ij(θ) / τ ), τ = 0.03 (flag --mem-tau).
  - The gradient reaches W_q, K¹, K² only through ω; addresses move only as fast as θ̄.
- After the freeze: θ ← θ̄ once (this changes ω once at that step; logged), then frozen (requires_grad = False,
  no EMA, no norm updates). Addressing is then a pure function of a.

### 2.2 Infancy and freeze schedule
- Infancy: step 0 → t_f. t_f = the first step ≥ mem_freeze · steps (default 0.6) at which neuro.done() is true
  (births can arrive after t1: "the next event catches up", and ramp births too).
- Sub-key balancing, every 250 optimizer steps during infancy, per half codebook, with a DEDICATED RNG:
  - usage EMA U_i (share of top-m selections; empty capacity slots excluded);
  - busy (U_i > 4 · mean U) → split into the least-used key j: K_j ← normalize(K_i + ε), K_i ← normalize(K_i - ε),
    ε ~ N(0, 0.02). The band of rows addressed through j gets a COPY of the band through i (V rows and S
    accumulators; half 1 = rows j·S.., half 2 = the strided column j), so the contexts handed to j find the same
    content. Both bands' optimizer state is reset (MemOpt rows) and the EMA copy gets the same edit.
  - dead (U_i < 0.05 · mean U, not used by a split) → K_i ← normalize(a recent q-half + N(0, 0.05)); its band's
    S is reset to S₀.
- Freeze at t_f: θ ← θ̄; freeze θ, the norm statistics, the sub-keys and the private keys (section 4); record the
  probe addresses R⁰ (section 6). The freeze flag is saved in checkpoints and RE-APPLIED on reload; grow_experts
  re-applies requires_grad = False if called after the freeze (it never should be).

### 2.3 Margin loss (infancy only, address space only)
- What must stay stable is the read SET, so the gap is at the set boundary:
  γ = σ_(k_m) - σ_(k_m+1) (6th vs 7th pair; the 7th is always among the 64 candidates since k_m + 1 ≤ m), and per
  half γ¹ = s¹_(m) - s¹_(m+1), γ² likewise. Ranks come from θ̄, scores from the online θ.
- Computed in a SEPARATE pass on sg(a): L_margin = λ_m · mean[relu(δ - γ) + ½ relu(δ - γ¹) + ½ relu(δ - γ²)],
  δ = 0.01, λ_m = 0.01 (flag --mem-margin λ_m,δ). Natural gaps at init: median 0.0035 (A), so δ is a mild push.
  Logged: the active share of the hinge; if it collapses sub-key usage, the balancing log shows it.
- After the freeze: γ, γ¹, γ² are only logged.

## 3. Shared pool (flag --mem-shared; rows = L · S² = 4 · 262,144 = 1,048,576)
- Table V_sh ∈ R^{1,048,576 × 256}, fp32, N(0, 0.02). A plain tensor with a manual grad buffer, NOT an
  nn.Parameter (so BrainOpt, clip_grad_norm_, carry_brain_state and model.zero_grad never see it).
- Read, one fused call per pass for all 12 heads: a custom autograd function MemRead(V, rows, ω) =
  embedding_bag(V, rows, per_sample_weights = ω, mode "sum") whose backward index_add_'s into ONE preallocated
  grad buffer G_sh (no fresh 1.07 GB allocation per call) and returns ∂/∂ω.
- Gate: g_lh = sigmoid(w_g,lh · [p̄_k ; r_lh] + b_g,lh), b_g init -2, w_g ∈ R^{64+256}.
- Memory output: M_k = Σ_l W_o,l ( Σ_h g_lh r_lh ), W_o,l ∈ R^{d×256}, zero-init.
- Timing, every pass: a read before pass 1 (from s_0) and after each pass k (from s_k), feeding pass k+1.
  The pass-0 read is token-keyed (s_0 depends only on the token); it is kept but reported separately
  (flag --mem-read0, default on).
- Destinations: (a) inject_{k+1} += M_k; (b) slot 15: s_k = cat(s_k[..., :15, :], s_k[..., 15:, :] + W_ms M_k)
  (out-of-place), W_ms ∈ R^{64×d}, zero-init. Slot 15 is shared with the normal workspace dynamics, not dedicated.

### 3.1 Write = sparse row-Adagrad with a prior (MemOpt; V_sh and V_e)
- G = the summed table gradient of the optimizer step (accumulated micro-batches; the loss is already / accum).
- Active rows A = {r : G_r ≠ 0}. For r ∈ A:
  - S_r ← S_r + mean_i G_ri², with S_r initialised to S₀ (not 0);
  - ΔV_r = -η · G_r / sqrt(S_r); ΔV_r is clipped to ||ΔV_r|| ≤ c_Δ (clip the UPDATE, not G); V_r += ΔV_r.
- Why S₀: with S₀ = 0 the first write of any row has norm η√d_V whatever |G| is (Adagrad is scale-invariant), so a
  row read with weight 1e-3 moves as hard as the main row (A, B, C). With S₀ the step is ≈ η·G/√S₀ for small G,
  i.e. proportional to the gradient (and so to ω).
- S₀: --mem-s0 (absolute). Default "auto": at the first step with writes, S₀ = 100 × median_{r∈A} mean_i G_ri²
  (≈ 100 typical prior writes), then fixed. Same rule for the private table (its own S₀).
- η = 0.01 (--mem-lr), c_Δ = 0.05 (--mem-clip; the init row norm is 0.02·√256 = 0.32). Writes start at
  --mem-write-start (default 200 steps), while W_o is still tiny.
- Rows not in A: bit-identical (no update, no decay).
- Honest note (reviewers): with ~16K tokens × 6 passes × 3 heads × 6 rows ≈ 1.8M hits per level per step on
  262K rows, almost every row is written every step. "Consolidation" is therefore mostly a global ~1/√t LR decay,
  and rows behave as prototypes (cluster centres), not single episodes. The dense grad buffer is the right storage
  (a sparse COO gradient at this density would be larger: ≈ 3.6 GB).

## 4. Private expert memory (flag --mem-private 512)
- Preallocated for the MAX expert count (768) at construction, so births never reallocate:
  keys P ∈ R^{768×512×64} (normalised in the forward), EMA keys P̄, values V_e ∈ R^{768×512×256} (N(0, 0.02)),
  accumulators S_e ∈ R^{768×512}, gates b ∈ R^{768}. P and V_e are manual tensors handled by MemOpt
  (V_e row-Adagrad as 3.1; P = Adam, no weight decay, per-row state), not BrainOpt.
- Query for (token t, chosen expert e) at pass k: the level-1 head-1 key half q¹ (64-d). Indices from the EMA side
  (q̄¹ with P̄_e); weights from the online side (q¹ with P_e). Note: q¹ is shared with the shared pool's level-1
  head-1 address, so private and shared addresses are correlated (less redundancy than independent queries).
- Batched in the capacity layout: Q ∈ R^{E×C×64}; empty capacity slots are masked out of the read, the usage stats
  and the balancing.
- Scores Q P_eᵀ by bmm → R^{E×C×512}; top-4 (composite stable tie-break); ω = softmax(online scores / τ).
- Read: ρ = Σ ω V_e[e·512 + idx] via MemRead on the flattened table (embedding_bag, one grad buffer).
- Output (W_p applied ONCE after the combination; identical result, 6× cheaper):
  y_t += W_p ( Σ_{e chosen} gate_te · σ(b_e) · ρ_te ), W_p ∈ R^{d×256} (98K), zero-init, b_e = -2.
- Keys learn during infancy (gradient through ω, EMA picks indices, per-expert dead / busy balancing every 250 steps)
  and freeze at t_f.
- Copy-on-split (grow_experts, birth_mode split): the twin gets exact copies of P_e, P̄_e, V_e, S_e, b_e and the
  MemOpt rows. Ramp-mode children are not selected at birth, so they start from their own fresh rows.

## 5. Redundancy and repair
- Redundancy: 3 heads per level (multi-hash), 4 levels, private memories.
- Repair default = OFF (measure only): moved probes are counted per level / head / pass.
- With --mem-repair β (default 0.25 when on): after the freeze, at every eval, for a probe π whose address set moved
  and a NEW row r ∈ R(π) \ R⁰(π) whose write count is below --mem-repair-usage (default: the 10% quantile), copy
  the DROPPED row's value: V_r ← (1 - β) V_r + β V_{r_drop} (r_drop = the dropped row with the largest ω⁰). Busy
  rows are never overwritten. R⁰ is NEVER reset (always the freeze-time addresses). Repairs use probe set Π_A; the
  guard measures on a disjoint set Π_B (no Goodhart). Counts logged per level.

## 6. Drift probes and the stability guard
- Probe sets Π_A, Π_B: 32 fixed val sequences each × 32 positions (1,024 positions per set), drawn once with a
  dedicated RNG; run in FIXED batches, eval mode, K = 6, head-group offset 0.
- Recorded per pass: the addresses of every level / head, ω, the expert set and the private rows.
- Stability (after the freeze), against the freeze-time R⁰ AND against the last eval:
  - set stab = mean over (π, pass, l, h) of |R ∩ R⁰| / k_m;
  - ω-weighted stab_ω = mean Σ_{r ∈ R ∩ R⁰} ω_r (rows that matter);
  - per level and per pass; pass-0 reported separately.
- Input drift: mean ||a - a⁰|| / ||a⁰|| per level (the cause of address moves; the freeze can't stop it).
- Head-group offset overlap: once per eval, the probe addresses at a random training offset vs offset 0
  (reviewer A: training uses a random group offset per step, so a training write and an eval read of the same
  context can address different rows).
- Margins (γ, γ¹, γ²: mean and 5% quantile), collisions (distinct probes per row), row usage / write-count histogram,
  dead / busy sub-keys (infancy), gate means per level / pass / kind, private hit rate, routing drift (share of probe
  contexts with the same experts as at the freeze).
- Guard: stab_ω < 0.95 (--mem-guard) → "MEMORY DRIFT" in the log and metrics. Expectation (A, C measurements):
  3% input drift ≈ 0.93 set overlap, 10% ≈ 0.79. ≥ 0.99 is NOT expected; the guard is an alarm, not a promise.
- Knockouts at every loop report and at the end: val with shared reads = 0; with private reads = 0; with level l off.

## 7. Parameters, FLOPs, VRAM (final size: 36 heads, 768 experts, B = 16, T = 512, accum 2)
- New dense parameters (BrainOpt, AdamW for keys without WD):
  - concept trunk 0.26M + level heads 0.2M + 3 × 0.49M ≈ 1.9M; W_fb tiny; inject rank-64 4 × 49K = 0.2M;
  - translators 12 × 57K = 0.7M; sub-keys 12 × 65K = 0.8M (+ EMA copies); gates and W_o 4 × 98K, W_ms 25K, W_p 98K.
  - W_q and the sub-keys go to an AdamW group with weight_decay 0 (Muon's orthogonalised equal steps and decay on
    normalised keys are both wrong for keys).
- Tables (MemOpt, manual tensors): V_sh 1.07 GB + G_sh 1.07 GB + S 4 MB; V_e 0.40 GB + G 0.40 GB; P 0.10 GB +
  P̄ 0.10 GB + G 0.10 GB + Adam 0.20 GB. ≈ 3.5 GB.
- Activations with the fixes (indices under no_grad; level heads + translators + read + gate + W_o of each pass in
  one checkpointed block with the indices as inputs; one MemRead per pass; private via MemRead):
  ≈ 1.0-1.5 GB over 6 grad passes (without checkpointing reviewer A estimates 4.5-5.5 GB).
- Total ≈ +4.5-5 GB → ~18-18.5 GB allocated (v4e: 13.4 GB). Above the 16.5 GB target. The smoke step measures the
  real peak; fallbacks in order: shared S = 362 (4 × 131K rows; saves ≈ 1.1 GB), private 256 rows (≈ 0.4 GB),
  --readout-frac 0.5 (already exists), --mem-heads 2.
- FLOPs per token per pass (MACs, corrected):
  - level heads 1.93M; translators 12 × (448·128 + 2·512·64) ≈ 1.5M (+ during infancy the θ̄ path and the margin pass
    ≈ +2.1M); reads 18K; W_o 4 × 98K = 0.39M; inject rank-64 0.2M; private scores 6 × 512 × 64 ≈ 0.2M (+0.2M EMA side
    in infancy); W_p 0.1M.
  - ≈ 4.3M after the freeze, ≈ 6.5M in infancy, vs the current pass ≈ 5.3M → +80-120% per pass.
  - Expected step time 1.8-2.5× v4d's. The level heads + translators dominate, not W_in.

## 8. Flags (all default off / old behaviour)
- --concepts "16,32,64,128", --concept-w "1,1,0.5,0.5", --concept-inject-rank 64. Requires --slots.
- --mem-shared S (0 = off), --mem-heads 3, --mem-topm 8, --mem-topk 6, --mem-tau 0.03, --mem-read0 1.
- --mem-private R (0 = off).
- --mem-lr 0.01, --mem-s0 auto, --mem-clip 0.05, --mem-write-start 200, --mem-dv 256.
- --mem-freeze 0.6, --mem-ema 0.999, --mem-margin "0.01,0.01", --mem-addr-sg 1.
- --mem-probes 32 (sequences per set), --mem-guard 0.95, --mem-repair 0 (β; 0 = measure only),
  --mem-repair-usage 0.1.

## 9. Tests (CPU, before any run)
1. All flags off: outputs, losses and parameters bit-identical to v4d code over 40 training steps (accumulation 2,
   growth on).
2. Concept targets vs a brute-force loop: EOT cuts (shifted suffix-min), EOT positions masked, n/4 masks.
3. Level conditioning: level 2's loss gradient reaches level 1's head; masked-moment hinge = the nonzero() version.
4. Product keys: top-k_m = brute-force top-k over all S² pairs; duplicated scores → the lowest flat index.
5. EMA vs online: perturbing θ alone never changes the indices; the gradient reaches W_q, K through ω.
6. Freeze: after t_f, 1,000 optimizer steps leave the translator, sub-keys, norm statistics, private keys
   bit-identical; the same input → bit-identical rows, including under autocast bf16 (addressing stays fp32).
7. Step-0/1 gradients: at step 0 only W_o, W_ms, W_p, W_in (B_l), W_fb get a nonzero gradient; from step 1 V_sh,
   V_e, the gates and the translators do too.
8. Zero-init exactness: memory on with W_o = W_ms = W_p = B_l = W_fb = 0 → outputs equal the no-memory model.
9. Sparse write: unread rows bit-identical; read rows follow the S₀ row-Adagrad formula with the update clip.
10. Causality: changing tokens after t changes neither the reads nor the outputs of position t.
11. Copy-on-split: the twin's P, P̄, V_e, S_e, b and MemOpt rows equal the parent's.
12. Sub-key split: the band copy makes the split key's contexts read the same values as before the split.
13. Repair (when on): only rows below the usage threshold change; busy rows bit-identical; R⁰ unchanged.
14. Margin scope: L_margin's gradient is zero for every non-translator parameter (sg(a)).
15. Exclusions: BrainOpt's parameter list, clip_grad_norm_ and carry_brain_state contain no table; tables are not
    rebuilt by build_opt on births.
16. Checkpoint reload: identical addresses, tables, accumulators, freeze flag (re-applied); outputs identical.
17. Memory / VRAM: CPU estimate of the saved-for-backward bytes; an end-to-end CPU smoke with growth, a freeze,
    probes, the guard and the loop report.

## 10. Runs (only after the code review OK)
- Data: X:\Datasets\wikipedia_en_20231101_wiki16k (wiki16k BPE, 400M train tokens, val = 2,000 held-out articles).
- 1) memory model 12K steps; 2) no-memory control 12K (v4d config, single 16-token latent head, same data);
  3) memory model 24K. v4d batch / sequence settings; detached with the viewer window; checks every 20 minutes.
- Logged: all memory metrics of section 6, stability from the freeze to the end, knockouts at the end.

## 11. Review log (revision 2)
See plan_v2/12_memory_review_resolutions.md for every finding and its resolution.
