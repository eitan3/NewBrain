# 22. Learned memory access gate (plan only, no code, no training) — revision 2, 2026-10-24

Builds ON TOP of plan 21 (life cycle, fixed addressing + restart, coarse pool, neighbour writes, resume). Read with plans 12,
15, 16, 20, 21 and the code: `memory.py` (select / online / read_shared / private, MemRead, MemOpt), `loop_brain.py`
(`_hier_step`, `run`, the halting and concept features), `spiking_lm.py` (RoutedMLP capacity).

**The owner's request:**
- Memory is not used for every token: the brain learns WHEN it needs memory and, later, WHAT is worth storing.
- Addresses come from the thought / context only, and the thoughts are not constrained.

**Revision 2** applies both reviews and the coordinator's decisions (section 12, review log). The main changes:
- the router is a PREDICTOR of the brain's own content gate, trained by BCE. There is no gradient through the forward
  read, no score multiplier and no IPW;
- a smaller detached feature set;
- c_f = 0.5 with a 10% floor;
- the write gate is OFF in v1, and only a step-size variant is specified;
- a Phase-0 go / no-go before any run.

With every new flag off the code is plan 21 bit for bit.

---------------------------------------------------------------------------------------------------------------------

## 0. Summary

| part | v1 decision |
|---|---|
| Router | s = w·φ + b on DETACHED features φ (93-d). fp32. Trained by BCE towards the detached per-position mean content gate ḡ (the brain's own learned "use this read"). It has its own loss weight, its own AdamW group, its own grad clip, and is excluded from the global clip_grad_norm_. |
| Selection (training) | Per read and per micro-batch: the top-c positions by s, plus a uniform 10% exploration floor drawn from the rest. Only these positions read AND write. Static shapes; the indices are cached under checkpointing. |
| Forward | Selected positions read exactly as in plan 21 (not scaled by s); the others get no memory output. |
| Schedule | c = 1 until 0.2·S, then a cosine to c_f = 0.5 at 0.5·S; the floor applies once c < 1. |
| Inference | A per-pass threshold θ̂_k = an EMA of the training top-k cut-off, updated only on the first (non-recompute) run of the block in training; never in eval or probes. |
| Write gate | OFF in v1. A step-size variant is specified behind a flag (section 5), with a toy test required before it is enabled. |
| Plan-21 coupling | Address reuse (readdr) off under the gate; the norm statistics over all positions; restarts / the reservoir from open positions; probes ungated; the router never frozen; the private memory gated by the same mask. |
| Go / no-go | Phase 0 (section 9): the top 20% of positions hold ≥ 60% of the positive benefit, and the AUC ≥ 0.75. |

---------------------------------------------------------------------------------------------------------------------

## 1. Notation

**Sizes and passes**
- B = 16, T = 512, N = B·T = 8,192 positions per micro-batch (accumulation 2).
- K passes. Plan 21 runs `--mem-read0 0`, so read k happens after pass k (k = 0..K−2) and feeds pass k+1.
- S is the total number of optimizer steps.

**Per read k and position t**
- p̄_{k,t} ∈ R^64: the slot mean of n_s(s_k).
- μ_l, ν_l: the concept mean / log-variance, level l = 1..4.
- r_{k,t,l,h}: the read vectors.
- g_{k,t,l,h} = σ(w_g,lh·[p̄; r] + b_g,lh): the content gates (plan 12 §3; plus the coarse pool's gates g^c).
- M̃_{k,t}: plan 21's memory output.
- ḡ_{k,t} := sg( mean over the fine (l, h) and coarse (l, h_c) gates of g ) ∈ (0, 1).
- O_k: the open set of read k.

**Other symbols**
- sg = stop-gradient; σ = logistic.

---------------------------------------------------------------------------------------------------------------------

## 2. Router = a predictor of the content gate

### 2.1 Features φ_{k,t} (all detached, fp32, causal)

| feature | dim | definition | where it comes from |
|---|---|---|---|
| thought | 64 | p̄_{k,t}, standardised by the translator's EMA norm (the p̄ part of â) | `prep` (all positions) |
| uncertainty | 4 | ν̄_l = mean_i ν_{l,i}, scaled (ν̄ + 6)/10 | hier, read k |
| concept change | 4 | Δμ_l = log1p(‖μ_{l,k} − μ_{l,k−1}‖ / max(‖μ_{l,k}‖, 1e-6)); 0 at k = 0 | hier (= u's 2nd component) |
| thought change | 1 | δ = log1p(‖p̄_k − p̄_{k−1}‖ / max(‖p̄_k‖, 1e-6)); 0 at k = 0 | run(): p̄ of the previous read |
| last-token surprise | 2 | σ_t = −log p₀(idx[t] ∣ <t) = pass-0 CE at t−1; a flag for t = 0 or idx[t−1] = EOT (σ_t := its EMA mean there) | pass-0 read-out (2.3) |
| surprise trace | 1 | tr_t = max(σ_t, γ·tr_{t−1}), γ = e^{−1/8} (≈ 8 tokens); reset at EOT and at each row start | windowed (2.3) |
| next-token entropy | 1 | H_{k,t} = −Σ_v p_k(v) log p_k(v) / log V, from pass k's read-out | per-pass read-out (2.3) |
| pass | 16 | one-hot of min(k, 15) | — |

- In total 93 dims. The scalar features are standardised by EMA statistics (rate 0.001, fp32, updated on the first run
  of the training block only, never frozen).
- The 1,536-d μ block of revision 1 is dropped: it let the router memorise topics, and it cost parameters.
- Causality:
  - p̄, ν, Δμ, δ and H_k at t are functions of tokens ≤ t (the core is causal);
  - σ_t uses the read-out at t−1 evaluated at the known input token idx[t];
  - tr_t is a max over σ_{≤t}.

### 2.2 The loss and the optimiser

    s_{k,t} = w·φ_{k,t} + b          (fp32; w ~ N(0, 0.02²) from a dedicated generator, b = logit(0.12) = −2 to match the content gates' init)
    L_r = λ_r · mean over the reads k and the positions t ∈ O_k of BCE(σ(s_{k,t}), ḡ_{k,t})

- Soft targets in (0, 1). The minimiser of BCE with soft targets is σ(s) = E[ḡ ∣ φ], so s ranks positions by the
  predicted content gate.
- Only open positions have a target: ḡ needs the read r.
- Isolation:
  - φ and ḡ are detached, so L_r's gradient reaches w, b only. The router parameters receive no other gradient,
    because the forward read does not depend on s.
  - Parameter group "mem.gate.": AdamW (lr 1e-3, β = (0.9, 0.99), wd 0), its own clip (norm 1.0), excluded from the
    global `clip_grad_norm_` and from the BrainOpt group lists that carry or rebuild state on growth.
  - λ_r = 0.1. The loss is added to the total for convenience; its gradient touches only the router.
- Reads outside the bptt window run under no_grad. Their L_r terms are computed under a local `torch.enable_grad()` on
  the detached (φ, ḡ), so every read trains the router.
- Where it is computed: `block(s)` (checkpointed) returns φ (N×93 fp32, ≈ 3 MB) and ḡ_O (n_open) as extra outputs. The
  logits s with gradient and the BCE are computed OUTSIDE the block, so no router gradient passes through the
  checkpoint. Inside the block the selection uses s computed under no_grad from the same weights.

### 2.3 Read-out order (reviewer A, E5)

**Training** (deep supervision + halting compute every pass's read-out; `readout_frac` must be 1, asserted under
`--mem-gate`). At pass k:
1. the core;
2. the read-out `ce_k, lp_k = _readout_ce(x_k, tgt)` (moved BEFORE `_hier_step`; it depends only on x_k);
3. `_hier_step` (the router, the selection, the read);
4. `halt_features(x_k, lp_k, prev_lp, k, cu)` AFTER `_hier_step`, because it needs the hier uncertainty cu.

Only the order of independent computations changes, so the values are identical (test T2).
- σ_t = ce_0[t−1] (shifted by one; t = 0 → the flag);
- H_{k,t} from lp_k (already detached).

**Eval / inference** (no targets):
- one chunked read-out per pass gives lp_k: the entropy for every read, plus σ from pass 0;
- the log-softmax is chunked over positions (the eval batch 32 × 512 × 16,384 fp32 would be 1 GB unchunked);
- the cost is ≈ +50-60% eval FLOPs (one LM head ≈ 6.3M MACs per token per pass);
- `--mem-gate-feat` can drop "ent" to bring eval back to a single pass-0 head (+10%).

**Trace.** The recursion tr_t = max(σ_t, γ tr_{t−1}) over T = 512 would be 512 small kernels. Instead:

    tr_t = max_{0 ≤ j < W, no EOT / row start in (t−j, t]} γ^j σ_{t−j},    W = 24

computed as one unfold (B, T, W), a γ^j weight, a document mask and a max. It equals the recursion exactly except for
terms below γ^W ≈ 0.05·σ (test T9).

### 2.4 Why a predictor, not gradient-through-forward (the reviewers' finding)

Revision 1 trained s through the forward (M = 1[open]·m(s)·M̃; st / sigmoid multipliers). Its gradient signal is
⟨g_t, M̃_t⟩ with g_t = ∂L/∂M_t.

**Proposition 1 (the forward-gradient signal vanishes once W_o and the content gates are learned).**
- (a) M_t = Σ_l W_o,l u_{t,l} with u_{t,l} = Σ_h g_{t,lh} r_{t,lh}, so Σ_t ⟨g_t, M̃_t⟩ = Σ_l ⟨∂L/∂W_o,l, W_o,l⟩_F. At a
  stationary point of W_o this is 0: the aggregate signal is zero.
- (b) The content gate's logit a_{t,lh} has ∂L/∂a = σ'(a)·⟨g_t, W_o,l r_{t,lh}⟩, the same per-position benefit signal,
  already consumed. At a stationary point of w_g, E[σ'(a)⟨g, W_o r⟩ [p̄; r]] = 0. The router's signal is then
  orthogonal to every feature the content gate sees (p̄, r); what is left is only the part explained by features outside
  the content gate (ν, Δμ, δ, the surprise) and the σ' curvature.

So as the brain learns, the forward-gradient router loses its signal exactly where the content gates do the work. ∎

The predictor uses the content gate's OUTPUT as its target instead: the brain's own, already learned "use this read".
This signal does not vanish when the content gates converge; it IS their converged answer. For comparison, the residual
−⟨g, M̃⟩ per position is still logged (section 7).

The st / sigmoid gradient modes stay documented alternatives. They are not built in v1.

### 2.5 The selection, consistency without IPW, and no collapse

Selection at read k (training, c < 1):

    n_c = q₆₄(c·N),   n_ε = q₆₄(0.10·N),   n_c + n_ε ≤ N                    (q₆₄: round to a multiple of 64)
    A_k = topk(s_k, n_c)                  (s computed under no_grad; ties broken by a fixed random key)
    E_k: u = rand(N, generator = gen_gate_dev); u[A_k] = −1; E_k = topk(u, n_ε).indices      (device-side, uniform)
    O_k = sort(A_k ∪ E_k)

- Exact: |O_k| = n_c + n_ε, A ∩ E = ∅. No host sync (n_c, n_ε are host integers).
- With c = 1: O_k = all positions and no gather (the plan-21 path).

**Proposition 2 (the BCE on the open sample is consistent; no IPW needed).**
- Assumptions:
  - (i) the decision 1[t ∈ O_k] depends on (φ_{k,1..N}, the exploration draw) only; it never depends on ḡ_{k,t};
  - (ii) ḡ_{k,t} depends on the batch only through position t's own read (no interaction between gates);
  - (iii) positivity: P(t ∈ O_k ∣ φ) ≥ ε' := n_ε/(N − n_c) > 0 (the floor).
- Claim: by (i)-(ii), ḡ_t ⊥ 1[open] ∣ φ_t (missing at random given φ), so E[ḡ ∣ φ, open] = E[ḡ ∣ φ].
  - A well-specified BCE fit on the open sample therefore converges to E[ḡ ∣ φ] on the whole support (by (iii)), with no
    importance weights.
  - This is the difference from revision 1: there the router was trained on a quantity (the gradient) whose observation
    the router itself controlled, and the target was a function of s. Here the target does not depend on s, and the
    selection depends only on φ.
- Misspecification (a linear router): the fit is the BCE projection under the open-sample density, i.e. weighted ×1/ε'
  towards the top region. The ranking near the boundary, the part that matters, gets the most data. ∎

**Proposition 3 (exact capacity, static shapes).** n_c, n_ε are functions of the step only (c(t) and the floor). topk
returns exactly n_c distinct indices, and the masked uniform topk returns exactly n_ε distinct indices outside A_k. The
gathered tensors have the static leading dimension n_c + n_ε within a step. The cached indices make the checkpoint
recompute shape-identical. ∎

**No collapse (the "rich-get-richer closed" risk).**
- No feedback through the router's gradient: s does not touch the forward, and the target is the content gate, so a
  closed region cannot lower its own target by being closed.
- What remains is a content loop:
  - rows used only by closed contexts get fewer writes and fade (λ);
  - their content gates then learn on explored positions only (10%), which lowers ḡ there;
  - which lowers s.
- Bounds against it:
  - (a) the floor: ≥ ε' of every region reads AND writes;
  - (b) c_f = 0.5 keeps half of all positions open;
  - (c) the infancy c = 1 until 0.2·S: the content gates and the predictor learn on dense data first;
  - (d) the training top-k always opens exactly n_c + n_ε positions, so "closes everything" is impossible by construction.
- Detector: the knockout "forced-open minus learned" (section 7). If forced-open beats the learned gate by > 0.01 nats,
  raise c_f.

### 2.6 Inference threshold (the train / inference mismatch)

Top-k over a batch is non-causal: it depends on future tokens and on other sequences. The score s itself is causal, so
"open iff s_{k,t} > θ̂_k" is causal.

    τ_k = the n_c-th largest s_k (from the topk above);   θ̂_k ← (1 − ρ_θ) θ̂_k + ρ_θ τ_k,   ρ_θ = 0.02
    k ≥ 10 share θ̂_10;   θ̂_k = −∞ while c = 1

- The update happens ONLY on the first run of the block of a TRAINING forward with targets: a `cache["first"]` flag.
  It never happens on the checkpoint recompute, in eval, in probes or in knockouts (test T6).
- Eval / probes / inference: O_k = {t : s_{k,t} > θ̂_k}, with no exploration.
  - Static shapes via `nonzero_static(size = ⌈1.5·c_f·N⌉)`, with the overflow (the lowest scores) dropped and logged.
  - Or a plain `nonzero` (one host sync per read, acceptable at eval).

**Proposition 4 (consistency).**
- Assumptions: a fixed router; scores with a continuous cdf F, density f(q) > 0 at q = F^{-1}(1 − c); the batch's
  positions mixing with an effective size N_eff (correlated thoughts: N_eff ≈ B·T/ℓ_corr ≈ 256, not N).
- Then:
  - τ → q, with sd ≈ √(c(1−c)/N_eff)/f(q) (central order statistic);
  - the EMA θ̂ has sd ≈ √(ρ_θ/(2 − ρ_θ)) × that, if consecutive batches are independent (reviewer A, E7: the per-batch
    sd uses N_eff, not N);
  - the inference open rate 1 − F(θ̂) → c;
  - the two rules disagree on a share |F(θ̂) − F(τ)| → 0.
- So the threshold rule is the large-batch limit of the training rule. ∎

Consequences:
- Per batch, the inference rate varies (more on fact-dense text, less on chit-chat). That is the intended behaviour.
- A moving router adds a lag of ≈ (the drift of q per batch)/ρ_θ.
- Logged: the train-time fraction of positions above θ̂_k per pass (it should track c); the eval open rate per pass.
- MoD's auxiliary BCE predictor of "was in the top-k" reduces to a monotone function of s, hence to this threshold
  (Raposo et al. 2024). It is not needed.

### 2.7 Granularity and plan-21 couplings

- Per read (each read has its own top-c, with the pass one-hot), not once per position. A joint budget over passes is
  impossible because the reads are sequential.
- Address reuse (`--mem-readdr-every`) is off under the gate (the open sets differ between reads). Fresh addressing on
  60% of positions costs about as much as plan 21's reuse schedule.
- The EMA norm statistics are over ALL positions (`prep` runs before the gather), so the schedule cannot shift the
  addressing frame.
- Restarts: key use (`pend`) and the reservoir come from open positions, so the keys follow the traffic that reads.
- Probes are ungated: in record mode `select` runs on every probe position, so plan 21's switch signal stab_w is
  unchanged. Additionally logged: stab_w over the open probe positions.
- The router, its feature statistics and θ̂ are never frozen (not at the switch, not at the freeze).
- The life cycle (λ, β_S, S₀): unchanged.
  - S₀ is set at write start (step 200) while c = 1.
  - Rows get ≈ 0.6× the writes at c_f: watch plan 21's ‖V‖ per write-count bucket.
  - The decay λ is tied to the address turnover, not to c.
- Neighbour writes: from open positions only (rows from `select` on O_k).
- Coarse pool: the same O_k.
- Private memory: the same mask.
  - q1 is scattered into zeros (N, 64), and `private()` receives the open mask o.
  - The per-slot output is multiplied by o[t_k], so closed tokens have no output, no G_e / G_P contribution and no
    `pend_e` count.
  - v1 does not save the private bmm.
- Chunk-level reads for the coarse levels: later, as in revision 1 §4 (the gain is < 2% of a pass, it changes the
  semantics, and it needs EOT-aware chunks).
- Halting: the trainings are independent (the router's input is detached, and the halt head uses detached CE).
  - At halting eval, reads at the exit pass are wasted compute, not errors.
  - Logged: the open rate by exit pass.
  - The halt features are not router inputs in v1.

### 2.8 The block with variable N under checkpointing

Inside `block(s)`:
1. `hier` (all N);
2. `prep` (all N, the norm statistics);
3. φ;
4. on the first run only: s under no_grad, O_k, the τ / θ̂ update and the feature-statistics update (`cache["sel_open"]`);
   on recompute, the cached O_k;
5. gather â, a, p̄ at O_k;
6. select / online / read_shared on n_open rows;
7. M = zeros(N, d).index_copy(0, O_k, M̃_O); q1 likewise.

Returns (mus, nus, M, q1, margin, φ, ḡ_O, O_k). The recompute is shape-identical: the indices are cached, the exploration
draw is cached and never redrawn, and the generator is not advanced on recompute.

---------------------------------------------------------------------------------------------------------------------

## 3. Compute and VRAM (c_f = 0.5 + floor 0.1 → 60% open)

| per token per read (MACs) | plan 21 | gated (60% open) |
|---|---|---|
| concept heads | 1.93M | 1.93M |
| fine + coarse addressing | ≈ 1.75M | ≈ 1.05M |
| reads, gates, W_o | ≈ 0.8M | ≈ 0.48M |
| private (v1, masked) | ≈ 0.3M | ≈ 0.3M |
| router + features | — | ≈ 0.01M |
| core pass | ≈ 5.3M | 5.3M |

- The memory part goes ≈ 2.85M → ≈ 1.84M (−35%); the whole pass ≈ 10.1M → ≈ 9.1M (−10%).
- The backward scatter / dω and the checkpoint recompute of the memory part: ×0.6.
- MemOpt: unchanged (almost every row is still touched).
- Wall time: the memory part is partly launch-bound (plan 13). Estimate: memory −15…−30%, step −5…−10% after 0.5·S, and
  +≈ 0.5% before 0.2·S (the router, the features, one extra topk).
- VRAM: −0.05…−0.15 GB (the cached indices, the recompute peaks). The tables are unchanged.
- Eval: + the per-pass chunked read-outs (+50-60% eval FLOPs with "ent"; +10% without).

Note: "memory work drops 5×" holds only for the FLOPs of the gated part at an open share of 0.2. With
v1's open share of 0.6 it is ≈ 1.7× on those FLOPs and far less in wall time. Section 3 states only these numbers.

---------------------------------------------------------------------------------------------------------------------

## 4. Addresses

Unchanged from plan 21: a = [μ_l ; p̄] → the EMA-standardised â → fixed random translators / sub-keys, fine + coarse pools,
uncapped restart. No token keys (`--mem-read0 0`). The gate only decides which positions use their address.

---------------------------------------------------------------------------------------------------------------------

## 5. Write gate — OFF in v1; only the step-size variant is specified

Revision 1 scaled each position's contribution to G by β_t. The reviewers' objection: row-Adagrad normalises by √S_r, and S
accumulates (βg)². A row written by uniformly high-β positions therefore gets the SAME step as with β = 1 (the scale
cancels), and mixed rows get distorted accumulators.

**Step-size variant** (`--mem-wgate 1`, default 0):
- **Surprise:** u_t = the final-pass CE of the next token (`ces[-1][t]`, detached).
- **Weights:** β̃_t = clip(exp(κ·ẑ_t), 0.25, 4), with ẑ the per-micro-batch z-score of u; κ = 0.5. The weights are
  normalised to mean 1 over the open positions of each read.
- **Per-row strength** (MemRead backward, two extra scalar index_adds into (rows,) fp32 buffers):
  - B_r += Σ_{(t,j)→r} ω_{t,j} β_t;
  - W_r += Σ_{(t,j)→r} ω_{t,j};
  - β̄_r = B_r / W_r (contribution-weighted; 1 where W_r = 0).
- **MemOpt** (G and S unweighted, as in plan 21):

      S_r ← (decayed) S_r + mean_i G_ri²            (unweighted g²: the normalisation is unchanged)
      ΔV_r = −η · β̄_r · G_r / √S_r,  clipped to ‖ΔV_r‖ ≤ c_Δ          (β applied AFTER the normalisation)

  It is a per-row learning-rate multiplier in [0.25, 4], so a surprising row actually moves farther. The clip c_Δ
  still bounds every step.
- **Brain gradients: per-step gradients identical; trajectories differ.** Within one step, the forward and every
  autograd gradient are unchanged: only the two scalar buffers and MemOpt change, and the buffers are read only in
  `after_step`. But β changes V, so from the next step on the reads, losses and gradients differ from a β ≡ 1 run.
- **Causal:** β is used after the step's predictions; there are no writes at inference.
- **Enabling it requires** the toy test T10 first.
  - Setup: a small table and two context types, "facts" (consistent targets, high CE early) and "noise" (random
    targets).
  - Pass condition: with the variant, fact recall reaches threshold in fewer steps than β ≡ 1, the noise rows' norms
    stay ≤ 1.1× the β ≡ 1 run, and nothing diverges over 2,000 steps.
- The β-on-G variant of revision 1 and Titans-style data-dependent forgetting are dropped.

---------------------------------------------------------------------------------------------------------------------

## 6. Flags (all off = plan 21 bit-identical)

| flag | default | meaning |
|---|---|---|
| `--mem-gate` | 0 | 1 = the predictor gate on |
| `--mem-gate-c` | "1.0,0.5" | c at the start / final |
| `--mem-gate-sched` | "0.2,0.5" | the cosine starts / ends (fractions of the run) |
| `--mem-gate-explore` | 0.10 | the uniform floor (once c < 1) |
| `--mem-gate-w` | 0.1 | λ_r (router BCE weight) |
| `--mem-gate-lr` | 1e-3 | the router's AdamW lr (own group, wd 0) |
| `--mem-gate-clip` | 1.0 | the router's own grad-norm clip |
| `--mem-gate-feat` | "pbar,nu,dmu,dthought,surp,trace,ent,pass" | the feature set |
| `--mem-gate-trace` | 8 | the trace time constant (tokens); W = 3× it |
| `--mem-gate-thr-ema` | 0.02 | ρ_θ |
| `--mem-gate-eval` | learned | learned / random / inverted / oracle / open / off (eval gating mode) |
| `--mem-wgate` | 0 | the step-size write variant (section 5) |
| `--mem-wgate-kappa` | 0.5 | κ |

- In cfg only when not at the default.
- The router is `LoopBrain.mem_gate`, built after every existing module with a forked RNG.
- θ̂_k, the feature EMA statistics, the tie-break key and gen_gate_dev's state go into Memory's extra / host state.
- `--mem-gate` asserts `readout_frac == 1` (training) and `mem_shared > 0`.

---------------------------------------------------------------------------------------------------------------------

## 7. Diagnostics and eval modes

**Eval gating modes** (`--mem-gate-eval`; all at the same open rate r = the learned gate's eval rate, except open / off):
- learned: s > θ̂_k;
- random: a seeded Bernoulli(r);
- inverted: the bottom-r by s;
- oracle: the top-r by the measured benefit b_t (non-causal, an upper bound);
- forced-open;
- off.

For each, the knockout delta on Wikipedia val and on world-facts.

**Measured benefit:** b_t := CE_t(memory off) − CE_t(forced open), from two eval forwards. It is per position (all
reads at t), including the interactions. Logged:
- the benefit capture C(gate) = Σ_{t open} b_t / Σ_t max(b_t, 0) for learned vs random vs oracle;
- the AUC of s̄_t (mean over reads) for 1[b_t in the top 20%];
- the share of the positive benefit held by the top 20% (the Phase-0 statistic, re-measured at every report).

**Knockouts restricted to open vs closed:** the memory-off ΔCE on the positions the learned gate opens vs those it
closes. A working gate gives open ≫ closed ≈ 0.

**Open rate per pass and by position type** (eval_world.py tags from the generator / world JSON):
- names, numbers;
- relation cue ("lives in", "works as");
- pre-attribute (the token before an attribute), attribute;
- other world text, Wikipedia, the Q/A "A:" position.

The expectation: pre-attribute and "A:" ≫ c_f; Wikipedia / chit-chat < c_f.

**Training (tensors, read only on logging steps):**
- L_r; the BCE calibration (mean σ(s) vs mean ḡ on open positions);
- the train-time fraction above θ̂_k per pass (it should be ≈ c);
- θ̂_k; the score quantiles (5/25/50/75/95%) per pass;
- the open-set overlap between reads k and k+1;
- ḡ mean on top-k vs explored positions;
- the residual −⟨g, M̃⟩ per position (a tensor hook on M; mean on top-k vs explored, and its rank correlation with s),
  for comparison with the predictor's ranking.

**Probes:** stab_w over all probe positions (unchanged) and over open ones; the open rate by halting exit pass.

---------------------------------------------------------------------------------------------------------------------

## 8. Tests (CPU)

1. **Flags-off identity**: bit-identical to plan 21 over 40 steps (growth, accumulation 2) and end to end.
2. **Gate on, c = 1**: the forward outputs, all non-router gradients and G are bit-identical to plan 21. This includes
   the read-out reorder of 2.3; the router gets a nonzero gradient from L_r only.
3. **The BCE router learns a planted ranking** (T3b-like, content gates present).
   - Setup: a tiny LoopBrain on synthetic sequences where a planted feature (a marker token followed by a "fact" whose
     value is only recoverable from memory) makes the read useful. The content gates and W_o train normally.
   - Pass condition: after N steps the router's AUC for the planted positions is ≥ 0.9, and ḡ on the planted positions
     exceeds ḡ elsewhere. The same with c = 0.5 and the floor (a selection-consistency check of Proposition 2).
4. **Capacity exact**: |A| = n_c, |E| = n_ε, A ∩ E = ∅, both multiples of 64; |O| static across reads and batches; a
   constant router gives the tie-key order, not the first positions; no host sync in the training path.
5. **Floor**: over 2,000 draws each non-top-k position's frequency is ε' ± 4 s.e.; deterministic for a seed; the
   device-side draw is not redrawn on recompute.
6. **The EMA updates only on the first run**:
   - θ̂_k, the feature statistics and gen_gate_dev's state change exactly once per training read;
   - they are unchanged by the checkpoint recompute, by eval, probes and knockouts.
7. **θ̂ consistency**: synthetic i.i.d. and AR(1)-correlated scores. θ̂ gives an inference open rate within the
   order-statistic bound of c, and the threshold vs top-k agreement is ≥ 1 − bound.
   - Causality: in threshold mode, changing tokens after t changes neither the decision at t (all passes) nor the outputs
     at t.
8. **Checkpoint, variable N**: the gated `_hier_step` with and without checkpointing gives identical outputs and
   gradients; different n_open across reads.
9. **Features**: σ_t = ce_0[t−1]; the chunked eval read-out equals the unchunked one; the windowed trace equals the
   recursion within γ^W; the resets at EOT / row starts.
10. **(Only before enabling `--mem-wgate`)** the toy write test of section 5; per-step brain gradients bit-identical with
    the variant on vs off, from the same state (the trajectories differ afterwards); β ≡ 1 → MemOpt bit-identical.
11. **Private gating**: closed tokens have zero private output, no G_e / G_P rows and no pend_e counts.
12. **Couplings**: the norm statistics equal the ungated ones; restart usage from open positions only; probes record
    every probe position; the router is trainable after the freeze; readdr off under the gate.
13. **Resume**: 40 straight steps = 20 + save + resume + 20, bit-identical on CPU, with the cut inside the cosine
    (c < 1), crossing a θ̂ update, a restart event and a probe eval. The router's optimizer state and the device
    generator are restored.
14. **Eval modes**: learned / random / inverted / oracle / open / off callable; the rates and the benefit capture are
    logged.
15. The existing tests (plan 21's + the earlier 158) pass.

---------------------------------------------------------------------------------------------------------------------

## 9. Phase 0 (go / no-go, before any gated run)

Measured separately on a plan-21 checkpoint (or the current memory run), with the memory forced open:
- b_t = CE_t(memory off) − CE_t(open) on the val and world-facts batches.
- **GO** if:
  - (1) the top 20% of positions by b_t hold ≥ 60% of Σ max(b_t, 0); AND
  - (2) a linear predictor on the section-2.1 features reaches AUC ≥ 0.75 for "top-20% benefit" on held-out batches.
- If (1) fails, the benefit is diffuse: gating can only save compute, not focus memory. Stay at c = 1, or use the gate
  purely for speed with c_f chosen from the benefit curve.
- If (1) holds but (2) fails, the features do not see the need yet. Revisit the features before building the gate.

---------------------------------------------------------------------------------------------------------------------

## 10. Recommended world-facts configuration (after GO and the tests)

    plan 21: --mem-life 1 --mem-addr-learn 0 --mem-restart 100 --mem-coarse 64 --mem-nbr 12 --mem-read0 0 --mem-probe-every 2
    gate:    --mem-gate 1 --mem-gate-c 1.0,0.5 --mem-gate-sched 0.2,0.5 --mem-gate-explore 0.10 --mem-gate-w 0.1
             --mem-gate-lr 1e-3 --mem-gate-clip 1.0 --mem-gate-feat pbar,nu,dmu,dthought,surp,trace,ent,pass
             --mem-gate-trace 8 --mem-gate-thr-ema 0.02
    write:   --mem-wgate 0
             --save-report-ckpts 1

**Comparison runs** (12K steps, the same data / seed): `world_v4d_12k` (the control), plan 21 ungated, and plan 21 +
the gate.

**Decisive numbers:**
- facts accuracy gated ≥ ungated;
- the benefit capture: learned ≫ random, and ≥ 0.7 × oracle;
- the knockout on open ≫ closed;
- pre-attribute / "A:" open rates ≫ 0.5;
- the step time.

---------------------------------------------------------------------------------------------------------------------

## 11. Implementation size (estimate)

| file | what | lines |
|---|---|---|
| loop_brain.py | the router module (fp32), the features + the windowed trace, the read-out reorder, the selection / exploration / caching and θ̂ in `_hier_step`, gather / scatter, L_r outside the block, eval gating modes, the residual hook | ≈ 150 |
| memory.py | the private open mask + usage masking; usage / reservoir from open positions; probes on every position; θ̂ / feature statistics / tie key / gen_gate_dev in extra + host state | ≈ 70 |
| train_lm.py | flags, c(t), the "mem.gate." AdamW group with its own clip (excluded from the global clip), logging, resume, eval modes and the benefit / capture at reports | ≈ 90 |
| eval_world.py | position-type tags, open rates, the knockout open vs closed, capture / AUC | ≈ 70 |
| (optional) write variant | the B_r / W_r buffers in MemRead, the β̄ step in `_row_adagrad` | ≈ 40 |
| tests | 8.1-8.15 | ≈ 380 |

About 380 lines of code (+40 optional) and 380 of tests: ≈ 1.5-2 days after plan 21 lands. It touches the same
functions, so it is built on plan 21's final code. The Phase-0 measurement is separate and comes first.

---------------------------------------------------------------------------------------------------------------------

## 12. Review log (revision 1 → 2)

The review documents themselves were not available to me. Rows R1-R12 are as summarised in the coordinator's
decisions. Rows E3 / E5 / E7 / E8 follow reviewer A's text as relayed by the coordinator.

| # | finding | resolution |
|---|---|---|
| R1 | The gradient-through-forward router (st / sigmoid multipliers) gets a signal ⟨g, M̃⟩ that goes to 0 once W_o and the content gates are learned (the aggregate equals ⟨∇W_o, W_o⟩; the content gate consumes the per-position part). | The router is a predictor of the detached content gate ḡ, trained by BCE (2.2); proof in Proposition 1 (2.4). The st / sigmoid modes are kept only as documented alternatives, not built. |
| R2 | The forward read scaled by the score changes the model and couples the router to the read. | Removed: selected positions read exactly as in plan 21; the others get no output. |
| R3 | IPW was needed only because the router trained on its own selected sample; it adds variance (16× weights). | Removed. The selection depends on φ only and the target does not depend on s, so the BCE is consistent on the open sample with the floor's positivity (Proposition 2). |
| R4 | The router's optimisation must not interfere with the brain: global clip, precision, loss scale. | Own loss weight λ_r, own AdamW group and clip, excluded from the global clip_grad_norm_; fp32; detached inputs and targets; BCE computed outside the checkpoint block. |
| R5 | The 1,536-d μ block is costly and invites topic memorisation (frequent-token opening). | Dropped. Features: p̄, ν̄, Δμ, δ, the surprise, the surprise trace, the next-token entropy, the pass one-hot (93-d). |
| R6 | c_f = 0.2 is too aggressive before we know the benefit concentrates; the 5% floor is thin. | c = 1 until 0.2·S, a cosine to c_f = 0.5 by 0.5·S; the floor 10%, uniform, drawn on device. |
| R7 | The θ̂ / statistics updates could run on the checkpoint recompute, in eval or in probes. | Updates only on the first run of a training block (a cache flag); test T6. |
| R8 | Plan-21 couplings were underspecified. | 2.7: readdr off, the norm statistics over all positions, restarts / reservoir from open positions, probes ungated, the router never frozen, the private memory gated by the same mask. |
| R9 | The β-on-G write gate is cancelled by row-Adagrad's normalisation (S accumulates β²g²). | The write gate is OFF in v1. Only the step-size variant is specified (β̄_r after the normalisation, S unweighted), behind a flag, with a toy test before enabling (section 5). |
| R10 | Diagnostics could not tell a good gate from a random one, or measure what was missed. | Eval modes learned / random / inverted / oracle / open / off; benefit capture; the AUC of s vs b; knockouts on open vs closed; open rates by position type; the train-time fraction above θ̂; score quantiles; the residual −⟨g, M̃⟩ (section 7). |
| R11 | No evidence yet that benefit is concentrated enough for gating to pay off. | Phase 0 go / no-go (section 9): top-20% share ≥ 60% and AUC ≥ 0.75. |
| R12 | The test list was too long and partly about removed mechanisms (IPW, multipliers). | Tests 8.1-8.15, focused on: identity, the planted ranking, capacity, floor, θ̂, first-run-only updates, resume. |
| E3 | Revision 1's Proposition 8 ("the write gate leaves the brain gradients identical") holds per step only: β changes V, so the trajectories differ from the next step on. | Reworded in section 5 to "per-step gradients identical; trajectories differ", and test 8.10 likewise. Moot while the write gate is off. |
| E5 | The read-out reorder had to be exact: the CE / log-probs before `_hier_step`, while the halting features need cu from it. | 2.3: the read-out (ce, lp) moves before `_hier_step`; `halt_features` stays after; T2 checks bit-identity. |
| E7 | The IPW variance formula applied the finite-population correction twice and used an uncentred Σx²; Horvitz–Thompson gives M²(1−f)S²/n. | Resolved by removing IPW (R3); the formula no longer appears. |
| E8 | (a) L_z pulls the scores toward 0, not toward θ (θ ≈ the 80th percentile > 0), so the re-testing of closed regions comes from the floor, not from L_z. (b) DKW slip: at N_eff = 256 the normal approximation gives about 0.05, while the uniform DKW bound gives 0.085. | (a) Moot: L_z is removed with the gradient-trained router; in v1 the re-testing of closed regions comes from the floor only (2.5). (b) Resolved: revision 2 uses the order-statistic (normal) sd with N_eff (Proposition 4) and quotes no DKW figure. |

## 13. References

- Raposo et al., *Mixture-of-Depths*, arXiv:2404.02258 (2024): top-k capacity routing, the non-causality of top-k, the
  causal predictor.
- Behrouz, Zhong, Mirrokni, *Titans*, arXiv:2501.00663 (2024/25); Behrouz et al., *Miras*, arXiv:2504.13173 (2025):
  surprise-weighted writes, forgetting.
- Lample et al., *Product-key memory*, arXiv:1907.05242 (2019).
- Banino et al., *PonderNet*, arXiv:2107.05407 (2021); Graves, *ACT*, arXiv:1603.08983 (2016).
- Zhou et al., *Expert-choice routing*, arXiv:2202.09368 (2022); Fedus et al., *Switch Transformers* (2022): capacity,
  overflow.
- Rubin (1976), *Inference and missing data* (missing-at-random, Proposition 2); Bahadur (1966), quantiles in large
  samples; Dvoretzky–Kiefer–Wolfowitz / Massart (the DKW bound).
