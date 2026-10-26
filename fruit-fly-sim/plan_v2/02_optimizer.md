# v2 design report 2: weight update rule / optimizer (design agent, 2026-10-21)

## Problems with global backprop + AdamW here
- Growth: new slices get exp_avg_sq = median of old -> first Adam steps mis-scaled; global LR rewarm disturbs all old weights.
- Conditional activity: inactive modules get zero grad (not no grad) -> AdamW decays m, v AND applies weight decay -> silent
  shrinking/forgetting; on wake-up a shrunken v gives oversized steps. Global step count t is wrong for rarely-active modules.
- Continual learning: plasticity loss (Dohare et al., Nature 2024: continual backprop = small steady random re-init).
  Our ReDo-every-500 (955 units at once) is the opposite. Nothing protects old knowledge across curriculum shifts.
- Spikes: STE/LSQ biased; Adam amplifies tiny surrogate grads of dead/saturated units; alpha shares LR with weights.

## Evidence for alternatives (LM quality)
- Muon: ~2x compute efficiency vs AdamW in Moonshot scaling runs, used for 16B MoE (arXiv 2502.16982). No 2nd moment ->
  new rows simply start at momentum 0. Untested with spiking/STE nets.
- SOAP/Shampoo: ~35-40% fewer steps at 360M-660M (arXiv 2409.11321); heavy state, awkward to resize.
- Sparse/lazy Adam: per-row step count, update m/v/wd only where grad != 0 (standard for embeddings/MoE).
- Plasticity: continual backprop, L2-to-init, shrink-and-perturb. Consolidation (EWC/SI) weak at LM scale; SI-style LR damping = heuristic.
- Research bets: e-prop ~= truncated BPTT on tiny PTB (113 vs 113 ppl, arXiv 1901.09049); predictive coding 176 vs 163 ppl
  (arXiv 2211.03481), slow; forward-forward / target prop: no competitive LM results.
- Fast weights: the gated linear recurrence S_t = a_t*S_{t-1} + k v^T already IS Hebbian fast-weight learning (Schlag et al. 2021).
- Conclusion: backprop for slow weights; brain-like ideas go into WHICH params move and HOW FAST (age, activity, importance, neuromodulation).

## RL link
Three-factor rule dw = eta * M * e. Policy gradient = reward-modulated plasticity with backprop eligibility traces.
GRPO: G=8 samples/prompt, A_i = (R_i - mean)/std, KL to reference (beta 0.02-0.05), LR ~0.1x pretrain, same optimizer.
On-policy RL forgets less than SFT (RL's Razor, arXiv 2509.04259). Neuromodulator can gate per-module LR.

## Recommendation: "MuonLazy + maturity"
- Muon on hidden 2-D matrices: B <- 0.95B + G; O = NS5(G + 0.95B); W <- W(1 - lr*wd) - lr*0.2*sqrt(max(m,n))*O*D_age
  (lr 2e-3, wd 0.1 carry over).
- Lazy AdamW (betas 0.9/0.95, per-row step) for embeddings, head, norms, gains. Spike alpha: own group 0.1x LR, no wd.
- Inactive modules: skip their step entirely (no decay of momentum or weights).
- Maturity D_age per output neuron: m_i = 1 + (k-1) exp(-age_i/tau), k=4, tau=1% of steps; divided by (1 + c*I_i),
  I_i = SI-style importance EMA(|g*dw|) normalised to mean 1, c=0.5. Newborns/copies age 0 (+ small noise). Replaces global rewarm.
- Renewal: continual backprop: each step re-init the lowest-utility unit (EMA(|h_i| * ||w_out_i||)) among units older than
  2000 steps, rate ~1e-5 per unit per step, function-preserving (zero output weights). Replaces ReDo-every-500.
- Bet A (--plastic): low-rank plastic adapter P on top layer, three-factor online update:
  e_t = 0.9 e_{t-1} + psi'(u) post (x) pre; P <- (1-1e-4)P + 1e-3 * M_t * e_t; M_t = GRPO advantage or token surprise.
- Bet B (--opt soap): SOAP, preconditioner freq 10.

## A/B (~1 GPU-hour)
Organism config, ~12 min each: AdamW / AdamW+lazy+maturity / MuonLazy+maturity / Muon only / (SOAP).
Metrics: val loss, loss spike in 200 steps after growth, firing/dead/saturated after growth, forgetting probe (easy tier after hard phase).
Win = > 0.01 better than AdamW without more dead units. Bet A: 20-min frozen-backbone GRPO vs plastic adapter on a toy reward.
