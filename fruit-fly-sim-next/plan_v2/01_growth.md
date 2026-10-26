# v2 design report 1: smooth growth / neurogenesis (design agent, 2026-10-21)

## Pre-allocated capacity
- Allocate every growable tensor at capacity C (~1.25x final; headroom for renewal). Born neurons = contiguous prefix with a
  per-layer counter n_born; forward uses views up[:n_c], down[:, :n_c], n_c = ceil(n_born/64)*64; mask covers <= 63 unborn in the
  last block. Mixer heads are already 64-channel blocks. Prefix slicing avoids wasted FLOPs (full masked matmul wastes up to 50% early).
- Memory: 16 B/param -> 64M-param capacity ~1 GB. Activations scale with n_c.
- torch.compile: bucket n_c to multiples of 256 (<= ~20 recompiles).
- Compaction: a dead neuron mid-prefix is swapped with the last born (rows/cols of params, Adam moments, alpha, ema_rate) -> optimizer never rebuilt.

## Birth protocol (neuron i born at t_b, age a)
- Output: d_i = 0 AND ramp m_i(a) = min(1, a/R), R = 1000 (MSG idea; with Adam, d moves ~lr per step regardless).
- Input: u_i = rho_med * normalize(0.5 v_k + 0.5 r); rho_med = median alive row norm; v_k = k-th top right singular vector
  of M_l = E[g_y x^T] (GradMax-lite, one hooked batch); r random orthogonal to v.
- Threshold: alpha_i = 2 * quantile(|x_i|, 1 - r_l), r_l = layer's current median firing rate; ema_rate_i = r_l.
  (Fixes the inherited-median-alpha bug in Spike.grow.)
- Optimizer: exp_avg 0; exp_avg_sq = elementwise median of alive rows.
- Per-neuron LR warm-in lambda_i(a) = 0.1 + 0.9 min(1, a/W), W = 500, applied after opt.step() as
  p[i] = p_old[i] + lambda_i (p[i] - p_old[i]) on newborn rows (Adam is scale-invariant, so scaling grads does nothing).
- No global LR rewarm.
- Explains the organism failure: inherited thresholds on mismatched input scales -> firing jump; big block insertion + Adam 3x first
  step + global rewarm -> saturation/drift -> mass ReDo.

## Where and when
- Literature: GradMax (Evci 2022, best for WHERE), SENN natural expansion score (Mitchell 2023, noisy), Firefly/Net2Net splitting
  (fallback), loss plateau (noisy), fixed schedule (robust: MSG and our runs).
- Primary rule: schedule decides WHEN, gradients decide WHERE.
  b(t) = (N1 - N0)/(t1 - t0) births per step for t0 = 5% .. t1 = 60% of tokens (fractional accumulation).
  Every 100 steps per site l: S_l = sum_{j<=8} sigma_j(M_l)^2 / cost_l; births allocated by softmax(log EMA(S_l) / 0.5).
  Heads born as whole 64-channel heads when a site accumulates 64 births. Depth: rare MSG-ramped layer births only after
  width saturates (all S_l < 0.5x their t0 value). Optional: b(t) *= clip(S_max(t)/S_max(t0), 0.5, 1.5).
- Fallback: activity pressure P_l = mean saturation + mean CBP utility; Net2Net-split the top-utility saturated neuron
  (copy u + 1% noise, halve d, copy Adam state).

## Renewal and death
- Utility U_i <- 0.99 U_i + 0.01 mean|s_i| * ||d_i||_1 (continual backprop). Maturity 2000 steps before replaceable.
- Per layer per step replace K = Poisson(1e-5 * n_born) lowest-utility mature neurons (+ always-eligible dead ones,
  ema_rate < 1e-3); hard cap 2 per layer per step; replacements go through the birth protocol.
- Death: mature neuron with U < 0.05x layer median for > 5k steps ramps out over R steps and frees its slot.

## Starting size
- VRAM is not the constraint; compute/tokens are. Decision runs: start 60% of final (~8M of 13M), grow linearly to 13M by 60%
  of tokens, capacity 16M. Next scale: start 30M, final 50M, capacity 64M (~1 GB state; 50M x 1B tokens ~ 3e17 FLOP ~ 3 h).
- Width before depth: start with the final layer count (inserted layers caused the shocks).

## Smoothness tests
1. CPU unit test: birth op leaves logits unchanged (atol 1e-5); compaction/swap leaves logits and Adam state unchanged.
2. KL(p_t || p_{t-1}) on a fixed probe batch on birth steps <= 1.5x median of non-birth steps.
3. No loss spikes: max (L_t - EMA_L)/sigma_L < 3 over growth windows vs a no-growth control.
4. Firing: layer rate drift < 0.03 per 1k steps; saturated < 5%; newborn rate within 0.5-2x layer median; dead < 5%; ReDo <= cap.

## Risks
GradMax gains modest in vision (unsure for LMs); compile recompiles; p_old copy traffic; ramp/warm-in vs WSD decay (stop births
before decay); symmetric spiking duplicates need noise.

## Validation (~1 GPU-hour, organism config, equal tokens, ~15 min each)
A smooth growth + GradMax + full birth protocol | B = A with uniform allocation | C = A without calibration and warm-in.
References: organism 1.459, full-size graded+floor 1.405. Go if A beats 1.459 by >= 0.02 and passes all smoothness tests;
if A ~= B, drop GradMax for the fallback rule.
