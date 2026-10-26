# R6 review, lens B: the read gate (blob_gate.py) in world_blob25d_12k

Read-only review. No repo edits, no GPU, the live run was not touched (only config.json, train.log and metrics.jsonl were
read; the live run has not reached step 500 yet). Evidence comes from the identical-code runs r3/r4/r5 (metrics.jsonl and diag.jsonl)
plus a CPU simulation of the real `BlobGate` class (`blob_gate.py` imported unmodified) that copies the
after_backward target / Huber / 1/propensity weights and the step_end AdamW / lambda / open_bias warm-up. Script:
`C:\Users\eitan\AppData\Local\Temp\claude\C--Eitan-Python-NewBrain\de301b72-951d-49a4-9895-252fb20d74d6\scratchpad\gate_sim.py`.

## Verdict

AUC 0.50 at step 750 is **not** just "too early". The fit timing and near-zero contents alone do not explain it. Two
implementation defects make the gate's output noise 100x to 1,000x larger than its target, which wipes out any
signal. The target itself is computed correctly: right sign, right scale, unbiased, and no gradient reaches the thought.
The simulation reproduces the r4 numbers at 450 steps after the fit almost exactly: open_bias −24.5 (r4: −25.7), gate loss
1.7–2.2e-3 (r4: 1.4e-3), lambda > 0 on every expert with open rate ≈ 0.5, and AUC ≈ 0.50 with decile slope ≈ 0. In the simulation this
happens **even though a learnable signal is planted**: the oracle AUC there is ≈ 0.6, and the gate starts with corr +0.33 to the signal and loses it.

(The trajectory forecast is in the section at the end.)

## How the target is computed (trace)

1. Exploration (blob_gate.py:435-436, 264-269). This is a per-pair Bernoulli draw with a known propensity
   p = share[stratum(A_g)] · 0.05/0.25, drawn only when `rec_on` is true, i.e. on in-window (grad-enabled) training passes. Out-of-window
   passes (k < K − bptt) never explore and never record.
2. The read scale (449-452) is s = an independent hash uniform in [0, 1) on exploration pairs and 1 elsewhere. It is cached in `sel`,
   so a recompute gives the same value.
3. Scale (304-336): out = s·(r^g + r^p). In backward, b = ⟨g, r⟩ = ∂L/∂s, where g = ∂(total/accum)/∂out (fp32, autocast off,
   r saved in bf16). The function writes b into `rec["bk"]` with copy_, which is idempotent. The s grad returned is None (s is not a
   real leaf, but b is the exact ∂L/∂s).
4. after_backward (580-595) computes b̃ = −b and b̃_cv = −b + (s − ½)·κ̂·‖r‖². This sign is correct: −l′(0) − ½κ‖r‖² does not depend on s.
   Then y = y_scale · b̃_cv with y_scale = batch·seq·accum = 16·512·2 = 16,384 (train_lm.py:1345). The loss
   is the per-token mean divided by accum, so y is in nats summed over tokens per read ("nats per token" as the plan defines it). ✓
   y is clipped at median ± 5·1.4826·MAD per call, δ = max(1.4826·MAD, 1e-3), w = 1/p, and the loss is Σ w·Huber / Σ w, backpropagated ÷ accum into the gate
   parameters only. The gate's AdamW (lr 1e-3, clip 1.0) steps once per optimizer step (step_end 609-613).
5. Timing: the fit is at step 300. The gate trains **and** the open_bias warm-up runs from step 300 to step 500 at the same time
   (warm_left is decremented in step_end only while `_gq` is non-empty, which starts at the fit). By step 750 the gate has had
   450 steps, 250 of them after the warm-up.

Checked and correct:
- Sign. b = dl/ds, b̃ = −b > 0 means the read helped. The κ̂ regression of b on s‖r‖² has slope +κ, which matches.
- Scale (N_tok). Correct.
- Unbiasedness of s ~ U(0, 1). Correct.
- Exploration selection is independent of ĝ, apart from the buffer overflow (finding 6).
- The gate cannot push the thought. φ, m, u_g, u_p and ĝ are all computed under no_grad (blob_gate.py:396-427). The stored
  `gate_rows` are detached copies. The gate is not a submodule (blob_memory.py:798), so it is outside model.parameters(),
  make_opt and the global clip. Its loss backward reaches only the gate's parameters, including W_cf, which sees only detached xf / u.
  The main loss cannot reach the gate either, because ĝ is computed under no_grad. **Confirmed.**

## Findings

### 1. HIGH (corrupting): the open_bias warm-up and gate training form a positive feedback loop ("bias chase"), steps 300-500 and lasting after that
blob_gate.py:427-432 and 622-628, plus `score()` at 235-236 (open_bias is part of ĝ, and ĝ is what the Huber loss compares with y).

```python
# inject_gated: the quantile of the RAW score over all pairs
bm._gq.append(nan_quantile(raw, 1.0 - bm.gate_open0))          # q77 of (ghat - open_bias)
# step_end, during warm:
new = torch.where(first, -q, gate.open_bias.lerp(-q, 0.2))
gate.open_bias.copy_(...)
# after_backward, the same steps: Huber(score(...) - y), score() includes + self.open_bias
```

Failure scenario: open_bias = −q77(raw), so the median pair's ĝ = q50 − q77 is below 0. The targets y are ≈ 0, so the Huber gradient
pushes ĝ upward on the exploration rows. The fastest way to do that is to raise raw for every pair (β, b1/a0, g0, the trunk).
That raises q77 by the same amount, open_bias follows it down (lerp 0.2), and the error is re-created at the next step.
raw therefore climbs at Adam's maximum rate for the whole 200-step warm-up. The simulation drives open_bias from −0.75 to −24.5 (r4: −25.7, which
`mem_gate_open_bias` shows) and ĝ's spread grows from 1.3 to 2.1 nats. The coherent max-rate updates inflate
the trunk / head weights, so after the warm-up the gate is very sensitive and very noisy. In the simulation the
planted signal's correlation drops from +0.33 to ≈ 0 during the warm-up (sim A, steps 0 → 150). With the warm-up turned off (sim C) the
correlation stays at +0.1 to +0.4.
The plan has no warm-up at all (§5.2: β₀ = −0.02, λ = 0, "nothing reads until the gate has seen a positive gain").
Fix: keep the bias out of the regression. Put it in the threshold (open = raw > λ_e − b), or train against y − open_bias, or freeze the gate's
optimizer during the warm-up. The simplest fix is to drop the warm-up (gate_warm 0), which follows the plan.

### 2. HIGH (corrupting): units mismatch. The gate regresses ~1e-3-nat targets with O(1)-nat outputs and Adam at lr 1e-3 (from step 300)
blob_gate.py:120-121 (prior weights ±0.5 on standardised features), 135 (lr 1e-3), 584 (y in raw nats).

- y: std ≈ 6.6e-3 (r4 diag `mem_gate_signal_var` 4.4e-5), mean −7e-5 to −1.1e-4. Any real per-pair signal is ~1e-4 to 1e-3.
- ĝ at init: the match-quality prior alone (pg, pp = ±0.5 on 6 standardised inputs) has std ≈ 1.2 nats, which is ~200x the
  target std and ~1,000x the signal. The plan calls β₀ = −0.02 nats "slightly negative". The prior is 60x larger than that.
- AdamW makes parameter steps of ≈ lr regardless of gradient size. On a 2,002-input trunk with noise-dominated gradients
  this moves ĝ by ~0.01 to 0.1 nats per step, an Adam noise floor 10x to 100x above the target.
- Huber δ = max(1.4826·MAD, 1e-3) ≈ 1e-3 to 3e-3. With |ĝ − y| ≈ 0.5 every row is in the linear region, so the gradient is a pure sign
  (median regression). The magnitude information in y is lost until ĝ is already within a few 1e-3 of y.
- The r4 evidence fits this: gate_loss 1.4e-3 implies |ĝ − y| ≈ 0.5 to 1 nat (with δ ≈ 2e-3). Decile slope 3e-5 means the ĝ
  deciles span ~10^4x more than the y deciles. A calibrated regressor on a noisy target would shrink toward the mean and
  give a slope near 1.
- Simulation, same data, y multiplied by 1,000 (sim B, still with the bias chase): corr(ĝ, signal) rises to +0.63 and
  AUC vs the signal sign to 0.79. The architecture can learn. The units cannot.

Fix: standardise the target, e.g. y_n = y / EMA(1.4826·MAD(y)) with δ = 1. A positive rescaling keeps the sign, so "read iff
predicted gain > λ_e = 0" still holds. Also scale the prior to the same units, or start pg/pp at about 0.01.

### 3. MEDIUM (interpretation): AUC vs 1[y > 0] cannot reach 0.60 at this per-pair SNR, even for an oracle
blob_write.py:597-609. The labels are single-sample signs of y, where noise std ≈ 6.6e-3 and |mean| ≈ 1e-4. If the true per-pair
gain spread is σ_s, an oracle's AUC is about 0.5 + σ_s/(π·σ_n). That gives ≈ 0.53 for σ_s = 0.1·σ_n. T0b measured 0.52 even with
knockout labels. So the plan's alert (pooled AUC < 0.60 after 0.3·S) fires whether or not the gate is good. Two more limits on the metric:
it pools the whole 250-step interval while the gate drifts, and it pools routed and shared rows. For a fair metric, track the decile slope
(calibration) and corr(ĝ, y) on a held-out sample. Also note `mem_gate_signal_*` is cumulative since the fit because health_every = 0
never resets `_hy`.

### 4. MEDIUM: lambda is "pinned" by ĝ noise, not by predicted gain (from step ~350)
blob_gate.py:279-283. Mean y is below 0, so a calibrated gate would put ĝ < 0 on most pairs, leaving λ = 0 and an open rate below 50%. Instead
λ_e > 0 on 100% of experts (lam_pos_frac 1.0), and the open rate sits at ≈ 0.5 to 0.52, because an overdispersed, roughly zero-mean ĝ
is above 0 for about half of all pairs. The controller itself works as written (0.01·(rate − 0.5) per step, ≥ 0). The pinning is a
symptom of findings 1 and 2. It will last until ĝ's noise falls below |E[y]| ~1e-4.

### 5. LOW-MEDIUM: the target is the gain in the total objective, not in LM nats (all steps)
train_lm.py:1690-1712 and loop_brain.py:1350-1405. b = ⟨∂total/∂out, r⟩, where total also includes deep supervision on earlier passes,
0.2·aux_x CE, the concept-hierarchy loss (c_w 0.3), the think-latent loss, the halt KL and the route aux. Reads change those losses too, which adds variance (and
non-LM "gain") to y. The plan says l = loss. To get LM nats only, a second backward would be needed, or accept and document it.

### 6. LOW: other observations
- Control variate inert: κ̂ ≈ 2e-9 in raw units, ≈ 3e-5 nats per ‖r‖² after scaling. (s − ½)κ̂‖r‖² ≈ 1e-5 vs y std 6.6e-3. Correct but
  useless. It is not a degenerate target. One κ̂ is shared across global and private (the plan specifies one per space type).
- Stratification on A_g is ineffective: global d1 ≈ 280 vs 2h0² = 32, so A_g ≈ 1e-4 for almost all pairs. Reads are almost all private
  (A_p ≈ 0.18). The strata edges start at 0.25/0.5/0.75 and lerp at 0.01 per step, so right after the fit nearly everything falls in stratum 0,
  giving an explore rate of 3% (open_rate_by_call ≈ 0.03 in the first calls). Propensities are still exact, so the target stays unbiased. Only efficiency suffers.
- Buffer overflow is 3 to 6% (broken-if is > 1%). compact_causal drops the latest positions first, exploration pairs included
  (`exk = ex & kept_all`), so late-position exploration rows are slightly under-represented. Mild selection bias.
- The gate's match / prev features are not RMS-normalised (log d1 ≈ 5.6, log(d2 − d1) can be −13.8, log(1 + age)). The docstring says
  every group is RMS-normalised. This adds to the gate's sensitivity in finding 2.
- `_yring` and `_stats_step` grow for 250 steps between metric calls (≈ 170 MB of GPU in small tensors per interval). It is not a gate-correctness issue,
  but it is a plausible contributor to the fragmentation in the r4 step-750 OOM (13 GB reserved but unallocated).
- Plan λ_r = 0.1 is not applied. This does not matter under Adam.

## Expected trajectory with the code unchanged
This comes from simulation A over 3,000 gate steps (simulation step s corresponds to run step 300 + s). Each simulation call has 1,024 tokens × 6 pairs, 4 calls per step, y noise
std 6.6e-3 (t3), and a planted signal of std ~1e-3 for which the oracle AUC vs y > 0 is ≈ 0.6:

| run step | sim ĝ std | gate loss | open rate | λ>0 | AUC(y>0) | corr(ĝ, true signal) |
|---|---|---|---|---|---|---|
| 750  | 0.7  | 2.2e-3 | ~0.5 | all | 0.48–0.52 | ≈ 0 |
| 1,000 | 0.46 | 1.6e-3 | ~0.5 | all | ~0.50 | 0.0–0.1 |
| 1,500 | 0.33 | 1.0e-3 | ~0.5 | all | ~0.50 | 0.0–0.1 |
| 3,000 | 0.26 | 8e-4 | ~0.5 (0.4–0.6 jitter) | all | 0.47–0.55 | 0.0–0.06 |
| 3,300 | 0.22 (plateau) | 7.5e-4 | ~0.5 | all | ~0.50 | ≈ 0.03 |

ĝ's spread levels off at ≈ 0.22 nats, about 40x the y std. That plateau is the Adam lr 1e-3 noise floor, so the gate does **not**
recover on its own. Without fixes, expect pooled AUC 0.50 ± 0.03, decile slope ≈ 0, open rate pinned at ≈ 0.5 with every λ_e
slightly above 0, and gate loss falling slowly toward ~7e-4 (the falling loss is shrinkage of the noise, not learning). The real run has
less learnable signal than the simulation (T0b: match features add nothing), so it should look the same or worse.

With fixes 1 and 2 (no warm-up chase, standardised y, prior in target units), sim B shows corr(ĝ, signal) +0.4 to +0.6 within
~300 to 500 gate steps. But AUC vs 1[y > 0] is still bounded by the per-pair SNR (finding 3): even a well-trained gate will read
~0.52 to 0.58 on this metric, not 0.60 or more. Judge the gate by decile slope and corr(ĝ, y) on held-out exploration pairs, plus a knockout check.

