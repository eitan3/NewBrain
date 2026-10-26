# Design v2: growing spiking "graph brain" LM

Revised after four independent reviews (mixer/spiking, looped thinking, modular routing, growth/plan).
Literature references below come partly from the reviewers' memory; check exact hyperparameters before relying on them.

Hardware: Windows 11 laptop, RTX 5090 Laptop GPU (24 GB, sm_120), torch 2.11 + cu128. Target 10-50M params.

## 0. Ground rules (apply to every stage)
- **No Python loop over time or tokens in training.** Every op is a big parallel op over batch x sequence.
  This is what made growing_brain.py 9x slower on GPU.
- **Add one novel piece at a time**, each with a go/no-go test against the last passing configuration.
- **Fixed evaluation harness:**
  - validation loss (nats)
  - synthetic reasoning suite
  - tokens/s, logged throughout (laptop GPUs throttle)
  - peak memory
  - firing rate, and estimated synaptic operations (SynOps) as the efficiency metric (spikes don't speed up dense GPU matmuls)
- **Unit tests:**
  - chunked scan vs a naive fp64 loop, including extreme decays, with gradcheck
  - training (chunked) path vs inference (recurrent) path give identical logits
  - every growth op preserves the output, AND the first optimizer step after growth is not oversized

## 1. Neuron / token mixer
Gated linear recurrence (the "membrane"):

    S_t = a_t * S_{t-1} + k_t v_t^T
    y_t = q_t S_t

- **Default: Mamba-2 style, one scalar decay per head.**
  - Computed with the segsum chunked form; every exponent is <= 0, so it is numerically safe.
  - Decay: `a_t = exp(-softplus(dt_t + b) * exp(A_log))`, dt in [1e-3, 1e-1] log-uniform, A in [1, 16].
    This spreads timescales across heads.
- **Option: RWKV-6/GLA per-channel decay.**
  - Requires materialised intra-chunk exponents or secondary chunking.
  - log-decay clamped (>= -5 per step); chunk size 32-64.
- **Option: decay in [-1, 1]**, for state tracking such as parity.
- **Precision:** cumsum, exp, inter-chunk state and S in fp32; only the matmuls in bf16.
- **q and k:** SiLU or L2-normalised, scaled by d_k^-0.5.
- **Token mixing:** a causal depthwise conv (kernel 4) instead of token-shift.
- **Kernels:** a plain-PyTorch implementation is the reference. Try triton-windows plus flash-linear-attention (fla)
  kernels as the fast path, after checking they work on sm_120 with torch 2.11.

## 2. Spikes
- **Order inside the block:** `y -> per-head RMSNorm -> * silu(gate) -> spike -> out_proj`.
  The norm fixes threshold drift across heads with different decays.
- **Main spike type: graded or ternary with a learned per-channel scale alpha:**

      s = alpha * clamp(round(x / alpha), lo, hi)

  with STE / LSQ-style gradients.
- **Binary {0,1}** is an ablation only; it is expected to lose clearly (SpikeGPT/SpikeLM experience).
- **Schedule:** start analog, then anneal into spiking with `x + lambda * (s(x) - x)`, lambda going from 0 to 1.
- **Firing rate:**
  - init thresholds for 10-25% firing
  - the spike-rate penalty switches on only after warmup
  - alarm if more than 5% of channels are dead (< 0.1% firing) or saturated (> 95% firing)
- **State:** the state S stays linear (no reset inside), so the scan stays parallel.

## 3. Thinking = looped DEPTH, not a per-token inner loop
- **Ticks loop over depth, not over tokens.** One tick = one full-sequence parallel scan through a shared "core".
  Every tick level has its own across-token state, so the amount of thinking never changes the memory timescale.
- **Structure** (Geiping et al. 2025 recurrent-depth recipe):

      e = prelude(x)                  2 blocks, run once
      h0 ~ N(0, sigma^2)              random initial state
      for k in 1..r:
          h = h + core(norm([h; e]))  input injected every tick; sandwich RMSNorm;
                                      core = 2-4 mixer + spiking-MLP blocks;
                                      small init on the residual output
      y = coda(norm(h)) -> head

- **Training depth:**
  - r is sampled per step from a log-normal-Poisson, mean about 8, capped at about 3x the mean.
  - Truncated backprop through the last 4-8 ticks (the earlier ticks run under no_grad).
  - Activation checkpointing.
- **Per-token depth ("freeze-and-carry"):** a token past its depth is frozen, and later ticks read its frozen value.
  This is the same rule at training and inference.
- **Halting is NOT trained jointly** (avoids collapse to 1 or to max ticks):
  - Phase A: random-depth training only, with no ponder loss.
  - At inference, exit when the answer stops changing: KL(p_k || p_{k-1}) < eps, or ||h_k - h_{k-1}|| / ||h_k|| < eps.
  - Optional: a halting head, or the spiking "decision population", trained as a detached probe
    (stop-gradient into the core).
- **Thinking budget** is an inference knob; train with depths beyond the current max.
- **Speed:** torch.compile + CUDA graphs on the core.

## 4. Modules on a grid + router
- **Layout:**
  - modules on a 2D **torus** with small-world shortcuts (1-2 long-range links per module)
  - sensory input **broadcast** to all modules (low-rank projection)
  - motor readout **pools** from all modules
  - rule: graph diameter <= minimum tick count
- **Compute:**
  - Start with dense compute plus a mask.
  - Neighbour mixing as one batched einsum over a neighbour table, or as conv2d with circular padding
    (shared neighbour kernel, NCA-like).
  - Gather + bmm only if profiling shows a win.
- **Router:**
  - DeepSeek-V3-style aux-loss-free balancing: sigmoid affinity, plus a per-module bias updated by the load error
    (the bias is used for selection only).
  - z-loss on router logits; a tiny sequence-level aux loss as a backstop.
  - Always-on shared modules (sensory/motor).
  - No expert-choice routing: it leaks future tokens in a causal LM.
- **Router stability across ticks:** tick index + previous mask as router input, or route once per token.
  Measure the Jaccard overlap of active sets between consecutive ticks.
- **Inactive modules:** write gate = 0, so the recurrence stays linear; a learned constant leak, applied lazily.
- **One compute budget:** k fixed, depth decides the amount of thinking. Avoid stacking competing compute penalties.
- **Research side-experiment:** seRNN-style dense wiring with a distance-weighted wiring cost at small scale,
  to see what modularity emerges on its own.

## 5. Growth
- **Rule:**
  - zero ONLY the output projection; input weights random (or GradMax init)
  - new spiking units' thresholds set to fire at about the median rate
  - option: MSG mask ramp 0 -> 1 over about 1k steps
- **Optimizer:**
  - new exp_avg_sq slices start at the mean of the existing ones
  - rebuild the optimizer with all its param-group settings
  - rewarm the learning rate (from 0.1x over 1-5% of the growth interval)
  - WSD (warmup-stable-decay) schedule
  - **Bug in growing_brain.py `_carry_optimizer`:** new slices get zero moments but the old step count, so their
    first update is about 3x the learning rate.
- **Router-aware growth:**
  - new modules enter with a large negative router bias, annealed in, or via sparse-upcycling split of an existing expert
  - rescale the load-balance loss at growth
- **Order and trigger:** grow width first (MLP, heads with a sensible decay init), then depth by stacking/duplicating
  layers. Use a fixed token schedule (e.g. at 25% and 50% of tokens); a plateau-based trigger only later.
- **Long-run health:**
  - plasticity: dead-unit fraction and effective rank; remedies ReDo / continual backprop / shrink-and-perturb
  - forgetting: replay 10-20% of earlier data when the curriculum changes

## 6. Stages, go/no-go, fallbacks
| Stage | Compare | Go if | Fallback |
|---|---|---|---|
| 0 Baselines | small transformer vs analog Mamba-2-style mixer, same params/tokens | within 0.05 nats and at least 50% of the tokens/s | fla GLA/RWKV kernels |
| 1 Spiking | analog / graded / ternary / binary | best spike type within 0.1 nats of analog; 5-30% firing; no dead layers | analog + activation-sparsity penalty |
| 2 Looping | looped vs unlooped (same params, and same FLOPs) | at least +10 points on addition / graph paths or length generalisation; monotone accuracy-vs-r curve; exit depth correlates with difficulty | fixed random depth + entropy exit |
| 3 Routing | routed grid vs dense at the same active FLOPs | within 0.03 nats; usage entropy above 0.9 of max; wall-clock not worse | top-k MoE in the MLP only, no grid |
| 4 Growth | grown from 1/4 size vs trained at full size from scratch | within 0.02 nats with at least 25% fewer FLOPs; loss spike after growth recovers quickly | width-only LiGO/bert2BERT, or train at full size |

**Compute** (assumes 20-40 effective TFLOPS):
- smoke test: 5M params / 50M tokens, about 10 min
- decision run: 20M params / 300M tokens on TinyStories, 1-4 h (looping multiplies by the mean tick count)
- 3 seeds when the difference is under 0.05 nats

**Reasoning suite:**
- reversed multi-digit addition (train up to 10 digits, test 12-20)
- k-hop graph reachability
- S3/S5 permutation composition and parity at growing lengths
- 4x4 Sudoku / small mazes (stretch goal)
- TinyStories perplexity vs test-time r

**Final:** a leave-one-out ablation of every component.
