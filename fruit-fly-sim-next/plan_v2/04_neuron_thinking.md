# v2 design report 4: neuron model v2 + thinking loop (design agent, 2026-10-21)

## A. Neuron / spike model
Likely causes (inferred from code):
1. MLP spikes have no norm before the threshold (absolute alpha) -> any residual-scale change (copied layers, new rows) moves
   all thresholds -> firing 20% -> 40% after growth. Mixer norms but then multiplies by silu(z), so it drifts too.
2. Clamp gives zero gradient outside [lo, hi] -> saturated channels stay saturated (25%), negative MLP channels die (15-20%).
3. alpha gets raw LSQ grads; missing LSQ scale g = 1/sqrt(N*Q_p) (Esser et al. 2020).
4. Newborns take the median alpha.

Recommended spike "normalised graded count" (NGC), after SpikingBrain (arXiv 2509.05276):
    m_t = mean_c |x_t,c|  (divisive normalisation per token per layer);  u = x / (m_t + eps)
    q = clamp(round(u / theta_c), lo, hi), theta_c learned (init ~20% firing), hi = 7 (3-bit count)
    s = theta_c * m_t * q;  leaky STE (grad 1 inside, 0.1 outside);  theta grad scaled by 1/sqrt(N*hi)
- Scale-invariant thresholds -> growth no longer moves firing rates. Apply after the gate in the mixer.
- Graded count = soft-reset integrate-and-fire run for hi virtual steps (I-LIF, SpikeYOLO) -> spike-count meaning kept, free.
- Uniform levels 0..7 (SpikingBrain: 94% of counts in [0,7]); optional mean-count penalty 1e-3 after warmup.
- Expected: maybe halves the +0.085 gap (unsure).
- Reset across tokens via sigma-delta error feedback with 2-3 parallel Jacobi sweeps: experimental, low priority.
- Skip ALIF (NGC already adapts). Keep floor homeostasis as rescue only; never let a rate rule own the threshold.
- Distillation from the analog 1.320 model: CE + 1.0 KL(p_analog || p_spike) (+0.1 MSE per block); SpikeLM/SpikeBERT/SpikingBrain;
  maybe recovers 30-50% more of the gap.
- Growth hygiene: calibrate newborn theta on its own first batch; per-channel analog->spike ramp ~1k steps;
  ReDo capped (<= 2 units/layer/step) and frozen 2k steps after growth.
- Validation: 13M, 3 seeds: graded+floor 1.405 -> +NGC -> +leaky STE/LSQ scale -> +hi=7 -> +KD -> organism.
  Targets: spiking gap <= 0.04; organism <= 1.40; dead + saturated < 5% after growth.

## B. Thinking loop v2
    e = prelude(x) (1-2 blocks); h0 ~ N(0, 0.02^2)
    for k in 1..r: h = h + core(RMSNorm(W_in [h; e]))   core = 2 blocks (SSD mixer + NGC MLP), out-proj init / sqrt(2 n_core r_mean)
    y = coda(RMSNorm(h)) -> head
- h = analog membrane across ticks; spikes = messages inside a tick; each tick = full-sequence scan with its own state; no tick
  embedding (depth extrapolation). Huginn (Geiping 2025), Ouro (arXiv 2510.25741).
- Train: r ~ log-normal-Poisson mean 6, cap 18; backprop last 4 ticks; halting not trained jointly.
- Developmental: grow the tick count 2 -> 4 -> 6 at the 25/50% growth points (cheapest function-preserving growth).
- Core growth: width as now; depth = block inside the core with zero out-proj; check ||h_k - h_{k-1}||/||h_k|| still decays.
- Routing per (token, tick) with the previous mask as router input; fixed k per tick; Jaccard > 0.6 near convergence.
- Exit per token (freeze-and-carry): spike-pattern Hamming change < 1% between ticks, or KL(p_k || p_{k-1}) < 5e-4.
- Optional detached halting probe (Ouro-style): target 1[CE_k - CE_{k+1} < 0.01].
- Benchmarks (CPU-generatable): k-hop pointer chasing on 32-node functional graphs (train k 1..8, test to 16; Saunshi ICLR 2025);
  reversed-digit addition (train 1-10 digits, test 12-20; Fan ICLR 2025).
- Pass: accuracy rises monotonically with r in {1,2,4,8,16,32}; ticks-to-90% grows with k; adaptive ticks vs difficulty
  Spearman > 0.7; stable to 3x training mean r. Baselines: r = 1 and equal-FLOP untied deep model; spiking vs analog core.
- Caveat: linear-recurrence state is finite; check pointer chasing at k = 1 first (recall vs thinking).
