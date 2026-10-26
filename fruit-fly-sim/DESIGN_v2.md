# NewBrain v2 design: the growing, spiking, partly-active organism

Draft 1, 2026-10-21. Merged from four design reports (plan_v2/01-04) and the evidence in PLAN_v2_brief.md.
The curriculum / RL teacher, the senses and associative memory are planned on the thread side and plug in at section 7.
Decisions that need Eitan are marked **[DECIDE]**.

## 0. Diagnosis: why the v1 organism underperformed
v1 organism = 1.459 (3 seeds) vs the predicted 1.431 from adding the separate costs of growth (+0.026) and spikes (+0.085).
Four concrete causes:
1. **Spike thresholds are absolute.** Any change in signal scale (copied layers, new rows) moves every threshold, so after growth
   firing jumped 20% -> 40%.
2. **The spike gradient is cut outside its range**, so saturated neurons stay saturated (25%) and silent ones stay silent (15%).
3. **Growth was abrupt.** Whole layers and hundreds of neurons arrived at once. Newborns got an inherited threshold, Adam's
   oversized first step, and a global LR rewarm that also shook every old weight.
4. **Renewal was abrupt.** ReDo wiped 955 neurons in one step right after growth.

Everything below is designed so these cannot happen.

## 1. The neuron (spiking unit)
- **Membrane:** the Mamba-2-style gated linear recurrence (unchanged; matched the transformer at 1.320 vs 1.314).
- **Spike = "normalised graded count" (NGC):**
  - Each token's signal is divided by its own mean magnitude (divisive normalisation, like real cortex).
  - Each neuron then emits 0..7 spikes against a learned threshold.
  - Scale-invariant, so growth can't shift firing rates.
  - Uses a leaky straight-through gradient (0.1 outside the range) so saturated and silent neurons can recover, plus a
    properly scaled threshold gradient.
- **Homeostasis:** "floor" only, i.e. rescue near-silent neurons. Never let a firing-rate rule own the threshold
  (that cost +0.23).
- **Optional: distillation from the analog brain** (it "teaches" the spiking one). Used by SpikeLM/SpikingBrain.
- **Target:** spiking cost <= 0.04 (from 0.085).

## 2. Only part of the brain is active (routing)
- **Routed MLP:** each layer's MLP becomes **1 always-on shared expert + 64 small experts; each token uses 6 of them**.
  - Same compute per token as now.
  - **8x more total neurons** (~70M params at the current width; fits easily in 24 GB).
- **Load balancing:** aux-loss-free, i.e. a per-expert bias nudged toward even load (DeepSeek-V3 style).
- **Health statistics count only tokens routed to that expert**, so a rarely-used expert is never mistaken for dead
  (otherwise we'd get the 955-wipe failure again).
- **Implementation:** sort tokens by expert, then one batched matmul (grouped GEMM if it works on sm_120, otherwise padded bmm).
  No Python loop over experts.
- **The 2D grid / small-world wiring stays a label and a prior only.** On a single GPU it saves nothing; flat routing comes first.
- **Later add-on:** a product-key memory layer (~1M slots), a cheap "lots of neurons" store for facts. It overlaps with the
  thread's associative-memory plan, so we should merge the two ideas.

## 3. Smooth growth (neurogenesis)
- **Pre-allocated brain:**
  - Tensors are allocated at capacity (~1.25x final).
  - Born neurons are a contiguous prefix; unborn neurons cost no compute (slice to the nearest 64).
  - No tensor or optimizer rebuild, ever.
- **Births are continuous:** a schedule releases a few neurons per step from 5% to 60% of training.
- **WHERE to grow** follows the gradient: every 100 steps, each site gets a "how much would a new neuron help" score
  (GradMax-style: top singular values of the gradient-input correlation), and births go mostly where the score is highest.
- **In the routed brain, growth = a new expert spawned next to the most overloaded expert.** The balancer bias is a free
  "pressure meter". The new expert starts as a noisy copy with zero output and slowly earns traffic.
- **Birth protocol** (every newborn, every renewal):
  1. output weight 0 and an output ramp 0 -> 1 over ~1000 steps;
  2. input weights in the most useful direction (from the gradient), scaled like their siblings;
  3. its own threshold, calibrated on its own activity;
  4. proper optimizer state;
  5. no global LR rewarm.
- **Shape:** width before depth. Start with the final number of layers (inserted layers caused the shocks).
  Depth growth is rare and ramped.
- **Death:** neurons that stay useless for a long time ramp out and free their slot, so the brain can reorganise,
  not just get bigger.

## 4. Smooth renewal
- **Continual backprop:** each step, at most a couple of the least-useful mature neurons per layer are re-born
  (through the birth protocol).
- Usefulness = activity x outgoing weight.
- Newborns are protected for 2000 steps.
- Expected rate ~1e-5 per neuron per step; hard cap 2 per layer per step.
- Replaces ReDo-every-500.

## 5. How weights learn (optimizer / update rule)
- **Keep backprop for the slow weights** (local rules like e-prop or predictive coding are still far behind on language).
  The fast weights are already brain-like: the recurrent state is a Hebbian outer-product memory.
- Brain-like ideas go into WHICH weights move and HOW FAST:
  - **Muon** for hidden matrices: ~2x compute efficiency over AdamW in recent LLM scaling runs; no second moment, so newborn
    rows need no guessed state.
  - **Lazy updates:** inactive experts are not touched at all. No silent weight decay or optimizer-state decay while they sleep.
  - **Maturity:** young neurons have high plasticity and important mature neurons consolidate. Per-neuron LR multiplier
    1..4x decaying with age, divided by an importance estimate.
  - **Thresholds:** their own learning-rate group.
- **[DECIDE] Conflict between the reports, newborn speed.**
  - The growth report wants newborns to start SLOW (LR warm-in 0.1 -> 1).
  - The optimizer report wants them FAST (4x, the young-brain idea).
  - My proposal: protect the *function* with the output ramp and let newborns *learn* fast (4x).
  - We test both (one A/B arm each).
- **RL later = reward-modulated plasticity.**
  - GRPO (8 answers per prompt, reward relative to the group, KL leash to the pre-RL brain), same optimizer, ~0.1x LR.
  - On-policy RL forgets less than supervised fine-tuning, a good fit for the teacher idea.
- **Research bet behind a flag:** a small "plastic" layer that learns online with a three-factor Hebbian rule
  (eligibility trace x reward). Fast adaptation to the teacher's rewards without backprop.

## 6. Thinking (latent reasoning)
- **Structure:**
  - prelude (1-2 blocks)
  - -> shared core (2 blocks) looped r times, each tick a full parallel pass
  - -> coda
  - -> head
- **Signals:** the membrane h is carried across ticks; spikes are the messages within a tick.
- **Training:** random r (mean 6, up to 18) and backprop through the last 4 ticks.
- **Developmental:** the number of ticks grows too, 2 -> 4 -> 6, alongside the neurons.
- **Stopping, per token:** stop when its spike pattern stops changing (< 1% of spikes differ between ticks) or its prediction
  stops changing.
- **Routing per tick:** different ticks can use different experts.
- **Proven on synthetic tasks first:**
  - k-hop pointer chasing (train k <= 8, test 16)
  - reversed-digit addition (train <= 10 digits, test 12-20)
  - Pass criteria: accuracy rises with thinking time, and harder problems use more ticks.

## 7. Interfaces to the thread-side plans
- **Curriculum and teacher RL:** the trainer gets a data/teacher interface (phases, reward function), and the GRPO loop
  from section 5.
- **Senses:**
  - Each new sense = an encoder plus its own always-on "sensory" shared expert and a group of routed experts.
  - Born through the same growth machinery, with zero output so nothing learned breaks.
  - Heads stay swappable.
- **Associative memory:** to be merged with the product-key memory layer of section 2.

## 8. Build plan with go/no-go (GPU estimates on the RTX 5090 laptop, ~15 min per TinyStories run)
- **P0 (CPU, no GPU):** capacity/prefix model, NGC spike, routed MLP (padded-bmm path + dense reference), birth protocol,
  renewal, MuonLazy + maturity.
  Unit tests: function preservation for every birth/renewal/compaction, routed == dense reference, optimizer state on swaps.
- **P1: four independent A/B studies at 13M, ~6-7 GPU-hours total, run one after another:**

  | Study | Arms | Go if |
  |---|---|---|
  | Spike | graded+floor -> NGC -> +leaky STE -> +0..7 -> +distillation | spiking gap <= 0.04, dead + saturated < 5% |
  | Routing | dense vs 64x6 routed vs aux-loss vs coarse 8x1 vs dense-8x upper bound | routed beats dense by >= 0.03, balanced, >= 70% speed |
  | Optimizer | AdamW vs AdamW+lazy+maturity vs Muon+lazy+maturity vs Muon only | beats AdamW by > 0.01 |
  | Smooth growth | smooth + gradient-placed vs smooth + uniform vs no birth protocol | beats v1 organism (1.459) by >= 0.02, passes smoothness tests (no loss spikes, KL bound, stable firing) |

- **P2: organism v2** = the winners combined, 3 seeds. Target <= 1.40 at 13M (v1: 1.459), healthy neurons throughout.
- **P3: thinking loop** on the synthetic tasks, then on language.
- **P4: scale up and raise:**
  - start ~30M -> grow to ~50M active, 8x total via routing;
  - curriculum + teacher RL + vision (thread plans).

## 9. Questions for Eitan
1. **[DECIDE]** Agree with "backprop for slow weights, brain-like control of which/how-fast weights learn" as the default,
   with local Hebbian learning as a flagged research bet? Or do you want local learning rules as a main track now?
2. **[DECIDE]** Newborn learning speed: protected-but-fast (my proposal) vs slow warm-in. Or just let the A/B decide?
3. **[DECIDE]** Order of the P1 studies. My suggestion: spike -> routing -> optimizer -> growth, since growth depends on the
   other three.
4. **[DECIDE]** Dataset after P2: TinyStories has too little difficulty range for a real curriculum. Move to a graded corpus
   (BabyLM-style child-directed speech -> stories -> simple Wikipedia) together with the thread's curriculum plan?
5. **[DECIDE]** Scale: is ~50M active / ~400M total (routed) a good next target for the laptop, or do you want to plan
   bigger (e.g. cloud GPUs later)?
