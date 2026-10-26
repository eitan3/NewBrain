# Uncertainty / "loop until sure" (steps B/C) + growth memory fix - 2026-10-23

Step A (measurement, analysis/uncertainty_probe.py on v3c): the best "unsure" signal is how much the prediction
still changes between passes (KL p_k || p_{k-1}); entropy marks inherently hard words; router uncertainty has no
signal. Untrained exit on KL gives fixed-K3 quality with ~15% fewer passes, but no gain at the best K: the brain
was never trained to spend passes on hard words.

## B. Implemented (flags, all off by default; 106 tests pass)
- `--deep-sup W`: LM loss on every earlier backpropagated pass (mean over the passes in the bptt window except the
  last, weight W; the last pass stays the main loss). Replaces the old single sampled pass at 0.2 when on.
- `--halt 1` (+ `--halt-prior 0.2`, `--halt-beta 0.01`): stop head = MLP(d+3 -> 64 -> 1) on DETACHED features
  [normf(x_k), log1p KL(p_k||p_{k-1}), entropy/log V, k/16] -> lambda_k. Halting distribution
  p_k = lambda_k * prod_{j<k}(1 - lambda_j), last pass takes the rest. Loss = sum_k p_k * CE_k (CE detached: the
  head never shapes the brain; only deep-sup changes the LM objective) + beta * KL(p || truncated geometric(prior)).
- Eval (`halt_eval`, in the loop report): K_max 16; each word exits at the first pass where its cumulative stop
  probability >= 0.5; reports the loss at each word's exit, mean passes, expected passes, a histogram, and the
  fixed-K losses of the same run. Read from one K_max run (as if stopped words froze: the probe showed freezing
  does not change the other words). Log line: "loop until sure: loss X with N passes per word".
- Cost: one decoder read-out per pass (d x V = 1.6M MACs per word per pass, ~+30% of a pass's body) in training
  when deep-sup or halt is on; the read-out of passes outside the bptt window runs without gradient.
- Not yet: C (feeding the uncertainty back into the router / workspace) and real compute savings at inference
  (skipping frozen words' expert work); both come after B shows the stop head learns something useful.

## Growth memory fix
- `free_cache(device)` = torch.cuda.empty_cache() after every birth event (v4b: ~22 GB held for ~11 GB used,
  38k -> 17k tok/s). PYTORCH_CUDA_ALLOC_CONF=expandable_segments is not supported on Windows, so not used.

## Proposed queue (not launched)
1. v4b + `--route-noise 0.3 --route-noise-decay 0.8` (single change vs v4b_split).
2. v4b + `--deep-sup 0.3 --halt 1` (halt-prior 0.2, halt-beta 0.01).
Same 200M-token recipe as v4b_split; compare at equal tokens, K=6, iso-FLOP, and (2) the loop-until-sure loss
and passes per word.
