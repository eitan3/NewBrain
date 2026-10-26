# NewBrain v2 planning brief

## What Eitan wants (his words, condensed)
- A **developing organism**, not just a model: born small, grows while it keeps learning, raised like a baby
  (words -> sentences -> stories ...), later taught with **rewards/penalties (RL)** by a teacher, not only supervised.
- **Senses**: text first, then vision, later audio/touch, all feeding one brain (like the fly's sensors). Swappable output heads.
- **Spiking neurons**, brain-like, graph-like local connectivity; **only part of the brain active** at a time (MoE-like).
- The model **decides how much to think** (internal latent reasoning).
- **Smooth** everything: growth neuron-by-neuron (not jumps), renewal of a few neurons every step (not hundreds every N steps).
  **Where and how it grows really matters.**
- **More starting neurons** than now.
- **Rethink the weight update / optimizer.**
- Plan carefully before building. Hardware: one Windows laptop, RTX 5090 Laptop GPU 24 GB, 192 GB RAM.

## What exists (fruit-fly-sim/)
spiking_lm.py (model), train_lm.py (trainer), test_spiking_lm.py (41 tests), DESIGN_graph_brain.md (v2 design reviewed by 4 agents,
stages 0-4). Mixer = Mamba-2-style gated linear recurrence, chunked scan in plain PyTorch, fp32 decay maths. Spikes on outputs
(analog/graded 0..4/ternary/binary, learned per-channel size alpha, data calibration at spike start, STE/LSQ). Growth: heads, MLP width,
layers (top or spread, fresh or copy init), function-preserving (zero output weights), Adam state carried (median 2nd moment), LR rewarm.
Homeostasis "full" (owns alpha) or "floor" (rescues only near-silent units). ReDo every 500 steps. Curriculum: easy/medium/hard thirds.

## Evidence (TinyStories, ~13M params, equal compute, val loss; 3 seeds where marked, seed spread +-0.002..0.006)
- transformer 1.314; Mamba-style mixer 1.320 (3 seeds) -> mixer = attention at this scale, 56% of its speed.
- growth v1 (born 2.9M, grow at 10/20/30%, layers on top) 1.373; growth v2 (born half, grow 25/50%, copies inserted in the middle) 1.346 (3 seeds).
- spikes: graded 1.382 (20% dead neurons), graded+floor homeostasis 1.405 (3 seeds; 11% dead), ternary 1.548 / +floor 1.541 (59% -> 0.2% dead),
  binary 1.528 / +floor 1.557 (21% saturated). "Full" homeostasis (thresholds set only by firing-rate rule): graded 1.612, ternary 1.745 -> bad.
- ORGANISM (graded spikes + floor homeostasis + growth v2): 1.459 (3 seeds). Worse than the sum of costs (predicted ~1.431):
  after growth, firing jumps 20% -> 40%, 25-28% channels saturated, ~15% dead; ReDo wiped 955 units at once after the first growth.
  Newborn neurons inherit the median threshold but their input scale differs.
- curriculum on TinyStories: neutral (growth+curr 1.349 vs 1.346) or slightly negative (full size 1.328 vs 1.318; costs on hard tier).
- GPU: kernel-launch-bound small loops were 9x slower than CPU in the first prototype; parallel chunked scans are fine.
