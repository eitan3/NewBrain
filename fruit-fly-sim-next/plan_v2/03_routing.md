# v2 design report 3: which part of the brain is active (routing) (design agent, 2026-10-21)

## Options at 10M-200M params on one 24 GB GPU
- Fine-grained top-k MoE in the MLP + shared expert: PRIMARY. Beats dense at equal compute; experts the size of the full MLP
  are rarely optimal (Ludziejewski/Krajewski 2024, arXiv 2402.07871); DeepSeekMoE fine experts + shared experts (2401.06066).
  Must sort tokens by expert + grouped GEMM; a Python loop over experts is launch-bound (the old 9x trap).
- Aux-loss-free balancing (Wang et al. 2024, arXiv 2408.15664): per-expert bias, selection only, += u*sign(load error), u=1e-3. USE.
- 2D torus modules + small-world links: no evidence it beats flat MoE; on one GPU locality saves nothing. Keep the grid as labels/priors only.
- Mixture-of-Depths: top-k over sequence not causal; with linear-recurrence mixer a skipped token saves nothing without gather. SKIP.
  Use "null experts" (MoE++ style, from memory) instead; freeze-and-carry in the thinking loop gives per-token depth.
- Expert choice: leaks the future. NO.
- Product-key memory layer (Berges et al. 2024, Memory Layers at Scale, 2412.09764): beats dense 2x compute and MoE at matched
  compute/params, esp. facts; gather-bound, needs sparse optimizer handling. LATER add-on (stage 3b), ~0.5-1M slots.
- Spike activity sparsity: no training speedup at 10-25% unstructured firing; only batch-1 inference could exploit it.
  Real compute savings come from module-level routing. Report SynOps.

## Interactions
- Growth "where pressure is highest": the aux-free bias is a pressure meter (most negative bias = overloaded expert). Spawn a new
  expert: up-weights = copy + noise, router row = copy + noise, down-projection = ZERO, router bias starts ~-3 and is annealed in
  over ~1k steps, Adam 2nd moments = mean of existing, rewarm. Grows TOTAL neurons at constant ACTIVE compute; active grows via k+1
  or a wider shared expert. One expert per layer every N steps = smooth.
- Spikes: router reads the analog normed residual, never spiked; alpha per expert; firing EMA and ReDo must count ONLY tokens routed
  to that expert (otherwise rarely-used experts look dead -> mass wipe, same failure as the 955-unit ReDo). Track dead experts
  (load < 0.1 k/E) separately from dead neurons inside experts.
- Thinking loop: route per tick inside the shared core with a tick embedding in the router input; balancer stats pooled over ticks;
  log Jaccard overlap of chosen experts between ticks.

## Recommended routed MLP (current size d=384, 6 layers)
- 1 shared expert (hidden 384) + E=64 routed experts (hidden 192), top-k=6 -> active hidden 1536 (= dense), total 12,672
  (8.25x MLP neurons, ~58M MLP params, ~70M total; Adam ~1 GB). E=128 (~16x) later.
- Gating: sigmoid affinity, top-k on score + bias, gate weights renormalised over chosen k (bias excluded).
- Balancing: aux-free u=1e-3, z-loss 1e-3, sequence aux ~1e-4 backstop. Optional 1-2 null experts.
- Routing per token per layer; torus = only an index with a weak prior (test after flat MoE passes).
- Implementation: sort tokens by expert; Path A torch grouped_mm (bf16, SM>=80; sm_120 support NOT confirmed -> microtest first);
  Path B capacity padding C = ceil(1.25 N k / E) -> (E,C,d) -> torch.bmm (always works; log drop rate); dense+mask = unit-test reference.
  Expect routed tokens/s 65-85% of dense (estimate). ScatterMoE (Triton) if triton-windows works.

## Experiment (~1.5 GPU-hours), analog SSD, seq 512, same tokens (~60-80M), seed 0
A dense 1536 | B shared 384 + 64x192 top-6 aux-free | C = B with aux loss 0.01 | D coarse 8x1536 top-1 |
E dense 12,288 (upper bound) | F = B + graded spikes + routed-only firing EMA.
Metrics: val loss, tokens/s, memory, usage entropy/logE, dead experts, max/mean load, drop rate, bias spread, routing Jaccard, spike health per expert.
Go: B beats A by >= 0.03, entropy > 0.9, <= 5% dead experts per layer, tokens/s >= 70% of A. Fallback: E=16 top-2 + shared.
