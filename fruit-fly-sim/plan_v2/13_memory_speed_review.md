# Memory speed review (no implementation) — 2026-10-23
Merged from three independent CPU-only reviews (nothing in fruit-fly-sim / runs changed, no GPU used):
- area 1 — addressing, reads, private memory: speed_wip\review_area1.md
- area 2 — writes / MemOpt, checkpoint recompute, probes / eval, host syncs: speed_wip\review_area2.md
- area 3 — design-level options + op counts + a step-1000 checkpoint probe: speed_wip\review_area3.md
Tags: [M-GPU] measured on the GPU, [M-CPU] measured on the CPU, [A] analytic (about ±40%).

## 0. What we know for sure
- Plain v4d at the starting size: 0.83 s/step [M-GPU, smoke]. Memory model: 1.72 s/step in the 20-step smoke, which had no table writes;
  the live run wiki_mem_12k gives ≈ 1.77 s/step once writes are on (7.3-7.7 min per 250 steps vs 6.9 min for the first 250) [M-GPU, log].
- So the memory overhead ≈ 0.94 s/step at the starting size.
- torch.compile / Triton are NOT available here: no triton package in the venv (a download), and the CPU path has no cl.exe on PATH.
  Every fused-kernel option is blocked unless Eitan approves installing triton-windows.

## 1. Where the time goes
Two views, which disagree on the cause; a GPU profile settles it (section 4).
- Op-count view (area 3) [M-CPU op counts on the real LoopBrain]: 9,356 ops per micro-batch without memory, 22,674 with memory
  (K = 6), 18,194 after the freeze. About +22.5K ops/step → 0.89 s / 22.5K ≈ 40 µs per op, the same per-op cost as the baseline
  (0.83 s / 19.1K ≈ 43 µs). On Windows about 25 µs of that is host / launch time, and the per-pass host sync in RoutedMLP
  (counts.tolist()) stops the GPU from running ahead. Estimated pure GPU work only 0.33-0.38 s/step. So the overhead is mostly
  "too many small ops", and the ops that pay off to remove are whole ops, not FLOPs.
- Work view (area 1 + area 2) [A]:

| Component (per step, starting size) | ms | notes |
|---|---|---|
| Shared sub-key full stable sort (512 → top 9), 4 levels × 3 heads × 2 halves × N=8192, every read | 120-240 | full sort; CPU: 83 → 24 ms per level-half with topk [M-CPU] |
| Shared read backward (6-column index_add_ loop + 6× F.embedding for dω) | 120-180 | atomics into hot rows |
| Online sub-key gathers (embedding of normalised keys) | 85-110 | |
| Private full sort over (E, C, 512) incl. empty slots | 70-120 | CPU: 213 → 25 ms with topk on kept rows [M-CPU] |
| Checkpoint recompute (_hier_step ≈ 8 ms × 9 + private ≈ 10 ms) | ≈ 85 | removing the hier checkpoint needs +7-9 GB → no |
| Margin / gaps pass (second fp32 query matmul, 17% of memory ops) | 50-60 | |
| Table writes: row-Adagrad on V_sh + V_e (≈ 20 full-table passes, ≈ 30 GB) | ≈ 45 | V_e swept for all 768 experts while 192 are alive |
| fp32 query einsums / concept heads | 75-105 | |
| Private-key Adam + EMA lerps | ≈ 7 | full 768-expert tensors |
| Balancing every 250 steps (Python loops, ≈ 80K .item(), ≈ 2M aten ops) | 40-80 amortised | 8.6 s on CPU at step 500 [M-CPU]; 10-20 s stall on GPU [A], matches +30 s in the 250-500 log block |
| Eval memory part + probes (A, B, offset rerun) + knockouts at reports | ≈ 25-30 amortised | evaluate ≈ 9-10 s / 250 steps, half memory |
- Host syncs: none added to a normal step; the first write step adds 2, a balancing step ≈ 80K.
- Size: memory cost barely depends on model size (E·C ≈ 73.7K at 192 and 768 experts; heads are batched). At the final size the
  freeze also removes the EMA / margin / balancing work: frozen memory ≈ 0.64 s/step instead of 0.89 [A, area 3].
  Projected final step ≈ 1.9 s (base v4d ≈ 1.25 s at final size from the v4d/v4e throughput drop; the biggest uncertainty).

## 2. Options, ranked
Exactness: E = bit-identical / same math, NE = near-exact (summation order, ties), C = changes the model.

### 2a. Same math (engineering)
| # | Option | gain s/step | exact | risk | VRAM | effort |
|---|---|---|---|---|---|---|
| 1 | topk(k) + small stable re-sort instead of the full sort, shared sub-keys | 0.08-0.20 | E on tie-free data; keep the lowest-index rule with a boundary-tie fallback | low | − | S |
| 2 | Private top-4 on kept slots only (not the whole (E, C, 512) incl. empty) via topk | 0.06-0.11 | E | low | − | S |
| 3 | Vectorise balance() (no per-key Python loop / .item()) | 0.04-0.08 + removes the 10-20 s stalls | E if the RNG draw order is kept | low | 0 | M |
| 4 | Margin computed inside the online pass (reuse q, gather the boundary keys once) | 0.04-0.07 (infancy) | NE | low | 0 | S |
| 5 | MemRead backward: one flattened index_add_ + dω by one gather+bmm | 0.03-0.08 | NE (atomic order) | low | small + | S |
| 6 | Row-Adagrad on touched rows only (one sync next to the existing loss .item()); sweep V_e / P / P̄ over live experts only; no torch.where/copy_ passes | 0.04-0.06 | E (bit-identical on CPU) | low | 0 | S |
| 7 | Batch the per-level Python loop in select() | 0.02-0.05 (launch count) | E | low | small + | S |
| 8 | After the freeze reuse the EMA addressing for the weights (online = EMA then) | 0.01-0.015 | E | low | 0 | S |
| 9 | Probes every 2nd eval (flag), fewer eval batches | 0.005-0.03 | training E | none | 0 | S |
| — | Fused Triton kernels (pk scoring+topk, embedding-bag read+bwd, row-Adagrad+Adam) | 0.1-0.2 more | NE | med | − | L — BLOCKED (no triton) |
| — | CUDA graphs | only the after-step part, ≈ 2-3 ms | E | — | — | not worth it (data-dependent shapes) |
| — | bf16 scoring + fp32 recheck | small after #1 | NE | med | − | not worth it |
- Total same-math package (#1-#9): ≈ 0.35-0.6 s/step of the 0.94 s overhead → ≈ 1.2-1.4 s/step at the starting size
  (≈ 1.45-1.6 s at the final size). If the op-count view is right, #1, #2, #4, #7 (whole ops removed) carry most of it.

### 2b. Changes the experiment (needs Eitan)
| Option | gain s/step (start / final) | quality / stability | constraint |
|---|---|---|---|
| Margin on 1 sampled pass | 0.13 / 0 (frozen) | margin isn't widening gaps yet anyway | ok |
| New addresses every 2nd pass (reads still every pass) | 0.09 | stability better | ok |
| Calmer balancing (fewer re-seeds per event) | 0.02-0.06 | less address churn (step 1000: one event re-seeded 17% of private keys, 24% of shared sub-keys) | ok |
| Re-address only tokens whose thought moved (capped) | 0.03-0.06 | best for stability | ok |
| H = 1 (or 2) heads per level | 0.05-0.12 / 0.04-0.09 (half for H=2) | quality loss expected (Lample 2019, PEER); less redundancy | ok |
| S = 256 | 0.02-0.06 | frees 1.6 GB; small quality loss | ok |
| top-k 4 | 0.02 | small | ok |
| Fewer reads per forward (5 / 3 / 2) | 0.16 / 0.38 / 0.5 | | breaks "read on all passes" |
| L = 3 / L = 2 levels | 0.12 / 0.25 | | breaks the 16/32/64/128 hierarchy |
| Private memory off / only last 3 passes | 0.12 / 0.06 | private ≈ 0 value at step 1000 (see 5) | breaks shared+private / all passes |
| Memory on 50% of tokens | 0.05-0.1 | | partly breaks |
| Smaller query (64-d), shared translator, cheaper level heads, bf16 tables | ≤ 0.03 each | | not worth it |
- Package (a) mild, no constraint touched: sampled margin + addresses every 2nd pass + calmer balancing ≈ −0.25 s.
  With the same-math package: ≈ 1.05-1.3 s/step at the start, ≈ 1.5-1.6 s at the final size.
- Package (b) aggressive, constraints kept: (a) + H = 2 (or 1) + S = 256 + top-k 4 → ≈ 1.0-1.1 s start, ≈ 1.4-1.5 s final.
- Package (c) breaks constraints (L = 2, two reads, no private): ≈ 1.0 s start, ≈ 1.45 s final before engineering — not recommended.

## 3. Recommendation
1. Same-math package #1-#9 first (≈ −0.35-0.6 s/step, results identical up to atomic summation order). Low risk, small effort.
2. Then package (a) if Eitan agrees: it also IMPROVES stability (less churn, fewer re-addresses).
3. Hold (b) until a later checkpoint (step ≥ 4,000) shows whether the thought-keyed reads gain value and whether cross-pass address
   overlap stays ≥ 0.6.
Estimated result: ≈ 1.2-1.4 s/step (same math) → ≈ 1.05-1.3 s/step with (a), vs 1.77 s now (−25% to −40%).

## 4. GPU profile to run (20 steps, torch.profiler) — in the gap after wiki_mem_12k, before the control starts
Run past step 200 (writes on) and include one balancing step. It should confirm:
- launch-bound vs work-bound: GPU idle gaps, op count, host time per op, the effect of the per-pass counts.tolist() sync;
- the sort / topk time, the MemRead backward (index_add_ contention), the gathers, the recompute time (≈ 85 ms expected);
- after-step time (≈ 45-55 ms expected) and the real touched fraction of G (5-30%?) and of the live part of G_e;
- balancing wall time and sync count (10-20 s, ≈ 80K expected);
- that a normal step has no extra syncs; peak memory with and without the private-read checkpoint;
- how much Eitan's other GPU apps slow the bandwidth-bound parts.

## 5. Findings beyond speed (for Eitan)
From the live run's step-1000 checkpoint, eval-only, early and weak evidence (area 3):
- Shared reads off: +0.093 nats; private memory off: ≈ 0.
- Almost all of the shared value comes from the PASS-0 read, which is keyed by the token alone (s_0 depends only on the token):
  keeping only read0 costs +0.0005; dropping read0 and keeping the thought-keyed reads costs +0.085. So far the memory acts mostly
  as a token-keyed lookup, which goes against "addresses only from thoughts". Proposal: report read0 separately in the knockouts,
  and decide whether --mem-read0 should be off.
- Address overlap between consecutive passes: 0.006 (read0 → read1), then 0.35 → 0.59 → 0.66 → 0.70 as the thought settles.
  The three heads share no rows; the top-4 rows carry 81-88% of the read weight.
- Reusing an address when the thought moved < 0.3-0.5 (50-69% reuse) did not change the loss.
- Balancing churn is high: one event at step 1000 re-seeded 17% of private keys and 24% of shared sub-keys.
