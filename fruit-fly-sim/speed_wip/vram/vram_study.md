# VRAM study: world_mem_life_12k at full size

This covers the looped LM at its end-of-growth size: `runs/world_mem_life_12k/last.pt` (step 7000, 35 heads and 740 experts) grown to 36 heads and 768 experts, with the optimizer state carried over the way neurogenesis does it. Settings: batch 16 × accum 2, seq 512, bptt 6, K up to 16, run with the exact flags from `config.json`. One GPU process was run at a time, with `set_per_process_memory_fraction(0.9)` (a cap of 23.09 GB, because 24463 MiB is 25.65 GB) and 2 CPU threads. No model code was edited. The what-if variants are monkeypatches in `variants.py`.

Scripts are in this folder: `vram_common.py` (loader and an exact copy of the train_lm step), `profile_step.py`, `profile_report.py`, `profile_time.py`, `analyze_snapshot.py`, `variants.py`, `check_chunk_ce.py` and `run_all.sh`. Raw results are in `step_profile_*.json`, `report_profile*.json`, `log_*.txt`, `snapshot_K10*.pickle` and `*.analysis.txt`. GB means 1e9 bytes.

## 0. Where the 23.1 GB came from

| point in the run | allocated (GB) | reserved (GB) |
|---|---|---|
| after load (persistent state) | 4.70 (+0.63 grads and opt state once stepping) | 5.6 |
| training steps K = 6/10/16, steady state | peak 20.1-20.25 | 20.5-21.1 |
| **`evaluate()` right after training** (eval_batch 32, K = 6, no_grad) | peak 8.71 | **23.05** (hit the cap) |
| free_cache, loop report, probes | peak 6.9 / 6.2 | 8.2-8.5 |
| next training steps | peak 20.15 | 20.55 → 21.1 |

The log line's `res 23.1GB` is `memory_reserved()` read straight after `evaluate()`. The training cache, about 20.9 GB of blocks sized for the training step, cannot serve eval's batch-32 tensors. Those are 0.54-1.07 GB (B,T,V) logits and log-probs. So the allocator stacks new segments on top of the cache. train_lm frees the cache around the report and probes, but not around `evaluate()`. Pressing on the cap also slows eval itself: `evaluate()` took 49.7 s near the cap and 21.1 s otherwise, because of allocator retries.

The real training step peaks at **20.1 GB allocated** whatever K is: K = 6 gives 20.12, K = 10 gives 20.14, K = 16 gives 20.25. K = 2 gives 12.07 and K = 4 gives 16.16. That works out to **about 2.0 GB per backpropagated pass × bptt 6**.

Measured speed with this harness: **8.4k tok/s**, the expected value over the K distribution (95% uniform on 2-10, 5% on 12-16). The run itself logged 8-9k.

## 1. Ranked consumers at the training peak (K = 10, batch 16, 2nd micro-batch)

The peak is in the last pass's read-out (`_halt_parts`) of micro-batch 2. These figures come from the memory snapshot (`snapshot_K10.analysis.txt`), grouped by model line. Per-pass figures are the 6-pass totals divided by 6.

| # | consumer | GB at peak | notes |
|---|---|---|---|
| 1 | **activations of the 6 backprop passes** | **~12.1** (2.0 per pass) | breakdown below |
| 1a | SSD mixer: scan, q/k/v, convs, mixer spikes | 3.8 (0.63/pass) | `ssd_scan` 52/54/57/61, lines 324-339, Spike 203/204, conv 251 |
| 1b | routed experts: dispatch buffer, up/down bmm, ExpertSpike | 3.0 (0.50/pass) | spiking_lm 672/675/677, 492-494 |
| 1c | RMSNorm fp32 copies (5 norms per pass + workspace norms) | 1.7 (0.28/pass) | spiking_lm 110/111 |
| 1d | workspace read/write, slot write, inject | ~1.0 (0.17/pass) | loop_brain 199-208, 242, 785 |
| 1e | concept heads (ConceptHier outputs kept for the loss) | 0.9 (0.15/pass) | memory.py 308/311 |
| 1f | router scores and wave terms | 0.45 (0.08/pass) | spiking_lm 603/612/613 |
| 1g | memory queries, write-back | 0.4 | memory.py 643, 330 |
| 2 | **logits / CE over the 16K vocab (deep-sup + halting read-outs + final CE)** | **2.7** transient at the peak, plus 0.27 bf16 logits and 0.54 saved log-probs for the final CE | current and previous pass (B,T,V) fp32 log-probs (1.07), and halting `p`, `p·lp`, `lp−prev` (1.6). The deep-sup read-out is already checkpointed. |
| 3 | memory tables and their buffers (persistent) | **3.54** | V 1.09 + **G 1.09** (dense preallocated grad buffer), V_e 0.40 + **G_e 0.40**, P / P_ema / P_m / P_v / G_P 0.10 each; S_acc, wcount, lam_r, psi_r 0.017 |
| 4 | concept loss at the end of the forward | 0.9 | memory.py:276, 6 passes × 4 levels of fp32 intermediates (the next peak once #2 is fixed) |
| 5 | BrainOpt state | 0.60 | Muon momentum 0.49, Adam m/v 0.11, counts 0.003 |
| 6 | parameters | 0.55 | 137.7M; experts 0.45 |
| 7 | grads from micro-batch 1, alive during micro-batch 2 | 0.55 | |
| 8 | optimizer-step transients | +1.9 (peak 7.5) | well below the training peak, so not binding |
| 9 | eval / probe transients | evaluate +3.4 (peak 8.7), loop report +1.6, probes +0.9 | not binding by allocation, but **binding by reserved** (section 0) |
| 10 | allocator fragmentation (reserved − peak allocated) | 0.4-1.0 during training; **+2.9 after evaluate()** | `expandable_segments` is **not supported on Windows** (tested: "expandable_segments not supported on this platform") |

Memory-table write density per step: 570k of 1.06M shared rows are touched (54%), and 143k of 393k private rows (36%).

Speed profile (`log_ptime.txt`, K = 6): about 1.76 s of GPU time in a 2.1 s step. The step is **GPU-bound on elementwise kernels** (mul, copy, elementwise, sum, div: over 50%). GEMMs are only about 20%, and there are 48 host syncs costing about 0.2 s.

## 2. Same-math savings

"meas" means measured on the full-size model. "est" means an estimate. Speed is expected tok/s over the K distribution; repeat runs vary by about ±3%.

| saving | peak alloc (GB) | Δ GB | reserved, training (GB) | tok/s | Δ speed | how known / notes |
|---|---|---|---|---|---|---|
| baseline 16×2 | 20.14 | — | 20.5-21.1 (23.05 after eval) | 8.4k | — | meas |
| **fused chunked CE** (`chunk_ce`: linear + log-softmax + CE + halting entropy/KL in 2048-token chunks; previous pass kept as bf16 logits; backward recomputes per chunk) | 18.92 | −1.2 | 20.8 | 8.2k | −2% | meas. Forward bit-identical; dW/dh within bf16 rounding (`check_chunk_ce.py`). The peak moves to the concept loss (#4). |
| + checkpoint the concept loss (its 6×4 level terms) | ≈18.0 | −0.9 more | — | ≈0% | — | est (cheap elementwise recompute) |
| **micro-batch 8 × accum 4** | **13.16** | **−7.0** | **13.5-13.8** | **8.0k** | **−4.5%** | meas. Same effective batch. The capacity C and the EMA norm statistics come per micro-batch, so not bit-identical. |
| micro 8×4 + chunk_ce | 12.81 | −7.3 | 15.2 | 7.9k | −6% | meas (more fragmentation than 8×4 alone) |
| **per-pass activation checkpointing** + chunk_ce, 16×2 | **10.36** | −9.8 | 12.3 | 7.4k | −12% | meas with homeostasis off (see note). An implementation must make the recompute side-effect-free: homeostasis edits `alpha.data` in place during the forward (the recompute then routes differently; it failed the checkpoint metadata check); balancer `pending`, `pend_e` / `lastq`, `last_idx` are touched twice; the routing `.tolist()` sync runs again. |
| ckpt_pass + chunk_ce, 32×1 | 14.23 | −5.9 | 14.8 | 7.3k | −13% | meas: a larger micro-batch does **not** buy speed (GPU-bound) |
| homeostasis off (reference for the ckpt row) | 20.13 | 0 | 21.1 | 8.2k | ≈0 | meas |
| Muon momentum in bf16 (`bf16_mom`) | −0.25 | −0.25 | | | ≈0 | est. 8-bit gives −0.37. Not strictly same math. |
| Adam m/v bf16 or 8-bit | −0.05 / −0.08 | | | | 0 | est, negligible |
| memory tables V / V_e in bf16 with an fp32 master only for touched rows | −0.75 GPU (master on pinned CPU, 1.5 GB host) | | | | −3 to −5% | est: about 0.7M touched rows × 1 KB × 2 directions ≈ 1.4 GB PCIe per step ≈ 60-100 ms. **Reads then see bf16-rounded V, so not same math.** Master kept on the GPU for touched rows: about 0. |
| sparse / compact grads instead of dense G, G_e | ≤ −0.5 (G) − 0.26 (G_e) | | | | −1 to −3% | est: 54% / 36% of rows are touched per step, so a compact buffer plus a row→slot map (unique over about 12M row-writes) saves little. Dense is right at this density. |
| G in bf16 | −0.75 | | | | 0 | **not same math** (index_add accumulation error), not recommended |
| CPU-pinned offload of cold rows (UVA gather) | up to −1.5 | | | | −15 to −25% | est: usage is near-uniform (busy 0.02, dead 0.02), so there is no cold set. Each read gathers about 0.6 GB per micro-batch over PCIe, about 0.3-0.5 s per step. |
| **empty_cache before evaluate()** (and after it, as already done for the report and probes) | 0 alloc | — | eval spike 23.05 → about 9 (eval on an empty cache) | | ≈0 (one 0.1-0.3 s re-grow per eval) | meas: free after eval drops 22.1 → 12.4 → 8.5. This is the direct fix for the 23.1 GB. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | — | — | — | — | — | **not supported on Windows** (warning, falls back to normal segments) |
| per-process fraction cap, e.g. `set_per_process_memory_fraction(0.78)` for 20.0 GB | 0 | hard cap on reserved | ≤20 | 0 while there is headroom | | meas side effect: near the cap the allocator frees and retries, and eval ran 2.4× slower. Use as a guard only, after the peak is lowered. |
| smaller eval chunks (eval_batch 32 → 16 or 8; probes already 8) | eval peak 8.7 → ~7.0 / ~6.1 | | stops the eval/train shape mismatch | | +few s per eval | est. With eval_batch 16 the eval tensors match the training block sizes. |

## 3. Recommendation

**Least speed loss, comfortably ≤ 20 GB reserved:**

1. **micro-batch 8 × accum 4.** This gives a peak of 13.2 GB allocated and 13.8 GB reserved in training, for −4.5% speed (measured).
2. **`torch.cuda.empty_cache()` right before `evaluate()`.** The trainer already frees around the report and probes. Optionally use eval_batch 16 too. This removes the 23 GB eval spike. Eval then runs on about 5.3 GB of state plus 3.4 GB of transients, about 9 GB.
3. **A guard cap** of `set_per_process_memory_fraction(20e9 / total)` (about 0.78) so reserved can never pass 20 GB.
4. Optionally add **chunked fused CE and a checkpointed concept loss**. This saves about 1-2 GB more for about 2% speed. Worth it if the new blob memory needs the room, otherwise skip it: with 8×4 it raised fragmentation (15.2 GB reserved vs 13.8).

Expected result: about **14 GB peak reserved in training, about 9 GB in eval, at about 8.0k tok/s**. That leaves about 6 GB of headroom under the 20 GB target and about 10 GB under the card.

Keep **per-pass activation checkpointing** in reserve. It reaches 10.4 GB at 16×2 for −12% speed. It is the lever if the blob memory plus a bigger model need more than 6 GB. It needs the side-effect guards listed in the table: deferred homeostasis and balancer updates, and no double usage counts.

**On the speed target (≥10k, aim 20k):** none of the memory savings buys speed. The step is about 83% GPU-busy and dominated by elementwise kernels: SSD scan math, RMSNorm fp32 casts, spikes, MoE scatter/gather, read-out softmax. Batch size barely matters (8×4 vs 16×2 vs 32×1 are within 10%). Reaching 10-20k needs kernel fusion instead: `torch.compile` of `Core.forward` / `ssd_scan`, a fused RMSNorm and spike, and fewer fp32 round trips. The 48 host syncs (about 0.2 s, roughly 10%) are a second target. The VRAM headroom from step 1 is what makes compile or fusion workspaces affordable.

## 4. Which savings apply to the new per-pass blob memory (~1M slots × 64-d keys + values, per-pass banks)

Size per 1M-slot bank, fp32:

| part | dv = 64 | dv = 256 |
|---|---|---|
| keys (64-d) | 0.26 | 0.26 |
| values | 0.26 | 1.02 |
| dense grad buffers (keys + values) | 0.51 | 1.28 |
| key Adam m/v | 0.51 | 0.51 |
| EMA keys | 0.26 | 0.26 |
| row-Adagrad S | 0.004 | 0.004 |
| **total** | **≈1.8 GB** | **≈3.3 GB** |

With K-many separate 1M-slot banks (10-16) this does not fit on the GPU at all. Either split 1M slots over the banks, or keep only the active banks resident.

| saving | applies to the blob memory? |
|---|---|
| compact / sparse grads instead of dense preallocated buffers | **Yes, high value** if a pass touches ≪50% of its bank (likely with per-pass banks). The grad buffer then scales with touched rows, not slots. |
| row-Adagrad (one scalar per row) for values, not Adam | **Yes** (already the MemOpt design): saves 2× the value size |
| bf16 / 8-bit Adam state for the keys | **Yes**: −0.26 / −0.38 GB per 1M slots |
| bf16 values read copy with an fp32 master on pinned CPU or for touched rows only | **Yes** if the bf16 rounding of reads is acceptable (not bit-same): −50% of values |
| CPU-pinned offload of idle banks, async prefetch of the next pass's bank (and again in reverse for backward) | **Yes, the main lever for per-pass banks.** One bank at ~25 GB/s takes about 40 ms (dv 64) to 130 ms (dv 256), overlappable with a ~200 ms pass. Not useful for the current shared pool, whose usage is uniform. |
| chunked key scoring (never build N × 1M scores: 8192 × 1M bf16 = 16 GB) | **Yes, mandatory**: product keys or budgeted chunked top-k (like `SELECT_BUDGET`) |
| checkpointing its per-pass read block | **Yes** (as `_hier_step` already does) |
| empty_cache before eval, fraction cap, micro-batch 8×4 | Yes (global) |
| chunked CE, expandable segments | No (CE is unrelated; expandable segments are unsupported on Windows) |

## Method notes

- One full step runs exactly as in train_lm lines 1345-1411: K forced, deep supervision, halting, concepts, memory reads and writes, clip, BrainOpt, `apply_balance`, `write_counts`, `mem.after_step`. Route noise 0.3, lam 1, `write_on`.
- Peaks per phase use `reset_peak_memory_stats` between phases. The snapshot was taken with `_record_memory_history(stacks="python")` and replayed in `analyze_snapshot.py`.
- `ckpt_pass` was measured with homeostasis off. Homeostasis off alone changes speed by ≤3%, which is within noise.
- Timings are the median of 3 steps per K, interpolated over the sampled-K distribution.
