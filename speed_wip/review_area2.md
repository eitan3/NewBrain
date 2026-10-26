# Speed review, area 2: memory writes / MemOpt, checkpointing, probes and eval, host syncs

Scope: `fruit-fly-sim/memory.py`, `loop_brain.py` (`_hier_step`, `run`), `train_lm.py` (step loop, eval, loop_report),
`spiking_lm.py` (`RoutedMLP.forward` → `pmem`), `brain_opt.py`. No code was changed. Every number is labelled:
**[A]** analytic, **[CPU]** measured on the CPU (Ultra 9 275HX, torch 2.11, `CUDA_VISIBLE_DEVICES=-1`; on Windows an
empty value does not hide the GPU). Scripts: `speed_wip/area2/bench_after_step.py`, `speed_wip/area2/bench_balance.py`.

GPU used for the analytic times: RTX 5090 Laptop (24 GB, ~896 GB/s peak). I assume about 650 GB/s is reachable for
streaming kernels, about 25 TFLOP/s fp32 without TF32, and about 80 TFLOP/s bf16. All of these are lower while the
GPU is shared (see section 7).

Run facts used (from `runs/wiki_mem_12k`, the same config): N = 8,192 tokens per micro-batch, router k = 6,
capacity 1.5, head_groups = 6, writes from step 200. The time per 250 steps was 6.88 min for steps 0-250 (only 50 of
them had writes), then 7.39, 7.33 and 7.73 min. So **steady state with writes is about 1.77 s/step, not 1.72.** The
1.72 s smoke had no table writes. The balancing counts in `mem_balance` were:

| step | shared splits / dead | private splits / dead |
|---|---|---|
| 250 | 707 / 0 | 5,160 / 0 |
| 500 | 866 / 4,273 | 5,818 / **39,225** |
| 750 | 845 / 2,707 | 6,563 / 14,784 |
| 1000 | 313 / 2,965 | 7,258 / 19,231 |

Per-step multiplicities [A]:
- E[K] = 6.4.
- Shared reads with gradient per micro-batch ≈ 4.5. These are the passes K-bptt..K-2 that address, plus read0 when K ≤ 6.
- Private reads with gradient per micro-batch ≈ 4.9.
- So per optimizer step (accum 2): **about 9 checkpointed shared blocks and about 10 checkpointed private reads.**

---

## 1. Summary table (per optimizer step, steady state after step 200)

| component | start (6 heads, 192 exp) | final (36 heads, 768 exp) | of which avoidable |
|---|---|---|---|
| `after_step` table sweeps (V_sh + V_e row-Adagrad) | ~30 GB moved, ~45 ms [A] | same, ~45 ms [A] | ~35-40 ms (fused), ~9 ms exact via e_live slicing at start |
| `_adam_keys` + `P_ema` lerp (full Emax) | ~4.8 GB, ~7 ms [A] | same | ~6 ms |
| EMA lerps wq / k1 / k2, usage EMAs, norm stats | <10 MB, <0.1 ms, ~25 launches | same | – |
| MemRead backward (G accumulation, the "write" half of each read) | shared ~6.6 GB × 9 + private ~2.2 GB × 10 ≈ 81 GB, ~120 ms [A] (atomics, could be more) | same | ~60-80 ms (fused bwd) |
| checkpoint recompute, `_hier_step` block | ~8 ms × 9 ≈ 75 ms [A] | same | 10-45 ms |
| checkpoint recompute, `Memory.private` f | ~1 ms × 10 ≈ 10 ms [A] | same | 10 ms (costs ~1.2 GB VRAM) |
| balancing, every 250 steps (infancy only, until freeze at 60%) | **~10-20 s per call → 40-80 ms/step amortised** (8.6 s CPU-measured Python alone at step-500 counts) | ×~4 keys → up to 40-80 s per call, 160-320 ms/step amortised if dead rates hold | almost all |
| memory probes at every eval (A, B, offset rerun of B) | ~2 s → ~8 ms/step [A] | same | 4-6 ms |
| eval total (20 × 32 seq at K=6 + probes) | ~10-12 s → ~45 ms/step [A]; about half of it is memory cost | same | – |
| loop_report memory knockouts (5 times per run) | ~12 s each, <1 ms/step amortised | same | – |
| `torch.save` every 1000 steps (2.1 GB file, ~1.9 GB memory tables) | ~3-5 s → 3-5 ms/step [A] | same | 2-4 ms |
| host syncs added per normal step | **0** (first write step: 2; balance step: ~80k) | same | – |
| kernel launches added per step (my paths) | ~300 after_step + ~550 MemRead bwd + ~200 private + recompute ≈ 1.3k [A, CPU op counts] | same | – |

At steady state, my area accounts for roughly **0.25-0.35 s of the ~0.94 s/step memory overhead**. That overhead is
1.77 s minus 0.83 s, and the rest belongs to area 1 (addressing sorts, reads). The biggest single surprise is the
**balancing loop**. It is not visible in s/step at small counts, but it is a 10-20 s stall every 250 steps, and with
up to 768 experts it gets worse.

---

## 2. Component details

### 2.1 `Memory._row_adagrad` (V_sh 1,048,576 × 256; V_e 393,216 × 256)
Per 131,072-row chunk, the full table goes through these operations:
- `g != 0`, `.any(1)`
- `g*g`, `.mean(1)`
- `-lr*g`, `/ s.sqrt()`, `.norm`, `dv * clamp`
- `v + dv`, `torch.where(act, v+dv, v)`, `v.copy_`, and finally `G.zero_()`

That is **about 20.5 table-sized passes** [A]:
- V_sh: T = 1.07 GB → 22 GB.
- V_e: T = 0.40 GB → 8.3 GB.
- Together about **30 GB/step ≈ 45 ms** at 650 GB/s.
- FLOPs are negligible (~10 per element ≈ 3.7 GFLOP), so it is purely bandwidth-bound.
- Launches: 233 aten ops for V_sh plus ~90 for V_e [CPU op count]. No host syncs, except on the first write step (`bool(act.any())` and the `median` for S0).

V_e is swept at **Emax = 768 experts even when only 192 are alive**. Rows of unborn experts never get a gradient, so
75% of that sweep does nothing at the starting size.

CPU timings (ms; exactness checked bit-for-bit against the current code):

| table, touched fraction f | current | touched-rows (index once) | dense without the two `where`/`copy` |
|---|---|---|---|
| V_sh f=1.0 | 815 | 762 | 528 |
| V_sh f=0.3 | 836 | 316 | 593 |
| V_sh f=0.05 | 1007 | 117 | 590 |
| V_e f=1.0 | 453 | 275 | 206 |
| V_e f=0.3 | 294 | 123 | 214 |
| V_e f=0.05 | 247 | 28 | 171 |

All variants are **bit-identical** (V, S, wcount, G). Why the dense variant without `where` is exact: on an inactive
row g = 0, so S += +0.0 and V += -0.0. Both leave every finite value unchanged.

How many rows are touched? The plan says "almost every row is written". The logs say otherwise early on:
- Rows never written by step 250: 46-61% per level.
- By step 1000: 4-8%.
- The median write count at step 1000 is 44 over about 800 write steps.

So **the touched fraction per step is well below 1**, probably about 5-30% for the shared table (44 writes / 800 steps
≈ 5.5% of steps per median row). This is the number the profiler run must confirm. The private table is close to
f = 1 over its live rows: there are about 1.6M hits/step on 98K live rows, and 0% were unwritten by step 1000.

### 2.2 `_adam_keys` (P: 393,216 × 64 at Emax) and the EMAs
- `_adam_keys`: ~33 ops over the full Emax tensors, about 45 passes of T = 100 MB → **4.5 GB ≈ 7 ms** [A].
  - CPU: 139 ms at E = 768, 30 ms at E = 192 (4.6× less, exact).
  - The rows of unborn experts have G = 0, so `where` keeps them, yet they are swept anyway.
- `P_ema.lerp_`: 0.3 GB ≈ 0.5 ms. It is exact to restrict it to live experts: lerp(a, a, w) = a, and unborn rows of P and P_ema stay equal, including after copy-on-split.
- wq_ema (2.75 MB), k1/k2_ema (1.6 MB each), norm stats, U (24×512) and U_e (768×512) together: <10 MB, about 25 launches, no syncs.

### 2.3 MemRead backward (where G is filled: the cost of the write path)
**Shared**, per call. There are 98,304 bags (N·L·H) × 6 rows × 256:
- 6 × [`g*w` tmp 100 MB w + r, `index_add_` atomic RMW 200 MB, source read 100 MB]
- plus 6 × [`F.embedding` gather 100 MB r + 100 MB w, product, sum]
- **≈ 6.6 GB per call and ~25 launches** [A].
- × 9 calls ≈ 59 GB ≈ 90 ms, more if fp32 atomics on hot rows serialise. The collision logs show 5-22 probes per row, so hot rows exist.

**Private**, per call. There are about 49K kept slots × 4 rows: ≈ 2.2 GB; × 10 ≈ 22 GB ≈ 34 ms.

A fused backward needs about 1.9 GB per shared call instead of 6.6. It reads g, gathers the V rows once for dw, and
atomically adds w·g into G.

### 2.4 Activation checkpoints (what is recomputed)
**`LoopBrain._hier_step`**, one block per pass. Backward reruns everything except `select`, which is cached in `cache["sel"]`:
- `n_s`
- `ConceptHier`: 15.8 GMAC bf16 ≈ 0.4 ms
- `prep`: a and a_hat, 2 × 59 MB fp32
- `online`: `_queries` 5.6 GMAC fp32 without TF32 ≈ 0.5 ms; `_key_scores` ×2 materialises (L,H,N,7,64) fp32 = 176 MB each, ≈ 2 GB of traffic
- `read_shared`: embedding_bag gathers 600 MB, plus gates and W_o
- `margin_loss`: a second `_queries` (another 5.6 GMAC) and 2 more `_key_scores`, ≈ 1.2 GB

Totals [A]:
- About **23 GFLOP fp32 + 32 GFLOP bf16 + ~5 GB of traffic ≈ 8 ms per block**, so **~75 ms/step** recomputed. The same costs again in the original forward.
- Without the checkpoint, the saved activations would be about 1.5-2 GB per pass × 4.5 passes ≈ 7-9 GB. That does not fit: the peak is already 16.5-16.7 GB on a shared 24 GB card.

**`Memory.private` f**, recomputed per call:
- GatherRows of P (50 MB), normalize, q1[t_k] gather, softmax
- embedding_bag over V_e (200 MB), `index_add` into z, W_p
- **≈ 0.6 GB ≈ 1 ms**; ×10 ≈ 10 ms/step [A]
- Dropping this checkpoint would keep about 250 MB per pass → **~1.2 GB more peak**.

Not recomputed: the private indices (the bmm with P_ema and the full `stable_topk` sort of 73,728 × 512 per pass,
which is area 1). They are outside `f`, which is correct.

### 2.5 Balancing (every 250 steps until the freeze)
`plan_balance` on CPU is cheap: 2.4 ms for the shared table, 21 ms at 192×512 and 80 ms at 768×512 [CPU].

The cost is the **per-key Python loops in `balance()`**. Each private dead key does:
- `bool(lastq_ok[e])`, which is a sync
- `torch.randn(64).to(Pv)`, an H2D copy
- `torch.tensor([ri], device=…)`, another H2D copy
- about 12 indexed writes

Each shared dead key does `bool(q.any())` (a sync), and every split or dead key calls `opt_reset`, which indexes with
a Python list (another H2D copy) and computes a `median` over the whole (4,3,512,64) state.

CPU-measured at the step-500 counts (864 / 3,408 shared splits / deads, 5,760 / 33,408 private):
- **8.6 s of pure Python and dispatch** (unprofiled), with about **2.0M aten ops and 80K `item()` calls** [CPU].
- On the GPU, each of the about 80K blocking calls adds a round trip of about 30-100 µs on WDDM, so a GPU balance at these counts costs about **10-20 s**.

This matches the extra time seen between steps 250 and 500: about +30 s versus steps 0-250, of which writes explain
only about 5-10 s. At 768 live experts the key count is 4× larger: tens of seconds per balance.

### 2.6 Probes and eval (every 250 steps)
Cost unit [A]: one micro-batch forward pass (16 seq × 1 pass) ≈ 40 ms. This comes from 1.77 s spread over about 43
units per step: forward 6.4 passes + 3 × 5 backward-with-recompute units, × 2 micro-batches.

**`evaluate`**: 20 batches × 32 seq × K=6 = 240 units ≈ **9-10 s**. The memory roughly doubles the cost per pass, so about 4-5 s of it is memory.

**`MemoryProbes.measure`**: 3 forwards of 32 seq at K=6 (set A, set B, and B again at a random head-group offset because n_groups = 6 > 1), ≈ 36 units ≈ 1.5 s, plus record overhead:
- a second `mem.gaps` per read, which is another `_queries` + `_key_scores`
- about 25 `.cpu()` copies per forward (a, rows, w, g, gaps ×3 per read; experts and prows per pass)
- per-pass private `rt[fpos]` sort

Then CPU-side `compare`, `torch.unique` for collisions (about 180K pairs per level), and `usage_stats`, which runs a
quantile over the 1M wcount values and makes 8 `.item()` / `.tolist()` calls. **≈ 2 s per eval → ~8 ms/step (0.45%).**
After the freeze, `compare` runs against ref as well, and `repair` (off by default) would add a `.nonzero().tolist()`
plus a Python loop.

**`loop_report`**, at 25/50/75/100% (4 times per run): about 540 pass-batches of 32 seq ≈ 43 s each. The memory
knockouts (shared, private, and 4 levels × val(6) × 4 batches) are 144 of those ≈ 12 s. **<1 ms/step amortised.**

**`think_diag` / `tick_losses` / `halt_eval`**: only inside `loop_report`, so the same amortised figure applies.

### 2.7 Host syncs inside a normal training step (memory code)
**None.** The design avoids them: `_sync_host` mirrors, a host-known chunking in `stable_topk`, no `nonzero`, and
margin stats kept as tensors. The only exceptions:
- first write step: `bool(act.any())` and the `median` for S0 (×2 tables)
- balance steps: about 80K syncs (2.5)
- `.cpu()` in the probe records (eval only)

The pre-existing syncs per step, outside my area, still run:
- `counts.tolist()` in `RoutedMLP` once per pass, about 13/step
- `sample_K` / aux_k / group offset `.item()`
- the loss `.item()`

### 2.8 Kernel launches (Python loops in these paths) [A + CPU op counts]

| path | launches |
|---|---|
| `_row_adagrad` | ~29 per chunk × (8 + 3) chunks ≈ 320 (233 measured for V_sh) |
| `_adam_keys` | 33 |
| EMAs / usage | ~25 |
| MemRead backward | shared ~25 × 9, private ~20 × 10 ≈ 430 |
| GatherRows backward | 1 × 10 |
| recompute blocks | about 60-80 ops × 9 + about 20 × 10 ≈ 900 |
| **total** | **≈ 1.7k per step** |

At about 8 µs per launch, that is ~14 ms of CPU issue time. It is hidden behind GPU work except where a sync drains
the queue (the `counts.tolist()` once per pass). The balancing adds about 2M launches every 250 steps.

---

## 3. Options

Gains are per optimizer step against ~1.77 s/step (steady state, starting size) unless stated. "Exact" means
bit-identical results.

| # | option | expected gain | exactness | risk | VRAM | effort |
|---|---|---|---|---|---|---|
| 1 | **Vectorise `balance()`**: build all split/dead index tensors once (plan on CPU → 1 H2D copy), do batched `index_copy_` / `index_put_` for P, P_ema, V_e bands, S, Adam rows and U; draw the noise as one `randn(n, 64, generator=gen)`; replace `bool(q.any())` / `bool(lastq_ok[e])` with masks; compute the `median` once per codebook | 10-20 s per balance → **40-80 ms/step (2-4.5%)** at the start, more as experts grow; also removes about 80K syncs | exact only if the RNG draw order is kept (draw per-key noise in the same order from `gen`, which is easy: one (n,64) `randn` gives the same stream as n successive (64,) draws on CPU). Splits that touch the same key twice in one plan need sequential semantics; `plan_balance` never reuses a key, so it is safe | low-medium (existing test 228 covers copy-on-split; add an old-vs-new equality test) | 0 | 0.5-1 day |
| 2 | **Restrict private sweeps to live experts**: `_row_adagrad(V_e[:e_live*R], …)`, `_adam_keys` and `P_ema.lerp_` on `[:e_live]` | ~9 + ~5 ms ≈ **14 ms (0.8%)** at the start, falling to 0 at 768 experts | **exact** (unborn rows have G = 0 and P = P_ema) | very low | 0 | 1 h |
| 3 | **Drop the `torch.where` + `copy_` passes in `_row_adagrad`** (add dv in place; S += mean(g²)) | 20.5 → 15.5 passes: **~11 ms (0.6%)** | **exact** (CPU-verified bit-identical) | very low | slightly less | 1 h |
| 4 | **Touched-row write**: `idx = (G != 0).any(1).nonzero()` once per table (1 sync, placed right before the loss `.item()` that syncs anyway), then gather / update / scatter only those rows and `index_fill_` G to zero on them. A cheaper variant: a `touched` bool mask set inside MemRead backward (`index_fill_`) instead of reading all of G | at f = 0.1-0.3 for V_sh: 45 → ~12-20 ms, so **~25-33 ms (1.5-2%)**; at f → 1 almost nothing (CPU: 815 → 762 ms at f = 1, → 117 ms at f = 0.05) | **exact** (CPU-verified) | low. The sync costs about 1 ms of bubble; the gains shrink as the table fills (f rises over training) | the gathered copy is f × 1.07 GB of temporaries (chunk it) | 0.5 day |
| 5 | **Fused row-Adagrad + Adam kernel** (Triton): read G, S, V once, write V, S, wcount, and zero G only on active rows | ~45 + 7 → ~9 + 1 ms: **~40 ms (2.3%)**. Combined with #4: ~5 ms | exact if the op order is replicated (fp32 `div`/`sqrt` are IEEE in Triton; avoid fast-math `rsqrt`) | medium: **no triton in the venv and no cl.exe**, so torch.compile is unavailable. It needs `pip install triton-windows` (a download, which needs the user's OK) or a hand-written CUDA extension (needs MSVC). Without that, #3 + #4 get most of the gain | 0 | 1-2 days plus toolchain |
| 6 | **Fused MemRead backward** (one kernel: dw from gathered rows and atomic w·g into G; or at least one `index_add_` over all 6 columns using a flattened (rows, g⊗w) instead of 6 separate ones, and `torch.bmm` for dw on gathered rows) | shared 6.6 → ~2-3 GB per call, private 2.2 → ~1 GB: **~50-70 ms (3-4%)** | near-exact (the atomic summation order already varies between runs; dw summation order changes) | medium (area 1 owns the read; coordinate) | less (no 100 MB temporaries) | 1 day (torch-only version: 0.5 day) |
| 7 | **Less recompute in `_hier_step`**: (a) the margin uses q computed from `a_hat.detach()`; give it a custom function that reuses the online q values and only returns dwq, saving 1 fp32 `_queries` per run (×2 for recompute); (b) a fused gathered-dot `_key_scores` (q·K[idx] without materialising (L,H,N,7,64)); (c) margin on a 1/4 token subsample | (a) ~9 ms exact-ish; (b) ~25-35 ms near-exact, and also makes not-checkpointing the cheap parts possible; (c) ~20 ms during infancy only, **changes the model** | (a) exact values, gradient summation order only; (b) near-exact; (c) changes the model | (a) low; (b) medium; (c) low | (b) frees about 0.7 GB per pass of transient memory | (a) 0.5 day; (b) 1 day; (c) 1 h |
| 8 | **Turn off the private-read checkpoint** (`mem.ckpt = False`) | ~10 ms (0.6%) | exact | VRAM: **+~1.2 GB** at a 16.7 GB peak on a shared 24 GB card. Only if other apps leave room | +1.2 GB | flag exists |
| 9 | **Probes less often** (every 2nd or 4th eval; offset rerun only every 4th eval; skip set A before the freeze, since it is only used for repair and A-stability after the freeze) | 8 → 2-4 ms/step (**0.2-0.35%**) | exact for training; less frequent metrics | none | 0 | 1 h |
| 10 | **Cheaper eval**: 20 → 10 eval batches, or eval every 500 steps | ~22 ms/step (1.3%) | training exact; noisier val | none | 0 | flag |
| 11 | **Checkpoint file**: exclude rebuildable / zero buffers (`G`, `G_e`, `G_P` are already non-persistent; `P_ema` must stay); save every 2000 steps or asynchronously (copy to pinned memory, write from a thread) | 2-4 ms/step | exact | low | +2 GB pinned host RAM for async | 0.5 day |
| 12 | **CUDA graphs** | see 4.1: not feasible for the step; feasible only for `after_step` (static shapes, no syncs) | exact | after_step is bandwidth-bound, so graphs save only launch issue time (~2 ms) | small | 0.5 day, low value |
| 13 | **Remove syncs** | normal steps: nothing to remove from memory code. Balance: see #1. Eval: batch the probe `.cpu()` copies into one per forward (minor) | – | – | – | – |

**Recommended order:** #2, #3 (exact, about 1 hour each, ~25 ms together) → #1 (largest, removes the 10-20 s
stalls) → #4 → #6 / #7a → #5 if a Triton toolchain is approved. The realistic total for this area is about
**0.10-0.15 s/step (6-8%)**, plus removing the balance stalls, which are worth 2-4.5% amortised and more at larger
expert counts.

---

## 4. Specific questions

### 4.1 CUDA graphs feasibility
- **The whole training step: no.** Shapes depend on the data. `RoutedMLP` sizes C and `n_keep` from `counts.tolist()` every pass. K is random per step (2-16). The bptt window changes the graph structure. The checkpoint recompute and the custom autograd functions with Python-side `ctx.V`/`ctx.G` would also need capture-safe handling.
- **`Memory.after_step`: yes.** Fixed shapes, no syncs after the first write step (S0 must be set before capture), and no allocation once warmed up. But it is bandwidth-bound: a graph saves only about 300 launches ≈ 2-3 ms of CPU issue time, which is already overlapped. **Low value.**
- **`balance()`: not as written** (data-dependent loops). After #1 it is a handful of batched ops, so it no longer matters.
- **Probe forwards: no** (same as the step).

### 4.2 Effect of the GPU being shared with other apps
- The memory paths are **bandwidth-bound** (sweeps, gathers, atomics). Any other GPU client, including the compositor, a browser or a second training process (`nvidia-smi` showed 19.8 GB used and 74% utilisation from the queue), takes bandwidth and SM time slices. WDDM time-slicing between contexts adds stalls of about 1-10 ms whenever another context has work.
- **VRAM headroom** is the binding constraint for #8 and for any reduction of the `_hier_step` checkpoint. Peak allocated 16.5-16.7 GB plus other apps leaves very little of 24 GB. If VRAM spills to shared system memory (the WDDM "shared GPU memory" fallback), steps can slow down 2-10×. Check `mem_reserved_gb` against the Task Manager "shared GPU memory" reading.
- **Sync-heavy code suffers most:** every `item()` waits for the queue to drain, and under contention that wait includes the other process's slice. This makes balancing (about 80K syncs) much worse on a shared GPU than the estimates above.
- Benchmarks must be run with the GPU otherwise idle. Report variance across at least 3 runs of 20 steps.

### 4.3 What a 20-step `torch.profiler` run should confirm
Run steps ≥ 200 so writes are on, include one step where `_step % 250 == 0`, and use
`record_shapes=True, with_stack=False`, with `torch.cuda.synchronize()` around the timed region.
1. **The touched fraction f per table per step.** Log `(G != 0).any(1).float().mean()` for G and for `G_e[:e_live*R]` just before `after_step`. This decides whether #4 is worth more than #3.
2. **`after_step` GPU time** (an NVTX range around it): expected 45-55 ms. If it is much higher, the GPU is contended or the 20-pass count is off.
3. **MemRead backward time and atomic throughput**: the `index_add_` kernel time (CUDA `indexFuncLargeIndex`) per call, expected 6 × ~1.5 ms for shared calls. If it is much higher, hot-row atomic contention confirms #6.
4. **Recompute share:** CUDA time inside `CheckpointFunction` / the second forward of `block` and `f`, expected ~75 ms and ~10 ms per step.
5. **Balance step**: wall time of `balance()` and the count of `cudaStreamSynchronize` / `cudaMemcpy` calls. Expected about 10-20 s and about 80K at the step-500 counts. This also confirms the size of the stall.
6. **Syncs per normal step**: the count of `cudaStreamSynchronize` / `cudaMemcpyAsync D2H`. It should equal the no-memory baseline (about 13 `counts.tolist()` + about 4 sampling / loss `.item()`).
7. **Kernel launch count per step**, memory against no-memory (expected +1.5-2k), and GPU idle gaps (CPU-bound stretches) between passes.
8. **Peak memory** with and without `mem.ckpt` (private), to decide #8.
9. One eval step: the split between `evaluate`, `probes.measure` and `usage_stats` (expected about 9 / 2 / 0.05 s).

---

## 5. Caveats
- The analytic GB/s and FLOP/s assume an uncontended GPU. The CPU timings are ratios and exactness checks only, not GPU times.
- Balancing on the GPU was not measured (GPU use was forbidden). 8.6 s is the CPU-only Python and dispatch floor; the GPU estimate adds about 80K blocking transfers.
- The run log supports a 1.77 s/step steady state after writes start. The 0.83 s no-memory figure was measured from another run.
