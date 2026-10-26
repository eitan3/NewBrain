# Speed review, area 1: addressing, reads and private memory

Scope: the concept heads, the translators and queries, sub-key scoring with its sort, the product-key merge, the
shared read (MemRead forward and backward, gates, W_o), the private memory (dense Q, bmm, sort, read) and the
margin/gaps pass. MemOpt, checkpointing policy, probes, host syncs elsewhere and design-level options belong to
areas 2 and 3. They come up here only where an area-1 option touches them.

Every number is tagged:
- **[A]** analytic: worked out from shapes, the code and stated hardware assumptions.
- **[M]** measured on the CPU.

No GPU was used. All CPU work ran with `CUDA_VISIBLE_DEVICES=-1`. An empty value did not hide the GPU on this
Windows machine: `torch.cuda.is_available()` still returned True, so `-1` was used.

## 0. Setup and assumptions

- **Config** (runs/wiki_mem_12k/config.json):
  - Batch: B=16, T=512, so N = 8192 tokens per micro-batch; accum 2.
  - Model: d=384, ds=64 (p̄), ain = 448; L=4 levels, H=3 memory heads; S=512, m=8, km=6; dv=256.
  - Private memory: R=512.
  - MoE: top-6 experts, capacity 1.5.
  - Passes: K is drawn uniformly from 2..10 (5% of steps draw 12..16), mean ≈ 6.4; bptt 6; read0 on; margin on
    during infancy (the first 60% of training).
  - The core's attention heads (6 at the start, 36 at the end) do **not** appear anywhere in area 1. The memory has
    its own H=3.
- **Starting size vs final size.** In area 1 only the private memory depends on the expert count.
  - E·C ≈ capacity · N · k = 1.5 · 8192 · 6 = 73,728 slots at both sizes (192 × 384 and 768 × 96). So the dense
    Q, the bmm and the score tensor are the **same size** at the start and at the end.
  - Only normalize(P_ema[:E]) grows: 25 MB at 192 experts, 100 MB at 768.
  - So the absolute area-1 cost per step is about the same at both sizes. Its share of the step is larger at the
    start, because the core is cheaper then.
- **Calls per micro-batch at K=6.** Every pass is inside the bptt window, so read0 has a gradient too.
  - Concept heads: 7 calls (read0 + 6 passes).
  - Addressed calls (select + online + shared read + margin): 6 (read0 + passes 0..4).
  - Private calls: 6 (passes 0..5).
  - With the current whole-block checkpoint, each gradient call runs its forward twice (forward plus recompute),
    except `select`, which is cached.
  - Private `f` is checkpointed as well, so it also runs twice.
  - **Per step = 2 micro-batches: 12 addressed calls, 14 concept-head calls, 12 private calls.**
  - At K=10 there are 10 addressed calls per micro-batch, 5 of them with gradient, so the costs are similar.
- **GPU** (Win32_VideoController; read through WMI, the GPU was not touched): RTX 5090 Laptop, 24 GB.
  - Bandwidth: 896 GB/s peak; I assume about 650 GB/s achieved for streaming and somewhat less for scattered
    gathers and atomics.
  - FP32 SIMT GEMM: ~20-25 TFLOPS achieved. Addressing runs in fp32 with TF32 off.
  - bf16 tensor cores: ~100 TFLOPS achieved.
  - Each aten op is roughly one CUDA launch, at ~8-10 µs of CPU time on Windows (WDDM).
- **CPU measurements**: Core Ultra 9 275HX, 8 threads, full size (N = 8192), median of 3 runs.
  - The training queue was running at the same time, so expect ±20% noise.
  - CPU times are only relative weights. Sorts, gathers and GEMMs do not speed up by the same factor on a GPU.
  - Script: `C:\Users\eitan\AppData\Local\Temp\claude\C--Eitan-Python-NewBrain\de301b72-951d-49a4-9895-252fb20d74d6\scratchpad\bench_area1.py`
- **Launch counts** were measured with a TorchDispatchMode that counts non-view aten ops [M].
- **Host syncs**: none of the area-1 components syncs in the training path [A, from reading the code].
  - `select`, `online`, `read_shared`, `margin_loss` and the private read all have zero `.item()`, `.tolist()`,
    `nonzero()` or boolean-index calls in training.
  - `.cpu()` appears only in probe recording (eval).
  - The only sync next to area 1 is RoutedMLP's `counts.tolist()` (one per pass). The private read runs right
    after it, while the GPU queue is nearly empty.
- **The 1.72 s vs 0.83 s smoke** ran with `--mem-write-start 200` and only 20 steps. So it had **no table writes**:
  MemRead's G `index_add_` and MemOpt were off. Real steps after step 200 cost more than the smoke shows: about
  +80-120 ms [A] in area 1 alone, from the G accumulation in the read backward.

## 1. Cost per component

Per call means one call at N = 8192 (forward only unless stated). Per step means ×12 addressed calls (×14 concept
heads, ×12 private), counting forward + recompute + backward.

### 1.1 Concept heads (`ConceptHier.forward`, memory.py)

- **FLOPs [A]**: trunk 1024→256, level 0 256→768, levels 1-3 (256+384)→768.
  - 15.84 GMAC forward per call; backward ≈ 2× forward.
  - Per step: (2 forwards + 1 backward) × 14 calls = 1.77 TFLOP, in bf16 under autocast.
  - Add inject write_back: 1.6 GMAC per call.
- **Bytes [A]**: ~0.3 GB per forward, mostly the fp32 `.to(acc)` copies of mu and nu (4 × N × 768).
  ~13 GB per step.
- **Launches [M]**: 26 forward, 93 forward+backward. ~1,670 per step with the recompute.
- **CPU [M]**: 82 ms forward, 246 ms forward+backward.
- **GPU estimate [A]**: 30-45 ms per step.

### 1.2 Translators and queries (`_queries`, memory.py)

- **FLOPs [A]**: one einsum `lnc,lhoc->lhno` is 5.64 GMAC (11.3 GFLOP, fp32).
  - Per micro-batch: EMA 6 + online forward/recompute 12 + online backward 12 (da_hat and dW_q) + margin
    forward/recompute 12 + margin backward 6 (dW_q) = 48 einsums.
  - Per step: 96 einsums = 1.08 TFLOP fp32.
- **Bytes [A]**: q (L,H,N,128) is 50 MB, plus three normalizes ≈ 0.25 GB per call.
- **Launches [M]**: 10 per call.
- **CPU [M]**: 36 ms per call.
- **GPU estimate [A]**: 45-60 ms per step. This is fp32 GEMM-bound, because TF32 is off.

### 1.3 Sub-key scores and the full sort (`select`, `stable_topk`)

**Score GEMMs**
- **FLOPs [A]**: 8 bmm calls per select (L levels × 2 halves), 6.44 GMAC fp32 in total.
- **Bytes [A]**: they write the score tensor, 2·L·H·N·S = 100.7 M floats = 403 MB.
- **CPU [M]**: 33 ms.
- **GPU estimate [A]**: ~0.6 ms of compute + 0.6 ms of writes per select, so 12-15 ms per step.

**Full stable sort**
- **What it does**: 8 sorts per select, each over 24,576 rows of 512.
- **Bytes [A]**: ≈ 4 GB of traffic per select. `torch.sort` returns the *full* sorted values (403 MB) and int64
  indices (805 MB), sorts them in place, then slices the first 9 columns. On top of that comes the radix-sort work
  in shared memory.
- **Launches [M]**: the whole select is 98 ops.
- **CPU [M]**: 83 ms per level and half, so 666 of the 807 ms of a select (82%).
- **GPU estimate [A]**: 10-20 ms per select, so **120-240 ms per step**. This is the largest single uncertain
  item.
- **VRAM**: transient peak ≈ 50 MB scores + 50 MB values + 100 MB indices per level and half.

**Product-key merge (`pk_topk`)**
- **What it does**: for 98,304 rows × 64 candidates per select: argsort of int64 flat indices, gathers, then a
  stable sort.
- **Bytes [A]**: ~0.3 GB per select.
- **Launches [M]**: 10 per level, 40 per select.
- **CPU [M]**: 14.7 ms per level.
- **GPU estimate [A]**: 1-2 ms per select, so 12-24 ms per step.

### 1.4 Online weights (`online` → `_key_scores`)

- **What it does**: the online queries (1.2) plus F.embedding gathers of the chosen sub-keys,
  (L,H,N,km+1,64) per half = 2 × 44 M floats = **352 MB**, then a multiply, a sum and a softmax.
- **Bytes [A]**:
  - Forward: ≈ 1.4 GB.
  - Backward: ≈ 1.5 GB plus `embedding_dense_backward`. On CUDA that sorts 2 × 688 K indices and then does a
    segmented reduce.
- **Launches [M]**: 31 forward, 129 forward+backward. ~1,920 per step with the recompute.
- **CPU [M]**: 99 ms forward, 364 ms forward+backward.
- **GPU estimate [A]**: 85-110 ms per step, not counting the query einsums.

### 1.5 Shared read (`read_shared` → `MemRead`, gates, W_o)

**Forward**
- `embedding_bag` gathers N·L·H·km = 589,824 rows of 1 KB = **604 MB** and writes r (100 MB).
- Gates: 31 MMAC. W_o: 3.22 GMAC (bf16).
- **Launches [M]**: 15.
- **CPU [M]**: 83 ms in total, of which 48.5 ms is the embedding_bag.

**Backward with writes on**
- **G**: 6 × [g·w temporary (100 MB write + read) + `index_add_` atomics (100 MB read-modify-write)] ≈ 3 GB.
  CPU [M]: 225 ms.
- **dw**: 6 × [F.embedding gather (100 MB written) + multiply (200 MB read, 100 MB written) + sum] ≈ 3 GB.
  CPU [M]: 126 ms.
- **Total**: ≈ 4.5-6 GB per backward. Launches [M]: 53.
- **After the freeze** (with addr_sg), w has no gradient, so dw is skipped. Only G remains.

**GPU estimate [A]**
- Forward: 1.1-1.5 ms × 24 (forward + recompute) = 26-36 ms per step.
- Backward: 10-15 ms × 12 = **120-180 ms per step**.
- Gates and W_o: ~20-30 ms per step.

### 1.6 Margin / gaps pass (`margin_loss` → `gaps`; infancy only)

- **What it does**: a second query einsum on sg(a), plus 4-slot sub-key gathers per half (201 MB).
- **Bytes [A]**: ≈ 0.8 GB forward, ≈ 1 GB backward plus the embedding backward sort.
- **Launches [M]**: 34 forward, 175 forward+backward. **The most launch-heavy component.**
- **CPU [M]**: 79 ms forward, 281 ms forward+backward.
- **GPU estimate [A]**: 50-60 ms per step, plus its query einsums (counted in 1.2).

### 1.7 Private memory (`Memory.private`)

**Scoring (no gradient, runs once per call)**
- Q = zeros(E, C, 64) (19 MB) plus a scatter.
- normalize(P_ema[:E]): 25 MB (192 experts) or 100 MB (768 experts), recomputed on every call even though P_ema
  changes only once per optimizer step.
- bmm (E,C,64)·(E,64,R): **2.42 GMAC at both sizes**, writing sc (E,C,R) = 37.7 M floats = 151 MB.
- **Full stable sort of every one of the 73,728 rows**, including the ~1/3 that are empty capacity slots, in 3
  chunks: ≈ 1.2 GB of traffic. Only the 49,152 kept rows are then used.

**Read `f`**
- Pk gather: 50 MB. MemRead of V_e: 196,608 rows = 201 MB. index_add into z. W_p: 0.81 GMAC.
- Backward: G_e `index_add_` loop (4 columns) + dw loop (4 × F.embedding) + G_P `index_add_` ≈ 2-2.5 GB.

**CPU [M]**

| piece | E=768 (C=96) | E=192 (C=384) |
|---|---|---|
| dense Q + normalize + bmm | 25.5 ms | 46.8 ms (noisy) |
| normalize(P_ema) alone | 10.4 ms | 4.6 ms |
| **full sort of all rows (current)** | **212.9 ms** | **251.3 ms** |
| topk + re-sort on the kept rows only | 25.5 ms | 21.2 ms |
| same, sync-free (GPU) variant | 40.2 ms | 68.4 ms |
| alternative: per-slot gathered keys, no dense Q | 494 ms | 788 ms |
| whole call forward (current) | 290 ms | 491 ms |
| whole call forward+backward (current) | 466 ms | 918 ms |

The alternative row confirms that the dense Q + bmm design is right. The waste is the sort, not the GEMM.

**Launches [M]**: 34 per call forward, 93 forward+backward.

**GPU estimate [A]**: sort 6-10 ms per call, so **70-120 ms per step**. Everything else ≈ 60-70 ms per step.

### 1.8 Totals

**CPU [M]**, per gradient pass-call:

| component | current code | ms |
|---|---|---|
| select | | 807 |
| online | fwd+bwd + recompute | 463 |
| shared read | fwd+bwd + recompute | 560 |
| margin | fwd+bwd + recompute | 360 |
| concept heads | | 328 |
| prep | | 31 |
| private | | ~506 |
| **total** | | **≈ 3.06 s** |

- Sorts are 666 + 213 = 879 ms of that (29%).
- With the exact topk work already prototyped in speed_wip (select 305 ms, private scoring −188 ms), the total
  is ≈ 2.37 s, **−23%** on the CPU.

**GPU per step [A]**: area 1 ≈ 0.72-0.93 s per step, with writes on and in infancy.

| component | ms per step |
|---|---|
| shared sorts | 120-240 |
| shared read backward | 120-180 |
| online gathers | 85-110 |
| private sort | 70-120 |
| queries | 45-60 |
| margin gathers | 50-60 |
| concept heads | 30-45 |
| everything else | ~100 |

This is the same order as the measured 0.89 s total memory overhead. The smoke had no writes, but it does include
area 2. The uncertainty is about ±40%: sort and atomic throughput on this GPU are not measured. One CUDA profile
once the queue frees up (torch.profiler with CUDA activities, one step) would settle the ranking.

**Launches [M/A]**: ≈ 4,900 aten ops per micro-batch, ≈ 9,800 per step, in area 1.

| component | ops per micro-batch |
|---|---|
| margin | 1,254 |
| online | 960 |
| concept heads | 833 |
| private | ~680 |
| select | 588 |
| shared read | 498 |
| prep | 108 |

At ~8-10 µs each that is 80-100 ms of CPU-side launch time per step. It is mostly hidden while the GPU is busy,
but it gets exposed after every per-pass `counts.tolist()` sync.

## 2. Options

Gains are per training step on the GPU [A], with "%" relative to 1.72 s (the starting size). At the final size
the absolute gains are about the same, so the percentages are smaller. Exactness:
- **exact**: bit-identical indices, values and gradients (verified on the CPU where noted).
- **near-exact**: the same math, but a different floating-point summation order, so last-bit differences. Indices
  can flip only at near-ties of about 1e-7.
- **changes model**: the function itself changes.

### O1. Top-k instead of the full sort, shared sub-keys (A1)

- **What changes**: `torch.topk(k+1)` plus a stable re-sort of the k winners by (score desc, index asc).
  - Ties at the k / k+1 boundary keep the lowest-index rule.
  - CPU path: affected rows fall back to the full sort, which needs one sync.
  - GPU path (sync-free): the boundary-valued entries are replaced by the lowest-index equal entries, found with
    a second top-k over distinct integer keys. No host sync.
- **Where**: memory.py `stable_topk` / `_topk_rows`. **Already implemented in speed_wip.**
- **Gain**: 80-200 ms (5-12%).
  - CPU [M]: 83 → 24 ms per level and half (sync-free variant 33 ms); select 807 → 305 ms.
  - On the GPU it removes ≈ 4 GB of traffic per select and the radix sort. topk on 512-element rows is 1-3
    passes over L2-resident data.
- **Exactness**: exact. Full-size select output is identical [M]. test_04, with heavy ties, passes.
- **Risk**: low.
  - The sync-free GPU path is untested on CUDA, because the GPU was not used.
  - NaN scores would be ordered differently from torch.sort. Scores are finite cosines.
- **VRAM**: lower. The 150 MB value/index sort outputs per chunk disappear.
- **Effort**: done. It still needs the equivalence tests and one GPU timing.

### O2. Top-k over the kept private slots only

- **What changes**: `stable_topk(sc[e_k, p_k], 4)` instead of a full sort of all (E,C) rows followed by
  indexing. The ~1/3 empty capacity slots are no longer sorted.
- **Where**: `Memory.private`. Already implemented in speed_wip.
- **Gain**: 60-110 ms (3.5-6%).
  - CPU [M]: 213 → 25.5 ms (E=768) and 251 → 21 ms (E=192).
- **Exactness**: exact, because the operation is per row.
- **Risk**: low.
- **VRAM**: about −100 MB transient.
- **Effort**: done.

### O3. Product-key merge without the 64-wide argsort

- **What changes**: sort each half's m=8 list by index, so the outer product is already in flat-index order; then
  `stable_topk(sig, km+1)` instead of argsort(64) + gather + stable sort(64).
- **Where**: `pk_topk`. Done in speed_wip.
- **Gain**: 5-12 ms (<1%). CPU [M]: 14.7 → 7.7 ms per level.
- **Exactness**: exact.
- **Risk**: low. **VRAM**: neutral. **Effort**: done.

### O4. Batching the per-level loops (A2)

- **What changes**: one stable_topk over all (half, level, head, token) rows and one pk_topk. The 8 score GEMMs
  stay separate calls with exactly the reference shapes, written into one buffer, so cuBLAS picks the same
  kernels.
- **Grouping**: levels are grouped so the buffer stays ≤ `select_budget` = 2^26 elements. At full size that is 2
  levels per group, the same peak as before.
- **Where**: `Memory.select` → `_select_batched` / `_addr_group`. Done in speed_wip, behind
  `mem.batch_select`.
- **Gain**: <10 ms. The op count does not drop much: 98 → 147 dispatch ops, because the tie-safe top-k adds ops.
  The value is fewer, larger kernels and no Python loop.
- **Exactness**: exact on the CPU [M].
  - On the GPU the score GEMMs are the same calls, so they should be bit-identical.
  - Collapsing them into one `matmul` over batch L·H would **not** be guaranteed bit-identical, because cuBLAS
    heuristics depend on the batch count. That variant was deliberately not used.
- **Risk**: low. **VRAM**: neutral at the default budget; +200 MB if the budget is raised to one group.
- **Effort**: done.

### O5. torch.compile / Triton

**Feasibility on this machine [M]**
- The venv has torch 2.11.0+cu128 and **no `triton` package**. GPU inductor needs Triton, so torch.compile
  cannot generate GPU kernels here.
  - Installing the community `triton-windows` wheel would be a download; it was not tried.
- CPU inductor fails with `InductorError: Compiler: cl is not found`. MSVC 14.38-14.40 is installed under
  `C:\Program Files\Microsoft Visual Studio\18\Community` but is not on PATH. It could work from a vcvars shell,
  but only for CPU, which does not matter for training.
- speed_wip has a `mem.compile_addr` flag (default off) that falls back to eager on failure.

**O5a. Generic torch.compile of the addressing.**
- Even with Triton, inductor lowers `topk` / `sort` to aten. The gain would come only from fusing the
  elementwise work: normalizes, the gather-mul-sum of `_key_scores`, softmax, the gaps arithmetic.
- **Gain**: 30-60 ms (2-3.5%).
- **Exactness**: near-exact (fused reductions).
- **Risk**: medium (dynamic shapes, graph breaks around the custom autograd Functions).
- **Effort**: medium, after Triton is installed.

**O5b. Fused product-key scoring + top-k kernel (Triton).**
- One program per (level, head, half, token-tile): it computes q·k over the 512 sub-keys in registers and keeps
  a running top-9 (with the lowest-index tie rule) in registers. Optionally the 8×8 product-key merge happens in
  the same kernel.
- It never writes the 403 MB score tensor.
- **Gain**: on top of O1, about another 30-60 ms (2-3.5%). Select becomes GEMM-bound: 12.9 GFLOP ≈ 0.6 ms.
- **Exactness**: near-exact. The fp32 dot-product order differs from cuBLAS, so a near-tie can flip a selected
  row compared with the current code.
  - Fine for a new run; not for continuing a frozen run.
- **Risk**: medium-high (custom kernel, Windows Triton).
- **VRAM**: −400 MB transient.
- **Effort**: high (2-4 days including the tie-rule tests).

**O5c. Fused embedding-bag read + backward (Triton).**
- **Forward**: as now.
- **Backward**: per bag, load g once; for each of the km rows load V[r_j] once, compute dw_j = g·V[r_j], and
  atomically add w_j·g into G[r_j].
- Traffic: ≈ 604 MB V reads + 100 MB g + 604 MB of atomics per backward, instead of ≈ 4.5-6 GB.
- **Gain**: 60-100 ms (3.5-6%) with writes on. Smaller after the freeze, when dw is gone. Apply the same kernel
  to the private V_e read for another ~15-25 ms.
- **Exactness**: near-exact for dw. G accumulation is already nondeterministic on the GPU because of atomics.
- **Risk**: medium. **VRAM**: −200 to −600 MB transient. **Effort**: medium-high (1-2 days).

### O6. bf16 scoring with an fp32 recheck

- **What changes**: compute the score GEMM and the top-k in bf16, keep the top m+1+δ candidates (for example 16),
  recompute their fp32 scores exactly, and take the exact top-9 among them.
- **Safety check**: provably exact only with a bound check: the bf16 score of candidate 17, plus 2× the bf16 error
  bound, must be below the fp32 9th score. Rows that fail fall back to fp32 (a sync, or a sync-free masked
  recompute).
- **Gain after O1**: only 10-25 ms (≈1%). The fp32 GEMM is cheap (0.6 ms); bf16 halves the 403 MB write; the
  recheck needs a 16 × 64 gather per row (≈ 0.8 GB per select), which **eats most of the saving**.
- **Exactness**: exact only with the fallback; near-exact without it.
- **Risk**: medium. It complicates the frozen-addressing reproducibility argument.
- **VRAM**: −200 MB transient. **Effort**: medium.
- **Verdict**: not worth doing after O1. Revisit only inside O5b, with bf16 inputs and fp32 accumulation.

### O7. A smaller query dimension (64+64 → 32+32 per head)

- **What changes**: W_q goes from 448→128 to 448→64; sub-keys go from 64 to 32 dimensions.
- **Where**: the hard-coded 64/128 in `Memory.__init__`, `_queries` and `_key_scores`.
- **What it halves**:
  - the query einsums: 45-60 → 23-30 ms;
  - the score GEMMs;
  - the online and margin gathers: 352/201 MB → 176/100 MB.
  - The private query (64) and the private keys would stay at 64.
- **Gain**: 50-80 ms (3-5%).
- **Exactness**: changes the model. It is less address capacity, but 32-dimensional sub-keys over 512 codes are
  still far from saturated. Expect a small or unmeasurable effect on retrieval quality, though that has to be
  tested.
- **Risk**: medium (quality). **VRAM**: −150 to −300 MB of activations.
- **Effort**: low (one parameter), plus a quality A/B run.

### O8. A cheaper private lookup

- **(a) Cache normalize(P_ema[:E]) once per optimizer step.** P_ema changes only in `after_step`. The same applies
  to normalize(k1_ema, k2_ema), with wq_ema read as-is.
  - **Gain**: 3-8 ms. **Exactness**: exact. **Risk**: low (invalidate in `after_step`, `balance`, `on_grow`,
    `freeze` and load). **Effort**: low.
- **(b) Score only the kept slots.** The dense Q wastes 1/3 of the 2.4-GMAC bmm. That is about 1 ms per step:
  **not worth it**. The gathered-keys alternative measured 20× slower on the CPU.
- **(c) Per-expert product keys** (512 = 16 × 32, or 2 halves of 23).
  - Scoring cost: 48 dot products per slot instead of 512.
  - After O2 the whole private scoring is only ≈ 20-30 ms per step, so the gain is ≤ 15-20 ms (≈1%).
  - **Exactness**: changes the model (key factorisation, balancing and copy-on-split logic).
  - **Effort**: medium-high. **Verdict**: not worth it.
- **(d) Hash / LSH addressing.** Keys would no longer be learned. It changes the model a lot and gains about the
  same as (c). **Not recommended.**

### O9. Reuse the EMA queries after the freeze

- **What changes**: after `freeze()` (with `addr_sg=1`), keys.wq == wq_ema bit for bit, and online runs with no
  gradient on the same detached a_hat. So online's q1/q2 equal select's, and `online` can take them from `select`.
- **Gain**: one 5.64-GMAC einsum plus normalizes per forward and recompute, ≈ 10-15 ms per step, for the last 40%
  of training.
- **Exactness**: exact.
- **Risk**: low. Guard it on `is_frozen and addr_sg`.
- **Effort**: low.

### O10. Merge the margin pass into the online pass

- **What changes**: compute the queries once, through a two-output custom op:
  - q, whose gradient reaches a_hat and W_q;
  - q_sg (same values), whose gradient reaches W_q only.

  Then compute the 2 extra per-half boundary slots in the same `_key_scores` gather as the online slots.
  - The pair-boundary scores sig[km-1] and sig[km] are already online slots, so only I1/I2[m-1:m+1] are new.
- **What it removes**: the margin's separate query einsum (forward, recompute and backward dW_q) and half of its
  gathers.
- **Gain**: 40-70 ms in infancy (2.5-4%); 0 after the freeze. It also removes ~800 launches per step, the
  largest launch block.
- **Exactness**: near-exact. dW_q is summed in a different order.
- **Risk**: medium (custom autograd; test 14, the margin scope, must still pass).
- **VRAM**: −100 MB. **Effort**: medium (1 day).

### O11. Sub-key score backward

- **What changes**: F.embedding's CUDA backward sorts 688 K indices per half to stay deterministic. A custom
  autograd with `index_add_` into the small (L·H·S, 64) key-gradient buffer would avoid the sort and the 352 MB
  "grad of the gathered rows" tensor, by accumulating g·q directly.
- **Gain**: 15-30 ms.
- **Exactness**: near-exact (atomic order).
- **Risk**: low-medium. **Effort**: low-medium.

### O12. MemRead backward in pure PyTorch (no Triton)

- **(a) dw with one chunked gather + bmm.** `bmm(F.embedding(rows, V), g[..., None])` in 2-3 row chunks, instead
  of 6 × (gather + multiply + sum).
  - CPU [M]: 126 → 67 ms. GPU [A]: ≈ 3 GB → 1.3 GB per backward.
  - **Gain**: 20-40 ms (1-2%), during infancy only (no dw after the freeze).
  - **Exactness**: near-exact (dot-product order). **VRAM**: +200 MB transient per chunk.
  - **Effort**: low. Apply the same to the private dw (4 columns): +5-10 ms.
- **(b) One flattened `index_add_` instead of the 6-column loop.**
  - CPU [M]: 225 → 180 ms. On the GPU the bytes are the same; it saves about 5 launches per call.
  - **Gain**: <5 ms. **Exactness**: near-exact on the CPU, the same class on the GPU (atomics).
  - **Verdict**: not worth it. Prefer O5c.

## 3. Ranked options (area 1)

| # | option | gain / step (GPU [A]) | % of 1.72 s | exactness | risk | VRAM | effort |
|---|---|---|---|---|---|---|---|
| 1 | O1 topk instead of the full sort (shared) | 80-200 ms | 5-12% | exact | low | lower | done in speed_wip |
| 2 | O2 private topk on the kept rows | 60-110 ms | 3.5-6% | exact | low | −100 MB | done in speed_wip |
| 3 | O5c fused MemRead read + backward (Triton) | 60-100 ms (+15-25 private) | 3.5-7% | near-exact | medium | −0.2 to −0.6 GB | 1-2 days; **blocked: no Triton** |
| 4 | O10 margin merged into online | 40-70 ms (infancy) | 2.5-4% | near-exact | medium | −100 MB | 1 day |
| 5 | O7 query dimension 32+32 | 50-80 ms | 3-5% | changes the model | medium | −0.15 to −0.3 GB | low + A/B run |
| 6 | O5b fused product-key scoring + top-k (Triton) | +30-60 ms beyond O1 | 2-3.5% | near-exact | med-high | −400 MB | 2-4 days; **blocked: no Triton** |
| 7 | O12a dw by gather + bmm (shared + private) | 25-50 ms (infancy) | 1.5-3% | near-exact | low | +200 MB (chunked) | low |
| 8 | O11 sub-key score backward without the sort | 15-30 ms | 1-2% | near-exact | low-med | −350 MB | low-med |
| 9 | O9 reuse EMA queries after the freeze | 10-15 ms (last 40%) | ~1% | exact | low | 0 | low |
| 10 | O3 pk_topk without argsort(64) | 5-12 ms | <1% | exact | low | 0 | done |
| 11 | O4 batched levels | <10 ms | <0.5% | exact | low | 0 (budgeted) | done |
| 12 | O8a cache the normalized EMA keys per step | 3-8 ms | <0.5% | exact | low | 0 | low |
| 13 | O6 bf16 scoring + fp32 recheck | 10-25 ms after O1 | ~1% | exact with fallback | medium | −200 MB | medium; skip |
| 14 | O8c/d private product keys or hash | ≤ 15-20 ms after O2 | ~1% | changes the model | med-high | − | medium-high; skip |

What to expect if they are combined:
- The **exact** set (O1 + O2 + O3 + O4 + O9 + O8a) ≈ **160-350 ms per step (9-20% of 1.72 s)**. That is roughly
  20-40% of the 0.89 s memory overhead.
- Adding the cheap near-exact items (O10, O12a, O11) brings it to ≈ 250-500 ms.
- The Triton kernels (O5b, O5c) add ≈ 100-160 ms more, but need Triton on Windows.

## 4. State of speed_wip (left as it was when the task changed)

**Implemented**
- In memory.py, all behind module and instance flags:
  - `memory.TOPK_FAST`: O1 and O3.
  - O2 (private top-k over the kept rows; always on).
  - `mem.batch_select` and `mem.select_budget`: O4.
  - `mem.compile_addr`: off by default; falls back to eager.
  - `mem.fused_adagrad`: an area-2 prototype: active-row-only MemOpt with chunks padded to the reference shape.
- In loop_brain.py: `model.mem_ckpt_mode = "mem"`, an area-2 checkpoint prototype.
- In train_lm.py: `--mem-probe-every` and `--mem-ckpt-mode`.

**Tests**
- `test_memory.py` without test_01/01b: **17 passed** in speed_wip.
- test_01/01b and test_spiking_lm.py were not re-run after the change of task.
- No new equivalence tests were added, and `speed.diff` was not written.

**Not touched**: nothing in fruit-fly-sim\ or runs\ was edited; no GPU was used.
