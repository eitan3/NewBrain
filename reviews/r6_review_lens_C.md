# r6 review, lens C: probation/writes (3), search/read (4), speed (6)

Read-only review of `fruit-fly-sim/blob_write.py`, `blob_memory.py` and the read paths in `blob_gate.py`, checked against the live run `world_blob25d_12k`. The memory was fitted at step 300 and writes started at step 301.

How the findings were checked:
- CPU snippets (2 threads, no GPU).
- The logs of the live run (read only).
- The `world_blob25d_12k_r5_stopped/last.pt` checkpoint (step 500, same code, G = 262144), loaded on the CPU with mmap.

## Findings by severity

| # | Severity | Where | Summary | Affects live run |
|---|---|---|---|---|
| C1 | **corrupts results** | blob_memory.py:396-397, 407-411; blob_write.py:455-458 | The int8 sketch's per-row scale is stored in fp16 and underflows to 0, so **86% of waiting entries have a zero sketch**. Their cosine is 0, so they can never be promoted. | yes, from step 301 |
| C2 | **corrupts results** | blob_memory.py:323-330, 1159-1160, 1368-1372 | Padding in the global shortlist is aliased to **slot 0** and counted as a valid candidate. A token with no real neighbour reads slot 0 at full strength (use_A 0). | yes, from the first spawn (step ~301) |
| C3 | corrupts results (moderate) | blob_write.py:171-193, 260-264; blob_memory.py:1065-1069, 1166 | `xfull` (the model-width 384-d copy) is never moved by merges or renewals. The 384-d re-rank and the gauss read use the spawn position while the 64-d key has moved: median travel 2.0, 51% of blobs moved > 1. | yes, from the first merge or renewal |
| C4 | corrupts results (moderate) | blob_memory.py:1068-1069; blob_gate.py:407, 501-504, 569, 636-640 | With rerank_full, `gd` is returned in **384-d order** but holds 64-d distances, so it is not ascending. The gate's match features, w1/ratio/gap in the confirmation tests, and the sigma2_g tracking all assume ascending order. | yes, from step 301 |
| C5 | minor (follows from C2) | blob_memory.py:1170-1175; blob_gate.py:630-635 | Slot-0 duplicates give gap = 0 or junk gaps in the running median `m` of the gauss kernel. | early steps |
| C6 | speed / stability | blob_memory.py:464-494 | `_exact` uses n = min(ptr, n). The size of the last block and of `gmin` change **every step** until the store is full (~step 3,250). This fragments the allocator (r4 died with 13 GB reserved but unallocated), and `--frag-gb` then forces `empty_cache` syncs. | yes, steps 301 to ~3,250 |
| C7 | speed | blob_memory.py:577-579 (called at blob_write.py:450) | `pr.age(step)`: `torch.as_tensor(<python int>, device=cuda)` is a blocking H2D copy, i.e. a second host sync per step. | yes |
| C8 | speed | blob_write.py:485-486; blob_memory.py:893-897 | The global spawn/merge (search, cdist, victims) runs on **all 1,664 candidates** although at most 64 can be active. The IVF workspaces swap between the read size (N = 8,192) and the write size (1,664) twice per step, about 300 MB each time. | yes |
| C9 | speed / memory | blob_memory.py:1065-1066 | The re-rank gathers `xfull[idx]` as fp32 with shape (8192, 64, 384), about 0.8 GB, plus the same size again for the difference. This happens on every gated pass, about 12 times per step. | yes |
| C10 | latent breaks run | blob_write.py:597-605 | `torch.quantile` on `_yring` fails above 2^24 elements (checked: "quantile() input tensor is too large"). That is about 6.4M rows per 250 steps today, so 2.6x headroom. With eval_every ≥ ~650 or a higher explore rate it raises at the metrics call. | not at eval 250 |
| C11 | minor | blob_write.py:603 | The bucketize warning comes from the values argument `yr[:, 0]` (a strided column), not from `q[1:-1]`. It is eval-only. Fix: `yr[:, 0].contiguous()`. | eval only |
| C12 | minor | see the minor list below | Small leaks, logging errors and doc drift. | partly |

## C1: the int8 sketch scale underflows (the main write-path bug)

```python
# blob_memory.py:392-397
def _q(self, x):
    if self.qdtype == "fp16":
        return x.to(torch.float16), None
    s = x.abs().amax(-1).clamp(min=1e-12) / 127.0
    return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8), s.to(torch.float16)   # <- fp16 scale
# blob_write.py:455-458
old = pr.sketch_f(j)                                       # int8 * fp16 scale
cos = F.cosine_similarity(Q["sketch"][:, :pr.kd], old, dim=-1)
...
promoted = hit & (cos >= bm.promo_theta) & ...
```

The sketch is `(-G) @ gproj`. Here G is a per-token gradient: the loss is the mean over 32,768 tokens, so the sketch coordinates are about 1e-6. That makes `amax/127` about 1e-8, which is below the smallest fp16 subnormal (5.96e-8), so the stored scale becomes 0. The sketch then dequantises to the zero vector, its cosine with any query is 0, and the entry can never pass `cos ≥ 0.3`.

**Evidence:**
- **r5 checkpoint (step 500):**
  - `pgrad_scale == 0` for **85.8%** of the 318,128 occupied entries.
  - The 90th percentile of the scale is 5.96e-8, i.e. 100% of scales are subnormal.
  - `key_scale` is fine: 0% zero.
- **Logs:** `gm_promo_cos_med = 0.0` exactly in every int8 run, including the live one at step 500.
- **CPU check:** element std 1e-6 → scale [0, 6e-8, 6e-8, 0], so cos(original, dequantised) = 0 for 2 of the 4 rows. At 3e-7, all 4 rows go to 0.

**Effect:**
- Promotion only works for the roughly 14% of candidates whose gradients are large enough. This is a size-biased subset, and it is why the budget is not filled (~37 of 64 per step).
- A recurring context whose entry is dead is pushed again on every return, so it fills the FIFO with duplicates.

**Fix:** the cosine is the sketch's only use. Push `F.normalize(pgrad)` (then the scale is about 1e-3), or keep the scale in fp32 or bf16. Apply the same guard to the keys for safety.

## C2: shortlist padding aliased to slot 0 is read as a valid neighbour

```python
# blob_memory.py:323-330  rerank()
safe = cand.clamp(min=0)                     # -1 (no candidate) -> slot 0
d2 = torch.where((cand >= 0) & ok[safe], d2, torch.full_like(d2, INF))
top = d2.topk(...)
return safe.gather(1, top.indices), top.values          # id 0 with d2 = inf
# blob_memory.py:1159-1160  _read()
keys = space.key.view(-1, self.kd)[flat].to(q.dtype)
valid = space.occ.view(-1)[flat]             # occ[0] is True -> padding counted as valid
# blob_memory.py:1166 (gauss, full): d = |qf - xfull[0]|^2 (finite)  -> w uniform over the copies
# kernel_weights: A = 1 for any valid row when use_A=False
# blob_memory.py:1370  _count(): ok = occ[gi[:, 0]] & m  -> slot 0's uses / touch for every empty token
```

**Failure:** take a token whose 16 probed cells hold fewer than k_g = 8 blobs.
- **No blob at all:** the token reads `u = v_0` with A = 1, so slot 0's content is injected at full strength (blended 50/50 with the private read).
- **Some blobs:** extra copies of slot 0 enter the softmax.

**Knock-on effects:**
- Slot 0 receives gradient (through BagRead), credit and strength from all of these tokens.
- Slot 0 is "touched" every step, so it can never be evicted.

**CPU repro:** one blob in slot 0 and five queries in other cells give `gi = [0,0,0,0]`, every `gd = inf`, `A = 1.0` and `|u| = |v_0|`.

**Checkpoint (step 500):**
- slot 0 has 46,193 uses, against a median of 629.
- Its strength is 0.036, against a 99th percentile of 0.0045, which ranks it **2nd of 7,128**.
- The live run logs `gm_A_g = 1.0`.

**Fix:** have `rerank` return -1 where d2 is inf, and use `valid = (flat >= 0) & occ[flat.clamp(min=0)]` with clamped gathers. Alternatively, AND `valid` with `isfinite(sel["gd"][:, :k])` in `_read`, `_count` and the `_make_rec` jg rows. The write path is already safe because it gates on `d1 < gamma h0²`.

## C3: xfull is stale after merges and renewals

`_merge` moves `K` by up to `min(sigma, 0.5 h0)` = 2, and `renew` moves it toward the queries' mean, but neither touches `bm.xfull`. Searching uses the moved 64-d key, while the top-64 re-rank and the gauss distances (`kernel_dist full`) use the 384-d point from spawn time. The two geometries drift apart: in the checkpoint, the median |key − spawn_key| is 2.0, the 90th percentile is 3.56, and 51% of blobs have moved more than 1 (typical d1 ≈ 9.4, so a distance of about 3.1).

**Fix:** when a merge happens, apply `xfull[j] += eta (xfull_new − xfull[j])`. For a renewal, accumulate the mean qfull of the confirming queries, or re-derive xfull.

## C4: 64-d distances in 384-d order feed the gate and the confirmations

`search_global` (lines 1068-1069) returns `idx.gather(1, top)` and `d2.gather(1, top)`, where `top` is ordered by the 384-d distance. Downstream code treats column 0 of `gd` as d1, the nearest neighbour:
- `_feats` (blob_gate.py:370-378): d1 = dd[:,0] and d2 = dd[:,1], so log(d2/d1) can be negative and log(d2 − d1) clamps to log(1e-6). This contradicts the fixed-sign prior (PRIOR_SIGN).
- `_make_rec`:
  - `w1g` uses the bounded kernel on these 64-d distances, not the gauss weights the read actually used. With dd < d1 this gives z > 1.
  - `ratio_g` can be < 1, so the `conf_margin` test fails spuriously.
  - `gap_g` can be < 0, and it feeds the sigma2_g median, which then drifts down (5.6 for the global space against 9.0 for private).

**Fix:** return the model-width distances (df) as well, and use the same distances as the read for `_feats`, w1, ratio and gap. Alternatively, take w1 and the gap from `kernel_weights`.

## C5: the gauss kernel's running median m

The code is safe when m = 0 or m is tiny: `where(m > 0, m, h0²)` falls back, and the step-end update refuses mq ≤ 0. One quirk: the fallback h0² = 16 is in 64-d units but is applied to 384-d distances. It only matters before the first gap statistics, when softmax is one-hot anyway. The real distortion is C2: while most tokens have no candidate, their copies of slot 0 contribute gap = 0, which suppresses or biases m early on. Live values at step 500: m_g = 23.7 and m_p = 13.4, which look healthy.

## C6–C9: speed

- **C6:** in `_exact`, `n = min(self._ptr, self.n)` changes every step. `lp` (the padded last block), `s` (1024 × last), the `cat` pad copy and `gmin` (Qn × ng) all change size every step, so the caching allocator cannot reuse blocks. r4 OOMed with "13.07 GiB reserved but unallocated"; the live run survives through `--frag-gb 3`, whose `empty_cache` (cudaFree) synchronizes the device. **Fix:** always scan whole kc blocks over a fixed allocation: never-written rows have ksq = inf, clamp to 6e4 and never win. Do the key dequantisation once rather than once per 1,024-query chunk.
  - The cost grows linearly until the ring is full at ~step 3,250. It then holds at about 1.2 TFLOP of fp16 plus ~20 GB of transient traffic per step, roughly 1% of a ~3.3 s step.
- **C7:** `ProbationStore.age` calls `torch.as_tensor(now, device=...)` with a Python int, which is a blocking H2D copy. The intended single sync is at blob_write.py:515. Fix: `bm.now - pr.step.long()`.
  - Other per-step syncs: the intended `.tolist()` and the `--nan-abort 1` check (one per step, by design). I found no `.item`, `.nonzero` or boolean indexing in the per-step path; `diag_snapshot`'s `torch.tensor(edges)` runs only every 250 steps.
- **C8:**
  - **All candidates sent to write_blobs:** `write_blobs` gets all 1,664 rows with `active=gw` (≤ 64). Compact them to a static 64 (argsort[:promo_budget]) before `search_global`, `cdist` and `victims`.
  - **Workspace churn:** cache `_workspace` per shape, e.g. key the name by N.
  - **Repeated table rebuilds:** `assign_cells` triggers a full `rebuild_cells` (two 196K-element argsorts plus a 134 MB gather) up to three times per step (merge, spawn and renewal). One rebuild per step would do.
- **C9:** compute the 384-d re-rank as `|q|² + |x|² − 2 q·x` with a bf16 bmm, or chunk it, instead of materialising (N, 64, 384) in fp32.
- **Per-step cost that grows as the stores fill:** only `_exact`, which is bounded once the ring is full, and `_yring`/`_stats_step`, which grow until the next metrics call (about 260 MB per 250 steps). The IVF tables, tiles, victims and `apply_value_grads` are all fixed size.

## Checked and found correct

- **`_exact` against `_exact_ref`:** the CPU comparison found 0 top-1 mismatches in every case tested:
  - full and partial last block (ptr 257, 700);
  - ptr below M (5);
  - occupancy of 2% and 0.05% (blocks with no occupied entry);
  - duplicate keys and ties;
  - a wrapped ring (ptr 3000 > n);
  - fp16 storage.
  - An empty store returns (Q, M) zeros with d2 = inf, and `step_writes` gates on `isfinite`.
  - The group and stride arithmetic is correct: every strided group contains at least one real entry.
  - Only artefact: when fewer than M entries are written, the ids are clamped to n−1 and duplicated. This only affects the logged margin.
- **Tiled IVF (`q_cap < 0`):** returns exactly the unbounded-q_cap shortlist, with and without `pos`, for 4 and 32 probes. The tile bound and the `searchsorted` mapping skip empty cells correctly.
- **New writes are visible at once:** `_spawn`, `_merge` and `renew` call `assign_cells(slots)`, which recomputes `cell_pref` and rebuilds the table. They do not wait for the 1,000-step refit. Overflow beyond cell_cap is counted (`gm_cell_overflow` = 0 at step 500), but cell_gini is 0.73, so watch it as the store fills.
- **Row-wise Adagrad:**
  - It updates touched rows only, and untouched rows stay bit-identical.
  - The state is reset on spawn (and on eviction), merge and split (both twins).
  - `apply_value_grads` runs before `step_writes`, so a new or evicted slot never receives stale gradients.
  - Design note: per-row normalisation gives rows read with tiny weights (the gauss tail, the slot-0 copies) the same clipped step of 0.05 as heavily read rows.
- **Reads only after the fit:** guarded in `inject`, `inject_gated`, `select_writes` and `step_writes`. `xf_norm_med` is set before the first write.
- **Promotion logic:**
  - top-1 within promo_gamma h0², cos ≥ 0.3, and a budget of 64 ranked by cos.
  - Losers are not consumed; the entry is killed only if the write was actually stored.
  - One promotion per entry.
  - Expiry is by FIFO.

## Minor items (C12)

- **`lost` overflows:** it is int8 and is incremented on every lost budget (blob_write.py:467). Entries that lose more than 127 times wrap negative. This only affects the logged `promoted_after_drop`.
- **Budget applied before the per-entry dedupe** (blob_write.py:461-472): duplicates use up budget slots, so fewer than 64 promotions can happen even when more qualify.
- **The ring wraps before the hold expires:** `n_prob = ceil(1638.4 × 3000)`, but 16 × ceil(0.1 × 512) × 2 = 1,664 candidates are pushed per step. The ring therefore wraps after about 2,954 steps, and the 3,000-step expiry almost never fires.
- **`bm._wstats` is never cleared** (blob_write.py:343): one small CUDA tensor is added per micro-batch for the whole run.
- **Wasted `nearest_cells` work:** `probation_push` computes it for every push even though the lookup is exact (blob_memory.py:441-442, 1091).
- **Doc drift:** the docstrings say content = 0.1 |xf| (blob_write.py:11, 420), but `v_init = 0.2` (logged `gm_v_norm_rel_med` 0.205).
- **fp16 overflow in `_exact` and `_exact_ref`:** extreme x_hat outliers (|q| ≳ 250) can overflow the fp16 dot product and push true neighbours out of the shortlist. Rare.
