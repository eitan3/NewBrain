# Plan 25c: per-pass blob memory, overview

Status: plan only. Nothing is built. The detailed version, with all the math, is being written to
`25c_blob_only_per_pass_plan.md`. Older versions (25, 25b rev1/rev2) are kept but superseded.

## 1. What the memory is
- The memory is **only blobs (splats)**, one blob space **per pass**. There is no hash/row table, no coarse pool and no
  private expert memory. The concept hierarchy stays, because it isn't memory.
- A blob = a centre (key, 64-d), a content vector (value, 256-d), plus bookkeeping (strength, age, use count). The
  width h0 is shared per bank (one number per pass), not per blob.
- Banks: 9 (passes 0-7 each have their own; bank 8 is shared by the rare long-thinking passes 8+).
- Size: about 1.5 KB per blob (key, value, its optimizer accumulator, bookkeeping). Starting point is about 32-64K blobs
  per bank, about 0.4-0.8M blobs in all, so roughly 0.6-1.2 GB. Bank size is limited by recall (more blobs = more
  look-alikes nearby; measured recall falls from 0.77 at 4K blobs to 0.57 at 32K) and by lookup cost, not by VRAM. The
  detailed plan fixes the exact numbers.
- The position lives in the pass's own thought space. The **key is pass T's thought** (the mean of workspace slots 0-14,
  already 64-d), taken **before pass T's own memory read is added**. It is only centred and scaled by fixed numbers
  measured once early in training. No learned projection, no token keys.

## 2. Reading (every pass)
1. After pass T, the brain has its thought (the query).
2. **Gate** (per pass, learned): predicts how much reading would help at each position.
   - Read where the predicted gain > 0.
   - Safety: at most 50% of positions per pass (speed), and 5% random reads (so the gain stays measurable).
   - No budget is imposed; the brain decides how much each pass reads.
3. **Lookup**: find the nearest blobs in pass T's bank (exact distances, top 4).
4. **Blend**: weight them by nearness (a Gaussian kernel of width h0), giving the read r.
5. r goes into pass T+1 (the inject and workspace slot 15), so it shapes the next thought.

Gradients:
- The main loss trains the blob contents, the output layer, and flows back into the thought through the read: both the
  content and the match weights (which blobs matched).
- The gate's own training never pushes the thought (stop-gradient).
- Keys are never trained by gradient.

## 3. Writing (inside the training step, no extra loop, no dataset pointers)
- After backward, each pass picks a few surprising positions per step, using only the brain's own signals (high
  loss, or loss that didn't improve across passes).
- New blob: key = that position's thought; content = what the gradient says would have helped (+ small noise).
- If a blob already sits very close, **merge** into it instead of making a duplicate (threshold chosen so unrelated
  facts are not merged).
- When a bank is full, **evict** the weakest blob, never one used in this step.

## 4. Keeping blobs findable while thoughts drift
- **Re-encode**: when a read clearly helped (confirmed recall), the blob's key moves toward the current thought. This
  is how recurring facts stay findable.
- Strength fades slowly (slow enough that a rare fact seen every ~1,300 steps survives). It is used only to choose
  what to evict.
- h0 (the blob width) is set from real same-fact distances and tracked, with a floor, so it can't shrink to zero.
- Moving all keys with the measured drift (transport) is in the appendix, for later.

## 5. Size and speed
- Freeing the old row memory saves about 3.5 GB. The banks get bigger, with exact numbers per pass in the detailed plan.
- VRAM rules: free the GPU cache before eval, cap at 20 GB, and use micro-batch 8x4 if the same-math check passes.
  Training should then sit around 14 GB plus the banks, leaving room for Windows.
- Speed: memory must not slow the step by more than 3%. Getting faster (10-20k tok/s) is a separate speed track
  (fused kernels, fewer CPU-GPU syncs). The study is running now.

## 6. Checks before any training run
- **T0a, paraphrase**: does the same fact in different wording or another document land near the same blob? The
  biggest risk.
- **T0b, gate**: can the gate tell where reading helps (fact words) from where it hurts (other text)?
- **T0h, writes**: do rare world facts actually get picked for writing, or does Wikipedia take all the write slots?
- **T0e, width and merging**: set h0 from same-fact distances, and check that the merge rule doesn't join
  unrelated facts.

## 7. The run and how we judge it
Compared at the same steps with the old run (`world_mem_life_12k`) and the control (`world_v4d_12k`):
- late facts >= 0.27 at step 7K (old run 0.280), and above the control's final 0.260 at the end;
- early/rare facts clearly above the old 0.016;
- less harm on non-fact world text (the old memory cost +0.13 nats there);
- speed within 3% of the old run; VRAM <= 20 GB.

## 8. Risks
- If the blobs don't hold recurring facts as well as the old table did, this run has no working memory. The results
  would show it, compared against the old run.
- The paraphrase check may fail: the old table was keyed mostly by wording.
