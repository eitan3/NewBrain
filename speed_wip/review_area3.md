# Speed review, area 3: design-level options for the memory system (no implementation)
2026-10-23. Code read: fruit-fly-sim/memory.py, loop_brain.py, spiking_lm.py, train_lm.py (byte-identical to
speed_wip/plan_v2/baseline_mem_v1). Nothing in fruit-fly-sim or runs was edited. All measurements ran on the CPU
from copies in speed_wip/area3/ (`opcount.py`, `opcount2.py`, `opphase.py`, `addr_probe.py`, `addr_probe2.py`;
raw results in the matching `.json` files). The live run's step-1000 `last.pt` was copied to
`area3/step1000.pt` and only read there.

GPU note: these runs used `CUDA_VISIBLE_DEVICES=""`, which (per the coordinator) does not hide the GPU on this
machine. Every tensor was created on the CPU (the checkpoint was loaded with `map_location="cpu"` and nothing was
moved to `.cuda()`), so no GPU work was issued. Later runs should use `CUDA_VISIBLE_DEVICES=-1`. torch.compile and
Triton are not available here, so any option that needs them is marked **blocked**.

---------------------------------------------------------------------------------------------------------------
## 0. Where the 0.89 s/step goes: a measured model

### 0.1 Op counts (the main finding)
I counted the aten ops dispatched (roughly one CUDA kernel launch each) for one training micro-batch: forward,
backward and `after_step`. The model was the real LoopBrain at a tiny N; op counts do not depend on N.

| model (one micro-batch) | K=6 | K=10 |
|---|---|---|
| no memory (latent think head, the 0.83 s baseline) | 9,356 | 11,475 |
| concepts only (L=4 heads, no tables) | 13,072 | 15,559 |
| **memory as run (L4 H3, S, R, read0, margin)** | **22,674** | **25,969** |
| as run, after the freeze (no margin; frozen addressing) | 18,194 | 22,239 |
| H=1 / H=2 | 22,724 / 22,702 (no change) | ≈ same |
| L=3 / L=2 / L=1 | 21,251 / 19,842 / 18,369 | 24,300 / 22,545 / 20,854 |
| margin off | 20,433 | 24,062 |
| no private memory | 21,310 | 24,407 |
| no shared pool | 15,986 | 18,542 |
| read0 off (shared read only; the pass-0 private and addressing still run) | 22,280 | 25,886 |
| no per-pass block checkpoint (VRAM +3-4 GB, not viable) | 20,084 | 23,747 |

Calibration with the K mix (uniform 2-10, 5% 12-16; E[K] = 6.4; per micro-batch on average 4.47 addressed reads
inside the gradient window and 1.93 outside it):
- Memory + concepts add about 22.5K ops per optimizer step (2 micro-batches): ~2,130 per gradient-window read
  (forward, checkpoint recompute and backward), ~640 per no-grad read, plus the last pass's concept heads.
- 0.89 s / 22.5K ops ≈ **40 µs per op**. The baseline gives the same figure (0.83 s / ~19.1K ops ≈ 43 µs per op).
- The tiny-tensor CPU run costs 16-20 µs per op, and a Windows/WDDM CUDA launch adds about 5-10 µs. So roughly
  **25 µs of each op is host time**. The other ~15 µs is GPU work that the per-pass host sync
  (`counts.tolist()` in RoutedMLP) does not let the GPU hide.

**Conclusion: the memory overhead is mostly set by the number of ops (about 60-65% host and launch overhead), not
by FLOPs.** An option that removes ops saves about 40 µs per op. An option that only removes FLOPs or bytes (fewer
heads, smaller S, a smaller query, bf16 tables, a fraction of tokens) saves only the exposed part of its GPU time,
somewhere from 0 to about 0.8 of it. Below I use about 0.5× and give a range. That changes once the "same math"
engineering (fusion, fewer launches, CUDA graphs) removes the host-bound part: then the GPU-only options become
worth more.

Per-phase split (opphase.py, K=6, infancy). The memory adds 13.3K ops per micro-batch:
- backward of the memory and concept blocks: about 5.5K (41%)
- `select` (the EMA addressing, about 310 ops per call, because of the per-level Python loop with two
  `stable_topk` and one `pk_topk` per level): 1.9K (14%)
- the margin pass, forward plus recompute plus its backward: about 2.2K (17%; infancy only)
- online w, forward plus recompute: 0.9K
- the shared read, forward plus recompute: 0.8K
- the concept heads, forward plus recompute: 0.7K, plus the concept loss
- private, prep, slot write, MemOpt: the rest

### 0.2 GPU work (analytic, N = 8192 tokens per micro-batch, RTX 5090 Laptop)
Infancy, per step: about 0.33-0.38 s of GPU work. Roughly:
- 0.12 s: the full `torch.sort` of S=512 scores for each (level, head, half, token), about 100M elements per
  addressed pass, plus the private sort of E·C×512 ≈ 38M elements per pass
- 0.11 s: shared and private gathers and `index_add_` atomics, about 3 GB per gradient-window read, plus the dw
  re-gathers
- 0.07 s: fp32 addressing matmuls with TF32 off, plus the (L,H,N,7,64) key-row gathers (online and margin)
- 0.02 s: the concept heads in bf16
- 0.03 s: MemOpt (the dense row-Adagrad over 1M×256 and 393K×256 rows, plus zeroing G)

After the freeze: about 0.25 s.

### 0.3 Start size vs final size
- **The memory cost barely depends on model size.** The private capacity layout E·C ≤ 1.5·N·k does not depend on E.
  Only the P_ema normalisation grows with E: about 0.3 ms per call at 768 experts.
- **Final size coincides with the freeze.** Births end at 0.6 and the freeze is the first step ≥ 0.6 with no births
  left. Frozen memory costs about 15.9K ops per step ≈ **0.64 s** (vs 0.89 s in infancy), because the margin and
  the gradient through addressing are gone.
- **The base cost grows with size.** v4e/v4d went from 40k to 25-27k tok/s from start to final size (×1.5), so I
  assume **base ≈ 1.25 s at final size** (not measured; this is the largest uncertainty here). Final s/step with
  memory as is ≈ 1.25 + 0.64 ≈ **1.9 s**.
- Not counted: the balancing every 250 steps. At step 1000 it did 19,231 private dead re-seeds, 7,258 private
  splits and 2,965 shared dead re-seeds. Each one is a Python iteration with about 10-20 small ops and a host sync
  (`bool(self.lastq_ok[e])`, `bool(q.any())`). That is about 5-15 s per event, i.e. **about 0.02-0.06 s/step
  amortised** (an estimate; see 10b).

### 0.4 Quality and stability evidence (step-1000 checkpoint, eval only, 8×512 val tokens, K=6)
Weak evidence: infancy, step 1000 of 12,207, no retraining. Base val loss 4.9714. Changes in nats:

| ablation at eval | Δ loss |
|---|---|
| shared reads all off | +0.093 |
| private off | −0.0005 (the private gate σ(b_e) ≈ 0.12 and W_p is still small) |
| **read0 only (passes 1-5 reads off)** | **+0.0005** |
| read0 + the last read | +0.001 |
| all reads except read0 | +0.085 |
| reads on the last 3 passes only / every 2nd pass | +0.091 / +0.085 |
| only head 1 gated (H=1, untrained) / heads 1-2 | +0.044 / +0.042 |
| top-4 of the 6 rows, renormalised | +0.0004 |
| address cache θ = 0.3 (re-address a token only if ‖Δa‖/‖a‖ ≥ 0.3; 50% of addressings reused) | −0.002 |
| address cache θ = 0.5 (69% reused) | −0.002 |
| every pass reuses read0's addresses | +0.008 |

Address statistics:
- **Consecutive passes, w-weighted row overlap:** read0→1: 0.006 (s_0 is token-keyed, s_1 is a thought);
  1→2: 0.35; 2→3: 0.59; 3→4: 0.66; 4→5: 0.70.
- **Input drift per pass:** 1.4, 0.40, 0.21, 0.16, 0.13. The thought settles, and the addresses settle with it.
- **The heads share no rows** (overlap 0.000): they are a true multi-hash.
- **w is not one-hot at τ = 0.03:** max w is about 0.45-0.52, the effective number of rows is about 3.3-4.2 of
  6, and the top-4 rows carry about 0.81-0.88 of the mass.
- **The read0 gate at level 4 is 0.97-0.998.**

**Key observation for the owner.** At step 1000 almost all of the memory's measured value (≈0.09 nats) comes from
the **pass-0 read, which is token-keyed**: s_0 = init_state(emb(token)), so read0's address is a learned function
of the token id alone. The thought-addressed reads on passes 1-5 are worth about 0.001 nats so far. That conflicts
in spirit with "addresses come only from thoughts (no token keys)": today the memory works mainly as a large
token-keyed table. It may change later in training (the thought reads' gates are rising, 0.15 → 0.3+), but keep
this in mind when judging any "fewer reads" option, and when crediting the memory with an effect: the knockout
should report read0 and the thought reads separately.

---------------------------------------------------------------------------------------------------------------
## 1. Option catalogue

Savings use 40 µs per removed op plus about 0.5× the removed GPU time. "Start" is infancy at the start size
(1.72 s). "Final" is post-freeze at the final size (about 1.9 s, assumed). Op deltas are per optimizer step.

### 1.1 Fewer translator heads per level (H 3 → 2 or 1)
- **Ops:** 0. The heads are batched in every op, which was measured.
- **GPU:** the parts that scale with H are the sub-key scoring and sorts, the translator, the key-row gathers, and
  the shared gather/scatter. They are about 0.22 s in infancy. H=1 removes 0.15 s, H=2 removes 0.07 s.
- **s/step:** H=1: **−0.05 to −0.12 s (3-7%)** at start, −0.04 to −0.09 s (2-5%) at final. H=2: about half that.
  After the engineering pass (GPU-bound) this approaches the full −0.15 s (10% or more of what remains).
- **Quality:** Lample et al. 2019 found multi-head PKM clearly better than single-head (more heads gave lower
  perplexity and better key usage; they use 4 heads). PEER (He 2024) also relies on many heads. Here the heads are
  disjoint multi-hashes, and the untrained H=1 eval costs +0.044 nats. After retraining expect a smaller but real
  loss: each token reads 6 rows instead of 18 per level, and each row averages about 3× fewer writes. H=2 is the
  safer step.
- **Stability:** the mean ω-stability per head is unchanged, but the read output loses its redundancy. With 3 heads
  one head's set flip moves about 1/3 of a level's read; with 1 head it moves all of it. The variance of
  "read-value stability" goes up about 3× (plan section 5 counted redundancy as a mitigation).
- **Constraint:** none violated. **VRAM:** −0.3 to −0.6 GB of activations (H=1). **Effort:** none (`--mem-heads`).

### 1.2 Fewer read passes, or reusing addresses across passes
**(a) Read on fewer passes (1 / 3 / 5 reads per forward).** This **violates "memory read on all passes".**
- Each skipped gradient-window read saves about 1,600 memory ops (≈860 after the freeze). A skipped no-grad read
  saves about 540. The concept heads still run every pass.
- "5 reads": −0.16 s (9%) start, −0.11 s (6%) final.
- "3 reads" (read0, middle, last): −0.38 s (22%) start, −0.26 s (14%) final.
- "read0 + last": −0.5 s (29%) start, −0.34 s (18%) final.
- "last only": about −0.6 s, but it drops read0 (+0.09 nats today).
- **Quality (step 1000):** read0 + last costs +0.001. Any variant without read0 loses about 0.09.
- **Stability:** fewer reads means fewer address sets to keep stable, and the last read sees the most settled
  thought (drift 0.13 per pass vs 0.4 early), so it is the most stable read to keep.
- **VRAM:** −0.2 to −0.5 GB. **Effort:** small (a pass mask in `run()`).

**(b) Static address refresh: reads on all passes, new addresses only on some passes.** Compatible with all
constraints.
- Compute `select` (rows) on read0, on read 1 (the first thought-keyed read), and then every r-th pass. On the
  other passes reuse the cached rows.
- Recompute w every pass with the online translator from the current thought. The read follows the thought within
  the cached set, and the keys still get gradient every pass.
- **Never** carry read0's addresses forward: that would spread token keys onto thought passes (read0→1 overlap is
  0.6%).
- Saves about 310 ops plus about 7 ms of GPU (fp32 scoring and the 100M-element sort) per skipped `select`.
  - r=2 (about 2.7 skipped selects per micro-batch): **−0.09 s (5%)** at start and at final. It is the same after
    the freeze, because select is unchanged by the freeze.
  - Refresh only at read0 and read1 (about 4.4 skipped): **−0.14 s (8%)**.
- **Quality:** at eval, θ=0.3-0.5 caching (50-69% reuse) was neutral (−0.002). Even full reuse cost only +0.008.
  This is weak evidence, because the thought reads carry little value yet.
- **Stability: positive.** Later passes reuse the anchor pass's set, so small late-pass drift (13-16% per pass)
  can no longer flip rows. The cross-training stability metric then depends mainly on the refresh passes.
- **Effort:** small-moderate (a cache in `run()`; the checkpointed block already passes `sel` through `cache`).

**(c) Thresholded per-token re-addressing** (re-address only tokens whose ‖Δa‖/‖a‖ ≥ θ, measured θ≈0.3).
- It saves **no ops**, and it saves GPU work only if the moved tokens are compacted. Compaction needs a static cap,
  for example "re-address the 50% most-moved tokens per level" via `topk` on the drift, to avoid a host sync.
- **s/step:** −0.03 to −0.06 s.
- **Stability:** best of the three. It acts as hysteresis: an address only changes when the thought really moved.
  This directly reduces the address flips that the ω-stability guard counts.
- **Effort:** moderate. Recommended only as a follow-up to (b).

### 1.3 Smaller S, or a smaller query
**S 512 → 256 per half** (4 × 65K rows):
- Ops: about 0 (MemOpt chunk loops −90 ops).
- GPU: sorts ≈ −45%, sub-key scoring −50%, MemOpt −75% (it is dense over S²). In total about −0.075 s of GPU.
- **s/step:** **−0.02 to −0.06 s (1-3%).**
- **VRAM: −1.6 GB** (V + G go from 2.15 to 0.54 GB; S_acc and wcount shrink too). That gives headroom, which
  matters at 16.7 GB peak.
- **Quality:** Lample et al. show that PKM gains grow with memory size (diminishing, but monotone up to about 1M
  slots). Here, however, the rows are already written about 7× per step (plan 3.1: "prototypes"). A 4× smaller
  table means about 4× more contexts per row. Expect a small loss.
- **Stability:** slightly better. Fewer, coarser cells mean the expected top-k boundary gaps grow, but only
  log-weakly with S (the gaps of extreme order statistics depend on about √(2 ln S)). Expect +5-10% on the gaps,
  not a cure.

**Query 128 → 64 (two 32-d halves):**
- GPU −0.03 s, ops 0 → **−0.01 to −0.03 s.**
- **Quality and stability:** 32-d cosines are noisier; the product space is less discriminative. Lample uses
  d_q = 512. Not recommended.
- **Effort:** both are flag-level changes (S) or small changes (query width).

### 1.4 One translator shared across levels (plus a level embedding)
- **Speed: about 0.** W_q is already one einsum over (L,H), and each level still needs its own query, because its
  input μ_l differs. Batching the per-level `select` loop into one bmm and one sort would save about 150 ops per
  call, but that is a same-math engineering change that does not need shared weights.
- **Quality:** a tied geometry across levels, plus 0.5M fewer parameters.
- **Stability:** neutral after the freeze.
- **Verdict: no speed reason to do it.**

### 1.5 Fewer concept levels, or a cheaper level-head trunk
**L 4 → 3 or 2: this violates the 16/32/64/128 hierarchy constraint.**
- Measured: about −1,430 ops per level per micro-batch at K=6. With the K mix that is about −1,200 per level, plus
  the GPU parts that scale with L.
- L=3: **−0.12 s (7%) start, −0.09 s (5%) final.**
- L=2: **−0.25 s (15%) start, −0.18 s (9%) final.**
- **Quality:** each level's concept cosine is 0.79-0.83 at step 1000, and the levels' read0 gates differ (level 4
  is used most), so the levels are not obviously redundant. Dropping a level removes a horizon and its 18 rows per
  token.
- **Stability:** neutral per level.
- **VRAM:** −0.25 GB per level of activations; the table shrinks by S² rows per level (−0.54 GB V+G per level).

**Cheaper trunk:** the trunk (1024→256, 0.26M MACs) is not the cost. The level heads (≈0.49M MACs each) are, and
the concept heads total only about 0.02 s of GPU per step. Low-rank or scalar-ν heads would save **< 0.01 s.** Not
worth it.

**Related (an estimate, not measured separately):** compute the concept NLL+hinge on 2 sampled gradient-window
passes instead of all of them (about 65 ops per level per pass, forward and backward) → about −0.06 to −0.08 s.
It changes the training signal's variance, not its expectation. It is not on the owner's constraint list, but it is
a training-signal change; ask first.

### 1.6 A cheaper private lookup
- **Private off entirely:** −1,200 ops per micro-batch plus about 0.05 s GPU → **−0.12 s (7%).** Its measured
  value at step 1000 is about 0, but its gate is still closed.
- **Private on the last 3 passes only:** −0.06 s (3.5%). This violates "read on all passes" for the private memory.
- **Per-expert product keys** (2 × 23 sub-keys per expert ≈ 529 rows): the scoring is 20× smaller and the sort
  shrinks from 512 to 23 → GPU-only, **−0.01 to −0.03 s.** Quality and stability are about the same (Lample: PKM ≈
  flat keys at equal size). Effort: moderate.
- **Hashing** (random hyperplanes on q̄¹): removes the scoring and the sort (−0.02 to −0.04 s), but the keys can no
  longer learn or balance, and a frozen hash has no margins (stability no better). Not recommended.
- **Read the private memory from the top-1 chosen expert only** (not all 6): GPU −0.03 s. Less private capacity.
- **Most of the private GPU cost is the full stable sort,** which a top-k plus tie-break fix removes. That is same
  math and belongs to the engineering review.

### 1.7 Reading memory only in the bptt window
- It saves the 1.93 no-grad reads per micro-batch: **−0.08 to −0.11 s (5-6%).**
- **It violates "read on all passes".** It also creates a train/eval mismatch (eval K=6 is always inside the
  window).
- Worst of all, for K > 6 (44% of steps) it drops read0, today's most valuable read.
- **Not recommended.** If anything, give the no-grad reads cached addresses (1.2b) instead.

### 1.8 Applying memory to a fraction of the tokens (for example a static 50% random subset per pass)
- **GPU-only:** about −0.17 s of GPU, plus a few more gather/scatter ops → **−0.05 to −0.1 s (3-6%)** now. More after
  the engineering pass.
- **Quality:** memory dropout. Each row gets half the training signal, and there is a train/eval mismatch (at eval
  every token reads). A rotating parity (each token reads every other pass) at least keeps reads on every pass.
- **Constraint:** partly violates "read on all passes" (per token).
- **Stability:** per-read unchanged.
- **Effort:** small.

### 1.9 bf16 tables
- **Read-side bf16 shadow of V** (fp32 master kept for MemOpt): the gather bytes halve (about −10 GB of traffic per
  step) → **−0.01 to −0.03 s (1-2%).** VRAM +0.5 GB for the shadow.
- **bf16-only V:** −0.54 GB shared and −0.2 GB private. It needs stochastic rounding in MemOpt, because updates of
  lr·G/√S on values of about 0.02 fall below bf16 resolution (8-bit mantissa, about 8e-5 at 0.02). Without it, the
  writes silently stall.
- **G must stay fp32** (atomics and accumulation).
- **Addressing stability is unaffected** (addressing stays fp32); only the values change.
- **Low priority.**

### 1.10 Other options found
**(a) The margin on one sampled gradient-window pass per micro-batch (infancy only).**
- The margin costs about 2.2K ops per micro-batch (17% of the memory ops). Sampling one pass and keeping the same
  weight is an unbiased estimate of the current mean over passes.
- **−0.13 s (8%) start, 0 at final.**
- Quality and stability: about none. At step 1000 the hinge is active on 55% of reads and the mean gap is still
  0.0036, the same as the natural gap at init, so the margin is not yet widening the gaps anyway.
- No constraint conflict. **Effort:** tiny.

**(b) Balancing churn** (stability as well as speed). At step 1000 one balance event re-seeded **19,231 private
keys (about 17% of the live private keys) and 2,965 shared sub-keys (about 24% of all sub-keys)**, plus 7,258
private splits.
- **Speed:** each item is a Python iteration with host syncs, about −0.02 to −0.06 s/step amortised if vectorised
  (engineering).
- **Design:** a 17-24% re-seed rate per event is large address churn in infancy. Each shared re-seed re-points a
  512-row band. A dead threshold of 0.05·mean on a usage EMA with about a 100-step horizon (usage_rate 0.01) seems
  too eager for R=512 keys per expert with about 1.7K hits per micro-batch-pass. Re-seeding only keys that are dead
  across two consecutive events, or using a slower EMA, would cut both the time and the churn. Not a constraint
  issue, but it bears directly on "stability of addressing matters a lot".

**(c) top-k 6 → 4.**
- The top-4 rows carry 81-88% of w, and the eval change is +0.0004.
- GPU −0.02 s, a few ops → **about −0.02 s.**
- Stability: each row flip then moves more of the mass (≈0.2 vs 0.15); the set boundary moves to 4|5. Neutral to
  slightly worse. Optional.

**(d) Same math, handed to the engineering review:**
- After the freeze (with addr_sg), online θ = θ̄, so the online queries and scores equal the EMA ones. `online()`
  can reuse select's q and its pk_topk scores, which removes a whole fp32 einsum and the key-row gathers per pass.
- In infancy, `gaps()` recomputes the online queries a third time.
- The full `stable_topk` sorts can become top-k with a tie-break.
- The per-level select loop can be batched.
- Without Triton or torch.compile (**blocked**), fusion must come from hand batching or CUDA graphs. CUDA graphs
  are hard here because of the data-dependent capacity C (host sync per pass).

**(e) Read0 is token-keyed:** see 0.4. It costs as much as any read (about 2.1K ops per micro-batch for K ≤ 6),
and it is the only read with measured value today. Whether to keep it is a principle question for the owner, not a
speed question. It should be reported separately in every knockout.

---------------------------------------------------------------------------------------------------------------
## 2. Summary table

| option | Δ s/step start (infancy) | Δ final (frozen) | quality | addressing stability | constraint | VRAM | effort |
|---|---|---|---|---|---|---|---|
| H=2 | −0.02 to −0.06 | −0.02 to −0.05 | small − | redundancy ↓ | ok | −0.2 GB | flag |
| H=1 | −0.05 to −0.12 | −0.04 to −0.09 | − (Lample/PEER: heads help) | read-value variance ↑ | ok | −0.5 GB | flag |
| address refresh every 2nd pass (reads every pass) | −0.09 | −0.09 | ≈0 (measured, weak) | **+** | ok | ≈0 | small |
| refresh only at read0 + read1 | −0.14 | −0.14 | ≈0 to small − | **+** | ok | ≈0 | small |
| thresholded re-addressing (capped) | −0.03 to −0.06 | same | ≈0 | **++** (hysteresis) | ok | ≈0 | moderate |
| 5 / 3 / 2 reads per forward | −0.16 / −0.38 / −0.5 | −0.11 / −0.26 / −0.34 | ≈0 now if read0 kept | + (fewer sets) | **violates all-pass reads** | −0.2 to −0.5 GB | small |
| S=256 | −0.02 to −0.06 | same | small − | slightly + | ok | **−1.6 GB** | flag |
| query 64-d | −0.01 to −0.03 | same | − | ≈ / − | ok | small | small |
| shared translator | ≈0 | ≈0 | ≈ | ≈ | ok | ≈0 | small |
| L=3 / L=2 | −0.12 / −0.25 | −0.09 / −0.18 | − | ≈ | **violates hierarchy** | −0.8 / −1.6 GB | flag |
| cheaper level heads | < −0.01 | < −0.01 | ≈ | ≈ | ok | ≈0 | small |
| concept loss on 2 sampled passes (est.) | −0.06 to −0.08 | same | noisier signal | ≈ | ok (ask) | small | small |
| private only last 3 passes | −0.06 | −0.06 | ? (≈0 now) | ≈ | **violates all-pass reads** | small | small |
| private per-expert PK | −0.01 to −0.03 | same | ≈ | ≈ | ok | ≈0 | moderate |
| private off | −0.12 | −0.12 | ≈0 now, unknown later | — | **violates shared+private** | −1.0 GB | flag |
| bptt-window reads only | −0.08 to −0.11 | same | − (loses read0 at K>6) | ≈ | **violates** | small | small |
| 50% of tokens per pass | −0.05 to −0.1 | same | − (dropout, mismatch) | ≈ | partly violates | −0.3 GB | small |
| bf16 read shadow / bf16 V | −0.01 to −0.03 | same | ≈ (needs SR) | none | ok | +0.5 / −0.7 GB | small-moderate |
| margin on 1 sampled pass | **−0.13** | 0 | ≈ | ≈ (margin not effective yet) | ok | −0.2 GB | tiny |
| top-k 4 | −0.02 | −0.02 | ≈ (+0.0004 eval) | ≈ / − | ok | small | flag |
| calmer balancing (dead across 2 events / slower EMA) | −0.02 to −0.06 (with vectorising) | 0 (infancy only) | ≈ | **+** (less churn) | ok | 0 | small |

GPU-only options (H, S, query, fraction, bf16, top-k) gain about 2× more after the engineering pass removes the
host-bound part. The op-removing options (fewer reads, address refresh, margin sampling, levels) lose part of their
value then, because the cost per op drops.

---------------------------------------------------------------------------------------------------------------
## 3. Recommended packages

Baselines: start 1.72 s (memory overhead 0.89 s). Final ≈ 1.25 s base (assumed) + 0.64 s frozen overhead
≈ 1.9 s. "+ eng" applies the other reviewer's same-math savings of 30-50% to the memory overhead that remains
after the package. The design and engineering savings are not fully additive, so these are ranges.

### (a) Mild, near-original behaviour (no constraint touched)
Contents:
1. The margin on one sampled gradient-window pass per micro-batch.
2. A static address refresh every 2nd pass, starting from read 1 (never from read0). The reads still happen on all
   passes, with w recomputed every pass.
3. Balancing: re-seed only keys that are dead in two consecutive events.
4. Optionally, the bf16 read shadow.
5. Keep H=3, L=4, S=512, the private memory and read0.

| | design only | + eng (30-50%) |
|---|---|---|
| start | about −0.22 to −0.25 s → **≈1.48 s** (−14%) | **≈1.06-1.31 s** |
| final | about −0.1 s → **≈1.8 s** (−5%) | **≈1.5-1.6 s** |

Expected quality: unchanged within noise. Stability: equal or better (the address refresh is a mild hysteresis,
and there is less balancing churn).

### (b) Aggressive (constraints respected where possible)
Contents:
1. Everything in (a).
2. H=1, or H=2 for a safer version.
3. A refresh only at read0 and read1, or capped thresholded re-addressing.
4. S=256 (frees 1.6 GB).
5. top-k 4.
6. The concept loss on 2 sampled passes (ask the owner).
7. Keep L=4, the private memory and reads on all passes.

| | design only | + eng (30-50%) |
|---|---|---|
| start | about −0.45 to −0.5 s (overlaps discounted) → **≈1.25-1.3 s** (−25%) | **≈1.0-1.1 s** |
| final | about −0.3 s → **≈1.6 s** (−16%) | **≈1.4-1.5 s** |

Expected quality: a small loss, mostly from H=1 (Lample/PEER); about half of that with H=2. S=256 costs a little.
Stability: better, through the refresh and hysteresis and the slightly larger gaps, but read-value redundancy is
lower with H=1.

### (c) Constraint-breaking, for reference only (the owner must approve each item)
Contents: L=2, reads only at read0 and the last pass, private memory off, plus (b).
- Start: about −0.7 s → **≈1.0 s.** Final: about −0.45 s → **≈1.45 s.**
- With eng: about 0.9 s at start and 1.35 s at final.

It keeps today's measured memory value (read0) but abandons "all passes", "16/32/64/128" and "shared + private".

### Recommendation
Do (a) now. Its items are cheap, safe and constraint-clean.

Before (b), measure the two key unknowns on a later checkpoint (step ≥ 4000):
1. How much the thought reads (passes 1+) are worth vs read0. Repeat `addr_probe.py`'s read masks.
2. The cross-pass overlap. If later reads still overlap ≥ 0.6, address refresh is free.

Ask the owner explicitly about read0 being token-keyed: it is currently the memory's whole measured benefit.
