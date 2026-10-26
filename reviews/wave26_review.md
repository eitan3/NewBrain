# wave26 review: the v2.1 memory code used by the live run

Code reviewed: the frozen snapshot `runs/world_wave26_12k/code_snapshot/` (wave_v21.py, wave_columns.py, wave_address.py,
wave_write.py, wave_fastgather.py, wave_memory.py, and the memory hooks in train_lm.py / loop_brain.py / spiking_lm.py).
Specs: `SPEC_v21_final.md` and `SPEC_wavesheet_v2_cloud.md`, including the owner decisions (z_conf back to 4 at 10:07;
split children copy the parent; the shared pair reads global under its own gate row).
Scope: only bugs that would corrupt the run's results or crash it later.
Method: I read the code, then ran CPU checks on a copy of the snapshot in `scratchpad/rev26/code`. Nothing under runs/
or fruit-fly-sim-wave/ was modified, and the GPU was not touched. The heavier checks loaded `ckpt_1000.pt` read-only
(the real step-1000 model: 225 experts, memory active, n_opt 1000) and ran 22 training micro-batches of the run's
data on the CPU: B 16 x T 512, the trainer's loss / accum 2, K in {4..10}.

Run state when reviewed: step 2250, 333 experts, `wmem_writes_p` 53,767, `wmem_writes_g` 20,897, `confirms_g` 21,759,
`confirms_p` 28,412, log_s -1.28.

## Verdict

Nothing found will crash the run or corrupt the brain. Gradient isolation, exact-zero unread pairs, writing only where
read, no writes from dropped sequences, the selection invariants, the D update, splits and resume all hold. One real
bug (F1) quietly degrades the global sheet's confirmations and write targets. The rest are low-severity or latent.

## Findings, ranked by severity

### F1 (MEDIUM, affects the live run): duplicate-index `index_put` zeroes the token's global read for h_g and the global write target
- **Where:** `wave_v21.py:524` in `after_backward`:
  `yg_tok = torch.zeros(N, d).index_put((tok,), y_g_raw)`. Also `wave_v21.py:623` in `_writes`:
  `yg_tok = torch.zeros(N, d).index_put((dec["tok"],), yg)[tg_]`.
- **What goes wrong:**
  - `tok = S // 7` repeats every token once per selected pair. `y_g_raw = vg * y_g_full[tok]` is nonzero only on that
    token's vg rows, and zero on its private-only rows and on the invalid filler rows of S.
  - A non-accumulating `index_put` with duplicate indices keeps an arbitrary one of the rows. That is undefined on
    CUDA; on CPU the last one wins.
  - So for a token that read the global sheet, `yg_tok` is often the zero row. Then:
    - `h_g = cosz(0, .) = 0`, so the token can never be a global confirmation (needs h_g >= theta);
    - the global write target loses its `max(h_g,0)[c>=.5] y_hat` "old content" term;
    - the result can differ between identical runs (the GPU picks any row).
- **Measured** on ckpt_1000, 6 micro-batches:
  - of about 4,070 valid GT tokens per micro-batch, about 970 carry a nonzero global read;
  - CPU semantics zero 24% of those (194-286 per micro-batch);
  - about 2,700 GT tokens per micro-batch (66%) have mixed vg / non-vg rows, so they are at risk on the GPU;
  - tokens with h_g >= 0.05 drop from about 211 to about 160 per micro-batch (-24%).
- **Live run:** yes, at every step since 500. Global confirmations and the global write targets are biased low. It does
  not crash, does not touch the brain, and the private sheet is unaffected.
- **Fix** (at the next restart, not hot-patched): keep the token-level read from `_hint` and drop the scatter.
  - In `_hint`: `rec["y_g_tok"] = y_g_full.detach()`.
  - In `after_backward`: `h_g = cosz(rec["y_g_tok"][GT], -e_g[GT])`.
  - In `_writes`: `yg_tok = rec["y_g_tok"][tg_]`.
  - `y_g_full` is already pre-keep, which matches `y_raw`. Equivalent alternative: index_put only the vg rows,
    `index_put((tok[vg],), y_g_raw[vg])` (their duplicates are identical).

### F2 (LOW, live run: small): the absolute 1e-6 floor on write targets
- **Where:** `wave_v21.py:626` (global) and `wave_v21.py:651` (private): `good = |E| >= 1e-6`.
- **Measured** on the run's own scale (ckpt_1000, 22 micro-batches, 431 private picks, 166 global targets):
  - **Private:** 17 / 431 = **3.9% skipped** (95% CI about 2.3-6.2%). Of these:
    - 12 have E exactly 0. These are K >= 10 micro-batches: the read pass 3 lies outside the bptt window (K - 6 > 3), so
      a pair gets gradient only if its expert is routed again at a pass inside the window. The relative patch skips
      these too, which is correct.
    - **5 / 431 = 1.2%** are real, nonzero gradients between 1e-10 and 1e-6 that only the absolute floor drops.
  - **Global:** 1 / 166 (0.6%) skipped.
  - **Typical sizes:** private |E| median 7e-6 to 2e-5, the lowest 10th percentile 2-7e-6, the per-micro-batch maximum
    5e-5 to 4e-4.
  - **Over all privately read pairs** (not only written ones), 8-31% fall below 1e-6. The writes pick high-surprise
    tokens, so their gradients are larger.
- **Why it matters later:** |E| scales as s|x| / (N accum). The skip share will grow if log_s falls, the loss and its
  gradients shrink, or the batch grows. The comment in the patch ("skipped every private write") does not hold for this
  run: `wmem_writes_p` is 53,767.
- **Fix:** the relative threshold in `scratchpad/patch_v21_private_writes.diff` is right (`tiny = 1e-6 * max|E|`).
  Apply it at the next restart; it is not urgent mid-run.

### F3 (LOW, latent, not reachable in this run): child rows' AdamW moments are lost if a birth comes before the first memory step after a resume
- **Where:** `wave_v21.py:862-867`. `on_grow` copies the Wp / Wg / bp_b / bg_b moments only when `self._opt` exists.
  After a resume, the saved optimizer state sits in `_opt_pending` until the first `_adamw`.
- **Check** (rev26/checks.py, `resume/grow_right_after_resume`): a resume followed by a birth before the first step
  diverges from the no-stop run. A plain resume, and a resume after a birth, are bit-identical over 4 steps.
- **Live run:** no. Checkpoints are at multiples of 1000, which are also multiples of `--neuro-every 25`. On the resumed
  step the trainer skips births (`not first`), and `_adamw` runs at the end of that step, before the next birth.
- **Fix if wanted:** materialise `_opt` (and load the pending state) at the top of `on_grow`.

### F4 (LOW, live run: probably 0 pairs): `wp` ignores rows without a private place
- **Where:** `wave_v21.py:265` (`wp = (xp | bp > tau) & present`).
- **Spec:** PART 2 says DUMMY when `Pinit[row] == 0`, and then `wp = False`.
- **Effect:** such pairs spend the K_pr budget and capacity reading DUMMY (y = 0, b = 0). Every expert that existed at
  step 500 got warm-up samples, and children copy `Pinit`, so this is about 0 in this run.
- **Fix:** `wp &= (Pinit[rows] > 0)` for k < 6.

### F5 (LOW / INFO, live run: yes, by design or unlisted): the hint is re-injected at every pass >= 3, not only passes 4-5
- **Where:** `wave_v21.py:420-446`.
- **Behaviour:** once the cache exists, every later call injects the cached `r3`. With K up to 10 (and 12-16 on 5% of
  steps) that is up to 13 injections; at eval (K = k_mean 6) it is 3 (passes 3-5).
- **Spec:** the brief and the module docstring say "passes 4-5 re-inject".
- **Effect:** not corruption, but a train / eval mismatch in how much hint the long-K steps get.
- **Fix:** if the spec is meant literally, skip `k > read_pass + 2`.

### F6 (LOW, live run: yes): the memory's GEMMs run in TF32, not fp32 'ieee'
- **Where:** `train_lm.py:1407-1408` set `allow_tf32 = True` globally. The memory switches autocast off but not TF32.
- **Spec:** PART 2/3 ask for fp32 'ieee' (address codes, private place argmax, Translate, D_grad).
- **Effect:** near-tie code / place assignments can flip. It is deterministic on one GPU, so write and read keys stay
  consistent, and nothing crashes.
- **Fix:** wrap the memory's `__call__`, `after_backward` and `after_step` in
  `torch.backends.cuda.matmul.allow_tf32 = False` and restore it on exit.

### F7 (LOW, only after a NaN): `nan_abort.pt` holds an already-mutated memory
- **Where:** `train_lm.py:2031` then `2036-2041`. `bm.after_backward()` (writes, confirms, sigma_b, address learning)
  runs before the NaN check.
- **Effect:** with a NaN or inf E, sigma_b, rho and the A / B decoders go NaN, and the abort checkpoint saves that
  memory. The abort itself does fire (the gate grads are in the norm). `last.pt` stays clean, so a resume from `last.pt`
  is fine.
- **Fix:** guard `after_backward` with `torch.isfinite(probe.grad).all()`, or note that `nan_abort.pt`'s memory is
  not "before the step".

### F8 (INFO, operational)
- **Memory files are never pruned.**
  - `prune_ckpts` globs `ckpt_*.pt`, and the stem `ckpt_1000.memory` is not all digits, so `ckpt_N.memory.pt`
    (1.94 GB each) is never pruned (`train_lm.py:261-264`).
  - About 6 report / save checkpoints come to about 12 GB. C: has 350 GB free, so there is no risk.
- **Do not hold `last.pt` open while the run saves it.**
  - On Windows, `os.replace(tmp, last.pt)` fails if another process has `last.pt` open (no FILE_SHARE_DELETE).
    `save_checkpoint` does not catch that, so the run would die.
  - Next saves: steps 3000, 4000, and so on.
  - I therefore used `ckpt_1000.pt`, not `last.pt`.
- **D update bookkeeping differs from the spec.**
  - `wave_v21.py:805-811`: n / R / Cv advance only on rows with a nonzero gradient. The spec increments n and decays
    R / Cv on every row.
  - Both leave a zero-gradient row unmoved. The update math otherwise matches PART 3 exactly (checked numerically).
  - Harmless.

## Checklist (all items asked)

| Item | Result |
|---|---|
| **Gradient isolation** | OK. The gate loss uses `autograd.backward(inputs=gate params)` on detached inputs. Selection and recall run under no_grad. `xt` and the cap are detached. The LM gradient reaches only A, B and log_s (memory-owned: outside BrainOpt and the brain clip, `train_lm.py:406-423, 2050`), plus D_grad via Translate. U23 passes. |
| **D only via D_grad** | OK. D is a buffer; Translate returns None for it. D changes only in `_d_update` and `on_grow`. |
| **Unread pairs exactly 0** | OK. U22 passes. Invalid, demoted and dropped-sequence pairs give o = 0 exactly (X = 0 → Translate 0 → cap factor 1). Rows for routed pairs that are not re-routed go to the zero row 7N. |
| **Write only where read** | OK. U25 passes. Global writes go to entry 0 of a gused token (one of its 4 read entries); private writes go to the read place with the read keys. |
| **Drop sequences write nothing** | OK. U25 passes (allow is masked by keep; confirmations need keep). They still feed the gate loss and the address statistics (as the spec intends). D_grad and the A / B grads get 0 from them. |
| **Capacity / selection** | OK. 60 random cases: S unique; slots unique and < NSLOT; per-row count <= C (C_sh for the shared row); valid <= K; vp <= 3N; the shared pair never private; vg tokens inside GT; no exploration at eval. A NaN gate score is never read (NaN > tau is False). |
| **Cap scale detached** | OK (`wave_v21.py:409`). |
| **Factored D update** | OK. Matches the PART 3 formulas row by row (allclose, rtol 1e-5); a zero row does not move. See F8 for the bookkeeping difference. |
| **Checkpoint / resume** | OK. `memory_embed 1`: `last.pt` carries the memory, and `--resume` loads it strictly; `last.memory.pt` is a redundant copy used only by `--memory-load` or a brain-only checkpoint. Extra state (n_opt, n_fwd, pointers, address n_upd, store sweep_ptr / n_step, policy n_calls, AdamW state) restores correctly. All generators are reseeded from n_fwd / n_step. A round trip continues bit-identically, before and after a birth. Latent exception: F3. |
| **Splits** | OK. Every per-expert tensor is copied: strip key and chain bytes, load, Wld_p, P_e, Ph, Pinit, D, R, Cv, n_d, Wp / Wg / bp_b / bg_b and their AdamW moments (U28 passes). `row_of` as the identity is correct: `RoutedMLP.grow_experts` appends new experts at E0.., and renewal never touches routed experts (`neurogenesis.py:248`, `not self.routed`). Emax = 768 = the final count, so the assert cannot fire. |
| **Eval vs train** | OK. `_live` is False at eval: no exploration, no keep/drop, no probe, no counter / kappa / store change (checked). A stray `after_backward` after eval does nothing. |
| **NaN / inf** | No realistic source found. log_s is clamped [ln 0.05, 0]. Every norm divisor is clamped. sigma_b is clamped at 1e-12 and bt clipped to ±5. null_sd is clamped at 0.005. kappa uses (q99 - med) clamped at 0.01. Loads enter only as log2(load+1). D's V >= 1e-30, followed by the RMS clip. Address homeostasis logs self-correct before som_use can underflow. See F7 for the case after a NaN. |
| **Growth / leaks over 12K steps** | None found. `_rows` and `_samp` are cleared every after_backward. The cache is replaced every forward. `_stat_acc` is reset at each log. `policy._mix` is a fixed-size sum. The counters are ints. Generator caches are bounded. |
| **Determinism** | Per-purpose generators are reseeded from python ints, and the sweep randoms from n_step. The one nondeterministic memory op is F1's duplicate `index_put`. |
| **Late triggers** | All checked. Write_start 500: lazy AdamW; null stats start at step 512 (c = 0 before). Null steps (every 16): fine. **K > bptt** (K >= 10): a graph is still built, passes in the window give E, and writes happen (checked). Births from step 610: copy, C and conf cap recomputed, strip sweep extends. 768 experts: C = 64, NSLOT 53,248, fine. |

## The 14 listed deviations

| # | Deviation | Harmful for this run? | Why |
|---|---|---|---|
| 1 | Selection, Translate, D update and gate loss are its own code; wave_gate.py is not wired | **No** | Selection invariants, the D math and the isolation all check out against the spec. F1 sits in this code, but it is not a consequence of skipping wave_gate.py. |
| 2 | row_of is the identity | **No** | Births append; routed experts are never renewed or compacted; ids stay contiguous up to 767. |
| 3 | Private keys use the global residual codebooks | **No** (quality only) | The keys are consistent between read and write (cached), and the codebooks drift at eta_slow like the global ones. The codebooks were fitted on x_hat - W[loc], not x_hat - P_e, so private code use may be less balanced. |
| 4 | Global write to entry 0, not the least-loaded entry | **No** | Entry 0 is always read; fix A's write conscience keeps the load flat (load_g max/mean 2.19, top-10 share 8.3% at step 2250). |
| 5 | theta fixed at 0.05 (no W2 calibration) | **No** | It only sets the confirmation and old-content thresholds; no crash or corruption. F1 matters more for h_g. |
| 6 | pi = routing slot rank | **No** | The rank comes from the noisy, biased selection score, not the gate probability; it only orders the 2 private writes. |
| 7 | Injection through the trainer's buf / x hook | **No** | The router never sees the hint and the brain gradient path is the LM's own. It re-injects at every pass >= 3, not only 4-5 (F5, low). |
| 8 | The hint is fp32 | **No** | About 88 MB more transient VRAM; numerically safer. |
| 9 | Read chunk 2048 | **No** | Speed only. |
| 10 | The global address learns before write_start | **No** | Intended warm-up: fast for 1,000 updates (= 500 steps at accum 2), then slow exactly when writes start. |
| 11 | The keep generator is on the CPU | **No** | Seeded from n_fwd, so deterministic and resumable; one tiny host-to-device copy per micro-batch. |
| 12 | Write-side syncs remain | **No** | Speed only. |
| 13 | Fix-A conscience: w_hits decays 0.99 per chunk of 32 writes | **No** | A micro-batch has <= 16 global writes, so the decay is per micro-batch (time constant about 50 steps, not 100); balance is fine in the metrics. |
| 14 | Train-only, swap and the budget controller are deferred | **No** | None are used. `set_train_only(ids)` and `read()` raise NotImplementedError, and nothing in the training path calls them. |

Not on the list but found, all low: F4 (wp ignores Pinit == 0), F5 (re-injection beyond pass 5), F6 (TF32), and the
D-update bookkeeping in F8.

## Private-write skip fraction (the absolute |target| < 1e-6 floor)

**3.9% of private write targets are skipped (17 / 431; 95% CI about 2.3-6.2%). Only 1.2% (5 / 431) are real nonzero
targets lost to the absolute floor.** The other 12 have E = 0 exactly (K >= 10 steps, pair not re-routed inside the
bptt window). Measured on ckpt_1000 with the trainer's loss scale.

Global: 0.6% (1 / 166).

Conclusion: the relative-threshold patch is safe, but it recovers only about 1% of private writes at the current scale.
It is not worth a mid-run restart on its own; apply it together with the F1 fix at the next restart.

## Files (scratchpad, rev26)
- `checks.py`: resume round trips, selection invariants, D math, eval purity, K > bptt.
- `skip_frac.py`, `skip_frac.log`, `skip_frac2.log`: the skip fraction measurement.
- `yg_dup.py`, `yg_dup.log`: F1 quantified.
- `code/`: the copy of the snapshot that was run.
