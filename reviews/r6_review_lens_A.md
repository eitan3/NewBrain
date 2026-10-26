# r6 review, lens A: result corruption (1) and run-breaking (5)

Run under review: `runs/world_blob25d_12k` (pid 45632). The effective config is in `metrics.jsonl`: preset 25d applied, so gate on, probation writes, int8 probation, Adagrad content optimizer, gauss kernel, no A, 384-d re-rank, G=196,608, P=256, qcap -64.

Method: read-only review of `blob_memory.py`, `blob_gate.py`, `blob_write.py`, `train_lm.py`, `loop_brain.py` and `spiking_lm.py`, plus a diff against `plan_v2/baseline_pre_blob`. CPU-only snippets were used for three things: flags-off identity, chunk-ce equivalence and the probation quantisation. I also compared logs and metrics from the earlier attempts (`_crash300`, `_r2` to `_r5`) and from the running run, which had reached step 750 when this report was written.

Severity labels: CORRUPTS = corrupts results, BREAKS = breaks the run, SPEED, MINOR.

---

## Summary

| # | Severity | Finding | Affects the running run |
|---|---|---|---|
| 1 | CORRUPTS | Probation gradient sketches are stored as int8 with an fp16 per-row scale. For realistic gradient sizes the scale underflows to 0, so the "returns with an agreeing gradient" test (cos >= 0.3) cannot pass for most entries. Every global and private write is then picked by outlier gradient size, not by agreement. | Yes, from step 300 (first writes). Evidence in the running run: `gm_promo_cos_med = 0.0` exactly at steps 500 and 750. |
| 2 | CORRUPTS (eval bias) | The gate's halting features are missing in every eval or val forward (`prev_halt` is never set without targets), so the gate runs off-distribution there. Eval reads saturate the 55% buffer, and the latest positions lose their reads first. | Yes, every val, report and ckpt eval from step 300. Running run: eval `gm_read_frac = 0.5500` (buffer cap) vs training open rate 0.47 to 0.51, which includes 5% exploration. |
| 3 | BREAKS (probable) | Projected VRAM peak at full growth (E=768, 36 heads) is 21.5 to 23.0 GB allocated, against a cap of 22.98 GB (0.896 x 23.89 GiB). In-step fragmentation is not covered by the frag guard. | Risk window is about step 5,500 to 7,500 and after. The projection matched the run exactly so far (17.5 GB at step 500, 19.0 at 750). |
| 4 | MINOR (latent crash) | `gm_metrics` calls `torch.quantile` on the 250-step `_yring`. torch refuses inputs above 16,777,216 values; we are at about 8M per interval, a 2x margin. | No, unless the eval interval or batch grows. |
| 5 | MINOR (design deviation) | The IVF refit rings are never filled on the gated path, so every refit (1000, 2000, ...) fits the stored keys only. | Yes, from step 1000. |
| 6 | MINOR | Content Adagrad `eps = 1e-8` is at or above the per-element value-gradient RMS (about 5e-9), so it is not scale-free. | Yes. |
| 7 | MINOR (causality purity) | `gate.observe_prior` updates the prior's standardisation from the current call's whole batch, later positions included, before scoring (BatchNorm-like). | Negligible. |
| 8 | SPEED | The frag guard fires on essentially every step (about 7% cost). This is the intended OOM fix. | Yes. |
| 9 | MINOR | `nan_abort.pt` is not an exact resume point. | Only on NaN. |
| 10 | MINOR | `_wstats` grows without bound. `_yring` holds up to about 0.24 GB of VRAM between evals. | Yes, small. |

Verified correct, details at the end: flags-off identity (bit-identical model and RNG), chunk-ce semantics, read causality, eval guards, no gradient into writes, checkpoint and resume completeness, births copying spaces and gate rows, int8 key scales.

---

## 1. CORRUPTS: int8 probation sketches underflow, so the promotion test is decided by gradient size

Where:
- `blob_memory.py:392-397` (`_q`) and `:407-411` (`sketch_f`)
- `blob_write.py:412-415` (the sketch is the raw `(-G) @ gproj`)
- `blob_write.py:455-458` (promotion = `hit & cos >= promo_theta`)
- `blob_write.py:506-511` (private writes only for promoted candidates)

```python
# blob_memory.py:392-411
def _q(self, x):
    if self.qdtype == "fp16":
        return x.to(torch.float16), None
    s = x.abs().amax(-1).clamp(min=1e-12) / 127.0
    return torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8), s.to(torch.float16)   # <- s underflows
...
def sketch_f(self, ids):
    g = self.pgrad[ids]
    ...
    return g.float() * self.pgrad_scale[ids].float()[..., None]                                 # <- 0 * q = 0

# blob_write.py:412-415  (raw, unscaled gradient of total/accum, mean over 16,384 tokens)
bm._wqueue.append(dict(key=bm.address(xbar), ..., q=-G, ..., sketch=(-G) @ bm.gproj, ...))

# blob_write.py:455-458
old = pr.sketch_f(j)
cos = F.cosine_similarity(Q["sketch"][:, :pr.kd], old, dim=-1)          # old == 0  ->  cos == 0
hit = act & torch.isfinite(d1) & (d1 < bm.promo_gamma * bm.h0_g ** 2) & pr.occ[j] & (pr.step[j].long() < step)
promoted = hit & (cos >= bm.promo_theta) & (pr.hits[j] + 1 >= bm.promo_min_returns)
```

**Failure scenario.** The sketch coordinates are about |G|/sqrt(384), where G is the summed gradient of the per-micro-batch mean loss at one token's expert inputs. The per-row scale is amax/127, stored in fp16. It rounds to 0 when amax < 3.8e-6, which means |G| below roughly 2e-5 to 3e-5.

CPU check with the real `ProbationStore` (scratchpad `ident.py` session):

| \|G\| | int8: stored/true norm | int8: cos | fp16: cos |
|---|---|---|---|
| 1e-8 | 0 | 0 | 0 |
| 1e-7 | 0 | 0 | 0 |
| 1e-6 | 0 | 0 | 0.95 |
| 1e-5 | 0 | 0 | 0.999 |
| 3e-5 | 1.48 | 0.99997 | 1.0 |
| 1e-4 | 1.14 | 0.99999 | 1.0 |

The logged learning signals put typical |G| far below that threshold: `gm_kappa` is about 2e-9, and `gm_y_mean` is about -7e-5 nats/token after the x16,384 `y_scale`, which gives dl/ds of about 4e-9 per pair.

What the metrics show (`gm_promo_cos_med` is the median gradient cosine among probation hits):
- Running run: exactly 0.0 at steps 500 and 750.
- r3, r4 and r5 (also int8): exactly 0.0.
- Promotions still happen, at about 4% of hits (`gm_promo_rate_of_hits` 0.039 and 0.043), about 37 per step. These can only be entries whose stored |G| was a large outlier, roughly 100x typical.

Consequence: the owner's "keep what comes back" rule has become "keep what comes back if its first visit had an outlier-large gradient". All global spawns and merges go through this test, and so do all private writes, because `ph & prom0[...]` requires promotion. Every memory-content result of this run measures a different write policy than the specified one.

The fp16 default would also fail below about 1e-7 per vector, because its coordinates flush to subnormal or zero, but it is far less extreme.

Fix (needs a restart, or a resume after which probation entries refill within about 3,000 steps): only the cosine uses the stored sketch, so store `pgrad / |pgrad|` (unit rows) in `probation_push`, or keep the scale in fp32. Normalising alone makes both dtypes work. The `y_scale` multiplier is an alternative.

Affects the running run from step 300.

---

## 2. CORRUPTS (eval bias): the gate's halting features exist in training but never in eval

Where:
- `loop_brain.py:607-610` (`per_pass` is False when not training, so `_targets` is None)
- `loop_brain.py:1326-1342` (`lams` is only filled when `tgt is not None`)
- `blob_gate.py:413-417` (the missing halt becomes halt=0, hnone=1)

```python
# loop_brain.py:607-610
per_pass = self.training and targets is not None and (self.cfg["deep_sup"] or self.cfg["halt"]
                                                       or self.cfg.get("think_w", 0))
self._targets = targets if per_pass else None
# loop_brain.py:1331-1342  (tgt = self._targets: None in every eval forward)
elif tgt is not None:
    ...
    if halt:
        lams.append(self.halt_lambda(feat))
if bm is not None and halt and lams and sub is None:   # never true in eval
    bm.prev_halt = lams[-1].detach()
# blob_gate.py:413-417
hp = bm.prev_halt
halt = hp.reshape(-1).float() if (hp is not None and hp.numel() == N) else None
hnone = torch.ones(N, device=dev) if halt is None else torch.zeros(N, device=dev)
halt = torch.zeros(N, device=dev) if halt is None else halt
```

**Failure scenario.** In training, the gate sees the real halting probability and hnone=0 at every pass from 1 on. In every eval forward (`evaluate`, `val_at_train_noise`, tier losses, loop report, `halt_eval`, `think_diag`, sampling) it sees halt=0 and hnone=1 at all passes, an input it only met at pass 0 in training. The per-expert thresholds lambda were tuned in training for a 50% open rate, so the open decisions shift.

Evidence:
- Running run: eval `gm_read_frac` (from `bm.last`, the last eval forward) is 0.55001 at steps 500 and 750. That is exactly `ceil(0.55 * n_pairs)`, the static buffer, so it is saturated.
- Training `gm_open_rate_r` is 0.469 and 0.515, and that figure includes the 5% exploration pairs.
- r3, r4 and r5 show the same pattern.

When the buffer saturates, `compact_causal` drops the latest positions first. Every reported val loss and loop-report number after step 300 is therefore measured with a gate policy different from the trained one, with the late tokens of each sequence memory-starved. This also confounds any "memory helps" reading taken from val.

Fix: either compute the halting lambda in eval forwards too (cheap with chunked read-outs), or remove halt/hnone from the gate features. Then log the eval and train open rate per pass to confirm the gap closes.

Magnitude on val is unknown. The direction of the open-rate shift is evidenced. Part of the gap may come from per-pass differences, because `bm.last` is the last pass only.

---

## 3. BREAKS (probable): VRAM projection over 12,207 steps vs the 0.896 x 23.89 GiB cap

Cap: `set_per_process_memory_fraction(0.896)` allows 21.40 GiB. That equals **22.98 GB** in the trainer's `mem_gb` units (1e9 bytes). The cap applies to reserved memory, so allocated memory plus in-step fragmentation must stay under it.

Measured trajectory (max allocated, cumulative):

| Step | Peak allocated | Note |
|---|---|---|
| 250 | 14.83 GB | before the fit, no reads |
| 500 | 17.52 GB | after the fit |
| 750 | 19.00 GB | spike anneal starts at step 611, +1.45 GB, same as in r4 and the old-memory baseline |

My a-priori projection from r5 (same flags, G=262,144, P=512) was 17.6 / 19.0. It used static savings of -1.81 GB in buffers and -0.40 GB in dense value-grad buffers, so the model of the memory is calibrated.

Growth still to come: E 204 to 768 and active heads 6 to 11 per pass, between steps 611 and 7,324.
- The old-memory baseline `world_mem_life_12k` (same model and growth schedule, ckpt_route 0) grew +4.03 GB from step 750 to 7,500: 0.62 GB per 1,000 steps, flat after 7,500.
- This run recomputes the router (ckpt_route 1), so its growth should be lower. Estimate: params+grads+Muon momentum for 85M new expert weights about +1.0 GB, saved (N,E) wave/fatigue/hist tensors +0.2 to 0.3 GB, extra head activations +1 to 2.5 GB. Total +2.5 to 4.0 GB.
- Projected final peak allocated: **21.5 to 23.0 GB vs 22.98 GB cap.** That leaves 0 to 1.5 GB for in-step fragmentation, which is not covered by the step-end frag guard.

The memory-specific blocks inside the training peak do not shrink with growth:
- `search_global` 384-d re-rank (`blob_memory.py:1065-1067`): `xfull[idx].float()` is (8192, 64, 384) fp32 = 0.8 GB, plus the difference and the square. About 2.4 GB is transient at every pass, on top of the activations of the earlier in-window passes.
- `_cache[call]["gate_rows"]` (`blob_gate.py:480`): `phi[tk]` is about 7.2k x 2,002 fp32 = 57 MB, plus xf/ug/up. About 90 MB per in-window call, about 0.5 GB alive until the next forward.
- `_yring`: up to about 0.24 GB by the end of each 250-step interval.

How to decide early: projected peak(7,500) ≈ mem_gb(1000) + 6.5 x [mem_gb(2000) - mem_gb(1000)]. If that exceeds about 21.5 GB, plan an intervention at a resume boundary (`last.pt` every 1,000 steps). Options:
- Chunk the 384-d re-rank in `search_global` over row blocks. This is value-identical per row, so the resume is unchanged. It saves about 1.6 GB of peak.
- Cap `gate_rows` (store only `okx` rows).
- If the desktop leaves room, raise `--mem-fraction`. Total is 23.89 GiB, so about 2.5 GiB is currently left outside the cap.

Other phases checked, none near the cap:
- IVF refit at every multiple of 1,000: inside `after_step`, after the frag guard; kmeans on 196,608 x 1,024 needs about 2.4 GB transient on a step-end base of about 6 to 8 GiB.
- Dense `apply_value_grads`: about 1.2 to 1.5 GB transient per space. This was the OOM site in r2 and r4 (578 MiB request with 10 to 13 GiB reserved-unallocated). It is now protected by the frag guard, and the request halves with P=256.
- Probation `_exact` lookup: about 1.3 GB transient.
- Eval: batch 16, no grad.
- Loop reports at 3,000, 6,000, 9,250 and 12,207: chunks of 8 sequences, under 1.5 GB transient.
- Checkpoint saves: storages are copied host-side, about 2 GB per file, with 361 GB free on C:.

---

## 4. MINOR (latent crash): `torch.quantile` size limit in `gm_metrics`

Where: `blob_write.py:597-602`.

```python
yr = torch.cat(bm._yring)
yr = yr[yr[:, 2] > 0]
...
q = torch.quantile(yr[:, 0], torch.linspace(0, 1, 11, device=yr.device))
```

The CPU check shows `torch.quantile` raising "input tensor is too large" above 16,777,216 elements. Kept exploration rows per eval interval: about 2.9k per call x about 5.5 in-window calls x 2 micro-batches x 250 steps, about 8M. That is a 2x margin today. A larger `--eval-every`, batch or `--gm-explore` would crash the run at an eval.

Fix: subsample before the quantile, as is already done for `_promo_log`.

---

## 5. MINOR: refit rings stay empty with the gate on

`_ring_add(qd, 0)` is only called in `BlobMemory.inject` (`blob_memory.py:1335-1336`), and only after the fit. The gated path (`blob_gate.inject_gated`) never calls it, and `observe_write` has no caller. So `refit()` (`blob_memory.py:973-981`) runs with nq = nw = 0 and fits only the stored keys, weighted by occupancy.

Cells with no keys keep their step-300 centroids. This is not the documented query/write mix. It is not a crash, and recall may even benefit, but it is a silent deviation from step 1000 on.

---

## 6. MINOR: the content Adagrad epsilon dominates

Where: `blob_memory.py:1390-1394`. The update is `upd = g * lr / (sqrt(v) + 1e-8)`.

The value gradients are raw: loss mean over 16,384 tokens, about 5e-9 per element by the kappa and y logs above. So sqrt(v) is at most about eps for the first tens of touches. The step is then about `lr * g / eps`, which depends on the gradient scale, instead of Adagrad's scale-free `lr * sign`-like step (the clip of 0.05 still bounds it).

Related: rows read with weight 0 (invalid candidates map to slot 0; padded buffer rows have s=0) are marked touched. That means `opt_n` is incremented and the v-cap is applied with a zero gradient.

---

## 7. MINOR: the gate prior uses current-batch statistics before scoring

Where: `blob_gate.py:421-427` (`observe_prior` before `gate.score`) and `:252-260`.

Each training call updates `fstat` from the whole batch, including later positions, then standardises this call's scores with it. The weight is 100% on the first call after the fit (step 300) and a 1% EMA after that.

This is a BatchNorm-style within-batch dependence. Its information content is negligible, but it means the read decisions are not strictly causal. Fix: observe after scoring.

---

## 8. SPEED: the frag guard empties the cache on nearly every step

Where: `train_lm.py:1741-1747`.

After `opt.step()` all activations are freed, so `reserved - allocated` is about the activation pool, well over 3 GB. `empty_cache()` therefore runs, and the next forward re-mallocs everything. Estimated cost about 7%: 10.1k vs 9.4k tok/s at step 500 in r4 vs r5, which is noisy.

The guard is what prevents the r2/r4 fragmentation OOM in `apply_value_grads`, so keep it. It could trigger only when `reserved > cap - frag_gb` to avoid the per-step cost. Windows has no `expandable_segments`.

---

## 9. MINOR: `nan_abort.pt` is not a faithful resume point

Where: `train_lm.py:1719-1730`. The file is saved after this step's data draws and forwards. The RNG has advanced, and several things were already updated mid-step: `cnt_open`/`cnt_all`, `uses`/`touch`, `fstat`, `xf_norm_med` and the `ch_mean` EMA.

`--resume` from it would replay the step with a different batch and double-count those statistics. That is fine for diagnosis, but it should not be resumed as if exact.

---

## 10. MINOR: host-list growth

- `bm._wstats` (`blob_write.py:343`) is appended twice per step and never read or cleared. By the end that is about 24k tiny CUDA tensors, about 12 MB plus Python overhead.
- `_yring` and `_stats_step` are cleared at each eval, about 0.24 GB on the GPU at peak; this is part of item 3.
- `train_lm.py:1574` reuses `rs` (the resume dict) for routing stats. It is harmless, and only keeps the CPU checkpoint alive until the first eval after a resume.

---

## Verified correct (no action)

**Flags-off identity.** CPU build plus one training forward/backward of a small LoopBrain with the run's switches (graded spikes, waves, ws-direct, head groups, ckpt-route, deep-sup, halt, latent think, 4 concept levels). Compared: `baseline_pre_blob`, current with `gm=0`, and current with `gm=1` before the fit.
- Identical: state-dict hash, RNG state after build, RNG state after the forward, loss and logits.
- Gradients agree to 1e-10 relative. That is the same as run-to-run nondeterminism of `index_add_`, since the gradient hashes differ even between two baseline runs.
- BlobMemory and BlobGate draw only from forked generators and are built last.
- Trainer changes are all behind `bm is not None` or flags at their defaults.

**chunk-ce.**
- Final CE: `chunked_readout(...).mean()` is the same as `F.cross_entropy` (same autocast head and fp32 log-softmax).
- Deep supervision: same per-pass CE means, and `aux_x` is None with deep_sup > 0.
- Halting features: the KL to the previous pass is rebuilt from the stored logits, and entropy/log V is unchanged.
- `select_writes` inputs: `(ce, raw H, V)` from the final pass, same math.
- CPU gradient comparison, chunk 0 vs chunk 37: relative difference 2.3e-7 (fp32 reassociation).
- The checkpoint recompute skips the extras, so there are no side effects.

**Read causality.**
- All store and statistic changes happen at step end: value grads, writes, promotions, merges, evictions, renewals, lambda, edges, kappa, sigma^2, gapm, gate AdamW and open_bias. Reads inside a step, including micro-batch 1, see only previous steps.
- IVF tiled mode (qcap -64) drops no pairs, so a query's shortlist is independent of other queries.
- Private search is per pair.
- The 55% buffer is ordered by position, so overflow drops later positions first.
- Exploration is a per-pair hash with a per-forward salt.
- The gate's token features (xf, side, previous-pass mu/halt/read) are per token and causal. The exception is item 7.

**Eval guards.**
- These are behind `_live()`, `rec_on` or `training`: `observe`, `_ring_add`, `_count`, `xf_norm_med`, `observe_prior`, `gate.count`, `_gq`, `_stats_step`, `_edges_A`, `_mgaps`, `_make_rec`, `capture_hooks`/`_kept_pairs`, `select_writes`, `after_step` and `write_blobs`.
- `_cache` is only filled with grad enabled.
- Eval forwards (val, tiers, train-noise val, loop report, halt_eval, think_diag, sample) cannot change any store.
- `gm_metrics` only resets logging accumulators.

**No gradient into writes.**
- Keys, `xfull` and indices are detached buffers.
- Value gradients flow only through the `BagRead` sink into dense fp32 buffers.
- `Scale` returns no gradient for s.
- The capture hooks and `collect_writes`/`step_writes` run under no_grad.
- The gate trains on detached rows with its own graph and AdamW, outside `model.parameters()`, BrainOpt and the global clip.

**No dataset labels.** Only next-token targets (surprise) and the tokenizer EOT id are used.

**Checkpoint and resume (`last.pt` every 1,000 steps, plus `--save-at 1000`).**
- `model.state_dict()` carries:
  - the global and private spaces: keys, values, Adagrad opt_v/opt_n, strength, clocks, occ, cell_pref/cell, spawn_key, `xfull`
  - probation: int8 keys, sketches, scales, step, hits, lost, ptr (the `_ptr` host mirror is restored)
  - mu/sigma, `fitted`, centroids, rings with counters, h0
  - kappa, sigma2, gapm (the running medians), `xf_norm_med`, credit, ch_mean/ch_n, wstats
  - via `_extra_state`: the gate's parameters and buffers (lambda, open_bias, warm_left, fstat, edges, cnt_*) with its AdamW state, the mask/sr/kmeans generator states and the salt RNG
- The derived IVF table, `key_bf` and `ksq` are rebuilt deterministically on load by a post-hook that runs after the children. The live table is always rebuilt after every key or occupancy change, so it equals a fresh rebuild.
- The trainer saves all global RNGs, BrainOpt, neuro, counters and group_passes.
- The save happens at the loop top, so the per-step host lists are empty.
- Only logging accumulators are lost on resume (`_yring`, `_wcount`, `_mstats`, `rate_sum`/`rate_n`).

**Births and splits.**
- Private spaces and gate rows are preallocated for 768+1, so no grow path is taken.
- A split halves the parent's strength and resets its Adagrad state, then copies the space to the child. It also copies the gate rows c/beta/ge with their AdamW moments, lambda and credit.
- Indices are append-only, so the shared expert stays at 768.
- Births happen at step start, after the previous step's value grads were applied and zeroed.

**Other checks.**
- int8 probation keys: x_hat coordinates are about N(0,1), so the scale is about 0.02 and fine. Only the sketches underflow (item 1).
- `hash_uniform`'s int64 wrap keeps the correct low 32 bits.
- `_put` handles an all-invalid index list.
- `sample()` with a 4-token prompt goes through all static shapes.
- `torch.nn.utils.get_total_norm` exists (torch 2.11).
