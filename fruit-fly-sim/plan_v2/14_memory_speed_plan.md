# Memory speed package: plan (2026-10-23)
Eitan chose "same math + mild" from 13_memory_speed_review.md. Built in a copy (C:\Eitan\Python\NewBrain\speed_pkg),
tested against plan_v2/baseline_mem_v2 (= fruit-fly-sim after the read0 fix), reviewed by the coordinator, and only then
copied into fruit-fly-sim. The memory runs relaunch with --mem-read0 0 + this package after the review OK.

## Same math (always on; results identical to the current code, up to atomic summation order where noted)
S1. Shared sub-key top-m: torch.topk(m+1) + a small stable re-sort by (score desc, index asc) instead of a full stable sort
    of 512. Tie rule kept (lowest index wins): when the k-th and (k+1)-th values are equal, those rows use the old path.
S2. Private top-4: computed only for the kept capacity slots (not the whole (E, C, 512) incl. empty slots), via topk + the
    same tie rule.
S3. Balancing vectorised: no per-key Python loop / .item(); the same RNG draws in the same order (bit-identical).
S4. Margin inside the online pass: reuse the online q and gather the boundary keys together with the read keys (NE).
S5. MemRead backward: one flattened index_add_ instead of 6; dω by one gather + bmm (NE: atomic order).
S6. Writes: row-Adagrad / private-key Adam / EMA lerps only over touched rows and live experts (one sync placed next to
    the existing per-step loss sync); no full-table torch.where / copy_ sweeps. Bit-identical.
S7. select(): the per-level Python loop batched.
S8. After the freeze: online = EMA addressing, so the weights reuse the EMA queries (no second query matmul).
S9. --mem-probe-every N (default 1 = as now).

## Mild (each its own flag; default OFF in code, ON in the new memory runs)
M1. --mem-readdr-every 2: new addresses (indices) only on every 2nd read; the other reads reuse the previous read's indices,
    while the online weights are still recomputed from the current thought. Reads still happen on every pass.
M2. --mem-margin-passes 1: the margin loss on 1 random pass per forward (from the grad window) instead of every pass.
M3. Calmer balancing: --mem-balance-every 1000, --mem-balance-max 0.02 (at most 2% of keys per codebook per event),
    no re-seed of a key used in the last window (U above a floor).

## Tests
- All existing tests (143) pass.
- Equivalence vs baseline_mem_v2 over several training steps with memory on (concepts + shared + private, growth on, a
  balancing event, the freeze): losses, outputs, gradients, tables, addresses identical (bit-identical where exact;
  allclose with a tight tolerance for S4/S5; identical indices on tie-free data for S1/S2) with the mild flags OFF.
- Tie test: the lowest-index rule still holds for S1/S2.
- Each mild flag: does what it says (index reuse on alternate reads, one margin pass, the balancing caps).

## Then
A 20-step GPU profile, old vs new (torch.profiler), in the next GPU gap, and the coordinator's review of diff + profile.
