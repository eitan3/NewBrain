# 12 memory plan: review findings and resolutions (revision 2, 2026-10-23)
Reviewers A, B, C (independent). All three verdicts: SOUND WITH FIXES. Plus the coordinator's notes (N1-N7).

## The fundamental limitation (all three reviewers)
- Freezing the translator freezes the MAPPING thought → address, not the thought. The trunk, level heads and
  workspace keep training after the freeze, so addresses drift with the input. Measured on random keys:
  1% input change ≈ 97.5% top-6 overlap, 3% ≈ 93%, 10% ≈ 79%. A 0.99 stability target needs < 0.3% drift.
- Resolution: the core design is kept (a standard product-key memory). The promise is changed into a measurement:
  ω-weighted stability vs the freeze-time addresses, input drift per level, offset overlap; guard = alarm at 0.95.
  Mitigations that don't constrain thoughts: sharper reads (τ 0.03), sg(a) on the addressing input after the freeze,
  set-boundary margins, redundancy. Constraining the thoughts was declined by Eitan; this is the decision point.

## Errors and resolutions
| # | Finding (who) | Resolution (section) |
|---|---|---|
| 1 | Margin on 1st-2nd gap protects the wrong boundary; δ = 0.05 is 10× natural gaps (A, C) | γ = 6th-7th pair gap + per-half m-th/(m+1)-th; separate sg(a) pass; δ = 0.01 (2.3) |
| 2 | Repair overwrites busy shared rows, blends the wrong target, resets R⁰ (hides slow drift), Goodharts the probes (A, B, C) | Default measure-only; if on: β 0.25, only low-usage rows, copy the dropped row, R⁰ fixed, disjoint probe sets (5) + N4 |
| 3 | Activations/VRAM underestimated (+4.5-5.5 GB act., peak 22-24 GB) (A, B, C) | Indices under no_grad; checkpointed per-pass memory block; one fused MemRead with one preallocated grad buffer; private via MemRead; revised ≈ 18-18.5 GB + fallbacks (7) |
| 4 | FLOPs wrong (level heads 1.93M, W_o double-counted, infancy paths missing; +90-130%) (A, B) | Recomputed (7); step time 1.8-2.5×; W_p after the combination (6× cheaper) (4) |
| 5 | Random head-group offset makes training writes and eval reads address differently (A) | Probes at a fixed offset, fixed batches; offset-overlap metric logged (6) |
| 6 | "Separate rows" per head false: heads share the level region (A, B) | Stated as multi-hash (2.1) |
| 7 | Adagrad scale-invariant: first write η√d_V whatever ω; G-clip ineffective (A, B, C) | S₀ prior (auto = 100× median G²) (N1), η 0.01, clip the update ΔV, write start 200 (3.1) |
| 8 | Tables would enter BrainOpt (Muon on 1M×256), the global clip, carry_brain_state, rebuilt on births (A, B, C) | Tables are manual tensors in MemOpt; keys on AdamW no WD (3, 4, 7); test 15 |
| 9 | Test 7 wrong (at step 0 only W_o/W_ms/W_p/W_in/W_fb get gradient) (A, B) | Fixed (test 7) |
| 10 | Row offset l·S² with l = 1..4 (A, B) | (l-1)·S² (2.1) |
| 11 | NLL denominator ×B (16× too small) (A, B) | Σ_{b,t} v · d (1.3) |
| 12 | Hinge std via nonzero() = host sync (A, B) | Masked moments (1.3) |
| 13 | bf16 autocast / TF32 make addressing not fp32 (A, B, C) | autocast off + TF32 off for addressing; test 6 under bf16 (0) |
| 14 | Sub-key split / re-seed re-points a band of rows to stale content (A, B) | Band copy of V and S, MemOpt reset; dead re-seed resets S (2.2); test 12 |
| 15 | Freeze at t1 misses later and ramp births; grow_experts would re-enable grads (A, B) | t_f = first step ≥ 0.6·steps with neuro.done() (N6); re-apply on grow and reload (2.2) |
| 16 | Tie-break by lowest index needs a composite sort (B, C) | Sort by flat index, then stable by score (2.1) |
| 17 | b(t) must be the SHIFTED suffix-min; use input idx; EOT positions predict the next doc (A, B) | Fixed (1.1) |
| 18 | sg?(μ_{l-1}) contradicts "with gradient" (A) | No sg (1.2) |
| 19 | W_p is 384×256 = 98K, not 25K (A, B) | Fixed (7) |
| 20 | Slot-15 in-place write; not truly dedicated (B, C) | Out-of-place cat; stated as shared (3) |
| 21 | Private selection must use EMA q̄¹; empty capacity slots in stats (B, C) | Fixed (4) |
| 22 | Private tables grow on births → reallocation / opt rebuild (B) | Preallocated at 768 experts (4) |
| 23 | EMA norm stats updated per pass/micro-batch (C) | Once per optimizer step (2) |
| 24 | Probes as 2,000 single positions are slow; RNG for balancing/probes perturbs the main stream (B, C) | 32 sequences × 32 positions per set; dedicated RNG; new modules built last (6, 1.4) |
| 25 | Pass-0 read is token-keyed (A, C) | Kept, reported separately, flag --mem-read0 (3) |
| 26 | Private query reuses q¹ → correlated with shared addresses (A) | Stated (4) |
| 27 | Freeze θ ← θ̄ changes ω once (A) | Logged at the freeze (2.1) |
| 28 | Writes are effectively dense (~7 hits/row/step): "consolidation" ≈ global LR decay, rows = prototypes (A, C) | Stated; dense grad buffer kept (sparse COO ≈ 3.6 GB) (3.1) + N2 |
| 29 | τ = 0.1 too soft for stable reads (C) | τ = 0.03; ω-weighted stability (2.1, 6) |

## Coordinator notes
- N1 S₀ with --mem-s0 (shared and private): done (3.1).
- N2 sparse table grads: NOT taken. At ~7 writes per row per step the sparse gradient would be ≈ 3.6 GB vs the dense
  1.07 GB buffer (my count and reviewer C). VRAM is cut instead by checkpointing + one fused grad buffer.
- N3 soft clamp ν = -1 + 5 tanh(raw/5): done (1.2).
- N4 repair only on low-usage rows or β = 0.25, flag, counts: done, default measure-only (5).
- N5 W_in rank 64 default: done (1.4).
- N6 freeze at 0.6 of each run, stability logged from the freeze to the end: done (2.2, 6, 10).
- N7 no-memory control: run 2 (10).
