# 16. Why does the memory address mostly the same rows? (investigation only, no code) — 2026-10-24

Scope: the shared product-key memory of `runs/wiki_mem_12k` (thought-only addressing `--mem-read0 0`, 4 levels × 3 heads,
S = 512 per half, top-m 8, top-k 6, τ 0.03). Scripts, raw outputs and JSON: `speed_wip/addr_inv/` (`capture.py`,
`analyze.py`, `gen_tables.py`, `res_*.txt|json`).

## Setup
- **Points**
  - fresh init of the 12k config (seed 0, step 0);
  - `speed_wip/area3/step1000.pt` (**old run, different config**: word-keyed read0 ON, older memory.py, 225 experts);
  - the 12k run at **step 6000** (`last.pt` copied 00:16);
  - the 12k run at **step 7000** (`last.pt` was rewritten while I worked, copied 00:46; 325 steps before the freeze at about 7325).
  - "init addr": the step-6000/7000 thoughts addressed with the **init** W_q, norm stats (m = 0, v = 1) and init sub-keys.
- **Fixed val batch everywhere**: 32 windows × 512 tokens = 16,384 positions (`val.bin`, RandomState(1)). CPU, 4 threads,
  eval, K = 6, no writes, no routing noise.
- **Reads**: reads after passes p1..p5 (k = 0..4). Every read is addressed fresh (readdr-every 1), so the p5 row sets are
  what fresh addressing would pick; in training, p5 reuses p4's rows. The old run also has the word-keyed read "w0".
- **Selection source**: selections come from the real `Memory.select` (hooked). Re-addressing offline reproduces it: top-1
  sub-key agrees 99.99%, row sets 99.9% (ties / fp order).
- **Metrics**
  - The selection-based numbers are mean values over the 3 heads unless stated otherwise.
  - "Distinct rows per 1,000 positions" = distinct rows among 1,000 random positions × 6 rows (ideal uniform: 5,932 per
    head; 17,396 pooled over a level's 3 heads, which share one row table).
  - "Sub-key dead in batch" = never in any top-8 over the 16,384 positions.
  - PR = PCA participation ratio (Σλ)²/Σλ²; erank = exp(entropy of the normalised singular values).
- **Caveats**
  - 16 K positions from 32 windows cover only about 45 documents. Since addresses are strongly document-dependent (below),
    absolute "distinct" counts are batch-dependent; compare points, not absolutes.
  - There is no optimizer state in the checkpoints, so key "gradients" are inferred from the online-vs-EMA drift.

## Headline: is it "not trained yet, so the same address every time"? — **Contradicted**

| point | distinct rows /1k pos per head, L0 / L1 / L2 / L3 (ideal 5932) | pooled over heads, L0 (ideal 17396) | sub-key dead in batch | a: PR / erank (of 448) | â: PR / erank | q: PR / erank (of 64) | MI(top-1 sub-key; doc) / (; token), nats |
|---|---|---|---|---|---|---|---|
| fresh init (step 0) | 4208 / 4391 / 4500 / 4541 | 12387 | 0.2-0.6% | 87 / 246 | 87 / 246 | 38 / 59 | 0.08 / 2.0 (token-keyed) |
| old run s1000* (read0 on) | 2648 / 2683 / 2662 / 2593 | 7865 | 3-9% | 3-7 / 97 | 3-7 / 104 | 7 / 33 | 1.8 / 0.26 |
| 12k s6000 | 2077 / 1995 / 1997 / 1805 | 6198 | 24-33% | 3-4 / 89 | 5-6 / 132 | 11-12 / 35 | 1.8-1.9 / 0.30 (doc-keyed) |
| 12k s7000 | 2137 / 2061 / 2132 / 1894 | 6376 | 26-34% | 3-4 / 90 | 4-7 / 134 | 11-13 / 36 | 1.8-2.0 / 0.28 |
| **s6000 thoughts + INIT W_q / norm / keys** | 3642 / 3041 / 3398 / 2871 | 10778 | 1-7% | 3-4 / 89 | = a | 15-18 / 46 | 1.8 / 0.28 |
| **s7000 thoughts + INIT W_q / norm / keys** | 3607 / 2889 / 3466 / 2621 | 10686 | 2-8% | 3-4 / 90 | = a | 15-18 / 46 | 1.8 / 0.27 |

Concentration (the share of a head's 98 K row draws on the top 1% of rows, and the Gini of the row counts):
- uniform random draws: 0.07, Gini 0.73;
- init: 0.45-0.52, Gini 0.94-0.96;
- old s1000: 0.82-0.83, Gini 0.989;
- s6000 / s7000: 0.91-0.95, Gini 0.994;
- init addressing of the s6000 thoughts: 0.65-0.80, Gini 0.98.

The full per-level table is T1.

**Plainly: the data contradicts the hypothesis.**
1. The untrained network spreads its addresses the most. Concentration grows with training: init → old s1000 → s6000;
   s6000 → s7000 is flat.
2. The trained thoughts are clearly differentiated:
   - row sets of different documents almost never overlap: 0.004 of 6 rows, against 0.25-0.32 for neighbouring tokens of the
     same document;
   - the top-1 sub-key alone carries about 1.9 of the 3.57 nats of document identity.
   - At init, the addresses were keyed by the current token (same token in another document: 0.17 overlap; document MI ≈ 0).
3. The collapse has two parts:
   1. **The thoughts became very low-dimensional / anisotropic:** a PR 87 → 3-4, â PR 5-6; the top 5 PCA directions hold
      62-72% of the variance of â. This alone takes the init addressing from about 4,400 to about 3,200 distinct rows per head
      per 1k positions.
   2. **The learned addressing (W_q, sub-keys) concentrates further:** about 3,200 → about 2,000. Dead sub-keys go from
      2-8% to 25-34%.

## Verdicts per hypothesis

**H1: the thoughts are not yet differentiated (untrained). — Contradicted.**
- The addresses carry document content: T6 MI(doc) 1.7-2.0 nats; same-document overlap decays with lag
  (0.31 → 0.10 from lag 1 to 256), and different-document overlap is 0.004.
- The untrained point is the most spread (T1).
- What *is* true: the trained thoughts live in very few high-variance directions (T4: a cos ≈ 0.11-0.20, â top-5 = 0.62-0.72).
  So "same address" means "same document / same slow topic state", not "same default thought".
- A partial exception: window-start positions (t < 4) of different documents share rows (overlap 0.05-0.14 vs 0.005 at
  t ≥ 256), i.e. a generic "no context yet" thought.

**H2: the query cloud is collapsed into a cone / shared mean, or low rank. — Supported as LOW RANK / ANISOTROPY, not as a shared mean.**
- **Rank**:
  - q has PR 11-13 of 64; the top 20 directions hold 95-96% of the variance (init: 63%);
  - the sub-keys spread over PR 45-46 of 64 (T4).
  - So most sub-key directions are never visited by any query.
- **A cone only on the early passes**: q resultant R = 0.49-0.64 at p1 (coarse levels worst), 0.21-0.26 at p3-p5 (T2, T4).
- **Shared mean (counterfactual T8, p4)**:
  - subtracting the batch mean of q per half raises distinct rows only ×1.1-1.2 (7.4 K → 8.3-9.2 K per head over 16 K
    positions);
  - PCA-whitening â (same trained W_q and keys) raises them ×2.0-2.3 (→ 15-17 K) and cuts dead sub-keys from 23-28% to 4-6%;
  - whitened â + random keys reaches 32-38 K, ×4.5-5.
- **Verdict**: the collapse is "a few dominant directions of the thought", which the per-coordinate EMA standardisation does not
  remove. It is not one shared mean direction.
- The coarse-level, early-pass reads are the most coned (T3: L3 at p1 = 1,160-1,295 distinct, vs about 2,100 for L0).

**H3: rich-get-richer key dynamics (busy keys attract, dead keys stranded). — Supported for the keys; not super-linear for the rows.**
- **Dead keys are outside the query cloud** (T5, s6000/s7000):
  - their best cosine to any query is 0.30, below the typical 8th-best score of 0.39-0.40, so they cannot enter a top-8;
  - busy keys reach 0.75-0.77, and Spearman(usage, max cos) = 0.93-0.94.
- **Dead keys don't move**: online-vs-EMA drift 0.05-0.06, vs 0.16-0.18 for mid/busy keys; Spearman(usage, drift) = 0.63-0.74.
  Only selected keys receive gradient (w over the chosen pairs + the margin at the boundary), so a key that falls out of
  every top-8 is frozen in place until a balancing event.
- **Busy keys sit toward the query mean**: cos to the mean query direction +0.10…+0.20 vs −0.05…−0.10 for dead ones.
- **Trained keys are worse than random keys** for these thoughts (T8: random unit keys give 2.1-2.3× more distinct rows and
  1-2% dead).
  - At old s1000 it was the opposite: there the keys had followed the query cone (keys R 0.52-0.64, PR 13-15), and random keys
    were worse.
  - So the keys over-fit an earlier, narrower query distribution and then got stranded when the queries moved.
- **Re-seeding the dead keys** from batch queries helps (7.4 K → 10-12 K) but leaves 12-16% dead: the cloud is too thin to
  give 40% of the keys a region of their own.
- **Rows, s6000 → s7000** (rows of the bands edited by the step-7000 balance excluded):
  - Spearman(count₆₀₀₀, growth) = 0.72-0.75, i.e. persistent;
  - but the top-1% rows took 8-10% of the growth, the same as their 8.6-9.9% share of the counts, and relative growth is
    *higher* for rare rows (count 1-10: +60%, 1000+: about +13% per 1000 steps).
  - So the row popularity is stable and proportional, not self-amplifying.
- **Gates**: they barely prefer popular rows (T10: Spearman +0.1…+0.33 at p1-p2, ≈ 0 or negative at p4-p5).

**H4: the product-key halves are correlated, so row usage is worse than per-half usage. — Supported, a secondary factor.**
- Over the selected pairs, MI(i; j) = 0.9-1.3 nats at s6000/s7000, i.e. the row perplexity is ×2.5-3.8 lower than independent
  halves would give (T7). It was 0.55-0.6 at init and 1.9-2.0 at old s1000.
- With the init addressing on the s6000 thoughts it is still 1.2: the dependence comes mainly from the low-rank thought (both
  halves are projections of the same few directions), not from learned key correlations.
- The per-half loss is larger: H(i) ≈ 4.2-4.6 nats vs log 512 = 6.24, about 1.7-2 nats per half.

**H5: a weak design signal; nothing in training rewards spreading. — Supported.**
- The only pressures on the addressing are:
  - the read loss on the *chosen* pairs (gradient only to selected keys);
  - the margin (a boundary gap, not spread);
  - balancing every 1,000 steps with a 2% cap and `--mem-balance-keep` (a key used once since the last event is never
    re-seeded).
- The usage-balance loss of plan 15 (a) is not implemented.
- The thought itself is trained only by the LM / concept losses, which evidently prefer a few dominant directions; there is no
  isotropy or whitening of the addressing input. The EMA norm is per coordinate.
- The dead share rises under this regime (log: 0.36 → 0.40), and the evidence above shows that nothing pulls a stranded key
  back.

## Other requested points

**d. Who uses the popular rows** (T9):
- The top-50 rows by write count (about 5,790 writes, i.e. written in essentially every step since write-start; median row
  53-103) take 1-2.6% of all row draws (≈ 100× uniform), and 10-15% of positions touch one.
- Their users are **broad**: 1,100-1,600 distinct tokens, a token entropy of 8.5-8.9 bits, all 32 windows, all passes about
  equally, and a median frequency rank of 140-300 (batch: 310).
- There is mild enrichment for newline ("NL" 6-12% vs 3.1% batch), ":" and window starts (pos < 16: 4-7% vs 3.1%). EOT /
  after-EOT positions are not enriched.
- So the top rows are a **generic "common-context" region of the low-rank thought space**: Zipf-like frequent contexts plus
  newline / list structure, not one degenerate default thought.
- Only 3-5 of the batch's own top-50 rows are among the all-time top-50, so locally popular rows vary with the document.

**e. Rich-get-richer between snapshots**: see H3 (proportional, persistent). Gate vs popularity: weak (T10).

**g1. 20 diverse contexts** (T11, pass p3, position 400 of 20 different windows):
- per head, 111-119 distinct rows of 120;
- 1-9 rows shared by 2 or more contexts, with a max multiplicity of 2-3;
- pooled over the 3 heads, 333-358 distinct of 360.
- Diverse contexts mostly get *different* rows; the skew is a frequency effect over many positions, not one shared row set.
- At init this was 357-358/360.

**g2. Distinct rows per 1,000 positions**: T1/T2/T3. About 2,000 of 5,932 per head now, 4,400 at init, 2,650 at old s1000
(and only 660 for its word-keyed w0 read).

**Level / pass dependence (5, 6):**
- The coarse level (128, L3) is the most collapsed, but only on the first passes: at p1, L3 has 1,160-1,295 distinct, q
  R = 0.64, 51% dead; at p3 the levels are equal (T3).
- Mid passes (p3-p4) are the most spread; p5 drops back a little.

## Conclusion: what causes the skew
The addressing is content-driven (document / slow topic state) and **the thought is trained, not untrained**. The skew
comes from three mechanisms, in order of size:
1. **Anisotropic, low-rank thoughts.** Training squeezes â into about 5-20 strong directions, and the EMA per-coordinate
   standardisation passes that straight into the queries (q in about 20 of 64 dims). A product-key codebook spread over 46
   dims can only be reached in a thin slab, so many keys are geometrically unreachable. This is the largest single factor:
   whitening alone gives ×2-2.3 distinct rows, and the init→trained thought change alone ×0.7.
2. **Stranded keys with no restoring force (rich-get-richer at the sub-key level).** Only selected keys get gradient. Keys
   adapted to an earlier narrower cloud (s1000) and are now left outside the cloud (best cos 0.30 < the 0.40 entry score);
   about 40% are dead by U, and trained keys now do worse than random ones (×0.45). The rate-limited, keep-protected
   balancing cannot keep up.
3. **Correlated halves.** MI(i; j) ≈ 1-1.3 nats costs another ×3 in row perplexity, mostly inherited from (1).

Not causes:
- an untrained network;
- a single shared-mean direction (centring gives only ×1.1-1.2);
- self-amplifying row popularity (growth is proportional);
- gates favouring popular rows.

The early-pass coarse-level cone and the window-start sharing are smaller, local effects.

## Fix directions to discuss (ranked by the evidence; no code proposed)
1. **Decorrelate / whiten the addressing input**, e.g. a running PCA-whitening or a batch-whitening of â in place of the
   per-coordinate EMA norm (in the counterfactual, ×2-2.3 distinct rows and dead sub-keys 25% → 5% with the SAME trained W_q
   and keys). Any change must respect the freeze (the statistics are frozen at t_f).
2. **Give unselected keys a restoring force**: the usage-balance loss of plan 15 (a) over all S scores (soft usage), or a
   cheaper pull of dead keys toward the query distribution. Also faster or uncapped dead re-seeding (re-seeding alone gave
   ×1.35-2; dropping `--mem-balance-keep` / raising `--mem-balance-dead-max` are the knobs).
   - With 1, random-quality keys would already give ×4.5-5 (whitened + random keys).
3. **Spread the queries in W_q**, i.e. regularise W_q (or q) toward isotropy over the batch (a decorrelation / covariance
   penalty on q per half). This also lowers the half-to-half MI (H4).
4. **Make the halves independent**: separate input projections or a decorrelation term between q¹ and q². This is a secondary
   gain (about ×1.5-3 in row perplexity at most), and partly solved by 1/3.
5. **Timing**: the freeze at about step 7,325 fixes the current keys, W_q and norm stats. Fixes 1-3 act on those, so in this
   run they would come too late. They are for the next run, from the start of infancy, as plan 15 says for (a).
6. **Not supported by the evidence**:
   - waiting for more training (the skew is flat from 6000 to 7000 and grew from init);
   - centring the queries only;
   - anti-popularity row penalties (row growth is not self-amplifying).

## Tables (all per level; "old s1000*" = the older run with word-keyed read0 ON and an older memory.py, a different config)

