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

### T1. Headline per level (means over passes p1-p5 and 3 heads; fixed batch of 16,384 positions)

| point | level | distinct rows /1k pos, per head (ideal 5932) | pooled over heads /1k (ideal 17396) | rows: top-1% share / Gini (uniform draws: 0.07 / 0.73) | sub-key ent/log S | sub-key dead in batch | a PR / erank | â PR / erank | q PR / erank (of 64) |
|---|---|---|---|---|---|---|---|---|---|
| init | 0 (16) | 4208 | 12387 | 0.52 / 0.955 | 0.885 | 0.006 | 87.1 / 245 | 87.1 / 245 | 37.3 / 59 |
| init | 1 (32) | 4391 | 12968 | 0.48 / 0.950 | 0.894 | 0.006 | 86.9 / 247 | 86.9 / 247 | 38.0 / 59 |
| init | 2 (64) | 4500 | 13262 | 0.46 / 0.946 | 0.901 | 0.004 | 88.7 / 247 | 88.7 / 247 | 38.1 / 59 |
| init | 3 (128) | 4541 | 13413 | 0.45 / 0.942 | 0.910 | 0.002 | 88.6 / 247 | 88.6 / 247 | 38.0 / 59 |
| old s1000* | 0 (16) | 2648 | 7865 | 0.83 / 0.989 | 0.896 | 0.088 | 6.6 / 103 | 6.7 / 110 | 7.0 / 33 |
| old s1000* | 1 (32) | 2683 | 7955 | 0.82 / 0.989 | 0.903 | 0.044 | 5.6 / 95 | 5.9 / 101 | 6.9 / 32 |
| old s1000* | 2 (64) | 2662 | 7893 | 0.82 / 0.989 | 0.892 | 0.030 | 4.2 / 99 | 4.4 / 106 | 7.4 / 34 |
| old s1000* | 3 (128) | 2593 | 7701 | 0.83 / 0.989 | 0.893 | 0.033 | 3.2 / 90 | 3.2 / 97 | 6.9 / 33 |
| s6000 | 0 (16) | 2077 | 6198 | 0.92 / 0.994 | 0.804 | 0.238 | 2.8 / 89 | 4.8 / 132 | 12.5 / 36 |
| s6000 | 1 (32) | 1995 | 5949 | 0.93 / 0.994 | 0.778 | 0.292 | 3.9 / 92 | 6.2 / 136 | 11.1 / 34 |
| s6000 | 2 (64) | 1997 | 5960 | 0.92 / 0.994 | 0.773 | 0.287 | 2.9 / 92 | 5.4 / 134 | 12.0 / 35 |
| s6000 | 3 (128) | 1805 | 5391 | 0.95 / 0.995 | 0.731 | 0.333 | 3.3 / 84 | 4.9 / 127 | 10.8 / 34 |
| s7000 | 0 (16) | 2137 | 6376 | 0.91 / 0.993 | 0.804 | 0.258 | 3.0 / 93 | 5.9 / 138 | 13.0 / 37 |
| s7000 | 1 (32) | 2061 | 6151 | 0.92 / 0.994 | 0.784 | 0.312 | 3.8 / 92 | 7.4 / 137 | 11.4 / 35 |
| s7000 | 2 (64) | 2132 | 6360 | 0.91 / 0.994 | 0.781 | 0.280 | 2.6 / 91 | 4.9 / 134 | 12.1 / 36 |
| s7000 | 3 (128) | 1894 | 5657 | 0.94 / 0.995 | 0.744 | 0.339 | 2.7 / 84 | 4.1 / 128 | 10.8 / 35 |
| s6000 thoughts + init addr | 0 (16) | 3642 | 10778 | 0.65 / 0.976 | 0.903 | 0.014 | 2.8 / 89 | 2.8 / 89 | 17.6 / 47 |
| s6000 thoughts + init addr | 1 (32) | 3041 | 9023 | 0.76 / 0.985 | 0.865 | 0.059 | 3.9 / 92 | 3.9 / 92 | 15.7 / 46 |
| s6000 thoughts + init addr | 2 (64) | 3398 | 10054 | 0.69 / 0.979 | 0.894 | 0.026 | 2.9 / 92 | 2.9 / 92 | 17.4 / 47 |
| s6000 thoughts + init addr | 3 (128) | 2871 | 8514 | 0.80 / 0.987 | 0.856 | 0.067 | 3.3 / 84 | 3.3 / 84 | 15.2 / 45 |
| s7000 thoughts + init addr | 0 (16) | 3607 | 10686 | 0.66 / 0.976 | 0.907 | 0.015 | 3.0 / 93 | 3.0 / 93 | 18.0 / 47 |
| s7000 thoughts + init addr | 1 (32) | 2889 | 8577 | 0.79 / 0.987 | 0.862 | 0.062 | 3.8 / 92 | 3.8 / 92 | 15.6 / 46 |
| s7000 thoughts + init addr | 2 (64) | 3466 | 10272 | 0.69 / 0.979 | 0.893 | 0.031 | 2.6 / 91 | 2.6 / 91 | 17.8 / 47 |
| s7000 thoughts + init addr | 3 (128) | 2621 | 7798 | 0.83 / 0.989 | 0.842 | 0.083 | 2.7 / 84 | 2.7 / 84 | 15.3 / 45 |

### T2. Usage per pass (mean over levels and heads): distinct rows /1k pos per head / sub-key dead / q resultant R (cone)

| point | w0 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| init | - | 4124 / 0.00 / 0.36 | 4449 / 0.00 / 0.40 | 4513 / 0.00 / 0.44 | 4491 / 0.01 / 0.46 | 4473 / 0.01 / 0.47 |
| old s1000* | 663 / 0.44 / 0.63 | 2757 / 0.03 / 0.43 | 2596 / 0.04 / 0.45 | 2618 / 0.05 / 0.48 | 2612 / 0.06 / 0.52 | 2648 / 0.06 / 0.55 |
| s6000 | - | 1680 / 0.40 / 0.51 | 1967 / 0.31 / 0.38 | 2145 / 0.25 / 0.23 | 2093 / 0.23 / 0.21 | 1957 / 0.25 / 0.27 |
| s7000 | - | 1798 / 0.40 / 0.49 | 2078 / 0.32 / 0.35 | 2221 / 0.26 / 0.22 | 2148 / 0.25 / 0.21 | 2034 / 0.26 / 0.26 |

### T3. Pass x level, distinct rows /1k pos per head (12k run)

| s6000 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|
| L0 | 2097 | 2115 | 2100 | 2063 | 2011 |
| L1 | 1853 | 2069 | 2090 | 2013 | 1949 |
| L2 | 1612 | 1999 | 2207 | 2170 | 1996 |
| L3 | 1160 | 1684 | 2183 | 2126 | 1871 |

| s7000 | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|
| L0 | 2126 | 2151 | 2162 | 2132 | 2113 |
| L1 | 1998 | 2145 | 2116 | 2049 | 1998 |
| L2 | 1774 | 2161 | 2322 | 2255 | 2148 |
| L3 | 1295 | 1854 | 2286 | 2155 | 1878 |

### T4. Thought / query geometry (mean over passes). top1/5/20 = variance share of the top PCA directions; cos = mean pairwise cosine of random positions (uncentred); R = length of the mean unit vector

| point | level | a: top1/5/20, cos | â: top1/5/20, cos | q: top1/5/20, cos, R | keys: PR (of 64), mean cos, R |
|---|---|---|---|---|---|
| init | 0 | 0.04/0.16/0.37, 0.196 | 0.04/0.16/0.37, 0.197 | 0.06/0.25/0.63, 0.195, 0.44 | 57.0, 0.000, 0.05 |
| init | 1 | 0.04/0.16/0.37, 0.182 | 0.04/0.16/0.37, 0.182 | 0.06/0.24/0.63, 0.190, 0.43 | 57.0, -0.000, 0.04 |
| init | 2 | 0.04/0.15/0.37, 0.177 | 0.04/0.15/0.37, 0.178 | 0.06/0.24/0.63, 0.177, 0.42 | 56.9, 0.000, 0.05 |
| init | 3 | 0.04/0.16/0.37, 0.171 | 0.04/0.16/0.37, 0.171 | 0.06/0.24/0.63, 0.164, 0.40 | 57.0, 0.000, 0.04 |
| old s1000* | 0 | 0.36/0.64/0.92, 0.327 | 0.36/0.63/0.91, 0.109 | 0.27/0.75/0.97, 0.233, 0.48 | 14.5, 0.272, 0.52 |
| old s1000* | 1 | 0.40/0.67/0.94, 0.337 | 0.39/0.66/0.93, 0.104 | 0.29/0.74/0.97, 0.252, 0.50 | 12.8, 0.370, 0.61 |
| old s1000* | 2 | 0.48/0.72/0.94, 0.326 | 0.47/0.71/0.93, 0.108 | 0.26/0.73/0.97, 0.238, 0.48 | 14.9, 0.370, 0.61 |
| old s1000* | 3 | 0.56/0.77/0.95, 0.364 | 0.55/0.76/0.94, 0.115 | 0.28/0.75/0.97, 0.237, 0.48 | 13.2, 0.403, 0.64 |
| s6000 | 0 | 0.56/0.82/0.94, 0.113 | 0.41/0.68/0.83, 0.069 | 0.16/0.55/0.95, 0.078, 0.27 | 46.6, 0.020, 0.15 |
| s6000 | 1 | 0.42/0.79/0.94, 0.183 | 0.34/0.62/0.81, 0.071 | 0.18/0.59/0.96, 0.100, 0.30 | 46.6, 0.017, 0.14 |
| s6000 | 2 | 0.55/0.81/0.94, 0.122 | 0.35/0.66/0.82, 0.075 | 0.16/0.56/0.96, 0.132, 0.34 | 46.3, 0.018, 0.14 |
| s6000 | 3 | 0.47/0.83/0.95, 0.203 | 0.36/0.70/0.84, 0.074 | 0.18/0.60/0.96, 0.171, 0.38 | 46.0, 0.021, 0.15 |
| s7000 | 0 | 0.55/0.78/0.93, 0.110 | 0.37/0.61/0.79, 0.064 | 0.16/0.53/0.94, 0.075, 0.27 | 45.7, 0.019, 0.15 |
| s7000 | 1 | 0.47/0.77/0.93, 0.181 | 0.31/0.58/0.78, 0.065 | 0.18/0.58/0.96, 0.086, 0.28 | 45.3, 0.016, 0.14 |
| s7000 | 2 | 0.61/0.80/0.94, 0.128 | 0.43/0.63/0.81, 0.065 | 0.17/0.56/0.95, 0.115, 0.32 | 45.0, 0.017, 0.14 |
| s7000 | 3 | 0.59/0.82/0.95, 0.215 | 0.47/0.67/0.82, 0.070 | 0.19/0.60/0.96, 0.155, 0.36 | 44.3, 0.020, 0.15 |

### T5. Key geometry by batch usage group (mean over heads and halves; the pass-p3 queries). dead = 0 selections in the batch, low <= median, busy = top 10%

| point | level | n dead/low/mid/busy (of 3072) | cos(key, mean query dir) | max cos to any query | median 8th-best score | drift abs(k_online - k_EMA) | Spearman(usage, max cos) | Spearman(usage, drift) |
|---|---|---|---|---|---|---|---|---|
| init | 0 | 2/1534/1224/312 | -0.31/-0.09/0.06/0.22 | 0.21/0.34/0.41/0.47 | 0.27 | 0.000/0.000/0.000/0.000 | 0.84 | 0.01 |
| init | 1 | 1/1535/1224/312 | -0.46/-0.09/0.07/0.21 | 0.16/0.34/0.41/0.47 | 0.27 | 0.000/0.000/0.000/0.000 | 0.83 | 0.02 |
| init | 2 | 1/1535/1224/312 | -0.41/-0.09/0.06/0.21 | 0.18/0.35/0.41/0.47 | 0.27 | 0.000/0.000/0.000/0.000 | 0.81 | 0.01 |
| init | 3 | 1/1535/1224/312 | -0.32/-0.09/0.06/0.20 | 0.19/0.35/0.42/0.47 | 0.27 | 0.000/0.000/0.000/0.000 | 0.81 | -0.01 |
| old s1000* | 0 | 23/1513/1224/312 | 0.30/0.41/0.52/0.50 | 0.50/0.69/0.82/0.84 | 0.70 | 0.130/0.147/0.103/0.088 | 0.63 | -0.32 |
| old s1000* | 1 | 17/1519/1224/312 | 0.34/0.53/0.58/0.52 | 0.55/0.75/0.83/0.84 | 0.70 | 0.104/0.137/0.113/0.102 | 0.58 | -0.16 |
| old s1000* | 2 | 10/1527/1223/312 | 0.17/0.51/0.57/0.51 | 0.48/0.73/0.82/0.83 | 0.67 | 0.165/0.138/0.106/0.105 | 0.59 | -0.20 |
| old s1000* | 3 | 25/1511/1224/312 | 0.23/0.57/0.58/0.48 | 0.48/0.77/0.83/0.85 | 0.69 | 0.135/0.148/0.114/0.090 | 0.56 | -0.19 |
| s6000 | 0 | 446/1091/1223/312 | -0.06/-0.03/0.01/0.10 | 0.30/0.38/0.54/0.76 | 0.39 | 0.061/0.130/0.174/0.173 | 0.93 | 0.65 |
| s6000 | 1 | 571/966/1223/312 | -0.09/-0.07/-0.02/0.12 | 0.30/0.37/0.53/0.75 | 0.39 | 0.052/0.116/0.168/0.174 | 0.93 | 0.73 |
| s6000 | 2 | 551/985/1224/312 | -0.10/-0.07/-0.02/0.14 | 0.30/0.37/0.53/0.74 | 0.38 | 0.053/0.125/0.172/0.179 | 0.94 | 0.71 |
| s6000 | 3 | 571/966/1223/312 | -0.09/-0.07/-0.01/0.16 | 0.30/0.37/0.52/0.77 | 0.40 | 0.049/0.111/0.165/0.173 | 0.93 | 0.74 |
| s7000 | 0 | 548/990/1222/312 | -0.05/-0.03/-0.00/0.14 | 0.30/0.38/0.55/0.76 | 0.40 | 0.062/0.136/0.169/0.173 | 0.94 | 0.63 |
| s7000 | 1 | 652/884/1224/312 | -0.07/-0.04/-0.01/0.15 | 0.30/0.37/0.54/0.77 | 0.40 | 0.056/0.117/0.163/0.168 | 0.93 | 0.71 |
| s7000 | 2 | 540/997/1223/312 | -0.08/-0.05/-0.01/0.17 | 0.30/0.38/0.55/0.75 | 0.39 | 0.053/0.123/0.167/0.174 | 0.94 | 0.69 |
| s7000 | 3 | 645/893/1222/312 | -0.08/-0.06/-0.01/0.20 | 0.30/0.37/0.54/0.77 | 0.40 | 0.050/0.110/0.161/0.167 | 0.94 | 0.73 |

### T6. Content dependence (mean over passes p1-p5 and heads): row-set overlap (shared rows / 6) by pair type; MI (nats, shuffle-corrected) of the top-1 sub-key (half 1) with the document / token (top-50 tokens) / position bucket (32). H(doc) = 3.57, H(token | top-50) = 3.35

| point | level | lag1 | lag8 | lag64 | lag256 | sameDoc>64 | diffSeq | sameTok_diffSeq | samePos<4_diffSeq | samePos>=256_diffSeq | MI doc | MI token | MI pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | 0 | 0.002 | 0.003 | 0.003 | 0.002 | 0.003 | 0.002 | 0.182 | 0.002 | 0.002 | 0.08 | 2.01 | 0.00 |
| init | 1 | 0.002 | 0.003 | 0.003 | 0.002 | 0.002 | 0.001 | 0.172 | 0.002 | 0.001 | 0.09 | 2.04 | 0.00 |
| init | 2 | 0.001 | 0.003 | 0.003 | 0.001 | 0.002 | 0.001 | 0.166 | 0.001 | 0.001 | 0.08 | 2.04 | 0.00 |
| init | 3 | 0.001 | 0.003 | 0.003 | 0.002 | 0.002 | 0.001 | 0.171 | 0.002 | 0.001 | 0.08 | 2.03 | -0.00 |
| old s1000* | 0 | 0.211 | 0.152 | 0.097 | 0.055 | 0.063 | 0.004 | 0.011 | 0.007 | 0.003 | 1.84 | 0.28 | 0.23 |
| old s1000* | 1 | 0.214 | 0.161 | 0.103 | 0.055 | 0.065 | 0.004 | 0.011 | 0.005 | 0.003 | 1.84 | 0.25 | 0.24 |
| old s1000* | 2 | 0.204 | 0.155 | 0.101 | 0.057 | 0.065 | 0.004 | 0.010 | 0.019 | 0.004 | 1.79 | 0.26 | 0.22 |
| old s1000* | 3 | 0.212 | 0.166 | 0.107 | 0.061 | 0.070 | 0.004 | 0.011 | 0.021 | 0.004 | 1.84 | 0.26 | 0.22 |
| s6000 | 0 | 0.311 | 0.235 | 0.161 | 0.097 | 0.119 | 0.004 | 0.016 | 0.138 | 0.005 | 1.94 | 0.34 | 0.19 |
| s6000 | 1 | 0.284 | 0.228 | 0.159 | 0.105 | 0.121 | 0.005 | 0.014 | 0.050 | 0.005 | 1.81 | 0.30 | 0.15 |
| s6000 | 2 | 0.253 | 0.202 | 0.142 | 0.089 | 0.107 | 0.006 | 0.016 | 0.121 | 0.006 | 1.77 | 0.28 | 0.14 |
| s6000 | 3 | 0.232 | 0.186 | 0.139 | 0.089 | 0.108 | 0.007 | 0.020 | 0.062 | 0.008 | 1.66 | 0.29 | 0.11 |
| s7000 | 0 | 0.323 | 0.244 | 0.164 | 0.098 | 0.122 | 0.003 | 0.015 | 0.139 | 0.005 | 1.98 | 0.31 | 0.20 |
| s7000 | 1 | 0.303 | 0.242 | 0.167 | 0.107 | 0.127 | 0.004 | 0.012 | 0.076 | 0.004 | 1.89 | 0.27 | 0.17 |
| s7000 | 2 | 0.267 | 0.209 | 0.145 | 0.089 | 0.106 | 0.005 | 0.013 | 0.130 | 0.005 | 1.88 | 0.28 | 0.16 |
| s7000 | 3 | 0.252 | 0.205 | 0.148 | 0.096 | 0.115 | 0.006 | 0.016 | 0.064 | 0.007 | 1.76 | 0.26 | 0.12 |
| s6000 thoughts + init addr | 0 | 0.143 | 0.097 | 0.058 | 0.029 | 0.039 | 0.001 | 0.004 | 0.078 | 0.001 | 1.78 | 0.31 | 0.18 |
| s6000 thoughts + init addr | 1 | 0.186 | 0.138 | 0.089 | 0.047 | 0.061 | 0.002 | 0.005 | 0.046 | 0.002 | 1.78 | 0.24 | 0.18 |
| s6000 thoughts + init addr | 2 | 0.152 | 0.106 | 0.066 | 0.035 | 0.043 | 0.001 | 0.004 | 0.047 | 0.001 | 1.78 | 0.30 | 0.18 |
| s6000 thoughts + init addr | 3 | 0.193 | 0.144 | 0.098 | 0.056 | 0.069 | 0.003 | 0.008 | 0.056 | 0.003 | 1.77 | 0.27 | 0.15 |
| s7000 thoughts + init addr | 0 | 0.157 | 0.107 | 0.065 | 0.032 | 0.043 | 0.001 | 0.004 | 0.058 | 0.001 | 1.86 | 0.30 | 0.20 |
| s7000 thoughts + init addr | 1 | 0.217 | 0.165 | 0.106 | 0.057 | 0.075 | 0.002 | 0.005 | 0.037 | 0.002 | 1.85 | 0.22 | 0.19 |
| s7000 thoughts + init addr | 2 | 0.164 | 0.117 | 0.073 | 0.039 | 0.049 | 0.001 | 0.004 | 0.070 | 0.001 | 1.84 | 0.30 | 0.19 |
| s7000 thoughts + init addr | 3 | 0.225 | 0.172 | 0.117 | 0.069 | 0.086 | 0.003 | 0.008 | 0.087 | 0.003 | 1.78 | 0.27 | 0.16 |
| old s1000 w0 (word-keyed read0, mean over levels) | all | 0.157 | 0.161 | 0.158 | 0.156 | 0.155 | 0.151 | 1.000 | 0.159 | 0.153 | 0.08 | 2.14 | 0.00 |

### T7. Product-key halves (H4): over the selected pairs (i, j), H(i), H(j) and the shuffle-corrected MI(i; j), in nats (log 512 = 6.24). exp(MI) = the factor lost in row perplexity relative to independent halves

| point | level | H(i) | H(j) | MI(i;j) | exp(MI) |
|---|---|---|---|---|---|
| init | 0 | 5.30 | 5.08 | 0.54 | 1.7 |
| init | 1 | 5.34 | 5.26 | 0.57 | 1.8 |
| init | 2 | 5.33 | 5.36 | 0.57 | 1.8 |
| init | 3 | 5.40 | 5.42 | 0.60 | 1.8 |
| old s1000* | 0 | 5.17 | 5.21 | 1.97 | 7.2 |
| old s1000* | 1 | 5.18 | 5.20 | 1.94 | 7.0 |
| old s1000* | 2 | 5.12 | 5.15 | 1.86 | 6.4 |
| old s1000* | 3 | 5.16 | 5.10 | 1.90 | 6.7 |
| s6000 | 0 | 4.57 | 4.39 | 1.33 | 3.8 |
| s6000 | 1 | 4.37 | 4.32 | 1.16 | 3.2 |
| s6000 | 2 | 4.32 | 4.30 | 1.09 | 3.0 |
| s6000 | 3 | 4.15 | 4.06 | 0.92 | 2.5 |
| s7000 | 0 | 4.58 | 4.46 | 1.35 | 3.9 |
| s7000 | 1 | 4.40 | 4.39 | 1.18 | 3.2 |
| s7000 | 2 | 4.42 | 4.37 | 1.13 | 3.1 |
| s7000 | 3 | 4.18 | 4.16 | 0.96 | 2.6 |
| s6000 thoughts + init addr | 0 | 5.31 | 5.18 | 1.19 | 3.3 |
| s6000 thoughts + init addr | 1 | 4.88 | 5.00 | 1.19 | 3.3 |
| s6000 thoughts + init addr | 2 | 5.20 | 5.11 | 1.20 | 3.3 |
| s6000 thoughts + init addr | 3 | 4.88 | 4.73 | 1.15 | 3.2 |
| s7000 thoughts + init addr | 0 | 5.30 | 5.20 | 1.24 | 3.4 |
| s7000 thoughts + init addr | 1 | 4.85 | 4.90 | 1.22 | 3.4 |
| s7000 thoughts + init addr | 2 | 5.18 | 5.14 | 1.23 | 3.4 |
| s7000 thoughts + init addr | 3 | 4.81 | 4.51 | 1.11 | 3.0 |

### T8. Counterfactual re-addressing of the SAME thoughts (pass p4; p1 is in the JSON): distinct rows per head over 16,384 positions (uniform random would give about 82,000) / sub-key dead in the batch

| point | level | trained | random_keys | reseed_dead | centred_q | centred_q+random_keys | whitened_ahat | whitened_ahat+random_keys | raw_a(no_norm) |
|---|---|---|---|---|---|---|---|---|---|
| init | 0 | 28682 / 0.01 | 31669 / 0.01 | 28682 / 0.01 | 55841 / 0.00 | 55950 / 0.00 | 71399 / 0.00 | 71346 / 0.00 | 28682 / 0.01 |
| init | 1 | 30596 / 0.01 | 31331 / 0.01 | 30596 / 0.01 | 56221 / 0.00 | 56280 / 0.00 | 71293 / 0.00 | 71390 / 0.00 | 30596 / 0.01 |
| init | 2 | 32616 / 0.01 | 32635 / 0.00 | 32616 / 0.01 | 56306 / 0.00 | 56556 / 0.00 | 71386 / 0.00 | 71456 / 0.00 | 32616 / 0.01 |
| init | 3 | 34108 / 0.00 | 33889 / 0.00 | 34108 / 0.00 | 55813 / 0.00 | 55849 / 0.00 | 71187 / 0.00 | 71331 / 0.00 | 34108 / 0.00 |
| old s1000* | 0 | 11005 / 0.10 | 6286 / 0.19 | 10713 / 0.08 | 9792 / 0.05 | 11324 / 0.03 | 34572 / 0.00 | 46512 / 0.00 | 3474 / 0.32 |
| old s1000* | 1 | 11584 / 0.05 | 5872 / 0.24 | 11324 / 0.04 | 9314 / 0.04 | 11527 / 0.04 | 29767 / 0.00 | 44772 / 0.00 | 3101 / 0.36 |
| old s1000* | 2 | 11727 / 0.04 | 6471 / 0.20 | 11378 / 0.03 | 9997 / 0.05 | 12248 / 0.04 | 29611 / 0.00 | 43487 / 0.00 | 3230 / 0.39 |
| old s1000* | 3 | 11365 / 0.04 | 5539 / 0.25 | 11321 / 0.03 | 9472 / 0.03 | 11508 / 0.04 | 25976 / 0.00 | 40672 / 0.00 | 2200 / 0.44 |
| s6000 | 0 | 7413 / 0.21 | 17287 / 0.02 | 10349 / 0.15 | 8265 / 0.19 | 19046 / 0.01 | 16003 / 0.04 | 36382 / 0.00 | 4679 / 0.34 |
| s6000 | 1 | 7110 / 0.24 | 15563 / 0.02 | 10640 / 0.16 | 7833 / 0.22 | 17164 / 0.01 | 14860 / 0.06 | 32942 / 0.00 | 3208 / 0.45 |
| s6000 | 2 | 7977 / 0.23 | 16816 / 0.02 | 10797 / 0.14 | 8850 / 0.19 | 18482 / 0.01 | 16241 / 0.05 | 34584 / 0.00 | 3420 / 0.40 |
| s6000 | 3 | 7276 / 0.24 | 15343 / 0.02 | 12068 / 0.16 | 8210 / 0.21 | 17527 / 0.01 | 14100 / 0.05 | 32055 / 0.00 | 1896 / 0.52 |
| s7000 | 0 | 7628 / 0.24 | 17164 / 0.01 | 10008 / 0.13 | 8612 / 0.21 | 19183 / 0.01 | 16971 / 0.05 | 37427 / 0.00 | 5301 / 0.33 |
| s7000 | 1 | 7316 / 0.28 | 15672 / 0.02 | 10416 / 0.14 | 8022 / 0.24 | 17174 / 0.01 | 15969 / 0.06 | 34711 / 0.00 | 3610 / 0.45 |
| s7000 | 2 | 8398 / 0.23 | 17542 / 0.01 | 10682 / 0.12 | 9214 / 0.20 | 18913 / 0.01 | 17480 / 0.05 | 36243 / 0.00 | 4158 / 0.37 |
| s7000 | 3 | 7404 / 0.26 | 15531 / 0.02 | 11633 / 0.14 | 8524 / 0.22 | 17487 / 0.01 | 14982 / 0.06 | 33665 / 0.00 | 1956 / 0.54 |

### T9. Who uses the top-50 rows by write count (s6000; all heads and passes of the batch)

| level | wcount: top-50 mean / median / q90 | share of all row draws (uniform: 0.0002) | positions touching one | token entropy (bits) / distinct tokens | median freq rank (batch: 310) | share pos<16 (base 0.031) | top tokens (share / batch share) | pass p1..p5 | batch top-50 ∩ wcount top-50 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 5795 / 103 / 2112 | 0.023 | 0.11 | 8.6 / 1068 | 299 | 0.072 | NL 0.080/0.031, , 0.042/0.034, . 0.037/0.028, _of 0.023/0.021, _the 0.021/0.038 | 0.19/0.21/0.21/0.20/0.19 | 5 |
| 1 | 5789 / 54 / 1897 | 0.026 | 0.15 | 8.7 / 1434 | 137 | 0.039 | NL 0.056/0.031, , 0.054/0.034, _the 0.040/0.038, . 0.038/0.028, _in 0.022/0.017 | 0.16/0.20/0.22/0.22/0.21 | 4 |
| 2 | 5787 / 94 / 2113 | 0.016 | 0.15 | 8.9 / 1575 | 203 | 0.040 | NL 0.059/0.031, , 0.037/0.034, . 0.029/0.028, _the 0.025/0.038, _of 0.025/0.021 | 0.14/0.20/0.24/0.22/0.19 | 4 |
| 3 | 5780 / 53 / 1927 | 0.010 | 0.10 | 8.5 / 1236 | 218 | 0.051 | NL 0.121/0.031, , 0.038/0.034, _of 0.021/0.021, . 0.020/0.028, _as 0.016/0.008 | 0.10/0.18/0.25/0.25/0.22 | 3 |

### T10. Gate vs row popularity: Spearman of the read gate with the mean log(1 + wcount) of its 6 rows (mean gate in brackets)

| point | level | p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|---|---|
| s6000 | 0 | +0.13 (0.80) | +0.11 (0.76) | +0.08 (0.70) | +0.05 (0.64) | +0.02 (0.60) |
| s6000 | 1 | +0.13 (0.73) | +0.11 (0.67) | +0.07 (0.59) | +0.02 (0.54) | -0.02 (0.49) |
| s6000 | 2 | +0.18 (0.74) | +0.15 (0.68) | +0.10 (0.61) | +0.03 (0.55) | -0.03 (0.50) |
| s6000 | 3 | +0.33 (0.69) | +0.33 (0.63) | +0.17 (0.55) | +0.00 (0.48) | -0.10 (0.44) |
| s7000 | 0 | +0.06 (0.80) | +0.04 (0.75) | -0.00 (0.69) | -0.03 (0.64) | -0.03 (0.61) |
| s7000 | 1 | +0.04 (0.69) | +0.04 (0.63) | -0.01 (0.55) | -0.03 (0.50) | -0.04 (0.46) |
| s7000 | 2 | +0.04 (0.70) | +0.07 (0.63) | +0.05 (0.56) | +0.02 (0.50) | -0.04 (0.46) |
| s7000 | 3 | +0.15 (0.71) | +0.22 (0.66) | +0.16 (0.59) | +0.04 (0.54) | -0.06 (0.50) |

### T11. g1: 20 diverse contexts (sequences 0-19, position 400, pass p3). Per head: distinct rows of the 120 / rows shared by 2 or more contexts / max multiplicity / contexts touching a shared row

| point | level | head 0; head 1; head 2 | pooled 3 heads: distinct of 360 |
|---|---|---|---|
| init | 0 | 119/1/2/2; 120/0/1/0; 119/1/2/2 | 358 |
| init | 1 | 120/0/1/0; 118/2/2/2; 119/1/2/2 | 357 |
| init | 2 | 120/0/1/0; 118/2/2/3; 120/0/1/0 | 358 |
| init | 3 | 120/0/1/0; 119/1/2/2; 119/1/2/2 | 358 |
| old s1000* | 0 | 113/5/3/5; 113/5/3/6; 113/6/3/6 | 339 |
| old s1000* | 1 | 110/8/3/5; 113/5/4/6; 115/4/3/3 | 338 |
| old s1000* | 2 | 116/4/2/2; 108/7/4/6; 113/4/4/4 | 336 |
| old s1000* | 3 | 114/5/3/5; 111/5/4/6; 108/7/4/8 | 333 |
| s6000 | 0 | 114/5/3/8; 117/3/2/2; 112/8/2/6 | 343 |
| s6000 | 1 | 118/2/2/3; 117/3/2/4; 115/5/2/8 | 350 |
| s6000 | 2 | 117/3/2/6; 118/2/2/4; 114/6/2/8 | 349 |
| s6000 | 3 | 112/7/3/10; 111/9/2/11; 119/1/2/2 | 342 |
| s7000 | 0 | 114/6/2/5; 118/2/2/2; 114/6/2/6 | 346 |
| s7000 | 1 | 117/3/2/4; 118/2/2/4; 118/2/2/4 | 352 |
| s7000 | 2 | 119/1/2/2; 118/2/2/4; 115/5/2/6 | 352 |
| s7000 | 3 | 117/3/2/3; 117/3/2/6; 120/0/1/0 | 354 |
