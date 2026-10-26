# Memory usage plan: make the memory store distinct content (plan only, no code) — 2026-10-23
Why: in wiki_mem_12k (thought-only addressing, speed package) the memory is read (gates 0.4-0.77, up from 0.12) but
barely helps (shared off +0.009, private off 0.000 at 25%). Use is very uneven at step 3,500:
- write counts per row: median 17, 90% quantile 816;
- 17-26% of the rows are never written;
- 36% of the shared sub-keys are dead (rising).
So most contexts share a few popular rows, which become blurry averages.
The freeze at step ≥ 7,325 is the first test. This plan is for the next run if the knockout stays ≈ 0.01 after the freeze.
Order of value: (a) > (d) > (c) > (b).

## (a) Usage-balance loss on the addresses
Per (level l, head h, half c), with the online scores s^c = q^c K^cᵀ ∈ R^{N×S} over ALL S sub-keys:
- Hard usage (no gradient): f_i = share of the top-m selections that went to sub-key i in this forward. It comes
  from select() (the EMA indices), and the pend counts already hold it.
- Soft usage (with gradient): P_i = mean over tokens of softmax(s^c / τ_u)_i, with τ_u = 0.1.
- Switch-style loss: L_u = λ_u · mean over (l, h, c) of S · Σ_i f_i P_i. Its minimum S·Σ (1/S)(1/S)·S = 1 is reached
  at uniform use. It pushes the soft probability down on overused keys and up on underused ones. This is the same
  form as the experts' aux loss, so the behaviour is known.
- Alternative, without hard counts: KL(P̄ ‖ uniform) = Σ_i P̄_i log(S·P̄_i). It is smoother, but it acts on the soft
  distribution, which can differ from the hard top-m. I recommend f·P because it targets what is actually used.
- Scope: computed on sg(a) (like the margin), so the gradient reaches W_q and the sub-keys only. It never moves the
  thought (Eitan: no constraint on thoughts).
- Row level: sub-key balance per half does not guarantee balance over rows (r = i·S + j), because the halves can be
  correlated. The loss is per half (cheap). Log the row-level usage entropy per level against log(S²), on the probes
  and over a training window. If the row entropy lags behind the per-half entropy, add a pair-level term later.
- Private memory: the same per expert over its R rows (f from the kept slots, P from the online scores over all R).
  Off by default, because the private memory currently gives ≈ 0 and its full online scores cost a (E, C, R)
  backward.
- Cost: one extra fp32 GEMM N×S per (l, h, c) with backward. That is the size of select()'s GEMM, about
  +0.05-0.1 s/step if done on every pass and every token. Mitigations:
  - only on the one margin pass (--mem-margin-passes 1 picks it already);
  - on a token subsample (e.g. 1,024 of 8,192, --mem-usage-tokens);
  - with both, ≈ +0.01-0.02 s/step.
- Weight: λ_u = 0.01 (the experts use 0.02; this loss acts on keys only). Log L_u, the min / max of f·S and the dead
  share. If the dead share is still > 10% at 30% of the run, raise λ_u to 0.03.
- Infancy only. After the freeze the keys and W_q are frozen, so the loss has nothing left to act on except the
  thought, and we don't constrain thoughts. It is OFF after the freeze, and the freeze-time spread is final.
  This is why (a) must be on from the start of the run.
- Balancing (re-seed / split) stays as the coarse fallback: with (a) on, every 1,000 steps, 2% split cap,
  --mem-balance-dead-max 0.10.
- Flags: --mem-usage-w 0.01, --mem-usage-tau 0.1, --mem-usage-tokens 1024, --mem-usage-private 0.
- Tests:
  - L_u = 1 at uniform f = P;
  - the gradient reaches W_q / K only (none to the thought, none to the tables);
  - on a toy skewed codebook, a few hundred steps of L_u alone reduce the max f·S;
  - flag off = identical.

## (b) Per-row learning rate by write count: NOT recommended as such
- Row-Adagrad already does it: the step is η·g / sqrt(S₀ + Σ g²). Heavily written rows take small steps, and rarely
  written rows take relatively large ones. An extra 1/(1 + n_r) factor would double-count.
- The real problem for rare rows is that they get few gradients, not that their steps are small. (a) addresses that.
- What IS worth checking is the prior S₀ = 100 × median G². For a row with few writes, the first steps are about
  η·g/√S₀, i.e. about 1/10 of a normalised step, so a rarely written row barely moves in its first ~100 writes.
  Proposal: --mem-s0-mult 10 (instead of 100). The first writes are then about 1/3 of a normalised step and still
  bounded by the update clip c_Δ = 0.05.
- Diagnostic to log: the mean |V_r - V_r(init)| by write-count bucket (1-10, 10-100, 100+). It shows whether rarely
  written rows ever leave their random init.

## (c) Heads and rows at the same VRAM (shared table 1.07 GB + grad 1.07 GB fixed)
| option | rows | per-level region | heads that share it | compute vs now | expected |
|---|---|---|---|---|---|
| now: H=3, S=512, d_V=256 | 1.05M | 262K | 3 (multi-hash) | 1.0 | collisions between heads |
| H=1, S=512, d_V=256 | 1.05M | 262K | 1 | about -0.05-0.12 s/step | no head collisions, less redundancy |
| H=2, S=512, d_V=256 | 1.05M | 262K | 2 | about -0.03-0.06 | middle |
| H=3, own region each, S=296 | 1.05M | 87.6K per head | 1 | about the same | no collisions, smaller key codebooks |
| H=2, S=724, d_V=128 | 2.1M | 524K | 2 | +gathers, -W_o | 2× slots, thinner rows |
| private R=512 → 128 | -0.3 GB | - | - | -private bmm | private ≈ 0 value now; spend the VRAM on shared rows |

- Recommendation for the next run: H=2 (S=512, d_V=256) + (a). It halves the head collisions and keeps some
  redundancy.
- Private R=128 with --mem-usage-private 0, unless the 12K post-freeze knockout shows private value.
- Variant for a later ablation: H=2, S=724, d_V=128 (2M rows), if (d) shows capacity is the limit.

## (d) A generated-world fact-recall test
Wikipedia val loss is a weak signal for "did it remember a fact", so add data where facts can only come from memory.
- World:
  - 5,000 people with invented syllable names (checked against the wiki16k vocabulary and Wikipedia titles, so no
    real names). Each person has a city, job, pet, birth year and a best friend (another person).
  - 300 cities, each with a country (of 40) and a river (of 100). 200 jobs.
- Text, 10 paraphrase templates per relation:
  - fact documents of 5-15 facts about random people;
  - mini-stories that use 2-4 facts;
  - two-hop sentences are NOT written: "X lives in C" and "C is in country K" appear separately.
- Exposure:
  - 1,000 "early" people appear only in the first 20% of training (retention test);
  - the other 4,000 appear throughout;
  - 20% of the tokens are world text, 80% Wikipedia.
  - With 200M tokens, that is ~40M world tokens and ~30K facts × ~12 tokens: about 110 exposures per fact for the
    late group, about 22 for the early group.
- Data: generate_world.py → X:\Datasets\wiki_world_mix (no download; wiki16k tokens; EOT between documents;
  the same meta.json format). The validation file has the questions held out:
  - the question forms never appear in training text, only the statements do;
  - 1,000 questions per type.
- Question types, scored by exact-match greedy decoding of the answer tokens and by the teacher-forced answer
  log-prob:
  1. single-hop: "Q: Where does <name> live? A: <city>" (also job, pet, birth year, friend);
  2. two-hop: "Q: In which country does <name> live?" (city → country);
  3. comparison: "Q: Who is older, <a> or <b>?";
  4. retention: single-hop on the early group only.
  The control on the same mix gives the baseline for each.
- Memory metrics on the questions:
  - accuracy with memory on;
  - accuracy with shared off, private off and each level off (the knockout delta on facts);
  - accuracy vs exposure count;
  - early-group retention at 25 / 50 / 75 / 100%.
  - The memory "works" if the shared-off delta on single-hop facts is clearly larger than on Wikipedia val.
- Runs (both 12K steps, same mix, same flags otherwise):
  1. control v4d on the world mix;
  2. memory with (a) + H=2 (+ --mem-s0-mult 10).
  Plus the eval script eval_world.py, which reads a checkpoint and writes the table above.

## Order of work (after the 12K and 24K results)
1. If the post-freeze knockout in wiki_mem_12k / 24k stays ≈ 0.01: implement (a) + the (b) diagnostics +
   --mem-s0-mult, with tests.
2. In parallel: generate the world mix and write eval_world.py (CPU).
3. Run the control and the memory model on the world mix (12K each), the memory one with (a) + H=2.
4. Decide on (c) variants from (d)'s capacity signals: accuracy vs exposure, and whether rows saturate.
