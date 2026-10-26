# 18. Address collapse: alternatives beyond whitening + the usage-balance loss (research + offline tests) — 2026-10-24

Scope: how to get addresses that are both spread and stable for the shared product-key memory (plan 12 §2, findings in
plan 16). Whitening and the usage-balance loss (plan 15 (a)) are covered by the plan-17 simulation (`addr_inv/sim.py`,
`sim_p3.json`). I quote its numbers only as reference points. Scripts and raw results: `speed_wip/addr_alt/`
(`alt.py`, `alt2.py`, `alt3.py`, `alt4.py`, `alt_common.py`, `*.json`, `*.log`). CPU, 4 threads, about 10 min in total.

## TL;DR
1. **Spread and stability lie on one trade-off curve, and every scheme tested lies on or below it.** Every way to spread
   the addresses also makes them more sensitive to small or slow changes of the thought. The current "collapsed" trained
   addressing is the most stable point measured. The question is which scheme gives the most spread at a given
   stability, not which gives the most spread.
2. **Best scheme measured: a data-fit, then frozen, product quantiser.**
   - Steps: basis-free truncated whitening of â (top r ≈ 32 of 448 directions), a fixed random projection / rotation
     into the two 64-d halves, and spherical k-means sub-key codebooks (S = 512 per half) fit once on a thought sample.
     Then everything is frozen: no gradient, no EMA.
   - On all three measures (spread, drift stability, noise stability) it beats the usage-loss simulation (plan 17
     `fP_opt`). At equal spread it beats whitening by a large margin on stability.
   - Dead sub-keys are about 0% by construction.
   - It is stable by construction from the moment it is fit (the mapping a → rows never changes again).
3. **Learning-free hashes (SimHash / ITQ / FlyHash / a random-projection PK) cannot collapse and are stable by
   construction, but give less stability per unit of spread than k-means.** They cut space uniformly instead of
   following the data.
   - **FSQ is a bad fit:** it reaches the ideal spread (5,800 / 5,932) only because it acts as a token-level random
     hash. Neighbouring tokens share 0.2% of rows, and 10% noise keeps 18%.
4. **Full whitening (ZCA) is unusable for stability**:
   - noise overlap at 3% noise drops to 0.03-0.6;
   - 6K→7K overlap drops to about 0.
   - Truncating the whitening to the top 32-64 directions keeps most of the spread gain and most of the stability.
5. **The thought drift itself dominates long-run stability, whatever the addressing.** With any addressing frozen at
   6K, the 7K thoughts keep only 0.00-0.32 of their 6K rows. The trained addressing's own learning costs only 0.05 more
   (0.324 frozen → 0.275 real). Re-fitting any statistic (whitening, codebook) over 1,000 steps costs a further
   40-60% of what remains. "Stable by construction" can therefore only remove the addressing-side part; it must be paired
   with an early, one-time fit (or a very slow one).

## Part 1: literature

The columns asked for are: mechanism · why it avoids collapse · effect on address STABILITY · cost · fit with our
constraints (A = addresses only from thoughts, S = stability, T = no loss on thoughts, R = read every pass,
G = one 24 GB GPU).

### Memory layers
- **Product-key memory** ([Lample et al. 2019](https://arxiv.org/abs/1907.05242)).
  - Mechanism:
    - query net + **BatchNorm on the query**;
    - two sub-key sets, random init (random keys "occupy the space relatively uniformly");
    - multi-head with shared values.
  - Why it helps: query BN raised usage at 1M slots from 25.8% to 80.3% (KL to uniform 2.06 → 0.95), and at 262K
    slots from 64% to 98%. BN is per-coordinate standardisation over the batch, like our EMA norm but placed AFTER the
    query projection.
  - Stability: batch statistics move every step. Fine for a feed-forward layer; for us it is the same problem as an EMA
    whitening (see test 3).
  - Cost: nil.
  - Fit:
    - our EMA norm already does the per-coordinate part;
    - plan 16 shows the problem is the cross-coordinate anisotropy, which per-coordinate BN cannot remove (centring q
      gave ×1.1-1.2 only).
- **Memory Layers at Scale** ([Berges et al. 2024, ICML 2025](https://arxiv.org/abs/2412.09764)).
  - Mechanism:
    - product keys, with **qk-normalisation** "when needed" for stability;
    - Memory+ adds an input-dependent SiLU gate on the output;
    - one memory pool shared across 3 layers;
    - values sharded across GPUs.
  - Collapse: they report no usage metrics. Random key-value pairs to "unbias key selection" and a softmax sink gave
    minor, inconsistent gains and were dropped.
  - Stability: not addressed.
  - Fit: the gate and sharing are already in our design. Nothing here targets collapse.
  - Related: [Lin et al. 2025, sparse memory finetuning](https://arxiv.org/abs/2510.15103) ranks slots by TF-IDF of
    access counts, which presupposes stable slot-to-context maps.
- **PEER** ([He 2024](https://arxiv.org/abs/2407.04153)).
  - Mechanism: product keys over 1M single-neuron experts; h independent query nets; **query BatchNorm on by default**.
  - Result: usage was already ≈ 100% even at 1M. BN lowers the unevenness (KL to uniform 1.52 → 1.06 at 1M) and
    perplexity slightly.
  - Stability / cost: same as PKM.
  - Fit: confirms that query-side normalisation is the standard lever, and that it balances usage rather than creating
    coverage.
- **UltraMem** ([Huang et al., ICLR 2025](https://arxiv.org/abs/2411.12364)).
  - Mechanism:
    - Tucker-decomposed query-key retrieval: row and column scores are combined through an r×r core (r = 2),
      approximated by rank-1 via SVD for the top-m;
    - an auxiliary loss keeps the core from degenerating;
    - **LayerNorm on queries and keys**, scaled so that candidate scores are about 1;
    - implicit value expansion (virtual rows via projectors, shuffled to remove the index topology).
  - Collapse: they name the product-key "bias on retrieval topology" (our H4: correlated halves) and fix it by giving
    the pair score more than additive structure. There are no usage metrics.
  - Stability: not addressed.
  - Cost: small. Fit: the topology point supports making the halves independent (test 2: `opq_km`).
- **Engram** ([DeepSeek 2026](https://arxiv.org/abs/2601.07372)).
  - Mechanism: hashed N-gram embeddings with multi-head hashing, a deterministic O(1) lookup, and a context gate.
  - Why no collapse: there are no keys, and the hash spreads by construction.
  - Stability: perfect by construction.
  - Fit: violates A (token-keyed). It is the extreme "stable, no collapse" point, and shows a fixed hash plus a learned
    gate works at scale.

### VQ codebook-collapse fixes (VQ-VAE family)
The analogy: our sub-keys are a codebook, and the thoughts are the encoder outputs. The key difference is that VQ fixes
may pull the encoder (a commitment loss); we may not (T).

- **k-means init / dead-code restart** ([Łańcucki et al. 2020](https://arxiv.org/abs/2005.08520); Jukebox-style random
  restarts, [Dhariwal et al. 2020](https://arxiv.org/abs/2005.00341)).
  - Mechanism: initialise / re-seed codes from encoder outputs (k-means++ or random batch points).
  - Why it helps: codes start inside the data cloud. Restarts recover stranded codes.
  - Stability: a restart re-maps the contexts near the re-seeded code. Periodic k-means re-fits re-map everything.
  - Cost: offline k-means on about 10⁴-10⁵ points is negligible.
  - Fit: good if done once. Our balancing re-seed is the restart. Plan 17's `kpp_reseed` is the dead-only variant.
    Test 1 below (`km_*`) is the full k-means codebook.
- **EMA codebook** ([van den Oord 2017](https://arxiv.org/abs/1711.00937) / [Razavi 2019](https://arxiv.org/abs/1906.00446)).
  - Mechanism: codes = EMA of the assigned encoder outputs (online k-means).
  - Why it helps: it follows the data.
  - Weaknesses:
    - it still leaves dead codes (codes that get no assignment get no update), the same as our selected-only gradient;
    - stability: codes keep moving (test 4: 1,000 steps of tracking keeps only 0.11-0.13 of the rows that freezing
      keeps at 0.20-0.22).
  - Fit: OK as a pre-fit tracker. It needs a freeze.
- **l2-normalised, low-dimensional codes** ([ViT-VQGAN, Yu et al. 2021](https://arxiv.org/abs/2110.04627)).
  - Mechanism: look up codes in a low-dim (e.g. 8-32) projected space, cosine distance.
  - Why it helps: low dimension leaves no empty directions, and the l2 norm removes norm effects.
  - Stability: neutral.
  - Fit: we already use cosine in 64-d per half. The finding that our q lives in about 12-20 of 64 directions says the
    same thing from the other side: match the code dimension to the data rank (test: truncated whitening to r = 32).
- **Affine reparameterisation** ([Huh et al. 2023](https://arxiv.org/abs/2305.08842)).
  - Mechanism: codes = A·c + b with shared A, b (or moment-matched to the encoder), so every code moves with the batch
    statistics.
  - Why it helps: shared parameters get gradient from every assignment, so dead codes drift back with the data.
  - Stability: the whole codebook moves together, which preserves relative order better than independent codes.
  - Fit: compatible with T. A cheap "shared-transform keys" option for learned keys. Not tested here (it is a learning
    method). See also SimVQ.
- **SimVQ** ([Zhu et al. 2024, ICCV 2025](https://arxiv.org/abs/2411.02038)).
  - Mechanism: codes = W·C with C a frozen random basis and only the linear map W learned.
  - Why it helps: the whole linear span is updated, so no code is starved.
  - Stability: W changes rotate all codes coherently.
  - Fit: good. "Frozen random keys + a learned shared map" is the learned cousin of the random-projection PK in test 1
    (rp_pk). It is the natural pick if learned keys are kept.
- **VQGAN-LC** ([Zhu et al., NeurIPS 2024](https://arxiv.org/abs/2406.11837)).
  - Mechanism: a 100K-entry codebook initialised from features of a pretrained encoder and FROZEN; only a projector is
    trained.
  - Result: utilisation above 99%.
  - Stability: frozen codes.
  - Fit: very close to the recommended "fit once, then freeze" k-means codebook.
- **Rotation trick** ([Fifty et al., ICLR 2025](https://arxiv.org/abs/2410.06424)).
  - Mechanism: a better gradient through the quantiser: rotate and rescale instead of straight-through.
  - Result: VQGAN codebook usage went from 2% to 27%.
  - Fit: changes the gradient into the encoder (the thought), so it is outside T.
- **Entropy / commitment regularisers** (e.g. the LFQ entropy term in MAGVIT-v2, below).
  - Mechanism: per-sample entropy low + batch-code entropy high.
  - Fit: when the gradient goes to the encoder, it violates T. Restricted to the keys and W_q it is exactly plan 15 (a).
- **CVQ-VAE** ([Zheng & Vedaldi, ICCV 2023](https://arxiv.org/abs/2307.15139)).
  - Mechanism: dead codes are pulled toward online-sampled encoder "anchors", with a strength set by low usage.
  - Result: 100% usage.
  - Stability: it only moves unused codes, so it barely re-maps live contexts.
  - Fit: good. It is the soft, continuous version of our dead re-seed and needs no thought gradient. Worth having if
    keys stay learned. It is the same idea as plan 16's "pull dead keys toward the query distribution".

### Learning-free quantisers (no learnable keys, so they cannot collapse in the "dead code" sense)
- **FSQ** ([Mentzer et al. 2023](https://arxiv.org/abs/2309.15505)).
  - Mechanism: project to d dims, bound, and round each dim to L levels; the implicit codebook is the grid.
  - Why no dead codes: the grid is fixed, and usage is near-uniform if the projected marginals fill the box.
  - Stability: by construction, but the cells are axis-aligned slabs, and in our test fine slabs cut through the noise
    directions (below).
  - Cost: trivial. Fit: A, T, G fine; S poor in our test.
- **LFQ** ([MAGVIT-v2, Yu et al. 2023](https://arxiv.org/abs/2310.05737)) and **BSQ**
  ([Zhao et al. 2024](https://arxiv.org/abs/2406.07548)).
  - Mechanism: sign quantisation of a d-dim projection (LFQ) or of its l2-normalised version (BSQ): a 2^d implicit
    codebook. They use an entropy loss on the encoder to balance bits.
  - Fit: without the entropy loss (T), LFQ is SimHash with a learned projection.
- **Random-projection LSH / SimHash** ([Charikar 2002](https://dl.acm.org/doi/10.1145/509907.509965); multi-probe
  [Lv et al. 2007](https://dl.acm.org/doi/10.5555/1325851.1325958)), and **cross-polytope LSH as in Reformer**
  ([Kitaev et al. 2020](https://arxiv.org/abs/2001.04451)).
  - Mechanism: bits = sign(random hyperplanes); multi-probe = the nearest buckets by margin. With 9 bits per half and
    multi-probe scores z·v over the 512 hypercube vertices, this is exactly a product key with fixed ±1 keys, so it
    drops into pk_topk unchanged.
  - Stability: perfect for the map. Cost: lower than now (an N×9 projection).
  - Fit: A, T, G fine. Bits need balanced thresholds (a fit median) and independent directions.
- **ITQ** ([Gong & Lazebnik 2011](https://doi.org/10.1109/CVPR.2011.5995432)) and spectral hashing
  ([Weiss et al. 2008](https://papers.nips.cc/paper/3383-spectral-hashing)).
  - Mechanism: PCA then a rotation that minimises the sign-quantisation error, giving balanced, decorrelated bits.
    It is fit once.
  - Fit: as SimHash, with a one-time fit.
- **FlyHash** ([Dasgupta, Stevens & Navlakha, Science 2017](https://www.science.org/doi/10.1126/science.aam9868)).
  - Mechanism: a sparse binary random expansion (PN → KC), then winner-take-all keeping the top 5%: a fixed,
    biologically motivated LSH.
  - Fit: thematically apt for this project. In our setting, the 512 Kenyon cells per half with top-m are a fixed-key PK
    (test 2).
- **Product quantisation / OPQ** ([Jégou et al. 2011](https://doi.org/10.1109/TPAMI.2010.57);
  [Ge et al. 2013](https://doi.org/10.1109/CVPR.2013.379)).
  - Mechanism: k-means per sub-space. OPQ first rotates so that the sub-spaces are balanced in variance and close to
    independent.
  - Fit: this is the theory behind our product key. OPQ directly targets H4 (correlated halves).
- **Hash layers** ([Roller et al. 2021](https://arxiv.org/abs/2106.04426)).
  - Mechanism: fixed (random or balanced) token-to-expert hashing beats learned routing in their tests.
  - Fit: token-keyed (A violated). Evidence that fixed assignment loses little.

### Retrieval memories (Hopfield / kNN-LM style)
- **kNN-LM** ([Khandelwal et al. 2020](https://arxiv.org/abs/1911.00172)) and **Memorizing Transformers**
  ([Wu et al. 2022](https://arxiv.org/abs/2203.08913)).
  - Mechanism: store (context vector, value) pairs; exact or approximate kNN over the stored keys.
  - Why no collapse: the keys ARE past contexts, so the key distribution equals the query distribution.
  - Stability: Memorizing Transformers note staleness: keys written by an older network drift from new queries.
    This is our 6K→7K problem, seen from the read side.
  - Cost:
    - per-token storage grows with the stream (64-d keys × 10⁶ tokens is about 256 MB fp32, which is OK);
    - kNN needs a search index (FAISS on GPU, or a product quantiser).
  - Fit: violates "a big learned table". It is the conceptual limit of "keys follow the data", and our k-means
    codebook is its compressed form.
- **Modern Hopfield networks** ([Ramsauer et al. 2020](https://arxiv.org/abs/2008.02217)).
  - Mechanism: softmax attention over stored patterns. Retrieval is stable only when patterns are well separated
    (β large).
  - Fit: the same message: separation (spread) and the basin size (stability) trade off.

### What the literature says, applied to our constraints
- Coverage fixes that work without touching the encoder (the thought) fall into four kinds:
  - put the keys where the data is (k-means init, VQGAN-LC, CVQ anchors, kNN memories);
  - share parameters across keys so no key is starved (affine reparameterisation, SimVQ);
  - standardise the query (BN / LN / qk-norm);
  - remove the keys altogether (FSQ / LFQ / LSH / hashing).
- None of these papers measures stability of the context-to-slot map over training. Our extra requirement is
  therefore the deciding one, so the tests below centre on it.

## Part 2: offline tests

### Protocol (identical to plan 17's sim.py, so the numbers can be compared directly)
- Data: captures `cap_6000.pt` and `cap_b.pt` (7K), at pass p3 (k = 2); p1 was re-run as a check and agrees.
- Fit and evaluation: everything fitted (whitening, codebooks, thresholds, quantile tables) is fit on the 6K thoughts
  of sequences 0-15 and then FROZEN. Evaluation is on sequences 16-31 (8,192 positions).
- Metrics, as means over the 4 levels × 3 heads:
  - `dist`: distinct rows per 1,000 positions per head (ideal 5,932); `top1`: the top-1% row share; `dead`: sub-keys
    never selected;
  - `mi`: the shuffle-corrected MI(i; j) of the halves;
  - `xdoc` / `lag1` / `lag256`: row-set overlap (/6) between positions of different documents, or of the same document
    at lag 1 / 256;
  - `n03` / `n10`: overlap under 3% / 10% relative Gaussian noise on a;
  - `drift`: the 6K-fitted addressing applied to the 7K thoughts of the same tokens.
- Inputs:
  - `ema`: the current â;
  - `pcaR`: â PCA-whitened and truncated to the top R directions;
  - `pzcaR`: the same, in the basis-free form V_R Λ_R^-½ V_Rᵀ;
  - `zca`: full whitening.
- Addressing schemes (all are a top-8-per-half / top-6-pair product key through pk_topk):
  - `trained`: the run's W_q + sub-keys;
  - `rp_pk`: a fixed Gaussian projection to 128, plus fixed random unit sub-keys;
  - `simhash`: 9 hyperplanes per half, median thresholds, multi-probe over the 512 vertices;
  - `fsq`: 3 directions per half, fit-quantile mapped, 8 equal-mass levels each (8³ = 512);
  - `km_wq` / `km_rp`: spherical k-means codebooks (S = 512) on the trained-W_q queries / on the rp_pk queries;
  - `kmbal_rp`: the same with Sinkhorn-balanced assignment during the fit;
  - `flyhash`: 512 KC per half, each summing 6 inputs, WTA;
  - `itq`: ITQ-rotated 9-bit codes;
  - `opq_km`: a random rotation, then the halves come from disjoint coordinates, then k-means per half;
  - `rp_pk_sep` / `simhash_b`: the halves come from disjoint directions.

### Test 1-2: spread vs stability at pass p3
Sorted by spread. Rows marked † are plan 17's sim_p3 numbers (same protocol), for reference.

| input | address | dist /1k (5932) | top1% | dead | MI(i;j) | xdoc | lag1 | lag256 | n03 | n10 | drift 6K→7K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ema | trained (now) | 1867 | 0.946 | 0.312 | 1.28 | 0.004 | 0.275 | 0.106 | **0.944** | **0.834** | **0.324** |
| pca16 | rp_pk | 2632 | 0.851 | 0.112 | 1.38 | 0.003 | 0.129 | 0.036 | 0.932 | 0.797 | 0.226 |
| ema | random keys † | 2927 | 0.791 | 0.053 | - | 0.001 | 0.155 | 0.044 | 0.926 | 0.781 | 0.202 |
| pca16 | km_rp | 3115 | 0.768 | 0.003 | 1.84 | 0.001 | 0.094 | 0.025 | 0.921 | 0.767 | 0.196 |
| ema | km_wq | 3179 | 0.745 | 0.002 | 1.56 | 0.001 | 0.144 | 0.037 | 0.928 | 0.786 | 0.196 |
| pca16 | flyhash | 3325 | 0.732 | 0.004 | 1.71 | 0.001 | 0.099 | 0.021 | 0.921 | 0.770 | 0.168 |
| pca16 | itq | 3403 | 0.713 | 0.183 | 0.53 | 0.001 | 0.088 | 0.020 | 0.913 | 0.743 | 0.151 |
| ema | usage loss (fP_opt) † | 3427 | 0.704 | 0.002 | - | 0.001 | 0.112 | 0.030 | 0.911 | 0.735 | 0.151 |
| **pca32** | **km_rp** | **3524** | 0.681 | 0.002 | 1.41 | 0.001 | 0.089 | 0.025 | **0.912** | 0.742 | **0.223** |
| pca16 | simhash | 3720 | 0.656 | 0.033 | 1.11 | 0.001 | 0.070 | 0.015 | 0.910 | 0.733 | 0.133 |
| zca | random keys † | 3741 | 0.625 | 0.006 | - | 0.001 | 0.106 | 0.044 | 0.489 | 0.101 | 0.011 |
| pca32 | rp_pk | 3795 | 0.634 | 0.015 | 0.99 | 0.001 | 0.071 | 0.017 | 0.902 | 0.717 | 0.176 |
| ema | km_rp | 3844 | 0.619 | 0.000 | 1.27 | 0.001 | 0.092 | 0.023 | 0.895 | 0.690 | 0.152 |
| pca32 | flyhash | 3916 | 0.610 | 0.002 | 1.18 | 0.001 | 0.067 | 0.016 | 0.900 | 0.709 | 0.168 |
| **pca32** | **opq_km** | **4383** | 0.514 | 0.000 | 0.87 | 0.000 | 0.058 | 0.012 | **0.902** | 0.711 | **0.162** |
| ema | whitened q + trained keys † | 4396 | 0.485 | 0.000 | - | 0.001 | 0.085 | 0.040 | 0.833 | 0.527 | 0.056 |
| pca32 | itq | 4479 | 0.494 | 0.023 | 0.42 | 0.000 | 0.040 | 0.009 | 0.876 | 0.644 | 0.107 |
| pca64 | km_rp | 4489 | 0.488 | 0.000 | 0.89 | 0.000 | 0.046 | 0.014 | 0.866 | 0.614 | 0.135 |
| ema | rp_pk | 4575 | 0.475 | 0.001 | 0.74 | 0.000 | 0.044 | 0.010 | 0.856 | 0.584 | 0.069 |
| pca32 | simhash | 4619 | 0.465 | 0.002 | 0.70 | 0.000 | 0.035 | 0.008 | 0.874 | 0.635 | 0.099 |
| pca64 | rp_pk | 4864 | 0.412 | 0.000 | 0.61 | 0.000 | 0.027 | 0.006 | 0.848 | 0.568 | 0.094 |
| zca | whiten q + random keys † | 4884 | 0.388 | 0.000 | - | 0.000 | 0.046 | 0.016 | 0.202 | 0.013 | 0.000 |
| zca | rp_pk | 5043 | 0.353 | 0.000 | 0.53 | 0.000 | 0.035 | 0.012 | 0.165 | 0.009 | 0.000 |
| ema | simhash | 5169 | 0.341 | 0.000 | 0.48 | 0.000 | 0.019 | 0.004 | 0.807 | 0.466 | 0.029 |
| pca64 | simhash | 5399 | 0.284 | 0.000 | 0.37 | 0.000 | 0.011 | 0.003 | 0.803 | 0.459 | 0.043 |
| pca32 | fsq | 5750 | 0.178 | 0.000 | 0.13 | 0.000 | 0.003 | 0.001 | 0.767 | 0.372 | 0.011 |
| ema | fsq | 5818 | 0.153 | 0.000 | 0.09 | 0.000 | 0.002 | 0.000 | 0.651 | 0.183 | 0.004 |

Other rows (in `alt_p3.log`, `alt2_p3.log`):
- `kmbal_rp` is identical to `km_rp` within ±1% (Sinkhorn balancing adds nothing; k-means is already balanced on this
  data);
- `rp_pk_sep` and `simhash_b` (halves from disjoint directions) cut MI(i; j) to 0.5-0.57 but did not move the
  trade-off;
- the pass-p1 re-run (`alt_p1.log`) gives the same ordering. There the trained addressing is more collapsed
  (1,499, 46% dead), and `km_wq` gives 3,363 with drift 0.198.

Reading the table:
1. **Every method lies on or below one trade-off curve.** More spread means lower `lag1` (fewer rows shared by
   neighbouring tokens of a document), lower `n03` / `n10` and lower `drift`. Cross-document overlap is ≈ 0 for all, so
   document separation is never the issue. The spread a method adds is resolution INSIDE a document's region.
2. **k-means codebooks sit on top of that curve.**
   - pca32 + km_rp vs the usage loss (fP_opt): spread 3,524 vs 3,427, drift 0.223 vs 0.151, n03 0.912 vs 0.911. It is
     at least as good on all three, and its drift stability is half as good again.
   - pca32 + opq_km vs whitened-q (plan 17's best-spread learned point): at the same spread (4,383 vs 4,396), drift
     0.162 vs 0.056 and n10 0.71 vs 0.53.
   - Why: k-means puts the codes where the thoughts are dense. Bucket borders then run through sparse regions, so small
     moves rarely cross them. Uniform hashes put borders everywhere.
3. **Truncating the whitening matters more than the hash family.**
   - ema → pca32 (for rp_pk / simhash) costs some spread and buys about 2-3× the drift stability.
   - Full ZCA wrecks stability for every scheme: the drift ≈ 0, and n03 is 0.03-0.26 for my schemes (plan 17's ZCA
     variants: 0.20-0.60).
   - The minor directions of the thought are noise / fast-changing. Equalising them gives them the same vote as the
     slow content directions.
   - R = 32 was the best value: R = 16 was less stable than R = 32 for k-means (0.196 vs 0.223), probably because the
     top few directions themselves rotate between 6K and 7K.
4. **Uniform learning-free hashes (SimHash, ITQ, FlyHash, rp_pk) lie below k-means** by about 0.03-0.07 of drift
   overlap at equal spread. FlyHash on pca32 behaves like rp_pk (3,916 / 0.168). ITQ balances bits (MI 0.42) but loses
   on stability.
5. **FSQ: not recommended.**
   - lag-1 overlap is 0.002-0.009: neighbouring tokens of the same document get different rows;
   - drift stability is 0.004-0.02; 10% noise keeps 18-56%.
   - Its near-ideal spread is a token-level random hash, not a content address. Few dims × fine equal-mass levels put
     the borders inside the dense core of the distribution.
6. Balancing (Sinkhorn) and independent halves did not move the trade-off curve. Dead sub-keys were already ≈ 0 with
   any data-fit or random codebook.

### Test 3-4: what happens if the fitted pieces keep updating (`alt3.py`, `alt4.py`)
The `drift` column above freezes the 6K-fitted addressing. In a run, the statistics are EMAs until the freeze. Below,
"refit" re-fits on the 7K fit half (whitening re-computed; k-means warm-started from the 6K codebook, 5 Lloyd
iterations). Each value is the overlap with the 6K-fitted addressing on the 6K thoughts.

| input + codebook | dist | frozen | codebook tracked | whitening refit | whitening 50/50 mix | both refit |
|---|---|---|---|---|---|---|
| trained W_q queries + k-means (no whitening) | 3184 | 0.198 | 0.112 | - | - | - |
| pzca32 + k-means | 3503 | 0.222 | 0.131 | 0.121 | 0.162 | 0.079 |
| pzca64 + k-means | 4468 | 0.137 | 0.066 | 0.075 | 0.099 | 0.044 |
| pca32 (eigenbasis) + rp_pk / k-means | 3795 / 3510 | 0.176 / 0.223 | - | 0.001 / 0.000 | - | - |
| pzca32 + rp_pk | 3766 | 0.179 | - | 0.043 | - | - |
| the real trained run (captured 6K rows vs captured 7K rows) | 1866 | - | - | - | - | **0.275** |

1. **The thought drift dominates.**
   - With all addressing frozen, the 7K thoughts keep at most 0.22-0.32 of their rows (at our spreads).
   - The real trained run, with its addressing still learning, keeps 0.275, only 0.05 below its frozen value of 0.324.
   - So the thought moves far more than the keys over 1,000 steps. No addressing scheme can be "stable over training"
     while the thought it reads keeps moving.
2. **Every refit piece costs about 40-50% of the remaining overlap per 1,000 steps**: codebook tracking 0.22 → 0.13,
   whitening refit → 0.12, both → 0.08.
   - The naive eigenbasis form (`pca32`) loses everything on refit (0.00): eigenvector order and sign flip. If whitening
     is used, it must be the basis-free form V_R Λ^-½ V_Rᵀ, or full ZCA.
3. So "stable by construction" means: fit once from a large thought sample, then never touch it. The freeze must
   come for the statistics too. A slow EMA (the 50/50 mix) keeps 0.16 instead of 0.12.

### Caveats
- The proxy says nothing about LM loss or recall. Higher spread only matters if the memory is capacity-limited.
  Plan 16's popular-row blur suggests that it is, but this is not shown.
- Only 16 eval sequences (about 23 documents), so absolute `dist` is batch-dependent: compare rows, not absolutes.
- The noise is isotropic in a-space. Real perturbations (dropout, routing noise, pass-to-pass changes) may load the
  minor directions more or less.
- There is one 1,000-step drift interval, before the freeze and at a high LR. Post-freeze drift (lower LR late in the
  cosine) is probably smaller but unmeasured.
- The k-means / whitening were fit on 8K positions (16 sequences). A real fit should use 10⁵-10⁶ thoughts from many
  documents. With so few documents, the codebook may over-fit them, which would flatter the stability numbers.

## Ranked recommendation

1. **Frozen, data-fit product quantiser ("k-means keys, fit once").** This is the recommended first try.
   - What:
     1. Collect about 200K-1M thoughts â (sg) per level over the last about 500 steps before the fit point.
     2. Compute the basis-free truncated whitening W = V_R Λ_R^-½ V_Rᵀ, R ≈ 32 (a sweep of 24-48 is cheap offline).
     3. Apply a fixed random orthogonal projection to 128 and split it into halves (OPQ-style; take disjoint
        coordinates per half if MI matters).
     4. Fit spherical k-means with S = 512 per half per head (different seeds and projections per head).
     5. Freeze everything.
   - Fit point: at the planned freeze (60%), replacing the EMA keys + W_q for the INDEX path. Optionally, a first fit
     at about 10-20% so that writes accumulate into a sensible map, with a row migration at the final fit: copy each old
     row's V to the new row that most of its sampled contexts now map to, the same mechanics as copy-on-split.
   - Spread / stability: same or more spread than the usage loss, with clearly better drift and noise stability
     (table). Dead sub-keys about 0 by construction.
   - Stable by construction: **yes** after the fit (no gradient, no EMA, no balancing). Only the thought drift remains.
   - Weights ω: keep a learned path for weights only. Either the online W_q / keys as now (the indices from the frozen
     PQ, ω from online scores on those pairs), or simply ω = softmax(cos(q, centroid)/τ) with a learned τ and the gate.
     With sg on a, nothing constrains the thought (T holds).
   - Cost:
     - a fit: k-means on 1M × 64 with S = 512, 25 iterations, is seconds on the GPU (about 1 GFLOP per iteration);
     - per step: the same select GEMM as now, plus a 448×448 matmul;
     - VRAM: under 10 MB.
   - Risks: fit too early and the thoughts move away (the s1000 stranding story again). Fit too late and infancy runs
     on the collapsed map. The two-fit + migration variant addresses this.
2. **Learning-free random-projection PK / SimHash / FlyHash on the truncated whitening (pzca32).**
   - Simplest, and the map has no learned part at all. Only the whitening statistics need a one-time fit.
   - About 0.03-0.07 lower drift overlap than k-means at equal spread. SimHash is the cheapest (an N×9 projection +
     a vertex GEMM). FlyHash is the thematic choice and behaves like rp_pk.
   - Stable by construction: **yes** (once the whitening is frozen).
   - Useful as the default index path from step 0, because it needs no data to start (the whitening can be fixed at
     an early checkpoint). It cannot strand keys.
3. **If keys must stay learned during infancy: SimVQ / affine-shared keys + CVQ-style anchoring of dead keys, plus
   k-means re-initialisation at the freeze.**
   - Restores coverage without a thought loss, and moves keys coherently.
   - Not stable by construction before the freeze, which is no worse than now.
   - Overlaps with plan 15 (a). Only worth it if (1)'s one-time fit proves too coarse.
4. **Not recommended:**
   - FSQ (a token-level hash in practice);
   - full ZCA whitening (no stability);
   - naive PCA-eigenbasis whitening refit (re-maps everything);
   - balanced (Sinkhorn) k-means (no gain);
   - LFQ / BSQ / the rotation trick / commitment-entropy losses that act on the encoder (violate the no-thought-loss
     rule).

**Trade-off, plainly.**
- Spread is bought with sensitivity: every +1,000 distinct rows per 1K positions costs about 0.04-0.09 of 6K→7K overlap
  along the best curves (k-means: R = 32 → 64, +965 rows per 1K, −0.088).
- The k-means PQ at R = 32 is the knee: about ×1.9 the spread of now, keeping about 70% of now's drift stability
  (0.22 vs 0.32) and 97% of its noise stability.
- Whitening + usage loss, as simulated in plan 17, gets more spread only by leaving the curve. Its whitened-q
  variants are near 0.05 drift overlap.
- For long-run stability, the binding factor is the thought drift, not the keys. A frozen PQ removes the key / statistics
  part completely, but the owner's "same context → same rows over long training" will additionally need either a lower
  spread target or (later) the light anchor on the thought.
- A cheap thing to log in the next run: the 1,000-step row overlap of the probe contexts under the frozen addressing.
  It separates thought drift from addressing drift online.
