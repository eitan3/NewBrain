# Long-term memory neurons + hierarchical concepts - plan v2 (no code yet) - 2026-10-23

Goal (Eitan): separate KNOWLEDGE from THINKING. Memories live in large, addressable tables; the looped core asks with
its thoughts on every pass and decides what to do with the answer.
Decisions so far:
- Addresses come ONLY from thoughts / concepts (no token or word keys).
- A translator maps thought → address. Its input may change; its OUTPUT MAPPING must not. The translator and the address
  space are frozen after infancy.
- Thoughts are NOT constrained: no commitment loss and no anchor training on the thoughts.
- Light margins, soft reads, redundancy, repair and full measuring are kept.
- Two kinds of memory: a shared pool, plus a small private memory for each expert.
- Concepts are hierarchical (16 / 32 / 64 / 128 tokens, like NCM), and memory is hierarchical on top of them.

## A. Hierarchical concepts (also the address input for memory)
- Levels L1..L4 predict the latent of the next 16, 32, 64 and 128 tokens.
  - Target for level n: z*_n(t) = the stop-grad mean of normf(x_K) over positions t+1..t+n, from the same forward
    (cumsum, as today's think target), so all levels are almost free.
- Each level is conditioned on the one below:
  head_Ln(pooled workspace, mu_{L(n-1)}) → (mu_Ln, logvar_Ln). That is coarse plans built on fine ones, as in NCM.
- Document boundaries: TinyStories stories are ~200 tokens, packed into 512-token sequences with <|endoftext|>.
  A window is cut at the first end-of-text token, so it never mixes two stories. A position with fewer than n/4 real
  future tokens is masked for that level, along with all positions past the sequence end.
- Loss: Gaussian NLL per level (logvar clamped) + the VICReg variance hinge (as in 09). Weight per level
  think_w × [1, 1, 0.5, 0.5]. The 128-token level stays at 0.5; its share of valid positions on TinyStories (cut
  at the end-of-text token, masked below n/4 real tokens) is measured and reported before the first run.
- Uncertainty per level, u_n = [mean logvar, how much mu_n moved since the last pass]:
  - to the stop head (2 features per level);
  - to the workspace (zero-init feedback, as today).
- Usefulness: as in v4d, the mu of each level is written into the next pass's core input (zero-init inject).
- Replaces the single 16-token latent head (v4d). Per-level diagnostics: NLL, target std / pred std / cosine, AUC
  of "more passes help", and how each level's uncertainty behaves.

## B. The translator (thought → address), one per concept level
- Input: [mu_Ln, pooled workspace].
- Pipeline:
  1. Normalisation with running mean / scale statistics (EMA), frozen at the freeze.
  2. A small projection to a 128-dim key, L2-normalised.
  3. Product keys: split the key into 2 halves of 64; each half is compared (cosine) with 1,024 sub-keys; top-8 per
     half → 64 candidate pairs → the top-k (k = 4-8) addresses out of 1,024² = 1M.
- 3 independent heads per level: separate projection and sub-keys, fixed per-head seeds. They give redundancy.
- Infancy (0 → the end of growth, ~60%):
  - the translator and sub-keys learn by gradient through the soft read weights;
  - addressing uses a SLOW EMA copy of them (decay 0.999), so addresses glide instead of jumping. Split of roles: the
    EMA copy chooses WHICH rows (the top-k indices); the ONLINE translator computes the soft weights on those same
    indices, so the gradient still reaches it. After the freeze both are the same frozen weights;
  - the sub-keys are kept balanced: dead sub-keys are re-seeded, busy ones split, using the expert lessons.
- Freeze (the end of growth): the translator weights, sub-keys and normalisation statistics are frozen bit-exact.
  From then on the same input always gives the same address.
- Margin (light, address-space side only): during infancy a small loss on the translator / sub-keys keeps the best
  and second-best address apart. It never acts on the thought. After the freeze the margin is only measured.
- Determinism: keys and scores in fp32, fixed tie-breaking (the lowest index wins).

## C. Memory kinds and sizes (VRAM: v4e code peaks at 13.4 GB allocated on 24 GB)
- (a) Shared pool:
  - 1M rows × 256 fp32 = 1.0 GB, split into one region per concept level:
    - L1 (16 tokens): 384K rows (details);
    - L2 (32): 256K;
    - L3 (64): 192K;
    - L4 (128): 192K (topics / situations).
  - Each level's product-key space maps onto its own region, so levels never collide with each other.
- (b) Private expert memory:
  - each expert has 512 rows × 256 for the stage tests (768 × 512 × 256 fp32 = 0.4 GB). 2,048 rows (1.6 GB) is a
    later option, once VRAM headroom is known (other apps + reserve);
  - it is read only when that expert fires for the word (6 of 768);
  - addressing: a per-expert key matrix (rows × 64), queried with the shared translator's L1/L2 key (first 64 dims),
    top-4. The keys are LEARNED during infancy like the shared sub-keys (gradient through the soft weights, EMA copy
    picks the rows, dead keys re-seeded, busy keys split) and frozen at the end of growth. They are not fixed random
    from the start.
  - Split births: the twin copies the parent's private memory and keys exactly (no knowledge lost).
  - Experts born after the freeze? Births end at 60% = the freeze point, so none are.
- Total for the stage tests: about +1.5 GB with optimizer state → ~15 GB allocated. With 2K private rows it would be
  ~16.2 GB allocated, which plus reserve and other apps risks out-of-memory, so it stays an option.
- Scaling later: new tables / heads with new seeds next to the old ones. Never re-mod or resize an existing table.

## D. Read and write (every pass)
- Read (shared): for each level and head, the top-k rows are read, weighted by a softmax of their scores. A gate per
  level from the thought, sigmoid(w·[thought, read] + b), b = -2, then W_o_level (zero-init).
- Read (private): inside the routed experts, each chosen expert reads its own top-4 rows and adds them to its output
  through its own zero-init projection.
- Destination:
  - the next pass's core input (the inject path, so the membranes carry it to later words);
  - a dedicated workspace memory slot.
- Write: sparse. Only the rows that were read get a gradient.
  - Row-wise Adagrad (1 float per row; the consolidation LR shrinks with accumulated writes), optionally with a small
    LR floor.
  - No momentum, no weight decay, excluded from z / aux losses and from BrainOpt.
- Init: rows ~ N(0, 0.02) and W_o = 0. A test checks that both get a gradient at step 0.
- Soft top-k reads and writes: a small drift of the thought shifts the weights between the same rows instead of
  jumping to unrelated rows.

## E. Stability toolkit (without constraining thoughts)
1. Frozen translator / sub-keys / norms after infancy: bit-exact mapping.
2. The slow EMA addressing copy during infancy.
3. Margins built into the address space (infancy only).
4. Soft top-k (4-8 rows) reads and writes.
5. Redundancy: 3 heads per level + coarse levels (slow-changing) + private memories.
6. Repair / migration: when the drift monitor finds probe contexts whose rows moved, it copies (blends) their values
   from the old rows to the new rows and logs the repair. Its trigger is per-probe row change after the freeze.
7. Content stability: sparse writes and consolidation, no decay.
8. Automatic guard: if probe stability falls below 99%, it raises the alarm, runs repair, and logs what moved.

## F. Measuring (every eval)
- Probe stability:
  - 2,000 fixed val contexts × all passes, re-addressed;
  - % of identical top-k rows per level and head, since the last eval and since the freeze;
  - target ≥ 99% after the freeze;
  - the code / sub-key assignment stability.
- Margins per level and head.
- Collisions: distinct probe contexts per row; the row usage histogram; dead and overloaded sub-keys (pre-freeze).
- Repairs: count and which levels.
- Memory change: the relative value change of the probe rows.
- Gates per level, pass and kind (shared / private); the private-memory hit rate.
- Knockouts: val and fact recall with the shared / private / each level switched off.
- Routing drift (for private memories): the % of probe contexts routed to the same experts as at the freeze.
- Fact recall and early-fact retention (world data, section H).

## G. Interactions and cost
- Workspace: plus one memory slot (17 slots, or reuse slot 16).
- think_use / inject: now per level, summed into the next pass's core input.
- Stop head: features gain 2 per level (uncertainty) + an optional retrieval confidence.
- Growth: births until 60%. The freeze at the end of growth means all structural change happens in infancy. Private
  memories are copied at splits.
- BrainOpt / carry_brain_state: the memory tables use their own sparse optimizer; translators and gates use BrainOpt
  until the freeze, after which they are excluded (frozen).
- Checkpoints: tables, keys, translators, norm statistics, EMA copies, freeze flag, Adagrad state and repair log.
- Cost per pass:
  - 4 levels × 3 heads × (2 × 1,024 × 64) key scores ≈ 1.6M MACs per word;
  - top-k gathers of 256 floats;
  - private memory 6 × 2,048 × 64 ≈ 0.8M MACs per word;
  - level heads ≈ 4 × 0.46M MACs.
  - Total ≈ +5M MACs per word per pass (~+15-25% of a pass) plus ~2.7 GB.
  - Option: shared memory reads only on passes 1, 3 and 5. Not proposed by default (Eitan: every pass).

## H. Test data: a generated world
- 5,000 people (age, city, job, pet, best friend), 300 cities (country, river), 200 jobs.
- Paraphrased fact sentences and mini-stories, mixed about 20% world / 80% TinyStories.
- Questions with the fact NOT in the sequence: recall, two-hop, comparisons.
- An early-facts subset shown only in the first 20% of training (retention).
- All compared runs use the same mix.
- Separation: knockout should kill recall while val and in-context reasoning stay; small core + memory vs bigger core.

## I. Tests (CPU, before any run)
- Frozen addressing gives bit-identical addresses over 1,000 training steps.
- Rows that weren't read are bit-identical after steps.
- Causality: changing tokens after t never changes word t's reads.
- Step-0 gradients reach the tables and W_o.
- Hierarchical targets: the window means with the EOT cut and masks are hand-checked.
- Level conditioning.
- Split births copy the private memory and keys exactly.
- Checkpoint reload keeps the addresses.
- Repair: blending the moved rows restores the probe reads.
- The margin loss touches only the translator / sub-keys.

## J. Staged plan
1. Hierarchical concepts alone (v4d recipe + 4 levels vs v4d's single level), a short paired run on TinyStories.
2. The world dataset + eval harness.
3. Shared memory (hierarchical) short paired runs on the world mix: none / shared.
4. + private expert memories: shared vs shared + private.
5. A long run (200M): retention, probe stability around the freeze, repairs, knockout separation.
6. Scale: table sizes; small core + memory vs bigger core.

## Open decisions for Eitan
- Private memory size per expert: 512 for the stage tests (proposed); 2,048 later if VRAM allows.
- Reads on every pass (proposed, per your instruction) vs selected passes, if cost matters.
- Level weights and whether the 128-token level is worth it on TinyStories (stories are ~200 tokens).
