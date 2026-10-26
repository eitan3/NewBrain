# Plan 26: brain-like memory to replace the blob store (wave memory sheet)

Status: design study. No model code was changed. CPU toys (≤ 2 threads, no GPU) are in
`C:\Eitan\Python\NewBrain\speed_wip\mem26\` (scripts plus `*.txt` outputs). This plan replaces 25d's blob store at
the same or smaller size and time. It does not add capacity on top of it.

---------------------------------------------------------------------------------------------------------------------

## 0. Summary for the morning review

> **Update after the pre-build checks (§8.3-8.4, 2026-10-25):**
>
> - **W1 failed for the §6.3 entry map.** With random-projection location and hard units, cloze top-1 is 0.24.
> - **Coarse-to-fine entry fixes it:** an online SOM location, then 3 residual code levels, one per wave pass, with
>   soft top-2 units. Cloze top-1 0.52 (held 0.68), above the 25d address (0.50 / 0.69) and about 80% of the exact
>   384-d ceiling (0.65).
> - **W2 passed:** 300-600K memories in 472 MB.
> - **W3 passed with phase only:** cos 0.85 vs 0.75.
> - **W5:** phase gives order selectivity.
> - **W4:** late facts survive (about 0.7), **rare facts do not** at steady state (about 0.3). The two-tier option
>   does not help; forgetting by use is the next fix to test.
> - Concepts do not improve the cue.
> - The full-module harness `wbench_module.py` scores any MemoryBase module on its real write and read paths.

**Recommended design: the Wave Memory Sheet (WMS).** It is a separate, swappable module.
- **The sheet:** a 32 × 32 torus of hypercolumns with 64 units each (65,536 memory neurons).
- **Start of a read:** a thought (the expert input xf, fixed-normalised, detached) lights an *entry front*, a 4 × 4
  patch of hypercolumns with one unit each.
- **3 passes:** activity then travels as a wave. In each pass the next front is the adjacent patch that the synapses
  drive most, and visited columns are refractory. So every pass lights different neurons: new routing each pass.
- **Where memories live:** in the sheet's synapses. Each synapse is a presence bit plus a 3-bit learned **phase
  shift**. Timing is a **phase code**: each active neuron is a complex number (magnitude = firing, angle = when in the
  wave it fires). Inputs that arrive in phase add, inputs out of phase cancel. That gives coincidence detection with
  batched complex multiply-adds. There are no clocks, no dt and no real delays.
- **What a read returns:** fronts 1-3 spell out the content as a 3-level residual code (unit = coarse code,
  phase = refinement). It is decoded by fixed codebooks into a 384-d read that goes into the expert input, as in 25d.
- **Gate and learning:** a small memory-side gate uses the wave's own coincidence strength. Writes happen in-step
  after backward, with a local Hebbian / phasor rule. Forgetting is homeostatic, also in-step. Reconsolidation
  happens on recall. There is no sleep phase, no replay from data and no row table.

**Size and time against today:**

| | blob store (25d, measured) | WMS (projected) |
|---|---|---|
| memory state | ≈ 1.9 GB store; peak allocated 19.45 GB vs 16.0 GB without memory | **0.47 GB** synapses + < 5 MB codebooks/maps + ≈ 60 MB per-read activations |
| step time | ≈ **1.27 s of 2.21 s per 16,384-token step (≈ 58%)** | **≈ 50-100 ms** per step (bandwidth-bound estimate; needs a Triton kernel) |
| memories held | ≈ 150K global blobs live (gm_live_g) + private | ≈ 150K at fill 0.12 (300K at fill 0.25 or with M = 128) |

**What the CPU toys say (§4):**
1. **Synapses are cheap.** A 1-bit hypercolumn attractor holds 16K patterns in 2 MB (≈ 8K memories/MB). A blob takes
   ≈ 5 KB (≈ 200/MB). Capacity grows ∝ M² (M = 256 holds 4× what M = 128 holds).
2. **Settling over 3 passes matters.** With 25% of the cue wrong: 0.79 content accuracy after 1 pass, 0.98 after 3
   (8K patterns).
3. **The wave version works.** On a 12×12×55 sheet, 32K wave paths are recalled perfectly from a clean cue, and at
   0.77-0.80 with 25% of the cue wrong.
4. **Phase is a trade.** At equal neurons, phase doubles the patterns and adds 3 bits per active unit. At equal synapse
   bits, 1-bit weights-only is ≈ 1.6× better (0.107 vs 0.067 bits per synapse bit). Integer pass-delays (delay-2
   synapses) also lose at equal bits. **Phase stays the timing code (owner direction), behind a flag
   `--wm-phase 3|0`.** The pilot runs both arms, so the per-byte cost is measured, not assumed.
5. **Skewed unit usage is the main capacity killer.** Willshaw at Zipf-1 usage: 0.70 at 4K patterns. A Bayesian-Hebbian
   (BCPNN) correction or usage homeostasis brings it back to 0.99. The design has homeostasis on entry locations and
   balanced codebooks.
6. **In-step self-replay works.** From a palimpsest fast tier into a separate long-term synapse set, with no data and no
   sleep: 0.81 of the oldest memories are kept in long-term (fast tier alone: 0.01), with 0 spurious writes. This is an
   option (`--wm-tiers 2`), off by default.

**Biggest risk (same as 25d):** the same fact in other wording must light the same entry patch. The toys tolerate
≤ 25% wrong entry *units*, but a wrong entry *location* loses the memory. CPU check W1 (§8) measures this on the
blob run's 6K checkpoint before anything is built. Mitigation: two entry locations (`--wm-entries 2`).

Ranking:
1. **WMS (phase)**: recommended, specified in §6.
2. **Hypercolumn attractor (BCPNN, all-to-all, 3 settling steps)**: the non-wave fallback. Best capacity per byte in
   the toys, same interfaces, no geometry. It costs more per read because of the dense fan-out.
3. **Delta-rule fast-weight module (DeltaNet / Titans style, separate matrix)**: simple and swappable, but its capacity
   per byte is far lower (linear in key dimension) and its writes interfere densely. Only useful as a small fast tier.

---------------------------------------------------------------------------------------------------------------------

## 1. Rules this design follows (owner + coordinator notes, all applied)

1. Addresses come only from the brain's own thoughts (xf, the expert input). No token keys, no dataset pointers. The
   address maps (centring, random projections, codebooks) are fit once and then fixed. Nothing on the address side is
   learned by gradient.
2. Thoughts are never constrained for the memory's sake. **No memory-side loss reaches the brain or the thoughts.** The
   read is discrete (unit and phase choices), so no gradient flows from the match into xf. The main loss may train the
   brain to *use* the read (ordinary main-loss learning) and trains the memory's own read-out scalars and gate.
3. **The memory is separate from the brain and expert weights and swappable.** It can be swapped, removed, or trained
   for one expert only, and the brain still works. Brain and expert weights change only through the main loss. No
   consolidation or replay goes into them.
4. Everything happens inside the normal training step: no sleep, no offline phase, no extra recompute loop. Replay, if
   used, comes only from the memory's own dynamics.
5. No row or hash tables and no product-key memory. Quantised or sparse codes appear only as the way synapses and wave
   patterns are written.
6. "3 passes" are recall dynamics *inside one memory access*. Per-outer-pass banks stay banned: the same sheet serves
   every outer pass, so 6 or 40 passes behave the same.
7. A replacement, not an addition: ≤ today's memory VRAM (≈ 1.9 GB) and ≤ today's memory step time. The target is far
   below both.
8. No dt-stepped or real-delay simulation. Timing is computed in closed form inside the 3 passes.

Note for the owner: the sheet's synapses are stored as packed arrays indexed by the firing neuron. Every event-driven
spiking simulator does this, because a spike delivers along its axon. No memory owns a slot. Each memory is spread
over ≈ 800 synapses that are shared with many other memories, and recall is dynamics, not lookup. If you count
"gather the synapses of the neurons that fired" as a row table, then every sparse SNN is one, and you should tell us.

---------------------------------------------------------------------------------------------------------------------

## 2. What today's memory costs (from logs only, no GPU job)

Sources: `runs\world_blob25d_12k\metrics.jsonl` (live run, 39 records to step 8,750) and `train.log`, plus the
no-memory control `runs\world_v4d_12k` (same model and growth flags, `--batch 16 --accum 2`, same data). The metrics
have no gm timing fields (only `tok_s`), so the share is computed from tok/s.

| step | blob 25d tok/s | no-memory control tok/s | old row memory (mem_life) tok/s |
|---|---|---|---|
| 250 (blob memory not yet fitted; fit at 300) | 15.6K (restarts r2/r3: 22.5K / 23.2K) | 21.7K | 10.9K |
| 1,000 | 8.2K | 22.8K | 9.8K |
| 3,000 | 8.1K | 19.3K | 9.7K |
| 7,000 | 7.4K | 17.6K | 8.6K |
| run average | 7.9K (to 8,750) | 18.8K (full 12K) | 9.0K |

- **The blob memory takes ≈ 55-60% of step time.** At 7K that is 2.21 s per step vs 0.93 s, so memory ≈ 1.27 s per
  16,384-token step. Before the memory switches on, the blob run (concepts on) matches the control (22.5-23.2K vs
  21.7K at step 250). So the concept hierarchy explains almost none of the gap.
- **Caveats:**
  - the control ran an older code snapshot (`baseline_mem_v3`) with `ckpt_route 0`; the blob run has `ckpt_route 1`,
    which adds recompute;
  - the GPU may have been shared at times.
  - The "8k vs 10k" in the brief understates the gap. 10K was never the no-memory speed of this config; ≈ 18-19K is.
- **VRAM:** peak allocated 19.45 GB (blob) vs 16.0 GB (control) at the end. It was 22.4 GB around step 7K, with eval
  included. So the store is ≈ 1.9 GB, and the peak is ≈ 3.5 GB above no memory.
- **Budget used here:** memory ≤ 1.9 GB and ≤ 1.27 s per step. **Target:** ≤ 0.6 GB and ≤ 0.1 s, so the run would go
  from ≈ 7.5K to ≈ 16-17K tok/s.

---------------------------------------------------------------------------------------------------------------------

## 3. Why GBs are not needed (information budget)

- **Facts are small.** A fact's answer is ≈ 14 bits (vocab 16,384). LMs store knowledge at ≈ 2 bits per parameter
  when trained to saturation (Allen-Zhu & Li 2024, *Physics of LMs 3.3: Knowledge capacity scaling laws*).
- **Synapses are efficient.** Sparse Hebbian associative memories reach up to ln 2 ≈ 0.69 bits per binary synapse
  asymptotically (Willshaw, Buneman & Longuet-Higgins 1969; Palm 1980). Sparse Hopfield: Tsodyks & Feigel'man 1988.
- **A blob is not.** It spends ≈ 5 KB (64-d key + 384-d content + optimiser and bookkeeping state; 1.9 GB / ≈ 350K)
  on a few tens of useful bits.
- **The job is fast, robust acquisition.** The memory's problem is one-shot acquisition, robustness to drift and
  wording, and harm control. Bulk capacity is not the problem.
- **Synapses suit that job.** They store a memory in ≈ 0.1-1 KB of synapse bits, and they come with pattern
  completion (multi-pass settling) and a free familiarity signal.

---------------------------------------------------------------------------------------------------------------------

## 4. CPU toy results (all in `speed_wip\mem26\`)

### 4.1 Hypercolumn attractor (toy_attractor.py)

- **Patterns:** H = 32 hypercolumns × M units, one active unit per hypercolumn. The first 16 hypercolumns are the cue,
  the last 16 the content.
- **Synapses:** Hebbian outer products, as float, int8, 1-bit (Willshaw) or BCPNN (log P_ij / P_i P_j).
- **Recall:** T settling steps with per-hypercolumn WTA and a persistent cue drive.
- **Measures:** content accuracy, by pattern age.

| config (synapses) | P | clean cue T1 / T3 | 25% cue wrong T1 / T3 | 50% wrong T1 / T3 |
|---|---|---|---|---|
| 32×128, 1-bit, 16.8M bits = 2 MB | 8K | 1.000 / 1.000 | 0.855 / 0.980 | 0.41 / 0.63 |
| same | 16K | 0.961 / 1.000 | 0.44 / 0.57 | 0.16 / 0.17 |
| 32×128 float counts | 8K | 0.987 / 1.000 | 0.79 / 0.98 | 0.42 / 0.60 |
| 32×256, 1-bit (67M bits = 8 MB) | 32K | 1.000 / 1.000 | 0.83 / **0.977** | 0.36 / 0.54 |
| same | 64K | 0.93 / 1.000 | 0.36 / 0.45 | 0.10 / 0.10 |
| 32×128, 1-bit, **Zipf-1 unit usage** | 4K | 0.67 / 0.68 | 0.33 / 0.33 | — |
| 32×128, **BCPNN**, Zipf-1 usage | 4K | 0.93 / 0.99 | 0.80 / 0.93 | 0.47 / 0.59 |
| 32×128, palimpsest γ = 0.99975/pattern | 32K | newest 10%: 0.99 / 1.00; oldest: 0.01 | | |

- **Familiarity** (WTA margin), stored vs novel cue: AUROC 0.97-1.00 up to near capacity.
- **Spontaneous replay** from a random state: it lands on a stored pattern only when the store forgets (palimpsest):
  ≈ 35% of starts, mostly recent ones (median age ≈ 200 patterns). Without forgetting, at load: 0%.

### 4.2 Wave sheet (toy_wave.py)

- **Sheet:** S × S torus of hypercolumns. A memory is a path: entry patch → 3 adjacent patches, one unit per
  hypercolumn, 1-bit synapses front t → t+1.
- **Optional extras:** delay-2 synapses t → t+2 and lateral within-front clean-up.
- **Recall:** light the entry, then 3 passes. Refractory visited hypercolumns; the next front = the strongest
  non-visited hypercolumns; WTA inside each.

| sheet (synapse bits) | P | clean front 1/2/3 | 25% entry units wrong | familiarity AUROC |
|---|---|---|---|---|
| 12×12×32, F = 9, delay-1 only (21M) | 16K | 0.999 / 1.000 / 0.997 | 0.71 / 0.69 / 0.65 | 1.00 |
| 12×12×32 + delay-2 + lateral (64M) | 32K | 0.905 / 0.998 / 0.989 | 0.35 / 0.55 / 0.39 | 1.00 |
| **12×12×55, delay-1 only (63M = equal bits)** | 32K | **1.000 / 1.000 / 1.000** | **0.80 / 0.79 / 0.77** | 1.00 |
| 8×8×64, F = 4, delay-1 | 16K | 0.99 / 0.98 / 0.97 | 0.60 / 0.52 / 0.48 | 0.99 |

- **Delays:** delay-2 synapses repair later fronts (front 2 beats front 1). At *equal synapse bits*, though, spending
  the bits on more units wins.
- **Front size:** bigger fronts (F = 9-16) are much more robust than F = 4, because each pass has more evidence.

### 4.3 Phase vs weights only (toy_phase.py, toy_phase_m128.py)

- **Phasor memory:** the threshold phasor associative memory idea, in hypercolumn form. Active unit z = e^{iφ} with
  Q = 8 phases; W_ij = Σ_μ z_i^μ conj(z_j^μ); recall u_j = Σ_i conj(W_ij) z_i, winner = argmax |u|,
  phase = angle(u); 3 passes.
- **Quantised synapse:** presence bit + 3-bit phase of the sum.
- **Information measure:** mutual information per content hypercolumn (unit + phase), summed over patterns, divided by
  synapse bits.

| config | synapse bits | best clean bits per synapse bit (P) | best at 25% cue wrong (P) |
|---|---|---|---|
| weights 1-bit, M = 128 | 16.8M | **0.107** (16K) | 0.052 (8K) |
| phasor 4-bit, M = 64 (equal bits) | 16.8M | 0.062 (8K) | 0.033 (4K) |
| weights 1-bit, M = 180 | 33M | **0.116** (32K) | 0.055 (16K) |
| phasor 4-bit, M = 90 (equal bits) | 33M | 0.063 (16K) | 0.034 (8K) |
| phasor 4-bit, M = 128 (equal *neurons*, 4× bits) | 67M | 0.067 (32K: unit 0.94, phase 0.95) | 0.035 (16K: 0.96) |
| phasor float, M = 128 (16-bit synapses) | 268M | 32K: unit 0.97 vs weights-only 0.09 (bit) / 0.48 (count) | |

- **Per neuron, phase wins:** about 2× the patterns, robust to noise, plus 3 extra bits of content per active unit.
- **Per synapse bit, phase loses:** ≈ 1.6-1.8× worse. Each pass also needs ≈ 4× the bytes moved and 4× the MACs
  (complex) of a 1-bit pass.
- **What weights alone cannot do:** distinguish two memories that light the *same* units in a different order, such
  as relation direction (X→Y vs Y→X) or temporal order. That needs a dedicated toy (W5).

### 4.4 Content code fidelity (toy_pq.py)

- **Code:** a front = 16 hypercolumns, each coding one 24-d slice of the 384-d content; fronts 1-3 code successive
  residuals.
- **Proxy data:** Gaussian with a flat or 1/f spectrum. Real write targets are untested.

| unit code per hypercolumn | cos after fronts 1 / 2 / 3, flat spectrum | 1/f spectrum |
|---|---|---|
| 6 bits (64 units) | 0.47 / 0.63 / 0.73 | 0.60 / 0.72 / 0.79 |
| 8 bits (≈ 64 units × 4 phases) | 0.54 / 0.71 / 0.81 | 0.65 / 0.78 / 0.85 |

64 units × 8 phases (9 bits) projects to ≈ 0.83-0.88 after 3 fronts. This is enough for the blob's role: in 25d the
content norm is only ≈ 0.18 of ‖xf‖ (gm_v_norm_rel_med), and the useful part is a direction. W3 measures it on real
contents.

### 4.5 In-step consolidation by self-replay (toy_consolidate.py)

- **Setup:** 16 writes per step into a fast palimpsest tier (life ≈ 2K patterns). In the same step, 32 spontaneous
  waves settle on the fast synapses, and the converged ones (stable and clear margin) are written into a separate
  1-bit long-term synapse set on the same units.
- **Results:**
  - 12,288 replays were written, 0 spurious.
  - Recall from long-term alone: 0.81 for the oldest 20%, 0.68 mid, 0.46 newest. Fast tier alone: 0.01 / 0.56 / 1.00.
  - Direct long-term writes (upper bound): 1.00 everywhere.
- **Conclusion:** consolidation from the memory's own replay works with no data and no sleep. The read must take
  whichever tier gives the stable wave; a naive sum of the two tiers is worse (0.56 newest).

---------------------------------------------------------------------------------------------------------------------

## 5. Candidates considered

For each candidate: storage and recall, fit to the rules, capacity, cost, learning, change, failure, and how to swap
it or train it per expert.

### 5.1 Wave Memory Sheet (recommended, §6)
Hypercolumn sheet, entry front from the thought, 3 wave passes, phase-coded synapses, local connectivity.

- **Capacity:** ≈ 150-300K memories in 0.47 GB.
- **Cost:** reads cost ≈ 3 × 16 active units × a 7 KB synapse window.
- **Learning:** one-shot Hebbian / phasor writes in-step; homeostatic forgetting; reconsolidation on recall.
- **Failure modes:**
  - entry-location disagreement across wording;
  - hub locations (handled by homeostasis);
  - content code fidelity;
  - phase overwrite noise.
- **Swap and per-expert training:** the whole sheet is a file. Per-expert private sheets are an option (§6.9).

### 5.2 Hypercolumn attractor (BCPNN / Willshaw, all-to-all), fallback
Same codes, but every unit connects to every other and recall is 3 settling steps.

- **Capacity:** best per byte in the toys (P ∝ M²; ≈ 8K memories/MB at 1 bit).
- **Cost:** each active unit fans out over all N units, so per-read bytes ∝ k·N. At N = 16K, k = 32 that is
  ≈ 64 KB per step per read. This is cheaper than WMS per pass, but there is no wave and no locality.
- **Swap and per-expert training:** like WMS.
- **When to use it:** if the wave geometry turns out to cost recall (entry location sensitivity is worse here only if
  hypercolumns are positional; they are not).
- References: Lansner & Ekeberg 1989; Lansner 2009 *Associative memory models: from the cell-assembly theory to
  biophysically detailed cortex simulations*; Ravichandran, Lansner & Herman 2020s (BCPNN representation learning);
  Tsodyks & Feigel'man 1988; Amit, Gutfreund & Sompolinsky 1985 (0.138 N).

### 5.3 Delta-rule fast-weight matrix as a separate module
F ∈ R^{384×D_k}, with φ = fixed random features of xf.

- **Write:** F ← F + β (v − F φ) φᵀ.
- **Read:** r = F φ.
- Literature: Schlag, Irie & Schmidhuber 2021 *Linear Transformers Are Secretly Fast Weight Programmers*; Yang et
  al. 2024 *Parallelizing Linear Transformers with the Delta Rule*; Gated DeltaNet (Yang, Kautz & Hatamizadeh,
  ICLR 2025); Ba et al. 2016 *Using Fast Weights to Attend to the Recent Past*; Hinton & Plaut 1987; Titans (Behrouz,
  Zhong & Mirrokni 2025): a gradient-updated MLP memory with surprise momentum and forgetting; Miconi et al. 2018/2019
  differentiable plasticity / backpropamine.
- **Capacity:** ≈ D_k associations (linear), and less with correlated thoughts. Every write touches every read, so
  the risk of harm to other text is high.
- **Use:** fine as a tiny fast tier (a few MB). Not competitive as the main store. A sparse φ (k-WTA) turns it into
  Kanerva SDM, which is mathematically §5.2's feed-forward half.

### 5.4 Modern / dense Hopfield, Kanerva machine, compressed blobs
- **Hopfield / Kanerva:** modern Hopfield (Krotov & Hopfield 2016; Demircigil et al. 2017; Ramsauer et al. 2020) stores
  patterns explicitly: softmax over stored keys, the same thing as blobs. Kanerva Machine (Wu, Wayne, Graves &
  Lillicrap 2018) and Larimar (Das et al. 2024) are episodic, per-context. Neither cuts storage.
- **Compressed blobs:** int8 keys, PQ content of ≈ 150 B instead of ≈ 5 KB. This keeps the 25d logic and makes the
  replacement ≈ 10× smaller, but it does **not** cut time. IVF search plus the gates dominate, and 3-step Hopfield
  settling would triple the search.
- **Use:** the fallback only if every synaptic option fails W1-W4.

### 5.5 Rejected
- Product-key memory / memory layers at scale (Lample et al. 2019; Berges et al. 2024): row tables.
- SDM with hard locations used as addressable slots: a row table.
- Consolidation into brain or expert weights, and complementary learning systems in the classic sense (McClelland,
  McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland 2016): banned. The two-tier variant here consolidates
  only inside the memory.
- Any sleep or offline replay: banned.
- Per-pass banks: banned.
- Real conduction delays or dt-stepped spiking: banned for cost. Their *function* is kept through phase.

### 5.6 Timing options (coordinator / owner question)
- **(1) Phase code: recommended.**
  - Neuron state z = a e^{iφ}: a = firing strength (0/1 after WTA), φ = wave timing (8 levels).
  - Synapse = presence p_ij plus learned phase shift θ_ij, which plays the role of a conduction delay (a delay
    shifts arrival time, a phase rotation shifts arrival angle).
  - Input u_j = Σ_i p_ij e^{i(φ_i + θ_ij)}. Coincident (in-phase) inputs add; others cancel. That is closed-form
    coincidence detection, with no clock and no dt.
  - Cost vs a real-valued pass: 4 real MACs per synapse instead of 1, and 4 bits of state instead of 1. The pass is
    bandwidth-bound, so ≈ 3-4× a 1-bit pass. At WMS sizes that is still ≈ 0.1 s per step at most (§6.8).
  - Gradient path: not needed for storage (the write is Hebbian). The read-out gains get the main loss; no gradient
    goes into thoughts.
  - Capacity gain: 2× patterns per neuron and +3 bits per active unit, but 1.6× worse per synapse bit (§4.3). It adds
    order and timing selectivity, which weights alone lack.
  - References: Frady & Sommer 2019 (PNAS), *Robust computation with rhythmic spike patterns* (threshold phasor
    associative memory, TPAM); Noest 1988 (phasor neural networks); Plate 1995/2003 holographic reduced
    representations; Frady, Kent, Olshausen & Sommer 2020 resonator networks; Löwe et al. 2022 (complex autoencoder) /
    rotating features (Löwe et al. 2023).
- **(2) Closed-form latency:** t_j = softmin_τ(t_i + d_ij) − threshold. One add and one (soft-)min per synapse:
  tropical (min, +) algebra, no tensor cores, but bandwidth-bound like (1).
  - Gradient: exists through the softmin (SpikeProp, Bohte, Kok & La Poutré 2002; tempotron, Gütig & Sompolinsky 2006;
    learnable delays: Hammouamri, Khalfaoui-Hassani & Masquelier, ICLR 2024).
  - Capacity: polychronous groups can outnumber neurons (Izhikevich 2006, *Polychronization*).
  - Verdict: equivalent in spirit to (1). With Hebbian writes it gains nothing over phase, and min-plus has no
    hardware support. Keep as a footnote.
- **(3) Pass index / integer delays 0/1/2:** the delay-2 matrices in toy W. They repair later fronts but **lose at
  equal bits** (§4.2). Dropped as a capacity tool. The pass index already exists implicitly as the front number.
- **Order of trials:** weights-only (`--wm-phase 0`) and phase (`--wm-phase 3`) are both built. The pilot compares them
  at equal bytes, and toy W5 tests whether order or timing adds recall the weights cannot.

---------------------------------------------------------------------------------------------------------------------

## 6. The recommended design: Wave Memory Sheet (WMS), specified for implementation

### 6.1 Module and state (all memory-side, saved and loaded as one file)

| name | shape / dtype | what it is | learned? |
|---|---|---|---|
| μ, σ | [384] fp32 | centring / scaling of xf, fit on 64K positions at `--wm-fit-step` (300) | fixed after fit |
| Lproj | [1024, 384] bf16 | fixed Gaussian projection to entry locations (32×32 torus positions) | fixed |
| b_loc | [1024] fp32 | entry-location homeostatic bias | in-step homeostasis (not gradient) |
| Dproj | [8, 384] bf16 | fixed Gaussian projection to the 8 wave directions (default path) | fixed |
| E | [16, 64, 24] fp32 | entry codebooks per slice (k-means on x̂ slices at fit) | fixed after fit |
| Ephase | [16, 2, 24] fp32 | fixed random pair (a_s, b_s) per slice: entry phase = angle(a_s·x̂_s + i b_s·x̂_s) | fixed |
| C | [3, 16, 64, 8, 24] fp32 | content codebooks: level t, slice s, unit m, phase q → 24-d (residual k-means on write targets collected during steps 300-500; before that, writes are off) | fixed after fit |
| Wsyn | [65536, 225, 32] uint8 | sheet synapses, one 4-bit nibble per (pre unit, target hypercolumn in the 15×15 window, target unit): bit 3 = presence, bits 0-2 = phase shift θ (×π/4) | in-step plasticity (not gradient) |
| use_hc | [1024] fp32 | EMA of each hypercolumn's write usage (homeostasis, logs) | in-step |
| α | [3] fp32 | read-out gain per front level (init 1, 0.5, 0.25) | **main loss** |
| γ | [1] fp32 | read scale (init 0.2 · median ‖xf‖ / ‖C-decoded unit vector‖) | **main loss** |
| gate | Linear(10 → 16) → GELU → Linear(16 → 1) | read gate on detached wave features | **main loss** |

- **Sheet geometry:**
  - 32 × 32 hypercolumns on a torus, M = 64 units each, so U = 65,536 units.
  - Unit id u = hc · 64 + m, with hc = y · 32 + x.
  - A front is a 4 × 4 patch.
  - Hypercolumn (y, x) codes slice s(y, x) = (y mod 4) · 4 + (x mod 4). Any 4 × 4 patch therefore covers all 16
    slices once.
- **Window:**
  - Target hypercolumns of unit u are the 15 × 15 block centred on its hypercolumn (offsets −7..+7, torus wrap).
  - Window index w = (dy + 7) · 15 + (dx + 7).
  - Nibble storage packs 2 target units per byte, so [225, 32] bytes = 225 × 64 nibbles.
  - Size: 65,536 × 225 × 32 B = **472 MB**.
- **Flags for a bigger or smaller sheet:**
  - `--wm-units 128` doubles units and synapses (944 MB, ≈ 2× memories, 2× read bytes).
  - `--wm-phase 0` stores 1-bit presence only (118 MB). Use the bytes saved for `--wm-units 128` with phase 0 at
    236 MB, which is the equal-byte control arm.

### 6.2 Where it sits in the brain (same read point as 25d)

- **Placement:**
  - In `RoutedMLP.forward`, after xf = core.n_m(h) and routing, before the experts.
  - One WMS read per token per core pass. All of a token's routed experts get the same xf, so one wave serves them all.
  - The read vector is added to each routed expert's buffer row before w_up, and to the shared expert's input.
- **Inject:** xf_e ← xf + g · γ · r. With no read (gate closed), xf is untouched.
- **No brain-side memory weights:** the codebook output is already in model space.
- **Detached:** x̂ = ((xf − μ)/σ).detach().

### 6.3 Forward: one read (B tokens in parallel; shapes per micro-batch pass, B = 8,192)

1. **Entry location.**
   - ℓ = x̂ Lprojᵀ + b_loc → [B, 1024].
   - p0 = argmax ℓ. With `--wm-entries 2`, also the second best, as a second wave whose features are max-pooled.
   - The entry patch P0 is the 16 hypercolumns of the 4×4 patch whose top-left corner is p0.
   - Feature: loc_margin = (ℓ₁ − ℓ₂)/std(ℓ).
2. **Entry units and phases.** For each h ∈ P0 with slice s:
   - m_h = argmax_m ⟨E[s, m], x̂_s⟩;
   - φ_h = round(angle(⟨a_s, x̂_s⟩ + i⟨b_s, x̂_s⟩) / (π/4)) mod 8.
   - Front F0 = 16 × (hc, m, φ) → int tensors [B, 16] × 3.
3. **Passes t = 1, 2, 3** ("new routing each pass"):
   1. **Candidates:** the 8 patches adjacent to F_{t−1} (shift ±4 in y and/or x), excluding any patch that overlaps a
      visited front (refractory). That gives 8 patches × 16 hypercolumns × 64 units = 8,192 candidate target units,
      all inside the 15 × 15 window of every unit in F_{t−1}.
   2. **Phasor input:** for each candidate unit j,
      u_j = Σ_{i ∈ F_{t−1}} p_ij · exp(i · (φ_i + θ_ij) · π/4).
      That is 16 terms, with nibbles gathered from Wsyn[u_i, w(i, j), j_nibble].
      The kernel loads 16 rows × 128 window-hypercolumns × 32 B = 64 KB per read and accumulates complex sums for
      8,192 targets in fp16. SRAM: 32 KB.
   3. **Next front:**
      - patch score S_k = Σ_{h ∈ patch k} max_m |u_{h,m}| + β_dir · ⟨Dproj[k], x̂⟩ / ‖x̂‖, with β_dir = 0.5.
        The direction term only breaks ties for novel cues and defines the default write path.
      - k* = argmax S_k.
      - Per hypercolumn h of patch k*: m_h = argmax_m |u_{h,m}|, φ_h = round(angle(u_{h,m_h}) / (π/4)) mod 8.
      - This gives F_t.
   4. **Features:**
      - coincidence c_t = mean_h max_m |u_{h,m}| / 16 ∈ [0, 1] (the share of the front's 16 inputs arriving in phase);
      - margin_t = mean_h (|u|₁ − |u|₂) / 16;
      - dirgap_t = (S_{k*} − S_second) / 16.
   5. **Early exit** (`--wm-early-exit 0.15`): if c_1 < θ_exit, stop after pass 1. The read is 0, and the gate sees
      only pass-1 features. Expected ≈ 60-80% of tokens exit (novel thoughts), which bounds the cost.
4. **Decode.**
   - r = Σ_{t=1..3} α_t · concat_s C[t, s, m_{h(s),t}, φ_{h(s),t}, :] → [B, 384].
   - Each front fills each slice once; level t is the t-th residual.
5. **Gate.**
   - f = [c_1, c_2, c_3, margin_1..3, dirgap_2, dirgap_3, loc_margin, ‖r‖/‖xf‖] (detached, 10 features).
   - g = σ(gate(f) + b0), with b0 = −1.5 at init (≈ 0.18 open).
   - The read is used where g > `--wm-gate-floor` (0.05); otherwise g = 0, for speed and exactness.
   - Exploration: on 5% of tokens g is replaced by U(0, 1) (`--wm-explore`), so the main loss still sees the
     gain-vs-g relation where the gate is closed.
   - The gate and α, γ are trained by the main loss: ∂L/∂g = ⟨∂L/∂xf_e, γ r⟩. The features are detached and r has
     no path to xf, so **no gradient reaches the thought**.
6. **Saved for writes and logs:** the fronts (4 × 16 × (hc int16, m uint8, φ uint8) = 256 B per read), c_t, g.
   That is 8,192 × 256 B ≈ 2 MB per micro-batch pass; ≈ 60 MB per step at full K.

The forward has no backward through Wsyn, E, C or Lproj: they are buffers. Autograd covers only α, γ, the gate and
the brain.

### 6.4 Write (plasticity) rule, in-step, after each micro-batch's backward

1. **Candidates.**
   - Surprise σ_t = CE_t − EMA of CE for that token id (a fixed 16,384-entry EMA, memory-side).
   - The top `--wm-write-w` = 32 positions per micro-batch (64 per step), ≤ 4 per sequence, as in 25d.
   - Only in-window passes are used, and the pass with the largest ‖∂L/∂xf‖ is taken. Brain signals only.
2. **Content target.**
   - c = −∂L/∂xf at (token, pass), normalised: ĉ = c/‖c‖ · κ, with κ = running median of ‖r‖ over confirmed reads
     (init 0.18 ‖xf‖, the 25d content scale).
   - If the token had an open read (g > 0.05): ĉ ← r_read + ĉ − proj. That is reconsolidation: the stored memory
     becomes the old recall plus the correction, so **memories change**.
3. **Encode:** residual-quantise ĉ, level t = 1..3, per slice s: (m, q) = argmin ‖res_s − C[t, s, m, q]‖, then
   res_s −= C[t, s, m, q]. This gives 3 × 16 (unit, phase) pairs.
4. **Path.**
   - **Familiar** (c_1 ≥ θ_fam = 0.5 on this write's forward wave): reuse its recalled patches P1-P3 (same
     hypercolumns). Then depress the old content units' links. For every (i ∈ F_{t−1}^old, j ∈ F_t^old) where j is
     replaced, clear the presence bit with probability 0.5 (anti-Hebbian replacement).
   - **New:** the fronts are the patches chosen by the forward wave. For a novel cue these follow the default
     direction term, so the path depends only on the thought. If the chosen patch has use_hc above 2× the mean, take
     the next direction (homeostasis).
   - Entry units and phases are the forward's own F0.
5. **Synapses** (Hebbian / phasor), for t = 1..3 and every pair (i ∈ F_{t−1}, j ∈ F_t), 16 × 16 = 256 pairs per
   transition, 768 per memory:
   - if p_ij = 0: p_ij ← 1, θ_ij ← (φ_j − φ_i) mod 8;
   - if p_ij = 1: θ_ij ← (φ_j − φ_i) mod 8 with probability ½ (stochastic phasor averaging; `--wm-phase-merge`).
   - All 64 memories of a step are applied in one scatter: 49K nibble writes, < 1 ms.
6. **Homeostatic forgetting** (in-step, every step, no offline phase).
   - Target presence fill `--wm-fill` = 0.12.
   - Each step, clear presence bits on a random fraction λ of the set synapses, with
     λ = (writes_per_step × 768) / (fill_target × 944M), adapted by a slow controller on the measured fill
     (sampled on 1M random nibbles per 100 steps).
   - λ ≈ 4.3e-4 per step, so an unrefreshed memory lives ≈ 2-3K steps. That covers the rare-fact recurrence
     (≈ 1.3K steps) and matches the blob store's evict ages (2.7-6.3K).
   - Recalled memories that helped are rewritten (step 2), so recurring facts live indefinitely.
7. **Entry homeostasis:** b_loc[p] ← b_loc[p] − 0.01 · (use_loc[p] / mean − 1), clipped to ±3, so entry use stays
   balanced. This was the main capacity killer in the toys (§4.1, Zipf rows).
8. **Address drift:**
   - Every confirmed recall rewrites the path from the *current* entry front, so the entry → front-1 synapses follow
     the drifting thoughts.
   - Old entry links fade through step 6.
   - This plays the role of 25d's key renewal, with no anti-capture machinery, because each memory is spread over
     768 synapses and a single rewrite cannot drag another memory away.

### 6.5 Optional fast + long-term tiers (`--wm-tiers 2`, default off)

- **Two synapse sets on the same units:**
  - Wfast: as above, with λ ≈ 3e-3, life ≈ 300 steps;
  - Wlong: λ ≈ 5e-5.
- **Wlong is written only from memory-generated events, inside the step:**
  - (i) confirmed recalls of fast memories;
  - (ii) self-replay: R = 64 waves per step started at random entry patches and random entry units on Wfast. Waves
    with c_t ≥ 0.6 for t = 1..3 are written into Wlong (toy §4.5: 0 spurious).
- **Read:** run the wave on both sets in the same kernel (two nibble loads) and take, per pass, the set with the
  larger c_t.
- **Size:** 2 × 472 MB = 944 MB, within budget. Add this only if the single tier forgets rare facts (D-W4 below).

### 6.6 Optional order selectivity
Phase already gives it. Toy W5 decides whether `--wm-phase 3` is worth its 4× bytes over `--wm-phase 0` with
`--wm-units 128`.

### 6.7 Swap, removal, and per-expert training
- **Swap / remove:**
  - The memory is one module, `WaveMemory`, with `state_dict()` = §6.1, saved with `--wm-save` and loaded with
    `--wm-load`. Removing it = `--wm 0`: every g = 0, so xf is unchanged.
  - The brain is made robust to memory absence by `--wm-drop 0.1`: on 10% of micro-batches the whole memory is off.
    This is ordinary main-loss training of the brain and adds no memory-side loss.
  - Eval reports memory-on, memory-off and swapped-memory losses.
- **Train one expert's memory only** (`--wm-private 1`):
  - **Sheets:** every routed expert e (and the shared expert) owns a small private sheet: 8 × 8 hypercolumns ×
    16 units = 1,024 units, a 2 × 2-hypercolumn front, the full 8 × 8 window. Synapses: 1,024 × 64 × 16 nibbles =
    512 KB per expert, 385 MB for 769.
  - **Codebooks:** the content codebooks are shared (slices = 4 × 96-d at the private level; a separate C_priv
    [3, 4, 16, 8, 96]).
  - **Reads:** the private read uses the same xf and a fixed per-expert Lproj_e [64, 384] (generated from a seed,
    not stored). It is added for that expert only.
  - **Writes:** a private write happens when the token was routed to e (the same candidate rule, credit-limited as in
    25d: ≈ 0.26 writes per expert per micro-batch).
- **Per-expert training flags:**
  - `--wm-train-experts 17,40` allows writes only to those private sheets;
  - `--wm-freeze-global` stops global writes;
  - `--freeze-brain` (main optimiser off, backward still run to get ∂L/∂xf write targets). Memory-only training of
    one expert then needs no change to brain weights.
  - Splits: a child expert copies the parent's private sheet (512 KB), as 25d copied private blobs.
- **VRAM total:** global 472 MB + private 385 MB + ≈ 70 MB ≈ **0.93 GB** (≤ 1.9 GB).

### 6.8 Cost estimate (per optimiser step: 16,384 tokens × ≈ 12.8 core passes ≈ 210K reads)
- **Global pass 1:** 210K × 64 KB = 13 GB.
- **Passes 2-3:** only for the ≈ 20-40% of reads that did not exit: 2 × 0.3 × 13 GB ≈ 8 GB.
- **Total:** ≈ 21 GB of reads, **≈ 25-40 ms** at ≈ 0.6-0.9 TB/s effective. Complex MACs ≈ 210K × 16 × 8,192 × 1.6
  ≈ 4.4e10 complex MAC, < 5 ms.
- **Private sheets:** ≈ 6 per read × tiny, ≈ 10-20 ms.
- **Writes, forgetting, homeostasis:** < 3 ms.
- **Gate / decode:** < 5 ms.
- **Total memory:** **≈ 50-100 ms per step**, vs ≈ 1.27 s today. Phase 0 halves the bytes.
- **Kernels:** a Triton kernel (`wm_pass`) does gather + complex accumulate + patch scoring per read. A PyTorch
  reference (chunked einsum) is used only for unit tests and toys.
- **Host syncs:** none. Early exit is masked, with a static buffer.
- Uncertainty: these are bandwidth estimates. The real kernel, and L2 reuse between tokens hitting the same region,
  could move them 2× either way. All are far below the budget.

### 6.9 Flags (defaults)
`--wm 1 --wm-sheet 32 --wm-units 64 --wm-front 4 --wm-window 15 --wm-passes 3 --wm-phase 3 --wm-entries 1
--wm-fit-step 300 --wm-content-fit 300,500 --wm-early-exit 0.15 --wm-gate 1 --wm-gate-floor 0.05 --wm-explore 0.05
--wm-drop 0.1 --wm-write-w 32 --wm-write-per-seq 4 --wm-fam 0.5 --wm-phase-merge 0.5 --wm-fill 0.12 --wm-beta-dir 0.5
--wm-tiers 1 --wm-replay 64 --wm-private 0 --wm-private-sheet 8 --wm-private-units 16 --wm-train-experts all
--wm-freeze-global 0 --wm-load '' --wm-save ''`, with `--gm 0` (the blob store off).

Logged every 250 steps:
- fill, λ, entry-use Gini, hub share;
- c_t histograms and early-exit rate;
- gate open rate and g-vs-gain slope (exploration pairs);
- writes new / reconsolidated / depressed;
- per-memory survival of 4,096 tracked writes (does the cue still recall the path after n steps);
- the memory-off and swapped losses at eval.

### 6.10 Failure modes and what catches them
| risk | sign | response |
|---|---|---|
| paraphrase lands on another entry patch | W1 location agreement < 0.5; late-fact gain only on train templates | `--wm-entries 2`; coarser location (16×16 positions); if it is still low, no thought-keyed memory can help (25d has the same issue) |
| hub entry locations / skewed units | entry Gini > 0.5, fill hotspots | b_loc homeostasis; BCPNN-style division of |u| by √use (`--wm-usage-norm`) |
| content too coarse | cos(decode, target) < 0.75 in W3 | 8 phases × 128 units, or 4 fronts (`--wm-passes 4`; the cost is linear) |
| harm on other text | world_other ΔCE > +0.04 | gate floor; early-exit threshold; the familiarity AUROC is ≈ 1 in the toys, but real novel thoughts are not random |
| phase not worth it | phase-0 arm ≥ phase-3 arm at equal bytes | switch the default to `--wm-phase 0 --wm-units 128` |
| rare facts forgotten before recurrence | tracked survival at 1.3K steps < 0.5 | lower fill target or λ; `--wm-tiers 2` |

---------------------------------------------------------------------------------------------------------------------

## 7. Learning, change and capacity, in one place
- **Learning during training:** one-shot Hebbian / phasor writes at surprising positions, with gradient-derived
  content (the brain's own error signal at the read point, as 25d). This is a three-factor rule: pre (front t−1) ×
  post (front t) × modulator (surprise, which selects the write; reconsolidation, which selects the depression). In the
  fly mushroom body, dopamine writes and updates memories with cell-type-specific rules (Aso & Rubin 2016; Hige et al.
  2015).
- **Change:** recall followed by a correction rewrites the content units (reconsolidation). Links that are no longer
  used fade through homeostatic forgetting (bounded synapses / palimpsest: Amit & Fusi 1994; Fusi, Drew & Abbott 2005;
  Benna & Fusi 2016 for two-timescale synapses).
- **Capacity:**
  - ≈ 768 presence bits per memory out of 944M; at fill 0.12, ≈ 150K live memories (300K with `--wm-units 128` or
    fill 0.25, at lower noise tolerance). That is ≈ 1.5-3 KB of synapse per memory, against ≈ 5 KB per blob.
  - Toys at small scale: 16-32K memories in 21-63M bits with clean recall (0.04-0.12 bits of content per synapse bit).
  - Uncertainty: the scaling from the toys to the 32×32×64 geometry is theory (P ∝ 1/per-memory fill). W2 measures it.

---------------------------------------------------------------------------------------------------------------------

## 8. Validation plan

### 8.1 CPU checks before any build (≤ 2 threads, no GPU, read-only on runs)
| id | question | how | GO |
|---|---|---|---|
| **W1** | does the same fact in other wording light the same entry patch and units? | `runs\world_blob25d_12k\ckpt_6000.pt` on CPU (read-only), ≈ 300 fact prompts × 2-3 wordings / documents (the fact-cloze eval set), K = 6; xf at the cue token per pass → fit μ, σ, E, Lproj on 64K positions; measure location agreement (top-1 / top-2) and the share of the 16 entry units that agree | top-2 location agreement ≥ 0.6 and unit agreement ≥ 0.75 (the toys tolerate 25% wrong) |
| **W2** | full-geometry capacity with skewed entries + homeostasis, phase 3 vs phase 0 at equal bytes | a sparse reimplementation of toy_wave (window-local nibbles, no dense N²), 32×32×64, P up to 300K, entry locations from W1's real xf distribution | ≥ 150K memories with front-3 recall ≥ 0.9 at 25% entry noise |
| **W3** | content fidelity on real targets | blob contents and captured gradients from the 6K checkpoint (contents are in the gm state) → fit C, measure cos after 3 fronts | ≥ 0.8 |
| **W4** | survival under the in-step write / forget / reconsolidate stream | toy stream with Zipf recurrence (late ≈ every 105 steps, rare ≈ every 1.3K), 64 writes per step, λ controller | rare-fact recall at its next occurrence ≥ 0.5; with `--wm-tiers 2` it must improve, else drop tiers |
| **W5** | does phase add order selectivity that weights cannot? | pairs of memories on the same unit sets in a different order (X→Y vs Y→X); recall the second element | phase ≥ 0.9, weights-only ≈ 0.5 → phase earns its bytes |

### 8.2 GPU tests after the build (when the GPU is free)
- **T-unit:** reference vs Triton kernel equality; memory-off gives exactly the no-memory forward; swap test (load
  another state, the loss changes only through r); per-expert flag test (writes touch only that sheet).
- **T-speed:** tok/s with `--wm 1` vs `--wm 0` on the same harness, ≥ 3 runs. GO: memory ≤ 10% of step time (vs ≈ 58%
  today) and peak VRAM ≤ the blob run's −1 GB.
- **T-pilot (2K steps, phase 3 vs phase 0 + units 128):** fill, c_t, early-exit, gate slope, W1 repeated live.
- **Main run `world_wms_12k`:** same data, flags and steps as `world_blob25d_12k`, with `--gm 0 --wm 1`. Evals at 25 /
  50 / 75 / 100% and 7K, same harness as 25d §19:

| metric | GO |
|---|---|
| late facts (cloze exact, n = 5,000) | ≥ blob run at 7K and at the end (old bars: ≥ 0.27 @7K, ≥ 0.28 end) |
| rare facts | ≥ blob run at the same steps (old row run 0.016 @7K; 25d's GO asked ≥ 0.035) |
| held-out wording / documents (val_world templates) | gain ≥ 50% of the in-template gain; this is the T0b generalisation failure, the decisive one |
| harm on other world text | memory-on − memory-off ≤ +0.04 nats; Δ wiki ≤ 0.01 |
| brain without memory | memory-off loss ≤ the no-memory control + 0.02 (it must still work); swapped-memory eval reported |
| per-expert training | 500 steps of `--freeze-brain --wm-train-experts e` on 5 experts: facts routed to e improve, everything else ≤ ±0.005 nats |
| speed / VRAM | ≥ 1.8× the blob run's tok/s; peak ≤ 18.5 GB |
| early stop | at 7K, late < 0.24 or rare ≤ the old row run's 0.016 → stop and report |

### 8.3 Results of the pre-build checks (run 2026-10-25, CPU, 2 threads, ckpt_6000 read-only)

All scripts and outputs are in `C:\Eitan\Python\NewBrain\speed_wip\mem26\`.

**Captures.**
- `w1_capture.py` / `w1c_capture.py` → `cap\w1_6k.npz`, `cap\w1c_6k.npz`.
- Source: blob run `ckpt_6000.pt`, its own `code_snapshot`, no bytecode written, blob reads off.
- Documents: the T0a builder. 200 late + 200 early probe people × 5 relations. Per fact: 3 training-template
  documents, 1 held-out-template document and 1 cloze prompt.
- Size: 33,752 fact sentences and 2,000 cloze prompts, with xf, mu_1..4, write-back and pooled workspace for passes
  0-5.

**W1: paraphrase → the same entry?** (`w1_analyze.py` → `w1_results.txt`) **FAIL** for the §6.3 entry map.
Agreement of two views of the same fact, passes 1-5:

| pair | xf cos | location, both sides top-2: randproj 1024 / k-means 1024 / k-means 256 | entry units (16 slices × 64) |
|---|---|---|---|
| train template vs another train template | 0.83-0.86 | 0.58-0.62 / 0.82-0.84 / 0.87-0.90 | 0.41-0.45 |
| train vs held-out template | 0.71-0.76 | 0.43-0.47 / 0.61-0.65 / 0.67-0.70 | 0.30-0.34 |
| train document vs cloze cue | 0.60-0.61 | 0.29-0.30 / 0.47-0.49 / 0.53-0.59 | 0.22 |
| control: same person, other relation | 0.65-0.75 | 0.28-0.41 / 0.50-0.63 / 0.57-0.74 | 0.22-0.31 |
| control: same template, other person | 0.20-0.25 | 0.02-0.03 / 0.19-0.26 / 0.35-0.53 | 0.06-0.08 |

- At pass 0 the template dominates (same template, other person: cos 0.63).
- Reading at a different pass than the write costs little: train@5 vs cloze@2 is the same as same-pass. Reading at
  pass 0 is poor.
- Coarser or soft unit codes (`w1_variants.py` → `w1b_variants.txt`) raise agreement, but they raise the
  same-person-other-relation control by the same amount. Example, 4 codes per slice: same fact 0.53, control 0.56.
- So in xf's cosine geometry, the cloze cue is about as close to the same person's other facts as to the asked fact.

**Relation probe** (linear ridge, split by person, chance 0.20, `wbench.py`):

| feature | accuracy |
|---|---|
| xf | 0.990 |
| mu_1..4 (previous pass) | 0.984 |
| workspace p | 0.988 |
| write-back | 0.959 |

The relation *is* in xf, linearly. It is just not dominant in xf's cosine geometry.

**W-bench** (`wbench.py`, reusable `Bench` class; all passes in `wbench_final.txt`; extra variants in
`wbench_extra_k3.txt`, `wbench_quick.txt`).
- Stored: 25,426 sentences = 15,109 distinct facts (probe + filler), from training-template documents, one entry per
  pass.
- Read: 2,000 held-out-template cues and 2,000 cloze cues.
- Metric: the right (person, relation) at top-1 / top-5. Cue variants are scored as idealised code-match recall.
- Mean over passes 1-5:

| method | bytes per entry | held top-1 / top-5 | cloze top-1 / top-5 |
|---|---|---|---|
| exact cosine nearest neighbour, xf 384-d (ceiling) | 768 | 0.778 / 0.942 | **0.649** / 0.825 |
| 25d blob address (fixed 64-d projection, L2 nearest) | 128 | 0.686 / 0.780 | 0.499 / 0.640 |
| (a) random projection, 1024 locations + 16×16 units (the §6.3 default) | 9 | 0.417 / 0.632 | 0.243 / 0.410 |
| (b) online SOM 16×16 location + 16×16 units | 9 | 0.394 / 0.595 | 0.202 / 0.360 |
| (d) soft top-2 units, k-means 256 + 16×16 | 9 | 0.454 / 0.677 | 0.270 / 0.464 |
| **(c+d) coarse-to-fine: SOM 256 + 3 residual levels × 32 slices × 16 codes, top-2** | **49** | **0.684 / 0.864** | **0.524 / 0.715** |

At pass 3 only:

| method | held top-1 | cloze top-1 |
|---|---|---|
| (c) k-means location + 2 levels | 0.50 | 0.33 |
| (c) SOM location + 2 levels | 0.55 | 0.37 |
| (c+d) SOM + 3 levels × 16 slices | 0.63 | 0.49 |
| (c+d) SOM + 3 levels × 32 slices | 0.71 | 0.55 |
| codes on PCA-whitened xf | | 0.21-0.35 (worse) |
| exact NN on PCA-whitened xf, 128 dims | 0.76 | 0.63 |

**Concepts as cue** (exact cosine, pass 3, cloze top-1; addressing from concepts is allowed):

| cue | cloze top-1 | held top-1 |
|---|---|---|
| xf + previous write-back | 0.64 | 0.81 (vs 0.78 for xf alone) |
| xf + mu_1..4 | 0.59 | |
| xf + pooled workspace | 0.56 | |
| mu_1..4 alone | 0.10 | |
| coarse-to-fine on xf + mu | 0.10 | |

**Concepts do not improve the cue. The cue stays xf only.**

**W-bench for a full module** (`wbench_module.py`, `Harness` class).
- It drives a `memory_api.MemoryBase` module, imported read-only from `fruit-fly-sim-wave` (no bytecode written).
- Writes go through the module's real write path (`bench_write(xf, target, e_k, t_k, call)` or `write(xf, target=…)`)
  into its fixed-size store. Each fact's target is a fixed random content vector at the 25d content scale; the
  labels are never shown to the module.
- Queries go through the module's real `read()` (r_shared, or the mean of r_pairs over the captured routing). Score:
  cosine(r, content) → top-1 / top-5, against `byte_budget()` bytes.
- Reference adapter `ExactNN` (FIFO store of xf + content):

| store | size | held top-1 | cloze top-1 |
|---|---|---|---|
| cap 32,768 | 50.3 MB | 0.776 | 0.655 |
| cap 8,192 (FIFO overflow) | 12.6 MB | 0.252 | 0.210 |

The first row equals the idealised ceiling, so the harness is correct. A module that returns one content vector has
top-5 = top-1; blended reads give a meaningful top-5. Cloze rows have no captured routing, so they get the shared
read only.

**W2: full-geometry capacity** (`w2_sheet.py`).
- Setup: 32×32 torus, 15×15 window, 4×4 fronts, entry + 3 passes, uniform entry locations, entry location given
  exactly, entry units corrupted. Measures: path / front-3 unit recall.
- (A first run had a bug: the 8 directions were not in angular order, so ±1 "turns" could reverse and paths
  overlapped themselves. Fixed; numbers below are after the fix.)

| arm (synapse store) | P | fill | 0% wrong | 25% wrong | 50% wrong |
|---|---|---|---|---|---|
| phase 3, M = 64 (472 MB) | 150K | 0.11 | 1.000 | 1.000 | 0.997 |
| same | 300K | 0.21 | 1.000 | 1.000 (phase 0.999) | 0.977 |
| same | 450K | 0.30 | 1.000 | 0.996 (phase 0.993) | 0.90 |
| weights only, M = 128 (472 MB, equal bytes) | 300K | 0.06 | 1.000 | 1.000 | 0.997 |
| same | 600K | 0.11 | 1.000 | 1.000 | 0.95 |

- **GO (≥ 150K at ≥ 0.9 with 25% noise): passed** in both arms, with large margin.
- At equal bytes, weights only holds about 1.3-2× more. But it stores 336 instead of 432 content bits per memory and
  has lower content fidelity (W3).
- Caveat: these are independent random codes with the location given exactly. W1 says real cues miss the location
  in 10-50% of cases.

**W3: content fidelity on real contents** (`w3_content.py`).
- Data: the live blob values of ckpt_6000 (127K global + 54K private). They are spread out: the top 32 principal
  components hold 25% of the variance.
- Residual code over 3 fronts, cos(reconstruction, content):
  - phase 0, 64 units: 0.51 / 0.65 / **0.75** (p10 0.72);
  - phase 3, 64 units × 8 phases: 0.61 / 0.76 / **0.85** (p10 0.83).
- **GO (≥ 0.8): passed only with phase 3.**

**W5: order selectivity** (`w5_order.py`). Pairs of memories on the same cue units, differing only in cue firing
order:

| P | weights only, paired | phasor, paired | both, unpaired |
|---|---|---|---|
| 2K | 0.49 | 1.000 | 1.00 |
| 8K | 0.52 | 1.000 | 1.00 |
| 16K | 0.49 | 1.000 | 0.96-1.00 |

Weights only sit at chance (0.5) on paired memories; phase earns its bytes for order. Caveat: in the brain, order is
largely already in xf (the SSD context).

**W4: survival under the in-step stream** (`w4_stream.py`, 16×16×64 sheet, 1/4 scale): results in §8.4.

**What changes in the design (§6) because of these checks:**
1. **Entry location:** an online SOM-like map of the thought (16 × 16 = 256 prototypes on the sheet torus;
   competitive + neighbour update and usage homeostasis, inside the step, memory-side). It replaces the fixed random
   projection, which scored 0.24 cloze.
2. **Coarse-to-fine waves:**
   - Pass 1 enters at the location (≈ the person).
   - Passes 2 and 3 are driven, besides the synapses, by the thought's **residual codes**: x̂ − prototype, then the
     level-2 and level-3 residuals; 32 slices × 16 codes, soft top-2.
   - Each hypercolumn gets 16 *key units* (driven by the thought's code) next to its 64 *content units*.
   - A unit fires when the synaptic wave and the thought's drive coincide (in phase).
   - This is the best measured cue: 0.52 cloze, 0.68 held, above the 25d address at 0.50 / 0.69.
3. **Phase stays on (`--wm-phase 3`):** W3 fidelity 0.85 vs 0.75, and W5 order. The weights-only arm stays as the
   per-byte control.
4. **No concepts in the cue.**
5. **Ceiling:** even exact 384-d nearest neighbour gets only 0.65 cloze top-1 on held-out wording. That bounds every
   memory keyed by this brain's thought at ckpt 6K. A WMS can at best approach it.

### 8.4 W4 results (`w4_stream.py`, 16×16×64 sheet = 1/4 of the full synapses, 16 writes per step)

**Setup.**
- Recurring facts: 500 late (every ≈ 105 steps, Poisson) and 500 rare (every ≈ 1,300 steps). The rest of the writes
  are junk.
- On each recurrence: first a recall from a noisy cue, then a rewrite on the same path (reconsolidation).
- Forgetting: homeostatic random clearing, targeting fill 0.12.
- Cue noise: entry location fails with probability `loc-fail`; each entry unit is wrong with probability
  `unit-wrong`.
  - "measured": loc-fail 0.12, unit-wrong 0.58 (W1 train-vs-train: k-means 256 top-2 location 0.88, hard units 0.42);
  - "optimistic": loc-fail 0.10, unit-wrong 0.25.

| run | late recall | rare recall | rare at gap > 1.4-1.7K |
|---|---|---|---|
| optimistic, 5K steps (fill still rising, forgetting weak) | 0.90 | 0.89 | 0.87 |
| measured, 5K steps (same caveat) | 0.80 | 0.72 | 0.58 |
| **measured, 12K steps (steady state: fill 0.13-0.15, λ ≈ 3.5-4e-4)** | **0.69-0.72** | **0.29-0.32** at steady state (0.49 over the run) | **0.23** |
| optimistic + two tiers (fast + long-term, in-step self-replay), 5K | 0.90 | 0.73 | 0.45 |

**Reading:**
- Late facts survive: about 0.7 recall from cues measured on real thoughts. The location failures alone cap recall
  at about 0.88.
- **Rare facts do not survive at steady state:** about 0.3 at their next exposure, below the W4 GO of 0.5. Uniform
  random forgetting at the rate that holds fill erases a 3-transition path in about 1-2K steps.
- The two-tier option made rare facts *worse* (0.45 at long gaps). The fast tier forgets them before a replay or
  confirmed recall copies them, and spontaneous replay rarely reaches a once-seen memory. **Drop the tiers.**

**Fixes to test before a build** (same toy, minutes each):
- forget by use instead of uniformly: protect synapses on confirmed paths with a 2-bit counter;
- lower the fill target with the full-size sheet (the full sheet has 4× the synapses for 4× the writes, so the same
  fill; only a larger M or a lower write rate helps);
- write rare (high-surprise) facts twice (two entry locations) so either path can survive.

Without one of these, rare facts are the WMS's weak point, as they were for the blob store.

**W4b: the fixes, tested** (`w4b_stream.py` → `w4b_variants.txt`).
- Same stream, measured cue noise, 12K steps, recall at the next exposure over the last 6K steps (steady state).
- **Equal bytes: 29 MB** at 1/4 scale. 1-bit synapses use M = 64; 2-bit counters use M = 45 (synapses ∝ M²).

| variant | late | rare | rare, gap > 1.3K |
|---|---|---|---|
| 1-bit, fill 0.12 (baseline) | 0.67 | 0.37 | 0.18 |
| 1-bit, fill 0.06 | 0.84 | 0.44 | 0.01 |
| 1-bit, write twice on surprise | 0.84 | 0.52 | 0.26 |
| 2-bit counters, random decrement | 0.77 | 0.30 | 0.00 |
| 2-bit + **forget by use** (confirmed recall increments its path) | 0.80 | 0.53 | 0.28 |
| 2-bit + forget by use + **surprise strength 2** | 0.80 | 0.66 | 0.45 |
| **2-bit + forget by use + surprise 2 + write twice (2 entry locations)** | **0.96** | **0.81** | **0.60** |
| same, both entry locations failing together (correlated, pessimistic) | 0.88 | 0.76 | 0.59 |
| 2-bit + forget by use + surprise 2, fill 0.06 | 0.86 | 0.55 | 0.13 |

**What each piece does:**
- **Forget by use and surprise strength** are what make rare facts survive. A synapse on a confirmed or surprising
  path needs 2-3 random decrements to vanish, while junk fades after one.
- **Writing twice** (a second path from the thought's second entry location, written only after a failed recall)
  mostly protects against entry-location failures.
- **A lower fill target** helps late facts but hurts long-gap rare facts: it means faster forgetting.

**W4 GO (rare ≥ 0.5) is met by the combined variant**, also with correlated entry failures.

**Adopted in §6 (supersedes §6.4 steps 5-6):**
- Wsyn becomes 2-bit counters + 3-bit phase: 5 bits per synapse, i.e. 5/4 of the 472 MB, or M = 57 at an equal
  472 MB.
- Write: new synapse count = 1, or 2 after a failed recall (surprise).
- A confirmed recall increments its path (cap 3).
- Forgetting: random decrement at the controller's λ (fill 0.12).
- On a failed recall, also write a second path from the second-best SOM location (`--wm-entries 2`). The reader
  tries both.

**Caveat:** this is a 1/4-scale toy with independent random codes and cue noise taken from W1. The gain from writing
twice depends on how often the two entry locations fail together on real thoughts; the correlated row bounds it.

### 8.4b Owner decisions on WaveSheet v2 (Eitan, 2026-10-25 08:06 UTC)
1. GO on WaveSheet, although its density per memory is about the same as the blobs'.
2. Rule 5 (no row tables) is satisfied by all three of:
   - gathering the synapse rows of the key neurons that fired;
   - the best-match content write;
   - the fixed 16-state efficacy LUT.
3. Regular training trains EVERYTHING (all memory parts and the brain). Training one expert's memory alone is a later
   fine-tune mode only. The global sheet is the default; private per-expert parts stay behind a flag, OFF, until he
   answers "global only or private too".
4. The wave stays a coarse-to-fine sweep at a fixed place. No travelling wave now; the hop arm (v1.1) is the maximum
   next step and stays unbuilt.
5. Read at pass 3 for every token; each expert's gate decides how much to use it, at passes 3-5. `--wave-intent`
   stays 1.0 (off).

### 8.4c Owner decisions, second round (Eitan, 2026-10-25 08:13 and 08:22 UTC)
- **Private memory per expert is ON.** The global sheet is small and only for common things; the expert sheets carry
  the specialised memories.
- **Size** (the coordinator's version, given to Eitan at 08:25 UTC):
  - global sheet: 256 places (16×16 torus), about 0.11 GB;
  - private sheets: 4 places per expert × 769, about 1.35 GB, with a small within-expert place map (4 prototypes per
    expert) and load-biased allocation;
  - per-expert output layers W_e: about 1.81 GB with AdamW and gradients;
  - VRAM target: peak about 20.1 GB = brain about 15.6 + sheets 1.47 + W_e 1.81 + read transients about 1.2.
  - The guard line is 21 GB; if the peak is over, shrink the private sheets first.
- **Pre-read gate:** a learned "need memory?" decision per (token, expert), placed before the read. If no, the read is
  not computed and the output is 0. The global read is computed if any of the token's experts wants it, with a fixed
  cap (≤ half the pairs). The gate trains on about 5% exploration reads, the help signal and confirmed recalls.
- **Output layer:** each expert has its own 384→384 layer. It is linear, with no activation (the output is added to
  the signed expert input), scaled by the memory's confidence, with a small scale at init. It is part of the memory:
  saved in the memory file, stepped by the memory's AdamW, swappable.
- **Writers:** each written memory goes to the private sheets of the token's TOP-2 routed experts, plus the global
  sheet. The global copy fades in about 1K steps unless it is confirmed.
  - Switch to top-1 if P(the fact's top-1 expert is in the cue's top-6) ≥ 0.9.
  - Report P(top-1 in top-6), P(top-1 or top-2 in top-6) and P(top-1 == top-1) for held and cloze cues.
  - The pre-read gate cap (≤ half the pairs), with about 5% exploration, was OK'd by Eitan.
  - The full design delta (v2.1) comes from the cloud as review input.
- **Splits:** a child copies its parent's private memory.

### 8.5 First experiment after the offline checks and G1/G2: a small brain + memory (Eitan's idea, 2026-10-25)
Runs only with Eitan's typed go, and only after the wave memory passes Stage A, the U/T tests and G1/G2.

**Why:** r7's big brain (768 experts) learned the facts in its own weights, so blob reads changed almost nothing
(memory-off late cloze 0.494 vs on 0.496). A brain with less capacity should have to rely on the memory, which also
makes it the cleanest test of whether a memory works. RETRO (Borgeaud et al. 2022) and memory layers (Berges et al.
2024) both found the smaller the main network, the more the memory gets used.

**Runs:**
- Data and steps as always: wiki_world_mix, 12,207 steps, r7's flags apart from size and memory.
- Small brain (at most 192 or 384 experts) with each of `--memory none`, `wave` and `blob`.
- The full-size brain with `--memory wave`.

**Judged on each run by memory on vs off on the same checkpoint:**
- late, early/rare and held-out-wording facts;
- loss (world and Wikipedia val);
- tok/s.

**What would show the idea works:** the memory helps the small brain clearly (a large on-vs-off gap), and the small
brain plus wave gets close to the full-size brain.

**Caveat:** the brain must stay large enough to form good thoughts (they are the memory's address), understand the
question and decode the recalled answer, so the brain should be small but not tiny.

---------------------------------------------------------------------------------------------------------------------

## 9. Where I am unsure
- **Entry-location robustness to wording (W1)** decides everything, as for 25d. No synaptic trick fixes an address
  that moves.
- **Toy-to-full scaling** (uniform random patterns vs real thoughts): the toys show skew destroys Willshaw capacity and
  BCPNN / homeostasis restores most of it. Real correlations are not Zipf on independent columns.
- **Content as the gradient direction:** a single 3-level code of a one-shot gradient is cruder than an
  Adagrad-refined blob content. Reconsolidation refines it only when the fact recurs.
- **Kernel speed estimates** are bandwidth arithmetic, not measurements.
- **Phase:** per byte it lost in the toys. It is kept by owner direction behind a flag, and the pilot decides.
- **The "row table" reading** of event-driven synapse gathers is the owner's call (§1).

---------------------------------------------------------------------------------------------------------------------

## 10. References (titles / authors / years; from memory, not re-verified online)
- Willshaw, Buneman & Longuet-Higgins 1969, *Non-holographic associative memory* (Nature).
- Palm 1980, *On associative memory* (Biol. Cybern.).
- Amit, Gutfreund & Sompolinsky 1985, *Storing infinite numbers of patterns in a spin-glass model of neural networks*.
- Tsodyks & Feigel'man 1988, *The enhanced storage capacity in neural networks with low activity level*.
- Kanerva 1988, *Sparse Distributed Memory* (MIT Press). Bose, Furber & Shapiro 2005, *An associative memory for the
  on-line recognition and prediction of temporal sequences* (N-of-M spiking SDM).
- Lansner & Ekeberg 1989 (BCPNN); Lansner 2009 (Trends Neurosci.); Ravichandran, Lansner & Herman 2020-2023 (BCPNN
  networks).
- Amit & Fusi 1994, *Learning in neural networks with material synapses*. Fusi, Drew & Abbott 2005, *Cascade models of
  synaptically stored memories*. Benna & Fusi 2016, *Computational principles of synaptic memory consolidation*.
- Hopfield 1982. Krotov & Hopfield 2016, *Dense associative memory*. Demircigil et al. 2017. Ramsauer et al. 2020,
  *Hopfield networks is all you need*.
- Noest 1988, *Phasor neural networks*. Frady & Sommer 2019, *Robust computation with rhythmic spike patterns* (PNAS,
  TPAM). Plate 1995 / 2003, *Holographic reduced representations*. Frady, Kent, Olshausen & Sommer 2020, *Resonator
  networks*. Löwe et al. 2022 / 2023 (complex autoencoder; rotating features).
- Izhikevich 2006, *Polychronization: computation with spikes*. Bohte, Kok & La Poutré 2002, *SpikeProp*. Gütig &
  Sompolinsky 2006, *The tempotron*. Hammouamri, Khalfaoui-Hassani & Masquelier 2024, *Learning delays in SNNs using
  dilated convolutions with learnable spacings*.
- Abeles 1991 (synfire chains); Diesmann, Gewaltig & Aertsen 1999, *Stable propagation of synchronous spiking*. Muller,
  Chavane, Reynolds & Sejnowski 2018, *Cortical travelling waves* (Nat. Rev. Neurosci.). Keller, Muller, Sejnowski &
  Welling 2024, *Traveling waves encode the recent past and enhance sequence learning* (ICLR).
- Dasgupta, Stevens & Navlakha 2017, *A neural algorithm for a fundamental computing problem* (fly hashing). Hige,
  Aso, Modi, Rubin & Turner 2015 (heterosynaptic plasticity in the MB). Aso & Rubin 2016, *Dopaminergic neurons write
  and update memories with cell-type-specific rules*.
- Hinton & Plaut 1987, *Using fast weights to deblur old memories*. Ba, Hinton, Mnih, Leibo & Ionescu 2016. Schlag,
  Irie & Schmidhuber 2021. Yang et al. 2024 (DeltaNet parallel); Yang, Kautz & Hatamizadeh 2025 (Gated DeltaNet).
  Behrouz, Zhong & Mirrokni 2025, *Titans*. Miconi, Clune & Stanley 2018; Miconi et al. 2019 (*Backpropamine*).
- McClelland, McNaughton & O'Reilly 1995 (CLS); Kumaran, Hassabis & McClelland 2016; Buzsáki 2015 (sharp-wave
  ripples).
- Lample et al. 2019, *Large memory layers with product keys*; Berges et al. 2024, *Memory layers at scale* (rejected:
  row tables). Kaiser, Nachum, Roy & Bengio 2017, *Learning to remember rare events*. Wu, Wayne, Graves & Lillicrap 2018,
  *The Kanerva Machine*; Das et al. 2024, *Larimar*.
- Allen-Zhu & Li 2024, *Physics of language models part 3.3: knowledge capacity scaling laws*.
