# 24. Is plan 21's consolidation life cycle the right approach? Review + offline toy (CPU only, no code) — 2026-10-24

Follows plans 20 / 21 (the life cycle) and 23 (the splat / kNN memory). Script `speed_wip/splat/consol_toy.py`;
results in `res_consol.json` / `log_consol.txt` (main drift) and `consol_harsh/` (harsh drift). CPU, 4 threads.

Structure (for the merge with the cloud study): §1 the critique · §2 alternatives + literature · §3 keys under drift
(splat / kNN) · §4 the toy · §5 recommendation + cost · §6 risks.

## TL;DR
1. **The one-time global switch protects the wrong thing.**
   - What fails under drift is not the content and not the addressing parameters. It is the *match* between a stored
     address and where its context's thought is now.
   - Freezing the addresses at a global switch does not help that match. In the key-memory toy, the plan-21 cycle (a)
     is indistinguishable from simple constant fading (d): 0.78 vs 0.77 early-fact retention under constant drift,
     0.89 vs 0.90 when the drift slows at the LR decay.
   - At full LR, the stability trigger (≥ 0.9) never fires, so the switch is in practice a fixed time (the LR-decay
     start).
2. **Per-memory consolidation by reuse (synaptic tagging / maturity) and a Benna-Fusi cascade did not help either**
   (0.71 / 0.71). Under a tight slot budget they slightly hurt the newest facts (0.89-0.91 vs 1.00): mature but stale
   keys keep slots that new facts need. Their real strength, protecting useful content from being forgotten, is not
   the binding problem under drift.
3. **What works is moving the memory's key to where its context's thought is now:**
   - **re-encode on a confirmed recall** (reconsolidation: when a read is confirmed useful, the key ← the current
     thought): early facts **0.92** (vs 0.78), all useful facts 0.94, new facts 1.00;
   - **replay / "sleep" re-registration** (periodically recompute the keys of mature memories from their stored source
     contexts with the current network): **1.00** under the main drift, and the only mechanism that survives the harsh
     drift (0.91-0.97 vs 0.25-0.58 for everything else).
4. **Facts seen only once are lost by every mechanism after a few intervals of drift** (early one-shot recall 0.00).
   Only re-registration can keep them, and only if they are kept long enough to be re-registered.
5. **Recommendation** for the splat / kNN memory:
   - no global switch;
   - constant mild fading for slot turnover;
   - **re-encode-on-confirmed-recall as the default** (≈ 70 ops per step, < 3 ms; fits both speed configs of plan 23);
   - **replay re-registration** through 1 replay window per micro-batch (≈ 2-6% throughput, 8 B per splat for the
     source pointer), the strongest mechanism;
   - per-memory maturity only as value protection (the Adagrad accumulator), not for keys;
   - keep plan 21's LR-decay coupling only for the row pool if rows stay.

---

## 1. Critique of the current mechanism (plan 21; memory.py `LifeCycle`)
**What it does.**
- Infancy: rows fade (λ = 1e-3 per step, lazy Λ) and the row-Adagrad accumulator fades (β 0.99, lazy Ψ).
- The switch fires at the first of: probe stability ≥ 0.9 on 3 consecutive probe evals, the LR-decay start (80%), or
  a cap (90%), and only once births are done.
- A linear ramp (5% of the run), then mature: no fade, Adagrad, and the addressing (keys / W_q / norm) frozen.

**1.1 A one-time global switch while the thoughts keep drifting.**
- The measured drift is 72% relative change and 40% 10-NN survival per 1,000 steps, and the probe stability is flat
  at 0.39-0.47 from step 2,500 to 7,000 (plan 20). So trigger (a) cannot fire at full LR, and the switch is the LR
  decay in practice.
- That is the right *time* (the drift does slow with the LR), but a global switch assumes that after it, addresses
  stay valid. They don't: whatever drift remains after the switch is not compensated, and nothing re-maps a frozen
  address.
- The toy's "LR decay" scenario (drift × 0.3 after 80%) shows the part that is recovered (0.78 → 0.89) and the part
  that is not.

**1.2 Memories written just before the freeze.**
- At the freeze, the rows hold the content of the last ~700 steps (λ's half-life) at addresses that the thoughts are
  still leaving.
- Plan 21 itself notes that at the fixed 7,325-step freeze about half of the rows written before it were orphaned
  within 500 steps.
- After the switch, Adagrad makes them *harder* to rewrite (1/√S): the most recent infancy content is consolidated
  exactly when it is least likely to be read again at its address.
- A per-memory rule (consolidate what is re-used successfully) would not consolidate them at all until they are hit
  again.

**1.3 Is the global fade too blunt?**
- Yes in principle: one λ forgets useful and useless memories alike.
- But the toy says this is not the binding problem under drift. Keeping useful memories alive longer (maturity,
  cascade) does not raise their recall (0.71 vs 0.78), because a kept but stale key is not found by the drifted
  query.
- Selectivity pays only together with key renewal (§4, b_re, e).
- For rows the bluntness is worse than for keys. A row is shared by many contexts, so "useful" is not even defined
  per row. That is an argument for per-memory items (keys / splats) rather than rows.

**1.4 What plan 21 gets right.**
- Fading during the fast-drift phase: stale content should go, and it recycles capacity.
- The decaying accumulator: memories stay rewritable.
- The coupling to the LR schedule.
- Dead-key restarts with band reset.

Those carry over. The freeze and the global switch are the parts to drop.

## 2. Alternatives, with the literature
1. **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly 1995, Psychological Review; Kumaran,
   Hassabis & McClelland 2016, TICS).
   - A fast hippocampal store for episodes; slow interleaved replay into the neocortical weights.
   - Mapping: the splat / kNN pool = the fast store; the brain's weights = the slow learner.
   - **Replay windows in the training batch** (a stored context fed again) are CLS replay. As a side effect they
     re-register the fast store's keys (§2.6).
2. **Per-memory consolidation: synaptic tagging and capture** (Frey & Morris 1997, Nature); spaced re-use (Cepeda et
   al. 2006).
   - Each memory has its own age / strength / use count. It consolidates when it is re-used successfully (spaced), and
     fades when unused.
   - Toy policy (b): maturity +1 per spaced successful re-use, fade λ₀·4^{−m}.
3. **Multi-timescale cascade synapses** (Fusi, Drew & Abbott 2005, Neuron; Benna & Fusi 2016, Nature Neuroscience).
   - Each memory is a chain of coupled variables (fast → slow) that hand off to each other, giving power-law
     forgetting and a large capacity without a switch.
   - Toy policy (c): u₁, u₂, u₃ with capacities 1 / 4 / 16, conductances 0.25 / 0.0625, leak 0.0156; strength = u₁.
4. **Reconsolidation / re-encoding on recall** (Nader, Schafe & LeDoux 2000, Nature: a recalled memory becomes labile
   and is re-stored).
   - Computational form: when a read is confirmed useful, re-write the memory at the CURRENT thought, so its key
     tracks the drift. Policy (b_re).
   - Variants:
     - add a key instead of moving it (b_tr, "trail");
     - re-encode on confidence alone without confirmation (b_un).
5. **Continual plasticity with no freeze.**
   - Representational drift is normal in the brain (Driscoll et al. 2017, Cell; Rule, O'Leary & Harvey 2019, Curr.
     Opin. Neurobiol.). Readouts stay valid because they are continually re-learned, not because representations are
     frozen.
   - Loss of plasticity is the known failure of "learn then freeze" in deep nets (Dohare et al. 2024, Nature,
     continual backprop).
   - Mapping: no global freeze; the memory's addresses are maintained continuously by re-encoding / re-registration.
6. **Replay / "sleep" re-registration.**
   - Hippocampal replay in sleep (Wilson & McNaughton 1994, Science); synaptic homeostasis (Tononi & Cirelli 2014,
     Neuron).
   - **The ML precedent: REALM's asynchronous index refresh** (Guu et al. 2020,
     [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)): the retrieval keys of the whole corpus are re-embedded with
     the current encoder every few hundred steps, precisely because stale keys break retrieval (also noted in
     Memorizing Transformers, Wu et al. 2022).
   - For us: store a source pointer (document, offset) per splat, and periodically recompute the key with the current
     brain. Policy (e).
7. **Importance-weighted protection of the values** (EWC, Kirkpatrick et al. 2017,
   [arXiv:1612.00796](https://arxiv.org/abs/1612.00796); Synaptic Intelligence, Zenke, Poole & Ganguli 2017,
   [arXiv:1703.04200](https://arxiv.org/abs/1703.04200); MAS, Aljundi et al. 2018,
   [arXiv:1711.09601](https://arxiv.org/abs/1711.09601)).
   - For the memory values, the row-Adagrad accumulator S_r is already an importance estimate (Σ g²), and 1/√S_r is
     EWC-like protection.
   - Plan 21's decaying accumulator (β 0.99) deliberately removes that protection in infancy.
   - Per-memory maturity is the natural switch for it: immature memory → discounted S (plastic); mature → accumulating
     S (protected).
   - This protects *content*; it does nothing for key staleness (not tested in the toy; values are fixed there).
8. **Generative / experience replay in continual learning** (Shin et al. 2017,
   [arXiv:1705.08690](https://arxiv.org/abs/1705.08690); GEM, Lopez-Paz & Ranzato 2017,
   [arXiv:1706.08840](https://arxiv.org/abs/1706.08840)). Replay protects the weights. Here it would also refresh the
   memory keys (item 6).

## 3. Keys of a splat / kNN memory under drift
- **The problem**: a key is the write-time thought. After drift, the same context's thought has moved by δ; recall
  survives while δ < the spacing to the nearest other key (plan 23 §2-3).
  - Fading and capacity decide which keys exist. Only re-encoding decides whether they are still *in the right place*.
- **Options for keeping a key valid**:
  1. **Re-encode on confirmed access** (b_re): key ← the current thought when the read was confirmed. This is the
     cheapest, and it renews exactly the memories in use.
     - "Confirmed" in the brain: the gate was open and the read helped (the delta target |v* − r| small, or the loss
       dropped vs the gate-closed counterfactual).
     - Unconfirmed (confidence-only) re-encoding works almost as well when the drift slows (0.94), but it can hijack:
       93-95 wrong re-encodes out of about 45K in the toy (0.2%).
  2. **A slow key per splat = the EMA of the thoughts that read it successfully.** This is (1) with η < 1. It is more
     robust to a wrong confirmation, but it lags the drift. Not separately tested (η = 1 was used); with re-encounters
     about once per interval, the lag of η = 0.5 would cost about half the gain.
  3. **Trail keys** (a confirmed recall *adds* a key): about 4.6 keys per useful fact. It costs capacity; good only when
     slots are plentiful (0.89 at a 20K budget, 0.56 at 6K).
  4. **Re-registration from the source context** (e): the only option that does not need the context to recur. It
     needs the context (a pointer into the training data) and a forward pass.
- **Interaction with fading and capacity**:
  - With re-encoding, a memory's age should count from its last *re-encode*, not its birth. A re-encoded memory is as
    good as new; the fade clock restarts.
  - Capacity then holds: the facts in use (renewed) + the recent facts (not yet faded) + the mature facts (if
    re-registered).
  - Without renewal, maturity / cascade keep stale keys alive and waste slots: in the tight budget, the newest
    one-shot facts drop to 0.75-0.79.

## 4. The toy (`consol_toy.py`)
**World.**
- 8 drift intervals × 4 steps (1 step ≈ 250 training steps, 1 interval ≈ 1,000 steps).
- 400 new facts per step (12,800 in all). 30% are "useful": each step they are re-encountered with probability 0.25
  (≈ once per interval), at their CURRENT thought, with their value. The rest are seen once.
- Memory: one key per item in the frozen whitened-32 space (fitted at t = 0); the read is the nearest key (opacity
  only breaks ties: plan 23 showed that blending or opacity-weighting neighbours hurts); slot budget 6,000 (tight) or
  20,000.
- Base rule for every policy: spawn new facts; on a re-encounter, read; if correct, reinforce (α + 1); otherwise spawn
  a new key at the current thought. Fading λ₀ = 0.25 per step = plan 21's 1e-3 per training step; prune at α < 0.05;
  evict the weakest when over budget.

**Drift.**
- A stationary warp process: a ← normalise(a + amp·F_k(a) + σ·noise), with a fresh smooth field each step.
- **Main calibration: by its effect on addressing.** One interval leaves a frozen product-key row overlap of 0.21
  (S = 512, as measured in plans 18 / 20). That implies 45% relative change and 56% 10-NN survival per interval.
- Two scenarios: the drift stays constant, or it drops to ×0.3 at 80% (the LR decay).
- A 4-step process cannot match both 72% and 0.21. At 72% relative change, the row overlap is ≤ 0.05 at every field
  length scale. That **harsh** variant is in `consol_harsh/`, reported below.

**Policies.** (a) plan 21 global; (b) per-memory maturity; (b_re) b + re-encode on confirmed recall;
(b_tr) b + trail keys; (b_un) b + confidence-only re-encode; (c) Benna-Fusi cascade; (d) constant mild fade λ₀/2;
(e) b + sleep re-registration of mature items at every interval end; (f) no fade.

### 4.1 Main drift, tight budget (6,000 slots for 12,800 facts): the final state after 8 intervals

| policy | early useful (born in interval 1) | all useful | newest useful / one-shot | early one-shot | confident-wrong reads | keys per useful fact | key writes per step (toy) |
|---|---|---|---|---|---|---|---|
| (a) plan 21 global switch | 0.78 | 0.81 | 1.00 / 1.00 | 0.00 | 0.001 | 1.09 | 463 |
| (b) per-memory maturity | 0.71 | 0.73 | 0.91 / 0.79 | 0.00 | 0.001 | 1.32 | 465 |
| **(b_re) + re-encode on confirmed recall** | **0.92** | **0.94** | 1.00 / 1.00 | 0.00 | 0.001 | 1.06 | 871 |
| (b_tr) + trail keys | 0.56 | 0.58 | 0.54 / 0.25 | 0.00 | 0.001 | 1.49 | 1,233 |
| (b_un) + confidence-only re-encode | 0.78 | 0.89 | 1.00 / 1.00 | 0.00 | 0.003 (93 hijacks) | 1.19 | 1,852 |
| (c) cascade (Benna-Fusi) | 0.71 | 0.73 | 0.89 / 0.76 | 0.00 | 0.001 | 1.34 | 465 |
| (d) constant mild fade | 0.77 | 0.79 | 1.00 / 1.00 | 0.00 | 0.001 | 1.19 | 463 |
| **(e) maturity + sleep re-registration** | **1.00** | **0.98** | 1.00 / 1.00 | 0.00 | 0.000 | 0.98 | 833 |
| (f) no fade | 0.71 | 0.73 | 0.89 / 0.75 | 0.00 | 0.001 | 1.34 | 465 |

With the drift slowing at the LR decay (same budget), early useful / all useful:
- (a) 0.89 / 0.92 · (b) 0.89 / 0.91 · **(b_re) 0.99 / 0.98** · (b_tr) 0.80 / 0.71 · (b_un) 0.94 / 0.96;
- (c) 0.89 / 0.91 · (d) 0.90 / 0.92 · **(e) 1.00 / 0.98** · (f) 0.89 / 0.91.

With the generous budget (20,000 slots), the differences between a / b / c / d / f vanish (0.80 each, constant drift):
capacity is not binding, and the stale-key effect dominates. b_re 0.92, b_tr 0.89, e 1.00.

### 4.2 Early useful facts over time (main drift, constant, tight budget; recall at the end of intervals 1-8)

| policy | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| (a) global | 1.00 | 1.00 | 0.98 | 0.87 | 0.60 | 0.79 | 0.51 | 0.78 |
| (b) maturity | 1.00 | 1.00 | 0.98 | 0.88 | 0.60 | 0.79 | 0.51 | 0.71 |
| (b_re) re-encode | 1.00 | 1.00 | 0.99 | 0.95 | 0.98 | 0.97 | 0.83 | 0.92 |
| (c) cascade | 1.00 | 1.00 | 1.00 | 0.92 | 0.56 | 0.77 | 0.51 | 0.71 |
| (d) fade | 1.00 | 1.00 | 1.00 | 0.92 | 0.56 | 0.77 | 0.51 | 0.77 |
| (e) sleep | 1.00 | 1.00 | 0.99 | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 |

The curves are not monotone. Each step draws a fresh drift field (some stronger than others), and the probe recall
at an interval end depends on how recently each fact was last re-encountered. There are about 360 early-useful facts
per probe, so differences under ±0.05 are noise. b_re and e are flat; a / b / c / d / f oscillate with the drift's
recent strength, because their keys are only as fresh as the last failed-then-respawned re-encounter.

### 4.3 Harsh drift (72% relative change per interval, row overlap 0.04), tight budget

| policy | constant drift: early useful / all useful / newest one-shot | drift ×0.3 after 80%: early useful / all useful |
|---|---|---|
| (a) global | 0.25 / 0.29 / 0.82 | 0.40 / 0.79 |
| (b) maturity | 0.25 / 0.28 / 0.50 | 0.39 / 0.64 |
| (b_re) re-encode | 0.28 / 0.33 / 0.61 | 0.58 / 0.76 |
| (c) cascade | 0.25 / 0.29 / 0.50 | 0.40 / 0.62 |
| (d) fade | 0.25 / 0.29 / 0.82 | 0.40 / 0.77 |
| **(e) sleep** | **0.91 / 0.87** / 0.64 | **0.97 / 0.95** |
| (f) no fade | 0.25 / 0.25 / 0.25 | 0.30 / 0.54 |

When the thoughts move more than a key's basin between two re-encounters, re-encoding on access is too late (the read
already fails). Only re-registration from the source keeps up.

### 4.4 What the toy says
- **The switch and the selective-retention mechanisms are second-order.**
  - (a) ≈ (d) everywhere. Maturity and the cascade (b, c, f) are ≤ (a): they keep stale keys and, when slots are
    tight, crowd out the newest one-shot facts (0.75-0.79 vs 1.00).
- **Key renewal is first-order.** b_re: +0.14 (constant) / +0.10 (LR decay) early retention vs plan 21, at about twice
  the key writes. e: +0.22 / +0.11, and the only robust option under the harsh drift.
- **Interference** stays low for all policies (confident-wrong reads ≤ 0.4%). Confidence-only re-encoding causes about
  0.2% hijacks per re-encode, and 2-4× more key writes. It needs a confirmation gate.
- **Capacity**:
  - b_re and e use the fewest keys per useful fact (0.98-1.06): one renewed key instead of a trail of stale ones.
  - Trail keys (b_tr) are the most expensive (1.5 keys per useful fact at the tight budget, 4.6 at the generous one).
  - Plain fading (a, d) recycles slots fastest, so it keeps more one-shot facts.
- **One-shot facts do not survive 8 intervals of drift anywhere** (0.00; c / f 0.06 at 20K slots with slowing drift).
  That is the honest limit of a key memory: what is never seen again can only be kept by re-registering it, and (e)
  only re-registers mature items.

## 5. Recommendation and cost (vs the ≥ 10k tok/s budget of plan 23 §7)

| mechanism (splat / kNN pool) | keep? | brain cost per step | toy effect (early useful, main / harsh drift) |
|---|---|---|---|
| global switch + freeze (plan 21) | **drop for keys** (keep the LR-decay coupling only for the row pool, if rows stay) | 0 | baseline 0.78 / 0.25 |
| constant mild fade (lazy Λ per splat; the clock restarts at a re-encode) | **yes** (slot turnover) | +2 ops per read (gather + exp) | ≈ baseline |
| per-memory maturity (spaced confirmed re-use) | **yes, but only for value plasticity** (discounted → accumulating Adagrad S = EWC-like protection), and as the eligibility for re-registration | +4-6 ops in the write path | −0.07 on its own (it keeps stale keys) |
| cascade (Benna-Fusi) | no (no gain over fading in the toy; 3 floats per splat + a lazy 3-mode closed form ≈ 8 ops per read) | ≈ 8 ops per read | −0.07 |
| **re-encode on confirmed recall** | **yes, the default** | the spawn path again: index_copy of the coefficients + re-filing in the T = 4 index tables ≈ 70 ops per step, < 3 ms; the subsample = the confirmed reads among the written positions (≈ 1-2% of positions) | **+0.14** / +0.03 (+0.18 with slowing drift) |
| confidence-only re-encode | no (hijacks; 2-4× the writes) | - | 0 / −0.04 |
| trail keys | only with spare slots | ×1.5-4.6 slots | −0.22 at a tight budget |
| **replay re-registration** ("sleep") | **yes, as the second stage** | see below | **+0.22 / +0.66** |

**Replay re-registration, concretely.**
- Store a source pointer (sequence id, offset: 8 B per splat, 8 MB per 1M) with every splat.
- Cheapest form: **one replay window per micro-batch.** B = 16 becomes 15 fresh + 1 replayed window, drawn from the
  windows of the mature splats (oldest re-registration first).
  - Its forward gives the current thoughts at every stored position of that window, so all of that window's splats
    are re-registered with the same lookup / write ops as a spawn.
  - It also replays that content into the weights (CLS replay).
  - Cost:
    - as a normal training row: −1/16 of the fresh data (≈ −6% data throughput at unchanged tok/s);
    - as an extra no-grad row: ≈ +2-4% step time;
    - the re-registration writes themselves: < 3 ms.
  - At 2 windows per step and 4-8 mature splats per window, about 8-16K re-registrations per 1,000 steps. That is the
    toy's rate (about 3,300 mature items re-registered per interval at 12.8K facts), scaled up.
- A separate "sleep phase" (forward-only over 50K windows every 1,000 steps) costs about 25M tokens ≈ 7-14 min per
  1,000 steps: +50%. **Too expensive.** The interleaved row is the affordable form.
- Fits both speed configs of plan 23 (A ≈ 13-16k, B ≈ 20k tok/s) with ≤ 4% loss.

**The recommended life cycle for the splat / kNN pool (replaces plan 21's switch for this pool):**
1. Every write: a spawn, or a reinforce + **re-encode the key on a confirmed recall** (the fade clock restarts).
2. Constant mild fading (λ ≈ 5e-4-1e-3 per step) of everything that was not renewed; evict the weakest when full.
3. Per-splat maturity from spaced confirmed re-use. It sets (i) the value plasticity (discounted vs accumulating
   Adagrad) and (ii) the eligibility for replay re-registration.
4. One replay window per micro-batch re-registers the keys of mature splats (and replays them into the weights).
5. The LR schedule enters only through the drift itself. The fade can be lowered at the LR decay, because the drift
   slows; no freeze.

For the row pool (if it stays alongside): plan 21 as is, but the switch can be the LR-decay start directly (the
stability trigger cannot fire at full LR). Expect it to orphan the pre-switch rows; the splat pool is the part that
remembers across the drift.

## 6. Risks and open points
1. **"Confirmed recall" in the brain needs a signal.** Candidates: the gate opened and the read reduced the loss (a
   gate-closed counterfactual on a subsample), or |v* − r| below a threshold. A wrong confirmation moves a key to the
   wrong context. The toy's confidence-only variant bounds that damage (0.2% hijacks), but its benefit was smaller.
2. **The toy's re-encounters come with the true identity of the fact.** In text, "the same context recurs" is fuzzy:
   a paraphrase, a different document about the same entity. Re-encoding then merges contexts, which may be what we
   want (one fact, many phrasings) or a source of interference.
3. **Re-registration needs the source context stored and stable** (the dataset is fixed, so pointers stay valid).
   For splats spawned at later passes, the key depends on the pass, so the replay forward must run K passes (it does).
4. **The drift calibration of the toy is uncertain by about 2× in both directions.** The main drift matches the
   addressing data (row overlap 0.21); the harsh one matches the relative change (72%). The ranking of mechanisms is
   the same in both; the size of the gains differs.
5. **Values are fixed in the toy.** Value protection (EWC-like / Adagrad by maturity) is argued, not tested.
6. **The go / no-go measurement of plan 23 §10 (real self-recall across a 1,000-step checkpoint pair) still comes
   first.** It also measures how often a re-encode would be needed: the self-recall at 250 / 500 / 1,000 steps.
