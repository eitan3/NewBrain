# Wave memory v2.2 spec (coordinator, 2026-10-25 16:07 UTC). Eitan approved all four items at 16:06.

## Starting point and process
- Base: `fruit-fly-sim-wave27\` = the fixed wave26 snapshot (wave26 code + the F1, F2 and F6 fixes).
- Time box: code plus CPU tests in about 45 minutes. Run only the run-critical tests; deep reviews come later.

## (1) New-facts eval (world_evals; run it first on world_wave26_12k final.pt)
- **Facts:** 2,000 NEW people whose names are not in the world file, with the same relation types as the world facts.
- **Templates:** study text uses the 10 train templates; questions use the 2 held-out templates.
- **Study phase:**
  - eval mode, no_grad, no optimizer;
  - assert the brain weights' checksum is unchanged afterwards;
  - memory writes ON with the same write rule as training, exploration off;
  - each fact shown 3 times, mixed into wiki filler;
  - each sequence stands alone, and a question never shares a sequence with its fact.
- **Memory file:** use a COPY of final.memory.pt; never touch the original. If writes are train-only, add an eval write path that calls the same write function under no_grad.
- **Quiz:** cloze exact and rank, in four modes: on, ko, private_only, global_only.
- **Interference:** re-run the old late/early facts with the memory as it is after the new writes.
- **Bar:** on − off ≥ +0.05 exact; off should be near chance.
- **GPU job:** 16 GB cap, 2 threads; tell the coordinator before starting.

## (2) Delta write: `--wave-write delta` (default in v2.2)
- At a write, take what the chosen place already recalls for this key (from the read/confirm path), and write only the error:
  - target unit on but not recalled: counter +1, set the phase;
  - recalled but not in the target: counter −1 (the erase);
  - already matching: the existing confirm +1 only.
- Erase only when the key match is confident (z ≥ z_conf), so different facts sharing a place don't erase each other.
- Storage is unchanged; no gradient flows into thoughts.

## (3) Normalized readout: `--wave-readout norm`
- For each (token, expert) read y:
  - `y_n = RMSNorm(y) * g`, where g is a learned vector [384], init 1;
  - multiply by the existing confidence c;
  - multiply by the gate `s = 2*tanh(SiLU(xf·w_e + b_e)/2)`, where w_e is a per-expert vector [384] (small random init) and b_e inits to 1;
  - then apply `W_e = I + D_e` as today.
- A zero or no read must give exactly 0 output (off == none holds).
- The main loss may flow through the gate into xf; that is allowed, since it is the read path. There is no separate gate loss.
- Log the read-norm / xf-norm ratio.

## (4) Small brain run: world_wave27_small_12k
- **Experts:** grow 48 → 192 (not 192 → 768), with the same growth-schedule fractions, top-6, plus the shared expert.
- **Memory keeps the same size:**
  - global sheet 256 places;
  - 193 strips × 16 places (3,088 places vs 3,076 today);
  - reads take the best 4 entries within the strip through the same place-map mechanism as the global sheet;
  - D becomes [193, 384, 384].
- **Fallback:** if 16-place strips are not a small change, use 4 places per strip (a smaller memory) and say which was chosen.
- **Everything else** as world_wave26_12k: wiki_world_mix, 12,207 steps, 16×2, LR, seeds; items 2 and 3 ON.
- **One run.** The small-brain no-memory control comes later, on Eitan's word only.

## CPU tests (run-critical only)
1. off == none, bit-identical.
2. Delta toy: A→v1 then A→v2 recalls v2 within 2 writes, and an unrelated B on the same place survives at ≥ 90%.
3. A zero read gives zero output, and the gate gradients are nonzero.
4. Storage bytes are fixed.
5. The new-facts eval leaves the weights unchanged.
6. The small config builds and runs a tiny CPU step.

## Approvals
- The 20-step GPU VRAM smoke and the launch both need Eitan's own typed go; the coordinator asks him once the code is reported ready.
