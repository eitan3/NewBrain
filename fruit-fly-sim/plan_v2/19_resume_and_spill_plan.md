# Resume-from-checkpoint + spill hygiene (plan only, no code) — 2026-10-24
Why: wiki_mem_12k spilled into shared GPU memory right after the freeze (2k tok/s, reserved 22.7 GB). With no resume,
stopping costs the whole run.

## 1. Full checkpoint (last.pt, every --ckpt-every steps, atomic: write tmp, then rename)
- model.state_dict(): already includes the memory buffers (V, S_acc, wcount, keys, EMA copies, norm stats, the freeze
  flag), the probe_state extra state and the halt/concept modules.
- Optimizer:
  - BrainOpt: state per name (m, v, n, mom, born), t, fine / lazy sets;
  - or the torch AdamW state_dict.
- Neurogenesis:
  - births / renewals, the schedule position, pending ramps, the done() flag;
  - head-group growth state (n_groups, group_offset);
  - the current model cfg (sizes after growth), so the model is rebuilt at the grown size before loading.
- Memory host state not in buffers:
  - mem._step, s0 (buffers: ok);
  - the balance windows win / win_e (non-persistent now: make them persistent or save them separately);
  - the margin / gen RNG states (mem.gen, mem.gen_margin);
  - MemoryProbes: the probe sets (deterministic from the seed) and the ref / last records (already in probe_state).
- Trainer state:
  - step, ema loss, tokens seen, flops, last_grow, n_eval;
  - the spill-guard state (tok_hist, spill_run, spill_freed);
  - report_steps already done;
  - the metrics.jsonl position (append mode is fine).
- RNG: python random, numpy (the data sampler RandomState), torch CPU, torch CUDA (all devices), and the model's own
  generators.
- Data: the sampler is stateless apart from its RNG, so the RNG state is enough.

## 2. --resume
- Path: --resume runs/<name>/last.pt.
- Rebuild the model from the saved cfg (the grown sizes), load the state, rebuild the optimizer with build_opt and
  load its state, restore Neurogenesis / trainer / RNG, then continue the loop at step + 1. The LR schedule is a
  function of step, so nothing extra is needed for it.
- Open the run folder in append mode; log "RESUMED at step N".
- Test:
  - CPU, tiny config: run 40 steps straight vs 20 steps + save + resume + 20 steps → bit-identical losses / state
    (with growth, a freeze and memory on);
  - the same with a forced balancing event and a probe eval across the cut.

## 3. Spill hygiene
- free_cache() right after the freeze probes (probes.freeze), and after every probe eval when reserved > 0.85 × total.
- Faster spill guard: act on the FIRST eval where reserved > --spill-gb AND tok/s fell > 35%, not after 3.
- Log torch.cuda.memory_reserved() before/after the probes for a few evals, to confirm the probes cause the jump.
- Cheaper option: run the probes with smaller batches (--mem-probe-batch 8), which lowers the peak during probes.
