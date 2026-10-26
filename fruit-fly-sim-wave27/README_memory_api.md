# The memory-module interface (fruit-fly-sim-wave)

This folder is a copy of `fruit-fly-sim-next` with the memory behind one interface, so that a different memory (the
wave memory, plan_v2/26) can be dropped in next to the blob memory. The settled parts of the wave memory are built
(the address, the read / write path, the pluggable storage and write policy); its storage is a PLACEHOLDER until
the storage / forgetting spec arrives (section "The wave memory" below).

Files:

| file | what |
|---|---|
| `memory_api.py` | `MemoryBase` (the interface), `memory_off`, memory.pt save / load, train-only freezing, byte budget |
| `wave_memory.py` | `make_wave_memory`; `WaveMemory` (the stub, `--wave-mode stub`); `CodedWaveMemory` (`--wave-mode on`) |
| `wave_address.py` | `WaveAddress`: the online SOM entry map + residual code levels; `Codes` |
| `wave_store.py` | the STORAGE interface `WaveStore`, the registry `STORES`, the placeholder `HebbianSheet` |
| `wave_write.py` | the WRITE-SELECTION interface `WritePolicy`, the registry `POLICIES` (surprise_h, surprise, random) |
| `wave_bench.py` | the adapter: `wbench_module.Harness` scores `--memory wave` end to end (read-only import of mem26) |
| `tests_wave.py` | the wave memory's tests (unit + tiny training runs), CPU |
| `tests_wave_bench.py` | the W-bench parity of the address (ckpt_6000 capture) + the module bench, CPU |
| `blob_memory.py` | `BlobMemory` now subclasses `MemoryBase`; new methods at the end of the class (trainer hooks, train-only) |
| `loop_brain.py` | builds the module from `--memory`; `reset_memory()`; the runtime off switch in `run()` |
| `train_lm.py` | the `--memory*` flags; the blob-specific trainer code moved into the module's hooks |
| `tests_memory_api.py` | the tests (a)-(g), CPU, tiny configs |

## Choosing the memory

```
--memory blob   (default) the blob memory, configured by the --gm ... flags exactly as before
                (--memory blob with --gm 0 = no memory, as before)
--memory wave   the wave memory: --wave-mode on (default, CodedWaveMemory) or stub (WaveMemory: reads zeros)
--memory none   no memory module at all: exactly the --gm 0 code path, whatever --gm says
```

With the default flags the trainer is bit-identical to `fruit-fly-sim-next` (per-step losses and the full
checkpoint: tests (a)).

In the model, the module is `LoopBrain.bm` (state_dict prefix `bm.`). The blob memory's cfg is unchanged (`gm=1` +
its `gm_*` keys); the wave memory is in cfg as `memory="wave"` + `wave_*` keys; `LoopBrain(**cfg)` rebuilds either.

## The interface (`memory_api.MemoryBase`)

`MemoryBase` is an `nn.Module` that registers nothing, so subclassing it does not change a state_dict.

**The read** - RoutedMLP calls the module once per pass, right after the routed experts' capacity layout is filled:

```
module(xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=k) -> (buf, x_shared)
```

| arg | meaning |
|---|---|
| `xf (N, D)` | the experts' input rows: each token's thought at this pass (N = B*T) |
| `buf (E, C, D)` | the routed experts' capacity layout, `buf[e_k, p_k] = xf[t_k]` |
| `x (B, T, D)` | the shared expert's input |
| `side` | the router's side input (workspace summary) or None |
| `e_k, p_k, t_k` | the kept (expert id, capacity slot, token) pairs, each `(n_pairs,)` |
| `slot_k, counts, N, E, C, n_keep` | top-k slot of each pair, tokens per expert, sizes |
| `call` | the pass index: a cache id (activation checkpointing re-runs a forward with the same id) |

The simple way to implement a memory is `read()`:

```
read(xf, e_k, t_k, p_k=None, slot_k=None, side=None, call=None)
    -> (r_pairs (n_pairs, D) or None, r_shared (N, D) or None)
```

and the base `__call__` injects it: `buf[e_k, p_k] += r_pairs` (before the experts' up projection),
`x_shared = x + r_shared`. A module that needs its own layout overrides `__call__` (the blob memory does: masked,
compacted, gated reads).

**Trainer hooks** (all optional, defaults do nothing), in the order of one optimizer step:

| hook | when |
|---|---|
| `trainer_setup(args)` | once, before the loop |
| `step_begin(step, args) -> [lines]` | each step, before the evaluation (the blob: step stamp, fit window, the one-time fit) |
| `begin_forward()` | LoopBrain.run, each model forward |
| `select_writes(logits, y, x, ce_h=None)` | between a micro-batch's loss and its backward |
| `after_backward()` | after each micro-batch's backward. Memory-side losses are backpropagated here; they must not reach the brain (test (f)) |
| `after_step(step, refit_every=0)` | after the brain's optimizer step: the memory's own update |
| `on_grow(src, E0, split)` | experts `E0 ..` were born from parents `src` |
| `describe(args)`, `log_record(step)`, `health()`, `diag_snapshot()`, `metrics(step)` | logging |
| `ready` (property) | True once the module reads (the blob: after its address fit) |

**State**: `memory_state_dict()` / `load_memory_state_dict(sd)` (default: the module's state_dict), `memory_cfg()`,
`nbytes()`.

**Train-only**: `set_train_only(expert_ids)` restricts the module's own learning to those experts' memory;
`trainable_parameters(expert_ids)` lists the nn.Parameters that stay trainable (the default: none).

**Runtime off**: `module.enabled = False` (or `with memory_off(model):`) makes LoopBrain run the no-memory code path;
the module is not called at all and nothing in it changes (test (d)).

## The wave memory (`--memory wave --wave-mode on`)

**Current defaults (WaveSheet v2, `SPEC_wavesheet_v2_cloud.md`; this section's later paragraphs describe the earlier
build and still apply where not overridden here).** `--wave-store columns` (`wave_columns.py`): one global sheet at
P = `--wave-side`^2 = 1,024 places x 3 layers x 16 columns, 4-bit key synapses (`s = 1 + 5 (cnt - 1) + ph`), chain
bits (`--wave-lambda` 4), the decoder A / B (lr 0.1 x `--wave-lr`, renormed after each memory step), the sweep of 16
places per step to f* = `--wave-fill` and f*ch; 459 MB of state at d 384 / 769 experts. The address codes entry 1's
OWN residual (x_hat - P[E1]) for the column store (`Codes.code_e`; `--wave-entry-keys shared` = the arm reusing the
entry-0 keys). `--wave-priv-levels 0` (no private stores). The recall runs once per forward at pass
`--wave-read-pass` 3 and is re-applied at passes 4-5. The keep mask (`--wave-drop` 0.1, `--wave-drop-mode seq`): 10%
of the sequences of each training forward get m = 0 (the capture leaf still sees them; `forward` = the old
whole-forward dropout). Per-expert gate rows `gate_e` (zero init, clamped to [-4, 4]). `--wave-store sheet` is the
pre-v2 sheet with `--wave-pack nibble|c2p3`. Train-only without private parts (the default): owned places (the
`--wave-owned` 16 top `owner_use[:, e]` places, never an empty one), writes / confirms / sweep there only, the gate
row e trainable, everything else frozen; the sheet / hebb stores without private parts raise a clear error.

Place balance (measured, `tests_wave_curve.py`): the entry map alone collapses the W-bench facts onto ~51 of the 1,024
places (8.6K facts at one place), so the columns crowd. Defaults now: `--wave-map-beta 0.25` (fix A: the map also
learns on the WRITTEN thoughts, per step of 32 writes, under a write-count conscience; the two-speed rate) and
`--wave-entries 4` (the read scans 4 places, keeps the clearest by chibar; writes at entry 0).
`--wave-alloc-gamma` (fix B: load-biased write entry) is an arm. The null control bins by the winner place's load
(`--wave-null-bins load`; `fill` = the spec's global-fill bins) and a confirmation needs z >= `--wave-z-conf` 2
(c_tok = clamp((z - `--wave-c-z0` 2) / 4, 0, 1)). `--wave-priv-levels 1` adds the private parts (763 MB in all at
d 384 / 769 experts; global only 459 MB). Tools: `tests_wave_curve.py` (T3a / T3b / the capacity curve with load,
place-agreement and oracle-place diagnostics), `tests_wave_conf.py` (z precision / recall, no-match cues),
`tests_wave_drift.py --write-cap/--read-cap` (brain drift), `stage_a.py reach` (the expert as the private address).

Owner's rules it keeps: separate from the brain's weights (its own state; memory.pt; swappable; one expert's memory
trainable alone); everything inside the normal training step, no sleep / offline phase; no memory-side loss, so no
gradient into the thoughts or the brain; addresses only from the brain's own thought xf (never token ids or data
pointers); no row or hash table (codes over neurons, a store of synapses); no bigger than the blob memory (full-size
defaults, d 384 and 769 experts: 913 MB of state vs the blob's ~1.9 GB).

**Address** (`wave_address.py`). x_hat = ((xf - mu) / sd) @ rot (running statistics, fixed seeded rotation; xf
detached). An entry map of `--wave-side`^2 = 256 prototypes on a torus (an online SOM: roughly the person) gives the
top-`--wave-entries` (2) locations under a usage-homeostasis bias; then `--wave-levels` (3) residual levels, each
the residual of x_hat after the previous level's prototypes (level 1: x_hat - the location's prototype), cut into
`--wave-slices` (32) slices with `--wave-codes` (16) prototypes each, soft top-`--wave-topk` (2) codes per slice.
Learning, in `after_backward` of every training micro-batch, on a deterministic strided subsample of that forward's
detached thoughts (`--wave-upd-rows`): a batched SOM step (winner + Gaussian torus neighbourhood annealed over
`--wave-anneal` updates, usage homeostasis on the winner and on the read bias), minibatch k-means per (level, slice)
with dead prototypes re-seeded on the worst-coded rows. The first update seeds everything from that batch's thoughts;
reads and writes start after `--wave-warm` updates. Deterministic under `--wave-seed`.

W-bench parity (`tests_wave_bench.py`, ckpt_6000 capture, passes 1-5, the address learning in-step style on the fit
thoughts, scored by wbench's own idealised code-match ranking of variant "(c+d) SOM + 3 residual levels x 32 slices,
top-2"): held top-1 0.699 / cloze top-1 0.532 (wbench: 0.684 / 0.524).

**Read** (`CodedWaveMemory.__call__`, the RoutedMLP hook, once per pass): codes of every token; the global store's read
R_g; the gate g = sigmoid(gate_router(f) + b0), f = DETACHED features (location margin, code margins and residual
fractions per level, log stored mass, |R_g| / content scale); exploration (`--wave-explore`, a random g) and a floor
(`--wave-gate-floor`). Each kept pair's expert input gets g (gain_g R_g + gain_p[e] R_p[e]), the shared expert's input
g (gain_g R_g + gain_p[shared] R_p[shared]). The gate is decided before the experts run (as the blob's). gate_router,
gain_g, gain_p are nn.Parameters in the brain's optimizer (BrainOpt: Adam), trained only by the main loss through the
read; the read has no path to xf (tests_wave (u5): d xf.grad is identical with and without the memory).

**Writes** (once per token, never per pass x expert). `select_writes` (between loss and backward): the write policy
picks this micro-batch's tokens from the final read-out's per-token CE / entropy (`--wave-write-w`, at most
`--wave-write-per-seq` per sequence, not at / right after an EOT). During the forward each in-window pass adds a zero
leaf at the read point; after backward its gradient is dL/d(expert input). For each selected token, the ONE pass with
the largest gradient is written: content = -gradient normalised to `--wave-content` x median |xf|, codes of that
pass; parts: the global store + the token's top-1 routed expert's private part + the shared expert's part. The
policy's `plan()` sets each write's strength and entries (e.g. a second entry after a failed recall), with the read's
gate value and cos(read, target) as the recall signal; the store's `confirm()` hook gets the confirmed recalls
(open and cos > `--wave-confirm`). Logged write mix (`wmem_write_ce / all_ce / write_ent / write_seq_frac / write_pos`,
per source when the trainer sets `policy.source_fn`).

**Storage** (`wave_store.py`, pluggable, `--wave-store`): `write(codes, content, strength, part, entries)`,
`read(codes, part, entries)`, `mass`, `confirm(codes, ok, part)`, `forget(step, parts)` (every step), `copy_parts`
(births), `byte_budget`, `fill`; the state is buffers. The only store is `hebb` = `HebbianSheet`, a PLACEHOLDER:
synapses from code units (global: location x level-slice x code; private: part x level-slice x code, the first
`--wave-priv-levels` levels) to the D content neurons, Hebbian writes, a read = the mean over the active units of
their mean stored content, an optional uniform decay (`--wave-forget`). On the full-module bench it recalls poorly
(`wave_bench.py`, passes 1-5, gate bypassed: global store held 0.134 / cloze 0.061 top-1, + private 0.123 / 0.049;
with the address still learning while the facts are written 0.022 / 0.017: the codes drift away from the stored
ones): a linear superposed read cannot pick one memory out of a crowded unit; the spec's store (e.g. counters + phase, confirm increments, random decrements to a fill target,
2 entries) plugs in behind the same interface.

**Memory dropout** `--wave-drop` (0.1): per training forward, the whole memory off (no read, no write; the address
still learns). Deterministic: (seed, forward count). **Off / reset / swap / train-only**: the memory_api switches, as
for the blob. Train-only freezes the address, the global store, the gate, gain_g, the write baseline and the content
scale; only the selected experts' private parts are written and only their gain_p entries learn (the others masked
and restored after each step).

Flags (cfg `wave_*`, all in `WAVE_DEFAULTS`): `--wave-mode on|stub --wave-seed --wave-side --wave-levels
--wave-slices --wave-codes --wave-topk --wave-entries --wave-warm --wave-anneal --wave-upd-rows --wave-store
--wave-store-dtype --wave-read-topk --wave-priv-levels --wave-priv-topk --wave-forget --wave-write --wave-write-w
--wave-write-per-seq --wave-content --wave-confirm --wave-drop --wave-drop-mode --wave-gate-b0 --wave-explore
--wave-gate-floor --wave-max-experts --wave-lambda --wave-pack --wave-entry-keys --wave-owned --wave-map-beta
--wave-alloc-gamma --wave-null-bins --wave-c-z0` (see `train_lm.py` for the rest). Logged per eval: `wmem_*` (address usage / dead codes, writes, dropped forwards, the write mix,
store fill, gate open rate, gains).

Cost (an estimate, not measured: the GPU was busy): per read, the address is 256 prototype distances + 3 x 32 x 16
code distances (~0.2 MFLOP per token); the placeholder's global read gathers 2 entries x 96 level-slices = 192
synapse rows of 384 fp16 (147 KB), the private reads 64 rows per pair (49 KB). At 16K tokens x ~12.8 passes per step
with 6 pairs per token: ~0.1-0.2 s per step of gathers at ~0.7 TB/s effective, under the blob's ~1.27 s. Writes and
address updates are a few thousand rows per micro-batch. (Determinism on CUDA needs
torch.use_deterministic_algorithms for the stores' accumulating index_put.)

## Swap flags

```
--memory-save 1         every checkpoint also gets the memory alone next to it: last.pt -> last.memory.pt
                        (ckpt_<step>.memory.pt, final.memory.pt, nan_abort.memory.pt); {kind, cfg, state, step, n_experts, d}
--memory-embed 0        brain-only checkpoints (no bm.* keys; implies --memory-save 1). --resume of such a
                        checkpoint reads <name>.memory.pt next to it (or --memory-load). Default 1: as before
--memory-load PATH      load a memory.pt into the model's memory module (same kind; sizes must fit)
--memory-reset 1        the memory module starts fresh (as built from cfg); before --memory-load if both.
                        A fresh blob memory mid-run moves its address fit to resume step + --gm-fit-window
--resume X --memory K   K different from the checkpoint's kind: the brain is kept, the memory is swapped
                        (fresh, or --memory-load). Needs --opt brain if the parameters change
--memory-eval-off 1     every evaluation (val, loop report, sample) with the memory switched off
--memory-train-only experts=1,3,shared
                        only these experts' memory learns (shared = the shared expert)
--memory-cap-mb M       the byte budget of the memory state (printed at start; 0 = no cap)
--memory-cap-strict 1   exit when over the cap
```

### Train-only mode

`--memory-train-only experts=<ids>` (usually with `--resume` of a run past the blob fit, or `--memory-load` of a
fitted memory):

* the brain: every model parameter gets `requires_grad False` and leaves the optimizer (BrainOpt groups / AdamW param
  groups); its buffers (homeostasis thresholds, router balance bias, load EMAs) are restored after every step;
  neurogenesis / scheduled growth / dead-unit re-init are off; a micro-batch whose loss reaches nothing trainable has
  no backward.
* the blob memory: only the contents (value + content-optimizer state) of the selected private spaces learn. The
  global space, the other private spaces, keys, occupancy, statistics, the probation store, the address map and the
  gate (weights + optimizer state) are frozen: writes and the IVF refit are off, other rows' content gradients are
  masked before the update, and the rest of the state is restored after each step end from a snapshot (one device copy
  of the memory's persistent buffers - for the full-size run about the memory's own size again in VRAM). The content
  gradient reaches the store through a detached match-weight leaf (the frozen brain gets no gradient).
* the wave stub: plumbing only (no parameters). The wave memory (on): the "Train-only" lines of its section.

## Tests

```
set CUDA_VISIBLE_DEVICES=-1 & python tests_memory_api.py [work_dir]
```

(a) default flags == fruit-fly-sim-next bit for bit (blob `--gm-preset 25d` and `--gm 0`: losses + full checkpoint;
the reference runs import fruit-fly-sim-next's own code in a subprocess, no .pyc written there); (b) `--memory none`
== `--gm 0`, and the wave stub == none; (c) memory.pt save / reset / load -> identical outputs, and a stop / resume
through brain-only checkpoints == the uninterrupted run; (d) memory off at eval == no module, stores untouched; (e)
train-only leaves every brain tensor and every other memory part bit-identical, the selected experts' contents learn;
(f) hooks: the gate's memory-side loss reaches no brain parameter and no xf; (g) the byte budget vs a cap. About 10
tiny runs, ~12 minutes on 2 CPU threads.

```
set CUDA_VISIBLE_DEVICES=-1 & python -B tests_wave.py [work_dir]          (~10 minutes)
set CUDA_VISIBLE_DEVICES=-1 & python -B tests_wave_bench.py               (W-bench parity + module bench, ~5 min)
```

tests_wave.py: (u1)-(u10) unit: address determinism, no graph, coarse-to-fine error, homeostasis on skewed thoughts,
no gradient into the thoughts (the gate and gains do learn), one write per token, byte budget (incl. the full-size
defaults under 1.9 GB), pluggable store / policy (strength, entries, confirm, forget, plan), dropout, train-only;
(e1)-(e6) tiny runs: end to end, determinism, stop / resume, memory.pt save / reset / load + off + eval untouched,
swap none -> wave, `--memory-train-only experts=1,shared`.
