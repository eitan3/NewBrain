# Launching training runs (launch_queue_v2.ps1)

The tools are `scripts\launch_queue_v2.ps1`, `scripts\auto_resume_v2.ps1` and `fruit-fly-sim-next\vram_guard.py`.

Use these for new runs. The older `launch_queue.ps1`, `auto_resume.ps1` and the `*_r7*` scripts are unchanged, and the live r7 run still uses them.

## Launch

```powershell
.\scripts\launch_queue_v2.ps1 -Queue r8 -Runs "world_blob25d_r8 | <train_lm flags, e.g. the r7 argv + --runs C:\Eitan\Python\NewBrain\runs>"
```

Optional parameters:

| Parameter | Effect |
|---|---|
| `-After <queue>` | Wait for another queue to finish first. Only one GPU job runs at a time. |
| `-Src <folder>` | Code folder to snapshot. Default `fruit-fly-sim-next`. |
| `-SkipGuard` | No VRAM guard. Use for tests and CPU runs. |
| `-GuardMarginGB 2` | How far below the cap the guard's peak must stay. |
| `-NoWatcher` | No auto-resume watcher. |
| `-MaxResume 2` | Resumes per run before the watcher gives up. |
| `-NoWatch` | No viewer window. |

What happens at launch:

1. **Code snapshot.** Every `<Src>\*.py` is copied to `runs\<name>\code_snapshot\`, together with `SHA256SUMS.txt`. The run starts from there, and so do all its resumes. Edits to the source folder after launch can never reach a running or resumed run.
   - A fresh run refuses a snapshot that already exists, so pick a new run name.
   - A run with `--resume` in its arguments reuses its snapshot.
   - A queue name can be used only once, because a reused log would confuse the watcher.
2. **Launch arguments.** The run's arguments are saved in `runs\<name>\launch_args.txt`. The watcher resumes with exactly these arguments plus `--resume`.
3. **Queue.** The queue (`runs\_queues\<queue>.cmd`) starts detached, with its log in `runs\_queues\<queue>.log` and its pid in `runs\_queues\<queue>.pid`.
4. **Watcher.** The auto-resume watcher starts detached, with its log in `runs\_queues\<queue>.watcher.log` and its pid in `runs\_queues\<queue>.watcher.pid`.

## The VRAM guard (vram_guard.py, inside the queue, right before each fresh run)

The guard runs on the GPU. It refuses to start next to a live run that holds `runs\gpu.lock`.

It runs about 20 training steps of `train_lm.main` from the run's snapshot, with the run's exact flags, but at the final size and in the worst case:

- `--start` is set to the final heads and experts (no births).
- Every step uses K = 16, and lam = 1 (spikes on).
- The blob memory is fitted at step 5.
- One IVF refit happens at step 15.
- evaluate() and the loop report run at step 10.
- The run folder is a temporary one: `runs\<name>\guard\`.

It records `max_memory_reserved` and compares it with cap − margin, where cap is `--mem-fraction` × total GPU memory and the margin is 2 GB.

- **OK:** the run starts.
- **REFUSE (or an OOM):** the queue log gets two lines:
  - `vram_guard: REFUSE: peak ... > cap ... - margin ...` followed by the cheapest savers that would fit, with their estimated GB and time cost;
  - `guard REFUSED <name>` and `finished <name> (exit 98)`.

  The run is not started and the watcher does not resume it. Add the suggested flags and launch under a new name.

The guard takes a few minutes, mostly for the model build, the fit and the 20 steps.

Saver estimates are from the CPU allocation tracker at the final size (768 experts, 36 heads, batch 16×2, K = 10). All savers are the same maths, verified bit-identical on CPU.

| Flag | Saves about | Time cost (est.) |
|---|---|---|
| `--vram-readout-lean 1` (default ON in fruit-fly-sim-next) | 3.2 GB | 0 |
| `--vram-ckpt-concept 1` | 0.9 GB | ~0.5% |
| `--vram-ckpt-gmdist 1` | 0.6 GB | ~0.5% |
| `--vram-ckpt-norm 1` | 1.5 GB | ~2% |
| `--vram-ckpt-experts 1` | 2.4 GB | ~3-4% |
| `--vram-ckpt-mixer 1` | 2.0 GB | ~3-4% |
| `--vram-savers 1` | all of the above | |

## Auto-resume (auto_resume_v2.ps1)

The watcher waits for `queue <q> done` in the queue log. It then decides for each run of the queue:

| Outcome | What the watcher does |
|---|---|
| exit 0 | Done. |
| NaN abort: `nan_abort.pt` newer than `last.pt`, or `NAN ABORT` in the tail of the run's train.log | Not resumed. The NaN would recur with the same data and RNG, so it is left for the owner. |
| guard refused (exit 98) | Not resumed. |
| no `last.pt`, or `-Max` resumes already used | Not resumed. |
| any other failure (OOM, crash, killed) | Resumed as a new queue `<queue>_resume<k>`: `launch_args.txt` + `--resume runs\<name>\last.pt`, from the run's code snapshot, no guard. |

The watcher waits for each resume queue's `done` in turn. It exits when nothing is left to resume, for example once every run has finished cleanly.

A resume repeats the steps since the last `last.pt`, which is saved every `--ckpt-every` steps (1000 by default). It also needs a minute or two to load.

## Stop

```powershell
.\scripts\stop_run.ps1 -Queue <queue>          # kills the queue and its run (pid tree)
```

Stop the watcher first if the run must not be resumed:

```powershell
taskkill /PID (Get-Content runs\_queues\<queue>.watcher.pid) /T /F
```

Otherwise the watcher sees the killed run as a failure and resumes it, at most `-Max` times. A watcher waiting on a queue that never logs `done` waits forever, so kill it the same way.

## Dry run (no GPU)

`runs\_dryrun_v2\dryrun.ps1` exercises the launcher and watcher on a fake tree: a fake `train_lm.py` and a mocked `vram_guard.py` in `fake_src\`, polling every 2 s. It checks that:

- snapshots are created and runs start from them;
- a fresh run refuses an existing snapshot;
- a failure is resumed from the snapshot with `--resume`;
- the watcher waits for each resume and stops at Max 2;
- a clean exit makes the watcher exit;
- a NaN abort is not resumed;
- a guard refusal is not resumed.

The result is in `runs\_dryrun_v2\dryrun.log`. To run it again, delete or rename `runs\_dryrun_v2\runs` first.
