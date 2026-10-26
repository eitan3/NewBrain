# NewBrain

A research project on a **brain-inspired language model**: a looped, spiking, mixture-of-experts network that grows as it trains, and a separate, swappable **memory** that lets it hold facts across sequences.

It started from the fruit-fly connectome simulation (Shiu et al., *Nature* 2024; see `fruit-fly-sim/README.md`) and grew into a language model built on the same ideas.

## The model

`LoopBrain` (`loop_brain.py`, `spiking_lm.py`) is a spiking language model that:

- runs the same layers in a **loop** (K passes, with learned halting and a workspace between passes);
- routes tokens to **experts** (mixture of experts) and **grows** them during training (for example 48 → 192 experts, by splitting);
- uses graded **spiking** units, annealed from analog to spikes early in training;
- adds concepts, "thinking" tokens and deep supervision on earlier passes.

The memory is a separate module, kept apart from the brain's weights so it can be swapped, saved per user, or trained per expert (`memory_api.py`). Several designs were tried:

| Memory | Folder | Idea |
|---|---|---|
| Blob memory | `fruit-fly-sim-next` (`blob_memory.py`) | Key/value slots with learned addressing |
| Wave memory v2 / v2.1 / v2.2 | `fruit-fly-sim-wave*` (`wave_*.py`) | Activity sheets with travelling waves |
| **FastMem** (current) | `fruit-fly-sim-fastmem*` (`fastmem.py`) | Fast-weight matrices (delta rule with decay), one per expert plus a global sheet, one memory per data stream |

The current line is FastMem with **packs** (`fruit-fly-sim-fastmem-c`, `--fastmem-pack 4`). Four consecutive sequences of a stream are scanned as one segment. That lets a question's loss train the memory write of a fact up to 3 sequences earlier.

Everything is trained on one laptop GPU (RTX 5090 Laptop, 24 GB).

## Latest result

`world_fastmem_pack_12k` (12,207 steps, 200M tokens, final val loss 3.93) is the **first run where the memory is used**.

On questions about facts stated 1-4 sequences earlier, the right answer ranks first among the candidates:

| | Rate | Change |
|---|---|---|
| Memory off | 4.0% | |
| Memory on | 6.1% | +2.0 points |
| Chance | 2.8% | |

The gain is positive at every checkpoint and every distance. The previous run, without packs, showed about 0. It is still below the +5 point goal, and exact answers are still about 1%.

## Repository layout

| Path | What |
|---|---|
| `fruit-fly-sim/` | The original fruit-fly simulation work, design notes (`DESIGN_*.md`) and the v2 plan (`plan_v2/`) |
| `fruit-fly-sim-next/` | The main model line (blob memory era) |
| `fruit-fly-sim-wave*/` | Wave-memory versions |
| `fruit-fly-sim-fastmem/`, `-b/`, `-c/` | FastMem: first build; packed-scan VRAM fix; packs (current) |
| `fruit-fly-sim-r7-lean/`, `-orig-reconstructed/` | Older snapshots kept for reference |
| `scripts/` | PowerShell tools: `launch_queue_v2.ps1` (launch with snapshot + VRAM guard), `auto_resume_v2.ps1` (restart after crashes), `watch_run_follow.ps1` (viewer), `stop_run.ps1`, `status.ps1`; see `README_launch.md` |
| `world_change/` | The changing-facts dataset builders, memory evals and diagnostics, and their results |
| `world_evals/`, `phase0_evals/` | Earlier evaluation results |
| `speed_wip/`, `speed_pkg/` | Speed and VRAM experiments |
| `reviews/`, `viz/`, `tokenizers/` | Code reviews, visualisations, the wiki16k tokenizer |

Each code folder is a full, self-contained copy of the model code. A run executes from a snapshot of one folder (`runs/<name>/code_snapshot/`), so editing a folder never changes a running run.

## Setup

- Windows 11, one NVIDIA GPU (~25 GB), Python 3.13 in a `uv` virtual environment (`.venv`).
- Main packages: `torch` 2.11 (CUDA 12.8), `numpy`, `tokenizers`.
- Datasets live outside the repository (on `X:\Datasets`): `wikipedia_en_20231101_wiki16k`, `wiki_world_mix`, `world_change_mix`, `world_change_mix_v2`. The builders are in `world_change/` and the `prepare_*.py` / `generate_world.py` scripts.

## Training

Runs are launched through a queue that snapshots the code, checks VRAM first, and restarts automatically after a crash:

```powershell
.\scripts\launch_queue_v2.ps1 -Queue <queue> -Src fruit-fly-sim-fastmem-c -Runs "<run_name> | <train_lm.py flags>"
.\scripts\watch_run_follow.ps1 -Name <run_name>     # a window that follows the run across restarts
```

The flags of the latest run are in `scripts/launch_fastmem_pack.ps1`. See `scripts/README_launch.md` for the guard, the watcher and resuming.

## Evaluating the memory

The changing-facts data gives each person new facts in every "world" and asks about them 1-4 sequences later. The answer is outside the model's 512-token context, so only the memory can carry it.

```bash
python -B world_change/eval_change_fastmem.py --ckpt <copy of a checkpoint> --data X:/Datasets/world_change_mix_v2 --pack 4
python -B world_change/diag_change_fastmem.py --ckpt <copy of a checkpoint> --data X:/Datasets/world_change_mix_v2
```

The eval compares memory **off**, **on without writes** and **on with writes**, by distance, relation and question form. Run evals on a copy of a checkpoint, on the CPU (`CUDA_VISIBLE_DEVICES=-1`, 2 threads), so they never touch a live run.

## Tests

CPU only:

```bash
cd fruit-fly-sim-fastmem-c
python -B tests_fastmem.py          # the memory module (causality, off == none, gradients, save / load)
python -B tests_fastmem_packed.py   # packed scan == padded scan
python -B tests_fastmem_pack.py     # packs: P = 1 matches the old code, gradient across sequences, data layout
```

Some older tests need small `.pt` fixtures that are not in the repository.

## Not in the repository

`.venv/`, `runs/` (checkpoints and logs), `data/`, datasets, model and data files (`*.pt`, `*.bin`, `*.npz`, ...), logs, and `Drosophila_brain_model/`. That last one is a separate clone of [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model). See `.gitignore`.
