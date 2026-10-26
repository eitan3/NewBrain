"""Train SpikingLM on tokenised TinyStories, optionally growing from a small start.

    python -u train_lm.py --run-name s0_ssd_grow --mixer ssd --start layers=2,heads=2,mlp=512 \
        --grow-at 0.1,0.2,0.3 --tokens 300e6

Everything goes into runs/<run-name>/: run.pid, config.json, metrics.jsonl, final.pt, sample.txt.
Progress lines are printed with flush, so the launcher's log file is always current.
One GPU job at a time: runs/gpu.lock holds the pid of the run using the GPU.
"""
import argparse
import contextlib
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from brain_opt import BrainOpt, carry_brain_state
from loop_brain import LoopBrain, chunked_readout
from memory_api import (MEM_PREFIX, MEMORY_KINDS, BufferFreeze, budget_line, byte_budget, freeze_for_train_only,
                        load_memory_file, memory_off, memory_path, parse_train_only, save_memory_file)
from memory import GATE_FEATS, GATE_MODES, LifeCycle, MemoryProbes, gate_c
from neurogenesis import Neurogenesis, reset_opt_rows
from spiking_lm import (RoutedMLP, SpikingLM, carry_optimizer, n_params, redo_dead, remap_name, remap_snapshot,
                        snapshot_optimizer)

HERE = Path(__file__).resolve().parent


def pid_alive(pid):
    if sys.platform == "win32":  # os.kill(pid, 0) would TERMINATE the process on Windows
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return not ok or code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_cmdline(pid):
    """The command line of a live process ("" if it has none), None if no such process. Raises if it cannot be
    queried (the caller then keeps the old pid-only rule)."""
    if sys.platform == "win32":  # (no psutil in the venv: one CIM query, only at launch)
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}'; "
                              f"if ($p) {{ 'FOUND ' + $p.Name + ' ' + $p.CommandLine }}"],
                             capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise OSError(out.stderr)
        text = out.stdout.strip()
        return text[len("FOUND "):] if text.startswith("FOUND") else None
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return None


def lock_holder_alive(pid):
    """The gpu.lock owner holds the GPU only if it is a live python process running train_lm.py (Windows reuses
    pids: a stale lock once blocked a launch because its pid now belonged to bash.exe)."""
    if not pid_alive(pid):
        return False
    try:
        cmd = pid_cmdline(pid)
    except Exception:  # (cannot tell: keep the lock, as before)
        return True
    return cmd is not None and "python" in cmd.lower() and "train_lm.py" in cmd


def claim_gpu(runs, name):
    lock = runs / "gpu.lock"
    if lock.exists():
        try:
            owner, holder = lock.read_text().split()[:2]
            if int(owner) != os.getpid():
                if lock_holder_alive(int(owner)):
                    sys.exit(f"GPU is in use by run {holder} (pid {owner}); one GPU job at a time.")
                lock.unlink()
                print(f"stale gpu.lock removed (run {holder}, pid {owner} is not a running train_lm.py)", flush=True)
        except ValueError:
            pass
    lock.write_text(f"{os.getpid()} {name}\n")
    return lock


class Tee:
    """Copy everything printed (and tracebacks) into runs/<name>/train.log, flushed per write."""

    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        try:
            self.console.write(text)
        except (OSError, ValueError, UnicodeEncodeError):
            pass
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        for s in (self.console, self.file):
            try:
                s.flush()
            except (OSError, ValueError):
                pass


def parse_moe(s):
    if not s:
        return None
    e, k, h, sh, bal = s.split(",")
    return dict(n_exp=int(e), k=int(k), hidden=int(h), shared=int(sh), balance=bal)


def parse_dims(s):
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))} if s else {}


class Data:
    def __init__(self, path, seq, device, curriculum=False, mode="random"):
        # token id width: uint16 unless the cache's meta.json says otherwise (large vocabularies: uint32)
        meta = path / "meta.json"
        mj = json.loads(meta.read_text()) if meta.exists() else {}
        dt = np.dtype(mj.get("dtype", "uint16"))
        self.train = np.memmap(path / "train.bin", dtype=dt, mode="r")
        self.val = np.memmap(path / "val.bin", dtype=dt, mode="r")
        self.seq, self.dev = seq, device
        # --data-mode stream (world_change_mix): train.bin = S streams of L tokens (stream-major); batch row b of
        # micro-batch c reads stream g*B + b, sequence (c // G) mod n_seq (G = S // B stream groups), i.e. every row
        # continues its own stream across micro-batches and optimizer steps. self.ctr = micro-batch counter (set
        # from the step in the train loop, so --resume continues the streams exactly).
        self.mode, self.ctr, self.last_streams = mode, 0, None
        if mode == "stream":
            st = mj["train"]
            self.S, self.L, self.n_seq = int(st["streams"]), int(st["stream_len"]), int(st["seqs_per_stream"])
            assert mj["seq"] == seq, f"stream data was built for --seq {mj['seq']}"
            assert self.S * self.L == len(self.train)
        # difficulty buckets (build_curriculum.py); phase k trains on the k+1 easiest buckets
        self.buckets = sorted(path.glob("curr_*.bin")) if curriculum else []
        self.buckets = [np.memmap(p, dtype=dt, mode="r") for p in self.buckets]
        self.phase = len(self.buckets) - 1
        # validation per difficulty tier (build_curriculum.py), reported for every run
        self.val_tiers = [np.memmap(p, dtype=dt, mode="r") for p in sorted(path.glob("val_curr_*.bin"))]

    def tier_batch(self, k, B, gen):
        d = self.val_tiers[k]
        x = torch.from_numpy(np.stack([d[i:i + self.seq + 1].astype(np.int64)
                                       for i in gen.randint(0, len(d) - self.seq - 1, B)]))
        x = x.pin_memory().to(self.dev, non_blocking=True) if self.dev == "cuda" else x
        return x[:, :-1], x[:, 1:]

    def batch(self, B, split="train", gen=None):
        gen = gen or np.random
        if split == "train" and self.buckets:
            sizes = np.array([len(b) for b in self.buckets[:self.phase + 1]], dtype=np.float64)
            which = gen.choice(len(sizes), B, p=sizes / sizes.sum())
            rows = [self.buckets[w][i:i + self.seq + 1] for w in which
                    for i in [gen.randint(0, len(self.buckets[w]) - self.seq - 1)]]
        elif split == "train" and self.mode == "stream":
            assert self.S % B == 0, f"--batch {B} must divide the {self.S} streams"
            G = self.S // B
            g, j = self.ctr % G, (self.ctr // G) % self.n_seq
            self.ctr += 1
            rows = [self.train[s * self.L + j * self.seq: s * self.L + j * self.seq + self.seq + 1]
                    for s in range(g * B, (g + 1) * B)]
            self.last_streams = (list(range(g * B, (g + 1) * B)), [j == 0] * B)   # (fastmem: stream ids, starts)
        else:
            d = self.train if split == "train" else self.val
            rows = [d[i:i + self.seq + 1] for i in gen.randint(0, len(d) - self.seq - 1, B)]
        x = torch.from_numpy(np.stack([r.astype(np.int64) for r in rows]))
        x = x.pin_memory().to(self.dev, non_blocking=True) if self.dev == "cuda" else x
        return x[:, :-1], x[:, 1:]


@torch.no_grad()
def tier_losses(model, data, batches, B, device):
    """Validation loss on the easy / medium / hard stories separately."""
    out = {}
    for k in range(len(data.val_tiers)):
        gen = np.random.RandomState(4321 + k)
        tot = 0.0
        for _ in range(batches):
            x, y = data.tier_batch(k, B, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                tot += model(x, y)[1].item()
        out[f"val_t{k}"] = tot / batches
    return out


@torch.no_grad()
def newborn_stats(model):
    """Expert growth health (routed brain, first routed layer): median traffic (load_ema, 1 = average) and the
    share of dead experts (load < 0.05) of the original experts and of the newborns by age in steps."""
    mlp = next((b.mlp for b in model.blocks if isinstance(b.mlp, RoutedMLP)), None)
    if mlp is None or not (mlp.age < 10**5).any():
        return None
    age, le = mlp.age.float(), mlp.load_ema.float()
    out = {}
    for name, sel in (("old", age >= 10**5), ("0-500", age < 500), ("500-1k", (age >= 500) & (age < 1000)),
                      ("1k-2k", (age >= 1000) & (age < 2000)), ("2k+", (age >= 2000) & (age < 10**5))):
        if sel.any():
            out[name] = dict(n=int(sel.sum()), load=le[sel].median().item(), dead=(le[sel] < 0.05).float().mean().item())
    return out


def free_cache(device):
    """After growth: the old (smaller) tensors and optimizer state stay in PyTorch's cache, and with repeated
    births the fragmented cache filled the GPU (v4b: ~22 GB held for ~11 GB used, speed 38k -> 17k tok/s)."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_rng_state(device):
    """Every RNG the training loop draws from (--resume): python, numpy (the data sampler), torch CPU / CUDA."""
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") and torch.cuda.is_available() else None)


def set_rng_state(st, device):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"].cpu())
    if st.get("cuda") is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def save_checkpoint(path, obj):
    """Atomic: written to a temporary file, then renamed over the old checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def send_to_trash(path):
    """Windows: to the Recycle Bin (SHFileOperationW, FOF_ALLOWUNDO), so a pruned checkpoint can be restored.
    Elsewhere: os.remove."""
    path = os.path.abspath(str(path))
    if sys.platform != "win32":
        os.remove(path)
        return
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
    op = SHFILEOPSTRUCTW(None, FO_DELETE, path + "\0", None,
                         FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI, False, None, None)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if rc or op.fAnyOperationsAborted:
        raise OSError(f"SHFileOperationW failed (code {rc}) for {path}")


def prune_ckpts(run, keep, trash=None):
    """Keep only the newest `keep` numbered checkpoints (ckpt_<step>.pt) of a run; older ones go to the Recycle Bin.
    last.pt / final.pt are never touched; keep <= 0 keeps everything. Returns the removed paths."""
    if keep is None or keep <= 0:
        return []
    trash = trash or send_to_trash
    numbered = []
    for p in Path(run).glob("ckpt_*.pt"):
        s = p.stem[len("ckpt_"):]
        if s.isdigit():
            numbered.append((int(s), p))
    numbered.sort()
    removed = []
    for _, p in numbered[:-keep]:
        try:
            trash(p)
            removed.append(p)
            print(f"keep-ckpts {keep}: moved {p.name} to the Recycle Bin", flush=True)
        except Exception as e:  # never kill a training run over housekeeping
            print(f"keep-ckpts: could not remove {p.name}: {e}", flush=True)
    return removed


def apply_vram_savers(model, args):
    """The --vram-* savers (activation checkpointing; same maths, no state change). Set on the classes, so experts /
    heads born later are covered too; every flag is set explicitly (off = the plain code path)."""
    import loop_brain as _lbm
    from blob_memory import BlobMemory
    from loop_brain import LoopBrain as _LB
    from spiking_lm import RMSNorm as _RN, RoutedMLP as _RM, SSDMixer as _SM
    allv = bool(getattr(args, "vram_savers", 0))
    on = lambda n: allv or bool(getattr(args, n, 0))
    _RN.ckpt = on("vram_ckpt_norm")
    _SM.ckpt = on("vram_ckpt_mixer")
    _RM.ckpt_experts = on("vram_ckpt_experts")
    _LB.ckpt_concept = on("vram_ckpt_concept")
    BlobMemory.ckpt_dist = on("vram_ckpt_gmdist")
    _lbm.READOUT_LEAN = on("vram_readout_lean")
    bm = getattr(model, "bm", None)
    rows = int(getattr(args, "vram_rerank_rows", 0) or 0) or (1024 if allv else 0)
    if bm is not None:
        bm.rerank_rows = rows
    names = [n for n, v in (("norm", _RN.ckpt), ("mixer", _SM.ckpt), ("experts", _RM.ckpt_experts),
                            ("concept", _LB.ckpt_concept), ("gmdist", BlobMemory.ckpt_dist),
                            ("readout-lean", _lbm.READOUT_LEAN)) if v]
    if names or rows:
        print(f"VRAM savers: checkpointed {', '.join(names) or '-'}" + (f"; re-rank rows {rows}" if rows else ""),
              flush=True)


def blob_memory_kwargs(args, tok, final):
    """The blob memory's LoopBrain keyword arguments from the --gm flags (plan_v2/25d; moved here unchanged from
    main() so a --resume can swap the memory kind: memory_api.py)."""
    kw = {}
    assert not (args.mem_shared or args.mem_private or args.mem_coarse or args.mem_gate), \
        "--gm 1 needs --mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-gate 0 (the old memory is exclusive)"
    assert args.gm_dv in (0, args.d), "--gm-dv must equal --d (contents at model width)"
    tg, tp = (int(v) for v in args.gm_topk.split(","))
    if args.gm_preset == "25d":  # (the decided phase-2 defaults; explicit flags of the others still apply)
        args.gm_gate, args.gm_track_sigma, args.gm_rerank_full = 1, 1, 1
        args.gm_write = "probation" if args.gm_write == "off" else args.gm_write
        args.gm_content_opt, args.gm_lr = "adagrad", (0.01 if args.gm_lr == 0.03 else args.gm_lr)
        args.gm_kernel, args.gm_use_A = "gauss", 0          # (the oracle: Gaussian top-8, c 0.1, no A)
    if args.gm_write == "probation" and not args.gm_probation_rate and not args.gm_probation_cap:
        args.gm_probation_rate = args.gm_cand_rate * args.batch * args.seq * args.accum
    gg, gp = (float(v) for v in args.gm_gamma.split(","))
    eot_id = tok.token_to_id("<|endoftext|>")
    kw.update(gm=1, gm_global=args.gm_global, gm_private=args.gm_private, gm_key=args.gm_key,
              gm_max_experts=final.get("experts", 0), gm_cand=args.gm_cand, gm_topk_g=tg, gm_topk_p=tp,
              gm_search=args.gm_search, gm_cells=args.gm_cells, gm_cell_cap=args.gm_cell_cap,
              gm_probe=args.gm_probe, gm_qcap=args.gm_qcap, gm_cell_topk=args.gm_cell_topk,
              gm_read_mode=args.gm_read_mode, gm_read_p=args.gm_read_p,
              gm_compute_frac=args.gm_compute_frac, gm_tau=args.gm_tau, gm_kernel_c=args.gm_kernel_c,
              gm_h0=args.gm_h0, gm_priv_bf16=args.gm_priv_bf16, gm_match_grad=args.gm_match_grad,
              gm_inj_scale=args.gm_inj_scale, gm_lr=args.gm_lr, gm_beta2=args.gm_beta2,
              gm_clip=args.gm_clip, gm_track_xf=args.gm_track_xf,
              gm_probation_rate=args.gm_probation_rate, gm_probation_life=args.gm_probation_life,
              gm_seed=args.gm_seed)
    p2 = dict(gm_gate=args.gm_gate, gm_explore=args.gm_explore, gm_buf=args.gm_buf, gm_gate_lr=args.gm_gate_lr,
              gm_gate_clip=args.gm_gate_clip, gm_write=args.gm_write, gm_cand_rate=args.gm_cand_rate,
              gm_write_w=args.gm_write_w, gm_private_rate=args.gm_private_rate,
              gm_promo_gamma=args.gm_promo_gamma, gm_promo_theta=args.gm_promo_theta,
              gm_promo_budget=args.gm_promo_budget, gm_promo_rank=args.gm_promo_rank,
              gm_promo_min_returns=args.gm_promo_min_returns, gm_prob_lookup=args.gm_prob_lookup,
              gm_probation_cap=args.gm_probation_cap, gm_probation_dim=args.gm_probation_dim,
              gm_probation_dtype=args.gm_probation_dtype, gm_probation_log=args.gm_probation_log,
              gm_gamma_g=gg, gm_gamma_p=gp, gm_theta_m=args.gm_theta_m, gm_protect_age=args.gm_protect_age,
              gm_protect_frac=args.gm_protect_frac, gm_strength_half=args.gm_strength_half,
              gm_track_sigma=args.gm_track_sigma, gm_rerank_full=args.gm_rerank_full,
              gm_rerank_dtype=args.gm_rerank_dtype, gm_rerank_top=args.gm_rerank_top, gm_v_init=args.gm_v_init,
              gm_content_opt=args.gm_content_opt, gm_eot=-1 if eot_id is None else int(eot_id),
              gm_kernel=args.gm_kernel, gm_kernel_cg=args.gm_kernel_cg, gm_kernel_dist=args.gm_kernel_dist,
              gm_use_A=args.gm_use_A, gm_gate_open0=args.gm_gate_open0, gm_gate_warm=args.gm_gate_warm)
    dflt = LoopBrain.GM_DEFAULTS
    kw.update({k: v for k, v in p2.items() if v != dflt[k]})   # (in cfg only when not at the default)
    return kw


def wave_memory_kwargs(args, tok=None, final=None):
    """The wave memory's LoopBrain keyword arguments (wave_memory.py; every --wave-* flag is a wave_* cfg key)."""
    from wave_memory import WAVE_DEFAULTS
    assert not (args.mem_shared or args.mem_private or args.mem_coarse or args.mem_gate), \
        "--memory wave needs --mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-gate 0 (the old memory is exclusive)"
    kw = {k: getattr(args, k) for k in WAVE_DEFAULTS if hasattr(args, k)}
    if args.wave_mode == "stub":
        return dict(memory="wave", wave_mode="stub")
    eot = tok.token_to_id("<|endoftext|>") if tok is not None else None
    kw.update(wave_eot=-1 if eot is None else int(eot),
              wave_max_experts=max(int(args.wave_max_experts or 0), int((final or {}).get("experts", 0))))
    return dict(memory="wave", **kw)


def fastmem_memory_kwargs(args, final=None):
    """FastMem's LoopBrain keyword arguments (fastmem.py; every --fastmem-* flag is a fastmem_* cfg key)."""
    from fastmem import FASTMEM_DEFAULTS
    assert not (args.mem_shared or args.mem_private or args.mem_coarse or args.mem_gate),         "--memory fastmem needs --mem-shared 0 --mem-private 0 --mem-coarse 0 --mem-gate 0 (the old memory is exclusive)"
    kw = {k: getattr(args, k) for k in FASTMEM_DEFAULTS if hasattr(args, k)}
    kw["fastmem_max_experts"] = max(int(args.fastmem_max_experts or 0), int((final or {}).get("experts", 0)))
    return dict(memory="fastmem", **kw)


def ckpt_memory_kind(cfg):
    """The memory kind of a checkpoint's model cfg (memory_api.MEMORY_KINDS)."""
    if cfg.get("memory") in ("wave", "fastmem"):
        return cfg["memory"]
    return "blob" if cfg.get("gm") else "none"


def is_memory_cfg_key(k):
    from wave_memory import WAVE_DEFAULTS
    from fastmem import FASTMEM_DEFAULTS
    return k in ("gm", "memory") or k in LoopBrain.GM_DEFAULTS or k in WAVE_DEFAULTS or k in FASTMEM_DEFAULTS


def load_brain_state(model, state, args, swap=False):
    """--resume: the model state. As before (strict) when the memory is in it; a brain-only checkpoint
    (--memory-embed 0) or a memory swap loads the brain strictly and the memory from --memory-load / the checkpoint's
    <name>.memory.pt / fresh."""
    bm = getattr(model, "bm", None)
    has_mem = any(k.startswith(MEM_PREFIX) for k in state)
    if swap:
        state = {k: v for k, v in state.items() if not k.startswith(MEM_PREFIX)}
    if bm is None or (has_mem and not swap):
        model.load_state_dict(state)
        return
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(MEM_PREFIX)]
    assert not unexpected and not bad, f"--resume: brain state mismatch (unexpected {unexpected[:5]}, missing {bad[:5]})"
    side = memory_path(args.resume)
    if not swap and not args.memory_load and side.exists():
        load_memory_file(side, model)
        print(f"memory: loaded {side} (a brain-only checkpoint)", flush=True)
    elif not args.memory_load and missing:
        print(f"memory: {bm.kind} starts fresh (not in {args.resume})", flush=True)


def build_opt(model, args):
    if args.opt == "brain":
        # maturity time constant: 1% of optimizer steps (effective batch = batch x accum)
        tau = max(1, int(0.01 * args.tokens / (args.batch * args.accum * args.seq)))
        opt = BrainOpt(model, args.lr, args.wd, muon=bool(args.muon), lazy=bool(args.lazy),
                       maturity=args.maturity, tau=tau)
        if memory_owns_params(model):  # (--memory wave: the memory's parameters have their own AdamW, wave_memory.py)
            opt.groups = {n: g for n, g in opt.groups.items() if not n.startswith(MEM_PREFIX)}
        return opt
    return make_opt(model, args.lr, args.wd)


def memory_owns_params(model):
    """True when the memory module steps its own parameters (the wave memory): they are then outside the brain's
    optimizer and the brain's gradient clip."""
    bm = getattr(model, "bm", None)
    return bm is not None and bm.kind in ("wave", "fastmem")


def brain_params(model):
    """(name, parameter) of the brain: everything except the memory gate's router (plan_v2/22: its own AdamW and clip)
    and, for --memory wave, the memory's own parameters (bm.*: its own AdamW and clip)."""
    skip = ("mem_gate.",) + ((MEM_PREFIX,) if memory_owns_params(model) else ())
    return [(n, p) for n, p in model.named_parameters() if not n.startswith(skip)]


def make_opt(model, lr, wd):
    wd_ok = lambda n, p: p.dim() >= 2 and not n.startswith(("emb.", "mem.keys.")) and "router_conn" not in n
    decay = [p for n, p in brain_params(model) if wd_ok(n, p)]
    other = [p for n, p in brain_params(model) if not wd_ok(n, p)]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def loop_report(model, data, args, batches=4):
    """Looped brain: val loss vs number of passes, loss after each pass, and the memory knock-out."""
    model.eval()
    core = model.blocks[0]
    out = {}
    # (plan_v2/21) the same batches, run --mem-probe-batch sequences at a time: the (B, T, V) read-outs of the long
    # loops are the eval peak; equal chunks, so a batch's loss is the mean of its chunks' losses
    cb = getattr(args, "mem_probe_batch", 0) or args.eval_batch
    cb = cb if cb < args.eval_batch and args.eval_batch % cb == 0 else args.eval_batch
    chunks = lambda x, y: [(x[i:i + cb], y[i:i + cb]) for i in range(0, x.size(0), cb)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
        def val(K):
            gen = np.random.RandomState(4242)
            tot = 0.0
            for _ in range(batches):
                x, y = data.batch(args.eval_batch, "val", gen)
                model.K = K
                if cb == args.eval_batch:
                    tot += model(x, y)[1].item()
                else:
                    tot += sum(model(xc, yc)[1].item() for xc, yc in chunks(x, y)) / (args.eval_batch // cb)
            return tot / batches
        out["loss_at_K"] = {K: val(K) for K in (1, 2, 4, 6, 8, 10, 12, 16)}
        if model.cfg.get("halt"):  # loop until sure: each word stops when its stop probability passes 0.5
            gen = np.random.RandomState(4242)
            hs = [model.halt_eval(xc, yc) for _ in range(batches)
                  for xc, yc in chunks(*data.batch(args.eval_batch, "val", gen))]
            out["halt"] = dict(loss=sum(h["halt_loss"] for h in hs) / len(hs),
                               passes=sum(h["halt_passes"] for h in hs) / len(hs),
                               expected_passes=sum(h["halt_expected_passes"] for h in hs) / len(hs),
                               passes_hist=[sum(c) for c in zip(*(h["passes_hist"] for h in hs))])
        x, y = data.batch(args.eval_batch, "val", np.random.RandomState(4242))
        hier = getattr(model, "hier", None)
        if getattr(model, "think", False) or hier is not None:  # does the uncertainty predict that more passes help?
            out["think"] = model.think_diag(x, y, chunk=cb)
        if getattr(model, "think_in", None) is not None or (hier is not None and hier.has_inject):
            model.think_zero = True  # usefulness: the loss the concept write-back saves
            out["think_usefulness"] = val(args.k_mean) - out["loss_at_K"].get(args.k_mean, val(args.k_mean))
            model.think_zero = False
        mem = getattr(model, "mem", None)
        if mem is not None:  # memory knock-outs: shared reads = 0, private reads = 0, one level's reads = 0
            kos = ([("shared", "ko_shared", True)] if mem.S else []) + ([("private", "ko_private", True)] if mem.R else []) \
                + ([(f"level{l + 1}", "ko_levels", (l,)) for l in range(mem.L)] if mem.S else []) \
                + ([("read0", "read0", False)] if mem.read0 else [])                 + ([("coarse", "ko_coarse", True)] if mem.Sc else [])  # the token-keyed read before pass 1; coarse
            for name, attr, value in kos:
                old = getattr(mem, attr)
                setattr(mem, attr, value)
                out[f"loss_mem_ko_{name}"] = val(args.k_mean)
                setattr(mem, attr, old)
        out["tick_loss"] = model.tick_losses(x, y, 10, chunk=cb)
        if core.slots:  # workspace knock-out probes
            for mode in ("last", "all"):
                model.knockout = mode
                out[f"loss_ws_knockout_{mode}"] = val(args.k_mean)
            model.knockout = None
        elif core.state == "carry":
            core.state = "reset"
            out["loss_state_knockout"] = val(args.k_mean)
            core.state = "carry"
        model.trace_tokens = [(0, 100), (0, 300), (1, 200)]   # per-pass expert choices of three words
        model.K = 10
        model(x[:2])
        out["expert_trace"] = model.trace
        model.trace_tokens, model.trace = None, None
    model.K = args.k_mean
    model.train()
    return out


@torch.no_grad()
def reset_mem_key_rows(opt, model, name, sl):
    """Fresh optimizer moments for sub-key rows the memory's balancing just split or re-seeded (index sl of the
    parameter `name`): zero momentum, the median second moment. BrainOpt (state by name) or torch AdamW."""
    p = dict(model.named_parameters()).get(name)
    if p is None or opt is None:
        return
    st = opt.state.get(name, {}) if hasattr(opt, "groups") else opt.state.get(p, {})
    for mk, vk in (("m", "v"), ("exp_avg", "exp_avg_sq")):
        if mk in st:
            st[mk][sl] = 0.0
        if vk in st:
            st[vk][sl] = st[vk].median()


@torch.no_grad()
def memory_metrics(model, probes):
    """Section-6 metrics of the memory (plan_v2/12): probes (stability vs the freeze and the last eval, input drift,
    offset overlap, margins, collisions, gates, private hit rate, routing drift; repair if on), usage, the training
    margins, the last balancing and the freeze. Returns (metrics, alarm)."""
    mem = model.mem
    if probes is not None:
        out, alarm = probes.measure(model)
    else:
        out, alarm = {f"mem_{k}": v for k, v in mem.usage_stats().items()}, False
    if mem.margin_stats:
        out.update({f"mem_margin_{k}": v.item() for k, v in mem.margin_stats.items()})
    for key in ("balance", "freeze", "freeze_step"):
        if key in mem.stats:
            out[f"mem_{key}"] = mem.stats[key]
    if "restart" in mem.stats:  # (not "mem_restart": that is a cfg key in the same record)
        out["mem_restart_last"] = mem.stats["restart"]
    if mem.life:  # the life cycle (plan_v2/21): phase, lambda, beta, Lambda(t), switch reason
        st = mem.life_state
        out["mem_life_state"] = dict(phase=st["phase"], decay=st["decay"], beta=st["beta"], Lam=st["Lam"],
                               reason=st["reason"], t_switch=st["t_switch"], stab_run=st["stab_run"])
    return out, alarm


def memory_line(m):
    """The memory part of the eval log line."""
    f = lambda v: "/".join(f"{x:.3f}" for x in v) if isinstance(v, (list, tuple)) else f"{v:.3f}"
    s = "  | mem"
    if "mem_reuse_overlap" in m:  # (a dict pass -> overlap: logged as its mean)
        m = dict(m, mem_reuse_ovl_mean=sum(m["mem_reuse_overlap"].values()) / len(m["mem_reuse_overlap"]))
    for key, name in (("mem_stab_w", "stab_w"), ("mem_stab_w_pass0", "pass0"), ("mem_stab_set", "set"),
                      ("mem_A_stab_w", "A"), ("mem_last_stab_w", "vs last"), ("mem_last_stab_w_open", "vs last open"),
                      ("mem_stab_w_open", "stab_w open"), ("mem_input_drift_by_level", "drift"),
                      ("mem_offset_overlap", "offset ovl"), ("mem_reuse_ovl_mean", "reuse ovl"), ("mem_gap_mean", "gaps"),
                      ("mem_margin_active", "margin act"),
                      ("mem_collisions_by_level", "coll"), ("mem_subkeys_busy", "busy"), ("mem_subkeys_dead", "dead"),
                      ("mem_private_hit", "priv hit"), ("mem_private_stab", "priv stab"), ("mem_routing_same", "routing same"),
                      ("mem_repairs_by_level", "repairs")):
        if key in m:
            s += f" {name} {m[key] if isinstance(m[key], int) else f(m[key])}"
    if "mem_life_state" in m:  # (plan_v2/21)
        lf = m["mem_life_state"]
        s += f" | life {lf['phase']} lambda {lf['decay'] or 0:.2e} beta {lf['beta'] or 1:.4f} Lambda {lf['Lam']:.2f}"
        if lf["reason"]:
            s += f" (switch: {lf['reason']} at {lf['t_switch']})"
        if "mem_vnorm_by_wcount" in m:
            s += f" |V| by count {f(m['mem_vnorm_by_wcount'])}"
    if "mem_restart_last" in m:
        s += " restarted " + "/".join(str(v) for v in m["mem_restart_last"].values())
    for key, name in (("mem_last_stab_w_c", "coarse vs last"), ("mem_stab_w_c", "coarse stab_w"),
                      ("mem_subkeys_dead_c", "coarse dead"), ("mem_gate_mean", "gate"), ("mem_gate_mean_c", "gate_c")):
        if key in m:
            s += f" {name} {f(m[key])}"
    if m.get("mem_frozen"):
        s += " [frozen]"
    if m.get("mem_alarm"):
        s += " MEMORY DRIFT"
    return s


def _auc(score, label):
    """Rank AUC of score for the boolean label (nan when one class is empty)."""
    score, label = score.double().flatten(), label.flatten()
    r = torch.empty_like(score)
    r[score.argsort()] = torch.arange(len(score), dtype=score.dtype, device=score.device)
    pos = label.sum().item()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return ((r[label].sum().item() - pos * (pos - 1) / 2) / (pos * neg))


def gate_train_stats(model):
    """The memory gate's training diagnostics of the last training forward (plan_v2/22 section 7; tensors, read here, on
    logging steps only): the router BCE and its calibration, per read: c, the open count, the fraction above theta_k,
    theta_k, the score quantiles, the open-set overlap with the previous read, g_bar on the top-c vs explored positions,
    and (gate_diag) the residual -<dL/dM, M> on top-c vs explored and its rank correlation with s."""
    sts = getattr(model, "gate_stats", None) or []
    out = {}
    if getattr(model, "gate_loss", None) is not None:
        out["gate_bce"] = model.gate_loss.item()
    v = lambda t: t.item() if torch.is_tensor(t) else t
    per = []
    for st in sts:
        d = {key: v(st[key]) for key in ("k", "c", "n_open", "n_c", "n_e", "above", "theta", "bce", "calib_pred",
                                        "calib_gbar", "gbar_top", "gbar_explore", "overlap_prev") if key in st}
        if not bool(st["theta_valid"]):
            d["above"] = float("nan")
        d["q"] = torch.quantile(st["s"].float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=st["s"].device)).tolist()
        if "resid" in st:
            r, s_all, isA = st["resid"], st["s_all"], st.get("resid_isA")
            if isA is not None:
                d["resid_top"] = r[isA].mean().item()
                d["resid_explore"] = r[~isA & (r != 0)].mean().item() if bool((~isA & (r != 0)).any()) else float("nan")
            nz = r != 0  # (closed positions have no read: M = 0 there)
            if int(nz.sum()) > 2:
                rr, ss = r[nz].double(), s_all[nz].double()
                rk = lambda t: t.argsort().argsort().double()
                a, b = rk(rr), rk(ss)
                a, b = a - a.mean(), b - b.mean()
                d["resid_spearman_s"] = (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()
        per.append(d)
    if per:
        out["gate_reads"] = per
        out["gate_c"] = per[0]["c"]
    return out


@torch.no_grad()
def gate_report(model, data, args, batches=2):
    """The memory gate's eval report (plan_v2/22 section 7) on validation batches at K = --k-mean: the loss of every
    eval gating mode (learned / random / inverted / oracle at the learned gate's per-read open count, forced open,
    memory off); the measured benefit b_t = CE_t(off) - CE_t(open); the benefit capture sum_t open_t b_t /
    sum_t max(b_t, 0) of learned / random / inverted / oracle (open_t = the share of reads that open t); the AUC of the
    mean score for "b_t in the top 20%"; the top-20% share of the positive benefit (the Phase-0 statistic); the
    knockout CE(off) - CE(learned) on the positions the learned gate opens (open_t >= 0.5) vs closes; the open rate
    per pass."""
    model.eval()
    model.K = args.k_mean
    cb = getattr(args, "mem_probe_batch", 0) or args.eval_batch
    cb = cb if cb < args.eval_batch and args.eval_batch % cb == 0 else args.eval_batch
    gen = np.random.RandomState(4243)
    acc = {key: [] for key in ("b", "s", "ko")}
    ce_m = {m: [] for m in GATE_MODES}
    frac = {m: [] for m in ("learned", "random", "inverted", "oracle")}
    by_pass = {}
    old = (model.mem_gate_eval, model.gate_eval_rate, model.gate_oracle_b)

    def run(mode, xc, yc, oracle=None):
        model.mem_gate_eval, model.gate_eval_rate, model.gate_oracle_b = mode, None, oracle
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
            logits = model(xc)
        ce = F.cross_entropy(logits.float().flatten(0, 1), yc.flatten(), reduction="none")
        n = ce.numel()
        masks = [(r["k"], torch.ones(n, dtype=torch.bool, device=ce.device) if r["mask"] is None else r["mask"])
                 for r in model.gate_open]
        f = torch.stack([m.float() for _, m in masks]).mean(0) if masks else torch.zeros_like(ce)
        s = torch.stack([r["s"] for r in model.gate_open]).mean(0) if model.gate_open else torch.zeros_like(ce)
        return ce, f, s, masks

    try:
        for _ in range(batches):
            x, y = data.batch(args.eval_batch, "val", gen)
            for i in range(0, x.size(0), cb):
                xc, yc = x[i:i + cb], y[i:i + cb]
                ce_open = run("open", xc, yc)[0]
                ce_off = run("off", xc, yc)[0]
                b = ce_off - ce_open
                ce_l, f_l, s_l, masks = run("learned", xc, yc)
                for k, mk in masks:
                    by_pass.setdefault(k, []).append(mk.float().mean().item())
                ce_m["open"].append(ce_open)
                ce_m["off"].append(ce_off)
                ce_m["learned"].append(ce_l)
                frac["learned"].append(f_l)
                for mode in ("random", "inverted", "oracle"):
                    ce, f, _, _ = run(mode, xc, yc, oracle=b if mode == "oracle" else None)
                    ce_m[mode].append(ce)
                    frac[mode].append(f)
                acc["b"].append(b)
                acc["s"].append(s_l)
                acc["ko"].append(ce_off - ce_l)
    finally:
        model.mem_gate_eval, model.gate_eval_rate, model.gate_oracle_b = old
        model.train()
    b = torch.cat(acc["b"]).double()
    pos = b.clamp(min=0).sum().item()
    out = {f"gate_loss_{m}": torch.cat(v).mean().item() for m, v in ce_m.items()}
    for m, v in frac.items():
        f = torch.cat(v).double()
        out[f"gate_capture_{m}"] = (f * b).sum().item() / max(pos, 1e-12)
        out[f"gate_open_rate_{m}"] = f.mean().item()
    n20 = max(1, int(round(0.2 * b.numel())))
    top = b.topk(n20).indices
    out["gate_top20_share"] = b[top].clamp(min=0).sum().item() / max(pos, 1e-12)
    lab = torch.zeros_like(b, dtype=torch.bool)
    lab[top] = True
    out["gate_auc_top20"] = _auc(torch.cat(acc["s"]), lab)
    ko, fl = torch.cat(acc["ko"]).double(), torch.cat(frac["learned"])
    op = fl >= 0.5
    out["gate_ko_open"] = ko[op].mean().item() if bool(op.any()) else float("nan")
    out["gate_ko_closed"] = ko[~op].mean().item() if bool((~op).any()) else float("nan")
    out["gate_open_by_pass"] = [sum(by_pass[k]) / len(by_pass[k]) for k in sorted(by_pass)]
    return out


def gate_line(m):
    """The memory gate's part of the eval log line."""
    s = ""
    if "gate_c" in m:
        rd = m.get("gate_reads", [])
        ab = [r["above"] for r in rd if r.get("above") == r.get("above")]
        s += f"  | gate c {m['gate_c']:.3f} bce {m.get('gate_bce', float('nan')):.4f}"
        if rd:
            s += f" open {sum(r['n_open'] for r in rd if 'n_open' in r) / max(1, sum('n_open' in r for r in rd)):.0f}"
        if ab:
            s += f" above-theta {sum(ab) / len(ab):.3f}"
    if "gate_capture_learned" in m:
        s += (f"  | gate eval: loss learned {m['gate_loss_learned']:.3f} open {m['gate_loss_open']:.3f} off "
              f"{m['gate_loss_off']:.3f} random {m['gate_loss_random']:.3f}; capture learned {m['gate_capture_learned']:.2f}"
              f" random {m['gate_capture_random']:.2f} oracle {m['gate_capture_oracle']:.2f}; top20 share "
              f"{m['gate_top20_share']:.2f} AUC {m['gate_auc_top20']:.2f}; ko open {m['gate_ko_open']:.3f} closed "
              f"{m['gate_ko_closed']:.3f}")
    return s


@torch.no_grad()
def gm_eval_delta(model, data, args, batches=None):
    """The blob memory's validation delta (eval mode: no writes, no store change): CE with every read knocked out minus
    CE with the memory on, on the usual validation batches and (--gm-heldout) on fixed windows of a held-out token
    file. Positive = the memory helps. (No answer / non-answer split: the trainer has no token tags.)"""
    bm = model.bm
    model.eval()
    model.K = args.k_mean
    out = {}
    sets = [("val", None)]
    if args.gm_heldout:
        sets.append(("heldout", np.memmap(args.gm_heldout, dtype=np.uint16, mode="r")))
    for name, arr in sets:
        gen = np.random.RandomState(4321)
        tot = {False: 0.0, True: 0.0}
        n = batches or max(1, args.eval_batches // 2)
        for _ in range(n):
            if arr is None:
                x, y = data.batch(args.eval_batch, "val", gen)
            else:
                ix = gen.randint(0, len(arr) - args.seq - 1, args.eval_batch)
                w = torch.from_numpy(np.stack([arr[i:i + args.seq + 1].astype(np.int64) for i in ix])).to(args.device)
                x, y = w[:, :-1], w[:, 1:]
            for ko in (False, True):
                bm.ko = ko
                rm = bm.read_mode
                if ko:
                    bm.read_mode = "none"
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    tot[ko] += model(x, y)[1].item()
                bm.read_mode, bm.ko = rm, False
        out[f"gm_{name}_ko_delta"] = (tot[True] - tot[False]) / n
    model.train()
    return out


class DiagWriter:
    """--diag-every: the line's tensors are flattened into ONE device vector and copied to pinned host memory with a
    non_blocking copy + an event; the line is written at the NEXT diag point (or at the end), when the copy has long
    finished, so the training loop never waits on it."""

    def __init__(self, path, device):
        self.path, self.cuda = path, str(device).startswith("cuda") and torch.cuda.is_available()
        self.pending = None

    def push(self, step, host, tens):
        self.flush()
        keys, shapes, flat = [], [], []
        for k, v in tens.items():
            vs = v if isinstance(v, (list, tuple)) else [v]
            vs = [t for t in vs if torch.is_tensor(t)]
            if not vs:
                continue
            t = torch.stack([x.detach().float().reshape(-1) for x in vs]) if isinstance(v, (list, tuple)) \
                else v.detach().float().reshape(-1)
            keys.append(k)
            shapes.append((isinstance(v, (list, tuple)), tuple(t.shape)))
            flat.append(t.reshape(-1))
        vec = torch.cat(flat) if flat else torch.zeros(0)
        if self.cuda:
            buf = torch.empty(vec.shape, dtype=vec.dtype, pin_memory=True)
            buf.copy_(vec, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        else:
            buf, ev = vec.clone(), None
        self.pending = (step, host, keys, shapes, buf, ev)

    def flush(self):
        if self.pending is None:
            return
        step, host, keys, shapes, buf, ev = self.pending
        if ev is not None:
            ev.synchronize()
        rec, o = dict(step=step, **host), 0
        for k, (is_list, shp) in zip(keys, shapes):
            n = 1
            for s_ in shp:
                n *= s_
            vals = buf[o:o + n].tolist()
            o += n
            rec[k] = vals if (is_list or n > 1) else vals[0]
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.pending = None


def diag_collect(model, neuro):
    """The training-path diagnostics: memory, gates, experts, concepts, loop (device tensors + a few host numbers)."""
    host, t = {}, {}
    if getattr(model, "bm", None) is not None:
        t.update({f"mem_{k}": v for k, v in model.bm.diag_snapshot().items()})
    mlp = model.blocks[0].mlp
    host["experts"] = mlp.E
    if neuro is not None:
        host["births"] = neuro.births
    load = mlp.load_ema.float()
    p = load / load.sum().clamp(min=1e-12)
    t["expert_load_entropy"] = -(p * (p + 1e-12).log()).sum() / math.log(max(2, mlp.E))
    t["expert_dead_frac"] = (load < 0.01).float().mean()
    t["expert_near_dead_frac"] = (load < 0.1).float().mean()
    t["expert_load_max"] = load.max()
    dt = getattr(model, "diag_train_out", None) or {}
    if dt.get("pass_ce"):
        t["loop_pass_ce"] = dt["pass_ce"]
    if dt.get("overlap_lag1"):
        t["routing_overlap_lag1"] = dt["overlap_lag1"]
    if getattr(model, "diag_levels", None):
        t["concept_level_loss"] = [v.float().mean() for v in model.diag_levels]
    hs = getattr(model, "halt_stats", None)
    if hs:
        t["halt_expected_passes"] = hs["expected_passes"]
    return host, t


@torch.no_grad()  # (without it eval kept a full training graph: K passes x eval batch)
def evaluate(model, data, args, track=False):
    model.eval()
    if args.arch == "loop":
        model.K = args.k_mean
    gen = np.random.RandomState(1234)  # same validation batches every time
    spikes = model.spikes()
    for s in spikes:
        s.track = track
    tot = 0.0
    for i in range(args.eval_batches):
        x, y = data.batch(args.eval_batch, "val", gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
            _, loss = model(x, y)
        tot += loss.item()
        if i == 0:
            for s in spikes:
                s.track = False
    stats = {}
    if track and spikes and spikes[0].rate is not None:
        r = torch.cat([s.rate for s in spikes])
        stats = dict(fire=r.mean().item(), dead=(r < 1e-3).float().mean().item(),
                     saturated=(r > 0.95).float().mean().item())
    if data.val_tiers:
        stats.update(tier_losses(model, data, max(1, args.eval_batches // 4), args.eval_batch, args.device))
    noise = getattr(args, "route_noise", 0.0)
    if noise and noise != args.eval_route_noise:  # routing temperature: also the val loss at the training noise
        routed = [m for m in model.modules() if isinstance(m, RoutedMLP)]
        for m in routed:
            m.eval_noise = noise
        gen, tn = np.random.RandomState(1234), 0.0
        for _ in range(args.eval_batches):
            x, y = data.batch(args.eval_batch, "val", gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                tn += model(x, y)[1].item()
        for m in routed:
            m.eval_noise = args.eval_route_noise
        stats["val_at_train_noise"] = tn / args.eval_batches
    model.train()
    return tot / args.eval_batches, stats


@torch.no_grad()
def sample(model, tok, device, prompt="Once upon a time", n=200, temp=0.8):
    model.eval()
    ids = torch.tensor([tok.encode(prompt).ids], device=device)
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(ids[:, -512:])[:, -1].float()
        nxt = torch.multinomial(torch.softmax(logits / temp, -1), 1)
        ids = torch.cat([ids, nxt], 1)
        if nxt.item() == 0:
            break
    model.train()
    return tok.decode(ids[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--runs", default=str(HERE.parent / "runs"))
    ap.add_argument("--data", default=str(HERE.parent / "data" / "ts4k"))
    ap.add_argument("--data-mode", default="random", choices=["random", "stream"],
                    help="stream: each batch row continues its own stream across steps (world_change_mix layout)")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json path (default: <data>/tokenizer.json)")
    ap.add_argument("--mixer", default="ssd", choices=["ssd", "attn"])
    ap.add_argument("--spike", default="analog", choices=["analog", "graded", "ternary", "binary", "ngc", "ngc4"])
    ap.add_argument("--opt", default="adamw", choices=["adamw", "brain"], help="brain = BrainOpt (brain_opt.py)")
    ap.add_argument("--muon", type=int, default=1, help="BrainOpt: Muon for hidden matrices (1) or AdamW everywhere (0)")
    ap.add_argument("--lazy", type=int, default=1, help="BrainOpt: leave parameters without gradient untouched")
    ap.add_argument("--maturity", type=float, default=0.0, help="BrainOpt: newborn LR multiplier k (e.g. 4); 0 = off")
    ap.add_argument("--neuro", action="store_true", help="smooth growth: neurons born continuously from --start to final")
    ap.add_argument("--neuro-where", default="grad", choices=["grad", "uniform"])
    ap.add_argument("--neuro-protocol", type=int, default=1, help="1 = full birth protocol, 0 = ablation")
    ap.add_argument("--neuro-every", type=int, default=25)
    ap.add_argument("--neuro-t", default="0.05,0.6", help="births happen between these fractions of training")
    ap.add_argument("--renew-rate", type=float, default=1e-5, help="continual-backprop renewal rate per neuron per step")
    ap.add_argument("--arch", default="stack", choices=["stack", "loop"], help="loop = looped brain (loop_brain.py)")
    ap.add_argument("--k-mean", type=int, default=6, help="loop: mean number of passes")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--bptt", type=int, default=6, help="loop: backprop through the last N passes")
    ap.add_argument("--state", default="carry", choices=["carry", "reset", "none"], help="loop: internal state")
    ap.add_argument("--d-state", type=int, default=96)
    ap.add_argument("--refractory", type=int, default=1, help="loop: forbid re-using an expert on the next pass")
    ap.add_argument("--capacity", type=float, default=1.25, help="loop: expert capacity factor (0 = dropless)")
    ap.add_argument("--route-u", type=float, default=1e-3, help="balancer step size (bias update rate)")
    ap.add_argument("--route-aux", type=float, default=0.0, help="loop: use a load-balancing loss with this weight")
    ap.add_argument("--route-lift", type=float, default=0.0,
                    help="aux balance mode: raise the selection bias of experts below this fraction of the average "
                         "load (e.g. 0.5), so unpicked newborns get traffic (0 = off)")
    ap.add_argument("--bias-logit", type=int, default=0,
                    help="1 = the selection bias (balancer, newborn debt, lift) acts on the router logits, not on the "
                         "sigmoid scores (use a larger --route-u, e.g. 0.01)")
    ap.add_argument("--route-z", type=float, default=-1.0, help="router z-loss weight (-1 = default 1e-3)")
    ap.add_argument("--route-shield", type=int, default=0,
                    help="1 = experts still ramping in get no router gradient (left out of the z-loss and aux P)")
    ap.add_argument("--route-both", type=int, default=0,
                    help="1 = with --route-aux, also run the auxfree bias balancer (balance mode 'both')")
    ap.add_argument("--birth-mode", default="ramp", choices=["ramp", "split", "legacy"],
                    help="expert births: ramp (zero-output child ramps in) or split (twins share the parent's words)")
    ap.add_argument("--birth-delta", type=float, default=0.5,
                    help="newborn expert's router row = parent's + a random change of this size (x the row norm)")
    ap.add_argument("--route-noise", type=float, default=0.0,
                    help="training-only Gaussian noise (std, router-logit units) on which experts are selected; "
                         "gate weights and the balance loss stay clean (0 = off)")
    ap.add_argument("--route-noise-decay", type=float, default=0.8,
                    help="the noise falls linearly to 0 at this fraction of training (0 = constant, a temperature)")
    ap.add_argument("--eval-route-noise", type=float, default=0.0,
                    help="routing noise at evaluation (0 = clean, deterministic)")
    ap.add_argument("--slots", type=int, default=0, help="loop: workspace slots (0 = vector state)")
    ap.add_argument("--waves", action="store_true", help="loop: learned expert connectivity + fatigue instead of refractory")
    ap.add_argument("--ws-direct", action="store_true", help="loop: workspace v2 (direct path + workspace, learned balance)")
    ap.add_argument("--pops", type=int, default=1, help="loop: routed temporal populations (1 = one shared)")
    ap.add_argument("--head-group", type=int, default=0,
                    help="loop: pass-routed head groups of this many heads (0 = dense): the first N heads always "
                         "on, the rest (up to --heads) in groups that take turns over the passes")
    ap.add_argument("--deep-sup", type=float, default=0.0,
                    help="loop: weight of the LM loss on every earlier backpropagated pass (0 = the old single "
                         "sampled pass at 0.2)")
    ap.add_argument("--halt", type=int, default=0, help="loop: 1 = learned 'stop now' head (PonderNet-style)")
    ap.add_argument("--halt-prior", type=float, default=0.2,
                    help="halting: geometric prior's stop probability per pass (0.2 = ~5 passes)")
    ap.add_argument("--halt-beta", type=float, default=0.01, help="halting: weight of the KL to the prior")
    ap.add_argument("--think-w", type=float, default=0.0,
                    help="loop: weight of the thinking-concept head (workspace -> bag of the next --think-n words)")
    ap.add_argument("--think-n", type=int, default=8, help="thinking concept: how many next words")
    ap.add_argument("--think-target", default="bag", choices=["bag", "latent"],
                    help="thinking concept target: bag of the next words, or the latent of the next --think-n words")
    ap.add_argument("--think-use", type=int, default=0,
                    help="latent: write the predicted concept back into the next pass's core input (usefulness)")
    ap.add_argument("--think-fb", type=int, default=0,
                    help="1 = write the thought's uncertainty back into the workspace each pass")
    ap.add_argument("--head-groups", type=int, default=0,
                    help="loop: number of routed head groups (0 = groups of --head-group heads); e.g. 6 with "
                         "--heads 36 --head-group 6 = 6 groups of 5, so K=6 evaluation uses each group once")
    ap.add_argument("--ckpt-route", type=int, default=0,
                    help="loop: recompute the router (tokens x experts tensors) in backward to save memory")
    ap.add_argument("--readout-frac", type=float, default=1.0,
                    help="loop: the per-pass read-outs (deep supervision, halting) use this random fraction of the "
                         "words, drawn once per forward (1 = all)")
    # ---- memory system (plan_v2/12_memory_full_plan.md; memory.py). Everything off by default.
    ap.add_argument("--concepts", default="",
                    help="loop: hierarchical concept heads with these horizons, e.g. 16,32,64,128 (replaces "
                         "--think-target latent; --think-w / --think-fb / --think-use act on the hierarchy; needs --slots)")
    ap.add_argument("--concept-weights", default="1,1,0.5,0.5",
                    help="loss weight per concept level (the plan's --concept-w: that name is taken by --concept)")
    ap.add_argument("--concept-inject-rank", type=int, default=64,
                    help="rank of the concepts' write-back into the next pass (0 = full d x d)")
    ap.add_argument("--mem-shared", type=int, default=0,
                    help="shared product-key pool: S sub-keys per half (rows = levels x S^2); 0 = off")
    ap.add_argument("--mem-heads", type=int, default=3, help="memory heads (translators) per concept level")
    ap.add_argument("--mem-topm", type=int, default=8, help="product keys: top-m per half")
    ap.add_argument("--mem-topk", type=int, default=6, help="product keys: rows read per head")
    ap.add_argument("--mem-tau", type=float, default=0.03, help="softmax temperature of the read weights")
    ap.add_argument("--mem-read0", type=int, default=1, help="1 = also read before the first pass (from s_0: keyed by the token alone); 0 = the first memory use is after pass 1")
    ap.add_argument("--mem-private", type=int, default=0, help="private memory rows per expert (0 = off)")
    ap.add_argument("--mem-lr", type=float, default=0.01, help="MemOpt (row-Adagrad) learning rate of the tables")
    ap.add_argument("--mem-s0", default="auto", help="Adagrad prior S0 (absolute), or auto = 100 x median G^2")
    ap.add_argument("--mem-clip", type=float, default=0.05, help="MemOpt: clip of each row's update norm")
    ap.add_argument("--mem-write-start", type=int, default=200, help="table writes start at this optimizer step")
    ap.add_argument("--mem-dv", type=int, default=256, help="memory value width")
    ap.add_argument("--mem-freeze", type=float, default=0.6,
                    help="addressing freezes at the first step >= this fraction of training with no births left")
    ap.add_argument("--mem-ema", type=float, default=0.999, help="EMA of the translator / keys / norm statistics")
    ap.add_argument("--mem-margin", default="0.01,0.01", help="sub-key margin loss lambda,delta (infancy)")
    ap.add_argument("--mem-addr-sg", type=int, default=1, help="1 = after the freeze the addressing input is sg(a)")
    ap.add_argument("--mem-probes", type=int, default=32, help="drift probes: validation sequences per probe set")
    ap.add_argument("--mem-guard", type=float, default=0.95, help="alarm when the w-weighted stability falls below")
    ap.add_argument("--mem-repair", type=float, default=0.0, help="repair blend beta after the freeze (0 = measure only)")
    ap.add_argument("--mem-repair-usage", type=float, default=0.1,
                    help="repair only rows whose write count is below this quantile")
    ap.add_argument("--spill-gb", type=float, default=21.5,
                    help="VRAM spill guard: on the first eval with reserved GB above this AND tok/s down by "
                         "--spill-drop, the cache is freed")
    ap.add_argument("--spill-drop", type=float, default=0.35,
                    help="VRAM spill guard: a tok/s drop below the running median by this fraction counts as a spill")
    ap.add_argument("--mem-probe-every", type=int, default=1,
                    help="drift probes on every N-th evaluation (and the last); the others log the usage only")
    # milder memory (plan_v2/14_memory_speed_plan.md): each off by default (= as before)
    ap.add_argument("--mem-readdr-every", type=int, default=1,
                    help="new addresses only on every N-th thought read from --mem-readdr-start on; the reads in "
                         "between reuse the previous read's indices, weights still from their own thought (1 = every read)")
    ap.add_argument("--mem-readdr-start", type=int, default=4,
                    help="with --mem-readdr-every: the reads after passes 1 .. P (pass index < P) always address anew")
    ap.add_argument("--mem-margin-passes", type=int, default=0,
                    help="the sub-key margin on this many random reads of the gradient window per forward (0 = all)")
    ap.add_argument("--mem-balance-every", type=int, default=250, help="sub-key balancing every N optimizer steps")
    ap.add_argument("--mem-balance-max", type=float, default=0.0,
                    help="balancing changes at most this fraction of the keys per codebook per event (0 = no cap)")
    ap.add_argument("--mem-balance-dead-max", type=float, default=0.0,
                    help="a separate cap for dead-key re-seeds: at most this fraction of the keys per codebook per "
                         "event (0 = dead re-seeds share --mem-balance-max with the splits)")
    ap.add_argument("--mem-balance-keep", type=int, default=0,
                    help="1 = never re-seed a key that was used since the last balancing event")
    # memory life cycle + key restart + coarse level + neighbour writes (plan_v2/21_lifecycle_plan.md): each off by
    # default (= as before)
    ap.add_argument("--mem-life", type=int, default=0,
                    help="1 = memory life cycle: decaying, rewritable rows in infancy, an adaptive switch, a ramp, "
                         "then the freeze (replaces --mem-freeze)")
    ap.add_argument("--mem-decay", type=float, default=1e-3, help="life cycle: row decay per step in infancy")
    ap.add_argument("--mem-acc-beta", type=float, default=0.99,
                    help="life cycle: MemOpt accumulator discount per step in infancy (1 = Adagrad)")
    ap.add_argument("--mem-switch-stab", type=float, default=0.9,
                    help="life cycle: switch when the probe stability vs the last probe eval is >= this ...")
    ap.add_argument("--mem-switch-n", type=int, default=3, help="... on this many consecutive probe evals")
    ap.add_argument("--mem-switch-max", type=float, default=0.9,
                    help="life cycle: switch at the latest at this fraction of training (also at the LR decay start)")
    ap.add_argument("--mem-ramp", type=float, default=0.05,
                    help="life cycle: ramp length (fraction of training): decay and discount -> 0, then the freeze")
    ap.add_argument("--mem-addr-learn", type=int, default=1,
                    help="0 = fixed random addressing: W_q / sub-keys get no gradient, no margin loss")
    ap.add_argument("--mem-restart", type=int, default=0,
                    help="re-seed every sub-key unused since the last event every N steps in infancy (uncapped, band "
                         "reset; replaces the shared-pool balancing; 0 = off)")
    ap.add_argument("--mem-restart-zero", type=int, default=0,
                    help="1 = a restart also clears V of the re-seeded key's band (0 = only its MemOpt state is reset: "
                         "the band's rows are shared with the other heads' live keys)")
    ap.add_argument("--mem-probe-batch", type=int, default=8,
                    help="drift probes and loop-report forwards: this many sequences at a time (0 = --eval-batch); "
                         "lowers the eval-time VRAM peak")
    ap.add_argument("--mem-coarse", type=int, default=0,
                    help="coarse product-key pool: S sub-keys per half per level, next to the fine pool (0 = off)")
    ap.add_argument("--mem-coarse-heads", type=int, default=1, help="coarse pool: heads per level")
    ap.add_argument("--mem-nbr", type=int, default=0,
                    help="neighbour writes: the pairs ranked --mem-topk + 1 .. N are written (not read) (0 = off)")
    ap.add_argument("--mem-nbr-w", type=float, default=0.15, help="neighbour writes: their total weight alpha")
    # the learned memory access gate (plan_v2/22_memory_gate_plan.md, revision 2): off by default (= as before)
    ap.add_argument("--mem-gate", type=int, default=0,
                    help="1 = the predictor gate: per read only the top-c positions by a router score (+ a uniform "
                         "floor) read and write the memory; inference: score > an EMA threshold")
    ap.add_argument("--mem-gate-c", default="1.0,0.5", help="gate: the open rate c at the start / final")
    ap.add_argument("--mem-gate-sched", default="0.2,0.5", help="gate: the cosine from c start to c final starts / "
                                                               "ends at these fractions of the run")
    ap.add_argument("--mem-gate-explore", type=float, default=0.10, help="gate: the uniform exploration floor (c < 1)")
    ap.add_argument("--mem-gate-w", type=float, default=0.1, help="gate: the router BCE weight")
    ap.add_argument("--mem-gate-lr", type=float, default=1e-3, help="gate: the router's AdamW lr (own group, wd 0)")
    ap.add_argument("--mem-gate-clip", type=float, default=1.0, help="gate: the router's own grad-norm clip")
    ap.add_argument("--mem-gate-feat", default=",".join(GATE_FEATS), help="gate: the router's features")
    ap.add_argument("--mem-gate-trace", type=int, default=8, help="gate: the surprise trace time constant (tokens)")
    ap.add_argument("--mem-gate-thr-ema", type=float, default=0.02, help="gate: the threshold EMA rate rho_theta")
    ap.add_argument("--mem-gate-eval", default="learned", choices=list(GATE_MODES), help="gate: the eval gating mode")
    ap.add_argument("--mem-gate-quant", type=int, default=64, help="gate: the open-set sizes are multiples of this")
    ap.add_argument("--mem-gate-report", type=int, default=1,
                    help="gate: 1 = the eval-mode / benefit report at the report points (with the loop report)")
    ap.add_argument("--mem-wgate", type=int, default=0, help="gate: the step-size write variant (not implemented in v1)")
    ap.add_argument("--mem-wgate-kappa", type=float, default=0.5, help="gate: the write variant's kappa (unused in v1)")
    # the global + per-expert blob memory (plan_v2/25d_global_expert_blob_plan.md, blob_memory.py): off by default
    # (= as before). Stable parts only: storage, fixed map + fit, IVF / private search, read + inject, content update,
    # births; the gate / writes / merge / eviction / renewal are not built (--gm-read-mode is the gate's placeholder).
    # the memory-module interface (memory_api.py, README_memory_api.md)
    ap.add_argument("--memory", default=None, choices=list(MEMORY_KINDS),
                    help="the memory module: blob (the default: the --gm flags, exactly as before), wave (wave_memory.py, "
                         "a stub), none (no module, = --gm 0). On --resume: default = the checkpoint's; a different "
                         "kind swaps the memory (the brain is kept, the new memory starts fresh or from --memory-load)")
    ap.add_argument("--memory-load", default="", help="load the memory module's state from this memory.pt")
    ap.add_argument("--memory-reset", type=int, default=0, help="1 = the memory module starts fresh (as built)")
    ap.add_argument("--memory-save", type=int, default=0,
                    help="1 = every checkpoint also gets its memory alone as <name>.memory.pt next to it")
    ap.add_argument("--memory-embed", type=int, default=1,
                    help="1 (default) = the memory's state stays inside the brain checkpoints too; 0 = brain-only "
                         "checkpoints (implies --memory-save 1; --resume then reads <name>.memory.pt or --memory-load)")
    ap.add_argument("--memory-eval-off", type=int, default=0,
                    help="1 = every evaluation (val, loop report, sample) runs with the memory switched off")
    ap.add_argument("--memory-train-only", default="",
                    help="experts=<id>[,<id>...] (id: int or 'shared'): only these experts' memory learns; the brain and "
                         "every other memory part are frozen (requires_grad False, out of the optimizer, buffers restored)")
    ap.add_argument("--memory-cap-mb", type=float, default=0.0, help="byte budget of the memory state (0 = no cap)")
    ap.add_argument("--memory-cap-strict", type=int, default=0, help="1 = exit when the memory state is over the cap")
    # FastMem (fastmem.py, SPEC_fastmem.md): --memory fastmem; every flag is a fastmem_* cfg key
    ap.add_argument("--fastmem-write-grad", default="full", choices=["full", "detached"],
                    help="fastmem: full = the main loss reaches xf through the writes of the in-step scan; detached = "
                         "the write side (k, v, beta, omega, lam) sees xf.detach()")
    ap.add_argument("--fastmem-proj", default="expert", choices=["expert", "shared"],
                    help="fastmem: Wq / Wk / Wv per memory row (expert, the spec) or one set for all rows (shared)")
    ap.add_argument("--fastmem-heads", type=int, default=0, help="fastmem: 0 = the --fastmem-size value")
    ap.add_argument("--fastmem-dk", type=int, default=0, help="fastmem: 0 = the --fastmem-size value")
    ap.add_argument("--fastmem-dv", type=int, default=0, help="fastmem: 0 = the --fastmem-size value")
    ap.add_argument("--fastmem-global", type=int, default=0, help="fastmem: global sheet heads (0 = the size's)")
    ap.add_argument("--fastmem-global-topk", type=int, default=0, help="fastmem: global heads per token (0 = the size's)")
    ap.add_argument("--fastmem-size", default="lite", choices=["lite", "full"],
                    help="fastmem: lite = 2 heads x 128 x 128, global 8 (top-2), bf16 maths, the scan once at pass 3; "
                         "full = 4 x 512 x 512, global 32 (top-4), fp32, the scan at every pass")
    ap.add_argument("--fastmem-stream", type=int, default=1,
                    help="fastmem: 1 = one memory per data stream (user), carried across micro-batches / steps, reset "
                         "at the stream start (--data-mode stream names the streams; else zero per sequence); 0 = one "
                         "memory for all (mean merge after the step)")
    ap.add_argument("--fastmem-streams", type=int, default=16, help="fastmem: the number of streams (users)")
    ap.add_argument("--fastmem-read-pass", type=int, default=-2,
                    help="fastmem: the pass that scans / writes (-1 = every pass, the final one writes; -2 = the size's)")
    ap.add_argument("--fastmem-dtype", default="", choices=["", "bf16", "fp32"], help="fastmem: the maths ('' = the size's)")
    ap.add_argument("--fastmem-explore", type=float, default=0.05, help="fastmem: pre-read gate exploration (constant)")
    ap.add_argument("--fastmem-lr", type=float, default=1e-3, help="fastmem: the memory's AdamW lr")
    ap.add_argument("--fastmem-d-lr", type=float, default=1e-4, help="fastmem: D's factored update lr")
    ap.add_argument("--fastmem-decay0", type=float, default=1e-4, help="fastmem: initial 1 - exp(lam)")
    ap.add_argument("--fastmem-ckpt", type=int, default=0, help="fastmem: 1 = recompute the memory graph in backward")
    ap.add_argument("--fastmem-seed", type=int, default=1234)
    ap.add_argument("--fastmem-max-experts", type=int, default=0, help="fastmem: memory rows (0 = the final experts)")
    ap.add_argument("--fastmem-scan-budget", type=int, default=1 << 22,
                    help="fastmem: the packed scan's chunk budget in elements of [pairs, H, L, L] (one chunk's quadratics "
                         "alive at a time; ~60 B x budget, CPU-measured); 0 = the old padded [R, B, H, Cs, Cs] scan. Not a cfg key "
                         "(the maths are the same): may change on --resume")
    # the wave memory (wave_memory.py, README_memory_api.md): --memory wave; every flag is a wave_* cfg key
    ap.add_argument("--wave-mode", default="v21", choices=["v21", "on", "stub"],
                    help="wave memory: v21 (WaveSheet v2.1: global sheet + per-expert strips, wave_v21.py), on (the "
                         "v2 build, wave_memory.CodedWaveMemory) or stub (reads zeros, no state)")
    ap.add_argument("--wave-seed", type=int, default=1234, help="wave: the memory's own seed (rotation, init, gate)")
    ap.add_argument("--wave-side", type=int, default=16,
                    help="wave: the entry map is side x side prototypes (torus; the column store's places P = side^2)")
    ap.add_argument("--wave-levels", type=int, default=3, help="wave: residual code levels")
    ap.add_argument("--wave-slices", type=int, default=32, help="wave: slices per level (must divide --d)")
    ap.add_argument("--wave-codes", type=int, default=16, help="wave: prototypes per (level, slice)")
    ap.add_argument("--wave-topk", type=int, default=2, help="wave: soft top-k codes per slice at the read")
    ap.add_argument("--wave-entries", type=int, default=4,
                    help="wave: entry locations per read (the column store scans them; writes at entry 0)")
    ap.add_argument("--wave-warm", type=int, default=200, help="wave: address updates (micro-batches) before reads")
    ap.add_argument("--wave-anneal", type=int, default=2000, help="wave: SOM neighbourhood / rate anneal (updates)")
    ap.add_argument("--wave-upd-rows", type=int, default=4096, help="wave: thoughts per address update")
    ap.add_argument("--wave-fast", type=int, default=1000,
                    help="wave: address updates at the fast rates (the warm-up: the spec's step < 500 at accum 2), then "
                         "slow; 0 = always fast")
    ap.add_argument("--wave-eta-slow", type=float, default=1.0 / 8192,
                    help="wave: the address's slow-phase rate per update (statistics, map, bias, codes; spec T = 8192)")
    ap.add_argument("--wave-store", default="columns",
                    help="wave: storage (wave_store.STORES: columns = the v2 column store, sheet = the pre-v2 wave "
                         "sheet, hebb = the placeholder)")
    ap.add_argument("--wave-lambda", type=float, default=4.0,
                    help="wave columns: the chain weight lambda (M = (|Z| + lambda nch) / (16 + 3 lambda); 0 = no chain)")
    ap.add_argument("--wave-pack", default="nibble", choices=["nibble", "c2p3"],
                    help="wave sheet: synapse layout (nibble = 4 bits: 3 counts x 5 phases + empty; c2p3 = 2-bit "
                         "counter + 3-bit phase)")
    ap.add_argument("--wave-entry-keys", default="own", choices=["own", "shared"],
                    help="wave columns: entry 1 reads with its own residual keys (own) or the entry-0 keys (shared arm)")
    ap.add_argument("--wave-null-bins", default="load", choices=["fill", "load"],
                    help="wave columns: null-control bins by the winner place's load (default: measured) or by the "
                         "global fill (the spec)")
    ap.add_argument("--wave-c-z0", type=float, default=2.0,
                    help="wave: the read's confidence c_tok = clamp((z - this) / 4, 0, 1) (spec 2)")
    ap.add_argument("--wave-map-beta", type=float, default=0.25,
                    help="wave columns, place balance A: the entry map also learns on the written thoughts under a "
                         "write-count conscience of this weight (0 = off)")
    ap.add_argument("--wave-alloc-gamma", type=float, default=0.0,
                    help="wave columns, place balance B: a write goes to the entry (of --wave-entries) maximising "
                         "loc score - gamma sc log(load + 1) (0 = off: entry 0)")
    ap.add_argument("--wave-owned", type=int, default=16,
                    help="wave columns, --memory-train-only: owned places per selected expert")
    ap.add_argument("--wave-sheet-side", type=int, default=32, help="wave sheet: side x side hypercolumns (torus)")
    ap.add_argument("--wave-sheet-units", type=int, default=64, help="wave sheet: content units per hypercolumn")
    ap.add_argument("--wave-sheet-topk", type=int, default=1, help="wave sheet: key codes per slice at a read")
    ap.add_argument("--wave-fill", type=float, default=0.12, help="wave sheet: the synapse fill held by forgetting")
    ap.add_argument("--wave-merge", type=float, default=0.5, help="wave sheet: P(a rewrite takes the new phase)")
    ap.add_argument("--wave-content-fast", type=int, default=2000, help="wave sheet: content-codebook fast batches")
    ap.add_argument("--wave-read-chunk", type=int, default=2048, help="wave: tokens per recall chunk (memory)")
    ap.add_argument("--wave-store-dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="wave: synapse dtype")
    ap.add_argument("--wave-read-topk", type=int, default=1, help="wave: codes per slice at a global-store read")
    ap.add_argument("--wave-priv-write", default="read", choices=["read", "top1"],
                    help="wave: private writes only into the read parts (pair gate open or explored), the first "
                         "--wave-priv-cap by routing slot, + shared if read (read); or top-1 expert + shared (top1)")
    ap.add_argument("--wave-priv-cap", type=int, default=2, help="wave: private parts written per token (read rule)")
    ap.add_argument("--wave-null-every", type=int, default=16, help="wave: the null control every N optimizer steps")
    ap.add_argument("--wave-write-start", type=int, default=500,
                    help="wave v21: reads / writes / gate from this optimizer step on (before: == none)")
    ap.add_argument("--wave-strip", "--wave-strip-places", dest="wave_strip", type=int, default=4,
                    help="wave v21: places per expert strip (v2.2 small brain: 16)")
    ap.add_argument("--wave-strip-entries", type=int, default=0,
                    help="wave v21: strip places a private read takes (best by prototype score; 0 = auto: min(4, "
                         "strip) for strips of more than 4 places, else 1)")
    ap.add_argument("--wave-latent", type=int, default=2048,
                    help="wave v21 (v2.2): the memories' latent width (write value [correction | xf | workspace summary"
                         " | a fixed random projection of the correction], the store's decoder, g, D_e [dl, d]); 0 = d")
    ap.add_argument("--wave-cols", type=int, default=64,
                    help="wave v21 (v2.2): columns per place (each holds --wave-latent / cols dims; 16 = v2.1's store; "
                         "a place's bytes scale with it: 64 -> 1,769,472 B per place)")
    ap.add_argument("--wave-side-dim", type=int, default=64, help="wave v21 (v2.2 latent): the workspace summary width")
    ap.add_argument("--wave-read-unit", default="slice", choices=["slice", "full"],
                    help="wave v21 --wave-readout norm: slice = the injected d-slice of y_n is ~ unit (y_n x sqrt(dl / "
                         "d)); full = |y_n| = 1 over the dl latent (the world_wave27b run: reads ~ 0.43 x smaller)")
    ap.add_argument("--wave-conf", default="soft", choices=["soft", "ramp"],
                    help="wave v21: the read confidence c: soft = sigmoid(2 (z - 3)) (v2.2), ramp = clamp((z - "
                         "--wave-c-z0) / 4, 0, 1) (v2.1); --wave-z-conf stays the erase / confirm threshold")
    ap.add_argument("--wave-readout", default="norm", choices=["norm", "raw"],
                    help="wave v21 (v2.2): norm = RMSNorm(y) g x c x the per-(token, expert) gate s = 2 tanh(silu(xf . "
                         "w_e + b_e) / 2); raw = v2.1's unit read")
    ap.add_argument("--wave-gate-mode", default="pre", choices=["pre", "post"],
                    help="wave: pre = the pre-read gate + per-expert output layer (wave_gate.py), post = the old gate")
    ap.add_argument("--wave-k-routed", type=int, default=6, help="wave (pre gate): routed experts per token")
    ap.add_argument("--wave-gate-cap-frac", type=float, default=0.5, help="wave (pre gate): read rows / all rows")
    ap.add_argument("--wave-moments-bf16", type=int, default=0, help="wave (pre gate): bf16 AdamW moments for D")
    ap.add_argument("--wave-priv-levels", type=int, default=1,
                    help="wave: code levels of the per-expert private stores (0 = no private stores)")
    ap.add_argument("--wave-priv-topk", type=int, default=1, help="wave: codes per slice at a private read")
    ap.add_argument("--wave-forget", type=float, default=0.0, help="wave: the hebb stores' decay per step")
    ap.add_argument("--wave-write", default="delta",
                    help="wave: write selection + plan (wave_write.POLICIES: spec = the v2 spec's W4, wave = W4b's "
                         "count 2 + the 2nd entry on surprise); v21: delta (v2.2 default: write only the error vs "
                         "what the place recalls, erase at z >= --wave-z-conf) or spec (the v2.1 set rule)")
    ap.add_argument("--wave-write-w", type=int, default=16, help="wave: writes per micro-batch (one per token)")
    ap.add_argument("--wave-strong-k", type=int, default=4, help="wave (spec policy): writes at strength 2 per micro-batch")
    ap.add_argument("--wave-write-per-seq", type=int, default=1, help="wave: at most this many writes per sequence")
    ap.add_argument("--wave-writes-per-seq", type=int, default=8,
                    help="wave v21 (v2.2): writes per sequence = the top N most surprising read tokens, >= 8 tokens "
                         "apart (1 = v2.1's one per sequence); the micro-batch cap is --wave-write-w x N")
    ap.add_argument("--wave-target", default="grad+old", choices=["grad+old", "grad"],
                    help="wave: the write target: -e_hat (+ max(h, 0) x the old recall where c_tok >= .5)")
    ap.add_argument("--wave-content", type=float, default=0.18, help="wave: content scale x median |xf|")
    ap.add_argument("--wave-confirm", type=float, default=0.5,
                    help="wave: a read counts as a confirmed recall when open and cos(read, target) > this")
    ap.add_argument("--wave-confirm-w", type=int, default=16,
                    help="wave: confirmed recalls per micro-batch (unwritten tokens)")
    ap.add_argument("--wave-z-conf", type=float, default=4.0,
                    help="wave: a confirmation needs z >= this (null control; 2 = the c_tok start, spec 4, arm 3)")
    ap.add_argument("--wave-null", type=int, default=256, help="wave: null-control cues per micro-batch (0 = off)")
    ap.add_argument("--wave-read-pass", type=int, default=3,
                    help="wave: the pass of the one recall per forward (earlier passes read nothing, later re-apply it)")
    ap.add_argument("--wave-cap", type=float, default=0.5, help="wave: the hint's norm cap, x |xf| (0 = none)")
    ap.add_argument("--wave-lr", type=float, default=1e-3, help="wave: the memory's own AdamW learning rate")
    ap.add_argument("--wave-recon", type=int, default=0,
                    help="wave: 1 = a confirmed recall also binds its path's absent synapses (reconsolidation arm)")
    ap.add_argument("--wave-drop", type=float, default=0.1,
                    help="wave: the keep mask's drop rate (seq: sequences with m = 0 per training forward)")
    ap.add_argument("--wave-drop-mode", default="seq", choices=["seq", "forward"],
                    help="wave: seq = the spec's per-sequence keep mask, forward = drop whole training forwards")
    ap.add_argument("--wave-gate-open0", type=float, default=0.25, help="wave: the open share during the gate warm-up")
    ap.add_argument("--wave-gate-warm", type=int, default=200, help="wave: steps that set the gate's open bias")
    ap.add_argument("--wave-explore", type=float, default=0.05, help="wave: tokens with a random gate (training)")
    ap.add_argument("--wave-gate-floor", type=float, default=0.5, help="wave: gate values below this read nothing")
    ap.add_argument("--wave-max-experts", type=int, default=0, help="wave: private parts (0 = the final experts)")
    ap.add_argument("--gm", type=int, default=0, help="1 = the blob memory (global + per-expert spaces)")
    ap.add_argument("--gm-global", type=int, default=262144, help="gm: global slots")
    ap.add_argument("--gm-private", type=int, default=512, help="gm: slots per private space (routed + shared experts)")
    ap.add_argument("--gm-key", type=int, default=64, help="gm: key width (fixed random projection of xf)")
    ap.add_argument("--gm-dv", type=int, default=0, help="gm: content width (must be --d; 0 = --d)")
    ap.add_argument("--gm-fit-step", type=int, default=300, help="gm: the map and the IVF cells are fitted at this step")
    ap.add_argument("--gm-fit-window", type=int, default=20, help="gm: statistics from the training forwards of this "
                                                                  "many steps before --gm-fit-step")
    ap.add_argument("--gm-search", default="ivf", choices=["ivf", "brute"], help="gm: global search (brute = tests only)")
    ap.add_argument("--gm-cells", type=int, default=1024, help="gm: IVF cells")
    ap.add_argument("--gm-cell-cap", type=int, default=512, help="gm: IVF slots per cell (static)")
    ap.add_argument("--gm-probe", type=int, default=16, help="gm: IVF cells probed per query")
    ap.add_argument("--gm-qcap", type=int, default=192, help="gm: IVF query pairs per cell (static)")
    ap.add_argument("--gm-cell-topk", type=int, default=32, help="gm: bf16 candidates per (query, probe) row")
    ap.add_argument("--gm-ivf-refit", type=int, default=1000, help="gm: IVF refit every N steps (0 = never)")
    ap.add_argument("--gm-cand", type=int, default=16, help="gm: candidates returned by the search per query")
    ap.add_argument("--gm-topk", default="8,4", help="gm: blobs blended per read: global,private")
    ap.add_argument("--gm-read-mode", default="all", choices=["all", "none", "random"],
                    help="gm: the read mask (placeholder for the gate)")
    ap.add_argument("--gm-read-p", type=float, default=0.5, help="gm: read probability of --gm-read-mode random")
    ap.add_argument("--gm-compute-frac", type=float, default=1.0,
                    help="gm: static compute buffer for the private reads, this fraction of the pairs (1 = all)")
    ap.add_argument("--gm-tau", type=float, default=0.1, help="gm: match-weight temperature")
    ap.add_argument("--gm-kernel-c", type=float, default=1.0, help="gm: match weights use (d_j - d_1) / (d_1 + c h0^2)")
    ap.add_argument("--gm-h0", type=float, default=4.0, help="gm: initial kernel width h0 (global and private)")
    ap.add_argument("--gm-priv-bf16", type=int, default=1, help="gm: private contents in bf16 (stochastic rounding)")
    ap.add_argument("--gm-match-grad", type=int, default=1, help="gm: the loss reaches xf through the match weights")
    ap.add_argument("--gm-inj-scale", type=int, default=0, help="gm: a learnable per-channel injection scale (init 1)")
    ap.add_argument("--gm-lr", type=float, default=0.03, help="gm: content RMSprop lr (lazy, touched rows only)")
    ap.add_argument("--gm-beta2", type=float, default=0.9, help="gm: content RMSprop beta2 (per touch)")
    ap.add_argument("--gm-clip", type=float, default=0.05, help="gm: per-row content update norm cap (0 = none)")
    ap.add_argument("--gm-track-xf", type=int, default=0, help="gm: keep the per-token xf sum over the passes")
    ap.add_argument("--gm-probation-rate", type=float, default=0.0,
                    help="gm: probation store sized for this many write candidates per step (0 = off)")
    ap.add_argument("--gm-probation-life", type=int, default=3000, help="gm: probation FIFO life in steps")
    ap.add_argument("--gm-seed", type=int, default=1234, help="gm: seed of the fixed projection / its generators")
    # phase 2 (plan 25d rev 2 sections 5-10, the owner's write decision, the review): each off by default (= phase 1);
    # --gm-preset 25d turns on the decided defaults (gate, probation writes, row-Adagrad, tracked width, 384-d re-rank)
    ap.add_argument("--gm-preset", default="", choices=["", "25d"], help="gm: the decided phase-2 defaults")
    ap.add_argument("--gm-gate", type=int, default=0, help="gm: the learned per-(token, expert) read gate")
    ap.add_argument("--gm-explore", type=float, default=0.05, help="gm gate: exploration rate (match-stratified)")
    ap.add_argument("--gm-buf", type=float, default=0.55, help="gm gate: static read buffer, fraction of the pairs")
    ap.add_argument("--gm-gate-lr", type=float, default=1e-3, help="gm gate: its own AdamW lr")
    ap.add_argument("--gm-gate-clip", type=float, default=1.0, help="gm gate: its own grad-norm clip")
    ap.add_argument("--gm-write", default="off", choices=["off", "probation", "direct"],
                    help="gm writes: probation = keep what comes back (the owner's default), direct = comparison")
    ap.add_argument("--gm-cand-rate", type=float, default=0.10, help="gm writes: candidates = this fraction of tokens")
    ap.add_argument("--gm-write-w", type=int, default=32, help="gm writes (direct): writes per micro-batch")
    ap.add_argument("--gm-private-rate", type=float, default=0.256, help="gm writes: private credit per expert per step")
    ap.add_argument("--gm-promo-gamma", type=float, default=1.0, help="gm probation: top-1 within this x h0^2")
    ap.add_argument("--gm-promo-theta", type=float, default=0.3, help="gm probation: gradient-sketch cosine >= theta")
    ap.add_argument("--gm-promo-budget", type=int, default=64, help="gm probation: promotions kept per step")
    ap.add_argument("--gm-promo-rank", default="cos", choices=["cos", "keycos"], help="gm probation: ranking")
    ap.add_argument("--gm-promo-min-returns", type=int, default=1, help="gm probation: returns needed (2 = option)")
    ap.add_argument("--gm-prob-lookup", default="exact", choices=["ivf", "exact"],
                    help="gm probation: lookup (exact = blocked fp16 scan; ivf = cells rebuilt after each push)")
    ap.add_argument("--gm-probation-cap", type=int, default=0, help="gm probation: entries (0 = rate x life)")
    ap.add_argument("--gm-probation-dim", type=int, default=64, help="gm probation: key / sketch dims (64 or 32)")
    ap.add_argument("--gm-probation-dtype", default="fp16", choices=["fp16", "int8"], help="gm probation: storage")
    ap.add_argument("--gm-probation-log", type=int, default=0, help="gm probation: logging-only surprise buffer")
    ap.add_argument("--gm-gamma", default="0.25,0.25", help="gm merges: gamma global,private (d1 < gamma h0^2)")
    ap.add_argument("--gm-theta-m", type=float, default=0.2, help="gm merges: content agreement cos > theta")
    ap.add_argument("--gm-protect-age", type=int, default=2000, help="gm eviction: young-blob protection (steps)")
    ap.add_argument("--gm-protect-frac", type=float, default=0.5, help="gm eviction: protected share limit")
    ap.add_argument("--gm-strength-half", type=float, default=4000.0, help="gm eviction: strength half-life (steps)")
    ap.add_argument("--gm-track-sigma", type=int, default=0, help="gm: the kernel width sigma^2 tracked (EMA)")
    ap.add_argument("--gm-rerank-full", type=int, default=0, help="gm: 384-d re-rank of the global top-64")
    ap.add_argument("--gm-rerank-dtype", default="bf16", choices=["bf16", "fp32"], help="gm: the 384-d keys' dtype")
    ap.add_argument("--gm-rerank-top", type=int, default=64, help="gm: candidates re-ranked at model width")
    ap.add_argument("--gm-v-init", type=float, default=0.2, help="gm writes: content norm = this x |xf| (s)")
    ap.add_argument("--gm-kernel", default="bounded", choices=["bounded", "gauss"],
                    help="gm: match weights; gauss = scale-free exp(-(d_j - d_1) / (c m)), m = running median d2 - d1")
    ap.add_argument("--gm-kernel-cg", type=float, default=0.1, help="gm gauss kernel: c (0.1 ~ top-1)")
    ap.add_argument("--gm-kernel-dist", default="full", choices=["full", "key"],
                    help="gm gauss kernel: global distances at model width after the re-rank (full) or the 64-d key")
    ap.add_argument("--gm-use-A", type=int, default=1, help="gm: the match-strength factor A (0 = off: A = 1)")
    ap.add_argument("--gm-gate-open0", type=float, default=0.23, help="gm gate: open rate set during the warm-up")
    ap.add_argument("--gm-gate-warm", type=int, default=200, help="gm gate: warm-up steps of the open-rate bias")
    ap.add_argument("--gm-content-opt", default="rmsprop", choices=["rmsprop", "adagrad"],
                    help="gm: the content optimizer (lazy, touched rows)")
    ap.add_argument("--gm-eval-every", type=int, default=0,
                    help="gm: memory on / off validation delta every N steps (0 = off; at the eval steps)")
    ap.add_argument("--gm-heldout", default="", help="gm: a token .bin (uint16) for the held-out on / off delta")
    ap.add_argument("--empty-cache-eval", "--vram-empty-cache", dest="empty_cache_eval", type=int, default=-1,
                    help="torch.cuda.empty_cache() before every evaluate() (-1 = on with --gm 1; the loop report "
                         "already frees the cache)")
    ap.add_argument("--mem-fraction", "--vram-frac", dest="mem_fraction", type=float, default=0.0,
                    help="torch.cuda.set_per_process_memory_fraction(f) (0 = off)")
    # run control for the real run (each off by default = as before)
    ap.add_argument("--chunk-ce", type=int, default=0,
                    help="loop arch: the training read-outs (final CE, deep supervision, halting) in N-token chunks "
                         "(no (B, T, V) fp32 log-probs; 0 = off, the plain path)")
    # VRAM savers (activation checkpointing, same maths, no saved-state change: each off by default = as before)
    ap.add_argument("--vram-ckpt-norm", type=int, default=0,
                    help="1 = every RMSNorm's fp32 intermediates recomputed in backward (not kept)")
    ap.add_argument("--vram-ckpt-mixer", type=int, default=0,
                    help="1 = the SSD mixer of each backprop pass recomputed in backward (spike thresholds as seen "
                         "by the first run; homeostasis not repeated)")
    ap.add_argument("--vram-ckpt-experts", type=int, default=0,
                    help="1 = the routed experts' hidden layer (up, spike, down) of each backprop pass recomputed in "
                         "backward (same threshold / homeostasis rule as --vram-ckpt-mixer)")
    ap.add_argument("--vram-ckpt-concept", type=int, default=0,
                    help="1 = the concept loss terms (per pass and level) recomputed in backward")
    ap.add_argument("--vram-readout-lean", type=int, default=1,
                    help="1 (default) = --chunk-ce's checkpointed chunks do not keep the previous pass's logits and their own "
                         "detached logits alive until backward (~2 x (B, T, V) bf16 per backprop pass)")
    ap.add_argument("--vram-ckpt-gmdist", type=int, default=0,
                    help="1 = the blob memory's model-width (n, k, d) read distances recomputed in backward")
    ap.add_argument("--vram-rerank-rows", type=int, default=0,
                    help="blob memory: query rows per block of the model-width re-rank (0 = auto, ~512 MB blocks; "
                         "the values are the same for any block size)")
    ap.add_argument("--vram-savers", type=int, default=0,
                    help="1 = all of the --vram-* savers above (--vram-rerank-rows 1024 unless set)")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="save the full resumable last.pt at this step and exit (the schedules keep the full length; "
                         "--resume continues to the end)")
    ap.add_argument("--frag-gb", type=float, default=0.0,
                    help="empty the CUDA cache around the memory's step end when reserved - allocated exceeds this many "
                         "GB (fragmentation guard; 0 = off)")
    ap.add_argument("--nan-abort", type=int, default=0,
                    help="1 = a NaN / inf loss or gradient norm saves nan_abort.pt (full state) and exits cleanly")
    ap.add_argument("--save-at", default="", help="extra ckpt_<step>.pt at these steps, e.g. 1000,3000 (--keep-ckpts)")
    ap.add_argument("--health-every", type=int, default=0,
                    help="health line every N steps into health.jsonl (VRAM peak, gate open rates, promotions, "
                         "waiting-area fill, blob lifetimes, eval-write guard counter)")
    ap.add_argument("--diag-every", type=int, default=0,
                    help="in-run diagnostics every N steps into diag.jsonl (device accumulators, deferred copies; "
                         "no state change; 0 = off)")
    ap.add_argument("--resume", default="", help="resume from this full checkpoint (runs/<name>/last.pt)")
    ap.add_argument("--save-report-ckpts", type=int, default=0,
                    help="1 = also save ckpt_<step>.pt at the 25 / 50 / 75 / 100%% report points")
    ap.add_argument("--keep-ckpts", type=int, default=5,
                    help="keep only the newest N numbered ckpt_<step>.pt (older -> Recycle Bin); last.pt / final.pt "
                         "are always kept; 0 = keep all")
    ap.add_argument("--sizes", default="", help="expert sizes, e.g. 96x16,192x32,384x16 (count per size)")
    ap.add_argument("--moe", default="", help="routed MLP: n_exp,k,hidden,shared,balance  e.g. 64,6,192,384,auxfree")
    ap.add_argument("--concept", default="", help="next-concept planning: path to a trained concept_tok.pt")
    ap.add_argument("--concept-levels", type=int, default=3, help="coarse levels the brain plans (1, 2, 4 codes)")
    ap.add_argument("--concept-split", type=int, default=2, help="plan from the output of this block into the next")
    ap.add_argument("--concept-w", type=float, default=0.3, help="weight of the concept prediction loss")
    ap.add_argument("--teacher", default="", help="distil from this run's final.pt (e.g. runs/s0_ssd/final.pt)")
    ap.add_argument("--kd", type=float, default=1.0, help="weight of KL(teacher || student)")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--mlp", type=int, default=1536)
    ap.add_argument("--start", default="", help="grow from these dims, e.g. layers=2,heads=2,mlp=512")
    ap.add_argument("--grow-at", default="0.1,0.2,0.3", help="fractions of the token budget")
    ap.add_argument("--tokens", type=float, default=300e6)
    ap.add_argument("--batch", type=int, default=32, help="micro-batch size")
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step (effective batch = batch x accum)")
    ap.add_argument("--eval-batch", type=int, default=0, help="validation batch size (0 = batch x accum)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--decay", type=float, default=0.2, help="WSD: final fraction of steps decays to 0")
    ap.add_argument("--rewarm", type=float, default=0.01, help="LR rewarm after growth, fraction of steps")
    ap.add_argument("--anneal", default="0.05,0.4", help="analog->spike blend from/to (fractions)")
    ap.add_argument("--homeo", type=float, default=0.0, help="homeostatic threshold rate (e.g. 0.01); 0 = off")
    ap.add_argument("--target", type=float, default=0.15, help="target firing rate for homeostasis")
    ap.add_argument("--homeo-mode", default="full", choices=["full", "floor"],
                    help="full: pull every channel to --target; floor: only rescue near-silent channels")
    ap.add_argument("--redo-every", type=int, default=0, help="re-init silent MLP units every N steps (needs --homeo)")
    ap.add_argument("--grow-where", default="top", choices=["top", "spread"])
    ap.add_argument("--grow-init", default="fresh", choices=["fresh", "copy"])
    ap.add_argument("--curriculum", default="", help="fractions where harder data buckets are added, e.g. 0.25,0.5")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000, help="save runs/<name>/last.pt every N steps")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import fastmem as _fastmem
    _fastmem.SCAN_BUDGET = int(args.fastmem_scan_budget)   # (the packed scan, VRAM; not in the cfg)
    args.eval_batch = args.eval_batch or args.batch * args.accum

    run = Path(args.runs) / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "train.log", "a", encoding="utf-8", errors="replace")
    sys.stdout, sys.stderr = Tee(sys.stdout, logf), Tee(sys.stderr, logf)
    (run / "run.pid").write_text(str(os.getpid()))
    lock = claim_gpu(Path(args.runs), args.run_name) if args.device == "cuda" else None
    (run / ("config_resume.json" if args.resume else "config.json")).write_text(
        json.dumps(vars(args) | {"argv": sys.argv}, indent=1))
    print(f"run {args.run_name}  pid {os.getpid()}  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    rs = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    # the memory module (memory_api.py): its kind (on --resume: the checkpoint's unless --memory names another)
    mem_explicit, ck_kind = args.memory is not None, (ckpt_memory_kind(rs["cfg"]) if rs is not None else None)
    if args.memory is None:
        args.memory = ck_kind if rs is not None else "blob"
    if args.memory == "none":
        args.gm = 0                         # (exactly the --gm 0 code path)
    if not args.memory_embed:
        args.memory_save = 1
    mem_swap = rs is not None and mem_explicit and args.memory != ck_kind
    train_only = parse_train_only(args.memory_train_only)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.mem_fraction > 0 and str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer or str(Path(args.data) / "tokenizer.json"))
    data = Data(Path(args.data), args.seq, args.device, curriculum=bool(args.curriculum), mode=args.data_mode)
    assert args.data_mode != "stream" or not args.curriculum, "--data-mode stream has no curriculum buckets"
    curr_at = [float(f) for f in args.curriculum.split(",")] if args.curriculum else []

    final = dict(layers=args.layers, heads=args.heads, mlp=args.mlp)
    moe = parse_moe(args.moe)
    if moe:
        final["experts"] = moe["n_exp"]
    start = final | parse_dims(args.start)
    if moe and args.sizes:
        sizes = [int(s) for part in args.sizes.split(",") for s in [part.split("x")[0]] * int(part.split("x")[1])]
        g = torch.Generator().manual_seed(args.seed)
        sizes = [sizes[i] for i in torch.randperm(len(sizes), generator=g)]  # mixed sizes in random order
        moe["hidden"], moe["n_exp"] = sizes, len(sizes)
        final["experts"] = len(sizes)
        start = final | parse_dims(args.start)
        moe["hidden"] = sizes[:start["experts"]]
    if rs is not None:  # --resume: the model at its grown size from the saved cfg, then its state
        cfg = copy.deepcopy(rs["cfg"])
        if mem_swap:  # (memory_api: the brain from the checkpoint, the memory module of another kind, fresh)
            assert cfg.get("arch") == "loop", "--memory swap needs --arch loop"
            cfg = {k: v for k, v in cfg.items() if not is_memory_cfg_key(k)}
            if args.memory == "blob":
                assert args.gm, "--resume --memory blob (a swap): give --gm 1 and its --gm-* flags"
                cfg.update(blob_memory_kwargs(args, tok, final))
            elif args.memory == "wave":
                cfg.update(wave_memory_kwargs(args, tok, final))
            elif args.memory == "fastmem":
                cfg.update(fastmem_memory_kwargs(args, final))
            print(f"MEMORY SWAP: the checkpoint's {ck_kind} memory -> {args.memory} (the brain is kept)", flush=True)
        model = (LoopBrain if cfg.get("arch") == "loop" else SpikingLM)(**cfg).to(args.device)
        load_brain_state(model, rs["state"], args, swap=mem_swap)
    elif args.arch == "loop":
        # capacity 2x the average load: near-dropless once balanced (drops would compound over passes), but
        # bounded memory: fully dropless sized every expert buffer by the busiest expert -> 28.6 GB at init
        moe["capacity"] = args.capacity
        moe["u"] = args.route_u
        if args.route_aux:
            moe["balance"], moe["aux_w"] = "aux", args.route_aux  # gradient-based load-balancing loss
        if args.route_aux and args.route_both:
            moe["balance"] = "both"
        if args.route_lift:
            moe["lift"] = args.route_lift
        if args.route_shield:
            moe["shield"] = True
        if args.bias_logit:
            moe["bias_logit"] = True
        if args.route_z >= 0:
            moe["z_w"] = args.route_z
        if args.birth_mode != "ramp":
            moe["birth_mode"] = args.birth_mode
        if args.birth_delta != 0.5:
            moe["birth_delta"] = args.birth_delta
        # head groups: --head-group N always-on heads (group 0), the other --heads - N round-robin into
        # --head-groups groups (default: groups of N)
        n_groups = 0
        if args.head_group:
            n_groups = args.head_groups or max(1, math.ceil((args.heads - args.head_group) / args.head_group))
        mem_kw = {}
        if args.concepts:  # memory system (plan_v2/12): the keyword arguments exist only when it is on
            # the EOT id cuts the concept windows at document ends: a wrong id would silently break the masks
            eot = tok.token_to_id("<|endoftext|>")
            assert eot is not None, "--concepts: the tokenizer has no <|endoftext|> token"
            meta = Path(args.data) / "meta.json"
            if meta.exists():  # (data without meta.json, e.g. ts4k: the tokenizer's id)
                m_eot = json.loads(meta.read_text()).get("eot_id")
                assert m_eot is not None, f"{meta} has no eot_id"
                assert int(m_eot) == eot, f"meta.json eot_id {m_eot} != the tokenizer's <|endoftext|> id {eot}"
            if args.think_w <= 0:
                raise SystemExit("--concepts needs --think-w > 0: the concept loss is scaled by --think-w "
                                 "(use e.g. --think-w 0.1 --think-fb 1 --think-use 1)")
            horizons = [int(n) for n in args.concepts.split(",")]
            weights = [float(w) for w in args.concept_weights.split(",")]
            if len(weights) != len(horizons):
                print(f"--concept-weights has {len(weights)} values for {len(horizons)} levels: using the default "
                      f"rule (1, 1, 0.5, ...)", flush=True)
                weights = None
            mem_kw = dict(concepts=horizons, concept_weights=weights, concept_inject_rank=args.concept_inject_rank,
                          concept_eot=eot, mem_shared=args.mem_shared, mem_heads=args.mem_heads,
                          mem_topm=args.mem_topm, mem_topk=args.mem_topk, mem_tau=args.mem_tau,
                          mem_read0=args.mem_read0, mem_private=args.mem_private, mem_dv=args.mem_dv,
                          mem_addr_sg=args.mem_addr_sg, mem_margin=tuple(float(v) for v in args.mem_margin.split(",")),
                          mem_ema=args.mem_ema, mem_max_experts=final.get("experts", 0), mem_life=args.mem_life,
                          mem_addr_learn=args.mem_addr_learn, mem_restart=args.mem_restart, mem_coarse=args.mem_coarse,
                          mem_coarse_heads=args.mem_coarse_heads, mem_nbr=args.mem_nbr, mem_nbr_w=args.mem_nbr_w)
            if args.mem_gate:  # (plan_v2/22: the model's keyword arguments exist only when the gate is on)
                assert args.readout_frac == 1, "--mem-gate needs --readout-frac 1"
                assert args.mem_shared > 0, "--mem-gate needs --mem-shared > 0"
                mem_kw.update(mem_gate=1, mem_gate_feat=args.mem_gate_feat, mem_gate_trace=args.mem_gate_trace,
                              mem_gate_thr_ema=args.mem_gate_thr_ema)
        if args.memory == "wave":  # (memory_api.py: the wave memory, a stub)
            mem_kw.update(wave_memory_kwargs(args, tok, final))
        elif args.memory == "fastmem":  # (fastmem.py, SPEC_fastmem.md)
            mem_kw.update(fastmem_memory_kwargs(args, final))
        elif args.gm:  # (plan_v2/25d: the model's keyword arguments exist only when the blob memory is on)
            mem_kw.update(blob_memory_kwargs(args, tok, final))
        model = LoopBrain(tok.get_vocab_size(), args.d, args.d_state, start["heads"], args.head_dim,
                          moe=dict(moe, n_exp=start["experts"]) if not args.sizes else moe, spike=args.spike,
                          k_mean=args.k_mean, k_min=args.k_min, k_max=args.k_max, bptt=args.bptt,
                          state="" if args.state == "none" else args.state,
                          refractory=args.refractory, slots=args.slots, waves=args.waves,
                          pops=args.pops, ws_direct=args.ws_direct, head_group=args.head_group,
                          head_groups=n_groups, ckpt_route=bool(args.ckpt_route), deep_sup=args.deep_sup,
                          halt=bool(args.halt), halt_prior=args.halt_prior, halt_beta=args.halt_beta,
                          think_w=args.think_w, think_n=args.think_n, think_fb=bool(args.think_fb),
                          think_target=args.think_target, think_use=bool(args.think_use),
                          readout_frac=args.readout_frac, **mem_kw).to(args.device)
    else:
      model = SpikingLM(tok.get_vocab_size(), args.d, start["layers"], start["heads"], args.head_dim, start["mlp"],
                      args.mixer, args.spike, moe=dict(moe, n_exp=start["experts"]) if moe else None,
                      concept=dict(path=str(Path(args.concept).resolve()), levels=args.concept_levels,
                                   split=args.concept_split) if args.concept else None).to(args.device)
    if args.memory_reset and getattr(model, "bm", None) is not None:   # (memory_api: a fresh module, same config)
        model.reset_memory()
        print(f"memory: {model.bm.kind} reset (fresh, as built)", flush=True)
    if args.memory_load:
        if getattr(model, "bm", None) is None:
            raise SystemExit(f"--memory-load {args.memory_load}: the model has no memory module")
        info = load_memory_file(args.memory_load, model)
        print(f"memory: loaded {args.memory_load} ({info['kind']}, saved at step {info.get('step')})", flush=True)
    if (rs is not None and (args.memory_reset or mem_swap) and not args.memory_load
            and getattr(model, "bm", None) is not None and model.bm.kind == "blob"
            and rs["step"] > args.gm_fit_step - args.gm_fit_window):
        # (a fresh blob memory mid-run: its fit window starts now, not at the configured step)
        args.gm_fit_step = rs["step"] + args.gm_fit_window
        print(f"memory: the blob address fit moves to step {args.gm_fit_step}", flush=True)
    model.set_homeo(args.homeo, args.target, args.homeo_mode)
    if args.mem_wgate:
        raise SystemExit("--mem-wgate: the step-size write variant is not implemented in v1 (plan_v2/22 section 5: off)")
    opt = build_opt(model, args)
    gate = getattr(model, "mem_gate", None)   # the memory access gate (plan_v2/22): its router has its own AdamW
    gate_opt = None
    if gate is not None:
        gate_opt = torch.optim.AdamW(gate.parameters(), lr=args.mem_gate_lr, betas=(0.9, 0.99), weight_decay=0.0)
        if rs is not None and rs.get("gate") is not None:
            gate_opt.load_state_dict(rs["gate"]["opt"])
            gate.load_host_state(rs["gate"]["host"])
        gc0, gc1 = (float(v) for v in args.mem_gate_c.split(","))
        gs0, gs1 = (float(v) for v in args.mem_gate_sched.split(","))
        model.mem_gate_explore, model.mem_gate_w = args.mem_gate_explore, args.mem_gate_w
        model.mem_gate_eval, model.gate_quant = args.mem_gate_eval, args.mem_gate_quant
    if rs is not None:  # the optimizer state (BrainOpt: per name + its step count; torch AdamW: its state_dict)
        if args.opt == "brain":
            opt.t = rs["opt"]["t"]
            opt.state = {n: {k: v.to(args.device) for k, v in st.items()} for n, st in rs["opt"]["state"].items()}
            if mem_swap:  # (the old memory's parameters are gone)
                opt.state = {n: st for n, st in opt.state.items() if n in opt.groups}
        else:
            try:
                opt.load_state_dict(rs["opt"]["state"])
            except ValueError as e:
                raise SystemExit(f"--resume: the AdamW state does not fit the model ({e}); a memory swap that changes "
                                 f"the parameters needs --opt brain") from e
    teacher = None
    if args.teacher:
        ck = torch.load(args.teacher, map_location=args.device)
        teacher = SpikingLM(**ck["cfg"]).to(args.device)
        teacher.load_state_dict(ck["state"], strict=False)
        teacher.eval().requires_grad_(False)
        print(f"distilling from {args.teacher} (kd weight {args.kd})", flush=True)

    tok_per_step = args.batch * args.accum * args.seq  # per optimizer step; all schedules count optimizer steps
    steps = int(args.tokens // tok_per_step)
    grow_steps = [int(float(f) * steps) for f in args.grow_at.split(",")] if args.start and not args.neuro else []
    t0, t1 = (float(f) for f in args.neuro_t.split(","))
    neuro = Neurogenesis(model, start, final, steps, every=args.neuro_every, where=args.neuro_where, t0=t0, t1=t1,
                         protocol=bool(args.neuro_protocol), renew_rate=args.renew_rate) if args.neuro else None
    if neuro is not None and rs is not None:  # (built on the grown model: its schedule comes from the checkpoint)
        neuro.load_state_dict(rs["neuro"], args.device)
    a0, a1 = (float(f) for f in args.anneal.split(","))
    print(f"start {start} -> final {final}; {n_params(model)/1e6:.2f}M params (+{model.emb.weight.numel()/1e6:.2f}M emb); "
          f"{steps} steps x {tok_per_step} tokens; grow at {grow_steps}", flush=True)
    mem = getattr(model, "mem", None)  # memory system (plan_v2/12): None unless --mem-shared / --mem-private
    probes, freeze_step, life = None, None, None
    if mem is not None:
        mem.lr, mem.clip, mem.s0_cfg = args.mem_lr, args.mem_clip, args.mem_s0
        mem.restart_zero = bool(args.mem_restart_zero)
        mem.balance_dead_max = args.mem_balance_dead_max
        mem.balance_every, mem.balance_max, mem.balance_keep = (args.mem_balance_every, args.mem_balance_max,
                                                                bool(args.mem_balance_keep))
        model.mem_readdr_every, model.mem_readdr_start, model.mem_margin_passes = (
            args.mem_readdr_every, args.mem_readdr_start, args.mem_margin_passes)
        if gate is not None and args.mem_readdr_every != 1:  # (plan_v2/22: the open sets differ between reads)
            print("--mem-gate: address reuse (--mem-readdr-every) is off under the gate", flush=True)
            model.mem_readdr_every = 1
        mem.opt_reset = lambda name, sl: reset_mem_key_rows(opt, model, name, sl)  # (reads the current opt)
        freeze_step = math.ceil(args.mem_freeze * steps)
        if args.mem_probes:
            probes = MemoryProbes(data.val, args.seq, n_seq=args.mem_probes, seed=args.seed,
                                  batch=args.mem_probe_batch or args.eval_batch,
                                  guard=args.mem_guard, repair=args.mem_repair, repair_usage=args.mem_repair_usage)
        if args.mem_life:  # (plan_v2/21: the switch replaces the fixed --mem-freeze)
            life = LifeCycle(mem, steps, decay=args.mem_decay, beta=args.mem_acc_beta, switch_stab=args.mem_switch_stab,
                             switch_n=args.mem_switch_n, switch_max=args.mem_switch_max, lr_decay=args.decay,
                             ramp=args.mem_ramp)
        if rs is not None:
            mem.load_host_state(rs["mem"])
            if probes is not None:
                probes.rng.set_state(rs["probes_rng"])
        tables = sum(b.numel() * b.element_size() for n, b in mem.named_buffers())
        print(f"memory: shared S={mem.S} ({mem.L * mem.S ** 2} rows), private R={mem.R} x {mem.Emax} experts, "
              f"tables + buffers {tables / 1e9:.2f} GB; writes from step {args.mem_write_start}, "
              + (f"life cycle on (switch at the first of: stability >= {args.mem_switch_stab} on {args.mem_switch_n} "
                 f"probe evals, step {life.t_decay:.0f}, step {life.t_max:.0f}; ramp {life.ramp} steps)"
                 if life is not None else f"freeze at the first step >= {freeze_step} with no births left")
              + (f"; coarse pool S={mem.Sc} x {mem.Hc} heads" if mem.Sc else "")
              + (f"; restart every {mem.restart} steps" if mem.restart else "")
              + ("; fixed addressing" if not mem.addr_learn else "")
              + (f"; neighbour writes to rank {mem.nbr} (alpha {mem.nbr_w})" if mem.nbr else "")
              + (f"; memory gate c {args.mem_gate_c} over {args.mem_gate_sched} of the run, floor {args.mem_gate_explore}, "
                 f"features {','.join(gate.feats)} ({gate.dim}-d)" if gate is not None else ""), flush=True)

    bm = getattr(model, "bm", None)  # the blob memory (plan_v2/25d): None unless --gm 1
    apply_vram_savers(model, args)
    if args.chunk_ce:
        assert args.arch == "loop" and not args.teacher, "--chunk-ce needs --arch loop and no --teacher (no logits)"
        model.chunk_ce = args.chunk_ce
    if bm is not None:  # (memory_api: the trainer's settings; the blob: the gate target in nats per token, accumulation)
        bm.trainer_setup(args)
    empty_eval = (args.empty_cache_eval if args.empty_cache_eval >= 0 else int(bm is not None)) > 0
    if bm is not None:
        print(bm.describe(args), flush=True)
        brep = byte_budget(model, args.memory_cap_mb * 1e6)
        print(budget_line(brep), flush=True)
        if not brep["ok"] and args.memory_cap_strict:
            raise SystemExit(f"--memory-cap-strict: the memory state is over --memory-cap-mb {args.memory_cap_mb}")
    flops, last_grow, t_start, t_log, ema = 0.0, -10**9, time.time(), time.time(), None
    report_steps = {round(steps * f / args.eval_every) * args.eval_every for f in (0.25, 0.5, 0.75)} | {steps}
    tok_since, settle, n_eval = 0, False, 0
    tok_hist, spill_run, spill_freed = [], 0, False  # VRAM spill guard
    start_step = 0
    if rs is not None:  # --resume (plan_v2/19): the trainer counters and every RNG, then the saved step's train step
        tr = rs["trainer"]
        start_step = rs["step"]
        flops, last_grow, ema, n_eval, tok_since = tr["flops"], tr["last_grow"], tr["ema"], tr["n_eval"], tr["tok_since"]
        tok_hist, spill_run, spill_freed, settle = list(tr["tok_hist"]), tr["spill_run"], tr["spill_freed"], tr["settle"]
        t_start = time.time() - tr["elapsed"]
        data.phase = tr["data_phase"]
        if tr.get("group_passes") is not None:
            model.group_passes = list(tr["group_passes"])
        set_rng_state(rs["rng"], args.device)
        print(f"RESUMED at step {start_step} from {args.resume}", flush=True)
    save_at = {int(v) for v in args.save_at.split(",") if v.strip()}
    diagw = None
    if args.diag_every:
        model.diag_train = True
        diagw = DiagWriter(run / "diag.jsonl", args.device)

    def model_state():
        """The model state of a checkpoint: everything (default), or the brain alone (--memory-embed 0)."""
        st = model.state_dict()
        if not args.memory_embed:
            for k in [k for k in st if k.startswith(MEM_PREFIX)]:
                del st[k]
        return st

    def save_mem(ckpt, step):
        """--memory-save 1: the memory alone next to the checkpoint (<name>.memory.pt)."""
        if args.memory_save and getattr(model, "bm", None) is not None:
            save_memory_file(memory_path(ckpt), model, step)

    def eval_mem():
        """--memory-eval-off 1: the evaluations run with the memory switched off."""
        return memory_off(model) if args.memory_eval_off else contextlib.nullcontext()

    buf_freeze = None
    if train_only is not None:  # (memory_api train-only mode: the brain frozen, only these experts' memory learns)
        if bm is None:
            raise SystemExit("--memory-train-only: the model has no memory module")
        if mem is not None or gate is not None:
            raise SystemExit("--memory-train-only: not with the old memory (--mem-*)")
        if not bm.ready:
            raise SystemExit("--memory-train-only: the memory is not ready (the blob: not fitted yet); resume from a "
                             "checkpoint past the fit or --memory-load a fitted memory")
        if neuro is not None or grow_steps:
            print("--memory-train-only: growth / neurogenesis off (the brain is frozen)", flush=True)
            neuro, grow_steps = None, []
        names = freeze_for_train_only(model, opt, train_only)
        bm.set_train_only(train_only)
        buf_freeze = BufferFreeze(model)
        print(f"MEMORY TRAIN-ONLY: experts {train_only}; trainable parameters {names or 'none'} (+ the memory's own "
              f"content update); the brain's {sum(p.numel() for p in model.parameters())} parameters and "
              f"{len(buf_freeze.items)} buffers frozen", flush=True)

    def full_ckpt(step):
        """Everything --resume needs (plan_v2/19): model (incl. the blob stores, probation, IVF, gate, generators via the
        state dict), optimizer, neurogenesis, memory host state, trainer counters, every RNG."""
        return dict(
            cfg=model.cfg, state=model_state(), step=step,
            opt=dict(t=opt.t, state=opt.state) if args.opt == "brain" else dict(state=opt.state_dict()),
            neuro=neuro.state_dict() if neuro is not None else None,
            mem=mem.host_state() if mem is not None else None,
            **(dict(gate=dict(opt=gate_opt.state_dict(), host=gate.host_state())) if gate is not None else {}),
            probes_rng=probes.rng.get_state() if probes is not None else None,
            trainer=dict(flops=flops, last_grow=last_grow, ema=ema, n_eval=n_eval, tok_since=tok_since,
                         tok_hist=list(tok_hist), spill_run=spill_run, spill_freed=spill_freed, settle=settle,
                         elapsed=time.time() - t_start, data_phase=data.phase,
                         group_passes=list(getattr(model, "group_passes", None) or []) or None),
            rng=get_rng_state(args.device))
    def near_cap():
        """--frag-gb: the CUDA cache is within frag_gb of the cap (reserved > cap - frag_gb): only then empty_cache."""
        return args.frag_gb > 0 and args.device == "cuda" and torch.cuda.memory_reserved() > (
            (args.mem_fraction or 1.0) * torch.cuda.get_device_properties(0).total_memory - args.frag_gb * 1e9)

    with open(run / "metrics.jsonl", "a") as mlog:
        for step in range(start_step, steps + 1):
            # (--resume: the saved step's growth, freeze, evaluation and checkpoint already happened before the save;
            # only its schedules and its train step run)
            first = rs is not None and step == start_step
            # ---- growth on a fixed schedule
            if step in grow_steps and not first:
                k = grow_steps.index(step) + 1
                target = {key: round(start[key] + (final[key] - start[key]) * k / len(grow_steps)) for key in final}
                target["mlp"] = 64 * round(target["mlp"] / 64)
                snap = snapshot_optimizer(model, opt) if args.opt == "adamw" else None
                dh, dm, dl = (target[x] - model.cfg[x] for x in ("heads", "mlp", "layers"))
                if dh > 0:
                    model.grow_heads(dh)
                if dm > 0:
                    model.grow_mlp(dm)
                if dl > 0:
                    mapping = model.grow_layers(dl, args.grow_where, args.grow_init)
                    if snap is not None:
                        snap = remap_snapshot(snap, mapping)
                    else:
                        opt.state = {remap_name(n, mapping): s for n, s in opt.state.items()}
                model.set_homeo(args.homeo, args.target, args.homeo_mode)
                model.to(args.device)
                if args.opt == "adamw":
                    opt = carry_optimizer(model, snap, make_opt(model, args.lr, args.wd))
                    last_grow = step
                else:
                    opt = carry_brain_state(opt, model, build_opt(model, args))
                    last_grow = step if not args.maturity else last_grow  # maturity replaces the global rewarm
                print(f"step {step}: GREW to layers={model.cfg['layers']} heads={model.cfg['heads']} "
                      f"mlp={model.cfg['mlp']}  ({n_params(model)/1e6:.2f}M params)", flush=True)

            # ---- smooth growth / renewal (neurogenesis)
            if neuro is not None and not first:
                snap = snapshot_optimizer(model, opt) if args.opt == "adamw" and step % neuro.every == 0 else None
                if neuro.step(step):
                    model.set_homeo(args.homeo, args.target, args.homeo_mode)
                    if args.opt == "adamw":
                        opt = carry_optimizer(model, snap, make_opt(model, args.lr, args.wd))
                    else:
                        opt = carry_brain_state(opt, model, build_opt(model, args))
                    free_cache(args.device)
                    # the last birth: free the cache once more after the next step, when the tensors at the
                    # pre-growth sizes (old gradients, activations) are gone too; the sizes do not change after it
                    settle = neuro.done()
                for i, idx in neuro.pop_renewed():
                    reset_opt_rows(opt, model, i, idx)

            # ---- memory freeze: the first step >= --mem-freeze x steps at which no more births can happen; with
            # --mem-life (plan_v2/21) the life cycle's switch and ramp decide it (and set this step's decay rates)
            if life is not None and not first:
                st0 = life.state["phase"]
                do_freeze = life.begin_step(step, neuro is None or neuro.done())
                if life.state["phase"] != st0:
                    print(f"step {step}: MEMORY SWITCH ({life.state['reason']}): ramp over {life.ramp} steps, then the "
                          f"freeze", flush=True)
            else:
                do_freeze = (mem is not None and not first and not mem.is_frozen and step >= freeze_step
                             and (neuro is None or neuro.done()))
            if do_freeze:
                info = probes.freeze(model) if probes is not None else (mem.freeze() or {})
                free_cache(args.device)  # (plan_v2/19: the probes' temporaries)
                print(f"step {step}: MEMORY FROZEN (addressing fixed; theta <- EMA changed the first read's weights by "
                      f"{info.get('mem_freeze_dw_l1', float('nan')):.4f} L1 (rows the same: "
                      f"{info.get('mem_freeze_rows_same', '-')}); set overlap over all reads "
                      f"{info.get('mem_freeze_set_overlap', float('nan')):.3f})", flush=True)
                mem.stats["freeze_step"] = step
                if life is not None:
                    life.mature()

            # ---- schedules
            frac = step / max(steps, 1)
            lr = args.lr * min(1.0, step / max(1, args.warmup * steps))
            if frac > 1 - args.decay:
                lr *= max(0.0, (1 - frac) / args.decay)
            since = step - last_grow
            if since < args.rewarm * steps:
                lr *= 0.1 + 0.9 * since / (args.rewarm * steps)
            if args.opt == "adamw":
                for g in opt.param_groups:
                    g["lr"] = lr
            if gate is not None:  # the gate's open rate of this step (plan_v2/22)
                model.mem_gate_c = gate_c(frac, gc0, gc1, gs0, gs1)
                model.gate_diag = (step + 1) % args.eval_every == 0 or step + 1 == steps   # (the residual: logged next)
            lam = 1.0 if args.spike == "analog" else min(1.0, max(0.0, (frac - a0) / max(a1 - a0, 1e-9)))
            noise = 0.0
            if args.route_noise:
                noise = args.route_noise * (max(0.0, 1 - frac / args.route_noise_decay) if args.route_noise_decay > 0 else 1.0)
            for mod in model.modules():
                if isinstance(mod, RoutedMLP):
                    mod.noise, mod.eval_noise = noise, args.eval_route_noise
            model.set_lam(lam)
            if args.spike.startswith("ngc") and last_grow >= 0 and step - last_grow <= 1000 and not first:
                for s in model.spikes():
                    s.ramp_in(1.0 / 1000)  # newborn channels join the normalisation over 1000 steps
            if data.buckets:
                phase = sum(frac >= f for f in curr_at)
                if phase != data.phase:
                    data.phase = phase
                    print(f"step {step}: curriculum phase {phase} (training on the {phase + 1} easiest of "
                          f"{len(data.buckets)} buckets)", flush=True)
            if (args.redo_every and args.homeo and lam > 0 and step and step % args.redo_every == 0 and not first
                    and train_only is None
                    and (args.homeo_mode == "full" or frac < 1 - args.decay)):  # no resets while LR decays
                n = redo_dead(model, opt)
                if n:
                    print(f"step {step}: re-initialised {n} silent MLP units", flush=True)

            if bm is not None:  # (memory_api step_begin; the blob: the step stamp, the fit window, the one-time fit)
                for line in bm.step_begin(step, args):
                    print(line, flush=True)

            # ---- eval / log
            if (step % args.eval_every == 0 or step == steps) and not first:
                if empty_eval:
                    free_cache(args.device)
                with eval_mem():
                    val, sp = evaluate(model, data, args, track=args.spike != "analog" and lam > 0)
                if (bm is not None and bm.kind == "blob" and args.gm_eval_every and step % args.gm_eval_every == 0
                        and bm.ready):
                    sp.update(gm_eval_delta(model, data, args))   # (eval mode, nothing written: memory on / off)
                dt = time.time() - t_log
                gts = {}
                rec = dict(step=step, tokens=step * tok_per_step, flops=flops, val=val, train=ema, lr=lr, lam=lam,
                           params=n_params(model), tok_s=tok_since / dt if dt > 0 else 0,
                           mem_gb=torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else 0,
                           mem_reserved_gb=torch.cuda.memory_reserved() / 1e9 if args.device == "cuda" else 0,
                           active=model.n_active(), elapsed=time.time() - t_start, **model.cfg, **sp)
                if model.cfg.get("concept") and model.concept_acc is not None:
                    rec["concept_acc"] = [a.item() for a in model.concept_acc]
                if args.route_noise:
                    rec["route_noise"] = noise
                if args.arch == "loop":
                    rec.update(model.diag)
                    if getattr(model, "think_last", None) is not None:  # thinking-concept loss, last train step
                        rec["think_loss"] = model.think_last.item()
                    if getattr(model, "think_stats", None):  # latent target: collapse detector and fit
                        rec.update({f"think_{k}": v.item() for k, v in model.think_stats.items()})
                    if getattr(model, "halt_stats", None):  # last training step: where the stop head puts its mass
                        rec["halt_train_expected_passes"] = model.halt_stats["expected_passes"].item()
                    if getattr(model, "train_route", None) is not None:  # last training step (eval has no noise)
                        rec["train_distinct_experts"] = model.train_route[0].item()
                        rec["train_drop"] = model.train_route[1].item()
                    if gate is not None:  # (plan_v2/22) the last training forward's gate diagnostics
                        gts = gate_train_stats(model)
                        rec.update(gts)
                    gn = model.head_group_grad_norms()  # from the last training step's backward
                    if gn is not None:
                        rec["head_group_grad_norm"] = gn
                    if step in report_steps:
                        free_cache(args.device)  # (plan_v2/21: the report's long loops are the eval peak)
                        with eval_mem():
                            rep = loop_report(model, data, args)
                        free_cache(args.device)
                        if gate is not None and args.mem_gate_report:
                            rep.update(gate_report(model, data, args))
                            free_cache(args.device)
                        rec.update(rep)
                        print("  loop report: loss by passes " + " ".join(f"K{k}:{v:.3f}" for k, v in rep["loss_at_K"].items())
                              + "  | after each pass " + " ".join(f"{v:.2f}" for v in rep["tick_loss"])
                              + (f"  | memory knocked out {rep['loss_state_knockout']:.3f}" if "loss_state_knockout" in rep else "")
                              + (f"  | workspace knocked out: last pass {rep['loss_ws_knockout_last']:.3f}, all passes {rep['loss_ws_knockout_all']:.3f}"
                                 if "loss_ws_knockout_last" in rep else "")
                              + (f"  | loop until sure: loss {rep['halt']['loss']:.3f} with {rep['halt']['passes']:.2f} passes per word"
                                 if "halt" in rep else "")
                              + (f"  | 'more passes help' AUC: thought vagueness {rep['think']['auc_concept_entropy']:.2f}, "
                                 f"thought change {rep['think']['auc_concept_change']:.2f}, output change "
                                 f"{rep['think']['auc_output_change']:.2f}" if "think" in rep else "")
                              + ("  | memory knocked out: " + " ".join(f"{k[12:]} {v:.3f}" for k, v in rep.items()
                                                                       if k.startswith("loss_mem_ko_"))
                                 if any(k.startswith("loss_mem_ko_") for k in rep) else "")
                              + gate_line(rep), flush=True)
                if bm is not None:  # (memory_api log_record)
                    rec.update(bm.log_record(step))
                if neuro is not None:
                    rec.update(births=neuro.births, renewals=neuro.renewals, mlp_widths=model.cfg["mlp"])
                rs = model.routing_stats()
                if rs:
                    rec["routing"] = rs
                    spk_r = "  route ent {:.2f} dead {:.2f} max/mean {:.1f} drop {:.3f}".format(
                        *(sum(r[key] for r in rs) / len(rs) for key in ("entropy", "dead", "maxload", "drop")))
                    nb = newborn_stats(model)
                    if nb:
                        rec["newborn"] = nb
                        spk_r += "  | load by age " + " ".join(
                            f"{k}:{v['load']:.2f}/{v['dead']:.2f}d(n{v['n']})" for k, v in nb.items())
                    if "route_noise" in rec:
                        spk_r += "  | train: noise {:.3f} experts/token {:.1f} drop {:.3f}".format(
                            rec["route_noise"], rec.get("train_distinct_experts", 0.0), rec.get("train_drop", 0.0))
                else:
                    spk_r = ""
                mem_line = ""
                if mem is not None:  # drift probes, stability guard, repair (after the routing stats: own forwards)
                    probe_now = n_eval % max(1, args.mem_probe_every) == 0 or step == steps
                    cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
                    res0 = None
                    if cuda and probe_now and probes is not None:  # (plan_v2/19, 21) free the cache around the probes
                        res0 = torch.cuda.memory_reserved()
                        free_cache(args.device)
                    mrec, alarm = memory_metrics(model, probes if probe_now else None)
                    if res0 is not None:  # the reserved VRAM before / after the probes, and their own peak
                        mrec["mem_reserved_probe_gb"] = [res0 / 1e9, torch.cuda.memory_reserved() / 1e9]
                        free_cache(args.device)
                    if life is not None and "mem_last_stab_w" in mrec:  # the life cycle's switch signal
                        life.observe(mrec["mem_last_stab_w"])
                    rec.update(mrec)
                    mem_line = memory_line(mrec)
                    if alarm:
                        print(f"step {step}: MEMORY DRIFT: w-weighted stability {mrec['mem_stab_w']:.3f} < guard "
                              f"{args.mem_guard} (input drift by level "
                              + "/".join(f"{v:.3f}" for v in mrec.get("mem_input_drift_by_level", [])) + ")", flush=True)
                mlog.write(json.dumps(rec) + "\n")
                mlog.flush()
                spk = f"  fire {sp['fire']:.2f} dead {sp['dead']:.3f} sat {sp['saturated']:.3f}" if "fire" in sp else ""
                if "concept_acc" in rec:
                    spk += "  concept acc " + "/".join(f"{a:.2f}" for a in rec["concept_acc"])
                if args.arch == "loop" and model.diag:
                    spk += "  experts/token {:.1f} overlap lag1/2/4 {:.2f}/{:.2f}/{:.2f}".format(
                        model.diag.get("distinct_experts", 0), model.diag.get("overlap_lag1", 0),
                        model.diag.get("overlap_lag2", 0), model.diag.get("overlap_lag4", 0))
                    if "wave_travel_cells" in model.diag:
                        spk += "  wave travel {:.2f} cells, neighbour share {:.2f} (gamma {:.2f} beta {:.2f})".format(
                            model.diag["wave_travel_cells"], model.diag.get("wave_neighbour_share", 0),
                            model.diag["wave_gamma"], model.diag["wave_beta"])
                if neuro is not None:
                    spk += f"  births {neuro.births} renew {neuro.renewals} params {n_params(model)/1e6:.2f}M"
                if "val_t0" in sp:
                    spk += "  easy/med/hard " + "/".join(f"{sp[f'val_t{k}']:.3f}" for k in range(len(data.val_tiers)))
                print(f"step {step:6d}/{steps}  tok {rec['tokens']/1e6:7.1f}M  val {val:.4f}  "
                      f"train {ema if ema is None else round(ema, 4)}  lr {lr:.2e}  lam {lam:.2f}  "
                      f"{rec['tok_s']/1e3:.0f}k tok/s  mem {rec['mem_gb']:.1f}GB res {rec['mem_reserved_gb']:.1f}GB  "
                      f"{rec['elapsed']/60:.1f} min{spk}{spk_r}{mem_line}" + (gate_line(gts) if gate is not None else ""),
                      flush=True)
                if str(args.device).startswith("cuda") and step:  # VRAM spill guard (Windows spills a full GPU into slow shared memory)
                    # (plan_v2/19) act on the FIRST eval with reserved > --spill-gb AND tok/s down by > --spill-drop
                    slow = len(tok_hist) >= 6 and rec["tok_s"] < (1 - args.spill_drop) * float(np.median(tok_hist[-12:]))
                    spill = rec["mem_reserved_gb"] > args.spill_gb and slow
                    spill_run = spill_run + 1 if spill else 0
                    if not slow:
                        tok_hist.append(rec["tok_s"])
                    if spill:
                        print(f"step {step}: VRAM SPILL: reserved {rec['mem_reserved_gb']:.1f} GB (limit {args.spill_gb}), "
                              f"{rec['tok_s']/1e3:.0f}k tok/s vs median {np.median(tok_hist[-12:])/1e3:.0f}k"
                              + (" - persists after free_cache" if spill_freed else " - freeing the cache"), flush=True)
                        if not spill_freed:
                            free_cache(args.device)
                            spill_freed = True
                    else:  # (recovered: a later spill frees the cache again)
                        spill_freed = False
                if args.save_report_ckpts and step in report_steps:  # (plan_v2/21: for eval_world.py)
                    torch.save({"cfg": model.cfg, "state": model_state(), "step": step}, run / f"ckpt_{step}.pt")
                    save_mem(run / f"ckpt_{step}.pt", step)
                    prune_ckpts(run, args.keep_ckpts)
                t_log, tok_since, n_eval = time.time(), 0, n_eval + 1
            if args.ckpt_every and step and step % args.ckpt_every == 0 and step < steps and not first:
                # the full checkpoint (plan_v2/19), atomic: everything --resume needs to continue with this step's
                # train step exactly as without the stop
                save_checkpoint(run / "last.pt", full_ckpt(step))
                save_mem(run / "last.pt", step)
            if step in save_at and not first:  # (--save-at: extra model checkpoints, pruned like the report ones)
                torch.save({"cfg": model.cfg, "state": model_state(), "step": step}, run / f"ckpt_{step}.pt")
                save_mem(run / f"ckpt_{step}.pt", step)
                prune_ckpts(run, args.keep_ckpts)
            if args.stop_at and step == args.stop_at and step < steps and not first:
                # (--stop-at: the full resumable state of this step, then a clean exit; --resume runs its train step)
                save_checkpoint(run / "last.pt", full_ckpt(step))
                save_mem(run / "last.pt", step)
                print(f"step {step}: STOPPED (--stop-at {args.stop_at}); resume with --resume {run / 'last.pt'}",
                      flush=True)
                if diagw is not None:
                    diagw.flush()
                if lock and lock.exists() and lock.read_text().split()[0] == str(os.getpid()):
                    lock.unlink()
                return
            if step == steps:
                break

            # ---- train step: `accum` micro-batches, one optimizer step. K, the deep-supervision pass and the
            # head-group rotation are drawn once per step, so every micro-batch runs the same network;
            # each micro loss is divided by accum, so the gradient is the mean over the effective batch.
            if args.arch == "loop":
                model.K = model.sample_K() if args.k_min != args.k_max else args.k_mean
                if args.accum > 1:  # (accum = 1 keeps the old in-run sampling, same random stream as before)
                    model.aux_k_fixed = model.sample_aux_k(model.K)
                model.group_offset = model.sample_group_offset()
            scoring = neuro is not None and neuro.want_score(step)
            if scoring:
                neuro.attach()
            if mem is not None:
                mem.write_on = step >= args.mem_write_start
            opt.zero_grad(set_to_none=True)
            if gate_opt is not None:
                gate_opt.zero_grad(set_to_none=True)
            micro = []
            data.ctr = step * args.accum  # (stream mode: micro-batch c = step * accum + i; a no-op for random windows)
            for _ in range(args.accum):
                x, y = data.batch(args.batch)  # two draws of 16 == one draw of 32 (same numpy stream)
                if bm is not None and hasattr(bm, "set_streams"):   # (fastmem per stream: each row's stream / user)
                    bm.set_streams(*(data.last_streams or (None,)))
                if bm is not None and near_cap():
                    torch.cuda.empty_cache()  # (also per micro-batch, but only near the cap: freeing every time is slow)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    logits, loss = model(x, y)
                    if bm is not None:  # (25d writes: this micro-batch's candidates, chosen before backward)
                        bm.select_writes(logits, y, x, ce_h=model.last_readout if logits is None else None)
                    total = loss
                    if teacher is not None:
                        with torch.no_grad():
                            t_logits = teacher(x)
                        kd = F.kl_div(F.log_softmax(logits.float(), -1).flatten(0, 1),
                                      F.log_softmax(t_logits.float(), -1).flatten(0, 1), log_target=True, reduction="batchmean")
                        total = loss + args.kd * kd
                    aux = model.aux_loss()
                    if aux is not None:
                        total = total + aux
                    if model.concept_loss is not None:
                        total = total + args.concept_w * model.concept_loss
                    if args.arch == "loop" and model.aux_x is not None:  # deep supervision on an earlier pass
                        if args.chunk_ce:
                            total = total + 0.2 * chunked_readout(model.heads["lm"], model.aux_x, y, args.chunk_ce)[0].mean()
                        else:
                            total = total + 0.2 * F.cross_entropy(model.heads["lm"](model.aux_x).float().flatten(0, 1),
                                                                  y.flatten())
                if train_only is not None and not total.requires_grad:
                    pass   # (train-only: nothing of this micro-batch reaches a trainable part)
                else:
                    (total / args.accum if args.accum > 1 else total).backward()
                if bm is not None:  # (25d: the gate's post-backward update, confirmation / write gradients)
                    bm.after_backward()
                micro.append(loss.detach())
                del logits, total
            if scoring:
                neuro.detach_and_score()
            if args.nan_abort:  # (one extra host sync per step, only with the flag)
                gn_ = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
                lt_ = sum(micro) / len(micro)
                if not math.isfinite(float(torch.stack([lt_.float(), gn_.float()]).sum())):
                    save_checkpoint(run / "nan_abort.pt", full_ckpt(step))
                    save_mem(run / "nan_abort.pt", step)
                    print(f"step {step}: NAN ABORT: loss {float(lt_)} grad norm {float(gn_)}; the state before this "
                          f"optimizer step is in {run / 'nan_abort.pt'}", flush=True)
                    if diagw is not None:
                        diagw.flush()
                    if lock and lock.exists() and lock.read_text().split()[0] == str(os.getpid()):
                        lock.unlink()
                    return
            if gate_opt is None:
                torch.nn.utils.clip_grad_norm_([p for _, p in brain_params(model)] if memory_owns_params(model)
                                               else model.parameters(), 1.0)
            else:  # (plan_v2/22) the router: its own clip and AdamW, outside the brain's global clip
                torch.nn.utils.clip_grad_norm_([p for _, p in brain_params(model)], 1.0)
                torch.nn.utils.clip_grad_norm_(list(gate.parameters()), args.mem_gate_clip)
                gate_opt.step()
            opt.step() if args.opt == "adamw" else opt.step(lr / args.lr)
            if args.arch == "loop":
                model.blocks[0].mlp.apply_balance()  # pending counts summed over all micro-batches
            if bm is not None:  # (plan_v2/25d) the content update on the touched rows, the IVF refit
                if near_cap():
                    torch.cuda.empty_cache()  # (only near the cap, like the per-micro-batch one: freeing every step is slow)
                bm.after_step(step, refit_every=args.gm_ivf_refit)
                if buf_freeze is not None:   # (train-only: the frozen brain's buffers as loaded)
                    buf_freeze.restore()
                if near_cap():
                    torch.cuda.empty_cache()
            if diagw is not None and (step + 1) % args.diag_every == 0:
                diagw.push(step + 1, *diag_collect(model, neuro))
            if args.health_every and (step + 1) % args.health_every == 0:
                hrec = dict(step=step + 1, vram_peak_reserved_gb=(torch.cuda.max_memory_reserved() / 1e9
                                                                  if str(args.device).startswith("cuda") and
                                                                  torch.cuda.is_available() else 0.0))
                if bm is not None:
                    hrec.update(bm.health())
                with open(run / "health.jsonl", "a") as hf:
                    hf.write(json.dumps(hrec) + "\n")
            lt, li = sum(micro) / len(micro), None
            if mem is not None:  # MemOpt table writes, EMA keys / statistics, private-key Adam, balancing
                cnt = mem.write_counts()  # the rows MemOpt touches: their counts come with the loss (ONE host sync)
                if cnt is not None:
                    vals = torch.cat([lt.double().view(1), cnt.double()]).tolist()
                    li, cnt = vals[0], [int(v) for v in vals[1:]]
                mem.after_step(lr, cnt)
            if settle:
                free_cache(args.device)
                settle = False
            li = lt.item() if li is None else li
            if not math.isfinite(li):
                print(f"step {step}: loss is {li}, stopping", flush=True)
                break
            ema = li if ema is None else 0.98 * ema + 0.02 * li
            flops += 6 * model.n_active() * tok_per_step * (model.K if args.arch == "loop" else 1)
            tok_since += tok_per_step

    if diagw is not None:
        diagw.flush()
    torch.save({"cfg": model.cfg, "state": model_state()}, run / "final.pt")
    save_mem(run / "final.pt", steps)
    with eval_mem():
        text = sample(model, tok, args.device)
    (run / "sample.txt").write_text(text, encoding="utf-8")
    print("sample:", text.replace("\n", " ")[:600], flush=True)
    print(f"done in {(time.time()-t_start)/60:.1f} min, total {flops:.3e} FLOPs", flush=True)
    if lock and lock.exists() and lock.read_text().split()[0] == str(os.getpid()):
        lock.unlink()


if __name__ == "__main__":
    main()
