"""Train SpikingLM on tokenised TinyStories, optionally growing from a small start.

    python -u train_lm.py --run-name s0_ssd_grow --mixer ssd --start layers=2,heads=2,mlp=512 \
        --grow-at 0.1,0.2,0.3 --tokens 300e6

Everything goes into runs/<run-name>/: run.pid, config.json, metrics.jsonl, final.pt, sample.txt.
Progress lines are printed with flush, so the launcher's log file is always current.
One GPU job at a time: runs/gpu.lock holds the pid of the run using the GPU.
"""
import argparse
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
from loop_brain import LoopBrain
from memory import LifeCycle, MemoryProbes
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
    def __init__(self, path, seq, device, curriculum=False):
        # token id width: uint16 unless the cache's meta.json says otherwise (large vocabularies: uint32)
        meta = path / "meta.json"
        dt = np.dtype(json.loads(meta.read_text()).get("dtype", "uint16")) if meta.exists() else np.uint16
        self.train = np.memmap(path / "train.bin", dtype=dt, mode="r")
        self.val = np.memmap(path / "val.bin", dtype=dt, mode="r")
        self.seq, self.dev = seq, device
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


def build_opt(model, args):
    if args.opt == "brain":
        # maturity time constant: 1% of optimizer steps (effective batch = batch x accum)
        tau = max(1, int(0.01 * args.tokens / (args.batch * args.accum * args.seq)))
        return BrainOpt(model, args.lr, args.wd, muon=bool(args.muon), lazy=bool(args.lazy),
                        maturity=args.maturity, tau=tau)
    return make_opt(model, args.lr, args.wd)


def make_opt(model, lr, wd):
    wd_ok = lambda n, p: p.dim() >= 2 and not n.startswith(("emb.", "mem.keys.")) and "router_conn" not in n
    decay = [p for n, p in model.named_parameters() if wd_ok(n, p)]
    other = [p for n, p in model.named_parameters() if not wd_ok(n, p)]
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
                      ("mem_A_stab_w", "A"), ("mem_last_stab_w", "vs last"), ("mem_input_drift_by_level", "drift"),
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
    ap.add_argument("--resume", default="", help="resume from this full checkpoint (runs/<name>/last.pt)")
    ap.add_argument("--save-report-ckpts", type=int, default=0,
                    help="1 = also save ckpt_<step>.pt at the 25 / 50 / 75 / 100%% report points")
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer or str(Path(args.data) / "tokenizer.json"))
    data = Data(Path(args.data), args.seq, args.device, curriculum=bool(args.curriculum))
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
        model = (LoopBrain if cfg.get("arch") == "loop" else SpikingLM)(**cfg).to(args.device)
        model.load_state_dict(rs["state"])
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
    model.set_homeo(args.homeo, args.target, args.homeo_mode)
    opt = build_opt(model, args)
    if rs is not None:  # the optimizer state (BrainOpt: per name + its step count; torch AdamW: its state_dict)
        if args.opt == "brain":
            opt.t = rs["opt"]["t"]
            opt.state = {n: {k: v.to(args.device) for k, v in st.items()} for n, st in rs["opt"]["state"].items()}
        else:
            opt.load_state_dict(rs["opt"]["state"])
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
              + (f"; neighbour writes to rank {mem.nbr} (alpha {mem.nbr_w})" if mem.nbr else ""), flush=True)

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
                    and (args.homeo_mode == "full" or frac < 1 - args.decay)):  # no resets while LR decays
                n = redo_dead(model, opt)
                if n:
                    print(f"step {step}: re-initialised {n} silent MLP units", flush=True)

            # ---- eval / log
            if (step % args.eval_every == 0 or step == steps) and not first:
                val, sp = evaluate(model, data, args, track=args.spike != "analog" and lam > 0)
                dt = time.time() - t_log
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
                    gn = model.head_group_grad_norms()  # from the last training step's backward
                    if gn is not None:
                        rec["head_group_grad_norm"] = gn
                    if step in report_steps:
                        free_cache(args.device)  # (plan_v2/21: the report's long loops are the eval peak)
                        rep = loop_report(model, data, args)
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
                                 if any(k.startswith("loss_mem_ko_") for k in rep) else ""),
                              flush=True)
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
                      f"{rec['elapsed']/60:.1f} min{spk}{spk_r}{mem_line}", flush=True)
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
                    torch.save({"cfg": model.cfg, "state": model.state_dict(), "step": step}, run / f"ckpt_{step}.pt")
                t_log, tok_since, n_eval = time.time(), 0, n_eval + 1
            if args.ckpt_every and step and step % args.ckpt_every == 0 and step < steps and not first:
                # the full checkpoint (plan_v2/19), atomic: everything --resume needs to continue with this step's
                # train step exactly as without the stop
                save_checkpoint(run / "last.pt", dict(
                    cfg=model.cfg, state=model.state_dict(), step=step,
                    opt=dict(t=opt.t, state=opt.state) if args.opt == "brain" else dict(state=opt.state_dict()),
                    neuro=neuro.state_dict() if neuro is not None else None,
                    mem=mem.host_state() if mem is not None else None,
                    probes_rng=probes.rng.get_state() if probes is not None else None,
                    trainer=dict(flops=flops, last_grow=last_grow, ema=ema, n_eval=n_eval, tok_since=tok_since,
                                 tok_hist=list(tok_hist), spill_run=spill_run, spill_freed=spill_freed, settle=settle,
                                 elapsed=time.time() - t_start, data_phase=data.phase,
                                 group_passes=list(getattr(model, "group_passes", None) or []) or None),
                    rng=get_rng_state(args.device)))
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
            micro = []
            for _ in range(args.accum):
                x, y = data.batch(args.batch)  # two draws of 16 == one draw of 32 (same numpy stream)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                    logits, loss = model(x, y)
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
                        total = total + 0.2 * F.cross_entropy(model.heads["lm"](model.aux_x).float().flatten(0, 1),
                                                              y.flatten())
                (total / args.accum if args.accum > 1 else total).backward()
                micro.append(loss.detach())
                del logits, total
            if scoring:
                neuro.detach_and_score()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step() if args.opt == "adamw" else opt.step(lr / args.lr)
            if args.arch == "loop":
                model.blocks[0].mlp.apply_balance()  # pending counts summed over all micro-batches
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

    torch.save({"cfg": model.cfg, "state": model.state_dict()}, run / "final.pt")
    text = sample(model, tok, args.device)
    (run / "sample.txt").write_text(text, encoding="utf-8")
    print("sample:", text.replace("\n", " ")[:600], flush=True)
    print(f"done in {(time.time()-t_start)/60:.1f} min, total {flops:.3e} FLOPs", flush=True)
    if lock and lock.exists() and lock.read_text().split()[0] == str(os.getpid()):
        lock.unlink()


if __name__ == "__main__":
    main()
