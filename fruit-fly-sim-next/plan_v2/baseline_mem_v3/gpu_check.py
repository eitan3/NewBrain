"""GPU check of the memory speed package (plan_v2/14_memory_speed_plan.md): old code vs new code on the REAL v4d
memory config, real data, the trainer's step (BrainOpt, apply_balance, write_counts + the loss sync, after_step),
under the trainer's autocast bf16.

    python -B gpu_check.py --device cuda --steps 20                  # the check (needs the GPU to itself)
    python -B gpu_check.py --device cuda --steps 20 --deterministic  # + deterministic kernels (bit-for-bit mode)
    python -B gpu_check.py --device cpu --tiny --steps 6             # a smoke test of the script (CPU, tiny)

Configurations (each --steps optimizer steps, the same seed, the same batches from train.bin, the same K / deep-
supervision pass / head-group offset per step, writes from step 0, balancing every --balance-every steps):
    a   OLD code (plan_v2/baseline_mem_v2)
    a2  OLD code again: the run-to-run noise (index_add_ / atomics make CUDA nondeterministic)
    b   NEW code, mild flags OFF (readdr_every 1, margin_passes 0, balance_max 0, keep off; margin_fused and
        READ_DW_BMM at their defaults = ON)
    c   NEW code, mild flags ON (readdr_every 2, readdr_start 3, margin_passes 1, balance_max 0.02, keep on)
"Identical" on CUDA = the old-vs-new difference is no larger than the old-vs-old one (unless --deterministic makes
old-vs-old bit-identical: then old-vs-new is judged bit for bit).

The model: train_lm.main's OWN argument parser and construction code are run (with the v4d flags + the memory
flags) up to the LoopBrain(...) call, whose arguments are captured (main is stopped there: nothing is built, the
run directory is a temporary one); both the old and the new LoopBrain are then built from those arguments. The lines
after the construction (set_homeo, build_opt, the memory settings, opt_reset) are replicated from train_lm.main, with
each code version's own build_opt / reset_mem_key_rows. Not reproduced: growth (neurogenesis births / scoring),
evaluation, the freeze (all outside a short window from the start).

Output: a table on stdout and the JSON (--out, default gpu_check.json next to this file).
"""
import argparse
import gc
import importlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BASE = HERE / "plan_v2" / "baseline_mem_v2"
MODS = ("spiking_lm", "loop_brain", "brain_opt", "neurogenesis", "train_lm", "memory", "concepts")
DATA = r"X:\Datasets\wikipedia_en_20231101_wiki16k"

# the v4d run's flags (runs/wiki_v4d_12k) + the memory
V4D = ("--arch loop --spike graded --homeo 0.01 --homeo-mode floor --opt brain --batch 16 --accum 2 --eval-batch 32 "
       "--route-aux 0.02 --capacity 1.5 --moe 768,6,192,384,auxfree --heads 36 --start heads=6,experts=192 --neuro "
       "--neuro-every 25 --neuro-t 0.05,0.6 --head-group 6 --head-groups 6 --state none --waves --slots 16 --ws-direct "
       "--anneal 0.05,0.2 --birth-mode split --birth-delta 0.1 --route-z 1e-4 --route-noise 0.3 --route-noise-decay 0 "
       "--deep-sup 0.3 --halt 1 --think-w 0.1 --think-fb 1 --think-target latent --think-n 16 --think-use 1 "
       "--concepts 16,32,64,128 --mem-shared 512 --mem-private 512 --mem-read0 0").split()
# --tiny: the same flags, small sizes (argparse: the last value wins)
TINY = ("--d 32 --d-state 8 --heads 8 --head-dim 8 --start heads=4,experts=8 --moe 12,2,16,16,auxfree "
        "--head-group 4 --head-groups 2 --slots 4 --think-n 4 --k-mean 4 --k-min 2 --k-max 6 --bptt 3 "
        "--concepts 2,4,8,16 --mem-shared 8 --mem-private 8 --mem-dv 16 --mem-topm 4 --mem-topk 3 --seq 64 "
        "--batch 2 --accum 2 --eval-batch 2").split()


def cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tiny", action="store_true", help="a tiny config (smoke test of this script)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--tokens", default="200e6", help="the run's token budget (schedules, BrainOpt tau): v4d 200e6")
    ap.add_argument("--at-frac", type=float, default=0.1,
                    help="the schedules (lr, spike blend lam, route noise) as at this fraction of the run "
                         "(0.1: lr at its peak, lam 0.33; 0 = the run's very first steps, lr ~ 0)")
    ap.add_argument("--balance-every", type=int, default=10)
    ap.add_argument("--K", type=int, default=0, help="fixed number of passes (0 = sampled as the trainer does, "
                                                     "one schedule shared by all configs)")
    ap.add_argument("--probe-batch", type=int, default=4, help="sequences in the address-agreement probe batch")
    ap.add_argument("--configs", default="a,a2,b,c")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG=:4096:8")
    ap.add_argument("--ref-mode", action="store_true",
                    help="config b with margin_fused = False and READ_DW_BMM = False (the reference mode of "
                         "test_speed.test_01: bit-identical to the old code where the math order allows)")
    ap.add_argument("--no-sync-count", action="store_true", help="skip the host-sync count (CUDA, step 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "gpu_check.json"))
    return ap.parse_args()


A = cli()
if A.deterministic:  # before CUDA is initialised
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# ---------------------------------------------------------------- the two code versions as module objects
def current():
    import brain_opt
    import loop_brain
    import memory
    import neurogenesis
    import spiking_lm
    import train_lm
    return SimpleNamespace(spiking_lm=spiking_lm, loop_brain=loop_brain, brain_opt=brain_opt,
                           neurogenesis=neurogenesis, train_lm=train_lm, memory=memory)


def baseline():
    """test_speed.baseline(): plan_v2/baseline_mem_v2 imported as separate module objects (the current modules are
    out of sys.modules while it imports, then put back; no bytecode written)."""
    saved = {n: sys.modules.pop(n) for n in MODS if n in sys.modules}
    dont = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(BASE))
    try:
        out = SimpleNamespace(**{n: importlib.import_module(n) for n in MODS[:6]})
    finally:
        sys.path.remove(str(BASE))
        sys.dont_write_bytecode = dont
        for n in MODS:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    assert Path(out.memory.__file__).parent == BASE
    return out


# ---------------------------------------------------------------- train_lm's own parser + construction code
class _Stop(Exception):
    pass


def capture_construction(train_lm, flags):
    """Runs train_lm.main() with these flags up to its LoopBrain(...) call and returns (args, LoopBrain positional
    args, LoopBrain kwargs). main's side effects before that point: a temporary run directory (train.log,
    config.json, run.pid, gpu.lock there), the seeds, TF32 on (as in training); stdout / stderr are restored."""
    box = {}

    def fake_loopbrain(*a, **kw):
        box["a"], box["kw"] = a, kw
        raise _Stop

    orig_parse = argparse.ArgumentParser.parse_args

    def parse(self, args=None, namespace=None):
        ns = orig_parse(self, args, namespace)
        box["args"] = ns
        return ns

    tmp = tempfile.mkdtemp(prefix="gpu_check_runs_")
    argv, out, err, lb = sys.argv, sys.stdout, sys.stderr, train_lm.LoopBrain
    sys.argv = ["train_lm.py", "--run-name", "gpu_check", "--runs", tmp] + flags
    train_lm.LoopBrain = fake_loopbrain
    argparse.ArgumentParser.parse_args = parse
    try:
        train_lm.main()
    except _Stop:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig_parse
        train_lm.LoopBrain = lb
        sys.argv, sys.stdout, sys.stderr = argv, out, err
    assert "kw" in box, "train_lm.main() did not reach LoopBrain(...)"
    return box["args"], box["a"], dict(box["kw"])


def build(mods, cap, device, new, mild, ref_mode):
    """The model and optimizer as train_lm.main builds them (lines after LoopBrain(...) replicated)."""
    args, a, kw = cap
    torch.manual_seed(A.seed)
    np.random.seed(A.seed)
    model = mods.loop_brain.LoopBrain(*a, **kw).to(device)
    model.set_homeo(args.homeo, args.target, args.homeo_mode)
    box = [mods.train_lm.build_opt(model, args)]
    mem = model.mem
    mem.lr, mem.clip, mem.s0_cfg = args.mem_lr, args.mem_clip, args.mem_s0
    mem.balance_every = A.balance_every                     # (forced: one event inside the window)
    if new:
        mem.balance_max, mem.balance_keep = (0.02, True) if mild else (0.0, False)
        model.mem_readdr_every, model.mem_readdr_start, model.mem_margin_passes = (2, 3, 1) if mild else (1, 4, 0)
        if ref_mode and not mild:
            mem.margin_fused = False
    mem.opt_reset = lambda name, sl: mods.train_lm.reset_mem_key_rows(box[0], model, name, sl)
    return model, box


# ---------------------------------------------------------------- data, schedule
def make_batches(train_lm, args, steps, device):
    data = train_lm.Data(Path(args.data), args.seq, "cpu")
    gen = np.random.RandomState(args.seed)  # == np.random.seed(seed) + data.batch(B), the trainer's stream
    bs = [data.batch(args.batch, "train", gen) for _ in range(steps * args.accum)]
    pg = np.random.RandomState(4242)
    probe = data.batch(A.probe_batch, "val", pg)
    if device == "cuda":
        bs = [(x.pin_memory(), y.pin_memory()) for x, y in bs]
    return bs, probe


def make_schedule(model, steps, accum):
    """K, the deep-supervision pass and the head-group offset per step, drawn once with the model's own sampling
    code on a dedicated RNG stream (the global stream is left alone): every config gets the same sequence."""
    out = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(A.seed + 777)
        for _ in range(steps):
            K = A.K or model.sample_K()
            out.append(dict(K=K, aux_k=model.sample_aux_k(K) if accum > 1 else None,
                            offset=model.sample_group_offset()))
    return out


def schedules(args, g, total):
    frac = g / max(total, 1)
    lr = args.lr * min(1.0, g / max(1, args.warmup * total))
    if frac > 1 - args.decay:
        lr *= max(0.0, (1 - frac) / args.decay)
    a0, a1 = (float(f) for f in args.anneal.split(","))
    lam = 1.0 if args.spike == "analog" else min(1.0, max(0.0, (frac - a0) / max(a1 - a0, 1e-9)))
    noise = 0.0
    if args.route_noise:
        noise = args.route_noise * (max(0.0, 1 - frac / args.route_noise_decay) if args.route_noise_decay > 0 else 1.0)
    return lr, lam, noise


# ---------------------------------------------------------------- one configuration
def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def probe_records(model, xb, yb, dev, K):
    """Eval forward (K = k_mean, offset 0, autocast) on the fixed probe batch with the memory's record on: the
    selected shared rows of every read and the private rows of every pass, at every position."""
    mem = model.mem
    was, K0, off0 = model.training, model.K, getattr(model, "probe_offset", None)
    model.eval()
    model.K = K
    model.probe_offset = None
    xb, yb = xb.to(dev), yb.to(dev)
    mem.record = dict(pos=torch.arange(xb.numel(), device=dev), reads=[], calls=[])
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            _, loss = model(xb, yb)
        rec = mem.record
    finally:
        mem.record = None
        model.K, model.probe_offset = K0, off0
        model.train(was)
    rows = [r["rows"] for r in rec["reads"] if torch.is_tensor(r.get("rows"))]
    prows = [c["prows"].cpu() for c in rec["calls"] if torch.is_tensor(c.get("prows"))]
    return dict(loss=float(loss), rows=rows, prows=prows)


def run_config(name, mods, cap, batches, sched, probe, dev, new, mild=False):
    args = cap[0]
    tl, mem_mod = mods.train_lm, mods.memory
    RoutedMLP = mods.spiking_lm.RoutedMLP
    rd = getattr(mem_mod, "READ_DW_BMM", None)
    if new and A.ref_mode and not mild:
        mem_mod.READ_DW_BMM = False
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.manual_seed_all(A.seed)
    res = dict(name=name, code="new" if new else "old", mild=mild)
    t_build = time.perf_counter()
    model, box = build(mods, cap, dev, new, mild, A.ref_mode)
    mem = model.mem
    res["flags"] = dict(margin_fused=getattr(mem, "margin_fused", None), READ_DW_BMM=getattr(mem_mod, "READ_DW_BMM", None),
                        readdr_every=getattr(model, "mem_readdr_every", None),
                        readdr_start=getattr(model, "mem_readdr_start", None),
                        margin_passes=getattr(model, "mem_margin_passes", None),
                        balance_max=getattr(mem, "balance_max", None), balance_keep=getattr(mem, "balance_keep", None),
                        balance_every=mem.balance_every)
    if sched[0] is None:
        sched[0] = make_schedule(model, A.steps, args.accum)
    total = int(args.tokens // (args.batch * args.accum * args.seq))
    g0 = int(round(A.at_frac * total))
    sync(dev)
    res["build_s"] = time.perf_counter() - t_build
    losses, times, Ks, bal_steps, sync_count = [], [], [], [], None
    try:
        for i in range(A.steps):
            sc = sched[0][i]
            lr, lam, noise = schedules(args, g0 + i, total)
            count_syncs = dev == "cuda" and i == 2 and not A.no_sync_count
            sync(dev)
            t0 = time.perf_counter()
            if count_syncs:
                try:
                    torch.cuda.set_sync_debug_mode("warn")
                    wctx = warnings.catch_warnings(record=True)
                    wlist = wctx.__enter__()
                    warnings.simplefilter("always")
                except Exception as e:  # noqa: BLE001 -- optional measurement
                    print(f"  (host-sync count unavailable: {e})", flush=True)
                    torch.cuda.set_sync_debug_mode(0)
                    count_syncs = False
            # ---- train_lm.main's step (the schedules, then `accum` micro-batches, one optimizer step)
            for mod in model.modules():
                if isinstance(mod, RoutedMLP):
                    mod.noise, mod.eval_noise = noise, args.eval_route_noise
            model.set_lam(lam)
            model.K = sc["K"]
            if args.accum > 1:
                model.aux_k_fixed = sc["aux_k"]
            model.group_offset = sc["offset"]
            mem.write_on = True                                 # writes from step 0
            opt = box[0]
            opt.zero_grad(set_to_none=True)
            micro = []
            for j in range(args.accum):
                x, y = batches[i * args.accum + j]
                if dev == "cuda":
                    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    logits, loss = model(x, y)
                    total_l = loss
                    aux = model.aux_loss()
                    if aux is not None:
                        total_l = total_l + aux
                    if getattr(model, "concept_loss", None) is not None:
                        total_l = total_l + args.concept_w * model.concept_loss
                    if model.aux_x is not None:
                        total_l = total_l + 0.2 * F.cross_entropy(model.heads["lm"](model.aux_x).float().flatten(0, 1),
                                                                  y.flatten())
                (total_l / args.accum if args.accum > 1 else total_l).backward()
                micro.append(loss.detach())
                del logits, total_l
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(lr / args.lr)
            model.blocks[0].mlp.apply_balance()
            lt, li = sum(micro) / len(micro), None
            if new:  # the touched-row counts read with the loss (one host sync), then after_step
                cnt = mem.write_counts()
                if cnt is not None:
                    vals = torch.cat([lt.double().view(1), cnt.double()]).tolist()
                    li, cnt = vals[0], [int(v) for v in vals[1:]]
                mem.after_step(lr, cnt)
            else:
                mem.after_step(lr)
            li = lt.item() if li is None else li
            if count_syncs:
                torch.cuda.set_sync_debug_mode(0)
                wctx.__exit__(None, None, None)
                sync_count = sum("synchroniz" in str(w.message) for w in wlist)
            sync(dev)
            dt = time.perf_counter() - t0
            balanced = mem.balance_every and (i + 1) % mem.balance_every == 0 and not mem.is_frozen
            losses.append(li)
            times.append(dt)
            Ks.append(sc["K"])
            if balanced:
                bal_steps.append(i)
            print(f"  [{name}] step {i:3d}  K {sc['K']:2d}  loss {li:.6f}  {dt:7.3f} s"
                  + ("  (balancing)" if balanced else "") + (f"  host syncs {sync_count}" if count_syncs else ""),
                  flush=True)
            if not math.isfinite(li):
                res["error"] = f"loss {li} at step {i}"
                break
        res["balance_stats"] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                                for k, v in dict(mem.stats.get("balance", {})).items()}
        res["probe"] = probe_records(model, probe[0], probe[1], dev, args.k_mean)
    finally:
        if dev == "cuda":
            torch.cuda.set_sync_debug_mode(0)
        if rd is not None:
            mem_mod.READ_DW_BMM = rd
        res.update(losses=losses, times=times, K=Ks, balance_steps=bal_steps, host_syncs_step2=sync_count)
        if dev == "cuda":
            res["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
            res["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        del model, box, mem
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- comparisons and report
def timing(r):
    t, bs = r.get("times", []), set(r.get("balance_steps", []))
    w = [x for i, x in enumerate(t) if i >= 3]
    wn = [x for i, x in enumerate(t) if i >= 3 and i not in bs]
    f = lambda v, fn: fn(v) if v else None
    return dict(mean=f(w, statistics.mean), median=f(w, statistics.median),
                mean_excl_balance=f(wn, statistics.mean), median_excl_balance=f(wn, statistics.median),
                balance_step_s=[t[i] for i in sorted(bs) if i < len(t)], n_timed=len(w))


def agreement(p, q):
    """Fraction of identical selected rows (element for element, read by read), the set overlap (rows of q among
    p's per position / level / head), and the same for the private rows."""
    if p is None or q is None:
        return None
    out = {}
    if p["rows"] and len(p["rows"]) == len(q["rows"]) and all(a.shape == b.shape for a, b in zip(p["rows"], q["rows"])):
        ra, rb = torch.stack(p["rows"]), torch.stack(q["rows"])
        out["rows_identical"] = float((ra == rb).float().mean())
        out["rows_set_overlap"] = float((ra[..., :, None] == rb[..., None, :]).any(-1).float().mean())
        out["rows_identical_by_read"] = [float((a == b).float().mean()) for a, b in zip(p["rows"], q["rows"])]
    elif p["rows"] or q["rows"]:
        out["rows_identical"] = None
        out["rows_note"] = f"different read structure: {len(p['rows'])} vs {len(q['rows'])} reads"
    if p["prows"] and len(p["prows"]) == len(q["prows"]) and all(a.shape == b.shape for a, b in zip(p["prows"], q["prows"])):
        out["prows_identical"] = float(torch.stack([(a == b).float().mean() for a, b in zip(p["prows"], q["prows"])]).mean())
    out["probe_loss_diff"] = abs(p["loss"] - q["loss"])
    return out


def loss_diff(r1, r2):
    a, b = r1.get("losses") or [], r2.get("losses") or []
    n = min(len(a), len(b))
    if not n:
        return None
    d = [abs(x - y) for x, y in zip(a[:n], b[:n])]
    return dict(max_abs=max(d), mean_abs=sum(d) / n, bit_identical=all(x == y for x, y in zip(a[:n], b[:n])),
                per_step=d, n=n)


def main():
    dev = A.device
    if dev == "cuda":
        assert torch.cuda.is_available(), "--device cuda: no CUDA device"
    if A.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"gpu_check: device {dev}, steps {A.steps}, tiny {A.tiny}, deterministic {A.deterministic}, "
          f"at-frac {A.at_frac}, balance every {A.balance_every}, ref-mode {A.ref_mode}", flush=True)
    new_m = current()
    old_m = baseline()
    flags = V4D + (TINY if A.tiny else []) + ["--data", A.data, "--device", dev, "--tokens", A.tokens,
                                             "--seed", str(A.seed)]
    cap = capture_construction(new_m.train_lm, flags)
    args = cap[0]
    kw_show = {k: v for k, v in cap[2].items()}
    print(f"LoopBrain{tuple(cap[1])} kwargs: {kw_show}", flush=True)
    batches, probe = make_batches(new_m.train_lm, args, A.steps, dev)
    sched = [None]
    plan = dict(a=(old_m, False, False), a2=(old_m, False, False), b=(new_m, True, False), c=(new_m, True, True))
    results = {}
    for name in A.configs.split(","):
        mods, new, mild = plan[name]
        print(f"=== config {name}: {'NEW' if new else 'OLD'} code{', mild flags ON' if mild else ''}", flush=True)
        try:
            results[name] = run_config(name, mods, cap, batches, sched, probe, dev, new, mild)
        except Exception as e:  # noqa: BLE001 -- OOM (or anything else): report it, go on
            oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
            print(f"  [{name}] {'OOM' if oom else 'ERROR'}: {type(e).__name__}: {str(e)[:500]}", flush=True)
            results[name] = dict(name=name, error=f"{type(e).__name__}: {str(e)[:2000]}", oom=oom,
                                 traceback=traceback.format_exc()[-4000:])
            if dev == "cuda":
                results[name]["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
                results[name]["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()

    # ---- comparisons
    R = results
    get = lambda n: R.get(n) if R.get(n) and not R[n].get("error") else None
    comp = {}
    for x, y in (("a", "a2"), ("a", "b"), ("a2", "b"), ("a", "c")):
        if get(x) and get(y):
            comp[f"{x}_vs_{y}"] = dict(loss=loss_diff(R[x], R[y]), address=agreement(R[x].get("probe"), R[y].get("probe")))
    verdict = {}
    oo, on = comp.get("a_vs_a2"), comp.get("a_vs_b")
    if oo and on:
        oo_bit = oo["loss"]["bit_identical"] and (oo["address"] or {}).get("rows_identical") == 1.0
        verdict["old_vs_old_bit_identical"] = oo_bit
        if oo_bit:
            verdict["mode"] = "bit-for-bit (old vs old is bit-identical)"
            verdict["old_vs_new_bit_identical"] = on["loss"]["bit_identical"] and \
                (on["address"] or {}).get("rows_identical") == 1.0
            verdict["pass"] = verdict["old_vs_new_bit_identical"]
        else:
            verdict["mode"] = "within noise (old vs new no larger than old vs old)"
            lo = on["loss"]["max_abs"] <= oo["loss"]["max_abs"]
            ao = (on["address"] or {}).get("rows_identical")
            ar = (oo["address"] or {}).get("rows_identical")
            verdict["loss_within_noise"] = lo
            verdict["address_within_noise"] = None if ao is None or ar is None else ao >= ar
            verdict["pass"] = lo and bool(verdict["address_within_noise"])
        if not verdict["pass"] and dev == "cpu" and not A.ref_mode:
            verdict["note"] = ("on the CPU old-vs-old is exact, so the near-exact items (margin_fused, READ_DW_BMM: "
                               "summation order) show up; use --ref-mode for the bit-for-bit CPU check")
    tim = {n: timing(r) for n, r in R.items() if r.get("times")}

    # ---- table
    print("\n" + "=" * 100)
    print(f"{'config':8s}{'code':6s}{'mild':6s}{'mean s/step':>12s}{'median':>10s}{'mean*':>10s}{'median*':>10s}"
          f"{'balance step s':>18s}{'peak alloc GB':>15s}{'reserved GB':>13s}{'syncs':>7s}")
    for n, r in R.items():
        if r.get("error"):
            print(f"{n:8s}{'':12s}  {'OOM' if r.get('oom') else 'ERROR'}: {r['error'][:70]}")
            continue
        t = tim.get(n, {})
        fm = lambda v: f"{v:10.3f}" if isinstance(v, float) else f"{'-':>10s}"
        bal = ",".join(f"{v:.3f}" for v in t.get("balance_step_s", [])) or "-"
        print(f"{n:8s}{r['code']:6s}{str(r['mild']):6s}{fm(t.get('mean')):>12s}{fm(t.get('median'))}"
              f"{fm(t.get('mean_excl_balance'))}{fm(t.get('median_excl_balance'))}{bal:>18s}"
              f"{r.get('peak_alloc_gb', float('nan')):15.2f}{r.get('peak_reserved_gb', float('nan')):13.2f}"
              f"{str(r.get('host_syncs_step2', '-')):>7s}")
    print("(steps 3..N timed; * = without the balancing steps; syncs = host syncs counted in step 2, CUDA only)")
    names = [n for n in ("a", "a2", "b", "c") if get(n)]
    if names:
        print("\nper-step loss:")
        print(f"{'step':>5s}{'K':>4s}" + "".join(f"{n:>14s}" for n in names))
        ref = R[names[0]]
        for i in range(len(ref["losses"])):
            print(f"{i:5d}{ref['K'][i]:4d}" + "".join(f"{R[n]['losses'][i]:14.6f}" if i < len(R[n]["losses"])
                                                     else f"{'-':>14s}" for n in names))
    for k, v in comp.items():
        ld, ad = v["loss"], v["address"] or {}
        print(f"{k:10s} loss max|d| {ld['max_abs']:.3e} mean|d| {ld['mean_abs']:.3e} bit-identical {ld['bit_identical']}"
              f" | rows identical {ad.get('rows_identical')} set overlap {ad.get('rows_set_overlap')}"
              f" private rows identical {ad.get('prows_identical')} | probe loss |d| {ad.get('probe_loss_diff')}")
    print("verdict:", json.dumps(verdict))

    # ---- JSON
    slim = {}
    for n, r in R.items():
        s = {k: v for k, v in r.items() if k != "probe"}
        if r.get("probe"):
            s["probe_loss"] = r["probe"]["loss"]
            s["probe_reads"] = len(r["probe"]["rows"])
        slim[n] = s
    out = dict(device=dev, gpu=torch.cuda.get_device_name() if dev == "cuda" else None, torch=torch.__version__,
               steps=A.steps, tiny=A.tiny, deterministic=A.deterministic, ref_mode=A.ref_mode, at_frac=A.at_frac,
               balance_every=A.balance_every, seed=A.seed, train_lm_flags=flags,
               loopbrain_args=[str(v) for v in cap[1]], loopbrain_kwargs={k: str(v) for k, v in cap[2].items()},
               schedule=sched[0], timing=tim, comparisons=comp, verdict=verdict, configs=slim)
    Path(A.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {A.out}", flush=True)


if __name__ == "__main__":
    main()
