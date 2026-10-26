"""VRAM study helpers: load runs/world_mem_life_12k/last.pt, grow to the end-of-growth size, and run ONE training
step exactly as train_lm.py does (lines 1345-1411), with per-phase memory probes. No model code is edited; optional
"what-if" variants are applied by monkeypatching inside this process only."""
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent  # fruit-fly-sim
sys.path.insert(0, str(SRC))
torch.set_num_threads(2)

from brain_opt import carry_brain_state  # noqa: E402
from loop_brain import LoopBrain  # noqa: E402
from memory import MemoryProbes  # noqa: E402
from spiking_lm import RoutedMLP  # noqa: E402
import train_lm  # noqa: E402

RUN = SRC.parent / "runs" / "world_mem_life_12k"
GB = 1e9


def gpu_guard():
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def mstat(tag=""):
    return dict(tag=tag, alloc=torch.cuda.memory_allocated() / GB, reserved=torch.cuda.memory_reserved() / GB,
                peak_alloc=torch.cuda.max_memory_allocated() / GB, peak_reserved=torch.cuda.max_memory_reserved() / GB)


def load(full=True, ckpt=None, verbose=True):
    """Model + BrainOpt (+ its state) + memory host state, as train_lm --resume does; full: grow to heads 36 /
    experts 768 (the end-of-growth size) and carry the optimizer state the way neurogenesis does."""
    args = SimpleNamespace(**json.loads((RUN / "config.json").read_text()))
    args.eval_batch = args.eval_batch or args.batch * args.accum
    rs = torch.load(ckpt or RUN / "last.pt", map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(rs["cfg"])
    model = LoopBrain(**cfg).to("cuda")
    model.load_state_dict(rs["state"])
    model.set_homeo(args.homeo, args.target, args.homeo_mode)
    opt = train_lm.build_opt(model, args)
    opt.t = rs["opt"]["t"]
    opt.state = {n: {k: v.to("cuda") for k, v in st.items()} for n, st in rs["opt"]["state"].items()}
    mem = model.mem
    mem.lr, mem.clip, mem.s0_cfg = args.mem_lr, args.mem_clip, args.mem_s0
    mem.restart_zero = bool(args.mem_restart_zero)
    mem.balance_dead_max = args.mem_balance_dead_max
    mem.balance_every, mem.balance_max, mem.balance_keep = args.mem_balance_every, args.mem_balance_max, bool(args.mem_balance_keep)
    model.mem_readdr_every, model.mem_readdr_start, model.mem_margin_passes = (
        args.mem_readdr_every, args.mem_readdr_start, args.mem_margin_passes)
    mem.opt_reset = lambda name, sl: train_lm.reset_mem_key_rows(opt, model, name, sl)
    mem.load_host_state(rs["mem"])
    step0 = rs["step"]
    h0, e0 = model.blocks[0].mix.h, model.blocks[0].mlp.E
    if full:
        core = model.blocks[0]
        if core.mix.h < args.heads:
            core.mix.grow_heads(args.heads - core.mix.h, ramp=True)
            model.cfg["heads"] = core.mix.h
        n_exp = parse_moe_n(args.moe)
        if core.mlp.E < n_exp:
            model.grow_experts(n_exp - core.mlp.E)
        model.to("cuda")
        opt = carry_brain_state(opt, model, train_lm.build_opt(model, args))
        mem.opt_reset = lambda name, sl: train_lm.reset_mem_key_rows(opt, model, name, sl)
    del rs
    torch.cuda.empty_cache()
    if verbose:
        print(f"loaded step {step0}: heads {h0} -> {model.blocks[0].mix.h}, experts {e0} -> {model.blocks[0].mlp.E}, "
              f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    data = train_lm.Data(Path(args.data), args.seq, "cuda")
    probes = MemoryProbes(data.val, args.seq, n_seq=args.mem_probes, seed=args.seed,
                          batch=args.mem_probe_batch or args.eval_batch, guard=args.mem_guard,
                          repair=args.mem_repair, repair_usage=args.mem_repair_usage)
    return SimpleNamespace(model=model, opt=opt, args=args, data=data, mem=mem, probes=probes, step0=step0)


def parse_moe_n(s):
    return int(s.split(",")[0])


def persistent_breakdown(S):
    """Bytes of the persistent state by kind: parameters, gradients (if any), BrainOpt state, memory tables."""
    m, opt = S.model, S.opt
    out = {}
    pn = dict(m.named_parameters())
    out["params"] = sum(p.numel() * p.element_size() for p in pn.values())
    out["grads"] = sum(p.grad.numel() * p.grad.element_size() for p in pn.values() if p.grad is not None)
    by = {}
    for name, st in opt.state.items():
        for k, v in st.items():
            by[k] = by.get(k, 0) + v.numel() * v.element_size()
    out["opt_state"] = sum(by.values())
    out["opt_state_by_key"] = by
    bufs = {}
    for n, b in m.mem.named_buffers(recurse=False):
        bufs[n] = b.numel() * b.element_size()
    out["mem_buffers"] = sum(bufs.values())
    out["mem_buffers_top"] = dict(sorted(bufs.items(), key=lambda kv: -kv[1])[:16])
    other = sum(b.numel() * b.element_size() for n, b in m.named_buffers() if not n.startswith("mem."))
    out["other_buffers"] = other
    out["mem_params"] = sum(p.numel() * p.element_size() for n, p in pn.items() if n.startswith("mem."))
    out["emb_head_params"] = sum(p.numel() * p.element_size() for n, p in pn.items() if n.startswith(("emb.", "heads.")))
    out["expert_params"] = sum(p.numel() * p.element_size() for n, p in pn.items() if ".mlp.w_" in n)
    return out


def set_step_state(S, step):
    """The per-step schedules of train_lm for this step (lr factor, lam, route noise, write_on)."""
    a = S.args
    steps = int(a.tokens // (a.batch * a.accum * a.seq))
    frac = step / steps
    lr = a.lr * min(1.0, step / max(1, a.warmup * steps))
    if frac > 1 - a.decay:
        lr *= max(0.0, (1 - frac) / a.decay)
    a0, a1 = (float(f) for f in a.anneal.split(","))
    lam = min(1.0, max(0.0, (frac - a0) / max(a1 - a0, 1e-9)))
    noise = a.route_noise * (max(0.0, 1 - frac / a.route_noise_decay) if a.route_noise_decay > 0 else 1.0)
    for mod in S.model.modules():
        if isinstance(mod, RoutedMLP):
            mod.noise, mod.eval_noise = noise, a.eval_route_noise
    S.model.set_lam(lam)
    S.mem.write_on = step >= a.mem_write_start
    return lr


def train_step(S, K, step=7000, batch=None, accum=None, probe=None, sync=True):
    """ONE optimizer step as in train_lm.main (no growth / eval / ckpt), K passes forced. probe(tag) is called
    between phases. Returns the loss."""
    m, opt, a, data, mem = S.model, S.opt, S.args, S.data, S.mem
    batch, accum = batch or a.batch, accum or a.accum
    pr = probe or (lambda tag: None)
    lr = set_step_state(S, step)
    m.train()
    m.K = K
    if accum > 1:
        m.aux_k_fixed = m.sample_aux_k(m.K)
    m.group_offset = m.sample_group_offset()
    opt.zero_grad(set_to_none=True)
    micro = []
    for i in range(accum):
        x, y = data.batch(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = m(x, y)
            total = loss
            aux = m.aux_loss()
            if aux is not None:
                total = total + aux
            if m.concept_loss is not None:
                total = total + a.concept_w * m.concept_loss
            if m.aux_x is not None:
                total = total + 0.2 * F.cross_entropy(m.heads["lm"](m.aux_x).float().flatten(0, 1), y.flatten())
        pr(f"fwd{i}")
        (total / accum if accum > 1 else total).backward()
        pr(f"bwd{i}")
        micro.append(loss.detach())
        del logits, total
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    pr("clip")
    opt.step(lr / a.lr)
    pr("opt")
    m.blocks[0].mlp.apply_balance()
    lt = sum(micro) / len(micro)
    cnt = mem.write_counts()
    li = None
    if cnt is not None:
        vals = torch.cat([lt.double().view(1), cnt.double()]).tolist()
        li, cnt = vals[0], [int(v) for v in vals[1:]]
    S.last_counts = cnt  # touched rows of V / V_e / P this step
    pr("write_counts")
    mem.after_step(lr, cnt)
    pr("mem_after_step")
    if sync:
        torch.cuda.synchronize()
    return li


def expected_time(times):
    """E[step time] under sample_K: 95% uniform on 2..10, 5% uniform on 12..16; times: {K: seconds}, linearly
    interpolated / extrapolated."""
    ks = sorted(times)
    f = lambda k: float(np.interp(k, ks, [times[x] for x in ks])) if ks[0] <= k <= ks[-1] else (
        times[ks[-1]] + (k - ks[-1]) * (times[ks[-1]] - times[ks[-2]]) / (ks[-1] - ks[-2]))
    return 0.95 * np.mean([f(k) for k in range(2, 11)]) + 0.05 * np.mean([f(k) for k in range(12, 17)])
