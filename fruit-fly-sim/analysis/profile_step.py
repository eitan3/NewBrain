"""Profile one full-size training step (GPU):  python analysis/profile_step.py <run_dir> [code_dir]

Builds the brain from the run's final checkpoint (its final size and settings) and times trainer-like steps
(2 micro-batches of 16 x 512, BrainOpt step, balancer), with K fixed to 6 (training mean ~6.4). Memory is the
peak allocated / reserved during the step. Variants switch parts off to attribute cost: per-pass read-outs
(deep supervision + stop head), the thinking head, the router checkpoint. Also the top CUDA ops by time.
code_dir: which copy of the code to import (e.g. the pre-optimisation originals), default this folder.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RUN = Path(sys.argv[1])
CODE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))
from brain_opt import BrainOpt  # noqa: E402
from loop_brain import LoopBrain  # noqa: E402

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "ts4k" / "train.bin"
dev = "cuda"
ck = torch.load(RUN / "final.pt", map_location="cpu", weights_only=False)
toks = np.memmap(DATA, dtype=np.uint16, mode="r")
rng = np.random.RandomState(0)


def batch(B=16, T=512):
    ix = rng.randint(0, len(toks) - T - 1, B)
    x = torch.from_numpy(np.stack([toks[i:i + T] for i in ix]).astype(np.int64)).to(dev)
    y = torch.from_numpy(np.stack([toks[i + 1:i + T + 1] for i in ix]).astype(np.int64)).to(dev)
    return x, y


def build(overrides):
    cfg = dict(ck["cfg"], **overrides)
    m = LoopBrain(**cfg).to(dev)
    own = m.state_dict()  # variants with parts switched off: load only the weights whose shapes still fit
    m.load_state_dict({k: v for k, v in ck["state"].items() if k in own and own[k].shape == v.shape}, strict=False)
    m.set_lam(1.0)
    for mod in m.modules():
        if hasattr(mod, "noise"):
            mod.noise = 0.3
    m.train()
    return m


def step(m, opt, K=6):
    m.K = K
    m.aux_k_fixed = m.sample_aux_k(K)
    m.group_offset = 0
    opt.zero_grad(set_to_none=True)
    for _ in range(2):
        x, y = batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = m(x, y)
            total = loss
            aux = m.aux_loss()
            if aux is not None:
                total = total + aux
            if m.aux_x is not None:
                total = total + 0.2 * F.cross_entropy(m.heads["lm"](m.aux_x).float().flatten(0, 1), y.flatten())
        (total / 2).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    m.blocks[0].mlp.apply_balance()


def measure(name, overrides, n=4):
    torch.cuda.empty_cache()
    m = build(overrides)
    if overrides.pop("_ckpt_route", False):
        m.blocks[0].mlp.ckpt_route = True
    opt = BrainOpt(m, 2e-3, 0.1)
    step(m, opt)  # warm-up (allocator, optimizer state)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(n):
        step(m, opt)
    torch.cuda.synchronize()
    dt = (time.time() - t) / n
    r = dict(step_s=round(dt, 3), tok_s=round(2 * 16 * 512 / dt), peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
             peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
    print(f"{name:34s} {r}", flush=True)
    del m, opt
    return r


res = {}
res["full (as trained)"] = measure("full (as trained)", {})
res["no per-pass read-outs"] = measure("no per-pass read-outs", dict(deep_sup=0.0, halt=False))
res["no thinking head"] = measure("no thinking head", dict(think_w=0.0, think_fb=False, think_use=False))
res["router checkpoint on"] = measure("router checkpoint on", dict(_ckpt_route=True))
res["base (no new parts)"] = measure("base (no new parts)", dict(deep_sup=0.0, halt=False, think_w=0.0, think_fb=False,
                                                                  think_use=False))
import inspect  # noqa: E402
if "readout_frac" in inspect.signature(LoopBrain.__init__).parameters:  # the optimised code
    res["readout 0.25"] = measure("readout 0.25", dict(readout_frac=0.25))
    res["readout 0.25 + router ckpt"] = measure("readout 0.25 + router ckpt", dict(readout_frac=0.25, _ckpt_route=True))

# top CUDA ops of one full step
m = build({})
opt = BrainOpt(m, 2e-3, 0.1)
step(m, opt)
from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    step(m, opt)
    torch.cuda.synchronize()
table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
print(table)
(RUN / "profile_step.json").write_text(json.dumps(res, indent=1))
(RUN / "profile_ops.txt").write_text(table)
