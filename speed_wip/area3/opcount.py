"""Area 3: count dispatched aten ops (~ CUDA kernel launches) per training micro-batch for memory variants.
CPU only, tiny N (op counts do not depend on N). Run with CUDA_VISIBLE_DEVICES="" and python -B."""
import sys, time, json
import torch
from torch.utils._python_dispatch import TorchDispatchMode
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\speed_wip\area3")
from loop_brain import LoopBrain

torch.manual_seed(0)


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0
        self.by = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        k = str(func.overloadpacket)
        self.by[k] = self.by.get(k, 0) + 1
        return func(*args, **(kwargs or {}))


def build(mem=True, H=3, L=4, S=64, R=16, read0=1, nexp=24, emax=96):
    kw = dict(think_w=0.1, think_n=16, think_fb=True, think_use=True, think_target="latent")
    if mem:
        conc = [16, 32, 64, 128][:L]
        kw.update(concepts=conc, concept_weights=None if L == 4 else [1.0] * L, concept_inject_rank=64,
                  concept_eot=0, mem_shared=S, mem_heads=H, mem_topm=8, mem_topk=6, mem_read0=read0,
                  mem_private=R, mem_max_experts=emax)
    m = LoopBrain(256, 384, 96, 6, 64, moe=dict(n_exp=nexp, k=6, hidden=192, shared=384, balance="aux",
                                                  aux_w=0.02, capacity=1.5),
                  spike="graded", k_mean=6, k_min=2, k_max=10, bptt=6, state="", refractory=1, slots=16,
                  waves=True, ws_direct=True, head_group=6, head_groups=1, deep_sup=0.3, halt=True, **kw)
    m.train()
    return m


def step(m, K, B=1, T=64):
    m.K = K
    idx = torch.randint(1, 256, (B, T))
    tgt = torch.randint(1, 256, (B, T))
    out = m(idx, tgt)
    loss = out[1] if isinstance(out, tuple) else out
    for extra in (getattr(m, "loop_extra", None), getattr(m, "loop_aux", None)):
        if extra is not None:
            loss = loss + extra
    loss.backward()
    if getattr(m, "mem", None) is not None:
        m.mem.after_step(1e-3)
    m.zero_grad(set_to_none=True)


def measure(m, K, reps=3):
    step(m, K)  # warm
    c = Count()
    with c:
        step(m, K)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); step(m, K); t.append(time.perf_counter() - t0)
    return c.n, min(t), c.by


if __name__ == "__main__":
    variants = {
        "nomem (latent think head)": dict(mem=False),
        "mem H3 L4 (as run)": dict(),
        "mem H1 L4": dict(H=1),
        "mem H2 L4": dict(H=2),
        "mem H3 L2": dict(L=2),
        "mem H3 L4 no private": dict(R=0),
        "mem H3 L4 no shared": dict(S=0),
        "concepts only (no mem)": dict(S=0, R=0),
    }
    res = {}
    for name, kw in variants.items():
        m = build(**kw)
        for K in (6, 10):
            n, t, by = measure(m, K)
            res[f"{name} K={K}"] = dict(ops=n, cpu_s=round(t, 4))
            print(f"{name:32s} K={K:2d} ops {n:6d}  cpu {t*1e3:7.1f} ms", flush=True)
            if name == "mem H3 L4 (as run)" and K == 6:
                top = sorted(by.items(), key=lambda kv: -kv[1])[:30]
                res["top_ops_mem_K6"] = top
        del m
    json.dump(res, open(r"C:\Eitan\Python\NewBrain\speed_wip\area3\opcount.json", "w"), indent=1)
