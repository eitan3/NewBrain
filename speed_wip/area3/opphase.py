"""Area 3: aten op counts per memory phase (forward incl. checkpoint recompute) for one training micro-batch."""
import sys, json, functools
import torch
from torch.utils._python_dispatch import TorchDispatchMode
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\speed_wip\area3")
import memory
import loop_brain
from opcount import build, step

STACK = ["other"]


def tag(obj, name, label):
    f = getattr(obj, name)

    @functools.wraps(f)
    def g(*a, **k):
        STACK.append(label)
        try:
            return f(*a, **k)
        finally:
            STACK.pop()
    setattr(obj, name, g)


tag(memory.ConceptHier, "forward", "concept_heads")
tag(memory.ConceptHier, "write_back", "concept_inject")
tag(memory.Memory, "prep", "prep")
tag(memory.Memory, "select", "select(EMA,no-grad)")
tag(memory.Memory, "online", "online_w")
tag(memory.Memory, "margin_loss", "margin")
tag(memory.Memory, "read_shared", "read_shared")
tag(memory.Memory, "private", "private")
tag(memory.Memory, "after_step", "after_step(MemOpt)")
tag(memory, "concept_targets", "concept_loss")
tag(memory, "concept_loss", "concept_loss")
tag(loop_brain.LoopBrain, "_slot_write", "slot_write")


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.by = {}
        self.bwd = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        lab = STACK[-1]
        if self.bwd and lab == "other":
            lab = "backward(unlabelled)"
        elif self.bwd:
            lab = lab + " [recompute]"
        self.by[lab] = self.by.get(lab, 0) + 1
        return func(*args, **(kwargs or {}))


def run(m, K):
    step(m, K)
    c = Count()
    m.K = K
    idx = torch.randint(1, 256, (1, 64)); tgt = torch.randint(1, 256, (1, 64))
    with c:
        out = m(idx, tgt)
        loss = out[1] + m.loop_extra + (m.loop_aux if m.loop_aux is not None else 0)
        c.bwd = True
        loss.backward()
        c.bwd = False
        if m.mem is not None:
            m.mem.after_step(1e-3)
    m.zero_grad(set_to_none=True)
    return dict(sorted(c.by.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    out = {}
    for name, kw in (("mem", {}), ("nomem", dict(mem=False))):
        m = build(**kw)
        for K in (6, 10):
            r = run(m, K)
            out[f"{name} K={K}"] = r
            print(name, K, sum(r.values()), json.dumps(r))
    json.dump(out, open(r"C:\Eitan\Python\NewBrain\speed_wip\area3\opphase.json", "w"), indent=1)
