"""CPU timing of Memory.balance() with the step-500 counts of wiki_mem_12k (866 / 4273 shared splits / deads,
5818 / 39225 private splits / deads at 192 live experts). CPU = Python + dispatcher overhead only (a lower bound for
the GPU run, which adds ~3 blocking host<->device transfers per key and ~15 kernel launches)."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
import torch
from torch.profiler import profile, ProfilerActivity
import memory as M

torch.manual_seed(0)
mem = M.Memory(384, 64, 4, H=3, S=512, R=512, n_exp=768, n_live=192)
mem._s0 = [1e-4, 1e-4]
st_v = torch.rand(4, 3, 512, 64)
st = {"m": torch.zeros(4, 3, 512, 64), "v": st_v}


def opt_reset(name, sl):  # what reset_mem_key_rows does (BrainOpt state by name)
    st["m"][sl] = 0.0
    st["v"][sl] = st["v"].median()


mem.opt_reset = opt_reset


def set_usage(U, n_busy, n_dead):
    U.fill_(1.0)
    C = U.shape[0]
    for c in range(C):
        U[c, :n_busy] = 10.0
        U[c, n_busy:n_busy + n_dead] = 0.001
    U /= U.sum(-1, keepdim=True)


Ush = mem.U.view(-1, 512)
set_usage(Ush, 866 // 24, 4273 // 24)
set_usage(mem.U_e[:192], 5818 // 192, 39225 // 192)
mem.qres.normal_()
mem.lastq.normal_()
mem.lastq_ok.fill_(True)
sp, de = M.plan_balance(mem.U.view(-1, 512).float())
spe, dee = M.plan_balance(mem.U_e[:192].float())
print(f"plan: shared splits {len(sp)} deads {len(de)}; private splits {len(spe)} deads {len(dee)}")
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU]) as p:
    s = mem.balance()
t = time.perf_counter() - t0
ops = sum(1 for e in p.events() if e.name.startswith("aten::") and e.cpu_parent is None)
items = sum(1 for e in p.events() if e.name in ("aten::item", "aten::_local_scalar_dense"))
print(f"balance(): {t:.2f} s CPU (profiled), top-level aten ops {ops}, item() calls {items}, stats {s}")
# second run unprofiled (usage reset again)
set_usage(Ush, 866 // 24, 4273 // 24)
set_usage(mem.U_e[:192], 5818 // 192, 39225 // 192)
t0 = time.perf_counter()
mem.balance()
print(f"balance() unprofiled: {time.perf_counter() - t0:.2f} s CPU")
