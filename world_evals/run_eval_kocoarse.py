"""eval_world.py with only the coarse-pool knockout added (mem_on + ko_coarse), per-process CUDA cap 0.85, 2 threads.
The other knockouts (shared / private / level1-4) are in world_mem_life_12k_step7000.*."""
import sys
sys.argv = [sys.argv[0]] + sys.argv[1:]
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
import torch
torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)
import eval_world as ew

_orig = ew.knockouts


def knockouts(model, gate_modes=True):
    out = _orig(model, gate_modes)[:1]
    if getattr(model.mem, "Sc", 0) or len(getattr(model.mem, "pools", [])) > 1:
        out.append(("ko_coarse", "ko_coarse", True))
    return out


ew.knockouts = knockouts
ew.main()
