"""Run fruit-fly-sim/eval_world.py with a per-process CUDA memory cap (0.85 of the GPU) and 2 CPU threads."""
import runpy, sys
import torch
torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
script = r"C:\Eitan\Python\NewBrain\fruit-fly-sim\eval_world.py"
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
