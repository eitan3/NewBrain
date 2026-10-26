"""Speed / accumulation study helpers: reuse the VRAM study's loader and step (../vram/vram_common.py), with the
0.78 memory-fraction guard. No model code is edited."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vram"))
import vram_common as vc  # noqa: E402  (sets 2 CPU threads, puts fruit-fly-sim on sys.path)
import torch  # noqa: E402

SCRATCH = Path(os.environ.get("SPEED_SCRATCH", r"C:\Users\eitan\AppData\Local\Temp\claude\C--Eitan-Python-NewBrain"
                                               r"\de301b72-951d-49a4-9895-252fb20d74d6\scratchpad"))


def guard():
    torch.cuda.set_per_process_memory_fraction(0.78)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(2)
