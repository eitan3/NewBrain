"""The full-size starting parameters (the same load as accum_run.py) for compare_runs.py's update-relative distances."""
import torch

import sp_common as sc
from sp_common import vc

sc.guard()
torch.manual_seed(0)
S = vc.load(full=True)
torch.save({n: p.detach().float().cpu().clone() for n, p in S.model.named_parameters()},
           sc.SCRATCH / "t1_init_params.pt")
print("saved")
