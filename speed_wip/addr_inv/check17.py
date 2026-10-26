import os, sys, math, re
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch, torch.nn.functional as F
torch.set_num_threads(4)
src = open("sim2.py").read()
fn = src[src.index("def sinkhorn_idx"):src.index("INPUTS = [")]
c6 = torch.load("cap_6000.pt", map_location="cpu", weights_only=False); st = c6["state"]
S, m, STEPS, BS, LR = 512, 8, 2000, 512, 0.003
exec(fn)
X = c6["X"]; B, T = X.shape; seq = torch.arange(B).repeat_interleave(T); FIT = (seq < 16).nonzero().squeeze(1)
a = c6["sels"][[s["k"] for s in c6["sels"]].index(2)]["a"][0].float()[FIT]
q = F.normalize(((a - st["norm_m"][0]) / torch.sqrt(st["norm_v"][0] + 1e-5)) @ st["wq_ema"][0, 0].T, dim=-1)
q1 = F.normalize(q[:, :64], dim=-1)
r1 = F.normalize(torch.randn(S, 64, generator=torch.Generator().manual_seed(0)), dim=-1)
def dup(K):
    c = K @ K.T; c.fill_diagonal_(-1); mx = c.max(1).values
    return int((mx > 0.95).sum()), int((mx > 0.99).sum()), round(float(mx.median()), 3)
tk = F.normalize(st["k1_ema"][0, 0], dim=-1)
print("trained", dup(tk))
for meth in ["attract", "fP", "B_ema_kmeans", "C_restart", "D_sinkhorn", "F_gumbel"]:
    K = dynamics(meth, q1, r1, 1)
    print(meth, "near-dup >0.95 / >0.99 / median max-cos:", dup(K), flush=True)
