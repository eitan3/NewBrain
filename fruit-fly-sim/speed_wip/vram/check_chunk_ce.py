"""Sanity check: ChunkedReadout == the reference bf16-logits log-softmax CE (values, dh, dW, halting parts)."""
import math
import torch
import torch.nn.functional as F
import vram_common  # noqa (sys.path)
import variants
torch.cuda.set_per_process_memory_fraction(0.9)
torch.manual_seed(0)
N, d, V = 5000, 384, 16384
h = torch.randn(N, d, device="cuda", requires_grad=True)
W = (torch.randn(V, d, device="cuda") * 0.05).requires_grad_()
t = torch.randint(0, V, (N,), device="cuda")
prev = (torch.randn(N, V, device="cuda")).to(torch.bfloat16)
g = torch.rand(N, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    lg = F.linear(h, W)
lp = F.log_softmax(lg.float(), -1)
ce = -lp.gather(1, t[:, None]).squeeze(1)
(ce * g).sum().backward()
dh0, dW0 = h.grad.clone(), W.grad.clone()
h.grad = W.grad = None
p = lp.exp()
ent0 = (-(p * lp).sum(-1) / math.log(V)).detach()
kl0 = torch.log1p((p * (lp - F.log_softmax(prev.float(), -1))).sum(-1).clamp(min=0)).detach()
ce1, lg1, ent1, kl1 = variants.ChunkedReadout.apply(h, W, t, prev, True, True)
(ce1 * g).sum().backward()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print("ce max abs diff", (ce1 - ce).abs().max().item(), "logits equal", torch.equal(lg1, lg))
print("ent diff", (ent1 - ent0).abs().max().item(), "kl diff", (kl1 - kl0).abs().max().item())
print("dh rel", rel(h.grad, dh0), "dW rel", rel(W.grad, dW0))
