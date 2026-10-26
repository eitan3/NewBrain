# Toy: does the st-multiplier router gradient <g, M~> measure the BENEFIT of reading, once a learned
# content gate g = sigmoid(w.[p; r] + b) and W_o (both present in memory.read_shared) can absorb the gain?
import torch, math
torch.manual_seed(0)
N, d = 4096, 16
fact = torch.rand(N) < 0.1                      # 10% positions where memory holds the answer
p = torch.randn(N, 8); p[:, 0] = fact.float() * 2 - 1 + 0.5 * torch.randn(N)   # thought feature, weakly says "fact"
ans = torch.randn(N, d)
r = torch.where(fact[:, None], ans + 0.3 * torch.randn(N, d), torch.randn(N, d))  # the read
y = torch.where(fact[:, None], ans, torch.randn(N, d) * 0.5)
Wo = torch.zeros(d, d, requires_grad=True)        # zero-init like W_o
gw = torch.zeros(8 + d, requires_grad=True); gb = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([Wo, gw, gb], lr=3e-3)
rs = torch.zeros(2, requires_grad=True)           # router: tabular score per region (fact / filler), st mode
ropt = torch.optim.Adam([rs], lr=1e-2); lz = 0.02
def loss_fn(gain, keep=None):
    g = torch.sigmoid(torch.cat([p, r], 1) @ gw + gb)
    M = (g[:, None] * r) @ Wo.T
    if keep is not None: M = M * keep[:, None]
    return ((y - gain[:, None] * M) ** 2).sum(1)   # per-position loss
for it in range(3001):
    s = rs[fact.long()]
    lg = torch.nn.functional.logsigmoid(s)
    m = torch.exp(lg - lg.detach())                # st: forward 1, grad (1 - sigma(s))
    L = loss_fn(m).mean() + lz * (s ** 2).mean()
    opt.zero_grad(); ropt.zero_grad(); L.backward(); opt.step(); ropt.step()
    if it % 500 == 0:
        with torch.no_grad():
            lo = loss_fn(torch.ones(N)); lc = loss_fn(torch.ones(N), keep=torch.zeros(N))
            ben = lc - lo
            # per-position router signal <g, M~> = -(d L_t / d gain) at gain 1
            gain = torch.ones(N, requires_grad=True)
        gain = torch.ones(N, requires_grad=True); lt = loss_fn(gain).sum(); (dg,) = torch.autograd.grad(lt, gain)
        print(f"it {it:5d} true benefit fact {ben[fact].mean():6.3f} filler {ben[~fact].mean():6.3f} | "
              f"st signal -dL/dgain fact {-dg[fact].mean():7.4f} filler {-dg[~fact].mean():7.4f} | "
              f"router s fact {rs[1].item():6.3f} filler {rs[0].item():6.3f}")
