# Same toy (clean fact feature, content gate can fit it). Compare router modes after 8K steps:
# st (+z), sigmoid (+z), sigmoid (no z), and "distill": router BCE onto the detached mean content gate.
import torch, sys
def run(mode, lz):
    torch.manual_seed(0)
    N, d = 4096, 16
    fact = torch.rand(N) < 0.1
    p = torch.randn(N, 8); p[:, 0] = fact.float() * 2 - 1 + 0.05 * torch.randn(N)
    ans = torch.randn(N, d)
    r = torch.where(fact[:, None], ans + 0.3 * torch.randn(N, d), torch.randn(N, d))
    y = torch.where(fact[:, None], ans, torch.randn(N, d) * 0.5)
    Wo = torch.zeros(d, d, requires_grad=True); gw = torch.zeros(8 + d, requires_grad=True); gb = torch.zeros(1, requires_grad=True)
    rs = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([Wo, gw, gb], lr=3e-3); ropt = torch.optim.Adam([rs], lr=1e-2)
    for it in range(8001):
        s = rs[fact.long()]
        g = torch.sigmoid(torch.cat([p, r], 1) @ gw + gb)
        M = (g[:, None] * r) @ Wo.T
        if mode == "st":
            lg = torch.nn.functional.logsigmoid(s); m = torch.exp(lg - lg.detach())
        elif mode == "sigmoid":
            m = torch.sigmoid(s)
        else:
            m = torch.ones(N)
        L = ((y - m[:, None] * M) ** 2).sum(1).mean() + lz * (s ** 2).mean()
        if mode == "distill":
            L = L + torch.nn.functional.binary_cross_entropy_with_logits(s, g.detach())
        opt.zero_grad(); ropt.zero_grad(); L.backward(); opt.step(); ropt.step()
    print(f"{mode:8s} lz={lz}: router s fact {rs[1].item():6.3f} filler {rs[0].item():6.3f}  "
          f"(content gate fact {g[fact].mean():.3f} filler {g[~fact].mean():.3f})")
for mode, lz in [("st", 0.02), ("st", 0.0), ("sigmoid", 0.02), ("sigmoid", 0.0), ("distill", 0.0)]:
    run(mode, lz)
