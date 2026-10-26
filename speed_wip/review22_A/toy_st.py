# Toy: is the st gain-gradient at gain 1 a "benefit" signal once the read scale (W_o / content gate) is trained?
import torch; torch.manual_seed(0)
N, d = 4000, 16
b = torch.rand(N) * 2                     # true per-position usefulness of the read
u = torch.nn.functional.normalize(torch.randn(N, d), dim=-1)
r = u + 0.3 * torch.randn(N, d)           # read content (noisy version of the needed direction)
y = b[:, None] * u + 0.5 * torch.randn(N, d)
def fit(gate_feat):
    W = torch.zeros(d, d, requires_grad=True); a = torch.zeros(gate_feat.shape[1], requires_grad=True); c0 = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([W, a, c0], 0.05)
    for _ in range(1500):
        g = torch.sigmoid(gate_feat @ a + c0)[:, None] if gate_feat.shape[1] else 1.0
        M = g * (r @ W.T); loss = 0.5 * ((y - M) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g = torch.sigmoid(gate_feat @ a + c0)[:, None] if gate_feat.shape[1] else 1.0
        M = g * (r @ W.T)
        gl = -(y - M)                                  # dl/dM per position
        st = (gl * M).sum(-1)                          # dl/d(gain) at gain 1  (st router gradient, before (1-sigma))
        ben = 0.5 * (y ** 2).sum(-1) - 0.5 * ((y - M) ** 2).sum(-1)   # exact L(closed)-L(open)
        first = -st                                    # Lemma 1 first-order benefit estimate
    return first, ben
def spear(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
for name, feat in [("no content gate", torch.zeros(N, 0)), ("content gate on noisy b", (b + 0.5 * torch.randn(N))[:, None]),
                   ("content gate on exact b", b[:, None])]:
    f, ben = fit(feat)
    top = ben.topk(N // 5).indices; topf = f.topk(N // 5).indices
    ov = len(set(top.tolist()) & set(topf.tolist())) / (N // 5)
    print(f"{name:28s} mean first-order={f.mean():+.3f} mean true benefit={ben.mean():+.3f} "
          f"spearman(first,true)={spear(f, ben):+.3f} spearman(first,b)={spear(f, b):+.3f} top20% overlap={ov:.2f}")
