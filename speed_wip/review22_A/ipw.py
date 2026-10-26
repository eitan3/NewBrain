import torch; torch.manual_seed(1)
M, n = 6528, 384                      # complement of A (N=8192, n_c=1664) and n_eps after q64 rounding
x = torch.randn(M) + 0.5
eps = n / M
est = []
for _ in range(20000):
    idx = torch.randperm(M)[:n]
    est.append(x[idx].sum() / eps)
est = torch.stack(est)
plan = (1 - eps) / eps * (x ** 2).sum() * (M - n) / (M - 1)
ht = M * M * (1 - eps) / n * x.var()
print("MC var", est.var().item(), "plan formula", plan.item(), "HT exact", ht.item(), "eps'", eps)
N, nc = 8192, 1664
w2 = nc + n / eps ** 2
print("Kish ESS over open set", N ** 2 / w2, "of", nc + n)
for nc_, ne_ in [(1664, 384), (1664, 448), (4096, 384)]:
    print(nc_, ne_, "omega", (N - nc_) / ne_)
