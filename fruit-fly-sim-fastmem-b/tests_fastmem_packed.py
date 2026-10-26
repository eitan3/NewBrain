"""CPU tests: the packed (bucketed) scan == the old padded scan (outputs, end states, grads), skewed / uniform /
worst-case routing; fp32 tight and bf16 loose. python tests_fastmem_packed.py"""
import math
import sys

import torch

import fastmem
from fastmem import scan_read_packed, scan_read_padded


def routing(R, B, T, C, mode, gen):
    """segb / segp / valid [R, C] in the capacity layout (slots in token order per row)."""
    segb = torch.zeros(R, C, dtype=torch.long)
    segp = torch.zeros(R, C, dtype=torch.long)
    valid = torch.zeros(R, C, dtype=torch.bool)
    for r in range(R):
        if mode == "uniform":
            lens = torch.binomial(torch.full((B,), float(T)), torch.full((B,), 2.0 / R), generator=gen).long()
        elif mode == "skewed":
            lens = torch.randint(0, 6, (B,), generator=gen)
            if r % 3 == 0:
                lens[torch.randint(0, B, (1,), generator=gen)] = int(torch.randint(T // 2, T + 1, (1,), generator=gen))
        else:   # worst: row 0 holds one whole sequence, the rest tiny
            lens = torch.randint(0, 3, (B,), generator=gen)
            if r == 0:
                lens[0] = T
        c = 0
        for b in range(B):
            n = int(min(lens[b], C - c))
            segb[r, c:c + n], segp[r, c:c + n], valid[r, c:c + n] = b, torch.arange(n), True
            c += n
    Cs = int(segp.max()) + 1
    return segb, segp, valid, Cs


def inputs(R, B, C, H, dk, dv, cd, gen):
    q = fastmem._l2n(torch.randn(R, C, H, dk, generator=gen)).to(cd)
    u = (fastmem._l2n(torch.randn(R, C, H, dk, generator=gen)) * torch.rand(R, C, H, 1, generator=gen).sqrt()).to(cd)
    w = torch.randn(R, C, H, dv, generator=gen).to(cd)
    lam = -1e-2 * torch.rand(R, C, H, generator=gen)
    M = torch.randn(R, B, H, dk, dv, generator=gen) * 0.1
    return q, u, w, lam, M


def run(fn, args, segb, segp, valid, B, Cs, prefix, merge, gO):
    q, u, w, lam, M = [a.clone().requires_grad_(i < 4 and a.is_floating_point()) for i, a in enumerate(args)]
    M = M.detach()
    O, dM = fn(q, u, w, lam, M, segb, segp, valid, B, Cs, prefix, merge)
    (O * gO).sum().backward()
    return O.detach(), dM, [t.grad if t.grad is not None else torch.zeros_like(t) for t in (q, u, w, lam)]


def compare(mode, cd, R=12, B=4, T=96, H=2, dk=32, dv=32, seed=0, budget=1 << 24):
    gen = torch.Generator().manual_seed(seed)
    C = math.ceil(1.5 * B * T * 2 / R) if mode != "worst" else T + 3 * B
    segb, segp, valid, Cs = routing(R, B, T, C, mode, gen)
    args = inputs(R, B, C, H, dk, dv, cd, gen)
    gO = torch.randn(R, C, H, dv, generator=gen) * valid[..., None, None]
    tol = 2e-4 if cd == torch.float32 else 3e-2
    worst = 0.0
    for prefix in (True, False):
        a = run(scan_read_padded, args, segb, segp, valid, B, Cs, prefix, prefix, gO)
        old = fastmem.SCAN_BUDGET
        fastmem.SCAN_BUDGET = budget
        try:
            b = run(scan_read_packed, args, segb, segp, valid, B, Cs, prefix, prefix, gO)
        finally:
            fastmem.SCAN_BUDGET = old
        pairs = [("O", a[0], b[0])] + ([("M_end", a[1], b[1])] if prefix else []) + \
                [(f"grad_{n}", x, y) for n, x, y in zip(("q", "u", "w", "lam"), a[2], b[2])]
        for n, x, y in pairs:
            x, y = x.float(), y.float()
            err = float((x - y).abs().max() / (x.abs().max() + 1e-6))
            worst = max(worst, err)
            assert err < tol, f"{mode} {cd} prefix {prefix} {n}: rel err {err:.2e} >= {tol}"
    print(f"ok  {mode:8s} {str(cd):15s} R {R} B {B} Cs {Cs} budget {budget}: worst rel err {worst:.1e}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    for mode in ("uniform", "skewed", "worst"):
        compare(mode, torch.float32)
        compare(mode, torch.bfloat16)
    compare("skewed", torch.float32, budget=2 * 16 * 16)     # tiny budget: many chunks, nested checkpoints
    compare("skewed", torch.float32, budget=1 << 24, seed=3)
    # under an outer checkpoint (as --fastmem-ckpt 1)
    from torch.utils.checkpoint import checkpoint
    gen = torch.Generator().manual_seed(7)
    segb, segp, valid, Cs = routing(10, 4, 64, 40, "skewed", gen)
    args = inputs(10, 4, 40, 2, 16, 16, torch.float32, gen)
    outs = []
    for fn in (scan_read_padded, scan_read_packed):
        q, u, w, lam = [a.clone().requires_grad_() for a in args[:4]]
        O = checkpoint(lambda *t: fn(*t, args[4], segb, segp, valid, 4, Cs, True, True)[0], q, u, w, lam,
                       use_reentrant=False)
        O.square().sum().backward()
        outs.append([O.detach(), q.grad, u.grad, w.grad, lam.grad])
    for x, y in zip(*outs):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-5), float((x - y).abs().max())
    print("ok  nested in an outer checkpoint (fastmem-ckpt 1)")
    print("ALL PACKED-SCAN TESTS PASSED")
