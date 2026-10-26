"""Task 2e: micro-benchmarks of fused / native ops that could replace small elementwise chains (fwd + bwd, CUDA
events, shapes of one pass at micro-batch 8: N = 4096 tokens, d = 384; mixer 11 of 36 heads computed per pass).
Reports time per call and the max deviation from the model's own code (same math?).

    python bench_fused.py
"""
import json
import math

import torch
import torch.nn.functional as F

import sp_common as sc
import spiking_lm
from spiking_lm import RMSNorm, Spike, ssd_scan

sc.guard()
dev = "cuda"
torch.manual_seed(0)
RES = {}


def bench(fn, iters=100, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters  # ms


def fwd_bwd(f, *inputs):
    def run():
        for t in inputs:
            t.grad = None
        y = f(*inputs)
        y.backward(gy if y.shape == gy.shape else torch.ones_like(y))
    return run


# ---------------------------------------------------------------- RMSNorm (5 + per pass, the fp32 chain)
N, d = 8 * 512, 384
for dt in (torch.bfloat16, torch.float32):
    x = torch.randn(N, d, device=dev, dtype=dt, requires_grad=True)
    gy = torch.randn(N, d, device=dev, dtype=dt)
    norm = RMSNorm(d).to(dev)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    w = norm.weight

    def frms(x):
        return F.rms_norm(x.float(), (d,), w, 1e-6).to(x.dtype)

    def frms_native(x):  # the input dtype directly (bf16 math inside the fused kernel: NOT the same math for bf16)
        return F.rms_norm(x, (d,), w.to(x.dtype), 1e-6)
    y0, y1, y2 = norm(x), frms(x), frms_native(x)
    x.grad = None
    norm(x).backward(gy)
    g0 = x.grad.clone()
    x.grad = None
    frms(x).backward(gy)
    g1 = x.grad.clone()
    r = dict(eager_ms=bench(fwd_bwd(norm, x)), f_rms_norm_fp32_ms=bench(fwd_bwd(frms, x)),
             f_rms_norm_native_ms=bench(fwd_bwd(frms_native, x)),
             max_dy_fp32=float((y0.float() - y1.float()).abs().max()), max_dx_fp32=float((g0.float() - g1.float()).abs().max()),
             max_dy_native=float((y0.float() - y2.float()).abs().max()))
    RES[f"rmsnorm_{str(dt)[6:]}"] = r
    print("rmsnorm", dt, r, flush=True)


# ---------------------------------------------------------------- graded spike (STE / LSQ), signed mixer spike
class SpikeFn(torch.autograd.Function):
    """The same maths as Spike.forward (graded, lam = 1, no homeostasis): s = round(clamp(u / a, lo, hi)) * a with the
    STE / LSQ gradients of the autograd graph: du = g * [lo <= u/a <= hi]; da = sum(g * (q - (u/a) * mask))."""

    @staticmethod
    def forward(ctx, u, a, lo, hi):
        v = u / a
        q = torch.clamp(v, lo, hi).round_()
        ctx.save_for_backward(u, a)
        ctx.lo, ctx.hi = lo, hi
        return (q * a).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        u, a = ctx.saved_tensors
        v = u / a
        mask = (v >= ctx.lo) & (v <= ctx.hi)
        gf = g.float()
        du = torch.where(mask, gf, 0.0).to(u.dtype)
        q = torch.clamp(v, ctx.lo, ctx.hi).round_()
        da = (gf * torch.where(mask, q - v, q)).sum(0)
        return du, da, None, None


C = 11 * 64
sp = Spike(C, "graded", signed=True).to(dev)
sp.calibrated.fill_(True)
sp.lam = 1.0
with torch.no_grad():
    sp.alpha.uniform_(0.3, 1.0)
u = torch.randn(N, C, device=dev, dtype=torch.bfloat16, requires_grad=True)
gy = torch.randn(N, C, device=dev, dtype=torch.bfloat16)
ch = torch.arange(C, device=dev)


def sp_eager(u):
    return sp(u)


def sp_fn(u):
    a = sp.alpha.abs() + 1e-4
    return SpikeFn.apply(u, a, -4.0, 4.0)


def sp_fq(u):  # fused fake-quant with a learnable scale (LSQ): one kernel each way; NOT the same edge gradient
    a = sp.alpha.abs() + 1e-4
    return torch._fake_quantize_learnable_per_channel_affine(u.float(), a, torch.zeros_like(a), 1, -4, 4, 1.0).to(u.dtype)


outs, grads = {}, {}
for n, f in (("eager", sp_eager), ("fn", sp_fn), ("fakequant", sp_fq)):
    u.grad = None
    sp.alpha.grad = None
    y = f(u)
    y.backward(gy)
    outs[n], grads[n] = y.detach().float(), (u.grad.float().clone(), sp.alpha.grad.float().clone())
r = dict(eager_ms=bench(fwd_bwd(sp_eager, u)), autograd_fn_ms=bench(fwd_bwd(sp_fn, u)),
         fakequant_ms=bench(fwd_bwd(sp_fq, u)))
for n in ("fn", "fakequant"):
    r[f"{n}_max_dy"] = float((outs[n] - outs["eager"]).abs().max())
    r[f"{n}_max_du"] = float((grads[n][0] - grads["eager"][0]).abs().max())
    r[f"{n}_rel_dalpha"] = float((grads[n][1] - grads["eager"][1]).norm() / grads["eager"][1].norm())
RES["spike_mixer"] = r
print("spike", r, flush=True)

# ---------------------------------------------------------------- SSD chunked scan (one pass of the mixer)
B, H, T, Nn, P = 8, 11, 512, 64, 64
q = F.normalize(torch.randn(B, H, T, Nn, device=dev), dim=-1).requires_grad_()
k = F.normalize(torch.randn(B, H, T, Nn, device=dev), dim=-1).requires_grad_()
v = torch.randn(B, H, T, P, device=dev, requires_grad=True)
la = (-torch.rand(B, H, T, device=dev) * 0.1).requires_grad_()
gy = torch.randn(B, H, T, P, device=dev)
_mask = {}


def segsum_cached(x):
    T_ = x.size(-1)
    if T_ not in _mask:
        _mask[T_] = torch.ones(T_, T_, dtype=torch.bool, device=x.device).tril().logical_not()
    cs = x.cumsum(-1)
    return (cs[..., :, None] - cs[..., None, :]).masked_fill(_mask[T_], float("-inf"))


def scan_eager():
    return ssd_scan(q, k, v, la, 64)[0]


def scan_cached():
    old = spiking_lm.segsum
    spiking_lm.segsum = segsum_cached
    try:
        return ssd_scan(q, k, v, la, 64)[0]
    finally:
        spiking_lm.segsum = old


def run_scan(f):
    def r_():
        for t in (q, k, v, la):
            t.grad = None
        f().backward(gy)
    return r_


y0, y1 = scan_eager(), scan_cached()
r = dict(eager_ms=bench(run_scan(scan_eager), 30), cached_mask_ms=bench(run_scan(scan_cached), 30),
         max_dy=float((y0 - y1).abs().max()))
torch.cuda.synchronize()
a_, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
a_.record()
for _ in range(30):
    with torch.no_grad():
        scan_eager()
b_.record()
torch.cuda.synchronize()
r["fwd_only_ms"] = a_.elapsed_time(b_) / 30
RES["ssd_scan_pass"] = r
print("ssd_scan", r, flush=True)

# ---------------------------------------------------------------- routed experts: padded bmm vs grouped mm
E, k_, h = 768, 6, 192
Ntok = 4096
xf = torch.randn(Ntok, d, device=dev, dtype=torch.bfloat16)
w_up = (torch.randn(E, d, h, device=dev) * 0.02).requires_grad_()
w_dn = (torch.randn(E, h, d, device=dev) * 0.02).requires_grad_()
idx = torch.randint(0, E, (Ntok * k_,), device=dev)
order = idx.argsort(stable=True)
e_sorted, tok = idx[order], order // k_
counts = torch.bincount(idx, minlength=E)
Ccap = int(math.ceil(1.5 * Ntok * k_ / E))
pos = torch.arange(Ntok * k_, device=dev) - (counts.cumsum(0) - counts)[e_sorted]
keep = pos < Ccap
e_k, p_k, t_k = e_sorted[keep], pos[keep], tok[keep]
n_keep = int(keep.sum())
gyo = torch.randn(n_keep, d, device=dev, dtype=torch.bfloat16)


def experts_bmm():
    buf = xf.new_zeros(E, Ccap, d)
    buf[e_k, p_k] = xf[t_k]
    hh = F.relu(torch.bmm(buf, w_up.to(buf.dtype)))
    out = torch.bmm(hh, w_dn.to(hh.dtype))
    return out[e_k, p_k]


kc = torch.minimum(counts, torch.full_like(counts, Ccap))
offs = kc.cumsum(0).to(torch.int32)


def experts_grouped():
    xs = xf[t_k]
    hh = F.relu(torch._grouped_mm(xs, w_up.to(torch.bfloat16), offs=offs))
    return torch._grouped_mm(hh, w_dn.to(torch.bfloat16), offs=offs)


r = {}
try:
    y0, y1 = experts_bmm(), experts_grouped()
    r["max_dy"] = float((y0.float() - y1.float()).abs().max())
    r["grouped_ms"] = bench(fwd_bwd(lambda: experts_grouped(), ), 30) if False else None
except Exception as ex:  # noqa: BLE001
    r["grouped_error"] = repr(ex)[:300]


def eb():
    w_up.grad = w_dn.grad = None
    experts_bmm().backward(gyo)


def eg():
    w_up.grad = w_dn.grad = None
    experts_grouped().backward(gyo)


r["bmm_padded_ms"] = bench(eb, 30)
if "grouped_error" not in r:
    try:
        r["grouped_ms"] = bench(eg, 30)
    except Exception as ex:  # noqa: BLE001
        r["grouped_bwd_error"] = repr(ex)[:300]
r["capacity"] = Ccap
r["kept_tokens"] = n_keep
RES["experts_pass"] = r
print("experts", r, flush=True)

(sc.HERE / "bench_fused.json").write_text(json.dumps(RES, indent=1))
