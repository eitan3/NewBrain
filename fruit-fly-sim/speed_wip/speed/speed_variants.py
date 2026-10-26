"""Task 2 speed what-ifs, applied in-process only (the model files are not edited). Source-level patches take the
method's own source (inspect.getsource), replace single lines and exec it in the module's namespace.

  topk_nosync   memory.TOPK_SYNC_FREE = True: stable_topk's tie handling by a second top-k instead of a host check
                (the module's own switch; identical indices and values by its docstring). Same math.
  route_nosync0 bincount -> zeros.index_add_ (no hidden max() sync), valid / has masks without a Python-scalar
                index_put (valid from arange(C) < min(count, C)). Same math, keeps the counts.tolist().
  route_static  + fixed capacity C = ceil(cap N k / E) (no counts.tolist(): every shape static); dropped slots go to
                an extra junk slot C with gate 0 / valid False. Same keep set as long as every expert gets >= 1 token
                in the pass (otherwise the original C is larger -> fewer drops): NEAR-same math.
  rms_fused     RMSNorm.forward -> F.rms_norm on the fp32 copy (one fused kernel), cast back. Same math up to rounding.
  spike_fn      Spike (graded, floor homeostasis) -> one autograd.Function with the same STE / LSQ gradients.
"""
import inspect
import math
import textwrap

import torch
import torch.nn.functional as F

import memory
import spiking_lm
from memory import Memory
from spiking_lm import RMSNorm, RoutedMLP, Spike


def _patch(cls, name, mod, repl):
    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    for old, new in repl:
        assert old in src, f"{name}: {old!r} not found"
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, f"<patched {cls.__name__}.{name}>", "exec"), mod.__dict__, ns)
    setattr(cls, name, ns[name])


ROUTE0 = [
    ("counts = torch.bincount(flat, minlength=E)",
     "counts = torch.zeros(E, dtype=torch.long, device=flat.device).index_add_(0, flat, torch.ones_like(flat))"),
    ("valid = torch.zeros(E, C, dtype=torch.bool, device=x.device)\n    valid[e_k, p_k] = True",
     "valid = torch.arange(C, device=x.device)[None, :] < counts.clamp(max=C)[:, None]"),
]
ROUTE_STATIC = [
    ("counts = torch.bincount(flat, minlength=E)",
     "counts = torch.zeros(E, dtype=torch.long, device=flat.device).index_add_(0, flat, torch.ones_like(flat))"),
    ("cl = counts.tolist()", "cl = None"),
    ("used = max(1, sum(c > 0 for c in cl))", "used = E"),
    ("C = max(1, min(max(cl), math.ceil(self.cap * N * k / used)))", "C = max(1, math.ceil(self.cap * N * k / used))"),
    ("C = max(1, max(cl))", "raise NotImplementedError('route_static needs a capacity')"),
    ("n_keep = sum(min(c, C) for c in cl)\n    kidx = None if n_keep == N * k else (~keep).to(torch.uint8).argsort(stable=True)[:n_keep]",
     "kidx = None"),
    ("e_k, p_k, t_k = kept(e_sorted), kept(pos), kept(tok)", "e_k, p_k, t_k = e_sorted, torch.where(keep, pos, C), tok"),
    ("buf = xf.new_zeros(E, C, D)", "buf = xf.new_zeros(E, C + 1, D)"),
    ("valid = torch.zeros(E, C, dtype=torch.bool, device=x.device)\n    valid[e_k, p_k] = True",
     "valid = torch.arange(C + 1, device=x.device)[None, :] < counts.clamp(max=C)[:, None]"),
    ("y.index_add_(0, t_k, out[e_k, p_k].to(acc) * kept(gate.flatten()[order])[:, None])",
     "gk = gate.flatten()[order] * keep\n    y.index_add_(0, t_k, out[e_k, p_k].to(acc) * gk[:, None])"),
    ("y = y + pmem(e_k, p_k, t_k, kept(gate.flatten()[order]), kept(order), N, k, E, C).to(acc)",
     "y = y + pmem(e_k, p_k, t_k, gk, order, N, k, E, C + 1).to(acc)"),
]
PRIV0 = [("has[e_k] = True", "has.index_fill_(0, e_k, True)")]


def apply_topk_nosync(S):
    memory.TOPK_SYNC_FREE = True


def apply_route_nosync0(S):
    _patch(RoutedMLP, "forward", spiking_lm, ROUTE0)
    _patch(Memory, "private", memory, PRIV0)


def apply_route_static(S):
    _patch(RoutedMLP, "forward", spiking_lm, ROUTE_STATIC)
    _patch(Memory, "private", memory, PRIV0)


def _rms(self, x):
    return F.rms_norm(x.float(), (self.weight.numel(),), self.weight, self.eps).to(x.dtype)


def apply_rms_fused(S):
    RMSNorm.forward = _rms


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a, lo, hi):
        v = u / a
        q = torch.clamp(v, lo, hi).round_()
        ctx.save_for_backward(u, a)
        ctx.lo, ctx.hi = lo, hi
        return q, (q * a).to(u.dtype)

    @staticmethod
    def backward(ctx, gq, g):
        u, a = ctx.saved_tensors
        v = u / a
        mask = (v >= ctx.lo) & (v <= ctx.hi)
        gf = g.float()
        q = torch.clamp(v, ctx.lo, ctx.hi).round_()
        du = torch.where(mask, gf, 0.0).to(u.dtype)
        da = (gf * torch.where(mask, q - v, q)).flatten(0, -2).sum(0)
        return du, da, None, None


_spike_orig = Spike.forward


def _spike(self, x, ch=None):
    if type(self) is not Spike or self.mode != "graded" or self.lam < 1 or self.homeo_mode != "floor" or self.track:
        return _spike_orig(self, x, ch)
    lo, hi = spiking_lm.LEVELS[self.mode]
    if self.signed and lo == 0:
        lo = -hi
    u = x if (self.signed or lo < 0) else F.relu(x)
    alpha = (self.alpha if ch is None else self.alpha[ch]).abs() + 1e-4
    q, s = _SpikeFn.apply(u, alpha, float(lo), float(hi))
    if self.homeo and self.training:   # (as Spike.forward)
        with torch.no_grad():
            r = (q != 0).to(self.ema_rate.dtype).flatten(0, -2).mean(0)
            if ch is None:
                self.ema_rate.lerp_(r, 0.01)
                a = self.alpha.data
                a.copy_(torch.where(self.ema_rate < self.floor, a * (1 - self.homeo), a))
            else:
                self.ema_rate[ch] = self.ema_rate[ch].lerp(r, 0.01)
                a = self.alpha.data[ch]
                self.alpha.data[ch] = torch.where(self.ema_rate[ch] < self.floor, a * (1 - self.homeo), a)
    return s


def apply_spike_fn(S):
    Spike.forward = _spike


def apply(S, names):
    for n in names:
        if not n:
            continue
        globals()[f"apply_{n}"](S)
        print("speed variant", n, flush=True)


def apply_topk_nosync2(S):
    """topk_nosync + the remaining sync of its path (rev.new_tensor(-1): a host -> device scalar copy) -> a Python
    scalar in torch.where. Same math."""
    memory.TOPK_SYNC_FREE = True
    src = textwrap.dedent(inspect.getsource(memory.stable_topk))
    old = "torch.where(flat == thr, rev, rev.new_tensor(-1))"
    assert old in src
    src = src.replace(old, "torch.where(flat == thr, rev, -1)")
    ns = {}
    exec(compile(src, "<patched stable_topk>", "exec"), memory.__dict__, ns)
    memory.stable_topk = ns["stable_topk"]
