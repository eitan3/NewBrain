"""Monkeypatched "step-level / batch-size-invariant" versions of the per-micro-batch side effects (Task 1). The model
code is NOT edited; install(S, names) patches classes in this process only. The harness must call
begin_step(S, step, xs, accum) before the first micro-batch, begin_micro(S, i) before each one and end_step(S) right
after opt.step() (before apply_balance / mem.after_step).

  noise    route noise: one (N_eff, E) normal draw per (step, pass) from a generator seeded by (step, pass); micro-batch
           i takes rows [i N, (i+1) N). Every global token gets the same noise whatever the split.
  homeo    homeostasis (Spike / ExpertSpike ema_rate + alpha floor): no in-forward edits; firing counts are summed per
           pass over the step's micro-batches; after the optimizer step each pass's EMA / floor update is applied REF
           times (REF = 2 = the number of micro-batch calls per pass of the 16x2 reference) with the step-level rate.
  memstats memory EMA norm statistics: summed over ALL micro-batches of the step (begin_forward no longer resets them;
           they are reset at the start of the step) instead of the last micro-batch only.
  qres     memory query reservoir: slot j holds the query of GLOBAL position linspace(0, N_eff - 1, nres)[j] (each
           micro-batch writes only the slots whose position it holds) instead of positions of the current micro-batch.
  margin   --mem-margin-passes draw: one draw per step (the margin generator is rewound for micro-batches 2..n).
  concept  concept loss: NLL normalised by the step's total valid count (known from the tokens before the step) and
           the variance hinge over fixed groups of 8 sequences (mean over groups) instead of over the micro-batch.
"""
import math

import torch
import torch.nn.functional as F

import loop_brain
import memory
from loop_brain import LoopBrain
from memory import Memory, masked_std
from spiking_lm import LEVELS, ExpertSpike, RoutedMLP, Spike

CTX = dict(step=0, mb=0, accum=1, names=set(), nv_tot=None, lvl=0, gm_state=None, ref=2, hgroup=8)
ALL = ("noise", "homeo", "memstats", "qres", "margin", "concept", "auxgroup")

_orig = dict(rmlp=RoutedMLP.forward, spike=Spike.forward, espike=ExpertSpike.forward, bf=Memory.begin_forward,
             usage=Memory._usage, closs=loop_brain.concept_loss)


# ------------------------------------------------------------------ route noise
def rmlp_forward(self, x, *a, **kw):
    if not (self.training and self.noise > 0):
        return _orig["rmlp"](self, x, *a, **kw)
    N = x.shape[0] * x.shape[1]
    k = getattr(self, "_rk", 0)
    self._rk = k + 1
    g = torch.Generator(device=x.device)
    g.manual_seed(1_000_003 * CTX["step"] + 7919 * k + 17)
    full = torch.randn(CTX["accum"] * N, self.E, generator=g, device=x.device)
    mine = full[CTX["mb"] * N:(CTX["mb"] + 1) * N]
    orig = torch.randn_like

    def rl(t, *aa, **kk):
        if tuple(t.shape) == tuple(mine.shape):
            return mine.to(t.dtype)
        return orig(t, *aa, **kk)
    torch.randn_like = rl
    try:
        return _orig["rmlp"](self, x, *a, **kw)
    finally:
        torch.randn_like = orig


# ------------------------------------------------------------------ homeostasis (deferred)
def _acc(self, key_shape, ch, fired, n):
    k = getattr(self, "_hk", 0)
    self._hk = k + 1
    acc = self.__dict__.setdefault("_hacc", {})
    if k not in acc:
        cshape = (self.E, 1) if isinstance(self, ExpertSpike) else (key_shape if ch is not None else ())
        acc[k] = [torch.zeros(key_shape, device=fired.device), torch.zeros(cshape, device=fired.device)]
    f, c = acc[k]
    if isinstance(self, ExpertSpike):
        f += fired
        c += n
    elif ch is None:
        f += fired
        c += n
        self._hch = False
    else:
        f.index_add_(0, ch, fired)
        c.index_add_(0, ch, torch.full_like(fired, float(n)))
        self._hch = True


def spike_forward(self, x, ch=None):
    if self.mode.startswith("ngc") or self.homeo_mode != "floor":
        return _orig["spike"](self, x, ch)
    lo, hi = LEVELS.get(self.mode, (0, 1))
    if self.signed and lo == 0:
        lo = -hi
    analog = x if (self.signed or lo < 0) else F.relu(x)
    if self.mode == "analog" or self.lam == 0:
        return _orig["spike"](self, x, ch)
    u = x
    if self.training and not self.is_calibrated():
        self.calibrate(u, lo < 0, ch=ch)
    alpha = (self.alpha if ch is None else self.alpha[ch]).abs() + 1e-4
    c = torch.clamp(u / alpha, lo, hi)
    q = c + (torch.round(c) - c).detach()
    s = (q * alpha).to(x.dtype)
    if self.track:
        self.rate = (q.detach() != 0).float().flatten(0, -2).mean(0)
    if self.homeo and self.training:
        with torch.no_grad():
            f = (q.detach() != 0).flatten(0, -2)
            _acc(self, self.ema_rate.shape, ch, f.sum(0).float(), f.size(0))
    return s if self.lam >= 1 else analog + self.lam * (s - analog)


def espike_forward(self, x, valid, nmask=None):
    if self.mode == "analog" or self.lam == 0 or self.track or not (self.homeo and self.training) \
            or not self.is_calibrated():
        return _orig["espike"](self, x, valid, nmask)
    analog = F.relu(x)
    v = valid[..., None]
    self.nmask_ref = nmask
    lo, hi = LEVELS[self.mode]
    alpha = (self.alpha.abs() + 1e-4).view(self.E, 1, self.hd)
    c = torch.clamp(x / alpha, lo, hi)
    q = c + (torch.round(c) - c).detach()
    s = (q * alpha).to(x.dtype)
    with torch.no_grad():
        fired = ((q.detach() != 0) & v).sum(1).float()            # (E, h)
        n = v.sum(1).float()                                       # (E, 1)
        _acc(self, (self.E, self.hd), None, fired, n)
    return s if self.lam >= 1 else analog + self.lam * (s - analog)


@torch.no_grad()
def homeo_apply(model, ref):
    for s in model.spikes():
        acc = s.__dict__.pop("_hacc", None)
        if not acc or not s.homeo:
            continue
        for k in sorted(acc):
            f, c = acc[k]
            if isinstance(s, ExpertSpike):
                r = f / c.clamp(min=1)
                used = c[:, 0] > 0
                ema = s.ema_rate.view(s.E, s.hd)
                for _ in range(ref):
                    ema.copy_(torch.where(used[:, None], ema.lerp(r.to(ema.dtype), 0.01), ema))
                    quiet = (ema < s.floor).flatten()
                    a = s.alpha.data
                    a.copy_(torch.where(quiet, a * (1 - s.homeo), a))
            else:
                if getattr(s, "_hch", False):
                    seen = c > 0
                    r = f / c.clamp(min=1)
                else:
                    seen = torch.ones_like(f, dtype=torch.bool)
                    r = f / c.clamp(min=1)
                for _ in range(ref):
                    s.ema_rate.copy_(torch.where(seen, s.ema_rate.lerp(r.to(s.ema_rate.dtype), 0.01), s.ema_rate))
                    a = s.alpha.data
                    a.copy_(torch.where(seen & (s.ema_rate < s.floor), a * (1 - s.homeo), a))


# ------------------------------------------------------------------ memory
def begin_forward(self, collect):
    self._collect = bool(collect) and not self._frozen


def _usage(self, o, q1, q2, n, sfx):
    m = self.m
    pend, qres = getattr(self, "pend" + sfx), getattr(self, "qres" + sfx)
    sel = torch.stack([o["I1"][..., :m], o["I2"][..., :m]], 2).flatten(3)
    pend.scatter_add_(-1, sel, torch.ones_like(sel, dtype=pend.dtype))
    gpos = torch.linspace(0, n * CTX["accum"] - 1, self.nres).long()      # (host)
    lo = CTX["mb"] * n
    slots = ((gpos >= lo) & (gpos < lo + n)).nonzero().squeeze(1)
    if slots.numel():
        pos = (gpos[slots] - lo).to(q1.device)
        sl = slots.to(q1.device)
        qres[:, :, 0, sl] = q1[:, :, pos]
        qres[:, :, 1, sl] = q2[:, :, pos]


# ------------------------------------------------------------------ concept loss
def concept_loss(mu, nu, tgt, valid):
    L = len(CTX["nv_tot"])
    l = CTX["lvl"] % L
    CTX["lvl"] += 1
    v = valid.to(mu.dtype)[..., None]
    nll = (0.5 * (nu + (tgt - mu) ** 2 * torch.exp(-nu)) * v).sum() / (mu.size(-1) * CTX["nv_tot"][l]) * CTX["accum"]
    G, B = CTX["hgroup"], mu.size(0)
    hs = []
    for g0 in range(0, B, G):
        sl = slice(g0, g0 + G)
        vv = v[sl]
        nvg = vv.sum().clamp(min=1)
        hs.append(F.relu(masked_std(tgt[sl], vv, nvg).detach() - masked_std(mu[sl], vv, nvg)).mean())
    # the trainer averages micro-batches (total / accum): scale so the step mean is over ALL groups of the step
    hinge = sum(hs) / len(hs)
    return nll + hinge


# ------------------------------------------------------------------ aux balance loss over fixed token groups
_orig["route"] = RoutedMLP.route


def route_stash(self, xf, side, extra, wave):
    score, zl, P = _orig["route"](self, xf, side, extra, wave)
    if self.training and self.balance in ("aux", "both"):
        self._stash = (score, zl)
    return score, zl, P


def rmlp_auxgroup(self, x, *a, **kw):
    y = _inner_rmlp(self, x, *a, **kw)
    st = self.__dict__.pop("_stash", None)
    if st is not None:
        score, zl = st
        N, E, k = score.size(0), self.E, self.k
        Gt = CTX["hgroup"] * x.shape[1]                      # tokens per group of 8 sequences
        ng = N // Gt
        sp = score / score.sum(1, keepdim=True)
        P = sp.view(ng, Gt, E).mean(1)                        # (groups, E)
        gid = torch.arange(N, device=x.device).repeat_interleave(k, output_size=N * k) // Gt
        f = torch.zeros(ng * E, device=x.device).index_add_(
            0, gid * E + self.last_idx.flatten(), torch.ones(N * k, device=x.device)).view(ng, E) / (Gt * k)
        self.aux = zl + self.aux_w * E * (f * P).sum(1).mean()
    return y


_inner_rmlp = None


# ------------------------------------------------------------------ install / hooks
_lb_forward = None


def lb_forward(self, idx, targets=None, head="lm"):
    for s in self.spikes():
        s._hk = 0
    self.blocks[0].mlp._rk = 0
    return _lb_forward(self, idx, targets, head)


def install(S, names):
    global _lb_forward
    names = set(ALL) if "all" in names else set(names)
    CTX["names"] = names
    if "noise" in names:
        RoutedMLP.forward = rmlp_forward
    if "auxgroup" in names:
        global _inner_rmlp
        _inner_rmlp = RoutedMLP.forward
        RoutedMLP.route = route_stash
        RoutedMLP.forward = rmlp_auxgroup
    if "homeo" in names:
        Spike.forward = spike_forward
        ExpertSpike.forward = espike_forward
    if "memstats" in names:
        Memory.begin_forward = begin_forward
    if "qres" in names:
        Memory._usage = _usage
    if "concept" in names:
        loop_brain.concept_loss = concept_loss
    _lb_forward = LoopBrain.forward
    LoopBrain.forward = lb_forward
    print("accum fixes:", sorted(names), flush=True)


def begin_step(S, step, xs, accum):
    CTX.update(step=step, accum=accum, lvl=0)
    m, mem = S.model, S.mem
    if "memstats" in CTX["names"] and not mem._frozen:
        mem.ns_sum.zero_()
        mem.ns_sq.zero_()
        mem.ns_cnt.zero_()
    if "concept" in CTX["names"]:
        hz, eot = m.cfg["concepts"], m.cfg.get("concept_eot", 0)
        tot = [0.0] * len(hz)
        for x, _ in xs:
            tg = memory.concept_targets(torch.zeros(*x.shape, 1, device=x.device), x, hz, eot)
            for l, (_, v) in enumerate(tg):
                tot[l] = tot[l] + v.sum()
        CTX["nv_tot"] = [t.float().clamp(min=1) for t in tot]   # device scalars (no host sync)


def begin_micro(S, i):
    CTX["mb"] = i
    mem = S.mem
    if "margin" in CTX["names"]:
        if i == 0:
            CTX["gm_state"] = mem.gen_margin.get_state()
        else:
            mem.gen_margin.set_state(CTX["gm_state"])


def end_step(S):
    if "homeo" in CTX["names"]:
        homeo_apply(S.model, CTX["ref"])
