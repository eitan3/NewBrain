"""Smooth growth (neurogenesis) and smooth renewal for SpikingLM (DESIGN_v2.md sections 3-4).

Every `every` steps a few MLP neurons are born, spread over the layers; heads are born one at a time,
staggered over layers. Births follow a linear schedule between `t0` and `t1` (fractions of training).

Birth protocol (each newborn, and each renewed neuron):
  - output weights 0 and an output ramp 0 -> 1 over `ramp` steps (function unchanged at birth, smooth after)
  - input weights in a useful direction: top singular vectors of M = E[g_out x_in^T] (GradMax-style),
    mixed 50/50 with a random direction, scaled like the siblings ("where": "grad"); random if "uniform"
  - its own spike threshold, calibrated on its own pre-activations to the layer's current firing rate
  - optimizer state: carried by name; new slices are born now (young-neuron LR under BrainOpt maturity)
WHERE: births go to layers in proportion to softmax(log score / 0.5), score = sum of the top-8 squared
singular values of M (how much a new neuron could reduce the loss); "uniform" spreads them evenly.
Renewal (continual backprop): every `every` steps, in each layer, at most `renew_cap` of the least useful
mature neurons (utility = EMA|output| x sum|output weights|, age > `mature` steps) are re-born.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from spiking_lm import MLP, RoutedMLP, SSDMixer


class Neurogenesis:
    def __init__(self, model, start, final, steps, every=25, t0=0.05, t1=0.6, where="grad", ramp=1000,
                 renew_rate=1e-5, renew_cap=2, mature=2000, score_every=100, protocol=True):
        self.m, self.steps, self.every, self.where, self.ramp_steps = model, steps, every, where, ramp
        self.protocol = protocol  # False = ablation: random input, inherited threshold, no output ramp
        self.t0, self.t1 = int(t0 * steps), int(t1 * steps)
        self.start, self.final = start, final
        self.renew_rate, self.renew_cap, self.mature, self.score_every = renew_rate, renew_cap, mature, score_every
        L = len(model.blocks)
        # routed brain: growth = births of experts (and heads); MLP-neuron births/renewal only for dense MLPs
        self.routed = isinstance(model.blocks[0].mlp, RoutedMLP)
        self.exp_start = model.blocks[0].mlp.E if self.routed else 0
        self.exp_target = final.get("experts", self.exp_start)
        self.mlp_target = final["mlp"]
        self.heads_target = final["heads"]
        self.born_at = [torch.zeros(b.mlp.up.out_features if not self.routed else 0, dtype=torch.long)
                        for b in model.blocks]
        self.score = np.ones(L)
        self.V = [None] * L      # top singular directions per layer (for newborn input weights)
        self.xs = [None] * L     # sample of MLP inputs per layer (for threshold calibration)
        self.hooks = []
        self.births = self.renewals = 0
        self.ramp_calls = 0
        # heads: birth times staggered over layers
        n_head_births = sum(final["heads"] - b.mix.h for b in model.blocks)
        self.head_times = list(np.linspace(self.t0, self.t1, n_head_births + 2)[1:-1].astype(int))
        for b in model.blocks:
            if isinstance(b.mlp, MLP):
                b.mlp.track_util = True

    # ---------------------------------------------------------------- scoring (gradient pressure)
    def want_score(self, step):
        return step % self.score_every == 0 and not self.routed

    def attach(self):
        """Hook MLP inputs and output gradients for one training step."""
        self.cache = [{} for _ in self.m.blocks]
        for i, b in enumerate(self.m.blocks):
            def fwd(mod, inp, out, i=i):
                self.cache[i]["x"] = inp[0].detach().flatten(0, -2)
                out.register_hook(lambda g, i=i: self.cache[i].__setitem__("g", g.detach().flatten(0, -2)))
            self.hooks.append(b.mlp.register_forward_hook(fwd))

    @torch.no_grad()
    def detach_and_score(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        for i, c in enumerate(self.cache):
            if "g" not in c:
                continue
            x, g = c["x"].float(), c["g"].float()
            M = g.T @ x / x.size(0)                      # (d, d): how output-gradient correlates with input
            _, S, Vh = torch.linalg.svd(M)
            s = (S[:8] ** 2).sum().item()
            self.score[i] = 0.7 * self.score[i] + 0.3 * s if self.score[i] != 1 else s
            self.V[i] = Vh[:16]                          # input directions that most reduce the loss
            self.xs[i] = x[torch.randperm(x.size(0), device=x.device)[:4096]]
        self.cache = None

    # ---------------------------------------------------------------- birth protocol
    @torch.no_grad()
    def _init_rows(self, i, idx):
        """Give neurons `idx` of layer i fresh input weights, zero output, ramp 0, own threshold."""
        mlp = self.m.blocks[i].mlp
        W = mlp.up.weight
        alive = torch.ones(W.size(0), dtype=torch.bool, device=W.device)
        alive[idx] = False
        rho = W[alive].norm(dim=1).median() if alive.any() else W.norm(dim=1).median()
        n = len(idx)
        r = torch.randn(n, W.size(1), device=W.device)
        if self.where == "grad" and self.protocol and self.V[i] is not None:
            v = self.V[i][torch.arange(n, device=W.device) % self.V[i].size(0)]
            v = v * torch.sign(torch.randn(n, 1, device=W.device))
            r = 0.5 * F.normalize(v, dim=1) + 0.5 * F.normalize(r - (r * v).sum(1, keepdim=True) * v, dim=1)
        W[idx] = (F.normalize(r, dim=1) * rho).to(W.dtype)
        mlp.down.weight[:, idx] = 0
        if self.protocol:
            mlp.ramp[idx] = 0
            mlp.ramping = True
        else:
            mlp.ramp[idx] = 1
            mlp.spike.alpha[idx] = mlp.spike.alpha[alive].abs().median() if alive.any() else mlp.spike.alpha[idx]
            return
        mlp.util[idx] = mlp.util.median() if alive.any() else 0
        sp = mlp.spike
        rate = sp.ema_rate[alive].median().clamp(0.02, 0.5).item() if alive.any() else sp.target
        sp.ema_rate[idx] = rate
        if hasattr(sp, "norm_w"):
            sp.norm_w[idx] = 0
        if sp.mode != "analog" and self.xs[i] is not None:
            pre = self.xs[i] @ W[idx].float().T                 # newborns' own pre-activations
            if sp.mode.startswith("ngc"):
                m = (self.xs[i] @ W.float().T).abs().mul(sp.norm_w.float()).sum(-1, keepdim=True) / sp.norm_w.sum().clamp(min=1)
                pre = pre / (m + 1e-6)
            lo = -1 if sp.mode == "ternary" else 0
            v = pre.abs() if lo < 0 else pre
            th = v.quantile(1 - rate, dim=0)
            sp.alpha[idx] = (2 * th).clamp(min=1e-3).to(sp.alpha.dtype)

    def _allocate(self, n):
        """Split n births over layers that still have room."""
        room = np.array([max(0, self.mlp_target - b.mlp.up.out_features) for b in self.m.blocks], dtype=float)
        if room.sum() == 0 or n == 0:
            return np.zeros(len(room), dtype=int)
        if self.where == "grad":
            w = np.exp((np.log(self.score + 1e-12) - np.log(self.score + 1e-12).max()) / 0.5)
        else:
            w = np.ones(len(room))
        w = w * (room > 0)
        p = w / w.sum()
        out = np.minimum(np.random.multinomial(n, p), room.astype(int))
        left = min(n, int(room.sum())) - out.sum()
        while left > 0:  # give births that did not fit to layers that still have room
            free = np.flatnonzero(out < room)
            j = free[np.argmax(w[free] + 1e-12)] if self.where == "grad" else np.random.choice(free)
            out[j] += 1
            left -= 1
        return out

    # ---------------------------------------------------------------- per-step driver
    @torch.no_grad()
    def step(self, step):
        """Ramps, births and renewal for this step. Returns True if the parameter set changed
        (the optimizer must then be rebuilt)."""
        self.ramps()
        if step % self.every:
            return False
        changed = self.births_step(step)
        self.renew(step)
        return changed

    @torch.no_grad()
    def ramps(self):
        """Newborns ramp in a little more every step (smooth, intended change of the function).
        No host syncs on most steps: "still ramping?" is read every `every` calls (a finished ramp that is
        still flagged multiplies by exactly 1.0), and the NGC weights are clamped at 1 anyway."""
        self.ramp_calls += 1
        check = self.ramp_calls % self.every == 0
        for b in self.m.blocks:  # ramps
            if isinstance(b.mlp, MLP) and b.mlp.ramping:
                b.mlp.ramp.add_(1.0 / self.ramp_steps).clamp_(max=1.0)
                if check:
                    b.mlp.ramping = bool((b.mlp.ramp < 1).any())
            mix = b.mix
            if getattr(mix, "ramping", False):
                mix.ramp.add_(1.0 / self.ramp_steps).clamp_(max=1.0)
                if check:
                    mix.ramping = bool((mix.ramp < 1).any())
        for s in self.m.spikes():  # newborn channels join the NGC normalisation gradually
            s.ramp_in(1.0 / self.ramp_steps)  # (a no-op once every weight is 1: they never exceed it)

    def done(self):
        """True once no more births can happen: the parameter set has stopped changing."""
        if self.head_times:
            return False
        if self.routed:
            return all(b.mlp.E >= self.exp_target for b in self.m.blocks)
        return all(b.mlp.up.out_features >= self.mlp_target for b in self.m.blocks)

    @torch.no_grad()
    def births_step(self, step):
        """Births at exactly zero output (function unchanged). Returns True if parameters changed."""
        changed = False
        # ---- expert births (routed brain): every layer grows to the scheduled number of experts
        # (also after t1: births only happen every `every` steps, so when t1 is not a multiple of it the
        # last scheduled ones would be skipped, e.g. 766 instead of 768 experts; the next event catches up)
        if self.routed and self.t0 <= step:
            f = min(1.0, (step - self.t0) / max(1, self.t1 - self.t0))
            target = int(round(self.exp_start + (self.exp_target - self.exp_start) * f))
            for b in self.m.blocks:
                if b.mlp.E < target:
                    self.births += target - b.mlp.E
                    # the loop's Core also extends its wave connectivity / sheet for the newborns
                    (b.grow_experts if hasattr(b, "grow_experts") else b.mlp.grow_experts)(target - b.mlp.E)
                    changed = True
            self.m.cfg["moe"] = dict(self.m.cfg["moe"], n_exp=self.m.blocks[0].mlp.E)
        # ---- MLP births on a linear schedule
        if not self.routed and self.t0 <= step <= self.t1:
            total_target = sum(self._target_width(step) for _ in self.m.blocks)
            have = sum(b.mlp.up.out_features for b in self.m.blocks)
            alloc = self._allocate(max(0, total_target - have))
            for i, n in enumerate(alloc):
                if n:
                    mlp = self.m.blocks[i].mlp
                    old = mlp.up.out_features
                    mlp.grow(int(n), ramp=self.protocol)
                    idx = torch.arange(old, old + int(n), device=mlp.up.weight.device)
                    self._init_rows(i, idx)
                    self.born_at[i] = torch.cat([self.born_at[i], torch.full((int(n),), step)])
                    self.births += int(n)
                    changed = True
        # ---- head births, one at a time, round-robin over layers that still need heads
        while self.head_times and step >= self.head_times[0]:
            self.head_times.pop(0)
            cands = [b for b in self.m.blocks if b.mix.h < self.heads_target]
            if cands:
                b = min(cands, key=lambda b: b.mix.h)
                b.mix.grow_heads(1, ramp=self.protocol and isinstance(b.mix, SSDMixer))
                changed = True
        if not self.routed:
            self.m.cfg["mlp"] = [b.mlp.up.out_features for b in self.m.blocks]
        heads = [b.mix.h for b in self.m.blocks]
        # the looped brain has one core and LoopBrain(**cfg) needs an int (a list broke checkpoint reloads)
        self.m.cfg["heads"] = heads[0] if self.m.cfg.get("arch") == "loop" else heads
        return changed

    @torch.no_grad()
    def renew(self, step):
        """Re-birth of the least useful mature neurons (a small, bounded change of the function)."""
        if step > self.mature and not self.routed:
            for i, b in enumerate(self.m.blocks):
                mlp = b.mlp
                n = mlp.up.out_features
                k = min(self.renew_cap, np.random.poisson(self.renew_rate * n * self.every))
                if not k:
                    continue
                age = step - self.born_at[i].to(mlp.util.device)
                util = mlp.util * mlp.down.weight.abs().sum(0).float()
                util = torch.where(age > self.mature, util, torch.full_like(util, float("inf")))
                dead = (mlp.spike.ema_rate < 1e-3) & (age > self.mature)
                util = torch.where(dead, torch.full_like(util, -1.0), util)
                idx = util.topk(k, largest=False).indices
                idx = idx[torch.isfinite(util[idx])]
                if len(idx):
                    self._init_rows(i, idx)
                    self.born_at[i][idx.cpu()] = step
                    self.renewals += len(idx)
                    self.renewed = [(i, idx)] + getattr(self, "renewed", [])

    def _target_width(self, step):
        f = min(1.0, max(0.0, (step - self.t0) / max(1, self.t1 - self.t0)))
        return int(round(self.start["mlp"] + (self.mlp_target - self.start["mlp"]) * f))

    def pop_renewed(self):
        """Renewed (layer, indices) since the last call, so the trainer can reset their optimizer state."""
        r = getattr(self, "renewed", [])
        self.renewed = []
        return r


@torch.no_grad()
def reset_opt_rows(opt, model, layer, idx):
    """Fresh optimizer state for re-born MLP neurons `idx` of block `layer` (zero momentum, median 2nd
    moment, born now). Works for BrainOpt (state by name) and torch AdamW (state by parameter)."""
    params = dict(model.named_parameters())
    pre = f"blocks.{layer}.mlp."
    for name, sl in ((pre + "up.weight", (idx,)), (pre + "down.weight", (slice(None), idx)), (pre + "spike.alpha", (idx,))):
        if name not in params:
            continue
        if hasattr(opt, "groups"):  # BrainOpt
            st = opt.state.get(name, {})
            for key, buf in st.items():
                if buf.shape != params[name].shape:  # a shared step count (per row / tensor): per element now
                    buf = st[key] = buf.expand(params[name].shape).clone()
                buf[sl] = {"born": float(opt.t), "v": buf.median().item()}.get(key, 0.0)
        else:
            st = opt.state.get(params[name], {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st:
                    st[key][sl] = st[key].median() if key == "exp_avg_sq" else 0