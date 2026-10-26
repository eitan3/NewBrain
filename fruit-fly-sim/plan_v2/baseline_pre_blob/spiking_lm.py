"""Spiking language model that can grow: stages S0/S1 of DESIGN_graph_brain.md.

- Token mixer: "ssd" = Mamba-2 style gated linear recurrence with one scalar decay per head
      S_t = a_t * S_{t-1} + dt_t * k_t v_t^T ,   y_t = q_t S_t
  computed with a chunked parallel scan (no Python loop over time), or "attn" = causal
  softmax attention (the transformer baseline).
- Spikes: Spike() is analog, graded (0..4), ternary (-1/0/1) or binary (0/1) with a learned
  per-channel size alpha and straight-through (LSQ) gradients. `lam` blends analog -> spiking.
  Order in the mixer: per-head RMSNorm -> * silu(gate) -> spike -> out_proj.
- Heads: model(idx) returns features; heads are swappable modules in `model.heads`.
- Growth: grow_heads / grow_mlp / grow_layers keep the function exactly (new parts only write
  through zero-initialised output weights). carry_optimizer() moves Adam state to the grown
  model by parameter name; new slices get exp_avg = 0 and exp_avg_sq = median of the old values.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

LEVELS = {"graded": (0, 4), "binary": (0, 1), "ternary": (-1, 1),
          "ngc": (0, 7), "ngc4": (0, 4)}  # ngc = normalised graded count (0..7 spikes per token)
ALPHA0 = {"graded": 0.5, "binary": 2.0, "ternary": 2.5}  # ~10-40% of N(0,1) inputs fire at init


# ---------------------------------------------------------------- scan
def segsum(x):
    """x (..., T) -> (..., T, T): out[i, j] = sum_{j < k <= i} x[k] for j <= i, else -inf."""
    T = x.size(-1)
    cs = x.cumsum(-1)
    out = cs[..., :, None] - cs[..., None, :]
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    return out.masked_fill(~mask, float("-inf"))


def ssd_scan(q, k, v, la, chunk=64, S0=None):
    """Chunked scan. q, k (B,H,T,N), v (B,H,T,P), la = log decay (B,H,T) <= 0.
    Returns y (B,H,T,P) and the final state (B,H,N,P). Every exp() has a non-positive argument."""
    B, H, T, N = q.shape
    P = v.size(-1)
    pad = (-T) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        la = F.pad(la, (0, pad))
    nc = (T + pad) // chunk
    q, k, v = (t.reshape(B, H, nc, chunk, -1) for t in (q, k, v))
    la = la.reshape(B, H, nc, chunk)
    cs = la.cumsum(-1)
    # inside each chunk: masked, decayed q.k scores
    y = ((q @ k.transpose(-1, -2)) * torch.exp(segsum(la))) @ v
    # state each chunk leaves behind at its end
    states = (k * torch.exp(cs[..., -1:] - cs)[..., None]).transpose(-1, -2) @ v  # (B,H,nc,N,P)
    if S0 is None:
        S0 = q.new_zeros(B, H, N, P)
    states = torch.cat([S0[:, :, None], states], 2)
    decay = torch.exp(segsum(F.pad(cs[..., -1], (1, 0))))                       # (B,H,nc+1,nc+1)
    states = torch.einsum("bhzc,bhcnp->bhznp", decay, states)
    S_prev, S_final = states[:, :, :-1], states[:, :, -1]
    y = y + (q * torch.exp(cs)[..., None]) @ S_prev
    return y.reshape(B, H, nc * chunk, P)[:, :, :T], S_final


def ssd_naive(q, k, v, la, S0=None):
    """Step-by-step recurrence (the inference path / reference for tests)."""
    B, H, T, N = q.shape
    S = q.new_zeros(B, H, N, v.size(-1)) if S0 is None else S0
    ys = []
    for t in range(T):
        S = torch.exp(la[..., t])[..., None, None] * S + k[:, :, t, :, None] * v[:, :, t, None, :]
        ys.append(torch.einsum("bhn,bhnp->bhp", q[:, :, t], S))
    return torch.stack(ys, 2), S


# ---------------------------------------------------------------- building blocks
def expand(module, name, dim, n, fill):
    """Append n entries along `dim` of parameter `name`, keeping the old values."""
    p = getattr(module, name)
    shape = list(p.shape)
    shape[dim] = n
    if fill == "zeros":
        extra = torch.zeros(shape)
    elif fill == "normal":
        extra = torch.randn(shape) * 0.02
    elif callable(fill):
        extra = fill(shape)
    else:
        extra = torch.full(shape, float(fill))
    setattr(module, name, nn.Parameter(torch.cat([p.data, extra.to(p)], dim)))


def expand_linear(lin, n_out=0, n_in=0):
    """New output rows are random (they must receive gradient); new input columns are zero."""
    if n_out:
        expand(lin, "weight", 0, n_out, "normal")
        lin.out_features += n_out
    if n_in:
        expand(lin, "weight", 1, n_in, "zeros")
        lin.in_features += n_in


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


class Spike(nn.Module):
    """Analog or quantised-spike activation. signed=True for the mixer output, False for MLP hidden."""

    def __init__(self, n, mode="analog", signed=False):
        super().__init__()
        self.mode, self.signed = mode, signed
        self.alpha = nn.Parameter(torch.full((n,), ALPHA0.get(mode, 1.0)))
        self.lam = 1.0       # 0 = analog, 1 = fully spiking (set by the trainer)
        self.track = False   # when True, store per-channel firing rates in self.rate
        self.rate = None
        # alpha is set from the real activations the first time spiking switches on in training
        self.register_buffer("calibrated", torch.tensor(mode == "analog"))
        # homeostasis: like a real neuron, each channel nudges its own threshold toward a target
        # firing rate (alpha is then set only by homeostasis, not by gradients). 0 = off.
        self.homeo, self.target = 0.0, 0.15
        # "full": homeostasis owns alpha and pulls every channel to `target` (hurt quality: s1h_*).
        # "floor": alpha stays learned by gradient; homeostasis only lowers the threshold of
        # channels that have gone nearly silent (ema rate < floor), like a rescue mechanism.
        self.homeo_mode, self.floor = "full", 0.01
        self.register_buffer("ema_rate", torch.full((n,), self.target))
        # weight of each channel in the NGC normalisation; newborn channels start at 0 and ramp to 1
        # (ramp_in), so a birth does not change how the existing channels are normalised
        self.register_buffer("norm_w", torch.ones(n))
        self._calibrated = False  # host mirror of `calibrated` once it is True (reading the buffer is a host sync)

    def is_calibrated(self):
        if not self._calibrated:
            self._calibrated = bool(self.calibrated)
        return self._calibrated

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._calibrated = False  # re-read the loaded flag

    @torch.no_grad()
    def calibrate(self, x, signed_levels, target=0.2, ch=None):
        """Per-channel alpha so that about `target` of inputs fire (a spike needs |x| > alpha/2).
        ch: the channels x holds (pass-routed head groups); the channels not computed now start like a
        typical calibrated one (as newborn channels do in grow), since the flag is shared by all channels."""
        v = x.detach().float().flatten(0, -2)
        v = v[torch.randperm(v.size(0), device=v.device)[:4096]]
        v = v.abs() if signed_levels else v
        th = v.sort(0).values[int((1 - target) * (v.size(0) - 1))]
        a = (2 * th).clamp(min=1e-3)
        if ch is None:
            self.alpha.copy_(a)
        else:
            self.alpha.fill_(a.median().item())
            self.alpha[ch] = a.to(self.alpha.dtype)
        self.calibrated.fill_(True)

    def forward(self, x, ch=None):
        """ch: optional channel indices of x's last dim (only some head groups are computed this pass);
        thresholds, normalisation weights and firing statistics are then those of these channels."""
        lo, hi = LEVELS.get(self.mode, (0, 1))
        if self.signed and lo == 0:
            lo = -hi
        analog = x if (self.signed or lo < 0) else F.relu(x)
        if self.mode == "analog" or self.lam == 0:
            if self.track:
                self.rate = (analog.detach() != 0).float().flatten(0, -2).mean(0)
            return analog
        ngc = self.mode.startswith("ngc")
        if ngc:
            # divisive normalisation: each token's signal is measured against its own mean magnitude,
            # so thresholds don't move when the overall scale changes (e.g. after growth)
            w = (self.norm_w if ch is None else self.norm_w[ch]).to(torch.float32)
            m = (x.float().abs() * w).sum(-1, keepdim=True) / w.sum() + 1e-6
            u = x.float() / m
        else:
            u = x
        if self.training and not self.is_calibrated():
            self.calibrate(u, lo < 0, ch=ch)
        alpha = (self.alpha if ch is None else self.alpha[ch]).abs() + 1e-4
        if self.homeo and self.homeo_mode == "full":
            alpha = alpha.detach()
        if ngc:
            # LSQ gradient scale for the threshold (Esser et al. 2020)
            g = 1.0 / math.sqrt(u.numel() / u.size(-1) * hi)
            alpha = alpha * g + (alpha - alpha * g).detach()
            lin = u / alpha
            c = torch.clamp(lin, lo, hi)
            # leaky straight-through only where the neuron is saturated (above hi, or below -hi when signed):
            # below 0 an unsigned neuron's output really is flat, and a fake gradient there broke training
            # (v2s_ngc: loss rose once spiking switched on)
            over = (lin > hi) | (lin < lo) if lo < 0 else (lin > hi)
            leak = 0.1 * (lin - c) * over
            c = c + leak - leak.detach()
        else:
            c = torch.clamp(u / alpha, lo, hi)
        q = c + (torch.round(c) - c).detach()   # straight-through round; autograd gives LSQ grads to alpha
        s = (q * alpha * m).to(x.dtype) if ngc else (q * alpha).to(x.dtype)
        if self.track:
            self.rate = (q.detach() != 0).float().flatten(0, -2).mean(0)
        if self.homeo and self.training:
            with torch.no_grad():
                r = (q.detach() != 0).to(self.ema_rate.dtype).flatten(0, -2).mean(0)
                if ch is None:
                    self.ema_rate.lerp_(r, 0.01)
                    if self.homeo_mode == "full":
                        # firing too much -> raise the threshold, too little -> lower it
                        self.alpha.data.mul_(torch.exp(self.homeo * (r - self.target) / self.target).clamp(0.9, 1.1))
                    else:  # (torch.where, not a boolean index: no host sync)
                        a = self.alpha.data
                        a.copy_(torch.where(self.ema_rate < self.floor, a * (1 - self.homeo), a))
                else:  # only the channels computed this pass are observed, so only they adapt
                    self.ema_rate[ch] = self.ema_rate[ch].lerp(r, 0.01)
                    if self.homeo_mode == "full":
                        self.alpha.data[ch] *= torch.exp(self.homeo * (r - self.target) / self.target).clamp(0.9, 1.1)
                    else:
                        a = self.alpha.data[ch]
                        self.alpha.data[ch] = torch.where(self.ema_rate[ch] < self.floor, a * (1 - self.homeo), a)
        return s if self.lam >= 1 else analog + self.lam * (s - analog)

    def grow(self, n):
        # new channels fire like a typical old one (if not calibrated yet, all get calibrated later)
        expand(self, "alpha", 0, n, self.alpha.data.abs().median().item())
        self.ema_rate = torch.cat([self.ema_rate, self.ema_rate.new_full((n,), self.target)])
        self.norm_w = torch.cat([self.norm_w, self.norm_w.new_zeros(n)])

    @torch.no_grad()
    def ramp_in(self, delta):
        """Newborn channels gradually join the NGC normalisation (call once per training step)."""
        self.norm_w.add_(delta).clamp_(max=1.0)


class CausalConv(nn.Module):
    """Depthwise causal conv over time, kernel 4 (replaces RWKV token-shift)."""

    def __init__(self, c, k=4):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.randn(c, k) / math.sqrt(k))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x, ch=None):  # (B,T,C); ch: which channels x holds (pass-routed head groups)
        w, b = (self.weight, self.bias) if ch is None else (self.weight[ch], self.bias[ch])
        x = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return F.conv1d(x, w[:, None, :].to(x.dtype), b.to(x.dtype), groups=x.size(1)).transpose(1, 2)

    def grow(self, n):
        expand(self, "weight", 0, n, lambda s: torch.randn(s) / math.sqrt(self.k))
        expand(self, "bias", 0, n, "zeros")


class SSDMixer(nn.Module):
    """Pass-routed head groups (n_groups > 0, the looped brain): the first `group0` heads are group 0 and
    always computed; every later head belongs to one of the routed groups 1..n_groups (round-robin by
    index, so a head's group never changes). forward(x, group=g) computes group 0 + group g only.
    A head that is not computed on a pass does nothing on that pass: no output, and no state either,
    because the scan state lives inside one forward (each pass re-scans the whole sequence from S = 0
    for the heads it computes; nothing is carried from pass to pass). The rule depends only on which
    group a pass uses, so it is the same in training and evaluation."""

    def __init__(self, d, heads, head_dim=64, spike="analog", chunk=64, group0=0, n_groups=0):
        super().__init__()
        self.h, self.p, self.chunk = heads, head_dim, chunk
        self.group0, self.n_groups = group0, n_groups
        self.register_buffer("hgroup", torch.tensor([self._group_of(i) for i in range(heads)], dtype=torch.long))
        self._active = {}  # (group, heads, device) -> (head idx, channel idx); avoids a host sync per pass
        di = heads * head_dim
        self.wz, self.wx, self.wq, self.wk = (nn.Linear(d, di, bias=False) for _ in range(4))
        self.wdt = nn.Linear(d, heads, bias=False)
        self.cx, self.cq, self.ck = CausalConv(di), CausalConv(di), CausalConv(di)
        self.A_log, self.dt_bias = nn.Parameter(torch.empty(heads)), nn.Parameter(torch.empty(heads))
        self._init_decay(self.A_log.data, self.dt_bias.data)
        self.Dskip = nn.Parameter(torch.ones(heads))
        self.norm_w = nn.Parameter(torch.ones(heads, head_dim))
        self.spike = Spike(di, spike, signed=True)
        self.wo = nn.Linear(di, d, bias=False)
        self.register_buffer("ramp", torch.ones(di))  # newborn heads ramp in (smooth growth)
        self.ramping = False

    @staticmethod
    def _init_decay(A_log, dt_bias):
        # A in [1, 16], dt log-uniform in [1e-3, 1e-1]: spreads memory timescales across heads
        A_log.copy_(torch.empty_like(A_log).uniform_(1, 16).log())
        dt = torch.exp(torch.empty_like(dt_bias).uniform_(math.log(1e-3), math.log(1e-1)))
        dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def _group_of(self, i):
        return 0 if (not self.n_groups or i < self.group0) else 1 + (i - self.group0) % self.n_groups

    def active(self, group):
        """(head indices, channel indices) computed on a pass of `group`; (None, None) = all heads."""
        if group is None or not self.n_groups:
            return None, None
        key = (int(group), self.h, self.hgroup.device)
        if key not in self._active:
            hs = ((self.hgroup == 0) | (self.hgroup == group)).nonzero().squeeze(1)
            ch = (hs[:, None] * self.p + torch.arange(self.p, device=hs.device)).flatten()
            self._active[key] = (hs, ch)
        return self._active[key]

    def idle_params(self):
        """Parameters not used on an average pass (idle routed head groups), for n_active."""
        if not self.n_groups:
            return 0.0
        routed = max(0, self.h - self.group0)
        per_head = sum(p.numel() for p in self.parameters()) / self.h  # every mixer parameter scales with h
        return per_head * routed * (1 - 1 / self.n_groups)

    def forward(self, x, write=None, group=None):
        """write: optional (B, T) weights in [0, 1]; 0 = this token is not integrated into the membranes.
        group: routed head group computed on this pass (with group 0); None = all heads (see class doc)."""
        if write is not None:  # unselected tokens also must not leak in through the short causal convs
            x = x * write[..., None].to(x.dtype)
        B, T, _ = x.shape
        hs, ch = self.active(group)
        H, P = (self.h if hs is None else len(hs)), self.p
        lin = lambda m, i: m(x) if i is None else F.linear(x, m.weight[i])
        sel = lambda t, i: t if i is None else t[i]
        heads = lambda t: t.view(B, T, H, P).transpose(1, 2)
        z = lin(self.wz, ch)
        v = heads(F.silu(self.cx(lin(self.wx, ch), ch)))
        q = F.normalize(heads(self.cq(lin(self.wq, ch), ch)).float(), dim=-1)
        k = F.normalize(heads(self.ck(lin(self.wk, ch), ch)).float(), dim=-1)
        dt = F.softplus(lin(self.wdt, hs).float() + sel(self.dt_bias, hs)).transpose(1, 2)          # (B,H,T)
        if write is not None:
            dt = dt * write.float()[:, None, :]
        la = -(dt * torch.exp(sel(self.A_log, hs))[:, None]).clamp(max=30.0)
        with torch.autocast(x.device.type, enabled=False):  # decay maths and state in fp32
            y, _ = ssd_scan(q, k, v.float() * dt[..., None], la, self.chunk)
        y = y + sel(self.Dskip, hs)[:, None, None] * v.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6) * sel(self.norm_w, hs)[:, None]  # per-head norm
        y = y.transpose(1, 2).reshape(B, T, H * P).to(z.dtype) * F.silu(z)
        y = self.spike(y) if ch is None else self.spike(y, ch)
        if self.ramping:
            y = y * sel(self.ramp, ch).to(y.dtype)
        return self.wo(y) if ch is None else F.linear(y, self.wo.weight[:, ch])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints from before head groups have no hgroup (all heads were dense = group 0)
        state_dict.setdefault(prefix + "hgroup", self.hgroup.clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._active = {}
        self.ramping = bool((self.ramp < 1).any())  # a checkpoint taken mid-ramp keeps ramping

    @torch.no_grad()
    def grow_heads(self, n, ramp=False):
        m = n * self.p
        self.hgroup = torch.cat([self.hgroup, torch.tensor([self._group_of(self.h + i) for i in range(n)],
                                                           dtype=torch.long, device=self.hgroup.device)])
        self._active = {}
        self.ramp = torch.cat([self.ramp, self.ramp.new_zeros(m) if ramp else self.ramp.new_ones(m)])
        self.ramping = self.ramping or ramp
        for lin in (self.wz, self.wx, self.wq, self.wk):
            expand_linear(lin, n_out=m)
        expand_linear(self.wdt, n_out=n)
        for c in (self.cx, self.cq, self.ck):
            c.grow(m)
        A_log, dt_bias = torch.empty(n), torch.empty(n)
        self._init_decay(A_log, dt_bias)
        expand(self, "A_log", 0, n, lambda s: A_log)
        expand(self, "dt_bias", 0, n, lambda s: dt_bias)
        expand(self, "Dskip", 0, n, 1.0)
        expand(self, "norm_w", 0, n, 1.0)
        self.spike.grow(m)
        expand_linear(self.wo, n_in=m)
        self.h += n


class AttnMixer(nn.Module):
    """Causal softmax attention with rotary positions (transformer baseline)."""

    def __init__(self, d, heads, head_dim=64, spike="analog", chunk=None):
        super().__init__()
        self.h, self.p = heads, head_dim
        di = heads * head_dim
        self.wq, self.wk, self.wv = (nn.Linear(d, di, bias=False) for _ in range(3))
        self.spike = Spike(di, spike, signed=True)
        self.wo = nn.Linear(di, d, bias=False)
        self._rope = None

    def rope(self, T, device):
        if self._rope is None or self._rope[0].size(0) < T:
            inv = 1.0 / (10000 ** (torch.arange(0, self.p, 2, device=device).float() / self.p))
            f = torch.outer(torch.arange(T, device=device).float(), inv)
            self._rope = (f.cos(), f.sin())
        return self._rope[0][:T], self._rope[1][:T]

    def forward(self, x):
        B, T, _ = x.shape
        heads = lambda t: t.view(B, T, self.h, self.p).transpose(1, 2)
        q, k, v = heads(self.wq(x)), heads(self.wk(x)), heads(self.wv(x))
        cos, sin = self.rope(T, x.device)

        def rot(t):
            a, b = t[..., ::2].float(), t[..., 1::2].float()
            return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2).to(t.dtype)

        y = F.scaled_dot_product_attention(rot(q), rot(k), v, is_causal=True)
        return self.wo(self.spike(y.transpose(1, 2).reshape(B, T, -1)))

    @torch.no_grad()
    def grow_heads(self, n):
        m = n * self.p
        for lin in (self.wq, self.wk, self.wv):
            expand_linear(lin, n_out=m)
        self.spike.grow(m)
        expand_linear(self.wo, n_in=m)
        self.h += n


class MLP(nn.Module):
    def __init__(self, d, hidden, spike="analog"):
        super().__init__()
        self.up = nn.Linear(d, hidden, bias=False)
        self.spike = Spike(hidden, spike, signed=False)
        self.down = nn.Linear(hidden, d, bias=False)
        # smooth growth: newborn units' output is multiplied by a ramp that rises 0 -> 1 after birth
        self.register_buffer("ramp", torch.ones(hidden))
        # continual-backprop utility: EMA of |output| per unit (tracked when track_util is on)
        self.register_buffer("util", torch.zeros(hidden))
        self.ramping, self.track_util = False, False

    def forward(self, x):
        s = self.spike(self.up(x))
        if self.ramping:
            s = s * self.ramp.to(s.dtype)
        if self.track_util and self.training:
            with torch.no_grad():
                self.util.mul_(0.99).add_(0.01 * s.detach().abs().float().flatten(0, -2).mean(0))
        return self.down(s)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.ramping = bool((self.ramp < 1).any())  # a checkpoint taken mid-ramp keeps ramping

    @torch.no_grad()
    def grow(self, n, ramp=False):
        expand_linear(self.up, n_out=n)
        self.spike.grow(n)
        expand_linear(self.down, n_in=n)
        self.ramp = torch.cat([self.ramp, self.ramp.new_zeros(n) if ramp else self.ramp.new_ones(n)])
        self.util = torch.cat([self.util, self.util.new_zeros(n)])
        self.ramping = self.ramping or ramp

    @torch.no_grad()
    def redo(self, min_rate=1e-3):
        """ReDo: re-initialise hidden units that stopped firing. A silent unit outputs 0, so giving
        it fresh input weights and zero output weights leaves the network's function unchanged."""
        dead = (self.spike.ema_rate < min_rate).nonzero().squeeze(1)
        if len(dead):
            self.up.weight[dead] = torch.randn_like(self.up.weight[dead]) * self.up.weight.std()
            self.down.weight[:, dead] = 0
            self.spike.alpha[dead] = self.spike.alpha.abs().median()
            self.spike.ema_rate[dead] = self.spike.target
        return dead


class ExpertSpike(Spike):
    """Spikes inside the routed experts. Input (E, C, h) with a mask of which capacity slots hold real
    tokens: thresholds are calibrated and firing statistics counted ONLY on tokens actually routed to an
    expert, so a rarely used expert is never mistaken for a dead one."""

    def __init__(self, n_exp, hidden, mode="analog"):
        super().__init__(n_exp * hidden, mode, signed=False)
        self.E, self.hd = n_exp, hidden

    def forward(self, x, valid, nmask=None):
        analog = F.relu(x)
        v = valid[..., None]
        self.nmask_ref = nmask
        if self.mode == "analog" or self.lam == 0:
            if self.track:
                self.rate = self._rates(analog != 0, v)
            return analog
        lo, hi = LEVELS[self.mode]
        if self.training and not self.is_calibrated():
            with torch.no_grad():
                vals = x.detach().float()
                vals = vals.abs() if lo < 0 else vals
                th = torch.nanquantile(vals.masked_fill(~v, float("nan")), 0.8, dim=1)  # (E, h): ~20% fire
                th = torch.where(torch.isnan(th), th.nanmedian(), th)
                self.alpha.copy_((2 * th).flatten().clamp(min=1e-3))
                self.calibrated.fill_(True)
        alpha = (self.alpha.abs() + 1e-4).view(self.E, 1, self.hd)
        c = torch.clamp(x / alpha, lo, hi)
        q = c + (torch.round(c) - c).detach()
        s = (q * alpha).to(x.dtype)
        if self.track or (self.homeo and self.training):
            r, used = self._rates(q.detach() != 0, v, per_expert=True)
            if self.track:
                self.rate = r[used][nmask[used] > 0] if nmask is not None else r[used].flatten()
            if self.homeo and self.training:
                with torch.no_grad():  # (torch.where, not boolean indexing: no host sync)
                    ema = self.ema_rate.view(self.E, self.hd)
                    ema.copy_(torch.where(used[:, None], ema.lerp(r.to(ema.dtype), 0.01), ema))
                    quiet = (ema < self.floor).flatten()
                    a = self.alpha.data
                    a.copy_(torch.where(quiet, a * (1 - self.homeo), a))
        return s if self.lam >= 1 else analog + self.lam * (s - analog)

    def _rates(self, fired, v, per_expert=False):
        n = v.sum(1).clamp(min=1)                                      # (E, 1) routed tokens per expert
        r = (fired & v).sum(1).float() / n                             # (E, h)
        used = v.sum(1).squeeze(-1) > 0
        if per_expert:
            return r, used
        nm = getattr(self, "nmask_ref", None)
        return r[used][nm[used] > 0] if nm is not None else r[used].flatten()

    @torch.no_grad()
    def grow_experts(self, src):
        a = self.alpha.data.view(self.E, self.hd)
        self.alpha = nn.Parameter(torch.cat([a, a[src]]).flatten())
        e = self.ema_rate.view(self.E, self.hd)
        self.ema_rate = torch.cat([e, e[src]]).flatten()
        self.norm_w = torch.cat([self.norm_w, self.norm_w.new_ones(len(src) * self.hd)])
        self.E += len(src)


class RoutedMLP(nn.Module):
    """Only part of the brain is active: 1 always-on shared expert + n_exp small experts, each token uses k.
    Tokens are sorted by expert and padded to a fixed capacity, so all experts run in one batched matmul.
    balance="auxfree": a per-expert bias (selection only) is nudged toward equal load (DeepSeek-V3 style);
    balance="aux": classic load-balancing loss instead."""

    def __init__(self, d, n_exp=64, k=6, hidden=192, shared=384, spike="analog", balance="auxfree",
                 capacity=1.25, u=1e-3, aux_w=0.01, z_w=1e-3, layers=6, d_side=0, lift=0.0, birth_delta=0.5,
                 shield=False, birth_mode="ramp", bias_logit=False):
        super().__init__()
        # lift (aux mode): the aux loss only pushes DOWN busy experts (its gradient on an expert is proportional
        # to that expert's traffic), so an expert nobody picks - e.g. a newborn whose router row starts off its
        # parent's - never gets picked or trained. lift > 0 raises the selection bias of experts whose load is
        # below lift x the average (rate u), and lets a raised bias relax back to 0 once the expert is busy.
        # birth_delta: size of a newborn's router-row change relative to the parent's row (ramp / split births).
        self.lift, self.birth_delta = lift, birth_delta
        # shield: experts still ramping in (debt > 0, so not selectable yet) are left out of the z-loss and of the
        # aux loss's P, so their router rows get no gradient at all (no Adam step, no weight decay) until they
        # can be picked. In v4 the P normalisation (raising an unpicked expert's score lowers everyone else's P)
        # inflated ramping newborns to scores ~0.7, they flooded when released and were then crushed to ~0.001.
        # balance="both": the auxfree bias balancer AND the aux loss.
        self.shield = shield
        # bias_logit: the selection bias (balancer, newborn debt, lift) is added to the router LOGIT instead of the
        # sigmoid score. In the loop the scores are squashed to ~0.01, so on the score scale a bias of a few 1e-3
        # already overrides the router (the lift run: all biases -> 0.7, "highest bias wins", 96x overload);
        # logits have a spread of ~3, so the same steps are gentle and a debt of -2 is a gradual ramp-in.
        self.bias_logit = bias_logit
        # hidden: one size for all experts, or a list with each expert's own number of neurons (padded to
        # the largest; unused neuron slots are masked off)
        sizes = list(hidden) if isinstance(hidden, (list, tuple)) else None
        if sizes:
            n_exp, hidden = len(sizes), max(sizes)
        self.E, self.k, self.h, self.cap, self.u = n_exp, k, hidden, capacity, u
        self.balance, self.aux_w, self.z_w = balance, aux_w, z_w
        self.shared = MLP(d, shared, spike) if shared else None
        self.router = nn.Linear(d, n_exp, bias=False)
        # optional second router input (the loop brain's working memory), starts silent
        self.router_s = nn.Linear(d_side, n_exp, bias=False) if d_side else None
        if self.router_s is not None:
            nn.init.zeros_(self.router_s.weight)
        # deferred balancing (loop brain): counts are summed over all passes, bias updated once per step
        self.defer = False
        self._pending_any = False  # a training forward added counts since the last apply_balance
        self.register_buffer("pending", torch.zeros(n_exp))
        self.register_buffer("pending_elig", torch.zeros(n_exp))  # tokens for which each expert was allowed
        self.w_up = nn.Parameter(torch.randn(n_exp, d, hidden) * 0.02)
        self.w_down = nn.Parameter(torch.randn(n_exp, hidden, d) * 0.02 / math.sqrt(2 * layers))
        self.register_buffer("bias", torch.zeros(n_exp))
        self.register_buffer("debt", torch.zeros(n_exp))  # newborn experts: bias still to be paid back
        self.register_buffer("load_ema", torch.ones(n_exp))  # traffic per expert (1 = average), for births
        self.register_buffer("age", torch.full((n_exp,), 10**6))  # training steps since birth
        self.birth_mode = birth_mode
        nmask = torch.ones(n_exp, hidden)
        if sizes:
            for i, s in enumerate(sizes):
                nmask[i, s:] = 0
        self.register_buffer("nmask", nmask)  # which neuron slots of each expert exist
        self.hetero = bool(sizes) and min(sizes) < hidden
        self.last_idx = None  # experts chosen per token in the last forward (for the refractory rule)
        self.spike_e = ExpertSpike(n_exp, hidden, spike)
        self.aux = None    # auxiliary loss of the last forward (added by the trainer)
        self.stats = None  # routing statistics of the last forward
        # routing "temperature": noise (std, in router-logit units) on WHICH experts are picked, set by the
        # trainer (constant, or annealed). eval_noise: the same at evaluation (default 0 = clean, deterministic).
        # Gate weights and the balance loss always use the clean scores.
        self.noise = 0.0
        self.eval_noise = 0.0

    def route(self, xf, side, extra, wave):
        """Router scores (N, E), z-loss and (aux balance mode) the mean routing probability per expert.
        wave = (prev_idx, fatigue, conn, gamma, beta): the loop's wave term gamma * conn[prev experts]
        summed - beta * fatigue, built here with embedding_bag (exactly prev_onehot @ conn, without the
        dense E x E matmul per token)."""
        acc = torch.promote_types(xf.dtype, torch.float32)             # float32 (or float64 in tests)
        logits = self.router(xf).to(acc)
        if side is not None and self.router_s is not None:
            logits = logits + self.router_s(side.reshape(xf.size(0), -1)).to(acc)
        live = (self.debt <= 0) if self.shield else None                 # selectable experts (shield)
        # router-stability (z) loss sees the router's own logits only, not the wave terms
        zl = self.z_w * torch.logsumexp(logits if live is None else logits.masked_fill(~live, float("-inf")),
                                        -1).pow(2).mean()
        if extra is not None:  # e.g. dense wave terms (learned expert connectivity, fatigue)
            logits = logits + extra.to(acc)
        if wave is not None:
            prev_idx, fatigue, conn, g, b = wave
            logits = logits + (g * F.embedding_bag(prev_idx, conn, mode="sum") - b * fatigue).to(acc)
        score = logits.sigmoid()
        P = None
        if self.balance in ("aux", "both"):
            sp = score if live is None else score * live
            P = (sp / sp.sum(1, keepdim=True)).mean(0)
        return score, zl, P

    def forward(self, x, forbid=None, side=None, extra=None, wave=None, pmem=None):
        """forbid: optional (B*T, E) bool mask of experts a token may not use now (refractory rule).
        side: optional (B, T, d_side) extra router input (the loop brain's working memory).
        wave: optional loop wave inputs, see route().
        pmem: optional private expert memory (memory.Memory.private): called with the capacity layout of the kept
        slots, returns the (N, D) memory output W_p(sum_e gate sigma(b_e) rho), added to y."""
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N, E, k = xf.size(0), self.E, self.k
        acc = torch.promote_types(x.dtype, torch.float32)              # float32 (or float64 in tests)
        if getattr(self, "ckpt_route", False) and torch.is_grad_enabled():
            # memory: the (tokens x E) routing intermediates are recomputed in backward, not stored
            score, zl, P = torch.utils.checkpoint.checkpoint(self.route, xf, side, extra, wave, use_reentrant=False)
        else:
            score, zl, P = self.route(xf, side, extra, wave)
        sel = score + self.bias
        temp = self.noise if self.training else self.eval_noise
        if self.bias_logit or temp > 0:  # selection on the logit scale (no gradient)
            lg = torch.logit(score.detach().clamp(1e-6, 1 - 1e-6))
            if temp > 0:  # noisy selection: perturb the logits, pick with the bias as usual
                lg = lg + temp * torch.randn_like(lg)
            sel = lg + self.bias if self.bias_logit else lg.sigmoid() + self.bias
        if forbid is not None:
            sel = sel.masked_fill(forbid, float("-inf"))
        idx = sel.topk(k, -1).indices                                   # selection uses the bias
        self.last_idx = idx
        gate = score.gather(1, idx)
        gate = gate / gate.sum(1, keepdim=True)                         # weights do not
        flat = idx.flatten()
        order = flat.argsort(stable=True)
        e_sorted, tok = flat[order], order // k
        counts = torch.bincount(flat, minlength=E)
        # the buffer shapes come from the counts: ONE host sync per forward (the counts), the rest on the host
        cl = counts.tolist()
        # capacity per expert, counted over experts that actually receive tokens: unselected newborn
        # experts must not shrink everyone else's capacity (that would change the function at birth)
        used = max(1, sum(c > 0 for c in cl))
        if self.cap and self.cap > 0:
            C = max(1, min(max(cl), math.ceil(self.cap * N * k / used)))
        else:  # dropless: every routed token is computed (drops would compound over the loop's passes)
            C = max(1, max(cl))
        pos = torch.arange(N * k, device=x.device) - (counts.cumsum(0) - counts)[e_sorted]
        keep = pos < C
        # the kept slots in order, without a boolean index (a host sync each): their number is known here
        n_keep = sum(min(c, C) for c in cl)
        kidx = None if n_keep == N * k else (~keep).to(torch.uint8).argsort(stable=True)[:n_keep]
        kept = lambda t: t if kidx is None else t[kidx]
        e_k, p_k, t_k = kept(e_sorted), kept(pos), kept(tok)
        buf = xf.new_zeros(E, C, D)
        buf[e_k, p_k] = xf[t_k]
        valid = torch.zeros(E, C, dtype=torch.bool, device=x.device)
        valid[e_k, p_k] = True
        h = self.spike_e(torch.bmm(buf, self.w_up.to(buf.dtype)), valid, self.nmask if self.hetero else None)
        if self.hetero:
            h = h * self.nmask[:, None, :].to(h.dtype)
        out = torch.bmm(h, self.w_down.to(h.dtype))
        y = torch.zeros(N, D, device=x.device, dtype=acc)
        y.index_add_(0, t_k, out[e_k, p_k].to(acc) * kept(gate.flatten()[order])[:, None])
        if pmem is not None:
            y = y + pmem(e_k, p_k, t_k, kept(gate.flatten()[order]), kept(order), N, k, E, C).to(acc)
        aux = zl
        if self.balance in ("aux", "both"):
            f = counts.float() / (N * k)
            aux = aux + self.aux_w * E * (f * P).sum()
        self.aux = aux
        if self.training and self.defer:
            self._pending_any = True  # (so apply_balance needs no host sync to know it has counts)
            with torch.no_grad():
                self.pending += counts.float()
                self.pending_elig += N - (forbid.sum(0).float() if forbid is not None else 0.0)
        elif self.training:
            with torch.no_grad():
                self._balance(counts.float(), counts.float().mean())
        if self.training and not self.defer:
            with torch.no_grad():
                self.load_ema.mul_(0.99).add_(0.01 * counts.float() / counts.float().mean().clamp(min=1e-6))
                self.age += 1
        self.stats = (counts.detach(), (~keep).sum().detach(), N * k)
        y = y.to(x.dtype).view(B, T, D)
        return y + self.shared(x) if self.shared is not None else y

    @torch.no_grad()
    def _balance(self, counts, mean):
        """Balancer step (auxfree) and newborn ramp-in (every mode) from per-expert counts."""
        if self.balance in ("auxfree", "both"):
            # the balancer leaves experts that are still ramping in alone (no overshoot)
            # proportional to the relative load error (capped): an expert at 10x the average load is
            # pushed down 4x faster than the old fixed-size sign step; balanced experts barely move
            err = (mean - counts) / mean.clamp(min=1e-6)
            self.bias += self.u * err.clamp(-4.0, 1.0) * (self.debt <= 0)
            # a newborn that already gets its fair share stops ramping; the balancer takes over
            self.debt.masked_fill_(counts >= mean, 0)
        elif self.lift > 0:  # aux mode: lift under-used experts (never below 0: overload is the aux loss's job)
            # a newborn that already gets its fair share stops paying back its debt (as in auxfree): fresh rows
            # score far above the trained ones, so paying back in full would flood them with tokens
            self.debt.masked_fill_(counts >= mean, 0)
            thr = self.lift * mean
            up = ((thr - counts) / thr.clamp(min=1e-6)).clamp(0.0, 1.0) * (self.debt <= 0)
            down = ((counts - mean) / mean.clamp(min=1e-6)).clamp(0.0, 1.0) * (self.bias > 0)
            self.bias += self.u * up
            self.bias -= torch.minimum(self.bias.clamp(min=0), self.u * down)
        # newborns ramp into routing over ~1000 steps. This must also run in "aux" mode (it used to be
        # auxfree-only, so with --route-aux a newborn stayed 2 below everyone and was never selected).
        # In aux mode without lift no balancer takes over, so the debt is paid in full (bias ends at 0 like
        # every other expert). Note: there the z-loss shrinks the fresh row while it ramps, and the aux loss
        # never pulls an unpicked expert up, so most newborns end up unpicked (v4 at step 2000): use lift.
        pay = self.debt.clamp(max=0.002)
        self.bias += pay
        self.debt -= pay

    @torch.no_grad()
    def apply_balance(self):
        """Deferred mode: one balancer / ramp / load / age update per optimizer step, from the counts
        summed over all passes of that step."""
        if not self._pending_any and self.pending.sum() == 0:
            return
        self._pending_any = False
        # load per ELIGIBLE token: under the refractory rule a popular expert is forbidden for some tokens,
        # and raw counts would make it look under-used (the balancer would then over-boost it)
        rate = self.pending / self.pending_elig.clamp(min=1)
        counts = rate * self.pending_elig.mean()
        mean = counts.mean().clamp(min=1e-6)
        self._balance(counts, mean)
        self.load_ema.mul_(0.99).add_(0.01 * counts / mean)
        self.age += 1
        self.pending.zero_()
        self.pending_elig.zero_()

    @torch.no_grad()
    def grow_experts(self, n):
        """Birth of n experts, each a noisy copy of a currently overloaded expert (most negative balancer
        bias = most pressure). Output weights are 0 and the router bias starts 2 below the parent, so the
        new expert is not selected at birth (function unchanged); the balancer then raises its bias step
        by step and it gradually takes over part of its parent's traffic."""
        noise = lambda t: t + torch.randn_like(t) * 0.01 * t.std()
        if self.birth_mode == "split":
            # split the BUSIEST experts (by measured traffic, each at most once per event) into twins.
            # The router row r becomes r - delta (parent) and r + delta (child), so each twin wins about
            # half of the parent's tokens (those with delta.x < 0 or > 0). The child copies the parent's
            # weights, so a token gets nearly the same computation whichever twin it reaches.
            src = self.load_ema.argsort(descending=True)[:n]
            if len(src) < n:
                src = src[torch.arange(n, device=src.device) % len(src)]
            row = self.router.weight.data[src]
            delta = F.normalize(torch.randn_like(row), dim=1) * row.norm(dim=1, keepdim=True) * self.birth_delta
            self.router.weight.data[src] = row - delta
            new_rows, new_down = row + delta, self.w_down.data[src].clone()
            new_bias, new_debt = self.bias[src].clone(), self.debt.new_zeros(n)
            self.load_ema[src] *= 0.5
            new_load = self.load_ema[src].clone()
        elif self.birth_mode == "ramp":
            # parents = busiest MATURE experts by measured traffic (never one still ramping in, and each
            # parent's traffic is halved so it is not picked again right away). The child gets its own
            # router direction (parent row + delta), zero output weights, and starts 2 below the parent's
            # bias, so it is not selected at birth (function unchanged); it then ramps in over ~1000 steps
            # with the balancer paused for it, and wins the tokens on its side of delta.
            young = (self.debt > 0) | (self.age < 500)  # never split an expert that is still settling in
            load = torch.where(young, torch.full_like(self.load_ema, -1.0), self.load_ema)
            src = load.argsort(descending=True)[:n]
            row = self.router.weight.data[src]
            delta = F.normalize(torch.randn_like(row), dim=1) * row.norm(dim=1, keepdim=True) * self.birth_delta
            new_rows, new_down = row + delta, torch.zeros_like(self.w_down.data[src])
            new_bias, new_debt = self.bias[src] - 2.0, self.debt.new_full((n,), 2.0)
            self.load_ema[src] *= 0.5
            new_load = self.load_ema[src].clone()
        else:
            # "legacy" (v2 first try): parent = most negative bias. That picked overloaded newborns, which
            # cascaded into chains of ever lower biases: 29 of 32 newborns per layer never got a token.
            src = self.bias.argsort()[:n]
            if len(src) < n:
                src = src[torch.arange(n, device=src.device) % len(src)]
            new_rows, new_down = noise(self.router.weight.data[src]), torch.zeros_like(self.w_down.data[src])
            new_bias, new_debt = self.bias[src] - 2.0, self.debt.new_full((n,), 2.0)
            new_load = self.load_ema.new_zeros(n)
        self.w_up = nn.Parameter(torch.cat([self.w_up.data, noise(self.w_up.data[src])]))
        self.w_down = nn.Parameter(torch.cat([self.w_down.data, new_down]))
        self.router.weight = nn.Parameter(torch.cat([self.router.weight.data, new_rows]))
        self.router.out_features += n
        if self.router_s is not None:
            self.router_s.weight = nn.Parameter(torch.cat([self.router_s.weight.data, self.router_s.weight.data[src]]))
            self.router_s.out_features += n
        self.pending = torch.cat([self.pending, self.pending.new_zeros(n)])
        self.pending_elig = torch.cat([self.pending_elig, self.pending_elig.new_zeros(n)])
        self.bias = torch.cat([self.bias, new_bias])
        self.debt = torch.cat([self.debt, new_debt])
        self.load_ema = torch.cat([self.load_ema, new_load])
        self.age = torch.cat([self.age, self.age.new_zeros(n)])
        self.spike_e.grow_experts(src)
        self.nmask = torch.cat([self.nmask, self.nmask[src]])
        self.E += n
        return src  # parents, so the loop can extend its expert-to-expert connectivity

    def dense_reference(self, x):
        """All experts computed densely, combined with the same gates (no capacity drops). For tests."""
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        score = self.router(xf).to(torch.promote_types(x.dtype, torch.float32)).sigmoid()
        idx = (score + self.bias).topk(self.k, -1).indices
        gate = torch.zeros_like(score).scatter(1, idx, score.gather(1, idx))
        gate = gate / gate.sum(1, keepdim=True)
        h = F.relu(torch.einsum("nd,edh->enh", xf, self.w_up))
        y = torch.einsum("enh,ehd,ne->nd", h, self.w_down, gate.to(h.dtype)).view(B, T, D)
        return y + self.shared(x) if self.shared is not None else y


MIXERS = {"ssd": SSDMixer, "attn": AttnMixer}


class Block(nn.Module):
    def __init__(self, d, heads, head_dim, mlp, mixer, spike, chunk, moe=None, layers=6):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mix = MIXERS[mixer](d, heads, head_dim, spike, chunk)
        self.mlp = RoutedMLP(d, spike=spike, layers=layers, **moe) if moe else MLP(d, mlp, spike)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.mlp(self.n2(x))


class SpikingLM(nn.Module):
    def __init__(self, vocab, d=384, layers=6, heads=6, head_dim=64, mlp=1536, mixer="ssd",
                 spike="analog", chunk=64, moe=None, concept=None):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, head_dim=head_dim, mlp=mlp,
                        mixer=mixer, spike=spike, chunk=chunk, moe=moe, concept=concept)
        self.emb = nn.Embedding(vocab, d)
        per = lambda v, i: v[i] if isinstance(v, (list, tuple)) else v  # widths may differ per layer
        self.blocks = nn.ModuleList(Block(d, per(heads, i), head_dim, per(mlp, i), mixer, spike, chunk, moe, layers)
                                    for i in range(layers))
        self.normf = RMSNorm(d)
        self.heads = nn.ModuleDict({"lm": nn.Linear(d, vocab, bias=False)})  # swappable output heads
        self.apply(self._init)
        self.concept_loss = None
        if concept:  # next-concept planning (concepts.py): frozen concept tokenizer + learned head
            from concepts import ConceptHead, ConceptTokenizer
            ck = torch.load(concept["path"], map_location="cpu")
            tok = ConceptTokenizer(**ck["cfg"])
            tok.load_state_dict(ck["state"])
            tok.requires_grad_(False).eval()
            self.concept_tok = tok
            self.concept_head = ConceptHead(d, tok, n_levels=concept.get("levels", 3))
            self.plan_proj = nn.Linear(tok.d, d, bias=False)
            nn.init.zeros_(self.plan_proj.weight)  # the plan starts silent and is learned
            self.concept_split = concept.get("split", 2)
            self.concept_corrupt = concept.get("corrupt", 0.1)
        for b in self.blocks:  # small residual-output init (GPT-2 style)
            outs = [b.mix.wo] + ([b.mlp.down] if isinstance(b.mlp, MLP) else
                                 [b.mlp.shared.down] if b.mlp.shared is not None else [])
            for lin in outs:
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * layers))

    def aux_loss(self):
        terms = [b.mlp.aux for b in self.blocks if isinstance(b.mlp, RoutedMLP) and b.mlp.aux is not None]
        return sum(terms) if terms else None

    def routing_stats(self):
        """Per routed layer: usage entropy / log E, dead-expert fraction, max/mean load, capacity drop rate."""
        out = []
        for b in self.blocks:
            if isinstance(b.mlp, RoutedMLP) and b.mlp.stats is not None:
                c, dropped, total = b.mlp.stats
                p = c.float() / c.sum()
                ent = -(p * (p + 1e-12).log()).sum() / math.log(len(p))
                out.append(dict(entropy=ent.item(), dead=(c < 0.1 * c.float().mean()).float().mean().item(),
                                maxload=(c.max() / c.float().mean()).item(), drop=dropped.item() / total))
        return out

    def n_active(self):
        """Parameters used per token (routed experts count k/E of their weights)."""
        n = n_params(self)
        for b in self.blocks:
            if isinstance(b.mlp, RoutedMLP):
                n -= (b.mlp.w_up.numel() + b.mlp.w_down.numel()) * (1 - b.mlp.k / b.mlp.E)
        for m in self.modules():  # pass-routed head groups: idle groups are not used on a pass
            if isinstance(m, SSDMixer):
                n -= m.idle_params()
        return int(n)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def features(self, idx):
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x)
            if self.cfg.get("concept") and i == self.concept_split:
                x = x + self._plan(x, idx)
        return self.normf(x)

    def _plan(self, x, idx):
        """Top-down plan: the state at the last token of chunk j predicts chunk j+1's concept codes
        (coarse to fine); the brain's own soft prediction is added to every token of chunk j+1.
        Also sets self.concept_loss (CE of the head against the frozen tokenizer's true codes)."""
        C = self.concept_tok.chunk
        B, T, D = x.shape
        n = math.ceil(T / C)
        if n * C != T:  # generation with a partial last chunk: pad (padding never reaches earlier tokens)
            x = F.pad(x, (0, 0, 0, n * C - T))
        h = F.layer_norm(x[:, C - 1::C], (D,))                                # (B, n, D) chunk-end states
        plan = self.concept_head.plan(h.flatten(0, 1)).view(B, n, C, -1)      # belief about chunk j+1
        plan = self.plan_proj(plan.to(x.dtype))
        plan = torch.cat([torch.zeros_like(plan[:, :1]), plan[:, :-1]], 1)  # chunk j+1 gets chunk j's plan
        self.concept_loss, self.concept_acc = None, None
        if self.training or T % C == 0:
            if T % C == 0 and n > 1:
                with torch.no_grad():
                    nxt = idx[:, C:].reshape(-1, C)                           # chunks 1..n-1
                    codes = self.concept_tok.codes(nxt, self.concept_head.K)
                logits = self.concept_head.teacher_logits(h[:, :-1].flatten(0, 1), codes,
                                                          self.concept_corrupt if self.training else 0.0)
                ces = [F.cross_entropy(lg.float().flatten(0, 1), c.flatten()) for lg, c in zip(logits, codes)]
                self.concept_loss = sum(ces) / len(ces)                       # every level weighted equally
                self.concept_acc = [(lg.argmax(-1) == c).float().mean().detach() for lg, c in zip(logits, codes)]
        return plan.reshape(B, n * C, D)[:, :T]

    def forward(self, idx, targets=None, head="lm"):
        out = self.heads[head](self.features(idx))
        if targets is None:
            return out
        return out, F.cross_entropy(out.float().flatten(0, 1), targets.flatten())

    def spikes(self):
        return [m for m in self.modules() if isinstance(m, Spike)]

    def set_lam(self, lam):
        for s in self.spikes():
            s.lam = lam

    def set_homeo(self, eta, target=0.15, mode="full"):
        for s in self.spikes():
            s.homeo, s.target, s.homeo_mode = eta, target, mode

    # ------------------------------------------------------------ growth
    @torch.no_grad()
    def grow_heads(self, n):
        for b in self.blocks:
            b.mix.grow_heads(n)
        self.cfg["heads"] += n

    @torch.no_grad()
    def grow_mlp(self, n):
        for b in self.blocks:
            b.mlp.grow(n)
        self.cfg["mlp"] += n

    @torch.no_grad()
    def grow_layers(self, n, where="top", init="fresh"):
        """Add n blocks whose output projections are zero, so each starts as the identity.
        where: "top" appends; "spread" inserts them evenly between existing blocks.
        init: "fresh" random input weights; "copy" copies the block below (stacking-style),
        so the new block starts from useful features. Returns {old index: new index}."""
        c = self.cfg
        L = len(self.blocks)
        after = [L - 1] * n if where == "top" else [round((i + 1) * L / (n + 1)) - 1 for i in range(n)]
        old = list(self.blocks)
        lam = self.spikes()[0].lam if self.spikes() else 1.0
        new_list, mapping = [], {}
        for i, b in enumerate(old):
            mapping[i] = len(new_list)
            new_list.append(b)
            for _ in range(after.count(i)):
                if init == "copy":
                    nb = copy.deepcopy(b)
                else:
                    nb = Block(c["d"], c["heads"], c["head_dim"], c["mlp"], c["mixer"], c["spike"], c["chunk"])
                    nb.apply(self._init)
                    for s in (nb.mix.spike, nb.mlp.spike):
                        s.lam = lam
                nn.init.zeros_(nb.mix.wo.weight)
                nn.init.zeros_(nb.mlp.down.weight)
                new_list.append(nb.to(self.emb.weight.device))
        self.blocks = nn.ModuleList(new_list)
        self.cfg["layers"] += n
        return mapping


def snapshot_optimizer(model, opt):
    """Adam state keyed by parameter name, taken before growing."""
    return {name: opt.state[p] for name, p in model.named_parameters() if p in opt.state and opt.state[p]}


def carry_optimizer(model, snap, new_opt):
    """Fill new_opt (built on the grown model) with the old state. Old slices keep their moments;
    new slices get exp_avg = 0 and exp_avg_sq = median of the old values, so their first update
    is not ~3x the learning rate (the bug in growing_brain.py's _carry_optimizer)."""
    for name, p in model.named_parameters():
        st = snap.get(name)
        if not st:
            continue
        ns = {"step": st["step"].clone()}
        for key in ("exp_avg", "exp_avg_sq"):
            old = st[key]
            buf = torch.zeros_like(p, memory_format=torch.preserve_format)
            if key == "exp_avg_sq" and old.shape != p.shape:
                buf.fill_(old.float().median().item())
            buf[tuple(slice(0, s) for s in old.shape)] = old
            ns[key] = buf
        new_opt.state[p] = ns
    return new_opt


def remap_name(name, mapping):
    """blocks.<i>.x -> blocks.<new i>.x after grow_layers inserted blocks in the middle."""
    if name.startswith("blocks."):
        _, i, rest = name.split(".", 2)
        name = f"blocks.{mapping[int(i)]}.{rest}"
    return name


def remap_snapshot(snap, mapping):
    """Rename blocks.<i>.* keys after grow_layers inserted blocks in the middle."""
    return {remap_name(name, mapping): st for name, st in snap.items()}


@torch.no_grad()
def redo_dead(model, opt, min_rate=1e-3):
    """Re-initialise silent MLP units in every block and reset their Adam state. Returns the count."""
    total = 0
    for b in model.blocks:
        if not isinstance(b.mlp, MLP):
            continue
        dead = b.mlp.redo(min_rate)
        if len(dead):
            for p, sl in ((b.mlp.up.weight, (dead,)), (b.mlp.down.weight, (slice(None), dead))):
                st = opt.state.get(p)
                if st:
                    st["exp_avg"][sl] = 0
                    st["exp_avg_sq"][sl] = st["exp_avg_sq"].median()
            total += len(dead)
    return total


def n_params(model, embeddings=False):
    """Trainable parameters (frozen parts such as the concept tokenizer are not counted)."""
    return sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and (embeddings or not n.startswith("emb.")))
