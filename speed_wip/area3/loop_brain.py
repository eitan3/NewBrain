"""The looped brain (DESIGN_v3_loop.md): one set of routed "areas" reused for K passes, Eitan's design.

    token -> embedding e -> x_0 = e, s_0 = f(e)
    repeat K times with the SAME weights:
        u = RMSNorm(x + e + read(s))               input re-injected every pass; latent state read additively
        m = Temporal(RMSNorm(u))                   leaky-membrane populations over the token sequence (word-to-
                                                   word context), graded spikes; with pops > 1 these are
                                                   ROUTED populations (each word picks 2 per pass; a population
                                                   does not integrate words it was not picked for)
        h = u + m
        o = Experts(RMSNorm(h))                    router picks N of E spiking experts (+1 shared, always on).
                                                   Expert choice per pass is shaped either by the hard
                                                   refractory rule (no expert twice in a row) or by WAVES:
                                                   score += gamma * C . used_prev - beta * fatigue, with C a
                                                   learned expert-to-expert connectivity (init: neighbours on a
                                                   16x24 sheet) and fatigue f <- 0.7 f + used
        latent state s:
          - vector mode (slots = 0): s = gated update from [h; o; s]          ("memory" variants of night 1)
          - WORKSPACE mode (slots > 0): s = 8 slots x 64; the pass result c = m + o is WRITTEN into the
            slots (attention-addressed, gated), and the thought x is updated ONLY from what is READ back
            out of the workspace ("think in s, speak in x")
        x = RMSNorm(x + alpha * update)
    logits = Decoder(RMSNorm(x_K))                 s is never decoded

The core is exposed as self.blocks[0] (.mix = temporal, .mlp = experts) so spikes, routing stats and the
optimizer work unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from memory import ConceptHier, Memory, concept_loss, concept_targets, masked_std
from spiking_lm import RMSNorm, RoutedMLP, SpikingLM, SSDMixer


def torus_kernel(E, rows=16, sigma=1.5):
    """Expert-to-expert connectivity init: experts on a rows x (E/rows) torus sheet, Gaussian neighbours."""
    cols = math.ceil(E / rows)
    idx = torch.arange(E)
    r, c = idx // cols, idx % cols
    dr = (r[:, None] - r[None]).abs()
    dc = (c[:, None] - c[None]).abs()
    dr, dc = torch.minimum(dr, rows - dr), torch.minimum(dc, cols - dc)
    K = torch.exp(-(dr ** 2 + dc ** 2).float() / (2 * sigma ** 2))
    K.fill_diagonal_(0)  # no self-excitation: an expert does not re-trigger itself
    return K / K.sum(1, keepdim=True)


def sheet_adj(pos, rows, cols, sigma=1.5):
    """Sheet neighbours from (continuous) expert positions on a fixed rows x cols torus: the same Gaussian
    and the same > 0.05 cut as the initial adj2, normalised by a FULL grid's row sum (a constant), so on
    the full initial grid it equals (torus_kernel > 0.05) and a newborn's neighbours depend only on
    distance, not on how crowded the sheet has become."""
    def gauss(dr, dc):
        dr, dc = dr.abs() % rows, dc.abs() % cols
        dr, dc = torch.minimum(dr, rows - dr), torch.minimum(dc, cols - dc)
        return torch.exp(-(dr ** 2 + dc ** 2) / (2 * sigma ** 2))
    g = torch.arange(rows * cols, device=pos.device)
    Z = gauss((g // cols).double(), (g % cols).double())[1:].sum()   # full grid, without self
    p = pos.double()
    K = gauss(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    K.fill_diagonal_(0)
    return (K / Z > 0.05).to(pos.dtype)


class TemporalPops(nn.Module):
    """Several temporal membrane populations; each word routes to `k` of them per pass."""

    def __init__(self, d, pops, heads, head_dim, spike, chunk, k=2):
        super().__init__()
        self.k = k
        self.pops = nn.ModuleList(SSDMixer(d, heads, head_dim, spike, chunk) for _ in range(pops))
        self.router = nn.Linear(d, pops, bias=False)
        self.register_buffer("bias", torch.zeros(pops))  # self-adjusting balance (selection only)
        self.h = sum(p.h for p in self.pops)
        self.last_load = None

    def forward(self, x):
        score = self.router(x).float().sigmoid()                      # (B, T, P)
        top = (score + self.bias).topk(self.k, -1).indices
        gate = torch.zeros_like(score).scatter(-1, top, score.gather(-1, top))
        gate = gate / gate.sum(-1, keepdim=True)
        self.last_load = (gate > 0).float().mean((0, 1)).detach()
        if self.training:
            with torch.no_grad():
                load = self.last_load
                self.bias += 1e-3 * ((load.mean() - load) / load.mean().clamp(min=1e-6)).clamp(-4, 1)
        out = 0
        for i, pop in enumerate(self.pops):  # unselected words are not integrated (dt = 0) and get no output
            out = out + pop(x, write=(gate[..., i] > 0).to(x.dtype)) * gate[..., i:i + 1].to(x.dtype)
        return out


class Core(nn.Module):
    """The routed "cortex" that every pass reuses."""

    def __init__(self, d, d_s, heads, head_dim, moe, spike, chunk, state, slots=0, waves=False, pops=1,
                 ws_direct=False, head_group=0, head_groups=0, sheet_cols=0):
        super().__init__()
        self.state, self.slots, self.waves, self.ws_direct = state, slots, waves, ws_direct
        self.n_u, self.n_t, self.n_m, self.n_x, self.n_e = (RMSNorm(d) for _ in range(5))
        # head_groups > 0: pass-routed head groups (the first head_group heads always on, the rest in
        # head_groups groups that take turns over the passes, see SSDMixer)
        self.n_groups = head_groups if pops == 1 else 0
        self.mix = (TemporalPops(d, pops, max(1, heads // 2), head_dim, spike, chunk) if pops > 1
                    else SSDMixer(d, heads, head_dim, spike, chunk, group0=head_group, n_groups=self.n_groups))
        side = d_s if (state and not slots) else (64 if slots else 0)
        self.mlp = RoutedMLP(d, spike=spike, layers=6, d_side=side, **moe)
        self.mlp.defer = True
        self.alpha = nn.Parameter(torch.full((d,), 0.2))
        if waves:  # learned expert-to-expert connectivity ("connectome"); name contains "router" -> AdamW
            E = self.mlp.E
            # a FIXED sheet (16 x sheet_cols, set by the starting pool): experts born later sit next to their
            # parent, so positions (and the diagnostics) do not shift when the pool grows
            self.sheet_rows, self.sheet_cols = 16, sheet_cols or math.ceil(E / 16)
            i = torch.arange(E)
            self.register_buffer("pos", torch.stack([(i // self.sheet_cols) % 16, i % self.sheet_cols], 1).float())
            fresh = self.sheet_cols == math.ceil(E / 16)  # the usual case: identical to the index-based torus
            K = torus_kernel(E) if fresh else self._sheet_kernel()
            self.router_conn = nn.Parameter(K)
            self.wave_gamma = nn.Parameter(torch.tensor(math.log(math.exp(3.0) - 1)))  # softplus -> 3.0
            # (at 1.0 the excitation was ~0.2 vs router-logit std ~0.4: waves were too weak to steer routing)
            self.register_buffer("adj2", (K > 0.05).float())  # sheet neighbours (diagnostics)
            self.wave_beta = nn.Parameter(torch.tensor(math.log(math.exp(2.0) - 1)))  # softplus -> 2.0
        if slots:  # multi-slot workspace
            self.ds = 64
            self.w_0 = nn.Linear(d, slots * self.ds, bias=False)        # s_0 from the token itself
            self.n_c = RMSNorm(d)
            self.w_wv = nn.Linear(d, slots * self.ds, bias=False)       # what to write, per slot
            self.w_wg = nn.Linear(d, slots)                             # how strongly to write, per slot
            self.w_rq = nn.Linear(d, self.ds, bias=False)               # what to read (core input)
            self.w_ro = nn.Linear(self.ds, d, bias=False)               # read-out -> core input
            self.w_xo = nn.Linear(slots * self.ds, d, bias=False)       # whole workspace -> thought x (full rank)
            if ws_direct:  # workspace v2: learned per-channel balance between direct path and workspace,
                # starting at 90% direct (the strict "only through the workspace" version was a bottleneck)
                self.ws_mix = nn.Parameter(torch.full((d,), math.log(0.1 / 0.9)))
            self.n_s = RMSNorm(self.ds)
        elif state:
            self.n_s = RMSNorm(d_s)
            self.w_s = nn.Linear(d_s, d, bias=False)
            self.w_0 = nn.Linear(d, d_s, bias=False)
            self.w_z = nn.Linear(2 * d + d_s, d_s)
            self.w_c = nn.Linear(2 * d + d_s, d_s)

    def _sheet_kernel(self, sigma=1.5):
        """Row-normalised Gaussian connectivity from the positions (initial router_conn when the sheet is
        not the default one, e.g. a pool built directly at a larger size on a fixed sheet)."""
        rows, cols = self.sheet_rows, self.sheet_cols
        d = self.pos[:, None] - self.pos[None]
        dr, dc = d[..., 0].abs() % rows, d[..., 1].abs() % cols
        dr, dc = torch.minimum(dr, rows - dr), torch.minimum(dc, cols - dc)
        K = torch.exp(-(dr ** 2 + dc ** 2) / (2 * sigma ** 2))
        K.fill_diagonal_(0)
        return K / K.sum(1, keepdim=True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # checkpoints from before the fixed sheet have no pos (pools never grew then: index grid = pos)
        if self.waves:
            state_dict.setdefault(prefix + "pos", self.pos.clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def grow_experts(self, n):
        """Expert births (RoutedMLP.grow_experts) plus the wave connectivity of the newborns. Each newborn
        copies its parent's outgoing row and incoming column of router_conn (the old block is unchanged);
        the entries between a newborn and its parent / siblings and its own diagonal are 0. Exact: the
        newborn is never selected at birth (bias 2 below), so its row adds nothing and it has no fatigue."""
        src = self.mlp.grow_experts(n)
        hook = getattr(self, "mem_hook", None)  # memory (plan_v2/12): copy-on-split of the private rows
        if hook is not None:
            hook(src, self.mlp.E - n)
        if self.waves:
            C = self.router_conn.data
            E0 = C.size(0)
            par = torch.cat([torch.arange(E0, device=C.device), src.to(C.device)])
            new = C[par][:, par].clone()                        # every entry from the parents' entries
            twin = par[:, None] == par[None, :]                 # newborn <-> parent / sibling / itself
            twin[:E0, :E0] = False                              # the old block stays exactly as it was
            new[twin] = 0
            self.router_conn = nn.Parameter(new)
            rows, cols = self.sheet_rows, self.sheet_cols
            jit = self.pos[src] + torch.randn(n, 2, device=self.pos.device, dtype=self.pos.dtype) * 0.3
            jit = torch.stack([jit[:, 0].remainder(rows), jit[:, 1].remainder(cols)], 1)  # wrap around the torus
            self.pos = torch.cat([self.pos, jit])
            self.adj2 = sheet_adj(self.pos, rows, cols).to(self.adj2.dtype)
        return src

    # ---------------------------------------------------------------- latent state
    def init_state(self, e):
        s0 = torch.tanh(self.w_0(self.n_e(e)))
        return s0.view(*e.shape[:-1], self.slots, self.ds) if self.slots else s0

    def read(self, s, q):
        """Attention read of the workspace with a query from the current activity."""
        att = torch.einsum("btd,btsd->bts", self.w_rq(q), self.n_s(s)) / math.sqrt(self.ds)
        return torch.einsum("bts,btsd->btd", att.softmax(-1), s)

    def write(self, s, c):
        """Each slot gets its own value and its own gate from the (normalised) pass result c; a convex
        update keeps |s| <= 1. (A shared value per pass made all slots converge to the same content.)"""
        c = self.n_c(c)
        val = torch.tanh(self.w_wv(c)).view(*c.shape[:-1], self.slots, self.ds)
        z = torch.sigmoid(self.w_wg(c))[..., None]
        return (1 - z) * s + z * val

    def side(self, s):
        if self.slots:
            return self.n_s(s.mean(-2))        # the router sees a summary of the workspace
        return self.n_s(s) if self.state else None

    # ---------------------------------------------------------------- one pass
    def forward(self, x, s, e, forbid=None, last=False, wave=None, ko=False, group=None, inject=None, pmem=None):
        """wave: a dense (tokens, E) term, or (prev_idx, fatigue) built inside the router (embedding_bag).
        group: routed head group computed on this pass (None = all heads).
        inject: optional (B, T, d) added to the core input u, which the membranes scan over the words (so later
        words can read it) - the thinking concept's write-back (plan_v2/09).
        pmem: optional private expert memory read, called inside the routed experts (memory.py)."""
        wave_t = None
        if isinstance(wave, tuple):
            g, b = F.softplus(self.wave_gamma), F.softplus(self.wave_beta)
            wave, wave_t = None, (wave[0], wave[1], self.router_conn, g, b)
        if self.slots and ko:  # knock-out probe: the workspace is reset to its starting state s_0
            s = self.init_state(e)
        if self.slots:
            r = self.read(s, x)
            u = self.n_u(x + self.n_e(e) + self.w_ro(r) + (inject if inject is not None else 0))
        else:
            u = self.n_u(x + self.n_e(e) + (self.w_s(self.n_s(s)) if self.state else 0))
        m = self.mix(self.n_t(u)) if group is None else self.mix(self.n_t(u), group=group)
        h = u + m
        o = self.mlp(self.n_m(h), forbid, self.side(s) if (self.state or self.slots) else None, extra=wave,
                     wave=wave_t, pmem=pmem)
        if self.slots:
            s = self.write(s, m + o)
            ws = self.w_xo(self.n_s(s).flatten(-2))
            if self.ws_direct:
                lam = torch.sigmoid(self.ws_mix)
                x = self.n_x(x + self.alpha * ((1 - lam) * (m + o) + lam * ws))
            else:  # strict: think in s, speak in x only through the workspace
                x = self.n_x(x + self.alpha * ws)
        else:
            x = self.n_x(x + self.alpha * (m + o))
            if self.state and not last:
                if self.state == "reset":
                    s = self.init_state(e)
                zc = torch.cat([h, o, s], -1)
                z = torch.sigmoid(self.w_z(zc))
                s = (1 - z) * s + z * torch.tanh(self.w_c(zc))
        return x, s


class LoopBrain(SpikingLM):
    def __init__(self, vocab, d=384, d_s=96, heads=6, head_dim=64, moe=None, spike="analog", chunk=64,
                 k_mean=6, k_min=2, k_max=10, bptt=6, state="carry", refractory=1, slots=0, waves=False,
                 pops=1, ws_direct=False, head_group=0, head_groups=0, sheet_cols=0, ckpt_route=False,
                 deep_sup=0.0, halt=False, halt_prior=0.2, halt_beta=0.01, think_w=0.0, think_n=8,
                 think_fb=False, think_target="bag", think_use=False, readout_frac=1.0, concepts=None,
                 concept_weights=None, concept_inject_rank=64, concept_eot=0, mem_shared=0, mem_heads=3, mem_topm=8,
                 mem_topk=6, mem_tau=0.03, mem_read0=1, mem_private=0, mem_dv=256, mem_addr_sg=1,
                 mem_margin=(0.01, 0.01), mem_ema=0.999, mem_max_experts=0, **_):
        nn.Module.__init__(self)
        moe = dict(moe or dict(n_exp=384, k=6, hidden=192, shared=384, capacity=0))
        if waves and not sheet_cols:  # fixed expert sheet, set by the starting pool (kept when it grows)
            sheet_cols = math.ceil(moe["n_exp"] / 16)
        self.cfg = dict(arch="loop", vocab=vocab, d=d, d_s=d_s, heads=heads, head_dim=head_dim, moe=moe,
                        spike=spike, chunk=chunk, k_mean=k_mean, k_min=k_min, k_max=k_max, bptt=bptt,
                        state=state, refractory=refractory, slots=slots, waves=waves, pops=pops, ws_direct=ws_direct,
                        head_group=head_group, head_groups=head_groups, sheet_cols=sheet_cols, ckpt_route=ckpt_route,
                        deep_sup=deep_sup, halt=halt, halt_prior=halt_prior, halt_beta=halt_beta,
                        think_w=think_w, think_n=think_n, think_fb=think_fb, think_target=think_target,
                        think_use=think_use, readout_frac=readout_frac,
                        layers=1, mlp=0, mixer="ssd", concept=None)
        self.has_state = bool(state) or bool(slots)
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Core(d, d_s, heads, head_dim, moe, spike, chunk, state, slots, waves, pops,
                                          ws_direct, head_group, head_groups, sheet_cols)])
        self.blocks[0].mlp.ckpt_route = ckpt_route
        self.normf = RMSNorm(d)
        self.heads = nn.ModuleDict({"lm": nn.Linear(d, vocab, bias=False)})   # the decoder (swappable)
        # "loop until sure" (PonderNet-style): after every pass a small head reads the thought x (normalised),
        # how much the prediction changed since the previous pass (KL), its entropy and the pass index, and
        # outputs the probability of stopping now. Its inputs are detached, so it never changes the brain.
        # thinking-concept head (plan_v2/08): on the WORKSPACE (internal thought, never decoded), predicts the bag
        # of the next think_n tokens - the direction of the thought. Its vagueness (entropy) and how much it
        # still moves between passes are the brain's uncertainty: fed to the stop head and (think_fb) written
        # back into the workspace so the brain can act on it.
        # think_target "latent" (plan_v2/09): instead of the bag of next words, predict the LATENT of the next
        # think_n words - the brain's own final representation of them (stop-gradient) - as a mean and a
        # log-variance; the variance is the thought's uncertainty. think_use: the predicted concept is written back
        # (zero-initialised) into the core input of the next pass, which the membranes scan over the words, so
        # later words can use it; its only training signal beyond the target is their loss (usefulness).
        # --concepts (plan_v2/12): the hierarchical concept heads replace the single thinking head (built at the end)
        self.think = bool(slots) and (think_w > 0 or think_fb or think_use) and not concepts
        self.think_latent = self.think and think_target == "latent"
        if not self.think:
            self.think_head = None
        elif self.think_latent:
            self.think_head = nn.Sequential(nn.Linear(slots * 64, 256), nn.GELU(), nn.Linear(256, 2 * d))
        else:
            self.think_head = nn.Sequential(nn.Linear(slots * 64, 128), nn.GELU(), nn.Linear(128, vocab))
        n_feat = d + 3 + (2 if self.think else 0)
        self.halt_head = nn.Sequential(nn.Linear(n_feat, 64), nn.GELU(), nn.Linear(64, 1)) if halt else None
        self.think_fb = nn.Linear(2, 64) if (self.think and think_fb) else None
        self.think_in = nn.Linear(d, d, bias=False) if (self.think_latent and think_use) else None
        self.apply(self._init)
        for lin in (self.think_fb, self.think_in):  # start silent: nothing changes until the brain learns to use it
            if lin is not None:
                nn.init.zeros_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
        core = self.blocks[0]
        mixers = list(core.mix.pops) if pops > 1 else [core.mix]
        for lin in [mx.wo for mx in mixers] + ([core.mlp.shared.down] if core.mlp.shared is not None else []):
            nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * k_mean))
        if state and not slots:
            nn.init.constant_(core.w_z.bias, -1.0)
            nn.init.zeros_(core.w_c.bias)
            nn.init.zeros_(core.w_s.weight)
        if slots:
            nn.init.constant_(core.w_wg.bias, -1.0)
        self.concept_loss = None
        self.aux_x = None
        self.K = k_mean
        self.diag = {}
        self.knockout = None      # None | "last" | "all": zero the workspace on the last / all passes (probes)
        self.trace_tokens = None  # [(b, t), ...]: record per-pass expert choices for these words
        self.trace = None
        # set by the trainer once per optimizer step, so all micro-batches of a step see the same:
        self.aux_k_fixed = None   # pass used for deep supervision (None = sampled inside run())
        self.group_offset = 0     # head-group rotation offset (training only; evaluation always uses 0)
        self.group_passes = [0] * self.blocks[0].n_groups  # training passes per routed head group (diagnostics)
        self._targets = None      # set by forward() when per-pass losses are needed (deep supervision / halting)
        self.loop_extra = None    # deep-supervision + halting loss of the last training forward
        self.halt_stats = None
        self.think_u, self.think_last, self.think_stats = [], None, None
        self.think_zero = False   # diagnostic: switch the concept write-back off (usefulness = loss it saves)
        # memory: the training read-out's (B, T, V) fp32 log-probs are recomputed in backward, not kept
        # (one copy per backprop pass); False = the plain path (tests compare the two)
        self.ckpt_readout = True
        self._halt_q = {}         # (passes, device) -> halting prior (built once: a list -> GPU copy is a host sync)
        # memory system (plan_v2/12, memory.py): all of it is built here, AFTER every existing module, so the existing
        # modules' initialisation and the random stream are exactly those of a model without it
        self.hier, self.mem = None, None
        self.probe_offset = None  # memory probes: an evaluation-time head-group offset (None = 0, the usual)
        self.mem_ckpt = True      # activation-checkpoint the per-pass concept / memory block
        if concepts:
            assert slots, "--concepts needs the workspace (--slots)"
            self._build_memory(d, slots, moe, halt, think_fb, think_use, concepts, concept_weights,
                               concept_inject_rank, concept_eot, mem_shared, mem_heads, mem_topm, mem_topk, mem_tau,
                               mem_read0, mem_private, mem_dv, mem_addr_sg, mem_margin, mem_ema, mem_max_experts)
        else:
            assert not (mem_shared or mem_private), "the memory needs --concepts (its queries come from the levels)"

    def _build_memory(self, d, slots, moe, halt, think_fb, think_use, concepts, concept_weights, rank, eot,
                      mem_shared, mem_heads, mem_topm, mem_topk, mem_tau, mem_read0, mem_private, mem_dv,
                      mem_addr_sg, mem_margin, mem_ema, mem_max_experts):
        concepts = [int(n) for n in concepts]
        L = len(concepts)
        weights = [float(w) for w in (concept_weights or [1.0, 1.0, 0.5, 0.5][:L] + [0.5] * max(0, L - 4))]
        assert len(weights) == L, "one concept weight per level"
        n_exp = moe["n_exp"] if not isinstance(moe.get("hidden"), (list, tuple)) else len(moe["hidden"])
        emax = max(int(mem_max_experts or 0), n_exp)
        self.cfg.update(concepts=concepts, concept_weights=weights, concept_inject_rank=rank, concept_eot=eot,
                        mem_shared=mem_shared, mem_heads=mem_heads, mem_topm=mem_topm, mem_topk=mem_topk,
                        mem_tau=mem_tau, mem_read0=mem_read0, mem_private=mem_private, mem_dv=mem_dv,
                        mem_addr_sg=mem_addr_sg, mem_margin=list(mem_margin), mem_ema=mem_ema, mem_max_experts=emax)
        base = torch.initial_seed()
        with torch.random.fork_rng(devices=[]):  # a dedicated stream: the global one is left exactly where it was
            torch.manual_seed((base + 7919) % (2 ** 63))
            self.hier = ConceptHier(d, slots * 64, L, fb=think_fb, inject=think_use, rank=rank)
            self.hier.apply(self._init)
            if self.hier.fb is not None:  # zero-init: nothing changes until the brain learns to use it
                nn.init.zeros_(self.hier.fb.weight)
                nn.init.zeros_(self.hier.fb.bias)
            if halt:  # the stop head's features gain u (2L numbers); rebuilt here, wider
                self.halt_head = nn.Sequential(nn.Linear(d + 3 + 2 * L, 64), nn.GELU(), nn.Linear(64, 1))
                self.halt_head.apply(self._init)
            if mem_shared or mem_private:
                self.mem = Memory(d, 64, L, H=mem_heads, S=mem_shared, m=mem_topm, km=mem_topk, tau=mem_tau, dv=mem_dv,
                                  R=mem_private, n_exp=emax, n_live=n_exp, read0=mem_read0, addr_sg=mem_addr_sg,
                                  margin=mem_margin, ema=mem_ema, seed=base)
                if mem_private:
                    self.blocks[0].mem_hook = self._mem_grow

    def _mem_grow(self, src, E0):
        self.mem.on_grow(src, E0, split=self.blocks[0].mlp.birth_mode == "split")

    def sample_aux_k(self, K):
        """Deep-supervision pass, the same distribution as inside run()."""
        return int(torch.randint(max(0, K - self.cfg["bptt"]), K - 1, ()).item()) if K >= 2 else -1

    def sample_group_offset(self):
        G = self.blocks[0].n_groups
        return int(torch.randint(G, ()).item()) if G else 0

    def pass_group(self, k):
        """Routed head group of pass k: 1 + (k + offset) % G. Training draws a random offset per optimizer
        step so every group sees every pass position (otherwise with K = 2 only groups 1-2 would ever
        train); evaluation uses offset 0, deterministic (K=6, G=5 -> groups 1,2,3,4,5,1)."""
        G = self.blocks[0].n_groups
        if not G:
            return None
        return 1 + (k + (self.group_offset if self.training else (self.probe_offset or 0))) % G

    @torch.no_grad()
    def head_group_grad_norms(self):
        """Gradient norm of each head group's output (wo) columns, from the last backward (diagnostics)."""
        mix = self.blocks[0].mix
        if not getattr(mix, "n_groups", 0) or mix.wo.weight.grad is None:
            return None
        g = mix.wo.weight.grad.float().view(mix.wo.weight.size(0), mix.h, mix.p)
        return [g[:, mix.hgroup == j].norm().item() for j in range(mix.n_groups + 1)]

    @torch.no_grad()
    def grow_heads(self, n, ramp=False):
        self.blocks[0].mix.grow_heads(n, ramp=ramp)
        self.cfg["heads"] = self.blocks[0].mix.h  # an int: LoopBrain(**cfg) must rebuild from a checkpoint

    @torch.no_grad()
    def grow_experts(self, n):
        src = self.blocks[0].grow_experts(n)
        self.cfg["moe"] = dict(self.cfg["moe"], n_exp=self.blocks[0].mlp.E)
        return src

    def aux_loss(self):
        aux, extra = getattr(self, "loop_aux", None), self.loop_extra
        if extra is None:
            return aux
        return extra if aux is None else aux + extra

    def forward(self, idx, targets=None, head="lm"):
        per_pass = self.training and targets is not None and (self.cfg["deep_sup"] or self.cfg["halt"]
                                                               or self.cfg.get("think_w", 0))
        # (the latent target needs the final pass, so it is computed in run() too)
        self._targets = targets if per_pass else None
        try:
            return super().forward(idx, targets, head)
        finally:
            self._targets = None

    def think_readout(self, s, prev_cl):
        """Thinking-concept prediction from the workspace: log-probs over the vocabulary (B, T, V) of the bag of
        the next words, and its uncertainty u = [normalised entropy, KL to the previous pass] (detached)."""
        cl = F.log_softmax(self.think_head(self.blocks[0].n_s(s).flatten(-2)).float(), -1)
        with torch.no_grad():
            p = cl.exp()
            ent = -(p * cl).sum(-1) / math.log(cl.size(-1))
            kl = (p * (cl - prev_cl)).sum(-1).clamp(min=0) if prev_cl is not None else torch.zeros_like(ent)
            u = torch.stack([ent, torch.log1p(kl)], -1)
        return cl, u

    def think_latent_readout(self, s, prev_mu):
        """Latent thinking concept: mean mu and log-variance lv (B, T, d) of the predicted latent of the next words,
        and its uncertainty u = [predicted variance (mean lv, scaled to ~[0, 1]), how much mu still moves]."""
        out = self.think_head(self.blocks[0].n_s(s).flatten(-2)).float()
        mu, lv = out.chunk(2, -1)
        lv = lv.clamp(-6.0, 4.0)
        with torch.no_grad():
            var_u = (lv.mean(-1) + 6.0) / 10.0
            move = ((mu - prev_mu).norm(dim=-1) / mu.norm(dim=-1).clamp(min=1e-6) if prev_mu is not None
                    else torch.zeros_like(var_u))
            u = torch.stack([var_u, torch.log1p(move)], -1)
        return mu, lv, u

    def _hier_step(self, s, read, address, k=0):
        """Concept level heads on the workspace s (plan_v2/12 section 1) and, with address, the memory: indices
        (EMA translator, no gradient, fp32), online weights, the shared read, gates and W_o (section 3). With gradient
        it is ONE activation-checkpointed block; the indices are chosen on its first run (no_grad) and reused when
        backward recomputes it. Returns (mus, nus, M, q1, margin) and the selection (or None)."""
        hier, mem, core = self.hier, self.mem, self.blocks[0]
        address = address and mem is not None
        collect = self.training and self._targets is not None and torch.is_grad_enabled()
        cache = {}

        def block(s):
            ns = core.n_s(s)
            mus, nus = hier(ns.flatten(-2))
            if not address:
                return mus, nus, None, None, None
            pbar = ns.mean(-2)
            a, a_hat = mem.prep(mus, pbar)
            if "sel" not in cache:
                cache["sel"] = mem.select(a.detach(), a_hat.detach(), collect)
            sel = cache["sel"]
            # after the freeze (with --mem-addr-sg) the addressing input is sg(a): w carries no gradient at all
            with torch.set_grad_enabled(torch.is_grad_enabled() and not (mem.is_frozen and mem.addr_sg)):
                on = mem.online(a_hat.detach() if (mem.is_frozen and mem.addr_sg) else a_hat, sel)
            M = g = None
            if read and mem.S and not mem.ko_shared:
                M, g = mem.read_shared(on["w"], sel, pbar)
            margin = None
            if collect and mem.S and mem.lam_m and not mem.is_frozen:  # separate pass on sg(a): keys only
                margin = mem.margin_loss(a_hat.detach(), sel)
            if mem.record is not None and "diag" not in cache:
                pos = mem.record["pos"]
                with torch.no_grad():
                    d = dict(k=k, a=a[:, pos].float().cpu())
                    if mem.S:
                        perm = lambda t: t[:, :, pos].permute(2, 0, 1, *range(3, t.dim())).float().cpu()
                        d.update(rows=sel["rows"][:, :, pos].permute(2, 0, 1, 3).cpu(), w=perm(on["w"]),
                                 g=g[pos].float().cpu() if g is not None else None)
                        d.update({n: perm(v) for n, v in zip(("gap", "gap1", "gap2"), mem.gaps(a_hat, sel))})
                    cache["diag"] = d
            return mus, nus, M, on["q1"], margin

        if torch.is_grad_enabled() and self.mem_ckpt:
            out = torch.utils.checkpoint.checkpoint(block, s, use_reentrant=False)
        else:
            out = block(s)
        if "diag" in cache:
            mem.record["reads"].append(cache["diag"])
        return out, cache.get("sel")

    def _private_fn(self, q1, sel):
        """The next pass's private read (called inside the routed experts), or None."""
        mem = self.mem
        if mem is None or not mem.R or mem.ko_private or q1 is None:
            return None
        qb1 = sel["qb1"]
        return lambda *a: mem.private(q1, qb1, *a)

    def _slot_write(self, s, M):
        """Memory output into the LAST workspace slot (slot 15 of 16), out of place: s[..., -1, :] += W_ms M."""
        w = self.mem.w_ms(M.to(self.mem.w_ms.weight.dtype)).to(s.dtype)
        return torch.cat([s[..., :-1, :], s[..., -1:, :] + w[..., None, :]], -2)

    def think_target(self, x_final):
        """z*_t = mean of the brain's own final representation normf(x_K) over the next n words (t+1 .. t+n),
        stop-gradient; a window cut by the sequence end uses the words that exist. Returns (z*, mask) where mask
        is False for the last position (nothing after it)."""
        n = self.cfg["think_n"]
        z = self.normf(x_final).detach().float()                         # (B, T, d)
        B, T, D = z.shape
        cs = torch.cat([z.new_zeros(B, 1, D), z.cumsum(1)], 1)           # cs[i] = sum of z[:i]
        t = torch.arange(T, device=z.device)
        lo, hi = (t + 1).clamp(max=T), (t + 1 + n).clamp(max=T)
        cnt = (hi - lo).clamp(min=1).to(z.dtype)
        target = (cs[:, hi] - cs[:, lo]) / cnt[None, :, None]
        return target, (hi > lo)

    @staticmethod
    def think_latent_loss(mu, lv, target, mask, n_valid=None):
        """Gaussian NLL of the target under the predicted mean / log-variance, plus a VICReg-style variance hinge
        on the predictions (each dimension should vary across words at least as much as the target does), so the
        prediction cannot collapse to a constant. n_valid: the mask is its first n_valid positions (known on the
        host: a slice instead of mask.nonzero(), which is a host sync)."""
        nll = 0.5 * (lv + (target - mu) ** 2 / lv.exp())
        m = mask[None, :, None].to(nll.dtype).expand_as(nll)
        nll = (nll * m).sum() / m.sum().clamp(min=1)
        sel = mask.nonzero().squeeze(1) if n_valid is None else slice(0, n_valid)
        mu_s, tg_s = mu[:, sel].flatten(0, 1), target[:, sel].flatten(0, 1)
        hinge = F.relu(tg_s.std(0).detach() - mu_s.std(0)).mean() if mu_s.size(0) > 1 else mu.sum() * 0
        return nll + hinge

    def think_loss(self, cl, targets):
        """Bag-of-next-words cross-entropy: mean over the next think_n tokens (targets[t] is the token after t,
        so the bag of position t is targets[t .. t+n-1]); positions past the sequence end are masked."""
        n, T = self.cfg["think_n"], targets.size(1)
        tot, cnt = 0.0, 0
        for j in range(n):
            if j >= T:
                break
            tot = tot - cl[:, :T - j].gather(-1, targets[:, j:, None]).sum()
            cnt += targets[:, j:].numel()
        return tot / cnt

    def readout_lp(self, x):
        return F.log_softmax(self.heads["lm"](self.normf(x)).float(), -1)

    def _readout_ce(self, x, tgt):
        lp = self.readout_lp(x)
        return -lp.gather(-1, tgt[..., None]).squeeze(-1), lp.detach()

    def pass_readout(self, x, prev_lp, k, cu=None):
        """Decoder read-out after a pass: log-probs (B, T, V) and the halting features (detached)."""
        lp = self.readout_lp(x)
        return lp, self.halt_features(x, lp, prev_lp, k, cu)

    def pass_readout_ce(self, x, tgt, prev_lp, k, cu=None):
        """Training read-out: per-word CE (with gradient), the detached log-probs and the halting features. With
        ckpt_readout the (..., V) fp32 log-probs are recomputed in backward instead of being kept for it."""
        if self.ckpt_readout and torch.is_grad_enabled():
            ce, lp = torch.utils.checkpoint.checkpoint(self._readout_ce, x, tgt, use_reentrant=False)
        else:
            ce, lp = self._readout_ce(x, tgt)
        return ce, lp, self.halt_features(x, lp, prev_lp, k, cu)

    def halt_features(self, x, lp, prev_lp, k, cu=None):
        """Halting features of a read-out (detached): thought, KL to the previous pass, entropy, pass index."""
        with torch.no_grad():
            p = lp.exp()
            ent = -(p * lp).sum(-1) / math.log(lp.size(-1))
            kl = (p * (lp - prev_lp)).sum(-1) if prev_lp is not None else torch.zeros_like(ent)
            feat = [self.normf(x).detach().float(), torch.log1p(kl.clamp(min=0))[..., None],
                    ent[..., None], torch.full_like(ent, k / 16.0)[..., None]]
            if self.think:  # the thought's own uncertainty (zeros if not computed)
                feat.append(cu if cu is not None else torch.zeros_like(feat[0][..., :2]))
            elif self.hier is not None:  # concept hierarchy: 2 numbers per level
                feat.append(cu if cu is not None else torch.zeros_like(feat[0][..., :2 * self.hier.L]))
            return torch.cat(feat, -1)

    def halt_lambda(self, feat):
        return torch.sigmoid(self.halt_head(feat.to(self.halt_head[0].weight.dtype)).squeeze(-1).float())

    @staticmethod
    def halt_distribution(lams):
        """lams: list of (B, T) stop probabilities for passes 0..K-1; the last pass takes the remaining mass."""
        p, alive = [], torch.ones_like(lams[0])
        for i, lam in enumerate(lams):
            lam = torch.ones_like(lam) if i == len(lams) - 1 else lam
            p.append(alive * lam)
            alive = alive * (1 - lam)
        return torch.stack(p, 0)                                          # (K, B, T), sums to 1

    def halt_prior_q(self, K, device):
        """Truncated geometric prior over K passes, cached per (K, device)."""
        key = (K, str(device))
        if key not in self._halt_q:
            lp_ = self.cfg["halt_prior"]
            q = torch.tensor([lp_ * (1 - lp_) ** i for i in range(K)], device=device)
            q[-1] = 1 - q[:-1].sum()
            self._halt_q[key] = q
        return self._halt_q[key]

    @torch.no_grad()
    def halt_eval(self, idx, targets, K_max=16, threshold=0.5):
        """Loop until sure: every word stops at the first pass where its cumulative stop probability reaches
        threshold (ceiling K_max). Loss of each word's answer at its own exit pass (read from one K_max run, as
        if the stopped words had frozen; the uncertainty probe showed freezing does not change the others) and
        the average number of passes, plus the fixed-K losses of the same run for comparison."""
        was = self.training
        self.eval()
        outs = self.run(idx, K_max, every_tick=True, raw=True)
        prev, lams, ces = None, [], []
        for k, x in enumerate(outs):
            lp, feat = self.pass_readout(x, prev, k, self.think_u[k] if (self.think or self.hier is not None) else None)
            lams.append(self.halt_lambda(feat))
            ces.append(-lp.gather(-1, targets[..., None]).squeeze(-1))
            prev = lp
        P = self.halt_distribution(lams)
        stop = (P.cumsum(0) >= threshold).float().argmax(0)              # first pass over the threshold
        ce = torch.stack(ces, 0)
        exit_loss = ce.gather(0, stop[None]).squeeze(0)
        self.train(was)
        steps = torch.arange(1, K_max + 1, device=P.device)[:, None, None]
        return dict(halt_loss=exit_loss.mean().item(), halt_passes=(stop.float() + 1).mean().item(),
                    halt_expected_passes=(P * steps).sum(0).mean().item(),
                    fixed_loss={k + 1: ce[k].mean().item() for k in range(K_max)},
                    passes_hist=torch.bincount(stop.flatten() + 1, minlength=K_max + 1)[1:].tolist())

    @torch.no_grad()
    def think_diag(self, idx, targets, K_max=16, kmax_eval=8):
        """Loop-report diagnostic: per word and pass k = 1..kmax_eval, the label "more passes will help by > 0.05
        nats" (LM loss at k minus the best later pass) and the AUC of each uncertainty signal for it: the
        concept head's entropy and change (thinking) and the LM prediction's change (output)."""
        was = self.training
        self.eval()
        outs = self.run(idx, K_max, every_tick=True, raw=True)
        cus = self.think_u
        prev, ces, kls = None, [], []
        for k, x in enumerate(outs):
            lp, _ = self.pass_readout(x, prev, k)
            ces.append(-lp.gather(-1, targets[..., None]).squeeze(-1))
            kls.append((lp.exp() * (lp - prev)).sum(-1) if prev is not None else torch.zeros_like(ces[-1]))
            prev = lp
        ce = torch.stack(ces, 0)
        best_later = torch.stack([ce[k + 1:].min(0).values for k in range(K_max - 1)], 0)
        gain = ce[:-1] - best_later                                               # (K-1, B, T)
        lab = (gain[1:kmax_eval] > 0.05).flatten()                                # passes 2..kmax_eval
        sig = dict(concept_entropy=torch.stack([c[..., 0] for c in cus[1:kmax_eval]]),
                   concept_change=torch.stack([c[..., 1] for c in cus[1:kmax_eval]]),
                   output_change=torch.stack(kls[1:kmax_eval]))

        def auc(score, label):
            r = torch.empty_like(score)
            r[score.argsort()] = torch.arange(len(score), dtype=score.dtype, device=score.device)
            pos = label.sum()
            neg = len(label) - pos
            if pos == 0 or neg == 0:  # undefined (e.g. nothing helped yet)
                return float("nan")
            return ((r[label].sum() - pos * (pos - 1) / 2) / (pos * neg)).item()

        out = {f"auc_{n}": auc(v.flatten().double(), lab) for n, v in sig.items()}
        out["helped_share"] = lab.float().mean().item()
        out["concept_entropy_by_pass"] = [c[..., 0].mean().item() for c in cus[:kmax_eval]]
        self.train(was)
        return out

    def sample_K(self):
        """Uniform in [k_min, k_max]; 5% of steps think long (12-16 passes) so longer thinking stays stable."""
        if torch.rand(()).item() < 0.05:
            return int(torch.randint(12, 17, ()).item())
        return int(torch.randint(self.cfg["k_min"], self.cfg["k_max"] + 1, ()).item())

    def run(self, idx, K=None, every_tick=False, raw=False):
        core = self.blocks[0]
        K = K or self.K
        e = self.emb(idx)
        x = e
        s = core.init_state(e) if self.has_state else None
        bptt = self.cfg["bptt"]
        forbid, seen, hist, outs, aux_terms = None, None, [], [], []
        fatigue, prev, prev_idx = None, None, None
        self.aux_x = None
        tgt = self._targets
        ds_w, halt = self.cfg.get("deep_sup", 0.0), self.cfg.get("halt", False)
        ces, lams, prev_lp = [], [], None
        c_w, c_losses, prev_cl, cu = self.cfg.get("think_w", 0.0), [], None, None
        self.think_u = []
        lat_preds, inject = [], None                                          # (mu, lv) of backprop passes
        sub, tgt_r = None, tgt
        frac = self.cfg.get("readout_frac", 1.0)
        if tgt is not None and frac < 1:  # the per-pass read-outs see a random subset of the words (the same on
            n = max(1, round(frac * tgt.numel()))                             # every pass of this forward)
            sub = torch.randperm(tgt.numel(), device=tgt.device)[:n].sort().values
            tgt_r = tgt.flatten()[sub]
        if tgt is not None and ds_w:  # deep supervision on every pass replaces the single sampled one
            aux_k = -1
        elif self.training and self.aux_k_fixed is not None:
            aux_k = self.aux_k_fixed
        else:
            aux_k = int(torch.randint(max(0, K - bptt), K - 1, ()).item()) if (self.training and K >= 2) else -1
        if self.trace_tokens:
            self.trace = []
        G = core.n_groups
        gpass = [0] * G
        hier, mem = self.hier, self.mem
        hier_preds, margins, prev_mu, pm = [], [], None, None   # concept / memory (plan_v2/12); pm: private read
        if mem is not None:
            mem.begin_forward(self.training and tgt is not None)
            if (mem.S and mem.read0) or mem.R:  # the read before pass 1, from s_0 (token-keyed; --mem-read0)
                with torch.set_grad_enabled(torch.is_grad_enabled() and K - bptt <= 0):
                    (_, _, M, q1, mg), sel = self._hier_step(s, read=bool(mem.read0), address=True, k=-1)
                    if mg is not None:
                        margins.append(mg)
                    if M is not None:
                        M = M.view(x.shape)
                        inject = M.to(x.dtype)
                        s = self._slot_write(s, M)
                    pm = self._private_fn(q1, sel)
        for k in range(K):
            ko = self.knockout == "all" or (self.knockout == "last" and k == K - 1)
            group = self.pass_group(k)
            if group is not None:
                gpass[group - 1] += 1
            with torch.set_grad_enabled(torch.is_grad_enabled() and k >= K - bptt):
                # wave = gamma * C . used_prev - beta * fatigue, built in the router from the previous
                # pass's expert indices (embedding_bag: exact, no dense E x E matmul)
                wave = (prev_idx, fatigue) if (core.waves and prev is not None) else None
                x, s = core(x, s, e, None if core.waves else forbid, last=(k == K - 1), wave=wave, ko=ko,
                            group=group, inject=inject, pmem=pm)
                if self.think_latent:  # the predicted latent of the next words, with its uncertainty
                    mu, lv, cu = self.think_latent_readout(s, prev_cl)
                    if tgt is not None and c_w and k >= K - bptt:
                        lat_preds.append((mu, lv))
                    prev_cl = mu.detach()
                    self.think_u.append(cu)
                    if self.think_fb is not None:
                        s = s + self.think_fb(cu.to(s.dtype))[..., None, :]
                    if self.think_in is not None and not self.think_zero:  # usefulness: next pass's membranes
                        inject = self.think_in(mu.to(x.dtype))                         # carry it to later words
                elif self.think:  # the thought's direction and uncertainty (after the pass wrote the workspace)
                    cl, cu = self.think_readout(s, prev_cl)
                    if tgt is not None and c_w and k >= K - bptt:
                        c_losses.append(self.think_loss(cl, tgt))
                    prev_cl = cl.detach()
                    self.think_u.append(cu)
                    if self.think_fb is not None:  # tell the workspace how sure the thought is
                        s = s + self.think_fb(cu.to(s.dtype))[..., None, :]
                    del cl
                elif hier is not None:  # concept hierarchy (+ the memory read that feeds the next pass)
                    if mem is not None and mem.record is not None:
                        mem.record["calls"].append(dict(
                            experts=core.mlp.last_idx[mem.record["pos"]].sort(-1).values.cpu(),
                            prows=mem.__dict__.pop("_prows", None)))
                    addr = mem is not None and k < K - 1
                    (mus, nus, M, q1, mg), sel = self._hier_step(s, read=addr, address=addr, k=k)
                    cu = hier.uncertainty(mus, nus, prev_mu)
                    if tgt is not None and c_w and k >= K - bptt:
                        hier_preds.append((mus, nus))
                    if mg is not None:
                        margins.append(mg)
                    prev_mu = [mu.detach() for mu in mus]
                    self.think_u.append(cu)
                    if hier.fb is not None:  # W_fb u (zero-init) into every workspace slot
                        s = s + hier.fb(cu.to(s.dtype))[..., None, :]
                    inject = None
                    if hier.has_inject and not self.think_zero:  # usefulness: sum_l B_l A_l mu_l
                        inject = hier.write_back(mus).to(x.dtype)
                    if M is not None:  # memory output: next pass's core input and the last workspace slot
                        M = M.view(x.shape)
                        inject = M.to(x.dtype) if inject is None else inject + M.to(x.dtype)
                        s = self._slot_write(s, M)
                    pm = self._private_fn(q1, sel) if addr else None
            if core.mlp.aux is not None and core.mlp.aux.requires_grad:
                aux_terms.append(core.mlp.aux)
            used = torch.zeros(x.shape[0] * x.shape[1], core.mlp.E, dtype=torch.bool, device=x.device)
            used.scatter_(1, core.mlp.last_idx, True)
            seen = used if seen is None else seen | used
            hist.append(used)
            if core.waves:
                prev = used.to(x.dtype if x.is_floating_point() else torch.float32)
                prev_idx = core.mlp.last_idx
                fatigue = prev if fatigue is None else 0.7 * fatigue + prev
            elif self.cfg["refractory"]:
                forbid = used
            if self.trace_tokens:
                T = x.shape[1]
                self.trace.append([core.mlp.last_idx[b * T + t].tolist() for b, t in self.trace_tokens])
            if k == aux_k and torch.is_grad_enabled():
                self.aux_x = self.normf(x)
            if tgt is not None:  # per-pass read-out: deep supervision / halting (gradient only where backprop runs)
                with torch.set_grad_enabled(torch.is_grad_enabled() and k >= K - bptt):
                    xr, cur = x, cu
                    if sub is not None:
                        xr = x.flatten(0, 1)[sub]
                        cur = cu.flatten(0, 1)[sub] if cu is not None else None
                    ce, prev_lp, feat = self.pass_readout_ce(xr, tgt_r, prev_lp, k, cur)  # prev_lp: detached
                    ces.append(ce)
                if halt:  # outside the bptt window too: the head must learn to stop on the EARLY passes (feat is detached)
                    lams.append(self.halt_lambda(feat))
            if every_tick:
                outs.append(x if raw else self.normf(x))
        self.loop_aux = sum(aux_terms) / len(aux_terms) if aux_terms else None
        self.loop_extra = None
        if tgt is not None:
            self.halt_stats = None  # (only training forwards reset it, so an eval forward keeps it for logging)
            extra = 0.0
            if ds_w:  # mean LM loss of the earlier backpropagated passes (the last pass is the main loss)
                early = [c.mean() for k, c in enumerate(ces[:-1]) if k >= K - bptt]
                if early:
                    extra = extra + ds_w * sum(early) / len(early)
            if halt and len(lams) >= 2:
                P = self.halt_distribution(lams)                                     # (K, B, T)
                ce = torch.stack(ces, 0).detach()                                    # the head does not shape the LM
                rec = (P * ce).sum(0).mean()
                col = (-1,) + (1,) * (P.dim() - 1)                                   # (K, B, T) or (K, words)
                q = self.halt_prior_q(len(lams), P.device)
                kl = (P * (P.clamp_min(1e-9).log() - q.log().view(col))).sum(0).mean()
                extra = extra + rec + self.cfg["halt_beta"] * kl
                with torch.no_grad():
                    steps = torch.arange(1, len(lams) + 1, device=P.device).view(col)
                    self.halt_stats = dict(expected_passes=(P * steps).sum(0).mean(), kl=kl.detach())
            if lat_preds:  # latent target: the final pass's own representation of the next words (stop-gradient)
                target, mask = self.think_target(x)
                n_valid = x.size(1) - 1 if self.cfg["think_n"] > 0 else 0       # the mask is this prefix
                for mu, lv in lat_preds:
                    c_losses.append(self.think_latent_loss(mu, lv, target, mask, n_valid))
                with torch.no_grad():
                    mu = lat_preds[-1][0]
                    sel = slice(0, n_valid)
                    self.think_stats = dict(
                        target_std=target[:, sel].flatten(0, 1).std(0).mean(),           # collapse detector
                        pred_std=mu[:, sel].flatten(0, 1).std(0).mean(),
                        cosine=F.cosine_similarity(mu[:, sel], target[:, sel], dim=-1).mean())
            if hier_preds:  # concept hierarchy: targets = windows of the final pass's own representation (sg)
                acc = hier_preds[0][0][0].dtype
                tg = concept_targets(self.normf(x).detach().to(acc), idx, self.cfg["concepts"], self.cfg["concept_eot"])
                wl = self.cfg["concept_weights"]
                for mus, nus in hier_preds:
                    c_losses.append(sum(w * concept_loss(mu, nu, t, v) for w, mu, nu, (t, v) in zip(wl, mus, nus, tg)))
                with torch.no_grad():  # collapse detectors and fit per level (tensors: no host sync)
                    st = {}
                    for l, (mu, (t, v)) in enumerate(zip(hier_preds[-1][0], tg)):
                        vv = v.to(acc)[..., None]
                        nv = vv.sum().clamp(min=1)
                        st[f"l{l + 1}_valid"] = v.to(acc).mean()
                        st[f"l{l + 1}_target_std"] = masked_std(t, vv, nv).mean()
                        st[f"l{l + 1}_pred_std"] = masked_std(mu, vv, nv).mean()
                        st[f"l{l + 1}_cosine"] = (F.cosine_similarity(mu, t, dim=-1) * vv[..., 0]).sum() / nv
                    self.think_stats = st
            if margins:  # sub-key margin loss (infancy), mean over the passes that computed it
                extra = extra + sum(margins) / len(margins)
            if c_losses:
                extra = extra + c_w * sum(c_losses) / len(c_losses)
                self.think_last = c_losses[-1].detach()
            self.loop_extra = extra if torch.is_tensor(extra) else None
        if G and self.training:
            self.group_passes = [a + b for a, b in zip(self.group_passes, gpass)]
        if self.training:  # training-time routing stats as tensors (read only on logging steps: no host sync)
            st = core.mlp.stats
            self.train_route = (seen.float().sum(1).mean(), st[1].float() / st[2])
        with torch.no_grad():
            self.diag = {}
            if G:
                self.diag["head_group_passes"] = gpass                          # this forward
                self.diag["head_group_passes_train"] = list(self.group_passes)  # all training passes so far
            # routing / wave diagnostics: only the evaluation forwards are logged, so training skips them
            # (~10 host syncs and a (tokens x E) @ (E x E) matmul per pass at E = 768)
            if not self.training:
                self.diag["distinct_experts"] = seen.float().sum(1).mean().item()
                kk = core.mlp.k
                for lag in (1, 2, 4):
                    if len(hist) > lag:
                        ov = [(hist[i] & hist[i - lag]).float().sum(1).mean() / kk for i in range(lag, len(hist))]
                        self.diag[f"overlap_lag{lag}"] = torch.stack(ov).mean().item()
                self.diag["overlap_k_k2"] = self.diag.get("overlap_lag2", 0.0)
                if core.waves and len(hist) >= 2:  # how far the activity's centroid moves on the torus sheet
                    rows, cols = core.sheet_rows, core.sheet_cols  # fixed sheet positions (stable under growth)
                    ar = 2 * math.pi * core.pos[:, 0].float() / rows
                    ac = 2 * math.pi * core.pos[:, 1].float() / cols

                    def centroid(u):
                        w = u.float()
                        return torch.atan2(w @ ar.sin(), w @ ar.cos()), torch.atan2(w @ ac.sin(), w @ ac.cos())

                    cs = [centroid(u) for u in hist]
                    steps = []
                    for (r0, c0), (r1, c1) in zip(cs, cs[1:]):
                        dr = torch.remainder(r1 - r0 + math.pi, 2 * math.pi) - math.pi
                        dc = torch.remainder(c1 - c0 + math.pi, 2 * math.pi) - math.pi
                        steps.append(torch.sqrt((dr * rows / (2 * math.pi)) ** 2 + (dc * cols / (2 * math.pi)) ** 2).mean())
                    self.diag["wave_travel_cells"] = torch.stack(steps).mean().item()  # random routing ~7.7
                    # share of next-pass experts that are sheet-neighbours of the previous pass's experts
                    loc = [((hist[i].to(core.adj2.dtype) @ core.adj2 > 0) & hist[i + 1]).float().sum(1).mean() / kk
                           for i in range(len(hist) - 1)]
                    self.diag["wave_neighbour_share"] = torch.stack(loc).mean().item()
                    self.diag["wave_gamma"] = F.softplus(core.wave_gamma).item()
                    self.diag["wave_beta"] = F.softplus(core.wave_beta).item()
        return outs if every_tick else self.normf(x)

    def features(self, idx):
        return self.run(idx)

    @torch.no_grad()
    def tick_losses(self, idx, targets, K):
        return [F.cross_entropy(self.heads["lm"](x).float().flatten(0, 1), targets.flatten()).item()
                for x in self.run(idx, K, every_tick=True)]
