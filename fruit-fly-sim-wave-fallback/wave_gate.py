"""The wave memory's PRE-READ GATE and PER-EXPERT OUTPUT LAYER (spec v2.1, binding brief).  New file: wave_memory.py /
train_lm.py are not touched here; the integration hooks are listed at the end of this docstring.

One module, `GatedReadout` (a submodule of the memory: its state_dict lives under bm.*, i.e. in the memory file), holds
  * the gate        h = SiLU(G1 [W_gx xhat ; f_tok] + b) (16), bhat_p = <Wp[e], h> + <Up, f_pair> + bp[e], bhat_g likewise
  * the layer       out = y + y @ D[e]   (D [S, d, d] fp32, init 0: exactly the identity; NO bias, no activation)
                    o = exp(log_s) * ||xf|| * out, then the 0.5 ||x|| cap with its scale DETACHED
  * its own optimizer (two groups: the gate + log_s lr 1e-3 wd 0; D lr 1e-4 wd 0.01 fused AdamW; clip 1.0 per group;
    optional bf16 moments for D), saved in the module's extra state.
S = Emax + 1 slots (769): rows 0..Emax-1 the routed experts, row Emax (= `shared`) the shared expert.

Per forward (pass `read_pass`, no grad in the decision, detached inputs only, never target-dependent):
  dec = gr.decide(xh, e_k, t_k, C, n_live, step=, call=, f_tok=, f_pair=)      static shapes, no host sync
     pairs = the N * K routed pairs + N shared pairs (t, Emax).  Capacity-aware: inside each expert group the eligible
     pairs are ranked (exploration first, then score); rank >= C_e (C routed, N shared) -> -inf; then a global
     topk(Kp = cap_frac * n_pairs).  dec.slot = the rank = a capacity slot, so zero drops by construction.
     Global: per token score = max bhat_g over its kept pairs with vg; topk(Kg = cap_frac * N); a pair's global hint is
     used only if its token is in that set (gsel_used); a kept pair without vg never triggers the global read.
     dec.p_idx [Kp]: the pair rows to read from the private sheets (dec.read_p masks the unread ones)
     dec.g_idx [Kg]: the token rows to read from the global sheet (dec.g_valid masks the unread ones)
  o = gr.readout(dec, y_p, y_g, xn)    y_p [Kp, d] private readouts of dec.p_idx, y_g [Kg, d] global readouts of
     dec.g_idx (both ALREADY scaled by the confidence c_tok * kappa); xn [N] = ||x|| (the cap and the scale).
     Capture leaves (dL/dy per read pair per sheet) are attached here when training.
  buf, x = gr.inject(dec, o, buf, x, e_k, p_k)     routed pairs into buf[e_k, p_k], shared pairs into x (per token)
  later passes: o2, sh2 = gr.reuse(dec, o, e_k2, t_k2) -> the (t, e) pairs that read at the read pass get the cached
     output, the others exactly 0; inject with gr.inject_reuse(...)
After the main backward (before the optimizer step):
  stats = gr.after_backward(conf_p=None, conf_g=None)   b = -<dL/dy, y>, b~ = clamp(b / EMA-RMS(b), -5, 5),
     Huber(bhat - b~) on read pairs (weight 4 on exploration) + 0.5 BCE of the confirmation heads; backward with
     inputs = the gate parameters only.  Then gr.step() (the module's own optimizer, both groups, clip 1.0 each).
"Write only where read": dec.write_mask() -> (pair rows read per sheet, global tokens read); write_ok(target).
Split: gr.on_grow(src, dst) copies D, Wp, Wg, bp, bg rows.

INTEGRATION (for the agent owning wave_memory.py / train_lm.py):
  wave_memory.CodedWaveMemory.__init__ : self.gr = GatedReadout(d, self.Emax, k_routed=<top-k>, seed=self.seed + 31)
  wave_memory.CodedWaveMemory._step_params : exclude every parameter of self.gr from `params` (e.g. skip names starting
      with "gr."), then call self.gr.step() (its own groups / clip / fused AdamW).
  wave_memory.CodedWaveMemory.__call__ (k == read_pass): instead of _apply_recall's gate:
      dec = self.gr.decide(xf, e_k, t_k, C, self.n_live, step=self.n_fwd, call=k, training=self._live,
                           f_tok=<[N,F] cheap token feats: e.g. log ||xf||, pass id>, f_pair=<[n_pairs,1] router prob>)
      private read: self.priv.read(codes.take(t_all[dec.p_idx]), part=dec.e)        (only Kp rows: the open set)
      global read : Rg rows for codes.take(dec.g_idx)   (only Kg rows)  - move the store read AFTER decide so the
                    unopened tokens are never read; scale both by c_tok * kappa (the confidence) before readout.
      o = self.gr.readout(dec, y_p, y_g, xn);  buf, x = self.gr.inject(dec, o, buf, x, e_k, p_k); cache (dec, o).
    k > read_pass: o2, sh2 = self.gr.reuse(dec, o, e_k, t_k); buf, x = self.gr.inject_reuse(o2, sh2, buf, x, e_k, p_k)
    before write_start (or the memory off): do not call any gr method (no RNG, no compute: "off == none").
  wave_memory.CodedWaveMemory.after_backward : stats = self.gr.after_backward(conf_p, conf_g) (the confirmed-recall
      flags per read pair, optional); the writes: only rows of dec.write_mask(), skipping gr.write_ok(target) == False.
  wave_memory.CodedWaveMemory.on_grow : self.gr.on_grow(src, dst)
  wave_memory.CodedWaveMemory.log_record : rec.update(self.gr.log_record())
  train_lm.py : nothing beyond the existing bm.* exclusion from the brain optimizer and its clip.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = float("-inf")


# ---------------------------------------------------------------------------------------------------- the layer
class _ExpertD(torch.autograd.Function):
    """out = Y + Y @ D[r0:r0+G] (blocked over experts; D fp32, compute in Y.dtype).  Backward writes dD straight into
    D.grad block by block (fp32 accumulate): no [S, d, d] transient, no autograd grad for D."""

    @staticmethod
    def forward(ctx, Y, D, r0, block):
        G = Y.size(0)
        out = torch.empty_like(Y)
        for b in range(0, G, block):
            e = min(G, b + block)
            out[b:e] = torch.baddbmm(Y[b:e], Y[b:e], D[r0 + b:r0 + e].to(Y.dtype))
        ctx.save_for_backward(Y)
        ctx.D, ctx.r0, ctx.block = D, r0, block
        return out

    @staticmethod
    def backward(ctx, gO):
        (Y,), D, r0, block = ctx.saved_tensors, ctx.D, ctx.r0, ctx.block
        G = Y.size(0)
        gY = torch.empty_like(gO)
        if D.grad is None:
            D.grad = torch.zeros_like(D)
        for b in range(0, G, block):
            e = min(G, b + block)
            gY[b:e] = torch.baddbmm(gO[b:e], gO[b:e], D[r0 + b:r0 + e].to(gO.dtype).transpose(1, 2))
            D.grad[r0 + b:r0 + e].baddbmm_(Y[b:e].float().transpose(1, 2), gO[b:e].float())
        return gY, None, None, None


def expert_d(Y, D, r0, block=64):
    return _ExpertD.apply(Y, D, int(r0), int(block))


# ---------------------------------------------------------------------------------------------------- decision
class GateDecision:
    """One forward's decision.  All tensors have static shapes ([Kp] pair rows, [Kg] token rows, [N] tokens)."""
    __slots__ = ("N", "n_routed", "shared", "p_idx", "e", "t", "slot", "valid", "read_p", "read_g", "ex_p", "ex_g",
                 "g_idx", "g_valid", "gsel_used", "vp_share", "inputs", "probes", "y_saved", "C", "n_live", "training")

    def write_mask(self):
        """-> (read_p [Kp], read_g [Kp], gsel_used [N]): write only where read (private per pair row, global per token)."""
        return self.read_p, self.read_g, self.gsel_used

    def gathered_rows(self):
        """Rows the stores are asked for (static): Kp private + Kg global (unread ones are masked to exact 0)."""
        return int(self.p_idx.numel()), int(self.g_idx.numel())


def _rank_in_group(group, prio, n_groups):
    """rank of each element inside its group by prio descending (stable; no host sync)."""
    P = group.numel()
    o1 = torch.argsort(-prio, stable=True)
    o2 = torch.argsort(group[o1], stable=True)
    order = o1[o2]
    counts = torch.bincount(group, minlength=n_groups)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.empty_like(group)
    rank[order] = torch.arange(P, device=group.device) - starts[group[order]]
    return rank


class GatedReadout(nn.Module):
    def __init__(self, d, n_exp_max=768, k_routed=6, d_x=32, f_tok=2, f_pair=1, hid=16, cap_frac=0.5, explore=0.05,
                 open0=0.25, cap=0.5, log_s0=math.log(0.3), lr_gate=1e-3, lr_d=1e-4, wd_d=0.01, ema=0.99,
                 moments_bf16=False, block=64, seed=0):
        super().__init__()
        self.d, self.shared, self.S, self.k_routed = int(d), int(n_exp_max), int(n_exp_max) + 1, int(k_routed)
        self.cap_frac, self.explore, self.cap, self.seed = float(cap_frac), float(explore), float(cap), int(seed)
        self.lr_gate, self.lr_d, self.wd_d, self.ema, self.block = lr_gate, lr_d, wd_d, ema, int(block)
        self.moments_bf16 = bool(moments_bf16)
        self.n_ftok, self.n_fpair = int(f_tok), int(f_pair)
        g = torch.Generator().manual_seed(self.seed)
        r = lambda *s, std: torch.randn(*s, generator=g) * std
        self.W_gx = nn.Parameter(r(d_x, d, std=1.0 / math.sqrt(d)))
        self.G1_w = nn.Parameter(r(hid, d_x + f_tok, std=1.0 / math.sqrt(d_x + f_tok)))
        self.G1_b = nn.Parameter(torch.zeros(hid))
        self.Wp = nn.Parameter(r(self.S, hid, std=0.01))
        self.Wg = nn.Parameter(r(self.S, hid, std=0.01))
        self.Up = nn.Parameter(torch.zeros(f_pair))
        self.Ug = nn.Parameter(torch.zeros(f_pair))
        # bias init so ~open0 of bhat > 0 at the start: bhat ~ N(b, sd) with sd = 0.01 * E||h|| (Wp ~ N(0, .01));
        # E||h|| measured once on a seeded unit-rms probe -> b = Phi^-1(1 - open0) * -sd.  Learnable, lr_gate.
        with torch.no_grad():
            hp = self._h(r(512, d, std=1.0), r(512, f_tok, std=1.0))
            sd = 0.01 * hp.norm(dim=-1).pow(2).mean().sqrt()
            z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - open0))
            b0 = float(-z * sd)
        self.bp = nn.Parameter(torch.full((self.S,), b0))
        self.bg = nn.Parameter(torch.full((self.S,), b0))
        self.cp = nn.Linear(hid, 1)          # confirmation heads (BCE vs the confirmed-recall flag)
        self.cg = nn.Linear(hid, 1)
        self.log_s = nn.Parameter(torch.tensor(float(log_s0)))
        self.D = nn.Parameter(torch.zeros(self.S, d, d))
        self.register_buffer("b_ms", torch.zeros(2))           # EMA of b^2 (private, global)
        self.register_buffer("b_ms_n", torch.zeros((), dtype=torch.long))
        self._opt, self._opt_pending, self._dmom = None, None, None
        self._pending = []                                     # this step's decisions (with capture leaves)
        self._log = dict(n=0, read=0.0, explore=0.0, corr=0.0, vp=0.0, collapse=0)

    # ------------------------------------------------------------------ parameters
    def gate_parameters(self):
        return [self.W_gx, self.G1_w, self.G1_b, self.Wp, self.Wg, self.Up, self.Ug, self.bp, self.bg,
                *self.cp.parameters(), *self.cg.parameters()]

    # ------------------------------------------------------------------ the gate
    def _h(self, xhat, f_tok):
        return F.silu(torch.cat([xhat @ self.W_gx.t(), f_tok], -1) @ self.G1_w.t() + self.G1_b)

    def _bhat(self, h_rows, e, f_pair):
        bp = (self.Wp[e] * h_rows).sum(-1) + f_pair @ self.Up + self.bp[e]
        bg = (self.Wg[e] * h_rows).sum(-1) + f_pair @ self.Ug + self.bg[e]
        return bp, bg

    @staticmethod
    def _xhat(xh):
        x = xh.detach().float()
        return x / x.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()

    @torch.no_grad()
    def decide(self, xh, e_k, t_k, C, n_live, step=0, call=0, training=True, f_tok=None, f_pair=None):
        """xh [N, d] the experts' input (detached here); e_k, t_k [P] the routed pairs; C (python int) the routed
        capacity; n_live (python int) live routed experts; f_tok [N, f_tok]; f_pair [P, f_pair] (shared pairs: 0)."""
        dev, N, P = xh.device, xh.size(0), e_k.numel()
        xhat = self._xhat(xh)
        f_tok = torch.zeros(N, self.n_ftok, device=dev) if f_tok is None else f_tok.detach().float()
        fp_r = torch.zeros(P, self.n_fpair, device=dev) if f_pair is None else f_pair.detach().float()
        tok = torch.arange(N, device=dev)
        e_all = torch.cat([e_k.long(), torch.full((N,), self.shared, device=dev, dtype=torch.long)])
        t_all = torch.cat([t_k.long(), tok])
        fp_all = torch.cat([fp_r, torch.zeros(N, self.n_fpair, device=dev)])
        h = self._h(xhat, f_tok)
        bp, bg = self._bhat(h[t_all], e_all, fp_all)
        n_pairs = P + N
        if training and self.explore > 0:
            gen = torch.Generator(device=dev).manual_seed(self.seed * 1_000_003 + int(step) * 101 + int(call))
            ex_p = torch.rand(n_pairs, generator=gen, device=dev) < self.explore
            ex_tok = torch.rand(N, generator=gen, device=dev) < self.explore
            ex_g = torch.cat([torch.zeros(P, dtype=torch.bool, device=dev), ex_tok])   # via the token's shared pair
        else:
            ex_p = ex_g = torch.zeros(n_pairs, dtype=torch.bool, device=dev)
        vp, vg = (bp > 0) | ex_p, (bg > 0) | ex_g
        ex = ex_p | ex_g
        prio = torch.where(vp | vg, torch.maximum(bp, bg) + 1e4 * ex.float(), torch.full_like(bp, NEG))
        rank = _rank_in_group(e_all, prio, self.S)
        cap_e = torch.where(e_all == self.shared, torch.full_like(rank, N), torch.full_like(rank, int(C)))
        prio = torch.where(rank < cap_e, prio, torch.full_like(prio, NEG))
        Kp = max(1, int(self.cap_frac * n_pairs))
        pv, p_idx = torch.topk(prio, Kp)
        valid = torch.isfinite(pv)
        e, t = e_all[p_idx], t_all[p_idx]
        read_p = valid & vp[p_idx]
        want_g = valid & vg[p_idx]
        gs = torch.full((N,), NEG, device=dev)
        gs = gs.scatter_reduce(0, t, torch.where(want_g, bg[p_idx] + 1e4 * ex_g[p_idx].float(), torch.full_like(pv, NEG)),
                               reduce="amax", include_self=True)
        Kg = max(1, int(self.cap_frac * N))
        gv, g_idx = torch.topk(gs, Kg)
        g_valid = torch.isfinite(gv)
        gsel_used = torch.zeros(N, dtype=torch.bool, device=dev).index_put_((g_idx,), g_valid)
        read_g = want_g & gsel_used[t]
        dec = GateDecision()
        dec.N, dec.n_routed, dec.shared, dec.C, dec.n_live, dec.training = N, P, self.shared, int(C), int(n_live), training
        dec.p_idx, dec.e, dec.t, dec.slot, dec.valid = p_idx, e, t, rank[p_idx], valid
        dec.read_p, dec.read_g, dec.ex_p, dec.ex_g = read_p, read_g, ex_p[p_idx] & valid, ex_g[p_idx] & valid
        dec.g_idx, dec.g_valid, dec.gsel_used = g_idx, g_valid, gsel_used
        dec.vp_share = ((bp > 0).float().mean(), (bg > 0).float().mean())
        dec.inputs = dict(xhat=xhat[t].to(torch.bfloat16), f_tok=f_tok[t], f_pair=fp_all[p_idx])
        dec.probes, dec.y_saved = None, None
        if training:                                               # (logging: tensors, summed; no host sync here)
            n_read = (read_p | read_g).float().sum()
            self._log["n"] += 1
            self._log["read"] = self._log["read"] + n_read / n_pairs
            self._log["explore"] = self._log["explore"] + (ex_p | ex_g).float().mean()
            self._log["vp"] = self._log["vp"] + dec.vp_share[0]
        return dec

    # ------------------------------------------------------------------ the readout
    def readout(self, dec, y_p, y_g, xn):
        """y_p [Kp, d] (private readouts of dec.p_idx rows), y_g [Kg, d] (global readouts of dec.g_idx tokens), both
        scaled by the confidence already; xn [N] ||x||.  -> o [Kp, d] (exact 0 on unread rows)."""
        Kp, d = y_p.size(0), self.d
        dt = y_p.dtype
        pos = torch.zeros(dec.N, dtype=torch.long, device=y_p.device).index_put_((dec.g_idx,),
                                                                                torch.arange(dec.g_idx.numel(), device=y_p.device))
        yp = y_p * dec.read_p[:, None].to(dt)
        yg = y_g.to(dt)[pos[dec.t]] * dec.read_g[:, None].to(dt)
        if dec.training and torch.is_grad_enabled():               # Capture: dL/dy per read pair per sheet
            pr = torch.zeros(Kp, d, device=y_p.device, dtype=dt, requires_grad=True)
            pg = torch.zeros(Kp, d, device=y_p.device, dtype=dt, requires_grad=True)
            dec.probes, dec.y_saved = (pr, pg), (yp.detach(), yg.detach())
            yp, yg = yp + pr, yg + pg
            self._pending.append(dec)
        y = yp + yg
        # the capacity layout: routed rows at (e, slot) of [n_live, C], shared rows at slot of [1, N], one dummy row
        nl, C, N = dec.n_live, dec.C, dec.N
        sh = dec.e == self.shared
        flat = torch.where(sh, nl * C + dec.slot, dec.e * C + dec.slot)
        flat = torch.where(dec.valid, flat, torch.full_like(flat, nl * C + N))
        Y = torch.zeros(nl * C + N + 1, d, device=y.device, dtype=dt).index_put((flat,), y, accumulate=True)
        out_r = expert_d(Y[:nl * C].view(nl, C, d), self.D, 0, self.block).reshape(nl * C, d)
        out_s = expert_d(Y[nl * C:nl * C + N].view(1, N, d), self.D, self.shared, 1).reshape(N, d)
        out = torch.cat([out_r, out_s, Y[-1:]])[flat]
        xt = xn[dec.t].to(dt)
        o = torch.exp(self.log_s).to(dt) * xt[:, None] * out
        if not self.cap:
            return o
        f = (self.cap * xt / o.detach().float().norm(dim=-1).clamp(min=1e-6).to(dt)).clamp(max=1.0)   # detached cap
        return o * f[:, None]

    def inject(self, dec, o, buf, x, e_k, p_k):
        """routed rows -> buf[e_k, p_k] (accumulate); shared rows -> x (per token).  Static shapes."""
        rt = dec.p_idx < dec.n_routed
        j = torch.where(rt, dec.p_idx, torch.zeros_like(dec.p_idx))
        buf = buf.index_put((e_k[j], p_k[j]), (o * rt[:, None].to(o.dtype)).to(buf.dtype), accumulate=True)
        xs = torch.zeros(dec.N, self.d, device=o.device, dtype=o.dtype).index_add(0, dec.t, o * (~rt)[:, None].to(o.dtype))
        return buf, x + xs.to(x.dtype).view(x.shape)

    def reuse(self, dec, o, e_k, t_k):
        """Later passes: a (t, e) pair that read at the read pass gets its cached output, others exactly 0.
        -> (o_routed [P2, d], o_shared [N, d])."""
        key = dec.t * self.S + dec.e
        live = dec.read_p | dec.read_g
        key = torch.where(live, key, torch.full_like(key, -1))
        sk, so = torch.sort(key)
        q = t_k.long() * self.S + e_k.long()
        i = torch.searchsorted(sk, q).clamp(max=sk.numel() - 1)
        hit = sk[i] == q
        o_r = o[so[i]] * hit[:, None].to(o.dtype)
        sh = (dec.e == self.shared) & live
        o_s = torch.zeros(dec.N, self.d, device=o.device, dtype=o.dtype).index_add(0, dec.t, o * sh[:, None].to(o.dtype))
        return o_r, o_s

    @staticmethod
    def inject_reuse(o_r, o_s, buf, x, e_k, p_k):
        buf = buf.index_put((e_k, p_k), o_r.to(buf.dtype), accumulate=True)
        return buf, x + o_s.to(x.dtype).view(x.shape)

    @staticmethod
    def write_ok(target):
        """Skip any write whose target norm is < 1e-6."""
        return target.norm(dim=-1) >= 1e-6

    # ------------------------------------------------------------------ the teaching signal
    def after_backward(self, conf_p=None, conf_g=None):
        """b = -<dL/dy, y>; b~ = clamp(b / EMA-RMS, -5, 5); Huber(bhat - b~) on read pairs (x4 exploration) + 0.5 BCE
        confirmation heads; backward into the gate parameters only.  conf_p / conf_g: lists (one per pending decision)
        of [Kp] bool confirmed-recall flags, or None.  -> stats dict (empty if nothing pending)."""
        decs, self._pending = [d_ for d_ in self._pending if d_.probes is not None and d_.probes[0].grad is not None], []
        if not decs:
            return {}
        bs, bh, w, cf, logit_c, rows = [], [], [], [], [], []
        losses = []
        for i, dec in enumerate(decs):
            (pr, pg), (yp, yg) = dec.probes, dec.y_saved
            b_p = -(pr.grad.float() * yp.float()).sum(-1)
            b_g = -(pg.grad.float() * yg.float()).sum(-1)
            rows.append((dec, b_p, b_g, i))
        with torch.no_grad():                                      # EMA-RMS per sheet
            ms = torch.stack([torch.cat([b_p[d_.read_p] for d_, b_p, _, _ in rows]).pow(2).mean().nan_to_num(0.0),
                              torch.cat([b_g[d_.read_g] for d_, _, b_g, _ in rows]).pow(2).mean().nan_to_num(0.0)])
            first = self.b_ms_n == 0
            self.b_ms.copy_(torch.where(first, ms, self.b_ms * self.ema + ms * (1 - self.ema)))
            self.b_ms_n.add_(1)
            rms = self.b_ms.sqrt().clamp(min=1e-12)
        tot, cnt = 0.0, 0.0
        corr_x, corr_y = [], []
        for dec, b_p, b_g, i in rows:
            inp = dec.inputs
            h = self._h(inp["xhat"].float(), inp["f_tok"])
            bhp, bhg = self._bhat(h, dec.e, inp["f_pair"])
            for bhat, b, m, ex, k_, conf, head in ((bhp, b_p, dec.read_p, dec.ex_p, 0, conf_p, self.cp),
                                                    (bhg, b_g, dec.read_g, dec.ex_g, 1, conf_g, self.cg)):
                bt = (b / rms[k_]).clamp(-5, 5)
                wt = m.float() * (1.0 + 3.0 * ex.float())
                tot = tot + (F.huber_loss(bhat, bt, reduction="none") * wt).sum()
                cnt = cnt + m.float().sum()
                if conf is not None and conf[i] is not None:
                    lc = head(h).squeeze(-1)
                    tot = tot + 0.5 * (F.binary_cross_entropy_with_logits(lc, conf[i].float(), reduction="none")
                                       * m.float()).sum()
                corr_x.append(bhat.detach()[m])
                corr_y.append(bt[m])
        loss = tot / cnt.clamp(min=1.0) if torch.is_tensor(cnt) else torch.zeros(())
        params = [p for p in self.gate_parameters() if p.requires_grad]
        if loss.requires_grad:
            torch.autograd.backward(loss, inputs=params)
        cx, cy = torch.cat(corr_x), torch.cat(corr_y)
        corr = float(torch.corrcoef(torch.stack([cx, cy]))[0, 1]) if cx.numel() > 2 else 0.0
        corr = 0.0 if math.isnan(corr) else corr
        self._log["corr"] += corr
        return dict(gate_loss=float(loss.detach()), gate_corr=corr, gate_rows=int(cnt))

    # ------------------------------------------------------------------ the optimizer
    def _groups(self):
        return [self.gate_parameters() + [self.log_s], [self.D]]

    def _make_opt(self):
        gp, dp = self._groups()
        opt_g = torch.optim.AdamW(gp, lr=self.lr_gate, betas=(0.9, 0.99), weight_decay=0.0)
        if self.moments_bf16:
            opt_d = None
            self._dmom = dict(m=torch.zeros_like(self.D, dtype=torch.bfloat16),
                              v=torch.zeros_like(self.D, dtype=torch.bfloat16), t=0)
        else:
            fused = self.D.device.type in ("cuda", "cpu")
            opt_d = torch.optim.AdamW(dp, lr=self.lr_d, betas=(0.9, 0.99), weight_decay=self.wd_d, fused=fused)
        self._opt = (opt_g, opt_d)
        if self._opt_pending is not None:
            try:
                pg, pd, mom = self._opt_pending
                opt_g.load_state_dict(pg)
                if opt_d is not None and pd is not None:
                    opt_d.load_state_dict(pd)
                if mom is not None and self._dmom is not None:
                    self._dmom = dict(m=mom["m"].to(self.D.device), v=mom["v"].to(self.D.device), t=mom["t"])
            except (ValueError, KeyError):
                pass
            self._opt_pending = None

    @torch.no_grad()
    def _bf16_adamw(self):
        g, st = self.D.grad, self._dmom
        st["t"] += 1
        b1, b2, t = 0.9, 0.99, st["t"]
        m = st["m"].float().mul_(b1).add_(g, alpha=1 - b1)
        v = st["v"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
        st["m"].copy_(m), st["v"].copy_(v)
        self.D.mul_(1 - self.lr_d * self.wd_d)
        self.D.addcdiv_(m / (1 - b1 ** t), (v / (1 - b2 ** t)).sqrt_().add_(1e-8), value=-self.lr_d)

    def step(self):
        """The module's own optimizer step: clip 1.0 per group, then AdamW (gate group; D group fused or bf16 moments)."""
        if self._opt is None:
            self._make_opt()
        gp, dp = self._groups()
        if any(p.grad is not None for p in gp):
            torch.nn.utils.clip_grad_norm_([p for p in gp if p.grad is not None], 1.0)
            self._opt[0].step()
        if self.D.grad is not None:
            torch.nn.utils.clip_grad_norm_(dp, 1.0)
            if self._opt[1] is not None:
                self._opt[1].step()
            else:
                self._bf16_adamw()
        for p in gp + dp:
            p.grad = None

    # ------------------------------------------------------------------ state / growth / logs
    def get_extra_state(self):
        if self._opt is not None:
            mom = None if self._dmom is None else dict(m=self._dmom["m"], v=self._dmom["v"], t=self._dmom["t"])
            opt = (self._opt[0].state_dict(), None if self._opt[1] is None else self._opt[1].state_dict(), mom)
        else:
            opt = self._opt_pending
        return dict(opt=opt)

    def set_extra_state(self, st):
        self._opt, self._opt_pending = None, st.get("opt")

    @torch.no_grad()
    def on_grow(self, src, dst):
        src = torch.as_tensor(src, dtype=torch.long)
        dst = torch.as_tensor(dst, dtype=torch.long)
        for p in (self.D, self.Wp, self.Wg, self.bp, self.bg):
            p[dst] = p[src]
        if self._opt is not None and self._opt[1] is not None:   # (the child's moments: the parent's)
            for p in (self.D,):
                st = self._opt[1].state.get(p, {})
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in st:
                        st[k][dst] = st[k][src]

    def log_record(self):
        n = max(1, self._log["n"])
        rec = dict(wgate_read=float(self._log["read"]) / n, wgate_explore=float(self._log["explore"]) / n,
                   wgate_vp=float(self._log["vp"]) / n, wgate_corr=float(self._log["corr"]) / n,
                   wgate_log_s=float(self.log_s.detach()))
        # collapse: the gate (exploration aside) opens almost nothing or saturates the cap
        rec["GATE_COLLAPSE"] = int(self._log["n"] > 0 and (rec["wgate_vp"] < 0.01 or rec["wgate_vp"] > 0.95))
        self._log = dict(n=0, read=0.0, explore=0.0, corr=0.0, vp=0.0, collapse=0)
        return rec

    def nbytes_report(self):
        D = self.D.numel() * 4
        gate = sum(p.numel() for p in self.gate_parameters()) * 4
        mom = 2 * self.D.numel() * (2 if self.moments_bf16 else 4)
        return dict(D=D, D_grad=D, D_moments=mom, D_total=2 * D + mom, gate=gate, gate_total=4 * gate,
                    total=2 * D + mom + 4 * gate)
