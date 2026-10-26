"""Wave memory v2.1 (SPEC_v21_final.md PARTS 2-4): pre-read gate + selection, Translate (per-expert D, a BUFFER),
the factored D update, the gate target / loss.  No host syncs in the step path (logging returns tensors).

Memory rows: every per-expert tensor (D, D_grad, R, Cv, n, Wp, Wg, bp_b, bg_b) is indexed by MEMORY ROW; rows
0..E_live-1 are the live routed experts (row_of[brain id] from the caller), row 768 (= SH) is the shared expert.

Per forward (pass 3):
  dec = gr.decide(xh, e_p, row_of, E_live, step=, micro=, training=, f_tok=, f_pair=)
       xh [N,384] experts' input (detached here); e_p [N,6] brain expert ids; row_of [769] int64 (768 -> 768);
       f_tok [N,Ft]; f_pair [N,6] or [N,6,Fp] router prob.  -> dec with GT [K_g], gvalid, PR [K_pr], vp [K],
       S [K], tok, slot, vg, xp_S, xg ...  The memory reads y_g for tokens GT (DUMMY where ~gvalid) and y_p for the
       pairs S[PR] (DUMMY where ~vp[PR]).
  r_pairs, r_shared = gr.readout(dec, y_g [K_g,384], y_p [K_pr,384], xnorm [N], keep_tok [N] or None)
       (hint + Capture + Translate + scale + detached cap; dec.r3 cached for passes 4-5)
  passes 4-5: r_pairs, r_shared = gr.reuse(dec, e_p_new)
after_backward:  stats = gr.gate_loss(dec, conf_p [K] bool | None, conf_g [N] bool | None)  (backward into gate params)
                 dec.E_pair [K,384] = dL/dy_pair (W1 signals);  dec.b_p, dec.b_g raw benefits
after_step:      memory AdamW over gr.param_groups() (gate lr 1e-3, log_s 1e-4; the memory owns it), then
                 gr.d_step(E_live)   (factored update over D[:E_live] + row 768, then D_grad.zero_())
on_grow:         gr.on_grow(parent_rows, child_rows, opt)    reset: gr.reset_rows(rows, opt)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SH = 768
LOW = torch.iinfo(torch.int64).min
INF = float("inf")


def tkey(x):
    """Tie-free int64 sort key of a float32 tensor: order = value, ties -> the lower index wins (larger key)."""
    x = x.detach().float().contiguous().view(-1)
    n = x.numel()
    b = x.view(torch.int32)
    k = torch.where(b >= 0, b, b ^ 0x7FFFFFFF)
    return (k.long() << 32) + (n - 1 - torch.arange(n, device=x.device))


class _ieee:
    """fp32 'ieee' matmul inside, the trainer's precision restored on exit (host-side setting, no sync)."""
    def __enter__(self):
        self.p = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def __exit__(self, *a):
        torch.set_float32_matmul_precision(self.p)


# ---------------------------------------------------------------------------------------------------- autograd
class Capture(torch.autograd.Function):
    """Identity on y; backward: E_buf += grad (fp32); the anchor (0-dim leaf) keeps the graph alive."""
    @staticmethod
    def forward(ctx, y, anchor, E_buf):
        ctx.E_buf = E_buf
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        ctx.E_buf.add_(g.float())
        return g, torch.zeros((), device=g.device, dtype=g.dtype), None


class Translate(torch.autograd.Function):
    """outr = Xr + Xr @ D[:E] (per expert), outs = Xs + Xs @ D[SH].  D is a buffer (never .grad); backward writes
    dD into grad_buf (if not None: eval / train-only pass None)."""
    @staticmethod
    def forward(ctx, Xr, Xs, D, E, grad_buf):
        ctx.save_for_backward(Xr, Xs)
        ctx.D, ctx.E, ctx.gb = D, E, grad_buf
        return torch.baddbmm(Xr, Xr, D[:E]), torch.addmm(Xs, Xs, D[SH])

    @staticmethod
    def backward(ctx, gr, gs):
        Xr, Xs = ctx.saved_tensors
        D, E, gb = ctx.D, ctx.E, ctx.gb
        if gb is not None:
            gb[:E].baddbmm_(Xr.transpose(1, 2), gr)
            gb[SH].addmm_(Xs.t(), gs)
        return torch.baddbmm(gr, gr, D[:E].transpose(1, 2)), torch.addmm(gs, gs, D[SH].t()), None, None, None


class GateDecision:
    pass


# ---------------------------------------------------------------------------------------------------- the module
class GatedReadout(nn.Module):
    def __init__(self, d=384, n_slots=769, f_tok=2, f_pair=1, d_x=32, hid=16, explore=0.05, open0=0.25, tau=0.0,
                 cap=0.5, log_s0=math.log(0.3), lr_d=1e-4, b2=0.99, eps=1e-30, clip_d=1.0, ema=0.99, seed=0):
        super().__init__()
        assert n_slots == SH + 1 or n_slots > 1
        self.d, self.S_rows, self.seed = int(d), int(n_slots), int(seed)
        self.explore, self.tau, self.cap = float(explore), float(tau), float(cap)
        self.lr_d, self.b2, self.eps, self.clip_d, self.ema = lr_d, b2, eps, clip_d, ema
        self.n_ftok, self.n_fpair = int(f_tok), int(f_pair)
        self.train_only = False
        g = torch.Generator().manual_seed(self.seed)
        r = lambda *s, std: torch.randn(*s, generator=g) * std
        self.W_gx = nn.Parameter(r(d_x, d, std=1.0 / math.sqrt(d)))
        self.G1_w = nn.Parameter(r(hid, d_x + f_tok, std=1.0 / math.sqrt(d_x + f_tok)))
        self.G1_b = nn.Parameter(torch.zeros(hid))
        self.Wp = nn.Parameter(r(n_slots, hid, std=0.01))
        self.Wg = nn.Parameter(r(n_slots, hid, std=0.01))
        self.Up = nn.Parameter(torch.zeros(f_pair))
        self.Ug = nn.Parameter(torch.zeros(f_pair))
        with torch.no_grad():       # bias init: ~open0 of bhat > tau at the start (sd of <W, h> on a unit-rms probe)
            hp = self._h(r(512, d, std=1.0), r(512, f_tok, std=1.0))
            sd = 0.01 * float(hp.norm(dim=-1).pow(2).mean().sqrt())
            b0 = self.tau - float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - open0))) * sd
        self._b0_val = b0
        self.bp_b = nn.Parameter(torch.full((n_slots,), b0))
        self.bg_b = nn.Parameter(torch.full((n_slots,), b0))
        # confirmation heads: chat_g = <Wc0,h> + <Uc0,f_tok> + bc0 ; chat_p = <Wc1,h> + <Uc1,f_pair> + bc1
        self.Wc0, self.Uc0, self.bc0 = nn.Parameter(torch.zeros(hid)), nn.Parameter(torch.zeros(f_tok)), nn.Parameter(torch.zeros(()))
        self.Wc1, self.Uc1, self.bc1 = nn.Parameter(torch.zeros(hid)), nn.Parameter(torch.zeros(f_pair)), nn.Parameter(torch.zeros(()))
        self.log_s = nn.Parameter(torch.tensor(float(log_s0)))
        # the init rows (reset): non-persistent copies
        self.register_buffer("_Wp0", self.Wp.detach().clone(), persistent=False)
        self.register_buffer("_Wg0", self.Wg.detach().clone(), persistent=False)
        # D: a buffer, its grad buffer (not saved), the factored second moments
        self.register_buffer("D", torch.zeros(n_slots, d, d))
        self.register_buffer("D_grad", torch.zeros(n_slots, d, d), persistent=False)
        self.register_buffer("R", torch.zeros(n_slots, d))
        self.register_buffer("Cv", torch.zeros(n_slots, d))
        self.register_buffer("n", torch.zeros(n_slots, dtype=torch.long))
        self.register_buffer("sigma_b", torch.zeros(2))
        self.register_buffer("collapse_run", torch.zeros((), dtype=torch.long))
        self._log = None

    # ------------------------------------------------------------------ parameters
    def gate_parameters(self):
        return [self.W_gx, self.G1_w, self.G1_b, self.Wp, self.Wg, self.Up, self.Ug, self.bp_b, self.bg_b,
                self.Wc0, self.Uc0, self.bc0, self.Wc1, self.Uc1, self.bc1]

    def param_groups(self, lr_gate=1e-3, lr_s=1e-4):
        """For the memory's (fused) AdamW: betas .9/.99, wd 0, clip 1.0 per group."""
        return [dict(params=self.gate_parameters(), lr=lr_gate, name="gate"),
                dict(params=[self.log_s], lr=lr_s, name="log_s")]

    def grad_buf(self, training):
        return self.D_grad if (training and not self.train_only) else None

    # ------------------------------------------------------------------ gate
    def _h(self, xhat, f_tok):
        return F.silu(torch.cat([xhat @ self.W_gx.t(), f_tok], -1) @ self.G1_w.t() + self.G1_b)

    def _bhat(self, h, rows, f_pair):
        """h [N,hid], rows [N,7] memory rows, f_pair [N,7,Fp] -> bp, bg [N,7] (bp[:,6] = -inf)."""
        bp = torch.einsum("nkh,nh->nk", self.Wp[rows], h) + f_pair @ self.Up + self.bp_b[rows]
        bg = torch.einsum("nkh,nh->nk", self.Wg[rows], h) + f_pair @ self.Ug + self.bg_b[rows]
        bp = torch.cat([bp[:, :6], torch.full_like(bp[:, 6:], -INF)], 1)
        return bp, bg

    @staticmethod
    def budgets(N, E_live, training, explore=0.05):
        K, K_pr, K_g = 7 * N // 2, 6 * N // 2, N // 2
        C = 8 * math.ceil(2 * K * (6 / 7) / E_live / 8)
        K_xp = int(round(explore * 6 * N)) if training else 0
        K_xg = int(round(explore * N)) if training else 0
        return K, K_pr, K_g, C, K_g, K_xp, K_xg

    @torch.no_grad()
    def decide(self, xh, e_p, row_of, E_live, step=0, micro=0, training=True, f_tok=None, f_pair=None):
        with _ieee(), torch.autocast(xh.device.type, enabled=False):
            return self._decide(xh, e_p, row_of, int(E_live), int(step), int(micro), bool(training), f_tok, f_pair)

    def _decide(self, xh, e_p, row_of, E_live, step, micro, training, f_tok, f_pair):
        dev, N, tau = xh.device, xh.size(0), self.tau
        K, K_pr, K_g, C, C_sh, K_xp, K_xg = self.budgets(N, E_live, training, self.explore)
        M = 7 * N
        ar = torch.arange(M, device=dev)
        x = xh.detach().float()
        xhat = x / x.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()
        f_tok = torch.zeros(N, self.n_ftok, device=dev) if f_tok is None else f_tok.detach().float()
        if f_pair is None:
            f_pair = torch.zeros(N, 6, self.n_fpair, device=dev)
        f_pair = f_pair.detach().float()
        if f_pair.dim() == 2:
            f_pair = f_pair[..., None]
        f_pair = torch.cat([f_pair, torch.zeros(N, 1, self.n_fpair, device=dev)], 1)          # [N,7,Fp]
        eid = torch.cat([e_p.long(), torch.full((N, 1), SH, device=dev, dtype=torch.long)], 1)  # [N,7]
        rows = row_of.long()[eid]                                                                # memory rows
        h = self._h(xhat, f_tok)
        bp, bg = self._bhat(h, rows, f_pair)
        # exploration (own generator, reseeded per (step, micro); off at eval)
        gen = torch.Generator(device=dev).manual_seed(self.seed * 1_000_003 + step * 1009 + micro)
        u6, uN = torch.rand(6 * N, generator=gen, device=dev), torch.rand(N, generator=gen, device=dev)
        xp6 = torch.zeros(6 * N, dtype=torch.bool, device=dev)
        if K_xp:
            xp6.index_put_((torch.topk(tkey(u6), K_xp).indices,), torch.tensor(True, device=dev))
        xp = torch.cat([xp6.view(N, 6), torch.zeros(N, 1, dtype=torch.bool, device=dev)], 1)
        xg = torch.zeros(N, dtype=torch.bool, device=dev)
        if K_xg:
            xg.index_put_((torch.topk(tkey(uN), K_xg).indices,), torch.tensor(True, device=dev))
        # global selection: gsel (NOT isfinite: forced tokens carry +inf)
        SG = bg.amax(1)
        gs = torch.where(xg, INF, torch.where(SG > tau, SG, -INF))
        GT0 = torch.topk(tkey(gs), K_g).indices
        gsel = torch.zeros(N, dtype=torch.bool, device=dev).index_put_((GT0,), xg[GT0] | (SG[GT0] > tau))
        wg = gsel[:, None] & (xg[:, None] | (bg > tau))
        wp = xp | (bp > tau)
        want = wp | wg
        forced = xp | (xg[:, None] & wg)
        r = torch.where(forced, INF, torch.where(want, torch.maximum(torch.where(wp, bp, -INF),
                                                                       torch.where(wg, bg, -INF)), -INF)).view(-1)
        grp = torch.where(eid == SH, E_live, rows).view(-1)
        key = tkey(r)
        grank = torch.empty(M, dtype=torch.long, device=dev).scatter_(0, torch.argsort(key, descending=True), ar)
        order = torch.argsort(grp * M + grank)
        gs2 = grp[order]
        rk = torch.empty(M, dtype=torch.long, device=dev).scatter_(0, order, ar - torch.searchsorted(gs2, gs2))
        capv = torch.where(grp == E_live, C_sh, C)
        elig = (rk < capv) & want.view(-1)
        S = torch.topk(torch.where(elig, key, LOW + ar), K).indices
        valid = elig[S]
        tok = S // 7
        gS = grp[S]
        vp0 = valid & wp.view(-1)[S]
        vg = valid & wg.view(-1)[S]
        NSLOT = E_live * C + C_sh
        slot = torch.where(valid, torch.where(gS == E_live, E_live * C, gS * C) + rk[S], NSLOT)
        xpS = xp.view(-1)[S]
        PR = torch.topk(tkey(torch.where(vp0, torch.where(xpS, INF, bp.view(-1)[S]), -INF)), K_pr).indices
        vp = torch.zeros(K, dtype=torch.bool, device=dev).index_put_((PR,), vp0[PR])
        gused = torch.zeros(N, dtype=torch.int32, device=dev).index_put_((tok,), vg.int(), accumulate=True) > 0
        GT = torch.topk(tkey(torch.where(gused, SG, -INF)), K_g).indices
        gvalid = gused[GT]
        # dense views AFTER all masks: vp_d / vg_d [N,7]; pslot [N,7] -> row of K (K = zero row); gslot [N] -> K_g
        vp_d = torch.zeros(M + 1, dtype=torch.bool, device=dev).index_put_((S,), vp)[:M].view(N, 7)
        vg_d = torch.zeros(M + 1, dtype=torch.bool, device=dev).index_put_((S,), vg)[:M].view(N, 7)
        pslot = torch.full((M,), K, dtype=torch.long, device=dev).index_put_((S,), torch.arange(K, device=dev)).view(N, 7)
        gslot = torch.full((N,), K_g, dtype=torch.long, device=dev).index_put_(
            (GT,), torch.where(gvalid, torch.arange(K_g, device=dev), K_g))
        dec = GateDecision()
        dec.__dict__.update(
            N=N, E_live=E_live, K=K, K_pr=K_pr, K_g=K_g, C=C, C_sh=C_sh, NSLOT=NSLOT, K_xp=K_xp, K_xg=K_xg,
            training=training, eid=eid, rows=rows, xp=xp, xg=xg, gsel=gsel, wp=wp, wg=wg, want=want,
            S=S, valid=valid, tok=tok, grp_S=gS, slot=slot, vp0=vp0, vp=vp, vg=vg, PR=PR, gused=gused, GT=GT,
            gvalid=gvalid, vp_d=vp_d, vg_d=vg_d, pslot=pslot, gslot=gslot, xpS=xpS, xgS=xg[tok],
            n_excluded=want.sum() - valid.sum(), n_demoted=(vp0 & ~vp).sum(),
            inputs=dict(xhat=xhat, f_tok=f_tok, f_pair=f_pair), r3=None, E_pair=None, y_raw=None)
        return dec

    # ------------------------------------------------------------------ hint + translation
    def readout(self, dec, y_g, y_p, xnorm, keep_tok=None, out_dtype=torch.bfloat16):
        """y_g [K_g,d] reads of the GT tokens, y_p [K_pr,d] reads of the S[PR] pairs (confidence-scaled; DUMMY
        rows = 0); xnorm [N] (detached here); keep_tok [N] float/bool or None.  -> (r_pairs [N,6,d], r_shared [N,d])."""
        with _ieee(), torch.autocast(y_g.device.type, enabled=False):
            N, K, d, E = dec.N, dec.K, self.d, dec.E_live
            dev = y_g.device
            y_g, y_p = y_g.float(), y_p.float()
            y_g_full = torch.zeros(N, d, device=dev).index_put((dec.GT,), y_g * dec.gvalid[:, None])
            y_p_full = torch.zeros(K, d, device=dev).index_put((dec.PR,), y_p * dec.vp[dec.PR][:, None])
            yg_raw = dec.vg[:, None] * y_g_full[dec.tok]
            yp_raw = dec.vp[:, None] * y_p_full
            y_pair = yg_raw + yp_raw
            if keep_tok is not None:
                y_pair = keep_tok.float()[dec.tok][:, None] * y_pair
            gb = self.grad_buf(dec.training)
            if dec.training and torch.is_grad_enabled():
                dec.E_pair = torch.zeros(K, d, device=dev)
                dec.y_raw = (yp_raw.detach(), yg_raw.detach())
                anchor = torch.zeros((), device=dev, requires_grad=True)
                y_pair = Capture.apply(y_pair, anchor, dec.E_pair)
            X = torch.zeros(dec.NSLOT + 1, d, device=dev).index_put((dec.slot,), y_pair)
            outr, outs = Translate.apply(X[:E * dec.C].view(E, dec.C, d), X[E * dec.C:dec.NSLOT], self.D, E, gb)
            o = torch.cat([outr.reshape(-1, d), outs, torch.zeros(1, d, device=dev)])[dec.slot]
            xt = xnorm.detach().float()[dec.tok]
            o = (torch.exp(self.log_s) * xt)[:, None] * o
            if self.cap:
                o = o * torch.clamp(self.cap * xt / o.norm(dim=-1).clamp_min(1e-6), max=1).detach()[:, None]
            r3 = torch.zeros(7 * N + 1, d, device=dev, dtype=out_dtype).index_put((dec.S,), o.to(out_dtype))
            dec.r3 = r3
            v = r3[:7 * N].view(N, 7, d)
            return v[:, :6], v[:, 6]

    @staticmethod
    def reuse(dec, e_p):
        """Passes 4-5: a (token, expert) that read at pass 3 gets its cached output, others exactly 0."""
        N, r3 = dec.N, dec.r3
        eq = e_p.long()[:, :, None] == dec.eid[:, None, :6]
        has = eq.any(-1)
        kidx = eq.to(torch.int8).argmax(-1)
        idx = torch.where(has, torch.arange(N, device=r3.device)[:, None] * 7 + kidx, 7 * N)
        return r3[idx], r3[:7 * N].view(N, 7, -1)[:, 6]

    # ------------------------------------------------------------------ gate target + loss
    def gate_loss(self, dec, conf_p=None, conf_g=None):
        """After the main backward.  b_p = -<E_pair, vp y_p_raw>, b_g = -<E_pair, vg y_g_raw> (raw = before keep);
        sigma_b EMA; bt = clamp(b / sigma_b, -5, 5); L = Huber + 0.5 BCE (weight 4 on exploration); backward into
        the gate params only.  conf_p [K] bool (per S row), conf_g [N] bool.  -> dict of tensors (no sync)."""
        if dec.E_pair is None:
            return {}
        with _ieee(), torch.autocast(dec.E_pair.device.type, enabled=False):
            E = dec.E_pair
            yp, yg = dec.y_raw
            b_p, b_g = -(E * yp).sum(-1), -(E * yg).sum(-1)
            dec.b_p, dec.b_g = b_p, b_g
            vp, vg = dec.vp.float(), dec.vg.float()
            with torch.no_grad():
                rms = torch.stack([(b_p.square() * vp).sum() / vp.sum().clamp_min(1),
                                   (b_g.square() * vg).sum() / vg.sum().clamp_min(1)]).sqrt()
                self.sigma_b.copy_(torch.where(self.sigma_b == 0, rms, self.ema * self.sigma_b + (1 - self.ema) * rms))
                sb = self.sigma_b.clamp_min(1e-12)
                bt_p, bt_g = (b_p / sb[0]).clamp(-5, 5), (b_g / sb[1]).clamp(-5, 5)
            inp = dec.inputs
            h_all = self._h(inp["xhat"], inp["f_tok"])
            bp, bg = self._bhat(h_all, dec.rows, inp["f_pair"])
            S, tok = dec.S, dec.tok
            bhp = torch.where(dec.vp, bp.view(-1)[S], torch.zeros_like(bt_p))       # (-inf rows masked out)
            bhg = bg.view(-1)[S]
            w_p = 1 + 3 * dec.xpS.float()
            w_g = 1 + 3 * dec.xgS.float()
            num = (w_p * vp * F.huber_loss(bhp, bt_p, reduction="none", delta=1.0)
                   + w_g * vg * F.huber_loss(bhg, bt_g, reduction="none", delta=1.0)).sum()
            L_b = num / (w_p * vp + w_g * vg).sum().clamp_min(1)
            L_c = torch.zeros((), device=E.device)
            h = h_all[tok]
            if conf_p is not None:
                cp = h @ self.Wc1 + inp["f_pair"].view(-1, self.n_fpair)[S] @ self.Uc1 + self.bc1
                L_c = L_c + (F.binary_cross_entropy_with_logits(cp, conf_p.float(), reduction="none") * vp).sum() / vp.sum().clamp_min(1)
            if conf_g is not None:
                cg = h @ self.Wc0 + inp["f_tok"][tok] @ self.Uc0 + self.bc0
                L_c = L_c + (F.binary_cross_entropy_with_logits(cg, conf_g.float()[tok], reduction="none") * vg).sum() / vg.sum().clamp_min(1)
            L = L_b + 0.5 * L_c
            torch.autograd.backward(L, inputs=self.gate_parameters())
            with torch.no_grad():
                nonexp = (dec.vp & ~dec.xpS).sum() + (dec.vg & ~dec.xgS).sum()
                ex = (dec.vp & dec.xpS).float()
                ex_mean = (bt_p * ex).sum() / ex.sum().clamp_min(1)
                cond = (nonexp.float() < 0.01 * dec.K) & (ex_mean > 0)
                self.collapse_run.copy_(torch.where(cond, self.collapse_run + 1, torch.zeros_like(self.collapse_run)))
            self._log = dict(gate_L=L.detach(), gate_Lb=L_b.detach(), gate_Lc=L_c.detach(),
                             read_share=(dec.vp.sum() + dec.vg.sum()).float() / dec.K,
                             priv_reads=dec.vp.sum(), glob_tokens=dec.gvalid.sum(),
                             explore_share=(dec.xpS & dec.valid).float().sum() / dec.K,
                             excluded=dec.n_excluded, demoted=dec.n_demoted, sigma_b=self.sigma_b.clone(),
                             GATE_COLLAPSE=self.collapse_run >= 200)
            return self._log

    def log_record(self):
        """Tensors (the caller converts at log time)."""
        return dict(self._log or {}, log_s=self.log_s.detach())

    # ------------------------------------------------------------------ D update
    @torch.no_grad()
    def d_step(self, E_live, chunk=256):
        """Factored second-moment update over D[:E_live] + row SH, then D_grad.zero_().  No syncs; zero-grad rows
        don't move; rows independent."""
        with _ieee():
            b2, eps, lr, clip, d = self.b2, self.eps, self.lr_d, self.clip_d, self.d
            spans = [(a, min(int(E_live), a + chunk)) for a in range(0, int(E_live), chunk)] + [(SH, SH + 1)]
            for a, b in spans:
                G = self.D_grad[a:b]
                self.R[a:b].mul_(b2).add_(torch.linalg.vector_norm(G, dim=-1).square() / d, alpha=1 - b2)
                self.Cv[a:b].mul_(b2).add_(torch.linalg.vector_norm(G, dim=-2).square() / d, alpha=1 - b2)
                self.n[a:b] += 1
                bc = 1 - b2 ** self.n[a:b].double()
                rh = self.R[a:b] / bc.float()[:, None]
                ch = self.Cv[a:b] / bc.float()[:, None]
                V = rh[:, :, None] * ch[:, None, :] / rh.mean(-1).clamp_min(eps)[:, None, None] + eps
                U = G * V.rsqrt_()
                rms = torch.linalg.vector_norm(U, dim=(-2, -1)) / d
                self.D[a:b].add_(U / (rms / clip).clamp_min(1)[:, None, None], alpha=-lr)
            self.D_grad.zero_()

    # ------------------------------------------------------------------ split / reset / state
    _ROWED = ("D", "R", "Cv", "n")
    _GATE_ROWED = ("Wp", "Wg", "bp_b", "bg_b")

    @torch.no_grad()
    def on_grow(self, parent_rows, child_rows, opt=None):
        """Clone first, then index_copy_ into every child row: D, R, Cv, n, Wp, Wg, bp_b, bg_b and their AdamW moments
        (opt: the memory's optimizer, or None).  A child may reuse its parent row."""
        dev = self.D.device
        src = torch.as_tensor(parent_rows, dtype=torch.long, device=dev)
        dst = torch.as_tensor(child_rows, dtype=torch.long, device=dev)
        for name in self._ROWED + self._GATE_ROWED:
            t = getattr(self, name)
            t.index_copy_(0, dst, t[src].clone())
            if opt is not None and name in self._GATE_ROWED:
                st = opt.state.get(t, {})
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in st:
                        st[k].index_copy_(0, dst, st[k][src].clone())
        self.D_grad[dst] = 0

    @torch.no_grad()
    def reset_rows(self, rows, opt=None):
        dev = self.D.device
        rows = torch.as_tensor(rows, dtype=torch.long, device=dev)
        for name in ("D", "D_grad", "R", "Cv", "n"):
            getattr(self, name)[rows] = 0
        self.Wp[rows] = self._Wp0[rows]
        self.Wg[rows] = self._Wg0[rows]
        b0 = self.bp_b.new_full((rows.numel(),), 0.0)
        self.bp_b[rows] = b0 + self._b0
        self.bg_b[rows] = b0 + self._b0
        if opt is not None:
            for name in self._GATE_ROWED:
                st = opt.state.get(getattr(self, name), {})
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in st:
                        st[k][rows] = 0

    @property
    def _b0(self):
        return float(self._b0_val) if hasattr(self, "_b0_val") else 0.0

    def nbytes_report(self):
        D = self.D.numel() * 4
        small = (self.R.numel() + self.Cv.numel()) * 4 + self.n.numel() * 8
        gate = sum(p.numel() for p in self.gate_parameters()) * 4
        return dict(D=D, D_grad=D, R_Cv_n=small, D_total=2 * D + small, gate=gate, gate_with_adamw=4 * gate,
                    file_D=D + small)
