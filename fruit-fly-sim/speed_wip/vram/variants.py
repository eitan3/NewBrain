"""Same-math what-ifs for the VRAM study, applied by monkeypatching in the profiling process only (the model code
is not edited). apply(S, names):

  chunk_ce   the per-pass read-outs (deep supervision + halting features) and the final LM loss through ONE fused,
             token-chunked linear + log-softmax + CE (+ entropy / KL for halting) autograd function: the (N, V) fp32
             log-probs are never materialised (only `CHUNK` rows at a time); the previous pass is kept as its bf16
             logits (N, V) instead of fp32 log-probs (identical: lp is a function of the bf16 logits); backward
             recomputes each chunk's logits. Same math up to the summation order of dW over chunks.
  ckpt_pass  activation checkpointing of every backpropagated pass (Core.forward under non-reentrant checkpoint,
             RNG state preserved). Measurement only: the recompute repeats the forward's side effects (homeostasis
             EMA, balancer counts, private-memory usage counters), which a real implementation must guard.
  bf16_mom   BrainOpt Muon momentum stored in bf16 (the update math promotes to fp32).
"""
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import loop_brain
from loop_brain import LoopBrain

CHUNK = 2048


class ChunkedReadout(torch.autograd.Function):
    """ce[n] = -log softmax(h W^T)[n, tgt[n]] with bf16 logits (as under autocast), per chunk of rows. Also (no grad):
    the bf16 logits of every row (for the next pass's KL) and the halting parts (log1p KL to prev, normalised
    entropy). Saves only h (bf16) and the targets."""

    @staticmethod
    def forward(ctx, h, W, tgt, prev, want_logits, want_halt):
        hb, Wb = h.to(torch.bfloat16), W.to(torch.bfloat16)
        N, V = hb.size(0), Wb.size(0)
        ce = torch.empty(N, dtype=torch.float32, device=h.device)
        logits = torch.empty(N, V, dtype=torch.bfloat16, device=h.device) if want_logits else None
        ent = torch.empty(N, dtype=torch.float32, device=h.device) if want_halt else None
        lkl = torch.empty(N, dtype=torch.float32, device=h.device) if want_halt else None
        logV = math.log(V)
        for c in range(0, N, CHUNK):
            sl = slice(c, c + CHUNK)
            lg = hb[sl] @ Wb.t()
            lp = torch.log_softmax(lg.float(), -1)
            ce[sl] = -lp.gather(1, tgt[sl, None]).squeeze(1)
            if want_halt:
                p = lp.exp()
                ent[sl] = -(p * lp).sum(-1) / logV
                if prev is not None:
                    lkl[sl] = torch.log1p((p * (lp - torch.log_softmax(prev[sl].float(), -1))).sum(-1).clamp(min=0))
                else:
                    lkl[sl] = 0.0
            if want_logits:
                logits[sl] = lg
        ctx.save_for_backward(hb, Wb, tgt)
        ctx.hdt = h.dtype
        return ce, logits, ent, lkl

    @staticmethod
    def backward(ctx, gce, *_):
        hb, Wb, tgt = ctx.saved_tensors
        N = hb.size(0)
        dh = torch.empty_like(hb)
        dW = torch.zeros(Wb.shape, dtype=torch.float32, device=hb.device)
        for c in range(0, N, CHUNK):
            sl = slice(c, c + CHUNK)
            p = torch.softmax((hb[sl] @ Wb.t()).float(), -1)
            p[torch.arange(p.size(0), device=p.device), tgt[sl]] -= 1.0
            g = (p * gce[sl, None]).to(torch.bfloat16)
            dh[sl] = g @ Wb
            dW += (g.t() @ hb[sl]).float()
        return dh.to(ctx.hdt), dW, None, None, None, None


def _chunk_ce(self, h, tgt, prev, want_logits, want_halt):
    W = self.heads["lm"].weight
    with torch.autocast("cuda", enabled=False):
        return ChunkedReadout.apply(h.flatten(0, -2), W, tgt.flatten(), prev, want_logits, want_halt)


def pass_readout_ce(self, x, tgt, prev_lp, k, cu=None):
    """(ce (B, T) with grad, the bf16 logits as 'prev_lp', the halting features) - as LoopBrain.pass_readout_ce."""
    h = self.normf(x)
    shp = tgt.shape
    ce, logits, ent, lkl = _chunk_ce(self, h, tgt, prev_lp, True, True)
    feat = self.halt_features(x, None, None, k, cu, parts=(lkl.view(shp), ent.view(shp)))
    return ce.view(shp), logits, feat


_orig_forward = LoopBrain.forward


def forward(self, idx, targets=None, head="lm"):
    if targets is None or head != "lm" or not self.training:
        return _orig_forward(self, idx, targets, head)
    per_pass = self.cfg["deep_sup"] or self.cfg["halt"] or self.cfg.get("think_w", 0)
    self._targets = targets if per_pass else None
    try:
        h = self.features(idx)                      # normf(x_K)
        ce, _, _, _ = _chunk_ce(self, h, targets, None, False, False)
        return h, ce.mean()                         # (the trainer only deletes the first output)
    finally:
        self._targets = None


def apply_chunk_ce(S):
    LoopBrain.pass_readout_ce = pass_readout_ce
    LoopBrain.forward = forward


def apply_ckpt_pass(S):
    # (measurement: homeostasis edits alpha in place during the forward, so a recompute would see other thresholds
    # and route differently; a real implementation defers those updates to once per step - here they are off)
    S.model.set_homeo(0.0, S.args.target, S.args.homeo_mode)
    core = S.model.blocks[0]
    inner = type(core).forward

    def fwd(self, *a, **kw):
        if torch.is_grad_enabled() and self.training:
            return torch.utils.checkpoint.checkpoint(lambda *aa: inner(self, *aa, **kw), *a, use_reentrant=False)
        return inner(self, *a, **kw)
    type(core).forward = fwd


def apply_bf16_mom(S):
    for st in S.opt.state.values():
        if "mom" in st:
            st["mom"] = st["mom"].to(torch.bfloat16)


def apply_nohomeo(S):
    S.model.set_homeo(0.0, S.args.target, S.args.homeo_mode)


def apply(S, names):
    for n in names:
        {"chunk_ce": apply_chunk_ce, "ckpt_pass": apply_ckpt_pass, "bf16_mom": apply_bf16_mom,
         "nohomeo": apply_nohomeo}[n](S)
        print("variant", n, "applied", flush=True)
