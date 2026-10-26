"""Phase 0 of plan_v2/22 (learned memory gate): is the memory's benefit concentrated in a PREDICTABLE subset of positions?

    python -B phase0_benefit.py --ckpt ../runs/wiki_mem_12k/last.pt --device cuda
    python -B phase0_benefit.py --ckpt ../runs/wiki_mem_12k/last.pt --device cpu --n-seq 4 --out <dir>/p0.json
    python -B phase0_benefit.py ... --extra-val X:/Datasets/wiki_world_mix/val_world.bin   (also the world documents)

Per position t of a fixed set of val windows (--n-seq x --seq, random offsets from --seed), in eval mode with K passes and
dropless MoE capacity (every routed token computed: a position's result does not depend on the rest of the batch):

  b_t         = CE_t(memory off: mem.ko_shared = mem.ko_private = True) - CE_t(memory on)     (> 0: the memory helped)
  b_shared_t  = CE_t(ko_shared) - CE_t(on),   b_private_t = CE_t(ko_private) - CE_t(on)

1. concentration: the share of the total POSITIVE benefit sum(max(b, 0)) and of the NET benefit sum(b) held by the top x%
   of positions (x = 1, 5, 10, 20, 30, 50), the fraction of positions with b > 0, and the bottom-x% share of the negative
   mass (a symmetric, noise-like b gives the same concentration on both sides).
2. benefit by position type (types of the INPUT token at t, i.e. what a gate at t can see; overlapping): token frequency
   decile (occurrence-weighted deciles of --freq-tokens train tokens: d0 = the most frequent 10% of the token mass, d9 =
   the rarest), numbers, capitalised word starts, positions 1-3 after a capitalised / rare (decile >= --rare-decile)
   token, punctuation, whitespace, EOT, the first 32 positions after an EOT, the first 32 positions of the window (no
   EOT before them in the window: the model sees no context either), ordinary text (none of these). Mean b and the share
   of the total. Also the frequency decile of the TARGET token (diagnostic only: not visible to a gate).
3. cheap features (all causal, detached; the per-read ones at the reads k = 0 .. K-2 that feed the next pass):
   nu_l (concept log-variance, (mean nu_l + 6)/10 as in u), dmu_l (concept change between passes, u's 2nd component),
   delta (thought change log1p(|pbar_k - pbar_k-1| / |pbar_k|), pbar = the pooled normalised workspace), ent (normalised
   next-token entropy of pass k's read-out), gbar (the model's own content-gate mean over levels x heads of the shared
   read), each at read r0 / r1 and the mean over reads; surprise sigma_t = CE at t-1 of pass 0's read-out (filled with
   its mean at document starts, plus the indicator doc_start), a decaying max-surprise trace tr_t = max(sigma_t,
   eta tr_t-1) (eta = --trace-decay, reset after EOT). ROC AUC for "b_t in the top 20%" per feature, and a logistic
   regression (numpy Newton, L2) per single feature and on all of them, cross-validated over --folds folds BY SEQUENCE
   (pooled out-of-fold AUC). Extras (not in the GO decision): log position in the document, log train frequency of the
   input token.
4. GO if the top 20% hold >= --go-share (0.60) of the positive benefit AND the best cross-validated AUC (requested
   features) >= --go-auc (0.75).

Code: --code auto loads the current fruit-fly-sim modules and falls back to plan_v2/baseline_mem_v3 (a pristine copy that
matches the plan-12/14 checkpoints) if they fail to import, to load the checkpoint exactly, or to run it; --code baseline
/ current forces one. Writes JSON (--out), a .txt table and a .npz of the per-position arrays. Never writes next to the
checkpoint.
"""
import argparse
import json
import math
import os
import sys
import time
import unicodedata

if "--threads" in sys.argv[:-1] or not any(a.startswith("--threads") for a in sys.argv):  # before torch loads OpenMP
    _t = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv[:-1] else "4"
    if _t != "0":
        os.environ.setdefault("OMP_NUM_THREADS", _t)
        os.environ.setdefault("MKL_NUM_THREADS", _t)
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "plan_v2" / "baseline_mem_v3"
TOPS = (1, 5, 10, 20, 30, 50)
CODE_MODULES = ("loop_brain", "memory", "spiking_lm")


# ---------------------------------------------------------------- code + checkpoint
def _import_code(where):
    """Import LoopBrain / RoutedMLP / SpikingLM from `where` (a directory), purging earlier copies of the modules."""
    for name in list(sys.modules):
        if name in CODE_MODULES:
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in (HERE, BASELINE)]
    sys.path.insert(0, str(where))
    import loop_brain
    import spiking_lm
    return loop_brain.LoopBrain, spiking_lm.RoutedMLP, spiking_lm.SpikingLM


def build_model(a, ck, run_cfg, where):
    """eval_world.load_model: LoopBrain(**cfg) + load_state_dict, the trainer's attributes from config.json."""
    LoopBrain, RoutedMLP, SpikingLM = _import_code(where)
    cfg = ck["cfg"]
    model = (LoopBrain if cfg.get("arch") == "loop" else SpikingLM)(**cfg)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    for attr in ("mem_readdr_every", "mem_readdr_start", "mem_margin_passes"):
        if attr in run_cfg and hasattr(model, attr):
            setattr(model, attr, run_cfg[attr])
    for mod in model.modules():
        if isinstance(mod, RoutedMLP):
            mod.noise, mod.eval_noise = 0.0, run_cfg.get("eval_route_noise", 0.0)
            if a.capacity == "dropless":
                mod.cap = 0
    mem = getattr(model, "mem", None)
    if mem is not None:
        mem.write_on = False
    model.to(a.device).eval()
    model.K = a.k
    return model, list(missing), list(unexpected)


def load_model(a):
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cpath = Path(a.config) if a.config else Path(a.ckpt).parent / "config.json"
    run_cfg = json.loads(cpath.read_text()) if cpath.exists() else {}
    tries = {"auto": [("current", HERE), ("baseline", BASELINE)], "current": [("current", HERE)],
             "baseline": [("baseline", BASELINE)]}[a.code]
    errors = {}
    for name, where in tries:
        try:
            model, missing, unexpected = build_model(a, ck, run_cfg, where)
            if missing or unexpected:
                raise RuntimeError(f"state dict mismatch: missing {missing[:6]} ({len(missing)}), "
                                   f"unexpected {unexpected[:6]} ({len(unexpected)})")
            if getattr(model, "mem", None) is None or getattr(model, "hier", None) is None:
                raise RuntimeError("the checkpoint has no memory / concept hierarchy: nothing to analyse")
            with torch.no_grad():   # smoke test: one short forward with every knockout this script uses
                x = torch.zeros(1, 16, dtype=torch.long, device=a.device)
                for ks, kp in ((False, False), (True, True)):
                    model.mem.ko_shared, model.mem.ko_private = ks, kp
                    model.run(x, a.k, every_tick=True, raw=True)
                model.mem.ko_shared = model.mem.ko_private = False
            return model, ck.get("step"), run_cfg, name, where, errors
        except Exception as ex:  # noqa: BLE001 - any failure of this code version -> the next one
            errors[name] = f"{type(ex).__name__}: {ex}"
            print(f"[code {name}] failed: {errors[name]}", flush=True)
    raise SystemExit(f"could not load {a.ckpt}: {errors}")


# ---------------------------------------------------------------- one batch
class Capture:
    """Forward hooks (no file edits): the pooled workspace pbar at every concept-head call and the shared-read gates."""

    def __init__(self, model):
        self.model, self.on = model, False
        self.pbar, self.gates = [], []
        self.h = model.hier.register_forward_hook(self._hier)
        mem = model.mem
        self._orig = mem.read_shared   # the bound method; an instance attribute shadows it

        def read_shared(*args, **kw):
            out = self._orig(*args, **kw)
            if self.on and isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                self.gates.append(out[1].detach().float())          # (N, L, H)
            return out
        mem.read_shared = read_shared

    def _hier(self, mod, inp, out):
        if self.on:
            p = inp[0].detach().float()
            self.pbar.append(p.view(*p.shape[:-1], -1, 64).mean(-2))  # (B, T, 64) = n_s(s).mean(-2)

    def reset(self, on):
        self.on, self.pbar, self.gates = on, [], []


@torch.no_grad()
def forward_ce(model, x, y, a, want=False):
    """CE per position of the final pass (B, T); with want also the per-pass entropies and pass-0 CE."""
    outs = model.run(x, a.k, every_tick=True, raw=True)
    ents, ce0 = [], None
    for k, xk in enumerate(outs):
        if not want and k < len(outs) - 1:
            continue
        lp = model.readout_lp(xk).float()
        if want:
            ents.append((-(lp.exp() * lp).sum(-1) / math.log(lp.size(-1))).cpu())
        ce = -lp.gather(-1, y[..., None]).squeeze(-1)
        if k == 0:
            ce0 = ce.cpu()
        del lp
    return ce.cpu(), ents, ce0


def run_batch(model, cap, x, y, a):
    mem = model.mem
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda" and a.amp == "bf16")
    res = {}
    with amp:
        cap.reset(True)
        res["on"], ents, ce0 = forward_ce(model, x, y, a, want=True)
        think_u = [u.detach().float().cpu() for u in model.think_u]
        pbar, gates = [p.cpu() for p in cap.pbar], [g.cpu() for g in cap.gates]
        cap.reset(False)
        for name, ks, kp in (("off", True, True), ("ko_shared", True, False), ("ko_private", False, True)):
            mem.ko_shared, mem.ko_private = ks, kp
            try:
                res[name] = forward_ce(model, x, y, a)[0]
            finally:
                mem.ko_shared = mem.ko_private = False
    return res, ents, ce0, think_u, pbar, gates


def per_read_features(ents, think_u, pbar, gates, K, B, T, L):
    """dict name -> (B, T) for the reads k = 0 .. K-2 (r0, r1, mean over reads)."""
    reads = list(range(max(1, K - 1)))
    pbar = pbar[-K:]                      # (a read0 call before pass 1 would come first)
    nr = len(reads)
    gates = gates[-nr:] if len(gates) >= nr else []
    per = {}                              # name -> list over reads of (B, T)
    for k in reads:
        u = think_u[k]                    # (B, T, 2L)
        for l in range(L):
            per.setdefault(f"nu{l + 1}", []).append(u[..., 2 * l])
            per.setdefault(f"dmu{l + 1}", []).append(u[..., 2 * l + 1])
        if k == 0:
            dl = torch.zeros(B, T)
        else:
            dl = torch.log1p((pbar[k] - pbar[k - 1]).norm(dim=-1) / pbar[k].norm(dim=-1).clamp(min=1e-6))
        per.setdefault("delta", []).append(dl)
        per.setdefault("ent", []).append(ents[k])
        if gates:
            per.setdefault("gbar", []).append(gates[k].mean((1, 2)).view(B, T))
    out = {}
    for n, v in per.items():
        out[f"{n}_r0"] = v[0]
        if len(v) > 1:
            out[f"{n}_r1"] = v[1]
        out[f"{n}_mean"] = torch.stack(v).mean(0)
    return out


# ---------------------------------------------------------------- statistics
def auc(score, label):
    """ROC AUC (Mann-Whitney, ties averaged)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0 or not np.isfinite(score).all():
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    ranks = (first + (cnt + 1) / 2.0)[inv]
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit_lr(X, y, l2=1.0, iters=40):
    """L2 logistic regression by Newton steps; X standardised by the caller. Returns w (d + 1, bias last)."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = np.zeros(Xb.shape[1])
    R = np.eye(Xb.shape[1]) * l2
    R[-1, -1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + R @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def cv_lr_auc(X, y, groups, folds, seed=0):
    """Pooled out-of-fold AUC of the logistic regression; folds by group (sequence). Also the full-data weights."""
    ug = np.unique(groups)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(ug)
    nf = max(2, min(folds, len(ug)))
    fold_of = {g: i % nf for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    oof = np.full(len(y), np.nan)
    for i in range(nf):
        tr, te = f != i, f == i
        if y[tr].min() == y[tr].max():
            continue
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8
        w = fit_lr((X[tr] - m) / s, y[tr].astype(float))
        oof[te] = np.concatenate([(X[te] - m) / s, np.ones((te.sum(), 1))], 1) @ w
    ok = np.isfinite(oof)
    m, s = X.mean(0), X.std(0) + 1e-8
    w_all = fit_lr((X - m) / s, y.astype(float))
    return auc(oof[ok], y[ok]), w_all, nf


def concentration(b):
    b = np.asarray(b, float)
    n = len(b)
    srt = np.sort(b)[::-1]
    pos = np.clip(srt, 0, None)
    neg = np.clip(-srt[::-1], 0, None)        # the most negative first
    tp, tn, tnet = pos.sum(), neg.sum(), b.sum()
    out = dict(n=n, mean=float(b.mean()), frac_pos=float((b > 0).mean()), frac_zero=float((b == 0).mean()),
               sum_pos=float(tp), sum_neg=float(tn), sum_net=float(tnet), top={})
    for x in TOPS:
        k = max(1, int(math.ceil(n * x / 100)))
        out["top"][x] = dict(share_pos=float(pos[:k].sum() / tp) if tp > 0 else float("nan"),
                             share_net=float(srt[:k].sum() / tnet) if tnet != 0 else float("nan"),
                             bottom_share_neg=float(neg[:k].sum() / tn) if tn > 0 else float("nan"))
    return out


# ---------------------------------------------------------------- token types
def token_tables(tok, V):
    txt = [tok.decode([i]) for i in range(V)]
    is_num = np.array([any(c.isdigit() for c in s) for s in txt])
    stripped = [s.strip() for s in txt]
    is_ws = np.array([len(s) > 0 and len(st) == 0 for s, st in zip(txt, stripped)])
    is_punct = np.array([len(st) > 0 and all(unicodedata.category(c)[0] in "PS" for c in st) for st in stripped])
    lead_space = np.array([s.startswith(" ") for s in txt])
    upper = np.array([st[:1].isupper() for st in stripped])
    ends_nl = np.array(["\n" in s for s in txt])
    return dict(txt=txt, num=is_num, ws=is_ws, punct=is_punct, lead_space=lead_space, upper=upper, nl=ends_nl)


def train_counts(path, V, n_tokens, dtype, seed=0, chunk=1 << 20):
    data = np.memmap(path, dtype=dtype, mode="r")
    n_chunks = max(1, int(n_tokens) // chunk)
    rng = np.random.RandomState(seed + 99)
    cnt = np.zeros(V, np.int64)
    for off in rng.randint(0, max(1, len(data) - chunk), n_chunks):
        cnt += np.bincount(np.asarray(data[off:off + chunk], dtype=np.int64), minlength=V)[:V]
    return cnt


def freq_deciles(cnt):
    """Occurrence-weighted deciles: d0 = the most frequent token types holding the first 10% of the token mass."""
    order = np.argsort(-cnt, kind="stable")
    mass = cnt[order].astype(float)
    before = np.concatenate([[0.0], np.cumsum(mass)[:-1]]) / max(1.0, mass.sum())
    dec = np.empty(len(cnt), np.int64)
    dec[order] = np.minimum(9, (before * 10).astype(np.int64))
    dec[cnt == 0] = 9
    return dec


def position_types(X, Y, eot, tt, dec, rare_dec):
    """Boolean masks (N_seq, T) per type of the input token at t."""
    S, T = X.shape
    num, ws, punct = tt["num"][X], tt["ws"][X], tt["punct"][X]
    is_eot = X == eot
    prev = np.concatenate([np.full((S, 1), -1), X[:, :-1]], 1)
    prev_break = (prev == -1) | (prev == eot) | np.where(prev >= 0, tt["nl"][np.clip(prev, 0, None)], False)
    word_start = tt["lead_space"][X] | prev_break
    cap = word_start & tt["upper"][X] & ~is_eot
    rare = (dec[X] >= rare_dec) & ~is_eot
    trig = cap | rare
    after = np.zeros_like(trig)
    for j in (1, 2, 3):
        after[:, j:] |= trig[:, :-j]
    # positions since the last EOT (in the window); -1 = no EOT before t in this window
    since = np.full((S, T), -1, np.int64)
    for s in range(S):
        last = -1
        for t in range(T):
            since[s, t] = t - last if last >= 0 else -1
            if X[s, t] == eot:
                last = t
    first32_eot = (since >= 1) & (since <= 32)
    first32_win = (since < 0) & (np.arange(T)[None] < 32)
    types = {"EOT": is_eot, "number": num & ~is_eot, "cap_word_start": cap, "rare_token": rare,
             "after_cap_or_rare_1-3": after & ~is_eot, "punctuation": punct, "whitespace": ws,
             "first32_after_EOT": first32_eot, "first32_of_window": first32_win}
    special = np.zeros_like(is_eot)
    for m in types.values():
        special |= m
    types["ordinary"] = ~special
    for d in range(10):
        types[f"freq_d{d}"] = dec[X] == d
    for d in range(10):
        types[f"target_freq_d{d}"] = dec[Y] == d
    pos_doc = np.where(since >= 0, since, np.arange(T)[None] + 1)   # tokens of context in this document (>= 1)
    return types, pos_doc


def type_table(b, types):
    tot, totp = b.sum(), np.clip(b, 0, None).sum()
    out = {}
    for n, m in types.items():
        m = m.ravel()
        if m.sum() == 0:
            out[n] = dict(n=0, frac=0.0, mean=float("nan"), share_net=0.0, share_pos=0.0, frac_pos=float("nan"))
            continue
        bb = b[m]
        out[n] = dict(n=int(m.sum()), frac=float(m.mean()), mean=float(bb.mean()),
                      share_net=float(bb.sum() / tot) if tot != 0 else float("nan"),
                      share_pos=float(np.clip(bb, 0, None).sum() / totp) if totp > 0 else float("nan"),
                      frac_pos=float((bb > 0).mean()))
    return out


# ---------------------------------------------------------------- main
def analyse(name, X, Y, R, feats, extras, groups, eot, tt, dec, a, R_alt=None, bs_alt=0):
    """R: dict variant -> (S, T) CE arrays. Returns the result dict and the table lines."""
    b = {"full": R["off"] - R["on"], "shared": R["ko_shared"] - R["on"], "private": R["ko_private"] - R["on"]}
    types, pos_doc = position_types(X, Y, eot, tt, dec, a.rare_decile)
    extras = dict(extras, log_pos_in_doc=np.log(pos_doc.astype(float)))
    res = dict(n_seq=int(X.shape[0]), seq=int(X.shape[1]), n_pos=int(X.size),
               ce={k: float(v.mean()) for k, v in R.items()}, concentration={}, types={}, auc={})
    L = []
    L.append(f"=== {name}: {X.shape[0]} windows x {X.shape[1]} positions = {X.size:,} positions; K={a.k}, "
             f"capacity {a.capacity}")
    L.append("mean CE: " + "  ".join(f"{k} {v:.4f}" for k, v in res["ce"].items())
             + f"   (off - on = {res['ce']['off'] - res['ce']['on']:+.4f})")
    if R_alt is not None:  # the same windows again at another batch size: numerical noise in b_t
        nc = R_alt["on"].shape[0]
        res["noise"] = {"windows": nc, "batch_alt": bs_alt, "batch": a.batch}
        L.append(f"0. reproducibility: the first {nc} windows again at batch size {bs_alt} (vs {a.batch}); the CE is "
                 f"deterministic for a fixed batch, but expert-routing near-ties can flip with the GEMM shapes")
        for n in R_alt:
            d = np.abs(R_alt[n] - R[n][:nc])
            res["noise"][f"ce_{n}"] = dict(mean_abs=float(d.mean()), max_abs=float(d.max()), frac_gt_0p01=float((d > 0.01).mean()))
            L.append(f"  CE {n:10s} |diff| mean {d.mean():.4f} max {d.max():.3f}  frac > 0.01: {(d > 0.01).mean():.3f}")
        for v, (hi, lo) in dict(full=("off", "on"), shared=("ko_shared", "on"), private=("ko_private", "on")).items():
            b1, b2 = (R[hi][:nc] - R[lo][:nc]).ravel(), (R_alt[hi] - R_alt[lo]).ravel()
            nz = b2 - b1
            r = float(np.corrcoef(b1, b2)[0, 1]) if b1.std() > 0 and b2.std() > 0 else float("nan")
            res["noise"][f"b_{v}"] = dict(retest_corr=r, mean_abs_b=float(np.abs(b1).mean()),
                                          mean_abs_diff=float(np.abs(nz).mean()), frac_diff_gt_0p01=float((np.abs(nz) > 0.01).mean()))
            L.append(f"  b_{v:8s} test-retest corr {r:.3f}   mean|b| {np.abs(b1).mean():.4f}   mean|b - b'| "
                     f"{np.abs(nz).mean():.4f}   frac |b - b'| > 0.01: {(np.abs(nz) > 0.01).mean():.3f}")
        L.append("")
    L.append("1. concentration of b_t = CE(off) - CE(on): share of the positive benefit (pos) / net benefit (net) held "
             "by the top x% of positions; neg = the bottom x%'s share of the negative mass")
    for v, bv in b.items():
        c = concentration(bv.ravel())
        res["concentration"][v] = c
        L.append(f"  b_{v:8s} mean {c['mean']:+.4f}  frac b>0 {c['frac_pos']:.3f}  sum+ {c['sum_pos']:.1f}  "
                 f"sum- {c['sum_neg']:.1f}  net {c['sum_net']:+.1f}")
        L.append("     top x%:  " + "  ".join(f"{x:>2d}%: pos {c['top'][x]['share_pos']:.3f} net {c['top'][x]['share_net']:+.3f}"
                                             f" neg {c['top'][x]['bottom_share_neg']:.3f}" for x in TOPS[:3]))
        L.append("              " + "  ".join(f"{x:>2d}%: pos {c['top'][x]['share_pos']:.3f} net {c['top'][x]['share_net']:+.3f}"
                                              f" neg {c['top'][x]['bottom_share_neg']:.3f}" for x in TOPS[3:]))
    L.append("")
    L.append("2. benefit by position type (input token at t; types overlap; share = of the total over all positions)")
    L.append(f"  {'type':24s} {'frac':>6s} | " + " | ".join(f"{'b_' + v + ' mean':>13s} {'net%':>6s} {'pos%':>6s}"
                                                         for v in b))
    for v, bv in b.items():
        res["types"][v] = type_table(bv.ravel(), types)
    for n in types:
        r0 = res["types"]["full"][n]
        L.append(f"  {n:24s} {r0['frac']:6.3f} | " + " | ".join(
            f"{res['types'][v][n]['mean']:+13.4f} {100 * res['types'][v][n]['share_net']:6.1f} "
            f"{100 * res['types'][v][n]['share_pos']:6.1f}" for v in b))
    L.append("")
    # ---- features
    names = list(feats)
    F = np.stack([feats[n].ravel() for n in names], 1).astype(float)
    keep = [i for i in range(F.shape[1]) if np.isfinite(F[:, i]).all() and F[:, i].std() > 0]
    dropped = [names[i] for i in range(len(names)) if i not in keep]
    names, F = [names[i] for i in keep], F[:, keep]
    E = np.stack([extras[n].ravel() for n in extras], 1).astype(float)
    g = np.repeat(groups, X.shape[1])
    L.append(f"3. ROC AUC for 'b_t in the top {a.top_label}%' (AUC: raw, > 0.5 = high feature -> high benefit; "
             f"cv: 1-feature logistic regression, {a.folds}-fold by sequence)")
    if dropped:
        L.append(f"  (constant or non-finite, dropped: {', '.join(dropped)})")
    for v, bv in b.items():
        bb = bv.ravel()
        k = max(1, int(math.ceil(len(bb) * a.top_label / 100)))
        y = np.zeros(len(bb), bool)
        y[np.argsort(-bb, kind="stable")[:k]] = True
        r = dict(per_feature={}, extras={})
        if v == "full":
            for i, n in enumerate(names):
                r["per_feature"][n] = dict(auc=auc(F[:, i], y), cv=cv_lr_auc(F[:, i:i + 1], y, g, a.folds)[0])
            for i, n in enumerate(extras):
                r["extras"][n] = dict(auc=auc(E[:, i], y), cv=cv_lr_auc(E[:, i:i + 1], y, g, a.folds)[0])
        cv_all, w_all, nf = cv_lr_auc(F, y, g, a.folds)
        cv_ext = cv_lr_auc(np.concatenate([F, E], 1), y, g, a.folds)[0]
        r.update(lr_all_cv=cv_all, lr_all_weights=dict(zip(names + ["bias"], map(float, w_all))),
                 lr_all_plus_extras_cv=cv_ext, folds=nf)
        best_single = max([d["cv"] for d in r["per_feature"].values() if np.isfinite(d["cv"])] or [float("nan")])
        r["best_cv"] = float(np.nanmax([cv_all, best_single])) if np.isfinite([cv_all, best_single]).any() else float("nan")
        res["auc"][v] = r
    full = res["auc"]["full"]
    L.append(f"  {'feature':16s} {'AUC':>6s} {'cv':>6s}")
    for n, d in sorted(full["per_feature"].items(), key=lambda kv: -abs(kv[1]["auc"] - 0.5)):
        L.append(f"  {n:16s} {d['auc']:6.3f} {d['cv']:6.3f}")
    for n, d in full["extras"].items():
        L.append(f"  {n + ' (extra)':16s} {d['auc']:6.3f} {d['cv']:6.3f}")
    for v in b:
        r = res["auc"][v]
        L.append(f"  logistic regression, all {len(names)} features, cv AUC (b_{v}): {r['lr_all_cv']:.3f}   "
                 f"+ extras: {r['lr_all_plus_extras_cv']:.3f}   ({r['folds']} folds)")
    top = sorted(((abs(w), n, w) for n, w in full["lr_all_weights"].items() if n != "bias"), reverse=True)[:8]
    L.append("  largest standardised LR weights (b_full): " + ", ".join(f"{n} {w:+.2f}" for _, n, w in top))
    c20 = res["concentration"]["full"]["top"][20]["share_pos"]
    best = full["best_cv"]
    go = bool(c20 >= a.go_share and best >= a.go_auc)
    res["go"] = dict(go=go, top20_share_pos=c20, best_cv_auc=best, go_share=a.go_share, go_auc=a.go_auc)
    L.append("")
    L.append(f"{'GO' if go else 'NO-GO'} ({name}): top 20% hold {c20:.3f} of the positive benefit (need >= {a.go_share}); "
             f"best cross-validated AUC {best:.3f} (need >= {a.go_auc})")
    return res, L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=str(HERE.parent / "runs" / "wiki_mem_12k" / "last.pt"))
    ap.add_argument("--config", default="", help="the run's config.json (default: next to --ckpt)")
    ap.add_argument("--data", default="X:/Datasets/wikipedia_en_20231101_wiki16k",
                    help="dataset dir: val.bin, train.bin (frequency counts), tokenizer.json, meta.json")
    ap.add_argument("--extra-val", default="", help="also analyse these .bin files (comma separated), e.g. "
                                                    "X:/Datasets/wiki_world_mix/val_world.bin (same tokenizer)")
    ap.add_argument("--code", default="auto", choices=["auto", "current", "baseline"])
    ap.add_argument("--k", type=int, default=6, help="loop passes")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"], help="CUDA autocast (eval_world: bf16)")
    ap.add_argument("--capacity", default="dropless", choices=["dropless", "train"])
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=0, help="windows per forward (0 = 8 on cuda, 2 on cpu)")
    ap.add_argument("--freq-tokens", type=float, default=20e6, help="train.bin tokens counted for the deciles")
    ap.add_argument("--rare-decile", type=int, default=9, help="'rare' = input-token frequency decile >= this")
    ap.add_argument("--trace-decay", type=float, default=0.88, help="eta of the max-surprise trace (~8 tokens)")
    ap.add_argument("--top-label", type=float, default=20.0, help="label: b_t in the top x%%")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--noise-check", type=int, default=8,
                    help="re-run the first N windows at another batch size (1, or 2 if --batch 1): numerical noise "
                         "and test-retest reliability of b_t (0 = off)")
    ap.add_argument("--go-share", type=float, default=0.60)
    ap.add_argument("--go-auc", type=float, default=0.75)
    ap.add_argument("--out", default="", help="JSON path (default: NewBrain/phase0_evals/<run>_<ckpt>_n<n>.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (0 = torch default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(min(a.threads, 4))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    a.batch = a.batch or (8 if a.device == "cuda" else 2)
    t0 = time.time()
    torch.manual_seed(a.seed)
    data = Path(a.data)
    meta = json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else {}
    eot, dt = int(meta.get("eot_id", 0)), np.dtype(meta.get("dtype", "uint16"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))

    model, step, run_cfg, code, where, code_errors = load_model(a)
    mem = model.mem
    V = model.cfg["vocab"]
    Lv = model.hier.L
    print(f"checkpoint {a.ckpt} (step {step}), code {code} ({where}), K={a.k}, device {a.device}, memory S={mem.S} "
          f"R={mem.R} L={mem.L} read0={mem.read0} frozen={mem.is_frozen}, readdr every {model.mem_readdr_every} "
          f"start {model.mem_readdr_start} ({time.time() - t0:.0f}s)", flush=True)
    cap = Capture(model)

    cnt = train_counts(data / "train.bin", V, a.freq_tokens, dt, a.seed)
    dec = freq_deciles(cnt)
    tt = token_tables(tok, V)
    logfreq = np.log1p(cnt.astype(float))
    print(f"train counts: {cnt.sum():,} tokens ({time.time() - t0:.0f}s)", flush=True)

    sets = [("wiki_val", data / "val.bin")] + [(Path(p).stem, Path(p)) for p in a.extra_val.split(",") if p.strip()]
    results, lines, arrays = {}, [], {}
    for set_name, path in sets:
        vals = np.memmap(path, dtype=dt, mode="r")
        if set_name == "wiki_val" and "val_wiki_tokens" in meta:
            vals = vals[:meta["val_wiki_tokens"]]
        rng = np.random.RandomState(a.seed)
        offs = rng.randint(0, len(vals) - a.seq - 1, a.n_seq)
        XY = np.stack([np.asarray(vals[o:o + a.seq + 1], dtype=np.int64) for o in offs])
        X, Y = XY[:, :-1], XY[:, 1:]
        R = {n: [] for n in ("on", "off", "ko_shared", "ko_private")}
        feats, sur0 = {}, []
        t1 = time.time()
        for i in range(0, a.n_seq, a.batch):
            x = torch.from_numpy(X[i:i + a.batch]).to(a.device)
            y = torch.from_numpy(Y[i:i + a.batch]).to(a.device)
            res, ents, ce0, think_u, pbar, gates = run_batch(model, cap, x, y, a)
            for n in R:
                R[n].append(res[n].numpy())
            B, T = x.shape
            pf = per_read_features(ents, think_u, pbar, gates, a.k, B, T, Lv)
            for n, v in pf.items():
                feats.setdefault(n, []).append(v.numpy())
            sur0.append(ce0.numpy())
            done = min(i + a.batch, a.n_seq)
            print(f"  [{set_name}] {done}/{a.n_seq} windows, {time.time() - t1:.0f}s "
                  f"(on {np.concatenate(R['on']).mean():.4f} off {np.concatenate(R['off']).mean():.4f})", flush=True)
        R = {n: np.concatenate(v) for n, v in R.items()}
        feats = {n: np.concatenate(v) for n, v in feats.items()}
        if not any(n.startswith("gbar") for n in feats):
            print("  (no shared-read gates captured: gbar features unavailable)", flush=True)
        # surprise sigma_t = CE of pass 0 at t-1 (target idx[t]); doc starts (t = 0 or idx[t-1] = EOT) -> mean + flag
        ce0 = np.concatenate(sur0)
        S_, T_ = X.shape
        sig = np.full((S_, T_), np.nan)
        sig[:, 1:] = ce0[:, :-1]
        start = np.zeros((S_, T_), bool)
        start[:, 0] = True
        start[:, 1:] = X[:, :-1] == eot
        sig[start] = np.nan
        sig = np.where(np.isnan(sig), np.nanmean(sig), sig)
        tr = np.zeros_like(sig)
        for t in range(T_):
            prev = np.where(start[:, t], 0.0, tr[:, t - 1] if t else 0.0)
            tr[:, t] = np.maximum(sig[:, t], a.trace_decay * prev)
        feats.update(surprise=sig, surprise_trace=tr, doc_start=start.astype(float))
        extras = dict(log_train_freq=logfreq[X])
        R_alt, bs_alt = None, 1 if a.batch > 1 else 2
        nc = min(a.noise_check, a.n_seq)
        if nc and (bs_alt == 1 or nc >= 2):
            nc -= nc % bs_alt
            R_alt = {n: [] for n in R}
            for i in range(0, nc, bs_alt):
                x = torch.from_numpy(X[i:i + bs_alt]).to(a.device)
                y = torch.from_numpy(Y[i:i + bs_alt]).to(a.device)
                rr = run_batch(model, cap, x, y, a)[0]
                for n in R_alt:
                    R_alt[n].append(rr[n].numpy())
            R_alt = {n: np.concatenate(v) for n, v in R_alt.items()}
        r, L = analyse(set_name, X, Y, R, feats, extras, np.arange(S_), eot, tt, dec, a, R_alt, bs_alt)
        r["time_s"] = time.time() - t1
        results[set_name], lines = r, lines + L + [""]
        arrays.update({f"{set_name}/X": X, f"{set_name}/Y": Y, **{f"{set_name}/ce_{n}": v for n, v in R.items()},
                       **{f"{set_name}/feat_{n}": v for n, v in feats.items()}})
        print("\n".join(L), flush=True)

    table = "\n".join(lines)
    ck = Path(a.ckpt)
    out = Path(a.out) if a.out else HERE.parent / "phase0_evals" / f"{ck.parent.name}_{ck.stem}_n{a.n_seq}.json"
    if out.resolve().parent == ck.resolve().parent:
        raise SystemExit("refusing to write next to the checkpoint")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ckpt=str(ck), step=step, code=code, code_dir=str(where), code_errors=code_errors,
                                   args=vars(a), results=results, table=table, runtime_s=time.time() - t0),
                              indent=1, default=float))
    out.with_suffix(".txt").write_text(table + "\n", encoding="utf-8")
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    print(f"wrote {out} (+ .txt, .npz) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
