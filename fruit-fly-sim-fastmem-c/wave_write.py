"""The wave memory's pluggable WRITE-SELECTION policy (README_memory_api.md, "wave memory").

    policy.select(ce, H, V, B, T, idx=None, eot=None) -> (tok (n,) long, strength (n,))
        which tokens of this micro-batch are written (once per token), from brain signals only: the final read-out's
        per-token CE and entropy (no address comes from here: the address is the thought's code)
    policy.plan(tok, strength, recall) -> (strength (n,), entries (n, E) bool or None)
        per write: its strength and which entry locations it uses. recall = dict(open (n,) gate value of the read at
        the written pass, cos (n,) cosine of that read with the write target, ok (n,) 1 = a confirmed recall (open
        and cos > --wave-confirm), E entries); e.g. "count 2 on surprise" or "a second entry path after a failed
        recall" (POLICIES["wave"], plan 26 section 8.4)
    policy.log() -> dict        the write mix (selected vs all: CE, entropy, sequences touched, per-source counts)

Known risk (plan 26 / the blob runs): selection by raw surprise gets swamped by Wikipedia tokens. Hence a per-sequence
quota, the entropy-normalised default, the logged mix, and `source_fn` (optional: idx (B, T) -> (B,) source labels,
set by the trainer when its data can tell the sources apart). Source labels are for LOGGING ONLY: they are read after
the selection is made and never enter the selection, the strength or any training signal.

POLICIES maps --wave-write names to classes. Rare-fact retention is unsolved (plan 26 section 8.4): the spec decides.
"""
import math

import torch
import torch.nn as nn

INF = float("inf")


class WritePolicy(nn.Module):
    def __init__(self, write_w=32, per_seq=4, seed=1234, n_bins=32, min_after_eot=8):
        super().__init__()
        self.write_w, self.per_seq, self.seed, self.min_after_eot = int(write_w), int(per_seq), int(seed), min_after_eot
        self.register_buffer("ch_mean", torch.zeros(n_bins))       # (the entropy-binned CE baseline; surprise_h)
        self.register_buffer("ch_n", torch.zeros(n_bins, dtype=torch.long))
        self.frozen = False                                         # (train-only: the baseline does not move)
        self.source_fn = None
        self.n_calls = 0                                            # host (extra state: the random policy's stream)
        self._mix = None

    # ------------------------------------------------------------------ the interface
    def score(self, ce, H, V):
        """(N,) higher = write first."""
        raise NotImplementedError

    def plan(self, tok, strength, recall):
        return strength, None

    @torch.no_grad()
    def select(self, ce, H, V, B, T, idx=None, eot=None, allow=None):
        """allow (N,) bool or None: the only eligible tokens (train-only: masked before the quota and the top-k)."""
        N = B * T
        dev = ce.device
        ce, H = ce.reshape(N).float(), H.reshape(N).float()
        sc = self.score(ce, H, V).view(B, T)
        elig = torch.ones(B, T, dtype=torch.bool, device=dev)
        if idx is not None and eot is not None:
            pos = torch.arange(T, device=dev).expand(B, T)
            last = torch.where(idx == eot, pos, torch.full_like(pos, -10 ** 6)).cummax(1).values
            elig = (idx != eot) & (pos - last >= self.min_after_eot)
        if allow is not None:
            elig = elig & allow.view(B, T)
        sc = torch.where(elig, sc, torch.full_like(sc, -INF))
        top = sc.topk(min(self.per_seq, T), -1)                     # the per-sequence quota
        cand = (torch.arange(B, device=dev)[:, None] * T + top.indices).flatten()
        cs = top.values.flatten()
        k = min(self.write_w, cand.numel())
        best = cs.topk(k)
        ok = best.values > -INF
        tok = cand[best.indices]
        strength = ok.float()
        self._log(ce, H, tok, ok, B, T, idx)
        self.n_calls += 1
        return tok, strength

    # ------------------------------------------------------------------ the write mix
    @torch.no_grad()
    def _log(self, ce, H, tok, ok, B, T, idx):
        okf = ok.float()
        seq = tok // T
        per = torch.zeros(B, device=ce.device).index_add_(0, seq, okf)
        full = lambda v: torch.full((), float(v), device=ce.device)      # (a fill, not a host-to-device copy)
        m = torch.stack([okf.sum(), (ce[tok] * okf).sum(), (H[tok] * okf).sum(), ce.sum(), H.sum(),
                         full(ce.numel()), (per > 0).float().sum(), full(B), (tok % T).float().mul(okf).sum() / T])
        self._mix = m if self._mix is None else self._mix + m
        if self.source_fn is not None and idx is not None:
            src = self.source_fn(idx).long()[seq]
            c = torch.zeros(int(src.max()) + 1 if src.numel() else 1, device=ce.device).index_add_(0, src, okf)
            self._src = c if getattr(self, "_src", None) is None or self._src.numel() != c.numel() else self._src + c

    def log(self, reset=True):
        if self._mix is None:
            return {}
        v = self._mix.tolist()
        n = max(v[0], 1e-9)
        out = dict(writes=v[0], write_ce=v[1] / n, write_ent=v[2] / n, all_ce=v[3] / max(v[5], 1),
                   all_ent=v[4] / max(v[5], 1), write_seq_frac=v[6] / max(v[7], 1), write_pos=v[8] / n)
        if getattr(self, "_src", None) is not None:
            out["write_by_source"] = self._src.tolist()
            if reset:
                self._src = None
        if reset:
            self._mix = None
        return out


class EntropySurprise(WritePolicy):
    """sigma = CE - C(H): CE above the running mean CE of tokens with the same predicted entropy (32 bins; the current
    baseline is used, then updated). A token is not surprising just because the text is hard (the blob's rule)."""

    def score(self, ce, H, V):
        nb = self.ch_mean.numel()
        b = (H / math.log(V) * nb).long().clamp(0, nb - 1)
        sig = ce - self.ch_mean[b]
        if not self.frozen:
            s_ = torch.zeros(nb, device=ce.device).index_add_(0, b, ce)
            c_ = torch.zeros(nb, device=ce.device).index_add_(0, b, torch.ones_like(ce))
            seen, first = c_ > 0, self.ch_n == 0
            mean = s_ / c_.clamp(min=1)
            self.ch_mean.copy_(torch.where(seen, torch.where(first, mean, self.ch_mean.lerp(mean, 0.01)), self.ch_mean))
            self.ch_n.add_(seen.long())
        return sig


class RawSurprise(WritePolicy):
    """The per-token CE itself (the known-swamped control)."""

    def score(self, ce, H, V):
        return ce


class RandomWrites(WritePolicy):
    """Uniformly random eligible tokens (a control). Deterministic: (seed, call count)."""

    def score(self, ce, H, V):
        g = torch.Generator(device=ce.device).manual_seed(self.seed * 1_000_003 + self.n_calls)
        return torch.rand(ce.shape, generator=g, device=ce.device)


class WaveSheetPolicy(EntropySurprise):
    """Plan 26 section 8.4 (the W4b winner): the entropy-normalised selection, then per write count 1 after a confirmed
    recall, 2 on surprise (the read failed: not open or cos(read, target) <= the confirm threshold), and on surprise
    also the path from the second entry location (the reader tries both)."""
    surprise_w = 2

    def plan(self, tok, strength, recall):
        ok = recall["ok"] > 0
        s = torch.where(ok, strength, strength * self.surprise_w)
        E = int(recall["E"])
        ent = torch.zeros(tok.numel(), E, dtype=torch.bool, device=tok.device)
        ent[:, 0] = True
        if E > 1:
            ent[:, 1] = ~ok
        return s, ent


class SpecPolicy(WritePolicy):
    """WaveSheet v2 spec W4 (the default): sigma = CE / (H + 1) at the final pass; candidates: sigma >= the micro-batch's
    90th percentile, position >= 1 (and not right after an EOT); per sequence the most surprising candidate, the best
    `write_w` (16) of those by sigma; strength 2 for the top `strong_k` (4), else 1. plan(): a confirmed recall is not
    written (strength 0: the confirm hook strengthened it); entry 0 only (the 2nd-entry write is the "wave" arm).
    v2.2 (per_seq > 1, --wave-writes-per-seq): the per_seq most surprising candidates per sequence, at least `gap` (8)
    tokens apart (per_seq greedy picks, each masking +-(gap - 1) around it: static shapes, no host sync); the
    micro-batch cap becomes write_w x per_seq; strength 2 for the top strong_k of all."""
    strong_k = 4
    gap = 8

    def score(self, ce, H, V):
        return ce / (H + 1.0)

    @torch.no_grad()
    def select(self, ce, H, V, B, T, idx=None, eot=None, allow=None):
        """allow (N,) bool or None: the only eligible tokens (train-only: masked before the quota and the top-k)."""
        N = B * T
        dev = ce.device
        ce, H = ce.reshape(N).float(), H.reshape(N).float()
        sig = self.score(ce, H, V).view(B, T)
        pos = torch.arange(T, device=dev).expand(B, T)
        elig = pos >= 1
        if idx is not None and eot is not None:
            last = torch.where(idx == eot, pos, torch.full_like(pos, -10 ** 6)).cummax(1).values
            elig = elig & (idx != eot) & (pos - last >= self.min_after_eot)
        if allow is not None:
            elig = elig & allow.view(B, T)
        k90 = max(1, int(0.9 * N))
        q90 = sig.reshape(-1).kthvalue(k90).values                     # (python int k: no host sync)
        sc = torch.where(elig & (sig >= q90), sig, torch.full_like(sig, -INF))
        P = max(1, int(self.per_seq))
        if P == 1:
            best = sc.max(1)                                            # (first index on ties)
            cand, vals = torch.arange(B, device=dev) * T + best.indices, best.values
        else:                             # (v2.2: the top P per sequence, a gap of `gap` tokens: P fixed greedy picks)
            ci, cv = [], []
            for _ in range(P):
                best = sc.max(1)
                ci.append(best.indices)
                cv.append(best.values)
                sc = torch.where((pos - best.indices[:, None]).abs() < self.gap, torch.full_like(sc, -INF), sc)
            cand = (torch.arange(B, device=dev)[:, None] * T + torch.stack(ci, 1)).flatten()
            vals = torch.stack(cv, 1).flatten()
        k = min(self.write_w * P, B * P)
        top = vals.topk(k)
        tok = cand[top.indices]
        ok = top.values > -INF
        strength = ok.float() * torch.where(torch.arange(k, device=dev) < self.strong_k, 2.0, 1.0)
        self._log(ce, H, tok, ok, B, T, idx)
        self.n_calls += 1
        return tok, strength

    def plan(self, tok, strength, recall):
        ok = recall["ok"] > 0
        ent = torch.zeros(tok.numel(), int(recall["E"]), dtype=torch.bool, device=tok.device)
        ent[:, 0] = True
        return torch.where(ok, torch.zeros_like(strength), strength), ent


POLICIES = dict(surprise_h=EntropySurprise, surprise=RawSurprise, random=RandomWrites, wave=WaveSheetPolicy,
                spec=SpecPolicy, delta=SpecPolicy)   # (delta: v2.2's write rule in wave_v21; the selection is the spec's)
