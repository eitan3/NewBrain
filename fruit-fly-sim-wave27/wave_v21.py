"""WaveSheet v2.1 (SPEC_v21_final.md): the global sheet + per-expert strips, the pre-read gate with the spec's
selection, Translate (I + D_e) with the factored D update, the write rule "only where read". --memory wave
--wave-mode v21 (the default).

LAYOUT (one flat key_syn / chain_bits array, the column-store code with a place offset; wave_columns.WaveColumns)
  global places 0..P_g-1 (P_g = --wave-side^2 = 256, the 16 x 16 torus of the global address, wave_address.py, fix A)
  strips: memory row m (0..S-1, S = Emax + 1; the shared expert's row Emax keeps its 4 places, unused by default) owns
          places P_g + 4 m + j, j < 4
  DUMMY = P_g + 4 S (3,332 at full size): all zero, never written or swept; a read there gives y = 0
  row_of [S] maps a brain expert id to its memory row (here the identity: this trainer's expert ids are contiguous,
  births append; the shared expert -> Emax)
PRIVATE ADDRESS P_e [S, 4, d] (x_hat space of the global address): place_p = argmax_j <x_hat, P_e[row, j]> -
  |P_e|^2 / 2 over the initialised j < Pinit[row] (DUMMY if Pinit[row] == 0); keys = the global address's residual
  codebooks on x_hat - P_e[row, j*] (one shared set of Q). Learning: before write_start, 1,024 sample tokens x their
  2 first routed pairs (first-sample init while Pinit < 4, rate max(1/32, 1/(1 + hits)), sample-hit conscience 0.05);
  after, the micro-batch's private writes (BMU with the write-load conscience 0.05 log((Wld_p + 1) / (mean + 1)),
  P += (x - P) / 128), one-hot GEMMs.
GATE h = silu(G1 [W_gx x_hat ; f_tok] + b) [N, 16]; bp = <Wp[row], h> + Up f_pair + bp_b[row], bg likewise; the
  shared pair (k = 6): its OWN global gate row (Wg / bg_b [Emax]), bp[:, 6] = -inf (no private read).
SELECTION: the spec's code (PART 2): tie-free int64 keys, exploration 5% of routed pairs (private) + 5% of tokens
  (global, all 7 pairs), three budgets K = 7N/2, K_pr = 3N, K_g = N/2, capacity C per memory row, C_sh = K_g, no drops
  after selection, unread = exactly 0.
READ: global on the K_g tokens of GT (4 entries, the clearest by chibar), private on the K_pr rows of PR (their place,
  1 entry); confidence c = clamp((z - 2) / 4, 0, 1) from per-sheet null statistics (bins by the winner place's load);
  y = c kappa_bar y_hat (unitless); y_pair = keep (vg y_g + vp y_p), + the capture leaf (E_pair = dL/dy_pair);
  Translate: X -> X + X D_e per memory row (D a buffer; its gradient into the D_grad buffer); o = exp(log_s) |x| out,
  the 0.5 |x| cap with a DETACHED scale. Pass 3 injects; passes 4-5 re-inject the cached o of the pairs that read.
WRITES (after_backward; dropped sequences write and confirm nothing): candidates = tokens that read (global or a
  private pair) and were not confirmed, sigma >= q90, one per sequence (v2.2: --wave-writes-per-seq 8, >= 8 tokens
  apart), 16 (x per seq) per micro-batch, strength 2 for the top 4;
  global iff the token read the global sheet (at entry 0, its keys; fix A's map learns on these writes), private: the
  first 2 read routed pairs by routing rank, at their read place with their keys. Targets normalize(-e_hat +
  max(h, 0) [c >= .5] y_hat). Confirmations at z >= --wave-z-conf with h >= theta.
D UPDATE (after_step, PART 3): the factored second-moment update, per memory row, wd 0, no cross-row coupling.
SWEEP: 8 global + 24 live-strip places per step.
v2.2 (SPEC_v22.md):
  --wave-write delta (default; "spec" = the v2.1 set rule): at a write, entry 0's trace (what the place recalls for the
    key) vs the target per (layer, column): target synapses cnt + 1 with the target phase (matching: the confirm + 1);
    the recalled-but-not-target unit's key synapses cnt - 1, only when z(entry 0) >= --wave-z-conf (WaveColumns.write_delta)
  --wave-readout norm (default; "raw" = v2.1): y_n = RMSNorm(y) g x c kappa_bar x s, s = 2 tanh(silu(xf . w_e + b_e) / 2)
    per (token, expert), then I + D_e; g / w_e / b_e in the memory's AdamW (decoder group), w_e / b_e copied on splits;
    a zero / no read -> exactly 0; log wmem_read_ratio = |o| / |xf| over the pairs that read
  --wave-strip-places J (= --wave-strip): strips of J places; J > 4: a private read takes the best min(4, J) places of
    the strip (by the prototype score, entry 0 = the argmax: the write place), each with its own residual keys, the
    clearest by chibar wins (the global sheet's entry mechanism); --wave-strip-entries overrides
  --wave-latent dl (default 2048; 0 = d): memories are dl-d latents. Write value (no learned encoder, no gradient):
    [the unit correction (d) | xf (d) | the workspace summary (--wave-side-dim) | correction @ a fixed random fill
    matrix (the rest)], each part unit-normalised, then normalised; the column store's decoder holds dl (16 columns x
    dl / 16; the key / chain bytes per place do NOT change: a memory is still 3 x 16 (unit, phase) codes). Read: the
    store's dl-d y -> RMSNorm, g [dl], c, s -> out = y_n[:d] + y_n D_e, D_e [dl, d] per row, init 0. The capture
    E_pair is dl-d; the correction is its first d dims.
  --wave-conf soft (default): c = sigmoid(2 (z - 3)); --wave-z-conf stays the erase / confirm threshold; log
    wmem_fire_rate = share of reads with c > 0.5 (the last recall)
"""
import contextlib
import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function as _rf

from memory_api import MemoryBase
from wave_address import Codes, WaveAddress
from wave_columns import F as NFAN, L as NLAY, W as NW, UC, UK, WaveColumns
from wave_write import SpecPolicy

LOW = torch.iinfo(torch.int64).min


@contextlib.contextmanager
def ieee_fp32():
    """(fix F6) the memory's GEMMs at fp32 'ieee': the trainer's TF32 setting is saved on the host and restored on exit,
    so the brain's matmul precision is unchanged."""
    tf32, prec = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prec)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _ieee(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        with ieee_fp32():
            return fn(*a, **k)
    return wrap
INF = float("inf")

V21_DEFAULTS = dict(
    wave_mode="v21", wave_seed=1234, wave_max_experts=0,
    wave_side=16, wave_levels=3, wave_slices=32, wave_codes=16, wave_topk=2, wave_entries=4, wave_warm=200,
    wave_anneal=2000, wave_upd_rows=4096, wave_fast=1000, wave_eta_slow=1.0 / 8192, wave_map_beta=0.25,
    wave_strip=4, wave_fill=0.12, wave_lambda=4.0, wave_read_chunk=2048, wave_write_start=500,
    wave_explore=0.05, wave_drop=0.1, wave_cap=0.5, wave_scale_init=0.3, wave_lr=1e-3, wave_dec_lr=1e-4,
    wave_z_conf=4.0, wave_c_z0=2.0, wave_null=256, wave_null_every=16, wave_write_w=16, wave_strong_k=4,
    wave_content=0.18, wave_priv_cap=2, wave_sweep_g=8, wave_sweep_p=24, wave_d_lr=1e-4, wave_eot=-1,
    wave_read_pass=3,
    # v2.2 (SPEC_v22.md): the delta write (else "spec" / "set": the v2.1 set rule), the normalized readout (else
    # "raw": v2.1), the strip's read entries (0 = auto: min(4, strip) for strips of more than 4 places, else 1)
    wave_write="delta", wave_readout="norm", wave_strip_entries=0,
    wave_writes_per_seq=8,               # (v2.2: the top 8 per sequence, >= 8 tokens apart; 1 = v2.1's one per sequence)
    # v2.2 latent memories (0 = d: v2.1); the workspace summary's width (the router's side input); the confidence
    # soft = sigmoid(2 (z - 3)), ramp = v2.1's clamp((z - c_z0) / 4, 0, 1)
    wave_latent=2048, wave_side_dim=64, wave_conf="soft",
    wave_cols=64)                        # (v2.2: columns per place; 16 = v2.1's store; 2048 / 64 = 32 dims per column)


def tkey(x):
    """Tie-free int64 sort keys of a float tensor (larger float -> larger key; ties -> the lower index wins)."""
    n = x.numel()
    b = x.detach().float().contiguous().view(-1).view(torch.int32)
    k = torch.where(b >= 0, b, b ^ 0x7FFFFFFF)
    return (k.long() << 32) + (n - 1 - torch.arange(n, device=x.device))


class Translate(torch.autograd.Function):
    """out_r = Xr[..., :d] + Xr @ D[e] per memory row (bmm), out_s likewise with D[shared]; X is the latent (dl = d:
    v2.1's Xr + Xr D; v2.2 latent: D [S, dl, d], the identity on the correction slice). D is a buffer: its gradient
    is accumulated into grad_buf (None at eval / train-only), never into D.grad."""

    @staticmethod
    def forward(ctx, Xr, Xs, D, E, sh, grad_buf):
        ctx.save_for_backward(Xr, Xs)
        ctx.D, ctx.E, ctx.sh, ctx.gb = D, E, sh, grad_buf
        d = D.size(-1)
        return Xr[..., :d] + torch.bmm(Xr, D[:E]), Xs[..., :d] + Xs @ D[sh]

    @staticmethod
    def backward(ctx, gr, gs):
        Xr, Xs = ctx.saved_tensors
        D, E, sh, gb = ctx.D, ctx.E, ctx.sh, ctx.gb
        pad = Xr.size(-1) - D.size(-1)
        with ieee_fp32():                                              # (fix F6: runs inside the trainer's backward)
            if gb is not None:
                gb[:E].baddbmm_(Xr.transpose(1, 2), gr)
                gb[sh].addmm_(Xs.t(), gs)
            return (F.pad(gr, (0, pad)) + torch.bmm(gr, D[:E].transpose(1, 2)), F.pad(gs, (0, pad)) + gs @ D[sh].t(),
                    None, None, None, None)


class WaveV21Memory(MemoryBase):
    kind = "wave"

    def __init__(self, d, n_exp, **c):
        super().__init__()
        c = dict(V21_DEFAULTS, **{k: v for k, v in c.items() if k in V21_DEFAULTS})
        self._cfg = c
        self.d, self.seed = int(d), int(c["wave_seed"])
        self.Emax = max(int(c["wave_max_experts"] or 0), int(n_exp))
        self.S = self.Emax + 1                          # memory rows; row Emax = the shared expert
        self.sh = self.Emax
        self.n_live = int(n_exp)
        self.address = WaveAddress(d, side=c["wave_side"], levels=c["wave_levels"], slices=c["wave_slices"],
                                   codes=c["wave_codes"], topk=c["wave_topk"], entries=c["wave_entries"],
                                   seed=self.seed, warm=c["wave_warm"], anneal=c["wave_anneal"], fast=c["wave_fast"],
                                   eta_slow=c["wave_eta_slow"])
        self.address.entry_keys = True
        self.Pg, self.J = self.address.L, int(c["wave_strip"])
        self.dl = int(c["wave_latent"] or d)                       # (v2.2: the memories' latent width)
        self.ds_side = int(c["wave_side_dim"])
        assert c["wave_conf"] in ("soft", "ramp"), f"--wave-conf {c['wave_conf']!r}: soft or ramp"
        self.conf_soft = c["wave_conf"] == "soft"
        se = int(c["wave_strip_entries"] or 0)
        self.Jr = max(1, min(self.J, se if se > 0 else (min(4, self.J) if self.J > 4 else 1)))
        self.delta = c["wave_write"] == "delta"
        assert c["wave_readout"] in ("norm", "raw"), f"--wave-readout {c['wave_readout']!r}: norm or raw"
        self.readout_norm = c["wave_readout"] == "norm"
        self._place_e, self._ratio, self._fire = None, None, None
        self.DUMMY = self.Pg + self.J * self.S
        self.store = WaveColumns(self.dl, 1, self.address, fill=c["wave_fill"], lam=c["wave_lambda"], seed=self.seed,
                                 read_chunk=c["wave_read_chunk"], places=self.DUMMY + 1, pad=64, cols=c["wave_cols"])
        self.store.n_read = int(c["wave_entries"])
        self.register_buffer("row_of", torch.arange(self.S))                 # (brain id -> memory row; shared: Emax)
        # the private address
        self.register_buffer("P_e", torch.zeros(self.S, self.J, d))
        self.register_buffer("Pinit", torch.zeros(self.S, dtype=torch.long))
        self.register_buffer("Ph", torch.zeros(self.S, self.J))               # (warm-up sample hits)
        self.register_buffer("Wld_p", torch.zeros(self.S, self.J))            # (write-load EMA, T_L 1,024)
        # the gate (lr --wave-lr) and log_s (lr --wave-dec-lr)
        g = torch.Generator().manual_seed(self.seed + 31)
        r = lambda *s, std: torch.randn(*s, generator=g) * std
        self.n_ftok, self.n_fpair, hid, dx = 2, 1, 16, 32
        self.W_gx = nn.Parameter(r(dx, d, std=1.0 / math.sqrt(d)))
        self.G1_w = nn.Parameter(r(hid, dx + self.n_ftok, std=1.0 / math.sqrt(dx + self.n_ftok)))
        self.G1_b = nn.Parameter(torch.zeros(hid))
        self.Wp = nn.Parameter(r(self.S, hid, std=0.01))
        self.Wg = nn.Parameter(r(self.S, hid, std=0.01))
        self.Up = nn.Parameter(torch.zeros(self.n_fpair))
        self.Ug = nn.Parameter(torch.zeros(self.n_fpair))
        with torch.no_grad():                         # (bias for ~25% yes: b0 = -Phi^-1(.75) x the score's sd)
            hp = self._h(r(512, d, std=1.0), r(512, self.n_ftok, std=1.0))
            b0 = float(-0.6745 * 0.01 * hp.norm(dim=-1).pow(2).mean().sqrt())
        self.bp_b = nn.Parameter(torch.full((self.S,), b0))
        self.bg_b = nn.Parameter(torch.full((self.S,), b0))
        self.Wc = nn.Parameter(torch.zeros(2, hid))                           # (confirmation heads: global, private)
        self.Uc = nn.Parameter(torch.zeros(2, max(self.n_ftok, self.n_fpair)))
        self.bc = nn.Parameter(torch.zeros(2))
        self.log_s = nn.Parameter(torch.tensor(math.log(float(c["wave_scale_init"]))))
        if self.readout_norm:             # (v2.2 readout: g [d] init 1, w_e [S, d] small random, b_e [S] init 1;
            g2 = torch.Generator().manual_seed(self.seed + 37)               # the main loss trains them, lr --wave-dec-lr)
            self.g_read = nn.Parameter(torch.ones(self.dl))
            self.w_e = nn.Parameter(torch.randn(self.S, d, generator=g2) * (0.01 / math.sqrt(d)))
            self.b_e = nn.Parameter(torch.ones(self.S))
        # D: a buffer + its gradient buffer + the factored second moments
        self.register_buffer("D", torch.zeros(self.S, self.dl, d))          # (v2.2 latent: D_e^T [dl, d], init 0)
        self.register_buffer("D_grad", torch.zeros(self.S, self.dl, d), persistent=False)
        self.register_buffer("R", torch.zeros(self.S, self.dl))
        # v2.2 latent write value (no learned encoder): [correction d | xf d | workspace summary ds_side] then, short of
        # dl, a fixed random projection of the correction fills the rest; over dl, a fixed random projection down
        # (derived from the seed: not saved)
        self.n_parts = 2 * d + self.ds_side
        if self.dl != d:
            g3 = torch.Generator().manual_seed(self.seed + 43)
            if self.n_parts <= self.dl:
                self.register_buffer("lat_fill", torch.randn(d, self.dl - self.n_parts, generator=g3) / math.sqrt(d),
                                     persistent=False)
            else:
                self.register_buffer("lat_proj", torch.randn(self.n_parts, self.dl, generator=g3)
                                     / math.sqrt(self.n_parts), persistent=False)
        self.register_buffer("Cv", torch.zeros(self.S, d))
        self.register_buffer("n_d", torch.zeros(self.S, dtype=torch.long))
        # confidence: per-sheet null statistics (0 global, 1 private) in 8 load bins; theta per sheet; sigma_b
        self.register_buffer("null_mu", torch.zeros(2, 8))
        self.register_buffer("null_sd", torch.ones(2, 8))
        self.register_buffer("null_n", torch.zeros(2, 8, dtype=torch.long))
        self.register_buffer("theta", torch.full((2,), 0.05))
        self.register_buffer("sigma_b", torch.zeros(2))
        self.register_buffer("kappa", torch.zeros(()))                       # (median |xf|: the content scale)
        self.policy = SpecPolicy(write_w=c["wave_write_w"], per_seq=int(c["wave_writes_per_seq"]), seed=self.seed)
        self.policy.strong_k = int(c["wave_strong_k"])
        self.write_start, self.explore, self.drop = int(c["wave_write_start"]), float(c["wave_explore"]), float(c["wave_drop"])
        self.cap, self.z_conf, self.c_z0 = float(c["wave_cap"]), float(c["wave_z_conf"]), float(c["wave_c_z0"])
        self.lr, self.dec_lr, self.d_lr = float(c["wave_lr"]), float(c["wave_dec_lr"]), float(c["wave_d_lr"])
        self.map_beta, self.n_null, self.null_every = float(c["wave_map_beta"]), int(c["wave_null"]), int(c["wave_null_every"])
        self.upd_rows, self.content, self.priv_cap = int(c["wave_upd_rows"]), float(c["wave_content"]), int(c["wave_priv_cap"])
        self.sweep_g, self.sweep_p = int(c["wave_sweep_g"]), int(c["wave_sweep_p"])
        self.eot = None if c["wave_eot"] < 0 else int(c["wave_eot"])
        self.read_pass = int(c["wave_read_pass"])
        self.train_forward = False
        self.bench_open = False
        self._opt, self._opt_pending = None, None
        self.n_opt, self.n_fwd, self.ptr_g, self.ptr_p = 0, 0, 0, 0
        self.cnt = dict(fwd=0, recalls=0, reads_p=0, reads_g=0, pairs=0, toks=0, explore_p=0, explore_g=0,
                        writes_g=0, writes_p=0, writes_both=0, confirms_g=0, confirms_p=0, excluded=0, demoted=0,
                        dropped_seq=0, seqs=0)
        self._cache, self._live, self._rows, self._samp, self._sel = None, False, [], None, None
        self.last_stats = {}

    # ------------------------------------------------------------------ state
    def get_extra_state(self):
        opt = self._opt.state_dict() if self._opt is not None else self._opt_pending
        return dict(address=self.address.host_state(), store=self.store.host_state(), n_opt=self.n_opt,
                    n_fwd=self.n_fwd, ptr_g=self.ptr_g, ptr_p=self.ptr_p, n_live=self.n_live, cnt=dict(self.cnt),
                    policy_calls=self.policy.n_calls, opt=opt)

    def set_extra_state(self, st):
        self.address.load_host_state(st.get("address", {}))
        self.store.load_host_state(st.get("store", {}))
        self.n_opt, self.n_fwd = int(st.get("n_opt", 0)), int(st.get("n_fwd", 0))
        self.ptr_g, self.ptr_p = int(st.get("ptr_g", 0)), int(st.get("ptr_p", 0))
        self.n_live = int(st.get("n_live", self.n_live))
        self.cnt = dict(self.cnt, **st.get("cnt", {}))
        self.policy.n_calls = int(st.get("policy_calls", 0))
        self._opt, self._opt_pending = None, st.get("opt")

    def memory_cfg(self):
        return dict(self._cfg)

    @property
    def ready(self):
        return self.address.ready

    @property
    def active(self):
        """Reads / writes / gate only from write_start on (before: == none; the addresses learn)."""
        return self.address.ready and self.n_opt >= self.write_start

    # ------------------------------------------------------------------ the gate
    def _h(self, xhat, f_tok):
        return F.silu(torch.cat([xhat @ self.W_gx.t(), f_tok], -1) @ self.G1_w.t() + self.G1_b)

    def _scores(self, h_rows, rows, f_pair):
        bp = (self.Wp[rows] * h_rows).sum(-1) + f_pair @ self.Up + self.bp_b[rows]
        bg = (self.Wg[rows] * h_rows).sum(-1) + f_pair @ self.Ug + self.bg_b[rows]
        return bp, bg

    def _gens(self, purpose):
        """One generator per purpose, reseeded per (micro-batch, purpose) from python ints."""
        g = torch.Generator(device=self.D.device)
        g.manual_seed(self.seed * 1_000_003 + self.n_fwd * 17 + purpose)
        return g

    # ------------------------------------------------------------------ the forward
    def begin_forward(self):
        self._cache, self._sel = None, None
        self._live = bool(self.training and self.train_forward and torch.is_grad_enabled())
        if self._live:
            self.n_fwd += 1
            self.cnt["fwd"] += 1

    def _layout(self, e_k, t_k, slot_k, N):
        """eid [N, 7] (routed slots 0..5, -1 where dropped by capacity; column 6 = the shared expert)."""
        k = 6
        eid = torch.full((N * 7 + 1,), -1, dtype=torch.long, device=e_k.device)
        pos = torch.where(slot_k < k, t_k * 7 + slot_k, torch.full_like(t_k, N * 7))
        eid = eid.index_put((pos,), e_k.long())[:N * 7].view(N, 7)
        eid[:, 6] = self.sh
        return eid

    @torch.no_grad()
    def _select(self, bp, bg, eid, N, training):
        """The spec's selection (PART 2), python-int budgets from N; -> dict of the decision tensors."""
        dev, M = bp.device, 7 * N
        E = self.n_live
        K, K_pr, K_g = max(1, 7 * N // 2), max(1, 3 * N), max(1, N // 2)
        K_xp = int(round(self.explore * 6 * N)) if training else 0
        K_xg = int(round(self.explore * N)) if training else 0
        C = 8 * math.ceil(2 * K * (6 / 7) / max(1, E) / 8)
        C_sh, NSLOT = K_g, E * C + K_g
        ar = torch.arange(M, device=dev)
        present = eid >= 0
        ge = self._gens(1)
        xp = torch.zeros(N, 7, dtype=torch.bool, device=dev)
        if K_xp:
            u = torch.rand(N, 6, generator=ge, device=dev)
            u = torch.where(present[:, :6], u, torch.full_like(u, -1.0))
            xp6 = torch.zeros(6 * N, dtype=torch.bool, device=dev)
            xp6.index_put_((torch.topk(tkey(u.view(-1)), K_xp).indices,), torch.ones((), dtype=torch.bool, device=dev))
            xp[:, :6] = xp6.view(N, 6) & present[:, :6]
        xg = torch.zeros(N, dtype=torch.bool, device=dev)
        if K_xg:
            xg.index_put_((torch.topk(tkey(torch.rand(N, generator=ge, device=dev)), K_xg).indices,),
                          torch.ones((), dtype=torch.bool, device=dev))
        bp = torch.where(present, bp, torch.full_like(bp, -INF))
        bp[:, 6] = -INF                                                    # (the shared pair: global only)
        bg = torch.where(present, bg, torch.full_like(bg, -INF))
        tau = 0.0
        SG = bg.amax(1)
        gs = torch.where(xg, torch.full_like(SG, INF), torch.where(SG > tau, SG, torch.full_like(SG, -INF)))
        GT0 = torch.topk(tkey(gs), K_g).indices
        gsel = torch.zeros(N, dtype=torch.bool, device=dev).index_put_((GT0,), xg[GT0] | (SG[GT0] > tau))
        wg = gsel[:, None] & (xg[:, None] | (bg > tau)) & present
        wp = (xp | (bp > tau)) & present
        want = wp | wg
        forced = xp | (xg[:, None] & wg)
        r = torch.where(forced, torch.full_like(bp, INF),
                        torch.where(want, torch.maximum(torch.where(wp, bp, torch.full_like(bp, -INF)),
                                                        torch.where(wg, bg, torch.full_like(bg, -INF))),
                                    torch.full_like(bp, -INF))).view(-1)
        rows = self.row_of[eid.clamp(min=0)]
        grp = torch.where(eid == self.sh, torch.full_like(eid, E), rows).view(-1)
        key = tkey(r)
        g_rank = torch.empty(M, dtype=torch.long, device=dev).scatter_(0, torch.argsort(key, descending=True), ar)
        order = torch.argsort(grp * (M + 1) + g_rank)
        gs2 = grp[order]
        rk = torch.empty(M, dtype=torch.long, device=dev).scatter_(0, order, ar - torch.searchsorted(gs2, gs2))
        cap = torch.where(grp == E, torch.full_like(grp, C_sh), torch.full_like(grp, C))
        elig = (rk < cap) & want.view(-1)
        Kk = min(K, M)
        S = torch.topk(torch.where(elig, key, LOW + ar), Kk).indices
        valid = elig[S]
        tok = S // 7
        gS = grp[S]
        vp0 = valid & wp.view(-1)[S]
        vg = valid & wg.view(-1)[S]
        slot = torch.where(valid, torch.where(gS == E, E * C, gS * C) + rk[S], torch.full_like(S, NSLOT))
        PR = torch.topk(tkey(torch.where(vp0, torch.where(xp.view(-1)[S], torch.full_like(bp.view(-1)[S], INF),
                                                            bp.view(-1)[S]), torch.full_like(vp0, -INF, dtype=bp.dtype))),
                        min(K_pr, Kk)).indices
        vp = torch.zeros(Kk, dtype=torch.bool, device=dev).index_put_((PR,), vp0[PR])
        gused = torch.zeros(N, dtype=torch.int32, device=dev).index_put_((tok,), vg.int(), accumulate=True) > 0
        GT = torch.topk(tkey(torch.where(gused, SG, torch.full_like(SG, -INF))), K_g).indices
        gvalid = gused[GT]
        stats = torch.stack([want.sum(), (want.view(-1) & ~elig).sum() + (elig.sum() - valid.sum()),
                             (vp0 & ~vp).sum(), vp.sum(), gvalid.sum(), xp.sum(), xg.sum()])
        return dict(S=S, valid=valid, tok=tok, slot=slot, vp=vp, vg=vg, PR=PR, GT=GT, gvalid=gvalid, gused=gused,
                    xp=xp, xg=xg, C=C, NSLOT=NSLOT, K=Kk, eid=eid, stats=stats, K_g=K_g)

    @torch.no_grad()
    def _private_place(self, xhat, eid):
        """place_p [N, 6] of the routed pairs (DUMMY where the row has no prototype or the pair is missing)."""
        N = xhat.size(0)
        E = self.n_live
        Pc = self.P_e[:E].reshape(E * self.J, -1)
        rows = self.row_of[eid[:, :6].clamp(min=0)]                         # (N, 6)
        Jr = self.Jr
        jstar = torch.zeros(N, 6, dtype=torch.long, device=xhat.device)
        jtop = torch.zeros(N, 6, Jr, dtype=torch.long, device=xhat.device)
        okt = torch.zeros(N, 6, Jr, dtype=torch.bool, device=xhat.device)
        for a in range(0, N, 2048):
            s = xhat[a:a + 2048] @ Pc.T - 0.5 * (Pc * Pc).sum(-1)            # (n, J E)
            r_ = rows[a:a + 2048]
            cols = r_[..., None] * self.J + torch.arange(self.J, device=xhat.device)
            sc = s.gather(1, cols.view(len(r_), -1)).view(len(r_), 6, self.J)
            ok_j = torch.arange(self.J, device=xhat.device) < self.Pinit[r_][..., None]
            scm = torch.where(ok_j, sc, torch.full_like(sc, -INF))
            jstar[a:a + 2048] = scm.argmax(-1)
            if Jr > 1:                    # (v2.2 strips: the best Jr places of the strip, entry 0 = the argmax)
                t = scm.topk(Jr, -1)
                jtop[a:a + 2048] = torch.cat([jstar[a:a + 2048, :, None], t.indices[..., 1:]], -1)
                okt[a:a + 2048] = t.values > -INF
        has = (self.Pinit[rows] > 0) & (eid[:, :6] >= 0) & (rows < E)
        place = torch.where(has, self.Pg + self.J * rows + jstar, torch.full_like(rows, self.DUMMY))
        if Jr > 1:
            pe = torch.where(has[..., None] & okt, self.Pg + self.J * rows[..., None] + jtop,
                             torch.full_like(jtop, self.DUMMY))
            self._place_e = torch.cat([place[..., None], pe[..., 1:]], -1)     # (N, 6, Jr)
        else:
            self._place_e = place[..., None]
        return place, jstar, rows

    def _codes_at(self, xhat, place, base):
        """Codes at `place` with the keys of x_hat - P(place) (global or strip prototypes): place (n,) / base (n, d) =
        one entry per row; place (n, Jr) / base (n, Jr, d) = Jr entries, each with its own residual keys (v2.2)."""
        a = self.address
        n = xhat.size(0)
        if place.dim() == 2:
            ke = torch.stack([a._top1_codes(xhat - base[:, j]) for j in range(place.size(1))], 1)   # (n, Jr, 96)
            keys, loc = ke[:, 0], place
        else:
            keys, loc = a._top1_codes(xhat - base), place[:, None]
            ke = keys[:, None]
        z = torch.zeros(n, loc.size(1), device=xhat.device)
        return Codes(loc, z, keys[:, :, None], torch.ones(n, keys.size(1), 1, device=xhat.device),
                     torch.zeros(n, keys.size(1), 1, device=xhat.device), torch.zeros(n, a.n_feats, device=xhat.device),
                     ke)

    def _proto(self, place):
        """The prototype (x_hat space) of strip places (DUMMY / global: zeros); place (n,) or (n, Jr)."""
        m = (place >= self.Pg) & (place < self.DUMMY)
        q = (place - self.Pg).clamp(min=0, max=self.S * self.J - 1)
        return self.P_e.view(-1, self.d)[q] * m[..., None].float()

    def _zc(self, chi, places, sheet):
        ld = self.store.load[places.clamp(max=self.store.P - 1)].float()
        b = torch.log2(ld + 1).floor().long().clamp(0, 7)
        z = (chi - self.null_mu[sheet][b]) / self.null_sd[sheet][b].clamp(min=0.005)
        if self.conf_soft:                # (v2.2: soft, reads around z = 3 contribute; the use-gate decides)
            c = torch.sigmoid(2 * (z - 3)) * (self.null_n[sheet][b] > 0).float()
        else:
            c = ((z - self.c_z0) / 4).clamp(0, 1) * (self.null_n[sheet][b] > 0).float()
        return z, c

    def _latent(self, corr, xf, side):
        """(v2.2) the write value (n, dl) from the brain state at the written tokens, no learned encoder: corr (n, d)
        the unit correction, xf (n, d) the expert input thought, side (n, ds_side) or None the workspace summary;
        each part unit-normalised; dl == d: corr itself (v2.1)."""
        if self.dl == self.d:
            return corr
        nrm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sd = side if side is not None else corr.new_zeros(corr.size(0), self.ds_side)
        z = torch.cat([corr, nrm(xf.float()), nrm(sd.float())], -1)
        if hasattr(self, "lat_fill"):
            return torch.cat([z, corr @ self.lat_fill], -1)
        return z @ self.lat_proj

    @torch.no_grad()
    def _recall(self, xf, e_k, t_k, slot_k):
        st, a = self.store, self.address
        N = xf.size(0)
        with _rf("wave.address"):
            xhat = a.standardize(xf)
            codes = a.encode(xhat, standardized=True)
            eid = self._layout(e_k, t_k, slot_k, N)
            place_p, jstar, rows = self._private_place(xhat, eid)
        with _rf("wave.decide"):
            xn = xf.detach().float().norm(dim=-1)
            f_tok = torch.stack([torch.log1p(xn), codes.feats[:, 0]], -1)
            f_pair = torch.cat([-torch.arange(6, device=xf.device).float().expand(N, 6) / 6,
                                torch.zeros(N, 1, device=xf.device)], 1)[..., None]       # (routing rank; shared 0)
            h = self._h(a.standardize(xf) / xhat.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt(), f_tok)
            rr = torch.where(eid >= 0, self.row_of[eid.clamp(min=0)], torch.zeros_like(eid))
            bp, bg = self._scores(h[:, None, :].expand(N, 7, -1), rr, f_pair)
            dec = self._select(bp, bg, eid, N, self._live)
        S, tok = dec["S"], dec["tok"]
        GT, gvalid, PR, vp = dec["GT"], dec["gvalid"], dec["PR"], dec["vp"]
        with _rf("wave.recall_global"):
            cg = codes.take(GT)
            cg.loc = torch.where(gvalid[:, None], cg.loc, torch.full_like(cg.loc, self.DUMMY))
            trg = st.trace(cg)
            wing = trg["loc"].gather(1, trg["best"][:, None]).squeeze(1)
            z_g, c_g = self._zc(trg["chibar"], wing, 0)
            kap_g = st.kappa_mean(trg)
        with _rf("wave.recall_private"):
            pr_pair = S[PR]                                                  # (flat pair ids t * 7 + k)
            pl = place_p.view(-1)[(pr_pair // 7) * 6 + (pr_pair % 7).clamp(max=5)]
            pl = torch.where(vp[PR] & ((pr_pair % 7) < 6), pl, torch.full_like(pl, self.DUMMY))
            xh_pr = xhat[pr_pair // 7]
            if self.Jr > 1:               # (v2.2: the best Jr places of the strip, read like the global entries)
                ple = self._place_e.view(-1, self.Jr)[(pr_pair // 7) * 6 + (pr_pair % 7).clamp(max=5)]
                ple = torch.where((pl < self.DUMMY)[:, None], ple, torch.full_like(ple, self.DUMMY))
                cp = self._codes_at(xh_pr, ple, self._proto(ple))
            else:
                cp = self._codes_at(xh_pr, pl, self._proto(pl))
            trp = st.trace(cp)
            winp = trp["loc"].gather(1, trp["best"][:, None]).squeeze(1)
            z_p, c_p = self._zc(trp["chibar"], winp, 1)
            kap_p = st.kappa_mean(trp)
        if self.bench_open:
            c_g, c_p = torch.ones_like(c_g), torch.ones_like(c_p)
        return dict(codes=codes, xhat=xhat, xn=xn, eid=eid, dec=dec, h_in=(xhat, f_tok, f_pair, rr),
                    trg=trg, cg=cg, z_g=z_g, c_g=c_g, kap_g=kap_g, trp=trp, cp=cp, z_p=z_p, c_p=c_p, kap_p=kap_p,
                    place_p=place_p, place_e=self._place_e, pl=pl, xf=xf.detach())

    def _hint(self, rec, keep, xf=None):
        """y (with grad to the decoder) -> o [K, d] per selected pair (exactly 0 on unread rows), r3 [7N + 1, d].
        --wave-readout norm (v2.2): y_n = RMSNorm(y) g (the unit convention: / sqrt(d), so |y_n| = 1 at g = 1, the
        downstream scale exp(log_s) |x| and the cap unchanged) x the confidence c kappa_bar; per pair s = 2 tanh(
        silu(xf . w_e + b_e) / 2) with the main loss's gradient into xf (the read path); RMSNorm(0) = 0 exactly."""
        dec, st = rec["dec"], self.store
        N, K, d = rec["xn"].numel(), dec["K"], self.d
        S, tok, GT, gvalid, PR, vp, vg = dec["S"], dec["tok"], dec["GT"], dec["gvalid"], dec["PR"], dec["vp"], dec["vg"]
        if self.readout_norm:
            unit = lambda y: y * torch.rsqrt(y.pow(2).sum(-1, keepdim=True) + 1e-12) * self.g_read
        else:
            unit = lambda y: y / y.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        y_g = (rec["c_g"] * rec["kap_g"])[:, None] * unit(st.decode(rec["trg"]))
        y_p = (rec["c_p"] * rec["kap_p"])[:, None] * unit(st.decode(rec["trp"]))
        y_g_full = y_g.new_zeros(N, self.dl).index_put((GT,), y_g * gvalid[:, None].float())
        y_p_full = y_p.new_zeros(K, self.dl).index_put((PR,), y_p * vp[PR][:, None].float())
        rec["y_raw"] = (vg[:, None].float() * y_g_full[tok]).detach(), (vp[:, None].float() * y_p_full).detach()
        rec["y_g_tok"] = y_g_full.detach()          # (fix F1: the token-level global read, pre-keep; no scatter over tok)
        kp = keep[tok][:, None] if keep is not None else 1.0
        y_pair = kp * (vg[:, None].float() * y_g_full[tok] + vp[:, None].float() * y_p_full)
        if self.readout_norm:             # (v2.2: the per-(token, expert) gate s; unread pairs stay exactly 0)
            rows = rec["h_in"][3].view(-1)[S]
            xs = (xf if xf is not None else rec["xf"]).float()[tok]
            s = 2 * torch.tanh(F.silu((xs * self.w_e[rows]).sum(-1) + self.b_e[rows]) / 2)
            y_pair = y_pair * s[:, None]
        if self._live and torch.is_grad_enabled():
            rec["probe"] = torch.zeros(K, self.dl, device=y_pair.device, requires_grad=True)
            y_pair = y_pair + rec["probe"]                                   # (Capture: E_pair = dL/dy_pair)
        E, C, NSLOT = self.n_live, dec["C"], dec["NSLOT"]
        X = y_pair.new_zeros(NSLOT + 1, self.dl).index_put((dec["slot"],), y_pair)
        gb = self.D_grad if (self._live and self.train_only is None) else None
        outr, outs = Translate.apply(X[:E * C].view(E, C, self.dl), X[E * C:NSLOT], self.D, E, self.sh, gb)
        o = torch.cat([outr.reshape(-1, d), outs, y_pair.new_zeros(1, d)])[dec["slot"]]
        xt = rec["xn"][tok]
        o = (torch.exp(self.log_s) * xt)[:, None] * o
        if self.cap:
            o = o * (self.cap * xt / o.norm(dim=-1).clamp(min=1e-6)).clamp(max=1.0).detach()[:, None]
        with torch.no_grad():             # (log: the read norm / the thought norm over the pairs that read)
            if self._live:
                rd = (vg | vp).float()
                self._ratio = (o.norm(dim=-1) / xt.clamp(min=1e-6) * rd).sum() / rd.sum().clamp(min=1)
                rg, rp = gvalid.float(), vp[PR].float()                # (the recall-fire rate: c > 0.5 over reads)
                self._fire = ((((rec["c_g"] > 0.5).float() * rg).sum() + ((rec["c_p"] > 0.5).float() * rp).sum())
                              / (rg.sum() + rp.sum()).clamp(min=1))
        r3 = o.new_zeros(7 * N + 1, d).index_put((S,), o)
        return r3

    @_ieee
    def __call__(self, xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=None):
        k = int(call or 0)
        with torch.autocast(xf.device.type, enabled=False):
            if self._live and k == self.read_pass and self._samp is None:
                self._samp = (xf.detach(), e_k, t_k, slot_k)                 # (address learning samples)
            if self._live and k == self.read_pass:
                self._rows.append(xf.detach()[::max(1, xf.size(0) // self.upd_rows)].float())
            if not self.active or k < self.read_pass or (self._cache is None and k != self.read_pass):
                return buf, x
            Nt = xf.size(0)
            if self._cache is None:
                rec = self._recall(xf, e_k, t_k, slot_k)
                self.cnt["recalls"] += int(self._live)
                keep = self._keep(x) if self._live else None
                rec["keep"] = keep
                if self.dl != self.d and side is not None:          # (v2.2 latent: the workspace summary per token)
                    sd = side.detach().reshape(Nt, -1).float()
                    rec["side"] = F.pad(sd, (0, max(0, self.ds_side - sd.size(1))))[:, :self.ds_side]
                with _rf("wave.readout"), torch.set_grad_enabled(self._live or torch.is_grad_enabled()):
                    rec["r3"] = self._hint(rec, keep, xf)   # (training: with a graph even if pass 3 is outside the
                                                            # gradient window: passes 4-5 carry it to the capture)
                self._cache = rec
                if self._live:
                    with torch.no_grad():
                        med = rec["xn"].median()
                        self.kappa.copy_(torch.where(self.kappa > 0, self.kappa.lerp(med, 0.01), med))
                    self._count(rec)
                idx = torch.where(slot_k < 6, t_k * 7 + slot_k, torch.full_like(t_k, 7 * Nt))
            else:
                rec = self._cache
                eq = rec["eid"][t_k, :6] == e_k[:, None]
                has = eq.any(-1)
                kidx = eq.to(torch.int8).argmax(-1).long()
                idx = torch.where(has, t_k * 7 + kidx, torch.full_like(t_k, 7 * Nt))
            r3 = rec["r3"]
            r_p = r3[idx]
            r_s = r3[torch.arange(Nt, device=xf.device) * 7 + 6]
        buf = buf.index_put((e_k, p_k), r_p.to(buf.dtype), accumulate=True)
        return buf, x + r_s.to(x.dtype).view(x.shape)

    def _keep(self, x):
        B = x.size(0) if x.dim() == 3 else 1
        T = x.numel() // (B * x.size(-1))
        g = torch.Generator().manual_seed(self.seed * 7919 + self.n_fwd)
        ks = torch.rand(B, generator=g) >= self.drop
        self.cnt["dropped_seq"] += int(B - int(ks.sum()))
        self.cnt["seqs"] += B
        return ks.float().repeat_interleave(T).to(x.device)

    @torch.no_grad()
    def _count(self, rec):
        s = rec["dec"]["stats"]
        N = rec["xn"].numel()
        self._stat_acc = s if getattr(self, "_stat_acc", None) is None else self._stat_acc + s
        self.cnt["pairs"] += 6 * N
        self.cnt["toks"] += N

    def read(self, xf, e_k, t_k, p_k=None, slot_k=None, side=None, call=None):
        raise NotImplementedError("WaveV21Memory.read: the Harness bench of v2.1 is P2 (deferred)")

    # ------------------------------------------------------------------ writes
    def select_writes(self, logits, targets, idx=None, ce_h=None):
        rec = self._cache
        if not self._live or rec is None or rec.get("probe") is None:
            self._sel = None
            return None
        B, T = targets.shape
        with torch.no_grad():
            if ce_h is not None:
                ce, H, V = ce_h[0].reshape(-1).float(), ce_h[1].reshape(-1).float(), int(ce_h[2])
            else:
                lp = torch.log_softmax(logits.detach().reshape(B * T, -1).float(), -1)
                ce = -lp.gather(1, targets.reshape(-1, 1)).squeeze(1)
                H = -(lp.exp() * lp).sum(-1)
                V = lp.size(1)
            dec = rec["dec"]
            N = B * T
            vp_d = torch.zeros(7 * N + 1, dtype=torch.bool, device=ce.device).index_put(
                (dec["S"],), dec["vp"])[:7 * N].view(N, 7)
            read_tok = dec["gused"] | vp_d[:, :6].any(1)
            if rec["keep"] is not None:
                read_tok = read_tok & (rec["keep"] > 0)
            rec["vp_d"], rec["read_tok"] = vp_d, read_tok
            rec["ce_h"] = (ce, H, V, B, T, idx)
        self._sel = True
        return self._sel

    @_ieee
    @torch.no_grad()
    def after_backward(self):
        rec, rows, samp = self._cache, self._rows, self._samp
        self._rows, self._samp = [], None
        if self._live and rows and self.train_only is None:
            X = torch.cat(rows)
            if X.size(0) > self.upd_rows:
                X = X[::math.ceil(X.size(0) / self.upd_rows)]
            self.address.update(X)
        if samp is not None and not self.active and self.train_only is None:
            self._learn_private_warm(*samp)
        if rec is not None and self._live and self.train_only is None and self.n_opt % self.null_every == 0:
            with _rf("wave.null"):
                self._null(rec)                          # (the null control needs no gradient)
        if rec is None or rec.get("probe") is None or rec["probe"].grad is None or self.train_only is not None:
            return None
        Ep = rec["probe"].grad.float()                                   # (K, d) E_pair
        dec = rec["dec"]
        N, K = rec["xn"].numel(), dec["K"]
        S, tok, GT, gvalid, PR, vp, vg = dec["S"], dec["tok"], dec["GT"], dec["gvalid"], dec["PR"], dec["vp"], dec["vg"]
        keep = rec["keep"] if rec["keep"] is not None else torch.ones(N, device=Ep.device)
        # W1: the global hint's gradient per token (dense gather over the token's 7 pair rows), cosines
        pslot = torch.full((7 * N + 1,), K, dtype=torch.long, device=Ep.device).index_put(
            (S,), torch.arange(K, device=Ep.device))[:7 * N].view(N, 7)
        Epad = torch.cat([Ep * vg[:, None].float(), Ep.new_zeros(1, self.dl)])
        e_g = Epad[pslot].sum(1)                                          # (N, d)
        y_g_raw, y_p_raw = rec["y_raw"]
        yg_tok = rec["y_g_tok"]                      # (fix F1: was index_put over the repeated tok -> zero rows won)
        cosz = lambda a_, b_: torch.where(a_.norm(dim=-1) > 1e-6, F.cosine_similarity(a_, b_, dim=-1),
                                          torch.zeros_like(a_[:, 0]))
        h_g = cosz(yg_tok[GT], -e_g[GT])
        h_p = cosz(y_p_raw[PR], -Ep[PR])
        # W3: confirmations (read sheets, kept sequences)
        ok_g = gvalid & (keep[GT] > 0) & (rec["z_g"] >= self.z_conf) & (h_g >= self.theta[0])
        ok_p = vp[PR] & (keep[tok[PR]] > 0) & (rec["z_p"] >= self.z_conf) & (h_p >= self.theta[1])
        ok_g = self._topmask(ok_g, h_g * rec["c_g"], 8)
        ok_p = self._topmask(ok_p, h_p * rec["c_p"], max(1, self.n_live // 24))
        # W3b: the gate loss
        with torch.enable_grad():
            self._gate_loss(rec, Ep, ok_g, ok_p)
        # W4-W6: the writes
        conf_tok = torch.zeros(N, dtype=torch.bool, device=Ep.device)
        conf_tok = conf_tok.index_put((GT,), ok_g, accumulate=True).index_put((tok[PR],), ok_p, accumulate=True)
        self._writes(rec, Ep, e_g, h_g, h_p, conf_tok)
        st = self.store
        if bool(ok_g.any()):
            tr = {k_: (v[ok_g] if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == ok_g.numel() else v)
                  for k_, v in rec["trg"].items()}
            st.confirm(rec["cg"].take(ok_g.nonzero().squeeze(-1)), torch.ones(int(ok_g.sum()), device=Ep.device),
                       trace=tr)
            self.cnt["confirms_g"] += int(ok_g.sum())
        if bool(ok_p.any()):
            tr = {k_: (v[ok_p] if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == ok_p.numel() else v)
                  for k_, v in rec["trp"].items()}
            st.confirm(rec["cp"].take(ok_p.nonzero().squeeze(-1)), torch.ones(int(ok_p.sum()), device=Ep.device),
                       trace=tr)
            self.cnt["confirms_p"] += int(ok_p.sum())
        return None

    @staticmethod
    def _topmask(ok, score, k):
        if not bool(ok.any()):
            return ok
        sc = torch.where(ok, score, torch.full_like(score, -INF))
        top = torch.topk(tkey(sc), min(k, sc.numel())).indices
        return torch.zeros_like(ok).index_put((top,), ok[top])

    def _gate_loss(self, rec, Ep, ok_g, ok_p):
        dec = rec["dec"]
        S, tok, vp, vg = dec["S"], dec["tok"], dec["vp"], dec["vg"]
        y_g_raw, y_p_raw = rec["y_raw"]
        b_p = -(Ep * y_p_raw).sum(-1)
        b_g = -(Ep * y_g_raw).sum(-1)
        with torch.no_grad():
            for i, (b, m) in enumerate(((b_g, vg), (b_p, vp))):
                rms = (b[m].pow(2).mean() if bool(m.any()) else torch.zeros((), device=b.device)).sqrt()
                self.sigma_b[i] = torch.where(self.sigma_b[i] == 0, rms, 0.99 * self.sigma_b[i] + 0.01 * rms)
        bt_g = (b_g / self.sigma_b[0].clamp(min=1e-12)).clamp(-5, 5)
        bt_p = (b_p / self.sigma_b[1].clamp(min=1e-12)).clamp(-5, 5)
        xhat, f_tok, f_pair, rr = rec["h_in"]
        h = self._h(xhat[tok] / xhat[tok].pow(2).mean(-1, keepdim=True).add(1e-12).sqrt(), f_tok[tok])
        rows = rr.view(-1)[S]
        fp = f_pair.view(-1, f_pair.size(-1))[S]
        bhp, bhg = self._scores(h, rows, fp)
        xp = dec["xp"].view(-1)[S]
        xg = dec["xg"][tok]
        wpf, wgf = vp.float() * (1 + 3 * xp.float()), vg.float() * (1 + 3 * xg.float())
        Lb = ((wpf * F.huber_loss(bhp, bt_p, reduction="none", delta=1.0)).sum()
              + (wgf * F.huber_loss(bhg, bt_g, reduction="none", delta=1.0)).sum()) / (wpf + wgf).sum().clamp(min=1)
        cg_lab = torch.zeros(rec["xn"].numel(), device=Ep.device).index_put((dec["GT"],), ok_g.float())[tok]
        cp_lab = torch.zeros(dec["K"], device=Ep.device).index_put((dec["PR"],), ok_p.float())
        lc_g = (h * self.Wc[0]).sum(-1) + f_tok[tok] @ self.Uc[0, :f_tok.size(1)] + self.bc[0]
        lc_p = (h * self.Wc[1]).sum(-1) + fp @ self.Uc[1, :fp.size(1)] + self.bc[1]
        Lc = ((F.binary_cross_entropy_with_logits(lc_g, cg_lab, reduction="none") * vg.float()).sum()
              + (F.binary_cross_entropy_with_logits(lc_p, cp_lab, reduction="none") * vp.float()).sum()) \
            / (vp.float() + vg.float()).sum().clamp(min=1)
        loss = Lb + 0.5 * Lc
        torch.autograd.backward(loss, inputs=self.gate_parameters())
        self.last_stats = dict(gate_loss=loss.detach())

    def gate_parameters(self):
        return [self.W_gx, self.G1_w, self.G1_b, self.Wp, self.Wg, self.Up, self.Ug, self.bp_b, self.bg_b, self.Wc,
                self.Uc, self.bc]

    def readout_parameters(self):
        """(v2.2 --wave-readout norm) g, w_e, b_e: trained by the main loss (the read path), no gate loss."""
        return [self.g_read, self.w_e, self.b_e] if self.readout_norm else []

    def _target(self, neg_e, w, y_old, xf, side, toks):
        """The unit write value: v2.1 (dl == d) normalize(-e_hat + w y_hat_old); v2.2 latent: that correction from
        the first d dims of -E / the old read, then _latent with the token's xf and workspace summary."""
        nrm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        if self.dl == self.d:
            return nrm(nrm(neg_e) + w[:, None] * nrm(y_old))
        d = self.d
        corr = nrm(nrm(neg_e[:, :d]) + w[:, None] * nrm(y_old[:, :d]))
        return nrm(self._latent(corr, xf, None if side is None else side[toks]))

    @torch.no_grad()
    def _entry0(self, tr, i, sheet):
        """(v2.2 delta write) rows i of a trace -> the recalled units / phases at entry 0 (n, L, C) and erase (n,) =
        the key match at entry 0 is confident (z >= --wave-z-conf)."""
        chi0 = tr["chi"][i, 0].mean(-1)
        z0, _ = self._zc(chi0, tr["loc"][i, 0], sheet)
        return tr["v"][i, 0], tr["ph"][i, 0], z0 >= self.z_conf

    @torch.no_grad()
    def _writes(self, rec, Ep, e_g, h_g, h_p, conf_tok):
        dec, st = rec["dec"], self.store
        ce, H, V, B, T, idx = rec["ce_h"]
        N = B * T
        allow = rec["read_tok"] & ~conf_tok
        wt, strength = self.policy.select(ce, H, V, B, T, idx=idx, eot=self.eot, allow=allow)
        ok = strength > 0
        if not bool(ok.any()):
            return
        wt, strength = wt[ok], strength[ok]
        scale = self.kappa.clamp(min=1e-6) * self.content
        # (fix F2: a "zero" target = 1e-6 x the micro-batch's largest |E_pair|; dL/dy of a mean loss is ~1 / N per pair,
        # so an absolute 1e-6 floor dropped real targets)
        tiny = 1e-6 * Ep.norm(dim=-1).amax().clamp(min=1e-30)
        nrm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        # global: tokens that read the global sheet; entry 0 with its keys
        gpos = torch.full((N,), -1, dtype=torch.long, device=Ep.device).index_put(
            (dec["GT"],), torch.arange(dec["GT"].numel(), device=Ep.device))
        gw = dec["gused"][wt]
        out_codes, out_tg, out_st, out_rec, n_g, n_p = [], [], [], [], 0, 0
        if bool(gw.any()):
            tg_ = wt[gw]
            gi = gpos[tg_]
            yg_tok = rec["y_g_tok"][tg_]                 # (fix F1)
            w = h_g[gi].clamp(min=0) * (rec["c_g"][gi] >= 0.5).float()
            tgt = self._target(-e_g[tg_], w, yg_tok, rec["xf"][tg_], rec.get("side"), tg_)
            good = e_g[tg_].norm(dim=-1) > tiny
            c0 = rec["codes"].take(tg_)
            out_codes.append(Codes(c0.loc[:, :1], c0.loc_s[:, :1], c0.code[..., :1], c0.code_w[..., :1],
                                   c0.code_d[..., :1], c0.feats, c0.code_e[:, :1]))
            out_tg.append(tgt * scale)
            out_st.append(strength[gw] * good.float())
            if self.delta:                # (v2.2: what entry 0 already recalls for these keys; erase iff confident)
                out_rec.append(self._entry0(rec["trg"], gi, 0))
            if self.map_beta:                          # (fix A: the global map learns on the global writes)
                self.address.learn_writes(rec["xf"][tg_[good]], self.map_beta)
            n_g = int((good & (strength[gw] > 0)).sum())
        # private: the first priv_cap read routed pairs by routing rank, at their read place with their keys
        vp_d = rec["vp_d"][wt, :6]
        rank = vp_d.long().cumsum(1)
        pick = vp_d & (rank <= self.priv_cap)
        if bool(pick.any()):
            ti, ki = pick.nonzero(as_tuple=True)
            t_ = wt[ti]
            flat = t_ * 7 + ki
            spos = torch.full((7 * N + 1,), -1, dtype=torch.long, device=Ep.device).index_put(
                (dec["S"],), torch.arange(dec["K"], device=Ep.device))[flat]
            prpos = torch.full((dec["K"],), -1, dtype=torch.long, device=Ep.device).index_put(
                (dec["PR"],), torch.arange(dec["PR"].numel(), device=Ep.device))[spos]
            Ei = Ep[spos]
            yp = rec["y_raw"][1][spos]
            w = h_p[prpos].clamp(min=0) * (rec["c_p"][prpos] >= 0.5).float()
            tgt = self._target(-Ei, w, yp, rec["xf"][t_], rec.get("side"), t_)
            good = Ei.norm(dim=-1) > tiny
            cp = rec["cp"].take(prpos)
            if cp.loc.size(1) > 1:        # (v2.2 strips: the write goes to entry 0, the strip's argmax place)
                cp = Codes(cp.loc[:, :1], cp.loc_s[:, :1], cp.code, cp.code_w, cp.code_d, cp.feats, cp.code_e[:, :1])
            out_codes.append(cp)
            out_tg.append(tgt * scale)
            out_st.append(strength[ti] * good.float())
            if self.delta:
                out_rec.append(self._entry0(rec["trp"], prpos, 1))
            pl = rec["pl"][prpos]
            m = pl < self.DUMMY
            self._learn_private_slow(rec["xhat"][t_[m & good]], pl[m & good])
            n_p = int((good & (strength[ti] > 0)).sum())
            self.cnt["writes_both"] += int((gw[ti]).sum())
        if out_codes:
            allc = Codes.cat(out_codes)
            sg = torch.cat(out_st)
            keep_ = allc.loc[:, 0] < self.DUMMY
            if self.delta:                # (v2.2: write only the error at entry 0)
                rv, rph, er = (torch.cat([r_[i] for r_ in out_rec]) for i in range(3))
                st.write_delta(allc, torch.cat(out_tg), sg * keep_.float(), rv, rph, er)
            else:
                st.write(allc, torch.cat(out_tg), sg * keep_.float())      # (one encode for all <= 48 targets)
        self.cnt["writes_g"] += n_g
        self.cnt["writes_p"] += n_p

    # ------------------------------------------------------------------ the private address
    @torch.no_grad()
    def _learn_private_warm(self, xf, e_k, t_k, slot_k, n=1024):
        """Before write_start: 1,024 sample tokens x their routed pairs of slot 0 and 1."""
        xhat = self.address.standardize(xf)
        N = xhat.size(0)
        stride = max(1, N // n)
        m = (slot_k < 2) & (t_k % stride == 0) & (e_k < self.n_live)
        if not bool(m.any()):
            return
        rows, xs = self.row_of[e_k[m]], xhat[t_k[m]]
        # first-sample init: one new prototype per row per micro-batch while Pinit < 4
        first = torch.full((self.S,), rows.numel(), dtype=torch.long, device=xs.device).scatter_reduce(
            0, rows, torch.arange(rows.numel(), device=xs.device), "amin")
        need = (first < rows.numel()) & (self.Pinit < self.J)
        nr = need.nonzero().squeeze(-1)
        if nr.numel():
            self.P_e[nr, self.Pinit[nr]] = xs[first[nr]]
            self.Pinit[nr] += 1
        # the rest: BMU among the initialised, sample-hit conscience 0.05, rate max(1/32, 1 / (1 + hits))
        d2 = (xs[:, None, :] - self.P_e[rows]).pow(2).sum(-1)              # (n, 4)
        ph = self.Ph[rows]
        pen = 0.05 * torch.log((ph + 1) / (ph.mean(-1, keepdim=True) + 1))
        okj = torch.arange(self.J, device=xs.device) < self.Pinit[rows][:, None]
        j = torch.where(okj, d2 + pen, torch.full_like(d2, INF)).argmin(-1)
        self._move(rows * self.J + j, xs, lambda hits: torch.clamp(1.0 / (1.0 + hits), min=1.0 / 32))
        self.Ph.view(-1).index_add_(0, rows * self.J + j, torch.ones_like(j, dtype=torch.float))

    @torch.no_grad()
    def _learn_private_slow(self, xs, places):
        """After write_start: the private writes; BMU with the write-load conscience, P += (x - P) / 128."""
        if xs.numel() == 0:
            return
        rows = (places - self.Pg) // self.J
        d2 = (xs[:, None, :] - self.P_e[rows]).pow(2).sum(-1)
        wl = self.Wld_p[rows]
        pen = 0.05 * torch.log((wl + 1) / (wl.mean(-1, keepdim=True) + 1))
        okj = torch.arange(self.J, device=xs.device) < self.Pinit[rows][:, None]
        j = torch.where(okj, d2 + pen, torch.full_like(d2, INF)).argmin(-1)
        self._move(rows * self.J + j, xs, lambda hits: torch.full_like(hits, 1.0 / 128), per_sample=True)
        self.Wld_p.mul_(1 - 1.0 / 1024).view(-1).index_add_(0, rows * self.J + (places - self.Pg) % self.J,
                                                           torch.ones(rows.numel(), device=xs.device))

    def _move(self, q, xs, rate_fn, per_sample=False):
        """P_e.flat[q] += rate (mean of its samples - P) (warm) or rate sum(x - P) (slow); one-hot GEMM sums."""
        flat = self.P_e.view(-1, self.d)
        oh = F.one_hot(q, flat.size(0)).float()                            # (n, S J)
        cnt = oh.sum(0)
        sums = oh.T @ xs
        hit = cnt > 0
        if per_sample:
            delta = (sums - cnt[:, None] * flat) * rate_fn(cnt)[:, None]
        else:
            delta = (sums / cnt.clamp(min=1)[:, None] - flat) * rate_fn(self.Ph.view(-1))[:, None]
        flat.add_(torch.where(hit[:, None], delta, torch.zeros_like(delta)))

    # ------------------------------------------------------------------ the null control
    @torch.no_grad()
    def _null(self, rec):
        st = self.store
        codes, n = rec["codes"], self.n_null
        N = len(codes)
        t = torch.arange(0, N, max(1, N // n), device=codes.loc.device)[:n]
        o = (t + N // 2) % N
        cg = Codes(codes.loc[t], codes.loc_s[t], codes.code[o], codes.code_w[o], codes.code_d[o], codes.feats[t],
                   codes.code_e[o])
        chi = st.null_observe(cg)
        self._null_stats(0, chi, st.last_null_place)
        pp = rec["place_p"][t, 0]                                          # (the slot-0 pair of the null tokens)
        m = pp < self.DUMMY
        if bool(m.any()):
            xh = rec["xhat"][o[m]]
            if self.Jr > 1:               # (v2.2: the null cue reads the same Jr strip entries as a private read)
                ple = rec["place_e"][t, 0][m]
                tr = st.trace(self._codes_at(xh, ple, self._proto(ple)))
                self._null_stats(1, tr["chibar"], tr["loc"].gather(1, tr["best"][:, None]).squeeze(1))
                return
            cp = self._codes_at(xh, pp[m], self._proto(pp[m]))
            tr = st.trace(cp)
            self._null_stats(1, tr["chibar"], pp[m])

    def _null_stats(self, sheet, chi, places):
        ld = self.store.load[places].float()
        b = torch.log2(ld + 1).floor().long().clamp(0, 7)
        oh = F.one_hot(b, 8).float()
        cnt = oh.sum(0)
        m_ = (oh.T @ chi) / cnt.clamp(min=1)
        v_ = ((oh.T @ (chi * chi)) / cnt.clamp(min=1) - m_ * m_).clamp(min=0)
        rate = 1 - 0.99 ** self.null_every
        seen, first = cnt > 0, self.null_n[sheet] == 0
        mu, sd = self.null_mu[sheet], self.null_sd[sheet]
        self.null_mu[sheet] = torch.where(seen, torch.where(first, m_, mu.lerp(m_, rate)), mu)
        self.null_sd[sheet] = torch.where(seen, torch.where(first, v_.sqrt(), (sd ** 2).lerp(v_, rate).sqrt()), sd)
        self.null_n[sheet] += seen.long()

    # ------------------------------------------------------------------ the step end
    @_ieee
    def after_step(self, step, refit_every=0):
        on = self.active
        self.n_opt += 1
        if not on or self.train_only is not None:
            for p in self.parameters():
                p.grad = None
            self.D_grad.zero_()
            return
        self._adamw()
        self.store.renorm()
        self._d_update()
        self.D_grad.zero_()
        with _rf("wave.sweep"):
            self._sweep()

    def _adamw(self):
        dec = [self.store.A, self.store.B, self.log_s]
        gate = self.gate_parameters()
        ro = self.readout_parameters()    # (v2.2: its own group and clip, lr --wave-dec-lr: |grad w_e| ~ |xf| would
        if self._opt is None:             # otherwise dominate the decoder group's clip)
            self._opt = torch.optim.AdamW([dict(params=gate, lr=self.lr), dict(params=dec, lr=self.dec_lr)]
                                          + ([dict(params=ro, lr=self.dec_lr)] if ro else []),
                                          betas=(0.9, 0.99), weight_decay=0.0)
            if self._opt_pending is not None:
                try:
                    self._opt.load_state_dict(self._opt_pending)
                except (ValueError, KeyError):
                    pass
                self._opt_pending = None
        for grp in (gate, dec, ro):
            ps = [p for p in grp if p.grad is not None]
            if ps:
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
        self._opt.step()
        self._opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.bp_b.clamp_(-4, 4)
            self.bg_b.clamp_(-4, 4)
            self.log_s.clamp_(math.log(0.05), 0.0)

    @torch.no_grad()
    def _d_update(self, b2=0.99, eps=1e-30, clip_d=1.0):
        """The factored update (PART 3): per memory row, R / Cv = EMA of the row / column mean squares of D_grad,
        V = R C / mean(R), U = G / sqrt(V), the row's RMS clipped; wd 0. Rows with a zero gradient do not move."""
        E, d, dl = self.n_live, self.d, self.dl                      # (D rows: dl (the latent), columns: d)
        for a, b in [(a_, min(E, a_ + 256)) for a_ in range(0, E, 256)] + [(self.sh, self.sh + 1)]:
            G = self.D_grad[a:b]
            live = G.flatten(1).abs().amax(-1) > 0
            R = self.R[a:b].mul(b2).add_(torch.linalg.vector_norm(G, dim=-1).square() / d, alpha=1 - b2)
            Cv = self.Cv[a:b].mul(b2).add_(torch.linalg.vector_norm(G, dim=-2).square() / dl, alpha=1 - b2)
            n = self.n_d[a:b] + live.long()
            self.R[a:b] = torch.where(live[:, None], R, self.R[a:b])
            self.Cv[a:b] = torch.where(live[:, None], Cv, self.Cv[a:b])
            self.n_d[a:b] = n
            bc = (1 - b2 ** n.clamp(min=1).double()).float()
            rh, ch = self.R[a:b] / bc[:, None], self.Cv[a:b] / bc[:, None]
            V = rh[:, :, None] * ch[:, None, :] / rh.mean(-1).clamp(min=eps)[:, None, None] + eps
            U = G * V.rsqrt()
            rms = torch.linalg.vector_norm(U, dim=(-2, -1)) / (d if dl == d else math.sqrt(dl * d))
            self.D[a:b].add_(U / (rms / clip_d).clamp(min=1)[:, None, None] * live[:, None, None].float(),
                             alpha=-self.d_lr)

    @torch.no_grad()
    def _sweep(self):
        """8 global places + 24 over the live strips per step (the column store's v2 rule, place by place)."""
        Pg, J, E = self.Pg, self.J, self.n_live
        pg = [(self.ptr_g + i) % Pg for i in range(min(self.sweep_g, Pg))]
        self.ptr_g = (self.ptr_g + len(pg)) % Pg
        n_strip = E * J
        pp = [Pg + (self.ptr_p + i) % n_strip for i in range(min(self.sweep_p, n_strip))]
        self.ptr_p = (self.ptr_p + len(pp)) % n_strip
        st = self.store
        save = st.owned
        st.owned = None
        ptr, ns = st.sweep_ptr, st.n_sweep
        for places in (pg, pp):
            st.n_sweep, st.sweep_ptr = 1, 0
            for p in places:
                st.sweep_ptr = p
                st.forget(0)
        st.sweep_ptr, st.n_sweep, st.owned = ptr, ns, save

    # ------------------------------------------------------------------ growth, train-only, logs
    @torch.no_grad()
    @_ieee
    def on_grow(self, src, E0, split=True):
        """Children copy the parent (owner decision 3): the strip's key + chain bytes and load, Wld_p, P_e, Ph, Pinit,
        the D row and R / Cv / n_d, the gate rows Wp / Wg / bp_b / bg_b and their AdamW moments."""
        src = [int(s) for s in (src.tolist() if torch.is_tensor(src) else src)]
        dst = list(range(E0, E0 + len(src)))
        assert not dst or dst[-1] < self.Emax, f"wave v21: {dst[-1] + 1} experts > --wave-max-experts {self.Emax}"
        self.n_live = E0 + len(src)
        if not dst:
            return
        st = self.store
        s_r, d_r = self.row_of[src], self.row_of[dst]
        kb, cb = NLAY * st.C * NFAN * UK, (NLAY - 1) * st.C * NW * UC
        for a, b in zip(s_r.tolist(), d_r.tolist()):
            for j in range(self.J):
                ps, pd = self.Pg + self.J * a + j, self.Pg + self.J * b + j
                st.key_syn[pd * kb:(pd + 1) * kb] = st.key_syn[ps * kb:(ps + 1) * kb].clone()
                st.chain_bits[pd * cb:(pd + 1) * cb] = st.chain_bits[ps * cb:(ps + 1) * cb].clone()
                st.load[pd] = st.load[ps]
        for t in (self.Wld_p, self.P_e, self.Ph, self.Pinit, self.D, self.R, self.Cv, self.n_d):
            t[d_r] = t[s_r].clone()
        for p in [self.Wp, self.Wg, self.bp_b, self.bg_b] + self.readout_parameters()[1:]:
            p.data[d_r] = p.data[s_r].clone()
            if self._opt is not None and p in self._opt.state:
                for k_ in ("exp_avg", "exp_avg_sq"):
                    if k_ in self._opt.state[p]:
                        self._opt.state[p][k_][d_r] = self._opt.state[p][k_][s_r].clone()

    def set_train_only(self, expert_ids):
        if expert_ids is not None:
            raise NotImplementedError("--memory-train-only with --wave-mode v21: train-only isolation is P2 (deferred)")
        super().set_train_only(None)

    def trainable_parameters(self, expert_ids):
        return []

    def describe(self, args=None):
        return (f"wave memory v2.1: global {self.Pg} places (address {self.address.side}x{self.address.side}, "
                f"{self.address.E} entries, fix A beta {self.map_beta}) + {self.S} strips x {self.J} places, DUMMY "
                f"{self.DUMMY}; budgets K 7N/2, K_pr 3N, K_g N/2, explore {self.explore}; write_start "
                f"{self.write_start}; v2.2: write {'delta' if self.delta else 'set'}, readout "
                f"{'norm' if self.readout_norm else 'raw'}, strip reads {self.Jr} entries; {self.nbytes() / 1e6:.1f} MB")

    def log_record(self, step=None):
        rec = {f"wmem_{k}": v for k, v in self.cnt.items()}
        rec.update({f"wmem_addr_{k}": v for k, v in self.address.stats().items()})
        rec["wmem_ready"], rec["wmem_active"] = int(self.ready), int(self.active)
        sa = getattr(self, "_stat_acc", None)
        if sa is not None:
            s = sa.tolist()
            pairs, toks = max(1, self.cnt["pairs"]), max(1, self.cnt["toks"])
            rec.update(wmem_want=s[0], wmem_excluded=s[1], wmem_demoted=s[2], wmem_read_p_share=s[3] / pairs,
                       wmem_read_g_share=s[4] / toks, wmem_explore_p=s[5] / pairs, wmem_explore_g=s[6] / toks)
            self._stat_acc = None
            self.cnt["pairs"] = self.cnt["toks"] = 0
        ld = self.store.load.float()
        lg, lp = ld[:self.Pg], ld[self.Pg:self.Pg + self.J * self.n_live]
        for nm, v in (("g", lg), ("p", lp)):
            if v.numel() and float(v.sum()) > 0:
                srt = v.sort(descending=True).values
                rec[f"wmem_load_{nm}_maxmean"] = float(srt[0] / v.mean().clamp(min=1e-9))
                rec[f"wmem_load_{nm}_top10"] = float(srt[:10].sum() / v.sum())
        wl = self.Wld_p[:self.n_live].sum(-1)
        if float(wl.sum()) > 0:
            srt = wl.sort(descending=True).values
            rec["wmem_expert_wload_maxmean"] = float(srt[0] / wl.mean().clamp(min=1e-9))
            rec["wmem_expert_wload_top10"] = float(srt[:10].sum() / wl.sum())
        rec["wmem_log_s"] = float(self.log_s.detach())
        if self._ratio is not None:
            rec["wmem_read_ratio"] = float(self._ratio)                      # (v2.2: |read| / |xf| over read pairs)
            rec["wmem_fire_rate"] = float(self._fire)                        # (v2.2: reads with c > 0.5)
        if self.readout_norm:
            rec["wmem_gate_s_b_mean"] = float(self.b_e[:self.n_live].detach().mean())
            rec["wmem_g_read_rms"] = float(self.g_read.detach().pow(2).mean().sqrt())
        rec["wmem_erased"] = int(self.store.stats_acc.get("erased", 0))
        if self.last_stats:
            rec["wmem_gate_loss"] = float(self.last_stats["gate_loss"])
        return rec

    def health(self):
        return {f"wmem_{k}": v for k, v in self.address.stats().items()}
