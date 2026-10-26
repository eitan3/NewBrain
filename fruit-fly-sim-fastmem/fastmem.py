"""FastMem (SPEC_fastmem.md + the lite / per-stream revision): a fast-weight synapse matrix per expert + a global one,
delta-rule writes with decay, read q M. --memory fastmem. Our own code, written from the spec's equations.

SIZES (--fastmem-size; the explicit --fastmem-heads / -dk / -dv / -global / -global-topk / -read-pass / -dtype override)
  lite (default): H 2 heads x dk = dv 128 per memory row, global 8 heads (top-2 per token); the memory maths in bf16
       (the triangular solve and every state accumulate in fp32); the scan once at --fastmem-read-pass 3
  full: H 4 x 512 x 512, global 32 (top-4), fp32, the scan at every pass (the first build)
STATE (buffers, saved in the memory file; fixed size)
  --fastmem-stream 1 (default): one memory per data stream (= per user): Ms [streams, S, H, dk, dv] and
     Mgs [streams, G, 1, dk, dv] fp32 (16 streams); S = Emax + 1 memory rows (rows 0..Emax-1 = the routed experts,
     brain id = row; row Emax = the shared expert). The trainer names each batch row's stream (set_streams, from the
     stream loader); a stream's memory is carried across micro-batches and optimizer steps (detached between
     micro-batches: the next micro-batch of the same stream reads the previous one's end state), reset to zero at
     the stream's start; no cross-stream merge. Without stream ids (evaluation, random windows) every sequence starts
     from zero memory.
  --fastmem-stream 0: one memory for all, M [S, H, dk, dv] / Mg [G, 1, dk, dv]; after the step M <- M + mean_b(M_end_b
     - M) (the first build)
MATHS (per memory row e and head h, from the expert input thought xf)
  q = L2norm(xf Wq_e,h), k = L2norm(xf Wk_e,h), v = xf Wv_e,h; beta = sigmoid(xf Wb + bb) (per head and key dim),
  omega = sigmoid(xf Ww + bw) (per head and value dim); lam = -exp(s_e,h) softplus(xf Wl_e,h + bl_e,h) (one scalar per
  head; init exp(lam) = 1 - 1e-4)
  WRITE (per routed token in order): U = sqrt(beta) k;  M <- (I - U^T U) exp(lam) M + U^T (omega v)
  READ: R = q M / sqrt(dk); out_h = RMSNorm(R_h) 2 tanh(silu(xf Wg_e,h + bg_e,h) / 2); concat the H heads; then D_e
  [H dv, d] (init 0, the factored update of wave_v21's D) into the expert's input, times the pre-read gate. The shared
  row -> the shared expert's input; the global sheet (each token its top-k heads, weighted by the router's softmax)
  -> the shared expert's input.
IN-STEP (a segment = one sequence's tokens routed to one memory row, in time order)
  the causal scan from the start state in the exact closed (UT / WY) form over the segment: g_i = sum_{j<=i} lam_j,
      L_ij = exp(g_i - g_j) u_i.u_j (j < i);  (I + L) Delta = W - exp(g) U M          (unit lower-triangular solve)
      O_i = exp(g_{i-1}) q_i M + sum_{j<i} exp(g_{i-1} - g_j) (q_i.u_j) Delta_j        (token i reads M + the writes
                                                                                         of the tokens before it)
      M_end = exp(g_n) M + sum_j exp(g_n - g_j) u_j Delta_j^T              (== the token recurrence: tests (t1))
  PASSES: --fastmem-read-pass p >= 0 (lite: 3): passes k < p' = min(p, K - 1) read the start state only (no in-step
  prefix); pass p' runs the scan (with a graph while training, also outside the backprop window), writes once (its
  end state) and its read is reused at the later passes (per (token, expert) as at p'; 0 for pairs not routed there).
  p = -1 (full): every pass scans, the final pass writes.
GRADIENT (no memory-side loss): the main loss trains the projections, the gates, the router and D through the read.
  --fastmem-write-grad full (default): the write side (k, v, beta, omega, lam) passes gradient into xf; detached: it
  sees xf.detach(). The start state and the carried / merged state are detached.
PRE-READ GATE (v2.1-style): h = silu(G1 rms(xf.detach()) + b) [N, 16]; score = Wp[row].h + bp_b[row] (row S = global);
  reads iff score > 0 or explored (a constant 5% of the pairs while training); the read x (hard + sigmoid(score) -
  sigmoid(score).detach()) (straight-through: the main loss trains the gate; a closed pair contributes exactly 0).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from memory_api import MemoryBase
from wave_v21 import ieee_fp32

SIZES = dict(lite=dict(heads=2, dk=128, dv=128, global_=8, topk=2, read_pass=3, dtype="bf16"),
             full=dict(heads=4, dk=512, dv=512, global_=32, topk=4, read_pass=-1, dtype="fp32"))
FASTMEM_DEFAULTS = dict(
    fastmem_seed=1234, fastmem_max_experts=0, fastmem_size="lite",
    fastmem_heads=0, fastmem_dk=0, fastmem_dv=0, fastmem_global=0, fastmem_global_topk=0,   # (0: the size's value)
    fastmem_read_pass=-2, fastmem_dtype="",                                                 # (-2 / "": the size's)
    fastmem_stream=1, fastmem_streams=16,
    fastmem_proj="expert",            # (expert: Wq / Wk / Wv per memory row (the spec); shared: one set for all rows)
    fastmem_write_grad="full",        # (full: the writes pass gradient into xf; detached: the write side sees xf.detach())
    fastmem_explore=0.05, fastmem_lr=1e-3, fastmem_d_lr=1e-4, fastmem_decay0=1e-4,
    fastmem_ckpt=0)                   # (1: the memory's per-pass graph recomputed in backward (VRAM saver))


def _l2n(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-12)


def _proj(X, W):
    """X [R, C, d] @ W [R, d, F] per row (W [1, d, F]: one matrix for all rows) -> [R, C, F]."""
    W = W.to(X.dtype)
    if W.size(0) == 1:
        return (X.reshape(-1, X.size(-1)) @ W[0]).view(X.size(0), X.size(1), -1)
    return torch.bmm(X, W)


def scan_read(q, u, w, lam, M, segb, segp, valid, B, Cs, prefix=True, merge=False):
    """The causal scan + read of one family of memory rows (module doc), row layout [R, C]:
    q, u [R, C, H, dk]; w [R, C, H, dv]; lam [R, C, H] (u / w / lam unused without prefix); the start state M
    [R, H, dk, dv] (shared by the sequences) or [R, B, H, dk, dv] (one per sequence / stream), no grad;
    segb / segp [R, C] the slot's sequence and its position in the (row, sequence) segment; valid [R, C].
    The matmuls run in q's dtype (bf16 for lite), the solve and the state accumulate in fp32 (or float64).
    prefix False: the read of M alone. -> O [R, C, H, dv] (fp32), and with merge: per sequence the end states
    M_end [R, B, H, dk, dv] (5-d M), else the factors of sum_b (M_end_b - M) (merge_into adds them); detached."""
    R, C, H, dk = q.shape
    dv = M.size(-1)
    cd, ft = q.dtype, torch.promote_types(q.dtype, torch.float32)
    per_seq = M.dim() == 5
    nb = B * Cs
    idx = torch.where(valid, segb * Cs + segp, torch.full_like(segp, nb))                # (invalid -> a dump slot)
    vm = valid[..., None, None].to(cd)

    def seg(t):                       # [R, C, H, f] -> [R, B, H, Cs, f] (zeros at the padding)
        f = t.size(-1)
        z = t.new_zeros(R, nb + 1, H * f).scatter(1, idx[..., None].expand(-1, -1, H * f), t.reshape(R, C, H * f))
        return z[:, :nb].view(R, B, Cs, H, f).transpose(2, 3)

    Mc = M.to(cd)
    qs = seg(q * vm)
    if per_seq:
        qMs = qs @ Mc                                                                      # [R, B, H, Cs, dv]
    else:
        qMs = seg(torch.matmul(q.transpose(1, 2), Mc).transpose(1, 2))
    dM = None
    if not prefix:
        Os = qMs.to(ft)
    else:
        us, ws = seg(u * vm), seg(w * vm)
        uMs = us @ Mc if per_seq else seg(torch.matmul(u.transpose(1, 2), Mc).transpose(1, 2))
        ls = seg((lam.to(ft) * vm[..., 0].to(ft))[..., None])[..., 0]
        g = ls.cumsum(-1)                                                                  # [R, B, H, Cs]
        gp = g - ls                                                                        # g_{i-1}
        low = torch.ones(Cs, Cs, dtype=torch.bool, device=u.device).tril(-1)
        ninf = torch.tensor(-math.inf, device=u.device, dtype=ft)
        DL = torch.exp(torch.where(low, g[..., :, None] - g[..., None, :], ninf))
        DP = torch.exp(torch.where(low, gp[..., :, None] - g[..., None, :], ninf))
        L = DL * (us @ us.transpose(-1, -2)).to(ft)
        P = DP * (qs @ us.transpose(-1, -2)).to(ft)
        rhs = ws.to(ft) - torch.exp(g)[..., None] * uMs.to(ft)
        Delta = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)   # (I + L)^-1 rhs
        Os = torch.exp(gp)[..., None] * qMs.to(ft) + (P.to(cd) @ Delta.to(cd)).to(ft)    # [R, B, H, Cs, dv]
        if merge:
            with torch.no_grad():
                gn = g[..., -1]                                                            # [R, B, H] (padding: lam 0)
                ut = torch.exp(gn[..., None] - g)[..., None] * us.to(ft)                   # [R, B, H, Cs, dk]
                if per_seq:
                    dM = torch.exp(gn)[..., None, None] * M.to(ft) + torch.einsum(
                        "rbhcd,rbhce->rbhde", ut.to(cd), Delta.detach().to(cd)).to(ft)
                else:
                    dM = ((torch.exp(gn) - 1).sum(1), ut, Delta.detach())
    O = Os.transpose(2, 3).reshape(R, nb, H * dv)
    O = torch.cat([O, O.new_zeros(R, 1, H * dv)], 1).gather(1, idx[..., None].expand(-1, -1, H * dv))
    return O.view(R, C, H, dv), dM


@torch.no_grad()
def merge_into(acc, M, parts):
    """acc [R, H, dk, dv] += coef M + sum_{b, c} ut^T Delta, in place (no state-sized temporaries but coef M's)."""
    coef, ut, Delta = parts
    R, B, H, Cs, dk = ut.shape
    acc.addcmul_(coef[..., None, None].to(acc.dtype).expand_as(M), M)
    a = acc.view(R * H, dk, -1)
    a.baddbmm_(ut.permute(0, 2, 4, 1, 3).reshape(R * H, dk, B * Cs).to(acc.dtype),
               Delta.permute(0, 2, 1, 3, 4).reshape(R * H, B * Cs, -1).to(acc.dtype))


class FastMem(MemoryBase):
    kind = "fastmem"

    def __init__(self, d, n_exp, **c):
        super().__init__()
        c = dict(FASTMEM_DEFAULTS, **{k: v for k, v in c.items() if k in FASTMEM_DEFAULTS})
        assert c["fastmem_size"] in SIZES, f"--fastmem-size {c['fastmem_size']!r}: lite or full"
        assert c["fastmem_proj"] in ("expert", "shared"), f"--fastmem-proj {c['fastmem_proj']!r}: expert or shared"
        assert c["fastmem_write_grad"] in ("full", "detached"), \
            f"--fastmem-write-grad {c['fastmem_write_grad']!r}: full or detached"
        z_ = SIZES[c["fastmem_size"]]
        pick = lambda k, s: z_[s] if c[k] in (({-2} if k == "fastmem_read_pass" else {0, ""}) | {None}) else c[k]
        self._cfg = c
        self.d, self.seed = int(d), int(c["fastmem_seed"])
        self.Emax = max(int(c["fastmem_max_experts"] or 0), int(n_exp))
        self.S, self.sh, self.n_live = self.Emax + 1, self.Emax, int(n_exp)
        H, dk, dv = int(pick("fastmem_heads", "heads")), int(pick("fastmem_dk", "dk")), int(pick("fastmem_dv", "dv"))
        G, self.gk = int(pick("fastmem_global", "global_")), int(pick("fastmem_global_topk", "topk"))
        self.H, self.dk, self.dv, self.G = H, dk, dv, G
        self.read_pass = int(pick("fastmem_read_pass", "read_pass"))
        dt = pick("fastmem_dtype", "dtype")
        assert dt in ("bf16", "fp32"), f"--fastmem-dtype {dt!r}: bf16 or fp32"
        self.cd = torch.bfloat16 if dt == "bf16" else torch.float32
        self.stream, self.n_streams = bool(c["fastmem_stream"]), int(c["fastmem_streams"])
        self.write_full = c["fastmem_write_grad"] == "full"
        self.explore, self.lr, self.d_lr = float(c["fastmem_explore"]), float(c["fastmem_lr"]), float(c["fastmem_d_lr"])
        self.ckpt = bool(c["fastmem_ckpt"])
        S, nP = self.S, (self.S if c["fastmem_proj"] == "expert" else 1)
        gen = torch.Generator().manual_seed(self.seed + 101)         # (own generator: the global stream is untouched)
        r = lambda *s, std: nn.Parameter(torch.randn(*s, generator=gen) * std)
        z = lambda *s, v=0.0: nn.Parameter(torch.full(s, float(v)))
        sd = 1.0 / math.sqrt(d)
        s0 = math.log(float(c["fastmem_decay0"]) / math.log(2.0))   # (exp(s0) softplus(0) = decay0: exp(lam) ~ 1 - 1e-4)
        # the expert rows (routed + shared)
        self.Wq, self.Wk, self.Wv = r(nP, d, H * dk, std=sd), r(nP, d, H * dk, std=sd), r(nP, d, H * dv, std=sd)
        self.Wb, self.bb = r(d, H * dk, std=0.1 * sd), z(H * dk)
        self.Ww, self.bw = r(d, H * dv, std=0.1 * sd), z(H * dv, v=2.0)
        self.Wl, self.bl, self.s = r(S, d, H, std=0.1 * sd), z(S, H), z(S, H, v=s0)
        self.Wg, self.bg = r(S, d, H, std=0.1 * sd), z(S, H, v=1.0)
        self.g_norm = z(H * dv, v=1.0)
        self.D = z(S, H * dv, d)                                      # (D_e^T: the output layer, init 0)
        # the global sheet (G single-head rows)
        self.Wq_g, self.Wk_g, self.Wv_g = r(G, d, dk, std=sd), r(G, d, dk, std=sd), r(G, d, dv, std=sd)
        self.Wb_g, self.bb_g = r(d, dk, std=0.1 * sd), z(dk)
        self.Ww_g, self.bw_g = r(d, dv, std=0.1 * sd), z(dv, v=2.0)
        self.Wl_g, self.bl_g, self.s_g = r(G, d, 1, std=0.1 * sd), z(G, 1), z(G, 1, v=s0)
        self.Wg_g, self.bg_g = r(G, d, 1, std=0.1 * sd), z(G, 1, v=1.0)
        self.g_norm_g = z(dv, v=1.0)
        self.D_g = z(G, dv, d)
        self.Wr = r(d, G, std=sd)                                     # (the global router)
        # the pre-read gate (rows 0..S-1 the expert rows, S the global sheet)
        self.G1_w, self.G1_b = r(16, d, std=sd), z(16)
        self.Wp, self.bp_b = r(S + 1, 16, std=0.01), z(S + 1)
        # state
        if self.stream:
            self.register_buffer("M", torch.zeros(self.n_streams, S, H, dk, dv))
            self.register_buffer("Mg", torch.zeros(self.n_streams, G, 1, dk, dv))
        else:
            self.register_buffer("M", torch.zeros(S, H, dk, dv))
            self.register_buffer("Mg", torch.zeros(G, 1, dk, dv))
        for nm, rows, cols in (("", S, H * dv), ("_g", G, dv)):       # (D's factored second moments, wave_v21 PART 3)
            self.register_buffer("Dr" + nm, torch.zeros(rows, cols))
            self.register_buffer("Dc" + nm, torch.zeros(rows, d))
            self.register_buffer("Dn" + nm, torch.zeros(rows, dtype=torch.long))
        self.n_passes, self.train_forward = None, False
        self.streams, self.stream_start = None, None
        self._live, self._acc, self._accg, self._nseq = False, None, None, 0
        self._fw, self._pend = None, None
        self._opt, self._opt_pending, self.n_fwd, self.n_opt = None, None, 0, 0
        self._stats = {}

    # ------------------------------------------------------------------ state
    def get_extra_state(self):
        opt = self._opt.state_dict() if self._opt is not None else self._opt_pending
        return dict(n_fwd=self.n_fwd, n_opt=self.n_opt, n_live=self.n_live, opt=opt)

    def set_extra_state(self, st):
        self.n_fwd, self.n_opt = int(st.get("n_fwd", 0)), int(st.get("n_opt", 0))
        self.n_live = int(st.get("n_live", self.n_live))
        self._opt, self._opt_pending = None, st.get("opt")

    def memory_cfg(self):
        return dict(self._cfg)

    ROW_PARAMS = ("Wl", "bl", "s", "Wg", "bg", "D")                  # (per memory row; Wq / Wk / Wv too with proj expert)
    ROW_BUFS = ("Dr", "Dc", "Dn")

    def _row_tensors(self):
        ps = list(self.ROW_PARAMS) + (["Wq", "Wk", "Wv"] if self.Wq.size(0) == self.S else [])
        return [(n, getattr(self, n)) for n in ps] + [("Wp", self.Wp), ("bp_b", self.bp_b)]

    def _mrow(self, row):
        """Index of one memory row in M (per stream: all the streams' copies of the row)."""
        return (slice(None), row) if self.stream else (row,)

    def expert_state(self, e):
        """One memory row's state alone (brain id e; -1 = the shared expert): its M (every stream's), its parameters
        and D's moments."""
        row = self.sh if e == -1 else int(e)
        out = {n: t.detach()[row].clone().cpu() for n, t in self._row_tensors()}
        out.update({n: getattr(self, n)[row].clone().cpu() for n in self.ROW_BUFS})
        out["M"] = self.M[self._mrow(row)].clone().cpu()
        return out

    @torch.no_grad()
    def load_expert_state(self, e, st):
        row = self.sh if e == -1 else int(e)
        for n, t in self._row_tensors() + [(n, getattr(self, n)) for n in self.ROW_BUFS]:
            t[row] = st[n].to(t.device, t.dtype)
        self.M[self._mrow(row)] = st["M"].to(self.M.device)

    @torch.no_grad()
    def reset_expert(self, e):
        """The row's fast weights start empty (every stream's; its learned projections / D are kept)."""
        self.M[self._mrow(self.sh if e == -1 else int(e))] = 0

    def stream_state(self, s):
        """One stream's (user's) memory alone: its M and Mg."""
        assert self.stream, "stream_state: --fastmem-stream 0"
        return dict(M=self.M[s].clone().cpu(), Mg=self.Mg[s].clone().cpu())

    @torch.no_grad()
    def load_stream_state(self, s, st):
        self.M[s] = st["M"].to(self.M.device)
        self.Mg[s] = st["Mg"].to(self.Mg.device)

    @torch.no_grad()
    def reset_stream(self, s):
        self.M[s] = 0
        self.Mg[s] = 0

    def set_streams(self, streams, starts=None):
        """The trainer, before a training micro-batch: the stream of each batch row (None: no carried memory) and
        which rows start their stream (their memory is reset first)."""
        self.streams = None if streams is None else [int(s) for s in streams]
        if self.streams is not None:
            assert max(self.streams) < self.n_streams, f"fastmem: stream {max(self.streams)} >= {self.n_streams}"
            for s, st in zip(self.streams, starts if starts is not None else [False] * len(self.streams)):
                if st:
                    self.reset_stream(s)

    # ------------------------------------------------------------------ the forward
    def begin_forward(self):
        self._live = bool(self.training and self.train_forward and torch.is_grad_enabled())
        self._fw = None
        if self._live:
            self.n_fwd += 1

    def _gen(self):
        g = torch.Generator(device=self.M.device)
        g.manual_seed(self.seed * 1_000_003 + self.n_fwd * 17 + 5)
        return g

    def _gate(self, h, rows, gen):
        """The pre-read gate's multiplier per (token, row) pair: hard + sigmoid - sigmoid.detach()."""
        sc = (self.Wp[rows] * h).sum(-1) + self.bp_b[rows]
        hard = sc.detach() > 0
        if self._live and self.explore > 0:
            hard = hard | (torch.rand(sc.shape, generator=gen, device=sc.device) < self.explore)
        sg = torch.sigmoid(sc)
        return hard.to(sc.dtype) + sg - sg.detach(), hard

    @staticmethod
    def _segments(tokpos, valid, T, B):
        """segb (the slot's sequence), segp (its position among the row's tokens of that sequence; slots are in token
        order within a row), Cs = the longest segment (one host sync)."""
        R, C = valid.shape
        segb = torch.where(valid, tokpos // T, torch.zeros_like(tokpos))
        slot = torch.arange(C, device=valid.device).expand(R, C)
        key = torch.arange(R, device=valid.device)[:, None] * B + segb
        first = torch.full((R * B,), C, dtype=torch.long, device=valid.device).scatter_reduce(
            0, key.reshape(-1), torch.where(valid, slot, torch.full_like(slot, C)).reshape(-1), "amin")
        segp = torch.where(valid, slot - first[key], torch.zeros_like(slot))
        return segb, segp, int(segp.max()) + 1

    def _family(self, X, Xw, valid, segb, segp, B, Cs, M, W, prefix, merge):
        """Projections + scan + read-out of one family in its row layout -> (y [R, C, H dv] fp32, dM or None)."""
        Wq, Wk, Wv, Wb, bb, Ww, bw, Wl, bl, s, Wg, bg, gn = W
        R, C, _ = X.shape
        H, dk, dv = M.size(-3), M.size(-2), M.size(-1)
        Xc = X.to(self.cd)
        q = _l2n(_proj(Xc, Wq).view(R, C, H, dk))
        u = w = lam = None
        if prefix:
            Xwc = Xw.to(self.cd)
            k = _l2n(_proj(Xwc, Wk).view(R, C, H, dk))
            v = _proj(Xwc, Wv).view(R, C, H, dv)
            beta = torch.sigmoid(Xwc @ Wb.to(self.cd) + bb.to(self.cd)).view(R, C, H, dk)
            omega = torch.sigmoid(Xwc @ Ww.to(self.cd) + bw.to(self.cd)).view(R, C, H, dv)
            lam = -torch.exp(s)[:, None, :] * F.softplus(_proj(Xw.float(), Wl) + bl[:, None, :])   # [R, C, H] fp32
            u, w = torch.sqrt(beta) * k, omega * v
        O, dM = scan_read(q, u, w, lam, M, segb, segp, valid, B, Cs, prefix, merge)
        Rh = O / math.sqrt(dk)
        Rh = Rh * torch.rsqrt(Rh.pow(2).mean(-1, keepdim=True) + 1e-12)
        gate = 2 * torch.tanh(F.silu(_proj(X, Wg) + bg[:, None, :]) / 2)                       # [R, C, H]
        y = (Rh * gate[..., None]).reshape(R, C, H * dv) * gn
        return y, dM

    def _cast(self, name):
        """A parameter in the maths dtype, cast once per forward (bf16 lite: no per-pass copies of the big weights)."""
        p = getattr(self, name)
        if self.cd == p.dtype:
            return p
        c = self._fw.setdefault("cast", {})
        if name not in c:
            c[name] = p.to(self.cd)
        return c[name]

    def _run(self, fn, *a):
        if self.ckpt and torch.is_grad_enabled():
            return checkpoint(fn, *a, use_reentrant=False)
        return fn(*a)

    def _start_state(self, B, dev):
        """This forward's start states: (M_rows, Mg_rows); per stream [S, B, H, dk, dv] / [G, B, 1, dk, dv] (the
        batch rows' streams, or zeros), else the shared M / Mg."""
        if not self.stream:
            return self.M, self.Mg
        if self._live and self.streams is not None:
            assert len(self.streams) == B, f"fastmem: {len(self.streams)} stream ids for a batch of {B}"
            ix = torch.tensor(self.streams, device=dev)
            return self.M[ix].transpose(0, 1).contiguous(), self.Mg[ix].transpose(0, 1).contiguous()
        return (self.M.new_zeros(self.S, B, self.H, self.dk, self.dv),
                self.Mg.new_zeros(self.G, B, 1, self.dk, self.dv))

    def __call__(self, xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=None):
        k = int(call or 0)
        K = self.n_passes or (k + 1)
        if self.read_pass < 0:                       # (full: every pass scans, the final one writes)
            prefix, write, reuse = True, k == K - 1, False
        else:
            rp = min(self.read_pass, K - 1)
            prefix, write, reuse = k == rp, k == rp, k > rp
        B = x.size(0)
        T = N // B
        if reuse:
            if self._fw is None or "r3" not in self._fw:
                return buf, x
            fw = self._fw
            eq = fw["eid"][t_k] == e_k[:, None]
            kidx = eq.to(torch.int8).argmax(-1).long()
            idx = torch.where(eq.any(-1), t_k * fw["kk"] + kidx, torch.full_like(t_k, N * fw["kk"]))
            buf = buf.index_put((e_k, p_k), fw["r3"][idx].to(buf.dtype), accumulate=True)
            return buf, x + fw["rs"].to(x.dtype).view(x.shape)
        live_w = write and self._live
        with torch.autocast(xf.device.type, enabled=False), ieee_fp32(), \
                torch.set_grad_enabled(torch.is_grad_enabled() or (live_w and self.read_pass >= 0)):
            if self._fw is None:
                self._fw = dict(M=self._start_state(B, xf.device))
            for nm in ("Wq", "Wk", "Wv", "Wq_g", "Wk_g", "Wv_g"):   # (cast once, outside the checkpointed parts)
                self._cast(nm)
            Mrows, Mgr = self._fw["M"]
            X = xf.float()
            Xw = X if self.write_full else X.detach()
            gen = self._gen() if self._live else None
            xd = X.detach()
            h = F.silu((xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + 1e-12)) @ self.G1_w.t() + self.G1_b)
            Wrow = lambda W_, sl: W_[sl] if W_.size(0) == self.S else W_     # (slices: views, no copy)
            merge = live_w
            # routed experts: the capacity layout of RoutedMLP
            one = torch.ones((), dtype=torch.bool, device=xf.device)
            ve = torch.zeros(E, C, dtype=torch.bool, device=xf.device).index_put((e_k, p_k), one)
            tk = torch.zeros(E, C, dtype=torch.long, device=xf.device).index_put((e_k, p_k), t_k)
            segb, segp, Cs = self._segments(tk, ve, T, B)
            rsl, ssl = slice(0, E), slice(self.sh, self.sh + 1)

            def routed(X, Xw):
                Xe = X.new_zeros(E, C, self.d).index_put((e_k, p_k), X[t_k])
                Xwe = Xw.new_zeros(E, C, self.d).index_put((e_k, p_k), Xw[t_k])
                W = (Wrow(self._cast("Wq"), rsl), Wrow(self._cast("Wk"), rsl), Wrow(self._cast("Wv"), rsl), self.Wb,
                     self.bb, self.Ww, self.bw, self.Wl[rsl], self.bl[rsl], self.s[rsl], self.Wg[rsl], self.bg[rsl], self.g_norm)
                y, dM = self._family(Xe, Xwe, ve, segb, segp, B, Cs, Mrows[rsl], W, prefix, merge)
                return torch.bmm(y, self.D[rsl]), dM

            oe, dMe = self._run(routed, X, Xw)
            mult_e, hard_e = self._gate(h[t_k], e_k, gen)
            r_p = oe[e_k, p_k] * mult_e[:, None]
            # the shared expert: one row, its segments = the sequences
            ar = torch.arange(N, device=xf.device)

            def shared(X, Xw):
                W = (Wrow(self._cast("Wq"), ssl), Wrow(self._cast("Wk"), ssl), Wrow(self._cast("Wv"), ssl), self.Wb, self.bb,
                     self.Ww, self.bw,
                     self.Wl[ssl], self.bl[ssl], self.s[ssl], self.Wg[ssl], self.bg[ssl], self.g_norm)
                y, dM = self._family(X[None], Xw[None], torch.ones(1, N, dtype=torch.bool, device=xf.device),
                                     (ar // T)[None], (ar % T)[None], B, T, Mrows[ssl], W, prefix, merge)
                return (y[0] @ self.D[self.sh]), dM

            os_, dMs = self._run(shared, X, Xw)
            mult_s, hard_s = self._gate(h, torch.full((N,), self.sh, device=xf.device, dtype=torch.long), gen)
            r_s = os_ * mult_s[:, None]
            # the global sheet: top-gk of G heads per token
            lg = xd @ self.Wr
            top = lg.detach().topk(self.gk, -1)
            pw = torch.softmax(lg.gather(1, top.indices), -1)                                   # [N, gk]
            hg = top.indices.reshape(-1)
            tg = ar.repeat_interleave(self.gk)
            order = torch.argsort(hg * N + tg)
            hg, tg = hg[order], tg[order]
            cnt = torch.bincount(hg, minlength=self.G)
            Cg = max(1, int(cnt.max()))
            pg = torch.arange(hg.numel(), device=xf.device) - (cnt.cumsum(0) - cnt)[hg]
            vg = torch.zeros(self.G, Cg, dtype=torch.bool, device=xf.device).index_put((hg, pg), one)
            tkg = torch.zeros(self.G, Cg, dtype=torch.long, device=xf.device).index_put((hg, pg), tg)
            sbg, spg, Csg = self._segments(tkg, vg, T, B)

            def glob(X, Xw):
                Xg = X.new_zeros(self.G, Cg, self.d).index_put((hg, pg), X[tg])
                Xwg = Xw.new_zeros(self.G, Cg, self.d).index_put((hg, pg), Xw[tg])
                W = (self._cast("Wq_g"), self._cast("Wk_g"), self._cast("Wv_g"), self.Wb_g, self.bb_g, self.Ww_g,
                     self.bw_g, self.Wl_g, self.bl_g, self.s_g, self.Wg_g, self.bg_g, self.g_norm_g)
                y, dM = self._family(Xg, Xwg, vg, sbg, spg, B, Csg, Mgr, W, prefix, merge)
                return torch.bmm(y, self.D_g), dM

            og, dMg = self._run(glob, X, Xw)
            mult_g, hard_g = self._gate(h, torch.full((N,), self.S, device=xf.device, dtype=torch.long), gen)
            wg = pw.reshape(-1)[order] * mult_g[tg]
            r_g = X.new_zeros(N, self.d).index_add(0, tg, og[hg, pg] * wg[:, None])
            r_sg = r_s + r_g
            if merge:
                self._write(dMe, dMs, dMg, E, B)
            if self.read_pass >= 0 and write:        # (the read pass: its read is reused at the later passes)
                kk = max(1, int(counts.sum()) // N)
                eid = torch.full((N * kk + 1,), -1, dtype=torch.long, device=xf.device)
                pos = torch.where(slot_k < kk, t_k * kk + slot_k, torch.full_like(t_k, N * kk))
                eid = eid.index_put((pos,), e_k.long())[:N * kk].view(N, kk)
                r3 = r_p.new_zeros(N * kk + 1, self.d).index_put((pos,), r_p)
                r3[N * kk] = 0
                self._fw.update(eid=eid, kk=kk, r3=r3, rs=r_sg)
            if self._live or not self.training:
                with torch.no_grad():
                    xn = X.detach().norm(dim=-1).clamp(min=1e-6)
                    self._stats = dict(open_e=hard_e.float().mean(), open_s=hard_s.float().mean(),
                                       open_g=hard_g.float().mean(),
                                       read_ratio=(r_p.norm(dim=-1) / xn[t_k]).mean(),
                                       read_ratio_s=(r_sg.norm(dim=-1) / xn).mean())
        buf = buf.index_put((e_k, p_k), r_p.to(buf.dtype), accumulate=True)
        return buf, x + r_sg.to(x.dtype).view(x.shape)

    @torch.no_grad()
    def _write(self, dMe, dMs, dMg, E, B):
        """The write of this forward: per stream the end states (applied after backward: the graph still holds the
        start states), else the merge factors into the step's accumulators."""
        if self.stream:
            if self.streams is not None:
                self._pend = (dMe, dMs, dMg, E)
            return
        if self._acc is None:
            self._acc, self._accg = torch.zeros_like(self.M), torch.zeros_like(self.Mg)
        merge_into(self._acc[:E], self.M[:E], dMe)
        merge_into(self._acc[self.sh:self.sh + 1], self.M[self.sh:self.sh + 1], dMs)
        merge_into(self._accg, self.Mg, dMg)
        self._nseq += B

    def read(self, xf, e_k, t_k, p_k=None, slot_k=None, side=None, call=None):
        raise NotImplementedError("FastMem.read: the module overrides __call__")

    @torch.no_grad()
    def after_backward(self):
        """Per stream: the micro-batch's end states become its streams' memory (detached; the next micro-batch of a
        stream reads them)."""
        self._fw = None
        if self._pend is None:
            return None
        dMe, dMs, dMg, E = self._pend
        self._pend = None
        ix = torch.tensor(self.streams, device=self.M.device)
        self.M[ix, :E] = dMe.transpose(0, 1).to(self.M.dtype)
        self.M[ix, self.sh] = dMs[0].to(self.M.dtype)
        self.Mg[ix] = dMg.transpose(0, 1).to(self.Mg.dtype)
        return None

    # ------------------------------------------------------------------ the step end
    def own_parameters(self):
        """The parameters of the memory's AdamW (D / D_g: the factored update)."""
        return [p for n, p in self.named_parameters() if n not in ("D", "D_g")]

    def after_step(self, step, refit_every=0):
        self.n_opt += 1
        with ieee_fp32():
            if self.train_only is None:
                if self._opt is None:
                    self._opt = torch.optim.AdamW(self.own_parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0,
                                                  foreach=False)      # (per tensor: no full-size temporaries)
                    if self._opt_pending is not None:
                        try:
                            self._opt.load_state_dict(self._opt_pending)
                        except (ValueError, KeyError) as ex:
                            print(f"fastmem: the memory's AdamW state did not load ({ex}): fresh moments", flush=True)
                        self._opt_pending = None
                ps = [p for p in self.own_parameters() if p.grad is not None]
                if ps:
                    torch.nn.utils.clip_grad_norm_(ps, 1.0)
                    self._opt.step()
                for nm in ("", "_g"):
                    self._d_update(nm)
            with torch.no_grad():
                if self._acc is not None and self._nseq:
                    self.M.add_(self._acc / self._nseq)
                    self.Mg.add_(self._accg / self._nseq)
                    self._acc.zero_()
                    self._accg.zero_()
                self._nseq = 0
        for p in self.parameters():
            p.grad = None

    @torch.no_grad()
    def _d_update(self, nm, b2=0.99, eps=1e-30, clip_d=1.0):
        """D's factored second-moment update per row (wave_v21 PART 3, reimplemented for a parameter's .grad): R / C =
        EMAs of the row / column mean squares, V = R C / mean(R), U = G / sqrt(V), the row's RMS clipped, wd 0."""
        D = getattr(self, "D" + nm)
        if D.grad is None:
            return
        Rb, Cb, nb = getattr(self, "Dr" + nm), getattr(self, "Dc" + nm), getattr(self, "Dn" + nm)
        G = D.grad
        dl, d = G.shape[1], G.shape[2]
        live = G.flatten(1).abs().amax(-1) > 0
        Rn = Rb * b2 + (1 - b2) * torch.linalg.vector_norm(G, dim=-1).square() / d
        Cn = Cb * b2 + (1 - b2) * torch.linalg.vector_norm(G, dim=-2).square() / dl
        Rb.copy_(torch.where(live[:, None], Rn, Rb))
        Cb.copy_(torch.where(live[:, None], Cn, Cb))
        nb.add_(live.long())
        bc = (1 - b2 ** nb.clamp(min=1).double()).float()
        rh, ch = Rb / bc[:, None], Cb / bc[:, None]
        V = rh[:, :, None] * ch[:, None, :] / rh.mean(-1).clamp(min=eps)[:, None, None] + eps
        U = G * V.rsqrt()
        rms = torch.linalg.vector_norm(U, dim=(-2, -1)) / math.sqrt(dl * d)
        D.add_(U / (rms / clip_d).clamp(min=1)[:, None, None] * live[:, None, None].float(), alpha=-self.d_lr)

    # ------------------------------------------------------------------ growth, train-only, logs
    @torch.no_grad()
    def on_grow(self, src, E0, split=True):
        """Children copy the parent: M row (every stream's), the per-row parameters (+ their AdamW moments) and D's
        moments."""
        src = [int(s) for s in (src.tolist() if torch.is_tensor(src) else src)]
        dst = list(range(E0, E0 + len(src)))
        assert not dst or dst[-1] < self.Emax, f"fastmem: {dst[-1] + 1} experts > --fastmem-max-experts {self.Emax}"
        self.n_live = E0 + len(src)
        if not dst:
            return
        s_r = torch.tensor(src, device=self.M.device)
        d_r = torch.tensor(dst, device=self.M.device)
        self.M[self._mrow(d_r)] = self.M[self._mrow(s_r)].clone()
        for n in self.ROW_BUFS:
            t = getattr(self, n)
            t[d_r] = t[s_r].clone()
        for _, p in self._row_tensors():
            p.data[d_r] = p.data[s_r].clone()
            if self._opt is not None and p in self._opt.state:
                for k_ in ("exp_avg", "exp_avg_sq"):
                    if k_ in self._opt.state[p]:
                        self._opt.state[p][k_][d_r] = self._opt.state[p][k_][s_r].clone()

    def set_train_only(self, expert_ids):
        if expert_ids is not None:
            raise NotImplementedError("--memory-train-only with --memory fastmem: not implemented")
        super().set_train_only(None)

    def describe(self, args=None):
        np_ = sum(p.numel() for p in self.parameters())
        return (f"fastmem ({self._cfg['fastmem_size']}): {self.S} expert rows x M [{self.H}, {self.dk}, {self.dv}] + "
                f"global [{self.G}, {self.dk}, {self.dv}] (top-{self.gk}); "
                + (f"per stream ({self.n_streams} streams, carried, reset at the stream start); " if self.stream
                   else "one memory (mean merge); ")
                + f"read pass {self.read_pass if self.read_pass >= 0 else 'every'}; maths "
                f"{'bf16' if self.cd == torch.bfloat16 else 'fp32'}; write grad {'full' if self.write_full else 'detached'}; "
                f"proj {self._cfg['fastmem_proj']}; explore {self.explore}; M {(self.M.numel() + self.Mg.numel()) * 4 / 1e6:.1f}"
                f" MB, {np_ / 1e6:.1f}M params; {self.nbytes() / 1e6:.1f} MB of state")

    def log_record(self, step=None):
        rec = {f"fmem_{k}": float(v) for k, v in self._stats.items()}
        with torch.no_grad():
            Mr = self.M[:, :self.n_live] if self.stream else self.M[:self.n_live]
            rec["fmem_M_rms"] = float(Mr.pow(2).mean().sqrt())
            rec["fmem_Mg_rms"] = float(self.Mg.pow(2).mean().sqrt())
            rec["fmem_decay_mean"] = float((torch.exp(self.s[:self.n_live]) * math.log(2)).mean())
            rec["fmem_D_rms"] = float(self.D[:self.n_live].pow(2).mean().sqrt())
        return rec
