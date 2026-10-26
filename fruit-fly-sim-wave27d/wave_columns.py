"""The WaveSheet v2 column store (SPEC_wavesheet_v2_cloud.md PART 1-3) behind the wave_store interface
(STORES["columns"]): one global sheet of places (the address's entry map: P = side^2 = 1024), each place 3 layers x 16
columns; a column's 64 units are driven by 16 key fan-ins (column c reads address slices (2c + q) % 32, q < 16) and,
for layers 1-2, by binary chain bits from the 3 neighbouring columns' winners of the layer below.

Storage (counted by byte_budget; saved with the memory):
  key_syn     uint8 [P * 3 * 16 * 16 * 16 rows x 32 B]: row(p, l, c, q, k) = ((((p * 3 + l) * 16 + c) * 16 + q) * 16 + k),
              64 units per row as nibbles (the low nibble is the even unit): s = 0 empty, else 1 + 5 (cnt - 1) + ph
              (cnt 1..3, ph 0..4); efficacy (cos, sin)(2 pi ph / 5) if s > 0
  chain_bits  uint8 [P * 2 * 16 * 3 * 64 rows x 8 B]: row(p, l - 1, c, i, v_below) = ((((p * 2 + l - 1) * 16 + c) * 3 + i)
              * 64 + v_below), bit v = target unit v of layer l, column c
  decoder     A, B [3, 16, 64, 24] nn.Parameters (the memory's own AdamW, lr 1e-4), renormed after each memory step
              to |A_v| = |B_v| = rho_l with B_v orthogonal to A_v; rho [3] (init .137 .115 .096) follows the encode
  null stats  med [3], q99 [3] of the column maxima of null cues (EMA .01): kappa per (layer, column)

Read (n_read entries (2; --wave-entries), each with its OWN residual keys K[:, j] = the codes of x_hat - P[E_j] (spec PART 2; Codes.code_e);
  shared_keys = True is the arm reusing the entry-0 keys at entry 1):
  layer 0: Z[c, v] = sum_q efficacy(key_syn[row(E, 0, c, q, K[0, (2c + q) % 32]), v]); M0 = |Z| / 16; v0 = argmax
  layer l: Zk with the level-l keys; nch[v] = sum_i bit v of chain row(E, l - 1, c, i, v_(l-1)[(c + i - 1) % 16]);
           M = (|Zk| + lambda nch) / (16 + 3 lambda); v_l = argmax
  chi[j, l] = mean_c max_v M; chibar_j = mean_l chi; w = argmax_j (entry 0 on ties)
  readout: winner entry; per (l, c) the top-2 units by M; p = softmax(16 M_top2); theta = atan2 of their Z;
           y = sum_l concat_c sum_top2 c_tok kappa_lc p (cos theta A[l, c, v] + sin theta B[l, c, v]) (c_tok: the memory;
           computed as a dense-coefficient GEMM: a deterministic decoder gradient)
Write (W4): the target, normalised, residual-encoded per layer and column into (v*, ph*) through the decoder; at E0 with
  the keys: key nibble = 1 + 5 (max(cnt_old, st) - 1) + ph*; chain bit (E0, l - 1, c, i, v*_(l-1)[(c + i - 1) % 16]) ->
  v*_l[c]. entries[:, 1]: also at E1 with the keys K[:, 1] (the T4 arm).
Train-only (owned, a (P,) bool or None): writes / confirms only at owned places, the sweep only over owned places.
Confirm (W3): at the winner entry, key synapses of the recalled path present with the recalled phase -> cnt + 1 (s += 5
  if s <= 10); the chain bits of the path set. --wave-recon 1: absent key synapses bind at cnt 1.
Forget (after_step): sweep 16 places per step from sweep_ptr: key block (p, l): q = clamp((fill - f*) / max(n1, 1e-6),
  0, 1), each present synapse decremented with prob q; chain block: if fill > f*ch, each set bit cleared with prob
  (fill - f*ch) / fill. f* = --wave-fill (0.12), f*ch = 1 - (1 - f*)^(16 / 64) = 0.0315.
Determinism: one put per unique row (gather, modify, pack, put), first index on ties, seeded generators.
"""
import math

import torch
import torch.nn as nn
from torch.profiler import record_function as _rf

import wave_fastgather as _fg

DECODE_CHUNK = 4096


TOP2_MAX = True        # (True: the recall's top-2 by two max reductions (faster; ties low-index-first, not bit-equal to topk))
DECODE_GATHER = True     # (forward: gather the top-2 unit rows, one bmm per (token, column); False = the dense einsum)


def _uniq(rows):
    """torch.unique(rows, return_inverse=True) with static shapes (no host sync): a stable sort, the run starts, a
    cumsum. -> urow (m,) the unique rows ascending, the slots past the unique count repeat the last unique row; inv
    (m,) each entry's slot; fill (m,) the slot whose bytes a slot puts (itself, or the last unique slot: a repeated row
    puts the very same bytes, so the duplicate put is deterministic)."""
    m = rows.numel()
    srt, perm = rows.sort(stable=True)
    new = torch.ones(m, dtype=torch.bool, device=rows.device)
    new[1:] = srt[1:] != srt[:-1]
    gid = new.long().cumsum(0) - 1
    inv = torch.empty_like(gid).scatter_(0, perm, gid)
    ur = torch.zeros_like(srt).scatter_(0, gid, srt)
    fill = torch.minimum(torch.arange(m, device=rows.device), gid[-1])
    return ur[fill], inv, fill


def _top2(M):
    """== M.topk(2, -1) (values, indices) by two max reductions (CUDA's sbtopk was the recall's slowest kernel: ~0.6
    ms per call at (2048, 64, 64)); ties: the lowest index first (topk's tie order is unspecified)."""
    if not TOP2_MAX:
        return M.topk(2, -1)
    v0, i0 = M.max(-1)
    v1, i1 = M.scatter(-1, i0[..., None], float("-inf")).max(-1)
    return torch.return_types.topk((torch.stack([v0, v1], -1), torch.stack([i0, i1], -1)))


class _ChunkedDecode(torch.autograd.Function):
    """y = einsum(nlcu,lcud->ncd, dA, A) + the same for B, with dA/dB the dense (detached) top-2 unit coefficients,
    built and used DECODE_CHUNK rows at a time: the dense (n, L, C, UC) coefficient tensors are never held whole
    (at 64 columns they were ~1.1 GB each). The decoder gradient sums the chunks in a fixed order (dense: no
    accumulating index backward). Forward (DECODE_GATHER): the 2 L unit rows of A and of B each (token, column) uses,
    gathered, and one bmm per (token, column) over those 4 L coefficients: the same sum without the one-hot zeros
    (equal up to float rounding: the summation order differs)."""

    @staticmethod
    def _dense(top, c, s, e):
        return torch.zeros(e - s, *top.shape[1:-1], UC, device=top.device, dtype=c.dtype).scatter_(-1, top[s:e], c[s:e])

    @staticmethod
    def _gather(A, B, top, cA, cB):
        """top / cA / cB (n, L, C, 2) -> (n, C, dc): bmm of (n C, 1, 4 L) coefficients x (n C, 4 L, dc) rows."""
        n, Lq, Cq = top.shape[:3]
        t = top.permute(0, 2, 1, 3)                                                  # (n, C, L, 2)
        cc = torch.arange(Cq, device=top.device)[None, :, None, None]
        ll = torch.arange(Lq, device=top.device)[None, None, :, None]
        G = torch.cat([A[ll, cc, t], B[ll, cc, t]], 2)                              # (n, C, 2 L, 2, dc)
        co = torch.cat([cA.permute(0, 2, 1, 3), cB.permute(0, 2, 1, 3)], 2)        # (n, C, 2 L, 2)
        return torch.bmm(co.reshape(n * Cq, 1, -1), G.reshape(n * Cq, -1, A.size(-1))).view(n, Cq, -1)

    @staticmethod
    def forward(ctx, A, B, top, cA, cB):
        n = top.size(0)
        y = A.new_empty(n, A.size(1), A.size(-1))
        for s in range(0, n, DECODE_CHUNK):
            e = min(n, s + DECODE_CHUNK)
            if DECODE_GATHER:
                y[s:e] = _ChunkedDecode._gather(A, B, top[s:e], cA[s:e], cB[s:e])
                continue
            dA, dB = _ChunkedDecode._dense(top, cA, s, e), _ChunkedDecode._dense(top, cB, s, e)
            y[s:e] = torch.einsum("nlcu,lcud->ncd", dA, A) + torch.einsum("nlcu,lcud->ncd", dB, B)
        ctx.save_for_backward(top, cA, cB)
        return y

    @staticmethod
    def backward(ctx, gy):
        top, cA, cB = ctx.saved_tensors
        n, (L, C) = top.size(0), top.shape[1:3]
        gA = gy.new_zeros(L, C, UC, gy.size(-1)) if ctx.needs_input_grad[0] else None
        gB = gy.new_zeros(L, C, UC, gy.size(-1)) if ctx.needs_input_grad[1] else None
        for s in range(0, n, DECODE_CHUNK):
            e = min(n, s + DECODE_CHUNK)
            if gA is not None:
                gA += torch.einsum("nlcu,ncd->lcud", _ChunkedDecode._dense(top, cA, s, e), gy[s:e])
            if gB is not None:
                gB += torch.einsum("nlcu,ncd->lcud", _ChunkedDecode._dense(top, cB, s, e), gy[s:e])
        return gA, gB, None, None, None
from wave_store import STORES, WaveStore

L, C, F, UK, UC, PHI, W = 3, 16, 16, 16, 64, 5, 3


class WaveColumns(WaveStore):
    placeholder = False

    def __init__(self, d, n_parts, address, fill=0.12, lam=4.0, seed=1234, read_chunk=512, sweep=16, use_loc=True,
                 dtype=None, decay=0.0, places=None, pad=0, cols=C, **_):
        super().__init__()
        self.C = int(cols)                 # (columns per place: 16 = v2 / v2.2; --wave-cols 64 = v2.2's 2048-d latent)
        assert n_parts == 1, "WaveColumns: the global sheet (one part)"
        assert d % self.C == 0, f"WaveColumns: d {d} must be a multiple of the columns {self.C}"
        assert address.S == 32 and address.M == UK and address.levels == L, \
            "WaveColumns needs the address with 3 levels x 32 slices x 16 codes"
        self.d, self.P, self.dc = d, int(places or address.L), d // self.C       # (places: the v2.1 strips' total)
        self.fill_target, self.lam_ch, self.seed = float(fill), float(lam), int(seed)
        self.fill_ch = 1 - (1 - self.fill_target) ** (UK / UC)
        self.read_chunk, self.n_sweep = int(read_chunk), int(sweep)
        self.register_buffer("key_syn", torch.zeros(self.P * L * self.C * F * UK, UC // 2, dtype=torch.uint8))
        self.register_buffer("chain_bits", torch.zeros(self.P * (L - 1) * self.C * W * UC, UC // 8, dtype=torch.uint8))
        if pad:                                    # (the v2.1 layout's 64-byte scratch pads: zero, counted, saved)
            self.register_buffer("key_pad", torch.zeros(pad, dtype=torch.uint8))
            self.register_buffer("chain_pad", torch.zeros(pad, dtype=torch.uint8))
        g = torch.Generator().manual_seed(self.seed + 41)
        A = torch.randn(L, self.C, UC, self.dc, generator=g)
        B = torch.randn(L, self.C, UC, self.dc, generator=g)
        self.register_buffer("rho", torch.tensor([0.137, 0.115, 0.096]))
        self.A, self.B = nn.Parameter(A), nn.Parameter(B)
        self.renorm()
        self.register_buffer("med", torch.zeros(L))
        self.register_buffer("q99", torch.ones(L))
        self.register_buffer("null_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer("scale", torch.zeros(()))     # (the write targets' norm: the read-out's scale)
        self.register_buffer("fill_est", torch.zeros(()))
        self.register_buffer("fill_ch_est", torch.zeros(()))
        self.register_buffer("load", torch.zeros(self.P, dtype=torch.int32))  # (memories per place: + 1 per write,
        # x the sweep's kept share; integer: a deterministic index_add)
        # derived (never saved): the nibble-byte look-up table (presence x cos / sin of both nibbles), the fan-ins
        byte = torch.arange(256)
        s = (byte[:, None] >> torch.tensor([0, 4])) & 15                                # (256, 2)
        ang = torch.remainder(s - 1, PHI).float() * (2 * math.pi / PHI)
        self.register_buffer("lut", torch.stack([ang.cos() * (s > 0), ang.sin() * (s > 0)], -1).reshape(256, 4),
                             persistent=False)
        # fan-in: column c reads address slices (2c + q + c // 16) % 32 (16 columns: (2c + q) % 32 as before; more
        # columns: each further block of 16 shifted by one slice, so no two columns read the same slice pattern)
        cc_ = torch.arange(self.C)[:, None]
        self.register_buffer("fan", (2 * cc_ + torch.arange(F)[None] + cc_ // 16) % 32, persistent=False)
        # LUT16 (spec PART 2): an int16 word = 4 nibbles = units 4w..4w+3 -> their (cos, sin) x presence (65536, 8)
        w16 = torch.arange(65536)
        s4 = (w16[:, None] >> (4 * torch.arange(4))) & 15                              # (65536, 4)
        a4 = torch.remainder(s4 - 1, PHI).float() * (2 * math.pi / PHI)
        self.register_buffer("lut16", torch.stack([a4.cos() * (s4 > 0), a4.sin() * (s4 > 0)], -1).reshape(65536, 8),
                             persistent=False)
        self.register_buffer("nib", _fg.nib_lut(self.lut16), persistent=False)   # (wave_fastgather's unpack table)
        self.register_buffer("bits", 1 << torch.arange(8, dtype=torch.uint8), persistent=False)
        self.recon = False
        self.shared_keys = False           # (arm: entry 1 reads with the entry-0 keys)
        self.n_read = 2                    # (entries read: the address's first n_read places; --wave-entries)
        self.owned = None                  # (train-only: the (P,) bool mask of the places that may change)
        self.sweep_ptr = 0
        self.n_step = 0
        self.stats_acc = dict(writes=0, confirms=0, swept=0)

    # ------------------------------------------------------------------ host state
    def host_state(self):
        return dict(sweep_ptr=self.sweep_ptr, n_step=self.n_step)

    def load_host_state(self, st):
        self.sweep_ptr, self.n_step = int(st.get("sweep_ptr", 0)), int(st.get("n_step", 0))

    @torch.no_grad()
    def renorm(self):
        """|A_v| = |B_v| = rho_l, B_v orthogonal to A_v (spec: after each memory step)."""
        A, B = self.A.data, self.B.data
        B -= (B * A).sum(-1, keepdim=True) / (A * A).sum(-1, keepdim=True).clamp(min=1e-12) * A
        r = self.rho.view(L, 1, 1, 1)
        A *= r / A.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        B *= r / B.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # ------------------------------------------------------------------ indices
    def _key_rows(self, E, keys, l):
        """E (n,) places, keys (n, 32) level-l codes -> (n, 16 c, 16 q) key rows."""
        k = keys[:, self.fan]                                                           # (n, C, F)
        c = torch.arange(self.C, device=E.device)[None, :, None]
        q = torch.arange(F, device=E.device)[None, None, :]
        return (((E[:, None, None] * L + l) * self.C + c) * F + q) * UK + k

    def _chain_rows(self, E, v_below, l):
        """the chain rows of layer l (1, 2) for winners v_below (n, 16) -> (n, 16 c, 3 i)."""
        c = torch.arange(self.C, device=E.device)
        i = torch.arange(W, device=E.device)
        src = v_below[:, (c[:, None] + i[None] - 1) % self.C]                                 # (n, C, W)
        return ((((E[:, None, None] * (L - 1) + (l - 1)) * self.C + c[None, :, None]) * W + i[None, None]) * UC + src)

    def _n_ent(self, codes):
        return min(self.n_read, codes.loc.size(1))

    def _all_keys(self, codes):
        """-> (n, E, 96) top-1 keys per entry (E = n_read entries): each entry's own residual codes (Codes.code_e),
        else (no code_e, or the shared_keys arm) the entry-0 codes at every entry."""
        k0, E = codes.code[:, :, 0], self._n_ent(codes)
        if codes.code_e is not None and not self.shared_keys and codes.code_e.size(1) >= E:
            return codes.code_e[:, :E]
        return torch.stack([k0] * E, 1)

    def _keys(self, codes, l, j=0):
        return self._all_keys(codes)[:, j, l * 32:(l + 1) * 32]

    def _Z(self, rows):
        """key rows (..., F) -> (re, im) (..., UC) summed over the fan-in: whole 32-byte rows gathered as int32 x 8
        (index_select on dim 0: coalesced), viewed as 16 int16 words, one embedding_bag over the fan-in with LUT16."""
        return _fg.key_Z(self.key_syn, rows, self.nib)                 # (wave_fastgather: bit-exact, no syncs)

    def _nch(self, rows):
        """chain rows (..., W) -> (..., UC) the number of set bits per target unit."""
        return _fg.chain_nch(self.chain_bits, rows)

    # ------------------------------------------------------------------ the recall
    @torch.no_grad()
    def _recall(self, codes, entries=None):
        out = []
        for a in range(0, len(codes), self.read_chunk):
            out.append(self._recall_chunk(codes.take(slice(a, a + self.read_chunk))))
        tr = {k: torch.cat([o[k] for o in out]) for k in out[0]}
        score = tr["chi"].mean(-1)                                                      # (n, E)
        if entries is not None:
            score = torch.where(entries, score, torch.full_like(score, -1.0))
        tr["best"] = score.argmax(-1)
        tr["chibar"] = score.gather(1, tr["best"][:, None]).squeeze(1)
        tr["loc"] = codes.loc[:, :score.size(1)]
        return tr

    def _recall_chunk(self, codes):
        n, E = len(codes), self._n_ent(codes)
        res = dict(v=[], top=[], M2=[], th=[], ph=[], chi=[], colmax=[])
        for j in range(E):
            Ej = codes.loc[:, j]
            vs, tops, M2s, ths, phs, chis, cms = [], [], [], [], [], [], []
            prev = None
            for l in range(L):
                with _rf("wave.cols.key_gather_lut"):
                    zr, zi = _fg.key_Z_keys(self.key_syn, Ej, self._keys(codes, l, j), l, self.fan, self.nib)
                mag = (zr * zr + zi * zi).sqrt()
                if l == 0:
                    M = mag / F
                else:
                    with _rf("wave.cols.chain"):
                        M = (mag + self.lam_ch * self._nch(self._chain_rows(Ej, prev, l))) / (F + self.lam_ch * W)
                t2 = _top2(M)                                                          # (n, C, 2)
                v = t2.indices[..., 0]
                th = torch.atan2(zi.gather(-1, t2.indices), zr.gather(-1, t2.indices))  # (n, C, 2)
                vs.append(v)
                tops.append(t2.indices)
                M2s.append(t2.values)
                ths.append(th)
                phs.append(torch.remainder(torch.round(th[..., 0] * PHI / (2 * math.pi)), PHI).long())
                cms.append(t2.values[..., 0])
                chis.append(t2.values[..., 0].mean(-1))
                prev = v
            res["v"].append(torch.stack(vs, 1))                                        # (n, L, C)
            res["top"].append(torch.stack(tops, 1))                                    # (n, L, C, 2)
            res["M2"].append(torch.stack(M2s, 1))
            res["th"].append(torch.stack(ths, 1))
            res["ph"].append(torch.stack(phs, 1))
            res["colmax"].append(torch.stack(cms, 1))                                  # (n, L, C)
            res["chi"].append(torch.stack(chis, 1))                                    # (n, L)
        return {k: torch.stack(v, 1) for k, v in res.items()}                         # (n, E, ...)

    def read_traced(self, codes, part=None, entries=None):
        tr = self._recall(codes, entries)
        tr["mass"] = tr["chibar"]
        tr["loc"] = codes.loc[:, :self._n_ent(codes)]
        tr["keys"] = self._all_keys(codes)                                             # (n, E, 96)
        return self.decode(tr), tr

    def read(self, codes, part=None, entries=None):
        return self.read_traced(codes, part, entries)[0]

    @torch.no_grad()
    def trace(self, codes, entries=None):
        """read_traced without the decode (the pre-read gate decodes with grad later)."""
        tr = self._recall(codes, entries)
        tr["mass"] = tr["chibar"]
        tr["loc"] = codes.loc[:, :self._n_ent(codes)]
        tr["keys"] = self._all_keys(codes)
        return tr

    @torch.no_grad()
    def kappa_mean(self, tr):
        """(n,) the mean over (layer, column) of kappa at the winner entry (1 before null statistics)."""
        b = tr["best"]
        cm = tr["colmax"][torch.arange(b.numel(), device=b.device), b]
        kap = ((cm - self.med[None, :, None]) / (self.q99 - self.med).clamp(min=0.01)[None, :, None]).clamp(0, 1)
        return torch.where(self.null_seen > 0, kap.mean((1, 2)), torch.ones_like(kap[:, 0, 0]))

    def mass(self, codes, part=None):
        return self._recall(codes)["chibar"]

    def decode(self, tr):
        """The read-out WITH gradient to the decoder (A, B); kappa per (layer, column) vs the null statistics; the
        memory scales it by c_tok. Unit-scale y times the write targets' scale (the memory's content scale)."""
        b = tr["best"]
        ar = torch.arange(b.numel(), device=b.device)
        top, M2, th, cm = tr["top"][ar, b], tr["M2"][ar, b], tr["th"][ar, b], tr["colmax"][ar, b]
        kap = ((cm - self.med[None, :, None]) / (self.q99 - self.med).clamp(min=0.01)[None, :, None]).clamp(0, 1)
        kap = kap * (self.null_seen > 0).float() + (self.null_seen == 0).float()       # (no null stats yet: 1)
        p = torch.softmax(16 * M2, -1)                                                 # (n, L, C, 2)
        cA = (kap[..., None] * p * th.cos()).detach()
        cB = (kap[..., None] * p * th.sin()).detach()
        # dense (detached) unit coefficients, then one GEMM per decoder: the decoder's gradient is a GEMM too (a fixed
        # summation order: no accumulating index backward; the top-2 units of a column are distinct)
        y = _ChunkedDecode.apply(self.A, self.B, top, cA, cB).flatten(-2)             # (n, D)
        return y * self.scale.clamp(min=1e-6) * (M2[..., 0].amax((-1, -2)) > 0).float()[:, None]

    @torch.no_grad()
    def null_observe(self, codes):
        """The null control's cues: their chibar (returned) and the EMA of the column maxima's median / q99."""
        tr = self._recall(codes)
        b = tr["best"]
        ar = torch.arange(b.numel(), device=b.device)
        cm = tr["colmax"][ar, b]                                                       # (n, L, C)
        flat = cm.transpose(0, 1).reshape(L, -1)
        k50, k99 = max(1, flat.size(1) // 2), max(1, int(0.99 * flat.size(1)))
        md, qq = flat.kthvalue(k50, -1).values, flat.kthvalue(k99, -1).values
        first = self.null_seen == 0
        self.med.copy_(torch.where(first, md, self.med.lerp(md, 0.01)))
        self.q99.copy_(torch.where(first, qq, self.q99.lerp(qq, 0.01)))
        self.null_seen += 1
        self.last_null_place = tr["loc"].gather(1, b[:, None]).squeeze(1) if "loc" in tr else codes.loc[:, 0]
        return tr["chibar"]

    # ------------------------------------------------------------------ encode (W4) and the synapse updates
    @torch.no_grad()
    def encode(self, Y, valid=None):
        """Y (n, D) -> v* (n, L, C), ph* (n, L, C); rho follows the mean projection (spec W4). valid (n,) bool or None:
        rho's mean over these rows only (none valid: rho unchanged; static shapes, no host sync)."""
        n = Y.size(0)
        res = (Y / Y.norm(dim=-1, keepdim=True).clamp(min=1e-12)).view(n, self.C, self.dc).clone()
        vs, phs = [], []
        A, B = self.A.detach(), self.B.detach()
        cc = torch.arange(self.C, device=Y.device)[None]
        for l in range(L):
            a = torch.einsum("ncd,cvd->ncv", res, A[l]) / self.rho[l]
            b = torch.einsum("ncd,cvd->ncv", res, B[l]) / self.rho[l]
            ph = torch.remainder(torch.round(torch.atan2(b, a) * PHI / (2 * math.pi)), PHI)
            ang = ph * (2 * math.pi / PHI)
            sc = a * ang.cos() + b * ang.sin()
            v = sc.argmax(-1)                                                          # (n, C)
            phv = ph.gather(-1, v[..., None]).squeeze(-1).long()
            coef = sc.gather(-1, v[..., None]).squeeze(-1)
            angv = phv.float() * (2 * math.pi / PHI)
            res -= angv.cos()[..., None] * A[l][cc, v] + angv.sin()[..., None] * B[l][cc, v]
            if self.owned is None and valid is None:   # (train-only: the decoder, rho included, is frozen)
                self.rho[l] = 0.99 * self.rho[l] + 0.01 * coef.mean()
            elif self.owned is None:
                nv = valid.sum() * coef.size(1)
                mc = (coef * valid[:, None].float()).sum() / nv.clamp(min=1)
                self.rho[l] = torch.where(nv > 0, 0.99 * self.rho[l] + 0.01 * mc, self.rho[l])
            vs.append(v)
            phs.append(phv)
        return torch.stack(vs, 1), torch.stack(phs, 1)

    def _acc(self, k, v):
        """a stats counter += a device tensor (no host sync; fill() / the log read it)."""
        self.stats_acc[k] = self.stats_acc.get(k, 0) + v

    def _key_update(self, rows, v, ph, st, mode, mask=None):
        """rows (m,) key rows, v (m,) unit, ph (m,) phase, st (m,) strength; one put per unique row.
        mode 'set': s = 1 + 5 (max(cnt_old, st) - 1) + ph; 'inc': present with phase ph -> cnt + 1 (cap 3);
        'bind' (recon): also absent -> cnt 1. mask (m,) bool or None: only these entries change (static shapes: the
        masked entries' rows are gathered and put back unchanged)."""
        if rows.numel() == 0:
            return
        urow, inv, fill = _uniq(rows)                                                  # (no torch.unique: no sync)
        byt = self.key_syn[urow].long()                                                # (m, 32)
        s = torch.stack([byt & 15, byt >> 4], -1).flatten(-2)                          # (m, 64)
        old = s[inv, v]
        cnt = torch.where(old > 0, (old - 1) // 5 + 1, torch.zeros_like(old))
        oph = torch.where(old > 0, (old - 1) % 5, torch.zeros_like(old))
        if mode == "set":
            new = 1 + 5 * (torch.maximum(cnt, st.long().clamp(1, 3)) - 1) + ph
        elif mode == "add":                            # (v2.2 delta write: cnt + 1 (at least the strength), phase set)
            new = 1 + 5 * (torch.maximum(cnt + 1, st.long().clamp(1, 3)).clamp(max=3) - 1) + ph
        elif mode == "dec":                            # (v2.2 delta erase: present -> cnt - 1, cnt 1 -> empty)
            new = torch.where(old > 5, old - 5, torch.zeros_like(old))
        else:
            hit = (old > 0) & (oph == ph)
            new = torch.where(hit, 1 + 5 * ((cnt + 1).clamp(max=3) - 1) + oph, old)
            if mode == "bind":
                new = torch.where(old == 0, 1 + ph, new)
        flat = inv * UC + v
        last = torch.full((s.numel(),), -1, dtype=torch.long, device=s.device)        # (duplicates: the last wins)
        pos = torch.arange(flat.numel(), device=s.device)
        if mask is not None:                           # (a masked entry never wins: -1 = the scatter's start)
            pos = torch.where(mask, pos, torch.full_like(pos, -1))
        last.scatter_reduce_(0, flat, pos, "amax")
        s = torch.where(last >= 0, new[last.clamp(min=0)], s.view(-1)).view(-1, UC // 2, 2)[fill]
        self.key_syn[urow] = (s[..., 0] | (s[..., 1] << 4)).to(torch.uint8)

    def _chain_set(self, rows, v, mask=None):
        """rows (m,) chain rows, v (m,) the target unit: set bit v (one put per unique row); mask as _key_update."""
        if rows.numel() == 0:
            return
        urow, inv, fill = _uniq(rows)
        on = ((self.chain_bits[urow][..., None] & self.bits) > 0).flatten(-2)          # (m, 64)
        if mask is None:
            on.view(-1)[inv * UC + v] = True
        else:
            on = on.view(-1).int().index_put_((inv * UC + v,), mask.int(), accumulate=True).view_as(on) > 0
        on = on[fill]
        self.chain_bits[urow] = (on.view(-1, UC // 8, 8).to(torch.uint8) * self.bits).sum(-1).to(torch.uint8)

    def _path(self, E, keys, v, ph):
        """The key synapses (rows, v, ph) and chain bits (rows, v) of a path: E (n,), keys (n, 96), v / ph (n, L, C)."""
        kr, kv, kp, cr, cv = [], [], [], [], []
        for l in range(L):
            rows = self._key_rows(E, keys[:, l * 32:(l + 1) * 32], l)                  # (n, C, F)
            kr.append(rows.flatten(1))
            kv.append(v[:, l, :, None].expand(-1, self.C, F).flatten(1))
            kp.append(ph[:, l, :, None].expand(-1, self.C, F).flatten(1))
            if l:
                crow = self._chain_rows(E, v[:, l - 1], l)                              # (n, C, W)
                cr.append(crow.flatten(1))
                cv.append(v[:, l, :, None].expand(-1, self.C, W).flatten(1))
        cat = lambda x: torch.cat(x, 1)
        return cat(kr), cat(kv), cat(kp), cat(cr), cat(cv)

    @torch.no_grad()
    def write(self, codes, content, strength, part=None, entries=None, trace=None):
        n = len(codes)
        if n == 0:
            return
        keep = strength > 0
        if self.owned is None:                         # (train-only: the read-out scale is frozen)
            self.scale.copy_(torch.where(self.scale > 0, self.scale.lerp(content.norm(dim=-1).mean(), 0.01),
                                         content.norm(dim=-1).mean()))
        v, ph = self.encode(content.float())
        keys = self._all_keys(codes)
        for e in range(self._n_ent(codes)):
            sel = keep if e == 0 else (keep & (entries[:, e] if entries is not None else torch.zeros_like(keep)))
            if entries is not None and e == 0:
                sel = sel & entries[:, 0]
            if self.owned is not None:
                sel = sel & self.owned[codes.loc[:, e]]
            rows = sel.nonzero().squeeze(-1)
            if rows.numel() == 0:
                continue
            kr, kv, kp, cr, cv = self._path(codes.loc[rows, e], keys[rows, e], v[rows], ph[rows])
            self.load.index_add_(0, codes.loc[rows, e], torch.ones(rows.numel(), dtype=torch.int32, device=rows.device))
            st = strength[rows][:, None].expand_as(kr)
            self._key_update(kr.reshape(-1), kv.reshape(-1), kp.reshape(-1), st.reshape(-1), "set")
            self._chain_set(cr.reshape(-1), cv.reshape(-1))
        self.stats_acc["writes"] += int(keep.sum())

    @torch.no_grad()
    def write_delta(self, codes, content, strength, rv, rph, erase, valid=None):
        """(v2.2, --wave-write delta) entry 0 only: write the error between the target and what the place already
        recalls for these keys. rv / rph (n, L, C): the recalled units / phases at entry 0 (the read's trace); erase
        (n,) bool: the key match is confident (z >= z_conf). Per (layer, column): the target's synapses cnt + 1 with
        the target phase (already matching: this is the confirm + 1); where the recalled unit is not the target's and
        erase: the recalled unit's key synapses cnt - 1 (the erase runs first; chain bits: the target's set, none
        cleared). Storage format unchanged. valid (n,) bool or None: the rows that are real targets (static shapes:
        the caller passes every candidate; the scale and rho means run over the valid rows, None = all). No host sync:
        every row goes through, the rows not written are masked."""
        n = len(codes)
        if n == 0:
            return
        keep = strength > 0
        if valid is not None:
            keep = keep & valid
        if self.owned is not None:
            keep = keep & self.owned[codes.loc[:, 0]]
        if self.owned is None:
            cn = content.norm(dim=-1)
            if valid is None:
                mn, has = cn.mean(), None
            else:
                nv = valid.sum()
                mn, has = (cn * valid.float()).sum() / nv.clamp(min=1), nv > 0
            new = torch.where(self.scale > 0, self.scale.lerp(mn, 0.01), mn)
            self.scale.copy_(new if has is None else torch.where(has, new, self.scale))
        v, ph = self.encode(content.float(), valid)
        E0, keys = codes.loc[:, 0], self._all_keys(codes)[:, 0]
        er = (erase & keep)[:, None, None] & (rv != v)                                 # (n, L, C)
        kr, kv, km = [], [], []
        for l in range(L):
            kr.append(self._key_rows(E0, keys[:, l * 32:(l + 1) * 32], l).reshape(-1))  # (n C F)
            kv.append(rv[:, l, :, None].expand(-1, self.C, F).reshape(-1))
            km.append(er[:, l, :, None].expand(-1, self.C, F).reshape(-1))
        kv = torch.cat(kv)
        self._key_update(torch.cat(kr), kv, torch.zeros_like(kv), torch.ones_like(kv, dtype=torch.float), "dec",
                         mask=torch.cat(km))
        kr, kv, kp, cr, cv = self._path(E0, keys, v, ph)
        self.load.index_add_(0, E0, keep.int())
        st = strength[:, None].expand_as(kr)
        self._key_update(kr.reshape(-1), kv.reshape(-1), kp.reshape(-1), st.reshape(-1), "add",
                         mask=keep[:, None].expand_as(kr).reshape(-1))
        self._chain_set(cr.reshape(-1), cv.reshape(-1), mask=keep[:, None].expand_as(cr).reshape(-1))
        self._acc("writes", keep.sum())
        self._acc("erased", er.sum())

    @torch.no_grad()
    def confirm(self, codes, ok, part=None, trace=None):
        if trace is None:
            return
        okm = ok > 0                                   # (static: every row goes through, the rest masked; no sync)
        if self.owned is not None:
            okm = okm & self.owned[trace["loc"].gather(1, trace["best"][:, None]).squeeze(1)]
        if okm.numel() == 0:
            return
        b = trace["best"]
        ar = torch.arange(b.numel(), device=b.device)
        E = trace["loc"][ar, b]
        kr, kv, kp, cr, cv = self._path(E, trace["keys"][ar, b], trace["v"][ar, b], trace["ph"][ar, b])
        self._key_update(kr.reshape(-1), kv.reshape(-1), kp.reshape(-1), torch.ones(kr.numel(), device=kr.device),
                         "bind" if self.recon else "inc", mask=okm[:, None].expand_as(kr).reshape(-1))
        self._chain_set(cr.reshape(-1), cv.reshape(-1), mask=okm[:, None].expand_as(cr).reshape(-1))
        self._acc("confirms", okm.sum())

    @torch.no_grad()
    def forget(self, step, parts=None):
        with _rf("wave.sweep"):
            return self._forget(step, parts)

    def _forget(self, step, parts=None):
        """The sweep: --sweep places per step (round robin); key blocks (p, l) to f*, chain blocks to f*ch."""
        self.n_step += 1
        blk = 0
        if self.owned is not None:                     # (train-only: the owned places only)
            own = self.owned.nonzero().squeeze(-1).tolist()
            if not own:
                return
            places = [own[(self.sweep_ptr + i) % len(own)] for i in range(min(self.n_sweep, len(own)))]
            self.sweep_ptr = (self.sweep_ptr + len(places)) % len(own)
        else:
            places = [(self.sweep_ptr + i) % self.P for i in range(self.n_sweep)]
            self.sweep_ptr = (self.sweep_ptr + self.n_sweep) % self.P
        kb = self.C * F * UK
        fills, fch = [], []
        for p in places:
            for l in range(L):
                sl = slice((p * L + l) * kb, (p * L + l + 1) * kb)
                byt = self.key_syn[sl].long()
                s = torch.stack([byt & 15, byt >> 4], -1)                              # (kb, 32, 2)
                pres = s > 0
                fill = pres.float().mean()
                n1 = ((s >= 1) & (s <= 5)).float().mean()
                q = ((fill - self.fill_target) / n1.clamp(min=1e-6)).clamp(0, 1)
                u = _fg.sweep_randoms(s.numel(), self.n_step * 1000 + blk, s.device, self.seed).view(s.shape)
                blk += 1
                dec = pres & (u.long() < torch.round(256 * q).long())
                s = torch.where(dec, torch.where(s > 5, s - 5, torch.zeros_like(s)), s)
                self.key_syn[sl] = (s[..., 0] | (s[..., 1] << 4)).to(torch.uint8)
                fills.append(fill)
                if l == 0:                             # (the load follows the occupancy the sweep leaves)
                    keep_ = ((s > 0).float().mean() / fill.clamp(min=1e-9)).clamp(max=1.0)
                    self.load[p] = torch.round(self.load[p].float() * keep_).to(torch.int32)
            cb = self.C * W * UC
            for l in range(L - 1):
                sl = slice((p * (L - 1) + l) * cb, (p * (L - 1) + l + 1) * cb)
                on = ((self.chain_bits[sl][..., None] & self.bits) > 0)
                fill = on.float().mean()
                pr = ((fill - self.fill_ch) / fill.clamp(min=1e-6)).clamp(0, 1)
                u = _fg.sweep_randoms(on.numel(), self.n_step * 1000 + 500 + blk, on.device, self.seed).view(on.shape)
                blk += 1
                on = on & ~(u.long() < torch.round(256 * pr).long())
                self.chain_bits[sl] = (on.to(torch.uint8) * self.bits).sum(-1).to(torch.uint8)
                fch.append(fill)
        self.fill_est.copy_(torch.stack(fills).mean())
        self.fill_ch_est.copy_(torch.stack(fch).mean())
        self.stats_acc["swept"] += len(places)

    @torch.no_grad()
    def sweep_places(self, places, group=4):
        """== `for p in places: sweep_ptr = p; n_sweep = 1; forget(0)` (owned None), bit for bit, batched `group`
        places at a time: the same per-block randoms (one generator draw per block, the same seeds: n_step + 1 + i per
        place i, blocks 0-2 key, 3-4 chain), the per-block fill / q as before (exact: 0/1 sums < 2^24), one gather and
        one put per batch instead of 5 per place; fill_est / fill_ch_est end as the last place's (the old per-call
        overwrite)."""
        if len(places) == 0:
            return
        kb, cb = self.C * F * UK, self.C * W * UC
        dev = self.key_syn.device
        if not torch.is_tensor(places):
            places = torch.tensor(places, device=dev)
        ks = self.key_syn.view(self.P, L, kb, UC // 2)
        cs = self.chain_bits.view(self.P, L - 1, cb, UC // 8)
        n0 = self.n_step
        for a in range(0, len(places), group):
            pt = places[a:a + group]
            g = pt.numel()
            byt = ks[pt]                                                               # (g, L, kb, 32) uint8
            s = torch.stack([byt & 15, byt >> 4], -1)                                  # (g, L, kb, 32, 2)
            pres = s > 0
            fill = pres.float().mean((2, 3, 4))                                        # (g, L)
            n1 = ((s >= 1) & (s <= 5)).float().mean((2, 3, 4))
            q = ((fill - self.fill_target) / n1.clamp(min=1e-6)).clamp(0, 1)
            u = torch.stack([torch.stack([_fg.sweep_randoms(kb * 64, (n0 + a + i + 1) * 1000 + l, dev, self.seed)
                                          for l in range(L)]) for i in range(g)]).view(s.shape)
            dec = pres & (u.long() < torch.round(256 * q).long()[:, :, None, None, None])
            s = torch.where(dec, torch.where(s > 5, s - 5, torch.zeros_like(s)), s)
            ks[pt] = (s[..., 0] | (s[..., 1] << 4)).to(torch.uint8)
            keep_ = ((s[:, 0] > 0).float().mean((1, 2, 3)) / fill[:, 0].clamp(min=1e-9)).clamp(max=1.0)
            self.load[pt] = torch.round(self.load[pt].float() * keep_).to(torch.int32)
            on = (cs[pt][..., None] & self.bits) > 0                                   # (g, L - 1, cb, 8, 8)
            fc = on.float().mean((2, 3, 4))                                            # (g, L - 1)
            pr = ((fc - self.fill_ch) / fc.clamp(min=1e-6)).clamp(0, 1)
            u = torch.stack([torch.stack([_fg.sweep_randoms(cb * 64, (n0 + a + i + 1) * 1000 + 500 + L + l, dev,
                                                            self.seed) for l in range(L - 1)])
                             for i in range(g)]).view(on.shape)
            on = on & ~(u.long() < torch.round(256 * pr).long()[:, :, None, None, None])
            cs[pt] = (on.to(torch.uint8) * self.bits).sum(-1).to(torch.uint8)
        self.n_step = n0 + len(places)
        self.fill_est.copy_(fill[-1].mean())
        self.fill_ch_est.copy_(fc[-1].mean())
        self.stats_acc["swept"] += len(places)

    def after_param_step(self):
        """(the memory calls this after its AdamW step) the decoder's norm constraint."""
        self.renorm()

    @torch.no_grad()
    def fill(self):
        return dict(fill=float(self.fill_est), fill_chain=float(self.fill_ch_est),
                    **{k: int(v) for k, v in self.stats_acc.items()})

    @torch.no_grad()
    def fill_sample(self, n=1 << 20, seed=0):
        g = torch.Generator().manual_seed(self.seed + 13 * seed)
        idx = torch.randint(0, self.key_syn.numel(), (min(n, self.key_syn.numel()),), generator=g).to(self.key_syn.device)
        b = self.key_syn.view(-1)[idx].long()
        return float((torch.stack([b & 15, b >> 4], -1) > 0).float().mean())

    def byte_budget(self):
        return (sum(b.numel() * b.element_size() for n, b in self.named_buffers() if n in self.state_dict())
                + sum(p.numel() * p.element_size() for p in self.parameters()))


STORES["columns"] = WaveColumns
