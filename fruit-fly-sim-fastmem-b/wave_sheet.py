"""The wave memory's STORAGE (plan_v2/26 sections 6 and 8.3-8.4): a phase-coded wave sheet behind the wave_store
interface (STORES["sheet"]).

Sheet: `side` x `side` hypercolumns (hc) on a torus; each hc has K KEY units (driven by the thought's address codes:
`kslots` slices per hc, `codes` codes each) and M CONTENT units. A front is a 4 x 4 patch of hcs (slot i = 4 a + b).
A memory is a wave path of 4 fronts: F0 at the entry location (the SOM location's patch), F1..F3 each one of the 8
patches adjacent to the previous front (shift 4, never overlapping a visited front).

Synapse: a 2-bit counter c (0 = absent, 1..3) + a 3-bit phase theta, between a pre unit and a content unit of an hc in
the pre unit's 15 x 15 window (packed: counters 4 per byte, phases 2 per byte: 0.75 B per synapse). Kinds:
  wave   F0's key units -> F1's content units; F(t-1)'s content units -> F(t)'s content units (t = 2, 3)
  local  an hc's key units -> its own content units (window centre, an offset no wave synapse uses): the thought's
         codes bound to the content at that hc ("a unit fires when the wave and the thought's drive coincide")

READ (3 wave passes inside one read, new routing each pass, coarse-to-fine):
  pass t = 1..3: for each of the 8 candidate patches, each content unit j sums phasors
      z_j = sum_{i in F(t-1)} a_i [c_ij > 0] exp(i pi/4 (phi_i + theta_ij))          (the wave)
          + sum_{k in the thought's level-t keys at j's hc} a_k [c_kj > 0] exp(i pi/4 theta_kj)   (the drive)
  key phase 0, content phases from the codes. Pass 1 starts at the SOM location (F0's keys = level-1 codes); passes
  2 and 3 are driven by the level-2 / level-3 residual codes. Patch score = sum over its hcs of max_m |z| (+ a tiny
  direction term from the codes: it only breaks ties, so an empty sheet gives a deterministic default path); per hc
  the winner unit m and its phase round(angle(z) / (pi/4)). coincidence c_t = mean_h max |z| / (the input weight).
  ENTRIES: the read runs from each of the address's `entries` locations and keeps, per token, the entry with the
  larger sum of c_t ("the reader tries both").
  DECODE (to xf space, as section 6): r = sum_t alpha_t concat_slot C[t, slot, m, phase] over the recalled units
  (hcs with no input give 0); C = content codebooks (levels x 16 slots x M units x 8 phases x D / 16), learned in-step
  by residual minibatch k-means on the write targets (fast during the first `fast` write batches, then slow).
  alpha (levels,) is an nn.Parameter: trained only by the main loss through the read.

WRITE(codes, content, strength, entries, trace): the target is residual-quantised into (unit, phase) per front and
  slot; the path is the forward wave's path for that thought (trace; an empty sheet: the default path); every synapse
  on it gets c = max(c, strength) (strength 1, or 2 on surprise: the policy decides) and, if new, theta = phi_post -
  phi_pre (an existing synapse takes the new phase with probability `merge`). entries[:, e]: also write the path from
  entry e (the policy: the second entry after a failed recall).
CONFIRM(codes, ok, trace): a confirmed recall increments the synapses on its recalled path that are present with the
  recalled phase (cap 3; the spec's W3), keyed by the CURRENT codes. --wave-recon 1 (an arm, default off: the spec
  found its influx shortens memory life): absent synapses of the path also bind at count 1, so the memory is
  re-written under the current code (reconsolidation) while the path from the old code is not refreshed and fades.
FORGET(step): random decrement of a fraction lambda of all synapses per step; lambda follows the measured write rate
  and the measured fill (sampled every `fill_every` steps) so the presence fill holds near `fill` (0.12).
"""
import math

import torch
import torch.nn as nn

from wave_store import STORES, WaveStore

DIRS = ((-4, 0), (-4, 4), (0, 4), (4, 4), (4, 0), (4, -4), (0, -4), (-4, -4))   # angular order
WIN = 15
CENTRE = 7 * WIN + 7                      # the window centre: the local key -> content synapses


class WaveSheet(WaveStore):
    placeholder = False

    def __init__(self, d, n_parts, address, side=32, units=64, merge=0.5, fill=0.12, fill_every=100,
                 read_topk=1, levels=3, fast=2000, n_cap_fast=4096, n_cap_slow=200000, seed=1234, read_chunk=512,
                 use_loc=True, dtype=None, decay=0.0, pack="nibble"):
        super().__init__()
        assert n_parts == 1, "WaveSheet: one part (the global sheet); private stores are another STORES entry"
        assert d % 16 == 0, f"WaveSheet: d {d} must be a multiple of 16 (16 content slots per front)"
        assert address.S % 16 == 0, f"WaveSheet: --wave-slices {address.S} must be a multiple of 16"
        assert side % address.side == 0 and side >= WIN, f"WaveSheet: sheet side {side} vs SOM side {address.side}"
        assert units % 4 == 0
        self.d, self.side, self.M, self.levels = d, side, units, min(levels, address.levels)
        self.kslots, self.codes, self.S_addr = address.S // 16, address.M, address.S
        self.K = self.kslots * self.codes
        self.U = self.K + self.M
        self.H = side * side
        self.stride = side // address.side
        self.som_side = address.side
        self.merge, self.fill_target, self.fill_every = float(merge), float(fill), int(fill_every)
        self.r, self.seed, self.read_chunk = int(read_topk), int(seed), int(read_chunk)
        self.fast, self.n_cap_fast, self.n_cap_slow = int(fast), n_cap_fast, n_cap_slow
        self.dc = d // 16
        assert pack in ("nibble", "c2p3"), f"--wave-pack {pack!r}: nibble or c2p3"
        self.pack = pack
        self.PH = 5 if pack == "nibble" else 8          # phases per synapse
        n_pre = self.H * self.U
        if pack == "nibble":                            # 4 bits: 0 empty, 1 + 5 (c - 1) + ph (c 1..3, ph 0..4)
            self.register_buffer("syn", torch.zeros(n_pre, WIN * WIN, self.M // 2, dtype=torch.uint8))
        else:                                           # 2-bit counter + 3-bit phase (0.75 B / synapse)
            self.register_buffer("cnt", torch.zeros(n_pre, WIN * WIN, self.M // 4, dtype=torch.uint8))
            self.register_buffer("ph", torch.zeros(n_pre, WIN * WIN, self.M // 2, dtype=torch.uint8))
        KP = self.M * self.PH
        self.register_buffer("C", torch.zeros(self.levels, 16, KP, self.dc))              # content codebooks
        self.register_buffer("C_n", torch.zeros(self.levels, 16, KP))
        self.register_buffer("C_use", torch.full((self.levels, 16, KP), 1.0 / KP))
        self.register_buffer("lam", torch.zeros(()))
        self.register_buffer("fill_est", torch.zeros(()))
        self.register_buffer("syn_written", torch.zeros((), dtype=torch.float64))   # since the last controller update
        self.alpha = nn.Parameter(torch.ones(self.levels))
        g = torch.Generator().manual_seed(self.seed + 31)
        self.register_buffer("Dproj", torch.randn(self.levels, 8, self.K, generator=g) / math.sqrt(16 * self.kslots))
        a = torch.arange(4)
        self.register_buffer("patch", torch.stack(torch.meshgrid(a, a, indexing="ij"), -1).view(16, 2),
                             persistent=False)
        self.register_buffer("dirs", torch.tensor(DIRS), persistent=False)
        ang = torch.arange(8) * (math.pi / 4)
        self.register_buffer("cos8", ang.cos(), persistent=False)
        self.register_buffer("sin8", ang.sin(), persistent=False)
        self.register_buffer("sh2", torch.tensor([0, 2, 4, 6], dtype=torch.uint8), persistent=False)
        self.register_buffer("sh4", torch.tensor([0, 4], dtype=torch.uint8), persistent=False)
        # read look-up tables over packed bytes (rebuilt, never saved): cos / sin of (pre phase + each synapse's phase)
        # of the 2 synapses of a byte, times the synapse's presence (nibble), or apart (c2p3: presence of the 4
        # counters of a counter byte)
        byte = torch.arange(256)
        PH = self.PH
        if pack == "nibble":
            s = (byte[:, None] >> torch.tensor([0, 4])) & 15                       # (256, 2)
            th, pres = torch.remainder(s - 1, 5), (s > 0).float()
        else:
            th, pres = (byte[:, None] >> torch.tensor([0, 4])) & 7, torch.ones(256, 2)
            self.register_buffer("lut_pres", (((byte[:, None] >> torch.tensor([0, 2, 4, 6])) & 3) > 0).float(),
                                 persistent=False)                                 # (256, 4)
        ang = (torch.remainder(torch.arange(PH)[:, None, None] + th[None], PH)) * (2 * math.pi / PH)
        self.register_buffer("lut_cos", (ang.cos() * pres[None]).reshape(PH * 256, 2), persistent=False)
        self.register_buffer("lut_sin", (ang.sin() * pres[None]).reshape(PH * 256, 2), persistent=False)
        self.recon = False                              # (--wave-recon: set by the memory)
        self.n_cw = 0                                   # host: content-codebook update batches (extra state)
        self.n_step = 0
        self.stats_acc = dict(writes=0, confirms=0, decrements=0)

    # ------------------------------------------------------------------ host state (CodedWaveMemory saves it)
    def host_state(self):
        return dict(n_cw=self.n_cw, n_step=self.n_step)

    def load_host_state(self, st):
        self.n_cw, self.n_step = int(st.get("n_cw", 0)), int(st.get("n_step", 0))

    # ------------------------------------------------------------------ packing
    def _unpack_c(self, x):
        return ((x.unsqueeze(-1) >> self.sh2) & 3).flatten(-2)

    def _unpack_p(self, x):
        return ((x.unsqueeze(-1) >> self.sh4) & 7).flatten(-2)

    def _pack_c(self, v):
        v = v.to(torch.uint8).view(*v.shape[:-1], -1, 4)
        return (v << self.sh2).sum(-1).to(torch.uint8)

    def _pack_p(self, v):
        v = v.to(torch.uint8).view(*v.shape[:-1], -1, 2)
        return (v << self.sh4).sum(-1).to(torch.uint8)

    def _flat_rows(self):
        """(rows, M // 2) views of the storage: nibble [syn], c2p3 [cnt (M // 4 bytes), ph]."""
        if self.pack == "nibble":
            return (self.syn.view(-1, self.M // 2),)
        return self.cnt.view(-1, self.M // 4), self.ph.view(-1, self.M // 2)

    def _rows_get(self, urow):
        """(R,) flat (pre, w) rows -> counts c (R, M) and phases p (R, M), long."""
        if self.pack == "nibble":
            s = self._unpack_p4(self._flat_rows()[0][urow]).long()
            return torch.where(s > 0, (s - 1) // 5 + 1, torch.zeros_like(s)), torch.where(s > 0, (s - 1) % 5,
                                                                                         torch.zeros_like(s))
        cf, pf = self._flat_rows()
        return self._unpack_c(cf[urow]).long(), self._unpack_p(pf[urow]).long()

    def _rows_put(self, urow, c, p):
        if self.pack == "nibble":
            s = torch.where(c > 0, 1 + 5 * (c.clamp(min=1) - 1) + p, torch.zeros_like(c))
            self._flat_rows()[0][urow] = self._pack_p(s)
            return
        cf, pf = self._flat_rows()
        cf[urow] = self._pack_c(c)
        pf[urow] = self._pack_p(p)

    def _unpack_p4(self, x):
        return ((x.unsqueeze(-1) >> self.sh4) & 15).flatten(-2)

    def syn_at(self, pre, w):
        """(c, p) (..., M) of the synapses of pre units `pre` at window offsets `w` (tests, diagnostics)."""
        row = pre * (WIN * WIN) + w
        c, p = self._rows_get(row.reshape(-1))
        return c.view(*row.shape, self.M), p.view(*row.shape, self.M)

    def _z(self, rows, w, pre_ph, pre_w, shape):
        """The phasor sums (re, im) over the pre units (dim 1) of the gathered synapse rows (rows (n, P, 1) or
        (n, 16, KR) with w broadcastable), each times pre_w; shape = the gathered (..., M) shape."""
        if self.pack == "nibble":
            idx = pre_ph * 256 + self.syn[rows, w].long()
            return (self.lut_cos[idx].view(shape) * pre_w, self.lut_sin[idx].view(shape) * pre_w)
        pw = self.lut_pres[self.cnt[rows, w].long()].view(shape) * pre_w
        idx = pre_ph * 256 + self.ph[rows, w].long()
        return pw * self.lut_cos[idx].view(shape), pw * self.lut_sin[idx].view(shape)

    @torch.no_grad()
    def fill_sample(self, n=1 << 20, seed=0):
        """The presence fill on n random synapses."""
        buf = self.syn if self.pack == "nibble" else self.cnt
        g = torch.Generator().manual_seed(self.seed + 13 * seed)
        idx = torch.randint(0, buf.numel(), (min(n, buf.numel()),), generator=g).to(buf.device)
        b = buf.view(-1)[idx][:, None]
        v = self._unpack_p4(b) if self.pack == "nibble" else self._unpack_c(b)
        return float((v > 0).float().mean())

    @property
    def n_synapses(self):
        return (self.syn.numel() * 2) if self.pack == "nibble" else self.cnt.numel() * 4

    # ------------------------------------------------------------------ geometry
    def _hc(self, yx):
        return (yx[..., 0] % self.side) * self.side + (yx[..., 1] % self.side)

    def _patch_hcs(self, corner):
        """corner (..., 2) -> (..., 16) hc ids."""
        return self._hc(corner.unsqueeze(-2) + self.patch)

    def _widx(self, hp, hq):
        S = self.side
        dy = (hq // S - hp // S + S // 2) % S - S // 2
        dx = (hq % S - hp % S + S // 2) % S - S // 2
        return (dy + 7) * WIN + (dx + 7)

    def _entry_corner(self, loc):
        ly, lx = loc // self.som_side, loc % self.som_side
        return torch.stack([ly * self.stride, lx * self.stride], -1)

    def _overlap(self, a, b):
        S = self.side
        dy = ((a[..., 0] - b[..., 0] + S // 2) % S - S // 2).abs()
        dx = ((a[..., 1] - b[..., 1] + S // 2) % S - S // 2).abs()
        return (dy < 4) & (dx < 4)

    def _keys(self, codes, lev, r):
        """The thought's level-`lev` key units per front slot: ids (n, 16, kslots * r), weights (same)."""
        n = len(codes)
        sl = torch.arange(16, device=codes.code.device)
        ids, ws = [], []
        for j in range(self.kslots):
            col = lev * self.S_addr + j * 16 + sl                                 # (16,)
            c = codes.code[:, col, :r]                                            # (n, 16, r)
            w = codes.code_w[:, col, :r]
            ids.append(j * self.codes + c)
            ws.append(w / w.sum(-1, keepdim=True).clamp(min=1e-9))
        return torch.cat(ids, -1), torch.cat(ws, -1)

    # ------------------------------------------------------------------ the wave (one read)
    @torch.no_grad()
    def _wave(self, codes, entries=None, r=None):
        """-> trace dict: corner (n, E, 4, 2), unit (n, E, 3, 16), phase (n, E, 3, 16), mag (n, E, 3, 16),
        c (n, E, 3), best (n,) entry index."""
        r = self.r if r is None else r
        outs = []
        for a in range(0, len(codes), self.read_chunk):
            outs.append(self._wave_chunk(codes.take(slice(a, a + self.read_chunk)), r))
        tr = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
        score = tr["c"].sum(-1)                                                    # (n, E)
        if entries is not None:
            score = torch.where(entries, score, torch.full_like(score, -1.0))
        tr["best"] = score.argmax(-1)
        return tr

    def _wave_chunk(self, codes, r):
        n, dev = len(codes), codes.loc.device
        E = codes.loc.size(1)
        out = dict(corner=[], dcorner=[], unit=[], phase=[], mag=[], c=[])
        keys = [self._keys(codes, lev, r) for lev in range(self.levels)]
        for e in range(E):
            corner = self._entry_corner(codes.loc[:, e])                           # (n, 2)
            hc0 = self._patch_hcs(corner)                                          # (n, 16)
            kid, kw = keys[0]
            pre = (hc0[:, :, None] * self.U + kid).reshape(n, -1)                  # F0's key units
            pre_hc = hc0[:, :, None].expand_as(kid).reshape(n, -1)
            pre_ph = torch.zeros_like(pre)
            pre_w = kw.reshape(n, -1)
            visited = [corner]
            corners, units, phases, mags, cs = [corner], [], [], [], []
            for t in range(self.levels):
                lk_id, lk_w = keys[t]
                best = None
                for di in range(8):
                    cand = corner + self.dirs[di]
                    ok = torch.ones(n, dtype=torch.bool, device=dev)
                    for v in visited:
                        ok &= ~self._overlap(cand, v)
                    hcs = self._patch_hcs(cand)                                    # (n, 16)
                    w = self._widx(pre_hc[:, :, None], hcs[:, None, :])            # (n, P, 16)
                    rows = pre[:, :, None]
                    zr, zi = self._z(rows, w, pre_ph[:, :, None, None], pre_w[:, :, None, None],
                                     (n, pre.size(1), 16, self.M))
                    zr, zi = zr.sum(1), zi.sum(1)                                  # (n, 16, M)
                    krow = hcs[:, :, None] * self.U + lk_id                        # (n, 16, KR)
                    kr, ki = self._z(krow, CENTRE, torch.zeros_like(krow)[..., None], lk_w[..., None],
                                     (*krow.shape, self.M))                        # (pre phase 0: keys)
                    zr, zi = zr + kr.sum(2), zi + ki.sum(2)
                    mag = (zr * zr + zi * zi).sqrt()
                    mx, m = mag.max(-1)                                            # (n, 16)
                    dterm = (self.Dproj[t, di][lk_id] * lk_w).sum((1, 2)) * 1e-3
                    sc = torch.where(ok, mx.sum(-1) + dterm, torch.full_like(dterm, -1e9))
                    ph = torch.remainder(torch.round(torch.atan2(zi.gather(-1, m[..., None]), zr.gather(
                        -1, m[..., None])).squeeze(-1) / (2 * math.pi / self.PH)).long(), self.PH)
                    cur = dict(sc=sc, corner=cand, hcs=hcs, m=m, ph=ph, mx=mx)
                    if best is None:
                        best = cur
                    else:
                        b = sc > best["sc"]
                        best = {k: torch.where(b.view(-1, *([1] * (v.dim() - 1))), v, best[k]) for k, v in cur.items()}
                wsum = pre_w.sum(-1) + lk_w.sum(-1).mean(-1)
                c_t = best["mx"].mean(-1) / wsum.clamp(min=1e-9)
                corners.append(best["corner"])
                units.append(best["m"])
                phases.append(best["ph"])
                mags.append(best["mx"])
                cs.append(c_t)
                visited.append(best["corner"])
                corner = best["corner"]
                pre = best["hcs"] * self.U + self.K + best["m"]                     # the recalled content units
                pre_hc, pre_ph = best["hcs"], best["ph"]
                pre_w = torch.ones(n, 16, device=dev)
            # the thought's own default path (a NEW memory's write path): per pass the direction its level codes
            # prefer (Dproj), among the non-overlapping ones; depends on the thought only
            dc = [corners[0]]
            vis = [corners[0]]
            for t in range(self.levels):
                lk_id, lk_w = keys[t]
                ds = torch.stack([(self.Dproj[t, di][lk_id] * lk_w).sum((1, 2)) for di in range(8)], -1)   # (n, 8)
                cand = dc[-1][:, None, :] + self.dirs[None]                        # (n, 8, 2)
                ok = torch.ones(n, 8, dtype=torch.bool, device=dev)
                for v in vis:
                    ok &= ~self._overlap(cand, v[:, None, :])
                k = torch.where(ok, ds, torch.full_like(ds, -1e9)).argmax(-1)
                nxt = cand[torch.arange(n, device=dev), k]
                dc.append(nxt)
                vis.append(nxt)
            out.setdefault("dcorner", []).append(torch.stack(dc, 1))
            out["corner"].append(torch.stack(corners, 1))
            out["unit"].append(torch.stack(units, 1))
            out["phase"].append(torch.stack(phases, 1))
            out["mag"].append(torch.stack(mags, 1))
            out["c"].append(torch.stack(cs, 1))
        return {k: torch.stack(v, 1) for k, v in out.items()}

    # ------------------------------------------------------------------ the interface
    def read_traced(self, codes, part=None, entries=None):
        tr = self._wave(codes, entries)
        ar = torch.arange(len(codes), device=codes.loc.device)
        tr["mass"] = tr["c"][ar, tr["best"]].mean(-1)
        return self.decode(tr), tr

    def decode(self, tr):
        """The recalled (unit, phase) per level and slot -> content in xf space: sum_l alpha_l concat_slot C[...];
        hcs without input give 0. Gradient only to alpha (the trace and the codebooks are fixed)."""
        b = tr["best"]
        ar = torch.arange(b.numel(), device=b.device)
        unit, phase, mag = tr["unit"][ar, b], tr["phase"][ar, b], tr["mag"][ar, b]     # (n, 3, 16)
        code = unit * self.PH + phase
        lv = torch.arange(self.levels, device=code.device)[None, :, None]
        sl = torch.arange(16, device=code.device)[None, None, :]
        vec = self.C[lv, sl, code] * (mag > 0)[..., None].float()                     # (n, 3, 16, dc)
        return (vec.flatten(-2) * self.alpha[None, :, None]).sum(1)                   # (n, D)

    def read(self, codes, part=None, entries=None):
        return self.read_traced(codes, part, entries)[0]

    def mass(self, codes, part=None):
        return self._wave(codes)["c"].amax(1).mean(-1)

    @staticmethod
    def take_trace(tr, idx):
        return {k: v[idx] for k, v in tr.items()}

    # ------------------------------------------------------------------ content codebooks (in-step, no gradient)
    @torch.no_grad()
    def quantize(self, content):
        """content (n, D) -> codes (n, levels, 16) of (unit * PH + phase), residual over the levels."""
        R = content.float().view(-1, 16, self.dc)
        out = []
        for lev in range(self.levels):
            B = self.C[lev]                                                        # (16, M8, dc)
            dd = (R * R).sum(-1, keepdim=True) - 2 * torch.einsum("nsd,smd->nsm", R, B) + (B * B).sum(-1)[None]
            a = dd.argmin(-1)
            out.append(a)
            R = R - B[torch.arange(16, device=R.device)[None], a]
        return torch.stack(out, 1)

    @torch.no_grad()
    def update_books(self, content):
        """Residual minibatch k-means on this batch of write targets (fast, then slow after `fast` batches)."""
        R = content.float().view(-1, 16, self.dc)
        n = R.size(0)
        if n == 0:
            return
        K8 = self.M * self.PH
        ar = torch.arange(16, device=R.device)
        if self.n_cw == 0:
            g = torch.Generator().manual_seed(self.seed + 7)
            Rr = R
            for lev in range(self.levels):
                idx = torch.randint(0, n, (16, K8), generator=g).to(R.device)
                self.C[lev].copy_(Rr[idx, ar[:, None]] * (1 + 0.01 * torch.randn(16, K8, 1, generator=g).to(R.device)))
                a = ((Rr[:, :, None] - self.C[lev][None]).pow(2).sum(-1)).argmin(-1)
                Rr = Rr - self.C[lev][ar[None], a]
        fast = self.n_cw < self.fast
        cap = self.n_cap_fast if fast else self.n_cap_slow
        for lev in range(self.levels):
            B = self.C[lev]
            dd = (R * R).sum(-1, keepdim=True) - 2 * torch.einsum("nsd,smd->nsm", R, B) + (B * B).sum(-1)[None]
            a = dd.argmin(-1)                                                      # (n, 16)
            oh = torch.nn.functional.one_hot(a, K8).float()
            cnt = oh.sum(0)
            sums = torch.einsum("nsm,nsd->smd", oh, R)
            new = self.C_n[lev] + cnt
            B.add_((cnt / new.clamp(min=1))[..., None] * (sums / cnt.clamp(min=1)[..., None] - B))
            self.C_n[lev].copy_(new.clamp(max=cap))
            self.C_use[lev].mul_(0.98).add_(0.02 * cnt / n)
            if fast:                                                               # re-seed dead codes (fast phase)
                err = (R - B[ar[None], a]).pow(2).sum(-1)
                dead = self.C_use[lev] < 0.05 / K8
                k = min(K8, n)
                worst = err.topk(k, 0).indices
                rank = (dead.long().cumsum(-1) - 1).clamp(min=0, max=k - 1)
                seed = R[worst.gather(0, rank.T).T, ar[:, None]]
                B.copy_(torch.where(dead[..., None], seed, B))
                self.C_use[lev].copy_(torch.where(dead, torch.full_like(cnt, 1.0 / K8), self.C_use[lev]))
                a = ((R[:, :, None] - B[None]).pow(2).sum(-1)).argmin(-1)
            R = R - B[ar[None], a]
        self.n_cw += 1

    # ------------------------------------------------------------------ synapse updates
    def _path_synapses(self, codes, corner, unit, phase, keys_r=1):
        """The synapses of one path per row: (pre, w, post unit, theta), each (n, n_syn)."""
        n = len(codes)
        hcs = self._patch_hcs(corner)                                              # (n, 4, 16)
        pre_l, w_l, post_l, th_l = [], [], [], []
        for t in range(self.levels):
            kid, _ = self._keys(codes, t, keys_r)                                  # (n, 16, kslots)
            post_hc, post_m, post_ph = hcs[:, t + 1], unit[:, t], phase[:, t]      # (n, 16)
            if t == 0:                                                             # F0 keys -> F1 contents
                k0, _ = self._keys(codes, 0, keys_r)
                p_unit = (hcs[:, 0, :, None] * self.U + k0).reshape(n, -1)
                p_hc = hcs[:, 0, :, None].expand_as(k0).reshape(n, -1)
                p_ph = torch.zeros_like(p_unit)
            else:                                                                  # F(t-1) contents -> F(t)
                p_unit = hcs[:, t] * self.U + self.K + unit[:, t - 1]
                p_hc, p_ph = hcs[:, t], phase[:, t - 1]
            P = p_unit.size(1)
            pre_l.append(p_unit[:, :, None].expand(n, P, 16).reshape(n, -1))
            w_l.append(self._widx(p_hc[:, :, None], post_hc[:, None, :]).reshape(n, -1))
            post_l.append(post_m[:, None, :].expand(n, P, 16).reshape(n, -1))
            th_l.append(torch.remainder(post_ph[:, None, :] - p_ph[:, :, None], self.PH).reshape(n, -1))
            # local: this front's level-t keys -> its own content unit, theta = phi
            lk = (post_hc[:, :, None] * self.U + kid).reshape(n, -1)
            pre_l.append(lk)
            w_l.append(torch.full_like(lk, CENTRE))
            post_l.append(post_m[:, :, None].expand_as(kid).reshape(n, -1))
            th_l.append(post_ph[:, :, None].expand_as(kid).reshape(n, -1))
        return tuple(torch.cat(x, 1) for x in (pre_l, w_l, post_l, th_l))

    @torch.no_grad()
    def _update_syn(self, pre, w, post, th, value, mode, gen):
        """mode 'set': c = max(c, value), new synapse phase th (existing: th with probability merge);
        mode 'inc': c = min(c + 1, 3), new synapse phase th, existing keep theirs."""
        pre, w, post, th, value = (x.reshape(-1) for x in (pre, w, post, th, value))
        keep = value > 0
        pre, w, post, th, value = pre[keep], w[keep], post[keep], th[keep], value[keep]
        if pre.numel() == 0:
            return 0
        row = pre * (WIN * WIN) + w
        urow, inv = torch.unique(row, return_inverse=True)                       # (unique rows: one put per row)
        c, p = self._rows_get(urow)                                               # (R, M) counts / phases
        old = c[inv, post]
        if mode == "set":
            newc = torch.maximum(old, value.long().clamp(max=3))
            take = (old == 0) | (torch.rand(old.shape, generator=gen, device=gen.device).to(old.device) < self.merge)
        elif self.recon:                       # (--wave-recon 1: absent synapses of the recalled path bind at 1)
            newc = (old + 1).clamp(max=3)
            take = old == 0
        else:                                  # (spec W3: present AND the recalled phase -> +1; nothing new binds)
            match = (old > 0) & (p[inv, post] == th.long())
            newc = torch.where(match, (old + 1).clamp(max=3), old)
            take = torch.zeros_like(match)
        c.view(-1).scatter_reduce_(0, inv * self.M + post, newc, "amax")
        sel = take.nonzero().squeeze(-1)
        tgt = (inv * self.M + post)[sel]
        win = torch.full((c.numel(),), -1, dtype=torch.long, device=c.device)   # (duplicates: the last writer wins,
        win.scatter_reduce_(0, tgt, torch.arange(tgt.numel(), device=c.device), "amax")   # deterministically)
        has = win >= 0
        p.view(-1)[has] = th[sel][win[has]].long()
        self._rows_put(urow, c, p)
        self.syn_written += float(pre.numel())
        return int(pre.numel())

    def _gen(self, salt):
        return torch.Generator().manual_seed(self.seed * 1_000_003 + self.n_cw * 7919 + self.n_step * 31 + salt)

    @torch.no_grad()
    def write(self, codes, content, strength, part=None, entries=None, trace=None):
        n = len(codes)
        if n == 0:
            return
        self.update_books(content)
        q = self.quantize(content)                                                 # (n, levels, 16)
        unit, phase = q // self.PH, q % self.PH
        if trace is None:
            trace = self._wave(codes)
        E = trace["corner"].size(1)
        if entries is None:
            entries = torch.zeros(n, E, dtype=torch.bool, device=content.device)
            entries[:, 0] = True
        g = self._gen(1)
        fam = trace.get("ok")                                   # (the memory marks confirmed recalls: familiar)
        fam = torch.zeros(n, dtype=torch.bool, device=content.device) if fam is None else fam > 0
        for e in range(E):
            rows = entries[:, e].nonzero().squeeze(-1)
            if rows.numel() == 0:
                continue
            c2 = codes.take(rows)
            # familiar (a confirmed recall from this entry): the recalled patches; new: the thought's default path
            use_rec = (fam[rows] & (trace["best"][rows] == e))[:, None, None]
            corner = torch.where(use_rec, trace["corner"][rows, e], trace["dcorner"][rows, e])
            pre, w, post, th = self._path_synapses(c2, corner, unit[rows], phase[rows])
            val = strength[rows].round().long()[:, None].expand_as(pre)
            self._update_syn(pre, w, post, th, val, "set", g)
        self.stats_acc["writes"] += n

    @torch.no_grad()
    def confirm(self, codes, ok, part=None, trace=None):
        """+1 (cap 3) along each confirmed read's recalled path, keyed by the current codes (reconsolidation)."""
        if trace is None:
            return
        rows = (ok > 0).nonzero().squeeze(-1)
        if rows.numel() == 0:
            return
        b = trace["best"][rows]
        c2 = codes.take(rows)
        pre, w, post, th = self._path_synapses(c2, trace["corner"][rows, b], trace["unit"][rows, b],
                                               trace["phase"][rows, b])
        self._update_syn(pre, w, post, th, torch.ones_like(pre), "inc", self._gen(2))
        self.stats_acc["confirms"] += int(rows.numel())

    @torch.no_grad()
    def forget(self, step, parts=None):
        """Random decrement of lambda x (all synapses); lambda from the write rate and the measured fill."""
        self.n_step += 1
        N = self.n_synapses
        if self.n_step % self.fill_every == 0:
            fill = self.fill_sample(seed=self.n_step)
            self.fill_est.fill_(fill)
            rate = float(self.syn_written) / self.fill_every
            base = rate / (self.fill_target * N)
            ratio = min(4.0, fill / self.fill_target) ** 2       # (below the target forgetting fades out)
            self.lam.fill_(base * ratio)
            self.syn_written.zero_()
        lam = float(self.lam)
        per = 2 if self.pack == "nibble" else 4                                   # synapses per byte
        buf = self.syn if self.pack == "nibble" else self.cnt
        k = int(lam * N / 4)            # bytes hit (one synapse each; lambda's c2p3 meaning, whatever the layout)
        if k <= 0:
            return
        g = torch.Generator().manual_seed(self.seed * 7 + self.n_step)
        ib = torch.unique(torch.randint(0, buf.numel(), (k,), generator=g)).to(buf.device)
        slot = torch.randint(0, per, (ib.numel(),), generator=g).to(buf.device)
        flat = buf.view(-1)
        b = flat[ib].long()
        if self.pack == "nibble":                                                  # dec: s - 5 if s > 5, else 0
            sh = 4 * slot
            v = (b >> sh) & 15
            nv = torch.where(v > 5, v - 5, torch.zeros_like(v))
            flat[ib] = ((b & ~(15 << sh)) | (nv << sh)).to(torch.uint8)
        else:
            sh = 2 * slot
            v = (b >> sh) & 3
            nv = (v - 1).clamp(min=0)
            flat[ib] = ((b & ~(3 << sh)) | (nv << sh)).to(torch.uint8)
        self.stats_acc["decrements"] += int(ib.numel())

    @torch.no_grad()
    def fill(self):
        return dict(fill=float(self.fill_est), lam=float(self.lam), n_cw=self.n_cw, **self.stats_acc)

    def copy_parts(self, src, dst):
        pass

    def byte_budget(self):
        return (sum(b.numel() * b.element_size() for n, b in self.named_buffers() if n in self.state_dict())
                + sum(p.numel() * p.element_size() for p in self.parameters()))


STORES["sheet"] = WaveSheet
