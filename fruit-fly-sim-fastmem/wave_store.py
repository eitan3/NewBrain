"""The wave memory's pluggable STORAGE (README_memory_api.md, "wave memory").

A store holds what the memory knows, addressed by the address codes (wave_address.Codes) of the brain's thoughts:

    store.write(codes, content, strength, part=None, entries=None, trace=None)
                                                                      one write per row (token): content (n, D),
                                                                      strength (n,) per write, part (n,) partition
                                                                      (None = the store has one), entries (n, E) bool:
                                                                      which entry locations of the row are written;
                                                                      trace: what the forward read recalled (its path)
    store.read(codes, part=None, entries=None) -> content (n, D)      (the store's state is buffers; learned read
                                                                      gains may be nn.Parameters)
    store.read_traced(codes, ...) -> (content, trace or None)         the read + what it recalled (the memory keeps it)
    store.mass(codes, part=None) -> (n,) or None                      how much is stored under the codes (gate feature)
    store.confirm(codes, ok, part=None, trace=None)                   recall-confirm hook: ok (n,) in [0, 1] for reads
                                                                      that turned out right (e.g. counters + 1 along
                                                                      the recalled path)
    store.forget(step, parts=None)                                    forgetting / decrement hook, every step
    store.copy_parts(src, dst)                                        expert births (a child copies its parent)
    store.byte_budget() -> bytes; store.fill() -> dict; state = the module's buffers (state_dict)

Several entries per write and per read: Codes.loc holds E entry locations per row; `entries` selects which of them a
write uses (a policy may write a second entry path after a failed recall) and a read uses all of them by default.

STORES maps --wave-store names to classes: "sheet" = wave_sheet.WaveSheet (plan 26 section 8.4: the phase-coded wave
sheet, the default), "hebb" = HebbianSheet (the earlier PLACEHOLDER, kept as a reference and for the private parts).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPES = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


class WaveStore(nn.Module):
    """The interface (module doc). Subclasses register their state as buffers."""
    placeholder = False

    def write(self, codes, content, strength, part=None, entries=None, trace=None):
        raise NotImplementedError

    def read(self, codes, part=None, entries=None):
        raise NotImplementedError

    def read_traced(self, codes, part=None, entries=None):
        """-> (content, trace): trace = what the read recalled (a dict of (n, ...) tensors, e.g. the wave's path;
        trace["mass"] (n,) if the store has a familiarity) or None. The memory keeps it for write / confirm."""
        return self.read(codes, part=part, entries=entries), None

    @staticmethod
    def take_trace(tr, idx):
        return None if tr is None else {k: v[idx] for k, v in tr.items()}

    def mass(self, codes, part=None):
        return None

    def confirm(self, codes, ok, part=None, trace=None):
        pass

    def host_state(self):
        return {}

    def load_host_state(self, st):
        pass

    def forget(self, step, parts=None):
        pass

    def copy_parts(self, src, dst):
        pass

    def byte_budget(self):
        return sum(b.numel() * b.element_size() for b in self.buffers())

    def fill(self):
        return {}


class HebbianSheet(WaveStore):
    """PLACEHOLDER storage (the end-to-end path runs; the spec decides the real one). Not a row table: a sheet of
    synapses from code neurons to the D content neurons, every memory spread over many synapses and every synapse
    holding many memories superposed.

    Pre-synaptic units: (part, location, level-slice j, code m) with location over the entry map when `use_loc`
    (the hypercolumn: a code unit means something only at its location), j over the first `levels` x slices of the
    address. A write (Hebbian, outer product of pre activity and content) adds strength x content to the synapses of
    its active units: for each written entry location, each j's top-1 code. A read is the activity-weighted mean of
    the active units' mean contents: each entry location x each j's top-r codes (soft weights). mass: the strength
    stored under the active units (mean per unit). forget: a uniform multiplicative decay `decay` per step (0 = none).

    Known weak: a linear superposed read cannot pick ONE memory out of a crowded unit (W-bench, pass 3, the address
    at parity: 0.11 held / 0.06 cloze top-1 vs 0.71 / 0.55 for the idealised code-match ranking of the same codes).
    Recall that sharp needs a competitive (winner-take-all / attractor) store: the storage spec."""
    placeholder = True

    def __init__(self, d, n_parts, address, use_loc=True, levels=None, read_topk=2, dtype="fp16", decay=0.0):
        super().__init__()
        self.d, self.P, self.use_loc = d, int(n_parts), bool(use_loc)
        self.Lx = address.L if use_loc else 1
        self.J = (address.levels if levels is None else int(levels)) * address.S
        self.M, self.r, self.decay = address.M, int(read_topk), float(decay)
        self.per_part = self.Lx * self.J * self.M
        self.register_buffer("syn", torch.zeros(self.P * self.per_part, d, dtype=DTYPES[dtype]))
        self.register_buffer("strength", torch.zeros(self.P * self.per_part))

    def _units(self, codes, part, entries, write):
        """(n, U) unit ids and weights of each row's active units."""
        n = len(codes)
        dev = codes.loc.device
        code = codes.code[:, :self.J, :1] if write else codes.code[:, :self.J, :self.r]
        cw = torch.ones_like(code, dtype=torch.float32) if write else codes.code_w[:, :self.J, :self.r]
        cw = cw / cw.sum(-1, keepdim=True)
        loc = codes.loc if self.use_loc else torch.zeros(n, 1, dtype=torch.long, device=dev)
        lw = torch.ones(loc.shape, device=dev)
        if entries is not None and self.use_loc:
            lw = lw * entries.float()
        base = (torch.zeros(n, dtype=torch.long, device=dev) if part is None else part.long()) * self.per_part
        j = torch.arange(self.J, device=dev)
        u = (base[:, None, None, None] + (loc[:, :, None, None] * self.J + j[None, None, :, None]) * self.M
             + code[:, None])                                               # (n, E, J, r)
        w = lw[:, :, None, None] * cw[:, None]
        return u.reshape(n, -1), w.reshape(n, -1)

    @torch.no_grad()
    def write(self, codes, content, strength, part=None, entries=None, trace=None):
        if len(codes) == 0:
            return
        u, w = self._units(codes, part, entries, write=True)
        ws = w * strength.float()[:, None]
        step = max(1, (1 << 24) // max(1, u.size(1) * self.d))           # (rows per chunk: <= 16M values)
        for a in range(0, u.size(0), step):
            upd = ws[a:a + step, :, None] * content[a:a + step].float()[:, None]      # (n, U, D)
            self.syn.index_put_((u[a:a + step].reshape(-1),), upd.reshape(-1, self.d).to(self.syn.dtype),
                                accumulate=True)
        self.strength.index_put_((u.reshape(-1),), ws.reshape(-1), accumulate=True)

    @torch.no_grad()
    def read(self, codes, part=None, entries=None):
        """The activity-weighted mean over the active units of each unit's MEAN stored content (synapses / strength:
        a crowded unit counts as much as a specific one; W-bench, pass 3: 3x the plain sum). Empty units add 0."""
        u, w = self._units(codes, part, entries, write=False)
        s = self.strength[u]
        wn = torch.where(s > 0, w / s.clamp(min=1e-9), torch.zeros_like(w))
        r = F.embedding_bag(u, self.syn, per_sample_weights=wn.to(self.syn.dtype), mode="sum").float()
        return r / w.sum(-1, keepdim=True).clamp(min=1e-9)

    @torch.no_grad()
    def mass(self, codes, part=None):
        u, w = self._units(codes, part, None, write=False)
        return (self.strength[u] * w).sum(-1) / w.sum(-1).clamp(min=1e-9)

    @torch.no_grad()
    def forget(self, step, parts=None):
        if not self.decay:
            return
        f = 1 - self.decay
        if parts is None:
            self.syn.mul_(f)
            self.strength.mul_(f)
            return
        for p in parts:
            sl = slice(p * self.per_part, (p + 1) * self.per_part)
            self.syn[sl].mul_(f)
            self.strength[sl].mul_(f)

    @torch.no_grad()
    def copy_parts(self, src, dst):
        for s, t in zip(src, dst):
            self.syn[t * self.per_part:(t + 1) * self.per_part] = self.syn[s * self.per_part:(s + 1) * self.per_part]
            self.strength[t * self.per_part:(t + 1) * self.per_part] = \
                self.strength[s * self.per_part:(s + 1) * self.per_part]

    @torch.no_grad()
    def fill(self):
        return dict(units_written=float((self.strength > 0).float().mean()), strength_sum=float(self.strength.sum()))

    def part_slice(self, p):
        return slice(p * self.per_part, (p + 1) * self.per_part)


STORES = dict(hebb=HebbianSheet)
