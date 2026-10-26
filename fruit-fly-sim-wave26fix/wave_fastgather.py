"""Fast, sync-free gathers for the WaveColumns recall (wave_columns.py) and the sweep's randoms.

  gather_rows(storage, row_idx, row_bytes=32)      whole rows of a uint8 store, as int32 [..., row_bytes // 4]
  key_Z(key_syn, rows, nib)                        == WaveColumns._Z(rows) bit for bit: (zr, zi) (..., 64) fp32
  key_Z_keys(key_syn, E, keys, l, fan, nib)        the same, the rows computed in-kernel from (E, the level-l keys)
  chain_nch(chain_bits, rows)                      == WaveColumns._nch(rows): (..., 64) fp32 set-bit counts
  nib_lut(lut16)                                   the (16, 2) per-nibble (cos, sin) x presence table, taken from LUT16
  sweep_randoms(n, step, device, seed)             uint8 [n] randoms made on the device (a per-device generator,
                                                   reseeded on the host per step with a python int)

Exactness: the old path sums LUT16 rows with embedding_bag(mode="sum"), a sequential fp32 sum over the fan-in q = 0..15
starting from +0. key_Z adds the very same per-nibble values (nib_lut reads them out of LUT16) in the same order, so the
result is bit-identical (tests_fastgather.py checks it on CPU and GPU). On CUDA a Triton kernel fuses the row gather,
the nibble unpack, the LUT look-up and the sum (no intermediate tensors); on CPU (or without Triton) a torch path
does the same arithmetic. No function here syncs the host (no .item, nonzero, bool-mask indexing, unique, or a
tensor-valued if).
"""
import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                   # pragma: no cover
    _HAS_TRITON = False

L, C, F, UK, UC = 3, 16, 16, 16, 64


# ---------------------------------------------------------------------------------------------------------- helpers
def _table(storage, row_bytes, dtype=torch.int32):
    """the uint8 store viewed (no copy) as [n_rows, row_bytes // itemsize] of dtype."""
    flat = storage.reshape(-1)                      # (a view for a contiguous store)
    isz = torch.empty((), dtype=dtype).element_size()
    return flat.view(dtype).view(-1, row_bytes // isz)


def gather_rows(storage_u8_flat, row_idx, row_bytes=32, dtype=torch.int32):
    """Whole rows of a uint8 store: row_idx (...) int32/int64 -> (..., row_bytes // 4) int32 (dtype=torch.int64 for
    the 8-byte chain rows: (..., 1)). One index_select on dim 0 of the int32 view: coalesced, no copies of the store."""
    tab = _table(storage_u8_flat, row_bytes, dtype)
    out = torch.index_select(tab, 0, row_idx.reshape(-1))
    return out.view(*row_idx.shape, tab.size(1))


def nib_lut(lut16):
    """(16, 2) fp32: the (cos, sin) x presence of nibble s, read out of LUT16 (word s = nibble 0 is s) so the values
    are the very floats the old embedding_bag path adds."""
    return lut16[:16, :2].contiguous()


def _rows_check(rows):
    return rows if rows.dtype in (torch.int32, torch.int64) else rows.long()


# ---------------------------------------------------------------------------------------------------------- torch path
_SH8 = {}


def _shifts(device):
    k = (device.type, device.index)
    if k not in _SH8:
        _SH8[k] = (4 * torch.arange(8, dtype=torch.int32, device=device),
                   torch.arange(64, dtype=torch.int64, device=device))
    return _SH8[k]


def _key_Z_torch(key_syn, rows, nib):
    sh = rows.shape
    w = gather_rows(key_syn, rows, 32)                                      # (..., F, 8) int32
    s = (w[..., None] >> _shifts(w.device)[0]).bitwise_and_(15)             # (..., F, 8, 8): unit 8 j + k
    s = s.flatten(-2).long()                                                # (..., F, 64)
    v = nib[s]                                                              # (..., F, 64, 2)
    acc = torch.zeros(*sh[:-1], UC, 2, dtype=nib.dtype, device=nib.device)
    for q in range(sh[-1]):                                                 # (sequential: embedding_bag's order)
        acc += v[..., q, :, :]
    return acc[..., 0], acc[..., 1]


def _chain_nch_torch(chain_bits, rows):
    w = gather_rows(chain_bits, rows, 8, torch.int64)[..., 0]               # (..., W) int64
    on = (w[..., None] >> _shifts(w.device)[1]).bitwise_and_(1)            # (..., W, 64)
    return on.sum(-2).float()


# ---------------------------------------------------------------------------------------------------------- triton
if _HAS_TRITON:
    @triton.jit
    def _kz_kernel(syn_ptr, rows_ptr, nib_ptr, out_ptr, NT, NF: tl.constexpr, BT: tl.constexpr):
        pid = tl.program_id(0)
        t = (pid * BT + tl.arange(0, BT)).to(tl.int64)
        tm = t < NT
        u = tl.arange(0, 64)
        wi = (u // 8)[None, :]
        shf = ((u % 8) * 4)[None, :]
        acc_r = tl.zeros((BT, 64), dtype=tl.float32)
        acc_i = tl.zeros((BT, 64), dtype=tl.float32)
        for q in tl.static_range(NF):
            r = tl.load(rows_ptr + t * NF + q, mask=tm, other=0).to(tl.int64)
            w = tl.load(syn_ptr + r[:, None] * 8 + wi, mask=tm[:, None], other=0)
            s = (w >> shf) & 15
            acc_r += tl.load(nib_ptr + s * 2)
            acc_i += tl.load(nib_ptr + s * 2 + 1)
        o = out_ptr + (t[:, None] * 64 + u[None, :]) * 2
        tl.store(o, acc_r, mask=tm[:, None])
        tl.store(o + 1, acc_i, mask=tm[:, None])

    @triton.jit
    def _kzk_kernel(syn_ptr, E_ptr, keys_ptr, nib_ptr, out_ptr, NT, l, kstride, BT: tl.constexpr):
        """rows in-kernel: t = (token n, column c); row = ((((E[n] 3 + l) 16 + c) 16 + q) 16 + keys[n, (2c + q) % 32]."""
        pid = tl.program_id(0)
        t = (pid * BT + tl.arange(0, BT)).to(tl.int64)
        tm = t < NT
        n = t // 16
        c = t % 16
        e = tl.load(E_ptr + n, mask=tm, other=0).to(tl.int64)
        base = ((e * 3 + l) * 16 + c) * 16
        u = tl.arange(0, 64)
        wi = (u // 8)[None, :]
        shf = ((u % 8) * 4)[None, :]
        acc_r = tl.zeros((BT, 64), dtype=tl.float32)
        acc_i = tl.zeros((BT, 64), dtype=tl.float32)
        for q in tl.static_range(16):
            k = tl.load(keys_ptr + n * kstride + (2 * c + q) % 32, mask=tm, other=0).to(tl.int64)
            r = (base + q) * 16 + k
            w = tl.load(syn_ptr + r[:, None] * 8 + wi, mask=tm[:, None], other=0)
            s = (w >> shf) & 15
            acc_r += tl.load(nib_ptr + s * 2)
            acc_i += tl.load(nib_ptr + s * 2 + 1)
        o = out_ptr + (t[:, None] * 64 + u[None, :]) * 2
        tl.store(o, acc_r, mask=tm[:, None])
        tl.store(o + 1, acc_i, mask=tm[:, None])

    @triton.jit
    def _nch_kernel(cb_ptr, rows_ptr, out_ptr, NT, NW: tl.constexpr, BT: tl.constexpr):
        pid = tl.program_id(0)
        t = (pid * BT + tl.arange(0, BT)).to(tl.int64)
        tm = t < NT
        u = tl.arange(0, 64)
        wi = (u // 32)[None, :]
        shf = (u % 32)[None, :]
        acc = tl.zeros((BT, 64), dtype=tl.float32)
        for i in tl.static_range(NW):
            r = tl.load(rows_ptr + t * NW + i, mask=tm, other=0).to(tl.int64)
            w = tl.load(cb_ptr + r[:, None] * 2 + wi, mask=tm[:, None], other=0)
            acc += ((w >> shf) & 1).to(tl.float32)
        tl.store(out_ptr + t[:, None] * 64 + u[None, :], acc, mask=tm[:, None])


def _use_triton(x, impl):
    if impl == "torch":
        return False
    ok = _HAS_TRITON and x.is_cuda
    if impl == "triton" and not ok:
        raise RuntimeError("wave_fastgather: the triton path needs CUDA and triton")
    return ok


_BT = 32


def key_Z(key_syn, rows, nib, impl="auto"):
    """key rows (..., F) int32/int64 into key_syn (uint8, 32-B rows) -> (zr, zi), each (..., 64) fp32, equal bit for
    bit to WaveColumns._Z(rows). nib = nib_lut(store.lut16). impl: 'auto' (triton on CUDA), 'triton', 'torch'."""
    rows = _rows_check(rows)
    if not _use_triton(rows, impl):
        return _key_Z_torch(key_syn, rows, nib)
    sh = rows.shape
    r2 = rows.reshape(-1, sh[-1]).contiguous()
    NT = r2.size(0)
    out = torch.empty(*sh[:-1], UC, 2, dtype=torch.float32, device=rows.device)
    if NT:
        _kz_kernel[(triton.cdiv(NT, _BT),)](_table(key_syn, 32), r2, nib, out, NT, NF=sh[-1], BT=_BT)
    return out[..., 0], out[..., 1]


def key_Z_keys(key_syn, E, keys, l, fan, nib, impl="auto"):
    """== key_Z(key_syn, WaveColumns._key_rows(E, keys, l), nib) without materialising the rows: E (n,) places,
    keys (n, 32) the level-l codes (any int dtype, may be a strided slice with unit last stride), fan the store's
    (16, 16) fan-in table (fan[c, q] = (2c + q) % 32; only used by the torch path)."""
    if not _use_triton(E, impl):
        k = keys[:, fan]
        c = torch.arange(C, device=E.device)[None, :, None]
        q = torch.arange(F, device=E.device)[None, None, :]
        rows = (((E[:, None, None].long() * L + l) * C + c) * F + q) * UK + k
        return key_Z(key_syn, rows, nib, impl="torch")
    if keys.stride(-1) != 1:
        keys = keys.contiguous()
    E = E.contiguous()
    n = E.numel()
    out = torch.empty(n, C, UC, 2, dtype=torch.float32, device=E.device)
    NT = n * C
    if NT:
        _kzk_kernel[(triton.cdiv(NT, _BT),)](_table(key_syn, 32), E, keys, nib, out, NT, int(l), keys.stride(0),
                                             BT=_BT)
    return out[..., 0], out[..., 1]


def chain_nch(chain_bits, rows, impl="auto"):
    """chain rows (..., W) into chain_bits (uint8, 8-B rows) -> (..., 64) fp32: the number of set bits per target unit
    over the W rows (== WaveColumns._nch(rows); small integers: exact in any order)."""
    rows = _rows_check(rows)
    if not _use_triton(rows, impl):
        return _chain_nch_torch(chain_bits, rows)
    sh = rows.shape
    r2 = rows.reshape(-1, sh[-1]).contiguous()
    NT = r2.size(0)
    out = torch.empty(*sh[:-1], UC, dtype=torch.float32, device=rows.device)
    if NT:
        _nch_kernel[(triton.cdiv(NT, _BT),)](_table(chain_bits, 8), r2, out, NT, NW=sh[-1], BT=_BT)
    return out


# ---------------------------------------------------------------------------------------------------------- the sweep
_GENS = {}


def sweep_randoms(n, step, device, seed):
    """uint8 [n] uniform randoms on `device`, from a per-device generator reseeded on the host with the python int
    seed * 9973 + step (the old sweep's seed); no CPU randoms, no copies, no syncs. Use as the old sweep did: a
    nibble / bit is hit iff u.long() < torch.round(256 * q).long() (q a device tensor: no sync)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    k = (device.type, device.index)
    g = _GENS.get(k)
    if g is None:
        g = _GENS[k] = torch.Generator(device=device)
    g.manual_seed(int(seed) * 9973 + int(step))
    return torch.randint(0, 256, (int(n),), generator=g, device=device, dtype=torch.uint8)


def sweep_threshold(q):
    """the hit threshold of probability q (a tensor): round(256 q) as int16 (compare with u.to(torch.int16) < thr)."""
    return torch.round(256 * q).to(torch.int16)
