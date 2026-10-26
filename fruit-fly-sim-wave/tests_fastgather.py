"""Tests for wave_fastgather.py: bit-exactness vs the live WaveColumns._Z / _nch (CPU and GPU), no host syncs, and
GPU timing at the full-size layout (3,333 places; 8,192 tokens x 4 entries x 3 layers, read_chunk 2048).

    python -B tests_fastgather.py [--no-gpu] [--no-bench]
"""
import math
import sys
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import wave_fastgather as fg  # noqa: E402
from wave_columns import WaveColumns, L, C, F, UK, UC, PHI, W  # noqa: E402

FAILS = []


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def luts(device):
    """the store's derived tables, built exactly as WaveColumns.__init__ builds them (on CPU, then moved)."""
    w16 = torch.arange(65536)
    s4 = (w16[:, None] >> (4 * torch.arange(4))) & 15
    a4 = torch.remainder(s4 - 1, PHI).float() * (2 * math.pi / PHI)
    lut16 = torch.stack([a4.cos() * (s4 > 0), a4.sin() * (s4 > 0)], -1).reshape(65536, 8)
    fan = (2 * torch.arange(C)[:, None] + torch.arange(F)[None]) % 32
    bits = 1 << torch.arange(8, dtype=torch.uint8)
    return lut16.to(device), fan.to(device), bits.to(device)


def fake_store(P, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    lut16, fan, bits = luts(device)
    ks = torch.randint(0, 256, (P * L * C * F * UK, UC // 2), generator=g, dtype=torch.uint8).to(device)
    cb = torch.randint(0, 256, (P * (L - 1) * C * W * UC, UC // 8), generator=g, dtype=torch.uint8).to(device)
    return types.SimpleNamespace(key_syn=ks, chain_bits=cb, lut16=lut16, fan=fan, bits=bits, P=P)


def biteq(a, b):
    return a.shape == b.shape and bool((a.view(torch.int32) == b.view(torch.int32)).all())


def test_exact(device, impls):
    st = fake_store(37, device, seed=1)
    nib = fg.nib_lut(st.lut16)
    s4 = (torch.arange(65536, device=device)[:, None] >> (4 * torch.arange(4, device=device))) & 15
    check(f"[{device}] LUT16 == nib_lut per nibble (bitwise)", biteq(st.lut16.view(65536, 4, 2), nib[s4]))
    g = torch.Generator().manual_seed(2)
    n = 300
    E = torch.randint(0, st.P, (n,), generator=g).to(device)
    keys = torch.randint(0, UK, (n, 96), generator=g).to(device)
    for l in range(L):
        rows = WaveColumns._key_rows(st, E, keys[:, l * 32:(l + 1) * 32], l)
        zr0, zi0 = WaveColumns._Z(st, rows)
        for impl in impls:
            zr, zi = fg.key_Z(st.key_syn, rows, nib, impl=impl)
            check(f"[{device}] key_Z {impl} l={l} bit-exact", biteq(zr.contiguous(), zr0.contiguous())
                  and bitieq(zi, zi0))
            zr, zi = fg.key_Z(st.key_syn, rows.int(), nib, impl=impl)
            check(f"[{device}] key_Z {impl} int32 rows l={l} bit-exact", bitieq(zr, zr0) and bitieq(zi, zi0))
            zr, zi = fg.key_Z_keys(st.key_syn, E, keys[:, l * 32:(l + 1) * 32], l, st.fan, nib, impl=impl)
            check(f"[{device}] key_Z_keys {impl} l={l} bit-exact", bitieq(zr, zr0) and bitieq(zi, zi0))
        if l:
            v = torch.randint(0, UC, (n, C), generator=g).to(device)
            cr = WaveColumns._chain_rows(st, E, v, l)
            n0 = WaveColumns._nch(st, cr)
            for impl in impls:
                check(f"[{device}] chain_nch {impl} l={l} exact", bitieq(fg.chain_nch(st.chain_bits, cr, impl=impl), n0))
    w = fg.gather_rows(st.key_syn, rows[:5])
    check(f"[{device}] gather_rows 32-B == key_syn[rows]", bool((w.view(torch.uint8).view(*rows[:5].shape, 32)
                                                                   == st.key_syn[rows[:5]]).all()))
    w8 = fg.gather_rows(st.chain_bits, cr[:5], 8, torch.int64)
    check(f"[{device}] gather_rows 8-B == chain_bits[rows]", bool((w8.view(torch.uint8).view(*cr[:5].shape, 8)
                                                                  == st.chain_bits[cr[:5]]).all()))


def bitieq(a, b):
    return bitieq_(a.contiguous(), b.contiguous())


def bitieq_(a, b):
    return a.shape == b.shape and bool((a.view(torch.int32) == b.view(torch.int32)).all())


bitieq.__doc__ = "bitwise fp32 equality (-0 vs +0 counts)"


def test_nosync(device):
    st = fake_store(9, device, seed=3)
    nib = fg.nib_lut(st.lut16)
    E = torch.randint(0, st.P, (64,), device=device)
    keys = torch.randint(0, UK, (64, 96), device=device)
    rows = WaveColumns._key_rows(st, E, keys[:, :32], 0)
    cr = WaveColumns._chain_rows(st, E, torch.randint(0, UC, (64, C), device=device), 1)
    fg.key_Z(st.key_syn, rows, nib); fg.key_Z_keys(st.key_syn, E, keys[:, :32], 0, st.fan, nib)  # noqa: E702
    fg.chain_nch(st.chain_bits, cr); fg.sweep_randoms(1000, 1, device, 5)                       # (compile first)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fg.key_Z(st.key_syn, rows, nib)
        fg.key_Z_keys(st.key_syn, E, keys[:, :32], 0, st.fan, nib)
        fg.chain_nch(st.chain_bits, cr)
        fg.gather_rows(st.key_syn, rows)
        fg.gather_rows(st.chain_bits, cr, 8, torch.int64)
        u = fg.sweep_randoms(4096 * 64, 7, device, 5)
        q = torch.rand((), device=device)
        hit = u.to(torch.int16) < fg.sweep_threshold(q)
        ok, err = True, ""
    except RuntimeError as e:
        ok, err = False, str(e)[:200]
    finally:
        torch.cuda.set_sync_debug_mode(0)
    check(f"[{device}] no host syncs (set_sync_debug_mode('error'))", ok, err)
    del hit


def test_sweep(device):
    a = fg.sweep_randoms(1 << 20, 3, device, 1234)
    b = fg.sweep_randoms(1 << 20, 3, device, 1234)
    c = fg.sweep_randoms(1 << 20, 4, device, 1234)
    m = a.float().mean().item()
    q = torch.tensor(0.3, device=device)
    rate = (a.long() < torch.round(256 * q).long()).float().mean().item()
    check(f"[{device}] sweep_randoms: uint8 on device, reproducible per (seed, step), uniform",
          a.dtype == torch.uint8 and a.device.type == torch.device(device).type and bool((a == b).all())
          and not bool((a == c).all()) and abs(m - 127.5) < 0.5 and abs(rate - 77 / 256) < 2e-3,
          f"mean {m:.2f}; P(u < round(256 .3)) {rate:.4f} vs {77 / 256:.4f}")


def bench(device):
    P = 256 + 769 * 4                                          # 3,333 places
    N, ENT, CH = 8192, 4, 2048
    print(f"\nbench: {P} places, key_syn {P * 393216 / 1e9:.2f} GB, chain {P * 49152 / 1e6:.0f} MB; "
          f"{N} tokens x {ENT} entries x {L} layers, read_chunk {CH}", flush=True)
    lut16, fan, bits = luts(device)
    st = types.SimpleNamespace(lut16=lut16, fan=fan, bits=bits, P=P,
                               key_syn=torch.randint(0, 256, (P * L * C * F * UK, UC // 2), dtype=torch.uint8,
                                                     device=device),
                               chain_bits=torch.randint(0, 256, (P * (L - 1) * C * W * UC, UC // 8),
                                                        dtype=torch.uint8, device=device))
    nib = fg.nib_lut(lut16)
    E = torch.randint(0, P, (CH,), device=device)
    keys = torch.randint(0, UK, (CH, 96), device=device)
    v = torch.randint(0, UC, (CH, C), device=device)
    rows = WaveColumns._key_rows(st, E, keys[:, :32], 0)
    cr = WaveColumns._chain_rows(st, E, v, 1)

    def tm(fn, reps=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1e3

    calls_key = (N // CH) * ENT * L                            # 48 key calls per recall
    calls_ch = (N // CH) * ENT * (L - 1)                       # 32 chain calls
    r = {}
    r["old key  (_key_rows + _Z)"] = tm(lambda: WaveColumns._Z(st, WaveColumns._key_rows(st, E, keys[:, :32], 0)))
    r["old _Z only"] = tm(lambda: WaveColumns._Z(st, rows))
    r["new key_Z (rows given)"] = tm(lambda: fg.key_Z(st.key_syn, rows, nib))
    r["new key_Z_keys (fused rows)"] = tm(lambda: fg.key_Z_keys(st.key_syn, E, keys[:, :32], 0, fan, nib))
    r["old chain (_chain_rows + _nch)"] = tm(lambda: WaveColumns._nch(st, WaveColumns._chain_rows(st, E, v, 1)))
    r["new chain (_chain_rows + chain_nch)"] = tm(lambda: fg.chain_nch(st.chain_bits, WaveColumns._chain_rows(st, E, v, 1)))
    for k_, t in r.items():
        print(f"  {k_:40s} {t:8.3f} ms / call (n = {CH})")
    old = calls_key * r["old key  (_key_rows + _Z)"] + calls_ch * r["old chain (_chain_rows + _nch)"]
    new = calls_key * r["new key_Z_keys (fused rows)"] + calls_ch * r["new chain (_chain_rows + chain_nch)"]
    print(f"  projected gathers per recall ({calls_key} key + {calls_ch} chain calls): old {old:.1f} ms, new {new:.1f} ms",
          flush=True)
    u_old = lambda: torch.randint(0, 256, (C * F * UK, 32, 2), dtype=torch.uint8).to(device)  # noqa: E731
    u_new = lambda: fg.sweep_randoms(C * F * UK * 64, 1, device, 1234)                        # noqa: E731
    print(f"  sweep randoms per (place, layer) block: old CPU+copy {tm(u_old, 10):.3f} ms, new {tm(u_new, 10):.3f} ms "
          f"(48 blocks per step)", flush=True)


def main():
    torch.set_num_threads(2)
    test_exact("cpu", ["torch"])
    test_sweep("cpu")
    if "--no-gpu" not in sys.argv and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 16e9 / total), torch.device("cuda", 0))
        dev = "cuda:0"
        test_exact(dev, ["torch", "triton"] if fg._HAS_TRITON else ["torch"])
        test_sweep(dev)
        test_nosync(dev)
        if "--no-bench" not in sys.argv:
            bench(dev)
    print(f"\n{len(FAILS)} failures" + (": " + "; ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
