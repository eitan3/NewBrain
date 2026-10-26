"""Run-critical CPU tests of wave memory v2.2 (SPEC_v22.md items 2-4; wave_v21.py --wave-write delta, --wave-readout
norm, --wave-strip-places 16). CPU, 2 threads; the tiny preset of tests_wave_v21 (P_g 16, 8 routed + shared).

    set CUDA_VISIBLE_DEVICES=-1 & python -B tests_wave_v22.py [--full 1] [--no-e2e]
        --full: the full-size small-brain memory's bytes (WaveV21Memory(384, 192, strip 16), ~1.8 GB RAM)
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

torch.set_num_threads(2)
import memory_api as api  # noqa: E402
import tests_wave as T  # noqa: E402
import tests_wave_v21 as V  # noqa: E402
from wave_address import Codes  # noqa: E402
from wave_v21 import WaveV21Memory  # noqa: E402

B, TT = V.B, V.TT
STRIP16 = dict(wave_strip=16)


def store_bytes(m):
    st = m.store
    return sum(b.numel() * b.element_size() for b in (st.key_syn, st.chain_bits, st.key_pad, st.chain_pad))


def grads(m, xs, mem):
    return [x.grad.clone() for x in T._micro(m, xs, B, TT, mem=mem, step=False)]


def run(check, full=False, e2e=True):
    # (0) the v2.2 latent dims; the write value's layout
    m = V.tiny()
    V.warm(m)
    lat = m._latent(torch.nn.functional.normalize(torch.randn(5, 32), dim=-1), torch.randn(5, 32), torch.randn(5, 64))
    tr = m.store.trace(m._cache["cg"]) if m._cache is not None else None
    y = m.store.decode(m._cache["trg"]) if m._cache is not None else torch.zeros(1, m.dl)
    check("(0) dims: latent 2048, 64 columns, store decoder [3, 64, 64, 32], g [2048], D [S, 2048, d] at 0, the read y 2048-d, the "
          "write value 2048-d with the correction as its first d dims",
          m.dl == 2048 and tuple(m.store.A.shape) == (3, m.store.C, 64, 2048 // m.store.C) and tuple(m.g_read.shape) == (2048,)
          and tuple(m.D.shape) == (m.S, 2048, 32) and y.size(-1) == 2048 and lat.size(-1) == 2048 and m.conf_soft,
          f"layout [corr 0:{m.d} | xf {m.d}:{2 * m.d} | side {2 * m.d}:{m.n_parts} | fill {m.n_parts}:{m.dl}]; trace "
          f"{tr is not None}")
    m._cache = None

    # (1) off == none, bit-identical: inactive (before write_start) and active with every gate closed (no read)
    for kw in ({}, STRIP16):
        m = V.tiny(**kw)
        xs = [T._clusters(B * TT, 32, 50 + j) for j in range(6)]
        a, b = grads(m, xs, True), grads(m, xs, False)
        ok_in = all(torch.equal(x, y) for x, y in zip(a, b))
        m = V.tiny(**kw)
        V.warm(m)
        with torch.no_grad():
            m.bp_b.fill_(-4.0)
            m.bg_b.fill_(-4.0)
        m.explore = 0.0
        a, b = grads(m, xs, True), grads(m, xs, False)
        ok_cl = all(torch.equal(x, y) for x, y in zip(a, b))
        check(f"(1) off == none bit-identical (thought grads; delta + norm{', strip 16' if kw else ''}): inactive, and "
              f"active with no read", ok_in and ok_cl and m.active, f"inactive {ok_in}, gates closed {ok_cl}")

    # (2) delta toy at one place: A -> v1, then A -> v2 recalls v2 within 2 writes; B (other keys) survives
    m = V.tiny()
    st = m.store
    g = torch.Generator().manual_seed(3)
    n = 2
    keys = torch.randint(0, 16, (n, 96), generator=g)
    place = 5
    cod = Codes(torch.full((n, 1), place), torch.zeros(n, 1), keys[:, :, None], torch.ones(n, 96, 1),
                torch.zeros(n, 96, 1), torch.zeros(n, m.address.n_feats), keys[:, None])
    yA1, yA2, yB = torch.randn(3, m.dl, generator=g)                  # (the latent width: 2048 by default)

    def match(i, y):
        rho = st.rho.clone()
        v, _ = st.encode(y[None])
        st.rho.copy_(rho)
        tr = st.trace(cod.take(torch.tensor([i])))
        return float((tr["v"][0, 0] == v[0]).float().mean())

    def dwrite(idx, ys, erase, delta=True):
        c = cod.take(torch.tensor(idx))
        tr = st.trace(c)
        if delta:
            st.write_delta(c, torch.stack(ys), torch.ones(len(idx)), tr["v"][:, 0], tr["ph"][:, 0],
                           torch.full((len(idx),), erase))
        else:
            st.write(c, torch.stack(ys), torch.ones(len(idx)))
    base = (st.key_syn.clone(), st.chain_bits.clone(), st.load.clone(), st.rho.clone(), st.scale.clone())
    res = {}
    for mode in ("delta", "set"):
        for t_, b_ in zip((st.key_syn, st.chain_bits, st.load, st.rho, st.scale), base):
            t_.copy_(b_)
        dwrite([0, 1], [yA1, yB], False, mode == "delta")
        a1, b0 = match(0, yA1), match(1, yB)
        hist = []
        for _ in range(2):
            dwrite([0], [yA2], True, mode == "delta")
            hist.append((match(0, yA2), match(0, yA1), match(1, yB)))
        res[mode] = (a1, b0, hist)
    a1, b0, hist = res["delta"]
    won = [h[0] > 0.9 and h[0] > h[1] for h in hist]
    check(f"(2) delta toy ({m.dl}-d latent): A -> v2 after A -> v1 recalls v2 within 2 writes (> 90% of columns, above v1); B on the same "
          "place survives >= 90% of its recall", any(won) and hist[-1][2] >= 0.9 * b0,
          f"A v1 recall {a1:.2f}, B {b0:.2f}; after each v2 write (v2, v1, B): "
          f"{[tuple(round(x, 2) for x in h) for h in hist]}; the set rule for contrast: "
          f"{[tuple(round(x, 2) for x in h) for h in res['set'][2]]}")

    # (3) a zero read gives zero output; s / g get gradients
    m = V.tiny()
    V.warm(m)
    with torch.no_grad():
        m.store.key_syn.zero_()
        m.store.chain_bits.zero_()
        m.bp_b.fill_(4.0)
        m.bg_b.fill_(4.0)
    E, Nt = m.n_live, B * TT
    e_k, p_k, t_k, slot_k, counts, C_ = T._route(Nt, E)
    m.train()
    m.train_forward = True
    m.begin_forward()
    mx = 0.0
    for k in range(6):
        xf = T._clusters(Nt, 32, 990 + k).requires_grad_(True)
        b0_ = xf.new_zeros(E, C_, 32).index_put((e_k, p_k), xf[t_k])
        bo, xo = m(xf, b0_, xf.view(B, TT, 32), None, e_k, p_k, t_k, slot_k, counts, Nt, E, C_, e_k.numel(), call=k)
        mx = max(mx, float((bo - b0_).detach().abs().max()), float((xo - xf.view(B, TT, 32)).detach().abs().max()))
    nread = int((m._cache["dec"]["vp"] | m._cache["dec"]["vg"]).sum())
    m._cache = None
    m = V.tiny()
    V.warm(m)
    T._micro(m, [T._clusters(B * TT, 32, 300 + j) for j in range(6)], B, TT, step=False)
    gn = {nm: float(p.grad.abs().sum()) if p.grad is not None else 0.0
          for nm, p in (("g", m.g_read), ("w_e", m.w_e), ("b_e", m.b_e))}
    in_opt = all(any(p is q for q in m.readout_parameters()) for p in (m.g_read, m.w_e, m.b_e))
    m.after_step(0)
    stepped = m._opt is not None and all(any(p is q for gr in m._opt.param_groups for q in gr["params"])
                                         for p in m.readout_parameters())
    check("(3) a zero read (empty store, every gate open) gives exactly 0 output; g, w_e, b_e get nonzero gradients "
          "from the main loss and step in the memory's own AdamW", nread > 0 and mx == 0.0 and all(v > 0 for v in
                                                                                                gn.values()) and in_opt and stepped,
          f"pairs read {nread}, max |out| {mx}; |grad| {', '.join(f'{k} {v:.2e}' for k, v in gn.items())}; ratio "
          f"{float(m._ratio):.3f}")

    # (4) storage bytes fixed (tiny strip 16: through writes, splits, sweeps) + the split copies w_e / b_e
    m = V.tiny(wave_max_experts=12, wave_z_conf=0.0, **STRIP16)     # (z_conf 0: the module's erase path runs too)
    b_0 = store_bytes(m)
    kb_, cb_ = V.kbcb(m)
    exp = (m.DUMMY + 1) * (kb_ * 32 + cb_ * 8) + 128
    V.warm(m)
    m.on_grow([2, 3], m.n_live)
    cp = torch.equal(m.w_e[2], m.w_e[8]) and torch.equal(m.b_e[3], m.b_e[9])
    V.warm(m, steps=3, seed=700)
    check("(4) storage bytes fixed (strip 16, writes + split + sweeps), = (DUMMY + 1) x 1,769,472 (64 columns) + 128; the split copies "
          "w_e / b_e", b_0 == store_bytes(m) == exp and m.n_live == 10 and m.cnt["writes_p"] > 0 and cp
          and m.store.stats_acc.get("erased", 0) > 0,
          f"{b_0:,} B -> {store_bytes(m):,} B (expected {exp:,}); DUMMY {m.DUMMY}; Jr {m.Jr}; writes_p "
          f"{m.cnt['writes_p']}; erased {m.store.stats_acc.get('erased', 0)}; w_e/b_e copied {cp}")
    if full:
        mf = WaveV21Memory(384, 192, wave_side=12, wave_strip=3, wave_strip_entries=3)   # (the run's layout)
        rep = api.byte_budget(mf)
        check("(4) full size (384, 192 experts, 64 columns, global 12 x 12, strips of 3)", True,
              f"places {mf.DUMMY + 1} ({mf.Pg} + {mf.S} x {mf.J} + DUMMY); storage {store_bytes(mf):,} B; D "
              f"{tuple(mf.D.shape)}; {api.budget_line(rep)}; D_grad {mf.D_grad.numel() * 4:,} B (not saved)")
        del mf

    # (6) writes per sequence: the top 8, >= 8 tokens apart, only eligible (read, kept, unconfirmed) tokens
    from wave_write import SpecPolicy
    pol = SpecPolicy(write_w=16, per_seq=8)
    Bp, Tp = 3, 1024
    g = torch.Generator().manual_seed(11)
    ce, H = torch.rand(Bp * Tp, generator=g) * 5, torch.rand(Bp * Tp, generator=g) * 3
    allow = torch.rand(Bp * Tp, generator=g) < 0.5
    tok, s_ = pol.select(ce, H, 50, Bp, Tp, allow=allow)
    tok = tok[s_ > 0]
    per = torch.bincount(tok // Tp, minlength=Bp)
    gaps = [int((t_.sort().values.diff()).min()) for t_ in (tok[tok // Tp == b] for b in range(Bp))]
    p1 = SpecPolicy(write_w=16, per_seq=1).select(ce, H, 50, Bp, Tp, allow=allow)[0]
    m = V.tiny(wave_writes_per_seq=8)
    V.warm(m)
    st = m.store
    before = (st.key_syn.clone(), st.chain_bits.clone())
    seen = {}
    real = m.policy.select

    def spy(*a, **k):
        out = real(*a, **k)
        seen["sel"], seen["allow"] = out, k.get("allow")
        return out
    m.policy.select = spy
    T._micro(m, [T._clusters(B * TT, 32, 500 + j) for j in range(6)], B, TT, step=False)
    m.policy.select = real
    rec, (wt, ws) = m._cache, seen["sel"]
    wt = wt[ws > 0]
    ch = V.changed_places(before, m)
    gl = set(rec["codes"].loc[rec["dec"]["gused"]].flatten().tolist())
    prp = set(rec["place_p"][rec["vp_d"][:, :6]].tolist())
    bad = [p for p in ch if (p < m.Pg and p not in gl) or (m.Pg <= p < m.DUMMY and p not in prp) or p == m.DUMMY]
    pm = torch.bincount(wt // TT, minlength=B)
    gm = [int(t_.sort().values.diff().min()) for t_ in (wt[wt // TT == b] for b in range(B)) if t_.numel() > 1]
    m.after_step(0)
    check("(6) writes per sequence: 8 per sequence, gaps >= 8, only allowed tokens (policy, T 1,024); in the module "
          "<= 8 per sequence, gaps >= 8, all written tokens read + kept + unconfirmed, writes only at read places; "
          "per_seq 1 == the v2.1 pick", per.tolist() == [8] * Bp and min(gaps) >= 8 and bool(allow[tok].all())
          and int(pm.max()) <= 8 and int(pm.max()) > 1 and all(x >= 8 for x in gm)
          and bool(seen["allow"][wt].all()) and ch and not bad and p1.numel() == Bp,
          f"policy per seq {per.tolist()}, min gaps {gaps}; module per seq {pm.tolist()}, min gaps {gm}, changed "
          f"places {len(ch)}, outside the read set {bad}")

    # (5) the small config end to end: train_lm tiny steps with births by split (experts 3 -> 12, top-6, strip 16)
    if e2e:
        import json
        import tests_memory_api as MA
        from test_memory import _tiny_data
        root = Path(tempfile.mkdtemp(prefix="v22_e2e_")).resolve()
        _tiny_data(root / "data")
        args = ("--moe 24,6,16,16,auxfree --start heads=4,experts=6 --memory wave --wave-mode v21 --wave-side 4 "
                "--wave-warm 2 --wave-anneal 10 --wave-fast 10 --wave-upd-rows 512 --wave-write-start 4 "
                "--wave-null-every 1 --wave-strip-places 16 --wave-write delta --wave-readout norm --memory-save 1 "
                "--concepts 2,4 --chunk-ce 512 --ckpt-route 1 --diag-every 5 --keep-ckpts 0").split()
        losses, rdir, _ = MA.run(root, "small", HERE, args, stop=16)
        ck = MA._load(rdir / "last.pt")
        mem = ck["state"] if "state" in ck else {}
        we = [k for k in mem if k.endswith("w_e")]
        ex = mem["bm._extra_state"]
        cnt = ex["cnt"]
        check("(5) small config: train_lm runs tiny CPU steps (experts 6 -> 24 by split, top-6, strip 16, delta + norm, "
              "8 writes / seq): finite losses, memory reads + writes after the splits",
              len(losses) >= 10 and all(v == v for v in losses.values()) and ex["n_live"] > 6 and cnt["writes_p"] > 0
              and cnt["writes_g"] > 0 and bool(we) and tuple(mem["bm.D"].shape[:1]) == (25,),
              f"steps {len(losses)}, last loss {list(losses.values())[-1]:.3f}; experts {ex['n_live']}; recalls "
              f"{cnt['recalls']}, writes g/p {cnt['writes_g']}/{cnt['writes_p']}, confirms {cnt['confirms_g']}/"
              f"{cnt['confirms_p']}; D {tuple(mem['bm.D'].shape)}; run dir {rdir}")


if __name__ == "__main__":
    R = []

    def check(name, ok, detail=""):
        R.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    run(check, full="--full" in sys.argv, e2e="--no-e2e" not in sys.argv)
    print("ALL PASS" if all(r[1] for r in R) else f"{sum(not r[1] for r in R)} FAIL of {len(R)}")
