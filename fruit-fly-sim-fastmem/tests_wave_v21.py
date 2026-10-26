"""P1 tests of WaveSheet v2.1 (wave_v21.py; SPEC_v21_final.md PART 5) owned by the store / write-rule build: U25, U28,
U34 (+ the shared expert's gate). CPU, 2 threads; the tiny preset (P_g 16, 8 routed + shared, 4 places per strip).

    set CUDA_VISIBLE_DEVICES=-1 & python -B tests_wave_v21.py [--full 1]      (--full: U34 at full size, ~2.5 GB RAM)
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
from wave_columns import F as NF, L as NL, UK, UC, W as NW  # noqa: E402
from wave_v21 import WaveV21Memory  # noqa: E402

KB16, CB16 = NL * 16 * NF * UK, (NL - 1) * 16 * NW * UC      # (key / chain rows per place at 16 columns)


def kbcb(m):
    """key / chain rows per place of m's store (its column count: --wave-cols)."""
    return NL * m.store.C * NF * UK, (NL - 1) * m.store.C * NW * UC
B, TT = 4, 64


def tiny(**kw):
    torch.manual_seed(0)
    c = dict(wave_side=4, wave_warm=2, wave_anneal=10, wave_write_start=4, wave_upd_rows=512, wave_null_every=1,
             wave_drop=0.0)
    if os.environ.get("WAVE_TEST_COLS"):              # (rerun the suite at another column count)
        c["wave_cols"] = int(os.environ["WAVE_TEST_COLS"])
    c.update(kw)
    return WaveV21Memory(32, 8, **c)


def warm(m, steps=8, seed=0):
    for s in range(steps):
        T._micro(m, [T._clusters(B * TT, 32, seed + 10 * s + j) for j in range(6)], B, TT)


def changed_places(before, m):
    st = m.store
    KB, CB = kbcb(m)
    kd = (before[0] != st.key_syn).view(st.P, KB, -1).any(-1).any(-1)
    cd = (before[1] != st.chain_bits).view(st.P, CB, -1).any(-1).any(-1)
    return (kd | cd).nonzero().squeeze(-1).tolist()


def run(check, full=False):
    # (U25) write only where read
    m = tiny()
    warm(m)
    st = m.store
    before = (st.key_syn.clone(), st.chain_bits.clone())
    seen = {}
    real = m._writes

    def spy(rec, *a):
        seen["rec"] = rec
        return real(rec, *a)
    m._writes = spy
    T._micro(m, [T._clusters(B * TT, 32, 500 + j) for j in range(6)], B, TT, step=False)
    m._writes = real
    rec = seen["rec"]
    dec = rec["dec"]
    ch = changed_places(before, m)
    N = rec["xn"].numel()
    gl = set(rec["codes"].loc[dec["gused"]].flatten().tolist())                # (places of tokens that read global)
    vp_d = rec["vp_d"]
    pr_places = set(rec["place_p"][vp_d[:, :6]].tolist())                     # (places of privately read pairs)
    bad = [p for p in ch if (p < m.Pg and p not in gl) or (m.Pg <= p < m.DUMMY and p not in pr_places)
           or p == m.DUMMY]
    sh_priv = bool((dec["vp"] & (dec["S"] % 7 == 6)).any())
    sh_glob = bool((dec["vg"] & (dec["S"] % 7 == 6)).any())
    check("(U25) writes only at places that were read: global places of global readers, strip places of privately "
          "read pairs, never DUMMY; the shared pair (k = 6) never reads privately and reads global under its own gate",
          ch and not bad and not sh_priv and sh_glob,
          f"changed places {len(ch)} (global {sum(p < m.Pg for p in ch)}, strips {sum(p >= m.Pg for p in ch)}); "
          f"outside the read set {bad}; shared private {sh_priv}, shared global {sh_glob}")
    m.after_step(0)
    # keep = 0 for every sequence -> no write, no confirmation
    m.drop = 1.0
    before = (st.key_syn.clone(), st.chain_bits.clone())
    c0 = dict(m.cnt)
    T._micro(m, [T._clusters(B * TT, 32, 600 + j) for j in range(6)], B, TT, step=False)
    ch = [p for p in changed_places(before, m)]
    check("(U25) dropped sequences (keep = 0): no write, no confirmation",
          not ch and m.cnt["writes_g"] == c0["writes_g"] and m.cnt["writes_p"] == c0["writes_p"]
          and m.cnt["confirms_g"] == c0["confirms_g"] and m.cnt["confirms_p"] == c0["confirms_p"],
          f"changed {ch[:5]}")
    m.after_step(0)
    m.drop = 0.0
    # no read -> nothing: every gate closed, no exploration
    with torch.no_grad():
        m.bp_b.fill_(-4.0)
        m.bg_b.fill_(-4.0)
    m.explore = 0.0
    before = (st.key_syn.clone(), st.chain_bits.clone())
    T._micro(m, [T._clusters(B * TT, 32, 700 + j) for j in range(6)], B, TT, step=False)
    rec = m._cache
    check("(U25) no read -> no write (gates closed, no exploration)",
          not changed_places(before, m) and int(rec["dec"]["vp"].sum()) == 0 and int(rec["dec"]["gvalid"].sum()) == 0)
    m.after_step(0)

    # (U28) split: children copy the parent (strip bytes, load, P_e, Pinit, Wld_p, D, R / Cv / n_d, gate rows)
    m = tiny(wave_max_experts=12)
    warm(m)
    with torch.no_grad():
        m.D[2].normal_()
    E0 = m.n_live
    m.on_grow([2, 2], E0)
    st = m.store
    KB, CB = kbcb(m)
    ok = True
    for child in (E0, E0 + 1):
        for j in range(m.J):
            ps, pc = m.Pg + m.J * 2 + j, m.Pg + m.J * child + j
            ok &= torch.equal(st.key_syn[ps * KB:(ps + 1) * KB], st.key_syn[pc * KB:(pc + 1) * KB])
            ok &= torch.equal(st.chain_bits[ps * CB:(ps + 1) * CB], st.chain_bits[pc * CB:(pc + 1) * CB])
            ok &= bool(st.load[ps] == st.load[pc])
        for t in (m.P_e, m.Pinit, m.Wld_p, m.D, m.R, m.Cv, m.n_d, m.Wp, m.Wg, m.bp_b, m.bg_b):
            ok &= torch.equal(t[2], t[child])
    check("(U28) split: both children copy the parent's strip (key + chain bytes, load), P_e, Pinit, Wld_p, D, R, Cv, "
          "n_d and gate rows", ok and m.n_live == E0 + 2 and bool(st.load[m.Pg + m.J * 2:m.Pg + m.J * 3].sum() > 0))
    warm(m, steps=2, seed=900)
    check("(U28) ... the grown module keeps running", m.n_live == E0 + 2)

    # (U23 core) the memory-side losses reach no thought / brain gradient; D is a buffer (never .grad)
    import copy
    m = tiny()
    warm(m)
    m._cache = None
    m2 = copy.deepcopy(m)
    xs = [T._clusters(B * TT, 32, 800 + j) for j in range(6)]
    la = T._micro(m, xs, B, TT, step=False)
    m2._gate_loss = lambda *a_, **k_: None                      # (the same micro-batch without the gate loss)
    lb = T._micro(m2, xs, B, TT, step=False)
    rec = m._cache
    same = all(torch.equal(a_.grad, b_.grad) for a_, b_ in zip(la, lb))
    det = all(not t.requires_grad and t.grad_fn is None for t in rec["h_in"])
    gate_g = any(p_.grad is not None and float(p_.grad.abs().sum()) > 0 for p_ in m.gate_parameters())
    rows_read = m.row_of[rec["eid"][:, :6].clamp(min=0)].view(-1)[(rec["dec"]["S"][rec["dec"]["vp"]] // 7) * 6
                                                                  + (rec["dec"]["S"][rec["dec"]["vp"]] % 7).clamp(max=5)]
    dg = float(m.D_grad[rows_read.unique()].abs().sum()) if rows_read.numel() else 0.0
    check("(U23) the gate loss: inputs detached (no path to xf), thought grads identical with / without it, the gate "
          "params get grads; D a buffer with no .grad, D_grad nonzero for read rows",
          same and det and gate_g and m.D.grad is None and not isinstance(m.D, torch.nn.Parameter) and dg > 0,
          f"thought grads identical {same}; gate grads {gate_g}; D_grad |sum| over read rows {dg:.3e}")
    m.after_step(0)
    # (U22) unread pairs exactly 0 at passes 3-5
    m = tiny(wave_read_pass=3)
    warm(m)
    outs = []
    real_call = m.__call__

    E, Nt = m.n_live, B * TT
    e_k, p_k, t_k, slot_k, counts, C_ = T._route(Nt, E)
    m.train()
    m.train_forward = True
    m.begin_forward()
    for k in range(6):
        xf = T._clusters(Nt, 32, 990 + k).requires_grad_(True)
        b0 = xf.new_zeros(E, C_, 32).index_put((e_k, p_k), xf[t_k])
        bo, xo = m(xf, b0, xf.view(B, TT, 32), None, e_k, p_k, t_k, slot_k, counts, Nt, E, C_, e_k.numel(), call=k)
        outs.append(((bo - b0)[e_k, p_k].detach(), (xo - xf.view(B, TT, 32)).reshape(Nt, 32).detach()))
    dec = m._cache["dec"]
    rd = torch.zeros(7 * Nt + 1, dtype=torch.bool).index_put((dec["S"],), dec["vp"] | dec["vg"])[:7 * Nt].view(Nt, 7)
    pr = rd[t_k, slot_k.clamp(max=6)] & (slot_k < 6)
    z_unread = all(float(o_[0][~pr].abs().max()) == 0.0 and float(o_[1][~rd[:, 6]].abs().max()) == 0.0
                   for o_ in outs[3:])
    nz = sum(float(o_[0][pr].norm()) + float(o_[1][rd[:, 6]].norm()) for o_ in outs[3:])
    pre = all(float(o_[0].abs().max()) == 0 and float(o_[1].abs().max()) == 0 for o_ in outs[:3])
    check("(U22) unread pairs exactly 0 at passes 3-5 (routed and shared); nothing before pass 3; read pairs nonzero",
          z_unread and pre and nz > 0, f"read pairs {int(pr.sum())} routed + {int(rd[:, 6].sum())} shared; inject norm "
          f"{nz:.3e}")
    m._cache = None

    # (U34) bytes
    if full:
        mf = WaveV21Memory(384, 768, wave_cols=16, wave_latent=0, wave_readout="raw")     # (the v2.1 layout)
        st = mf.store
        stor = sum(b.numel() for b in (st.key_syn, st.chain_bits, st.key_pad, st.chain_pad))
        rep = api.byte_budget(mf)
        root = Path(tempfile.mkdtemp(prefix="v21_u34_", dir=os.environ.get("V21_TMP") or None))

        class _Fake(torch.nn.Module):
            def __init__(self, bm):
                super().__init__()
                self.bm, self.cfg = bm, dict(d=384)
                self.blocks = [type("B", (), dict(mlp=type("M", (), dict(E=768))()))()]
        api.save_memory_file(root / "m.memory.pt", _Fake(mf), step=1)
        fsize = (root / "m.memory.pt").stat().st_size
        bb = rep["persistent"] + rep["params"]
        check("(U34) storage exactly 1,474,412,672 B; the memory file within 1% + 64 KB of byte_budget()",
              stor == 1_474_412_672 and abs(fsize - bb) <= 0.01 * bb + 65536,
              f"storage {stor:,} B; DUMMY {mf.DUMMY}; file {fsize:,} B vs budget {bb:,} B; {api.budget_line(rep)}")
        (root / "m.memory.pt").replace(root / "m.memory.pt.done")


if __name__ == "__main__":
    R = []

    def check(name, ok, detail=""):
        R.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    run(check, full="--full" in sys.argv)
    print("ALL PASS" if all(r[1] for r in R) else f"{sum(not r[1] for r in R)} FAIL of {len(R)}")
