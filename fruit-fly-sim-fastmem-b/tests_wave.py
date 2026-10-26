"""Tests of the wave memory (--memory wave --wave-mode on: wave_address.py, wave_store.py, wave_write.py, wave_memory.py).
CPU only, tiny configs.

    set CUDA_VISIBLE_DEVICES=-1 & python -B tests_wave.py [work_dir]

unit (in-process):
  (u1) the address is deterministic under its seed (buffers and codes), another seed differs
  (u2) the address takes no gradient: encode / update on a thought that requires grad build no graph
  (u3) coarse-to-fine: each residual level lowers the reconstruction error of the thought
  (u4) homeostasis: on skewed thoughts the entry map's read usage stays spread, no dead codes
  (u5) no gradient into the thoughts: the read has no path to xf (d xf.grad is the same with the memory as without),
       while the gate and the gains get the main loss's gradient
  (u6) one write per selected token (never per pass x expert): 3 passes x 2 experts, n selected -> n global writes
  (u7) the byte budget == memory.pt's tensors; the cap check; the full-size default (d 384, 769 experts) < 1.9 GB
  (u8) storage and write policy are pluggable: a registered store / policy gets strength, entries, confirm, forget
  (u9) the keep mask (--wave-drop-mode seq): deterministic, about the rate of SEQUENCES, never in eval; dropped
       sequences read exactly 0 while the capture leaf still gets their gradient; forward mode as before
  (u10) train-only at module level (private parts): only the selected private parts and gain rows change
  (u11) train-only, global column store (the default, no private parts): owned = the top owner_use places (never an
       empty one); only owned places' synapses / chain bits, gate row e and counters change; hints of tokens whose
       entries are all outside owned are bit-identical; the sheet store without private parts: a clear error
end to end (train_lm subprocesses, tests_memory_api's driver):
  (e1) a tiny training run: finite losses, writes, the address ready, the wmem_* log
  (e2) determinism: the same run twice == bit for bit (losses + checkpoint)
  (e3) stop at K + resume == the uninterrupted run (losses + checkpoint + memory)
  (e4) memory.pt save / reset / load -> identical outputs; memory off == no module; eval leaves the memory untouched
  (e5) swap: a --memory none checkpoint resumed with --memory wave (the brain kept, the memory fresh)
  (e6) --memory-train-only experts=1,shared: the brain, the address, the global store, the gate are bit-identical;
       the other experts' private parts too; the selected parts learned
  (e7) the same with the default global column store (no private parts): the brain bit-identical, only bm.gate_e
       trainable, the store changes only at owned places
schedule: (r1) passes 0-2 untouched, one recall at pass 3, re-applied at 3-5; (r2) the memory's own AdamW; (r3) the
  spec write policy; (r4) the null control
columns: (c1)-(c5) the store; (c6) entry 1 reads with its own residual keys
The blob's default-flags identity is tests_memory_api.py (a); run it too.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import math
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

torch.set_num_threads(2)
import memory_api as api  # noqa: E402
import tests_memory_api as T  # noqa: E402
from wave_address import WaveAddress  # noqa: E402
from wave_memory import WAVE_DEFAULTS, CodedWaveMemory  # noqa: E402
from wave_store import STORES, HebbianSheet  # noqa: E402
from wave_write import POLICIES, EntropySurprise  # noqa: E402

N, K = T.N, T.K
TINY_SHEET = "--wave-side 4 --wave-slices 32 --wave-codes 16 --wave-priv-levels 1"    # (the column store, P = 16)
COLS = dict(wave_store="columns", wave_slices=32, wave_codes=16, wave_priv_levels=0)   # (the default store, tiny)
WAVE = ("--memory wave " + TINY_SHEET + " --wave-warm 4 --wave-anneal 20 --wave-fast 10 --wave-content-fast 10 "
        "--wave-read-pass 1 --wave-gate-warm 4 --wave-upd-rows 512 --wave-write-w 8 --concepts 2,4 --think-w 0.1 "
        "--chunk-ce 512 --ckpt-route 1 --diag-every 5 --keep-ckpts 0 --wave-write-start 6").split()
SMALL = dict(wave_store="sheet", wave_priv_levels=1, wave_warm=2, wave_side=4, wave_slices=16, wave_codes=8,
             wave_sheet_side=16, wave_sheet_units=8, wave_anneal=10, wave_fast=0, wave_content_fast=8, wave_gate_warm=3, wave_upd_rows=512, wave_drop=0.0,
             wave_explore=0.0, wave_write_w=6, wave_read_pass=1, wave_write="wave", wave_write_per_seq=4,
             wave_priv_write="top1", wave_read_chunk=512, wave_null_every=1, wave_gate_mode="post")
PRE = dict(wave_store="columns", wave_slices=32, wave_codes=16, wave_priv_levels=1, wave_gate_mode="pre",
           wave_explore=0.05)                                                   # (the run's default: pre gate, private)


# ====================================================================== helpers
def _clusters(n, d, seed, k=12, skew=False):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(k, d, generator=g) * 3
    p = torch.full((k,), 1.0 / k)
    if skew:
        p = torch.tensor([0.6] + [0.4 / (k - 1)] * (k - 1))
    lab = torch.multinomial(p, n, replacement=True, generator=g)
    return C[lab] + torch.randn(n, d, generator=g)


def _route(Ntok, E, k=2):
    """k experts per token (slot 0 = its top-1), the capacity layout (e_k, p_k, t_k, slot_k)."""
    t = torch.arange(Ntok)
    e = torch.stack([(t + j) % E for j in range(k)], 1)          # (N, k)
    e_k, t_k = e.flatten(), t[:, None].expand(-1, k).flatten()
    slot_k = torch.arange(k).repeat(Ntok)
    order = e_k.argsort(stable=True)
    e_k, t_k, slot_k = e_k[order], t_k[order], slot_k[order]
    counts = torch.bincount(e_k, minlength=E)
    p_k = torch.arange(e_k.numel()) - (counts.cumsum(0) - counts)[e_k]
    return e_k, p_k, t_k, slot_k, counts, int(counts.max())


def _micro(m, xs, B, Tt, sel_ce=None, mem=True, wbuf=None, step=True):
    """One simulated micro-batch: begin_forward, len(xs) passes through the module's RoutedMLP hook, a loss on the
    injected inputs, select_writes, backward, after_backward. Returns the thoughts (with their .grad)."""
    E, D = m.n_live, xs[0].size(1)
    Ntok = B * Tt
    e_k, p_k, t_k, slot_k, counts, C = _route(Ntok, E)
    m.train()
    m.train_forward = True
    m.begin_forward()
    loss, leaves = 0.0, []
    for k, x0 in enumerate(xs):
        xf = x0.clone().requires_grad_(True)
        leaves.append(xf)
        buf = xf.new_zeros(E, C, D)
        buf = buf.index_put((e_k, p_k), xf[t_k])
        x = xf.view(B, Tt, D)
        if mem:
            buf, xs_ = m(xf, buf, x, None, e_k, p_k, t_k, slot_k, counts, Ntok, E, C, e_k.numel(), call=k)
        else:
            xs_ = x
        w = torch.linspace(-1, 1, D) * (k + 1)
        loss = loss + (buf * w).sum() + (xs_ * w.flip(0)).sum() * 0.5
    g = torch.Generator().manual_seed(7)
    ce = sel_ce if sel_ce is not None else torch.rand(Ntok, generator=g) * 5
    H = torch.rand(Ntok, generator=g) * 3
    if mem:
        m.select_writes(None, torch.zeros(B, Tt, dtype=torch.long), None, ce_h=(ce, H, 50))
    loss.backward()
    if mem:
        m.after_backward()
        if step:
            m.after_step(0)
    return leaves


def _module(**kw):
    torch.manual_seed(0)
    m = CodedWaveMemory(32, 4, **dict(SMALL, **kw))
    return m


def _warm(m, steps=4, B=4, Tt=64, seed=0, passes=3):
    for s in range(steps):
        _micro(m, [_clusters(B * Tt, 32, seed + 10 * s + j) for j in range(passes)], B, Tt)


def _sd(m):
    return {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in m.state_dict().items()}


# ====================================================================== unit tests
def unit_tests(check):
    # (u1) determinism
    X = [_clusters(1024, 32, s) for s in range(6)]
    a1, a2, a3 = WaveAddress(32, side=4, slices=8, seed=5, warm=2), WaveAddress(32, side=4, slices=8, seed=5, warm=2), \
        WaveAddress(32, side=4, slices=8, seed=6, warm=2)
    for x in X:
        a1.update(x), a2.update(x), a3.update(x)
    same = all(torch.equal(a, b) for a, b in zip(a1.state_dict().values(), a2.state_dict().values()))
    c1, c2 = a1.encode(X[0]), a2.encode(X[0])
    check("(u1) address deterministic under its seed (state + codes)", same and torch.equal(c1.code, c2.code)
          and torch.equal(c1.loc, c2.loc) and a1.ready)
    check("(u1) ... another seed gives another map", not torch.equal(a1.W, a3.W))
    # (u2) no gradient
    xg = X[1].clone().requires_grad_(True)
    c = a1.encode(xg)
    a1.update(xg)
    check("(u2) address: encode / update build no graph on a thought that requires grad",
          not any(t.requires_grad for t in (c.loc_s, c.code_w, c.code_d, c.feats)) and xg.grad is None
          and not any(b.requires_grad for b in a1.buffers()))
    # (u3) coarse to fine
    Xh = a1.standardize(X[2])
    c = a1.encode(X[2])
    rec = a1.W[c.loc[:, 0]].clone()
    errs = [float((Xh - rec).pow(2).sum(-1).mean())]
    S, ds = a1.S, a1.ds
    for lev in range(a1.levels):
        part = a1.books[lev][torch.arange(S)[None], c.code[:, lev * S:(lev + 1) * S, 0]]   # (n, S, ds)
        rec = rec + part.reshape(-1, S * ds)
        errs.append(float((Xh - rec).pow(2).sum(-1).mean()))
    check("(u3) each residual level lowers the reconstruction error", all(b < a for a, b in zip(errs, errs[1:])),
          "error after location / levels: " + " ".join(f"{e:.2f}" for e in errs))
    # (u4) homeostasis on skewed thoughts
    a4 = WaveAddress(32, side=4, slices=8, seed=1, warm=2, anneal=20)
    for s in range(40):
        a4.update(_clusters(1024, 32, 100 + s, skew=True))
    st = a4.stats()
    check("(u4) skewed thoughts (60% one cluster): read usage spread, no dead codes",
          st["loc_use_entropy"] > 0.8 and st["dead_codes"] == 0.0, f"{st}")

    # (u5) no gradient into the thoughts; gate + gains learn from the main loss
    m = _module()
    _warm(m)
    B, Tt = 4, 64
    xs = [_clusters(B * Tt, 32, 900 + j) for j in range(3)]
    m.n_null = 0                                       # (null statistics pinned: every read confident, c_tok = 1)
    m.null_mu.fill_(-10.0)
    m.null_sd.fill_(1.0)
    m.null_n.fill_(1)
    with_m = _micro(m, xs, B, Tt, step=False)
    m_gate = [p.grad for p in m.gate_router.parameters()] + [m.gain_g.grad, m.gain_p.grad]
    no_m = _micro(m, xs, B, Tt, mem=False)
    same_g = all(torch.allclose(a.grad, b.grad, rtol=0, atol=0) for a, b in zip(with_m, no_m))
    check("(u5) d loss / d xf identical with and without the memory read (no path from the read to the thought)",
          same_g and m.ready, f"ready {m.ready}")
    check("(u5) ... the gate and the read gains get the main loss's gradient",
          all(g is not None and float(g.abs().sum()) > 0 for g in m_gate))
    m.zero_grad()

    # (u6) one write per selected token
    m = _module()
    _warm(m)
    w0, s0, p0 = m.cnt["writes_g"], m.store.stats_acc["writes"], m.cnt["writes_p"]
    _micro(m, [_clusters(B * Tt, 32, 950 + j) for j in range(3)], B, Tt)
    nw, ns, npv = m.cnt["writes_g"] - w0, m.store.stats_acc["writes"] - s0, m.cnt["writes_p"] - p0
    check("(u6) one write per selected token (3 passes x 2 experts each): global writes == write_w (the sheet got "
          "exactly those rows); private: top-1 expert + shared, one each",
          nw == SMALL["wave_write_w"] and ns == nw and npv == 2 * nw,
          f"writes {nw}, sheet rows {ns}, private rows {npv} (want {2 * nw})")

    # (u6b) the default private-write rule: only into the parts that were read (pair gate open or explored), at most
    # --wave-priv-cap (2) per token by routing slot, + the shared part if its gate read; exploration only in training
    m = _module(wave_priv_write="read", wave_explore=0.3)
    _warm(m)
    seen = []
    real_t = m._priv_read_targets

    def spy(rec, tok, strength):
        out = real_t(rec, tok, strength)
        seen.append((rec, tok, strength, out))
        return out
    m._priv_read_targets = spy
    _micro(m, [_clusters(B * Tt, 32, 960 + j) for j in range(3)], B, Tt)
    rec, tok_, st_, (rows_, parts_) = seen[-1]
    okr, per = True, {}
    for r_, p_ in zip(rows_.tolist(), parts_.tolist()):
        t_ = int(tok_[r_])
        per[r_] = per.get(r_, 0) + (p_ != m.shared_part)
        if p_ == m.shared_part:
            okr &= bool(rec["g"][t_] > 0)
        else:
            hit = (rec["t_k"] == t_) & (rec["e_k"] == p_)
            okr &= bool((rec["gp"][hit] > 0).any())
    expl = bool(((rec["gp"] > 0) & (rec["gp"] <= m.floor)).any())
    m.eval()
    m.begin_forward()
    e_k, p_k, t_k, slot_k, counts, C_ = _route(B * Tt, m.n_live)
    xq = _clusters(B * Tt, 32, 961)
    with torch.no_grad():
        rec_e = m._recall_of(xq)
        g_e, gp_e = m._gate(rec_e["feats"], B * Tt, 1, e_k, t_k)
    no_ex = bool((((gp_e == 0) | (gp_e > m.floor))).all()) and bool((((g_e == 0) | (g_e > m.floor))).all())
    check("(u6b) private writes only where read (open or explored pair, <= 2 routed parts per token, shared if read); "
          "explored pairs read below the floor in training, no exploration at eval",
          okr and rows_.numel() > 0 and max(per.values()) <= 2 and expl and no_ex,
          f"{rows_.numel()} private writes for {int((st_ > 0).sum())} written tokens; explored-below-floor pairs {expl}; "
          f"eval gates all 0 or > floor {no_ex}")
    m._priv_read_targets = real_t

    # (u12) the pre-read gate (wave_gate.GatedReadout, the default): no gradient into the thoughts; one decision +
    # recall per forward; ||o|| = exp(log_s) ||x|| ||u|| (D = 0, |u| <= 1: one ||x||, under the cap); exploration
    # ~5% in training, none at eval; writes only where read
    from wave_gate import GatedReadout as _GR
    if not hasattr(_GR, "after_backward"):                     # (wave_gate.py has moved to the v2.1 API; v21: tests_wave_v21)
        check("(u12) pre gate of the v2 module: skipped (wave_gate.GatedReadout has the v2.1 API now)", True)
    else:
        m_keep = m
        mg = _module(**PRE)
        _warm(mg)
        mg.n_null = 0
        mg.null_mu.fill_(-10.0)
        mg.null_sd.fill_(1.0)
        mg.null_n.fill_(1)
        xs = [_clusters(B * Tt, 32, 1900 + j) for j in range(3)]
        r0 = mg.cnt["recalls"]
        with_m = _micro(mg, xs, B, Tt, step=False)
        n_rec = mg.cnt["recalls"] - r0
        rec = mg._recall
        no_m = _micro(mg, xs, B, Tt, mem=False)
        same_g = all(torch.equal(a_.grad, b_.grad) for a_, b_ in zip(with_m, no_m))
        with torch.no_grad():
            o = mg._readout_pre(rec)
            dec = rec["dec"]
            u_n = (rec["c_g"] * rec["kap"])                       # (|u_g| per global row; private rows add <= 1)
            ratio = o.norm(dim=-1) / rec["xn"][dec.t].clamp(min=1e-6)
        s0 = float(torch.exp(mg.gr.log_s))
        ex_tr = float(dec.ex_p.float().mean()) if dec.training else -1
        mg.eval()
        mg.begin_forward()
        e_k, p_k, t_k, slot_k, counts, C_ = _route(B * Tt, mg.n_live)
        with torch.no_grad():
            de = mg._recall_pre(xs[0], e_k, t_k, slot_k, C_, 1)["dec"]
        check("(u12) pre gate: d xf.grad identical with and without the memory; one recall per forward; ||o|| / ||x|| <= "
              "exp(log_s) x 2 and <= the 0.5 cap; exploration in training only",
              same_g and n_rec == 1 and float(ratio.max()) <= min(0.5, 2 * s0) + 1e-5 and float(ratio.max()) > 0
              and not bool(de.ex_p.any()) and ex_tr >= 0,
              f"recalls {n_rec}; ||o||/||x|| max {float(ratio.max()):.3f} (s {s0:.2f}); global |u| mean "
              f"{float(u_n.mean()):.3f}; explored share of read rows {ex_tr:.3f}; eval explored {int(de.ex_p.sum())}")
        mg.zero_grad()
        m = m_keep

    # (u7) byte budget
    root = Path(tempfile.mkdtemp(prefix="wave_u7_"))

    class _Fake(torch.nn.Module):          # (save_memory_file wants model.bm, blocks[0].mlp.E and cfg["d"])
        def __init__(self, bm):
            super().__init__()
            self.bm, self.cfg = bm, dict(d=32)
            self.blocks = [type("B", (), dict(mlp=type("M", (), dict(E=4))()))()]
    f = _Fake(m)
    api.save_memory_file(root / "m.memory.pt", f, step=1)
    saved = sum(v.numel() * v.element_size() for v in torch.load(root / "m.memory.pt", weights_only=False)["state"].values()
                if torch.is_tensor(v))
    rep = api.byte_budget(m, cap_bytes=10e6)
    check("(u7) the budget's persistent + parameter bytes == the tensors in memory.pt",
          rep["persistent"] + rep["params"] == saved, api.budget_line(rep))
    check("(u7) cap check", rep["ok"] and not api.byte_budget(m, cap_bytes=rep["total"] // 2)["ok"])
    big = CodedWaveMemory(384, 769, **{k: WAVE_DEFAULTS[k] for k in WAVE_DEFAULTS})
    rb = api.byte_budget(big, cap_bytes=1.9e9)
    check("(u7) the full-size defaults (d 384, 769 experts): state under 1.9 GB", rb["ok"], api.budget_line(rb))
    del big
    big = CodedWaveMemory(384, 769, **dict(WAVE_DEFAULTS, wave_priv_levels=1))
    rb = api.byte_budget(big, cap_bytes=1.9e9)
    check("(u7) ... global + private parts (--wave-priv-levels 1, 769 experts): state under 1.9 GB", rb["ok"],
          api.budget_line(rb))
    del big
    m2 = _module()
    api.load_memory_file(root / "m.memory.pt", _Fake(m2))
    check("(u7) memory.pt load -> the same state (incl. host counters)",
          all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values())
              if torch.is_tensor(a)) and m2.address.n_upd == m.address.n_upd and m2.ready)

    # (u8) pluggable storage and policy
    calls = dict(write=[], read=0, confirm=0, forget=0, plan=0)

    class Probe(HebbianSheet):
        def write(self, codes, content, strength, part=None, entries=None, trace=None):
            calls["write"].append((len(codes), None if entries is None else tuple(entries.shape), float(strength.sum())))
            return super().write(codes, content, strength, part=part, entries=entries)

        def read(self, codes, part=None, entries=None):
            calls["read"] += 1
            return super().read(codes, part=part, entries=entries)

        def confirm(self, codes, ok, part=None, trace=None):
            calls["confirm"] += 1

        def forget(self, step, parts=None):
            calls["forget"] += 1

    class TwoEntriesOnSurprise(EntropySurprise):
        """'write count 2 on surprise' + 'a second entry only after a failed recall' (the shape of the toy's rule)."""
        def plan(self, tok, strength, recall):
            calls["plan"] += 1
            ent = torch.ones(tok.numel(), recall["E"], dtype=torch.bool)
            ent[:, 1:] = (recall["cos"] < 0.5)[:, None]
            return strength * 2, ent

    STORES["probe"], POLICIES["two_entries"] = Probe, TwoEntriesOnSurprise
    try:
        m = _module(wave_store="probe", wave_write="two_entries")
        _warm(m)
        ok = (calls["read"] > 0 and calls["confirm"] > 0 and calls["forget"] > 0 and calls["plan"] > 0
              and (SMALL["wave_write_w"], (SMALL["wave_write_w"], m.address.E), 2.0 * SMALL["wave_write_w"])
              in calls["write"])
        check("(u8) a registered store / policy: write (strength per write, entries per write), read, confirm, forget, "
              "plan all reached", ok, f"{dict(calls, write=calls['write'][-2:])}")
    finally:
        del STORES["probe"], POLICIES["two_entries"]

    # (u9) the keep mask: per sequence (the default), and the whole-forward mode
    xk = torch.zeros(8, 4, 32)

    def keeps(m_, n=100):
        m_.train()
        m_.train_forward = True
        out = []
        for _ in range(n):
            m_.begin_forward()
            out.append(m_._seq_keep(xk)[::4].tolist())
        return out
    m, m2 = _module(wave_drop=0.3), _module(wave_drop=0.3)
    seq, seq2 = keeps(m), keeps(m2)
    rate = 1 - sum(map(sum, seq)) / (8 * len(seq))
    per_tok = bool((m._seq_keep(xk).view(8, 4) == m._seq_keep(xk).view(8, 4)[:, :1]).all())
    m.eval()
    m.begin_forward()
    check("(u9) keep mask (seq): deterministic, ~0.3 of the sequences, every token of a sequence alike, never in eval",
          seq == seq2 and 0.24 < rate < 0.36 and per_tok and m._seq_keep(xk) is None, f"rate {rate:.3f} of 800")
    md = _module(wave_drop=0.5)
    _warm(md)
    md.n_null = 0                                      # (null statistics pinned: c_tok = 1)
    md.null_mu.fill_(-10.0)
    md.null_sd.fill_(1.0)
    md.null_n.fill_(1)
    B_, T_ = 8, 32
    e_k, p_k, t_k, slot_k, counts, C_ = _route(B_ * T_, md.n_live)
    md.train()
    md.train_forward = True
    md.begin_forward()
    xf = _clusters(B_ * T_, 32, 777).requires_grad_(True)
    b0 = xf.new_zeros(md.n_live, C_, 32).index_put((e_k, p_k), xf[t_k])
    bo, xo = md(xf, b0, xf.view(B_, T_, 32), None, e_k, p_k, t_k, slot_k, counts, B_ * T_, md.n_live, C_, e_k.numel(),
                call=1)
    kp = md._keep.view(B_, T_)[:, 0] > 0
    dxs = (xo - xf.view(B_, T_, 32)).detach().abs().sum((1, 2))
    (bo.sum() + xo.sum()).backward()
    gp = md._probe.grad.view(B_, T_, 32).abs().sum((1, 2))
    check("(u9) ... dropped sequences read exactly 0, kept ones read, the capture leaf gets every sequence's gradient",
          bool(kp.any()) and bool((~kp).any()) and float(dxs[~kp].max()) == 0.0 and float(dxs[kp].sum()) > 0
          and float(gp.min()) > 0, f"kept {kp.int().tolist()}")
    md._recs, md._sel, md._rows = {}, None, []
    m = _module(wave_drop=0.3, wave_drop_mode="forward")
    m.train()
    m.train_forward = True
    fw = []
    for _ in range(400):
        m.begin_forward()
        fw.append(m._dropped)
    check("(u9) ... --wave-drop-mode forward: whole forwards, rate ~ 0.3", 0.22 < sum(fw) / 400 < 0.38,
          f"rate {sum(fw) / 400:.3f}")

    # (u10) train-only at module level
    m = _module()
    _warm(m)
    m.set_train_only([1, -1])
    before = _sd(m)
    _micro(m, [_clusters(B * Tt, 32, 980 + j) for j in range(3)], B, Tt, step=False)
    with torch.no_grad():
        m.gain_p.sub_(0.1 * m.gain_p.grad)            # (as an optimizer would; then the step end)
    m.after_step(0)
    after = _sd(m)
    parts = [1, m.shared_part]
    bad = []
    for k, v in after.items():
        if not torch.is_tensor(v):
            continue
        if k in ("priv.syn", "priv.strength"):
            for p in range(m.priv.P):
                if p not in parts and not torch.equal(before[k][m.priv.part_slice(p)], v[m.priv.part_slice(p)]):
                    bad.append(f"{k}[{p}]")
        elif k == "gain_p":
            keep = [p for p in range(v.numel()) if p not in parts]
            if not torch.equal(before[k][keep], v[keep]):
                bad.append(k)
        elif not torch.equal(before[k], v):
            bad.append(k)
    learned = not torch.equal(before["priv.strength"][m.priv.part_slice(m.shared_part)],
                              after["priv.strength"][m.priv.part_slice(m.shared_part)])
    check("(u10) train-only experts 1 + shared: address, global store, gate, other parts unchanged; shared part written",
          not bad and learned, f"changed outside the selection: {bad}")
    m.set_train_only(None)

    # (u11) train-only with the global column store (no private parts): owned places, gate row e
    m = _module(**COLS)
    _warm(m, steps=8)
    m.n_null = 0
    m.null_mu.fill_(-10.0)
    m.null_sd.fill_(1.0)
    m.null_n.fill_(1)
    e_sel = 1
    used = m.owner_use[:, e_sel] > 0
    m.n_owned = 4                                       # (4 of the 16 places: tokens outside them exist)
    m.set_train_only([e_sel])
    own = m.store.owned.clone()
    st = m.store
    xq = _clusters(B * Tt, 32, 991)
    e_k, p_k, t_k, slot_k, counts, C_ = _route(B * Tt, m.n_live)
    rp0, rs0 = m.read(xq, e_k, t_k)
    before = _sd(m)
    for j in range(3):                                   # (writes need surprising tokens: a few micro-batches)
        _micro(m, [_clusters(B * Tt, 32, 1980 + 10 * j + i) for i in range(3)], B, Tt, step=False)
        with torch.no_grad():
            m.gate_e.sub_(0.1 * m.gate_e.grad.sign())   # (as an optimizer would, then the step end)
        m.after_step(0)
    after = _sd(m)
    rp1, rs1 = m.read(xq, e_k, t_k)
    kb, cb = 3 * 16 * 16 * 16, 2 * 16 * 3 * 64          # (key / chain rows per place)
    bad, changed = [], {}
    for k, v in after.items():
        if not torch.is_tensor(v):
            continue
        if k in ("store.key_syn", "store.chain_bits"):
            per = kb if k == "store.key_syn" else cb
            diff = (before[k] != v).view(st.P, per, -1).any(-1).any(-1)
            changed[k] = int(diff.sum())
            if bool((diff & ~own).any()):
                bad.append(k)
        elif k == "gate_e":
            keep = torch.ones(v.numel(), dtype=torch.bool)
            keep[e_sel] = False
            changed[k] = int(not torch.equal(before[k][e_sel], v[e_sel]))
            if not torch.equal(before[k][keep], v[keep]):
                bad.append(k)
        elif k in ("store.fill_est", "store.fill_ch_est"):
            continue                                     # (counters of the sweep)
        elif k == "store.load":                          # (the per-place memory counter: owned places only)
            if not torch.equal(before[k][~own], v[~own]):
                bad.append(k)
        elif not torch.equal(before[k], v):
            bad.append(k)
    codes_q = m.address.encode(xq)
    out = ~own[codes_q.loc[:, :m.store.n_read]].any(1)
    same_out = torch.equal(rs0[out], rs1[out])
    check("(u11) train-only, global column store: owned = top owner_use places (none empty); only owned places' "
          "synapses / chain bits and gate row e change; the hints of tokens with both entries outside owned unchanged",
          0 < int(own.sum()) <= 4 and not bool((own & ~used).any()) and not bad and changed["gate_e"] == 1
          and changed["store.key_syn"] > 0 and same_out and bool(out.any()),
          f"owned {int(own.sum())} of {int(used.sum())} used places (share {m._owned_share}); changed places "
          f"{changed}; outside the selection {bad}; tokens outside owned {int(out.sum())}")
    check("(u11) ... trainable: gate_e alone", [n for n, p in m.named_parameters()
                                                if any(p is q for q in m.trainable_parameters([e_sel]))] == ["gate_e"])
    m.set_train_only(None)
    try:
        ms = _module(wave_priv_levels=0)
        ms.set_train_only([1])
        err = ""
    except NotImplementedError as ex:
        err = str(ex)
    check("(u11) ... the sheet store without private parts: a clear error", "no train-one-expert mode" in err, err[:90])


# ====================================================================== the schedule + the memory's own optimizer
def schedule_tests(check):
    """(r1) passes before --wave-read-pass get nothing (bit-identical inputs); ONE recall per forward at the read
    pass, re-applied at the later passes; the capture leaf's gradient is the sum over the applied passes.
    (r2) the memory's parameters step in its own AdamW: train_lm's BrainOpt / AdamW / clip exclude bm.* for wave."""
    m = _module(wave_read_pass=3)
    _warm(m, steps=3, passes=6)
    B, Tt, E, D = 2, 64, m.n_live, 32
    Ntok = B * Tt
    e_k, p_k, t_k, slot_k, counts, C = _route(Ntok, E)
    n_read = [0]
    real = m.store.read_traced

    def counted(*a, **k):
        n_read[0] += 1
        return real(*a, **k)
    m.store.read_traced = counted
    m.train()
    m.train_forward = True
    m.begin_forward()
    same, outs, loss = [], [], 0.0
    for k in range(6):
        xf = _clusters(Ntok, D, 2000 + k).requires_grad_(True)
        buf0 = xf.new_zeros(E, C, D).index_put((e_k, p_k), xf[t_k])
        bo, xs_ = m(xf, buf0, xf.view(B, Tt, D), None, e_k, p_k, t_k, slot_k, counts, Ntok, E, C, e_k.numel(), call=k)
        same.append(torch.equal(bo, buf0))
        outs.append((bo - buf0).detach())
        loss = loss + (bo * (k + 1)).sum()
    loss.backward()
    probe = m._probe
    want = sum(range(4, 7)) * torch.ones(Ntok, D)          # d loss / d probe = sum over passes 3..5 of (k + 1) per pair
    got = probe.grad / torch.bincount(t_k, minlength=Ntok).float()[:, None]
    m.store.read_traced = real
    check("(r1) schedule (the default --wave-read-pass 3): passes 0-2 untouched, one recall per forward (at pass 3) "
          "re-applied at passes 3-5, the capture leaf's gradient = the sum over the applied passes",
          same[0] and same[1] and same[2] and not same[3] and n_read[0] == 1 and torch.allclose(got, want)
          and WAVE_DEFAULTS["wave_read_pass"] == 3,
          f"inputs untouched per pass {same}, store reads {n_read[0]}, capture grad / pair {float(got.mean()):.1f} "
          f"(want {float(want.mean()):.1f})")
    m._recs, m._sel, m._rows = {}, None, []
    # (r2) own optimizer: the brain optimizer leaves bm.* out
    import train_lm as tl

    class _M(torch.nn.Module):
        def __init__(self, bm):
            super().__init__()
            self.bm, self.w = bm, torch.nn.Linear(4, 4)
    model = _M(m)
    names = [n for n, _ in tl.brain_params(model)]
    from brain_opt import BrainOpt
    args = type("A", (), dict(opt="brain", tokens=1000, batch=1, accum=1, seq=10, lr=1e-3, wd=0.1, muon=1, lazy=1,
                              maturity=0.0))()
    bo_ = tl.build_opt(model, args)
    check("(r2) the memory's parameters are outside the brain optimizer and the brain clip (BrainOpt groups, "
          "brain_params), and step in the memory's own AdamW",
          not any(n.startswith("bm.") for n in names) and not any(n.startswith("bm.") for n in bo_.groups)
          and any(n.startswith("w.") for n in bo_.groups), f"brain params {names}; BrainOpt groups {list(bo_.groups)}")
    assert isinstance(bo_, BrainOpt)
    # (r3) the spec write policy (W4)
    from wave_write import SpecPolicy
    pol = SpecPolicy(write_w=16, per_seq=1)
    B, T = 20, 32
    g = torch.Generator().manual_seed(4)
    ce, H = torch.rand(B * T, generator=g) * 6, torch.rand(B * T, generator=g) * 3
    tok, s = pol.select(ce, H, 100, B, T)
    sig = (ce / (H + 1)).view(B, T)
    q90 = sig.reshape(-1).kthvalue(int(0.9 * B * T)).values
    seqs = tok // T
    okp = (len(set(seqs.tolist())) == tok.numel() == 16 and bool((sig.view(-1)[tok] >= q90).all())
           and int((s == 2).sum()) == 4 and bool((s[:4] == 2).all()) and bool((tok % T >= 1).all()))
    s2, ent = pol.plan(tok, s, dict(ok=(torch.arange(16) % 2 == 0).float(), E=2))
    check("(r3) spec write policy: one write per sequence (16) at sigma = CE/(H+1) >= q90, strength 2 for the top 4, "
          "confirmed recalls not written, entry 0 only",
          okp and bool((s2[::2] == 0).all()) and bool((s2[1::2] == s[1::2]).all()) and bool(ent[:, 0].all())
          and not bool(ent[:, 1].any()), f"sequences {sorted(set(seqs.tolist()))[:6]}..., strengths {s.tolist()}")
    # (r4) the null control: an empty store reads exactly 0 (c_tok = 0)
    m = _module(wave_write_w=0)
    _warm(m, steps=3)
    xq = _clusters(256, 32, 4321)
    e_k, p_k, t_k, slot_k, counts, C = _route(256, m.n_live)
    m.eval()
    rp, rs = m.read(xq, e_k, t_k)
    check("(r4) null control: nothing stored -> c_tok = 0 -> the read is exactly 0 (pairs and shared)",
          float(rp.abs().max()) == 0.0 and float(rs.abs().max()) == 0.0 and int(m.null_n.sum()) > 0,
          f"max |read| {float(rs.abs().max()):.2e}; null updates {m.null_n.tolist()}")


# ====================================================================== the v2 column store (wave_columns.py)
def column_tests(check):
    import math as _m
    from wave_address import Codes
    from wave_columns import C as NC, F as NF, PHI, UC, WaveColumns
    torch.manual_seed(0)
    a = WaveAddress(32, side=4, slices=32, codes=16, seed=3, warm=1)
    for s in range(6):
        a.update(_clusters(1024, 32, 500 + s, k=40))

    def store(**kw):
        return WaveColumns(32, 1, a, seed=5, **kw)

    # (c1) U20: the LUT read == the arithmetic decode; two nibbles of one byte written in one batch both land
    st = store()
    g = torch.Generator().manual_seed(1)
    st.key_syn.copy_(torch.randint(0, 256, st.key_syn.shape, generator=g, dtype=torch.uint8))
    rows = torch.randint(0, st.key_syn.size(0), (50, NF), generator=g)
    zr, zi = st._Z(rows)
    byt = st.key_syn[rows].long()
    s = torch.stack([byt & 15, byt >> 4], -1).flatten(-2)                        # (50, F, 64)
    ang = torch.remainder(s - 1, PHI).float() * (2 * _m.pi / PHI)
    ar_, ai_ = (ang.cos() * (s > 0)).sum(1), (ang.sin() * (s > 0)).sum(1)
    st2 = store()
    st2._key_update(torch.tensor([7, 7]), torch.tensor([10, 11]), torch.tensor([3, 1]), torch.tensor([2., 1.]), "set")
    b7 = int(st2.key_syn[7, 5])
    both = (b7 & 15) == 1 + 5 * 1 + 3 and (b7 >> 4) == 1 + 0 + 1
    check("(c1) columns: the byte LUT read == the arithmetic nibble decode; two nibbles of one byte written in one "
          "batch both land", torch.allclose(zr, ar_, atol=1e-5) and torch.allclose(zi, ai_, atol=1e-5) and both,
          f"max |diff| {float((zr - ar_).abs().max()):.1e}; byte {b7:#04x}")

    # (c2) U10-like: clean recall >= 0.99 at <= 26 memories per place; keys matching with prob 0.6
    st = store()
    n = 16 * 26
    loc = torch.arange(n) % 16
    keys = torch.randint(0, 16, (n, 96), generator=g)
    content = torch.nn.functional.normalize(torch.randn(n, 32, generator=g), dim=-1)

    def codes_of(keys_):
        cw = torch.ones(n, 96, 2) * torch.tensor([1.0, 0.0])
        return Codes(torch.stack([loc, (loc + 1) % 16], 1), torch.zeros(n, 2), torch.stack([keys_, keys_], -1), cw,
                     torch.zeros(n, 96, 2), torch.zeros(n, a.n_feats))
    cs = codes_of(keys)
    v_true, ph_true = st.encode(content)
    st.write(cs, content, torch.ones(n))
    tr = st._recall(cs)
    acc_clean = float((tr["v"][:, 0] == v_true).float().mean())
    # the spec's U10 hard point: 131 memories / place, cue keys matching w.p. 0.3, vs the column model (stage_a.py)
    n = 16 * 131
    loc = torch.arange(n) % 16
    keys = torch.randint(0, 16, (n, 96), generator=g)
    content_h = torch.nn.functional.normalize(torch.randn(n, 32, generator=g), dim=-1)
    sth, st0 = store(), store(lam=0.0)
    cs_h = codes_of(keys)
    v_h, _ = sth.encode(content_h)
    for x in (sth, st0):
        x.write(cs_h, content_h, torch.ones(n))
    noisy = torch.where(torch.rand(n, 96, generator=g) < 0.3, keys, torch.randint(0, 16, (n, 96), generator=g))
    acc_n = (sth._recall(codes_of(noisy))["v"][:, 0] == v_h).float().mean((0, 2))
    acc_0 = (st0._recall(codes_of(noisy))["v"][:, 0] == v_h).float().mean((0, 2))
    import stage_a
    pred = stage_a.column_model(131, [0.3] * 3, 0.0, n_same=0, trials=40)
    check("(c2) columns: clean unit accuracy >= 0.99 at 26 memories / place; at 131 / place with keys matching w.p. "
          "0.3: layer 2 within +-0.05 of the spec's U10 figure (~0.64), the chain (lambda 4) above chain off",
          acc_clean >= 0.99 and abs(float(acc_n[2]) - 0.64) <= 0.05 and float(acc_n[2]) > float(acc_0[2]),
          f"clean {acc_clean:.3f}; a=0.3, n=131: per layer {[round(float(x), 3) for x in acc_n]} vs column model "
          f"{[round(float(x), 3) for x in pred]} (stage_a, optimistic), chain off {[round(float(x), 3) for x in acc_0]}")
    # (c3) U12: the decoder norm constraint after a step + renorm
    with torch.no_grad():
        st.A.add_(0.1 * torch.randn_like(st.A))
    st.renorm()
    nrm = (st.A.norm(dim=-1) - st.rho.view(3, 1, 1)).abs().max()
    orth = ((st.A * st.B).sum(-1)).abs().max()
    check("(c3) columns: decoder |A_v| = |B_v| = rho_l and B_v orthogonal to A_v after renorm (within 1e-5)",
          float(nrm) < 1e-5 and float(orth) < 1e-5, f"norm err {float(nrm):.1e}, A.B {float(orth):.1e}")
    # (c4) U11 / U17-lite: the sweep holds the key fill near f* and the chain fill near f*ch
    st = store(fill=0.12, sweep=16)
    fills = []
    for step in range(120):
        m_ = 64
        loc_s = torch.randint(0, 16, (m_,), generator=g)
        ks = torch.randint(0, 16, (m_, 96), generator=g)
        cw = torch.ones(m_, 96, 2) * torch.tensor([1.0, 0.0])
        c4 = Codes(torch.stack([loc_s, loc_s], 1), torch.zeros(m_, 2), torch.stack([ks, ks], -1), cw,
                   torch.zeros(m_, 96, 2), torch.zeros(m_, a.n_feats))
        st.write(c4, torch.randn(m_, 32, generator=g), torch.ones(m_))
        st.forget(step)
        if step >= 60:
            fills.append((float(st.fill_est), float(st.fill_ch_est)))
    fk = sum(f[0] for f in fills) / len(fills)
    fc = sum(f[1] for f in fills) / len(fills)
    check("(c4) columns: the sweep holds the key fill at f* (0.12 +- 0.02) and the chain fill at f*ch (0.0315 +- "
          "0.005) under a steady write stream", abs(fk - 0.12) <= 0.02 and abs(fc - st.fill_ch) <= 0.005,
          f"key fill {fk:.3f}, chain fill {fc:.4f} (f*ch {st.fill_ch:.4f})")
    # (c5) determinism
    s1, s2 = store(), store()
    n26 = len(cs)
    for x in (s1, s2):
        x.write(cs, content, torch.ones(n26))
        x.confirm(cs, torch.ones(n26), trace=dict(x.read_traced(cs)[1]))
        x.forget(0)
    check("(c5) columns: deterministic (write + confirm + sweep twice from the same state)",
          all(torch.equal(u, v) for u, v in zip(s1.state_dict().values(), s2.state_dict().values())))
    # (c6) entry 1 reads with its OWN residual keys (spec PART 2): code_e[:, 1] = the codes of x_hat - P[E1]
    a = WaveAddress(32, side=4, slices=32, codes=16, seed=3, warm=1, anneal=10)    # (distinct places: short anneal)
    for s in range(20):
        a.update(_clusters(1024, 32, 500 + s, k=40))
    a.entry_keys = True
    X6 = _clusters(64, 32, 610, k=40)
    c6 = a.encode(X6)
    Xh = a.standardize(X6)
    own1 = a._top1_codes(Xh - a.W[c6.loc[:, 1]])
    st6 = store()
    ct6 = torch.nn.functional.normalize(torch.randn(64, 32, generator=g), dim=-1)
    v6, _ = st6.encode(ct6)
    ent6 = torch.zeros(64, 2, dtype=torch.bool)
    ent6[:, 1] = True                                   # (written at entry 1 only, with its keys K[:, 1])
    st6.write(c6, ct6, torch.ones(64), entries=ent6)
    tr6 = st6._recall(c6)
    acc6 = float((tr6["v"][:, 1] == v6).float().mean())
    st6s = store()
    st6s.shared_keys = True
    st6s.write(c6, ct6, torch.ones(64), entries=ent6)
    acc6s_cross = float((st6s._recall(c6)["v"][:, 1] == v6).float().mean())
    check("(c6) columns: entry 1 reads / writes with its own residual keys (code_e[:, 1] == the codes of "
          "x_hat - P[E1]; entry 0 == code[..., 0]); a memory written at E1 is recalled there; the shared-keys arm",
          torch.equal(c6.code_e[:, 0], c6.code[:, :, 0]) and torch.equal(c6.code_e[:, 1], own1)
          and float((c6.code_e[:, 1] != c6.code_e[:, 0]).float().mean()) > 0.3 and acc6 >= 0.95
          and acc6s_cross >= 0.95,
          f"entry-1 keys differ from entry 0 on {float((c6.code_e[:, 1] != c6.code_e[:, 0]).float().mean()):.2f} of "
          f"the codes; recall at E1: own keys {acc6:.3f}, shared-keys arm {acc6s_cross:.3f}")
    # (c7) the recall chunk (--wave-read-chunk, default 2048 for speed) does not change the recall
    st7 = store()
    st7.write(cs_h, content_h, torch.ones(len(cs_h)))
    ra = st7._recall(cs_h)
    st7.read_chunk = 2048
    rb_ = st7._recall(cs_h)
    check("(c7) columns: recall with chunk 512 == chunk 2048 (bit-identical)",
          all(torch.equal(ra[k_], rb_[k_]) for k_ in ra), f"{len(cs_h)} cues")
    assert NC == 16 and UC == 64


# ====================================================================== the wave sheet (wave_sheet.py)
def sheet_tests(check):
    from wave_sheet import CENTRE, WaveSheet
    torch.manual_seed(0)
    a = WaveAddress(32, side=4, slices=16, codes=8, seed=3, warm=1)
    for s in range(8):
        a.update(_clusters(1024, 32, 300 + s, k=40))

    def sheet(**kw):
        return WaveSheet(32, 1, a, side=16, units=16, seed=5, **kw)

    # (s1) packing + a written path: counters = strength, phases = phi_post - phi_pre
    sh = sheet()
    v = torch.randint(0, 4, (5, sh.M))
    p = torch.where(v > 0, torch.randint(0, sh.PH, (5, sh.M)), torch.zeros_like(v))
    urow = torch.arange(5) * 7 + 3                          # (put / get round trip on 5 storage rows)
    sh._rows_put(urow, v, p)
    c_rt, p_rt = sh._rows_get(urow)
    rt = torch.equal(c_rt, v) and torch.equal(p_rt, p)
    sh.syn.zero_() if sh.pack == "nibble" else (sh.cnt.zero_(), sh.ph.zero_())
    X = _clusters(64, 32, 400, k=40)
    codes = a.encode(X)
    ct = torch.nn.functional.normalize(torch.randn(64, 32, generator=torch.Generator().manual_seed(1)), dim=-1)
    tr = sh._wave(codes)
    s = torch.full((64,), 2.0)
    ent = torch.zeros(64, 2, dtype=torch.bool)
    ent[:, 0] = True
    sh.write(codes, ct, s, entries=ent, trace=tr)
    q = sh.quantize(ct)
    pre, w, post, th = sh._path_synapses(codes, tr["dcorner"][:, 0], q // sh.PH, q % sh.PH)   # (new: default path)
    c, ph = sh.syn_at(pre, w)                                                 # (n, syn, M)
    got_c = c.gather(-1, post[..., None]).squeeze(-1)
    got_p = ph.gather(-1, post[..., None]).squeeze(-1)
    frac_c = float((got_c == 2).float().mean())
    frac_p = float((got_p == th).float().mean())
    check("(s1) the sheet: packing round-trips; every synapse of a written path has count = the write's strength (2) "
          "and phase = phi_post - phi_pre (collisions between paths may merge phases)",
          rt and frac_c == 1.0 and frac_p > 0.8, f"count ok {frac_c:.3f}, phase ok {frac_p:.3f}")
    # (s2) recall: the same thoughts read back their contents; both entries tried
    R, tr2 = sh.read_traced(codes)
    top = (torch.nn.functional.normalize(R, dim=-1) @ ct.T).argmax(1)
    acc = float((top == torch.arange(64)).float().mean())
    check("(s2) the sheet recalls what was written (64 memories, same thoughts): content top-1 among the 64",
          acc > 0.8, f"top-1 {acc:.3f}; coincidence c_t {tr2['c'].amax(1).mean(0).tolist()}")
    # (s3) the second entry: a memory written only at entry 2 is recalled when the reader tries both
    sh2 = sheet()
    tr = sh2._wave(codes)
    ent2 = torch.zeros(64, 2, dtype=torch.bool)
    ent2[:, 1] = True
    sh2.write(codes, ct, torch.ones(64), entries=ent2, trace=tr)
    R2, t2 = sh2.read_traced(codes)
    acc2 = float(((torch.nn.functional.normalize(R2, dim=-1) @ ct.T).argmax(1) == torch.arange(64)).float().mean())
    check("(s3) entries: written only from the 2nd entry location, recalled (the reader picks entry 2)",
          acc2 > 0.8 and float((t2["best"] == 1).float().mean()) > 0.8,
          f"top-1 {acc2:.3f}, entry 2 chosen {float((t2['best'] == 1).float().mean()):.3f}")
    # (s4) confirm: +1 along the recalled path, cap 3
    sh3 = sheet()
    tr = sh3._wave(codes)
    sh3.write(codes, ct, torch.ones(64), entries=ent, trace=tr)
    _, trr = sh3.read_traced(codes)
    b = trr["best"]
    ar = torch.arange(64)
    pre, w, post, _ = sh3._path_synapses(codes, trr["corner"][ar, b], trr["unit"][ar, b], trr["phase"][ar, b])
    cnt = lambda: sh3.syn_at(pre, w)[0].gather(-1, post[..., None]).squeeze(-1)
    c0 = cnt()
    for _ in range(4):
        sh3.confirm(codes, torch.ones(64), trace=trr)
    c1 = cnt()
    check("(s4) confirm: +1 per confirmed recall along the recalled path, capped at 3",
          int(c1.max()) == 3 and bool((c1 >= c0).all()) and float((c1 == 3).float().mean()) > 0.9,
          f"before mean {float(c0.float().mean()):.2f}, after 4 confirms mean {float(c1.float().mean()):.2f}")
    # (s5) forgetting holds the fill near the target under a steady write stream
    sh4 = sheet(fill=0.12, fill_every=20)
    g = torch.Generator().manual_seed(9)
    fills = []
    both = torch.ones(48, 2, dtype=torch.bool)
    for step in range(900):
        Xs = torch.randn(48, 32, generator=g) * 3                       # (spread-out thoughts: the whole sheet)
        cs = a.encode(Xs)
        sh4.write(cs, torch.nn.functional.normalize(torch.randn(48, 32, generator=g), dim=-1), torch.ones(48),
                  entries=both)
        sh4.forget(step)
        if step >= 500 and step % 20 == 19:
            fills.append(float(sh4.fill_est))
    mf = sum(fills) / len(fills)
    check("(s5) random-decrement forgetting holds the synapse fill near 0.12 (steady write stream past the target)",
          0.09 <= mf <= 0.15, f"fill over steps 500-900: {[round(f, 3) for f in fills]}, lambda {float(sh4.lam):.2e}")
    # (s6) determinism of the sheet
    s5a, s5b = sheet(), sheet()
    for st_ in (s5a, s5b):
        st_.write(codes, ct, torch.ones(64))
        st_.forget(0)
    check("(s6) the sheet is deterministic (write + read + forget twice from the same state)",
          all(torch.equal(x, y) for x, y in zip(s5a.state_dict().values(), s5b.state_dict().values()))
          and torch.equal(s5a.read(codes), s5b.read(codes)))
    # (s7) the local key synapses sit at the window centre, which no wave synapse uses
    pre, w, post, _ = sh._path_synapses(codes, tr["corner"][:, 0], q // sh.PH, q % sh.PH)
    is_key = (pre % sh.U) < sh.K
    check("(s7) local key -> content synapses only at the window centre; wave synapses never there",
          bool(((w == CENTRE) <= is_key).all()))

    # (s8) the gate: about --wave-gate-open0 (25%) of the (token, expert) pairs open after its warm-up
    m = _module(wave_explore=0.0, wave_lr=0.0)          # (the init: the gate does not learn here)
    _warm(m, steps=2)
    opens = []
    for st_ in range(6):
        _micro(m, [_clusters(4 * 64, 32, 700 + 10 * st_ + j) for j in range(3)], 4, 64)
        opens.append(m.log_record()["wmem_gate_open"])
    check("(s8) gate: ~25% of the reads open once the open bias is set (like 25d), not ~100%",
          0.12 <= opens[-1] <= 0.4, f"open rate per step {[round(o, 3) for o in opens]}, bias {float(m.open_bias):.3f}")


# ====================================================================== end to end
def e2e_tests(check, root):
    from test_memory import _tiny_data
    _tiny_data(root / "data")
    la, a_d, _ = T.run(root, "w_a", HERE, WAVE + ["--memory-save", "1"])
    log = (a_d / "train.log").read_text()
    import json
    recs = [json.loads(l) for l in (a_d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    last = recs[-1]
    check("(e1) tiny --memory wave run: finite losses, the address ready, writes, the wmem_* log",
          len(la) == N and all(math.isfinite(v) for v in la.values()) and last.get("wmem_ready") == 1
          and last.get("wmem_writes_g", 0) > 0 and "wave memory" in log
          and 0 < last.get("wmem_recalls", 0) <= last.get("wmem_fwd", 0),
          f"last loss {la.get(N - 1)}; writes {last.get('wmem_writes_g')} / private {last.get('wmem_writes_p')}; "
          f"write mix ce {last.get('wmem_write_ce', 0):.3f} vs all {last.get('wmem_all_ce', 0):.3f}; "
          f"gate open {last.get('wmem_gate_open')}; dropped {last.get('wmem_dropped')}/{last.get('wmem_seqs')} sequences; "
          f"recalls {last.get('wmem_recalls')} over {last.get('wmem_fwd')} training forwards")
    lb, b_d, _ = T.run(root, "w_b", HERE, WAVE + ["--memory-save", "1"])
    ok, info = T.losses_equal(la, lb, range(N))
    bad = [b for b in T.ckpt_equal(T._load(a_d / "last.pt"), T._load(b_d / "last.pt")) if not b[0].startswith("cfg/run")]
    check("(e2) determinism: the same run twice, losses + checkpoint bit-identical", ok and not bad,
          f"{info}; {len(bad)} differing: {bad[:4]}")
    ls1, s_d, _ = T.run(root, "w_s", HERE, WAVE + ["--memory-save", "1"], stop=K)
    ls2, s2_d, _ = T.run(root, "w_s2", HERE, WAVE + ["--memory-save", "1", "--resume", str(s_d / "last.pt")])
    ok, info = T.losses_equal(la, {**ls1, **ls2}, range(N))
    bad = T.ckpt_equal(T._load(a_d / "last.pt"), T._load(s2_d / "last.pt"))
    check("(e3) stop at K + resume == uninterrupted: losses + full checkpoint (incl. the memory)", ok and not bad,
          f"{info}; {len(bad)} differing: {bad[:4]}")

    # (e4) save / reset / load; off; eval leaves the memory untouched
    ck = T._load(a_d / "last.pt")
    m = T._build(ck)
    x, y = T._batch()
    s0 = T._mem_buffers(m)
    h0, l0 = T._out(m, x, y)
    s1 = T._mem_buffers(m)
    check("(e4) an eval forward leaves every memory buffer bit-identical", all(torch.equal(s0[k], s1[k]) for k in s0))
    m.reset_memory()
    m.eval()
    h1, l1 = T._out(m, x, y)
    check("(e4) reset: a fresh memory (not ready, output changes)", not m.bm.ready and not torch.equal(h0, h1),
          f"loss {l0:.6f} -> {l1:.6f}")
    api.load_memory_file(api.memory_path(a_d / "last.pt"), m)
    h2, l2 = T._out(m, x, y)
    check("(e4) load memory.pt after the reset: outputs bit-identical", torch.equal(h0, h2) and l0 == l2)
    with api.memory_off(m):
        h_off, l_off = T._out(m, x, y)
    m_nm = T._build(ck)
    m_nm.bm = None
    h_nm, l_nm = T._out(m_nm, x, y)
    check("(e4) memory off == the brain without a memory module; on != off", torch.equal(h_off, h_nm) and
          not torch.equal(h0, h_off), f"loss on {l0:.6f}, off {l_off:.6f}, no module {l_nm:.6f}")

    if os.environ.get("WAVE_E2E_QUICK"):                # (e1-e4 only: the must-pass set)
        return
    # (e5) swap: a --memory none checkpoint, resumed with --memory wave
    T.run(root, "n_s", HERE, WAVE[2:] + ["--memory", "none"], stop=K)
    lw, w_d, _ = T.run(root, "n_to_w", HERE, WAVE + ["--resume", str(root / "runs" / "n_s" / "last.pt")])
    log = (w_d / "train.log").read_text()
    ckw = T._load(w_d / "last.pt")
    brain_a = {k: v for k, v in T._load(root / "runs" / "n_s" / "last.pt")["state"].items()}
    check("(e5) swap none -> wave on resume: runs, the brain loaded, a wave memory in the checkpoint",
          "MEMORY SWAP" in log and any(k.startswith("bm.address.") for k in ckw["state"])
          and set(brain_a) <= set(ckw["state"]) and all(math.isfinite(v) for v in lw.values()))

    # (e6) train-only experts 1 + shared
    le, e_d, _ = T.run(root, "w_to", HERE, WAVE + ["--resume", str(s_d / "last.pt"), "--memory-train-only",
                                                    "experts=1,shared", "--memory-eval-off", "1"])
    a_ck, e_ck = T._load(s_d / "last.pt"), T._load(e_d / "last.pt")
    ba = {k: v for k, v in a_ck["state"].items() if not k.startswith("bm.")}
    be = {k: v for k, v in e_ck["state"].items() if not k.startswith("bm.")}
    bad = [k for k in ba if not torch.equal(ba[k], be[k])]
    check("(e6) train-only: every brain parameter and buffer bit-identical", not bad and set(ba) == set(be),
          f"{len(ba)} tensors; differing {bad[:5]}")
    ma = {k[3:]: v for k, v in a_ck["state"].items() if k.startswith("bm.")}
    me = {k[3:]: v for k, v in e_ck["state"].items() if k.startswith("bm.")}
    E1 = int(a_ck["cfg"]["wave_max_experts"])
    P = E1 + 1
    per = ma["priv.strength"].numel() // P
    sel = [1, E1]
    other, changed = [], {}
    for k, v in me.items():
        if not torch.is_tensor(v):
            continue
        a = ma[k]
        if k in ("priv.syn", "priv.strength", "gain_p"):
            rows = a.view(P, -1) if k != "gain_p" else a.view(P, 1)
            rows_e = v.view(P, -1) if k != "gain_p" else v.view(P, 1)
            keep = torch.ones(P, dtype=torch.bool)
            keep[sel] = False
            if not torch.equal(rows[keep], rows_e[keep]):
                other.append(k)
            changed[k] = [int(not torch.equal(rows[p], rows_e[p])) for p in sel]
        elif not torch.equal(a, v):
            other.append(k)
    log = (e_d / "train.log").read_text()
    check("(e6) train-only: address, global store, gate, gain_g, write baseline, other experts' parts unchanged",
          not other, f"differing: {other}")
    check("(e6) ... the selected parts learned (private writes / gains of expert 1 and the shared expert)",
          any(sum(v) for v in changed.values()), f"{changed}")
    check("(e6) ... the trainer's line: only bm.gain_p trainable",
          "MEMORY TRAIN-ONLY: experts [1, -1]; trainable parameters ['bm.gain_p']" in log)
    assert per > 0

    # (e7) train-only with the global column store, no private parts (the default config)
    G = list(WAVE)
    G[G.index("--wave-priv-levels") + 1] = "0"
    _, g_d, _ = T.run(root, "g_s", HERE, G + ["--memory-save", "1"], stop=K)
    lg, gt_d, _ = T.run(root, "g_to", HERE, G + ["--resume", str(g_d / "last.pt"), "--memory-train-only", "experts=1",
                                                 "--memory-eval-off", "1"])
    a_ck, e_ck = T._load(g_d / "last.pt"), T._load(gt_d / "last.pt")
    ba = {k: v for k, v in a_ck["state"].items() if not k.startswith("bm.")}
    be = {k: v for k, v in e_ck["state"].items() if not k.startswith("bm.")}
    bad = [k for k in ba if not torch.equal(ba[k], be[k])]
    log = (gt_d / "train.log").read_text()
    ma = {k[3:]: v for k, v in a_ck["state"].items() if k.startswith("bm.")}
    me = {k[3:]: v for k, v in e_ck["state"].items() if k.startswith("bm.")}
    other = [k for k, v in me.items() if torch.is_tensor(v) and k not in ("store.key_syn", "store.chain_bits", "gate_e",
                                                                         "store.fill_est", "store.fill_ch_est",
                                                                         "store.load")
             and not torch.equal(ma[k], v)]
    ge = (ma["gate_e"] != me["gate_e"]).nonzero().flatten().tolist()
    check("(e7) train-only expert 1, global column store: the brain bit-identical; only bm.gate_e trainable; gate rows "
          "other than 1 and every other memory tensor (address, decoder, gate MLP, null stats, owner_use) unchanged",
          not bad and not other and set(ge) <= {1} and "trainable parameters ['bm.gate_e']" in log
          and all(math.isfinite(v) for v in lg.values()),
          f"brain differing {bad[:3]}; memory differing {other}; gate rows changed {ge}; "
          f"{[l for l in log.splitlines() if 'owned places' in l][:1]}")


def run_all(root=None):
    root = Path(root or tempfile.mkdtemp(prefix="wave_")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    R = []

    def check(name, ok, detail=""):
        R.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    unit_tests(check)
    schedule_tests(check)
    column_tests(check)
    sheet_tests(check)
    if not os.environ.get("WAVE_UNIT_ONLY"):
        e2e_tests(check, root)
    return R, root


def test_wave(tmp_path):
    R, _ = run_all(tmp_path / "wave")
    bad = [r for r in R if not r[1]]
    assert not bad, "\n".join(f"{n}: {d}" for n, _, d in bad)


if __name__ == "__main__":
    R, root = run_all(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"\nwork dir {root}")
    print("ALL PASS" if all(r[1] for r in R) else f"{sum(not r[1] for r in R)} FAIL of {len(R)}")
