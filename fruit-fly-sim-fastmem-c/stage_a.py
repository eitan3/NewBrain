"""Stage A of the WaveSheet v2 spec (SPEC_wavesheet_v2_cloud.md, PART 1): the pre-build gates, offline on the blob run's
ckpt_6000 (runs/world_blob25d_12k, read-only, CPU, 2 threads; the run's own code_snapshot, no bytecode written).

    python -B stage_a.py grad      capture e = dL/dm (m: a zero leaf added at the experts' inputs at passes 3-5, the
                                   frozen brain, the training loss of each row) at every fact sentence's cue position
                                   of the T0a documents, and at every cloze prompt's last position (the answer's CE);
                                   also |xf| there per pass.  -> cap/stage_a_grad_6k.npz
    python -B stage_a.py t2        T2: -dL/dm stability (same fact over its 3 train exposures, vs a different fact with
                                   the same template, train vs cloze) + random-init decoder capture of the real targets
    python -B stage_a.py t6        T6: the oracle - inject m = alpha |x| (-e_hat of the fact's first train exposure) at
                                   the cloze prompt's last position, passes 3-5, all experts and the shared expert, g = 1;
                                   cloze top-1 vs no injection, another fact with the same template, and the mean -e
    python -B stage_a.py t1        T1: the address at 32 x 32 places: parity (wbench's idealised code-match scoring),
                                   place agreement, per-level key agreement a_l and P(stored key in cue top-2) for
                                   train-template returns, held-out templates, cloze cues, same-person-other-relation;
                                   and the column model (spec PART 2) with those a_l
    python -B stage_a.py reach     the expert as the private address: P(the fact's training top-1 expert in the cue's
                                   top-6), P(top-1 or top-2 in top-6), P(top-1 == top-1), held + cloze cues, passes 1-5

The documents, cloze prompts and their order are exactly the W-bench capture's (speed_wip/mem26/w1c_capture.py logic,
copied in capture_xf.py; nothing is written to mem26 or runs).
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
RUN = Path(r"C:\Eitan\Python\NewBrain\runs\world_blob25d_12k")
SNAP = RUN / "code_snapshot"
MEM26 = Path(r"C:\Eitan\Python\NewBrain\speed_wip\mem26")
OUT = HERE / "cap" / "stage_a_grad_6k.npz"
RELS = ["city", "job", "pet", "year", "friend"]
T00 = time.time()


def log(*a):
    print(f"[{time.time() - T00:.0f}s]", *a, flush=True)


# ====================================================================== the frozen brain + an injector at the read point
class Injector(torch.nn.Module):
    """Stands in for the run's blob memory at RoutedMLP's bmem hook: adds m (N, D) to every kept pair's expert input
    and to the shared expert's input at the passes in `passes`. m: a zero leaf (grad capture) or a fixed hint."""
    enabled = True

    def __init__(self, passes=(3, 4, 5)):
        super().__init__()
        self.passes, self.m, self.xnorm = set(passes), None, {}
        self.train_forward = False

    def begin_forward(self):
        self.xnorm, self.xf3 = {}, None

    def forward(self, xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=None):
        self.xnorm[call] = xf.detach().float().norm(dim=-1)
        if call == 3:
            self.xf3 = xf.detach().float()
        if call not in self.passes or self.m is None:
            return buf, x
        m = self.m.to(buf.dtype)
        buf = buf.index_put((e_k, p_k), m[t_k], accumulate=True)
        return buf, x + m.view(x.shape).to(x.dtype)


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def load_brain(ckpt="ckpt_6000.pt"):
    sys.path.insert(0, str(SNAP))
    from loop_brain import LoopBrain
    from spiking_lm import RoutedMLP
    ck = torch.load(RUN / ckpt, map_location="cpu", weights_only=False, mmap=True)
    model = LoopBrain(**ck["cfg"])
    model.load_state_dict(ck["state"], strict=False)
    del ck
    model.bm = None
    if getattr(model, "mem", None) is not None:
        model.mem.write_on = False
    model.eval()
    model.K = 6
    for mod in model.modules():
        if isinstance(mod, RoutedMLP):
            mod.noise, mod.eval_noise, mod.cap = 0.0, 0.0, 0
    for p in model.parameters():
        p.requires_grad_(False)
    inj = Injector()
    model.bm = inj
    return model, inj


def load_docs():
    sys.path.insert(0, str(HERE))
    import capture_xf as cx
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(cx.WORLD / "tokenizer.json"))
    build_docs, tokenize_docs, pack = cx.load_t0a()
    docs, cloze, probe = build_docs(tok, 200, 200)
    tokenize_docs(tok, docs)
    rows, place = pack(docs, 512, np.random.default_rng(1))
    return tok, docs, cloze, rows, place


def fact_table(docs, place):
    where, recs = [], []
    for di, d in enumerate(docs):
        r, s0 = place[di]
        for f in d["facts"]:
            if f["vtok"] < 0:
                continue
            where.append((r, s0 + f["ctok"], s0 + f["vtok"]))
            recs.append((di, f["person"], RELS.index(f["rel"]), f["template"], int(f["probe"]),
                         ["d1", "d2", "d3", "held"].index(d["role"])))
    return np.array(where), np.array(recs, np.int64)


def cloze_answers(docs, cloze):
    """The answer token of each cloze prompt: the value token of its held document's probe fact."""
    ans = {}
    for d in docs:
        if d["role"] != "held":
            continue
        for f in d["facts"]:
            if f["probe"] and f["vtok"] >= 0:
                ans[(f["person"], f["rel"])] = d["ids"][f["vtok"]]
    return np.array([ans.get((c["person"], c["rel"]), -1) for c in cloze], np.int64)


# ====================================================================== grad: e = dL/dm
def cmd_grad(a):
    tok, docs, cloze, rows, place = load_docs()
    where, recs = fact_table(docs, place)
    ans = cloze_answers(docs, cloze)
    log(f"{len(docs)} docs, {len(where)} fact sentences, {len(cloze)} cloze ({(ans < 0).sum()} without an answer), "
        f"{len(rows)} rows")
    if a.max_rows:
        rows, cloze = rows[:a.max_rows], cloze[:a.max_rows]
        keep = where[:, 0] < a.max_rows
        where, recs = where[keep], recs[keep]
        ans = ans[:a.max_rows]
    model, inj = load_brain()
    n = len(where)
    E = np.zeros((n, 384), np.float32)
    XN = np.zeros((n, 6), np.float32)
    CE = np.zeros(n, np.float32)
    B = a.batch
    for b in range(0, len(rows), B):
        x = torch.from_numpy(rows[b:b + B])
        inp, tgt = x[:, :-1], x[:, 1:]
        Bt, T = inp.shape
        inj.m = torch.zeros(Bt * T, 384, requires_grad=True)
        logits = _logits(model(inp))
        lp = torch.log_softmax(logits.float(), -1)
        nll = -lp.gather(-1, tgt[..., None]).squeeze(-1)                       # (B, T)
        nll.mean().backward()
        g = inj.m.grad.view(Bt, T, 384)
        sel = np.nonzero((where[:, 0] >= b) & (where[:, 0] < b + B))[0]
        rr, cc = where[sel, 0] - b, where[sel, 1]
        E[sel] = g[rr, cc].numpy()
        CE[sel] = nll.detach()[rr, cc].numpy()
        for k in range(6):
            XN[sel, k] = inj.xnorm[k].view(Bt, T)[rr, cc].numpy()
        inj.m = None
        if (b // B) % 10 == 0:
            log(f"  rows {b}/{len(rows)}")
    # cloze: the answer's CE at the last position
    nc = len(cloze)
    EC = np.zeros((nc, 384), np.float32)
    XC = np.zeros((nc, 6), np.float32)
    LP = np.zeros(nc, np.float32)
    for b in range(0, nc, 32):
        C = cloze[b:b + 32]
        T = max(len(c["ids"]) for c in C)
        x = np.zeros((len(C), T), np.int64)
        for i, c in enumerate(C):
            x[i, :len(c["ids"])] = c["ids"]
        last = torch.tensor([len(c["ids"]) - 1 for c in C])
        x = torch.from_numpy(x)
        inj.m = torch.zeros(len(C) * T, 384, requires_grad=True)
        logits = _logits(model(x))
        lp = torch.log_softmax(logits.float()[torch.arange(len(C)), last], -1)
        aa = torch.from_numpy(np.maximum(ans[b:b + 32], 0))
        nll = -lp.gather(-1, aa[:, None]).squeeze(-1)
        nll.sum().backward()
        g = inj.m.grad.view(len(C), T, 384)
        EC[b:b + len(C)] = g[torch.arange(len(C)), last].numpy()
        LP[b:b + len(C)] = (lp.argmax(-1) == aa).float().numpy()
        for k in range(6):
            XC[b:b + len(C), k] = inj.xnorm[k].view(len(C), T)[torch.arange(len(C)), last].numpy()
        inj.m = None
    log(f"cloze top-1 without memory: {LP[ans >= 0].mean():.3f}")
    np.savez(a.out, e=E, xnorm=XN, ce=CE, rec=recs, where=where, e_cloze=EC, xnorm_cloze=XC, cloze_top1=LP, ans=ans,
             cloze_rec=np.array([(c["person"], RELS.index(c["rel"]), c["template"]) for c in cloze], np.int64))
    log(f"wrote {a.out}")


# ====================================================================== T0: the crowd (real wiki thoughts) + baselines
WIKI_VAL = Path(r"X:\Datasets\wikipedia_en_20231101_wiki16k\val.bin")
WIKI_OUT = HERE / "cap" / "wiki_xf3_6k.npy"


def cmd_wiki(a):
    """300K real Wikipedia thoughts (xf at pass 3, positions >= 8) of ckpt_6000 for T0 / T3b's crowd load."""
    mm = np.memmap(WIKI_VAL, dtype=np.uint16, mode="r")
    n_win = (a.crowd + 503) // 504
    st = np.linspace(0, len(mm) - 600, n_win).astype(np.int64)
    model, inj = load_brain()
    out = []
    for b in range(0, n_win, a.batch):
        x = torch.from_numpy(np.stack([np.asarray(mm[s:s + 512]).astype(np.int64) for s in st[b:b + a.batch]]))
        with torch.no_grad():
            model(x)
        out.append(inj.xf3.view(x.size(0), 512, -1)[:, 8:].reshape(-1, 384).half().numpy())
        if (b // a.batch) % 10 == 0:
            log(f"  wiki windows {b}/{n_win}")
    X = np.concatenate(out)[:a.crowd]
    np.save(WIKI_OUT, X)
    log(f"wrote {WIKI_OUT}: {X.shape}")


def cmd_t0(a):
    """T0 at pass 3 (the deployed schedule's recall point): ExactNN at the spec's byte budget (0.46 GB of fp16 key +
    content = 299K entries, FIFO) and the address's ideal code ranking, with no load and with the crowd load (300K
    real wiki thoughts written first, random contents, never the answer)."""
    sys.path.insert(0, str(MEM26))
    import tests_wave_bench as TB
    import wbench as W
    import wbench_module as WM
    b = W.Bench(passes=(3,))
    h = WM.Harness(b)
    crowd = np.load(WIKI_OUT).astype(np.float32)
    cap = int(0.4598e9 // (2 * 384 * 2))
    k = 3
    xs = torch.from_numpy(b.feats("store", k, ["xf"]))
    tg = torch.from_numpy(h.content[[h.lab_pos[int(l)] for l in b.store_lab]])
    rng = np.random.default_rng(0)
    for load in ("no load", "crowd 300K"):
        m = WM.ExactNN(cap=cap)
        if load != "no load":
            for i in range(0, len(crowd), 8192):
                c = rng.standard_normal((min(8192, len(crowd) - i), 384)).astype(np.float32)
                c *= np.linalg.norm(h.content[0]) / np.linalg.norm(c, axis=1, keepdims=True)
                m.bench_write(torch.from_numpy(crowd[i:i + 8192]), torch.from_numpy(c))
        for i in range(0, len(xs), 8192):
            m.bench_write(xs[i:i + 8192], tg[i:i + 8192])
        out = []
        for q, qlab in (("held", b.held_lab), ("cloze", b.cz_lab)):
            _, rs = m.read(torch.from_numpy(b.feats(q, k, ["xf"])), None, None)
            out.append(h.score(rs.numpy(), qlab)[0])
        print(f"T0 ExactNN, {cap} entries (0.46 GB), {load}: held top-1 {out[0]:.3f}, cloze top-1 {out[1]:.3f}", flush=True)
    # the ideal codes (the address at 32 x 32, the W-bench's code-match ranking), no load / crowd
    addr = TB.train_address(b.feats("fit", k, ["xf"]), seed=k, side=a.side)
    enc = lambda X: (lambda c: (c.loc.numpy(), c.code.numpy()))(addr.encode(torch.from_numpy(X)))
    st_c = enc(b.feats("store", k, ["xf"]))
    cr_c = enc(crowd)
    score = TB.fast_score(L=addr.L, M=addr.M)
    for load in ("no load", "crowd 300K"):
        out = []
        for q, qlab in (("held", b.held_lab), ("cloze", b.cz_lab)):
            qc = enc(b.feats(q, k, ["xf"]))
            S = score(qc, st_c)
            best_fact = S.max(1)
            if load != "no load":                         # a crowd entry that outscores every fact entry = a miss
                bc = np.full(len(qlab), -np.inf, np.float32)
                for i in range(0, len(crowd), 20000):
                    bc = np.maximum(bc, score(qc, (cr_c[0][i:i + 20000], cr_c[1][i:i + 20000])).max(1))
            top = b.store_lab[S.argmax(1)]
            ok = top == qlab
            if load != "no load":
                ok &= best_fact > bc
            out.append(float(ok.mean()))
        print(f"T0 ideal codes (address {a.side}x{a.side}), {load}: held top-1 {out[0]:.3f}, cloze top-1 {out[1]:.3f}",
              flush=True)
    print("T0 blob module at 0.46 GB: NOT RUN offline (its read needs the trainer-side fit / IVF / kernel state); proxy: "
          "the W-bench's idealised 25d-address ranking, pass 3: held 0.70 / cloze 0.50 (wbench_final.txt per pass), an "
          "upper-bound-like number for the blob read at the same capacity", flush=True)


# ====================================================================== T2
def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def encode_decoder(Y, A, B, rho):
    """The spec's W4 encode with a decoder (A, B [3, 16, 64, 24], |A_v| = |B_v| = rho_l, B_v orthogonal to A_v): per
    layer and column the (unit v*, 5-phase ph*) that best projects the residual; res -= cos A + sin B at (v*, ph*);
    the stored memory is (v*, ph*) only, so the read-out is the sum of those decoder vectors. Returns the read-out and
    the mean projection per layer (the spec's rho update target)."""
    n = len(Y)
    res = Y.reshape(n, 16, 24).copy()
    rec = np.zeros_like(res)
    ci = np.arange(16)[None]
    coefs = []
    for l in range(3):
        a_ = np.einsum("ncd,cvd->ncv", res, A[l]) / rho[l]
        b_ = np.einsum("ncd,cvd->ncv", res, B[l]) / rho[l]
        ph = np.remainder(np.round(np.arctan2(b_, a_) * 5 / (2 * np.pi)), 5)
        ang = ph * 2 * np.pi / 5
        sc = a_ * np.cos(ang) + b_ * np.sin(ang)
        v = sc.argmax(-1)                                                          # (n, 16) first index on ties
        angv = np.take_along_axis(ang, v[..., None], -1)[..., 0]
        coefs.append(float(np.take_along_axis(sc, v[..., None], -1).mean()))
        part = np.cos(angv)[..., None] * A[l][ci, v] + np.sin(angv)[..., None] * B[l][ci, v]   # (n, 16, 24)
        rec += part
        res -= part
    return rec.reshape(n, 384), coefs


def random_decoder(seed, rho=(0.137, 0.115, 0.096), aniso=False):
    g = np.random.default_rng(seed)
    A = g.standard_normal((3, 16, 64, 24))
    if aniso:
        A *= np.linspace(2.0, 0.2, 24)[None, None, None]
    B = g.standard_normal((3, 16, 64, 24))
    B -= (B * A).sum(-1, keepdims=True) / (A * A).sum(-1, keepdims=True) * A    # B orthogonal to A
    for l in range(3):
        A[l] *= rho[l] / np.linalg.norm(A[l], axis=-1, keepdims=True)
        B[l] *= rho[l] / np.linalg.norm(B[l], axis=-1, keepdims=True)
    return A, B, np.array(rho)


def cmd_t2(a):
    Z = np.load(OUT)
    e, rec = Z["e"], Z["rec"]
    eh = unit(-e)
    lab = rec[:, 1] * 10 + rec[:, 2]
    probe_tr = (rec[:, 4] == 1) & (rec[:, 5] <= 2)
    # same fact over its 3 train exposures
    by = {}
    for i in np.nonzero(probe_tr)[0]:
        by.setdefault(int(lab[i]), []).append(i)
    same, diff = [], []
    tmpl = {}
    for i in np.nonzero(probe_tr)[0]:
        tmpl.setdefault((int(rec[i, 2]), int(rec[i, 3])), []).append(i)
    g = np.random.default_rng(0)
    for l, idx in by.items():
        if len(idx) < 3:
            continue
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                same.append(float(eh[idx[x]] @ eh[idx[y]]))
        # a different fact with the same (relation, template) as idx[0]
        pool = [j for j in tmpl[(int(rec[idx[0], 2]), int(rec[idx[0], 3]))] if lab[j] != l]
        if pool:
            diff.append(float(eh[idx[0]] @ eh[pool[int(g.integers(len(pool)))]]))
    same, diff = np.array(same), np.array(diff)
    # train vs cloze
    ec = unit(-Z["e_cloze"])
    crec = Z["cloze_rec"]
    clab = crec[:, 0] * 10 + crec[:, 1]
    tc = [float(eh[by[int(l)][0]] @ ec[i]) for i, l in enumerate(clab) if int(l) in by and Z["ans"][i] >= 0]
    tc = np.array(tc)
    ok1, ok2, ok3 = np.median(same) >= 0.3, np.median(same) - np.median(diff) >= 0.2, np.median(tc) >= 0.25
    print(f"{'PASS' if ok1 else 'FAIL'}  T2 same-fact cos over 3 train exposures: median {np.median(same):.3f} "
          f"(n {len(same)}, p10 {np.percentile(same, 10):.3f}, p90 {np.percentile(same, 90):.3f}) >= 0.3")
    print(f"{'PASS' if ok2 else 'FAIL'}  T2 same-fact minus different-fact-same-template: {np.median(same):.3f} - "
          f"{np.median(diff):.3f} = {np.median(same) - np.median(diff):.3f} >= 0.2")
    print(f"{'PASS' if ok3 else 'FAIL'}  T2 train vs cloze: median cos {np.median(tc):.3f} (n {len(tc)}) >= 0.25")
    # random-init decoder capture of the real targets
    caps = []
    Y = eh[probe_tr][:4000]
    for aniso in (False, True):
        A, B, rho = random_decoder(1, aniso=aniso)
        for _ in range(30):                         # (rho_l -> the mean projection: the spec's .99/.01 EMA, settled)
            _, co = encode_decoder(Y[:1000], A, B, rho)
            new = np.maximum(np.array(co), 1e-3)
            A *= (new / rho)[:, None, None, None]
            B *= (new / rho)[:, None, None, None]
            rho = new
        R, _ = encode_decoder(Y, A, B, rho)
        caps.append(float(np.median((unit(R) * Y).sum(-1))))
    ok4 = min(caps) >= 0.78
    print(f"{'PASS' if ok4 else 'FAIL'}  T2 random-init decoder capture of the real targets: median cos "
          f"{caps[0]:.3f} (isotropic decoder), {caps[1]:.3f} (anisotropic) >= 0.78")
    # extra: mean-removed versions (the shared component of -e)
    mu = eh.mean(0)
    print(f"  (|mean of -e_hat| over all facts {np.linalg.norm(mu):.3f}; same-fact median after removing it "
          f"{np.median([float(unit(eh[i] - mu) @ unit(eh[j] - mu)) for i, j in pairs(by)]):.3f})")


def pairs(by):
    out = []
    for idx in by.values():
        if len(idx) >= 2:
            out.append((idx[0], idx[1]))
    return out


# ====================================================================== T6
def cmd_t6(a):
    Z = np.load(OUT)
    tok, docs, cloze, rows, place = load_docs()
    e, rec = Z["e"], Z["rec"]
    eh = unit(-e)
    lab = rec[:, 1] * 10 + rec[:, 2]
    first = {}
    tmpl_of = {}
    for i in np.nonzero((rec[:, 4] == 1) & (rec[:, 5] == 0))[0]:            # the d1 (first train) exposure
        first.setdefault(int(lab[i]), i)
        tmpl_of.setdefault((int(rec[i, 2]), int(rec[i, 3])), []).append(i)
    mean_e = unit(eh[(rec[:, 4] == 1)].mean(0))
    ans = Z["ans"]
    crec = Z["cloze_rec"]
    model, inj = load_brain()
    g = np.random.default_rng(0)
    items = [i for i in range(len(cloze)) if ans[i] >= 0 and int(crec[i, 0] * 10 + crec[i, 1]) in first]
    other = []
    for i in items:
        f = first[int(crec[i, 0] * 10 + crec[i, 1])]
        pool = [j for j in tmpl_of[(int(rec[f, 2]), int(rec[f, 3]))] if lab[j] != lab[f]]
        other.append(pool[int(g.integers(len(pool)))] if pool else f)
    items, other = np.array(items), np.array(other)
    xnc = Z["xnorm_cloze"]

    def run(hint_of):
        """hint_of(j) -> (384,) unit hint for item j or None; returns top-1 over the items."""
        hits = []
        for b in range(0, len(items), 32):
            ii = items[b:b + 32]
            C = [cloze[i] for i in ii]
            T = max(len(c["ids"]) for c in C)
            x = np.zeros((len(C), T), np.int64)
            for r, c in enumerate(C):
                x[r, :len(c["ids"])] = c["ids"]
            last = np.array([len(c["ids"]) - 1 for c in C])
            hs = [hint_of(b + r) for r in range(len(C))]
            if hs[0] is None:
                inj.m = None
            else:
                m = torch.zeros(len(C), T, 384)
                for r in range(len(C)):
                    m[r, last[r]] = torch.from_numpy(hs[r].astype(np.float32))
                inj.m = m.view(-1, 384)
            with torch.no_grad():
                logits = _logits(model(torch.from_numpy(x)))
            pred = logits[torch.arange(len(C)), torch.from_numpy(last)].argmax(-1).numpy()
            hits.append(pred == ans[ii])
        inj.m = None
        return float(np.concatenate(hits).mean())

    base = run(lambda j: None)
    print(f"T6 cloze top-1 without a hint: {base:.3f} (n {len(items)})", flush=True)
    xn = xnc[items, 3:6].mean(1)                     # |x| at the read point, passes 3-5 (mean)
    best = {}
    for alpha in (0.1, 0.2, 0.3, 0.5):
        r_same = run(lambda j: alpha * xn[j] * eh[first[int(crec[items[j], 0] * 10 + crec[items[j], 1])]])
        r_oth = run(lambda j: alpha * xn[j] * eh[other[j]])
        r_mean = run(lambda j: alpha * xn[j] * mean_e)
        best[alpha] = (r_same, r_oth, r_mean)
        print(f"  alpha {alpha}: same fact {r_same:.3f} ({100 * (r_same - base):+.1f} pts), other fact same template "
              f"{r_oth:.3f}, mean -e {r_mean:.3f}", flush=True)
    al = max(best, key=lambda k: best[k][0])
    s, o, mm = best[al]
    ok = (s - base) >= 0.20 and (s - max(o, mm)) >= 0.10
    print(f"{'PASS' if ok else 'FAIL'}  T6 oracle at alpha {al}: gain {100 * (s - base):+.1f} pts (>= +20), over the best "
          f"control {100 * (s - max(o, mm)):+.1f} pts (>= +10)")


# ====================================================================== T1
def cmd_t1(a):
    sys.path.insert(0, str(MEM26))
    import tests_wave_bench as TB
    import wbench as W
    b = W.Bench(passes=tuple(int(x) for x in a.passes.split(",")))
    res = b.run(f"wave_address at {a.side}x{a.side} (in-step learning)", TB.make_module_codes(side=a.side), 49)
    m = lambda q: float(np.mean([res[(q, k)][0] for k in b.passes]))
    held, cloze = m("held"), m("cloze")
    ok = held >= 0.66 and cloze >= 0.50
    print(f"{'PASS' if ok else 'FAIL'}  T1 (a) parity at {a.side}x{a.side}: held {held:.3f} (>= 0.66), cloze {cloze:.3f} "
          f"(>= 0.50), passes {b.passes}", flush=True)
    # agreements at pass 3
    k = 3
    xs = b.feats("fit", k, ["xf"])
    addr = TB.train_address(xs, seed=k, side=a.side)
    r = b.rec
    st_i = b.store_idx
    enc = lambda X: addr.encode(torch.from_numpy(X))
    cs = enc(b.feats("store", k, ["xf"]))
    lab_s = b.store_lab
    probe_tr = (r[st_i, 4] == 1)
    by = {}
    for j in np.nonzero(probe_tr)[0]:
        by.setdefault(int(lab_s[j]), []).append(j)
    S = addr.S

    def agree(cq, iq, ref):
        """cq: cue codes (rows iq), ref: stored row index per cue -> place top-1 / top-2-any, a_l, P(key in top-2)."""
        loc1 = (cq.loc[iq, 0] == cs.loc[ref, 0]).float().mean().item()
        loc2 = (cq.loc[iq][:, :, None] == cs.loc[ref][:, None, :]).any(-1).any(-1).float().mean().item()
        al, at2 = [], []
        same_place = cq.loc[iq, 0] == cs.loc[ref, 0]
        for l in range(addr.levels):
            sl = slice(l * S, (l + 1) * S)
            eq = (cq.code[iq, sl, 0] == cs.code[ref, sl, 0]).float()
            in2 = (cq.code[iq, sl, :2] == cs.code[ref, sl, :1]).any(-1).float()
            al.append(eq[same_place].mean().item() if same_place.any() else float("nan"))
            at2.append(in2[same_place].mean().item() if same_place.any() else float("nan"))
        return loc1, loc2, al, at2

    rows = {}
    # train-template returns: the 2nd train exposure as the cue for the 1st
    iq = np.array([v[1] for v in by.values() if len(v) >= 2])
    ref = np.array([v[0] for v in by.values() if len(v) >= 2])
    rows["train-template return"] = agree(cs, iq, ref)
    ch = enc(b.feats("held", k, ["xf"]))
    first = {l: v[0] for l, v in by.items()}
    ok_h = [i for i, l in enumerate(b.held_lab) if int(l) in first]
    rows["held-out template"] = agree(ch, np.array(ok_h), np.array([first[int(b.held_lab[i])] for i in ok_h]))
    cc = enc(b.feats("cloze", k, ["xf"]))
    ok_c = [i for i, l in enumerate(b.cz_lab) if int(l) in first]
    rows["cloze cue"] = agree(cc, np.array(ok_c), np.array([first[int(b.cz_lab[i])] for i in ok_c]))
    # same person, other relation: cue = the person's fact of relation r+1
    sp_q, sp_r = [], []
    for l, v in by.items():
        p, rel = divmod(l, 10)
        l2 = p * 10 + (rel + 1) % 5
        if l2 in by:
            sp_q.append(by[l2][0])
            sp_r.append(v[0])
    rows["same person, other relation"] = agree(cs, np.array(sp_q), np.array(sp_r))
    print(f"T1 agreements at pass {k}, {a.side}x{a.side} places (key agreement at the right place):")
    for nm, (l1, l2, al, at2) in rows.items():
        print(f"  {nm:30s} place top-1 {l1:.3f} top-2 {l2:.3f} | a_l " + " ".join(f"{x:.3f}" for x in al)
              + " | P(stored key in cue top-2) " + " ".join(f"{x:.3f}" for x in at2), flush=True)
    a_ret = rows["cloze cue"][2]
    a_same = rows["same person, other relation"][2]
    for n in (26, 66, 131):
        accs = column_model(n, a_ret, float(np.nanmean(a_same)))
        print(f"  column model (cloze-cue a_l {[round(x, 3) for x in a_ret]}, same-person a {np.nanmean(a_same):.3f}), "
              f"n {n} memories/place: per-layer winner accuracy " + " ".join(f"{x:.3f}" for x in accs), flush=True)


def column_model(n, a_l, a_same, n_same=8, trials=400, seed=0, lam=4.0, W=3, F=16, Uk=16, Uc=64, PHI=5, chain=True):
    """Spec PART 2 column model: one place, 16 columns x 3 layers. A memory = per (l, c) a unit v and phase ph; its
    keys: per (l, slice) a code in 0..Uk-1, column c reads slices (2c + q) % 32, q < F. Key synapse (l, c, q, key) ->
    (v, ph) (last write wins); chain bit (l-1, c, i, v_{l-1}[(c + i - 1) % 16]) -> v_l[c]. Recall from a cue whose keys
    match the target's with probability a_l (else random; the n_same same-person memories share each key with the
    target with probability a_same): layer 0 M = |Z| / F; layers 1-2 M = (|Zk| + lam * nch) / (F + lam * W);
    winner = argmax. Returns the per-layer winner accuracy of the target."""
    g = np.random.default_rng(seed)
    acc = np.zeros(3)
    for _ in range(trials):
        nm = n + n_same
        keys = g.integers(0, Uk, (nm, 3, 32))
        tgt = 0
        for s_ in range(1, 1 + n_same):                     # same-person memories share keys with the target
            share = g.random((3, 32)) < a_same
            keys[s_] = np.where(share, keys[tgt], keys[s_])
        v = g.integers(0, Uc, (nm, 3, 16))
        ph = g.integers(0, PHI, (nm, 3, 16))
        ksyn = -np.ones((3, 16, F, Uk, Uc), np.int64)        # phase of the synapse, -1 = absent
        chainb = np.zeros((2, 16, W, Uc, Uc), bool)
        cols = (2 * np.arange(16)[:, None] + np.arange(F)[None]) % 32   # (16, F) slices
        for mi in g.permutation(nm):
            for l in range(3):
                kk = keys[mi, l][cols]                       # (16, F)
                ksyn[l, np.arange(16)[:, None], np.arange(F)[None], kk, v[mi, l][:, None]] = ph[mi, l][:, None]
                if chain and l:
                    for i in range(W):
                        src = v[mi, l - 1][(np.arange(16) + i - 1) % 16]
                        chainb[l - 1, np.arange(16), i, src, v[mi, l]] = True
        cue = np.where(g.random((3, 32)) < np.array(a_l)[:, None], keys[tgt], g.integers(0, Uk, (3, 32)))
        prev = None
        for l in range(3):
            kk = cue[l][cols]                                # (16, F)
            P = ksyn[l, np.arange(16)[:, None], np.arange(F)[None], kk]   # (16, F, Uc)
            z = np.where(P >= 0, np.exp(2j * np.pi * np.maximum(P, 0) / PHI), 0).sum(1)
            M = np.abs(z) / F
            if l and chain:
                nch = np.zeros((16, Uc))
                for i in range(W):
                    src = prev[(np.arange(16) + i - 1) % 16]
                    nch += chainb[l - 1, np.arange(16), i, src]
                M = (np.abs(z) + lam * nch) / (F + lam * W)
            win = M.argmax(-1)
            acc[l] += (win == v[tgt, l]).mean()
            prev = win
    return acc / trials


# ====================================================================== reach: the expert as the private address
def cmd_reach(a):
    """For held and cloze cues, per pass 1-5 (pass 3 the main number): P(the top-1 expert of the fact's training
    sentence is in the cue's top-6), P(its top-1 or top-2 is), P(top-1 == top-1); averaged over (cue, training
    sentence of that fact) pairs. Routing: the W-bench capture's cue_e (training and held sentences, eval routing) and,
    for the cloze prompts' last position, this frozen ckpt_6000 brain's routing (eval: no noise, no capacity drop)."""
    sys.path.insert(1, str(MEM26))
    import wbench as W
    b = W.Bench(passes=(1, 2, 3, 4, 5))
    Z = np.load(MEM26 / "cap" / "w1c_6k.npz")
    cue_e = Z["cue_e"].astype(np.int64)                                     # (6 passes, sentences, 6), slot 0 = top-1
    tok, docs, cloze, rows, place = load_docs()
    same = np.array_equal(np.array([(c["person"], RELS.index(c["rel"])) for c in cloze]), b.crec[:, :2])
    log(f"cloze prompts in the bench's order: {same}")
    model, inj = load_brain()
    mlp = model.blocks[0].mlp
    ids = []
    mlp.register_forward_hook(lambda m, inp, out: ids.append(m.last_idx.detach().clone()))
    ce = np.zeros((6, len(cloze), 6), np.int64)
    with torch.no_grad():
        for i in range(0, len(cloze), 32):
            C = cloze[i:i + 32]
            T = max(len(c["ids"]) for c in C)
            x = np.zeros((len(C), T), np.int64)
            for j, c in enumerate(C):
                x[j, :len(c["ids"])] = c["ids"]
            ids.clear()
            model(torch.from_numpy(x))
            last = np.array([len(c["ids"]) - 1 for c in C])
            for k in range(6):
                ce[k, i:i + len(C)] = ids[k].view(len(C), T, -1)[np.arange(len(C)), last].numpy()
    log("cloze routing captured")
    train = {}
    for r_, lab in zip(b.store_idx, b.store_lab):
        train.setdefault(int(lab), []).append(int(r_))
    out = {}
    for q, qe, qlab in (("held", lambda k: cue_e[k, b.held_idx], b.held_lab), ("cloze", lambda k: ce[k], b.cz_lab)):
        for k in (1, 2, 3, 4, 5):
            E6 = qe(k)
            m1, m12, eq = [], [], []
            for i, l in enumerate(qlab):
                for r_ in train.get(int(l), ()):
                    t = cue_e[k, r_]
                    m1.append(t[0] in E6[i])
                    m12.append(t[0] in E6[i] or t[1] in E6[i])
                    eq.append(t[0] == E6[i, 0])
            out[(q, k)] = (np.mean(m1), np.mean(m12), np.mean(eq), len(m1))
    for q in ("held", "cloze"):
        k3 = out[(q, 3)]
        print(f"REACH {q} cues, pass 3: P(train top-1 in cue top-6) {k3[0]:.3f}, P(train top-1 or top-2 in cue top-6) "
              f"{k3[1]:.3f}, P(top-1 == top-1) {k3[2]:.3f} ({k3[3]} pairs) | per pass 1-5: "
              + "; ".join(f"k{k}: {out[(q, k)][0]:.3f} / {out[(q, k)][1]:.3f} / {out[(q, k)][2]:.3f}" for k in (1, 2, 3, 4, 5)),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grad", "t2", "t6", "t1", "wiki", "t0", "reach"])
    ap.add_argument("--crowd", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0, help="grad: only the first rows / cloze prompts (a smoke)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--side", type=int, default=32)
    ap.add_argument("--passes", default="1,2,3,4,5")
    a = ap.parse_args()
    dict(grad=cmd_grad, t2=cmd_t2, t6=cmd_t6, t1=cmd_t1, wiki=cmd_wiki, t0=cmd_t0, reach=cmd_reach)[a.cmd](a)


if __name__ == "__main__":
    main()
