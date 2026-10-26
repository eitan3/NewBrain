"""T3 capacity curve (coordinator addition) and the T3a / T3b-crowd W-bench of the full wave memory (the default: the
v2 column store, global sheet, gate bypassed, the pass-3 schedule's recall point), through wbench_module.Harness (its
captured routing and its scoring), vs baselines at equal bytes, with the W-bench facts stored ON TOP of real Wikipedia
thought crowds (ckpt_6000, xf at pass 3: stage_a.py wiki). CPU, 2 threads.

    python -B tests_wave_curve.py --point 0|1|2|3|a|b [--arm wave|exact|ideal] [--sweep 1] [--kw wave_lambda=0]

Points: total memories = facts (25,426 sentences) + crowd, at ~26 / 66 / 131 / 200 memories per place of the spec's
1,024 places; a = T3a (no load: the facts alone), b = T3b-crowd (300K wiki thoughts written first). Crowd contents
are random vectors at the content scale (never an answer). Writes go through the module's real write path
(bench_write: a read first, the policy's plan, confirm, write), in batches of 4,096; --sweep 1 (the default) runs the
step end (after_step: the sweep of 16 places) once per 32 writes, the spec's 32 writes per step ("live writes").
Reports cloze / held top-1, the live writes, confirms, the store's bytes and key / chain fill, and (wave arm) the
diagnostics that separate the store from the address: the place hit rates (a query's entry 0 / either entry is a
place where its fact was written) and the ORACLE-place recall (the read at the fact's own write place, keys = the
query's residual there, entry 0 only).
Baselines: ExactNN (FIFO, fp16 key + content) at the spec's 0.46 GB; the address's ideal codes at 32 x 32 places
(W-bench code-match ranking incl. the crowd). The blob module itself is not runnable offline (stage_a.py t0).
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
MEM26 = Path(r"C:\Eitan\Python\NewBrain\speed_wip\mem26")
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(MEM26))
import wbench as W  # noqa: E402
import wbench_module as WM  # noqa: E402
import wave_bench as WB  # noqa: E402
from memory_api import byte_budget  # noqa: E402
from wave_address import Codes  # noqa: E402
from wave_memory import WAVE_DEFAULTS  # noqa: E402

PER_PLACE = (26, 66, 131, 200)
N_FACTS = 25426
N_CROWD_B = 300_000


def n_crowd_of(point, n_have):
    if point == "a":
        return 0
    if point == "b":
        return min(N_CROWD_B, n_have)
    return max(0, PER_PLACE[int(point)] * 1024 - N_FACTS)


@torch.no_grad()
def place_diag(m, h, b, k, wplace, q, qlab):
    """Of the queries q, against the places where each fact was actually written (wplace: per stored sentence, -1 =
    not written): place top-1 / top-2 / top-E hit (E = the entries read), the oracle-place top-1 (the read at the
    fact's first write place, keys = the query's residual there, that entry only), and the confident-false rate (top-1
    wrong while the read's best cosine >= 0.5)."""
    a, st = m.address, m.store
    places = {}
    for lab, p in zip(b.store_lab, wplace):
        if p >= 0:
            places.setdefault(int(lab), []).append(int(p))
    X = torch.from_numpy(b.feats(q, k, ["xf"]))
    c = a.encode(X)
    E = st._n_ent(c)
    loc = c.loc[:, :E].numpy()
    has = np.array([int(l) in places for l in qlab])
    hit = lambda j: np.array([any(loc[i, jj] in places.get(int(l), ()) for jj in range(j)) for i, l in enumerate(qlab)])
    h1, h2, hE = hit(1), hit(min(2, E)), hit(E)
    orc = torch.tensor([places[int(l)][0] if int(l) in places else int(loc[i, 0]) for i, l in enumerate(qlab)])
    keys = a._top1_codes(a.standardize(X) - a.W[orc])                     # (the query's residual at that place)
    oc = Codes(torch.stack([orc] * E, 1), c.loc_s, c.code, c.code_w, c.code_d, c.feats, torch.stack([keys] * E, 1))
    ent = torch.zeros(len(qlab), E, dtype=torch.bool)
    ent[:, 0] = True
    tr = st._recall(oc, ent)
    tr["mass"], tr["loc"], tr["keys"] = tr["chibar"], oc.loc, st._all_keys(oc)
    r_orc = st.decode(tr).float().numpy()
    orc1 = h.score(r_orc, qlab)[0]
    return dict(h1=float(h1[has].mean()), h2=float(h2[has].mean()), hE=float(hE[has].mean()), E=E, orc=float(orc1),
                r_orc=r_orc)


def false_rate(h, r, qlab):
    """top-1 wrong while the read's best cosine with a fact's content is >= 0.5 (a confident false recall)."""
    rn = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)
    S = rn @ (h.content / np.linalg.norm(h.content, axis=1, keepdims=True)).T
    want = np.array([h.lab_pos[int(l)] for l in qlab])
    return float(((S.argmax(1) != want) & (S.max(1) >= 0.5)).mean())


def load_stats(places, P):
    """'used / max / max:mean / top-10 share' of written places (array, -1 = not written)."""
    w = places[places >= 0]
    if not len(w):
        return "none"
    cnt = np.sort(np.bincount(w, minlength=P))[::-1]
    return (f"{int((cnt > 0).sum())} places, max {int(cnt[0])}, max/mean {cnt[0] / cnt.mean():.1f}, top-10 share "
            f"{cnt[:10].sum() / cnt.sum():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", required=True, choices=["0", "1", "2", "3", "a", "b"])
    ap.add_argument("--arm", default="wave", choices=["wave", "exact", "ideal"])
    ap.add_argument("--sweep", type=int, default=1, help="wave: the step end once per 32 writes (1) or never (0)")
    ap.add_argument("--kw", default="", help="wave: extra cfg, e.g. wave_lambda=0,wave_entry_keys=shared, "
                                             "wave_map_beta=4 (place balance A), wave_entries=4,wave_alloc_gamma=1 (B)")
    a = ap.parse_args()
    k = 3
    b = W.Bench(passes=(k,))
    h = WM.Harness(b)
    crowd_all = np.load(HERE / "cap" / "wiki_xf3_6k.npy", mmap_mode="r")
    n_crowd = n_crowd_of(a.point, len(crowd_all))
    crowd = np.asarray(crowd_all[:n_crowd], dtype=np.float32)
    total = n_crowd + N_FACTS
    g = np.random.default_rng(0)
    scale = float(np.linalg.norm(h.content[0]))
    xs = torch.from_numpy(b.feats("store", k, ["xf"]))
    tg = torch.from_numpy(h.content[[h.lab_pos[int(l)] for l in b.store_lab]])
    t0 = time.time()
    name = {"a": "T3a (no load)", "b": "T3b-crowd (300K wiki first)"}.get(a.point, f"point {a.point}")
    tag = (f"{name}: {total / 1024:.0f} memories/place (x1024 places), {n_crowd} crowd + {N_FACTS} facts")

    def crowd_content(n):
        c = g.standard_normal((n, 384)).astype(np.float32)
        return torch.from_numpy(c * scale / np.linalg.norm(c, axis=1, keepdims=True))

    def score(read):
        out = []
        for q, qlab in (("held", b.held_lab), ("cloze", b.cz_lab)):
            r = read(q, b.feats(q, k, ["xf"]))
            out.append(h.score(r, qlab)[0])
        return out

    if a.arm == "wave":
        kw = dict(wave_priv_levels=0)
        for p in filter(None, a.kw.split(",")):
            key, v = p.split("=")
            kw[key] = type(WAVE_DEFAULTS[key])(v)
        m = WB.factory(b, int(h.cue_e.max()) + 1, **kw)()
        st = m.store
        n_w = [0]
        wp_crowd, wp_fact = [], []

        def write(X, T, e_k=None, t_k=None, into=None):
            m.bench_write(X, T, e_k, t_k, call=k)
            into.append(m.last_write_place.numpy())
            n_w[0] += len(X)
            while a.sweep and n_w[0] >= 32:
                m.after_step(0)
                n_w[0] -= 32
        for i in range(0, n_crowd, 4096):
            write(torch.from_numpy(crowd[i:i + 4096]), crowd_content(min(4096, n_crowd - i)), into=wp_crowd)
            if i % (16 * 4096) == 0:
                print(f"  crowd {i}/{n_crowd} ({time.time() - t0:.0f}s)", flush=True)
        for i in range(0, len(xs), 4096):
            e_k, t_k = h._route(b.store_idx[i:i + 4096], k)
            write(xs[i:i + 4096], tg[i:i + 4096], e_k, t_k, into=wp_fact)
        idx = dict(held=b.held_idx, cloze=None)
        reads = {q: h._read_vecs(m, b.feats(q, k, ["xf"]), idx[q], k) for q in ("held", "cloze")}
        held, cloze = (h.score(reads[q], ql)[0] for q, ql in (("held", b.held_lab), ("cloze", b.cz_lab)))
        fr = {q: false_rate(h, reads[q], ql) for q, ql in (("held", b.held_lab), ("cloze", b.cz_lab))}
        rep = byte_budget(m)
        wf = np.concatenate(wp_fact)
        wc = np.concatenate(wp_crowd) if wp_crowd else np.zeros(0, np.int64)
        diag = {q: place_diag(m, h, b, k, wf, q, ql) for q, ql in (("held", b.held_lab), ("cloze", b.cz_lab))}
        cntp = np.bincount(np.concatenate([wf, wc])[np.concatenate([wf, wc]) >= 0], minlength=st.P)
        first = {}
        for lab_, p_ in zip(b.store_lab, wf):
            if p_ >= 0:
                first.setdefault(int(lab_), int(p_))
        for q, ql in (("held", b.held_lab), ("cloze", b.cz_lab)):     # (recall of facts written at light places)
            lo = np.array([int(l) in first and cntp[first[int(l)]] <= 131 for l in ql])
            diag[q]["lo"] = ((h.score(reads[q][lo], ql[lo])[0], h.score(diag[q]["r_orc"][lo], ql[lo])[0])
                             if lo.any() else (float("nan"),) * 2, int(lo.sum()))
        dl = "; ".join(f"{q}: light-place (<= 131 memories) facts top-1 {d['lo'][0][0]:.3f}, oracle {d['lo'][0][1]:.3f} (n {d['lo'][1]}), place top-1 {d['h1']:.3f} top-2 {d['h2']:.3f} top-{d['E']} {d['hE']:.3f}, oracle-place "
                       f"top-1 {d['orc']:.3f}, confident-false {fr[q]:.3f}" for q, d in diag.items())
        print(f"WAVE  {tag}{' [' + a.kw + ']' if a.kw else ''}: held top-1 {held:.3f}, cloze top-1 {cloze:.3f} | "
              f"live writes {st.stats_acc['writes']}, confirms {st.stats_acc['confirms']}, swept "
              f"{st.stats_acc['swept']} places, key fill {st.fill_sample():.4f}, chain fill "
              f"{float(st.fill_ch_est):.4f}, state {rep['total'] / 1e6:.1f} MB (store {st.byte_budget() / 1e6:.1f} MB) "
              f"| load: facts {load_stats(wf, st.P)}; crowd {load_stats(wc, st.P)}; both "
              f"{load_stats(np.concatenate([wf, wc]), st.P)} | {dl} ({time.time() - t0:.0f}s)", flush=True)
    elif a.arm == "exact":
        for nm, nbytes in (("spec 0.46 GB", 0.4598e9),):
            cap = int(nbytes // (2 * 384 * 2))
            m = WM.ExactNN(cap=cap)
            for i in range(0, n_crowd, 8192):
                m.bench_write(torch.from_numpy(crowd[i:i + 8192]), crowd_content(min(8192, n_crowd - i)))
            for i in range(0, len(xs), 8192):
                m.bench_write(xs[i:i + 8192], tg[i:i + 8192])
            held, cloze = score(lambda q, X: m.read(torch.from_numpy(X), None, None)[1].numpy())
            live = int(min(int(m.n), cap))
            print(f"EXACT {tag}, {nm} ({cap} entries): held top-1 {held:.3f}, cloze top-1 {cloze:.3f} | live "
                  f"entries {live}, {byte_budget(m)['total'] / 1e6:.0f} MB ({time.time() - t0:.0f}s)", flush=True)
    else:
        import tests_wave_bench as TB
        addr = TB.train_address(b.feats("fit", k, ["xf"]), seed=k, side=WAVE_DEFAULTS["wave_side"])
        enc = lambda X: (lambda c: (c.loc.numpy(), c.code.numpy()))(addr.encode(torch.from_numpy(X)))
        st_c, sc = enc(b.feats("store", k, ["xf"])), TB.fast_score(L=addr.L, M=addr.M)
        cr = enc(crowd) if n_crowd else None
        out = []
        for q, qlab in (("held", b.held_lab), ("cloze", b.cz_lab)):
            qc = enc(b.feats(q, k, ["xf"]))
            S = sc(qc, st_c)
            ok = b.store_lab[S.argmax(1)] == qlab
            if cr is not None:
                bc = np.full(len(qlab), -np.inf, np.float32)
                for i in range(0, n_crowd, 20000):
                    bc = np.maximum(bc, sc(qc, (cr[0][i:i + 20000], cr[1][i:i + 20000])).max(1))
                ok &= S.max(1) > bc
            out.append(float(ok.mean()))
        print(f"IDEAL {tag}, {addr.side}x{addr.side} places: held top-1 {out[0]:.3f}, cloze top-1 {out[1]:.3f} "
              f"(49 B / entry of codes) ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
