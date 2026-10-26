"""Diagnostics for the FastMem changing-facts runs (CPU, 2 threads; run on a COPY of a checkpoint, never the live run).

    python -B diag_change_fastmem.py --ckpt <copy>/last.pt [--n-ic 240] [--streams 8] [--seqs 32]

(1) IN-CONTEXT control, memory off: a val fact sentence (train template) placed in the SAME 512-token sequence just before
    its question (the held-out Q:/cloze form, val people), vs the fact ~250 filler tokens earlier, vs no fact; also the
    adjacent case with memory on (in-sequence scan only). Shows whether the brain can answer at all when the fact is visible.
(2) MEMORY PROBE: on_write over the val streams; at every question sequence, a second forward that reads ANOTHER stream's
    memory (streams rolled by one; not committed) and the zero-memory read. Gold logp / crank: true vs shuffled vs zero.
(3) TRAIN SIDE: (name, relation) -> answer repetition across train worlds (can weights learn answers?), and the main CE
    (memory off, eval mode, the eval K) on val sequences vs train sequences already trained on vs not yet reached.
"""
import sys
from pathlib import Path

import eval_change_fastmem as E                     # (sets threads / CUDA off / the snapshot path from --ckpt)
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import json
import time
import types
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_change_mix as B                        # noqa: E402  (Render: the dataset's own templates)

ew, memory_api = E.ew, E.memory_api
SEQ, EOT = 512, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="X:/Datasets/world_change_mix")
    ap.add_argument("--src", default="X:/Datasets/wiki_world_mix")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n-ic", type=int, default=240)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--ce-seqs", type=int, default=8)
    ap.add_argument("--parts", default="1,2,3")
    ap.add_argument("--out", default="")
    a, _ = ap.parse_known_args()
    parts = set(a.parts.split(","))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t_all = time.time()
    data = Path(a.data)
    meta = json.loads((data / "meta.json").read_text())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))
    world = json.loads((Path(a.src) / "world.json").read_text())
    names_all = json.loads((data / "names.json").read_text())
    names = names_all["val"]
    R = B.Render(tok, world)
    S_val, Lv = meta["val"]["streams"], meta["val"]["stream_len"]
    val = np.fromfile(data / "val.bin", dtype=np.uint16).reshape(S_val, Lv)
    items = dict(np.load(data / "val_items.npz"))
    wo = dict(np.load(data / "val_worlds.npz"))
    first = lambda s: tok.encode(s).ids[0]
    cf_fixed = [sorted({first(" " + c["name"]) for c in world["cities"]}), sorted({first(" " + j) for j in world["jobs"]}),
                sorted({first(" " + p) for p in world["pets"]}), sorted({first(f" {y}") for y in range(1920, 2006)})]
    name_first = [first(" " + n) for n in names]

    def cands(ii):
        r = int(items["rel"][ii])
        if r == 4:
            cast = wo["cast"][items["world"][ii]]
            return sorted({name_first[c] for c in cast if c >= 0 and c != items["name"][ii]})
        return cf_fixed[r]

    la = types.SimpleNamespace(ckpt=a.ckpt, config="", capacity="dropless", device="cpu", k=a.k)
    model, step, _, _, _ = ew.load_model(la)
    bm = model.bm
    mpath = memory_api.memory_path(Path(a.ckpt))
    memory_api.load_memory_file(mpath, model, strict=True)
    model.eval()
    h0 = E.brain_checksum(model), E.mem_param_checksum(bm)
    print(f"[diag] step {step}, K {model.K} ({time.time() - t_all:.0f}s)", flush=True)
    out, lines = dict(step=step), [f"DIAG changing facts, FastMem: {a.ckpt} (step {step}), K {a.k}"]

    def agg(res):
        v = np.array([(lp / n, ex, cr == 1, cr, 1 / nc) for lp, ex, cr, nc, n in res], float)
        return dict(n=len(v), exact=v[:, 1].mean(), crank1=v[:, 2].mean(), crank_mean=v[:, 3].mean(),
                    logp_tok=v[:, 0].mean(), chance1=v[:, 4].mean())

    def fmt(name, d):
        return (f"{name:28s} n {d['n']:4d}  exact {d['exact']:.4f}  crank1 {d['crank1']:.4f} (chance {d['chance1']:.4f})"
                f"  mean crank {d['crank_mean']:.1f}  logp/tok {d['logp_tok']:.3f}")

    def score(lg, toks, pos, n, cf):
        lp = F.log_softmax(lg[pos - 1:pos - 1 + n].float(), -1)
        gold = torch.as_tensor(toks[pos:pos + n].astype(np.int64))
        g0 = lp[0, gold[0]]
        crank = int((lp[0, torch.as_tensor(cf)] > g0).sum()) + 1
        return float(lp.gather(1, gold[:, None]).sum()), bool((lp.argmax(-1) == gold).all()), crank, len(cf), n

    # ------------------------------------------------------------ (1) in-context control
    if "1" in parts:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(items["stream"]), a.n_ic, replace=False)
        enc = R.enc

        def obj(ii):
            r, g = int(items["rel"][ii]), int(items["gold"][ii])
            return R.obj(r, g + 1920 if r == 3 else g, names)

        def build(ii, variant):
            r, p, o = int(items["rel"][ii]), names[int(items["name"][ii])], obj(ii)
            fpcs = R.fact_t[B.RELS[r]][int(rng.integers(B.gw.N_TRAIN_T))]
            fid, _ = R.sentence(fpcs, p, o, False)
            if int(items["form"][ii]) == 0:
                qid, (qoff, qn) = R.sentence(R.q_t[B.RELS[r]], p, o, False)
            else:
                cl = R.cloze_t[B.RELS[r]]
                qid, (qoff, qn) = R.sentence(cl[int(rng.integers(len(cl)))], p, o, False)
            s, j = int(items["stream"][ii]), int(items["seq"][ii])
            wiki = [int(t) for t in val[s, j * SEQ:j * SEQ + SEQ + 1]]   # (the question's own sequence as filler)
            fact = [EOT] + fid + [EOT] if variant != "nofact" else []
            gap = wiki[:250] if variant == "far" else []
            tail = fact + gap + [EOT] + qid + [EOT]
            prefix = wiki[250 if variant == "far" else 0:][:SEQ + 1 - len(tail)]
            seq = np.array(prefix + tail, np.int64)
            assert len(seq) == SEQ + 1
            pos = len(prefix) + len(fact) + len(gap) + 1 + qoff
            assert list(seq[pos:pos + qn]) == qid[qoff:qoff + qn]
            return seq, pos, qn

        variants = [("adjacent", "adj", "off"), ("far250", "far", "off"), ("nofact", "nofact", "off"),
                    ("adjacent_mem_on", "adj", "on")]
        R1 = {}
        for vname, variant, mm in variants:
            t0 = time.time()
            rng = np.random.default_rng(1)                   # (same templates per variant)
            built = [(build(ii, variant), ii) for ii in pick]
            res, by_rel = [], defaultdict(list)
            ctx = memory_api.memory_off(model) if mm == "off" else torch.no_grad()
            with ctx, torch.no_grad():
                for b0 in range(0, len(built), 8):
                    chunk = built[b0:b0 + 8]
                    toks = np.stack([c[0][0] for c in chunk])
                    lg = model(torch.from_numpy(toks[:, :SEQ]))
                    for k, ((seq, pos, n), ii) in enumerate(chunk):
                        sc = score(lg[k], toks[k], pos, n, cands(ii))
                        res.append(sc)
                        by_rel[E.RELS[int(items["rel"][ii])]].append(sc)
            R1[vname] = dict(ALL=agg(res), **{k: agg(v) for k, v in sorted(by_rel.items())})
            lines.append(fmt(f"(1) {vname}", R1[vname]["ALL"]) + f"  ({time.time() - t0:.0f}s)")
            for k, v in sorted(by_rel.items()):
                lines.append(fmt(f"      {vname} {k}", agg(v)))
            print(lines[-len(by_rel) - 1], flush=True)
        out["in_context"] = R1

    # ------------------------------------------------------------ (2) memory probe
    if "2" in parts:
        Bn, J = a.streams, a.seqs
        by_seq = defaultdict(list)
        for ii in range(len(items["stream"])):
            s, j = int(items["stream"][ii]), int(items["seq"][ii])
            if s < Bn and j < J:
                by_seq[j].append((s, ii))
        res = {m: [] for m in ("true", "shuffled", "zero")}
        dist_d = defaultdict(list)
        stats = defaultdict(list)
        by_key = defaultdict(lambda: {m: [] for m in ("true", "shuffled", "zero")})
        distr = {m: [] for m in ("true", "shuffled", "zero")}      # friend: rank among the stream's recent names

        def rank_in(lg_row, pos, gold0, cset):
            lp = lg_row[pos - 1].float()
            return int((lp[torch.as_tensor(sorted(cset))] > lp[gold0]).sum()) + 1
        perm = [(b + 1) % Bn for b in range(Bn)]
        t0 = time.time()
        live = E.Live(model)
        with live, torch.no_grad():
            for j in range(J):
                toks = val[:Bn, j * SEQ:j * SEQ + SEQ + 1].astype(np.int64)
                x = torch.from_numpy(toks[:, :-1])
                qs = by_seq.get(j, [])
                if qs and j > 0:
                    bm.set_streams(perm, [False] * Bn)        # another stream's memory, not committed
                    lg_s = model(x)
                    bm._pend = None
                    live_bf = bm.begin_forward                # zero start memory (= on_nowrite), not committed
                    bm.begin_forward = _dead_begin(bm)
                    lg_z = model(x)
                    bm._pend = None
                    bm.begin_forward = live_bf
                live.before(Bn, j)
                lg = model(x)
                stats["open_e"].append(float(bm._stats.get("open_e", float("nan"))))
                stats["open_g"].append(float(bm._stats.get("open_g", float("nan"))))
                stats["read_ratio"].append(float(bm._stats.get("read_ratio", float("nan"))))
                stats["read_ratio_s"].append(float(bm._stats.get("read_ratio_s", float("nan"))))
                live.after()
                if qs and j > 0:
                    for s, ii in qs:
                        pos, n, cf = int(items["pos"][ii]), int(items["n_ans"][ii]), cands(ii)
                        a_t, a_s, a_z = (score(L[s], toks[s], pos, n, cf) for L in (lg, lg_s, lg_z))
                        res["true"].append(a_t)
                        res["shuffled"].append(a_s)
                        res["zero"].append(a_z)
                        dist_d[int(items["dist"][ii])].append((a_t[0] - a_s[0]) / n)
                        r_ = int(items["rel"][ii])
                        for key in (f"rel_{E.RELS[r_]}", f"dist_{int(items['dist'][ii])}"):
                            for m, a_ in (("true", a_t), ("shuffled", a_s), ("zero", a_z)):
                                by_key[key][m].append(a_)
                        if r_ == 4:              # distractors: cast names seen in this stream's last 4 sequences
                            g0 = int(toks[s, pos])
                            recent = set(val[s, max(0, j - 4) * SEQ:j * SEQ].tolist())
                            cs = {c for c in cf if c in recent} | {g0}
                            if len(cs) >= 2:
                                for m, L in (("true", lg), ("shuffled", lg_s), ("zero", lg_z)):
                                    distr[m].append((rank_in(L[s], pos, g0, cs), len(cs)))
                if j % 8 == 0 or j == J - 1:
                    print(f"  probe seq {j + 1}/{J}, {len(res['true'])} q ({time.time() - t0:.0f}s)", flush=True)
        R2 = {m: agg(v) for m, v in res.items()}
        d = np.array([(t[0] - s_[0]) / t[4] for t, s_ in zip(res["true"], res["shuffled"])])
        dz = np.array([(t[0] - z[0]) / t[4] for t, z in zip(res["true"], res["zero"])])
        R2["true_minus_shuffled_logp_tok"] = dict(mean=d.mean(), sem=d.std() / np.sqrt(len(d)), frac_pos=(d > 0).mean(),
                                                  abs_mean=np.abs(d).mean())
        R2["true_minus_zero_logp_tok"] = dict(mean=dz.mean(), abs_mean=np.abs(dz).mean())
        R2["by_dist_true_minus_shuffled"] = {k: float(np.mean(v)) for k, v in sorted(dist_d.items())}
        R2["gate_read"] = {k: float(np.nanmean(v)) for k, v in stats.items()}
        R2["by_key"] = {k: {m: agg(v) for m, v in d_.items()} for k, d_ in sorted(by_key.items())}
        R2["friend_vs_recent_names"] = {m: dict(n=len(v), top1=float(np.mean([r == 1 for r, _ in v])) if v else None,
                                                mean_rank=float(np.mean([r for r, _ in v])) if v else None,
                                                chance_top1=float(np.mean([1 / c for _, c in v])) if v else None,
                                                mean_set=float(np.mean([c for _, c in v])) if v else None)
                                        for m, v in distr.items()}
        for m in ("true", "shuffled", "zero"):
            lines.append(fmt(f"(2) memory {m}", R2[m]))
        tm = R2["true_minus_shuffled_logp_tok"]
        lines.append(f"(2) gold logp/tok true - shuffled: {tm['mean']:+.4f} +- {tm['sem']:.4f} (mean |diff| "
                     f"{tm['abs_mean']:.4f}, true better in {tm['frac_pos']:.3f}); true - zero: "
                     f"{R2['true_minus_zero_logp_tok']['mean']:+.4f} (|diff| {R2['true_minus_zero_logp_tok']['abs_mean']:.4f})"
                     f"; by dist {', '.join(f'{k}: {v:+.4f}' for k, v in R2['by_dist_true_minus_shuffled'].items())}")
        lines.append("(2) gate / read: " + ", ".join(f"{k} {v:.4f}" for k, v in R2["gate_read"].items())
                     + f"  ({time.time() - t0:.0f}s)")
        for k, d_ in R2["by_key"].items():
            lines.append(f"(2) {k:12s} n {d_['true']['n']:4d}  crank1 true/shuf/zero " + "/".join(
                f"{d_[m]['crank1']:.4f}" for m in ("true", "shuffled", "zero")) + "  exact " + "/".join(
                f"{d_[m]['exact']:.4f}" for m in ("true", "shuffled", "zero")) + "  logp/tok " + "/".join(
                f"{d_[m]['logp_tok']:.3f}" for m in ("true", "shuffled", "zero")) + f"  (chance {d_['true']['chance1']:.4f})")
        fr = R2["friend_vs_recent_names"]
        lines.append("(2) friend gold vs the cast names seen in the stream's last 4 seqs: " + ", ".join(
            f"{m} top1 {fr[m]['top1']:.4f} mean rank {fr[m]['mean_rank']:.2f}" for m in ("true", "shuffled", "zero")
            if fr[m]["n"]) + f" (n {fr['true']['n']}, set {fr['true']['mean_set'] or 0:.1f}, chance top1 "
            f"{fr['true']['chance_top1'] or 0:.4f})")
        print("\n".join(lines[-20:]), flush=True)
        out["probe"] = R2

    # ------------------------------------------------------------ (3) train side
    if "3" in parts:
        t0 = time.time()
        ti = np.load(data / "train_items.npz")
        nm, rl, gd, wd = ti["name"], ti["rel"], ti["gold"], ti["world"]
        key = nm.astype(np.int64) * 5 + rl
        n_keys = len(np.unique(key))
        # modal answer per (name, rel) over all train questions, and how often the gold equals it
        Ga = int(gd.max()) + 1
        modal_hits = int(np.bincount(key * Ga + gd, minlength=(int(key.max()) + 1) * Ga).reshape(-1, Ga).max(1).sum())
        cast = np.load(data / "train_worlds.npz")["cast"]
        wpn = np.bincount(cast[cast >= 0])
        wpn = wpn[wpn > 0]
        # held out: the modal answer of (name, rel) over the EVEN worlds, hit rate on the ODD worlds' questions
        ev = wd % 2 == 0
        G = int(gd.max()) + 1
        nk = int(key.max()) + 1
        cnt = np.bincount(key[ev] * G + gd[ev], minlength=nk * G).reshape(nk, G)
        modal = cnt.argmax(1)
        ho = {}
        for r in range(5):
            m = (~ev) & (rl == r)
            ho[E.RELS[r]] = dict(heldout_modal_hit=float((modal[key[m]] == gd[m]).mean()),
                                 uniform=1 / [300, 200, 40, 86, 1000][r])
        R3 = dict(train_people=len(names_all["train"]), val_people=len(names), overlap=len(set(names_all["train"]) &
                  set(names)), name_rel_keys=n_keys, questions=int(len(key)),
                  worlds_per_name=dict(mean=float(wpn.mean()), min=int(wpn.min()), max=int(wpn.max())),
                  modal_answer_rate=modal_hits / len(key), heldout=ho)
        lines.append(f"(3) train people {R3['train_people']}, val people {R3['val_people']} (shared {R3['overlap']}); "
                     f"each train name in {wpn.mean():.0f} worlds (min {wpn.min()}, max {wpn.max()}); a question's gold "
                     f"= the most common answer of its (name, relation) over ALL train (in-sample): "
                     f"{R3['modal_answer_rate']:.4f}")
        lines.append("(3) held-out (modal from even worlds, scored on odd worlds) vs uniform: " + ", ".join(
            f"{k} {v['heldout_modal_hit']:.4f}/{v['uniform']:.4f}" for k, v in ho.items()))
        # CE: val vs train seen vs train unseen, memory off; all tokens and answer tokens
        S_tr, Lt = meta["train"]["streams"], meta["train"]["stream_len"]
        train = np.memmap(data / "train.bin", dtype=np.uint16, mode="r").reshape(S_tr, Lt)
        seen_upto = step * 2                                  # (accum 2: micro-batch c = sequence c of every stream)
        q_by = defaultdict(list)
        sel = {"val": (val, list(range(100, 100 + a.ce_seqs)), items),
               "train_seen": (train, list(range(seen_upto - 200, seen_upto - 200 + a.ce_seqs)), ti),
               "train_unseen": (train, list(range(20000, 20000 + a.ce_seqs)), ti)}
        R3["ce"] = {}
        for nmk, (arr, seqs, it) in sel.items():
            js = set(seqs)
            st, sq, ps, na, rr = it["stream"], it["seq"], it["pos"], it["n_ans"], it["rel"]
            m = (st < 8) & np.isin(sq, list(js))
            amask = {}
            for s, j, p, n, r in zip(st[m], sq[m], ps[m], na[m], rr[m]):
                amask.setdefault((int(s), int(j)), []).extend((q, int(r)) for q in range(int(p), int(p) + int(n)))
            ce_all, ce_ans, ce_rel = [], [], defaultdict(list)
            with memory_api.memory_off(model), torch.no_grad():
                for j in seqs:
                    toks = np.stack([np.asarray(arr[s, j * SEQ:j * SEQ + SEQ + 1]) for s in range(8)]).astype(np.int64)
                    lg = model(torch.from_numpy(toks[:, :-1]))
                    ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), torch.from_numpy(toks[:, 1:]).reshape(-1),
                                         reduction="none").view(8, SEQ)
                    ce_all.append(ce.mean().item())
                    for s in range(8):
                        for p, r in amask.get((s, j), []):
                            ce_ans.append(ce[s, p - 1].item())
                            ce_rel[E.RELS[r]].append(ce_ans[-1])
            R3["ce"][nmk] = dict(all=float(np.mean(ce_all)), answers=float(np.mean(ce_ans)) if ce_ans else None,
                                 n_ans_tok=len(ce_ans), seqs=[seqs[0], seqs[-1]],
                                 by_rel={k: float(np.mean(v)) for k, v in sorted(ce_rel.items())})
            lines.append(f"(3) CE memory off, {nmk:12s} seqs {seqs[0]}-{seqs[-1]} x 8 streams: all tokens "
                         f"{R3['ce'][nmk]['all']:.4f}, answer tokens {R3['ce'][nmk]['answers']:.4f} "
                         f"(n {len(ce_ans)}; " + ", ".join(f"{k} {v:.2f}" for k, v in R3["ce"][nmk]["by_rel"].items())
                         + ")")
            print(lines[-1], flush=True)
        lines.append(f"(3) ({time.time() - t0:.0f}s)")
        out["train_side"] = R3

    assert (E.brain_checksum(model), E.mem_param_checksum(bm)) == h0, "weights changed"
    lines.append(f"weights unchanged; total {time.time() - t_all:.0f}s")
    txt = "\n".join(lines)
    print(txt, flush=True)
    o = Path(a.out) if a.out else Path(__file__).resolve().parent / f"diag_fmb_s{step}.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    o.with_suffix(".txt").write_text(txt + "\n")
    print(f"wrote {o}", flush=True)


def _dead_begin(bm):
    def begin_forward():
        bm._live = False
        bm._fw = None
    return begin_forward


if __name__ == "__main__":
    main()
