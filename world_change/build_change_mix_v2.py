"""world_change_mix_v2: world_change_mix + IN-CONTEXT questions (a fact asked again inside its own sequence).

    python -B build_change_mix_v2.py                             (X:/Datasets/world_change_mix_v2, 200M train tokens)
    python -B build_change_mix_v2.py --train-seqs 64 --val-seqs 32 --out <tmp>   (a quick test)

V2 CHANGE vs v1 (everything else identical): in ~--ic-frac (0.2) of the world sequences that state facts, 1-2 of the
sequence's stated facts are ALSO asked in the SAME sequence: one combined block
    [EOT] fact sentence [EOT] <gap Wikipedia filler tokens> [EOT] question sentence [EOT]
with gap uniform in --ic-gap (50..400), capped by the room left in the sequence after the fitting step (the gap filler
counts as Wikipedia, not as world tokens). Such a fact is stated only there (not in a normal fact block); its normal
cross-sequence question (1-4 sequences later) is scheduled as before. If no room for a gap >= 50 is left, the fact is
emitted as a plain one-fact block (no in-context question; counted in stats ic_no_room). In the fitting step the
combined block is a fact block (dropping it removes its fact's later questions). All in-context randomness uses a
separate generator (seed + 100). Items get "ic" (1 = in-sequence question: dist 0,
fact_seq == seq; 0 = the v1 cross-sequence questions).

CPU only, 2 threads, nothing downloaded. Reuses fruit-fly-sim-wave27c/generate_world.py (templates, name maker, name
checker; imported read-only, no bytecode) and X:/Datasets/wiki_world_mix (tokenizer, world.json: its 300 cities, 200
jobs, 40 pets are the attribute vocabularies; its 5,000 people are EXCLUDED from the name pools).

WORLD: a cast of 10-20 invented names drawn from a name pool (train pool --train-names, val pool --val-names, disjoint,
and disjoint from wiki_world_mix's people / cities / countries / rivers, the tokenizer vocabulary and Wikipedia words).
Every world samples FRESH attributes for its cast (city, job, pet, birth year 1920-2005, best friend = another cast
member), so the same name has different facts in different worlds. A world is W = 4-8 consecutive 512-token sequences
of ONE stream. Sequences 0..W-2 each state 9-14 of the world's facts (TRAIN templates 0-9, each fact stated once); every
stated fact is asked d = 1..4 sequences later (uniform, capped by the world end and by 16 questions per sequence) with a
HELD-OUT form: "Q: <question> A: <answer>." (generate_world.Q_FORMS) or a held-out cloze template (10/11, subject
before the answer). So the answer is never inside the question's 512-token context: only an across-sequence memory
can carry it. Nothing but the order links fact and question (no ids, labels or markers in the text).
Concurrency: a world's cast never shares a name with any world active at the same positions in another stream (the
memory is shared by all batch rows), so at any time the latest statement of (name, relation) is the gold answer.

SEQUENCE: exactly 512 tokens = Wikipedia filler chunks with world blocks spliced in ([EOT] block [EOT]); the filler is
sized so that ~50% of the tokens are Wikipedia (train: wiki16k train.bin from token 320,000,000 on -- the part
wiki_world_mix does not use -- wrapping to 0; val: wiki16k val.bin).

FORMAT (uint16, eot_id 0, vocab 16384, tokenizer.json = wiki16k):
  train.bin / val.bin: S streams, stream-major, each stream L = n_seq * 512 + 1 tokens:
      stream s, sequence j = tokens[s*L + j*512 : s*L + j*512 + 513]   (x = [:512], y = [1:513])
  A plain random-window loader still works on these files (it just ignores the stream structure).
  A STREAM loader (patch train_lm_stream.patch, --data-mode stream) feeds batch row b, micro-batch c the sequence
  j = c of stream b (S == --batch), i.e. every row continues its own stream across micro-batches and steps.
  {split}_items.npz: one entry per question: stream, seq, pos (sequence offset of the first answer token, >= 1; the
      logit predicting it is at x position pos - 1), n_ans, rel (0 city 1 job 2 pet 3 year 4 friend), form (0 Q:,
      1 cloze), dist (sequences after the fact), fact_seq, fact_pos, world, name (pool index), gold (object index:
      city / job / pet / year - 1920 / friend's pool index), ic (1 = in-sequence question, dist 0)
  {split}_worlds.npz: stream, start, length, cast (W x 20, pool index, -1 pad), city / job / pet / year / friend (pool
      index of the friend), all W x 20
  names.json: {"train": [...], "val": [...]}, meta.json (counts + this format), samples.txt (decoded examples).
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
GW_DIR = Path(r"C:\Eitan\Python\NewBrain\fruit-fly-sim-wave27c")
sys.path.insert(0, str(GW_DIR))
import numpy as np                                  # noqa: E402
from tokenizers import Tokenizer                    # noqa: E402

import generate_world as gw                         # noqa: E402  (read-only import)

SEQ = 512
RELS = gw.RELS                                      # city, job, pet, year, friend
YEAR0, YEAR1 = 1920, 2006
CAST_MAX = 20


class Enc:
    """Token ids of string pieces, cached. Sentences are built as concatenations of pieces split at pre-token
    boundaries (ByteLevel regex pre-tokenizer: a space-prefixed word / a punctuation run starts a new pre-token), so
    the concatenation equals tokenizing the whole text (asserted on samples)."""

    def __init__(self, tok):
        self.tok, self.c = tok, {}

    def __call__(self, s):
        v = self.c.get(s)
        if v is None:
            v = self.c[s] = self.tok.encode(s).ids
        return v


def split_template(t):
    """'{p} lives in {o}.' -> ['{p}', ' lives in', ' {o}', '.'] (trailing spaces moved onto the next piece)."""
    parts = [x for x in re.split(r"(\{[poa]\})", t) if x]
    out, carry = [], ""
    for x in parts:
        x = carry + x
        carry = ""
        if not x.startswith(("{", " {")) and x.endswith(" "):
            x, carry = x[:-1], " "
        if x:
            out.append(x)
    return out


class Render:
    def __init__(self, tok, world):
        self.enc = Enc(tok)
        self.cities = [c["name"] for c in world["cities"]]
        self.jobs, self.pets = world["jobs"], world["pets"]
        self.fact_t = {r: [split_template(t) for t in gw.T_PERSON[r][:gw.N_TRAIN_T]] for r in RELS}
        held = gw.cloze_templates(gw.T_PERSON, RELS, "{p}")     # held-out templates with the subject first
        self.cloze_t = {r: [split_template(gw.T_PERSON[r][ti]) for ti, _, _ in held[r]] for r in RELS}
        self.q_t = {r: split_template(gw.Q_FORMS[r] + " {o}.") for r in RELS}

    def obj(self, rel, v, names):
        return (self.cities[v] if rel == 0 else self.jobs[v] if rel == 1 else self.pets[v] if rel == 2
                else str(v) if rel == 3 else names[v])

    def sentence(self, pieces, p, o, lead):
        """-> (ids, offset of the object's first token, n object tokens)"""
        ids, ans = [], None
        for i, x in enumerate(pieces):
            sp = " " if (lead and i == 0 and not x.startswith(" ")) else ""
            if x in ("{p}", " {p}"):
                s = (" " if x[0] == " " else sp) + p
            elif x in ("{o}", " {o}"):
                s = (" " if x[0] == " " else sp) + o
                ans = (len(ids), len(self.enc(s)))
            elif x in ("{a}", " {a}"):
                s = (" " if x[0] == " " else sp) + gw.article(o)
            else:
                s = sp + x
            ids += self.enc(s)
        return ids, ans

    def text(self, pieces, p, o, lead):
        s = "".join(x.replace("{p}", p).replace("{o}", o).replace("{a}", gw.article(o)) for x in pieces)
        return (" " if lead and not s.startswith(" ") else "") + s


def make_pool(a, tok, world, wiki_train, wiki_val):
    t0 = time.time()
    words = gw.wiki_word_set(np.fromfile(wiki_val, dtype=np.uint16), 10 ** 9, tok)
    if a.check_tokens:
        words |= gw.wiki_word_set(np.memmap(wiki_train, dtype=np.uint16, mode="r"), int(a.check_tokens), tok)
    old = {p["name"].lower() for p in world["people"]} | {c["name"].lower() for c in world["cities"]} \
        | {c.lower() for c in world["countries"]} | {r.lower() for r in world["rivers"]}
    ok = gw.name_checker(tok, words | old)
    names = gw.make_names(np.random.default_rng(a.seed), a.train_names + a.val_names, ok)
    assert not ({n.lower() for n in names} & old)
    print(f"names: {len(names)} ({len(words):,} wiki words + {len(old):,} wiki_world_mix names excluded, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return names[:a.train_names], names[a.train_names:]


class Filler:
    def __init__(self, arr, start):
        self.a, self.pos = arr, int(start) % len(arr)

    def take(self, n):
        out = []
        while n > 0:
            k = min(n, len(self.a) - self.pos)
            out.append(np.asarray(self.a[self.pos:self.pos + k]))
            self.pos = (self.pos + k) % len(self.a)
            n -= k
        return np.concatenate(out) if out else np.zeros(0, np.uint16)


def build_split(a, split, R, names, tok, filler, S, n_seq, seed, eot):
    rng = np.random.default_rng(seed)
    irng = np.random.default_rng(seed + 100)       # v2: all in-context-question randomness
    enc = R.enc
    nm_lead = names
    L = n_seq * SEQ + 1
    out = np.zeros((S, L), np.uint16)
    items = {k: [] for k in ("stream", "seq", "pos", "n_ans", "rel", "form", "dist", "fact_seq", "fact_pos", "world",
                             "name", "gold", "ic")}
    ic_gaps = []
    worlds = {k: [] for k in ("stream", "start", "length", "cast", "city", "job", "pet", "year", "friend")}
    active_end = np.zeros(S, np.int64)             # per stream: end (exclusive) of its current world
    active_cast = [set() for _ in range(S)]
    pending = [None] * S                           # per stream: the current world's generator state
    n_world_tok = n_tok = 0
    check_left = a.check_blocks
    stats = Counter()
    t0 = time.time()

    def new_world(s, j):
        busy = set().union(*(active_cast[t] for t in range(S) if t != s and active_end[t] > j))
        free = np.array([i for i in range(len(names)) if i not in busy])
        assert len(free) >= a.cast[1], "name pool too small for the concurrent casts"
        C = int(rng.integers(a.cast[0], a.cast[1] + 1))
        W = int(rng.integers(a.world_seqs[0], a.world_seqs[1] + 1))
        cast = rng.choice(free, C, replace=False)
        att = dict(city=rng.integers(0, len(R.cities), C), job=rng.integers(0, len(R.jobs), C),
                   pet=rng.integers(0, len(R.pets), C), year=rng.integers(YEAR0, YEAR1, C))
        fr = np.array([(i + int(rng.integers(1, C))) % C for i in range(C)])
        att["friend"] = cast[fr]
        wid = len(worlds["stream"])
        for k, v in (("stream", s), ("start", j), ("length", W)):
            worlds[k].append(v)
        for k in ("cast", "city", "job", "pet", "year", "friend"):
            v = cast if k == "cast" else att[k]
            row = np.full(CAST_MAX, -1, np.int64)
            row[:C] = v
            worlds[k].append(row)
        facts = [(i, r) for i in range(C) for r in range(5)]
        order = rng.permutation(len(facts))
        stated = [[] for _ in range(W)]
        asked = [[] for _ in range(W)]
        k = 0
        for sj in range(W - 1):
            n = int(rng.integers(a.facts_per_seq[0], a.facts_per_seq[1] + 1))
            for _ in range(n):
                if k >= len(facts):
                    break
                f = facts[order[k]]
                k += 1
                stated[sj].append(f)
                ds = [d for d in range(1, min(4, W - 1 - sj) + 1)]
                rng.shuffle(ds)
                for d in ds:
                    if len(asked[sj + d]) < a.q_per_seq:
                        asked[sj + d].append((f, sj, int(rng.random() < 0.5)))
                        break
                else:
                    stats["unasked_cap"] += 1
        values = [att["city"], att["job"], att["pet"], att["year"], att["friend"]]
        pending[s] = dict(wid=wid, j0=j, W=W, cast=cast, values=values, stated=stated, asked=asked, fact_pos={})
        active_end[s] = j + W
        active_cast[s] = set(int(c) for c in cast)

    def obj_of(wd, i, r):
        v = int(wd["values"][r][i])
        return R.obj(r, v, nm_lead), (v - YEAR0 if r == 3 else v)

    def build_seq(s, j):
        nonlocal n_world_tok, n_tok, check_left
        wd = pending[s]
        sj = j - wd["j0"]
        blocks = []        # (ids, [(kind, info, answer offset in block)], [(text, ids) to check], ic info or None)
        st = list(wd["stated"][sj])
        # v2: in-context questions -- 1-2 of this sequence's facts move into combined fact+gap+question blocks
        ic_f = []
        if st:
            stats["seqs_with_facts"] += 1
            if irng.random() < a.ic_frac:
                k = min(len(st), int(irng.integers(1, 3)))
                pick = set(int(x) for x in irng.choice(len(st), k, replace=False))
                ic_f = [f for x, f in enumerate(st) if x in pick]
                st = [f for x, f in enumerate(st) if x not in pick]
        # facts: 1-3 blocks (fact-doc style with an optional header, or story-ish with neutral filler sentences)
        if st:
            nb = int(rng.integers(1, min(3, len(st)) + 1))
            cuts = sorted(rng.choice(np.arange(1, len(st)), nb - 1, replace=False)) if nb > 1 else []
            for g in np.split(np.arange(len(st)), cuts):
                ids, recs, txt = [], [], []
                if rng.random() < 0.4:
                    h = gw.HEADERS[rng.integers(len(gw.HEADERS))] + "\n"
                    ids += enc(h)
                    txt.append(h)
                story = rng.random() < 0.3
                for gi in g:
                    i, r = st[gi]
                    p = nm_lead[int(wd["cast"][i])]
                    o, _ = obj_of(wd, i, r)
                    lead = bool(ids) and not txt[-1].endswith("\n")
                    if story and rng.random() < 0.5:
                        fs = gw.FILLER[rng.integers(len(gw.FILLER))]
                        fs = (" " if lead else "") + fs
                        ids += enc(fs)
                        txt.append(fs)
                        lead = True
                    pcs = R.fact_t[RELS[r]][rng.integers(gw.N_TRAIN_T)]
                    sid, ans = R.sentence(pcs, p, o, lead)
                    recs.append(("fact", (i, r), len(ids) + ans[0]))
                    ids += sid
                    txt.append(R.text(pcs, p, o, lead))
                blocks.append((ids, recs, [("".join(txt), ids)], None))
        # questions: 1-2 blocks; Q: lines joined by newlines, cloze sentences by spaces
        qs = list(wd["asked"][sj])
        if qs:
            rng.shuffle(qs)
            nb = 1 if len(qs) < 4 or rng.random() < 0.5 else 2
            for g in np.array_split(np.arange(len(qs)), nb):
                ids, recs, txt = [], [], []
                for gi in g:
                    (i, r), fsj, form = qs[gi]
                    p = nm_lead[int(wd["cast"][i])]
                    o, gold = obj_of(wd, i, r)
                    if form == 0:
                        pcs = R.q_t[RELS[r]]
                        if ids:
                            ids += enc("\n")
                            txt.append("\n")
                        lead = False
                    else:
                        cl = R.cloze_t[RELS[r]]
                        pcs = cl[rng.integers(len(cl))]
                        lead = bool(ids)
                    sid, ans = R.sentence(pcs, p, o, lead)
                    recs.append(("q", ((i, r), fsj, form, gold, ans[1]), len(ids) + ans[0]))
                    ids += sid
                    txt.append(R.text(pcs, p, o, lead))
                blocks.append((ids, recs, [("".join(txt), ids)], None))
        # v2: combined in-context blocks (gap filler inserted after fitting; ids here = fact + question sentence)
        for i, r in ic_f:
            p = nm_lead[int(wd["cast"][i])]
            o, gold = obj_of(wd, i, r)
            fpcs = R.fact_t[RELS[r]][irng.integers(gw.N_TRAIN_T)]
            fid, fans = R.sentence(fpcs, p, o, False)
            form = int(irng.random() < 0.5)
            if form == 0:
                qpcs = R.q_t[RELS[r]]
            else:
                cl = R.cloze_t[RELS[r]]
                qpcs = cl[irng.integers(len(cl))]
            qid, qans = R.sentence(qpcs, p, o, False)
            blocks.append((fid + qid, [("fact", (i, r), fans[0]), ("icq", ((i, r), sj, form, gold, qans[1]), qans[0])],
                           [(R.text(fpcs, p, o, False), fid), (R.text(qpcs, p, o, False), qid)], dict(nf=len(fid))))

        def blen(b):        # tokens a block takes without its gap filler (+2 outer EOTs, +2 inner EOTs if combined)
            return len(b[0]) + 2 + (2 if b[3] is not None and "gap" not in b[3] else 0)

        rng.shuffle(blocks)
        # fit: drop question blocks, then fact blocks (combined blocks count as fact blocks), until the world part
        # leaves >= min_wiki filler tokens
        while blocks and sum(blen(b) for b in blocks) > SEQ - a.min_wiki:
            qi = [k for k, b in enumerate(blocks) if b[1] and b[1][0][0] == "q"]
            k = qi[-1] if qi else len(blocks) - 1
            stats["dropped_block_" + ("q" if qi else "fact")] += 1
            if not qi:   # a dropped fact block: its later questions go too
                gone = {rec[1] for rec in blocks[k][1]}
                for jj in range(sj + 1, wd["W"]):
                    wd["asked"][jj] = [q for q in wd["asked"][jj] if q[0] not in gone]
            blocks.pop(k)
        # v2: gaps of the combined blocks, uniform in --ic-gap capped by the room left (each later one keeps >= min)
        icb = [k for k, b in enumerate(blocks) if b[3] is not None]
        irng.shuffle(icb)
        gsum = 0
        for n, k in enumerate(icb):
            ids, recs, chk, info = blocks[k]
            nf = info["nf"]
            avail = SEQ - sum(blen(b) for b in blocks)
            cap = min(a.ic_gap[1], avail - a.ic_gap[0] * (len(icb) - 1 - n))
            if cap < a.ic_gap[0]:       # no room: a plain one-fact block, no in-context question
                blocks[k] = (ids[:nf], recs[:1], chk[:1], None)
                stats["ic_no_room"] += 1
                continue
            g = int(irng.integers(a.ic_gap[0], cap + 1))
            w = filler.take(g)
            ids2 = ids[:nf] + [eot] + w.tolist() + [eot] + ids[nf:]
            kind, qinfo, qoff = recs[1]
            blocks[k] = (ids2, [recs[0], (kind, qinfo, nf + 2 + g + qoff)], chk, dict(nf=nf, gap=g))
            ic_gaps.append(g)
            gsum += g
        total = sum(blen(b) for b in blocks)
        wtok = total - gsum              # world tokens (the gap filler is Wikipedia)
        fill = SEQ - total
        # Wikipedia chunks around the blocks: ~target share overall, here whatever is left
        cuts = np.sort(rng.integers(0, fill + 1, len(blocks)))
        sizes = np.diff(np.concatenate([[0], cuts, [fill]]))
        parts, pos = [], 0
        for bi in range(len(blocks) + 1):
            w = filler.take(int(sizes[bi]))
            parts.append(w)
            pos += len(w)
            if bi < len(blocks):
                ids, recs, chk, info_b = blocks[bi]
                base = pos + 1
                parts.append(np.array([eot] + ids + [eot], np.uint16))
                for kind, info, off in recs:
                    if kind == "fact":
                        wd["fact_pos"][info] = (sj, base + off)
                    else:
                        (i, r), fsj, form, gold, n_ans = info
                        ic = int(kind == "icq")
                        fpos = wd["fact_pos"].get((i, r), (fsj, -1))
                        assert fpos[0] == fsj
                        assert not ic or (fsj == sj and 0 <= fpos[1] < base + off)
                        for kk, v in (("stream", s), ("seq", j), ("pos", base + off), ("n_ans", n_ans), ("rel", r),
                                      ("form", form), ("dist", sj - fsj), ("fact_seq", fsj + wd["j0"]),
                                      ("fact_pos", fpos[1]), ("world", wd["wid"]), ("name", int(wd["cast"][i])),
                                      ("gold", gold), ("ic", ic)):
                            items[kk].append(v)
                if check_left > 0:
                    check_left -= 1
                    for txt, cids in chk:
                        ref = tok.encode(txt).ids
                        assert ref == cids, ("piecewise tokenization differs", txt, ref[:40], cids[:40])
                pos += len(ids) + 2
        if any(b[3] is not None for b in blocks):
            stats["seqs_with_ic"] += 1
        seq = np.concatenate(parts)
        assert len(seq) == SEQ, len(seq)
        n_world_tok += wtok
        n_tok += SEQ
        stats["facts_stated"] += sum(1 for b in blocks for rec in b[1] if rec[0] == "fact")
        return seq

    for j in range(n_seq):
        for s in range(S):
            if active_end[s] <= j:
                new_world(s, j)
            out[s, j * SEQ:(j + 1) * SEQ] = build_seq(s, j)
        if j % max(1, n_seq // 10) == 0:
            print(f"  {split}: seq {j}/{n_seq} x {S} streams, world share {n_world_tok / max(n_tok, 1):.3f}, "
                  f"{len(worlds['stream']):,} worlds, {len(items['stream']):,} questions ({time.time() - t0:.0f}s)",
                  flush=True)
    out[:, -1] = eot
    # questions whose world ran past the stream end are simply never emitted (their sequence does not exist)
    it = {k: np.asarray(v, np.int64) for k, v in items.items()}
    wo = {k: np.asarray(v, np.int64) for k, v in worlds.items()}
    st = dict(split=split, streams=S, seqs_per_stream=n_seq, stream_len=L, tokens=int(out.size),
              world_share=n_world_tok / n_tok, worlds=len(wo["stream"]), questions=len(it["stream"]),
              facts_stated=int(stats.pop("facts_stated", 0)), people=len(names),
              people_used=int(len(np.unique(wo["cast"][wo["cast"] >= 0]))),
              questions_by_rel={RELS[r]: int((it["rel"] == r).sum()) for r in range(5)},
              questions_by_dist={int(d): int((it["dist"] == d).sum()) for d in range(0, 5)},
              questions_by_form={"Q": int((it["form"] == 0).sum()), "cloze": int((it["form"] == 1).sum())},
              questions_ic={"ic": int((it["ic"] == 1).sum()), "cross": int((it["ic"] == 0).sum())},
              ic_seq_frac=stats["seqs_with_ic"] / max(stats["seqs_with_facts"], 1),
              ic_gap=dict(min=int(min(ic_gaps, default=0)), mean=float(np.mean(ic_gaps)) if ic_gaps else 0.0,
                          max=int(max(ic_gaps, default=0))),
              stats=dict(stats), time_s=time.time() - t0)
    return out, it, wo, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="X:/Datasets/wiki_world_mix")
    ap.add_argument("--wiki", default="X:/Datasets/wikipedia_en_20231101_wiki16k")
    ap.add_argument("--out", default="X:/Datasets/world_change_mix_v2")
    ap.add_argument("--streams", type=int, default=16, help="parallel streams = the trainer's --batch")
    ap.add_argument("--train-seqs", type=int, default=24414, help="sequences per train stream (12,207 steps x accum 2)")
    ap.add_argument("--val-seqs", type=int, default=256, help="sequences per val stream")
    ap.add_argument("--train-names", type=int, default=1000)
    ap.add_argument("--val-names", type=int, default=600)
    ap.add_argument("--cast", type=int, nargs=2, default=[10, 20])
    ap.add_argument("--world-seqs", type=int, nargs=2, default=[4, 8])
    ap.add_argument("--facts-per-seq", type=int, nargs=2, default=[9, 14])
    ap.add_argument("--q-per-seq", type=int, default=16)
    ap.add_argument("--min-wiki", type=int, default=64, help="minimum Wikipedia tokens per sequence")
    ap.add_argument("--wiki-start", type=float, default=320e6, help="train filler start in wiki16k train.bin")
    ap.add_argument("--check-tokens", type=float, default=20e6, help="Wikipedia train tokens scanned for the name check")
    ap.add_argument("--check-blocks", type=int, default=3000, help="blocks whose piecewise ids are checked vs encode")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--ic-frac", type=float, default=0.2, help="v2: share of fact sequences with in-context questions")
    ap.add_argument("--ic-gap", type=int, nargs=2, default=[50, 400],
                    help="v2: Wikipedia tokens between an in-context fact sentence and its question")
    a = ap.parse_args()
    t0 = time.time()
    src, wiki, out = Path(a.src), Path(a.wiki), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smeta = json.loads((src / "meta.json").read_text())
    assert smeta["dtype"] == "uint16" and smeta["vocab"] == 16384
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    eot = tok.token_to_id(gw.EOT)
    assert eot == smeta["eot_id"] == 0
    world = json.loads((src / "world.json").read_text())
    train_names, val_names = make_pool(a, tok, world, wiki / "train.bin", wiki / "val.bin")
    R = Render(tok, world)
    for r in RELS:
        assert R.cloze_t[r], r
    res = {}
    for split, names, S, n, fil, seed in (
            ("val", val_names, a.streams, a.val_seqs, Filler(np.fromfile(wiki / "val.bin", np.uint16), 0), a.seed + 1),
            ("train", train_names, a.streams, a.train_seqs,
             Filler(np.memmap(wiki / "train.bin", dtype=np.uint16, mode="r"), a.wiki_start), a.seed + 2)):
        arr, it, wo, st = build_split(a, split, R, names, tok, fil, S, n, seed, eot)
        arr.tofile(out / f"{split}.bin")
        np.savez(out / f"{split}_items.npz", **it)
        np.savez(out / f"{split}_worlds.npz", **wo)
        res[split] = (arr, it, wo, st)
        print(f"{split}: {st}", flush=True)
    (out / "names.json").write_text(json.dumps({"train": train_names, "val": val_names}))
    (out / "tokenizer.json").write_bytes((src / "tokenizer.json").read_bytes())

    # samples: for val, stream 0: the first 3 questions with their fact sentence (decoded windows)
    lines = []
    arr, it, wo, st = res["val"]
    L = arr.shape[1]
    cross = np.flatnonzero(it["ic"] == 0)[:6].tolist()
    icq = np.flatnonzero(it["ic"] == 1)[:3].tolist()
    for q in cross + icq:
        if q == (icq[0] if icq else -1):
            lines.append("\n=== in-context questions (ic=1: fact and question in the SAME sequence) ===")
        s, j, p, n, fs, fp = (int(it[k][q]) for k in ("stream", "seq", "pos", "n_ans", "fact_seq", "fact_pos"))
        fseq = arr[s, fs * SEQ: fs * SEQ + SEQ]
        qseq = arr[s, j * SEQ: j * SEQ + SEQ]
        fwin = tok.decode(fseq[max(0, fp - 14): fp + n + 3].tolist(), skip_special_tokens=False)
        qwin = tok.decode(qseq[max(0, p - 14): p + n + 2].tolist(), skip_special_tokens=False)
        lines.append(f"[stream {s}] FACT seq {fs} pos {fp}: ...{fwin!r}...\n"
                     f"           QUESTION seq {j} pos {p} (dist {j - fs}, {f'ic, pos - fact_pos {p - fp}' if it['ic'][q] else 'cross-seq'}, {'Q' if it['form'][q] == 0 else 'cloze'}, "
                     f"{RELS[it['rel'][q]]}): ...{qwin!r}...  answer {tok.decode(qseq[p:p + n].tolist())!r}")
    lines.append("\n=== val stream 0, sequence 1, full ===\n" + tok.decode(arr[0, SEQ:2 * SEQ].tolist(), skip_special_tokens=False))
    (out / "samples.txt").write_text("\n".join(lines), encoding="utf-8")

    meta = dict(
        name="world_change_mix_v2", tokenizer="wiki16k", vocab=16384, eot_id=eot, dtype="uint16", seq=SEQ,
        format="stream-major: train.bin / val.bin hold `streams` streams of stream_len = seqs_per_stream * seq + 1 "
               "tokens; stream s sequence j = tokens[s*stream_len + j*seq : s*stream_len + j*seq + seq + 1]. "
               "Stream loader (--data-mode stream): batch row b of micro-batch c = stream b (streams == --batch), "
               "sequence c mod seqs_per_stream.",
        streams=a.streams,
        train=dict(res["train"][3], tokens=int(res["train"][0].size)),
        val=dict(res["val"][3], tokens=int(res["val"][0].size)),
        train_tokens=int(res["train"][0].size), val_tokens=int(res["val"][0].size),
        args=vars(a), source=f"generated (build_change_mix_v2.py) + Wikipedia filler from {wiki}",
        note="v2 = world_change_mix + in-context questions. items field ic: 1 = the question is asked in the SAME "
             "sequence as its fact (dist 0, fact_seq == seq), in a combined block [EOT] fact [EOT] <gap Wikipedia "
             "tokens, uniform in args.ic_gap capped by the room left> [EOT] question [EOT], in ~args.ic_frac of the "
             "fact-bearing world sequences (1-2 facts each; such a fact is stated only there and is still asked "
             "again 1-4 sequences later as an ic=0 question); 0 = the v1 cross-sequence questions (dist 1-4). "
             "pos - fact_pos of an ic item = gap + the question-side tokens between.",
        relations=RELS, year0=YEAR0, done=True)
    for split in ("train", "val"):
        meta[split]["stream_len"] = int(res[split][0].shape[1])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("\n".join(lines[:10]))
    print(f"done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
