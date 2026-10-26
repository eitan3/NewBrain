"""NEW-FACTS memory eval for the wave-memory run (SPEC_v22 item 1): can the trained memory store facts it never saw?

    CPU check: python -B run_newfacts_wave.py --n-facts 20 --seq 128 --study-batch 4        (CUDA hidden, 2 threads)
    GPU:       python -B run_newfacts_wave.py --device cuda                                  (16 GB cap, 2 threads)

Model: <ckpt> (default runs/world_wave26_12k/final.pt) with its code from <run>/code_snapshot, loaded through
run_eval_wave.py (same knockouts mem_on / ko_wave / private_only / global_only, same read masks). The memory is loaded
from a COPY of <ckpt>.memory.pt (--mem-copy, default in the scratch dir); the original's sha256 is checked before and
after. Nothing is ever written into runs/.

FACTS: --n-facts NEW people (default 2,000), syllable names from generate_world.make_names with a different seed,
rejected if in the world file (people, cities, countries, rivers), in the tokenizer vocabulary, a single token, or a
word of Wikipedia val + the first --check-tokens of the Wikipedia training stream. Person i gets ONE fact, relation
i % 5 (city / job / pet / year / friend; objects from the world's cities / jobs / pets / years 1920-2005; a friend is
another new person).
STUDY: each fact 3x (--reps), each time a random TRAIN template (0-9), as its own document ([EOT] fact [EOT]) inside
Wikipedia val filler: one fact per sequence, each sequence stands alone (no state carries between rows or micro-
batches; the model has --state none), study order shuffled. The brain is in eval mode, all brain parameters
requires_grad False, no optimizer; the memory module alone is put in its training (live) state with exploration 0 and
sequence drop 0, so its own forward / select_writes / after_backward run the SAME write rule as training (surprise
policy, 1 write per sequence, 16 per micro-batch, confirmations, the global map + private address learning on writes).
The write target needs E_pair = dL/dy_pair: one backward of the LM loss restricted to the memory's capture leaf
(loss.backward(inputs=[probe])). After every micro-batch: memory parameter .grad = None (the gate loss computes them;
they are never applied), D_grad zeroed; after_step (AdamW, D update, renorm, sweep) is never called.
The brain checksum (parameters and buffers, everything outside bm.*) is asserted unchanged.
QUIZ (after study, then the world's old facts for interference): cloze with the HELD-OUT templates (10, 11; the ones
with the subject before the answer), after [EOT]: exact (all answer tokens argmax), first token, log p / token; ranking
of the gold among the relation's candidates (city 300, job 200 with article, pet 40, year 86, friend: gold + 100
random new names) on the first --rank-limit items per relation. Modes: mem_on / ko_wave / private_only / global_only.
Also before study (mem_on) for the new facts and the old ones, so the old-fact change = interference.
Writes JSON + .txt to world_evals/<run>_<ckpt>_newfacts[_tag].json.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
if not any(x == "--out" or x.startswith("--out=") for x in sys.argv):
    sys.argv.append("--out=")                       # (else run_eval_wave adds its own --out: never reuse its file name)
import run_eval_wave as rw                         # noqa: E402  (env: CUDA hidden unless --device cuda, 2 threads,
                                                    # the 16 GB cap before any CUDA allocation; the knockout patches)
import argparse                                     # noqa: E402
import hashlib                                      # noqa: E402
import json                                         # noqa: E402
import re                                           # noqa: E402
import shutil                                       # noqa: E402
import time                                         # noqa: E402
import types                                        # noqa: E402
from collections import defaultdict                 # noqa: E402

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import torch.nn.functional as F                     # noqa: E402

import eval_world as ew                             # noqa: E402  (the snapshot's)
import generate_world as gw                         # noqa: E402  (the snapshot's: templates, names)
import memory_api                                   # noqa: E402
from wave_v21 import WaveV21Memory                  # noqa: E402

SCRATCH = Path(r"C:\Users\eitan\AppData\Local\Temp\claude\C--Eitan-Python-NewBrain"
               r"\de301b72-951d-49a4-9895-252fb20d74d6\scratchpad\newfacts")
MODES = [("mem_on", None), ("ko_wave", "off"), ("private_only", "private_only"), ("global_only", "global_only")]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\Eitan\Python\NewBrain\runs\world_wave26_12k\final.pt")
    ap.add_argument("--config", default="")
    ap.add_argument("--data", default="X:/Datasets/wiki_world_mix")
    ap.add_argument("--wiki-train", default="X:/Datasets/wikipedia_en_20231101_wiki16k/train.bin")
    ap.add_argument("--check-tokens", type=float, default=20e6, help="Wikipedia train tokens scanned for the name check")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--capacity", default="dropless")
    ap.add_argument("--n-facts", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seq", type=int, default=256, help="study sequence length (tokens)")
    ap.add_argument("--study-batch", type=int, default=16, help="sequences per study micro-batch (training: 16)")
    ap.add_argument("--addr-learn", type=int, default=1,
                    help="1 = keep after_backward's address update on the read rows (as in training)")
    ap.add_argument("--rank-limit", type=int, default=40, help="ranked new-fact items per relation")
    ap.add_argument("--friend-distractors", type=int, default=100)
    ap.add_argument("--old-limit", type=int, default=0, help="old world cloze items per type (0 = all)")
    ap.add_argument("--pre", type=int, default=1, help="1 = also quiz before study (mem_on)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rank-batch", type=int, default=256)
    ap.add_argument("--mem-copy", default="", help="where the memory file copy goes (default: scratch dir)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=2)
    a, _ = ap.parse_known_args()
    assert a.device == rw.DEV, "--device is read by run_eval_wave at import"
    return a


# ---------------------------------------------------------------- checksums
def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def brain_checksum(model):
    hp, hb = hashlib.blake2b(digest_size=8), hashlib.blake2b(digest_size=8)
    for n, p in sorted(model.named_parameters()):
        if not n.startswith("bm."):
            hp.update(n.encode())
            rw._h_obj(hp, p)
    for n, b in sorted(model.named_buffers()):
        if not n.startswith("bm."):
            hb.update(n.encode())
            rw._h_obj(hb, b)
    return hp.hexdigest(), hb.hexdigest()


def mem_param_checksum(bm):
    h = hashlib.blake2b(digest_size=8)
    for n, p in sorted(bm.named_parameters()):
        h.update(n.encode())
        rw._h_obj(h, p)
    rw._h_obj(h, bm.D)
    return h.hexdigest()


# ---------------------------------------------------------------- the new facts
def new_world(a, tok, world, cands):
    t0 = time.time()
    words = set()
    vw = np.fromfile(Path(a.data) / "val.bin", dtype=np.uint16)
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    words |= gw.wiki_word_set(vw[:meta["val_wiki_tokens"]], 10 ** 9, tok)
    if a.check_tokens and Path(a.wiki_train).exists():
        tr = np.memmap(a.wiki_train, dtype=np.uint16, mode="r")
        words |= gw.wiki_word_set(tr, int(a.check_tokens), tok)
    old = {p["name"].lower() for p in world["people"]} | {c["name"].lower() for c in world["cities"]} \
        | {c.lower() for c in world["countries"]} | {r.lower() for r in world["rivers"]}
    ok = gw.name_checker(tok, words | old)
    rng = np.random.default_rng(a.seed)
    names = gw.make_names(rng, a.n_facts, ok)
    assert not ({n.lower() for n in names} & old)
    years = [int(y) for y in cands["year"]]
    tp = gw.cloze_templates(gw.T_PERSON, gw.RELS, "{p}")
    facts = []
    for i, nm in enumerate(names):
        r = i % 5
        rel = gw.RELS[r]
        if rel == "city":
            g = int(rng.integers(len(world["cities"])))
            o = world["cities"][g]["name"]
        elif rel == "job":
            g = int(rng.integers(len(world["jobs"])))
            o = world["jobs"][g]
        elif rel == "pet":
            g = int(rng.integers(len(world["pets"])))
            o = world["pets"][g]
        elif rel == "year":
            g = int(rng.integers(len(years)))
            o = str(years[g])
        else:
            g = i
            while g == i:
                g = int(rng.integers(len(names)))
            o = names[g]
        ti, pre, ans_t = tp[rel][i // 5 % len(tp[rel])]
        cset = rel + ("_a" if ans_t.startswith(" {a}") else "")
        facts.append(dict(person=i, name=nm, relation=rel, obj=o, gold=g, cand_set=cset, template=ti,
                          prefix=pre.format(p=nm), answer=ans_t.format(o=o, a=gw.article(o))))
    print(f"[newfacts] {len(names)} new names ({len(words):,} wiki words + {len(old):,} world names excluded; "
          f"{time.time() - t0:.0f}s), e.g. {names[:5]}; relations {dict((r, sum(f['relation'] == r for f in facts)) for r in gw.RELS)}",
          flush=True)
    return names, facts


def study_rows(a, tok, facts, eot, filler):
    """[(row ids [T + 1], fact index, span (start, end) in x positions, answer start in x positions)]"""
    rng = np.random.default_rng(a.seed + 1)
    T = a.seq
    out = []
    for fi, f in enumerate(facts):
        for _ in range(a.reps):
            t = gw.T_PERSON[f["relation"]][int(rng.integers(gw.N_TRAIN_T))]
            s = t.format(p=f["name"], o=f["obj"], a=gw.article(f["obj"]))
            s = s[0].upper() + s[1:]
            enc = tok.encode(s)
            ids = [eot] + enc.ids + [eot]
            oc = len(t[:t.index("{o}")].format(p=f["name"], a=gw.article(f["obj"]), o=f["obj"]))   # (first {o})
            assert s[oc:oc + len(f["obj"])].lower() == f["obj"].lower()
            ans = [j for j, (u, v) in enumerate(enc.offsets) if v > oc and u < oc + len(f["obj"])]
            pre_n = int(rng.integers(8, max(9, T + 1 - len(ids) - 8)))
            post_n = T + 1 - len(ids) - pre_n
            p0 = int(rng.integers(0, len(filler) - pre_n - 1))
            p1 = int(rng.integers(0, len(filler) - post_n - 1))
            row = np.concatenate([filler[p0:p0 + pre_n], np.array(ids), filler[p1:p1 + post_n]]).astype(np.int64)
            assert len(row) == T + 1
            out.append((row, fi, (pre_n + 1, pre_n + 1 + len(enc.ids)), pre_n + 1 + ans[0]))
    perm = rng.permutation(len(out))
    return [out[i] for i in perm]


# ---------------------------------------------------------------- the study phase (memory writes, brain frozen)
def study(a, model, rows):
    bm = model.bm
    dev = a.device
    keep = dict(explore=bm.explore, drop=bm.drop)
    bm.explore, bm.drop = 0.0, 0.0                   # (exploration off; no sequence drop: every fact may write)
    req = {}
    for n, p in model.named_parameters():
        if not n.startswith("bm."):
            req[n] = p.requires_grad
            p.requires_grad_(False)
    orig_bf = bm.begin_forward

    def begin_forward():                             # (a forward without targets: the memory's live state anyway)
        bm.train_forward = True
        orig_bf()
    bm.begin_forward = begin_forward
    cap = dict(tok=None)
    orig_sel = bm.policy.select

    def sel(*aa, **kk):
        tok_, st_ = orig_sel(*aa, **kk)
        cap["tok"] = (tok_.detach().cpu(), st_.detach().cpu())
        return tok_, st_
    bm.policy.select = sel
    if not a.addr_learn:
        bm._rows = []
    c0 = dict(bm.cnt)
    sw0 = bm.store.stats_acc.get("writes", 0)
    sc0 = bm.store.stats_acc.get("confirms", 0)
    T = a.seq
    n_on_fact = n_on_ans = n_sel = n_probe = 0
    per_fact = defaultdict(int)
    t0 = time.time()
    bm.train()
    try:
        for b in range(0, len(rows), a.study_batch):
            chunk = rows[b:b + a.study_batch]
            xy = torch.from_numpy(np.stack([r[0] for r in chunk])).to(dev)
            x, y = xy[:, :-1], xy[:, 1:]
            cap["tok"] = None
            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    logits = model(x)
                bm.select_writes(logits, y, x)
                rec = bm._cache
                if rec is not None and rec.get("probe") is not None:
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss.backward(inputs=[rec["probe"]])
                    n_probe += int(rec["probe"].grad is not None)
            if not a.addr_learn:
                bm._rows = []
            bm.after_backward()
            for p in bm.parameters():                # (the gate loss's gradients: never applied)
                p.grad = None
            bm.D_grad.zero_()
            if cap["tok"] is not None:
                tk, st = cap["tok"]
                for t_, s_ in zip(tk.tolist(), st.tolist()):
                    if s_ <= 0:
                        continue
                    r, pos = divmod(t_, T)
                    row = chunk[r]
                    n_sel += 1
                    if row[2][0] - 1 <= pos < row[2][1]:
                        n_on_fact += 1
                        per_fact[row[1]] += 1
                    if pos == row[3] - 1:
                        n_on_ans += 1
            del logits, xy, x, y
            bm._cache = None
            if (b // a.study_batch) % 50 == 0:
                print(f"  study {b + len(chunk)}/{len(rows)} seqs, writes g {bm.cnt['writes_g'] - c0['writes_g']} "
                      f"p {bm.cnt['writes_p'] - c0['writes_p']} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        bm.eval()
        bm._live = False
        bm.begin_forward = orig_bf
        bm.policy.select = orig_sel
        bm.explore, bm.drop = keep["explore"], keep["drop"]
        for n, p in model.named_parameters():
            if n in req:
                p.requires_grad_(req[n])
        bm._cache, bm._rows, bm._samp = None, [], None
    d = {k: bm.cnt[k] - c0.get(k, 0) for k in ("writes_g", "writes_p", "writes_both", "confirms_g", "confirms_p",
                                                 "recalls", "dropped_seq", "seqs")}
    d.update(store_writes=bm.store.stats_acc.get("writes", 0) - sw0,
             store_confirms=bm.store.stats_acc.get("confirms", 0) - sc0,
             write_tokens_selected=n_sel, write_tokens_in_fact_span=n_on_fact, write_tokens_before_answer=n_on_ans,
             facts_with_a_write=len(per_fact), micro_batches_with_grad=n_probe, seqs=len(rows), time_s=time.time() - t0)
    return d


# ---------------------------------------------------------------- the quiz
def quiz(a, model, ns, items, rank_set, mode):
    res = {}
    with rw.KO(model, "wave", mode):
        t0 = time.time()
        sc = ew.score(model, [it["seq"] for it in items], [it["n_ans"] for it in items], ns, ns.eot)
        ranks = ew.rank_items(model, rank_set, ns, ns.eot) if rank_set else []
    by = defaultdict(list)
    for it, r in zip(items, sc):
        by[it["key"]].append(r)
        by[it["key"].split("/")[0].split("_")[0] + "/ALL"].append(r)
    for k, rr in sorted(by.items()):
        res[k] = dict(n=len(rr), exact=float(np.mean([r[2] for r in rr])), first_tok=float(np.mean([r[1] for r in rr])),
                      logp_tok=float(np.mean([r[0] / r[3] for r in rr])))
    rk = defaultdict(list)
    for it, r in zip(rank_set, ranks):
        rk[it["key"]].append(r)
        rk[it["key"].split("/")[0].split("_")[0] + "/ALL"].append(r)
    rko = {}
    for k, rr in sorted(rk.items()):
        r = np.array([x[0] for x in rr], float)
        C = np.array([x[1] for x in rr], float)
        rko[k] = dict(n=len(rr), top1=float((r == 1).mean()), top5=float((r <= 5).mean()), mean_rank=float(r.mean()),
                      chance_top1=float((1 / C).mean()), chance_rank=float(((C + 1) / 2).mean()))
    return dict(cloze=res, rank=rko, time_s=time.time() - t0)


def main():
    a = args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(2)
    t_all = time.time()
    ckpt = Path(a.ckpt)
    data = Path(a.data)
    meta = json.loads((data / "meta.json").read_text())
    eot = int(meta["eot_id"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))
    world = json.loads((data / "world.json").read_text())
    cand_txt = json.loads((data / "cloze_candidates.json").read_text())

    # ---- the memory copy (the original is only read)
    orig = memory_api.memory_path(ckpt)
    h_orig0 = sha256_file(orig)
    cp = Path(a.mem_copy) if a.mem_copy else SCRATCH / f"{ckpt.parent.name}_{ckpt.stem}.memory.copy.pt"
    cp.parent.mkdir(parents=True, exist_ok=True)
    assert "runs" not in cp.resolve().parts, "the copy never goes into runs/"
    if not cp.exists() or cp.stat().st_size != orig.stat().st_size or sha256_file(cp) != h_orig0:
        shutil.copyfile(orig, cp)
    h_copy = sha256_file(cp)
    assert h_copy == h_orig0
    print(f"[newfacts] memory original {orig} sha256 {h_orig0}; working copy {cp} ({h_copy})", flush=True)

    # ---- load (checkpoint on the CPU, memory from the copy, then the device)
    la = types.SimpleNamespace(ckpt=str(ckpt), config=a.config, capacity=a.capacity, device="cpu", k=a.k)
    model, step, run_cfg, missing, unexpected = rw._orig_load(la)
    bm = model.bm
    assert type(bm) is WaveV21Memory
    obj = memory_api.load_memory_file(cp, model, strict=True)
    del obj
    model.to(a.device).eval()
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    h_mem0 = rw.mem_checksum(bm)
    h_brain0 = brain_checksum(model)
    h_mpar0 = mem_param_checksum(bm)
    print(f"[newfacts] loaded {ckpt} (step {step}); memory active {bm.active} n_live {bm.n_live} explore {bm.explore} "
          f"drop {bm.drop}; brain checksum params {h_brain0[0]} buffers {h_brain0[1]}; memory state {h_mem0}; "
          f"threads {torch.get_num_threads()} device {a.device} ({time.time() - t_all:.0f}s)", flush=True)

    # ---- facts, study text, quiz items
    names, facts = new_world(a, tok, world, cand_txt)
    filler = np.fromfile(data / "val.bin", dtype=np.uint16)[:meta["val_wiki_tokens"]].astype(np.int64)
    rows = study_rows(a, tok, facts, eot, filler)
    cand_ids = {k: [tok.encode(c).ids for c in v] for k, v in cand_txt.items()}
    name_ids = [tok.encode(" " + n).ids for n in names]
    items, rank_set = [], []
    nrk = defaultdict(int)
    rng = np.random.default_rng(a.seed + 2)
    for f in facts:
        pre = [eot] + tok.encode(f["prefix"]).ids
        aid = tok.encode(f["answer"]).ids
        assert tok.encode(f["prefix"] + f["answer"]).ids == pre[1:] + aid, (f["prefix"], f["answer"])
        key = f"new_{f['relation']}/t{f['template']}"
        items.append(dict(seq=pre + aid, n_ans=len(aid), key=f"new_{f['relation']}"))
        if nrk[f["relation"]] < a.rank_limit:
            nrk[f["relation"]] += 1
            if f["relation"] == "friend":
                nd = min(a.friend_distractors, len(names) - 2)
                pool = [int(j) for j in rng.choice(len(names), nd + 2, replace=False)
                        if j not in (f["gold"], f["person"])][:nd]
                ids, gold = [name_ids[f["gold"]]] + [name_ids[j] for j in pool], 0
            else:
                ids, gold = cand_ids[f["cand_set"]], f["gold"]
            assert ids[gold] == aid, (f, ids[gold], aid)
            rank_set.append(dict(prefix_ids=pre, cand_ids=ids, gold=gold, key=f"new_{f['relation']}"))
    old = defaultdict(list)
    for l in open(data / "cloze.jsonl", encoding="utf-8"):
        it = json.loads(l)
        if it["type"].startswith("cloze_"):
            old[it["type"]].append(it)
    old_items = []
    for t, lst in old.items():
        for it in (lst[:a.old_limit] if a.old_limit else lst):
            old_items.append(dict(seq=[eot] + tok.encode(it["prefix"]).ids + it["answer_ids"], n_ans=len(it["answer_ids"]),
                                  key=f"old_{t[6:]}/{it['group']}"))
    ns = types.SimpleNamespace(device=a.device, batch=a.batch, rank_batch=a.rank_batch, eot=eot)
    print(f"[newfacts] study: {len(rows)} sequences of {a.seq} tokens ({a.reps} per fact), micro-batch {a.study_batch}; "
          f"quiz: {len(items)} new cloze items, {len(rank_set)} ranked ({sum(len(r['cand_ids']) for r in rank_set):,} "
          f"candidate sequences); {len(old_items)} old cloze items", flush=True)
    ex = rows[0]
    print("  study row example:", repr(tok.decode(ex[0][ex[2][0] - 4:ex[2][1] + 4].tolist())), "| answer starts at",
          repr(tok.decode([int(ex[0][ex[3]])])), flush=True)

    R = dict(pre={}, post={})
    if a.pre:
        t0 = time.time()
        R["pre"]["mem_on"] = quiz(a, model, ns, items + old_items, rank_set, None)
        print(f"[newfacts] pre-study mem_on: new exact {R['pre']['mem_on']['cloze']['new/ALL']['exact']:.4f} top1 "
              f"{R['pre']['mem_on']['rank'].get('new/ALL', {}).get('top1', float('nan')):.4f}; old exact "
              f"{R['pre']['mem_on']['cloze']['old/ALL']['exact']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    assert rw.mem_checksum(bm) == h_mem0, "the pre-study quiz changed the memory"

    # ---- study
    S = study(a, model, rows)
    h_mem1 = rw.mem_checksum(bm)
    h_brain1 = brain_checksum(model)
    h_mpar1 = mem_param_checksum(bm)
    print(f"[newfacts] study done: {S}", flush=True)
    print(f"[newfacts] brain params {h_brain0[0]} -> {h_brain1[0]} "
          f"({'UNCHANGED' if h_brain0[0] == h_brain1[0] else 'CHANGED'}); brain buffers {h_brain0[1]} -> {h_brain1[1]} "
          f"({'UNCHANGED' if h_brain0[1] == h_brain1[1] else 'CHANGED'}); memory params + D {h_mpar0} -> {h_mpar1} "
          f"({'UNCHANGED' if h_mpar0 == h_mpar1 else 'CHANGED'}); memory state {h_mem0} -> {h_mem1} "
          f"({'CHANGED (writes)' if h_mem0 != h_mem1 else 'UNCHANGED'})", flush=True)
    assert h_brain0[0] == h_brain1[0], "brain weights changed during study"
    assert h_mpar0 == h_mpar1, "memory parameters / D changed during study"

    # ---- quiz, four modes
    for name, mode in MODES:
        t0 = time.time()
        R["post"][name] = q = quiz(a, model, ns, items + old_items, rank_set, mode)
        print(f"[newfacts] post {name:13s}: new exact {q['cloze']['new/ALL']['exact']:.4f} top1 "
              f"{q['rank'].get('new/ALL', {}).get('top1', float('nan')):.4f} rank "
              f"{q['rank'].get('new/ALL', {}).get('mean_rank', float('nan')):.1f}; old exact "
              f"{q['cloze']['old/ALL']['exact']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    h_mem2 = rw.mem_checksum(bm)
    assert h_mem2 == h_mem1, "the quiz changed the memory"
    h_orig1 = sha256_file(orig)
    assert h_orig1 == h_orig0, "the ORIGINAL memory file changed"
    peak = torch.cuda.max_memory_allocated() / 1e9 if a.device == "cuda" else None

    # ---- report
    on, off = R["post"]["mem_on"], R["post"]["ko_wave"]
    lines = [f"NEW-FACTS eval: {ckpt} (step {step}); {len(facts)} new facts x {a.reps} study reps, seq {a.seq}, "
             f"micro-batch {a.study_batch}, device {a.device}",
             f"writes: {S['writes_g']} global, {S['writes_p']} private ({S['writes_both']} both), confirms "
             f"{S['confirms_g']} g / {S['confirms_p']} p; write tokens selected {S['write_tokens_selected']}, in a fact "
             f"span {S['write_tokens_in_fact_span']}, right before the answer {S['write_tokens_before_answer']}; facts "
             f"with >= 1 write {S['facts_with_a_write']}/{len(facts)}; study {S['time_s']:.0f}s",
             f"checks: brain params {'UNCHANGED' if h_brain0[0] == h_brain1[0] else 'CHANGED'}, brain buffers "
             f"{'UNCHANGED' if h_brain0[1] == h_brain1[1] else 'CHANGED'}, memory params+D "
             f"{'UNCHANGED' if h_mpar0 == h_mpar1 else 'CHANGED'}, memory state {h_mem0} -> {h_mem1}, original memory "
             f"file sha256 {h_orig0} -> {h_orig1} ({'UNTOUCHED' if h_orig0 == h_orig1 else 'CHANGED'})",
             "", f"{'key':22s} " + " ".join(f"{n:>13s}" for n in ["pre on"] * bool(a.pre) + [m for m, _ in MODES])
             + f" {'on-off':>8s}"]
    keys = sorted(on["cloze"], key=lambda k: (not k.startswith("new"), k.endswith("ALL"), k))
    for k in keys:
        vals = ([R["pre"]["mem_on"]["cloze"].get(k, {}).get("exact", float("nan"))] if a.pre else []) + \
            [R["post"][m]["cloze"][k]["exact"] for m, _ in MODES]
        lines.append(f"{k + ' exact':22s} " + " ".join(f"{v:13.4f}" for v in vals)
                     + f" {on['cloze'][k]['exact'] - off['cloze'][k]['exact']:+8.4f}   (n {on['cloze'][k]['n']}, "
                       f"lp/tok on {on['cloze'][k]['logp_tok']:.3f} off {off['cloze'][k]['logp_tok']:.3f})")
    for k in sorted(on["rank"], key=lambda k: (k.endswith("ALL"), k)):
        vals = ([R["pre"]["mem_on"]["rank"].get(k, {}).get("top1", float("nan"))] if a.pre else []) + \
            [R["post"][m]["rank"][k]["top1"] for m, _ in MODES]
        lines.append(f"{k + ' top1':22s} " + " ".join(f"{v:13.4f}" for v in vals)
                     + f" {on['rank'][k]['top1'] - off['rank'][k]['top1']:+8.4f}   (n {on['rank'][k]['n']}, chance "
                       f"{on['rank'][k]['chance_top1']:.4f}; mean rank on {on['rank'][k]['mean_rank']:.1f} off "
                       f"{off['rank'][k]['mean_rank']:.1f} chance {on['rank'][k]['chance_rank']:.1f})")
    d_new = on["cloze"]["new/ALL"]["exact"] - off["cloze"]["new/ALL"]["exact"]
    lines += ["", f"BAR new facts on - off exact {d_new:+.4f} (bar >= +0.05: {'PASS' if d_new >= 0.05 else 'FAIL'}); "
              f"off exact {off['cloze']['new/ALL']['exact']:.4f}, off top1 {off['rank'].get('new/ALL', {}).get('top1', float('nan')):.4f} "
              f"(chance {off['rank'].get('new/ALL', {}).get('chance_top1', float('nan')):.4f})"]
    if a.pre:
        pre = R["pre"]["mem_on"]["cloze"]
        lines.append("interference (old facts, mem_on post - pre): " + ", ".join(
            f"{k} {on['cloze'][k]['exact'] - pre[k]['exact']:+.4f}" for k in keys if k.startswith("old")))
    lines.append(f"total {time.time() - t_all:.0f}s" + (f", CUDA peak allocated {peak:.2f} GB" if peak else ""))
    txt = "\n".join(lines)
    print(txt, flush=True)
    out = Path(a.out) if a.out else HERE / f"{ckpt.parent.name}_{ckpt.stem}_newfacts{('_' + a.tag) if a.tag else ''}.json"
    assert "runs" not in out.resolve().parts
    out.write_text(json.dumps(dict(args=vars(a), step=step, study=S, results=R, facts=facts[:50],
                                   checks=dict(brain0=h_brain0, brain1=h_brain1, mem0=h_mem0, mem1=h_mem1, mem2=h_mem2,
                                               mem_params0=h_mpar0, mem_params1=h_mpar1, orig0=h_orig0, orig1=h_orig1),
                                   cuda_peak_gb=peak), indent=1))
    out.with_suffix(".txt").write_text(txt + "\n")
    print(f"wrote {out} (+ .txt)", flush=True)


if __name__ == "__main__":
    main()
