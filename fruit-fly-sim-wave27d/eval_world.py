"""Generated-world fact recall of a LoopBrain checkpoint (plan_v2/15 section d; data from generate_world.py).

    python -B eval_world.py --ckpt ../runs/wiki_v4d_12k/final.pt --device cpu
    python -B eval_world.py --ckpt <copy of last.pt> --config ../runs/wiki_mem_12k/config.json --limit 50

PRIMARY metric: cloze items (cloze.jsonl): the prefix of a HELD-OUT paraphrase template up to the answer, after
<|endoftext|> (a document start, nothing else prepended), e.g. "<Name>'s closest companion is" -> " <friend>". The
statement form is what training text looks like (the "Q: ... A:" form never is), but these exact wordings never
appear in training. Types: cloze_city / job / pet / year / friend (late group), cloze_early_retention (early group) and
geo_country (city -> country, the second hop on its own). Two-hop has no person -> country statement template (the
world never states it), so it stays with the Q: metric. Per item:
- (a) cloze exact match: greedy decoding reproduces the answer tokens (= every answer token is the argmax under teacher
  forcing given the gold prefix: one forward), first-token accuracy, teacher-forced answer log-prob;
- (b) ranking among all candidates of that relation by the SUM of the answer-token log-probs (each candidate appended
  to the prefix and run: the looped model has no cache): 300 cities, 200 jobs (with their article when the template
  has one), 40 pets, the 86 birth years present, 40 countries; friend: the gold + --friend-distractors random other
  people (default 100; 0 = all 5,000). Top-1 / top-5 / mean rank with their chance levels (1/C, 5/C, (C+1)/2).
  Ranking runs on the first N items per type: --rank-limit N, or by default N = max(--rank-min, --rank-budget / C)
  (about --rank-budget candidate sequences per type: short sequences are slow on CPU, the SSD pads to 64-token
  chunks; defaults 3000 / 20 -> city 20, job 20, friend 29, early 20, year 34, pet / country 75 items).
SECONDARY: the "Q: ... A:" questions of questions.jsonl: exact match, first token, answer log-prob, comparison choice
accuracy (log p(older) > log p(other), chance 0.5), and two-hop ranking among the 40 countries (q_two_hop).
Both per type and per group, plus accuracy vs exposure count (exposures.npz, --exposure-at tokens).
With a memory (mem.S / mem.R): everything again with the shared reads off, the private reads off and each level off
(mem.ko_shared / ko_private / ko_levels), and the deltas vs memory on (negative d logp / d top1 = the memory helped).
Also the val loss on val_world.bin (fresh world documents with held-out templates) vs Wikipedia val (the first
val_wiki_tokens of val.bin, or --wiki-val), under every knockout too.

MoE capacity: --capacity dropless (default) computes every routed token, so an item's result does not depend on what
else is in the batch; --capacity train keeps the trained capacity factor.
Writes JSON (--out) and a table (.txt). Never writes next to the checkpoint.
"""
import argparse
import json
import os
import re
import sys
import time

if "--threads" in sys.argv[:-1] or not any(a.startswith("--threads") for a in sys.argv):  # before torch loads OpenMP
    _t = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv[:-1] else "4"
    if _t != "0":
        os.environ.setdefault("OMP_NUM_THREADS", _t)
        os.environ.setdefault("MKL_NUM_THREADS", _t)
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from loop_brain import LoopBrain
from spiking_lm import RoutedMLP, SpikingLM

HERE = Path(__file__).resolve().parent
EXP_BUCKETS = [(0, 1), (1, 10), (10, 30), (30, 100), (100, 300), (300, 10 ** 9)]


def load_model(a):
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = (LoopBrain if cfg.get("arch") == "loop" else SpikingLM)(**cfg)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    run_cfg = {}
    cpath = Path(a.config) if a.config else Path(a.ckpt).parent / "config.json"
    if cpath.exists():
        run_cfg = json.loads(cpath.read_text())
    # attributes the trainer sets (train_lm.py): the milder memory schedule, the evaluation routing noise
    for attr in ("mem_readdr_every", "mem_readdr_start", "mem_margin_passes"):
        if attr in run_cfg and hasattr(model, attr):
            setattr(model, attr, run_cfg[attr])
    for mod in model.modules():
        if isinstance(mod, RoutedMLP):
            mod.noise, mod.eval_noise = 0.0, run_cfg.get("eval_route_noise", 0.0)
            if a.capacity == "dropless":
                mod.cap = 0
    mem = getattr(model, "mem", None)
    if mem is not None:
        mem.write_on = False
    model.to(a.device).eval()
    if hasattr(model, "K"):
        model.K = a.k
    return model, ck.get("step"), run_cfg, missing, unexpected


def knockouts(model, gate_modes=True):
    """[(name, attr, value)] like train_lm.loop_report; the first entry is the intact model. With the memory gate
    (plan_v2/22): also the eval gating modes forced open / memory off / random / inverted (attr "model.<name>": an
    attribute of the model)."""
    mem = getattr(model, "mem", None)
    out = [("mem_on" if mem is not None else "model", None, None)]
    if mem is None:
        return out
    if gate_modes and getattr(model, "mem_gate", None) is not None:
        out += [(f"gate_{m}", "model.mem_gate_eval", m) for m in ("open", "off", "random", "inverted")]
    if mem.S:
        out.append(("ko_shared", "ko_shared", True))
    if mem.R:
        out.append(("ko_private", "ko_private", True))
    if mem.S:
        out += [(f"ko_level{l + 1}", "ko_levels", (l,)) for l in range(mem.L)]
    if mem.read0:
        out.append(("ko_read0", "read0", False))
    return out


class KO:
    def __init__(self, model, attr, value):
        self.mem, self.attr, self.value = getattr(model, "mem", None), attr, value
        if attr and attr.startswith("model."):  # (the gate's eval mode: an attribute of the model)
            self.mem, self.attr = model, attr[len("model."):]

    def __enter__(self):
        if self.attr:
            self.old = getattr(self.mem, self.attr)
            setattr(self.mem, self.attr, self.value)

    def __exit__(self, *exc):
        if self.attr:
            setattr(self.mem, self.attr, self.old)


@torch.no_grad()
def gate_type_rates(model, typed, a, eot, batch=None):
    """The memory gate's open rate by position type (plan_v2/22 section 7). typed: [(token list, {position: type})]
    (positions index the input x = seq[:-1]). Every forward records the open mask of each read (model.gate_open); a
    position's rate is its mean over the reads. Returns {type: dict(n, rate, by_pass)}."""
    batch = batch or a.batch
    acc = defaultdict(lambda: [0, 0.0, None])
    order = sorted(range(len(typed)), key=lambda i: len(typed[i][0]))
    for b in range(0, len(order), batch):
        chunk = [typed[i] for i in order[b:b + batch]]
        T = max(len(sq) for sq, _ in chunk) - 1
        x = torch.full((len(chunk), T), eot, dtype=torch.long)
        for r, (sq, _) in enumerate(chunk):
            x[r, :len(sq) - 1] = torch.tensor(sq[:-1])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
            model(x.to(a.device))
        reads = [torch.ones(x.numel(), dtype=torch.bool) if rr["mask"] is None else rr["mask"].cpu()
                 for rr in model.gate_open]
        if not reads:
            continue
        mk = torch.stack(reads).float().view(len(reads), len(chunk), T)
        for r, (_, types) in enumerate(chunk):
            for pos, ty in types.items():
                d = acc[ty]
                d[0] += 1
                d[1] += mk[:, r, pos].mean().item()
                d[2] = mk[:, r, pos].clone() if d[2] is None else d[2] + mk[:, r, pos]
    return {ty: dict(n=n, rate=tot / n, by_pass=(bp / n).tolist()) for ty, (n, tot, bp) in sorted(acc.items())}


@torch.no_grad()
def score(model, seqs, n_ans, a, eot, batch=None):
    """seqs: token lists (prefix + answer); the last n_ans[i] tokens are the answer. Returns per sequence
    (sum log p, first-token correct, all argmax correct, n answer tokens). Sequences are batched by length."""
    batch = batch or a.batch
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    res = [None] * len(seqs)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        chunk = [seqs[i] for i in idx]
        T = max(len(s) for s in chunk) - 1
        x = torch.full((len(chunk), T), eot, dtype=torch.long)
        rows, pos, gold = [], [], []
        for r, s in enumerate(chunk):  # right padding: the model is causal, padding never reaches earlier tokens
            x[r, :len(s) - 1] = torch.tensor(s[:-1])
            n = n_ans[idx[r]]
            for j in range(len(s) - 1 - n, len(s) - 1):
                rows.append(r)
                pos.append(j)
                gold.append(s[j + 1])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
            logits = model(x.to(a.device))
        sel = logits[torch.tensor(rows, device=logits.device), torch.tensor(pos, device=logits.device)].float().cpu()
        lp = F.log_softmax(sel, -1)
        g = torch.tensor(gold)
        tok_lp = lp.gather(1, g[:, None])[:, 0].tolist()
        hit = (lp.argmax(-1) == g).tolist()
        acc = {}
        for o, l, h in zip(rows, tok_lp, hit):
            d = acc.setdefault(o, [0.0, h, True, 0])   # (first answer token = the first entry of its row)
            d[0] += l
            d[2] = d[2] and h
            d[3] += 1
        for r in range(len(chunk)):
            d = acc[r]
            res[idx[r]] = (d[0], bool(d[1]), bool(d[2]), d[3])
    return res


def rank_items(model, items, a, eot):
    """items: dicts with prefix_ids, cand_ids (list of token lists), gold (index). Scores every candidate continuation
    (sum log p) and returns per item (rank of the gold: 1 = best, ties count against the gold; number of candidates)."""
    seqs, n_ans, own = [], [], []
    for k, it in enumerate(items):
        for c in it["cand_ids"]:
            seqs.append(it["prefix_ids"] + c)
            n_ans.append(len(c))
            own.append(k)
    sc = score(model, seqs, n_ans, a, eot, batch=a.rank_batch)
    per = defaultdict(list)
    for k, r in zip(own, sc):
        per[k].append(r[0])
    out = []
    for k, it in enumerate(items):
        v = np.array(per[k])
        out.append((int((v >= v[it["gold"]]).sum()), len(v)))
    return out


def summarize_rank(items, ranks):
    groups = defaultdict(list)
    for k, it in enumerate(items):
        groups[f"{it['type']}/{it['group']}"].append(k)
    out = {}
    for key, ks in sorted(groups.items()):
        r = np.array([ranks[k][0] for k in ks], dtype=float)
        C = np.array([ranks[k][1] for k in ks], dtype=float)
        out[key] = dict(n=len(ks), n_cands=float(C.mean()), top1=float((r == 1).mean()), top5=float((r <= 5).mean()),
                        mean_rank=float(r.mean()), chance_top1=float((1 / C).mean()),
                        chance_top5=float((np.minimum(5, C) / C).mean()), chance_rank=float(((C + 1) / 2).mean()))
    return out


@torch.no_grad()
def greedy(model, prefix, n, a):
    ids = list(prefix)
    for _ in range(n):
        x = torch.tensor([ids], device=a.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
            ids.append(int(model(x)[0, -1].argmax()))
    return ids[len(prefix):]


@torch.no_grad()
def val_loss(model, data, a, seed=4242):
    if len(data) <= a.seq + 1:
        return float("nan")
    gen = np.random.RandomState(seed)
    tot = 0.0
    for _ in range(a.val_batches):
        rows = [data[i:i + a.seq + 1].astype(np.int64) for i in gen.randint(0, len(data) - a.seq - 1, a.val_batch)]
        xy = torch.from_numpy(np.stack(rows)).to(a.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
            tot += model(xy[:, :-1], xy[:, 1:])[1].float().item()
    return tot / a.val_batches


def summarize(qs, res, alt=None, exposure=None):
    """Per (type, group) and per type: n, exact match, first-token acc, mean answer log-prob (sum and per token)."""
    groups = defaultdict(list)
    for k, (q, r) in enumerate(zip(qs, res)):
        groups[(q["type"], q["group"])].append(k)
        groups[(q["type"], "all")].append(k)
        groups[("ALL", q["group"])].append(k)
        if q["type"].startswith("single_"):
            groups[("single_all", q["group"])].append(k)
        if q["type"] in ("cloze_city", "cloze_job", "cloze_pet", "cloze_year", "cloze_friend"):
            groups[("cloze_single_all", q["group"])].append(k)
        if q["type"] in ("early_retention", "cloze_early_retention"):
            groups[(q["type"][:-10] + "_" + q.get("relation", "?"), "early")].append(k)
    out = {}
    for (t, g), ks in sorted(groups.items()):
        rr = [res[k] for k in ks]
        d = dict(n=len(ks), exact=float(np.mean([r[2] for r in rr])), first_tok=float(np.mean([r[1] for r in rr])),
                 logp=float(np.mean([r[0] for r in rr])), logp_tok=float(np.sum([r[0] for r in rr]) / np.sum([r[3] for r in rr])))
        if alt is not None and t == "compare":
            ch = [res[k][0] > alt[k] for k in ks if k in alt]
            if ch:
                d["choice"] = float(np.mean(ch))
        out[f"{t}/{g}"] = d
    if exposure is not None:  # accuracy vs exposure of the asked fact (single-hop + retention)
        for lo, hi in EXP_BUCKETS:
            ks = [k for k, q in enumerate(qs) if q["type"].startswith(("single_", "early_retention", "cloze_"))
                  and lo <= exposure[q["fact"]] < hi]
            if ks:
                out[f"exposure_{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = dict(
                    n=len(ks), exact=float(np.mean([res[k][2] for k in ks])),
                    first_tok=float(np.mean([res[k][1] for k in ks])), logp=float(np.mean([res[k][0] for k in ks])),
                    logp_tok=float(np.sum([res[k][0] for k in ks]) / np.sum([res[k][3] for k in ks])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="final.pt / last.pt (copy last.pt of a running job first)")
    ap.add_argument("--config", default="", help="the run's config.json (default: next to --ckpt)")
    ap.add_argument("--data", default="X:/Datasets/wiki_world_mix")
    ap.add_argument("--wiki-val", default="", help="Wikipedia val.bin (default: the first val_wiki_tokens of val.bin)")
    ap.add_argument("--k", type=int, default=6, help="loop passes")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=0, help="questions per type (0 = all)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shots", type=int, default=0, help="solved same-type questions prepended (format demo)")
    ap.add_argument("--rank-limit", type=int, default=0, help="ranking: items per type (0 = by --rank-budget)")
    ap.add_argument("--rank-budget", type=int, default=3000, help="ranking: candidate sequences per type (approx.)")
    ap.add_argument("--rank-min", type=int, default=20, help="ranking: at least this many items per type")
    ap.add_argument("--friend-distractors", type=int, default=100,
                    help="friend ranking: random other people besides the gold (0 = all 5,000 names)")
    ap.add_argument("--rank-batch", type=int, default=256)
    ap.add_argument("--questions", type=int, default=1, help="1 = also the secondary Q: A: metrics")
    ap.add_argument("--capacity", default="dropless", choices=["dropless", "train"])
    ap.add_argument("--knockouts", type=int, default=1, help="1 = also every memory knockout (if the model has one)")
    ap.add_argument("--gate-modes", type=int, default=1,
                    help="memory gate (plan_v2/22): 1 = also the eval gating modes as knockouts, and the open rates by "
                         "position type")
    ap.add_argument("--gate-items", type=int, default=400, help="memory gate: items / sequences for the open rates")
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--val-batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--exposure-at", default="200000000", help="exposures.npz snapshot (tokens) for the buckets")
    ap.add_argument("--show", type=int, default=2, help="greedy-decoded examples printed per type")
    ap.add_argument("--out", default="", help="JSON path (default: NewBrain/world_evals/<run>_<ckpt>.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads (torch + OpenMP), so a CPU eval does not slow a training run (0 = torch default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(min(a.threads, 4))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    torch.manual_seed(a.seed)
    data = Path(a.data)
    meta = json.loads((data / "meta.json").read_text())
    eot = int(meta["eot_id"])
    dt = np.dtype(meta.get("dtype", "uint16"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))

    model, step, run_cfg, missing, unexpected = load_model(a)
    mem = getattr(model, "mem", None)
    print(f"checkpoint {a.ckpt} (step {step if step is not None else 'final'}), K={a.k}, device {a.device}, "
          f"memory {'S=%d R=%d L=%d' % (mem.S, mem.R, mem.L) if mem is not None else 'none'}; "
          f"missing {len(missing)} unexpected {len(unexpected)} keys ({time.time()-t0:.0f}s)", flush=True)
    if missing or unexpected:
        print("  missing:", missing[:8], "unexpected:", unexpected[:8], flush=True)

    # ---- primary: cloze items (held-out templates, after EOT)
    clz_all = [json.loads(l) for l in open(data / "cloze.jsonl", encoding="utf-8")]
    cand_txt = json.loads((data / "cloze_candidates.json").read_text())
    cand_ids = {k: [tok.encode(c).ids for c in v] for k, v in cand_txt.items()}
    by_t = defaultdict(list)
    for it in clz_all:
        by_t[it["type"]].append(it)
    clz = [it for t, lst in by_t.items() for it in (lst[:a.limit] if a.limit else lst)]
    c_seqs = [[eot] + tok.encode(it["prefix"]).ids + it["answer_ids"] for it in clz]
    c_n = [len(it["answer_ids"]) for it in clz]
    rng = np.random.default_rng(a.seed)
    rank_set = []
    for t, lst in by_t.items():
        C = len(cand_ids[lst[0]["cand_set"]]) if lst[0]["cand_set"] != "friend" or not a.friend_distractors \
            else a.friend_distractors + 1
        if t == "cloze_early_retention":  # mixed relations: the mean candidate count
            C = np.mean([len(cand_ids[it["cand_set"]]) if it["cand_set"] != "friend" else a.friend_distractors + 1
                         for it in lst])
        n_rk = a.rank_limit or max(a.rank_min, int(a.rank_budget // C))
        for it in lst[:n_rk]:
            ids, gold = cand_ids[it["cand_set"]], it["gold"]
            if it["cand_set"] == "friend" and a.friend_distractors:  # the gold + random other people
                excl = {gold, it["person"]}
                pool = [int(j) for j in rng.choice(len(ids), a.friend_distractors + 2, replace=False) if j not in excl]
                sel = [gold] + pool[:a.friend_distractors]
                ids, gold = [ids[j] for j in sel], 0
            rank_set.append(dict(type=it["type"], group=it["group"], prefix_ids=[eot] + tok.encode(it["prefix"]).ids,
                                 cand_ids=ids, gold=gold))
    # ---- secondary: the Q: A: questions
    qs, seqs, n_ans, alt_idx, alt_seqs, alt_n, prefix = [], [], [], [], [], [], {}
    if a.questions:
        allq = [json.loads(l) for l in open(data / "questions.jsonl", encoding="utf-8")]
        by_type = defaultdict(list)
        for q in allq:
            by_type[q["type"]].append(q)
        demos = {}
        for t, lst in by_type.items():
            demos[t] = lst[-a.shots:] if a.shots else []      # demonstrations from the end, questions from the start
            pool = lst[:len(lst) - a.shots] if a.shots else lst
            qs += pool[:a.limit] if a.limit else pool
        for t in by_type:
            ids = [eot]
            for d in demos[t]:
                ids += tok.encode(d["prompt"] + d["answer"] + "\n").ids
            prefix[t] = ids
        seqs = [prefix[q["type"]] + tok.encode(q["prompt"]).ids + q["answer_ids"] for q in qs]
        n_ans = [len(q["answer_ids"]) for q in qs]
        for k, q in enumerate(qs):  # comparison: the other name as the alternative answer
            if q["type"] == "compare":
                names = re.match(r"Q: Who is older, (.+) or (.+)\? A:$", q["prompt"]).groups()
                other = [n for n in names if " " + n != q["answer"]][0]
                ids = tok.encode(" " + other).ids
                alt_idx.append(k)
                alt_seqs.append(prefix["compare"] + tok.encode(q["prompt"]).ids + ids)
                alt_n.append(len(ids))
        countries = cand_txt["country"]
        th = by_type.get("two_hop", [])
        for q in th[:a.rank_limit or max(a.rank_min, a.rank_budget // len(countries))]:  # two-hop ranking among the 40 countries (Q: form)
            rank_set.append(dict(type="q_two_hop", group=q["group"], cand_ids=cand_ids["country"],
                                 prefix_ids=prefix["two_hop"] + tok.encode(q["prompt"]).ids,
                                 gold=countries.index(q["answer"])))
    exposure = None
    ex_path = data / "exposures.npz"
    if ex_path.exists():
        z = np.load(ex_path)
        key = f"at_{int(float(a.exposure_at))}"
        exposure = z[key] if key in z else z["final"]
    wval = np.memmap(a.wiki_val or data / "val.bin", dtype=dt, mode="r")
    if not a.wiki_val and "val_wiki_tokens" in meta:
        wval = wval[:meta["val_wiki_tokens"]]
    vworld = np.memmap(data / "val_world.bin", dtype=dt, mode="r")
    n_rank = sum(len(it["cand_ids"]) for it in rank_set)
    per_t = ", ".join(f"{t} {min(len(v), a.limit or len(v))}" for t, v in by_t.items())
    fd = "all" if not a.friend_distractors else a.friend_distractors
    print(f"cloze: {len(clz)} items ({per_t}); ranking: {len(rank_set)} items, {n_rank:,} candidate sequences "
          f"(rank-limit {a.rank_limit or f'budget {a.rank_budget} min {a.rank_min}'}, friend distractors {fd}); Q: {len(qs)} questions, shots {a.shots}", flush=True)

    # ---- examples (real greedy decoding)
    examples = []
    if a.show:
        shown = defaultdict(int)
        for it in clz:
            if shown[it["type"]] < a.show:
                shown[it["type"]] += 1
                out = greedy(model, [eot] + tok.encode(it["prefix"]).ids, len(it["answer_ids"]) + 2, a)
                examples.append(dict(type=it["type"], prompt=it["prefix"], answer=it["answer"], greedy=tok.decode(out)))
                print(f"  [{it['type']}] {it['prefix']!r} -> {it['answer']!r}; greedy {tok.decode(out)!r}", flush=True)
        for q in qs:
            if shown[q["type"]] < a.show:
                shown[q["type"]] += 1
                out = greedy(model, prefix[q["type"]] + tok.encode(q["prompt"]).ids, len(q["answer_ids"]) + 2, a)
                examples.append(dict(type=q["type"], prompt=q["prompt"], answer=q["answer"], greedy=tok.decode(out)))
                print(f"  [{q['type']}] {q['prompt']}{q['answer']!r} -> greedy {tok.decode(out)!r}", flush=True)

    # ---- every knockout
    results = {}
    kos = knockouts(model, bool(a.gate_modes)) if a.knockouts else knockouts(model)[:1]
    nan = float("nan")
    for name, attr, value in kos:
        t1 = time.time()
        with KO(model, attr, value):
            cres = score(model, c_seqs, c_n, a, eot)
            t_c = time.time() - t1
            ranks = rank_items(model, rank_set, a, eot)
            t_r = time.time() - t1 - t_c
            res = score(model, seqs, n_ans, a, eot) if seqs else []
            ares = score(model, alt_seqs, alt_n, a, eot) if alt_seqs else []
            vw, vk = val_loss(model, vworld, a), val_loss(model, wval, a)
        alt = {k: r[0] for k, r in zip(alt_idx, ares)}
        results[name] = dict(cloze=summarize(clz, cres, None, exposure), ranking=summarize_rank(rank_set, ranks),
                             questions=summarize(qs, res, alt, exposure) if qs else {}, val_world=vw, val_wiki=vk,
                             time_s=time.time() - t1, time_cloze_s=t_c, time_rank_s=t_r)
        cz, rk = results[name]["cloze"], results[name]["ranking"]
        print(f"{name}: cloze exact {cz.get('ALL/late', {}).get('exact', nan):.3f} (late) "
              f"{cz.get('ALL/early', {}).get('exact', nan):.3f} (early); city top1 "
              f"{rk.get('cloze_city/late', {}).get('top1', nan):.3f}; val world {vw:.4f} wiki {vk:.4f} "
              f"(cloze {t_c:.0f}s, ranking {t_r:.0f}s, total {time.time()-t1:.0f}s)", flush=True)

    # ---- the memory gate (plan_v2/22): open rate by position type (learned mode)
    gate_rates = None
    if a.gate_modes and getattr(model, "mem_gate", None) is not None:
        digit = {}

        def kind(t):
            if t not in digit:
                digit[t] = bool(re.search(r"[0-9]", tok.decode([t])))
            return digit[t]
        typed = []
        for it, sq in list(zip(clz, c_seqs))[:a.gate_items]:  # cloze: [eot] + prefix + answer
            n_pre = len(sq) - len(it["answer_ids"])
            ty = {p: ("number" if kind(sq[p]) else "other_world") for p in range(1, n_pre - 1)}
            ty[n_pre - 1] = "pre_attribute"
            ty.update({p: "attribute" for p in range(n_pre, len(sq) - 1)})
            typed.append((sq, ty))
        for q, sq in list(zip(qs, seqs))[:a.gate_items]:  # Q: ... A: (the "A:" position predicts the answer)
            n_pre = len(sq) - len(q["answer_ids"])
            ty = {p: ("number" if kind(sq[p]) else "q_text") for p in range(1, n_pre - 1)}
            ty[n_pre - 1] = "q_A:"
            typed.append((sq, ty))
        for name, d in (("world_val", vworld), ("wikipedia", wval)):
            gen = np.random.RandomState(a.seed + 77)
            for _ in range(max(1, min(a.gate_items // 50, 8))):
                if len(d) > a.seq + 1:
                    i0 = int(gen.randint(0, len(d) - a.seq - 1))
                    typed.append((d[i0:i0 + a.seq + 1].astype(np.int64).tolist(), {p: name for p in range(a.seq)}))
        gate_rates = gate_type_rates(model, typed, a, eot)
        print("memory gate open rate by position type: " + "  ".join(f"{k} {v['rate']:.3f} (n {v['n']})"
                                                                     for k, v in gate_rates.items()), flush=True)

    # ---- tables
    base = results[kos[0][0]]

    def rk_cols(r):
        if r is None:
            return ""
        return (f"{r['n']:4d} {r['n_cands']:5.0f} {r['top1']:6.3f} {r['chance_top1']:6.3f} {r['top5']:6.3f} "
                f"{r['chance_top5']:6.3f} {r['mean_rank']:7.1f} {r['chance_rank']:7.1f}")

    head_rk = f"{'n_rk':>4s} {'C':>5s} {'top1':>6s} {'chance':>6s} {'top5':>6s} {'chance':>6s} {'rank':>7s} {'chance':>7s}"
    lines = ["PRIMARY: cloze (held-out templates, after <|endoftext|>); ranking by the summed answer log-prob",
             f"{'type/group':30s} {'n':>5s} {'exact':>6s} {'tok1':>6s} {'lp/tok':>7s} | " + head_rk]
    ckeys = [k for k in base["cloze"] if not k.startswith(("exposure", "cloze_early_retention/all"))
             and not k.endswith("/all") and k != "ALL/geo"]
    ckeys = sorted(ckeys, key=lambda k: (k.startswith("ALL"), k.startswith(("cloze_early_", "cloze_single")), k))
    for k in ckeys:
        d = base["cloze"][k]
        lines.append(f"{k:30s} {d['n']:5d} {d['exact']:6.3f} {d['first_tok']:6.3f} {d['logp_tok']:7.3f} | "
                     + rk_cols(base["ranking"].get(k)))
    qkeys = []
    if qs:
        lines += ["", "SECONDARY: Q: ... A: questions (choice = compare: log p(older) > log p(other), chance 0.5)",
                  f"{'type/group':30s} {'n':>5s} {'exact':>6s} {'tok1':>6s} {'lp/tok':>7s} {'logp':>8s} {'choice':>6s}"]
        qkeys = [k for k in base["questions"] if not k.startswith(("exposure", "early_", "ALL")) and k.endswith("/all")]
        qkeys += [k for k in ("single_all/late", "early_retention/early", "ALL/late", "ALL/early")
                  if k in base["questions"]]
        for k in qkeys:
            d = base["questions"][k]
            lines.append(f"{k:30s} {d['n']:5d} {d['exact']:6.3f} {d['first_tok']:6.3f} {d['logp_tok']:7.3f} "
                         f"{d['logp']:8.3f} {d.get('choice', nan):6.3f}")
        for k in [k for k in base["ranking"] if k.startswith("q_two_hop")]:
            lines.append(f"{k + ' (40 countries)':30s} {'':5s} {'':6s} {'':6s} {'':7s} | {head_rk}")
            lines.append(f"{'':30s} {'':5s} {'':6s} {'':6s} {'':7s} | " + rk_cols(base["ranking"][k]))
    exp_rows = [k for k in base["cloze"] if k.startswith("exposure")]
    if exp_rows:
        lines.append(f"cloze vs exposure (at {a.exposure_at} tokens; single-hop + retention): n exact tok1 lp/tok")
        for k in exp_rows:
            d = base["cloze"][k]
            lines.append(f"  {k:28s} {d['n']:5d} {d['exact']:6.3f} {d['first_tok']:6.3f} {d['logp_tok']:7.3f}")
    lines.append(f"val loss: world {base['val_world']:.4f}  wikipedia {base['val_wiki']:.4f}  "
                 f"(gap {base['val_world'] - base['val_wiki']:+.4f})")
    if len(kos) > 1:  # knockout deltas (knockout - memory on): negative d lp/tok or d top1 = the memory helped
        rkeys = sorted(base["ranking"])
        dq = [k for k in ("single_all/late", "early_retention/early", "two_hop/all", "compare/all", "ALL/late")
              if k in base["questions"]]

        def short(k):
            for x in ("cloze_", "/late", "/early", "/all", "/geo"):
                k = k.replace(x, "")
            return k[:10]
        lines += ["", "knockout deltas (knockout - memory on; negative d lp/tok, d exact, d top1 = the memory helped)"]
        for title, part, keys, field in (("cloze d lp/tok", "cloze", ckeys, "logp_tok"),
                                         ("cloze d exact", "cloze", ckeys, "exact"),
                                         ("rank d top1", "ranking", rkeys, "top1"),
                                         ("rank d mean rank", "ranking", rkeys, "mean_rank"),
                                         ("Q d lp/tok", "questions", dq, "logp_tok"),
                                         ("Q d exact", "questions", dq, "exact")):
            if not keys:
                continue
            lines.append(f"{title:18s} " + " ".join(f"{short(k):>10s}" for k in keys))
            for name, _, _ in kos[1:]:
                r = results[name]
                lines.append(f"  {name:16s} " + " ".join(f"{r[part][k][field] - base[part][k][field]:+10.4f}"
                                                         for k in keys))
        lines.append(f"{'val d loss':18s} {'world':>10s} {'wikipedia':>10s}")
        for name, _, _ in kos[1:]:
            r = results[name]
            lines.append(f"  {name:16s} {r['val_world'] - base['val_world']:+10.4f} "
                         f"{r['val_wiki'] - base['val_wiki']:+10.4f}")
            r["delta"] = {part: {k: {f: r[part][k][f] - base[part][k][f] for f in base[part][k]
                                     if isinstance(base[part][k][f], float) and f in r[part][k]}
                                 for k in base[part] if k in r[part]}
                          for part in ("cloze", "ranking", "questions")}
            r["delta"]["val_world"] = r["val_world"] - base["val_world"]
            r["delta"]["val_wiki"] = r["val_wiki"] - base["val_wiki"]
    table = "\n".join(lines)
    print(table, flush=True)

    ck = Path(a.ckpt)
    out = Path(a.out) if a.out else HERE.parent / "world_evals" / f"{ck.parent.name}_{ck.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ckpt=str(ck), step=step, k=a.k, device=a.device, limit=a.limit, shots=a.shots,
                                   capacity=a.capacity, data=str(data), n_questions=len(qs), n_cloze=len(clz),
                                   rank_limit=a.rank_limit, rank_budget=a.rank_budget, friend_distractors=a.friend_distractors,
                                   results=results, gate_open_by_type=gate_rates,
                                   examples=examples, table=table, runtime_s=time.time() - t0), indent=1))
    out.with_suffix(".txt").write_text(table + "\n")
    print(f"wrote {out} (+ .txt) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
