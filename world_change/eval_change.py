"""CHANGING-FACTS memory eval: stream the val worlds of X:/Datasets/world_change_mix through a checkpoint, in order,
with memory writes ON during the stream, and score the later questions (answers outside the 512-token context).

    CPU check: python -B eval_change.py --ckpt <run>/final.pt --streams 2 --max-seqs 6          (CUDA hidden, 2 threads)
    GPU:       python -B eval_change.py --ckpt <run>/final.pt --device cuda                      (16 GB cap, 2 threads)

Loading = world_evals/run_eval_wave.py (imported through run_newfacts_wave.py): the model code from <run>/code_snapshot
(no bytecode), the checkpoint on the CPU, <ckpt>.memory.pt loaded strict (read only; its sha256 is checked before and
after with --hash-file 1), then .to(device); same knockout context (rw.KO: ko_wave = bm.enabled False). Device env
(CUDA hidden unless --device cuda, 2 threads, set_per_process_memory_fraction 16 GB) is set by run_eval_wave at import.

Passes over the SAME val streams (batch row b = stream b, sequence j at step j, as the stream loader trains):
  off        ko_wave: the memory module is not called (no reads, no writes)
  on_static  mem_on, no writes: the checkpoint's memory as loaded (it never saw these worlds)
  on_write   mem_on; at each step the batch is first SCORED (eval forward, no writes), then STUDIED: the write rule of
             run_newfacts_wave.study (memory module in its live training state, exploration 0, sequence drop 0, brain
             requires_grad False, one probe backward for the write target, after_backward; memory parameter grads
             dropped, D_grad zeroed, after_step never called). So a question in sequence j can only use what the
             memory wrote from sequences < j. The writes go to the in-RAM memory only (nothing is saved).
The brain (every parameter / buffer outside bm.*) and the memory parameters + D are checksum-asserted unchanged.

Metrics per question (answer = the object tokens, e.g. " Flegrerstad"; gold logits at x positions pos-1 ..):
  exact (all answer tokens argmax), first (first token argmax), logp/tok, vrank (rank of the gold first token in the
  16,384 vocabulary), crank (rank of the gold first token among the distinct first tokens of the relation's
  candidates: city 300, job 200, pet 40, year 86, friend = the world's other cast members; chance = (n+1)/2).
Reported by relation, form (Q / cloze) and distance (1-4 sequences after the fact), and on_write - off.
Output: world_change/<run>_<ckpt>_change[_tag].json + .txt (never into runs/).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALS = HERE.parent / "world_evals"
sys.path.insert(0, str(EVALS))
sys.dont_write_bytecode = True
import run_newfacts_wave as rn                      # noqa: E402  (-> run_eval_wave: env, 16 GB cap, snapshot path)
rw = rn.rw
import argparse                                     # noqa: E402
import json                                         # noqa: E402
import time                                         # noqa: E402
import types                                        # noqa: E402
from collections import defaultdict                 # noqa: E402

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import torch.nn.functional as F                     # noqa: E402

RELS = ["city", "job", "pet", "year", "friend"]


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\Eitan\Python\NewBrain\runs\world_wave26_12k\final.pt")
    ap.add_argument("--config", default="")
    ap.add_argument("--data", default="X:/Datasets/world_change_mix")
    ap.add_argument("--src", default="X:/Datasets/wiki_world_mix", help="world.json (the attribute vocabularies)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--capacity", default="dropless")
    ap.add_argument("--streams", type=int, default=0, help="first N val streams (0 = all; = the batch)")
    ap.add_argument("--max-seqs", type=int, default=0, help="first N sequences per stream (0 = all)")
    ap.add_argument("--modes", default="off,on_static,on_write")
    ap.add_argument("--addr-learn", type=int, default=1)
    ap.add_argument("--hash-file", type=int, default=1, help="sha256 of the memory file before / after")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=2)
    a, _ = ap.parse_known_args()
    assert a.device == rw.DEV, "--device is read by run_eval_wave at import"
    return a


# ---------------------------------------------------------------- the write step (run_newfacts_wave.study, per batch)
class Study:
    def __init__(self, model, addr_learn):
        self.model, self.bm, self.addr_learn = model, model.bm, addr_learn

    def __enter__(self):
        bm = self.bm
        self.keep = dict(explore=bm.explore, drop=bm.drop)
        bm.explore, bm.drop = 0.0, 0.0
        self.req = {}
        for n, p in self.model.named_parameters():
            if not n.startswith("bm."):
                self.req[n] = p.requires_grad
                p.requires_grad_(False)
        self.orig_bf = bm.begin_forward

        def begin_forward():
            bm.train_forward = True
            self.orig_bf()
        bm.begin_forward = begin_forward
        self.c0 = dict(bm.cnt)
        self.n_probe = 0
        return self

    def step(self, x, y, dev):
        bm = self.bm
        if not self.addr_learn:
            bm._rows = []
        bm.train()
        try:
            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    logits = self.model(x)
                bm.select_writes(logits, y, x)
                rec = bm._cache
                if rec is not None and rec.get("probe") is not None:
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                    loss.backward(inputs=[rec["probe"]])
                    self.n_probe += int(rec["probe"].grad is not None)
            if not self.addr_learn:
                bm._rows = []
            bm.after_backward()
            for p in bm.parameters():
                p.grad = None
            bm.D_grad.zero_()
        finally:
            bm._cache = None
            bm.eval()
            bm._live = False
        del logits

    def __exit__(self, *exc):
        bm = self.bm
        bm.eval()
        bm._live = False
        bm.begin_forward = self.orig_bf
        bm.explore, bm.drop = self.keep["explore"], self.keep["drop"]
        for n, p in self.model.named_parameters():
            if n in self.req:
                p.requires_grad_(self.req[n])
        bm._cache, bm._rows, bm._samp = None, [], None

    def counts(self):
        return {k: self.bm.cnt[k] - self.c0.get(k, 0) for k in ("writes_g", "writes_p", "writes_both", "confirms_g",
                                                                "confirms_p", "seqs") if k in self.bm.cnt}


# ---------------------------------------------------------------- scoring one batch
@torch.no_grad()
def score_batch(model, x, toks, its, cand_first, dev):
    """x: (B, T) inputs; toks: (B, T + 1) numpy; its: [(row, item index, pos, n_ans, cand first-token ids)]."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        logits = model(x)
    out = []
    for r, ii, pos, n, cands in its:
        lg = logits[r, pos - 1:pos - 1 + n].float()
        lp = F.log_softmax(lg, -1)
        gold = torch.as_tensor(toks[r, pos:pos + n].astype(np.int64), device=lg.device)
        hit = lp.argmax(-1) == gold
        g0 = lp[0, gold[0]]
        vrank = int((lp[0] > g0).sum()) + 1
        cf = torch.as_tensor(cands, device=lg.device)
        crank = int((lp[0, cf] > g0).sum()) + 1
        out.append((ii, float(lp.gather(1, gold[:, None]).sum()), bool(hit[0]), bool(hit.all()), n, vrank, crank,
                    len(cands)))
    del logits
    return out


def summarize(res, items):
    by = defaultdict(list)
    for ii, lp, first, exact, n, vr, cr, nc in res:
        rel, form, dist = RELS[items["rel"][ii]], "Q" if items["form"][ii] == 0 else "cloze", int(items["dist"][ii])
        for k in ("ALL", f"rel_{rel}", f"form_{form}", f"dist_{dist}"):
            by[k].append((lp / n, first, exact, vr, cr, nc))
    out = {}
    for k, v in by.items():
        v = np.array(v, float)
        out[k] = dict(n=len(v), exact=float(v[:, 2].mean()), first=float(v[:, 1].mean()), logp_tok=float(v[:, 0].mean()),
                      vrank_mean=float(v[:, 3].mean()), vrank_top1=float((v[:, 3] == 1).mean()),
                      crank_mean=float(v[:, 4].mean()), crank_top1=float((v[:, 4] == 1).mean()),
                      crank_chance=float(((v[:, 5] + 1) / 2).mean()), crank_top1_chance=float((1 / v[:, 5]).mean()))
    return out


def main():
    a = args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(2)
    t_all = time.time()
    ckpt, data = Path(a.ckpt), Path(a.data)
    meta = json.loads((data / "meta.json").read_text())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))
    world = json.loads((Path(a.src) / "world.json").read_text())
    names = json.loads((data / "names.json").read_text())["val"]

    # ---- the val streams + items
    SEQ, S, n_seq, L = meta["seq"], meta["val"]["streams"], meta["val"]["seqs_per_stream"], meta["val"]["stream_len"]
    val = np.fromfile(data / "val.bin", dtype=np.uint16).reshape(S, L)
    B = a.streams or S
    J = min(a.max_seqs or n_seq, n_seq)
    items = dict(np.load(data / "val_items.npz"))
    wo = dict(np.load(data / "val_worlds.npz"))
    first = lambda s: tok.encode(s).ids[0]
    cf_fixed = [sorted({first(" " + c["name"]) for c in world["cities"]}), sorted({first(" " + j) for j in world["jobs"]}),
                sorted({first(" " + p) for p in world["pets"]}), sorted({first(f" {y}") for y in range(1920, 2006)})]
    name_first = [first(" " + n) for n in names]
    by_seq = defaultdict(list)
    for ii in range(len(items["stream"])):
        s, j = int(items["stream"][ii]), int(items["seq"][ii])
        if s >= B or j >= J:
            continue
        r = int(items["rel"][ii])
        if r == 4:
            cast = wo["cast"][items["world"][ii]]
            cands = sorted({name_first[c] for c in cast if c >= 0 and c != items["name"][ii]})
        else:
            cands = cf_fixed[r]
        by_seq[j].append((s, ii, int(items["pos"][ii]), int(items["n_ans"][ii]), cands))
    n_items = sum(len(v) for v in by_seq.values())
    print(f"[change] {data}: {B} streams x {J} sequences of {SEQ} = {B * J * SEQ:,} tokens, {n_items:,} questions",
          flush=True)

    # ---- load (run_eval_wave: checkpoint + memory file strict, CPU then device)
    mpath = rw.memory_api.memory_path(ckpt)
    h_file0 = rn.sha256_file(mpath) if a.hash_file and mpath.exists() else None
    la = types.SimpleNamespace(ckpt=str(ckpt), config=a.config, capacity=a.capacity, device=a.device, k=a.k)
    model, step, *_ = rw.load_model(la)
    model.eval()
    bm = model.bm
    h_brain0, h_mpar0, h_mem0 = rn.brain_checksum(model), rn.mem_param_checksum(bm), rw.mem_checksum(bm)
    print(f"[change] loaded {ckpt} (step {step}); brain {h_brain0}; memory params+D {h_mpar0}; memory state {h_mem0}; "
          f"memory file sha256 {h_file0} ({time.time() - t_all:.0f}s)", flush=True)

    modes = a.modes.split(",")
    assert modes.index("on_write") == len(modes) - 1 if "on_write" in modes else True, "on_write must run last"
    R, W = {}, None
    for mode in modes:
        t0 = time.time()
        res = []
        ctx = rw.KO(model, "wave", "off" if mode == "off" else None)
        study = Study(model, a.addr_learn) if mode == "on_write" else None
        with ctx:
            if study:
                study.__enter__()
            try:
                for j in range(J):
                    toks = val[:B, j * SEQ:j * SEQ + SEQ + 1].astype(np.int64)
                    xy = torch.from_numpy(toks).to(a.device)
                    x, y = xy[:, :-1], xy[:, 1:]
                    res += score_batch(model, x, toks, by_seq.get(j, []), None, a.device)
                    if study:
                        study.step(x, y, a.device)
                    if j % max(1, J // 10) == 0 or j == J - 1:
                        e = np.mean([r[3] for r in res]) if res else float("nan")
                        print(f"  {mode}: seq {j + 1}/{J}, {len(res)} questions, exact {e:.4f}"
                              + (f", writes {study.counts()}" if study else "") + f" ({time.time() - t0:.0f}s)",
                              flush=True)
            finally:
                if study:
                    W = study.counts()
                    W["probe_batches"] = study.n_probe
                    study.__exit__()
        R[mode] = summarize(res, items)
        R[mode]["time_s"] = time.time() - t0
        h = rw.mem_checksum(bm)
        if mode != "on_write":
            assert h == h_mem0, f"{mode} changed the memory"
    h_brain1, h_mpar1, h_mem1 = rn.brain_checksum(model), rn.mem_param_checksum(bm), rw.mem_checksum(bm)
    assert h_brain1 == h_brain0, "brain weights / buffers changed"
    assert h_mpar1 == h_mpar0, "memory parameters / D changed"
    h_file1 = rn.sha256_file(mpath) if h_file0 else None
    assert h_file1 == h_file0, "the memory FILE changed"
    peak = torch.cuda.max_memory_allocated() / 1e9 if a.device == "cuda" else None

    # ---- report
    keys = sorted(R[modes[0]], key=lambda k: (k != "ALL", k))
    keys = [k for k in keys if k != "time_s"]
    lines = [f"CHANGING-FACTS eval: {ckpt} (step {step}); {B} streams x {J} seqs, {n_items} questions; device {a.device}",
             f"checks: brain {'UNCHANGED' if h_brain1 == h_brain0 else 'CHANGED'}, memory params+D "
             f"{'UNCHANGED' if h_mpar1 == h_mpar0 else 'CHANGED'}, memory state {h_mem0} -> {h_mem1}, memory file "
             f"{h_file0} -> {h_file1}; writes {W}",
             f"{'key':14s} " + " ".join(f"{m + ' exact':>16s}" for m in modes) + " " + " ".join(
                 f"{m + ' crank1':>17s}" for m in modes) + f" {'chance':>7s}"]
    for k in keys:
        lines.append(f"{k:14s} " + " ".join(f"{R[m][k]['exact']:16.4f}" for m in modes) + " "
                     + " ".join(f"{R[m][k]['crank_top1']:17.4f}" for m in modes)
                     + f" {R[modes[0]][k]['crank_top1_chance']:7.4f}   (n {R[modes[0]][k]['n']}; lp/tok "
                     + ", ".join(f"{m} {R[m][k]['logp_tok']:.3f}" for m in modes) + "; mean crank "
                     + ", ".join(f"{m} {R[m][k]['crank_mean']:.1f}" for m in modes) + ")")
    if "on_write" in R and "off" in R:
        d = R["on_write"]["ALL"]["exact"] - R["off"]["ALL"]["exact"]
        dc = R["on_write"]["ALL"]["crank_top1"] - R["off"]["ALL"]["crank_top1"]
        lines.append(f"on_write - off: exact {d:+.4f}, crank top1 {dc:+.4f}, logp/tok "
                     f"{R['on_write']['ALL']['logp_tok'] - R['off']['ALL']['logp_tok']:+.4f}")
    lines.append(f"total {time.time() - t_all:.0f}s" + (f", CUDA peak {peak:.2f} GB" if peak else ""))
    txt = "\n".join(lines)
    print(txt, flush=True)
    out = Path(a.out) if a.out else HERE / f"{ckpt.parent.name}_{ckpt.stem}_change{('_' + a.tag) if a.tag else ''}.json"
    assert "runs" not in out.resolve().parts
    out.write_text(json.dumps(dict(args=vars(a), step=step, results=R, writes=W, checks=dict(
        brain0=h_brain0, brain1=h_brain1, mem_params0=h_mpar0, mem_params1=h_mpar1, mem0=h_mem0, mem1=h_mem1,
        file0=h_file0, file1=h_file1), cuda_peak_gb=peak), indent=1))
    out.with_suffix(".txt").write_text(txt + "\n")
    print(f"wrote {out} (+ .txt)", flush=True)


if __name__ == "__main__":
    main()
