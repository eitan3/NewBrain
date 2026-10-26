"""(fruit-fly-sim-wave copy of speed_wip/mem26/w1c_capture.py, unchanged logic; the default output goes to this folder's cap/,
never into mem26. Used by tests_wave_drift.py for the ckpt_9250 capture: python -B capture_xf.py --ckpt ckpt_9250.pt
--out cap/w1c_9250.npz)

W1c capture (adds the concept vectors mu_1..4, their write-back sum_l B_l A_l mu_l, and the pooled workspace p fed to the concept heads; layout xf|mu1|mu2|mu3|mu4|wb|p); W1 capture (plan 26 §8): xf (the RoutedMLP input, the WMS address) on paraphrase / cross-document views of the
same facts, from the blob run's 6K checkpoint. CPU only (CUDA hidden), 2 threads, checkpoint read-only via mmap,
nothing written under runs\\ (no bytecode: run with python -B; the code snapshot is imported read-only).

Documents are built exactly as the earlier T0a check (speed_wip/splat/realcheck/t0/t0a_capture.py build_docs):
per probe fact 3 documents with 3 different training templates + 1 with a held-out template, plus a cloze prompt.
The blob memory's reads are switched OFF (model.bm = None): the WMS would replace it.

    python -B w1_capture.py --n-late 200 --n-early 200 --mode eval
"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.dont_write_bytecode = True
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
HERE = Path(__file__).resolve().parent
RUN = Path(r"C:\Eitan\Python\NewBrain\runs\world_blob25d_12k")
SNAP = RUN / "code_snapshot"
T0 = Path(r"C:\Eitan\Python\NewBrain\speed_wip\splat\realcheck\t0")
WORLD = Path(r"X:\Datasets\wiki_world_mix")
WIKI_VAL = Path(r"X:\Datasets\wikipedia_en_20231101_wiki16k\val.bin")
sys.path.insert(0, str(SNAP))
T00 = time.time()


def log(*a):
    print(f"[{time.time() - T00:.0f}s]", *a, flush=True)


def load_t0a():
    """build_docs / tokenize_docs / pack from the earlier T0a capture, without importing its common.py."""
    src = (T0 / "t0a_capture.py").read_text()
    src = src.replace("from common import *  # noqa", "")
    ns = dict(np=np, world=lambda: json.loads((WORLD / "world.json").read_text()))
    exec(compile(src, "t0a_capture.py", "exec"), ns)
    return ns["build_docs"], ns["tokenize_docs"], ns["pack"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-late", type=int, default=200)
    ap.add_argument("--n-early", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-fit", type=int, default=48, help="world val + wiki val windows (512) for fitting the maps")
    ap.add_argument("--mode", default="eval", choices=["eval", "train"])
    ap.add_argument("--ckpt", default="ckpt_6000.pt")
    ap.add_argument("--out", default=str(HERE / "cap" / "w1c_9250.npz"))
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(WORLD / "tokenizer.json"))
    build_docs, tokenize_docs, pack = load_t0a()
    docs, cloze, probe = build_docs(tok, a.n_late, a.n_early)
    bad = tokenize_docs(tok, docs)
    rows, place = pack(docs, 512, np.random.default_rng(1))
    log(f"{len(docs)} docs ({bad} unaligned facts), {len(cloze)} cloze, {len(rows)} rows")

    from loop_brain import LoopBrain
    from spiking_lm import RoutedMLP
    assert not torch.cuda.is_available()
    ck = torch.load(RUN / a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model = LoopBrain(**ck["cfg"])
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    step = ck.get("step")
    del ck
    log(f"loaded {a.ckpt} step {step}; missing {len(missing)} unexpected {len(unexpected)}")
    model.bm = None                       # blob memory reads off
    if getattr(model, "mem", None) is not None:
        model.mem.write_on = False
    model.eval()
    model.K = 6
    for mod in model.modules():
        if isinstance(mod, RoutedMLP):
            mod.noise, mod.eval_noise, mod.cap = 0.0, (0.3 if a.mode == "train" else 0.0), (1.5 if a.mode == "train" else 0)
    mlp = model.blocks[0].mlp
    xs, ids = [], []
    mlp.register_forward_pre_hook(lambda m, inp: xs.append(inp[0].detach().float().reshape(-1, inp[0].shape[-1])))
    mlp.register_forward_hook(lambda m, inp, out: ids.append(m.last_idx.detach().clone()))
    th_in, th_mu = [], []
    hier = model.hier
    hier.register_forward_pre_hook(lambda m, inp: th_in.append(inp[0].detach().float().reshape(-1, inp[0].shape[-1])))

    def _hk(m, inp, out):
        mus = out[0]
        wb = m.write_back(mus).detach().float() if m.has_inject else torch.zeros_like(mus[0])
        th_mu.append(torch.cat([mu.detach().float() for mu in mus] + [wb], -1).reshape(-1, 5 * mus[0].shape[-1]))
    hier.register_forward_hook(_hk)
    g = np.random.default_rng(11)

    @torch.no_grad()
    def fwd(x):
        xs.clear(); ids.clear(); th_in.clear(); th_mu.clear()
        model.probe_offset = int(g.integers(0, 6)) if a.mode == "train" else 0
        torch.manual_seed(int(g.integers(1 << 30)))
        x = torch.from_numpy(x)
        B, T = x.shape
        model.run(x, 6)
        assert len(xs) == 6 and len(th_mu) >= 6, (len(xs), len(th_mu))
        xf = torch.stack(xs).view(6, B, T, -1)
        mu = torch.stack(th_mu[-6:]).view(6, B, T, -1)
        ws = torch.stack(th_in[-6:]).view(6, B, T, -1)
        return torch.cat([xf, mu, ws], -1).numpy(), torch.stack(ids).view(6, B, T, -1).numpy()

    # fit sample: all positions >= 8 of n_fit/2 world-val + n_fit/2 wiki-val windows
    fitw = []
    for path in (WORLD / "val_world.bin", WIKI_VAL):
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        st = np.linspace(9 * 512, len(mm) - 600, a.n_fit // 2).astype(np.int64)
        fitw += [np.asarray(mm[s:s + 512]).astype(np.int64) for s in st]
    fitw = np.stack(fitw)
    F_ = []
    for b in range(0, len(fitw), a.batch):
        xf, _ = fwd(fitw[b:b + a.batch])
        F_.append(xf[:, :, 8::4].reshape(6, -1, xf.shape[-1]).astype(np.float16))
    fit = np.concatenate(F_, 1)
    log(f"fit sample {fit.shape}")
    # document cue / value positions
    where, recs = [], []
    for di, d in enumerate(docs):
        r, s0 = place[di]
        for f in d["facts"]:
            if f["vtok"] < 0:
                continue
            where.append((r, s0 + f["ctok"], s0 + f["vtok"]))
            recs.append((di, f["person"], ["city", "job", "pet", "year", "friend"].index(f["rel"]), f["template"],
                         int(f["probe"]), ["d1", "d2", "d3", "held"].index(d["role"])))
    where = np.array(where)
    n = len(where)
    cue = np.zeros((6, n, 384 + 5 * 384 + 1024), np.float16)
    val = np.zeros((6, n, 384), np.float16)
    cue_e = np.zeros((6, n, 6), np.int16)
    for b in range(0, len(rows), a.batch):
        xf, ei = fwd(rows[b:b + a.batch])
        m = (where[:, 0] >= b) & (where[:, 0] < b + a.batch)
        wi = np.nonzero(m)[0]
        cue[:, wi] = xf[:, where[wi, 0] - b, where[wi, 1]]
        val[:, wi] = xf[:, where[wi, 0] - b, where[wi, 2], :384]
        cue_e[:, wi] = ei[:, where[wi, 0] - b, where[wi, 1]]
        if (b // a.batch) % 20 == 0:
            log(f"  doc rows {b}/{len(rows)}")
    # cloze prompts (last position)
    cz = []
    for b in range(0, len(cloze), 32):
        C = cloze[b:b + 32]
        T = max(len(c["ids"]) for c in C)
        x = np.zeros((len(C), T), np.int64)
        for i, c in enumerate(C):
            x[i, :len(c["ids"])] = c["ids"]
        xf, _ = fwd(x)
        last = np.array([len(c["ids"]) - 1 for c in C])
        cz.append(xf[:, np.arange(len(C)), last].astype(np.float16))
    Path(a.out).parent.mkdir(exist_ok=True)
    np.savez(a.out, step=step, feat_layout=np.array([384, 384, 384, 384, 384, 384, 1024]), fit=fit, cue=cue, val=val, cue_e=cue_e, rec=np.array(recs, np.int64),
             cloze=np.concatenate(cz, 1),
             cloze_rec=np.array([(c["person"], ["city", "job", "pet", "year", "friend"].index(c["rel"]), c["template"])
                                 for c in cloze], np.int64), probe=np.array(probe))
    log(f"wrote {a.out}: facts {n}, cloze {len(cloze)}")


if __name__ == "__main__":
    main()
