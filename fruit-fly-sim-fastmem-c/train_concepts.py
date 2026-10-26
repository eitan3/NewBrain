"""Stage 1 of next-concept modelling: train the multiscale residual VQ concept tokenizer on TinyStories.

    python -u train_concepts.py --run-name concepts_c32 --steps 6000

Writes runs/<run-name>/concept_tok.pt (cfg + state) and train.log. Reports exact token reconstruction
accuracy when decoding from the first k levels (k = 1..K) and codebook usage per level.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from concepts import ConceptTokenizer, nearest, resize
from train_lm import Tee

HERE = Path(__file__).resolve().parent


def batch(data, n, chunk, gen, dev):
    ix = gen.randint(0, len(data) - chunk, n)
    return torch.from_numpy(np.stack([data[i:i + chunk].astype(np.int64) for i in ix])).to(dev)


@torch.no_grad()
def evaluate(tok, val, chunk, dev, n=4096):
    tok.eval()
    x = batch(val, n, chunk, np.random.RandomState(7), dev)
    z = tok.encode_latent(x)
    acc = []
    for k in range(1, len(tok.levels) + 1):
        _, recon, _ = tok.quantize(z, k)
        acc.append((tok.decode(recon).argmax(-1) == x).float().mean().item())
    codes, _, _ = tok.quantize(z)
    usage = [len(torch.unique(c)) / tok.V for c in codes]
    tok.train()
    return acc, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="concepts_c32")
    ap.add_argument("--runs", default=str(HERE.parent / "runs"))
    ap.add_argument("--data", default=str(HERE.parent / "data" / "ts4k"))
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--nested", type=float, default=0.5)
    ap.add_argument("--revive-every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run = Path(a.runs) / a.run_name
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "train.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, logf), Tee(sys.stderr, logf)
    (run / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(0)
    train = np.memmap(Path(a.data) / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(Path(a.data) / "val.bin", dtype=np.uint16, mode="r")
    cfg = dict(vocab=4096, d=a.d, chunk=a.chunk, codebook=a.codebook)
    tok = ConceptTokenizer(**cfg).to(a.device)
    opt = torch.optim.AdamW(tok.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen = np.random.RandomState(0)
    used = torch.zeros(len(tok.levels), a.codebook, device=a.device)
    t0 = time.time()
    print(f"concept tokenizer: levels {tok.levels}, codebook {a.codebook}, d {a.d}, "
          f"{sum(p.numel() for p in tok.parameters())/1e6:.2f}M params", flush=True)
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / 500) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = batch(train, a.batch, a.chunk, gen, a.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
            ce, vq, codes, _ = tok(x, a.nested)
            loss = ce + vq
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, c in enumerate(codes):
                used[k].index_fill_(0, c.flatten(), 1.0)
            if step == 1:
                used.zero_()  # first step: every code is "dead", so all codebooks start from real residuals
            if step == 1 or step % a.revive_every == 0:  # dead codes get a fresh vector from this batch's residuals
                z = tok.encode_latent(x)
                resid, recon = z, torch.zeros_like(z)
                for k, n in enumerate(tok.levels):
                    u = resize(resid, n).reshape(-1, tok.d)
                    dead = (used[k] == 0).nonzero().squeeze(1)
                    if len(dead):
                        tok.codebooks.data[k, dead] = u[torch.randint(0, len(u), (len(dead),), device=u.device)]
                    idx = nearest(u, tok.codebooks[k])
                    recon = recon + resize(tok.codebooks[k][idx].view(z.size(0), n, -1), tok.chunk)
                    resid = z - recon
                used.zero_()
        if step % 500 == 0 or step == a.steps:
            acc, usage = evaluate(tok, val, a.chunk, a.device)
            print(f"step {step:5d}  ce {ce.item():.3f}  vq {vq.item():.4f}  lr {lr:.1e}  "
                  f"recon acc by levels used {' '.join(f'{v:.3f}' for v in acc)}  "
                  f"codebook use {' '.join(f'{u:.2f}' for u in usage)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save({"cfg": cfg, "state": tok.state_dict()}, run / "concept_tok.pt")
    print("saved", run / "concept_tok.pt", flush=True)


if __name__ == "__main__":
    main()
