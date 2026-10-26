"""Train a small byte-level BPE tokenizer on TinyStories and encode it to uint16 token files.

    python prepare_tinystories.py --src data --out data/ts4k --vocab 4096

Writes tokenizer.json, train.bin, val.bin (uint16, stories joined by <|endoftext|> = id 0).
"""
import argparse
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


def stories(path, chunk_bytes=64 << 20):
    """Yield lists of stories from a big text file without loading it all."""
    rest = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            parts = (rest + block).split(EOT)
            rest = parts.pop()
            yield [s.strip() for s in parts if s.strip()]
    if rest.strip():
        yield [rest.strip()]


def encode_file(tok, src, dst):
    eot = tok.token_to_id(EOT)
    n, t0 = 0, time.time()
    with open(dst, "wb") as out:
        for batch in stories(src):
            ids = []
            for enc in tok.encode_batch(batch):
                ids.extend(enc.ids)
                ids.append(eot)
            np.asarray(ids, dtype=np.uint16).tofile(out)
            n += len(ids)
            print(f"  {dst.name}: {n/1e6:.0f}M tokens ({time.time()-t0:.0f}s)", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--out", default="data/ts4k")
    ap.add_argument("--vocab", type=int, default=4096)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_txt, val_txt = src / "TinyStoriesV2-GPT4-train.txt", src / "TinyStoriesV2-GPT4-valid.txt"

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def sample():  # tokenizer trained on the first ~200 MB of stories
        seen = 0
        for batch in stories(train_txt):
            yield from batch
            seen += sum(map(len, batch))
            if seen > 200e6:
                return

    t0 = time.time()
    tok.train_from_iterator(sample(), trainer)
    tok.save(str(out / "tokenizer.json"))
    print(f"tokenizer: {tok.get_vocab_size()} tokens in {time.time()-t0:.0f}s", flush=True)
    nv = encode_file(tok, val_txt, out / "val.bin")
    nt = encode_file(tok, train_txt, out / "train.bin")
    print(f"done: train {nt:,} tokens, val {nv:,} tokens", flush=True)


if __name__ == "__main__":
    main()
