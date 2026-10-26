"""Stream English Wikipedia (no full download), train a byte-level BPE tokenizer on a sample, and cache only the
token ids for training. Resumable: rerun the same command and it continues where it stopped.

    python prepare_wikipedia.py --out X:/Datasets/wikipedia_en_20231101_wiki16k --vocab 16384 --train-tokens 400e6

Reads wikimedia/wikipedia 20231101.en parquet files over the network in row-group blocks (huggingface_hub + pyarrow);
no raw text is stored. Writes tokenizer.json (also copied to NewBrain/tokenizers/<name>/), train.bin, val.bin
(articles joined by <|endoftext|> = id 0; uint16 below 65,536 tokens, else uint32) and meta.json.
Training reads files 1-40 (train-00000 .. train-00039); validation = the first --val-articles articles of file 41
(train-00040), which training never reads; the tokenizer sample also comes from training files only.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
REPO = "datasets/wikimedia/wikipedia/20231101.en"
N_FILES = 41
HERE = Path(__file__).resolve().parent


def open_file(fs, file_idx):
    return fs.open(f"{REPO}/train-{file_idx:05d}-of-{N_FILES:05d}.parquet", "rb", block_size=8 << 20)


def groups(fs, file_idx, start_group=0):
    """Yield (row group index, article texts) for one parquet file (~1,000 articles per row group)."""
    with open_file(fs, file_idx) as f:
        pf = pq.ParquetFile(f)
        for g in range(start_group, pf.num_row_groups):
            texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            yield g, [t.strip() for t in texts if t and t.strip()]


def encode(tok, texts, eot, dtype):
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eot)
    return np.asarray(ids, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="X:/Datasets/wikipedia_en_20231101_wiki16k")
    ap.add_argument("--name", default="wiki16k", help="tokenizer name (copied to NewBrain/tokenizers/<name>)")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--train-tokens", type=float, default=400e6)
    ap.add_argument("--val-articles", type=int, default=2000)
    ap.add_argument("--tok-sample-bytes", type=float, default=200e6)
    ap.add_argument("--run-name", default="", help="(ignored; lets launch_queue.ps1 run this script)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def save_meta():
        meta_path.write_text(json.dumps(meta, indent=1))

    # ---- 1. tokenizer (skipped on resume)
    tok_path = out / "tokenizer.json"
    if not tok_path.exists():
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOT],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

        def sample():  # the first ~200 MB of articles of the training files
            seen = 0
            for fi in range(N_FILES - 1):
                for _, texts in groups(fs, fi):
                    yield from texts
                    seen += sum(map(len, texts))
                    if seen > a.tok_sample_bytes:
                        return

        t0 = time.time()
        tok.train_from_iterator(sample(), trainer)
        tok.save(str(tok_path))
        print(f"tokenizer: {tok.get_vocab_size()} tokens in {time.time()-t0:.0f}s", flush=True)
    tok = Tokenizer.from_file(str(tok_path))
    repo_tok = HERE.parent / "tokenizers" / a.name
    repo_tok.mkdir(parents=True, exist_ok=True)
    shutil.copy(tok_path, repo_tok / "tokenizer.json")
    eot = tok.token_to_id(EOT)
    dtype = np.uint16 if tok.get_vocab_size() <= 65536 else np.uint32
    meta.update(source="wikimedia/wikipedia 20231101.en (streamed, parquet row groups)", tokenizer=a.name,
                vocab=tok.get_vocab_size(), eot_id=eot, dtype=np.dtype(dtype).name)
    save_meta()

    # ---- 2. validation: the first --val-articles articles of the last file
    if not meta.get("val_done"):
        n_art, chunks = 0, []
        for _, texts in groups(fs, N_FILES - 1):
            take = texts[:a.val_articles - n_art]
            chunks.append(encode(tok, take, eot, dtype))
            n_art += len(take)
            if n_art >= a.val_articles:
                break
        val = np.concatenate(chunks)
        val.tofile(out / "val.bin")
        meta.update(val_done=True, val_tokens=int(len(val)), val_articles=n_art,
                    val_source=f"first {n_art} articles of train-{N_FILES-1:05d} (file 41)")
        save_meta()
        print(f"val: {len(val):,} tokens, {n_art} articles", flush=True)

    # ---- 3. training tokens, resumable: progress = (file, next row group, tokens written)
    pos = meta.get("train_pos", dict(file=0, group=0, tokens=0, articles=0))
    train_bin = out / "train.bin"
    if train_bin.exists():  # cut any partial write after the last saved position
        with open(train_bin, "r+b") as f:
            f.truncate(pos["tokens"] * np.dtype(dtype).itemsize)
    limit, t0 = int(a.train_tokens), time.time()
    with open(train_bin, "ab") as outf:
        for fi in range(pos["file"], N_FILES - 1):
            if pos["tokens"] >= limit:
                break
            for g, texts in groups(fs, fi, pos["group"] if fi == pos["file"] else 0):
                ids = encode(tok, texts, eot, dtype)[:limit - pos["tokens"]]
                ids.tofile(outf)
                outf.flush()
                pos = dict(file=fi, group=g + 1, tokens=pos["tokens"] + len(ids), articles=pos["articles"] + len(texts))
                meta.update(train_pos=pos, train_files=f"train-00000 .. train-{fi:05d}")
                save_meta()
                print(f"  train: {pos['tokens']/1e6:.1f}M tokens, {pos['articles']} articles, file {fi+1}/40 "
                      f"group {g} ({time.time()-t0:.0f}s)", flush=True)
                if pos["tokens"] >= limit:
                    break
            else:
                pos = dict(pos, file=fi + 1, group=0)
    meta.update(train_tokens=pos["tokens"], train_articles=pos["articles"], done=pos["tokens"] >= limit)
    save_meta()
    print(f"done: {meta}", flush=True)


if __name__ == "__main__":
    main()
