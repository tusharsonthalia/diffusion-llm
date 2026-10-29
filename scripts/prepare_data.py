"""Download OpenWebText and tokenize it with GPT-2 into flat uint16 token files.

    python scripts/prepare_data.py --out-dir data/openwebtext --shards 4

Output (in --out-dir):
    train.bin   uint16 token stream: every document followed by <|endoftext|>
    val.bin     same, for the held-out documents
    meta.json   tokenizer, vocab sizes, special ids, token counts, source shards

OpenWebText is the corpus MDLM / BD3-LM pretrain on. It is published on the HF Hub as
80 parquet shards (Skylion007/openwebtext, ~110M GPT-2 tokens each, ~9B total). For
timing and kernel workloads a few shards are plenty; pass --shards 80 for all of it.

Loading (sequence packing): np.memmap(path, dtype=np.uint16, mode="r"), then cut
contiguous windows of seq_len tokens. Documents are separated by EOT, so windows
span document boundaries, which is what the reference's "wrap" mode does.

GPT-2 has 50257 tokens (ids 0..50256, EOT = 50256). The [MASK] token is appended by
the model at id 50257, so vocab_size for MDLM / BD3-LM is 50258. All ids fit in
uint16, halving the file size versus int32.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

REPO_ID = "Skylion007/openwebtext"
NUM_SHARDS = 80


def shard_filename(i: int) -> str:
    return f"plain_text/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet"


def tokenize_docs(enc: tiktoken.Encoding, texts: list[str], num_threads: int) -> np.ndarray:
    """Encode documents and join them, each followed by EOT, into one uint16 array."""
    # encode_ordinary: special-token strings inside the text are treated as plain text
    ids = enc.encode_ordinary_batch(texts, num_threads=num_threads)
    eot = enc.eot_token
    total = sum(len(d) for d in ids) + len(ids)
    out = np.empty(total, dtype=np.uint16)
    pos = 0
    for d in ids:
        out[pos:pos + len(d)] = d
        pos += len(d)
        out[pos] = eot
        pos += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("data/openwebtext"))
    parser.add_argument("--shards", type=int, default=4, help=f"how many of the {NUM_SHARDS} shards to use")
    parser.add_argument("--val-docs", type=int, default=5000, help="documents held out from shard 0")
    parser.add_argument("--batch-docs", type=int, default=10_000, help="documents tokenized per batch")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF download cache (default: HF_HOME)")
    args = parser.parse_args()

    assert 1 <= args.shards <= NUM_SHARDS, f"--shards must be in [1, {NUM_SHARDS}]"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab <= np.iinfo(np.uint16).max + 1, "token ids must fit in uint16"

    counts = {"train": 0, "val": 0}
    docs = {"train": 0, "val": 0}
    start = time.perf_counter()

    # write fresh files (re-running overwrites rather than appends)
    with (args.out_dir / "train.bin").open("wb") as f_train, (args.out_dir / "val.bin").open("wb") as f_val:
        files = {"train": f_train, "val": f_val}

        for shard in range(args.shards):
            path = hf_hub_download(REPO_ID, shard_filename(shard), repo_type="dataset", cache_dir=args.cache_dir)
            texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()

            # hold out the first val_docs documents of shard 0
            if shard == 0:
                splits = [("val", texts[:args.val_docs]), ("train", texts[args.val_docs:])]
            else:
                splits = [("train", texts)]

            for split, split_texts in splits:
                for i in range(0, len(split_texts), args.batch_docs):
                    tokens = tokenize_docs(enc, split_texts[i:i + args.batch_docs], args.threads)
                    tokens.tofile(files[split])
                    counts[split] += len(tokens)
                    docs[split] += len(split_texts[i:i + args.batch_docs])

            elapsed = time.perf_counter() - start
            print(f"shard {shard + 1}/{args.shards}: {counts['train']:,} train / {counts['val']:,} val tokens "
                  f"({elapsed:.0f}s)", flush=True)

    meta = {
        "dataset": REPO_ID,
        "shards": [shard_filename(i) for i in range(args.shards)],
        "tokenizer": "gpt2 (tiktoken)",
        "dtype": "uint16",
        "tokenizer_vocab_size": enc.n_vocab,  # 50257
        "eot_token": enc.eot_token,            # 50256, appended after every document
        "mask_index": enc.n_vocab,             # 50257, added by the model
        "model_vocab_size": enc.n_vocab + 1,   # 50258, including [MASK]
        "num_tokens": counts,
        "num_docs": docs,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
