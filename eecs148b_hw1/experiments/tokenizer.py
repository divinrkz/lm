"""
TinyStories tokenizer experiments:
  (a) Sample 10 documents, report bytes/token (compression ratio).
  (b) Encode train/validation CSVs, save NumPy uint16 arrays.

Run from repo root:
  uv run python -m eecs148b_hw1.bpe.tokenizer_experiments
  uv run python eecs148b_hw1/bpe/tokenizer_experiments.py

uint16 fits vocab sizes < 65536; use uint32 only if you exceed that.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

# Allow `python eecs148b_hw1/bpe/tokenizer_experiments.py` without installing the package.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from eecs148b_hw1.bpe.tokenizer import BPETokenizer

EOT = "<|endoftext|>"


def load_texts(csv_path: Path, text_column: str = "text") -> list[str]:
    texts: list[str] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if text_column not in fields:
            raise ValueError(f"Column {text_column!r} not in {fields}")
        for row in reader:
            t = row.get(text_column, "")
            if t is not None and str(t).strip():
                texts.append(str(t))
    return texts


def experiment_a(
    tok: BPETokenizer,
    texts: list[str],
    n_sample: int = 10,
    seed: int = 42,
) -> tuple[float, list[str]]:
    rng = random.Random(seed)
    k = min(n_sample, len(texts))
    sample = rng.sample(texts, k)
    total_bytes = 0
    total_tokens = 0
    for doc in sample:
        b = doc.encode("utf-8")
        ids = tok.encode(doc)
        total_bytes += len(b)
        total_tokens += len(ids)
    ratio = total_bytes / total_tokens if total_tokens else float("nan")
    return ratio, sample


def encode_corpus_joined(texts: list[str], tok: BPETokenizer, separator: str = EOT) -> list[int]:
    """Join documents with the special token between them, then encode."""
    if not texts:
        return []
    corpus = separator.join(texts) + separator
    return tok.encode(corpus)


def save_uint16(ids: list[int], path: Path) -> None:
    arr = np.asarray(ids, dtype=np.uint32)
    mx = int(arr.max()) if arr.size else 0
    if mx >= 2**16:
        raise ValueError(f"Max token id {mx} does not fit in uint16 (need < 65536)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.uint16))


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyStories tokenizer experiments")
    parser.add_argument("--vocab", default="out/tinystories_vocab.json")
    parser.add_argument("--merges", default="out/tinystories_merges.txt")
    parser.add_argument("--train-csv", default="data/tinystories/train.csv")
    parser.add_argument("--val-csv", default="data/tinystories/validation.csv")
    parser.add_argument("--out-dir", default="out/tokenized")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    vocab_path = root / args.vocab
    merges_path = root / args.merges
    train_csv = root / args.train_csv
    val_csv = root / args.val_csv
    out_dir = root / args.out_dir

    tok = BPETokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=[EOT],
    )
    vocab_max = max(tok.vocab.keys())
    print(f"Loaded tokenizer: {len(tok.vocab)} types, max id = {vocab_max}")

    if not train_csv.is_file():
        print(f"Train CSV not found: {train_csv}", file=sys.stderr)
        sys.exit(1)

    print("Loading training texts...")
    train_texts = load_texts(train_csv)
    print(f"  documents: {len(train_texts):,}")

    ratio, _sample_docs = experiment_a(tok, train_texts, n_sample=10, seed=args.seed)
    print()
    print("(a) Compression ratio (UTF-8 bytes / token) on 10 random documents:")
    print(f"    {ratio:.4f} bytes/token  (seed={args.seed})")

    print()
    print("(b) Encoding train split and saving uint16 .npy ...")
    train_ids = encode_corpus_joined(train_texts, tok)
    save_uint16(train_ids, out_dir / "train_tokens.npy")
    print(f"    train: {len(train_ids):,} tokens -> {out_dir / 'train_tokens.npy'}")

    if val_csv.is_file():
        val_texts = load_texts(val_csv)
        val_ids = encode_corpus_joined(val_texts, tok)
        save_uint16(val_ids, out_dir / "validation_tokens.npy")
        print(f"    val:   {len(val_ids):,} tokens -> {out_dir / 'validation_tokens.npy'}")
    else:
        print(f"    (validation CSV missing: {val_csv})")

    print()
    print(
        "uint16 is appropriate when every token id is < 65536 "
        f"(here max id = {vocab_max})."
    )


if __name__ == "__main__":
    main()
