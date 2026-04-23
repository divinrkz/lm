from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from tqdm import tqdm

from eecs148b_hw1.tokenizer.tokenizer import BPETokenizer

EOT = "<|endoftext|>"

def sample_texts(csv_path: Path, k: int, seed: int, text_column: str = "text") -> list[str]:
    rng = random.Random(seed)
    texts: list[str] = []
    n = 0
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if text_column not in fields:
            raise ValueError(f"Column {text_column!r} not in {fields}")
        for row in reader:
            t = (row.get(text_column) or "").strip()
            if not t:
                continue
            n += 1
            if len(texts) < k:
                texts.append(t)
            else:
                j = rng.randint(1, n)
                if j <= k:
                    texts[j - 1] = t
    return texts


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

def experiment_sample_and_ratio(
    tok: BPETokenizer,
    sample_docs: list[str],
    *,
    id_preview: int = 24,
) -> float:
    """Encode each document, print ids (truncated), return aggregate bytes/token."""
    total_bytes = 0
    total_tokens = 0
    for i, doc in enumerate(sample_docs):
        b = doc.encode("utf-8")
        ids = tok.encode(doc)
        total_bytes += len(b)
        total_tokens += len(ids)
        head = ids[:id_preview]
        more = f" ... (+{len(ids) - id_preview} ids)" if len(ids) > id_preview else ""
        snippet = doc.replace("\n", " ")[:120] + ("…" if len(doc) > 120 else "")
        print(f"\n--- document {i + 1} ---")
        print(f"  preview: {snippet!r}")
        print(f"  utf-8 bytes: {len(b):,}  tokens: {len(ids):,}")
        print(f"  ids[:{id_preview}]: {head}{more}")
    ratio = total_bytes / total_tokens if total_tokens else float("nan")
    return ratio


def encode_corpus(texts: list[str], tok: BPETokenizer, separator: str = EOT) -> list[int]:
    if not texts:
        return []
    joined = separator.join(
        tqdm(texts, desc="Joining documents", unit="doc", leave=False, disable=not sys.stderr.isatty())
    )
    corpus = joined + separator
    return tok.encode(corpus)


def save_uint16(ids: list[int], path: Path) -> None:
    arr = np.asarray(ids, dtype=np.uint32)
    mx = int(arr.max()) if arr.size else 0
    if mx >= 2**16:
        raise ValueError(f"max token id {mx} does not fit in uint16 (need < 65536)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.uint16))


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyStories tokenizer")
    parser.add_argument("--vocab", default="out/tinystories_vocab.json")
    parser.add_argument("--merges", default="out/tinystories_merges.txt")
    parser.add_argument("--train-csv", default="datasets/tinystories/train.csv")
    parser.add_argument("--val-csv", default="datasets/tinystories/validation.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--n-sample", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-preview", type=int, default=24)
    parser.add_argument("--encode-splits", action="store_true")
    parser.add_argument("--out-dir", default="out/tokenized")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    vocab_path = root / args.vocab
    merges_path = root / args.merges
    train_csv = root / args.train_csv
    val_csv = root / args.val_csv
    out_dir = root / args.out_dir

    if not vocab_path.is_file() or not merges_path.is_file():
        print(
            f"Missing vocab/merges ({vocab_path}, {merges_path}). "
            "Train first: uv run python -m eecs148b_hw1.tokenizer.train",
            file=sys.stderr,
        )
        sys.exit(1)

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

    print(
        f"\nsampling {args.n_sample} documents from {train_csv} (seed={args.seed})..."
    )
    sample_docs = sample_texts(
        train_csv, args.n_sample, args.seed, text_column=args.text_column
    )
    if len(sample_docs) < args.n_sample:
        print(
            f"Warning: only {len(sample_docs)} non-empty rows (requested {args.n_sample}).",
            file=sys.stderr,
        )

    eot_label = "no <|endoftext|> between docs here"
    print(f"\nencoded token IDs (each document encoded separately; {eot_label}).")
    ratio = experiment_sample_and_ratio(
        tok, sample_docs, id_preview=args.id_preview
    )

    print("\n" + "----")
    print("compression ratio (aggregate over sampled documents):")
    print(f"  UTF-8 bytes / token = {ratio:.4f}")
    print("  (tokens are BPE ids; lower bytes/token means more compression.)")
    print("----")

    if args.encode_splits:
        print("\nencoding full train split (loads all rows into memory)...")
        train_texts = load_texts(train_csv, text_column=args.text_column)
        train_ids = encode_corpus(train_texts, tok)
        save_uint16(train_ids, out_dir / "train_tokens.npy")
        print(f"  train: {len(train_ids):,} tokens -> {out_dir / 'train_tokens.npy'}")

        if val_csv.is_file():
            val_texts = load_texts(val_csv, text_column=args.text_column)
            val_ids = encode_corpus(val_texts, tok)
            save_uint16(val_ids, out_dir / "val_tokens.npy")
            print(f"  val:   {len(val_ids):,} tokens -> {out_dir / 'val_tokens.npy'}")
        else:
            print(f"  (validation CSV missing: {val_csv})")


if __name__ == "__main__":
    main()
