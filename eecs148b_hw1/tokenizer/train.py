from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from pathlib import Path
from tqdm import tqdm

import regex as re
from .util import REGEX_P, _apply_merge, _has_adjacent_pair, save_artifacts, segment


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    corpus_override: str | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if corpus_override is not None:
        corpus = corpus_override
    else:
        with open(input_path, encoding="utf-8") as f:
            corpus = f.read()

    segments = segment(corpus, special_tokens)
    word_freq, atomic_words = pretokenize(segments)
    print('am here')

    pieces_map: dict[str, list[bytes]] = {}
    for word, cnt in word_freq.items():
        if cnt <= 0:
            continue
        if word in atomic_words:
            pieces_map[word] = [word.encode("utf-8")]
        else:
            pieces_map[word] = [bytes([b]) for b in word.encode("utf-8")]

    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    pair_stats: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    for word, cnt in word_freq.items():
        if cnt <= 0:
            continue
        seq = pieces_map[word]
        for i in range(len(seq) - 1):
            pair_stats[(seq[i], seq[i + 1])] += cnt

    tqdm.write(
        f"[train_bpe] {input_path!s}: {len(corpus):,} chars, "
        f"{len(word_freq):,} pretoken types, {num_merges:,} merges to learn"
    )
    merge_iter = tqdm(
        range(num_merges),
        desc="BPE merges",
        unit="merge",
        mininterval=0.5,
        disable=not sys.stderr.isatty(),
    )

    for _ in merge_iter:
        positive = [p for p, c in pair_stats.items() if c > 0]
        if not positive:
            break
        best = max(positive, key=lambda p: (pair_stats[p], p))
        merges.append(best)
        a, b = best
        for word, cnt in word_freq.items():
            if cnt <= 0:
                continue
            old = pieces_map[word]
            if not _has_adjacent_pair(old, a, b):
                continue
            for i in range(len(old) - 1):
                pair = (old[i], old[i + 1])
                pair_stats[pair] -= cnt
                if pair_stats[pair] == 0:
                    del pair_stats[pair]
            new = _apply_merge(old, a, b)
            pieces_map[word] = new
            for i in range(len(new) - 1):
                pair_stats[(new[i], new[i + 1])] += cnt

    vocab = _build_vocab(special_tokens, merges)
    return vocab, merges


def _build_vocab(
    special_tokens: list[str],
    merges: list[tuple[bytes, bytes]],
) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1
    for a, b in merges:
        vocab[next_id] = a + b
        next_id += 1
    return vocab



def pretokenize(segments: list[tuple[bool, str]]) -> tuple[dict[str, int], set[str]]:
    word_freq: dict[str, int] = collections.defaultdict(int)
    atomic_words: set[str] = set()

    for is_special, chunk in segments:
        if not chunk:
            continue
        if is_special:
            atomic_words.add(chunk)
            word_freq[chunk] += 1
        else:
            for m in re.finditer(REGEX_P, chunk):
                pretoken = m.group(0)
                word_freq[pretoken] += 1

    return dict(word_freq), atomic_words


def _corpus_with_document_boundaries(raw: str, eot: str) -> str:
    """If ``eot`` is absent, join blank-line-separated blocks (plain HF-style text)."""
    if eot in raw:
        return raw
    blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]
    if len(blocks) < 2:
        blocks = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) > 40]
    return eot.join(blocks) + eot


def _corpus_from_tinystories_csv(path: Path, eot: str, text_column: str = "text") -> str:
    """One row = one story; join with ``eot`` so the special token is a true document boundary."""
    pieces: list[str] = []
    sep = ""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if text_column not in fields:
            raise ValueError(f"CSV {path} has no {text_column!r} column; found {fields!r}")
        for row in reader:
            t = (row.get(text_column) or "").strip()
            if not t:
                continue
            pieces.append(sep)
            pieces.append(t)
            sep = eot
        pieces.append(eot)
    return "".join(pieces)


def main() -> None:
    default_train =   Path("datasets") / "tinystories" / "train.csv"
    default_vocab =  Path("out") / "tinystories_vocab.json"
    default_merges =  Path("out") / "tinystories_merges.txt"

    p = argparse.ArgumentParser(description="train byte-level BPE on TinyStories.")
    p.add_argument(  "--input", type=Path,  default=default_train)
    p.add_argument("--text-column", default="text")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument( "--special-token", default="<|endoftext|>")
    p.add_argument("--vocab-json", type=Path, default=default_vocab)
    p.add_argument("--merges-txt", type=Path, default=default_merges)
    p.add_argument("--no-inject-eot", action="store_true", help="For .txt only: do not insert the special token between blank-line-separated blocks when missing.")
   
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(
            f"Corpus not found: {inp}\n"
        )

    eot: str = args.special_token
    if inp.suffix.lower() == ".csv":
        corpus_text = _corpus_from_tinystories_csv(inp, eot, text_column=args.text_column)
    else:
        raw = inp.read_text(encoding="utf-8")
        corpus_text = raw if args.no_inject_eot else _corpus_with_document_boundaries(raw, eot)
        del raw

    vocab, merges = train_bpe(
        inp,
        vocab_size=args.vocab_size,
        special_tokens=[eot],
        corpus_override=corpus_text,
    )

    args.vocab_json.parent.mkdir(parents=True, exist_ok=True)
    args.merges_txt.parent.mkdir(parents=True, exist_ok=True)
    save_artifacts(vocab, merges, args.vocab_json, args.merges_txt)
    tqdm.write(f"wrote vocab ({len(vocab)} types) -> {args.vocab_json}")
    tqdm.write(f"wrote merges ({len(merges)} rules) -> {args.merges_txt}")

if __name__ == "__main__":
    main()
