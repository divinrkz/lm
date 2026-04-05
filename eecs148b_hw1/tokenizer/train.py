from __future__ import annotations

import sys
from pathlib import Path

import collections
import regex as re
import json
import os
from functools import lru_cache
from tqdm import tqdm
from eecs148b_hw1.tokenizer.util import _has_adjacent_pair, _apply_merge, save_artifacts, load_artifacts


REGEX_P = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()

    segments = segment(corpus, special_tokens)
    word_freq, atomic_words = pretokenize(segments)

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


def segment(text: str, special_tokens: list[str]) -> list[tuple[bool, str]]:
    if not special_tokens:
        return [(False, text)]

    specials = sorted(special_tokens, key=len, reverse=True)
    n = len(text)
    i = 0
    out: list[tuple[bool, str]] = []

    while i < n:
        matched: str | None = None
        for s in specials:
            if text.startswith(s, i):
                matched = s
                break
        if matched is not None:
            out.append((True, matched))
            i += len(matched)
            continue

        next_i = n
        for s in specials:
            j = text.find(s, i)
            if j != -1 and j < next_i:
                next_i = j

        out.append((False, str(text[i:next_i])))
        i = next_i
    return out


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



# Running as `python eecs148b_hw1/bpe/scripts.py` does not set the package; add repo root.
if __name__ == "__main__":
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    corpus = root / "tests" / "fixtures" / "tinystories_sample_5M.txt"
    out_dir = root / "out"
    out_dir.mkdir(exist_ok=True)

    vocab, merges = train_bpe(
        corpus,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    save_artifacts(
        vocab,
        merges,
        out_dir / "tinystories_vocab.json",
        out_dir / "tinystories_merges.txt",
    )


if __name__ == "__main__":
    main()
