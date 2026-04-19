import collections
import json
import os
from functools import lru_cache
from tqdm import tqdm

import regex as re

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
        )
    else:
        merge_iter = range(num_merges)

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

@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    """Map each byte 0..255 to a single Unicode character (GPT-2 style, for vocab/merge files)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(x) for x in cs]))

def unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}

def _bytes_to_string(b: bytes) -> str:
    b2u = bytes_to_unicode()
    return "".join(b2u[x] for x in b)

def save_bpe_artifacts(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    *,
    vocab_indent: int | None = 2,
) -> None:
    vocab_out: dict[str, int] = {}
    for idx, tok in sorted(vocab.items()):
        vocab_out[_bytes_to_string(tok)] = idx
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=vocab_indent)
        f.write("\n")
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{_bytes_to_string(a)} {_bytes_to_string(b)}\n")

def load_bpe_artifacts(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load files written by :func:`save_bpe_artifacts` (GPT-2-style token strings)."""
    dec = unicode_to_bytes()
    with open(vocab_path, encoding="utf-8") as f:
        raw: dict[str, int] = json.load(f)
    vocab: dict[int, bytes] = {}
    for token_str, tid in raw.items():
        tid_int = int(tid)
        vocab[tid_int] = bytes([dec[c] for c in token_str])
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                continue
            a, b = parts
            merges.append((bytes([dec[c] for c in a]), bytes([dec[c] for c in b])))
    return vocab, merges

def _has_adjacent_pair(seq: list[bytes], a: bytes, b: bytes) -> bool:
    for i in range(len(seq) - 1):
        if seq[i] == a and seq[i + 1] == b:
            return True
    return False


def _apply_merge(pieces: list[bytes], a: bytes, b: bytes) -> list[bytes]:
    if len(pieces) < 2:
        return pieces
    out: list[bytes] = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and pieces[i] == a and pieces[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


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
