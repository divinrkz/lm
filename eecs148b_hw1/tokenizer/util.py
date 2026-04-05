import os
import json

from functools import lru_cache

__all__ = [
    "bytes_to_unicode",
    "unicode_to_bytes",
    "_bytes_to_string",
    "_has_adjacent_pair",
    "_apply_merge",
    "save_artifacts",
    "load_artifacts",
]

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


def _has_adjacent_pair(seq: list[bytes], a: bytes, b: bytes) -> bool:
    for i in range(len(seq) - 1):
        if seq[i] == a and seq[i + 1] == b:
            return True
    return False


def _apply_merge(seq: list[bytes], a: bytes, b: bytes) -> list[bytes]:
    """Replace every left-to-right adjacent (a, b) with a single piece a + b."""
    out: list[bytes] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def save_artifacts(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike, merges_path: str | os.PathLike,
    *, vocab_indent: int | None = 2,
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


def load_artifacts(vocab_path: str | os.PathLike, merges_path: str | os.PathLike,
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


