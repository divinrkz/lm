from __future__ import annotations

from typing import Iterable, Iterator

import regex as re

from eecs148b_hw1.tokenizer.util import REGEX_P, _apply_merge, load_artifacts, segment

__all__ = ["BPETokenizer"]


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = merges
        self._merge_rank: dict[tuple[bytes, bytes], int] = {p: i for i, p in enumerate(merges)}
        self.special_tokens = list(special_tokens) if special_tokens else []
        self._cache: dict[str, list[int]] = {}

        existing = set(self.vocab.values())
        next_id = max(self.vocab.keys(), default=-1) + 1
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in existing:
                self.vocab[next_id] = st_bytes
                existing.add(st_bytes)
                next_id += 1
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        vocab, merges = load_artifacts(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_word(self, pieces: list[bytes]) -> list[bytes]:
        merge_rank = self._merge_rank
        while len(pieces) >= 2:
            pairs = [tuple(pieces[i : i + 2]) for i in range(len(pieces) - 1)]
            candidates = [p for p in pairs if p in merge_rank]
            if not candidates:
                break
            best = min(candidates, key=lambda p: (merge_rank[p], p))
            pieces = _apply_merge(pieces, best[0], best[1])
        return pieces

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        for is_special, chunk in segment(text, self.special_tokens):
            if not chunk:
                continue
            if is_special:
                ids.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue
            for m in re.finditer(REGEX_P, chunk):
                pretoken = m.group(0)
                # Check cache first
                if pretoken in self._cache:
                    ids.extend(self._cache[pretoken])
                    continue
                pieces = [bytes([b]) for b in pretoken.encode("utf-8")]
                merged = self._bpe_encode_word(pieces)
                token_ids = [self.bytes_to_id[p] for p in merged]
                self._cache[pretoken] = token_ids
                ids.extend(token_ids)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab.get(i, b"") for i in ids)
        return raw.decode("utf-8", errors="replace")
