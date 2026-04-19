from __future__ import annotations

import sys
from pathlib import Path

# Running as `python eecs148b_hw1/bpe/scripts.py` does not set the package; add repo root.
if __name__ == "__main__":
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from eecs148b_hw1.bpe.util import save_bpe_artifacts, train_bpe


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
    save_bpe_artifacts(
        vocab,
        merges,
        out_dir / "tinystories_vocab.json",
        out_dir / "tinystories_merges.txt",
    )


if __name__ == "__main__":
    main()
