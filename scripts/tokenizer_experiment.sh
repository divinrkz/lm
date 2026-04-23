
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env PYTHONPATH="$ROOT" uv run python eecs148b_hw1/experiments/tokenizer.py --vocab out/tinystories_vocab.json --merges out/tinystories_merges.txt --train-csv datasets/tinystories/train.csv --val-csv datasets/tinystories/validation.csv --text-column text --n-sample 10 --seed 42 --id-preview 24 --encode-splits --out-dir out/tokenized
 "$@"
