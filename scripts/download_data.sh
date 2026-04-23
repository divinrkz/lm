#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env PYTHONPATH="$ROOT" uv run python eecs148b_hw1/data/tinystories.py "$@"
