#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Running a path puts only that file's directory on sys.path; set PYTHONPATH so `eecs148b_hw1` imports work.
exec env PYTHONPATH="$ROOT" uv run python eecs148b_hw1/data/tinystories.py "$@"
