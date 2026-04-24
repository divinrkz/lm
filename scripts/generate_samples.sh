#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
exec uv run python eecs148b_hw1/generate.py \
  --prompt "Once upon a time, there was a" \
  --max_tokens 100 \
  --temperature 0.7 \
  --top_p 0.9 \
  --ckpt_path checkpoints/model_5000.pt \
  --wandb --wandb_project lm --wandb_run_name "tiny-run-samples-3"
  "$@"
