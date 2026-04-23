#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
exec uv run python eecs148b_hw1/train.py \
  --train_data bpe/tokenized/train_tokens.npy \
  --val_data bpe/tokenized/val_tokens.npy \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_heads 8 \
  --num_layers 4 \
  --batch_size 128 \
  --max_steps 5000 \
  --lr 3e-4 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --log_every 10 \
  --eval_every 100 \
  --ckpt_dir checkpoints \
  --save_every 100 \
  --wandb --wandb_project lm --wandb_run_name "run2"
  --overfit_batch true
  "$@"
