#!/usr/bin/env bash
# Ablation 2: remove positional embeddings (NoPE).
# Same hyperparameters as the sinusoidal-PE baseline so the learning curves
# are directly comparable.
#
# Usage:
#   scripts/train_nope.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"

RUN_NAME="nope-lr3e-4"
CKPT_DIR="checkpoints/${RUN_NAME}"
mkdir -p "${CKPT_DIR}"

echo "=========================================="
echo "Running NoPE ablation"
echo "  run_name = ${RUN_NAME}"
echo "  ckpt    = ${CKPT_DIR}"
echo "=========================================="

uv run python eecs148b_hw1/train.py \
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
  --ckpt_dir "${CKPT_DIR}" \
  --save_every 1000 \
  --no_pos_emb \
  --wandb --wandb_project lm --wandb_run_name "${RUN_NAME}"
