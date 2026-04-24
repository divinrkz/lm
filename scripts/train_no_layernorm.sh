#!/usr/bin/env bash
# Ablation 1: remove LayerNorm.
# Usage:
#   scripts/train_no_ln.sh                # runs both: baseline LR and a lower LR
#   scripts/train_no_ln.sh 3e-4           # single run at given LR
#   LRS="3e-4 1e-4 3e-5" scripts/train_no_ln.sh   # custom sweep


ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"

if [ $# -ge 1 ]; then
  LRS="$*"
else
  LRS="${LRS:-3e-4 1e-4}"
fi

for LR in $LRS; do
  # Sanitize LR for filesystem/run names: 3e-4 -> 3e-4 (already safe). Keep as-is.
  RUN_NAME="no-layernorm-lr${LR}"
  CKPT_DIR="checkpoints/${RUN_NAME}"
  mkdir -p "${CKPT_DIR}"

  echo "=========================================="
  echo "Running LayerNorm ablation @ lr=${LR}"
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
    --lr "${LR}" \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --log_every 10 \
    --eval_every 100 \
    --ckpt_dir "${CKPT_DIR}" \
    --save_every 1000 \
    --no_ln \
    --wandb --wandb_project lm --wandb_run_name "${RUN_NAME}"
done
