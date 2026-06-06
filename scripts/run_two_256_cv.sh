#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/autodl-tmp/CNN_vs_ViT_vs_Swin_Transformer/Dataset_BUSI_with_GT}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"${PYTHON}" scripts/prepare_busi_seg.py --data-dir "${DATA_DIR}"

for experiment in unet_256_bce_dice unet_256_dice; do
  echo "== Training ${experiment} =="
  "${PYTHON}" scripts/train_cv.py \
    --experiment "${experiment}" \
    --data-dir "${DATA_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num-workers "${NUM_WORKERS}" \
    --amp
done
