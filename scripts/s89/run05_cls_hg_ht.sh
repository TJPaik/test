#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:=4}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for SEED in 0 1 2 3; do
  PYENV_VERSION=torch \
  SPLIT_SEED="${SEED}" \
  RUN_TAG="cls_hg_ht_NEW_SPLIT_seed${SEED}" \
  MAX_EPOCHS=100 \
  CLS_BATCH_SIZE=32 \
  CLS_DATASETS=hypergraph \
  HYP_MODELS=HyperTransformer \
  python main.py || exit 1
done
