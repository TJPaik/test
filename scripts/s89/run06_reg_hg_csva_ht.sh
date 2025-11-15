#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:=5}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for SEED in 0 1 2 3; do
  PYENV_VERSION=torch \
  SPLIT_SEED="${SEED}" \
  RUN_TAG="reg_hg_csva_ht_NEW_SPLIT_seed${SEED}" \
  MAX_EPOCHS=100 \
  REG_BATCH_SIZE=64 \
  REG_DATASETS=hypergraph_CSVA \
  HYP_MODELS=HyperTransformer \
  python main_regression.py || exit 1
done
