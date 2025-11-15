#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:=5}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SLOW_HG="SheafHyperGNN,HyperGT,PhenomNN"

for SEED in 0 1 2 3; do
  PYENV_VERSION=torch \
  SPLIT_SEED="${SEED}" \
  RUN_TAG="cls_hg_slow_NEW_SPLIT_seed${SEED}" \
  MAX_EPOCHS=100 \
  CLS_BATCH_SIZE=24 \
  CLS_DATASETS=hypergraph \
  HYP_MODELS="${SLOW_HG}" \
  python main.py || exit 1
done
