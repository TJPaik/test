#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:=7}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SLOW_HG="SheafHyperGNN,HyperGT,PhenomNN"

for SEED in 0 1 2 3; do
  PYENV_VERSION=torch \
  SPLIT_SEED="${SEED}" \
  RUN_TAG="reg_hg_cva_slow_NEW_SPLIT_seed${SEED}" \
  MAX_EPOCHS=100 \
  REG_BATCH_SIZE=48 \
  REG_DATASETS=hypergraph_CVA \
  HYP_MODELS="${SLOW_HG}" \
  python main_regression.py || exit 1
done
