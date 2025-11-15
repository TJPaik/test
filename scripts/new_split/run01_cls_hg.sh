#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:=0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

FAST_HG="HyperTransformer,DPHGNN,HJRL,HyperND,AllSetformer,AllDeepSets,TFHNN"
SLOW_HG="SheafHyperGNN,HyperGT,PhenomNN"
HYP_MODELS="${FAST_HG},${SLOW_HG}"

for SEED in 0 1 2 3; do
  PYENV_VERSION=torch \
  SPLIT_SEED="${SEED}" \
  RUN_TAG="cls_hg_NEW_SPLIT_seed${SEED}" \
  MAX_EPOCHS=100 \
  CLS_BATCH_SIZE=32 \
  CLS_DATASETS=hypergraph \
  HYP_MODELS="${HYP_MODELS}" \
  python main.py || exit 1
done
