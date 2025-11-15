#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:=7}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

GRAPH_MODELS="GCN,GIN,GAT,BipartiteTransformer,BipartiteGNN"

for SEED in 0 1 2 3; do
  PYENV_VERSION=torch \
  SPLIT_SEED="${SEED}" \
  RUN_TAG="reg_clique_csva_NEW_SPLIT_seed${SEED}" \
  MAX_EPOCHS=100 \
  REG_BATCH_SIZE=64 \
  REG_DATASETS=clique_CSVA \
  GRAPH_MODELS="${GRAPH_MODELS}" \
  python main_regression.py || exit 1
done
