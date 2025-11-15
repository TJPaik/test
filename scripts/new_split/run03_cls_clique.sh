#!/usr/bin/env bash
set -euo pipefail

: "${GPU_ID:=2}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

GRAPH_MODELS="GCN,GIN,GAT,BipartiteTransformer,BipartiteGNN"

for SEED in 0 1 2 3; do
  PYENV_VERSION=torch \
  SPLIT_SEED="${SEED}" \
  RUN_TAG="cls_clique_NEW_SPLIT_seed${SEED}" \
  MAX_EPOCHS=100 \
  CLS_BATCH_SIZE=32 \
  CLS_DATASETS=clique \
  GRAPH_MODELS="${GRAPH_MODELS}" \
  python main.py || exit 1
done
