## 실험 실행 가이드

회귀(`main_regression.py`)와 분류(`main.py`) 모두 PyTorch Lightning 기반이므로, 실행 전 `PYENV_VERSION=torch` 환경에서 필요한 패키지가 설치되어 있어야 합니다. 아래 표는 자주 쓰는 환경변수와 역할입니다.

### 데이터 생성 명령 요약

```bash
# 분류용 하이퍼그래프/바이파르타이트
PYENV_VERSION=torch PYTHONPATH="$PWD:$PYTHONPATH" \
python circuitgraph/classification_builder.py \
  --dataset_dir ./AnalogGenie/Dataset \
  --hypergraph_out classification_hypergraph_dataset.pt \
  --bipartite_out classification_bipartite_dataset.pt

# 분류용 clique
PYENV_VERSION=torch PYTHONPATH="$PWD:$PYTHONPATH" \
python circuitgraph/clique_dataset_builder.py \
  --input classification_hypergraph_dataset.pt \
  --output classification_clique_dataset.pt

# 회귀용 하이퍼그래프/바이파르타이트 (AICircuit_subset 예시)
PYENV_VERSION=torch PYTHONPATH="$PWD:$PYTHONPATH" \
python circuitgraph/regression_builder.py \
  --dataset_root ./AICircuit_subset/Dataset \
  --hypergraph_out regression_hypergraph_dataset.pt \
  --bipartite_out regression_bipartite_dataset.pt

# 회귀용 clique 확장 (집합 및 디자인별)
designs=(CSVA CVA LNA Mixer TSVA)
for d in "${designs[@]}"; do
  PYENV_VERSION=torch PYTHONPATH="$PWD:$PYTHONPATH" \
    python circuitgraph/clique_dataset_builder.py \
      --input regression_hypergraph_dataset_${d}.pt \
      --output regression_clique_dataset_${d}.pt
done

# 전체 세트 일괄 재생성
PYENV_VERSION=torch PYTHONPATH="$PWD:$PYTHONPATH" python -m circuitgraph.datasets
```

| 환경변수 | 용도 |
| --- | --- |
| `MAX_EPOCHS` | 학습 epoch 수 (기본: 분류 5, 회귀 3) |
| `HYP_MODELS` | 하이퍼그래프용 모델 목록을 콤마로 지정. 예: `HyperTransformer,HyperGT` |
| `GRAPH_MODELS` | 그래프/바이파르타이트 단계에서 사용할 GNN/Transformer 목록. 예: `GCN,GIN,GAT,BipartiteTransformer,BipartiteGNN` |
| `FAST_DEV_RUN` | `1`이면 Lightning fast_dev_run 모드로 빠른 검증 |
| `BATCH_SIZE` | 전역 배치 크기 기본값 (없으면 16). 회귀/분류 각각 `REG_BATCH_SIZE`, `CLS_BATCH_SIZE`로 별도 지정 가능 |
| `CLS_DATASETS` | 분류 스크립트에서 실행할 데이터셋 subset (`hypergraph,bipartite,clique`). 미지정 시 순차 실행 |
| `REG_DATASETS` | 회귀 스크립트가 순회할 데이터셋 subset (`hypergraph,bipartite,clique,*_CSVA` 등). 미지정 시 자동 |
| `REG_USE_LOG1P` | 타깃을 `sign(y)*log1p(|y|)` 스케일로 사용할지(기본 1). `0`이면 원 값 사용 |
| `REG_MAX_SAMPLES` | 회귀 데이터에서 무작위로 사용할 최대 샘플 수. 미지정 시 전체 |
| `REG_DESIGNS` | `CSVA,CVA,...` 형태로 지정하면 각 디자인별 하이퍼/바이파르타이트/클릭/스타 세트를 순차 실행 |
| `CLS_MAX_SAMPLES` | 분류 데이터에서 무작위로 사용할 최대 샘플 수. 미지정 시 전체 |
| `RUN_TAG` | `RUN_TAG=gin_gpu4` 처럼 지정하면 로그 버전에 태그가 붙어(`logs/<dataset>/GCN__timestamp__gin_gpu4`) 실험 식별이 쉬움 |

### 1. 분류 실험 (`main.py`)

데이터셋별 `.pt` 파일과 기본 모델:

| 데이터셋 키 | 파일 | 기본 모델 |
| --- | --- | --- |
| `hypergraph` | `classification_hypergraph_dataset.pt` | HyperTransformer, HyperGT, DPHGNN, HJRL, AllSetformer, SheafHyperGNN, TFHNN, EHNN, HyperND, PhenomNN (필요 시 `HYP_MODELS`) |
| `bipartite` | `classification_bipartite_dataset.pt` | LaplacianPositionalTransformer + GCN, GIN, GAT, BipartiteTransformer, BipartiteGNN (`GRAPH_MODELS`) |
| `clique` | `classification_clique_dataset.pt` | LaplacianPositionalTransformer + GCN, GIN, GAT, BipartiteTransformer, BipartiteGNN (`GRAPH_MODELS`) |

실행 예시:

```bash
PYENV_VERSION=torch \
MAX_EPOCHS=5 \
CLS_BATCH_SIZE=8 \
HYP_MODELS="HyperTransformer,HyperGT,SheafHyperGNN" \
    GRAPH_MODELS="GCN,GIN,GAT" \
python main.py
```

위 명령은 세 데이터셋을 순차로 학습하며, 하이퍼그래프 단계는 지정한 모델만, 바이파르타이트/클릭 단계는 `GRAPH_MODELS`에 포함된 그래프 GNN과 기존 LaplacianPositionalTransformer를 함께 평가합니다. `CLS_BATCH_SIZE`가 없으면 기본 16이 사용됩니다.

핵심 포인트:
- `HYP_MODELS` 미지정 시 레지스트리에 등록된 모든 하이퍼그래프 모델을 시도합니다.
- `FAST_DEV_RUN=1`을 넣으면 각 모델이 train/val/test 루프를 한 번씩만 돌며 코드 변경 검증에 유용합니다.

### 2. 회귀 실험 (`main_regression.py`)

데이터셋별 `.pt` 파일과 기본 모델:

| 데이터셋 키 | 파일 | 기본 모델 |
| --- | --- | --- |
| `hypergraph[_DESIGN]` | `regression_hypergraph_dataset(_DESIGN).pt` | HyperTransformer, HyperGT, DPHGNN, HJRL, SheafHyperGNN, HyperND, EHNN/ED-HNN, AllSetformer, AllDeepSets, TFHNN, PhenomNN (`HYP_MODELS`) |
| `bipartite[_DESIGN]` | `regression_bipartite_dataset(_DESIGN).pt` | LaplacianPositionalTransformer + GCN, GIN, GAT, BipartiteTransformer, BipartiteGNN (`GRAPH_MODELS`) |
| `clique[_DESIGN]` | `regression_clique_dataset(_DESIGN).pt` | LaplacianPositionalTransformer + GCN, GIN, GAT, BipartiteTransformer, BipartiteGNN (`GRAPH_MODELS`) |

실행 예시:

```bash
PYENV_VERSION=torch \
MAX_EPOCHS=3 \
HYP_MODELS="HyperTransformer,TFHNN" \
GRAPH_MODELS="GCN,GAT" \
REG_DESIGNS=CSVA,CVA,LNA,Mixer,TSVA \
REG_DATASETS= \
REG_BATCH_SIZE=8 \
REG_USE_LOG1P=1 \
REG_MAX_SAMPLES=512 \
python main_regression.py
```

설명:
- `REG_DATASETS`로 특정 파이프라인만 선택(예: `clique_CSVA`). 비워두면 `REG_DESIGNS` 순서에 따라 `hypergraph→bipartite→clique` 순으로 반복합니다.
- `REG_USE_LOG1P`는 타깃 스케일링을 제어하며, 원래 값으로 회귀하고 싶으면 `0`.
- `REG_MAX_SAMPLES`는 대규모 데이터셋(예: TSVA)을 부분 샘플링할 때 사용. 지우면 전체 샘플 사용.
- `REG_DESIGNS`를 지정하면 CSVA/CVA/LNA/Mixer/TSVA 각각의 `.pt`를 자동 순회하며, `GRAPH_MODELS`가 없으면 GCN·GIN·GAT 세 모델 모두 실행됩니다.
- 배치 크기는 `REG_BATCH_SIZE` 또는 전역 `BATCH_SIZE`로 조정.

### 3. 레거시 회귀 스크립트 (`main_regression_prev.py`)

해당 파일은 예전 논문 실험 설정을 그대로 유지하며, 별도 환경변수는 없습니다. 회로별 `.pt`를 순회하므로, 필요 시 다음처럼 실행합니다.

```bash
PYENV_VERSION=torch MAX_EPOCHS=10 python main_regression_prev.py
```

모델·데이터 구성이 스크립트 내부 `CONFIG`에 하드코딩돼 있으므로, 새로운 조합이 필요하면 해당 dict를 수정하십시오.

### 4. 실행 요약

1. 원하는 모델 세트: `HYP_MODELS`로 지정 (양쪽 스크립트 공통).
2. 배치 크기: `CLS_BATCH_SIZE` / `REG_BATCH_SIZE` / `BATCH_SIZE`.
3. 데이터 서브셋: 회귀는 `REG_DATASETS`; 분류는 현재 전체 루프 고정.
4. 빠른 검증: `FAST_DEV_RUN=1`.
5. 회귀 추가 스위치: `REG_USE_LOG1P`, `REG_MAX_SAMPLES`.

위 설정을 조합하면 “데이터셋별·모델별” 실험을 스크립트 수정 없이 수행할 수 있습니다. 필요 시 `logs/` 디렉터리에 생성되는 TensorBoard 런을 통해 각 조합의 loss/MSE/F1 등을 확인하세요.

### 로그/결과 확인

- 각 실행은 `logs/<dataset_name>[/<_reg>]/<model>__YYYYMMDD-HHMMSS__[RUN_TAG]/` 형태로 저장됩니다. 예: `logs/hypergraph/HyperTransformer__20251113-135247__gin_gpu4`.
- 폴더 안 `hparams.yaml`에 학습 시 사용한 환경변수(`model_name`, `batch_size`, `max_epochs`, `RUN_TAG` 등)이 기록되므로 어떤 실험인지 쉽게 확인할 수 있습니다.
- `tensorboard --logdir logs` 명령으로 TensorBoard를 띄우면 모든 실행의 `train_loss`, `val_loss`, `val_acc`, `test_acc`, `test_macro_f1`(분류) 또는 `test_loss`, `test_r2`(회귀)를 비교할 수 있습니다.
