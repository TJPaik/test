# Project Overview

This project uses graph neural networks (GNNs) and transformers to perform classification and regression tasks on electronic circuits. The circuits are represented as hypergraphs and bipartite graphs. The project includes scripts for parsing circuit netlists, building the graph datasets, and training various models.

The core technologies used are:
- **Python**
- **PyTorch**
- **PyTorch Lightning**
- **PyTorch Geometric**
- **Scikit-learn**

The project is structured as follows:
- `main.py`: The main script for training and evaluating the models.
- `circuitgraph/`: A module for parsing circuit netlists and building graph datasets.
- `models/`: Contains the GNN and transformer model implementations for both hypergraph and bipartite graph structures.
- `history2/`: Contains previous versions of the code.

# Building and Running

## 1. Generate Datasets

To generate the classification and regression datasets, run the following command:

```bash
python3 -m circuitgraph.datasets
```

This will create the following files:
- `classification_hypergraph_dataset.pt`
- `classification_bipartite_dataset.pt`
- `regression_hypergraph_dataset.pt`
- `regression_bipartite_dataset.pt`

## 2. Train and Evaluate Models

To train and evaluate the models, run the `main.py` script:

```bash
python3 main.py
```

This will train the models defined in the `CONFIG` dictionary in `main.py` on the corresponding datasets.

# Development Conventions

- **Models**: Models are organized by graph type (bipartite or hypergraph) in the `models/` directory.
- **Data**: The `circuitgraph/` module is responsible for all data processing and graph construction.
- **Training**: `pytorch-lightning` is used for training, with the main training logic in `main.py`.
- **Dependencies**: The core dependencies (`torch`, `pytorch-lightning`, `torch-geometric`, `scikit-learn`) are confirmed to be installed.
