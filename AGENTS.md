# Repository Guidelines

HyperAnalog pairs PyTorch Geometric graph learners with netlist-derived datasets; keep contributions focused on data fidelity, reproducible training, and lightweight artifacts for both GPU and CPU workflows.

## Project Structure & Module Organization
- `circuitgraph/`: parsers, graph builders, and dataset generation scripts; extend these when adding new netlist features or labels.
- `models/hypergraph` and `models/bipartite`: Lightning-ready architectures; colocate shared layers in helpers rather than duplicating code.
- `main.py`: orchestrates data modules, Lightning trainers, and logging; treat it as the canonical experiment script.
- `history2/` holds dataset backups; keep large `.pt` files out of Git and reference external storage paths instead.

## Build, Test, and Development Commands
- `python3 -m circuitgraph.datasets`: regenerates `classification_*` and `regression_*` `.pt` files from `AnalogGenie/Dataset` and `AICircuit`; rerun whenever schema changes.
- `python3 main.py`: trains and validates both hypergraph and bipartite models, writing TensorBoard runs under `logs/`.
- `CUDA_VISIBLE_DEVICES=0 python3 main.py` (or omit for CPU) ensures Lightning targets the intended accelerator.
- Install dependencies per environment: `pip install torch torch_geometric pytorch-lightning torchmetrics torch-scatter scikit-learn`.

## Coding Style & Naming Conventions
- Follow PEP8 with 4-space indents, `snake_case` functions, and `CamelCase` modules that subclass Lightning or torch layers.
- Type hints are expected on public functions; keep docstrings concise and describe tensor shapes.
- Place configuration constants (paths, hyperparameters) near the bottom `CONFIG` dict in `main.py` and mirror naming when adding new entries.

## Testing Guidelines
- There is no standalone unit-test suite yet; rely on Lightning validation/test loops triggered by `python3 main.py`.
- Before long runs, smoke-test structural changes by temporarily enabling `pl.Trainer(fast_dev_run=1)` or reducing `MAX_EPOCHS`.
- Validate dataset builders by opening the produced `.pt` file with `torch.load` and asserting feature counts match the consuming model.

## Commit & Pull Request Guidelines
- Write imperative commit messages prefixed with the touched area (e.g., `graph: deduplicate pg edges`); keep one logical change per commit.
- PRs should describe the motivation, data sources touched, training metrics (val accuracy/R²), and attach relevant `logs/` graphs or confusion matrices.
- Link issues or TODO items in the PR body, list new dependencies, and call out any data files that must be regenerated.

## Security & Configuration Tips
- Never commit proprietary netlists or `.pt` datasets; reference their absolute or mounted paths via environment variables instead.
- When handling GPU runs on shared machines, pin deterministic seeds inside `GraphDataModule.setup` and document them in PR notes for reproducibility.
