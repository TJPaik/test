import os
from datetime import datetime
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch_geometric.data import Dataset
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from torchmetrics import Accuracy, ConfusionMatrix, R2Score, F1Score

# Import models from their new locations
from models.hypergraph.transformer import HyperTransformer
from models.hypergraph.dhg_wrappers import DHGModelWrapper, build_dhg_model
from models.bipartite.bipartite_network import BipartiteGNN
from models.bipartite.transformer import BipartiteTransformer, LaplacianPositionalTransformer
from models.graph.basic import GraphBackbone, GraphGCN, GraphGIN, GraphGAT

from pytorch_lightning.loggers import TensorBoardLogger
from torch_scatter import scatter_mean


class GenericDataset(Dataset):
    def __init__(self, pt_file):
        super().__init__()
        self.data = torch.load(pt_file, weights_only=False)
        for a in self.data:
            pe1 = torch.sin(torch.pi * a.x[:, -3] / (a.x[:, -1] + 1e-5)).view(-1, 1)
            pe2 = torch.sin(torch.pi * a.x[:, -2] / (a.x[:, -1] + 1e-5)).view(-1, 1)
            a.x = torch.cat([a.x, pe1, pe2], dim=1)
            # a.y = torch.log1p(a.y)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 32, max_samples: int | None = None):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        env_cap = os.environ.get("CLS_MAX_SAMPLES")
        self.max_samples = max_samples if max_samples is not None else (int(env_cap) if env_cap else None)
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        full_dataset = GenericDataset(self.dataset_path)
        indices = list(range(len(full_dataset)))
        if self.max_samples and len(indices) > self.max_samples:
            generator = torch.Generator().manual_seed(42)
            keep_idx = torch.randperm(len(indices), generator=generator)[: self.max_samples].tolist()
            indices = [indices[i] for i in keep_idx]
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        # This is a simple collate function. For graph data, you might need a more complex one from torch_geometric.
        # However, the original code processes one graph at a time, so we will do the same.
        return batch

def compute_hypergraph_stats(subset: Subset):
    dataset = getattr(subset, 'dataset', subset)
    indices = getattr(subset, 'indices', range(len(subset)))
    max_hyperedge_order = 0
    for idx in indices:
        data = dataset[idx]
        hyperedge_ids = data.hyperedge_index[1]
        if hyperedge_ids.numel() == 0:
            continue
        counts = torch.bincount(
            hyperedge_ids,
            minlength=int(hyperedge_ids.max().item()) + 1
        )
        if counts.numel() > 0:
            max_hyperedge_order = max(max_hyperedge_order, int(counts.max().item()))
    return {"max_hyperedge_order": max_hyperedge_order}


def create_logger(dataset_name: str, model_name: str) -> TensorBoardLogger:
    run_tag = os.environ.get("RUN_TAG", "").strip()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [model_name, timestamp]
    if run_tag:
        parts.append(run_tag)
    version = "__".join(parts)
    return TensorBoardLogger("logs", name=dataset_name, version=version)

class LitGraphModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, num_classes=None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.metadata = (metadata or {}).copy()
        self.save_hyperparameters(ignore=["model"])

        # Determine the number of classes
        if num_classes is not None:
            self.num_classes = num_classes
        elif hasattr(model, 'lin') and hasattr(model.lin, 'out_features'):
            self.num_classes = model.lin.out_features
        else:
            # Default to a reasonable number if we can't determine it
            self.num_classes = 10
            print(f"Warning: Could not determine number of classes. Using default: {self.num_classes}")

        # Calculate class weights for balanced loss
        self.register_buffer('class_weights', torch.ones(self.num_classes))
        self.classification = True
        if self.classification:
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = torch.nn.MSELoss()

        hp = {
            "learning_rate": learning_rate,
            "num_classes": self.num_classes,
        }
        if metadata:
            hp.update(metadata)
        self.save_hyperparameters(hp)

        # Metrics
        self.train_r2 = R2Score()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        self.val_r2 = R2Score()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        self.test_r2 = R2Score()
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.test_macro_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_per_class_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average=None)

        # For tracking class distribution
        self.class_counts = torch.zeros(self.num_classes)
        self.class_weights_updated = False

    def forward(self, batch):
        # The models expect a single data object, not a batch. We iterate.
        outputs = []
        for data in batch:
            data = data.to(self.device)
            if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer, GraphBackbone)):
                 edge_attr = getattr(data, "edge_attr", None)
                 out = self.model(data.x, data.edge_index, edge_attr)
            else: # Hyper models
                 out = self.model(data.x, data.hyperedge_index, data)

            # Ensure out has a batch dimension if needed
            if out.dim() == 1:
                out = out.unsqueeze(0)  # Add batch dimension

            outputs.append(out)
        return outputs

    def training_step(self, batch, batch_idx):
        # Log learning rate to progress bar
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)

        loss = 0
        outs = []
        targets = []
        for data in batch:
            data = data.to(self.device)
            if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer, GraphBackbone)):
                 edge_attr = getattr(data, "edge_attr", None)
                 out = self.model(data.x, data.edge_index, edge_attr)
            else: # HyperGNN, HyperTransformer
                 out = self.model(data.x, data.hyperedge_index, data)

            # Ensure out has a batch dimension
            if out.dim() == 1:
                out = out.unsqueeze(0)  # Add batch dimension
            else:
                # If out already has a batch dimension, make sure it's the first dimension
                out = out.reshape(1, -1)


            if self.classification:
                target_idx = torch.argmax(data.y).unsqueeze(0)
                loss += self.criterion(out, target_idx)
                # Get predictions and targets
                pred = torch.argmax(out, dim=-1)
                target = torch.argmax(data.y, dim=-1)
                # Ensure pred and target have the same shape
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                if target.dim() == 0:
                    target = target.unsqueeze(0)

                # Update accuracy metric
                self.train_accuracy.update(pred, target)
                self.train_conf_matrix.update(pred, target)
            else:
                target = data.y.unsqueeze(0)
                loss += self.criterion(out, target)
                outs.append(out)
                targets.append(target)
        if not self.classification:
            outs = torch.cat(outs, dim=0)
            targets = torch.cat(targets, dim=0)
            self.train_r2.update(outs, targets)

        avg_loss = loss / len(batch)
        self.log('train_loss', avg_loss, prog_bar=True)
        return avg_loss

    def on_train_epoch_end(self):
        if self.classification:
            # Log training accuracy
            self.log('train_acc', self.train_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True)

            # Compute and print confusion matrix for training data
            cm = self.train_conf_matrix.compute()
            # print(f"\nTraining Confusion Matrix (Epoch {self.current_epoch}):\n{cm}")

        # Update class weights every 5 epochs to adapt to changing class distributions
        if self.current_epoch % 5 == 0:
            if self.classification:
                # Get class distribution from confusion matrix
                class_counts = torch.sum(cm, dim=1)
                total_samples = torch.sum(class_counts)

                # Avoid division by zero
                class_counts = torch.clamp(class_counts, min=1.0)

                # Calculate inverse frequency weights with smoothing
                weights = torch.sqrt(total_samples / (class_counts + 10))  # Square root for less extreme weights

                # Normalize weights
                weights = weights / torch.sum(weights) * self.num_classes

                # Update class weights
                self.class_weights.copy_(weights)
                self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                self.class_weights_updated = True

                print(f"\nUpdated class weights (Epoch {self.current_epoch}): {self.class_weights}")
            else:
                self.log('train_r2', self.train_r2.compute(), prog_bar=True)

        if self.classification:
            # Reset metrics for the next epoch
            self.train_accuracy.reset()
            self.train_conf_matrix.reset()

    def validation_step(self, batch, batch_idx):
        loss = 0
        outs = []
        targets = []
        for data in batch:
            data = data.to(self.device)
            if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer, GraphBackbone)):
                 edge_attr = getattr(data, "edge_attr", None)
                 out = self.model(data.x, data.edge_index, edge_attr)
            else: # HyperGNN, HyperTransformer
                 out = self.model(data.x, data.hyperedge_index, data)

            # Ensure out has a batch dimension
            if out.dim() == 1:
                out = out.unsqueeze(0)  # Add batch dimension

            # Calculate loss
            if self.classification:
                target_idx = torch.argmax(data.y).unsqueeze(0)
                loss += self.criterion(out, target_idx)
                # Get predictions and targets
                pred = torch.argmax(out, dim=-1)
                target = torch.argmax(data.y, dim=-1)
                # Ensure pred and target have the same shape
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                if target.dim() == 0:
                    target = target.unsqueeze(0)

                # Update accuracy metric
                self.val_accuracy.update(pred, target)
                self.val_conf_matrix.update(pred, target)
            else:
                target = data.y.unsqueeze(0)
                loss += self.criterion(out, target)

                outs.append(out)
                targets.append(target)
        if not self.classification:
            outs = torch.cat(outs, dim=0)
            targets = torch.cat(targets, dim=0)
            self.val_r2.update(outs, targets)

        avg_loss = loss / len(batch)
        self.log('val_loss', avg_loss, prog_bar=True)

    def on_validation_epoch_end(self):
        if self.classification:
            self.log('val_acc', self.val_accuracy.compute(), prog_bar=True)
            # Compute and print confusion matrix
            cm = self.val_conf_matrix.compute()
            # Do not print during sanity check
            if not self.trainer.sanity_checking:
                pass
                # print(f"\nValidation Confusion Matrix (Epoch {self.current_epoch}):\n{cm}")

            self.val_accuracy.reset()
            self.val_conf_matrix.reset()
        else:
            self.log('val_r2', self.val_r2.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = 0
        # Using similar logic as validation_step
        outs = []
        targets = []
        for data in batch:
            data = data.to(self.device)
            if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer, GraphBackbone)):
                 edge_attr = getattr(data, "edge_attr", None)
                 out = self.model(data.x, data.edge_index, edge_attr)
            else: # HyperGNN, HyperTransformer
                 out = self.model(data.x, data.hyperedge_index, data)

            # Ensure out has a batch dimension
            if out.dim() == 1:
                out = out.unsqueeze(0)  # Add batch dimension

            # Calculate loss
            if self.classification:
                target_idx = torch.argmax(data.y).unsqueeze(0)
                loss += self.criterion(out, target_idx)
                # Get predictions and targets
                pred = torch.argmax(out, dim=-1)
                target = torch.argmax(data.y, dim=-1)
                # Ensure pred and target have the same shape
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                if target.dim() == 0:
                    target = target.unsqueeze(0)

                # Update accuracy metric
                self.test_accuracy.update(pred, target)
                self.test_conf_matrix.update(pred, target)
                self.test_macro_f1.update(pred, target)
                self.test_per_class_f1.update(pred, target)
            else:
                target = data.y.unsqueeze(0)
                loss += self.criterion(out, target)
                outs.append(out)
                targets.append(target)
        if not self.classification:
            outs = torch.cat(outs, dim=0)
            targets = torch.cat(targets, dim=0)
            self.test_r2.update(outs, targets)

        avg_loss = loss / len(batch)
        self.log('test_loss', avg_loss)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_accuracy.compute())
        if self.classification:
            cm = self.test_conf_matrix.compute()
            print(f"\nTest Confusion Matrix:\n{cm}")
            macro_f1 = self.test_macro_f1.compute()
            per_class_f1 = self.test_per_class_f1.compute()
            self.log('test_macro_f1', macro_f1)
            print(f"Test Macro F1: {macro_f1:.4f}")
            print(f"Per-class F1: {per_class_f1}")

            self.test_accuracy.reset()
            self.test_conf_matrix.reset()
            self.test_macro_f1.reset()
            self.test_per_class_f1.reset()
        else:
            self.log('test_r2', self.test_r2.compute(), prog_bar=True)

    def configure_optimizers(self):
        # Use AdamW optimizer with moderate weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.001,  # Moderate weight decay
            betas=(0.9, 0.999)
        )

        # Use CosineAnnealingWarmRestarts for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=1,  # Keep the same cycle length
            eta_min=1e-6  # Minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

if __name__ == '__main__':
    # --- Configuration ---
    MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 5))

    hypergraph_model_registry = {
        "HyperTransformer": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: HyperTransformer(
                in_channels=in_c,
                hidden_channels=512,
                out_channels=out_c,
                num_layers=8,
                num_heads=16,
                dropout=0.3,
            ),
        },
        "HyperGT": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("HyperGT", in_c, out_c, stats=stats),
        },
        "DPHGNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("DPHGNN", in_c, out_c, stats=stats),
        },
        "HJRL": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("HJRL", in_c, out_c, stats=stats),
        },
        "AllSetformer": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("AllSetformer", in_c, out_c, stats=stats),
        },
        "SheafHyperGNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("SheafHyperGNN", in_c, out_c, stats=stats),
        },
        "TFHNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("TFHNN", in_c, out_c, stats=stats),
        },
        "EHNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model(
                "EHNN",
                in_c,
                out_c,
                overrides={"max_edge_order_hint": (stats or {}).get("max_hyperedge_order", 32)},
                stats=stats,
            ),
        },
        "HyperND": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("HyperND", in_c, out_c, stats=stats),
        },
        "PhenomNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("PhenomNN", in_c, out_c, stats=stats),
        },
    }

    graph_model_registry = {
        "GCN": {
            "constructor": lambda in_c, out_c, *_args, **__kwargs: GraphGCN(
                in_channels=in_c,
                hidden_channels=256,
                out_channels=out_c,
                num_layers=3,
                dropout=0.3,
            )
        },
        "GIN": {
            "constructor": lambda in_c, out_c, *_args, **__kwargs: GraphGIN(
                in_channels=in_c,
                hidden_channels=256,
                out_channels=out_c,
                num_layers=3,
                dropout=0.3,
            )
        },
        "GAT": {
            "constructor": lambda in_c, out_c, *_args, **__kwargs: GraphGAT(
                in_channels=in_c,
                hidden_channels=256,
                out_channels=out_c,
                num_layers=3,
                dropout=0.2,
                heads=4,
            )
        },
        "BipartiteTransformer": {
            "constructor": lambda in_c, out_c, edge_c, *_args, **__kwargs: BipartiteTransformer(
                in_channels=in_c,
                edge_attr_channels=max(edge_c, 1),
                hidden_channels=256,
                out_channels=out_c,
                num_layers=2,
                num_heads=4,
                dropout=0.2,
            )
        },
        "BipartiteGNN": {
            "constructor": lambda in_c, out_c, edge_c, *_args, **__kwargs: BipartiteGNN(
                in_channels=in_c,
                edge_attr_channels=max(edge_c, 1),
                hidden_channels=256,
                out_channels=out_c,
                num_layers=3,
            )
        },
    }

    enabled_hg_names = os.environ.get("HYP_MODELS")
    if enabled_hg_names:
        requested = [name.strip() for name in enabled_hg_names.split(",") if name.strip()]
        hypergraph_models = {name: hypergraph_model_registry[name] for name in requested if name in hypergraph_model_registry}
        if not hypergraph_models:
            hypergraph_models = hypergraph_model_registry
    else:
        hypergraph_models = hypergraph_model_registry

    enabled_graph_names = os.environ.get("GRAPH_MODELS")
    if enabled_graph_names:
        requested = [name.strip() for name in enabled_graph_names.split(",") if name.strip()]
        missing = [name for name in requested if name not in graph_model_registry]
        if missing:
            raise ValueError(f"GRAPH_MODELS에 알 수 없는 모델이 포함되어 있습니다: {missing}")
        graph_models = {name: graph_model_registry[name] for name in requested}
        if not graph_models:
            graph_models = graph_model_registry
    else:
        graph_models = graph_model_registry

    bipartite_models = {
        "LaplacianPositionalTransformer": {
            "constructor": lambda in_c, out_c, edge_c, *_, **__: LaplacianPositionalTransformer(
                in_channels=in_c,
                edge_attr_channels=edge_c,
                hidden_channels=256,
                out_channels=out_c,
            )
        },
    }
    bipartite_models.update(graph_models)

    clique_models = {
        "LaplacianPositionalTransformer": {
            "constructor": lambda in_c, out_c, edge_c, *_, **__: LaplacianPositionalTransformer(
                in_channels=in_c,
                edge_attr_channels=edge_c,
                hidden_channels=256,
                out_channels=out_c,
            )
        },
    }
    clique_models.update(graph_models)

    # Define datasets and the models to run on them
    CONFIG = {
        "hypergraph": {
            "data_path": "classification_hypergraph_dataset.pt",
            "models": hypergraph_models,
        },
        "bipartite": {
            "data_path": "classification_bipartite_dataset.pt",
            "models": bipartite_models,
        },
        "clique": {
            "data_path": "classification_clique_dataset.pt",
            "models": clique_models,
        },
    }

    requested_cls = os.environ.get("CLS_DATASETS")
    if requested_cls:
        dataset_items = []
        missing = []
        for name in requested_cls.split(","):
            key = name.strip()
            if not key:
                continue
            if key not in CONFIG:
                missing.append(key)
            else:
                dataset_items.append((key, CONFIG[key]))
        if missing:
            raise ValueError(f"CLS_DATASETS에 알 수 없는 항목이 포함되어 있습니다: {missing}")
    else:
        dataset_items = list(CONFIG.items())

    batch_size = int(os.environ.get("CLS_BATCH_SIZE", os.environ.get("BATCH_SIZE", 16)))

    # --- Training Loop ---
    for dataset_name, dataset_config in dataset_items:
        print(f"--- Training on {dataset_name} dataset ---")

        data_module = GraphDataModule(dataset_config["data_path"], batch_size=batch_size) # Moderate batch size for stable training
        data_module.setup()

        sample_data = data_module.train_dataset[0]
        in_channels = sample_data.x.shape[1]

        if dataset_name.startswith("hypergraph"):
            out_channels = sample_data.y.shape[0]
        else:
            out_channels = sample_data.y.shape[0]
            edge_attr_channels = sample_data.edge_attr.shape[1] if hasattr(sample_data, "edge_attr") and sample_data.edge_attr is not None else 0

        print(f"Dataset dimensions - in_channels: {in_channels}, out_channels: {out_channels}")
        if dataset_name.startswith(("bipartite", "clique")):
            print(f"Edge attribute channels: {edge_attr_channels}")

        dataset_stats = compute_hypergraph_stats(data_module.train_dataset) if dataset_name.startswith("hypergraph") else {}

        for model_name, model_config in dataset_config["models"].items():
            print(f"--- Training model: {model_name} ---")

            if dataset_name.startswith(("bipartite", "clique")):
                model = model_config["constructor"](in_channels, out_channels, edge_attr_channels, sample_data, dataset_stats)
            else:
                model = model_config["constructor"](in_channels, out_channels, sample_data, dataset_stats)

            metadata = {
                "task": "classification",
                "dataset": dataset_name,
                "model_name": model_name,
                "batch_size": batch_size,
                "max_epochs": MAX_EPOCHS,
            }
            lit_model = LitGraphModel(model, num_classes=out_channels, learning_rate=1e-4, metadata=metadata)

            tb_logger = create_logger(dataset_name, model_name)
            precision_setting = "16-mixed"
            if isinstance(model, DHGModelWrapper):
                precision_setting = 32
            fast_dev = bool(int(os.environ.get("FAST_DEV_RUN", "0")))
            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                callbacks=[TQDMProgressBar(refresh_rate=5)],
                accelerator="auto",
                gradient_clip_val=0.5,
                enable_progress_bar=True,
                log_every_n_steps=1,
                precision=precision_setting,
                logger=tb_logger,
                fast_dev_run=fast_dev
            )

            trainer.fit(lit_model, datamodule=data_module)

            print(f"--- Testing model: {model_name} ---")
            trainer.test(lit_model, datamodule=data_module)
