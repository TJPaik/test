import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch_geometric.data import Dataset
from pytorch_lightning.callbacks import TQDMProgressBar
from torchmetrics import R2Score

from models.hypergraph.transformer import HyperTransformer
from models.hypergraph.dhg_wrappers import DHGModelWrapper, build_dhg_model
from models.bipartite.bipartite_network import BipartiteGNN
from models.bipartite.transformer import BipartiteTransformer, LaplacianPositionalTransformer
from models.graph.basic import GraphBackbone, GraphGCN, GraphGIN, GraphGAT

from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("medium")


class GenericDataset(Dataset):
    def __init__(self, pt_file):
        super().__init__()
        self.data = torch.load(pt_file, weights_only=False)
        self.target_dim = 0
        self.use_log1p = bool(int(os.environ.get("REG_USE_LOG1P", "1")))
        for a in self.data:
            pe1 = torch.sin(torch.pi * a.x[:, -3] / (a.x[:, -1] + 1e-5)).view(-1, 1)
            pe2 = torch.sin(torch.pi * a.x[:, -2] / (a.x[:, -1] + 1e-5)).view(-1, 1)
            a.x = torch.cat([a.x, pe1, pe2], dim=1)
            self.target_dim = max(self.target_dim, a.y.view(-1).numel())

        for a in self.data:
            y = a.y.view(-1).float()
            if self.use_log1p:
                y = torch.sign(y) * torch.log1p(torch.abs(y))
            if y.numel() < self.target_dim:
                pad = torch.zeros(self.target_dim - y.numel(), dtype=y.dtype)
                y = torch.cat([y, pad], dim=0)
            a.y = y

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 32, max_samples: int | None = None):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        env_cap = os.environ.get("REG_MAX_SAMPLES")
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
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
        counts = torch.bincount(hyperedge_ids, minlength=int(hyperedge_ids.max().item()) + 1)
        if counts.numel() > 0:
            max_hyperedge_order = max(max_hyperedge_order, int(counts.max().item()))
    return {"max_hyperedge_order": max_hyperedge_order}


class LitGraphModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, num_targets=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.target_dim = num_targets
        self.criterion = torch.nn.MSELoss()
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()
        self.nan_tensor = None

    def _forward_single(self, data):
        data = data.to(self.device)
        if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer, GraphBackbone)):
            edge_attr = getattr(data, "edge_attr", None)
            out = self.model(data.x, data.edge_index, edge_attr)
        else:
            out = self.model(data.x, data.hyperedge_index, data)
        return out.view(-1)

    def _shared_step(self, batch, stage: str):
        preds = []
        targets = []
        for data in batch:
            pred = self._forward_single(data)
            preds.append(pred.unsqueeze(0))
            targets.append(data.y.view(1, -1).to(self.device))
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        loss = self.criterion(preds, targets)
        if stage == "train":
            self.train_r2.update(preds, targets)
            self.log('train_loss', loss, prog_bar=True)
        elif stage == "val":
            self.val_r2.update(preds, targets)
            self.log('val_loss', loss, prog_bar=True)
        else:
            self.test_r2.update(preds, targets)
            self.log('test_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True)
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
        self._log_r2(self.train_r2, 'train_r2', prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        self._log_r2(self.val_r2, 'val_r2', prog_bar=True)

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        self._log_r2(self.test_r2, 'test_r2', prog_bar=True)

    def _log_r2(self, metric: R2Score, name: str, prog_bar: bool = False):
        try:
            value = metric.compute()
        except ValueError:
            if self.nan_tensor is None:
                self.nan_tensor = torch.tensor(float('nan'), device=self.device)
            value = self.nan_tensor
        self.log(name, value, prog_bar=prog_bar)
        metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.001,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == '__main__':
    MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 3))

    hypergraph_model_registry = {
        "HyperTransformer": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: HyperTransformer(
                in_channels=in_c,
                hidden_channels=256,
                out_channels=out_c,
                num_layers=2,
                num_heads=4,
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
        "SheafHyperGNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("SheafHyperGNN", in_c, out_c, stats=stats),
        },
        "HyperND": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("HyperND", in_c, out_c, stats=stats),
        },
        "EHNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("EHNN", in_c, out_c, stats=stats),
        },
        "ED-HNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("EHNN", in_c, out_c, stats=stats),
        },
        "AllSetformer": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("AllSetformer", in_c, out_c, stats=stats),
        },
        "AllDeepSets": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("AllDeepSets", in_c, out_c, stats=stats),
        },
        "TFHNN": {
            "constructor": lambda in_c, out_c, _sample, stats=None, **__: build_dhg_model("TFHNN", in_c, out_c, stats=stats),
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
    }

    enabled_hg_names = os.environ.get("HYP_MODELS")
    if enabled_hg_names:
        requested = [name.strip() for name in enabled_hg_names.split(",") if name.strip()]
        missing = [name for name in requested if name not in hypergraph_model_registry]
        if missing:
            raise ValueError(f"HYP_MODELS에 알 수 없는 모델이 포함되어 있습니다: {missing}")
        hypergraph_models = {name: hypergraph_model_registry[name] for name in requested}
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

    designs_env = os.environ.get("REG_DESIGNS")
    design_list = [d.strip() for d in designs_env.split(",") if d.strip()] if designs_env else []

    def register_dataset(store, key, path, models):
        if not path or not os.path.isfile(path):
            print(f"[WARN] dataset file '{path}' missing; skipping {key}")
            return
        store[key] = {"data_path": path, "models": models}

    CONFIG = {}
    register_dataset(CONFIG, "hypergraph", "regression_hypergraph_dataset.pt", hypergraph_models)
    register_dataset(CONFIG, "bipartite", "regression_bipartite_dataset.pt", bipartite_models)
    register_dataset(CONFIG, "clique", "regression_clique_dataset.pt", clique_models)

    for design in design_list:
        tag = design.upper()
        register_dataset(
            CONFIG,
            f"hypergraph_{tag}",
            f"regression_hypergraph_dataset_{tag}.pt",
            hypergraph_models,
        )
        register_dataset(
            CONFIG,
            f"bipartite_{tag}",
            f"regression_bipartite_dataset_{tag}.pt",
            bipartite_models,
        )
        register_dataset(
            CONFIG,
            f"clique_{tag}",
            f"regression_clique_dataset_{tag}.pt",
            clique_models,
        )

    requested_datasets = os.environ.get("REG_DATASETS")
    if requested_datasets:
        order = []
        missing = []
        for name in requested_datasets.split(","):
            key = name.strip()
            if not key:
                continue
            if key not in CONFIG:
                missing.append(key)
            else:
                order.append((key, CONFIG[key]))
        if missing:
            raise ValueError(f"REG_DATASETS에 알 수 없는 항목이 포함되어 있습니다: {missing}")
        dataset_items = order
    elif design_list:
        dataset_items = []
        dataset_priority = ["hypergraph", "bipartite", "clique"]
        for design in design_list:
            tag = design.upper()
            for prefix in dataset_priority:
                key = f"{prefix}_{tag}"
                if key in CONFIG:
                    dataset_items.append((key, CONFIG[key]))
    else:
        dataset_items = list(CONFIG.items())

    batch_size = int(os.environ.get("REG_BATCH_SIZE", os.environ.get("BATCH_SIZE", 16)))

    for dataset_name, dataset_config in dataset_items:
        print(f"--- Training on {dataset_name} dataset ---")
        data_module = GraphDataModule(dataset_config["data_path"], batch_size=batch_size)
        data_module.setup()

        sample_data = data_module.train_dataset[0]
        in_channels = sample_data.x.shape[1]
        out_channels = sample_data.y.shape[0]

        if hasattr(sample_data, "edge_attr") and sample_data.edge_attr is not None:
            edge_attr_channels = sample_data.edge_attr.shape[1]
        else:
            edge_attr_channels = 0

        print(f"Dataset dimensions - in_channels: {in_channels}, targets: {out_channels}")
        is_graph_dataset = dataset_name.startswith(("bipartite", "clique", "star"))
        if is_graph_dataset:
            print(f"Edge attribute channels: {edge_attr_channels}")

        dataset_stats = compute_hypergraph_stats(data_module.train_dataset) if dataset_name.startswith("hypergraph") else {}

        for model_name, model_config in dataset_config["models"].items():
            print(f"--- Training model: {model_name} ---")
            if is_graph_dataset:
                model = model_config["constructor"](in_channels, out_channels, edge_attr_channels, sample_data, dataset_stats)
            else:
                model = model_config["constructor"](in_channels, out_channels, sample_data, dataset_stats)

            lit_model = LitGraphModel(model, num_targets=out_channels, learning_rate=1e-4)

            tb_logger = TensorBoardLogger("logs", name=f"{dataset_name}_reg")
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
                fast_dev_run=fast_dev,
            )

            trainer.fit(lit_model, datamodule=data_module)
            print(f"--- Testing model: {model_name} ---")
            trainer.test(lit_model, datamodule=data_module)
