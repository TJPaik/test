
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch_geometric.data import Dataset
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from torchmetrics import Accuracy, ConfusionMatrix, R2Score

# Import models from their new locations
from models.hypergraph.hypernetwork import HyperGNN
from models.hypergraph.transformer import HyperTransformer
from models.bipartite.bipartite_network import BipartiteGNN
from models.bipartite.transformer import BipartiteTransformer, LaplacianPositionalTransformer

from pytorch_lightning.loggers import TensorBoardLogger
from torch_scatter import scatter_mean


def get_sinusoidal_anchor_pe(data, hyperedge_distances):
    """
    하이퍼엣지의 앵커별 거리에 사인 함수를 적용하여
    노드의 위치 인코딩(PE)을 생성합니다.

    Args:
        data (Data): PyG 데이터 객체. num_nodes와 hyperedge_index를 포함.
        hyperedge_distances (torch.Tensor): 각 하이퍼엣지의 앵커별 거리 정보.
                                            shape: [num_hyperedges, num_anchors(2)]

    Returns:
        torch.Tensor: 각 노드에 대한 사인 기반 위치 인코딩.
                      shape: [num_nodes, num_anchors(2)]
    """
    # 1. 파라미터 D를 각 앵커(열)의 최대 절대 거리로 설정합니다.
    #    0으로 나누는 것을 방지하기 위해 작은 값(epsilon)을 더하거나 1로 대체합니다.
    D = torch.max(torch.abs(hyperedge_distances), dim=0).values
    D[D == 0] = 1.0  # 만약 최대 거리가 0이면 1로 설정하여 0으로 나누는 것을 방지

    # 2. 각 하이퍼엣지에 대해 PE를 계산합니다. PE_e = sin(π * d_e / D)
    #    (Broadcasting을 통해 각 열이 해당 D값으로 나누어집니다)
    hyperedge_pe = torch.sin(torch.pi * hyperedge_distances / D)

    # 3. 각 노드에 속한 하이퍼엣지 PE들의 평균을 계산하여 노드의 최종 PE로 사용합니다.
    node_indices, hyperedge_indices = data.hyperedge_index
    pe_per_connection = hyperedge_pe[hyperedge_indices]
    node_pe = scatter_mean(pe_per_connection, node_indices, dim=0, dim_size=data.num_nodes)

    return node_pe




class GenericDataset(Dataset):
    def __init__(self, pt_file):
        super().__init__()
        self.data = torch.load(pt_file, weights_only=False)
        for a in self.data:
            pe1 = torch.sin(torch.pi * a.x[:, -3] / (a.x[:, -1] + 1e-5)).view(-1, 1)
            pe2 = torch.sin(torch.pi * a.x[:, -2] / (a.x[:, -1] + 1e-5)).view(-1, 1)
            a.x = torch.cat([a.x, pe1, pe2], dim=1)
            a.y = torch.log1p(a.y)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 32):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

    def setup(self, stage: str = None):
        full_dataset = GenericDataset(self.dataset_path)
        indices = list(range(len(full_dataset)))
        if len(indices) < 2:
            train_indices, val_indices, test_indices = indices, [], []
        else:
            train_indices, test_indices = train_test_split(indices, test_size=self.test_ratio, random_state=42)
            adjusted_val_ratio = self.val_ratio / max(1e-8, (1 - self.test_ratio))
            if len(train_indices) >= 2:
                train_indices, val_indices = train_test_split(train_indices, test_size=adjusted_val_ratio, random_state=42)
            else:
                val_indices = train_indices[:]
                train_indices = []

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        # This is a simple collate function. For graph data, you might need a more complex one from torch_geometric.
        # However, the original code processes one graph at a time, so we will do the same.
        return batch

class LitGraphModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, num_classes=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

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
        self.classification = False
        if self.classification:
            self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = torch.nn.MSELoss()

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

        # For tracking class distribution
        self.class_counts = torch.zeros(self.num_classes)
        self.class_weights_updated = False

    def forward(self, batch):
        # The models expect a single data object, not a batch. We iterate.
        outputs = []
        for data in batch:
            if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer)):
                 out = self.model(data.x, data.edge_index, data.edge_attr)
            else: # HyperGNN, HyperTransformer
                 out = self.model(data.x, data.hyperedge_index)

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
            if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer)):
                 out = self.model(data.x, data.edge_index, data.edge_attr)
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
            if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer)):
                 out = self.model(data.x, data.edge_index, data.edge_attr)
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
            if isinstance(self.model, (BipartiteGNN, BipartiteTransformer, LaplacianPositionalTransformer)):
                 out = self.model(data.x, data.edge_index, data.edge_attr)
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

            self.test_accuracy.reset()
            self.test_conf_matrix.reset()
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
    MAX_EPOCHS = 100  # Increased for better convergence

    # Define datasets and the models to run on them
    CONFIG = {
        # "hypergraph_CSVA": {
        #     "data_path": "regression_hypergraph_dataset_CSVA.pt",
        #     "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        # },
        # "hypergraph_CVA": {
        #     "data_path": "regression_hypergraph_dataset_CVA.pt",
        #     "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        # },
        # "hypergraph_LNA": {
        #     "data_path": "regression_hypergraph_dataset_LNA.pt",
        #     "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        # },
        # "hypergraph_Mixer": {
        #     "data_path": "regression_hypergraph_dataset_Mixer.pt",
        #     "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        # },
        # "hypergraph_PA": {
        #     "data_path": "regression_hypergraph_dataset_PA.pt",
        #     "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        # },
        # "hypergraph_Receiver": {
        #     "data_path": "regression_hypergraph_dataset_Receiver.pt",
        #     "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        # },
        # "hypergraph_Transmitter": {
        #     "data_path": "regression_hypergraph_dataset_Transmitter.pt",
        #     "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        # },
        "hypergraph_TSVA": {
            "data_path": "regression_hypergraph_dataset_TSVA.pt",
            "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        },
        "hypergraph_VCO": {
            "data_path": "regression_hypergraph_dataset_VCO.pt",
            "models": { "HyperTransformer": { "constructor": lambda in_c, out_c: HyperTransformer(in_channels=in_c, hidden_channels=512, out_channels=out_c, num_layers=8, num_heads=16, dropout=0.3), }, }
        },

        "bipartite_CSVA": {
            "data_path": "regression_bipartite_dataset_CSVA.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
        "bipartite_CVA": {
            "data_path": "regression_bipartite_dataset_CVA.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
        "bipartite_LNA": {
            "data_path": "regression_bipartite_dataset_LNA.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
        "bipartite_Mixer": {
            "data_path": "regression_bipartite_dataset_Mixer.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
        "bipartite_PA": {
            "data_path": "regression_bipartite_dataset_PA.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
        "bipartite_Receiver": {
            "data_path": "regression_bipartite_dataset_Receiver.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
        "bipartite_Transmitter": {
            "data_path": "regression_bipartite_dataset_Transmitter.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
        "bipartite_TSVA": {
            "data_path": "regression_bipartite_dataset_TSVA.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
        "bipartite_VCO": {
            "data_path": "regression_bipartite_dataset_VCO.pt",
            "models": { "LaplacianPositionalTransformer": { "constructor": lambda in_c, out_c, edge_c: LaplacianPositionalTransformer(in_channels=in_c, edge_attr_channels=edge_c, hidden_channels=256, out_channels=out_c), },}
        },
    }



    # --- Training Loop ---
    for dataset_name, dataset_config in CONFIG.items():
        print(f"--- Training on {dataset_name} dataset ---")

        # Initialize DataModule with moderate batch size
        data_module = GraphDataModule(dataset_config["data_path"], batch_size=4) # Moderate batch size for stable training

        # Setup the data module to load the dataset
        data_module.setup()

        # Get a sample from the dataset to determine dimensions
        sample_data = data_module.train_dataset[0]

        # Extract dimensions from the sample
        in_channels = sample_data.x.shape[1]  # Number of input features

        if dataset_name.startswith("hypergraph"):
            out_channels = sample_data.y.shape[0]  # Number of classes for hypergraph
        else:  # bipartite
            out_channels = sample_data.y.shape[0]  # Number of classes for bipartite
            edge_attr_channels = sample_data.edge_attr.shape[1]  # Edge feature dimensions

        print(f"Dataset dimensions - in_channels: {in_channels}, out_channels: {out_channels}")
        if dataset_name.startswith("bipartite"):
            print(f"Edge attribute channels: {edge_attr_channels}")

        for model_name, model_config in dataset_config["models"].items():
            print(f"--- Training model: {model_name} ---")

            # Initialize model with dimensions from the dataset
            if dataset_name.startswith("bipartite"):
                model = model_config["constructor"](in_channels, out_channels, edge_attr_channels)
            else:  # hypergraph
                model = model_config["constructor"](in_channels, out_channels)

            lit_model = LitGraphModel(model, num_classes=out_channels, learning_rate=1e-4)

            tb_logger = TensorBoardLogger("logs", name=dataset_name)
            # Initialize Trainer with enhanced configuration
            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                callbacks=[
                    TQDMProgressBar(refresh_rate=5),
                    # ModelCheckpoint(
                    #     monitor='val_acc',
                    #     mode='max',
                    #     save_top_k=1,  # Save top 3 models
                    #     filename='{epoch}-{val_acc:.4f}'
                    # ),
                    # pl.callbacks.EarlyStopping(
                    #     monitor='val_acc',
                    #     mode='max',
                    #     patience=125,  # Increased patience
                    #     min_delta=0.005  # Smaller delta to catch smaller improvements
                    # )
                ],
                strategy="ddp_find_unused_parameters_true",
                accelerator="auto",
                gradient_clip_val=0.5,  # Reduced gradient clipping for more stable training
                enable_progress_bar=True,
                log_every_n_steps=1,
                precision=16,  # Use mixed precision for faster training
                logger=tb_logger
            )

            # Train and validate
            trainer.fit(lit_model, datamodule=data_module)

            # Test
            print(f"--- Testing model: {model_name} ---")
            trainer.test(lit_model, datamodule=data_module)
