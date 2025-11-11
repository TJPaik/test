import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, TransformerEncoder, TransformerEncoderLayer, LayerNorm, Embedding
from torch_geometric.nn import HypergraphConv

# Device selection (CUDA if available, else CPU)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# Try to import matplotlib, but continue if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Plotting functionality will be disabled.")
    MATPLOTLIB_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
SKLEARN_AVAILABLE = True



class HypergraphTransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, num_heads, dropout=0.1):
        super().__init__()
        self.hypergraph_conv = HypergraphConv(in_channels, in_channels, use_attention=True)
        self.layer_norm1 = LayerNorm(in_channels)
        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=in_channels * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.layer_norm2 = LayerNorm(in_channels)

    def forward(self, x, hyperedge_index, hyperedge_attr):
        # Hypergraph convolution
        x_conv = self.hypergraph_conv(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
        x = self.layer_norm1(x + x_conv)

        # Transformer encoder layer
        x_transformer = self.transformer_encoder_layer(x.unsqueeze(0)).squeeze(0)
        x = self.layer_norm2(x + x_transformer)

        return x

class HyperTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1,
                 num_heads=2, dropout=0.1):
        super(HyperTransformer, self).__init__()

        print(f"Initializing HyperTransformer with in_channels={in_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}")

        self.embedding = torch.nn.Sequential(
            Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            LayerNorm(hidden_channels)
        )

        self.lin_edge_attr = torch.nn.Sequential(
            Linear(6, hidden_channels),
            torch.nn.ReLU(),
            LayerNorm(hidden_channels)
        )

        self.blocks = torch.nn.ModuleList([
            HypergraphTransformerBlock(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            LayerNorm(hidden_channels)
        )

        self.fc = Linear(hidden_channels, out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, hyperedge_index, data):
        x = self.embedding(x)

        if hasattr(data, 'hyperedge_attr'):
            edge_attr = self.lin_edge_attr(data.hyperedge_attr.float())
        else:
            num_edges = torch.max(hyperedge_index[1]) + 1
            edge_attr = torch.zeros(num_edges, x.size(1), device=x.device)

        for block in self.blocks:
            x = block(x, hyperedge_index, edge_attr)

        x_mean = torch.mean(x, dim=0)
        x_max, _ = torch.max(x, dim=0)
        x = x_mean + 0.5 * x_max

        x = self.output_proj(x)
        x = self.fc(x)

        return x


# Function to train and evaluate the model
def train_and_evaluate(dataset, test_size=0.2, epochs=100, lr=0.001):
    # Split dataset into train and test sets
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Get dimensions from the first sample
    in_channels = dataset[0].x.shape[1]
    out_channels = dataset[0].y.shape[0]

    # Initialize model with simplified parameters
    model = HyperTransformer(
        in_channels=in_channels,
        hidden_channels=256,  # Reduced hidden dimension
        out_channels=out_channels,
        num_layers=4,  # Single transformer layer
        num_heads=4,   # Fewer attention heads
        dropout=0.1    # Reduced dropout
    )
    # Move model to device
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for data in train_dataset:
            optimizer.zero_grad()

            # Move data to device
            x = data.x.to(DEVICE)
            hyperedge_index = data.hyperedge_index.to(DEVICE)

            # Forward pass
            out = model(x, hyperedge_index)

            # Compute loss - convert one-hot encoded target to class index
            target_idx = torch.argmax(data.y).to(DEVICE).unsqueeze(0).long()
            loss = criterion(out.unsqueeze(0), target_idx)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Plot training loss if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('transformer_training_loss.png')
    else:
        print("Matplotlib not available. Skipping loss plot.")

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_dataset:
            x = data.x.to(DEVICE)
            hyperedge_index = data.hyperedge_index.to(DEVICE)
            out = model(x, hyperedge_index)
            pred = torch.argmax(out).item()
            true = torch.argmax(data.y).item()

            y_true.append(true)
            y_pred.append(pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('transformer_confusion_matrix.png')
    else:
        print("Matplotlib not available. Skipping confusion matrix plot.")

    return model, accuracy

# Function to test if model is invariant to node and edge ordering
def test_model_invariance(model, data):
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device

    # Get original output
    with torch.no_grad():
        original_output = model(data.x.to(device), data.hyperedge_index.to(device))

    print("\nTesting model invariance to node and edge ordering...")

    # Test node ordering invariance
    print("Testing node ordering invariance...")
    # Create a permutation of node indices
    num_nodes = data.x.size(0)
    node_perm = torch.randperm(num_nodes)

    # Permute node features
    permuted_x = data.x[node_perm].to(device)

    # Permute hyperedge_index (update node indices in hyperedge_index)
    node_perm_dict = {int(old_idx): int(new_idx) for old_idx, new_idx in enumerate(node_perm)}
    permuted_hyperedge_index = data.hyperedge_index.clone().to(device)
    for i in range(permuted_hyperedge_index.size(1)):
        if permuted_hyperedge_index[0, i] < num_nodes:  # Check if it's a node index
            old_idx = int(permuted_hyperedge_index[0, i])
            permuted_hyperedge_index[0, i] = node_perm_dict[old_idx]

    # Get output with permuted nodes
    with torch.no_grad():
        node_permuted_output = model(permuted_x, permuted_hyperedge_index)

    # Check if outputs are the same (or very close)
    node_diff = torch.abs(original_output - node_permuted_output).max().item()
    node_invariant = node_diff < 1e-5
    print(f"Node ordering invariance: {'Yes' if node_invariant else 'No'}")
    print(f"Maximum difference in outputs: {node_diff:.8f}")

    # Test edge ordering invariance
    print("\nTesting edge ordering invariance...")
    # Get unique hyperedge indices
    if data.hyperedge_index.size(1) > 0:
        unique_edges = torch.unique(data.hyperedge_index[1])
        num_edges = len(unique_edges)

        if num_edges > 1:  # Only test if there are at least 2 hyperedges
            # Create a permutation of hyperedge indices
            edge_perm = torch.randperm(num_edges)
            edge_perm_dict = {int(unique_edges[old_idx]): int(unique_edges[edge_perm[old_idx]])
                             for old_idx in range(num_edges)}

            # Permute hyperedge_index (update edge indices)
            permuted_hyperedge_index = data.hyperedge_index.clone().to(device)
            for i in range(permuted_hyperedge_index.size(1)):
                old_edge_idx = int(permuted_hyperedge_index[1, i])
                if old_edge_idx in edge_perm_dict:
                    permuted_hyperedge_index[1, i] = edge_perm_dict[old_edge_idx]

            # Get output with permuted edges
            with torch.no_grad():
                edge_permuted_output = model(data.x.to(device), permuted_hyperedge_index)

            # Check if outputs are the same (or very close)
            edge_diff = torch.abs(original_output - edge_permuted_output).max().item()
            edge_invariant = edge_diff < 1e-5
            print(f"Edge ordering invariance: {'Yes' if edge_invariant else 'No'}")
            print(f"Maximum difference in outputs: {edge_diff:.8f}")
        else:
            print("Not enough hyperedges to test edge ordering invariance.")
    else:
        print("No hyperedges found in the data.")

    return node_invariant, edge_invariant if 'edge_invariant' in locals() else (node_invariant, None)

# Main execution
if __name__ == "__main__":
    print("Working with real dataset...")

    # Load the existing hypergraph dataset
    print("Loading hypergraph dataset...")
    hypergraph_dataset = torch.load("classification_hypergraph_dataset.pt", weights_only=False)
    print(f"Dataset loaded with {len(hypergraph_dataset)} samples")
    # Check if we have enough samples in the real dataset
    if len(hypergraph_dataset) > 10:
        # Train on a small subset of the real dataset for testing
        print("Using a small subset of the real dataset for testing")
        # subset_size = min(100, )
        subset_size = len(hypergraph_dataset)
        subset_indices = torch.randperm(len(hypergraph_dataset))[:subset_size].tolist()
        subset_dataset = [hypergraph_dataset[i] for i in subset_indices]

        # Train with fewer epochs for testing
        model_real, acc_real = train_and_evaluate(subset_dataset, epochs=200)
        print(f"Real dataset subset accuracy: {acc_real:.4f}")

        # Test invariance on the first sample of the subset dataset
        test_model_invariance(model_real, subset_dataset[0])
    else:
        print("Real dataset is too small, using synthetic dataset instead.")
