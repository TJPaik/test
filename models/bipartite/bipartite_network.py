import os
import torch
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.data import Data

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


# Define a Graph Neural Network for bipartite graphs
class BipartiteGNN(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_channels, hidden_channels, out_channels, num_layers=2):
        super(BipartiteGNN, self).__init__()

        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_attr_channels, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATConv(hidden_channels, hidden_channels, edge_dim=hidden_channels))
        self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, edge_dim=hidden_channels))
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Last layer
        self.convs.append(GATConv(hidden_channels, hidden_channels, edge_dim=hidden_channels))
        self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Output layer
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Apply graph convolutions
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling (mean of all node features)
        x = torch.mean(x, dim=0)

        # Apply final linear layer
        x = self.lin(x)

        return x

# Function to train and evaluate the model
def train_and_evaluate(dataset, test_size=0.2, epochs=100, lr=0.01):
    # Split dataset into train and test sets
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Get dimensions from the first sample
    in_channels = dataset[0].x.shape[1]
    edge_attr_channels = dataset[0].edge_attr.shape[1]
    out_channels = dataset[0].y.shape[0]

    # Initialize model
    model = BipartiteGNN(
        in_channels=in_channels, 
        edge_attr_channels=edge_attr_channels,
        hidden_channels=64, 
        out_channels=out_channels
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for data in train_dataset:
            optimizer.zero_grad()

            # Forward pass
            out = model(data.x, data.edge_index, data.edge_attr)

            # Compute loss - convert one-hot encoded target to class index
            target_idx = torch.argmax(data.y).unsqueeze(0)
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
        plt.savefig('bipartite_training_loss.png')
    else:
        print("Matplotlib not available. Skipping loss plot.")

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_dataset:
            out = model(data.x, data.edge_index, data.edge_attr)
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
        plt.savefig('bipartite_confusion_matrix.png')
    else:
        print("Matplotlib not available. Skipping confusion matrix plot.")

    return model, accuracy

# Function to test if model is invariant to edge ordering
def test_model_invariance(model, data):
    model.eval()  # Set model to evaluation mode

    # Get original output
    with torch.no_grad():
        original_output = model(data.x, data.edge_index, data.edge_attr)

    print("\nTesting model invariance to node and edge ordering...")

    # Test node ordering invariance
    print("Testing node ordering invariance...")
    # Create a permutation of node indices
    num_nodes = data.x.size(0)
    node_perm = torch.randperm(num_nodes)

    # Permute node features
    permuted_x = data.x[node_perm]

    # Permute edge_index (update node indices in edge_index)
    node_perm_dict = {int(old_idx): int(new_idx) for old_idx, new_idx in enumerate(node_perm)}
    permuted_edge_index = data.edge_index.clone()
    for i in range(permuted_edge_index.size(1)):
        # Update source node
        old_src = int(permuted_edge_index[0, i])
        permuted_edge_index[0, i] = node_perm_dict[old_src]

        # Update target node
        old_tgt = int(permuted_edge_index[1, i])
        permuted_edge_index[1, i] = node_perm_dict[old_tgt]

    # Get output with permuted nodes
    with torch.no_grad():
        node_permuted_output = model(permuted_x, permuted_edge_index, data.edge_attr)

    # Check if outputs are the same (or very close)
    node_diff = torch.abs(original_output - node_permuted_output).max().item()
    node_invariant = node_diff < 1e-5
    print(f"Node ordering invariance: {'Yes' if node_invariant else 'No'}")
    print(f"Maximum difference in outputs: {node_diff:.8f}")

    # Test edge ordering invariance
    print("\nTesting edge ordering invariance...")
    # Create a permutation of edge indices
    num_edges = data.edge_index.size(1)

    if num_edges > 1:  # Only test if there are at least 2 edges
        edge_perm = torch.randperm(num_edges)

        # Permute edge_index and edge_attr
        permuted_edge_index = data.edge_index[:, edge_perm]
        permuted_edge_attr = data.edge_attr[edge_perm]

        # Get output with permuted edges
        with torch.no_grad():
            edge_permuted_output = model(data.x, permuted_edge_index, permuted_edge_attr)

        # Check if outputs are the same (or very close)
        edge_diff = torch.abs(original_output - edge_permuted_output).max().item()
        edge_invariant = edge_diff < 1e-5
        print(f"Edge ordering invariance: {'Yes' if edge_invariant else 'No'}")
        print(f"Maximum difference in outputs: {edge_diff:.8f}")
    else:
        print("Not enough edges to test edge ordering invariance.")
        edge_invariant = None

    return node_invariant, edge_invariant

# Main execution
if __name__ == "__main__":
    print("Working with real dataset...")

    # Load the existing bipartite dataset
    print("Loading bipartite dataset...")
    # Try to load with weights_only=False (needed for PyTorch 2.6+)
    bipartite_dataset = torch.load("classification_bipartite_dataset.pt", weights_only=False)
    print(f"Dataset loaded with {len(bipartite_dataset)} samples")

    # Check if we have enough samples in the real dataset
    if len(bipartite_dataset) > 10:
        # Train on the real dataset
        model_real, acc_real = train_and_evaluate(bipartite_dataset, epochs=50)
        print(f"Real dataset accuracy: {acc_real:.4f}")

        # Test invariance on the first sample of the dataset
        test_model_invariance(model_real, bipartite_dataset[0])
    else:
        print("Real dataset is too small, using synthetic dataset instead.")

