import torch
import numpy as np
from torch_geometric.nn import HypergraphConv
import numpy as np
import torch
from torch_geometric.nn import HypergraphConv

# Try to import matplotlib, but continue if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Plotting functionality will be disabled.")
    MATPLOTLIB_AVAILABLE = False
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.data import Data

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
SKLEARN_AVAILABLE = True

# Define a Hypergraph Neural Network for classification
class HyperGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(HyperGNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # First layer
        self.convs.append(HypergraphConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(HypergraphConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Last layer
        self.convs.append(HypergraphConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Output layer
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, hyperedge_index):
        # Apply hypergraph convolutions
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, hyperedge_index)
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
    out_channels = dataset[0].y.shape[0]

    # Initialize model
    model = HyperGNN(in_channels=in_channels, hidden_channels=64, out_channels=out_channels)
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
            out = model(data.x, data.hyperedge_index)

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
        plt.savefig('training_loss.png')
    else:
        print("Matplotlib not available. Skipping loss plot.")

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_dataset:
            out = model(data.x, data.hyperedge_index)
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
        plt.savefig('confusion_matrix.png')
    else:
        print("Matplotlib not available. Skipping confusion matrix plot.")

    return model, accuracy

# Function to test if model is invariant to node and edge ordering
def test_model_invariance(model, data):
    model.eval()  # Set model to evaluation mode

    # Get original output
    with torch.no_grad():
        original_output = model(data.x, data.hyperedge_index)

    print("\nTesting model invariance to node and edge ordering...")

    # Test node ordering invariance
    print("Testing node ordering invariance...")
    # Create a permutation of node indices
    num_nodes = data.x.size(0)
    node_perm = torch.randperm(num_nodes)

    # Permute node features
    permuted_x = data.x[node_perm]

    # Permute hyperedge_index (update node indices in hyperedge_index)
    node_perm_dict = {int(old_idx): int(new_idx) for old_idx, new_idx in enumerate(node_perm)}
    permuted_hyperedge_index = data.hyperedge_index.clone()
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
            permuted_hyperedge_index = data.hyperedge_index.clone()
            for i in range(permuted_hyperedge_index.size(1)):
                old_edge_idx = int(permuted_hyperedge_index[1, i])
                if old_edge_idx in edge_perm_dict:
                    permuted_hyperedge_index[1, i] = edge_perm_dict[old_edge_idx]

            # Get output with permuted edges
            with torch.no_grad():
                edge_permuted_output = model(data.x, permuted_hyperedge_index)

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
    # Load the existing hypergraph dataset
    print("Loading hypergraph dataset...")

    hypergraph_dataset = torch.load("classification_hypergraph_dataset.pt", weights_only=False)
    print(f"Dataset loaded with {len(hypergraph_dataset)} samples")

    print("Working with real dataset...")
    # Check if we have enough samples in the real dataset
    if len(hypergraph_dataset) > 10:
        # Train on the real dataset
        model_real, acc_real = train_and_evaluate(hypergraph_dataset, epochs=50)
        print(f"Real dataset accuracy: {acc_real:.4f}")

        # Test invariance on the first sample of the dataset
        test_model_invariance(model_real, hypergraph_dataset[0])
    else:
        print("Real dataset is too small, using synthetic dataset instead.")

