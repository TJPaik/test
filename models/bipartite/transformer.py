import os
import torch
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, TransformerEncoder, TransformerEncoderLayer, LayerNorm
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from scipy import sparse

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


# Define a Bipartite Transformer Network for classification
class BipartiteTransformer(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_channels, hidden_channels, out_channels, num_layers=1, 
                 num_heads=2, dropout=0.2):
        super(BipartiteTransformer, self).__init__()

        print(f"Initializing BipartiteTransformer with in_channels={in_channels}, edge_attr_channels={edge_attr_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}")

        # Node and edge feature encoders
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_attr_channels, hidden_channels)

        # Initial graph convolution to capture graph structure
        self.graph_conv = GATConv(hidden_channels, hidden_channels, edge_dim=hidden_channels)
        self.batch_norm = BatchNorm1d(hidden_channels)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(hidden_channels)
        )

        # Output layer
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Apply graph convolution to capture graph structure
        x = self.graph_conv(x, edge_index, edge_attr)
        x = self.batch_norm(x)
        x = F.relu(x)

        # Prepare for transformer (batch_size, seq_len, hidden_dim)
        # In our case, seq_len is the number of nodes
        batch_size = 1  # We process one graph at a time
        seq_len = x.size(0)  # Number of nodes

        # Check if sequence length is too large
        max_seq_len = 100  # Set a reasonable maximum sequence length
        if seq_len > max_seq_len:
            # If too many nodes, sample or use global pooling to reduce
            indices = torch.randperm(seq_len)[:max_seq_len]
            x = x[indices]
            seq_len = max_seq_len

        # Reshape for transformer
        x = x.view(batch_size, seq_len, -1)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Global pooling (mean of all node features)
        x = torch.mean(x, dim=1).squeeze(0)

        # Apply final linear layer
        x = self.lin(x)

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
    edge_attr_channels = dataset[0].edge_attr.shape[1]
    out_channels = dataset[0].y.shape[0]

    # Initialize model with simplified parameters
    model = BipartiteTransformer(
        in_channels=in_channels, 
        edge_attr_channels=edge_attr_channels,
        hidden_channels=32,  # Reduced hidden dimension
        out_channels=out_channels,
        num_layers=1,  # Single transformer layer
        num_heads=2,   # Fewer attention heads
        dropout=0.1    # Reduced dropout
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
        plt.savefig('bipartite_transformer_training_loss.png')
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
        plt.savefig('bipartite_transformer_confusion_matrix.png')
    else:
        print("Matplotlib not available. Skipping confusion matrix plot.")

    return model, accuracy

# LaplacianPositionalTransformer Implementation
# This section implements a transformer model that uses Laplacian eigenvectors as positional encoding.
# The Laplacian matrix captures the graph structure, and its eigenvectors provide a spectral representation
# of the graph that can be used as positional encoding in the transformer.

# Function to calculate Laplacian matrix and its eigenvectors for a graph
def calculate_laplacian_eigenvectors(edge_index, num_nodes, k=8):
    """
    Calculate the Laplacian matrix and its eigenvectors for a graph.

    Args:
        edge_index (torch.Tensor): Edge indices of shape [2, num_edges]
        num_nodes (int): Number of nodes in the graph
        k (int): Number of eigenvectors to compute (excluding the first one)

    Returns:
        torch.Tensor: Eigenvectors of the Laplacian matrix of shape [num_nodes, k]
    """
    # Convert edge_index to adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # Make it undirected

    # Calculate degree matrix
    degree = adj.sum(dim=1)
    degree_matrix = torch.diag(degree)

    # Calculate Laplacian matrix: L = D - A
    laplacian = degree_matrix - adj

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

    # Sort eigenvalues and eigenvectors
    indices = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Skip the first eigenvector (corresponds to eigenvalue 0)
    # and take the next k eigenvectors
    k = min(k, num_nodes - 1)  # Ensure k is not larger than num_nodes - 1
    selected_eigenvectors = eigenvectors[:, 1:k+1]

    return selected_eigenvectors

# Function to precompute Laplacian eigenvectors for a dataset
def precompute_laplacian_eigenvectors(dataset, k=8):
    """
    Precompute Laplacian eigenvectors for all graphs in a dataset.

    Args:
        dataset (list): List of graph data objects
        k (int): Number of eigenvectors to compute (excluding the first one)

    Returns:
        list: List of graph data objects with precomputed Laplacian eigenvectors
    """
    processed_dataset = []

    for data in dataset:
        num_nodes = data.x.size(0)
        eigenvectors = calculate_laplacian_eigenvectors(data.edge_index, num_nodes, k)

        # Create a new data object with the same attributes plus eigenvectors
        new_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y,
            laplacian_eigenvectors=eigenvectors
        )
        processed_dataset.append(new_data)

    return processed_dataset

# Define a Laplacian Positional Transformer Network for classification
class LaplacianPositionalTransformer(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_channels, hidden_channels, out_channels, 
                 num_layers=1, num_heads=2, dropout=0.2, pos_encoding_dim=8):
        super(LaplacianPositionalTransformer, self).__init__()

        print(f"Initializing LaplacianPositionalTransformer with in_channels={in_channels}, "
              f"edge_attr_channels={edge_attr_channels}, hidden_channels={hidden_channels}, "
              f"out_channels={out_channels}, pos_encoding_dim={pos_encoding_dim}")

        # Node feature encoder
        self.node_encoder = Linear(in_channels, hidden_channels // 2)

        # Edge feature encoder
        self.edge_encoder = Linear(edge_attr_channels, hidden_channels)

        # Positional encoding projection
        self.pos_encoder = Linear(pos_encoding_dim, hidden_channels // 2)

        # Combined features projection (after concatenation)
        self.combined_projection = Linear(hidden_channels, hidden_channels)

        # Initial graph convolution to capture graph structure
        self.graph_conv = GATConv(hidden_channels, hidden_channels, edge_dim=hidden_channels)
        self.batch_norm = BatchNorm1d(hidden_channels)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(hidden_channels)
        )

        # Output layer
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, laplacian_eigenvectors=None):
        # Encode node features
        node_features = self.node_encoder(x)

        # Encode edge features
        edge_attr = self.edge_encoder(edge_attr)

        # Process positional encoding if available
        if laplacian_eigenvectors is not None:
            # Check if the positional encoding dimension matches
            if laplacian_eigenvectors.size(1) != self.pos_encoder.in_features:
                # Pad or truncate the eigenvectors to match the expected dimension
                if laplacian_eigenvectors.size(1) < self.pos_encoder.in_features:
                    # Pad with zeros if we have fewer eigenvectors than expected
                    padding = torch.zeros(laplacian_eigenvectors.size(0), 
                                         self.pos_encoder.in_features - laplacian_eigenvectors.size(1),
                                         device=laplacian_eigenvectors.device)
                    laplacian_eigenvectors = torch.cat([laplacian_eigenvectors, padding], dim=1)
                else:
                    # Truncate if we have more eigenvectors than expected
                    laplacian_eigenvectors = laplacian_eigenvectors[:, :self.pos_encoder.in_features]

            # Encode positional information
            pos_encoding = self.pos_encoder(laplacian_eigenvectors)

            # Concatenate node features and positional encoding
            combined_features = torch.cat([node_features, pos_encoding], dim=1)

            # Project to hidden dimension
            node_features = self.combined_projection(combined_features)
        else:
            # If no positional encoding, project node features to full hidden dimension
            zeros = torch.zeros(node_features.size(0), node_features.size(1), 
                               device=node_features.device)
            combined_features = torch.cat([node_features, zeros], dim=1)
            node_features = self.combined_projection(combined_features)

        # Apply graph convolution to capture graph structure
        x = self.graph_conv(node_features, edge_index, edge_attr)
        x = self.batch_norm(x)
        x = F.relu(x)

        # Prepare for transformer (batch_size, seq_len, hidden_dim)
        # In our case, seq_len is the number of nodes
        batch_size = 1  # We process one graph at a time
        seq_len = x.size(0)  # Number of nodes

        # Check if sequence length is too large
        max_seq_len = 100  # Set a reasonable maximum sequence length
        if seq_len > max_seq_len:
            # If too many nodes, sample or use global pooling to reduce
            indices = torch.randperm(seq_len)[:max_seq_len]
            x = x[indices]
            seq_len = max_seq_len

        # Reshape for transformer
        x = x.view(batch_size, seq_len, -1)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Global pooling (mean of all node features)
        x = torch.mean(x, dim=1).squeeze(0)

        # Apply final linear layer
        x = self.lin(x)

        return x

# Function to train and evaluate the Laplacian Positional Transformer model
def train_and_evaluate_laplacian_transformer(dataset, test_size=0.2, epochs=100, lr=0.001, pos_encoding_dim=8):
    # Precompute Laplacian eigenvectors for the dataset
    print("Precomputing Laplacian eigenvectors for the dataset...")
    processed_dataset = precompute_laplacian_eigenvectors(dataset, k=pos_encoding_dim)

    # Split dataset into train and test sets
    train_indices, test_indices = train_test_split(range(len(processed_dataset)), test_size=test_size, random_state=42)
    train_dataset = [processed_dataset[i] for i in train_indices]
    test_dataset = [processed_dataset[i] for i in test_indices]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Get dimensions from the first sample
    in_channels = processed_dataset[0].x.shape[1]
    edge_attr_channels = processed_dataset[0].edge_attr.shape[1]
    out_channels = processed_dataset[0].y.shape[0]

    # Initialize model with improved parameters
    model = LaplacianPositionalTransformer(
        in_channels=in_channels, 
        edge_attr_channels=edge_attr_channels,
        hidden_channels=64,  # Increased hidden dimension
        out_channels=out_channels,
        num_layers=2,  # Two transformer layers
        num_heads=4,   # More attention heads
        dropout=0.2,   # Slightly increased dropout
        pos_encoding_dim=pos_encoding_dim
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
            out = model(data.x, data.edge_index, data.edge_attr, data.laplacian_eigenvectors)

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
        plt.title('Laplacian Positional Transformer Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('laplacian_transformer_training_loss.png')
    else:
        print("Matplotlib not available. Skipping loss plot.")

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in test_dataset:
            out = model(data.x, data.edge_index, data.edge_attr, data.laplacian_eigenvectors)
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
        plt.title('Laplacian Positional Transformer Confusion Matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('laplacian_transformer_confusion_matrix.png')
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
    # Check if we have enough samples in the real dataset

    # Load the existing bipartite dataset
    print("Loading bipartite dataset...")
    bipartite_dataset = torch.load("classification_bipartite_dataset.pt", weights_only=False)
    print(f"Dataset loaded with {len(bipartite_dataset)} samples")

    if len(bipartite_dataset) > 10:
        # Train on a small subset of the real dataset for testing
        print("Using a small subset of the real dataset for testing")
        subset_size = min(100, len(bipartite_dataset))
        # subset_size = len(bipartite_dataset)
        subset_indices = torch.randperm(len(bipartite_dataset))[:subset_size].tolist()
        subset_dataset = [bipartite_dataset[i] for i in subset_indices]

        # Train the original BipartiteTransformer model
        print("\n=== Training BipartiteTransformer model ===")
        model_real, acc_real = train_and_evaluate(subset_dataset, epochs=130)
        print(f"BipartiteTransformer accuracy: {acc_real:.4f}")

        # Test invariance on the first sample of the subset dataset
        test_model_invariance(model_real, subset_dataset[0])

        # Train the new LaplacianPositionalTransformer model
        print("\n=== Training LaplacianPositionalTransformer model ===")
        laplacian_model, laplacian_acc = train_and_evaluate_laplacian_transformer(
            subset_dataset, 
            epochs=130, 
            pos_encoding_dim=8
        )
        print(f"LaplacianPositionalTransformer accuracy: {laplacian_acc:.4f}")

        # Compare the two models
        print("\n=== Model Comparison ===")
        print(f"BipartiteTransformer accuracy: {acc_real:.4f}")
        print(f"LaplacianPositionalTransformer accuracy: {laplacian_acc:.4f}")
        print(f"Improvement: {(laplacian_acc - acc_real) * 100:.2f}%")

        # Test invariance on the first sample of the subset dataset for the Laplacian model
        processed_dataset = precompute_laplacian_eigenvectors([subset_dataset[0]], k=8)
        test_data = processed_dataset[0]

        # Define a wrapper function to match the interface expected by test_model_invariance
        class LaplacianModelWrapper(torch.nn.Module):
            def __init__(self, model, laplacian_eigenvectors):
                super(LaplacianModelWrapper, self).__init__()
                self.model = model
                self.laplacian_eigenvectors = laplacian_eigenvectors

            def forward(self, x, edge_index, edge_attr):
                return self.model(x, edge_index, edge_attr, self.laplacian_eigenvectors)

        wrapped_model = LaplacianModelWrapper(laplacian_model, test_data.laplacian_eigenvectors)
        print("\nTesting invariance for LaplacianPositionalTransformer:")
        test_model_invariance(wrapped_model, test_data)
