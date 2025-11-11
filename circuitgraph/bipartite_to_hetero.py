import torch
from torch_geometric.data import HeteroData

# Load the dataset
dataset_path = "classification_bipartite_dataset.pt"
data_list = torch.load(dataset_path, weights_only=False)

# Function to convert Data to HeteroData
def convert_to_hetero_data(data):
    hetero_data = HeteroData()

    # Split nodes into D (components, indices 0-6) and N (nets, indices 7-9)
    node_types = torch.argmax(data.x, dim=1)

    # D nodes (components: indices 0-6)
    d_mask = node_types <= 6
    d_indices = torch.nonzero(d_mask).flatten()
    hetero_data['D'].x = data.x[d_mask]

    # N nodes (nets: indices 7-9)
    n_mask = node_types >= 7
    n_indices = torch.nonzero(n_mask).flatten()
    hetero_data['N'].x = data.x[n_mask]

    # Create mapping from original indices to new indices
    d_idx_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(d_indices)}
    n_idx_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(n_indices)}

    # Process edges (D to N, N to D, D to D, and N to N)
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    d_to_n_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    n_to_d_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    d_to_d_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    n_to_n_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        src_type = node_types[src].item()
        dst_type = node_types[dst].item()

        if src_type <= 6 and dst_type >= 7:  # U to V
            d_to_n_mask[i] = True
        elif src_type >= 7 and dst_type <= 6:  # V to U
            n_to_d_mask[i] = True
        elif src_type <= 6 and dst_type <= 6:  # U to U
            d_to_d_mask[i] = True
        elif src_type >= 7 and dst_type >= 7:  # V to V
            n_to_n_mask[i] = True

    # D to N edges
    if d_to_n_mask.sum() > 0:
        d_to_n_edge_index = edge_index[:, d_to_n_mask]
        d_to_n_edge_attr = edge_attr[d_to_n_mask]

        # Remap indices
        src_mapped = torch.tensor([d_idx_map[src.item()] for src in d_to_n_edge_index[0]])
        dst_mapped = torch.tensor([n_idx_map[dst.item()] for dst in d_to_n_edge_index[1]])

        hetero_data['D', 'to', 'N'].edge_index = torch.stack([src_mapped, dst_mapped])
        hetero_data['D', 'to', 'N'].edge_attr = d_to_n_edge_attr

    # N to D edges
    if n_to_d_mask.sum() > 0:
        n_to_d_edge_index = edge_index[:, n_to_d_mask]
        n_to_d_edge_attr = edge_attr[n_to_d_mask]

        # Remap indices
        src_mapped = torch.tensor([n_idx_map[src.item()] for src in n_to_d_edge_index[0]])
        dst_mapped = torch.tensor([d_idx_map[dst.item()] for dst in n_to_d_edge_index[1]])

        hetero_data['N', 'to', 'D'].edge_index = torch.stack([src_mapped, dst_mapped])
        hetero_data['N', 'to', 'D'].edge_attr = n_to_d_edge_attr

    # D to D edges
    if d_to_d_mask.sum() > 0:
        d_to_d_edge_index = edge_index[:, d_to_d_mask]
        d_to_d_edge_attr = edge_attr[d_to_d_mask]

        # Remap indices
        src_mapped = torch.tensor([d_idx_map[src.item()] for src in d_to_d_edge_index[0]])
        dst_mapped = torch.tensor([d_idx_map[dst.item()] for dst in d_to_d_edge_index[1]])

        hetero_data['D', 'to', 'D'].edge_index = torch.stack([src_mapped, dst_mapped])
        hetero_data['D', 'to', 'D'].edge_attr = d_to_d_edge_attr

    # N to N edges
    if n_to_n_mask.sum() > 0:
        n_to_n_edge_index = edge_index[:, n_to_n_mask]
        n_to_n_edge_attr = edge_attr[n_to_n_mask]

        # Remap indices
        src_mapped = torch.tensor([n_idx_map[src.item()] for src in n_to_n_edge_index[0]])
        dst_mapped = torch.tensor([n_idx_map[dst.item()] for dst in n_to_n_edge_index[1]])

        hetero_data['N', 'to', 'N'].edge_index = torch.stack([src_mapped, dst_mapped])
        hetero_data['N', 'to', 'N'].edge_attr = n_to_n_edge_attr

    # Add graph label
    hetero_data.y = data.y

    return hetero_data

# Convert all data to HeteroData
hetero_data_list = [convert_to_hetero_data(data) for data in data_list]

# Save the HeteroData list to a file
hetero_dataset_path = "classification_hetero_dataset.pt"
torch.save(hetero_data_list, hetero_dataset_path)
print(f"Saved HeteroData list to {hetero_dataset_path}")

