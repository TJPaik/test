import argparse
from itertools import combinations
from typing import List

import torch
from torch_geometric.data import Data


def hypergraph_to_clique(data: Data) -> Data:
    """Convert a hypergraph Data object (with hyperedge_index) into a simple graph via clique expansion."""
    if not hasattr(data, "hyperedge_index"):
        raise ValueError("Input data object must contain 'hyperedge_index'.")

    node_indices = data.hyperedge_index[0]
    hyperedge_ids = data.hyperedge_index[1]
    unique_hyperedges = torch.unique(hyperedge_ids, sorted=True)

    edge_cols: List[torch.Tensor] = []
    for he in unique_hyperedges:
        mask = hyperedge_ids == he
        nodes = torch.unique(node_indices[mask])
        if nodes.numel() < 2:
            continue

        # Build undirected clique edges for the nodes belonging to this hyperedge
        pair_list = torch.tensor(list(combinations(nodes.tolist(), 2)), dtype=torch.long)
        if pair_list.numel() == 0:
            continue
        forward = pair_list.t().contiguous()
        reverse = forward.flip(0)
        edge_cols.append(forward)
        edge_cols.append(reverse)

    if edge_cols:
        edge_index = torch.cat(edge_cols, dim=1)
        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attr = torch.ones(edge_index.size(1), 1, dtype=data.x.dtype) if edge_index.numel() > 0 else torch.empty((0, 1), dtype=data.x.dtype)
    clique_data = Data(
        x=data.x.clone(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=data.y.clone(),
    )
    return clique_data


def create_clique_dataset(
    hypergraph_dataset_path: str,
    clique_dataset_path: str = "classification_clique_dataset.pt",
):
    print(f"Loading hypergraph dataset from {hypergraph_dataset_path}")
    hypergraph_dataset = torch.load(hypergraph_dataset_path, weights_only=False)
    clique_dataset = []
    for idx, sample in enumerate(hypergraph_dataset):
        clique_data = hypergraph_to_clique(sample)
        clique_dataset.append(clique_data)
        if (idx + 1) % 100 == 0:
            print(f"Converted {idx + 1} graphs to clique form...")

    print(f"Saving clique dataset with {len(clique_dataset)} graphs to {clique_dataset_path}")
    torch.save(clique_dataset, clique_dataset_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert hypergraph dataset to clique-expanded graph dataset.")
    parser.add_argument(
        "--input",
        default="classification_hypergraph_dataset.pt",
        help="Path to the existing hypergraph dataset (.pt).",
    )
    parser.add_argument(
        "--output",
        default="classification_clique_dataset.pt",
        help="Path to save the clique-expanded dataset (.pt).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_clique_dataset(args.input, args.output)
