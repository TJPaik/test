import os
import torch
from tqdm.auto import tqdm

from circuitgraph.netlist_parser import parse_spectre_netlist

from circuitgraph.hypergraph import build_hypergraph
from circuitgraph.bipartite_graph import build_bipartite

from circuitgraph.classification_constants import LABELS, RANGES
from circuitgraph.utils import update_i_source, collect_nets

# ---------- labels ----------
def get_label_from_data_index(data_index: int):
    """
    Returns 10-dim one-hot (or None) using your provided ranges.
    """
    NUM_CLASSES = len(LABELS)
    for s, e, k in RANGES:
        if s <= data_index <= e:
            one_hot = torch.zeros(NUM_CLASSES, dtype=int)
            one_hot[LABELS[k]] = 1
            return one_hot
    return None

# ---------- dataset builders ----------
def create_classification_dataset(dataset_dir: str,
                                  hypergraph_dataset_path: str = 'classification_hypergraph_dataset.pt',
                                  bipartite_dataset_path: str = 'classification_bipartite_dataset.pt'):
    hypergraph_list, bipartite_list = [], []

    for i in tqdm(sorted(os.listdir(dataset_dir)), desc="Building classification dataset", unit="circuit"):
        dpath = os.path.join(dataset_dir, i)
        if not os.path.isdir(dpath) or not i.isdigit():
            continue
        label = get_label_from_data_index(int(i))
        if label is None:
            continue

        # pick netlist file
        files = os.listdir(dpath)
        cir_path = None
        for x in files:
            if 'full' in x:
                cir_path = os.path.join(dpath, x)
                break
        if cir_path is None:
            cir_path = os.path.join(dpath, f"{i}.cir")

        insts = parse_spectre_netlist(cir_path)
        insts = update_i_source(insts)
        net_info = collect_nets(insts)
        #if not ("VDD" in net_info and "VSS" in net_info):
        #    print(f"[Warning] {i} does not have either VDD or VSS.")
        #    continue
        
        hypergraph        = build_hypergraph(insts, net_info, is_classification = True)
        hypergraph.y = label
        
        bipartite_graph   = build_bipartite(insts, net_info, is_classification = True)
        bipartite_graph.y   = label

        hypergraph_list.append(hypergraph)
        bipartite_list.append(bipartite_graph)

    torch.save(hypergraph_list, hypergraph_dataset_path)
    torch.save(bipartite_list,  bipartite_dataset_path)
