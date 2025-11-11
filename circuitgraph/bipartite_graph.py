import torch
from typing import Dict

try:
    from torch_geometric.data import Data
except Exception as e:
    raise RuntimeError("Please install torch-geometric: pip install torch-geometric") from e

from .instances import Inst

from circuitgraph.utils import initialize_global_variables
from circuitgraph.constants import EDGE_TYPES
import circuitgraph.constants as constants



def build_bipartite(insts: Dict[str, Inst], net_info: Dict[str, list], is_classification: bool) -> Data:
    """
    Nodes: instances + nets
    Edges: net <-> inst, with edge_attr one-hot over ['D','G','S','B','P']
    """
    initialize_global_variables(is_classification)
    
    def init_nodes():
        def onehots():
            h = {}
            variables = ['nmos', 'pmos'] + [item for item in constants.INSTANCE_TYPES if item != 'mos'] + ['signal', 'power', 'ground']
            for idx, var in enumerate(variables):
                vec = torch.zeros(len(variables), dtype=torch.float)
                vec[idx] = 1.0
                h[var] = vec
            return h

        enc = onehots()
        node_index = {}
        idx = 0

        for inst in insts.values():
            if inst.name not in node_index:
                if inst.kind == "mos":
                    oh = enc['nmos'] if 'nmos' in inst.model.lower() else enc['pmos']
                else:
                    oh = enc[inst.kind]
                node_index[inst.name] = [idx, oh.to(torch.float)]
                idx += 1

        for net_name in net_info.keys():
            if net_name not in node_index:
                if net_name.lower().startswith('vdd'):
                    oh = enc['power']
                elif net_name.lower().startswith('vss') or net_name.lower().startswith('gnd'):
                    oh = enc['ground']
                else:
                    oh = enc['signal']
                node_index[net_name] = [idx, oh.to(torch.float)]
                idx += 1
        return node_index

    def edge_type_vec(corresponding):
        v = torch.zeros(5, dtype=torch.float)
        for tp in corresponding:
            v[EDGE_TYPES.index(tp)] = 1.0
        assert v.sum() > 0
        return v

    node_info = init_nodes()
    edge_rows = []
    for net_name, (net_id, pinset) in net_info.items():
        types = {p for _, p in pinset if p in EDGE_TYPES}
        if not types:
            types.add('P')
        net_idx = node_info[net_name][0]
        for inst_name, _pin in pinset:
            inst_idx = node_info[inst_name][0]
            edge_rows.append([net_idx, inst_idx, edge_type_vec(types)])

    data = Data()
    data.x = torch.stack([x[1] for x in node_info.values()], dim=0)
    src = torch.tensor([[e[0] for e in edge_rows], [e[1] for e in edge_rows]], dtype=torch.long)
    rev = torch.tensor([[e[1] for e in edge_rows], [e[0] for e in edge_rows]], dtype=torch.long)
    data.edge_index = torch.cat((src, rev), dim=1)
    ea = torch.stack([e[2] for e in edge_rows], dim=0)
    data.edge_attr = torch.cat((ea, ea), dim=0)
    return data

