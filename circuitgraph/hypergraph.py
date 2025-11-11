import torch
from collections import defaultdict
from typing import Dict, List, Iterable, Tuple
from heapq import heappush, heappop

try:
    from torch_geometric.data import Data
except Exception as e:
    raise RuntimeError("Please install torch-geometric: pip install torch-geometric") from e

from circuitgraph.instances import Inst

from circuitgraph.regression_constants import REGRESSION_ALIAS_TO_CANON_DICT, REGRESSION_NODE_ATTRIBUTES

from circuitgraph.utils import identify_pmos, find_pg_pin, initialize_global_variables
import circuitgraph.constants as constants
from circuitgraph.constants import NODE_MOS, NODE_BJT


def build_net_graph(insts: Iterable[Inst]) -> Dict[str, set]:
    """
    Build an directed graph over nets: if an instance touches k nets,
    we connect all pairs of those k nets (hyperedge projection).
    """
    graph = defaultdict(set)
    reversed_edges = defaultdict(set)
    for inst in insts.values():
        nets = list({n for pin, n in inst.pins.items() if pin != 'B'})  # unique
        
        if inst.kind == "mos" or inst.kind == "bjt":
            # Direction: 
            #   nmos D/G -> S
            #   pmos S -> D/G    
            is_pmos = identify_pmos(inst.model)
            if is_pmos:
                graph[inst.pins['S']].add(inst.pins['D'])
                graph[inst.pins['S']].add(inst.pins['G'])
                graph[inst.pins['G']].add(inst.pins['D'])
                #graph[inst.pins['S']].add(inst.pins['B'])
                
                #graph[inst.pins['D']].add(inst.pins['S'])
                #graph[inst.pins['G']].add(inst.pins['S'])
                #graph[inst.pins['D']].add(inst.pins['G'])
                #graph[inst.pins['B']].add(inst.pins['S'])
                
                if inst.kind == "mos":
                    reversed_edges[inst.pins['S']].add(inst.pins['B'])
                    reversed_edges[inst.pins['B']].add(inst.pins['S'])
                    reversed_edges[inst.pins['G']].add(inst.pins['B'])
                    reversed_edges[inst.pins['D']].add(inst.pins['B'])
                    reversed_edges[inst.pins['B']].add(inst.pins['G'])
                    reversed_edges[inst.pins['B']].add(inst.pins['D'])
                    
                reversed_edges[inst.pins['G']].add(inst.pins['D'])
                reversed_edges[inst.pins['D']].add(inst.pins['G'])
                reversed_edges[inst.pins['D']].add(inst.pins['S'])
                reversed_edges[inst.pins['G']].add(inst.pins['S'])
            else:
                graph[inst.pins['D']].add(inst.pins['S'])
                graph[inst.pins['D']].add(inst.pins['G'])
                graph[inst.pins['G']].add(inst.pins['S'])
                #graph[inst.pins['B']].add(inst.pins['S'])
                
                #graph[inst.pins['S']].add(inst.pins['D'])
                #graph[inst.pins['S']].add(inst.pins['G'])
                #graph[inst.pins['G']].add(inst.pins['D'])
                #graph[inst.pins['S']].add(inst.pins['B'])
                if inst.kind == "mos":
                    reversed_edges[inst.pins['S']].add(inst.pins['B'])
                    reversed_edges[inst.pins['B']].add(inst.pins['S'])
                    
                    reversed_edges[inst.pins['G']].add(inst.pins['B'])
                    reversed_edges[inst.pins['D']].add(inst.pins['B'])
                    reversed_edges[inst.pins['B']].add(inst.pins['G'])
                    reversed_edges[inst.pins['B']].add(inst.pins['D'])
                    
                reversed_edges[inst.pins['G']].add(inst.pins['D'])
                reversed_edges[inst.pins['D']].add(inst.pins['G'])
                reversed_edges[inst.pins['S']].add(inst.pins['D'])
                reversed_edges[inst.pins['S']].add(inst.pins['G'])
        else:
            # connect all pairs (clique on nets touched by this inst)
            for i in range(len(nets)):
                for j in range(i+1, len(nets)):
                    a, b = nets[i], nets[j]
                    graph[a].add(b)
                    graph[b].add(a)
   
        # ensure isolated single-net appearances exist in graph
        for n in nets:
            graph.setdefault(n, graph[n])
    return graph, reversed_edges


def _dist_validation_test(dist, net_info):
    for v in dist.values():
        if v == float('inf'):
            return False
    if not len(dist) == len(net_info):
        return False
    return True

# graph: dict[net] -> list[(neighbor_net, weight)]
def dijkstra_from_pg(
    graph: Dict[str, List[str]],
    vdd_nets: Iterable[str],
    vss_nets: Iterable[str],
    is_from_vdd: bool,
    prev_dist = None
) -> Dict[str, float]:
    # 1) shortest distances from any VDD source
    dist = {n: float('inf') for n in graph} if prev_dist is None else prev_dist
    for n in graph:
        if n not in dist:
            dist[n] = float('inf')
    
    sources = vdd_nets if is_from_vdd else vss_nets
    
    pq = []
    for s in sources:
        if s in graph:
            dist[s] = 0
            heappush(pq, (0, s))

    other_pg_net = find_pg_pin(graph, constants.GROUND_NETS) if is_from_vdd else find_pg_pin(graph, constants.POWER_NETS)
    while pq:
        du, u = heappop(pq)
        if du != dist[u]:
            continue  # stale
        if u == other_pg_net: # will caculate at the last part
            continue
        for adj in graph[u]:
            if adj == other_pg_net:
                continue
            nd = du + 1  # unit weight
            if nd < dist.get(adj, float('inf')):
                dist[adj] = nd
                heappush(pq, (nd, adj))
                
            elif nd == dist.get(adj, float('inf')) and prev_dist is not None:
                heappush(pq, (nd, adj))

    
    # 2) assign VSS nets to "farthest from VDD"
    #    i.e., use the maximum finite shortest-path distance
    #if (prev_dist is None and dist[other_pg_net] != float('inf')) or dist[other_pg_net] == float('inf'):
    if other_pg_net is not None:
        if prev_dist is None or dist[other_pg_net] == float('inf'):
            finite_dists = [dist[node] for node in graph[other_pg_net] if dist[node] < float('inf')]
            farthest = max(finite_dists) if finite_dists else float('inf')
            dist[other_pg_net] = farthest + 1

    return dist


def calculate_net_distance(graph: Dict[str, set], reversed_edges: Dict[str, set], net_info,
                     vdd_nets: Iterable[str], vss_nets: Iterable[str], is_from_vdd: bool) -> Dict[str, int]:
    dist = dijkstra_from_pg(graph, vdd_nets, vss_nets, is_from_vdd)
    
    if not _dist_validation_test(dist, net_info):
        undirected_graph = graph
        for net, connected_nets in reversed_edges.items():
            for connected_net in connected_nets:
                undirected_graph[net].add(connected_net)
        dist = dijkstra_from_pg(undirected_graph, vdd_nets, vss_nets, is_from_vdd, dist)
    
    if not _dist_validation_test(dist, net_info):
        return None
        
    return dist



def device_distance(inst: Inst, net_dist: Dict[str, int]) -> float:
    """
    Device distance = min distance over its connected nets.
    Returns inf if none of its nets are in the net distance map.
    """
    dists = [net_dist.get(n, float('inf')) for n in inst.pins.values()]
    return min(dists) if dists else float('inf')

def distance_to_power_ground(
    insts: Iterable[Inst], net_info
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Compute:
      - net_distance: distance of each net to the nearest VDD net
      - inst_distance: distance of each instance (min over its nets)
    """
    graph, reversed_edges = build_net_graph(insts)
    net_distance = calculate_net_distance(graph, reversed_edges, net_info, constants.POWER_NETS, constants.GROUND_NETS, is_from_vdd = True)
    if net_distance is None:
        net_distance = calculate_net_distance(graph, reversed_edges, net_info, constants.POWER_NETS, constants.GROUND_NETS, is_from_vdd = False)
        
    
    # Adjust distance between p/g
    '''
    power_pin = find_pg_pin(graph, vdd_nets)
    ground_pin = find_pg_pin(graph, vss_nets)
    if vdd_net_distance[ground_pin] != vss_net_distance[power_pin]:
        assert vdd_net_distance[ground_pin] == vss_net_distance[power_pin]
        pg_distance = min(vdd_net_distance[ground_pin], vss_net_distance[power_pin])
        vdd_net_distance[ground_pin] = pg_distance
        vss_net_distance[power_pin] = pg_distance
        
        for k, v in vdd_net_distance.items():
            if not k == ground_pin:
                assert v <= vdd_net_distance[ground_pin] 
        for k, v in vss_net_distance.items():
            if not k == power_pin:
                assert v <= vss_net_distance[power_pin] 
    '''
    
    #vdd_inst_distance = {inst.name: device_distance(inst, vdd_net_distance) for inst in insts.values()}
    #vss_inst_distance = {inst.name: device_distance(inst, vss_net_distance) for inst in insts.values()}
    
    inst_distance = {inst.name: device_distance(inst, net_distance) for inst in insts.values()}
    return net_distance, inst_distance
    #return vdd_net_distance, vss_net_distance, vdd_inst_distance, vss_inst_distance
    #return vdd_net_distance, None, vdd_inst_distance, None



def get_node_distance(node_kind: str, inst: Inst, is_pmos: bool,
                         inst_distance: Dict[str, int], net_distance: Dict[str, int],
                         is_from_power: bool
                         ) -> int:
    
    distance = inst_distance[inst.name]
    if node_kind in constants.NODE_MOS or node_kind in constants.NODE_BJT:
        # Find pin with reference distance
        reference_pin = None
        for pin, net in inst.pins.items():
            if net_distance[net] == distance:
                reference_pin = pin
                break
        assert reference_pin is not None
        
        
        if not reference_pin in node_kind:
            other_pin_distance = [net_distance[inst.pins[node]] for node in list(node_kind)]
            
            smaller_distance = min(other_pin_distance)
            longer_distance = max(other_pin_distance)
            
            distance = smaller_distance
        
        '''
        move_unit = 1 if not is_pmos else -1
        move_unit = move_unit if is_from_power else -1 * move_unit
        
        if not reference_pin in node_kind:
            if reference_pin == "G" or reference_pin == "D":
                distance += move_unit # GD -> SB going down
            elif reference_pin == "S":
                distance -= move_unit # S -> GD going up
            elif reference_pin == "B":
                if "S" in node_kind:
                    distance -= move_unit # B -> S going up
                else:
                    distance -= move_unit * 2 # B -> GD going up twice             
        '''     
    return distance
'''
def get_node_distance_from_power_ground(node_kind: str, inst: Inst, is_pmos: bool,
                         vdd_inst_distance: Dict[str, int], vdd_net_distance: Dict[str, int],
                         vss_inst_distance: Dict[str, int], vss_net_distance: Dict[str, int],
                         ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    
    distance_from_vdd = get_node_distance(node_kind, inst, is_pmos, vdd_inst_distance, vdd_net_distance, is_from_power = True)
    #distance_from_vss = get_node_distance(node_kind, inst, is_pmos, vss_inst_distance, vss_net_distance, is_from_power = False)
    
    power_pin = find_pg_pin(vdd_net_distance, POWER_NETS)
    ground_pin = find_pg_pin(vdd_net_distance, GROUND_NETS)
    #vdd_distance = vss_net_distance[power_pin]
    vss_distance = vdd_net_distance[ground_pin]
    
    #assert vdd_distance == vss_distance
    
    #assert all(0 <= x <= float('inf') for x in [distance_from_vdd, distance_from_vss, vdd_distance, vss_distance])
    
    #return torch.tensor([distance_from_vdd]), torch.tensor([distance_from_vss]), torch.tensor([vdd_distance]), torch.tensor([vss_distance])
    return torch.tensor([distance_from_vdd]), None, None, torch.tensor([vss_distance])
'''

def get_node_distance_from_power_ground(node_kind: str, inst: Inst, is_pmos: bool,
                         inst_distance: Dict[str, int], net_distance: Dict[str, int],
                         ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    distance_from_vdd = get_node_distance(node_kind, inst, is_pmos, inst_distance, net_distance, is_from_power = True)
    return torch.tensor([distance_from_vdd])


def incidence_to_hypergraph_data(incidence_matrix: torch.Tensor,
                                 node_info: Dict[str, List],
                                 #net_features_lookup: Dict[Tuple[int, int], torch.tensor]
                                 hyperedge_features: torch.Tensor
                                 ) -> Data:
    H = incidence_matrix
    assert H.dim() == 2, "H must be 2-D [N x M]"
    N, M = H.size()

    if H.is_sparse:
        if H.layout == torch.sparse_coo:
            H = H.coalesce()
            node_idx, he_idx = H.indices()
        elif H.layout == torch.sparse_csr:
            H = H.to_sparse_coo().coalesce()
            node_idx, he_idx = H.indices()
        else:
            dense = H.to_dense()
            nz = (dense != 0)
            node_idx, he_idx = nz.nonzero(as_tuple=True)
    else:
        nz = (H != 0)
        node_idx, he_idx = nz.nonzero(as_tuple=True)

    data = Data()
    data.x = torch.stack([x[1] for x in node_info.values()], dim=0)
    data.edges = None
    data.num_nodes = N
    data.num_hyperedges = M
    
    data.hyperedge_index = torch.stack([node_idx.long(), he_idx.long()], dim=0)
    data.incidence_matrix = incidence_matrix
    #data.num_hyperedges = data.hyperedge_index.shape[1]
    
    #edge_attr_list = []
    #for n, e in zip(node_idx.tolist(), he_idx.tolist()):
    #    feat = net_features_lookup[(n, e)]
    #    edge_attr_list.append(feat)
    #data.hyperedge_attr = torch.stack(edge_attr_list, dim=0)
    
    data.hyperedge_attr = hyperedge_features
    return data

def build_hypergraph(insts: Dict[str, Inst], net_info: Dict[str, list], is_classification: bool) -> Data:
    """
    Vertices: MOS terminals (GS/GD/DS/SB) and passives; Hyperedges: nets
    """
    initialize_global_variables(is_classification)
    power_pin = find_pg_pin(net_info, constants.POWER_NETS)
    ground_pin = find_pg_pin(net_info, constants.GROUND_NETS)
    
    net_pg_distance, inst_pg_distance = distance_to_power_ground(insts, net_info)
    if power_pin in net_info:
        farthest = net_pg_distance[ground_pin]
    else:
        farthest = max([distance for distance in net_pg_distance.values()])
        net_pg_distance = {k:farthest - v for k,v in net_pg_distance.items()}
        inst_pg_distance = {k:farthest - v for k,v in inst_pg_distance.items()}
    
    def initialize_node():
        def kind_one_hots():
            h = {}
            variables = NODE_MOS + [item for item in constants.INSTANCE_TYPES if item != 'mos']
            for idx, var in enumerate(variables):
                vec = torch.zeros(len(variables), dtype=torch.float)
                vec[idx] = 1.0
                h[var] = vec
            return h
            
        onehots = kind_one_hots() 
        
        def get_node_features(node_kind, is_pmos, inst):
            if isinstance(is_pmos, bool):
                pmos_nmos = torch.tensor([1, 0]) if is_pmos else torch.tensor([0, 1])
            else:
                pmos_nmos = torch.tensor([0, 0])
                
            node_feature = torch.cat((onehots[node_kind], pmos_nmos)).to(torch.float)
            if inst.attrs is not None and not is_classification:
                node_attr = torch.zeros(len(REGRESSION_NODE_ATTRIBUTES), dtype=torch.float)
                for k, v in inst.attrs.items():
                    if k in REGRESSION_ALIAS_TO_CANON_DICT:
                        canon_k = REGRESSION_ALIAS_TO_CANON_DICT[k]
                    else:
                        canon_k = k
                    if canon_k in REGRESSION_NODE_ATTRIBUTES:
                        node_attr[REGRESSION_NODE_ATTRIBUTES.index(canon_k)] = float(v)
                node_feature = torch.cat(
                    (node_feature,
                    node_attr
                    )
                )
                
                
            #distance_from_vdd, distance_from_vss, vdd_distance, vss_distance = get_node_distance_from_power_ground(
            #    node_kind, inst, is_pmos, vdd_inst_distance, vdd_net_distance, vss_inst_distance, vss_net_distance
            #)

            distance_from_vdd = get_node_distance(node_kind, inst, is_pmos, inst_pg_distance, net_pg_distance, True)
            distance_from_vdd = torch.tensor([distance_from_vdd])
            vss_distance = torch.tensor([farthest])
            
            node_feature = torch.cat(
                (node_feature,
                 distance_from_vdd,
                 vss_distance - distance_from_vdd,
                 #distance_from_vss,
                 #vdd_distance,
                 vss_distance
                 )
            )
            return node_feature
            

        node_index = {}
        idx = 0
        for inst in insts.values():
            if inst.kind == "mos":
                is_pmos = identify_pmos(inst.model)
                for r in NODE_MOS:
                    node_name = f"{inst.name}_{r}"
                    if node_name not in node_index:
                        node_feature = get_node_features(r, is_pmos, inst)
                        node_index[node_name] = [idx, node_feature]
                        idx += 1
            elif inst.kind == "bjt":
                assert is_classification
                is_pmos = identify_pmos(inst.model)
                for r in NODE_BJT:
                    node_name = f"{inst.name}_{r}"
                    if node_name not in node_index:
                        node_feature = get_node_features(r, is_pmos, inst)
                        node_feature[len(NODE_MOS) + constants.INSTANCE_TYPES.index("bjt") - 1] = 1
                        node_index[node_name] = [idx, node_feature]
                        idx += 1
            else:
                if inst.name not in node_index:
                    node_feature = get_node_features(inst.kind, None, inst)
                    node_index[inst.name] = [idx, node_feature]
                    idx += 1
        return node_index

    def get_net_features(net):
        # return one hot vector (IS_SIGNAL, IS_POWER, IS_GROUND)
        net_distance = net_pg_distance[net]
        vdd_vss_distance = farthest # Need to check.
        
        if net in constants.POWER_NETS:
            return torch.tensor([0, 1, 0, net_distance, vdd_vss_distance - net_distance, vdd_vss_distance])
        elif net in constants.GROUND_NETS:
            return torch.tensor([0, 0, 1, net_distance, vdd_vss_distance - net_distance, vdd_vss_distance])
        else:
            return torch.tensor([1, 0, 0, net_distance, vdd_vss_distance - net_distance, vdd_vss_distance])

    node_info = initialize_node()
    incidence = torch.zeros([len(node_info), len(net_info)])
    hyperedge_features = {}
    #net_features_lookup = {}
    for inst in insts.values():
        if inst.kind == "mos":
            for r in NODE_MOS:
                nidx = node_info[f"{inst.name}_{r}"][0]
                for terminal in r:  # e.g., 'G','S' for "GS"
                    net = inst.pins[terminal]
                    eidx = net_info[net][0]
                    incidence[nidx][eidx] = 1
                    hyperedge_features[eidx] = get_net_features(net)
                    #net_features_lookup[(nidx, eidx)] = get_net_features(net)
                    
        elif inst.kind == "bjt":
            for r in NODE_BJT:
                nidx = node_info[f"{inst.name}_{r}"][0]
                for terminal in r:  # e.g., 'G','S' for "GS"
                    net = inst.pins[terminal]
                    eidx = net_info[net][0]
                    incidence[nidx][eidx] = 1
                    hyperedge_features[eidx] = get_net_features(net)
                    #net_features_lookup[(nidx, eidx)] = get_net_features(net)
        else:
            nidx = node_info[inst.name][0]
            for net in inst.pins.values():
                eidx = net_info[net][0]
                incidence[nidx][eidx] = 1
                hyperedge_features[eidx] = get_net_features(net)
                #net_features_lookup[(nidx, eidx)] = get_net_features(net)
    hyperedge_features = torch.stack(
        [torch.tensor(v) for _, v in sorted(hyperedge_features.items(), key = lambda x:x[0])], dim=0)
    
    
    return incidence_to_hypergraph_data(incidence, node_info, hyperedge_features)