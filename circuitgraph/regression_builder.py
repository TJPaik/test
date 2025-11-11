import os
import csv
import copy
import torch
import ast, operator, re
from typing import Dict, Any
from tqdm.auto import tqdm

from circuitgraph.instances import Inst
from circuitgraph.netlist_parser import parse_spectre_netlist
from circuitgraph.hypergraph import build_hypergraph 
from circuitgraph.bipartite_graph import build_bipartite

from circuitgraph.regression_constants import TARGET_LIST, ALIAS_TO_CANON, CANON_TARGETS
from circuitgraph.utils import collect_nets

# ---- 1) Unit table (Spectre/SPICE style) ----
_UNITS = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,    # milli (note: Mega is 'meg' not 'M' here)
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9,
    "t": 1e12,
}
# Accept case-insensitive units
_UNITS = {k.lower(): v for k, v in _UNITS.items()}

_num_unit_re = re.compile(
    r"""(?x)
    (?P<num>        # number part
        (?:\d+\.\d*|\.\d+|\d+)
        (?:[eE][+\-]?\d+)?   # optional exponent
    )
    (?P<unit>[A-Za-z]+)      # trailing unit letters (no digits)
    """
)

_ident_re = re.compile(r"\b([A-Za-z_]\w*)\b")

# ---- 2) Safe arithmetic evaluator ----
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def _safe_eval(parsed):
    """Recursively evaluate a sanitized AST Expression containing only numbers/ops."""
    if isinstance(parsed, ast.Expression):
        return _safe_eval(parsed.body)
    if isinstance(parsed, ast.Constant):   # py3.8+: numbers show as Constant
        if isinstance(parsed.value, (int, float)):
            return float(parsed.value)
        raise ValueError("Non-numeric constant found.")
    if isinstance(parsed, ast.Num):        # for older Python
        return float(parsed.n)
    if isinstance(parsed, ast.BinOp) and type(parsed.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(parsed.op)](
            _safe_eval(parsed.left), _safe_eval(parsed.right)
        )
    if isinstance(parsed, ast.UnaryOp) and type(parsed.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(parsed.op)](_safe_eval(parsed.operand))
    if isinstance(parsed, ast.Paren):
        return _safe_eval(parsed.value)
    # No names, calls, attrs, etc. allowed at this stage
    raise ValueError(f"Disallowed expression node: {type(parsed).__name__}")

# ---- 3) Helpers to normalize an expression string ----
def _apply_units(expr: str) -> str:
    """Turn literals like '45n' or '3.3k' into '(45*1e-9)' / '(3.3*1e3)'."""
    def repl(m):
        num = m.group("num")
        unit = m.group("unit").lower()
        if unit not in _UNITS:
            # If it's not a recognized unit, leave as-is (e.g., model name/param id)
            return m.group(0)
        return f"({num}*{_UNITS[unit]:.16g})"
    return _num_unit_re.sub(repl, expr)

def _substitute_idents(expr: str, params_numeric: Dict[str, float], variable_dict: Dict[str, float]) -> str:
    """Replace identifiers with numeric values (post-resolved)."""
    def repl(m):
        name = m.group(1)
        # leave Python keywords/operators alone (none match ident regex anyway)
        if name in params_numeric:
            return f"({params_numeric[name]:.16g})"
        elif name in variable_dict:
            return f"({variable_dict[name]:.16g})"
        return name  # keep unknown words (should not remain by eval time)
    return _ident_re.sub(repl, expr)

def _to_float(expr: str) -> float:
    parsed = ast.parse(expr, mode="eval")
    return float(_safe_eval(parsed))

# ---- 4) Resolve param_dict first (in case params depend on other params) ----
def resolve_params(param_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    param_dict can have numbers or strings with units/expressions.
    Returns all as float after resolving dependencies.
    """
    resolved: Dict[str, float] = {}
    visiting = set()

    def resolve_one(k: str) -> float:
        if k in resolved:
            return resolved[k]
        if k in visiting:
            raise ValueError(f"Circular dependency in param_dict at '{k}'")
        visiting.add(k)

        v = param_dict[k]
        if isinstance(v, (int, float)):
            out = float(v)
        else:
            # string: may contain idents/units/ops
            expr = str(v)
            expr = _apply_units(expr)
            # Substitute only the idents we can resolve now:
            def id_repl(m):
                name = m.group(1)
                if name == k:
                    # self-ref would be a cycle
                    return name
                if name in param_dict:
                    val = resolve_one(name)  # recursively resolve dependency
                    return f"({val:.16g})"
                return name
            expr = _ident_re.sub(id_repl, expr)
            out = _to_float(expr)

        visiting.remove(k)
        resolved[k] = out
        return out

    for key in list(param_dict.keys()):
        resolve_one(key)
    return resolved

# ---- 5) Public function: evaluate attrs dict to floats ----
def evaluate_attrs(attrs: Dict[str, str], param_dict: Dict[str, Any], variable_dict: Dict[str, float]) -> Dict[str, float]:
    """
    For each key in attrs (e.g., 'as', 'ad', 'ps', 'pd', 'w', 'l', ...),
    substitute parameters + units and compute a float.
    """
    # First resolve all params to numeric floats
    param_float_dict = resolve_params(param_dict)

    out: Dict[str, float] = {}
    for k, raw in attrs.items():
        if k == 'region' or k == 'type' or k == 'fq' or k == 'q':
            # if k == 'region' all the raw values are 'sat'.
            # if k == 'type' all the raw values are 'dc'.
            # if k == 'fq' all the raw values is 'fin'.
            # if k == 'fq' all the raw values is 'q'.
            continue 
        if raw in param_float_dict:
            out[k] = param_float_dict[raw]
        elif raw in variable_dict:
            out[k] = variable_dict[raw]
        else:
            expr = str(raw)
            expr = _apply_units(expr)              # turn 45n -> (45*1e-9)
            expr = _substitute_idents(expr, param_float_dict, variable_dict)    # WN1 -> (3.0)
            val = _to_float(expr)                  # safe-eval
            out[k] = val
    return out

def _update_insts_with_param(insts: Dict[str, Inst], param_dict: Dict[str, float], variable_dict: Dict[str, float]) -> Dict[str, Inst]:
    updated = copy.deepcopy(insts)
    for inst in updated.values():
        numeric_attrs = evaluate_attrs(inst.attrs, param_dict, variable_dict)
        inst.attrs = numeric_attrs
        
    return updated

def parse_simulation_script(simulation_script_path: str) -> Dict[str, float]:
    variable_dict = {}
    with open(simulation_script_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        toks = line.split()
        if not toks:
            continue
        if toks[0].startswith("desVar"):
            variable_dict[toks[1].strip("\"")] = _to_float(_apply_units(toks[2]))
    return variable_dict


def set_design_path_name(path, design_name):
    dirpath, filename = os.path.split(path)
    basename, ext = os.path.splitext(filename)
    new_filename = f"{basename}_{design_name}{ext}"
    return os.path.join(dirpath, new_filename)

def create_regression_dataset(dataset_dir: str,
                              hypergraph_dataset_path: str = 'regression_hypergraph_dataset.pt',
                              bipartite_dataset_path: str = 'regression_bipartite_dataset.pt'):
    """
    Skeleton for AICircuit: walks CSVs & netlists so you can attach target tensors.
    """
    
    data_dir = os.path.join(dataset_dir, "Dataset")
    netlist_dir = os.path.join(dataset_dir, "Simulation", "Netlists")
    simulation_dir = os.path.join(dataset_dir, "Simulation", "Ocean")
    netlists = sorted(os.listdir(netlist_dir))
    
    for name in tqdm(netlists, desc="Building regression dataset", unit="ckt"):
        
        netlist_path = os.path.join(netlist_dir, name, "netlist")
        csv_path     = os.path.join(data_dir, name, f"{name}.csv")
        simulation_path = os.path.join(simulation_dir, name, "oceanScript.ocn")
        if not (os.path.isfile(netlist_path) and os.path.isfile(csv_path)):
            continue

        base_insts = parse_spectre_netlist(netlist_path)
        variable_dict = parse_simulation_script(simulation_path)
        net_info = collect_nets(base_insts)

        hypergraph_list, bipartite_list = [], []
        
        design_hypergraph_dataset_path = set_design_path_name(hypergraph_dataset_path, name)
        design_bipartite_dataset_path = set_design_path_name(bipartite_dataset_path, name)
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
            
            for row in tqdm(rows, desc=f"  ↳ {name} rows", unit="row", leave=False):
                params = {k: v for k, v in zip(header, row)}
                insts = _update_insts_with_param(base_insts, params, variable_dict)
                
                '''
                specifications = torch.zeros(len(CANON_TARGETS), dtype=torch.float)
                for k, v in params.items():
                    if k in TARGET_LIST:
                        canon_k = ALIAS_TO_CANON.get(k, k)
                        assert canon_k in CANON_TARGETS
                        idx = CANON_TARGETS.index(canon_k)
                        specifications[idx] = float(v)
                '''
                specifications = []
                for k, v in params.items():
                    if k in TARGET_LIST:
                        canon_k = ALIAS_TO_CANON.get(k, k)
                        assert canon_k in CANON_TARGETS
                        specifications.append(float(v))
                specifications = torch.tensor(specifications).to(torch.float)
                
                hypergraph        = build_hypergraph(insts, net_info, is_classification = False); 
                hypergraph.y      = specifications 
                
                bipartite_graph   = build_bipartite(insts, net_info, is_classification = False);  
                bipartite_graph.y = specifications
                
                hypergraph_list.append(hypergraph)
                bipartite_list.append(bipartite_graph)
            


        torch.save(hypergraph_list, design_hypergraph_dataset_path)
        torch.save(bipartite_list,  design_bipartite_dataset_path)