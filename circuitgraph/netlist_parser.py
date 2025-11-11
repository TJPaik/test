import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from circuitgraph.instances import Inst

# ---------- helpers (kept local to avoid over-splitting) ----------
def _parse_param_file(param_path: str) -> dict:
    unit_scale = {
        'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'meg': 1e6, 'g': 1e9, '': 1.0
    }
    def parse_value(val_str: str) -> float:
        m = re.fullmatch(r'([\d\.]+)([a-zA-Z]*)', val_str)
        if not m:
            raise ValueError(f"Cannot parse: {val_str}")
        number, suffix = m.groups()
        return float(number) * unit_scale.get(suffix.lower(), 1.0)

    params = {}
    with open(param_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(('*', '//', ';')):
                continue
            for item in line.split():
                if '=' in item:
                    k, v = item.split('=', 1)
                    params[k.strip()] = parse_value(v.strip())
    return params

# ---------- SPICE (X/MOS + basic R/C/I) ----------
def parse_spice_netlist(netlist_path: str, param_path: str) -> Tuple[Dict[str, Inst], List[str]]:
    MOS_RE     = re.compile(r"^[xX]\w+")
    SUBCKT_RE  = re.compile(r"^\s*\.subckt\s+", re.IGNORECASE)
    ENDS_RE    = re.compile(r"^\s*\.ends\b", re.IGNORECASE)

    param_dict = _parse_param_file(param_path)

    def calculate_value(v_str: str) -> float:
        v_str = v_str.strip("'\"")
        if "*" in v_str:
            comp1, comp2 = v_str.split('*', 1)
            if comp1 in param_dict:
                base, mult = comp1, comp2
            elif comp2 in param_dict:
                base, mult = comp2, comp1
            else:
                raise ValueError(f"Cannot parse param expression: {v_str}")
            return param_dict[base] * float(mult)
        elif v_str in param_dict:
            return param_dict[v_str]
        else:
            return float(v_str)

    insts: Dict[str, Inst] = {}
    ports: List[str] = []

    with open(netlist_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith(('*', '//', ';')):
            continue
        if ENDS_RE.match(line):
            continue
        elif SUBCKT_RE.match(line):
            toks = line.split()
            ports = toks[2:] if len(toks) >= 3 else []
            continue

        if MOS_RE.match(line):
            toks = line.split()
            name = toks[0]
            if len(toks) < 6:
                raise ValueError(f"MOS line malformed: {line}")
            d, g, s, b = toks[1], toks[2], toks[3], toks[4]
            model = toks[5]
            params = {}
            for t in toks[6:]:
                if '=' in t:
                    k, v = t.split('=', 1)
                    params[k] = calculate_value(v)
            insts[name] = Inst(
                name=name, kind="mos", model=model,
                pins={"D": d, "G": g, "S": s, "B": b},
                attrs=params
            )
        else:
            toks = line.split()
            name = toks[0]
            n1, n2 = toks[1], toks[2]
            if name[0] in ("I", "i"):
                model = "CUR"
            elif name[0] in ("C", "c"):
                model = "CAP"
            elif name[0] in ("R", "r"):
                model = "RES"
            else:
                raise ValueError(f"Unrecognized instance line: {line}")
            params = {"value": calculate_value(toks[3])}
            insts[name] = Inst(
                name=name, kind=model.lower(), model=model,
                pins={"P1": n1, "P2": n2}, attrs=params
            )
    return insts, ports

# ---------- Spectre (with subckt flattening) ----------
def parse_spectre_netlist(netlist_path: str) -> Dict[str, Inst]:
    SUBCKT_RE = re.compile(r"^\s*subckt\s+(\w+)\b(.*)$", re.IGNORECASE)
    ENDS_RE   = re.compile(r"^\s*ends\b(?:\s+\w+)?\s*$", re.IGNORECASE)
    COMMENT_LINE_RE = re.compile(r"^\s*(\*|//|;)")
    INLINE_COMMENT_RE = re.compile(r"//.*$")
    
    IS_REGRESSION_DATASET = True if "AICircuit" in netlist_path else False

    def join_continuations(lines: List[str]) -> List[str]:
        joined, buf = [], ""
        for raw in lines:
            line = INLINE_COMMENT_RE.sub("", raw.rstrip("\n"))
            if not buf:
                buf = line
            else:
                buf += " " + line.lstrip()
            if buf.rstrip().endswith("\\"):
                buf = buf.rstrip()[:-1]
                continue
            joined.append(buf.strip())
            buf = ""
        if buf.strip():
            joined.append(buf.strip())
        return joined

    ATTR_RE = re.compile(r"(\w+)\s*=\s*([^=\s][^=]*?)(?=\s+\w+\s*=|$)")
    def parse_attrs(attr_str: str) -> Optional[Dict[str, str]]:
        attrs = {}
        for m in ATTR_RE.finditer(attr_str):
            k = m.group(1)
            v = m.group(2).strip()
            attrs[k] = v
        return attrs or None

    def split_pins_model_and_attrs(line: str):
        lp = line.find("(")
        rp = line.find(")", lp + 1) if lp != -1 else -1
        if lp == -1 or rp == -1:
            toks = line.split()
            pins = []
            model = toks[1] if len(toks) > 1 else ""
            attr_str = line.split(model, 1)[1] if model and model in line else ""
            return pins, model, attr_str
        pins_str = line[lp+1:rp].strip()
        after_rparen = line[rp+1:].strip()
        if not after_rparen:
            return [], "", ""
        model_and_rest = after_rparen.split(None, 1)
        model = model_and_rest[0]
        attr_str = model_and_rest[1] if len(model_and_rest) > 1 else ""
        pins = pins_str.split()
        return pins, model, attr_str

    def is_mos_model(model: str) -> bool:
        m = model.lower()
        return ("mos" in m) or (m in {"nfet", "pfet"})
    
    def is_bjt_model(model: str) -> bool:
        m = model.lower()
        return ("npn" in m) or ("pnp" in m)

    with open(netlist_path, "r") as f:
        raw_lines = f.readlines()
    filtered = [ln for ln in raw_lines if not COMMENT_LINE_RE.match(ln)]
    lines = join_continuations(filtered)

    insts: Dict[str, Inst] = {}
    internal_subckts: Dict[str, Tuple[Dict[str, Inst], List[str]]] = {}
    subckt_active = False
    subckt_name = None
    subckt_ports: List[str] = []
    subckt_insts: Dict[str, Inst] = {}
    from collections import defaultdict as _dd
    instance_index_dict = _dd(int)

    for line in lines:
        if not line:
            continue
        if ENDS_RE.match(line):
            if subckt_active and subckt_name is not None:
                internal_subckts[subckt_name] = (subckt_insts, subckt_ports)
            subckt_active = False
            subckt_name = None
            subckt_ports = []
            subckt_insts = {}
            continue

        m = SUBCKT_RE.match(line)
        if m:
            subckt_active = True
            subckt_name = m.group(1)
            rest = (m.group(2) or "").strip()
            subckt_ports = rest.split() if rest else []
            subckt_insts = {}
            continue

        toks = line.split()
        if not toks:
            continue

        raw_inst_name = toks[0]
        name_prefix = raw_inst_name[0]
        name = f"{name_prefix}{instance_index_dict[name_prefix]}"
        instance_index_dict[name_prefix] += 1

        pins_list, model, attr_str = split_pins_model_and_attrs(line)

        if is_mos_model(model) and len(pins_list) >= 4:
            pins = {"D": pins_list[0], "G": pins_list[1], "S": pins_list[2], "B": pins_list[3]}
            kind = "mos"
        elif is_bjt_model(model) and len(pins_list) >= 4:
            pins = {"D": pins_list[0], "G": pins_list[1], "S": pins_list[2]}
            kind = "bjt"
        else:
            pins = {f"P{i}": net for i, net in enumerate(pins_list)}
            kind = model.lower()
        
        
        
        if IS_REGRESSION_DATASET:
            # Skip the followings
            if kind == 'mutual_inductor' or kind == "balun" or kind == "port":
                continue
        
        attrs = parse_attrs(attr_str) if attr_str else None

        if model in internal_subckts:
            child_insts, child_ports = internal_subckts[model]
            port_map = {port: pins.get(f"P{i}", f"P{i}") for i, port in enumerate(child_ports)}
            new_insts = {}
            for child in child_insts.values():
                qname = f"{name}_{child.name}"
                if qname in new_insts:
                    raise AssertionError(f"Duplicate instance name: {qname}")
                remapped_pins = {role: port_map.get(net, net) for role, net in child.pins.items()}
                new_insts[qname] = Inst(
                    name=qname, kind=child.kind, model=child.model,
                    pins=remapped_pins, attrs=child.attrs
                )
            if subckt_active:
                subckt_insts.update(new_insts)
            else:
                insts.update(new_insts)
            continue

        target_dict = subckt_insts if subckt_active else insts
        if name in target_dict:
            raise AssertionError(f"Duplicate instance name: {name}")
        target_dict[name] = Inst(name=name, kind=kind, model=model, pins=pins, attrs=attrs)

    return insts
