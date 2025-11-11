
from typing import Dict
from circuitgraph.instances import Inst

from circuitgraph.classification_constants import CLASSIFICATION_POWER_NETS, CLASSIFICATION_GROUND_NETS, CLASSIFICATION_INSTANCE_TYPES
from circuitgraph.regression_constants import REGRESSION_POWER_NETS, REGRESSION_GROUND_NETS, REGRESSION_INSTANCE_TYPES
import circuitgraph.constants as constants

# ---------- nets ----------
def collect_nets(insts: Dict[str, Inst]) -> Dict[str, list]:
    """
    Returns: net_info: dict net -> [net_id, set((inst_name, pin_name), ...)]
    """
    net_info = {}
    net_idx = 0
    for ins in insts.values():
        for pin_name, net in ins.pins.items():
            if net not in net_info:
                net_info[net] = [net_idx, set()]
                net_idx += 1
            net_info[net][1].add((ins.name, pin_name))
    return net_info

# ---------- util ----------
def identify_pmos(model):
    if "pmos" in model.lower() or "pnp" in model.lower():
        return True
    elif "nmos" in model.lower() or "npn" in model.lower():
        return False
    raise ValueError(f"Invalid kind, {model}, for mos/bjt.")
    

def find_pg_pin(any_dict, pg_nets) -> str:
    for pg_candidate in pg_nets:    
        if pg_candidate in any_dict:
            return pg_candidate
    #raise ValueError("No power/ground pin has found.")
    return None # Return None if no power/ground pin has found.

def initialize_global_variables(is_classification: bool):
    if is_classification:
        constants.POWER_NETS = CLASSIFICATION_POWER_NETS
        constants.GROUND_NETS = CLASSIFICATION_GROUND_NETS
        constants.INSTANCE_TYPES = CLASSIFICATION_INSTANCE_TYPES
    else:
        constants.POWER_NETS = REGRESSION_POWER_NETS
        constants.GROUND_NETS = REGRESSION_GROUND_NETS
        constants.INSTANCE_TYPES = REGRESSION_INSTANCE_TYPES

def update_i_source(insts):
    # Only for AnalogGenie Dataset
    # Add isource
    change_list = []
    added_insts = {}
    for name, inst in insts.items():
        pin_idx = 0
        for pin, net in inst.pins.items():
            if net.startswith('I'):
                # Create isource
                terminal_pg = None
                
                if inst.kind == "mos" or inst.kind == "bjt":
                    is_pmos = identify_pmos(inst.model)
                    if not is_pmos:
                        if pin in ["D", "G"]:
                            terminal_pg = "VDD"
                        else:
                            terminal_pg = "VSS"
                    else:
                        if pin in ["D", "G"]:
                            terminal_pg = "VSS"
                        else:
                            terminal_pg = "VDD"
                else:
                    if pin_idx == 0:
                        terminal_pg = "VDD"
                    else:
                        terminal_pg = "VSS"
                
                new_net = f"n_{net}"
                if net not in added_insts:
                    i_source = Inst(name=net, kind="isource", model="isource", pins={"P0": terminal_pg, "P1":new_net}, attrs=None)
                    added_insts[net] = i_source
                change_list.append([name, pin, new_net])
            
            pin_idx += 1
    
    for name, i_source in added_insts.items():
        insts[name] = i_source
    for name, pin, new_net in change_list:
        insts[name].pins[pin] = new_net
        
        
    for name, inst in insts.items():
        for pin, net in inst.pins.items():
            assert not net.startswith('I')
    return insts