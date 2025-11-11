from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Inst:
    name: str
    kind: str                 # 'mos', 'capacitor', 'resistor', ...
    model: str
    pins: Dict[str, str]      # role -> net name
    attrs: Dict[str, Any]     # parameters (parsed or raw); may be empty
