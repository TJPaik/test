import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter_add


def _resolve_dhgbench_paths() -> Tuple[Path, Path]:
    """Return (repo_root, package_root) for DHG-Bench."""
    env_path = os.environ.get("DHG_BENCH_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2] / "DHG-Bench",
            here.parents[3] / "DHG-Bench",
            Path.cwd() / "DHG-Bench",
        ]
    )
    for candidate in candidates:
        pkg_root = candidate / "dhgbench"
        if pkg_root.is_dir():
            return candidate, pkg_root
    raise FileNotFoundError(
        "Unable to locate DHG-Bench. Set DHG_BENCH_PATH or place the repo next to DAC_analogHyper."
    )


DHG_BENCH_ROOT, DHG_BENCH_PKG = _resolve_dhgbench_paths()
if str(DHG_BENCH_PKG) not in sys.path:
    sys.path.insert(0, str(DHG_BENCH_PKG))

import yaml  # noqa: E402

from lib_models.HNN.allset import SetGNN  # noqa: E402
from lib_models.HNN.dphgnn import DPHGNN  # noqa: E402
from lib_models.HNN.ehnn import EHNN  # noqa: E402
from lib_models.HNN.hjrl import HJRL  # noqa: E402
from lib_models.HNN.hypergt import HyperGT  # noqa: E402
from lib_models.HNN.hypernd import HyperND  # noqa: E402
from lib_models.HNN.preprocessing import (  # noqa: E402
    dphgnn_preprocessing,
    hjrl_preprocessing,
    ehnn_preprocessing,
    hypergt_preprocessing,
    phenomNN_preprocessing,
)
from lib_models.HNN.phenomnn import PhenomNN  # noqa: E402
from lib_models.HNN.sheafhypergnn import SheafHyperGNN  # noqa: E402
from lib_models.HNN.tfhnn import TFHNN  # noqa: E402


CONFIG_DIR = DHG_BENCH_PKG / "lib_yamls" / "hg_yamls"


def _load_method_config(method: str, profile: str = "default") -> Dict[str, Any]:
    cfg_path = CONFIG_DIR / f"config_{method.lower()}.yaml"
    if not cfg_path.is_file():
        return {}
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict):
        return {}
    if profile in data:
        return dict(data[profile])
    return dict(data.get("default", {}))


def _build_args(method: str, overrides: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    cfg = _load_method_config(method)
    overrides = overrides or {}
    cfg.update(overrides)
    device = cfg.get("device")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    cfg.setdefault("method", method)
    cfg.setdefault("task_type", "hg_cls")
    cfg.setdefault("dname", f"analog_{method.lower()}")
    return SimpleNamespace(**cfg)


def _graph_readout(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1:
        return logits
    return logits.mean(dim=0)


def _build_v2e_norm(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    device = edge_index.device
    if edge_index.numel() == 0:
        return torch.zeros(0, device=device)
    cidx = edge_index[1].min()
    adjusted = (edge_index[1] - cidx).long()
    num_edges = int(adjusted.max().item()) + 1
    ones = torch.ones(edge_index.size(1), device=device)
    node_deg = scatter_add(ones, edge_index[0], dim=0, dim_size=num_nodes)
    edge_deg = scatter_add(ones, adjusted, dim=0, dim_size=num_edges)
    node_inv = torch.pow(torch.clamp(node_deg, min=1.0), -0.5)
    edge_inv = torch.pow(torch.clamp(edge_deg, min=1.0), -0.5)
    return node_inv[edge_index[0]] * edge_inv[adjusted]



class DHGModelWrapper(nn.Module):
    def __init__(
        self,
        method: str,
        in_channels: int,
        out_channels: int,
        overrides: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.method = method
        self.args = _build_args(method, overrides)
        self.stats = stats or {}
        self.device_str = self.args.device
        self.device_obj = torch.device(self.device_str)
        self.preprocessor: Optional[Callable[[Data, SimpleNamespace], Data]] = None
        self._prep_cache_name = f"_dhg_{method.lower()}_cached"
        self.model = self._build_model(in_channels, out_channels)

    def _build_model(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.method == "HyperGT":
            self.preprocessor = hypergt_preprocessing
            return HyperGT(in_channels, out_channels, self.args)
        if self.method == "DPHGNN":
            self.preprocessor = dphgnn_preprocessing
            self.args.num_features = in_channels
            return DPHGNN(in_channels, out_channels, self.args)
        if self.method == "HJRL":
            self.preprocessor = hjrl_preprocessing
            return HJRL(in_channels, out_channels, self.args)
        if self.method == "SheafHyperGNN":
            return SheafHyperGNN(in_channels, out_channels, self.args)
        if self.method == "EHNN":
            cache_dir = str(DHG_BENCH_PKG / "lib_ehnn_cache")
            self.preprocessor = lambda data, args: ehnn_preprocessing(data, args, folder=cache_dir)  # noqa: E731
            max_order = int(self.stats.get("max_hyperedge_order", 32))
            dummy_cache = {
                "edge_orders": torch.tensor([max_order], dtype=torch.long),
                "overlaps": torch.tensor([0], dtype=torch.long),
            }
            return EHNN(in_channels, out_channels, self.args, ehnn_cache=dummy_cache)
        if self.method == "HyperND":
            return HyperND(in_channels, out_channels, self.args)
        if self.method in {"AllSetformer", "AllDeepSets"}:
            return SetGNN(in_channels, out_channels, self.args)
        if self.method == "TFHNN":
            return TFHNN(in_channels, out_channels, self.args)
        if self.method in {"PhenomNN", "PhenomNNS"}:
            self.preprocessor = phenomNN_preprocessing
            return PhenomNN(in_channels, out_channels, self.args)
        raise ValueError(f"Unsupported DHG method: {self.method}")

    def _prepare_data(self, data: Data) -> Data:
        if self.method != "SheafHyperGNN":
            cached = getattr(data, self._prep_cache_name, None)
            if cached is not None:
                return cached
        prepared = copy.deepcopy(data)
        if self.preprocessor is not None:
            prepared = self.preprocessor(prepared, self.args)
        prepared = prepared.to(self.device_obj)
        if self.method == "EHNN" and hasattr(prepared, "ehnn_cache"):
            for key, value in prepared.ehnn_cache.items():
                if isinstance(value, torch.Tensor):
                    prepared.ehnn_cache[key] = value.to(self.device_obj)
        if self.method in {"AllSetformer", "AllDeepSets"} and not hasattr(prepared, "norm"):
            prepared.norm = _build_v2e_norm(prepared.hyperedge_index, prepared.x.size(0))
        if self.method in {"PhenomNN", "PhenomNNS"}:
            prepared.adj = [tensor.to(self.device_obj) for tensor in prepared.adj]
            prepared.G = [tensor.to(self.device_obj) for tensor in prepared.G]
        if self.method != "SheafHyperGNN":
            setattr(data, self._prep_cache_name, prepared)
        return prepared

    def forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor, data: Data):
        prepared = self._prepare_data(data)
        if self.method == "SheafHyperGNN":
            self.model.hyperedge_attr = None
        outputs = self.model(prepared)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        if logits.is_sparse:
            logits = logits.to_dense()
        graph_logits = _graph_readout(logits)
        return graph_logits


def build_dhg_model(
    method: str,
    in_channels: int,
    out_channels: int,
    overrides: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> DHGModelWrapper:
    return DHGModelWrapper(method, in_channels, out_channels, overrides, stats=stats)
