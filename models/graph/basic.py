import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATConv, global_mean_pool


class GraphBackbone(nn.Module):
    """Base class for simple graph encoders that produce graph-level predictions."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _pool_nodes(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)


class GraphGCN(GraphBackbone):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_attr_channels: int | None = None,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GraphGCN num_layers must be >= 2")
        self.dropout = dropout
        if edge_attr_channels is not None and not isinstance(edge_attr_channels, int):
            edge_attr_channels = 0
        self.edge_attr_channels = edge_attr_channels or 0
        self.edge_encoder = None
        if self.edge_attr_channels > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.edge_attr_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.head = nn.Linear(hidden_channels, out_channels)

    def _edge_weight(self, edge_attr: torch.Tensor | None) -> torch.Tensor | None:
        if edge_attr is None:
            return None
        if self.edge_encoder is not None:
            weight = self.edge_encoder(edge_attr).squeeze(-1)
            return F.softplus(weight)
        if edge_attr.dim() == 1:
            return edge_attr
        if edge_attr.size(-1) == 1:
            return edge_attr.view(-1)
        return None

    def forward(self, x, edge_index, edge_attr=None):
        edge_weight = self._edge_weight(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = self._pool_nodes(x)
        logits = self.head(graph_emb)
        return logits.squeeze(0)


class GraphGIN(GraphBackbone):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_attr_channels: int | None = None,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GraphGIN num_layers must be >= 2")
        self.dropout = dropout
        if edge_attr_channels is not None and not isinstance(edge_attr_channels, int):
            edge_attr_channels = 0
        self.edge_attr_channels = edge_attr_channels or 0
        self.use_edge_attr = self.edge_attr_channels > 0
        self.convs = nn.ModuleList()
        self.convs.append(self._build_conv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(self._build_conv(hidden_channels, hidden_channels))
        self.convs.append(self._build_conv(hidden_channels, hidden_channels))
        self.head = nn.Linear(hidden_channels, out_channels)

    def _build_conv(self, in_dim: int, out_dim: int) -> GINConv:
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        if self.use_edge_attr:
            return GINEConv(mlp, train_eps=True, edge_dim=self.edge_attr_channels)
        return GINConv(mlp, train_eps=True)

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            if self.use_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = self._pool_nodes(x)
        logits = self.head(graph_emb)
        return logits.squeeze(0)


class GraphGAT(GraphBackbone):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        heads: int = 4,
        edge_attr_channels: int | None = None,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GraphGAT num_layers must be >= 2")
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads for GraphGAT")
        self.dropout = dropout
        self.heads = heads
        self.out_per_head = hidden_channels // heads
        self.feature_dim = self.out_per_head * heads
        if edge_attr_channels is not None and not isinstance(edge_attr_channels, int):
            edge_attr_channels = 0
        self.edge_attr_channels = edge_attr_channels or 0
        self.edge_dim = self.edge_attr_channels if self.edge_attr_channels > 0 else None
        conv_kwargs = {}
        if self.edge_dim is not None:
            conv_kwargs["edge_dim"] = self.edge_dim
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                self.out_per_head,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                concat=True,
                **conv_kwargs,
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    self.feature_dim,
                    self.out_per_head,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=True,
                    concat=True,
                    **conv_kwargs,
                )
            )
        self.convs.append(
            GATConv(
                self.feature_dim,
                self.out_per_head,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                concat=True,
                **conv_kwargs,
            )
        )
        self.head = nn.Linear(self.feature_dim, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr if self.edge_dim is not None else None)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = self._pool_nodes(x)
        logits = self.head(graph_emb)
        return logits.squeeze(0)
