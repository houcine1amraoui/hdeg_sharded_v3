from dataclasses import dataclass
from torch import Tensor


@dataclass
class GraphStructure:
    node_features: Tensor      # (B, N, d)
    edge_index: Tensor         # (B, 2, E)
    edge_weight: Tensor | None = None