from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class GraphData:
    x: torch.Tensor           # [N, C] node features
    pos: torch.Tensor          # [N, 3] positions (x, y, t)
    edge_index: torch.Tensor   # [E, 2] edges
    batch: torch.Tensor | None = None  # [N] sample index per node
    bboxes: torch.Tensor | None = None
    batch_bb: torch.Tensor | None = None

    def clone(self):
        return GraphData(
            **{k: v.clone() if isinstance(v, torch.Tensor) else v
               for k, v in self.__dict__.items()}
        )

    def to(self, device):
        return GraphData(
            **{k: v.to(device) if isinstance(v, torch.Tensor) else v
               for k, v in self.__dict__.items()}
        )