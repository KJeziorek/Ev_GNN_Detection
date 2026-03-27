from __future__ import annotations

import torch
import torch.nn as nn

from utils.data import GraphData


class PointNetConv(nn.Module):
    """
    PointNet-style graph convolution.

    Args:
        input_dim:    Concatenated input size (node features + positional diff).
        output_dim:   Output feature size.
        bias:         Bias for the global_nn aggregation layer.
    Example::
        conv = MyPointNetConv(16, 64)
        for batch in loader:
            conv(x, pos, edge_index)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        # Float layers — only things created at construction time
        self.linear = nn.Linear(in_channels + 2, out_channels, bias=True)
        self.global_nn = nn.Linear(out_channels, out_channels, bias=bias)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        data: GraphData
        ) -> GraphData:

        x, pos, edge_index = data.x, data.pos[:, :2], data.edge_index

        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]
        x_j = x[edge_index[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)
        msg = self.linear(msg)
        out = self._scatter_amax(msg, edge_index)
        out = self.global_nn(out)
        data.x = out
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scatter_amax(
        self,
        msg: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Max-pool messages into per-node features."""
        unique_nodes, indices = torch.unique(edge_index[:, 0], return_inverse=True)
        expanded = indices.unsqueeze(1).expand(-1, self.out_channels)
        out = torch.zeros((unique_nodes.size(0), self.out_channels),
                          dtype=msg.dtype, device=msg.device)
        out = out.scatter_reduce(0, expanded, msg, reduce="amax", include_self=False)
        return out

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, out_channels={self.out_channels}, bias={self.bias})")
