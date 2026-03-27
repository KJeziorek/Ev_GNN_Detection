from __future__ import annotations

import torch
import torch.nn as nn

from utils.data import GraphData


class MinMaxConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        pos_dim = 2  # dx, dy, dist, dir_x, dir_y, dt

        # shared message MLP (one is enough — let max/min select from same space)
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + pos_dim, out_channels),
            nn.ReLU(inplace=True),
            # nn.Linear(out_channels, out_channels),
        )

        # post-aggregation: max + min + center node
        self.global_nn = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            # nn.ReLU(inplace=True),
            # nn.Linear(out_channels, out_channels, bias=bias),
        )

    def forward(self, data: GraphData) -> GraphData:
        x, pos_xy, edge_index = data.x, data.pos[:, :2], data.edge_index
        t = data.pos[:, 2:3]
        src, dst = edge_index[:, 0], edge_index[:, 1]

        # --- positional encoding ---
        rel_pos = pos_xy[dst] - pos_xy[src]
        # dist = rel_pos.norm(dim=1, keepdim=True) + 1e-8
        # rel_pos_norm = rel_pos / dist
        # dt = t[dst] - t[src]
        # pos_enc = torch.cat([rel_pos, dist, rel_pos_norm, dt], dim=1)

        # --- messages ---
        msg = torch.cat([x[dst], rel_pos], dim=1)
        msg = self.msg_mlp(msg)

        # --- dual aggregation ---
        agg_max = self._scatter_amax(msg, edge_index)
        agg_min = self._scatter_amin(msg, edge_index)

        # --- combine: max + min + center ---
        out = torch.cat([agg_max, agg_min], dim=1)
        out = self.global_nn(out)

        data.x = out
        return data

    def _scatter_amax(self, msg, edge_index):
        unique_nodes, indices = torch.unique(edge_index[:, 0], return_inverse=True)
        expanded = indices.unsqueeze(1).expand(-1, msg.size(1))
        out = torch.full((unique_nodes.size(0), msg.size(1)),
                         float('-inf'), dtype=msg.dtype, device=msg.device)
        out = out.scatter_reduce(0, expanded, msg, reduce="amax", include_self=False)
        out = out.clamp(min=-1e6, max=1e6)
        return out

    def _scatter_amin(self, msg, edge_index):
        unique_nodes, indices = torch.unique(edge_index[:, 0], return_inverse=True)
        expanded = indices.unsqueeze(1).expand(-1, msg.size(1))
        out = torch.full((unique_nodes.size(0), msg.size(1)),
                         float('inf'), dtype=msg.dtype, device=msg.device)
        out = out.scatter_reduce(0, expanded, msg, reduce="amin", include_self=False)
        out = out.clamp(min=-1e6, max=1e6)
        return out