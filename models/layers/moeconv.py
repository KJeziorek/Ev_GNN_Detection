from __future__ import annotations

import torch
import torch.nn as nn

from utils.data import GraphData


class MoEConv(nn.Module):
    """
    Mixture-of-Experts graph convolution.

    Each edge has a positional difference (diff_pos = pos_j - pos_i).  A small
    learnable gating network maps that diff_pos to scores over `num_kernels`
    expert weight matrices.  The top-`top_k` experts are selected, their
    outputs are combined with softmax weights, and the result is max-pooled
    per node before passing through a global MLP with a skip connection.

    Args:
        in_channels:   Node-feature dimension (positional diff is appended
                       internally, so the experts operate on in_channels + 2).
        out_channels:  Output feature dimension.
        num_kernels:   Number of expert weight matrices (k).
        top_k:         How many experts to activate per edge (default 1).
        bias:          Whether the global_nn linear layers use a bias.

    Example::
        conv = MoEConv(16, 64, num_kernels=4, top_k=1)
        data = conv(data)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 4,
        top_k: int = 1,
        dim: int = 2,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.top_k = top_k
        self.dim = dim
        self.bias = bias

        # k expert weight matrices — shape (num_kernels, in_channels+2, out_channels)
        # The +2 accounts for the 2-D positional diff concatenated with node features.
        self.expert_weights = nn.Parameter(
            torch.empty(num_kernels, in_channels, out_channels)
        )
        nn.init.kaiming_uniform_(self.expert_weights, a=0.01)

        # Gating network: maps diff_pos (2-D) → per-kernel logits
        self.gate = nn.Linear(dim, num_kernels, bias=True)

        # Global MLP applied after per-node aggregation (same structure as PointNetConv)
        self.global_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * 2, out_channels, bias=bias),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: GraphData) -> GraphData:
        x, pos, edge_index = data.x, data.pos[:, :self.dim], data.edge_index

        pos_i = pos[edge_index[:, 0]]          # (E, dim)
        pos_j = pos[edge_index[:, 1]]          # (E, dim)
        diff_pos = pos_j - pos_i               # (E, dim)
        x_j = x[edge_index[:, 1]]             # (E, in_channels)

        # Edge input: node features + positional difference
        # msg_input = torch.cat((x_j, diff_pos), dim=1)  # (E, in_channels+2)
        msg_input = x_j

        # ---- Gating ----
        gate_logits = self.gate(diff_pos)                                  # (E, num_kernels)
        _, topk_idx = torch.topk(gate_logits, self.top_k, dim=1)          # (E, top_k)

        # ---- Expert application ----
        # Fuse all experts into one GEMM: (E, in) @ (in, num_kernels*out) → (E, num_kernels*out)
        # Memory: O(E × num_kernels × out) — avoids OOM for large E.
        E = msg_input.size(0)
        expert_w = self.expert_weights.to(msg_input.dtype)                 # (num_kernels, in, out)
        w_fused = expert_w.permute(1, 0, 2).reshape(self.in_channels, self.num_kernels * self.out_channels)
        all_out = (msg_input @ w_fused).view(E, self.num_kernels, self.out_channels)  # (E, num_kernels, out)

        # Gather top-k expert outputs and sum them — no softmax weighting.
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, self.out_channels)    # (E, top_k, out)
        msg = all_out.gather(1, idx).sum(dim=1)                           # (E, out)

        # ---- Aggregation + global MLP ----
        out = self._scatter_amax(msg, edge_index)

        out_skip = out
        out = self.global_nn(out)
        out = out + out_skip  # skip connection

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
                f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"num_kernels={self.num_kernels}, top_k={self.top_k}, bias={self.bias})")
