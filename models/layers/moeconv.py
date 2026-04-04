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
        gate_logits = self.gate(diff_pos)                          # (E, num_kernels)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=1)  # (E, top_k)
        topk_weights = torch.softmax(topk_vals, dim=1)             # (E, top_k)

        # ---- Expert application ----
        # Group edges by their selected expert instead of gathering a full
        # (E, top_k, in_dim, out_dim) weight tensor — that would be O(E × channels²)
        # and blows up memory for large E and channel sizes.
        E = msg_input.size(0)
        msg = torch.zeros(E, self.out_channels,
                          dtype=msg_input.dtype, device=msg_input.device)

        for ki in range(self.top_k):
            expert_idx = topk_idx[:, ki]      # (E,) — which expert each edge uses
            weight_col = topk_weights[:, ki]  # (E,) — its softmax coefficient

            acc = torch.zeros_like(msg).to(msg_input.dtype)
            for k in range(self.num_kernels):
                mask = expert_idx == k
                if mask.any():
                    # autocast may run matmul in fp16 even with fp32 inputs;
                    # cast result back to acc.dtype to avoid index-put dtype mismatch.
                    acc[mask] = (msg_input[mask] @ self.expert_weights[k].to(msg_input.dtype)).to(acc.dtype)

            msg = msg + acc * weight_col.unsqueeze(-1)

        # ---- Aggregation + global MLP ----
        out = self._scatter_amax(msg, edge_index)

        out_skip = out.clone()
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
