import math

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d

from models.layers.my_pointnet import MyPointNetConv


class HEAD(nn.Module):
    """GNN detection head: per-node cls / reg / obj predictions."""

    def __init__(self, in_channels=64, num_classes=101):
        super().__init__()

        self.stem = MyPointNetConv(in_channels + 2, in_channels)
        self.stem_norm = BatchNorm1d(in_channels)

        self.conv1 = MyPointNetConv(in_channels + 2, in_channels)
        self.conv1_norm = BatchNorm1d(in_channels)

        self.conv2 = MyPointNetConv(in_channels + 2, in_channels)
        self.conv2_norm = BatchNorm1d(in_channels)

        self.regr = MyPointNetConv(in_channels + 2, 4)
        self.cls = MyPointNetConv(in_channels + 2, num_classes, bias=True)
        self.obj = MyPointNetConv(in_channels + 2, 1, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for m in self.modules():
            if isinstance(m, BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Zero-init obj/cls output weights + prior-prob bias so that
        # at init every cell predicts sigmoid(-4.6) ≈ 1% confidence.
        # Weights at zero still receive gradients (d_loss/d_w = d_loss/d_out * input).
        for head in (self.obj, self.cls):
            nn.init.zeros_(head.global_nn.weight)
            nn.init.constant_(head.global_nn.bias, bias_value)

    def forward(self, data):
        """
        Args:
            data: PyG Data with x, pos, edge_index

        Returns:
            (cls, reg, obj) — each [N_nodes, C]
        """
        if data.x.size(0) == 0:
            device = data.x.device
            return (
                torch.zeros(0, self.cls.output_dim, device=device),
                torch.zeros(0, 4, device=device),
                torch.zeros(0, 1, device=device),
            )

        pos2d = data.pos[:, :2]
        ei = data.edge_index

        x = self.stem(data.x, pos2d, ei)
        x = torch.nn.functional.relu(self.stem_norm(x))

        x_copy = x.clone()

        x1 = self.conv1(x_copy, pos2d, ei)
        x1 = torch.nn.functional.relu(self.conv1_norm(x1))

        x2 = self.conv2(x, pos2d, ei)
        x2 = torch.nn.functional.relu(self.conv2_norm(x2))

        reg = self.regr(x1.clone(), pos2d, ei)
        obj = self.obj(x1, pos2d, ei)
        cls = self.cls(x2, pos2d, ei)

        return cls, reg, obj
