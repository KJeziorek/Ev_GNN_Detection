import torch
import torch.nn as nn
from torch.nn import BatchNorm1d

from models.layers.my_pointnet import MyPointNetConv
from models.layers.my_linear import MyLinear
from models.layers.my_pooling import LIFSpikePool
from utils.data import GraphData


class BlockConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = MyPointNetConv(in_channels + 2, out_channels)
        self.norm1 = BatchNorm1d(out_channels)
        self.conv2 = MyPointNetConv(out_channels + 2, out_channels)
        self.norm2 = BatchNorm1d(out_channels)

        self.linear = MyLinear(in_channels, out_channels)
        self.norm_linear = BatchNorm1d(out_channels)

    def forward(self, data):
        if data.x.size(0) == 0:
            data.x = torch.zeros(0, self.out_channels, device=data.x.device)
            return data

        x_skip = self.norm_linear(self.linear(data.x))

        x = self.conv1(data.x, data.pos[:, :2], data.edge_index)
        x = torch.nn.functional.relu(self.norm1(x))

        x = self.conv2(x, data.pos[:, :2], data.edge_index)
        x = self.norm2(x)

        x = torch.nn.functional.relu(x + x_skip)
        data.x = x
        return data


class BACKBONE(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = BlockConv(1, 16)
        self.pool1 = LIFSpikePool(16, 16, 80, 60, (240, 180))

        self.block2 = BlockConv(16, 32)
        self.pool2 = LIFSpikePool(32, 32, 40, 30, (240, 180))

        self.block3 = BlockConv(32, 64)
        self.pool3 = LIFSpikePool(64, 64, 20, 15, (240, 180))

        self.block4 = BlockConv(64, 128)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        """
        Multi-scale forward pass.

        Args:
            x:          [N, 1]     input features (e.g. polarity)
            pos:        [N, 3+]    positions, last column = timestamp
            edge_index: [E, 2]     input edges (src, dst)

        Returns:
            List of (x, pos, edge_index) triples at 3 scales:
              scale 0 — after pool1  (80×60 grid, 16 ch → 32 ch)
              scale 1 — after pool2  (40×30 grid, 32 ch → 64 ch)
              scale 2 — after pool3  (20×15 grid, 64 ch → 64 ch)
        """
        # --- Scale 0: full res → 80×60 ---

        data = self.block1(data)
        data = self.pool1(data)
        data = self.block2(data)

        feat_s0 = data.clone()

        # --- Scale 1: 80×60 → 40×30 ---
        data = self.pool2(data)
        data = self.block3(data)

        feat_s1 = data.clone()

        # --- Scale 2: 40×30 → 20×15 ---
        data = self.pool3(data)
        data = self.block4(data)

        feat_s2 = data.clone()

        return [feat_s0, feat_s1, feat_s2]