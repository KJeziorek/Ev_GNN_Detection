import torch
import torch.nn as nn

from models.layers.network_blocks import BlockConv
from models.layers.pooling import GraphPooling
from models.layers.norm import BatchNorm
from models.layers.linear import LinearX
from utils.data import GraphData

class BACKBONE(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = BlockConv(1, 32)
        self.pool1 = GraphPooling((240/80, 180/60, 1))

        self.block2 = BlockConv(32, 64)
        self.pool2 = GraphPooling((80/40, 60/30, 1))

        self.block3 = BlockConv(64, 128)
        self.pool3 = GraphPooling((40/20, 30/15, 1))

        self.block4 = BlockConv(128, 256)

        # self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, LinearX):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data: GraphData):
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

        return [feat_s2]