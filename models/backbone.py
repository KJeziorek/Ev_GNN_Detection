import torch
import time

from torch.nn import BatchNorm1d
import torch.nn as nn

from models.layers.my_pointnet import MyPointNetConv
from models.layers.my_linear import MyLinear
from models.layers.my_pooling import LIFSpikePool

class BlockConv(torch.nn.Module):
    def __init__(self, 
                    in_channels, 
                    out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = MyPointNetConv(in_channels+2, out_channels)
        self.norm1 = BatchNorm1d(out_channels)

        self.conv2 = MyPointNetConv(out_channels+2, out_channels)
        self.norm2 = BatchNorm1d(out_channels)

        self.linear = MyLinear(in_channels, out_channels)
        self.norm_linear = BatchNorm1d(out_channels)

    def forward(self, x, 
                    pos, 
                    edge_index):

        x_skip = x.clone()
        x_skip = self.linear(x_skip)
        x_skip = self.norm_linear(x_skip)


        x = self.conv1(x, pos[:,:2], edge_index)
        x = self.norm1(x)
        x = torch.nn.functional.relu(x)

        x = self.conv2(x, pos[:,:2], edge_index)
        x = self.norm2(x)

        x = torch.nn.functional.relu(x + x_skip)
        return x

class BACKBONE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = BlockConv(1,16)
        self.pool1 = LIFSpikePool(16, 16, 24, 18, (240,180))
        self.block2 = BlockConv(16,32)
        self.block3 = BlockConv(32,64)
        self.block4 = BlockConv(64,64)
        
        self.initialize_weights()
         
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, pos, edge_index):
        x1 = self.block1(x, pos, edge_index)
        x1, pos, edge_index = self.pool1(x1, pos, edge_index.T, pos[:, -1])
        edge_index = edge_index.T
        x2 = self.block2(x1, pos, edge_index)
        x3 = self.block3(x2, pos, edge_index)
        x4 = self.block4(x3, pos, edge_index)
        return x4