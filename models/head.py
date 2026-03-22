import math
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d

from models.layers.my_pointnet import MyPointNetConv
from models.layers.my_linear import MyLinear
from models.layers.my_pooling import LIFSpikePool


class HEAD(torch.nn.Module):
    def __init__(self, 
                 in_channels=64, 
                 num_classes=101):
        super().__init__()

        self.stem = MyPointNetConv(in_channels+2, in_channels)
        self.stem_norm = BatchNorm1d(in_channels)

        self.conv1 = MyPointNetConv(in_channels+2, in_channels)
        self.conv1_norm = BatchNorm1d(in_channels)

        self.conv2 = MyPointNetConv(in_channels+2, in_channels)
        self.conv2_norm = BatchNorm1d(in_channels)

        self.regr = MyPointNetConv(in_channels+2, 4)
        self.cls = MyPointNetConv(in_channels+2, num_classes)
        self.obj = MyPointNetConv(in_channels+2, 1)

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
        if data.x.size(0) == 0:
            device = data.x.device
            empty_cls = torch.zeros(0, self.cls.output_dim, device=device)
            empty_reg = torch.zeros(0, 4, device=device)
            empty_obj = torch.zeros(0, 1, device=device)
            empty_pos = torch.zeros(0, 3, device=device)
            return (empty_cls, empty_reg, empty_obj, empty_pos)

        x = self.stem(data.x, data.pos[:,:2], data.edge_index)
        x = self.stem_norm(x)
        x = torch.nn.functional.relu(x)

        x_copy = x.clone()

        x1 = self.conv1(x_copy, data.pos[:,:2], data.edge_index)
        x1 = self.conv1_norm(x1)
        x1 = torch.nn.functional.relu(x1)

        x2 = self.conv2(x, data.pos[:,:2], data.edge_index)
        x2 = self.conv2_norm(x2)
        x2 = torch.nn.functional.relu(x2)

        x1_copy = x1.clone()

        reg = self.regr(x1_copy, data.pos[:,:2], data.edge_index)
        obj = self.obj(x1, data.pos[:,:2], data.edge_index)
        cls = self.cls(x2, data.pos[:,:2], data.edge_index)

        return (cls, reg, obj)