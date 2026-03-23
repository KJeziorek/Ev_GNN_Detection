from __future__ import annotations

import torch
import torch.nn as nn

from utils.data import GraphData
from models.layers.pointnet import PointNetConv
from models.layers.linear import LinearX
from models.layers.norm import BatchNorm


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A PointNet -> Batchnorm -> activation block"""

    def __init__(
        self, in_channels, out_channels, bias=False, act="relu"
    ):
        super().__init__()

        self.conv = PointNetConv(in_channels, out_channels, bias)
        self.norm = BatchNorm(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, data: GraphData):
        data = self.norm(self.conv(data))
        data.x = self.act(data.x)
        return data


class BlockConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, act="relu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = PointNetConv(in_channels, out_channels)
        self.norm1 = BatchNorm(out_channels)

        self.conv2 = PointNetConv(out_channels, out_channels)
        self.norm2 = BatchNorm(out_channels)

        self.linear = LinearX(in_channels, out_channels)
        self.norm_linear = BatchNorm(out_channels)

        self.act = get_activation(act, inplace=True)

    def forward(self, data: GraphData):
        # skip branch: project input features to out_channels
        skip = data.clone()
        skip = self.norm_linear(self.linear(skip))

        # main branch
        data = self.norm1(self.conv1(data))
        data.x = self.act(data.x)

        data = self.norm2(self.conv2(data))

        # residual add
        data.x = self.act(data.x + skip.x)
        return data