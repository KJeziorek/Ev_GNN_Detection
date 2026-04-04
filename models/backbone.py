import torch
import torch.nn as nn

from models.layers.network_blocks import BlockConv
from models.layers.pooling import GraphPooling
from models.layers.norm import BatchNorm
from models.layers.linear import LinearX
from utils.data import GraphData

class BACKBONE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        backbone_cfg = cfg.get("backbone", {})
        channels   = backbone_cfg.get("channels",   [16, 64, 128, 256, 256])
        pool_sizes = backbone_cfg.get("pool_sizes",  [
            [5.0, 5.0, 10.0],
            [2.0, 2.0,  1.0],
            [2.0, 2.0,  1.0],
            [1.0, 1.0,  1.0],
        ])

        # First block takes 1-channel input (polarity)
        in_ch = 1
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()

        # Build: block0, (pool0, block1), (pool1, block2), ...
        self.blocks.append(BlockConv(in_ch, channels[0]))
        for i, (pool_size, out_ch) in enumerate(zip(pool_sizes, channels[1:])):
            self.pools.append(GraphPooling(tuple(pool_size)))
            self.blocks.append(BlockConv(channels[i], out_ch))

        # How many tail-end feature maps to return to the head.
        # Must equal len(head.in_channels) which must match backbone.channels[-num_outputs:].
        self.num_outputs = len(cfg.get("head", {}).get("in_channels", [channels[-1]]))

        self.initialize_weights()

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
        # scale = data.pos.new_tensor([240.0, 180.0])

        pos = data.pos.clone()
        # data.x = torch.cat((data.x, data.pos[:, :2] / scale), dim=1)
        # data.pos[:, :2] = data.pos[:, :2] / 5.0
        data = self.blocks[0](data)
        data.pos = pos

        features = []
        for pool, block in zip(self.pools, self.blocks[1:]):
            data = pool(data)
            # scale = scale / pool.pool_size[:2].float().to(scale.device)
            # data.x = torch.cat((data.x, data.pos[:, :2] / scale), dim=1)
            data = block(data)
            features.append(data.clone())
        return features[-self.num_outputs:]