import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.backbone import BACKBONE
from models.head import YOLOXHead
from models.yolox.utils.compat import meshgrid

logger = logging.getLogger(__name__)


class Detection(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = BACKBONE()
        self.head = YOLOXHead(num_classes=100, strides=[12], in_channels=[256])

    def forward(self, data):
        fpn_outs = self.backbone(data)
        if self.training:
            assert data.target is not None
            loss, iou_loss, conf_loss, cls_loss, num_fg = self.head(
                fpn_outs, data.target, data
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs