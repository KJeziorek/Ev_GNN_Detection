"""
Detection model with incremental accumulator head.

Drop-in replacement for models/detection.py Detection class.
Returns the same output dict format so your training loop
doesn't need to change.
"""

import torch
import torch.nn as nn

from models.backbone import BACKBONE
from models.incremental_head import IncrementalDetectionHead


class DetectionIncremental(nn.Module):
    """
    BACKBONE -> IncrementalDetectionHead

    Training output dict:
        total_loss, iou_loss, conf_loss, cls_loss, num_fg
        (same keys as the original Detection model)

    Inference output:
        list[dict]  per image with  {boxes, scores, labels}
    """

    def __init__(self, num_classes=100, conf_threshold=0.3):
        super().__init__()

        self.backbone = BACKBONE()

        # backbone final output: 256-ch features at stride 12
        # grid_size = (H, W) of the accumulator = backbone output spatial dims
        #   240 / (3*2*2) = 20  ->  W = 20
        #   180 / (3*2*2) = 15  ->  H = 15
        self.head = IncrementalDetectionHead(
            num_classes=num_classes,
            strides=(12,),
            in_channels=(256,),
            grid_size=(15, 20),
            ema_alpha=0.10,
            decay_lambda=0.005,
            conf_threshold=conf_threshold,
        )

    def forward(self, data):
        fpn_outs = self.backbone(data)

        if self.training:
            assert data.target is not None
            loss, reg_loss, hm_loss, cls_loss, num_fg = self.head(
                fpn_outs, data.target, data
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": reg_loss,
                "conf_loss": hm_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    @torch.no_grad()
    def forward_with_detections(self, data):
        """
        Returns both loss dict AND detection list.
        Used during validation to compute both val_loss and mAP.

        Returns:
            (loss_dict, detections)
            loss_dict:  {total_loss, iou_loss, conf_loss, cls_loss, num_fg}
            detections: list[dict] per image  {boxes (xyxy), scores, labels}
        """
        fpn_outs = self.backbone(data)

        (loss, reg_loss, hm_loss, cls_loss, num_fg), detections = self.head(
            fpn_outs, data.target, data, return_detections=True
        )

        loss_dict = {
            "total_loss": loss,
            "iou_loss": reg_loss,
            "conf_loss": hm_loss,
            "cls_loss": cls_loss,
            "num_fg": num_fg,
        }
        return loss_dict, detections

    # ==================================================================
    #  Streaming inference helpers (for hardware / latency experiments)
    # ==================================================================

    def init_streaming(self, device="cuda", dtype=torch.float32):
        """Initialise persistent accumulator state for event-by-event inference."""
        self.eval()
        self._stream_state = self.head.init_state(device, dtype)
        self._prev_t = None
        return self._stream_state

    @torch.no_grad()
    def stream_event(self, backbone_data, read_every=1, _count=[0]):
        """
        Feed a single backbone-output event into the accumulator.

        Args:
            backbone_data: GraphData with 1 node (post-backbone)
            read_every:    how often to extract detections (1 = every event)

        Returns:
            detections or None  (None when skipping readout for speed)
        """
        # decide whether to apply global decay (once per new timestamp)
        t = backbone_data.pos[0, 2].item() if backbone_data.pos.shape[1] > 2 else 0
        decay = self._prev_t is None or t != self._prev_t
        self._prev_t = t

        self._stream_state = self.head.process_single_event(
            self._stream_state, backbone_data, apply_global_decay=decay
        )

        _count[0] += 1
        if _count[0] % read_every == 0:
            return self.head.read_detections_from_state(self._stream_state)
        return None
