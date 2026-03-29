import torch
import torch.nn as nn

# ======================================================================
# Standalone IoU utilities
# ======================================================================

def bboxes_iou(bboxes_a, bboxes_b, xyxy=False):
    """
    Pairwise IoU between two sets of boxes.

    Args:
        bboxes_a: [N, 4]
        bboxes_b: [M, 4]
        xyxy:     if False, inputs are (cx, cy, w, h); if True, (x1, y1, x2, y2)

    Returns:
        [N, M] IoU matrix
    """
    # Upcast to float32: area products overflow fp16 for any box > ~255 pixels wide
    bboxes_a = bboxes_a.float()
    bboxes_b = bboxes_b.float()

    if not xyxy:
        a_x1y1 = bboxes_a[:, :2] - bboxes_a[:, 2:4] / 2
        a_x2y2 = bboxes_a[:, :2] + bboxes_a[:, 2:4] / 2
        b_x1y1 = bboxes_b[:, :2] - bboxes_b[:, 2:4] / 2
        b_x2y2 = bboxes_b[:, :2] + bboxes_b[:, 2:4] / 2
    else:
        a_x1y1, a_x2y2 = bboxes_a[:, :2], bboxes_a[:, 2:4]
        b_x1y1, b_x2y2 = bboxes_b[:, :2], bboxes_b[:, 2:4]

    tl = torch.max(a_x1y1[:, None, :], b_x1y1[None, :, :])  # [N, M, 2]
    br = torch.min(a_x2y2[:, None, :], b_x2y2[None, :, :])  # [N, M, 2]
    inter = (br - tl).clamp(min=0).prod(dim=2)                # [N, M]

    area_a = (a_x2y2[:, 0] - a_x1y1[:, 0]) * (a_x2y2[:, 1] - a_x1y1[:, 1])  # [N]
    area_b = (b_x2y2[:, 0] - b_x1y1[:, 0]) * (b_x2y2[:, 1] - b_x1y1[:, 1])  # [M]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


class IOUloss(nn.Module):
    """IoU / GIoU loss on (cx, cy, w, h) boxes."""

    def __init__(self, reduction="none", loss_type="iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        # Upcast to float32: area products overflow fp16 for any box > ~255 pixels wide
        pred   = pred.float()
        target = target.float()

        # Convert cxcywh → xyxy
        p_x1y1 = pred[:, :2] - pred[:, 2:4] / 2
        p_x2y2 = pred[:, :2] + pred[:, 2:4] / 2
        t_x1y1 = target[:, :2] - target[:, 2:4] / 2
        t_x2y2 = target[:, :2] + target[:, 2:4] / 2

        tl = torch.max(p_x1y1, t_x1y1)
        br = torch.min(p_x2y2, t_x2y2)
        inter = (br - tl).clamp(min=0).prod(dim=1)

        area_p = (p_x2y2[:, 0] - p_x1y1[:, 0]) * (p_x2y2[:, 1] - p_x1y1[:, 1])
        area_t = (t_x2y2[:, 0] - t_x1y1[:, 0]) * (t_x2y2[:, 1] - t_x1y1[:, 1])
        union = area_p + area_t - inter + 1e-7
        iou = inter / union

        if self.loss_type == "giou":
            enc_tl = torch.min(p_x1y1, t_x1y1)
            enc_br = torch.max(p_x2y2, t_x2y2)
            enc_area = (enc_br - enc_tl).clamp(min=0).prod(dim=1)
            loss = 1.0 - (iou - (enc_area - union) / (enc_area + 1e-7))
        else:
            loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
