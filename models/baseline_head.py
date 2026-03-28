import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.layers.linear import LinearX
from models.layers.pointnet import PointNetConv
from models.layers.norm import BatchNorm
from models.layers.network_blocks import BaseConv
from utils.focal_loss import FocalLoss

# ======================================================================
# Standalone IoU utilities (no external YOLOX dependency needed)
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
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-16)


class IOUloss(nn.Module):
    """IoU / GIoU loss on (cx, cy, w, h) boxes."""

    def __init__(self, reduction="none", loss_type="iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
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
        union = area_p + area_t - inter + 1e-16
        iou = inter / union

        if self.loss_type == "giou":
            enc_tl = torch.min(p_x1y1, t_x1y1)
            enc_br = torch.max(p_x2y2, t_x2y2)
            enc_area = (enc_br - enc_tl).clamp(min=0).prod(dim=1)
            loss = 1.0 - (iou - (enc_area - union) / (enc_area + 1e-16))
        else:
            loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ======================================================================
# Point-based YOLOX Head
# ======================================================================

class YOLOXHead(nn.Module):
    """
    Point-based YOLOX head for GNN event detection.

    Each surviving event after the backbone is treated as a detection anchor.
    There is no dense grid: positions are irregular event coordinates in the
    stride-reduced space (e.g. [0, 80) × [0, 60) at stride 3).

    Decode:
        cx = (pos_x + raw_dx) × stride          (pixels)
        cy = (pos_y + raw_dy) × stride          (pixels)
        w  = exp(raw_log_w)   × stride          (pixels)
        h  = exp(raw_log_h)   × stride          (pixels)

    Labels expected in [B, max_det, 5] = [class_id, cx, cy, w, h] in pixels
    (as produced by convert_to_training_format).
    """

    def __init__(
        self,
        num_classes,
        strides=[3, 6, 12],
        in_channels=[64, 128, 256],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # ---- network layers (same as your template) ----
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.obj_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i]),
                    out_channels=int(in_channels[i]),
                )
            )
            self.cls_convs.append(
                BaseConv(
                    in_channels=int(in_channels[i]),
                    out_channels=int(in_channels[i]),
                )
            )
            self.reg_convs.append(
                BaseConv(
                    in_channels=int(in_channels[i]),
                    out_channels=int(in_channels[i]),
                )
            )
            self.obj_convs.append(
                BaseConv(
                    in_channels=int(in_channels[i]),
                    out_channels=int(in_channels[i]),
                )
            )
            
            self.cls_preds.append(LinearX(int(in_channels[i]), self.num_classes))
            self.reg_preds.append(LinearX(int(in_channels[i]), 4))
            self.obj_preds.append(LinearX(int(in_channels[i]), 1))


        # ---- losses ----
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.obj_focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        self.iou_loss = IOUloss(reduction="none", loss_type="giou")

        self.initialize_parameters()
    # ------------------------------------------------------------------
    # Bias init (same as original YOLOX)
    # ------------------------------------------------------------------

    def initialize_parameters(self, prior_prob=0.01):
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        def init_conv_module(module):
            """Recursively init all Linear layers inside a conv module with Kaiming."""
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, BatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def init_pred_head(module, bias_init=0.0):
            """Init a prediction head — works for nn.Linear, nn.Sequential, or custom conv."""
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, bias_init)

        # --- backbone shared convs ---
        for conv_list in [self.stems, self.cls_convs, self.reg_convs, self.obj_convs]:
            for base_conv in conv_list:
                init_conv_module(base_conv)

        # --- prediction heads ---
        for pred in self.cls_preds:
            init_pred_head(pred, bias_init=bias_value)

        for pred in self.obj_preds:
            init_pred_head(pred, bias_init=bias_value)

        for pred in self.reg_preds:
            init_pred_head(pred, bias_init=0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data_in, labels=None, data_orig=None):
        """
        Args:
            data_in:  list of 3 PyG Data objects (one per scale from backbone)
                      each has .x, .pos [:, (x, y, t)], .batch, .edge_index
            labels:   [B, max_det, 5]  (class_id, cx, cy, w, h) in pixels
                      None during inference.

        Returns:
            training:  (loss, iou_loss, obj_loss, cls_loss, num_fg_ratio)
            inference: list[dict] per image with keys "boxes", "scores", "labels"
        """
        all_outputs = []     # raw [reg(4), obj(1), cls(C)] per event
        all_positions = []   # event (x, y) in stride-reduced space
        all_strides = []     # stride per event (scalar repeated)
        all_batches = []     # batch index per event

        for k, (cls_conv, reg_conv, obj_conv, stride_this_level, data) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.obj_convs, self.strides, data_in)
        ):
            data = self.stems[k](data)
            cls_data = data.clone()
            obj_data = data.clone()

            cls_feat = cls_conv(cls_data)
            reg_feat = reg_conv(data)
            obj_feat = obj_conv(obj_data)

            cls_output = self.cls_preds[k](cls_feat).x       # [N_k, num_cls]
            reg_output = self.reg_preds[k](reg_feat).x       # [N_k, 4]
            obj_output = self.obj_preds[k](obj_feat).x       # [N_k, 1]

            N_k = reg_output.shape[0]
            if N_k == 0:
                continue

            output = torch.cat([reg_output, obj_output, cls_output], dim=1)

            all_outputs.append(output)
            all_positions.append(data.pos[:, :2])             # (x, y) reduced
            all_strides.append(
                output.new_full((N_k,), stride_this_level)
            )
            all_batches.append(data.batch)

        # Handle the (unlikely) case where every scale is empty
        if len(all_outputs) == 0:
            device = data_in[0].x.device
            if self.training:
                z = torch.tensor(0.0, device=device, requires_grad=True)
                return z, z, z, z, 0.0
            else:
                B = int(data_in[0].batch.max().item()) + 1 if data_in[0].batch is not None else 1
                return [
                    {"boxes": torch.zeros(0, 4, device=device),
                     "scores": torch.zeros(0, device=device),
                     "labels": torch.zeros(0, dtype=torch.long, device=device)}
                    for _ in range(B)
                ]

        outputs   = torch.cat(all_outputs, dim=0)     # [N_total, 5+C]
        positions = torch.cat(all_positions, dim=0)    # [N_total, 2]
        strides   = torch.cat(all_strides, dim=0)     # [N_total]
        batches   = torch.cat(all_batches, dim=0)      # [N_total]

        # Decode bbox: raw → pixel-space (cx, cy, w, h).
        # Channels 4+ (obj, cls) stay as raw logits.
        decoded = self.decode_outputs(outputs, positions, strides)

        if self.training:
            return self.get_losses(decoded, strides, batches, labels, positions)
        else:
            return self.postprocess(decoded, batches)

    # ------------------------------------------------------------------
    # Decode:  raw network output → pixel-space boxes
    # ------------------------------------------------------------------

    def decode_outputs(self, outputs, positions, strides):
        """
        Args:
            outputs:   [N, 5+C]  raw (dx, dy, log_w, log_h, obj, cls…)
            positions: [N, 2]    event (x, y) in stride-reduced coordinates
            strides:   [N]       stride per event

        Returns:
            [N, 5+C]  with channels 0-3 decoded to pixel-space (cx, cy, w, h)
                       and channels 4+ unchanged (raw logits).
        """
        decoded = outputs.clone()
        s = strides                                         # shorthand
        decoded[:, 0] = (positions[:, 0] + outputs[:, 0]) * s   # cx  (pixels)
        decoded[:, 1] = (positions[:, 1] + outputs[:, 1]) * s   # cy  (pixels)
        decoded[:, 2] = torch.exp(outputs[:, 2].clamp(max=10.0)) * s   # w
        decoded[:, 3] = torch.exp(outputs[:, 3].clamp(max=10.0)) * s   # h
        return decoded

    # ------------------------------------------------------------------
    # Loss (point-based SimOTA)
    # ------------------------------------------------------------------

    def get_losses(self, decoded, strides, batches, labels, positions):
        dtype = decoded.dtype
        device = decoded.device
        B = int(batches.max().item()) + 1

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        # Accumulate per-batch (no global mask needed)
        fg_bbox_preds = []    # decoded boxes for fg events
        fg_cls_preds = []     # cls logits for fg events
        cls_targets = []
        reg_targets = []
        obj_targets_list = [] # per-batch [N_b, 1]
        obj_preds_list = []   # per-batch [N_b, 1]

        num_fg = 0.0
        num_gts = 0.0

        for b in range(B):
            mask_b = batches == b
            N_b = mask_b.sum().item()

            bbox_preds_b = decoded[mask_b, :4]
            obj_preds_b  = decoded[mask_b, 4:5]
            cls_preds_b  = decoded[mask_b, 5:]
            strides_b    = strides[mask_b]
            anchor_xy_b  = positions[mask_b] * strides_b.unsqueeze(-1)

            num_gt = int(nlabel[b])
            num_gts += num_gt

            if num_gt == 0 or N_b == 0:
                obj_targets_list.append(torch.zeros(N_b, 1, device=device, dtype=dtype))
                obj_preds_list.append(obj_preds_b)
                # Nothing to append to fg lists — no foreground
                continue

            gt_bboxes = labels[b, :num_gt, 1:5]
            gt_classes = labels[b, :num_gt, 0]

            try:
                (
                    gt_matched_classes, fg_mask_local,
                    pred_ious_this_matching, matched_gt_inds, num_fg_img,
                ) = self.get_assignments(
                    gt_bboxes, gt_classes,
                    bbox_preds_b, anchor_xy_b,
                )
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise
                torch.cuda.empty_cache()
                (
                    gt_matched_classes, fg_mask_local,
                    pred_ious_this_matching, matched_gt_inds, num_fg_img,
                ) = self.get_assignments(
                    gt_bboxes, gt_classes,
                    bbox_preds_b, anchor_xy_b,
                    mode="cpu",
                )

            torch.cuda.empty_cache()
            num_fg += num_fg_img

            # Obj: all events in this batch
            obj_target_b = fg_mask_local.unsqueeze(-1).to(dtype)
            obj_targets_list.append(obj_target_b)
            obj_preds_list.append(obj_preds_b)

            # Fg only: directly index with fg_mask_local
            if num_fg_img > 0:
                fg_bbox_preds.append(bbox_preds_b[fg_mask_local])
                fg_cls_preds.append(cls_preds_b[fg_mask_local])

                cls_target = (
                    F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).float()
                    * pred_ious_this_matching.unsqueeze(-1)
                )
                reg_target = gt_bboxes[matched_gt_inds]

                cls_targets.append(cls_target)
                reg_targets.append(reg_target)

        num_fg = max(num_fg, 1)

        # ---- IoU loss ----
        if len(fg_bbox_preds) > 0:
            fg_bbox_preds = torch.cat(fg_bbox_preds, dim=0)
            reg_targets = torch.cat(reg_targets, dim=0)
            loss_iou = self.iou_loss(fg_bbox_preds, reg_targets).sum() / num_fg
        else:
            loss_iou = torch.tensor(0.0, device=device, requires_grad=True)

        # ---- Obj loss ----
        all_obj_preds = torch.cat(obj_preds_list, dim=0)
        all_obj_targets = torch.cat(obj_targets_list, dim=0)
        loss_obj = self.obj_focal_loss(all_obj_preds, all_obj_targets).sum() / num_fg

        # ---- Cls loss ----
        if len(fg_cls_preds) > 0:
            fg_cls_preds = torch.cat(fg_cls_preds, dim=0)
            cls_targets = torch.cat(cls_targets, dim=0)
            loss_cls = self.bcewithlog_loss(fg_cls_preds, cls_targets).sum() / num_fg
        else:
            loss_cls = torch.tensor(0.0, device=device, requires_grad=True)

        reg_weight = 1.0
        total_loss = reg_weight * loss_iou + loss_obj + loss_cls

        return (
            total_loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            num_fg / max(num_gts, 1),
        )
    # ------------------------------------------------------------------
    # Inside-GT-box assignment (sparse event-based)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_assignments(
        self,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        anchor_centers,
        mode="gpu",
    ):
        """
        Assigns each event as foreground if its pixel-space position falls
        inside at least one GT bounding box.  Events inside multiple boxes
        are assigned to the smallest-area GT.

        No IoU cost, no geometry radius, no dynamic-k — so the assignment
        is stable from epoch 1 and scales to any number of sparse events.
        """
        if mode == "cpu":
            gt_bboxes_per_image    = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes             = gt_classes.cpu().float()
            anchor_centers         = anchor_centers.cpu().float()

        # GT box corners  [G, 1]
        gt_x1 = (gt_bboxes_per_image[:, 0] - gt_bboxes_per_image[:, 2] / 2).unsqueeze(1)
        gt_y1 = (gt_bboxes_per_image[:, 1] - gt_bboxes_per_image[:, 3] / 2).unsqueeze(1)
        gt_x2 = (gt_bboxes_per_image[:, 0] + gt_bboxes_per_image[:, 2] / 2).unsqueeze(1)
        gt_y2 = (gt_bboxes_per_image[:, 1] + gt_bboxes_per_image[:, 3] / 2).unsqueeze(1)

        ax = anchor_centers[:, 0].unsqueeze(0)  # [1, N]
        ay = anchor_centers[:, 1].unsqueeze(0)  # [1, N]

        # inside[g, n] = True if event n is inside GT box g  [G, N]
        inside = (ax >= gt_x1) & (ax <= gt_x2) & (ay >= gt_y1) & (ay <= gt_y2)

        fg_mask = inside.any(dim=0)   # [N]
        num_fg  = int(fg_mask.sum().item())

        if num_fg == 0:
            return (
                gt_classes.new_zeros(0),
                fg_mask,
                gt_bboxes_per_image.new_zeros(0),
                gt_classes.new_zeros(0, dtype=torch.long),
                0,
            )

        # Assign each fg event to the smallest GT it sits inside
        gt_areas  = gt_bboxes_per_image[:, 2] * gt_bboxes_per_image[:, 3]  # [G]
        area_cost = gt_areas.unsqueeze(1).expand_as(inside).clone().float()
        area_cost[~inside] = float('inf')
        matched_gt_inds = area_cost.argmin(dim=0)[fg_mask]                  # [num_fg]

        gt_matched_classes = gt_classes[matched_gt_inds]

        # Per-pair IoU(matched_gt_i, pred_i) — used to soft-weight cls targets
        fg_preds    = bboxes_preds_per_image[fg_mask]                       # [num_fg, 4]
        matched_gts = gt_bboxes_per_image[matched_gt_inds]                  # [num_fg, 4]

        a_x1y1 = matched_gts[:, :2] - matched_gts[:, 2:4] / 2
        a_x2y2 = matched_gts[:, :2] + matched_gts[:, 2:4] / 2
        b_x1y1 = fg_preds[:, :2]    - fg_preds[:, 2:4]    / 2
        b_x2y2 = fg_preds[:, :2]    + fg_preds[:, 2:4]    / 2
        inter   = (torch.min(a_x2y2, b_x2y2) - torch.max(a_x1y1, b_x1y1)).clamp(min=0).prod(1)
        area_a  = (a_x2y2 - a_x1y1).prod(1)
        area_b  = (b_x2y2 - b_x1y1).prod(1)
        pred_ious_this_matching = (inter / (area_a + area_b - inter + 1e-16)).clamp(min=0)

        if mode == "cpu":
            gt_matched_classes      = gt_matched_classes.cuda()
            fg_mask                 = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds         = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    # ------------------------------------------------------------------
    # Post-processing (inference)
    # ------------------------------------------------------------------

    def postprocess(
        self, decoded, batches, conf_thre=0.01, nms_thre=0.65,
    ):
        """
        NMS on decoded point predictions.

        Args:
            decoded: [N_total, 5+C]  decoded boxes (cx,cy,w,h) + raw logits
            batches: [N_total]       batch index per event

        Returns:
            list[dict] per image with "boxes" (xyxy), "scores", "labels"
        """
        B = int(batches.max().item()) + 1
        results = []

        for b in range(B):
            mask_b = batches == b
            det = decoded[mask_b]                              # [N_b, 5+C]

            if det.shape[0] == 0:
                results.append({
                    "boxes":  det.new_zeros(0, 4),
                    "scores": det.new_zeros(0),
                    "labels": det.new_zeros(0, dtype=torch.long),
                })
                continue

            # cxcywh → xyxy
            boxes_xyxy = det[:, :4].clone()
            boxes_xyxy[:, 0] = det[:, 0] - det[:, 2] / 2
            boxes_xyxy[:, 1] = det[:, 1] - det[:, 3] / 2
            boxes_xyxy[:, 2] = det[:, 0] + det[:, 2] / 2
            boxes_xyxy[:, 3] = det[:, 1] + det[:, 3] / 2

            obj_score = det[:, 4].sigmoid()
            cls_scores = det[:, 5:].sigmoid()
            class_conf, class_pred = cls_scores.max(dim=1)
            score = obj_score * class_conf

            keep = score >= conf_thre
            boxes_xyxy = boxes_xyxy[keep]
            score = score[keep]
            class_pred = class_pred[keep]

            if boxes_xyxy.shape[0] == 0:
                results.append({
                    "boxes":  det.new_zeros(0, 4),
                    "scores": det.new_zeros(0),
                    "labels": det.new_zeros(0, dtype=torch.long),
                })
                continue

            nms_idx = torchvision.ops.batched_nms(
                boxes_xyxy, score, class_pred, nms_thre,
            )
            results.append({
                "boxes":  boxes_xyxy[nms_idx],
                "scores": score[nms_idx],
                "labels": class_pred[nms_idx],
            })

        return results
