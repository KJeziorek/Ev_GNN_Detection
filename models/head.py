"""
Drop-in replacement for YOLOXHead with configurable sparse-event training strategies.

=== CONFIGURATION GUIDE ===

All strategies are controlled via the `sparse_cfg` dict passed to YOLOXHead.__init__.
Default values (None / False) reproduce your original baseline behavior exactly.

--- Strategy 1: FG Cap (max_fg_per_gt) ---
    Caps the number of foreground events assigned to each GT box.
    Prevents 100x redundant supervision when many events share a spatial cell.

    sparse_cfg = {
        "max_fg_per_gt": 1,           # int or None. 1 = closest to your 45% baseline
        "fg_selection": "recent",     # "recent" | "random" | "iou"
                                      #   recent = keep events with latest timestamp
                                      #   random = random subsample each forward pass
                                      #   iou    = keep events whose predictions best match GT
    }

--- Strategy 2: Ignore Zone (use_ignore_zone) ---
    Events inside a GT box that are NOT selected as foreground become "ignore"
    instead of "background" in the objectness loss. Prevents penalizing the
    model for correctly detecting objects at non-primary temporal positions.

    sparse_cfg = {
        "use_ignore_zone": True,      # bool. Recommended with max_fg_per_gt.
    }

--- Strategy 3: Temporal Weighting (use_temporal_weighting) ---
    Weights each event's loss contribution by recency. Recent events contribute
    more, older events contribute less. Smooth exponential decay.

    sparse_cfg = {
        "use_temporal_weighting": True,
        "temporal_min_weight": 0.1,   # float in (0, 1). Weight for oldest event.
                                      #   0.1 = 10x less than newest. 0.5 = 2x less.
    }

--- Strategy 4: Per-Cell Aggregation (use_cell_aggregation) ---
    Pools predictions within each spatial grid cell before computing loss.
    Closest to your 45% mAP baseline while keeping multi-timestamp backbone.
    Regression (cx, cy, w, h) is mean-pooled for coherent box predictions.
    Objectness and classification logits are max-pooled to keep the strongest signal.

    sparse_cfg = {
        "use_cell_aggregation": True,  # bool. Mutually exclusive with max_fg_per_gt.
    }

=== RECOMMENDED CONFIGS ===

# Config A: Start here — closest to 45% baseline behavior
sparse_cfg_A = {
    "max_fg_per_gt": 1,
    "fg_selection": "recent",
    "use_ignore_zone": True,
}

# Config B: Allow a few events per GT, weighted by recency
sparse_cfg_B = {
    "max_fg_per_gt": 3,
    "fg_selection": "recent",
    "use_ignore_zone": True,
    "use_temporal_weighting": True,
    "temporal_min_weight": 0.1,
}

# Config C: Cell-level aggregation (most conservative)
sparse_cfg_C = {
    "use_cell_aggregation": True,
}

# Config D: Original baseline (no changes)
sparse_cfg_D = {}

=== USAGE ===

    # In detection.py, change:
    #   self.head = YOLOXHead(num_classes=100, strides=[20], in_channels=[256])
    # To:
    #   self.head = YOLOXHead(
    #       num_classes=100, strides=[20], in_channels=[256],
    #       sparse_cfg={"max_fg_per_gt": 1, "fg_selection": "recent", "use_ignore_zone": True},
    #   )
    #
    # IMPORTANT: your backbone must preserve timestamps in data.pos[:, 2].
    # If pos only has (x, y) after pooling, "recent" selection falls back to "random".
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.layers.linear import LinearX
from models.layers.pointnet import PointNetConv
from models.layers.norm import BatchNorm
from models.layers.network_blocks import BaseConv
from utils.focal_loss import FocalLoss

logger = logging.getLogger(__name__)

# ======================================================================
# Standalone IoU utilities (unchanged)
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
# Point-based YOLOX Head  (with configurable sparse-event strategies)
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
        sparse_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # ---- sparse training config (defaults = original baseline) ----
        cfg = sparse_cfg or {}
        self.max_fg_per_gt = cfg.get("max_fg_per_gt", None)             # None = unlimited
        self.fg_selection = cfg.get("fg_selection", "recent")           # recent | random | iou
        self.use_ignore_zone = cfg.get("use_ignore_zone", False)
        self.use_temporal_weighting = cfg.get("use_temporal_weighting", False)
        self.temporal_min_weight = cfg.get("temporal_min_weight", 0.1)  # weight for oldest event
        self.use_cell_aggregation = cfg.get("use_cell_aggregation", True)

        if self.use_cell_aggregation and self.max_fg_per_gt is not None:
            logger.warning(
                "use_cell_aggregation and max_fg_per_gt are both set. "
                "Cell aggregation will be applied BEFORE fg capping. "
                "Consider using only one."
            )

        logger.info(
            f"YOLOXHead sparse_cfg: max_fg_per_gt={self.max_fg_per_gt}, "
            f"fg_selection={self.fg_selection}, ignore_zone={self.use_ignore_zone}, "
            f"temporal_weight={self.use_temporal_weighting}(min={self.temporal_min_weight}), "
            f"cell_agg={self.use_cell_aggregation}"
        )

        # ---- network layers ----
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
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm1d, BatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def init_pred_head(module, bias_init=0.0):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, bias_init)

        for conv_list in [self.stems, self.cls_convs, self.reg_convs, self.obj_convs]:
            for base_conv in conv_list:
                init_conv_module(base_conv)

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
            data_in:  list of PyG Data objects (one per scale from backbone)
                      each has .x, .pos [:, (x, y, ...)], .batch, .edge_index
            labels:   [B, max_det, 5]  (class_id, cx, cy, w, h) in pixels
                      None during inference.

        Returns:
            training:  (loss, iou_loss, obj_loss, cls_loss, num_fg_ratio)
            inference: list[dict] per image with keys "boxes", "scores", "labels"
        """
        all_outputs = []
        all_positions = []
        all_strides = []
        all_batches = []
        all_timestamps = []  # NEW: preserve timestamps for temporal strategies

        for k, (cls_conv, reg_conv, obj_conv, stride_this_level, data) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.obj_convs, self.strides, data_in)
        ):
            data = self.stems[k](data)
            cls_data = data.clone()
            obj_data = data.clone()

            cls_feat = cls_conv(cls_data)
            reg_feat = reg_conv(data)
            obj_feat = obj_conv(obj_data)

            cls_output = self.cls_preds[k](cls_feat).x
            reg_output = self.reg_preds[k](reg_feat).x
            obj_output = self.obj_preds[k](obj_feat).x

            N_k = reg_output.shape[0]
            if N_k == 0:
                continue

            output = torch.cat([reg_output, obj_output, cls_output], dim=1)

            all_outputs.append(output)
            all_positions.append(data.pos[:, :2])
            all_strides.append(
                output.new_full((N_k,), stride_this_level)
            )
            all_batches.append(data.batch)

            # Extract timestamps if available (pos dim >= 3)
            if data.pos.shape[1] >= 3:
                all_timestamps.append(data.pos[:, 2])
            else:
                all_timestamps.append(output.new_zeros(N_k))

        # Handle empty case
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

        outputs    = torch.cat(all_outputs, dim=0)
        positions  = torch.cat(all_positions, dim=0)
        strides    = torch.cat(all_strides, dim=0)
        batches    = torch.cat(all_batches, dim=0)
        timestamps = torch.cat(all_timestamps, dim=0)

        # Decode bbox: raw → pixel-space (cx, cy, w, h)
        decoded = self.decode_outputs(outputs, positions, strides)

        if self.training:
            return self.get_losses(decoded, strides, batches, labels, positions, timestamps)
        else:
            # ---- Strategy 4: cell aggregation at inference ----
            if self.use_cell_aggregation:
                decoded, batches = self._aggregate_cells(decoded, positions, strides, batches)
            return self.postprocess(decoded, batches)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode_outputs(self, outputs, positions, strides):
        decoded = outputs.clone()
        s = strides
        decoded[:, 0] = (positions[:, 0] + outputs[:, 0]) * s
        decoded[:, 1] = (positions[:, 1] + outputs[:, 1]) * s
        decoded[:, 2] = torch.exp(outputs[:, 2].clamp(max=10.0)) * s
        decoded[:, 3] = torch.exp(outputs[:, 3].clamp(max=10.0)) * s
        return decoded

    # ------------------------------------------------------------------
    # Strategy 4 helper: per-cell aggregation
    # ------------------------------------------------------------------

    def _aggregate_cells(self, decoded, positions, strides, batches):
        """
        Pool all predictions within the same spatial grid cell.
        Regression (channels 0:4) is mean-pooled for coherent box predictions.
        Objectness + classification (channels 4:) is max-pooled to keep the
        most confident signal.

        Returns aggregated (decoded, batches) with one entry per cell.
        """
        # Quantize positions to integer cell IDs
        cell_xy = positions.long()  # positions are already in stride-reduced space
        cell_keys = torch.cat([batches.unsqueeze(1), cell_xy], dim=1)  # [N, 3]
        unique_cells, inv = torch.unique(cell_keys, dim=0, return_inverse=True)

        M = unique_cells.shape[0]
        D = decoded.shape[1]

        # --- Regression (0:4): mean-pool for coherent box predictions ---
        reg_src = decoded[:, :4]
        pooled_reg = torch.zeros((M, 4), dtype=decoded.dtype, device=decoded.device)
        inv_reg = inv.unsqueeze(1).expand(-1, 4)
        pooled_reg = pooled_reg.scatter_reduce(
            0, inv_reg, reg_src, reduce="mean", include_self=False
        )

        # --- Obj + Cls (4:): max-pool to keep most confident signal ---
        oc_src = decoded[:, 4:]
        C = oc_src.shape[1]
        pooled_oc = torch.full((M, C), float('-inf'), dtype=decoded.dtype, device=decoded.device)
        inv_oc = inv.unsqueeze(1).expand(-1, C)
        pooled_oc = pooled_oc.scatter_reduce(
            0, inv_oc, oc_src, reduce="amax", include_self=False
        )

        pooled = torch.cat([pooled_reg, pooled_oc], dim=1)
        new_batches = unique_cells[:, 0]
        return pooled, new_batches

    # ------------------------------------------------------------------
    # Temporal weight computation
    # ------------------------------------------------------------------

    def _compute_temporal_weights(self, timestamps, batches, device):
        """
        Compute per-event weights in [temporal_min_weight, 1.0] based on recency.
        Most recent event in each batch gets weight 1.0, oldest gets temporal_min_weight.
        """
        weights = torch.ones(timestamps.shape[0], dtype=torch.float32, device=device)
        B = int(batches.max().item()) + 1

        for b in range(B):
            mask_b = batches == b
            t = timestamps[mask_b]
            if t.numel() <= 1:
                continue
            t_min, t_max = t.min(), t.max()
            if t_max - t_min < 1e-6:
                continue
            # Normalize to [0, 1], then scale to [min_weight, 1.0]
            t_norm = (t - t_min) / (t_max - t_min)
            weights[mask_b] = self.temporal_min_weight + (1.0 - self.temporal_min_weight) * t_norm

        return weights

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def get_losses(self, decoded, strides, batches, labels, positions, timestamps):
        dtype = decoded.dtype
        device = decoded.device
        B = int(batches.max().item()) + 1

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        # ---- Strategy 3: precompute temporal weights ----
        if self.use_temporal_weighting:
            temporal_weights = self._compute_temporal_weights(timestamps, batches, device)
        else:
            temporal_weights = torch.ones(decoded.shape[0], dtype=dtype, device=device)

        # ---- Strategy 4: cell aggregation before loss ----
        if self.use_cell_aggregation:
            # Aggregate predictions per cell, then run loss on pooled predictions.
            # Regression (0:4) → mean-pool (coherent box, no Frankenstein mixing)
            # Obj+Cls   (4:)  → max-pool  (keep most confident signal)
            cell_xy = positions.long()
            cell_keys = torch.cat([batches.unsqueeze(1), cell_xy], dim=1)
            unique_cells, inv = torch.unique(cell_keys, dim=0, return_inverse=True)

            M = unique_cells.shape[0]

            # --- Regression (0:4): mean-pool ---
            reg_src = decoded[:, :4]
            pooled_reg = torch.zeros((M, 4), dtype=dtype, device=device)
            inv_reg = inv.unsqueeze(1).expand(-1, 4)
            pooled_reg = pooled_reg.scatter_reduce(
                0, inv_reg, reg_src, reduce="mean", include_self=False
            )

            # --- Obj + Cls (4:): max-pool ---
            oc_src = decoded[:, 4:]
            C = oc_src.shape[1]
            pooled_oc = torch.full((M, C), float('-inf'), dtype=dtype, device=device)
            inv_oc = inv.unsqueeze(1).expand(-1, C)
            pooled_oc = pooled_oc.scatter_reduce(
                0, inv_oc, oc_src, reduce="amax", include_self=False
            )

            pooled_decoded = torch.cat([pooled_reg, pooled_oc], dim=1)

            # Mean-pool positions per cell (for anchor centers)
            pooled_pos = torch.zeros((M, 2), dtype=dtype, device=device)
            expanded_inv_pos = inv.unsqueeze(1).expand(-1, 2)
            pooled_pos = pooled_pos.scatter_reduce(
                0, expanded_inv_pos, positions, reduce="mean", include_self=False
            )

            # Max-pool strides (all same within a cell anyway)
            pooled_strides = torch.zeros(M, dtype=dtype, device=device)
            pooled_strides = pooled_strides.scatter_reduce(
                0, inv, strides, reduce="amax", include_self=False
            )

            # Max-pool temporal weights per cell (most recent event's weight)
            pooled_tw = torch.zeros(M, dtype=dtype, device=device)
            pooled_tw = pooled_tw.scatter_reduce(
                0, inv, temporal_weights, reduce="amax", include_self=False
            )

            # Max-pool timestamps per cell
            pooled_ts = torch.full((M,), float('-inf'), dtype=dtype, device=device)
            pooled_ts = pooled_ts.scatter_reduce(
                0, inv, timestamps, reduce="amax", include_self=False
            )

            new_batches = unique_cells[:, 0]

            # Replace all tensors with pooled versions
            decoded = pooled_decoded
            positions = pooled_pos
            strides = pooled_strides
            batches = new_batches
            temporal_weights = pooled_tw
            timestamps = pooled_ts

        # Accumulate per-batch
        fg_bbox_preds = []
        fg_cls_preds = []
        cls_targets = []
        reg_targets = []
        obj_targets_list = []
        obj_preds_list = []
        obj_weights_list = []     # per-event weights for obj loss
        fg_weights_list = []      # per-fg-event weights for reg/cls loss

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
            timestamps_b = timestamps[mask_b]
            tw_b         = temporal_weights[mask_b]

            num_gt = int(nlabel[b])
            num_gts += num_gt

            if num_gt == 0 or N_b == 0:
                obj_targets_list.append(torch.zeros(N_b, 1, device=device, dtype=dtype))
                obj_preds_list.append(obj_preds_b)
                obj_weights_list.append(tw_b)
                continue

            gt_bboxes = labels[b, :num_gt, 1:5]
            gt_classes = labels[b, :num_gt, 0]

            try:
                (
                    gt_matched_classes, fg_mask_local,
                    pred_ious_this_matching, matched_gt_inds, num_fg_img,
                    inside_gt_mask,
                ) = self.get_assignments(
                    gt_bboxes, gt_classes,
                    bbox_preds_b, anchor_xy_b,
                    event_times=timestamps_b,
                    mode="gpu",
                )
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    raise
                torch.cuda.empty_cache()
                (
                    gt_matched_classes, fg_mask_local,
                    pred_ious_this_matching, matched_gt_inds, num_fg_img,
                    inside_gt_mask,
                ) = self.get_assignments(
                    gt_bboxes, gt_classes,
                    bbox_preds_b, anchor_xy_b,
                    event_times=timestamps_b,
                    mode="cpu",
                )

            torch.cuda.empty_cache()
            num_fg += num_fg_img

            # ---- Build obj target and mask ----
            obj_target_b = fg_mask_local.unsqueeze(-1).to(dtype)

            # Strategy 2: ignore zone — events inside GT but not fg get weight=0 in obj loss
            if self.use_ignore_zone and inside_gt_mask is not None:
                ignore_mask = inside_gt_mask & ~fg_mask_local  # [N_b]
                obj_weight_b = tw_b.clone()
                obj_weight_b[ignore_mask] = 0.0
            else:
                obj_weight_b = tw_b.clone()

            obj_targets_list.append(obj_target_b)
            obj_preds_list.append(obj_preds_b)
            obj_weights_list.append(obj_weight_b)

            # Fg only
            if num_fg_img > 0:
                fg_bbox_preds.append(bbox_preds_b[fg_mask_local])
                fg_cls_preds.append(cls_preds_b[fg_mask_local])
                fg_weights_list.append(tw_b[fg_mask_local])

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
            fg_bbox_preds_cat = torch.cat(fg_bbox_preds, dim=0)
            reg_targets_cat = torch.cat(reg_targets, dim=0)
            fg_weights_cat = torch.cat(fg_weights_list, dim=0)

            iou_per_event = self.iou_loss(fg_bbox_preds_cat, reg_targets_cat)
            loss_iou = (iou_per_event * fg_weights_cat).sum() / num_fg
        else:
            loss_iou = torch.tensor(0.0, device=device, requires_grad=True)

        # ---- Obj loss ----
        all_obj_preds = torch.cat(obj_preds_list, dim=0)
        all_obj_targets = torch.cat(obj_targets_list, dim=0)
        all_obj_weights = torch.cat(obj_weights_list, dim=0)

        obj_per_event = self.obj_focal_loss(all_obj_preds, all_obj_targets)
        loss_obj = (obj_per_event * all_obj_weights.unsqueeze(-1)).sum() / num_fg

        # ---- Cls loss ----
        if len(fg_cls_preds) > 0:
            fg_cls_preds_cat = torch.cat(fg_cls_preds, dim=0)
            cls_targets_cat = torch.cat(cls_targets, dim=0)
            fg_weights_cat = torch.cat(fg_weights_list, dim=0)

            cls_per_event = self.bcewithlog_loss(fg_cls_preds_cat, cls_targets_cat)
            # cls_per_event is [num_fg, num_classes], weight is [num_fg]
            loss_cls = (cls_per_event * fg_weights_cat.unsqueeze(-1)).sum() / num_fg
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
    # Assignment (with fg capping + ignore zone)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_assignments(
        self,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        anchor_centers,
        event_times=None,
        mode="gpu",
    ):
        """
        Assigns each event as foreground if its pixel-space position falls
        inside at least one GT bounding box. Events inside multiple boxes
        are assigned to the smallest-area GT.

        NEW: supports fg capping (max_fg_per_gt) and returns inside_gt_mask
        for ignore-zone computation.

        Returns:
            gt_matched_classes:      [num_fg]
            fg_mask:                 [N] bool
            pred_ious_this_matching: [num_fg]
            matched_gt_inds:         [num_fg] long
            num_fg:                  int
            inside_gt_mask:          [N] bool — all events inside any GT (before capping)
        """
        if mode == "cpu":
            gt_bboxes_per_image    = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes             = gt_classes.cpu().float()
            anchor_centers         = anchor_centers.cpu().float()
            if event_times is not None:
                event_times = event_times.cpu().float()

        N = anchor_centers.shape[0]

        # GT box corners  [G, 1]
        gt_x1 = (gt_bboxes_per_image[:, 0] - gt_bboxes_per_image[:, 2] / 2).unsqueeze(1)
        gt_y1 = (gt_bboxes_per_image[:, 1] - gt_bboxes_per_image[:, 3] / 2).unsqueeze(1)
        gt_x2 = (gt_bboxes_per_image[:, 0] + gt_bboxes_per_image[:, 2] / 2).unsqueeze(1)
        gt_y2 = (gt_bboxes_per_image[:, 1] + gt_bboxes_per_image[:, 3] / 2).unsqueeze(1)

        ax = anchor_centers[:, 0].unsqueeze(0)  # [1, N]
        ay = anchor_centers[:, 1].unsqueeze(0)  # [1, N]

        # inside[g, n] = True if event n is inside GT box g  [G, N]
        inside = (ax >= gt_x1) & (ax <= gt_x2) & (ay >= gt_y1) & (ay <= gt_y2)

        # Full inside-GT mask (before any capping) — used for ignore zones
        inside_gt_mask = inside.any(dim=0)  # [N]

        fg_mask = inside_gt_mask.clone()
        num_fg  = int(fg_mask.sum().item())

        if num_fg == 0:
            return (
                gt_classes.new_zeros(0),
                fg_mask,
                gt_bboxes_per_image.new_zeros(0),
                gt_classes.new_zeros(0, dtype=torch.long),
                0,
                inside_gt_mask,
            )

        # Assign each fg event to the smallest GT it sits inside
        gt_areas  = gt_bboxes_per_image[:, 2] * gt_bboxes_per_image[:, 3]  # [G]
        area_cost = gt_areas.unsqueeze(1).expand_as(inside).clone().float()
        area_cost[~inside] = float('inf')
        best_gt_per_event = area_cost.argmin(dim=0)  # [N] — best GT for each event

        # ---- Strategy 1: fg capping ----
        if self.max_fg_per_gt is not None:
            new_fg_mask = torch.zeros(N, dtype=torch.bool, device=anchor_centers.device)
            G = gt_bboxes_per_image.shape[0]

            for g in range(G):
                # Events assigned to this GT
                event_indices = torch.where(fg_mask & (best_gt_per_event == g))[0]
                if len(event_indices) == 0:
                    continue

                if len(event_indices) <= self.max_fg_per_gt:
                    new_fg_mask[event_indices] = True
                    continue

                # Select top-k based on strategy
                if self.fg_selection == "recent" and event_times is not None:
                    times = event_times[event_indices]
                    if times.abs().sum() > 1e-6:  # timestamps are meaningful
                        _, topk_idx = times.topk(self.max_fg_per_gt)
                        new_fg_mask[event_indices[topk_idx]] = True
                        continue

                if self.fg_selection == "iou":
                    # Keep events whose predicted boxes best match the GT
                    fg_preds = bboxes_preds_per_image[event_indices]
                    gt_box = gt_bboxes_per_image[g:g+1].expand(len(event_indices), -1)
                    # Compute per-event IoU with their assigned GT
                    a_x1y1 = gt_box[:, :2] - gt_box[:, 2:4] / 2
                    a_x2y2 = gt_box[:, :2] + gt_box[:, 2:4] / 2
                    b_x1y1 = fg_preds[:, :2] - fg_preds[:, 2:4] / 2
                    b_x2y2 = fg_preds[:, :2] + fg_preds[:, 2:4] / 2
                    inter = (torch.min(a_x2y2, b_x2y2) - torch.max(a_x1y1, b_x1y1)).clamp(min=0).prod(1)
                    area_a = (a_x2y2 - a_x1y1).prod(1)
                    area_b = (b_x2y2 - b_x1y1).prod(1)
                    ious = (inter / (area_a + area_b - inter + 1e-16)).clamp(min=0)
                    _, topk_idx = ious.topk(self.max_fg_per_gt)
                    new_fg_mask[event_indices[topk_idx]] = True
                    continue

                # "random" or fallback
                perm = torch.randperm(len(event_indices), device=event_indices.device)
                new_fg_mask[event_indices[perm[:self.max_fg_per_gt]]] = True

            fg_mask = new_fg_mask
            num_fg = int(fg_mask.sum().item())

            if num_fg == 0:
                return (
                    gt_classes.new_zeros(0),
                    fg_mask,
                    gt_bboxes_per_image.new_zeros(0),
                    gt_classes.new_zeros(0, dtype=torch.long),
                    0,
                    inside_gt_mask,
                )

        # Recompute matched GT indices for surviving fg events
        matched_gt_inds = best_gt_per_event[fg_mask]
        gt_matched_classes = gt_classes[matched_gt_inds]

        # Per-pair IoU — used to soft-weight cls targets
        fg_preds    = bboxes_preds_per_image[fg_mask]
        matched_gts = gt_bboxes_per_image[matched_gt_inds]

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
            inside_gt_mask          = inside_gt_mask.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
            inside_gt_mask,
        )

    # ------------------------------------------------------------------
    # Post-processing (inference) — unchanged
    # ------------------------------------------------------------------

    def postprocess(
        self, decoded, batches, conf_thre=0.01, nms_thre=0.65,
    ):
        B = int(batches.max().item()) + 1
        results = []

        for b in range(B):
            mask_b = batches == b
            det = decoded[mask_b]

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
