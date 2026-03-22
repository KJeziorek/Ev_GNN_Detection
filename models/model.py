import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.backbone import BACKBONE
from models.head import HEAD
from models.yolox.utils.losses import IOUloss
from models.yolox.utils.boxes import bboxes_iou
from models.yolox.utils.compat import meshgrid

logger = logging.getLogger(__name__)


class Detection(nn.Module):
    """
    GNN-based event detection model with YOLOX-style loss and inference.

    Architecture:
        GraphData → Backbone (3 scales) → GNN Heads (per-node cls/reg/obj)
        → Scatter to dense [B, C, H, W] grids → YOLOX grid decode + SimOTA loss

    Training:  returns dict with total_loss, iou_loss, conf_loss, cls_loss, ...
    Inference: returns list of dicts with boxes, scores, labels (after NMS)
    """

    def __init__(self, num_classes=2, spatial_range=(240, 180)):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_range = spatial_range

        # GNN backbone (multi-scale)
        self.backbone = BACKBONE()

        # GNN detection heads (one per scale)
        self.head1 = HEAD(32, num_classes)    # scale 0: 32ch from block2
        self.head2 = HEAD(64, num_classes)    # scale 1: 64ch from block3
        self.head3 = HEAD(128, num_classes)   # scale 2: 128ch from block4

        # Grid configs: (H, W) per scale, matching backbone pool layers
        self.grid_configs = [(60, 80), (30, 40), (15, 20)]

        # Strides per scale: spatial_range / grid_width
        self.strides = [
            spatial_range[0] / 80,    # 3.0
            spatial_range[0] / 40,    # 6.0
            spatial_range[0] / 20,    # 12.0
        ]

        # YOLOX loss infrastructure
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * 3
        self.decode_in_inference = True

    def forward(self, data, targets=None):
        """
        Args:
            data:    GraphData with x, pos, edge_index, batch, bboxes, batch_bb
            targets: optional [B, max_obj, 5] labels [class_id, cx, cy, w, h].
                     If None during training, built from data.bboxes/batch_bb.
        Returns:
            training:  dict with loss components
            inference: list of dicts with boxes/scores/labels (after NMS)
        """
        # Save original batch tensor before backbone mutates data in-place
        original_batch = data.batch.clone() if data.batch is not None else None

        # 1. Backbone → multi-scale graph features
        feats = self.backbone(data)

        # 2. GNN heads → per-node predictions, scatter to dense grids
        heads = [self.head1, self.head2, self.head3]
        dense_maps = []
        for i, (feat, head) in enumerate(zip(feats, heads)):
            cls, reg, obj = head(feat)
            dense = self._scatter_to_dense(
                cls, reg, obj, feat.pos, feat.batch, i
            )
            dense_maps.append(dense)

        # 3. YOLOX-style loss or inference
        if self.training:
            if targets is None:
                data.batch = original_batch
                targets = self._prepare_labels(data)
            return self._training_step(dense_maps, targets)
        else:
            return self._inference_step(dense_maps)

    # ------------------------------------------------------------------
    # Scatter sparse GNN predictions → dense grid
    # ------------------------------------------------------------------

    def _scatter_to_dense(self, cls, reg, obj, pos, batch, scale_idx):
        """
        Scatter per-node GNN predictions into a dense [B, C, H, W] grid.

        Uses scatter_mean (scatter_add normalised by per-cell count) so that
        cells with many contributing nodes don't produce extreme values
        before the exp() decode.
        """
        H, W = self.grid_configs[scale_idx]
        stride = self.strides[scale_idx]
        C = 4 + 1 + self.num_classes
        device = cls.device
        dtype = cls.dtype

        if batch is None or batch.numel() == 0:
            B = 1
            batch_ids = torch.zeros(cls.shape[0], dtype=torch.long, device=device)
        else:
            B = batch.max().item() + 1
            batch_ids = batch

        if cls.shape[0] == 0:
            return torch.zeros(B, C, H, W, device=device, dtype=dtype)

        # Route to grid cells (stride-aligned with YOLOX grid)
        col = (pos[:, 0] / stride).long().clamp(0, W - 1)
        row = (pos[:, 1] / stride).long().clamp(0, H - 1)

        # Channel order: [reg(4), obj(1), cls(num_cls)]
        combined = torch.cat([reg, obj, cls], dim=-1)

        flat_idx = batch_ids * (H * W) + row * W + col

        # Scatter-add
        dense_flat = torch.zeros(B * H * W, C, device=device, dtype=dtype)
        dense_flat.scatter_add_(
            0, flat_idx.unsqueeze(-1).expand_as(combined), combined
        )

        # Normalise by count → scatter_mean
        counts = torch.zeros(B * H * W, 1, device=device, dtype=dtype)
        counts.scatter_add_(
            0, flat_idx.unsqueeze(-1),
            torch.ones(flat_idx.shape[0], 1, device=device, dtype=dtype),
        )
        dense_flat = dense_flat / counts.clamp(min=1)

        return dense_flat.view(B, H, W, C).permute(0, 3, 1, 2)

    # ------------------------------------------------------------------
    # Label preparation
    # ------------------------------------------------------------------

    def _prepare_labels(self, data):
        """
        Convert collated bboxes to YOLOX label format.

        Input:  data.bboxes  [N_bb, 5] = [class_id, x_tl, y_tl, w, h]
                data.batch_bb [N_bb]    = sample index per bbox
        Output: [B, max_obj, 5]        = [class_id, cx, cy, w, h]
        """
        bboxes = data.bboxes
        batch_bb = data.batch_bb
        B = data.batch.max().item() + 1 if data.batch is not None else 1

        if bboxes is None or bboxes.numel() == 0:
            return torch.zeros(B, 1, 5, device=data.x.device)

        # Top-left → centre format
        labels = bboxes.clone()
        labels[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2  # cx = x + w/2
        labels[:, 2] = bboxes[:, 2] + bboxes[:, 4] / 2  # cy = y + h/2

        # Pad to [B, max_obj, 5]
        counts = torch.zeros(B, dtype=torch.long, device=bboxes.device)
        counts.scatter_add_(
            0, batch_bb, torch.ones_like(batch_bb, dtype=torch.long)
        )
        max_obj = max(counts.max().item(), 1)

        padded = torch.zeros(B, max_obj, 5, device=bboxes.device, dtype=bboxes.dtype)
        for b in range(B):
            mask = batch_bb == b
            n = mask.sum()
            if n > 0:
                padded[b, :n] = labels[mask]

        return padded

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _training_step(self, dense_maps, targets):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (dense, stride) in enumerate(zip(dense_maps, self.strides)):
            output, grid = self.get_output_and_grid(
                dense, k, stride, dense.type()
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(stride).type_as(dense)
            )
            if self.use_l1:
                batch_size = dense.shape[0]
                hsize, wsize = dense.shape[-2:]
                reg_out = dense[:, :4].view(batch_size, 1, 4, hsize, wsize)
                reg_out = reg_out.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_out.clone())
            outputs.append(output)

        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.get_losses(
            x_shifts, y_shifts, expanded_strides,
            targets, torch.cat(outputs, 1),
            origin_preds, dtype=dense_maps[0].dtype,
        )

        return {
            "total_loss": loss,
            "iou_loss": iou_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss,
            "l1_loss": l1_loss,
            "num_fg": num_fg,
        }

    # ------------------------------------------------------------------
    # Inference step
    # ------------------------------------------------------------------

    def _inference_step(self, dense_maps):
        outputs = []
        for dense in dense_maps:
            reg = dense[:, :4]
            obj = dense[:, 4:5]
            cls = dense[:, 5:]
            output = torch.cat([reg, obj.sigmoid(), cls.sigmoid()], 1)
            outputs.append(output)

        self.hw = [x.shape[-2:] for x in outputs]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)

        if self.decode_in_inference:
            outputs = self.decode_outputs(outputs, dtype=dense_maps[0].type())

        return self.postprocess(outputs)

    def postprocess(self, prediction, conf_thre=0.01, nms_thre=0.65):
        """NMS post-processing on decoded [B, N_anchors, 5+num_cls] predictions."""
        # cxcywh → xyxy
        boxes = prediction.clone()
        boxes[..., :2] = prediction[..., :2] - prediction[..., 2:4] / 2
        boxes[..., 2:4] = prediction[..., :2] + prediction[..., 2:4] / 2

        results = []
        for i in range(boxes.shape[0]):
            det = boxes[i]
            obj_scores = det[:, 4]
            cls_scores = det[:, 5:]

            class_conf, class_pred = cls_scores.max(dim=1)
            score = obj_scores * class_conf
            keep = score >= conf_thre

            det = det[keep]
            score = score[keep]
            class_pred = class_pred[keep]

            if det.shape[0] == 0:
                results.append({
                    "boxes": torch.zeros(0, 4, device=det.device),
                    "scores": torch.zeros(0, device=det.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=det.device),
                })
                continue

            nms_idx = torchvision.ops.batched_nms(
                det[:, :4], score, class_pred, nms_thre,
            )
            results.append({
                "boxes": det[nms_idx, :4],
                "scores": score[nms_idx],
                "labels": class_pred[nms_idx],
            })

        return results

    # ------------------------------------------------------------------
    # YOLOX grid decode
    # ------------------------------------------------------------------

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = (
                torch.stack((xv, yv), 2)
                .view(1, 1, hsize, wsize, 2)
                .type(dtype)
            )
            self.grids[k] = grid
        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        xy = (output[..., :2] + grid) * stride
        wh = torch.exp(output[..., 2:4].clamp(max=10.0)) * stride
        output = torch.cat([xy, wh, output[..., 4:]], dim=-1)
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4].clamp(max=10.0)) * strides,
            outputs[..., 4:],
        ], dim=-1)
        return outputs

    # ------------------------------------------------------------------
    # YOLOX loss computation (SimOTA)
    # ------------------------------------------------------------------

    def get_losses(self, x_shifts, y_shifts, expanded_strides,
                   labels, outputs, origin_preds, dtype):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes, fg_mask,
                        pred_ious_this_matching, matched_gt_inds, num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides,
                        x_shifts, y_shifts, cls_preds, obj_preds,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        raise
                    logger.error(
                        "OOM during label assignment — falling back to CPU"
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes, fg_mask,
                        pred_ious_this_matching, matched_gt_inds, num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides,
                        x_shifts, y_shifts, cls_preds, obj_preds, "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = (
                    F.one_hot(
                        gt_matched_classes.to(torch.int64), self.num_classes
                    )
                    * pred_ious_this_matching.unsqueeze(-1)
                )
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts,
                      eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    # ------------------------------------------------------------------
    # SimOTA assignment
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, gt_bboxes_per_image,
                        gt_classes, bboxes_preds_per_image, expanded_strides,
                        x_shifts, y_shifts, cls_preds, obj_preds, mode="gpu"):
        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(
            gt_bboxes_per_image, bboxes_preds_per_image, False
        )

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(
                    1, num_in_boxes_anchor, 1
                ),
                reduction="none",
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg, gt_matched_classes,
            pred_ious_this_matching, matched_gt_inds,
        ) = self.simota_matching(
            cost, pair_wise_ious, gt_classes, num_gt, fg_mask
        )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes, fg_mask, pred_ious_this_matching,
            matched_gt_inds, num_fg,
        )

    def get_geometry_constraint(self, gt_bboxes_per_image, expanded_strides,
                                x_shifts, y_shifts):
        expanded_strides_per_image = expanded_strides[0]
        x_centers = (
            (x_shifts[0] + 0.5) * expanded_strides_per_image
        ).unsqueeze(0)
        y_centers = (
            (y_shifts[0] + 0.5) * expanded_strides_per_image
        ).unsqueeze(0)

        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_l = gt_bboxes_per_image[:, 0:1] - center_dist
        gt_r = gt_bboxes_per_image[:, 0:1] + center_dist
        gt_t = gt_bboxes_per_image[:, 1:2] - center_dist
        gt_b = gt_bboxes_per_image[:, 1:2] + center_dist

        c_l = x_centers - gt_l
        c_r = gt_r - x_centers
        c_t = y_centers - gt_t
        c_b = gt_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt,
                        fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(
                cost[:, multiple_match_mask], dim=0
            )
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (
            (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        )
        return (
            num_fg, gt_matched_classes,
            pred_ious_this_matching, matched_gt_inds,
        )
