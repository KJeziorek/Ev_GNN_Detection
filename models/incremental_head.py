"""
Incremental Detection Head with EMA Spatial Accumulator.

Drop-in replacement for models/head.py YOLOXHead.
Uses the same BaseConv / LinearX / BatchNorm layers from the codebase.

Training:  all events processed in parallel via temporally-weighted scatter_add.
Inference: batch mode same as training, or event-by-event EMA for hardware.

Grid state per cell: [conf, cx, cy, w, h, cls_0..cls_K]
All channels are EMA'd — no counts, no overflow, bounded forever.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.layers.linear import LinearX
from models.layers.norm import BatchNorm
from models.layers.network_blocks import BaseConv


# ======================================================================
# CenterNet-style Focal Loss
# ======================================================================

class CenterNetFocalLoss(nn.Module):
    """
    Penalty-reduced focal loss for gaussian heatmap targets.

    Positive locations (target >= 0.99): standard focal term.
    Negative locations: down-weighted by (1 - target)^beta,
        so cells near a GT center are penalised less.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, 1, H, W]  raw logits (pre-sigmoid)
            target: [B, 1, H, W]  gaussian heatmap in [0, 1]
        """
        pred_sig = pred.sigmoid()

        pos_mask = target.ge(0.99).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = -(
            (1 - pred_sig).pow(self.alpha)
            * torch.log(pred_sig.clamp(min=1e-8))
            * pos_mask
        )

        neg_weight = (1 - target).pow(self.beta)
        neg_loss = -(
            neg_weight
            * pred_sig.pow(self.alpha)
            * torch.log((1 - pred_sig).clamp(min=1e-8))
            * neg_mask
        )

        return pos_loss + neg_loss


# ======================================================================
# Incremental Detection Head
# ======================================================================

class IncrementalDetectionHead(nn.Module):
    """
    Each event predicts a *vote* — an offset toward the object center it
    belongs to, plus size / class / confidence.  Votes are scattered into
    a fixed spatial grid.  Detections emerge from grid peaks.

    Training:   scatter_add with exponential-decay temporal weights.
    Inference:  same scatter (batch), or per-event EMA for hardware.
    Hardware:   grid = SRAM.  Per event: MLP -> 1 read-modify-write.
    """

    def __init__(
        self,
        num_classes,
        strides=(12,),
        in_channels=(256,),
        grid_size=(15, 20),       # (H, W) — matches backbone output at stride 12
        ema_alpha=0.10,           # EMA blending factor (hardware inference)
        decay_lambda=0.005,       # temporal decay rate (training weighting)
        conf_threshold=0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stride = strides[0]
        self.grid_h, self.grid_w = grid_size
        self.ema_alpha = ema_alpha
        self.decay_lambda = decay_lambda
        self.conf_threshold = conf_threshold

        ch = in_channels[0]
        # Total grid channels: conf(1) + reg(4) + cls(C)
        self.cell_channels = 1 + 4 + num_classes

        # ---- Voting network (mirrors YOLOX branch structure) ----
        self.stem = BaseConv(ch, ch)

        # Vote branch — offset to center + confidence weight
        self.vote_conv = BaseConv(ch, ch)
        self.vote_offset = LinearX(ch, 2)   # dx, dy to object center
        self.vote_conf = LinearX(ch, 1)     # how much this vote matters

        # Regression branch — object size at the voted center
        self.reg_conv = BaseConv(ch, ch)
        self.vote_size = LinearX(ch, 2)     # log(w), log(h)

        # Classification branch
        self.cls_conv = BaseConv(ch, ch)
        self.vote_cls = LinearX(ch, num_classes)

        # ---- Losses ----
        self.heatmap_focal = CenterNetFocalLoss(alpha=2.0, beta=4.0)

        self._init_weights()

    # ------------------------------------------------------------------
    #  Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, prior_prob=0.01):
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # Shared convs — Kaiming, matching existing YOLOX head pattern
        for base_conv in [self.stem, self.vote_conv, self.reg_conv, self.cls_conv]:
            nn.init.kaiming_normal_(
                base_conv.conv.linear.weight, nonlinearity="relu"
            )
            nn.init.zeros_(base_conv.conv.linear.bias)
            nn.init.kaiming_normal_(
                base_conv.conv.global_nn.weight, nonlinearity="relu"
            )

        # Prediction heads — small weights
        for pred in [self.vote_offset, self.vote_size]:
            nn.init.normal_(pred.weight, std=0.01)
            nn.init.zeros_(pred.bias)

        # Confidence / cls — prior-bias trick (initial sigmoid ~ 0.01)
        nn.init.normal_(self.vote_conf.weight, std=0.01)
        nn.init.constant_(self.vote_conf.bias, bias_value)

        nn.init.normal_(self.vote_cls.weight, std=0.01)
        nn.init.constant_(self.vote_cls.bias, bias_value)

    # ------------------------------------------------------------------
    #  Forward (matches YOLOXHead signature exactly)
    # ------------------------------------------------------------------

    def forward(self, data_in, labels=None, data_orig=None,
                return_detections=False):
        """
        Args:
            data_in: list[GraphData]  — one per scale from backbone
            labels:  [B, max_det, 5]  — (class_id, cx, cy, w, h) in pixels
            data_orig: unused, kept for interface compatibility
            return_detections: if True AND training, return (loss_tuple, detections)
                               useful for validation where you need both

        Returns:
            training:  (total_loss, reg_loss, heatmap_loss, cls_vote_loss, fg_ratio)
                       or ((total_loss, ...), detections) if return_detections=True
            inference: list[dict] per image  {boxes, scores, labels}
        """
        data = data_in[0]  # single scale

        # ---- empty guard ----
        if data.x.shape[0] == 0:
            device = data.x.device
            if self.training:
                z = torch.tensor(0.0, device=device, requires_grad=True)
                return z, z, z, z, 0.0
            B = int(data.batch.max().item()) + 1 if data.batch is not None else 1
            return [
                {
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
                for _ in range(B)
            ]

        # ---- 1. Voting MLPs ----
        data = self.stem(data)

        # vote branch (two heads share the same conv features)
        vote_feat = self.vote_conv(data.clone())
        offsets = self.vote_offset(vote_feat.clone()).x      # [N, 2]
        conf_logits = self.vote_conf(vote_feat).x            # [N, 1]

        # regression branch
        reg_feat = self.reg_conv(data.clone())
        size_logits = self.vote_size(reg_feat).x             # [N, 2]

        # classification branch
        cls_feat = self.cls_conv(data.clone())
        cls_logits = self.vote_cls(cls_feat).x               # [N, C]

        # ---- 2. Voted centres ----
        event_pos = data.pos[:, :2]                          # [N, 2] grid coords
        voted_centres = event_pos + offsets                   # [N, 2]
        sizes = torch.exp(size_logits.clamp(max=8.0))        # [N, 2] w,h grid

        # ---- 3. Pack per-event vote vector ----
        #  [conf(1), cx(1), cy(1), w(1), h(1), cls(C)]
        vote_data = torch.cat(
            [conf_logits, voted_centres, sizes, cls_logits], dim=-1
        )

        # ---- 4. Timestamps ----
        timestamps = data.pos[:, 2] if data.pos.shape[1] > 2 else None

        # ---- 5. Scatter into grid ----
        grid = self._scatter_to_grid(vote_data, voted_centres, data.batch, timestamps)

        B = int(data.batch.max().item()) + 1

        if self.training or (labels is not None and return_detections):
            losses = self._compute_losses(
                grid, event_pos, offsets, data.batch, labels, B
            )
            if return_detections:
                detections = self._extract_detections(grid, B)
                return losses, detections
            return losses
        return self._extract_detections(grid, B)

    # ------------------------------------------------------------------
    #  Grid accumulation  (training: weighted scatter_add)
    # ------------------------------------------------------------------

    def _scatter_to_grid(self, vote_data, voted_centres, batch, timestamps):
        """
        Scatter votes into [B, C+1, H, W] grid.

        The extra +1 channel accumulates the temporal weight sum,
        used to normalise the other channels (soft count that
        naturally decays — no overflow ever).
        """
        N, C = vote_data.shape
        B = int(batch.max().item()) + 1
        device, dtype = vote_data.device, vote_data.dtype

        # -- temporal weights --
        if timestamps is not None and timestamps.numel() > 0:
            ts = timestamps.to(device=device, dtype=dtype)
            t_max = torch.full((B,), -1e9, device=device, dtype=dtype)
            t_max.scatter_reduce_(0, batch, ts, reduce="amax", include_self=False)
            dt = (t_max[batch] - ts).clamp(min=0.0)
            weights = torch.exp(-self.decay_lambda * dt)
        else:
            weights = torch.ones(N, device=device, dtype=dtype)

        # -- weight the votes and append weight-sum channel --
        weighted = vote_data * weights.unsqueeze(-1)                 # [N, C]
        full = torch.cat([weighted, weights.unsqueeze(-1)], dim=-1)  # [N, C+1]

        # -- grid-cell indices --
        gi = voted_centres[:, 0].long().clamp(0, self.grid_w - 1)
        gj = voted_centres[:, 1].long().clamp(0, self.grid_h - 1)
        flat = batch * (self.grid_h * self.grid_w) + gj * self.grid_w + gi

        # -- scatter_add --
        C1 = C + 1
        grid_flat = torch.zeros(
            B * self.grid_h * self.grid_w, C1, device=device, dtype=dtype
        )
        grid_flat.scatter_add_(0, flat.unsqueeze(-1).expand(-1, C1), full)

        return grid_flat.reshape(B, self.grid_h, self.grid_w, C1).permute(0, 3, 1, 2)

    # ------------------------------------------------------------------
    #  Read normalised grid
    # ------------------------------------------------------------------

    def _read_grid(self, grid):
        """
        Returns normalised (conf_logit, reg, cls_logit) from accumulated grid.
        Divides by weight_sum so that the values are temporal-weighted averages.
        Empty cells (no votes) get a large negative conf logit so sigmoid → 0.
        """
        C = self.num_classes
        ws = grid[:, -1:]
        empty = (ws < 1e-4)
        ws = ws.clamp(min=1e-6)
        conf = grid[:, 0:1] / ws
        conf = torch.where(empty, torch.tensor(-10.0, device=conf.device, dtype=conf.dtype), conf)
        reg = grid[:, 1:5] / ws
        cls = grid[:, 5 : 5 + C] / ws
        return conf, reg, cls, ws

    # ------------------------------------------------------------------
    #  Gaussian heatmap targets
    # ------------------------------------------------------------------

    def _build_targets(self, labels, B, device, dtype):
        heatmap = torch.zeros(B, 1, self.grid_h, self.grid_w, device=device, dtype=dtype)
        reg_tgt = torch.zeros(B, 4, self.grid_h, self.grid_w, device=device, dtype=dtype)
        cls_tgt = torch.zeros(
            B, self.num_classes, self.grid_h, self.grid_w, device=device, dtype=dtype
        )
        center_mask = torch.zeros(B, self.grid_h, self.grid_w, dtype=torch.bool, device=device)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        num_gts = 0

        s = float(self.stride)

        for b in range(B):
            num_gt = int(nlabel[b])
            if num_gt == 0:
                continue
            num_gts += num_gt

            gt_cls = labels[b, :num_gt, 0].long()
            gt_box = labels[b, :num_gt, 1:5]          # cx, cy, w, h  pixels

            gcx = gt_box[:, 0] / s
            gcy = gt_box[:, 1] / s
            gw = gt_box[:, 2] / s
            gh = gt_box[:, 3] / s

            for g in range(num_gt):
                cx_g, cy_g = gcx[g].item(), gcy[g].item()
                w_g, h_g = gw[g].item(), gh[g].item()

                sigma_x = max(w_g / 6.0, 0.5)
                sigma_y = max(h_g / 6.0, 0.5)
                radius = int(max(sigma_x, sigma_y) * 3) + 1
                ci, cj = int(cx_g), int(cy_g)

                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        gxi = ci + dx
                        gyj = cj + dy
                        if 0 <= gxi < self.grid_w and 0 <= gyj < self.grid_h:
                            val = math.exp(
                                -0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2)
                            )
                            if val > heatmap[b, 0, gyj, gxi].item():
                                heatmap[b, 0, gyj, gxi] = val

                if 0 <= ci < self.grid_w and 0 <= cj < self.grid_h:
                    reg_tgt[b, :, cj, ci] = torch.tensor(
                        [cx_g, cy_g, w_g, h_g], device=device, dtype=dtype
                    )
                    cls_tgt[b, gt_cls[g], cj, ci] = 1.0
                    center_mask[b, cj, ci] = True

        return heatmap, reg_tgt, cls_tgt, center_mask, num_gts

    # ------------------------------------------------------------------
    #  Loss
    # ------------------------------------------------------------------

    def _compute_losses(self, grid, event_pos, offsets, batch, labels, B):
        device, dtype = grid.device, grid.dtype

        conf_acc, reg_acc, cls_acc, _ = self._read_grid(grid)

        heatmap, reg_tgt, cls_tgt, center_mask, num_gts = self._build_targets(
            labels, B, device, dtype
        )

        if num_gts == 0:
            z = torch.tensor(0.0, device=device, requires_grad=True)
            return z, z, z, z, 0.0

        # 1 — heatmap focal loss on confidence
        loss_hm = self.heatmap_focal(conf_acc, heatmap).sum() / max(num_gts, 1)

        # 2 — regression at GT centres
        loss_reg = torch.tensor(0.0, device=device, dtype=dtype)
        for b in range(B):
            m = center_mask[b]
            if m.any():
                loss_reg = loss_reg + F.smooth_l1_loss(
                    reg_acc[b, :, m], reg_tgt[b, :, m], reduction="sum"
                )
        loss_reg = loss_reg / max(num_gts, 1)

        # 3 — classification at GT centres
        loss_cls = torch.tensor(0.0, device=device, dtype=dtype)
        for b in range(B):
            m = center_mask[b]
            if m.any():
                loss_cls = loss_cls + F.binary_cross_entropy_with_logits(
                    cls_acc[b, :, m], cls_tgt[b, :, m], reduction="sum"
                )
        loss_cls = loss_cls / max(num_gts, 1)

        # 4 — per-event vote-offset loss (fg events -> GT centre)
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        loss_vote = torch.tensor(0.0, device=device, dtype=dtype)
        num_fg = 0
        s = float(self.stride)

        for b in range(B):
            mask_b = batch == b
            if not mask_b.any():
                continue
            num_gt = int(nlabel[b])
            if num_gt == 0:
                continue

            pos_b = event_pos[mask_b]        # [N_b, 2] grid coords
            off_b = offsets[mask_b]           # [N_b, 2]

            gt_box = labels[b, :num_gt, 1:5]
            gt_cx = gt_box[:, 0] / s
            gt_cy = gt_box[:, 1] / s
            gt_w = gt_box[:, 2] / s
            gt_h = gt_box[:, 3] / s

            # which events sit inside a GT box?  (grid coords)
            inside = (
                (pos_b[:, 0:1] >= (gt_cx - gt_w / 2).unsqueeze(0))
                & (pos_b[:, 0:1] <= (gt_cx + gt_w / 2).unsqueeze(0))
                & (pos_b[:, 1:2] >= (gt_cy - gt_h / 2).unsqueeze(0))
                & (pos_b[:, 1:2] <= (gt_cy + gt_h / 2).unsqueeze(0))
            )  # [N_b, G]

            is_fg = inside.any(dim=1)
            if not is_fg.any():
                continue

            fg_idx = is_fg.nonzero(as_tuple=True)[0]
            fg_pos = pos_b[fg_idx]                                    # [K, 2]
            gt_centres = torch.stack([gt_cx, gt_cy], dim=-1)          # [G, 2]
            assigned = torch.cdist(fg_pos, gt_centres).argmin(dim=1)  # [K]

            target_off = gt_centres[assigned] - fg_pos
            loss_vote = loss_vote + F.smooth_l1_loss(
                off_b[fg_idx], target_off, reduction="sum"
            )
            num_fg += fg_idx.shape[0]

        loss_vote = loss_vote / max(num_fg, 1)

        # ---- total ----
        total = 1.0 * loss_hm + 2.0 * loss_reg + 1.0 * loss_cls + 3.0 * loss_vote

        return (
            total,
            loss_reg,                       # maps to "iou_loss" slot
            loss_hm,                        # maps to "conf_loss" slot
            loss_cls + loss_vote,            # maps to "cls_loss" slot
            num_fg / max(num_gts, 1),        # maps to "num_fg" slot
        )

    # ------------------------------------------------------------------
    #  Detection extraction  (inference — batch mode)
    # ------------------------------------------------------------------

    def _extract_detections(self, grid, B):
        conf_acc, reg_acc, cls_acc, _ = self._read_grid(grid)
        confidence = conf_acc.sigmoid()          # [B, 1, H, W]
        s = float(self.stride)
        results = []

        for b in range(B):
            conf_b = confidence[b, 0]            # [H, W]

            # 3x3 local-max suppression on the grid
            local_max = F.max_pool2d(
                conf_b.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1
            ).squeeze()
            is_peak = (conf_b == local_max) & (conf_b > self.conf_threshold)

            peaks = is_peak.nonzero(as_tuple=False)  # [K, 2]  (j, i)
            if peaks.shape[0] == 0:
                results.append(
                    {
                        "boxes": grid.new_zeros(0, 4),
                        "scores": grid.new_zeros(0),
                        "labels": grid.new_zeros(0, dtype=torch.long),
                    }
                )
                continue

            pj, pi = peaks[:, 0], peaks[:, 1]
            scores = conf_b[pj, pi]

            # decode boxes:  grid coords -> pixel coords
            cx = reg_acc[b, 0, pj, pi] * s
            cy = reg_acc[b, 1, pj, pi] * s
            w = reg_acc[b, 2, pj, pi] * s
            h = reg_acc[b, 3, pj, pi] * s

            boxes_xyxy = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
            )

            cls_at = cls_acc[b, :, pj, pi].t()       # [K, C]
            lbl = cls_at.argmax(dim=-1)

            keep = torchvision.ops.batched_nms(boxes_xyxy, scores, lbl, 0.5)

            results.append(
                {
                    "boxes": boxes_xyxy[keep],
                    "scores": scores[keep],
                    "labels": lbl[keep],
                }
            )

        return results

    # ==================================================================
    #  Hardware-inference helpers  (event-by-event EMA)
    # ==================================================================

    def init_state(self, device, dtype=torch.float32):
        """Create a blank accumulator grid for streaming inference."""
        return torch.zeros(
            1, self.cell_channels, self.grid_h, self.grid_w,
            device=device, dtype=dtype,
        )

    @torch.no_grad()
    def process_single_event(self, state, data, apply_global_decay=True):
        """
        Update the accumulator with ONE event.

        Hardware cost:
            MLP forward  —  few-hundred MACs
            Grid update  —  1 SRAM read + 1 MAC per channel + 1 SRAM write

        Args:
            state:              [1, C, H, W]  persistent SRAM
            data:               GraphData with 1 node
            apply_global_decay: do state *= (1-alpha) before the update
                                (call once per unique timestamp, not per event)
        Returns:
            updated state
        """
        alpha = self.ema_alpha

        # global decay (hardware: background SRAM scan, or lazy-on-read)
        if apply_global_decay:
            state = state * (1.0 - alpha)

        # ---- voting MLP (identical to training path) ----
        data = self.stem(data)

        vote_feat = self.vote_conv(data.clone())
        offsets = self.vote_offset(vote_feat.clone()).x
        conf_logits = self.vote_conf(vote_feat).x

        reg_feat = self.reg_conv(data.clone())
        sizes = torch.exp(self.vote_size(reg_feat).x.clamp(max=8.0))

        cls_feat = self.cls_conv(data.clone())
        cls_logits = self.vote_cls(cls_feat).x

        voted = data.pos[:, :2] + offsets                     # [1, 2]
        vote = torch.cat([conf_logits, voted, sizes, cls_logits], dim=-1)  # [1, C]

        gi = voted[0, 0].long().clamp(0, self.grid_w - 1)
        gj = voted[0, 1].long().clamp(0, self.grid_h - 1)

        # EMA write:  cell += alpha * vote  (the *(1-alpha) was applied above)
        state[0, :, gj, gi] += alpha * vote[0]

        return state

    @torch.no_grad()
    def read_detections_from_state(self, state):
        """
        Read current detections from the EMA state.
        (No weight-sum normalisation needed — EMA is self-normalising.)
        """
        confidence = state[:, 0:1].sigmoid()
        s = float(self.stride)
        results = []

        for b in range(state.shape[0]):
            conf_b = confidence[b, 0]
            local_max = F.max_pool2d(
                conf_b.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1
            ).squeeze()
            is_peak = (conf_b == local_max) & (conf_b > self.conf_threshold)
            peaks = is_peak.nonzero(as_tuple=False)

            if peaks.shape[0] == 0:
                results.append(
                    {
                        "boxes": state.new_zeros(0, 4),
                        "scores": state.new_zeros(0),
                        "labels": state.new_zeros(0, dtype=torch.long),
                    }
                )
                continue

            pj, pi = peaks[:, 0], peaks[:, 1]
            scores = conf_b[pj, pi]

            cx = state[b, 1, pj, pi] * s
            cy = state[b, 2, pj, pi] * s
            w = state[b, 3, pj, pi] * s
            h = state[b, 4, pj, pi] * s

            boxes_xyxy = torch.stack(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
            )

            cls_logits = state[b, 5 : 5 + self.num_classes, pj, pi].t()
            lbl = cls_logits.argmax(dim=-1)

            keep = torchvision.ops.batched_nms(boxes_xyxy, scores, lbl, 0.5)
            results.append(
                {
                    "boxes": boxes_xyxy[keep],
                    "scores": scores[keep],
                    "labels": lbl[keep],
                }
            )

        return results
