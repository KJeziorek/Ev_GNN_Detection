"""
Detection diagnostic script.

Run after a few epochs of training to identify WHY mAP is low.
Checks (in order of likelihood):

  1. Coordinate / format bugs  (most common — kills mAP silently)
  2. Prediction statistics      (are boxes reasonable?)
  3. IoU distribution           (localization vs classification problem?)
  4. Confidence calibration     (threshold too high/low?)
  5. Class accuracy             (100-class confusion?)
  6. Per-sample visualization   (visual sanity check)

Usage:
    python -m scripts.diagnose --config configs/ncaltech101.yaml \
                               --checkpoint checkpoints/ncaltech101/best.pth
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import numpy as np
from collections import defaultdict

from datasets.ncaltech101 import NCaltech101
# Switch between Detection / DetectionAccumulated here:
# from models.detection_accumulated import DetectionAccumulated as DetectionModel
from models.detection import Detection as DetectionModel


# ======================================================================
#  Helpers
# ======================================================================

def labels_to_dicts(labels):
    """[B, max_det, 5] → list[dict] with boxes (xyxy) + labels."""
    B = labels.shape[0]
    out = []
    for b in range(B):
        lab = labels[b]
        valid = lab.sum(dim=1) > 0
        lab = lab[valid]
        if lab.shape[0] == 0:
            out.append({"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})
            continue
        cls_ids = lab[:, 0].long()
        cx, cy, w, h = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
        boxes = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)
        out.append({"boxes": boxes, "labels": cls_ids})
    return out


def box_iou(boxes1, boxes2):
    """Pairwise IoU between two sets of xyxy boxes. [N,4] x [M,4] → [N,M]"""
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-8)


# ======================================================================
#  Diagnostic checks
# ======================================================================

@torch.no_grad()
def run_diagnostics(model, dataloader, device, num_batches=50):
    model.eval()

    # Accumulators
    stats = defaultdict(list)
    all_ious = []
    all_scores = []
    all_cls_correct = []
    all_pred_counts = []
    all_gt_counts = []
    total_samples = 0
    samples_with_detections = 0
    format_issues = []

    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        data = data.to(device)
        B = int(data.batch.max().item()) + 1

        # --- Get predictions ---
        if hasattr(model, 'forward_with_detections'):
            _, predictions = model.forward_with_detections(data)
        else:
            predictions = model(data)

        targets = labels_to_dicts(data.target)

        for b in range(B):
            total_samples += 1
            pred = predictions[b]
            gt = targets[b]

            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_labels = pred["labels"].cpu()
            gt_boxes = gt["boxes"].cpu()
            gt_labels = gt["labels"].cpu()

            n_pred = pred_boxes.shape[0]
            n_gt = gt_boxes.shape[0]
            all_pred_counts.append(n_pred)
            all_gt_counts.append(n_gt)

            if n_pred > 0:
                samples_with_detections += 1
                all_scores.extend(pred_scores.tolist())

                # --- CHECK 1: Box sanity ---
                widths = pred_boxes[:, 2] - pred_boxes[:, 0]
                heights = pred_boxes[:, 3] - pred_boxes[:, 1]
                stats["pred_cx"].extend(((pred_boxes[:, 0] + pred_boxes[:, 2]) / 2).tolist())
                stats["pred_cy"].extend(((pred_boxes[:, 1] + pred_boxes[:, 3]) / 2).tolist())
                stats["pred_w"].extend(widths.tolist())
                stats["pred_h"].extend(heights.tolist())
                stats["pred_area"].extend((widths * heights).tolist())

                # Negative width/height = format bug
                if (widths <= 0).any() or (heights <= 0).any():
                    format_issues.append(f"batch {batch_idx} img {b}: negative w/h in predictions")

            if n_gt > 0:
                gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
                gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
                stats["gt_cx"].extend(((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2).tolist())
                stats["gt_cy"].extend(((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2).tolist())
                stats["gt_w"].extend(gt_w.tolist())
                stats["gt_h"].extend(gt_h.tolist())
                stats["gt_area"].extend((gt_w * gt_h).tolist())

                if (gt_w <= 0).any() or (gt_h <= 0).any():
                    format_issues.append(f"batch {batch_idx} img {b}: negative w/h in GT")

            # --- CHECK 2: IoU distribution ---
            if n_pred > 0 and n_gt > 0:
                ious = box_iou(pred_boxes, gt_boxes)  # [n_pred, n_gt]
                best_iou_per_pred, best_gt_idx = ious.max(dim=1)
                best_iou_per_gt, best_pred_idx = ious.max(dim=0)

                all_ious.extend(best_iou_per_gt.tolist())

                # --- CHECK 3: Class accuracy ---
                for gi in range(n_gt):
                    pi = best_pred_idx[gi].item()
                    if best_iou_per_gt[gi] > 0.5:
                        cls_ok = (pred_labels[pi] == gt_labels[gi]).item()
                        all_cls_correct.append(cls_ok)

    # ==================================================================
    #  Print report
    # ==================================================================

    print("\n" + "=" * 70)
    print("  DETECTION DIAGNOSTIC REPORT")
    print("=" * 70)

    # ---- 0. Format issues ----
    print(f"\n--- FORMAT BUGS ---")
    if format_issues:
        print(f"  ⚠️  Found {len(format_issues)} format issues!")
        for issue in format_issues[:10]:
            print(f"    {issue}")
    else:
        print(f"  ✓ No negative-size boxes found")

    # ---- 1. Detection rate ----
    print(f"\n--- DETECTION RATE ---")
    print(f"  Total samples examined:        {total_samples}")
    print(f"  Samples with ≥1 detection:     {samples_with_detections} "
          f"({100*samples_with_detections/max(total_samples,1):.1f}%)")
    print(f"  Avg predictions per sample:    {np.mean(all_pred_counts):.1f}")
    print(f"  Avg GT boxes per sample:       {np.mean(all_gt_counts):.1f}")

    if samples_with_detections == 0:
        print(f"\n  ❌ NO DETECTIONS AT ALL — check confidence threshold!")
        print(f"     This is your problem. The model predicts nothing.")
        print(f"     Try lowering conf_threshold in the head.")
        return

    # ---- 2. Score distribution ----
    scores = np.array(all_scores)
    print(f"\n--- CONFIDENCE SCORES ---")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    for thr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        pct = (scores >= thr).mean() * 100
        print(f"  Score ≥ {thr}:  {pct:.1f}% of predictions")

    # ---- 3. Box statistics comparison ----
    print(f"\n--- BOX STATISTICS (pred vs GT) ---")
    for key_base in ["cx", "cy", "w", "h", "area"]:
        pred_vals = np.array(stats.get(f"pred_{key_base}", [0]))
        gt_vals = np.array(stats.get(f"gt_{key_base}", [0]))
        if len(pred_vals) > 0 and len(gt_vals) > 0:
            print(f"  {key_base:>5s}:  pred [{pred_vals.min():.1f}, {pred_vals.mean():.1f}, {pred_vals.max():.1f}]"
                  f"   gt [{gt_vals.min():.1f}, {gt_vals.mean():.1f}, {gt_vals.max():.1f}]")

    # Check for coordinate space mismatch
    pred_cx = np.array(stats.get("pred_cx", []))
    gt_cx = np.array(stats.get("gt_cx", []))
    if len(pred_cx) > 0 and len(gt_cx) > 0:
        pred_range = pred_cx.max() - pred_cx.min()
        gt_range = gt_cx.max() - gt_cx.min()
        ratio = pred_range / max(gt_range, 1e-6)
        if ratio < 0.1 or ratio > 10:
            print(f"\n  ⚠️  COORDINATE SPACE MISMATCH LIKELY!")
            print(f"     Pred X range: {pred_cx.min():.1f}–{pred_cx.max():.1f}")
            print(f"     GT X range:   {gt_cx.min():.1f}–{gt_cx.max():.1f}")
            print(f"     Ratio: {ratio:.2f}x — predictions and GT are in different coordinate systems!")

    # ---- 4. IoU distribution ----
    print(f"\n--- IoU DISTRIBUTION (best IoU per GT box) ---")
    if len(all_ious) > 0:
        ious = np.array(all_ious)
        print(f"  Mean best IoU:  {ious.mean():.4f}")
        print(f"  Median:         {np.median(ious):.4f}")
        for thr in [0.1, 0.25, 0.5, 0.75]:
            pct = (ious >= thr).mean() * 100
            print(f"  IoU ≥ {thr}:  {pct:.1f}% of GT boxes matched")

        if ious.mean() < 0.1:
            print(f"\n  ❌ VERY LOW IoU — predictions are not near GT boxes at all.")
            print(f"     Likely a coordinate system or decode bug.")
        elif ious.mean() < 0.3:
            print(f"\n  ⚠️  LOW IoU — boxes are in roughly right area but poorly localized.")
            print(f"     Check stride / decode logic.")
        elif (ious >= 0.5).mean() > 0.5:
            print(f"\n  ✓ Localization looks decent (>{(ious>=0.5).mean()*100:.0f}% IoU≥0.5)")
    else:
        print(f"  No IoU data (no samples with both preds and GTs)")

    # ---- 5. Classification accuracy (at IoU≥0.5) ----
    print(f"\n--- CLASSIFICATION (at IoU ≥ 0.5) ---")
    if len(all_cls_correct) > 0:
        cls_acc = np.mean(all_cls_correct) * 100
        print(f"  Correct class:  {cls_acc:.1f}%  ({sum(all_cls_correct)}/{len(all_cls_correct)})")
        if cls_acc < 10:
            print(f"  ❌ Classification near random — 100 classes, expected ~1% random.")
            print(f"     But {cls_acc:.1f}% suggests cls head is barely learning.")
        elif cls_acc < 50:
            print(f"  ⚠️  Classification is the bottleneck, not localization.")
    else:
        print(f"  No matched boxes at IoU≥0.5 to evaluate classification")

    # ---- 6. Summary diagnosis ----
    print(f"\n--- DIAGNOSIS SUMMARY ---")

    if samples_with_detections < total_samples * 0.5:
        print(f"  → Problem: too few detections ({samples_with_detections}/{total_samples})")
        print(f"    Fix: lower conf_threshold, check obj loss convergence")
    elif len(all_ious) > 0 and np.mean(all_ious) < 0.15:
        print(f"  → Problem: predictions exist but wrong location (mean IoU={np.mean(all_ious):.3f})")
        print(f"    Fix: check decode_outputs coordinate transform, stride values,")
        print(f"         and that GT labels are in the same pixel space as decoded boxes")
    elif len(all_ious) > 0 and (np.array(all_ious) >= 0.5).mean() < 0.3:
        print(f"  → Problem: rough localization but poor IoU")
        print(f"    Fix: train longer, increase reg_weight, check anchor/stride alignment")
    elif len(all_cls_correct) > 0 and np.mean(all_cls_correct) < 0.3:
        print(f"  → Problem: localization OK but classification failing")
        print(f"    Fix: train longer, check class label encoding, try larger cls head")
    else:
        print(f"  → No single dominant failure mode identified")
        print(f"    Continue training or tune hyperparameters")


# ======================================================================
#  Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ncaltech101.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ncaltech101/best-v6.ckpt")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--split", choices=["val", "test", "train"], default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")

    # Dataset
    datamodule = NCaltech101(cfg)
    datamodule.setup()
    print(f"Classes: {datamodule.num_classes}")

    if args.split == "val":
        loader = datamodule.val_dataloader()
    elif args.split == "test":
        loader = datamodule.test_dataloader()
    else:
        loader = datamodule.train_dataloader()
    print(f"Using {args.split} split, {len(loader)} batches")

    # Model
    model = DetectionModel(cfg)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint formats
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "state_dict" in state:
            # Lightning checkpoint: strip "model." prefix
            sd = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
            model.load_state_dict(sd)
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint loaded — running with random weights")

    model = model.to(device)
    run_diagnostics(model, loader, device, num_batches=args.num_batches)
