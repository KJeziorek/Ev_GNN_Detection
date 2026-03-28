"""
Post-processing parameter sweep to maximize mAP.

Runs the model once to collect all raw predictions, then sweeps
conf_threshold × nms_threshold (and optionally score_power, top_k)
offline — no repeated forward passes.

Usage:
    python -m scripts.sweep_params --config configs/ncaltech101.yaml \
                                   --checkpoint checkpoints/ncaltech101/best.pth
"""

import argparse
import os
import sys
import itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import numpy as np
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
import torchvision

from datasets.ncaltech101 import NCaltech101
from models.detection import Detection as DetectionModel
# from models.detection_accumulated import DetectionAccumulated as DetectionModel


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


def apply_nms(raw_preds, conf_thre, nms_thre, top_k=300, min_box_size=0.0):
    """
    Apply conf filtering + NMS to raw decoded predictions.

    Args:
        raw_preds: list of dict per image, each with:
            "decoded": [N, 5+C]  (cx, cy, w, h, obj_logit, cls_logits...)
        conf_thre:  confidence threshold
        nms_thre:   NMS IoU threshold
        top_k:      max detections per image before NMS
        min_box_size: filter boxes smaller than this (pixels)

    Returns:
        list[dict] with {boxes (xyxy), scores, labels}
    """
    results = []
    for raw in raw_preds:
        det = raw["decoded"]  # [N, 5+C]
        if det.shape[0] == 0:
            results.append({
                "boxes": det.new_zeros(0, 4),
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

        # confidence filter
        keep = score >= conf_thre
        boxes_xyxy = boxes_xyxy[keep]
        score = score[keep]
        class_pred = class_pred[keep]

        if boxes_xyxy.shape[0] == 0:
            results.append({
                "boxes": det.new_zeros(0, 4),
                "scores": det.new_zeros(0),
                "labels": det.new_zeros(0, dtype=torch.long),
            })
            continue

        # min box size filter
        if min_box_size > 0:
            w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
            h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
            size_keep = (w >= min_box_size) & (h >= min_box_size)
            boxes_xyxy = boxes_xyxy[size_keep]
            score = score[size_keep]
            class_pred = class_pred[size_keep]

        # top-k before NMS
        if score.shape[0] > top_k:
            topk_idx = score.topk(top_k).indices
            boxes_xyxy = boxes_xyxy[topk_idx]
            score = score[topk_idx]
            class_pred = class_pred[topk_idx]

        # NMS
        if boxes_xyxy.shape[0] > 0:
            nms_idx = torchvision.ops.batched_nms(boxes_xyxy, score, class_pred, nms_thre)
            results.append({
                "boxes": boxes_xyxy[nms_idx],
                "scores": score[nms_idx],
                "labels": class_pred[nms_idx],
            })
        else:
            results.append({
                "boxes": det.new_zeros(0, 4),
                "scores": det.new_zeros(0),
                "labels": det.new_zeros(0, dtype=torch.long),
            })

    return results


# ======================================================================
#  Collect raw predictions (one forward pass)
# ======================================================================

@torch.no_grad()
def collect_raw_predictions(model, dataloader, device, num_batches=None):
    """
    Run model forward and collect decoded (pre-NMS) predictions + GT.

    Returns:
        all_raw:     list of list[dict] per batch, per image
                     each dict has "decoded": [N, 5+C] tensor
        all_targets: list of list[dict] per batch, per image
                     each dict has "boxes", "labels"
    """
    model.eval()
    all_raw = []
    all_targets = []

    # We need access to the head's internal decoded tensor before NMS.
    # Monkey-patch the head to capture it.
    head = model.head if hasattr(model, 'head') else model.backbone  # fallback

    # For the standard YOLOX head, we replicate the forward up to decode
    # then save the raw decoded output.

    pbar = tqdm(dataloader, desc="Collecting predictions")
    for batch_idx, data in enumerate(pbar):
        if num_batches is not None and batch_idx >= num_batches:
            break

        data = data.to(device)
        B = int(data.batch.max().item()) + 1

        # --- run backbone ---
        fpn_outs = model.backbone(data)

        # --- run head layers to get decoded predictions ---
        head_mod = model.head
        all_outputs = []
        all_positions = []
        all_strides_list = []
        all_batches = []

        for k, (cls_conv, reg_conv, stride_k, feat_data) in enumerate(
            zip(head_mod.cls_convs, head_mod.reg_convs, head_mod.strides, fpn_outs)
        ):
            feat_data = head_mod.stems[k](feat_data)
            cls_data = feat_data.clone()

            cls_feat = cls_conv(cls_data)
            reg_feat = reg_conv(feat_data)
            obj_feat = reg_feat.clone()

            cls_output = head_mod.cls_preds[k](cls_feat).x
            reg_output = head_mod.reg_preds[k](reg_feat).x
            obj_output = head_mod.obj_preds[k](obj_feat).x

            N_k = reg_output.shape[0]
            if N_k == 0:
                continue

            output = torch.cat([reg_output, obj_output, cls_output], dim=1)
            all_outputs.append(output)
            all_positions.append(feat_data.pos[:, :2])
            all_strides_list.append(output.new_full((N_k,), stride_k))
            all_batches.append(feat_data.batch)

        if len(all_outputs) == 0:
            for b in range(B):
                all_raw.append([{"decoded": torch.zeros(0, 5 + head_mod.num_classes, device=device)}])
            all_targets.append(labels_to_dicts(data.target))
            continue

        outputs = torch.cat(all_outputs, dim=0)
        positions = torch.cat(all_positions, dim=0)
        strides = torch.cat(all_strides_list, dim=0)
        batches = torch.cat(all_batches, dim=0)

        decoded = head_mod.decode_outputs(outputs, positions, strides)

        # Split per image
        batch_raw = []
        for b in range(B):
            mask_b = batches == b
            batch_raw.append({"decoded": decoded[mask_b].cpu()})

        all_raw.append(batch_raw)
        all_targets.append(labels_to_dicts(data.target))

    return all_raw, all_targets


# ======================================================================
#  Evaluate one parameter combination
# ======================================================================

def evaluate_params(all_raw, all_targets, conf_thre, nms_thre,
                    top_k=300, min_box_size=0.0):
    """Evaluate mAP for a given parameter combination."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for batch_raw, batch_targets in zip(all_raw, all_targets):
        preds = apply_nms(batch_raw, conf_thre, nms_thre, top_k, min_box_size)

        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        gt_cpu = [{k: v.cpu() for k, v in t.items()} for t in batch_targets]

        metric.update(preds_cpu, gt_cpu)

    results = metric.compute()
    return {
        "mAP": results["map"].item(),
        "mAP_50": results["map_50"].item(),
        "mAP_75": results["map_75"].item(),
        "mAP_s": results["map_small"].item(),
        "mAP_m": results["map_medium"].item(),
        "mAP_l": results["map_large"].item(),
        "mar_100": results["mar_100"].item(),
    }


# ======================================================================
#  Grid sweep
# ======================================================================

def run_sweep(all_raw, all_targets):
    """Sweep conf × nms × top_k and print sorted results."""

    conf_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    nms_thresholds = [0.3, 0.45, 0.5, 0.65, 0.8]
    top_k_values = [100, 300]
    min_box_sizes = [0.0]

    combos = list(itertools.product(
        conf_thresholds, nms_thresholds, top_k_values, min_box_sizes
    ))

    print(f"\nSweeping {len(combos)} parameter combinations...")
    print(f"  conf_thre:    {conf_thresholds}")
    print(f"  nms_thre:     {nms_thresholds}")
    print(f"  top_k:        {top_k_values}")
    print(f"  min_box_size: {min_box_sizes}")

    results = []
    pbar = tqdm(combos, desc="Sweeping")
    for conf_thre, nms_thre, top_k, min_box_size in pbar:
        metrics = evaluate_params(
            all_raw, all_targets,
            conf_thre=conf_thre,
            nms_thre=nms_thre,
            top_k=top_k,
            min_box_size=min_box_size,
        )
        results.append({
            "conf": conf_thre,
            "nms": nms_thre,
            "top_k": top_k,
            "min_box": min_box_size,
            **metrics,
        })
        pbar.set_postfix({"best_mAP50": max(r["mAP_50"] for r in results)})

    # Sort by mAP@50
    results.sort(key=lambda r: r["mAP_50"], reverse=True)

    # Print top results
    print("\n" + "=" * 100)
    print("  TOP 15 PARAMETER COMBINATIONS (sorted by mAP@50)")
    print("=" * 100)
    print(f"  {'conf':>6s}  {'nms':>5s}  {'top_k':>5s}  {'min_b':>5s}  "
          f"{'mAP':>6s}  {'mAP50':>6s}  {'mAP75':>6s}  "
          f"{'mAP_s':>6s}  {'mAP_m':>6s}  {'mAP_l':>6s}  {'mAR100':>6s}")
    print("-" * 100)

    for r in results[:15]:
        print(f"  {r['conf']:6.3f}  {r['nms']:5.2f}  {r['top_k']:5d}  {r['min_box']:5.1f}  "
              f"{r['mAP']:6.4f}  {r['mAP_50']:6.4f}  {r['mAP_75']:6.4f}  "
              f"{r['mAP_s']:6.4f}  {r['mAP_m']:6.4f}  {r['mAP_l']:6.4f}  {r['mar_100']:6.4f}")

    # Print worst too (helps understand sensitivity)
    print(f"\n  WORST 5:")
    print("-" * 100)
    for r in results[-5:]:
        print(f"  {r['conf']:6.3f}  {r['nms']:5.2f}  {r['top_k']:5d}  {r['min_box']:5.1f}  "
              f"{r['mAP']:6.4f}  {r['mAP_50']:6.4f}  {r['mAP_75']:6.4f}  "
              f"{r['mAP_s']:6.4f}  {r['mAP_m']:6.4f}  {r['mAP_l']:6.4f}  {r['mar_100']:6.4f}")

    # Analysis
    best = results[0]
    print(f"\n--- BEST PARAMS ---")
    print(f"  conf_threshold = {best['conf']}")
    print(f"  nms_threshold  = {best['nms']}")
    print(f"  top_k          = {best['top_k']}")
    print(f"  min_box_size   = {best['min_box']}")
    print(f"  → mAP@50 = {best['mAP_50']:.4f}   mAP@50:95 = {best['mAP']:.4f}")

    # Sensitivity analysis: which param matters most?
    print(f"\n--- SENSITIVITY ANALYSIS ---")
    for param_name, param_values in [("conf", conf_thresholds), ("nms", nms_thresholds),
                                      ("top_k", top_k_values)]:
        print(f"\n  {param_name}:")
        for val in param_values:
            subset = [r for r in results if r[param_name] == val]
            if subset:
                avg_map50 = np.mean([r["mAP_50"] for r in subset])
                best_map50 = max(r["mAP_50"] for r in subset)
                print(f"    {val:>8}  →  avg mAP@50={avg_map50:.4f}  best={best_map50:.4f}")

    # Recall analysis
    print(f"\n--- RECALL CHECK (mAR@100) ---")
    best_recall = max(results, key=lambda r: r["mar_100"])
    print(f"  Best recall:  mAR@100 = {best_recall['mar_100']:.4f}")
    print(f"    at conf={best_recall['conf']}, nms={best_recall['nms']}")
    print(f"    mAP@50 at this config: {best_recall['mAP_50']:.4f}")

    if best_recall["mar_100"] < 0.3:
        print(f"\n  ⚠️  Recall is very low even at best settings.")
        print(f"     This means the model cannot localize GT boxes well.")
        print(f"     The problem is in the MODEL, not post-processing.")
        print(f"     Check: coordinate decode, stride, loss convergence.")
    elif best_recall["mar_100"] > 0.5 and best["mAP_50"] < 0.2:
        print(f"\n  ⚠️  High recall but low precision.")
        print(f"     The model finds objects but assigns wrong classes or")
        print(f"     produces too many false positives.")
        print(f"     Check: cls head convergence, obj score calibration.")

    return results


# ======================================================================
#  Main
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ncaltech101.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ncaltech101/best.pth")
    parser.add_argument("--num_batches", type=int, default=None,
                        help="Limit batches (None = full val set)")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = {**cfg.get("data", {}), **cfg.get("norm", {}),
              **cfg.get("augmentation", {}), **cfg.get("graph", {})}

    device = cfg.get("device", "cuda")

    # Dataset
    datamodule = NCaltech101(ds_cfg)
    datamodule.setup()
    print(f"Classes: {datamodule.num_classes}")

    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()
    print(f"Using {args.split} split, {len(loader)} batches")

    # Model
    model = DetectionModel()

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "state_dict" in state:
            sd = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
            model.load_state_dict(sd)
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint loaded — running with random weights")

    model = model.to(device)

    # Step 1: collect raw predictions (single forward pass)
    print("\nStep 1: Collecting raw (pre-NMS) predictions...")
    all_raw, all_targets = collect_raw_predictions(
        model, loader, device, num_batches=args.num_batches
    )

    total_preds = sum(r["decoded"].shape[0] for batch in all_raw for r in batch)
    total_gts = sum(len(t["labels"]) for batch in all_targets for t in batch)
    print(f"  Collected {total_preds} predictions across {total_gts} GT boxes")

    # Step 2: sweep parameters
    print("\nStep 2: Sweeping post-processing parameters...")
    results = run_sweep(all_raw, all_targets)
