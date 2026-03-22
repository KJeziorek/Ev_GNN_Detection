"""Training script for event-based GNN detection.

Usage:
    python scripts/train.py --config configs/ncaltech101.yaml
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from time import time

from torchmetrics.detection import MeanAveragePrecision

from datasets.ncaltech101 import NCaltech101
from models.model import Detection


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_conf = 0.0
    total_cls = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        outputs = model(batch)
        loss = outputs["total_loss"]

        optimizer.zero_grad()
        if loss.grad_fn is not None and torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        total_iou += outputs["iou_loss"].item() if torch.is_tensor(outputs["iou_loss"]) else outputs["iou_loss"]
        total_conf += outputs["conf_loss"].item() if torch.is_tensor(outputs["conf_loss"]) else outputs["conf_loss"]
        total_cls += outputs["cls_loss"].item() if torch.is_tensor(outputs["cls_loss"]) else outputs["cls_loss"]
        n_batches += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(loader)}] loss={loss.item():.4f} "
                  f"iou={outputs['iou_loss']:.4f} "
                  f"conf={outputs['conf_loss']:.4f} "
                  f"cls={outputs['cls_loss']:.4f} "
                  f"num_fg={outputs['num_fg']:.1f}")

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "conf": total_conf / n,
        "cls": total_cls / n,
    }


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_conf = 0.0
    total_cls = 0.0
    n_batches = 0

    map_metric = MeanAveragePrecision(iou_type="bbox").to(device)

    for batch in loader:
        batch = batch.to(device)

        # Run in training mode briefly to compute losses on val set
        model.train()
        loss_outputs = model(batch)
        model.eval()

        total_loss += loss_outputs["total_loss"].item()
        total_iou += loss_outputs["iou_loss"].item() if torch.is_tensor(loss_outputs["iou_loss"]) else loss_outputs["iou_loss"]
        total_conf += loss_outputs["conf_loss"].item() if torch.is_tensor(loss_outputs["conf_loss"]) else loss_outputs["conf_loss"]
        total_cls += loss_outputs["cls_loss"].item() if torch.is_tensor(loss_outputs["cls_loss"]) else loss_outputs["cls_loss"]
        n_batches += 1

        # Inference predictions for mAP
        preds = model(batch)

        # Build ground-truth targets from batch bboxes
        B = batch.batch.max().item() + 1 if batch.batch is not None else 1
        targets = []
        for b in range(B):
            if batch.bboxes is not None and batch.bboxes.numel() > 0:
                mask = batch.batch_bb == b
                bbs = batch.bboxes[mask]
                # bboxes format: [class_id, x_tl, y_tl, w, h] -> xyxy
                boxes = torch.stack([
                    bbs[:, 1],
                    bbs[:, 2],
                    bbs[:, 1] + bbs[:, 3],
                    bbs[:, 2] + bbs[:, 4],
                ], dim=1)
                targets.append({
                    "boxes": boxes,
                    "labels": bbs[:, 0].long(),
                })
            else:
                targets.append({
                    "boxes": torch.zeros(0, 4, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                })

        map_metric.update(preds, targets)

    map_results = map_metric.compute()
    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "conf": total_conf / n,
        "cls": total_cls / n,
        "mAP": map_results["map"].item(),
        "mAP_50": map_results["map_50"].item(),
        "mAP_75": map_results["map_75"].item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ncaltech101.yaml")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)

    data_cfg = {**raw_cfg["data"], **raw_cfg["graph"]}
    data_cfg["batch_size"] = raw_cfg["training"]["batch_size"]
    data_cfg["num_workers"] = raw_cfg["training"]["num_workers"]

    train_cfg = raw_cfg["training"]
    log_cfg = raw_cfg["logging"]
    device = raw_cfg.get("device", "cuda")
    num_classes = raw_cfg["model"]["num_classes"]
    spatial_range = tuple(raw_cfg["model"]["spatial_range"])

    torch.manual_seed(raw_cfg["data"].get("seed", 42))

    # ── Dataset ───────────────────────────────────────────────────────────
    dm = NCaltech101(data_cfg)
    dm.setup()

    print(f"Classes: {dm.num_classes}")
    print(f"Train: {len(dm.train_data)}, Val: {len(dm.val_data)}, Test: {len(dm.test_data)}")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # ── Model ─────────────────────────────────────────────────────────────
    model = Detection(num_classes=num_classes, spatial_range=spatial_range).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    # ── Checkpointing setup ───────────────────────────────────────────────
    ckpt_dir = Path(log_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    patience = train_cfg.get("patience", 20)

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        t1 = time()
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{train_cfg['epochs']} ({t1 - t0:.1f}s) lr={lr:.2e}\n"
            f"  train: loss={train_metrics['loss']:.4f} "
            f"iou={train_metrics['iou']:.4f} "
            f"conf={train_metrics['conf']:.4f} "
            f"cls={train_metrics['cls']:.4f}\n"
            f"  val:   loss={val_metrics['loss']:.4f} "
            f"iou={val_metrics['iou']:.4f} "
            f"conf={val_metrics['conf']:.4f} "
            f"cls={val_metrics['cls']:.4f}\n"
            f"  mAP={val_metrics['mAP']:.4f} "
            f"mAP_50={val_metrics['mAP_50']:.4f} "
            f"mAP_75={val_metrics['mAP_75']:.4f}"
        )

        # ── Checkpointing ────────────────────────────────────────────────
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, ckpt_dir / "best.pt")
            print(f"  * New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        if epoch % log_cfg.get("save_every", 10) == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
            }, ckpt_dir / f"epoch_{epoch}.pt")

        # ── Early stopping ────────────────────────────────────────────────
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
