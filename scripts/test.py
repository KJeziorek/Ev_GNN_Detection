"""
Training script for point-based GNN YOLOX detection on NCaltech101.

Usage:
    python -m scripts.train --config configs/ncaltech101.yaml
"""

import argparse
import os
import sys
import time

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection import MeanAveragePrecision

from models.detection import Detection
from datasets.ncaltech101 import NCaltech101


def labels_to_torchmetrics(labels):
    """
    Convert training-format labels to torchmetrics target format.

    Args:
        labels: [B, max_det, 5]  (class_id, cx, cy, w, h) in pixels.
                Padding rows have all zeros.

    Returns:
        list[dict] with "boxes" (xyxy), "labels" (int) per image.
    """
    targets = []
    for b in range(labels.shape[0]):
        lab = labels[b]
        valid = lab.sum(dim=1) > 0
        lab = lab[valid]

        if lab.shape[0] == 0:
            targets.append({
                "boxes":  torch.zeros(0, 4, device=labels.device),
                "labels": torch.zeros(0, dtype=torch.long, device=labels.device),
            })
        else:
            cls_ids = lab[:, 0].long()
            cx, cy, w, h = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
            boxes_xyxy = torch.stack([
                cx - w / 2, cy - h / 2,
                cx + w / 2, cy + h / 2,
            ], dim=1)
            targets.append({"boxes": boxes_xyxy, "labels": cls_ids})
    return targets


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg):
        self.device = cfg.get("device", "cuda")
        train_cfg = cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.log_interval = train_cfg.get("log_interval", 50)
        self.patience = train_cfg.get("patience", 20)

        log_cfg = cfg.get("logging", {})
        self.save_dir = log_cfg.get("checkpoint_dir", "checkpoints")
        self.save_every = log_cfg.get("save_every", 10)
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = model.to(self.device)

        lr = train_cfg.get("learning_rate", 1e-3)
        self.base_lr = lr
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        self.best_map = 0.0

    # ---- Warmup LR ----

    def _warmup_lr(self, epoch, step, steps_per_epoch):
        total_warmup_steps = self.warmup_epochs * steps_per_epoch
        current_step = epoch * steps_per_epoch + step
        if current_step < total_warmup_steps:
            lr = self.base_lr * current_step / max(total_warmup_steps, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    # ---- Train one epoch ----

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss_sum = 0.0
        iou_loss_sum = 0.0
        obj_loss_sum = 0.0
        cls_loss_sum = 0.0
        num_batches = 0
        steps_per_epoch = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Train {epoch+1}/{self.epochs}", leave=True)

        for step, data in enumerate(pbar):
            if epoch < self.warmup_epochs:
                self._warmup_lr(epoch, step, steps_per_epoch)

            data = data.to(self.device)
            outputs = self.model(data)

            total_loss = outputs["total_loss"]
            iou_loss = outputs["iou_loss"]
            obj_loss = outputs["conf_loss"]
            cls_loss = outputs["cls_loss"]

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            iou_loss_sum += iou_loss.item() if isinstance(iou_loss, torch.Tensor) else iou_loss
            obj_loss_sum += obj_loss.item() if isinstance(obj_loss, torch.Tensor) else obj_loss
            cls_loss_sum += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss_sum/num_batches:.4f}",
                "iou":  f"{iou_loss_sum/num_batches:.4f}",
                "obj":  f"{obj_loss_sum/num_batches:.4f}",
                "cls":  f"{cls_loss_sum/num_batches:.4f}",
                "lr":   f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })

        if epoch >= self.warmup_epochs:
            self.scheduler.step()

        return total_loss_sum / max(num_batches, 1)

    # ---- Validate one epoch ----

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.map_metric.reset()

        pbar = tqdm(self.val_loader, desc=f"Val   {epoch+1}/{self.epochs}", leave=True)

        for data in pbar:
            data = data.to(self.device)

            predictions = self.model(data)
            targets = labels_to_torchmetrics(data.target)

            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in predictions]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            self.map_metric.update(preds_cpu, targets_cpu)

        metrics = self.map_metric.compute()
        map_50 = metrics["map_50"].item()
        map_75 = metrics["map_75"].item()
        map_all = metrics["map"].item()

        print(
            f"Epoch {epoch+1} val: "
            f"mAP@50={map_50:.4f} "
            f"mAP@75={map_75:.4f} "
            f"mAP@50:95={map_all:.4f}"
        )
        return map_50, map_all

    # ---- Save / Load ----

    def save_checkpoint(self, epoch, map_50, is_best=False):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "map_50": map_50,
        }
        path = os.path.join(self.save_dir, "last.pth")
        torch.save(state, path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best.pth")
            torch.save(state, best_path)
            print(f"  -> New best model saved (mAP@50={map_50:.4f})")

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        print(f"Loaded checkpoint from {path} (epoch {state['epoch']+1})")
        return state["epoch"]

    # ---- Main loop ----

    def fit(self):
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"{'='*60}")

            train_loss = self.train_one_epoch(epoch)
            map_50, map_all = self.validate(epoch)

            is_best = map_50 > self.best_map
            if is_best:
                self.best_map = map_50
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            self.save_checkpoint(epoch, map_50, is_best=is_best)

            if (epoch + 1) % self.save_every == 0:
                path = os.path.join(self.save_dir, f"epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), path)

            if epochs_no_improve >= self.patience:
                print(f"Early stopping after {self.patience} epochs without improvement.")
                break

        print(f"\nTraining complete. Best mAP@50={self.best_map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ncaltech101.yaml")
    parser.add_argument("--resume", type=str, default="/home/imperator/Code/Ev_GNN_Detection/checkpoints/ncaltech101/best.pth")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Flatten nested config for dataset (it expects a flat dict)
    ds_cfg = {**cfg.get("data", {}), **cfg.get("norm", {}),
              **cfg.get("augmentation", {}), **cfg.get("graph", {})}

    # Build datamodule and setup splits
    datamodule = NCaltech101(ds_cfg)
    datamodule.setup()

    print(f"Classes: {datamodule.num_classes}")
    print(f"Train: {len(datamodule.train_data)}, Val: {len(datamodule.val_data)}, Test: {len(datamodule.test_data)}")

    # Build model
    model = Detection()

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
        cfg=cfg,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.validate(0)
