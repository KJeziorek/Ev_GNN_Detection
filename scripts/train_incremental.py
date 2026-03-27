"""
Train the incremental detection model on NCaltech101.

Validation computes mAP (COCO-style) using torchmetrics.
Training uses the same dataset, collate_fn, output dict keys
as the original YOLOX pipeline.

Usage:
    python train_incremental.py

Requirements:
    pip install torchmetrics pycocotools
"""

import argparse
import os
import sys
import time
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import lightning as L
from torchmetrics.detection import MeanAveragePrecision

from datasets.ncaltech101 import NCaltech101
from models.detection_incremental import DetectionIncremental


# ======================================================================
#  Helpers
# ======================================================================

def targets_to_coco(target_tensor):
    """
    Convert [B, max_det, 5] labels to per-image dicts for torchmetrics.

    Input format:   [class_id, cx, cy, w, h]  in pixels
    Output format:  {boxes: [K,4] xyxy, labels: [K]}
    """
    B = target_tensor.shape[0]
    nlabel = (target_tensor.sum(dim=2) > 0).sum(dim=1)
    out = []

    for b in range(B):
        n = int(nlabel[b])
        if n == 0:
            out.append({
                "boxes":  target_tensor.new_zeros(0, 4),
                "labels": target_tensor.new_zeros(0, dtype=torch.long),
            })
            continue

        cls_ids = target_tensor[b, :n, 0].long()
        cx = target_tensor[b, :n, 1]
        cy = target_tensor[b, :n, 2]
        w  = target_tensor[b, :n, 3]
        h  = target_tensor[b, :n, 4]

        boxes_xyxy = torch.stack([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ], dim=-1)

        out.append({
            "boxes":  boxes_xyxy,
            "labels": cls_ids,
        })

    return out


# ======================================================================
#  Lightning Module
# ======================================================================

class DetectionModule(L.LightningModule):
    def __init__(self, num_classes=100, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = DetectionIncremental(
            num_classes=num_classes,
            conf_threshold=0.3,
        )

        # COCO-style mAP — computed per-epoch, not per-step
        self.val_map = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
        )

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)

        self.log("train/total_loss", outputs["total_loss"], prog_bar=True,
                 batch_size=int(batch.batch.max()) + 1)
        self.log("train/reg_loss",  outputs["iou_loss"], prog_bar=True,
                 batch_size=int(batch.batch.max()) + 1)
        self.log("train/hm_loss",   outputs["conf_loss"], prog_bar=True,
                 batch_size=int(batch.batch.max()) + 1)
        self.log("train/cls_loss",  outputs["cls_loss"], prog_bar=True,
                 batch_size=int(batch.batch.max()) + 1)
        self.log("train/num_fg",    outputs["num_fg"],
                 batch_size=int(batch.batch.max()) + 1)

        return outputs["total_loss"]

    # ------------------------------------------------------------------
    #  Validation — loss + mAP
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        loss_dict, detections = self.model.forward_with_detections(batch)

        B = int(batch.batch.max()) + 1

        # ---- log val loss ----
        self.log("val/total_loss", loss_dict["total_loss"], prog_bar=True,
                 batch_size=B)
        self.log("val/reg_loss",   loss_dict["iou_loss"],  batch_size=B)
        self.log("val/hm_loss",    loss_dict["conf_loss"], batch_size=B)
        self.log("val/cls_loss",   loss_dict["cls_loss"],  batch_size=B)

        # ---- accumulate for mAP ----
        #  predictions: list[dict] with {boxes (xyxy), scores, labels}
        #  ground truth: convert data.target [B, max_det, 5] to same format
        gt_list = targets_to_coco(batch.target)

        # make sure both live on the same device
        preds_cpu = []
        for det in detections:
            preds_cpu.append({
                "boxes":  det["boxes"].detach().cpu(),
                "scores": det["scores"].detach().cpu(),
                "labels": det["labels"].detach().cpu(),
            })

        gt_cpu = []
        for gt in gt_list:
            gt_cpu.append({
                "boxes":  gt["boxes"].detach().cpu(),
                "labels": gt["labels"].detach().cpu(),
            })

        self.val_map.update(preds_cpu, gt_cpu)

    def on_validation_epoch_end(self):
        metrics = self.val_map.compute()

        self.log("val/mAP",      metrics["map"],       prog_bar=True)
        self.log("val/mAP_50",   metrics["map_50"],    prog_bar=True)
        self.log("val/mAP_75",   metrics["map_75"])
        self.log("val/mAP_s",    metrics["map_small"])
        self.log("val/mAP_m",    metrics["map_medium"])
        self.log("val/mAP_l",    metrics["map_large"])

        self.val_map.reset()

    # ------------------------------------------------------------------
    #  Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # separate backbone and head params for different lr if needed
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ======================================================================
#  Main
# ======================================================================

def main(cfg):
    datamodule = NCaltech101(cfg)
    datamodule.setup()

    num_classes = datamodule.num_classes
    print(f"Number of classes: {num_classes}")

    model = DetectionModule(num_classes=num_classes, lr=1e-3)

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=5.0,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val/mAP_50",
                mode="max",
                save_top_k=3,
                filename="epoch{epoch:02d}-mAP50{val/mAP_50:.3f}",
                auto_insert_metric_name=False,
            ),
            L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val/mAP_50",
                mode="max",
                patience=15,
            ),
        ],
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ncaltech101.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Flatten nested config for dataset (it expects a flat dict)
    ds_cfg = {**cfg.get("data", {}), **cfg.get("norm", {}),
              **cfg.get("augmentation", {}), **cfg.get("graph", {})}
    main(ds_cfg)
