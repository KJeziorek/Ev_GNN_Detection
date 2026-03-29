import torch
import lightning as L
import wandb
from functools import partial
from typing import Any
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.detection import Detection
from models.ema import ModelEMA
from training.lr_scheduler import LRScheduler


def lr_lambda_fn(scheduler: Any, step: int) -> float:
    return scheduler.update_lr(step)


class LNDetection(L.LightningModule):
    def __init__(self, cfg, train_dataloader_len=None):
        super().__init__()
        self.cfg = cfg
        self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

        self.model = Detection(cfg)
        self.ema_model = ModelEMA(self.model, decay=cfg.get("ema_decay", 0.9998))

        train_cfg = cfg.get("training", {})
        self.batch_size = train_cfg.get("batch_size", 8)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.max_epoch = train_cfg.get("epochs", 100)
        self.warmup_lr = train_cfg.get("warmup_lr", 0)
        self.min_lr_ratio = train_cfg.get("min_lr_ratio", 0.05)
        # base lr is per-image; LRScheduler scales it
        self.basic_lr_per_img = train_cfg.get("learning_rate", 1e-3) / self.batch_size
        self.scheduler_name = train_cfg.get("scheduler", "yoloxwarmcos")
        self.no_aug_epochs = train_cfg.get("no_aug_epochs", 15)
        self.weight_decay = train_cfg.get("weight_decay", 5e-4)
        self.momentum = train_cfg.get("momentum", 0.9)
        self.use_ema = cfg.get("ema", True)

        self.train_dataloader_len = train_dataloader_len
        self._val_pred = None
        self._test_pred = None

    # ---- Optimizer & scheduler ----

    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        # When using warmup, set initial lr=1 so LambdaLR acts as an absolute setter
        lr = 1 if self.warmup_epochs > 0 else self.basic_lr_per_img * batch_size
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def get_lr_scheduler(self, lr: float, iters_per_epoch: int) -> LRScheduler:
        return LRScheduler(
            self.scheduler_name,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )

    def configure_optimizers(self):
        optimizer = self.get_optimizer(self.batch_size)
        custom_sched = self.get_lr_scheduler(
            lr=self.basic_lr_per_img * self.batch_size,
            iters_per_epoch=self.train_dataloader_len,
        )
        torch_sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(lr_lambda_fn, custom_sched),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": torch_sched, "interval": "step", "frequency": 1},
        }

    # ---- Device management for EMA (not an nn.Module, so Lightning won't move it) ----

    def on_train_start(self):
        self.ema_model.ema = self.ema_model.ema.to(self.device)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):  # noqa: ARG002
        return batch.to(device)

    # ---- Training ----

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        total_loss = outputs["total_loss"]
        self.log("train/total_loss", total_loss, prog_bar=True, batch_size=self.batch_size)
        self.log("train/iou_loss", outputs["iou_loss"], prog_bar=True, batch_size=self.batch_size)
        self.log("train/conf_loss", outputs["conf_loss"], prog_bar=True, batch_size=self.batch_size)
        self.log("train/cls_loss", outputs["cls_loss"], prog_bar=True, batch_size=self.batch_size)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):  # noqa: ARG002
        if self.use_ema:
            self.ema_model.update(self.model)

    # ---- Checkpoint ----

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            # Replace main model weights with EMA weights so the saved checkpoint
            # represents the smoothed model on resume.
            ema_state = self.ema_model.ema.state_dict()
            checkpoint["state_dict"] = {f"model.{k}": v for k, v in ema_state.items()}

    # ---- Helpers ----

    def _infer_model(self, device):
        if self.use_ema:
            return self.ema_model.ema.to(device)
        return self.model

    def _get_gt(self, batch):
        """Extract GT boxes in xyxy format from the raw bboxes tensor."""
        gts = []
        unique_indices = batch.batch_bb.unique(sorted=True)
        for idx in unique_indices:
            mask = batch.batch_bb == idx
            bbox = batch.bboxes[mask].clone()   # (x, y, w, h, cls_idx)
            bbox[:, 2:4] += bbox[:, :2]         # → (x1, y1, x2, y2, cls_idx)
            gts.append({
                "boxes": bbox[:, :4].cpu(),
                "labels": bbox[:, 4].cpu().long(),
            })
        return gts

    # ---- Validation ----

    def validation_step(self, batch, batch_idx):
        preds = self._infer_model(batch.x.device)(batch)
        gts = self._get_gt(batch)
        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        self.map_metric.update(preds_cpu, gts)
        if self.cfg.get("log_detections", True) and self._val_pred is None:
            self._val_pred = {"batch": batch, "gts": gts, "preds": preds_cpu}

    def on_validation_epoch_end(self):
        maps = self.map_metric.compute()
        self.log("val/mAP",   maps["map"])
        self.log("val/mAP50", maps["map_50"])
        self.log("val/mAP75", maps["map_75"])
        self.log("val/mAP_S", maps["map_small"])
        self.log("val/mAP_M", maps["map_medium"])
        self.log("val/mAP_L", maps["map_large"])
        self.map_metric.reset()
        if self._val_pred is not None:
            self._log_detections(self._val_pred, "val")
            self._val_pred = None

    # ---- Test ----

    def test_step(self, batch, batch_idx):
        preds = self._infer_model(batch.x.device)(batch)
        gts = self._get_gt(batch)
        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        self.map_metric.update(preds_cpu, gts)
        if self.cfg.get("log_detections", False) and self._test_pred is None:
            self._test_pred = {"batch": batch, "gts": gts, "preds": preds_cpu}

    def on_test_epoch_end(self):
        maps = self.map_metric.compute()
        self.log("test/mAP",   maps["map"])
        self.log("test/mAP50", maps["map_50"])
        self.log("test/mAP75", maps["map_75"])
        self.log("test/mAP_S", maps["map_small"])
        self.log("test/mAP_M", maps["map_medium"])
        self.log("test/mAP_L", maps["map_large"])
        self.map_metric.reset()
        if self._test_pred is not None:
            self._log_detections(self._test_pred, "test")
            self._test_pred = None

    # ---- Detection logging ----

    def _log_detections(self, data: dict, split: str) -> None:
        batch = data["batch"]
        gts   = data["gts"]
        preds = data["preds"]

        num_classes = self.cfg.get("model", {}).get("num_classes", self.cfg.get("num_classes", 100))
        class_id_to_label = {i: str(i) for i in range(num_classes)}

        images = []
        for i, ev_img in enumerate(batch.frame):  # ev_img: numpy [H, W, 3] uint8
            pred_boxes = []
            for bb, label, score in zip(
                preds[i]["boxes"].cpu().numpy(),
                preds[i]["labels"].cpu().numpy(),
                preds[i]["scores"].cpu().numpy(),
            ):
                x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                pred_boxes.append({
                    "position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
                    "class_id": int(label),
                    "bbox_caption": f"{label} {score:.2f}",
                    "scores": {"score": float(score)},
                    "domain": "pixel",
                })

            gt_boxes = []
            for bb, label in zip(gts[i]["boxes"].numpy(), gts[i]["labels"].numpy()):
                x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                gt_boxes.append({
                    "position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
                    "class_id": int(label),
                    "bbox_caption": str(int(label)),
                    "scores": {"score": 1.0},
                    "domain": "pixel",
                })

            images.append(
                wandb.Image(ev_img, boxes={
                    "predictions": {"box_data": pred_boxes, "class_labels": class_id_to_label},
                    "ground_truth": {"box_data": gt_boxes,  "class_labels": class_id_to_label},
                })
            )

        self.logger.experiment.log({f"{split}/predictions": images})