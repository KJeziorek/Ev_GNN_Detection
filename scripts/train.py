"""
Lightning training script for GNN YOLOX detection on NCaltech101.

Usage:
    python -m scripts.train --config configs/ncaltech101.yaml
    python -m scripts.train --config configs/ncaltech101.yaml --resume checkpoints/last.ckpt
    python -m scripts.train --config configs/ncaltech101.yaml --run-name my_exp --tags exp1 v2
    python -m scripts.train --config configs/ncaltech101.yaml --offline
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from datasets.ncaltech101 import NCaltech101
from training.trainer import LNDetection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str,  default="configs/ncaltech101.yaml")
    parser.add_argument("--resume",   type=str,  default=None,
                        help="Path to .ckpt file to resume training from")
    parser.add_argument("--run-name", type=str,  default=None,
                        help="W&B run name (defaults to W&B auto-generated name)")
    parser.add_argument("--tags",     nargs="*", default=None,
                        help="W&B tags, e.g. --tags baseline v1")
    parser.add_argument("--offline",  action="store_true",
                        help="Run W&B in offline mode (sync later with wandb sync)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    log_cfg   = cfg.get("logging",  {})

    # ---- Datamodule ----
    datamodule = NCaltech101(cfg)
    datamodule.setup()

    print(f"Classes : {datamodule.num_classes}")
    print(f"Train   : {len(datamodule.train_data)}")
    print(f"Val     : {len(datamodule.val_data)}")
    print(f"Test    : {len(datamodule.test_data)}")

    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()

    # ---- Model ----
    lightning_model = LNDetection(cfg, train_dataloader_len=len(train_loader))

    total     = sum(p.numel() for p in lightning_model.model.parameters())
    trainable = sum(p.numel() for p in lightning_model.model.parameters() if p.requires_grad)
    print(f"Params  : {total:,} total | {trainable:,} trainable")

    # ---- Logger ----
    wandb_logger = WandbLogger(
        project=log_cfg.get("wandb_project", "ev-gnn-detection"),
        name=args.run_name,
        tags=args.tags,
        offline=args.offline,
        config=cfg,
        save_dir=log_cfg.get("log_dir", "logs"),
    )

    # ---- Callbacks ----
    ckpt_dir = log_cfg.get("checkpoint_dir", "checkpoints/ncaltech101")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val/mAP50",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/mAP50",
            patience=train_cfg.get("patience", 20),
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=train_cfg.get("epochs", 200),
        accelerator="gpu",
        devices=1,
        precision=train_cfg.get("precision", "32-true"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 10.0),
        logger=wandb_logger,
        callbacks=callbacks,
        check_val_every_n_epoch=train_cfg.get("val_every_n_epoch", 5),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=True,
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )