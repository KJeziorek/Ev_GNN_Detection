import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
from time import time
from datasets.ncaltech101 import NCaltech101
from models.model import Detection


# ── Load config ──────────────────────────────────────────────────────────────
cfg_path = Path(__file__).resolve().parent.parent / "configs" / "ncaltech101.yaml"
with open(cfg_path) as f:
    raw_cfg = yaml.safe_load(f)

# Flatten nested config into a single dict for the DataModule
cfg = {**raw_cfg["data"], **raw_cfg["graph"]}
cfg["batch_size"] = raw_cfg["training"]["batch_size"]
cfg["num_workers"] = raw_cfg["training"]["num_workers"]

device = raw_cfg.get("device", "cuda")
num_classes = raw_cfg["model"]["num_classes"]
spatial_range = tuple(raw_cfg["model"]["spatial_range"])

# ── Setup dataset ────────────────────────────────────────────────────────────
dm = NCaltech101(cfg)
dm.setup()

print(f"Classes: {dm.num_classes}")
print(f"Train: {len(dm.train_data)}, Val: {len(dm.val_data)}, Test: {len(dm.test_data)}")

# ── Setup model ──────────────────────────────────────────────────────────────
model = Detection(num_classes=num_classes, spatial_range=spatial_range).to(device)


# ── Run evaluation on a few batches ──────────────────────────────────────────
test_loader = dm.test_dataloader()

for i, batch in enumerate(test_loader):
    t0 = time()

    batch = batch.to(device)

    with torch.no_grad():
        outputs = model(batch)
    
    print(outputs)
    t1 = time()

    # outputs is a list of dicts with boxes/scores/labels
    # n_dets = sum(r["boxes"].shape[0] for r in outputs)
    # print(f"Batch {i}: detections={n_dets}, time={t1 - t0:.3f}s")

    if i >= 4:
        break
