"""Main training script.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_seed, load_config, count_parameters
from models import GCNModel, GATModel
from training import Trainer


MODEL_REGISTRY = {
    "GCN": GCNModel,
    "GAT": GATModel,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    # --- Build model ---
    mc = config["model"]
    model_cls = MODEL_REGISTRY[mc["name"]]
    model = model_cls(
        in_channels=mc["in_channels"],
        hidden_channels=mc["hidden_channels"],
        out_channels=mc["out_channels"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
    )
    print(f"Model: {mc['name']} | Parameters: {count_parameters(model):,}")

    # --- Load datasets ---
    # TODO: Replace with your actual dataset loading
    # train_dataset = GraphDetectionDataset(root="data", split="train")
    # val_dataset   = GraphDetectionDataset(root="data", split="val")
    raise NotImplementedError(
        "Load your datasets here. See datasets/graph_dataset.py for the template."
    )

    # --- Train ---
    # trainer = Trainer(model, train_dataset, val_dataset, config)
    # best_val_loss = trainer.fit()
    # print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
