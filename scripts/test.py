"""Quick smoke test: load one NCaltech101 sample and plot events + bbox."""
import sys
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, ".")

from datasets.ncaltech101 import NCaltech101


def load_config(path="configs/ncaltech101.yaml"):
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Flatten into a single dict for the datamodule
    cfg = {}
    for section in raw.values():
        if isinstance(section, dict):
            cfg.update(section)
        else:
            pass  # skip scalar top-level keys like 'device'
    return cfg


def plot_events(sample, class_name="", idx=0):
    pos = sample.pos  # [N, 3] -> x, y, t
    x_feat = sample.x  # [N, C] node features (polarity)
    bbox = sample.bboxes  # [1, 5] -> cls, x, y, w, h

    ex = pos[:, 0].numpy()
    ey = pos[:, 1].numpy()
    pol = x_feat[:, 0].numpy() if x_feat.shape[1] > 0 else None

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = pol if pol is not None else "blue"
    ax.scatter(ex, ey, c=colors, cmap="coolwarm", s=0.3, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()

    if bbox is not None and bbox.numel() > 0:
        for bb in bbox:
            cls_id, bx, by, bw, bh = bb.tolist()
            rect = patches.Rectangle(
                (bx, by), bw, bh,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(bx, by - 2, f"cls={int(cls_id)}", color="lime", fontsize=9)

    ax.set_title(f"Sample #{idx} — {class_name}  ({len(ex)} nodes)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"scripts/sample_{idx}.png", dpi=150)
    plt.show()
    print(f"Saved to scripts/sample_{idx}.png")


def main():
    cfg = load_config()
    dm = NCaltech101(cfg)
    dm.setup()

    idx_to_class = {v: k for k, v in dm.class_to_idx.items()}

    # Load a few samples from train set
    for i in range(30):
        sample = dm.train_data[i]
        cls_id = int(sample.bboxes[0, 0].item())
        class_name = idx_to_class.get(cls_id, "?")

        print(f"\n--- Sample {i} ---")
        print(f"  Class: {class_name} (id={cls_id})")
        print(f"  Nodes (x): {sample.x.shape}")
        print(f"  Positions (pos): {sample.pos.shape}")
        print(f"  Edges: {sample.edge_index.shape}")
        print(f"  Bboxes: {sample.bboxes}")

        plot_events(sample, class_name=class_name, idx=i)


if __name__ == "__main__":
    main()
