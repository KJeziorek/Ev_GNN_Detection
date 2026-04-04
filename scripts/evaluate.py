import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from datasets.ncaltech101 import NCaltech101
from models.backbone import BACKBONE
from utils.data import GraphData


# ── helpers ──────────────────────────────────────────────────────────────────

def plot_graph(data: GraphData, title: str, ax, max_edges: int = 5000):
    """Scatter plot of nodes coloured by feature norm, with edges drawn."""
    pos = data.pos[:, :2].detach().cpu().numpy()
    feat_norm = data.x.detach().cpu().norm(dim=-1).numpy()
    edges = data.edge_index.detach().cpu().numpy()
    n_edges = edges.shape[0]

    # subsample edges if too many
    if n_edges > max_edges:
        idx = np.random.choice(n_edges, max_edges, replace=False)
        edges = edges[idx]

    for src, dst in edges:
        ax.plot([pos[src, 0], pos[dst, 0]],
                [pos[src, 1], pos[dst, 1]],
                c='gray', alpha=0.15, linewidth=0.3)

    sc = ax.scatter(pos[:, 0], pos[:, 1], c=feat_norm, s=3, cmap='viridis', zorder=2)
    plt.colorbar(sc, ax=ax, label='feature norm')
    ax.set_title(f"{title}\nN={pos.shape[0]}  E={n_edges}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')


def plot_graph_3d(data: GraphData, title: str, ax, max_edges: int = 8000):
    """3D scatter + edges showing the (x, y, t) structure of the graph."""
    pos = data.pos.detach().cpu().numpy()                  # [N, 3] -> x, y, t
    feat_norm = data.x.detach().cpu().norm(dim=-1).numpy()
    edges = data.edge_index.detach().cpu().numpy()
    n_edges = edges.shape[0]

    # subsample edges if too many
    if n_edges > max_edges:
        idx = np.random.choice(n_edges, max_edges, replace=False)
        edges = edges[idx]

    # draw edges as a Line3DCollection (much faster than individual plot calls)
    segments = np.stack([pos[edges[:, 0]], pos[edges[:, 1]]], axis=1)  # [E, 2, 3]
    lc = Line3DCollection(segments, colors='gray', alpha=0.12, linewidths=0.3)
    ax.add_collection3d(lc)

    # scatter nodes
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=feat_norm, s=2, cmap='viridis', depthshade=True)
    plt.colorbar(sc, ax=ax, label='feature norm', shrink=0.5)

    ax.set_title(f"{title}\nN={pos.shape[0]}  E={n_edges}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.view_init(elev=25, azim=-60)


# ── config ───────────────────────────────────────────────────────────────────

cfg_path = Path(__file__).resolve().parent.parent / "configs" / "ncaltech101.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

cfg["training"]["batch_size"] = 1
cfg["training"]["num_workers"] = 0

device = cfg.get("device", "cuda")

# ── dataset ──────────────────────────────────────────────────────────────────

dm = NCaltech101(cfg)
dm.setup()

test_loader = dm.test_dataloader()
batch = next(iter(test_loader)).to(device)

# ── run backbone stage-by-stage ──────────────────────────────────────────────

backbone = BACKBONE(cfg).to(device)

# load pretrained weights from Detection checkpoint
ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "ncaltech101" / "best.pth"
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    full_sd = state["model"] if "model" in state else state
    # extract backbone.* keys and strip the prefix
    bb_sd = {k.replace("backbone.", "", 1): v for k, v in full_sd.items() if k.startswith("backbone.")}
    backbone.load_state_dict(bb_sd)
    print(f"Loaded backbone weights from {ckpt_path}")
else:
    print(f"WARNING: {ckpt_path} not found — using random weights")

backbone.eval()

with torch.no_grad():
    scale = batch.pos.new_tensor([240.0, 180.0])

    stages = [("Input", batch.clone())]

    data = batch.clone()
    pos = data.pos.clone()
    data.pos[:, :2] = data.pos[:, :2] / scale
    data = backbone.blocks[0](data)
    data.pos = pos
    stages.append(("After block0", data.clone()))

    for i, (pool, block) in enumerate(zip(backbone.pools, backbone.blocks[1:])):
        data = pool(data)
        scale = scale / pool.pool_size[:2].float().to(scale.device)
        # scale = scale / torch.tensor([pool.pool_x, pool.pool_y]).to(scale.device)
        stages.append((f"After pool{i}", data.clone()))

        pos = data.pos.clone()
        data.pos[:, :2] = data.pos[:, :2] / scale
        data = block(data)
        data.pos = pos
        stages.append((f"After block{i+1}", data.clone()))

# ── plot ─────────────────────────────────────────────────────────────────────

n = len(stages)
ncols = 5
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
axes = axes.flatten()

for ax, (title, data) in zip(axes, stages):
    plot_graph(data, title, ax)

fig.suptitle("Graph visualisation through backbone stages", fontsize=16)
fig.tight_layout()
plt.savefig("backbone_graph_vis.png", dpi=150)
plt.show()
print("Saved to backbone_graph_vis.png")

# ── 3D plot ─────────────────────────────────────────────────────────────────

fig3d = plt.figure(figsize=(7 * ncols, 7 * nrows))

for i, (title, data) in enumerate(stages):
    ax = fig3d.add_subplot(nrows, ncols, i + 1, projection='3d')
    plot_graph_3d(data, title, ax)

fig3d.suptitle("3D graph visualisation (x, y, t) through backbone stages", fontsize=16)
fig3d.tight_layout()
plt.savefig("backbone_graph_vis_3d.png", dpi=150)
plt.show()
print("Saved to backbone_graph_vis_3d.png")
