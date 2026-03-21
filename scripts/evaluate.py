import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets.graph_gen.matrix_neighbour import GraphGenerator
from datasets.ncaltech101 import NCaltech101
from models.backbone import BACKBONE
from models.layers.my_pooling import LIFSpikePool

raw_file = "/home/imperator/Datasets/NCaltech101/Caltech101/accordion/image_0001.bin"
ann_file = "/home/imperator/Datasets/NCaltech101/Caltech101_annotations/accordion/annotation_0001.bin"


gen = GraphGenerator(width=240, height=180)  # NCaltech101 resolution

ds = NCaltech101()
layer = BACKBONE().to('cuda')
events = ds.load_events(raw_file)
ann = ds.load_annotations(ann_file)


# from time import time

# for i in range(100):
#     t0 = time()
#     X, P, E = gen.generate_edges(events, radius_x=5, radius_y=5, radius_t=0.001)
#     gen.clear()
#     out = layer(X.to('cuda'), P.to('cuda'), E.to('cuda'))
#     t1 = time()
#     print(t1 - t0, out)


X, P, E = gen.generate_edges(events, radius_x=5, radius_y=5, radius_t=0.001)
pool1 = LIFSpikePool(1, 1, 60, 45, (240,180), num_bins=50)
X1, P1, E1 = pool1(X, P, E.T, P[:, -1])
E1 = E1.T

print(X.shape, P.shape, E.shape)
print(X1.shape, P1.shape, E1.shape)

# --- Visualise graphs before and after pooling (3D: x, y, t) ---
def plot_graph_3d(ax, pos, edges, title, node_size=1, edge_alpha=0.15):
    """Plot a 3D graph given positions [N, 3+] (x, y, t) and edges [E, 2]."""
    pos_np = pos[:, :3].detach().cpu().numpy()
    edges_np = edges.detach().cpu().numpy()

    xs, ys, ts = pos_np[:, 0], pos_np[:, 1], pos_np[:, 2]

    # Draw edges
    for i in range(edges_np.shape[0]):
        src, dst = int(edges_np[i, 0]), int(edges_np[i, 1])
        if src == dst:
            continue
        ax.plot(
            [xs[src], xs[dst]],
            [ys[src], ys[dst]],
            [ts[src], ts[dst]],
            color='steelblue', linewidth=0.3, alpha=edge_alpha,
        )

    # Draw nodes coloured by timestamp
    sc = ax.scatter(xs, ys, ts, c=ts, cmap='plasma',
                    s=node_size, edgecolors='none', depthshade=True)
    plt.colorbar(sc, ax=ax, label='timestamp', fraction=0.03, pad=0.1)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    ax.invert_yaxis()

fig = plt.figure(figsize=(18, 8))

# Subsample edges for the pre-pooling plot if there are too many
max_edges = 20000
E_plot = E
if E.shape[0] > max_edges:
    idx = np.random.choice(E.shape[0], max_edges, replace=False)
    E_plot = E[idx]

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_graph_3d(ax1, P, E_plot,
              f'Before pooling  ({X.shape[0]} nodes, {E.shape[0]} edges)',
              node_size=0.5)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_graph_3d(ax2, P1, E1,
              f'After pooling  ({X1.shape[0]} nodes, {E1.shape[0]} edges)',
              node_size=6)

fig.suptitle('Graph before & after LIF spike pooling', fontsize=14)
fig.tight_layout()
plt.savefig('graph_pooling_comparison.png', dpi=200)
plt.show()
print("Saved to graph_pooling_comparison.png")