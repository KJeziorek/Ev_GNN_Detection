

cd datasets/graph_gen && pip install pybind11 && python setup.py build_ext --inplace



For now the best config:

REMEMBER TO REMOVE AUGMENTATION FOR FINAL 100 EPOCHS

using classic PointNetConv

# NCaltech101 dataset & training configuration

data:
  data_dir: /home/imperator/Datasets/ncaltech101
  sensor_width: 240
  sensor_height: 180
  num_events: 50000
  sample_len: 99999
  slice_method: last_by_count  # first_by_time, last_by_time, first_by_count, last_by_count

norm:
  norm_w: 240
  norm_h: 180
  norm_t: 1000

augmentation:
  hflip_p: 0.5
  crop_size: [0.75, 0.75]
  crop_p: 0.2
  zoom_range: [0.5, 1.5]
  zoom_subsample: true
  translate_size: [0.1, 0.1]

graph:
  radius_x: 3
  radius_y: 3
  radius_t: 10

model:
  num_classes: 100
  spatial_range: [240, 180]

backbone:
  # channels[i] = output channels of block i (block0 input is always 1)
  channels: [36, 72, 72, 144, 144]
  # pool_sizes[i] = [stride_x, stride_y, stride_t] applied before block i+1
  pool_sizes:
    - [5.0, 5.0, 10.0]   # pool1: 240→48, 180→36
    - [2.0, 2.0, 10.0]    # pool2: 48→24,  36→18
    - [2.0, 2.0, 10.0]    # pool3: 24→12,  18→9
    - [1.0, 1.0, 1.0]    # pool4: identity (no spatial reduction)

head:
  # Number of detection scales = len(strides) = len(in_channels).
  # The backbone returns the last N feature maps (tail of backbone.pool_sizes stages).
  # in_channels must match backbone.channels[-N:] exactly.
  #
  # Single-scale (current):
  #   strides: [20]          in_channels: [256]   → backbone.channels[-1:]
  #
  # Two-scale example:
  #   strides: [10, 20]      in_channels: [256, 256]  → backbone.channels[-2:]
  #
  # Three-scale example:
  #   strides: [5, 10, 20]   in_channels: [128, 256, 256]  → backbone.channels[-3:]
  strides: [10, 20]
  in_channels: [144, 144]
  sparse_cfg:
    max_fg_per_gt: null        # null = unlimited
    fg_selection: recent       # recent | random | iou
    use_ignore_zone: false
    use_temporal_weighting: false
    temporal_min_weight: 0.1
    use_cell_aggregation: false

training:
  batch_size: 64
  num_workers: 16
  epochs: 801
  learning_rate: 0.001
  weight_decay: 0.00005
  momentum: 0.9
  scheduler: yoloxwarmcos
  warmup_epochs: 5
  warmup_lr: 0
  min_lr_ratio: 0.05
  no_aug_epochs: 15
  patience: 20
  precision: "16-mixed"
  gradient_clip_val: 100.0
  val_every_n_epoch: 10
  log_every_n_steps: 10

logging:
  log_dir: logs/ncaltech101
  checkpoint_dir: checkpoints/ncaltech101
  wandb_project: ev-gnn-detection
  save_every: 10

device: cuda


COnv:


class PointNetConv(nn.Module):
    """
    PointNet-style graph convolution.

    Args:
        input_dim:    Concatenated input size (node features + positional diff).
        output_dim:   Output feature size.
        bias:         Bias for the global_nn aggregation layer.
    Example::
        conv = MyPointNetConv(16, 64)
        for batch in loader:
            conv(x, pos, edge_index)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        # Float layers — only things created at construction time
        self.linear = nn.Linear(in_channels + 2, out_channels, bias=True)
        # self.global_nn = nn.Linear(out_channels, out_channels, bias=bias)
        self.global_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels*2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels*2, out_channels, bias=bias),
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        data: GraphData
        ) -> GraphData:

        x, pos, edge_index = data.x, data.pos[:, :2], data.edge_index
        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]
        x_j = x[edge_index[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)
        msg = self.linear(msg)
        out = self._scatter_amax(msg, edge_index)

        out_skip = out.clone()  # for potential skip connection
        out = self.global_nn(out)
        out = out + out_skip  # skip connection

        data.x = out
        return data



