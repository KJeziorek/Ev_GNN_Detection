import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import QuantizableLayer
from .my_linear import MyLinear


class SurrogateSpike(torch.autograd.Function):
    """Hard threshold in forward, sigmoid surrogate gradient in backward."""

    @staticmethod
    def forward(ctx, potential, threshold, temperature):
        fired = (potential > threshold).float()
        ctx.save_for_backward(potential, threshold, temperature)
        return fired

    @staticmethod
    def backward(ctx, grad_output):
        potential, threshold, temperature = ctx.saved_tensors
        sig = torch.sigmoid((potential - threshold) / temperature.clamp(min=1e-3))
        surrogate = sig * (1.0 - sig) / temperature.clamp(min=1e-3)
        grad_potential = grad_output * surrogate
        grad_threshold = -grad_potential
        return grad_potential, grad_threshold, None


class LIFSpikePool(QuantizableLayer):
    """
    True LIF spiking pooling layer for event data — multi-fire.

    Grid of neurons at fixed positions.  Input events route to their
    grid cell via integer division, integrate into membrane.  Neurons
    can fire MULTIPLE TIMES within a single forward pass: events are
    split into temporal bins and processed sequentially with
    decay → integrate → threshold → emit → reset per bin.

    Training  (forward)
    -------------------
    Events sorted by timestamp and split into T temporal bins.
    Each bin: decay membrane, scatter_add, surrogate-threshold, emit,
    differentiable reset.  A neuron receiving enough input across
    bins fires multiple spikes — each becomes a distinct output event.

    Edges are built from three sources:
    (a) Coarsened input edges — input topology projected through grid
    (b) Grid adjacency — 8-connected neighbours with recent activity
    (c) Temporal self-edges — consecutive spikes from the same cell

    Inference  (step + flush)
    -------------------------
    Identical logic with persistent membrane + temporal decay.
    Input edges coarsened per micro-batch.  Cross-step edges from
    grid adjacency history.

    FPGA mapping
    ------------
    Routing    → address decoder  (floor division, combinational)
    Membrane   → accumulator register per cell
    Decay      → fixed-point exponential LUT per cell
    Threshold  → comparator per cell
    Emit       → linear MAC unit (shared or per-cell)
    Edge gen   → coarsen via cell LUT + fired-cell bitmask + adj ROM
    """

    def __init__(self, in_channels, out_channels,
                 grid_h=24, grid_w=18,
                 spatial_range=(240, 180),
                 fired_history_len=4,
                 num_bins=8):
        super().__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_neurons = grid_h * grid_w
        self.fired_history_len = fired_history_len
        self.num_bins = num_bins
        self.out_channels = out_channels

        # Fixed grid positions — deterministic HW mapping
        gx = torch.linspace(0, spatial_range[0], grid_w)
        gy = torch.linspace(0, spatial_range[1], grid_h)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        self.register_buffer(
            'neuron_pos',
            torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        )  # [num_neurons, 2]

        self.cell_w = spatial_range[0] / max(grid_w - 1, 1)
        self.cell_h = spatial_range[1] / max(grid_h - 1, 1)

        # Precompute grid adjacency (8-connected)
        self.register_buffer(
            'grid_adj', self._build_grid_adjacency(grid_h, grid_w)
        )  # [2, A]

        # Integration: transform input features before accumulation
        self.lin_integrate = MyLinear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

        # Emission: transform membrane → output spike features
        self.lin_emit = MyLinear(out_channels, out_channels)

        # Per-neuron LIF parameters
        self.tau = nn.Parameter(torch.ones(self.num_neurons, 1) * 2.0)
        self.threshold = nn.Parameter(torch.ones(self.num_neurons) * 1.0)
        self.temperature = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grid_adjacency(H, W):
        """Build sparse edge list for 8-connected grid adjacency.
        Returns [2, A] COO format."""
        src, dst = [], []
        for r in range(H):
            for c in range(W):
                idx = r * W + c
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            src.append(idx)
                            dst.append(nr * W + nc)
        if len(src) == 0:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.stack([
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
        ], dim=0)

    # ------------------------------------------------------------------
    # Routing (shared by train & eval)
    # ------------------------------------------------------------------

    def _route_to_grid(self, pos):
        """
        Map event positions → grid cell indices via floor division.
        O(1) per event.

        Args:
            pos: [N, 2+]
        Returns:
            cell_idx: [N]  long, in 0 .. num_neurons-1
        """
        col = (pos[:, 0] / self.cell_w).clamp(0, self.grid_w - 1).long()
        row = (pos[:, 1] / self.cell_h).clamp(0, self.grid_h - 1).long()
        return row * self.grid_w + col

    # ------------------------------------------------------------------
    # Training forward — multi-fire via temporal bins
    # ------------------------------------------------------------------

    def forward(self, x, pos, edge_index, timestamps, num_bins=None):
        """
        Multi-fire training forward.

        Events are sorted by timestamp and split into T temporal bins.
        Each bin is processed as: decay → integrate → threshold → emit
        → reset.  Neurons that accumulate enough charge across bins
        fire multiple times — each spike becomes a separate output event.

        Args:
            x:          [N, C_in]   input event features
            pos:        [N, 2+]     input event positions
            edge_index: [2, E]      input edges (PyG COO)
            timestamps: [N]         per-event timestamps
            num_bins:   int|None    temporal bins (default: self.num_bins)

        Returns:
            x_out:          [S, C_out]   features of all emitted spikes
            pos_out:        [S, 3]       positions + fire time (x, y, t)
            edge_index_out: [2, E']      edges among spikes
        """
        T = num_bins if num_bins is not None else self.num_bins
        N = x.size(0)
        C = self.out_channels
        M = self.num_neurons
        device = x.device

        # ---- Sort events by time ----
        order = timestamps.argsort()
        x = x[order]
        pos = pos[order]
        timestamps = timestamps[order]

        # Remap edge indices to sorted order
        inv_order = torch.empty(N, device=device, dtype=torch.long)
        inv_order[order] = torch.arange(N, device=device)
        edge_index = torch.stack([
            inv_order[edge_index[0]],
            inv_order[edge_index[1]],
        ], dim=0)

        # ---- Route + transform all events at once ----
        cell_idx = self._route_to_grid(pos)                       # [N]
        messages = self.bn(self.lin_integrate(x))                 # [N, C]

        # ---- Assign events to temporal bins ----
        t_min = timestamps[0]
        t_max = timestamps[-1]
        t_range = (t_max - t_min).clamp(min=1e-6)
        bin_idx = (
            (timestamps - t_min) / t_range * T
        ).long().clamp(0, T - 1)                                  # [N]

        # Precompute bin boundaries (latest timestamp per bin for decay)
        bin_times = torch.full((T,), t_min.item(), device=device)
        for t in range(T):
            mask_t = bin_idx == t
            if mask_t.any():
                bin_times[t] = timestamps[mask_t][-1]

        # ---- Process temporal bins sequentially ----
        membrane = torch.zeros(M, C, device=device, dtype=x.dtype)
        threshold = self.threshold.abs()
        prev_time = t_min

        # Accumulators for output spikes
        all_feat = []       # list of [S_t, C]
        all_pos = []        # list of [S_t, 3]  (x, y, t_fire)
        all_cell = []       # list of [S_t]  global cell ids
        all_bin = []        # list of [S_t]  bin index

        for t in range(T):
            event_mask = bin_idx == t
            if not event_mask.any():
                continue

            # -- Temporal decay --
            dt = bin_times[t] - prev_time
            prev_time = bin_times[t]
            if dt > 0:
                decay = torch.exp(
                    -dt / self.tau.abs().clamp(min=1e-3)
                ).squeeze(-1)                                     # [M]
                membrane = membrane * decay.unsqueeze(-1)

            # -- Integrate this bin's events (non-inplace for autograd) --
            idx_t = cell_idx[event_mask]
            msg_t = messages[event_mask]
            increment = torch.zeros(
                M, C, device=device, dtype=x.dtype
            )
            increment.scatter_add_(
                0, idx_t.unsqueeze(-1).expand_as(msg_t), msg_t
            )
            membrane = membrane + increment                       # autograd-safe

            # -- Threshold (differentiable) --
            potential = membrane.norm(dim=-1)                     # [M]
            fired = SurrogateSpike.apply(
                potential, threshold, self.temperature
            )                                                     # [M] {0,1}
            fired_mask = fired.bool()

            if fired_mask.any():
                # Gate → emit
                gated = fired.unsqueeze(-1) * membrane            # [M, C]
                spike_feat = F.relu(
                    self.lin_emit(gated[fired_mask])
                )                                                 # [S_t, C]
                spike_cell = fired_mask.nonzero(as_tuple=True)[0] # [S_t]

                all_feat.append(spike_feat)
                # Position = [x, y, t_fire] where t_fire is the
                # bin's representative timestamp
                n_spikes_t = spike_cell.size(0)
                fire_time = bin_times[t].expand(n_spikes_t, 1)
                all_pos.append(torch.cat([
                    self.neuron_pos[fired_mask],                   # [S_t, 2]
                    fire_time,                                     # [S_t, 1]
                ], dim=1))                                         # [S_t, 3]
                all_cell.append(spike_cell)
                all_bin.append(
                    torch.full_like(spike_cell, t)
                )

                # Differentiable reset: zero out fired neurons
                # Gradient flows through (1 - fired) for non-fired,
                # and through gated emission for fired.
                membrane = membrane * (1.0 - fired.unsqueeze(-1))

        # ---- Collect outputs ----
        if len(all_feat) == 0:
            return (
                torch.zeros(0, C, device=device),
                torch.zeros(0, 3, device=device),
                torch.zeros(2, 0, device=device, dtype=torch.long),
            )

        x_out = torch.cat(all_feat)                              # [S, C]
        pos_out = torch.cat(all_pos)                              # [S, 3]
        spike_cells = torch.cat(all_cell)                         # [S]
        spike_bins = torch.cat(all_bin)                           # [S]
        S = x_out.size(0)

        # ---- Build output edges ----
        edge_out = self._build_multifire_edges(
            edge_index, cell_idx, spike_cells, spike_bins, S
        )

        return x_out, pos_out, edge_out

    def _build_multifire_edges(self, edge_index, cell_idx,
                               spike_cells, spike_bins, S):
        """
        Build output edges for multi-fire forward from three sources:

        1. Coarsened input edges — project input topology through grid.
           For each distinct (cell_a, cell_b) pair, connect their spikes.
        2. Grid adjacency — connect spikes from 8-connected cells that
           both fired (same or adjacent temporal bins).
        3. Temporal self-edges — consecutive spikes from the same cell
           are connected, ensuring multi-fire neurons form a chain.

        All edges are causal: source_bin <= dest_bin.

        Args:
            edge_index:  [2, E]  input event edges
            cell_idx:    [N]     cell assignment per input event
            spike_cells: [S]     global cell id per spike
            spike_bins:  [S]     temporal bin per spike
            S:           int     total number of spikes

        Returns:
            [2, E']  output edges in local spike-index space
        """
        device = spike_cells.device
        M = self.num_neurons
        edges_list = []

        if S == 0:
            return torch.zeros(2, 0, device=device, dtype=torch.long)

        # === Map: cell → latest spike index ===
        # Used for coarsened input edges and grid adjacency
        cell_to_latest = torch.full(
            (M,), -1, device=device, dtype=torch.long
        )
        # scatter with ascending indices → last write wins = latest spike
        spike_indices = torch.arange(S, device=device)
        cell_to_latest.scatter_(0, spike_cells, spike_indices)

        cell_has_spike = cell_to_latest >= 0

        # === Source 1: Coarsened input edges ===
        if edge_index.numel() > 0:
            src_cell = cell_idx[edge_index[0]]                    # [E]
            dst_cell = cell_idx[edge_index[1]]                    # [E]

            # Distinct cell pairs where both have spikes
            diff = src_cell != dst_cell
            both_have = cell_has_spike[src_cell] & cell_has_spike[dst_cell]
            keep = diff & both_have

            if keep.any():
                pairs = torch.stack(
                    [src_cell[keep], dst_cell[keep]], dim=0
                )
                pairs = torch.unique(pairs, dim=1)                # [2, E']

                local_src = cell_to_latest[pairs[0]]
                local_dst = cell_to_latest[pairs[1]]
                # diff already ensures different cells → different spike
                # indices, so no self-loop filtering needed here
                edges_list.append(torch.stack(
                    [local_src, local_dst], dim=0
                ))

        # === Source 2: Grid adjacency ===
        if self.grid_adj.numel() > 0:
            adj_src_cell = self.grid_adj[0]                       # [A]
            adj_dst_cell = self.grid_adj[1]                       # [A]

            both_active = (
                cell_has_spike[adj_src_cell] &
                cell_has_spike[adj_dst_cell]
            )
            if both_active.any():
                ls = cell_to_latest[adj_src_cell[both_active]]
                ld = cell_to_latest[adj_dst_cell[both_active]]

                # Causal: source bin <= dest bin
                # (grid adj connects distinct cells → distinct spikes,
                #  so no self-loop filtering needed here)
                causal = spike_bins[ls] <= spike_bins[ld]
                if causal.any():
                    edges_list.append(
                        torch.stack([ls[causal], ld[causal]], dim=0)
                    )

        # === Source 3: Temporal self-edges ===
        # Connect consecutive spikes from the same cell in time order.
        # Sort by (cell, bin) → consecutive entries from same cell are
        # adjacent → connect them.
        if S > 1:
            sort_key = spike_cells.long() * (self.num_bins + 1) + spike_bins
            sorted_order = sort_key.argsort()
            sorted_cells = spike_cells[sorted_order]

            same_cell = sorted_cells[:-1] == sorted_cells[1:]
            if same_cell.any():
                orig_src = sorted_order[:-1][same_cell]
                orig_dst = sorted_order[1:][same_cell]
                edges_list.append(
                    torch.stack([orig_src, orig_dst], dim=0)
                )

        # === Source 4: Self-loops for every output spike ===
        # Ensures each node aggregates its own features in downstream
        # message-passing (standard in GNN layers).
        self_idx = torch.arange(S, device=device)
        edges_list.append(torch.stack([self_idx, self_idx], dim=0))

        # === Merge & deduplicate ===
        all_edges = torch.cat(edges_list, dim=1)
        return torch.unique(all_edges, dim=1)

    # ------------------------------------------------------------------
    # Edge coarsening (used by inference path)
    # ------------------------------------------------------------------

    def _coarsen_edges(self, edge_index, cell_idx, fired_mask):
        """
        Project input edges through grid routing.  Keep edges where
        both endpoint cells fired and are distinct.  Remap to local
        fired-neuron indices.

        Returns [2, E'] in local space.
        """
        if edge_index.numel() == 0:
            return torch.zeros(
                2, 0, device=edge_index.device, dtype=torch.long
            )

        src_cell = cell_idx[edge_index[0]]
        dst_cell = cell_idx[edge_index[1]]

        diff = src_cell != dst_cell
        both = fired_mask[src_cell] & fired_mask[dst_cell]
        keep = diff & both

        src_cell = src_cell[keep]
        dst_cell = dst_cell[keep]

        if src_cell.numel() == 0:
            return torch.zeros(
                2, 0, device=edge_index.device, dtype=torch.long
            )

        coarse = torch.stack([src_cell, dst_cell], dim=0)
        coarse = torch.unique(coarse, dim=1)

        cell_to_local = torch.cumsum(fired_mask.long(), dim=0) - 1
        return torch.stack([
            cell_to_local[coarse[0]],
            cell_to_local[coarse[1]],
        ], dim=0)

    def _coarsen_edges_inference(self, edge_index, cell_idx, fired_mask):
        """
        Same as _coarsen_edges but returns GLOBAL cell indices
        (remapped later in flush).
        """
        if edge_index.numel() == 0:
            return torch.zeros(
                2, 0, device=fired_mask.device, dtype=torch.long
            )

        src_cell = cell_idx[edge_index[0]]
        dst_cell = cell_idx[edge_index[1]]

        diff = src_cell != dst_cell
        both = fired_mask[src_cell] & fired_mask[dst_cell]
        keep = diff & both

        if not keep.any():
            return torch.zeros(
                2, 0, device=fired_mask.device, dtype=torch.long
            )

        coarse = torch.stack([src_cell[keep], dst_cell[keep]], dim=0)
        return torch.unique(coarse, dim=1)

    # ------------------------------------------------------------------
    # Event-by-event inference
    # ------------------------------------------------------------------

    def init_state(self, device, dtype=torch.float32):
        """Call once before the eval loop."""
        C = self.out_channels

        self._membrane = torch.zeros(
            self.num_neurons, C, device=device, dtype=dtype
        )
        self._current_time = 0.0

        # Sliding window of recently-fired cell masks
        self._fired_history = torch.zeros(
            self.fired_history_len, self.num_neurons,
            device=device, dtype=torch.bool
        )
        self._history_write = 0

        # Accumulators between flush() calls
        self._pending_feat = []
        self._pending_pos = []
        self._pending_edges = []
        self._pending_cell_ids = []

    @torch.no_grad()
    def step(self, x_event, pos_event, edge_index, timestamp):
        """
        Process a micro-batch of events with edges.

        Edge strategy (matches forward):
        1. Coarsen input edges through grid routing
        2. Grid-adjacency edges to recently-fired cells

        Args:
            x_event:    [B, C_in]  (or [C_in])
            pos_event:  [B, 2+]    (or [2+])
            edge_index: [2, E]     edges among the B input events
            timestamp:  float

        Returns:
            n_fired: int
        """
        if x_event.dim() == 1:
            x_event = x_event.unsqueeze(0)
            pos_event = pos_event.unsqueeze(0)

        device = x_event.device

        # -- Temporal decay --
        dt = timestamp - self._current_time
        self._current_time = timestamp
        if dt > 0:
            decay = torch.exp(
                -dt / self.tau.abs().clamp(min=1e-3)
            ).squeeze(-1)
            self._membrane *= decay.unsqueeze(-1)

        # -- Route & integrate --
        cell_idx = self._route_to_grid(pos_event)
        messages = self.lin_integrate(x_event)
        self._membrane.scatter_add_(
            0,
            cell_idx.unsqueeze(-1).expand_as(messages),
            messages,
        )

        # -- Apply BN (eval mode: running stats + affine) --
        membrane_normed = self.bn(self._membrane)

        # -- Hard threshold --
        potential = membrane_normed.norm(dim=-1)
        fired_mask = potential > self.threshold.abs()

        n_fired = int(fired_mask.sum().item())
        if n_fired == 0:
            return 0

        fired_idx = fired_mask.nonzero(as_tuple=True)[0]

        # -- Emit --
        spike_feat = F.relu(self.lin_emit(membrane_normed[fired_idx]))
        spike_xy = self.neuron_pos[fired_idx]                     # [S, 2]
        fire_time = torch.full(
            (spike_xy.size(0), 1), timestamp,
            device=device, dtype=spike_xy.dtype
        )
        spike_pos = torch.cat([spike_xy, fire_time], dim=1)       # [S, 3]

        # -- Edges: coarsen input + grid-adjacency history --
        step_edges = self._coarsen_edges_inference(
            edge_index, cell_idx, fired_mask
        )
        history_edges = self._history_edges(fired_mask)

        parts = [e for e in (step_edges, history_edges) if e.numel() > 0]
        if parts:
            combined = torch.unique(torch.cat(parts, dim=1), dim=1)
        else:
            combined = torch.zeros(2, 0, device=device, dtype=torch.long)

        # -- Store pending --
        self._pending_feat.append(spike_feat)
        self._pending_pos.append(spike_pos)
        self._pending_edges.append(combined)
        self._pending_cell_ids.append(fired_idx)

        # -- Update history --
        self._fired_history[self._history_write] = fired_mask
        self._history_write = (
            (self._history_write + 1) % self.fired_history_len
        )

        # -- Reset fired neurons --
        self._membrane[fired_mask] = 0.0

        return n_fired

    def _history_edges(self, current_fired):
        """
        Connect newly-fired cells to recently-fired grid neighbours.
        Returns [2, E'] in global cell-index space.
        """
        if self.grid_adj.numel() == 0:
            return torch.zeros(
                2, 0, device=current_fired.device, dtype=torch.long
            )

        recent = self._fired_history.any(dim=0)

        adj_src = self.grid_adj[0]
        adj_dst = self.grid_adj[1]
        keep = current_fired[adj_src] & recent[adj_dst]

        if not keep.any():
            return torch.zeros(
                2, 0, device=current_fired.device, dtype=torch.long
            )

        return torch.stack([adj_src[keep], adj_dst[keep]], dim=0)

    def flush(self):
        """
        Collect all spikes since last flush into a single graph.
        Remaps global cell edges → local spike indices.

        Returns:
            x:          [S, C]   spike features
            pos:        [S, 3]   spike positions + fire time (x, y, t)
            edge_index: [2, E]   local edges
        """
        device = self._membrane.device
        C = self.out_channels

        if len(self._pending_feat) == 0:
            return (
                torch.zeros(0, C, device=device),
                torch.zeros(0, 3, device=device),
                torch.zeros(2, 0, device=device, dtype=torch.long),
            )

        x_out = torch.cat(self._pending_feat, dim=0)
        pos_out = torch.cat(self._pending_pos, dim=0)
        all_cell_ids = torch.cat(self._pending_cell_ids, dim=0)
        S = x_out.size(0)

        # cell → latest local spike index
        cell_to_local = torch.full(
            (self.num_neurons,), -1, device=device, dtype=torch.long
        )
        spike_indices = torch.arange(S, device=device)
        cell_to_local.scatter_(0, all_cell_ids, spike_indices)

        if len(self._pending_edges) > 0:
            all_e = torch.cat(self._pending_edges, dim=1)
            all_e = torch.unique(all_e, dim=1)

            local_src = cell_to_local[all_e[0]]
            local_dst = cell_to_local[all_e[1]]

            # Keep edges where both endpoints have pending spikes
            valid = (local_src >= 0) & (local_dst >= 0)

            edge_out = torch.stack(
                [local_src[valid], local_dst[valid]], dim=0
            )
        else:
            edge_out = torch.zeros(2, 0, device=device, dtype=torch.long)

        # Add self-loops for every output spike
        self_idx = torch.arange(S, device=device)
        self_loops = torch.stack([self_idx, self_idx], dim=0)
        edge_out = torch.cat([edge_out, self_loops], dim=1)
        edge_out = torch.unique(edge_out, dim=1)

        self._pending_feat.clear()
        self._pending_pos.clear()
        self._pending_edges.clear()
        self._pending_cell_ids.clear()

        return x_out, pos_out, edge_out

    # ------------------------------------------------------------------
    # Quantization support
    # ------------------------------------------------------------------

    def _get_quantizable_modules(self):
        return [self.lin_integrate, self.lin_emit]
