import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import QuantizableLayer
from .my_linear import MyLinear


class SurrogateSpike(torch.autograd.Function):
    """Hard threshold forward, sigmoid surrogate gradient backward."""

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
    LIF spiking pooling layer for batched event-graph data.

    Grid of neurons at fixed spatial positions. Input events are routed to
    grid cells via floor division, integrated into membrane potential, and
    emitted as output spikes when threshold is crossed.

    Multi-fire: events are split into temporal bins and processed
    sequentially (decay -> integrate -> threshold -> emit -> reset).
    All samples in a batch are processed in parallel within each bin.

    Input/output format matches the batched collate_fn from NCaltech101:
        x:          [N_total, C]    concatenated node features
        pos:        [N_total, 3]    positions (x, y, t)
        edge_index: [E_total, 2]    edges (src, dst), globally indexed
        batch:      [N_total]       sample index per node
    """

    def __init__(self, in_channels, out_channels,
                 grid_h=24, grid_w=18,
                 spatial_range=(240, 180),
                 fired_history_len=4,
                 num_bins=50):
        super().__init__()

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_neurons = grid_h * grid_w
        self.fired_history_len = fired_history_len
        self.num_bins = num_bins
        self.out_channels = out_channels

        # Fixed grid positions
        gx = torch.linspace(0, spatial_range[0], grid_w)
        gy = torch.linspace(0, spatial_range[1], grid_h)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        self.register_buffer(
            'neuron_pos',
            torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).round()
        )  # [num_neurons, 2]

        self.cell_w = spatial_range[0] / max(grid_w - 1, 1)
        self.cell_h = spatial_range[1] / max(grid_h - 1, 1)

        # 8-connected grid adjacency [A, 2]
        self.register_buffer(
            'grid_adj', self._build_grid_adjacency(grid_h, grid_w)
        )

        # Integration transform
        self.lin_integrate = MyLinear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

        # Emission transform
        self.lin_emit = MyLinear(out_channels, out_channels)
        self.bn_emit = nn.BatchNorm1d(out_channels)

        # Per-neuron LIF parameters
        self.tau = nn.Parameter(torch.ones(self.num_neurons, 1) * 2.0)
        self.threshold = nn.Parameter(torch.ones(self.num_neurons) * 1.0)
        self.temperature = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _build_grid_adjacency(H, W):
        """8-connected grid adjacency. Returns [A, 2] (src, dst)."""
        edges = []
        for r in range(H):
            for c in range(W):
                idx = r * W + c
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            edges.append((idx, nr * W + nc))
        if len(edges) == 0:
            return torch.zeros(0, 2, dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long)

    def _route_to_grid(self, pos):
        """Map event (x, y) positions to grid cell indices. pos: [N, 2+]."""
        col = (pos[:, 0] / self.cell_w).clamp(0, self.grid_w - 1).long()
        row = (pos[:, 1] / self.cell_h).clamp(0, self.grid_h - 1).long()
        return row * self.grid_w + col

    # ------------------------------------------------------------------
    # Batched forward (parallel across samples)
    # ------------------------------------------------------------------

    def forward(self, data, num_bins=None):
        """
        Batched multi-fire LIF forward. All samples processed in parallel.

        Membrane is [B*M, C] where each sample owns M neurons. Events are
        routed to sample-specific neurons via batch*M + cell_idx. Temporal
        bins are computed per-sample, then all samples advance together.

        Args:
            x:          [N_total, C_in]  concatenated node features
            pos:        [N_total, 3]     positions (x, y, t)
            edge_index: [E_total, 2]     edges (src, dst), globally indexed
            batch:      [N_total]        sample index per node
            num_bins:   int|None         temporal bins (default: self.num_bins)

        Returns:
            dict with keys:
                x:          [S_total, C_out]
                pos:        [S_total, 3]       (x, y, t_fire)
                edge_index: [E'_total, 2]
                batch:      [S_total]
        """
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        T = num_bins if num_bins is not None else self.num_bins
        N = x.size(0)
        C = self.out_channels
        M = self.num_neurons
        device = x.device

        if N == 0:
            data.x = torch.zeros(0, C, device=device)
            data.pos = torch.zeros(0, 3, device=device)
            data.edge_index = torch.zeros(0, 2, device=device, dtype=torch.long)
            data.batch = torch.zeros(0, device=device, dtype=torch.long)
            return data

        B = batch.max().item() + 1

        # ---- Sort events by time (globally) ----
        timestamps = pos[:, 2]
        order = timestamps.argsort()
        x = x[order]
        pos = pos[order]
        batch = batch[order]
        timestamps = timestamps[order]

        # Remap edges to sorted order
        inv_order = torch.empty(N, device=device, dtype=torch.long)
        inv_order[order] = torch.arange(N, device=device)
        if edge_index.numel() > 0:
            edge_index = torch.stack([
                inv_order[edge_index[:, 0]],
                inv_order[edge_index[:, 1]],
            ], dim=1)

        # ---- Route & transform all events ----
        cell_idx = self._route_to_grid(pos)                           # [N] local cell
        global_cell = batch * M + cell_idx                            # [N] batch-aware cell
        messages = self.bn(self.lin_integrate(x))                     # [N, C]

        # ---- Per-sample temporal binning ----
        # Compute t_min, t_max per sample
        t_min_per_sample = torch.full((B,), float('inf'), device=device)
        t_max_per_sample = torch.full((B,), float('-inf'), device=device)
        t_min_per_sample.scatter_reduce_(0, batch, timestamps, reduce='amin')
        t_max_per_sample.scatter_reduce_(0, batch, timestamps, reduce='amax')

        t_min_ev = t_min_per_sample[batch]                            # [N]
        t_range_ev = (t_max_per_sample[batch] - t_min_ev).clamp(min=1e-6)
        bin_idx = ((timestamps - t_min_ev) / t_range_ev * T).long().clamp(0, T - 1)

        # Per-sample bin boundary times: [B, T]
        # For decay we need the latest timestamp in each (sample, bin)
        bin_times = t_min_per_sample.unsqueeze(1).expand(B, T).clone()  # [B, T]
        flat_sb = batch * T + bin_idx                                   # [N]
        bin_times_flat = bin_times.reshape(-1)                          # [B*T]
        bin_times_flat.scatter_reduce_(0, flat_sb, timestamps, reduce='amax')
        bin_times = bin_times_flat.reshape(B, T)                       # [B, T]

        # ---- Batched membrane: [B*M, C] ----
        membrane = torch.zeros(B * M, C, device=device, dtype=x.dtype)

        # Broadcast LIF params across batch
        tau_bm = self.tau.abs().clamp(min=1e-3).squeeze(-1).repeat(B)  # [B*M]
        threshold_bm = self.threshold.abs().repeat(B)                  # [B*M]

        prev_time = t_min_per_sample.clone()                           # [B]

        # Accumulators
        all_feat = []
        all_pos = []
        all_global_cell = []    # batch-aware cell ids for edge building
        all_bin = []
        all_batch_idx = []      # which sample each spike belongs to

        for t in range(T):
            event_mask = bin_idx == t
            if not event_mask.any():
                continue

            # -- Decay (per-sample dt, broadcast to all M neurons) --
            dt = bin_times[:, t] - prev_time                          # [B]
            prev_time = bin_times[:, t]
            dt_bm = dt.repeat_interleave(M)                           # [B*M]
            decay_mask = dt_bm > 0
            if decay_mask.any():
                decay = torch.ones(B * M, device=device, dtype=x.dtype)
                decay[decay_mask] = torch.exp(-dt_bm[decay_mask] / tau_bm[decay_mask])
                membrane = membrane * decay.unsqueeze(-1)

            # -- Integrate (scatter_mean to avoid large accumulations) --
            gc_t = global_cell[event_mask]                            # batch-aware cells
            msg_t = messages[event_mask]
            increment = torch.zeros(B * M, C, device=device, dtype=x.dtype)
            increment.scatter_add_(0, gc_t.unsqueeze(-1).expand_as(msg_t), msg_t)
            counts = torch.zeros(B * M, 1, device=device, dtype=x.dtype)
            counts.scatter_add_(0, gc_t.unsqueeze(-1), torch.ones_like(gc_t, dtype=x.dtype).unsqueeze(-1))
            increment = increment / counts.clamp(min=1)
            membrane = membrane + increment

            # -- Threshold --
            potential = torch.linalg.vector_norm(membrane, dim=-1).clamp(min=1e-6)  # [B*M]
            fired = SurrogateSpike.apply(potential, threshold_bm, self.temperature)
            fired_mask = fired.bool()                                 # [B*M]

            if fired_mask.any():
                gated = fired.unsqueeze(-1) * membrane
                spike_feat = self.lin_emit(gated[fired_mask])

                # Which global cells fired
                fired_gc = fired_mask.nonzero(as_tuple=True)[0]       # global cell ids
                spike_batch = fired_gc // M                           # sample index
                spike_local_cell = fired_gc % M                       # local cell id

                all_feat.append(spike_feat)

                n_spikes = fired_gc.size(0)
                fire_time = bin_times[spike_batch, t].unsqueeze(1)    # [S_t, 1]
                all_pos.append(torch.cat([
                    self.neuron_pos[spike_local_cell],                # [S_t, 2]
                    fire_time,
                ], dim=1))

                all_global_cell.append(fired_gc)
                all_bin.append(torch.full((n_spikes,), t, device=device, dtype=torch.long))
                all_batch_idx.append(spike_batch)

                # Reset
                membrane = membrane * (1.0 - fired.unsqueeze(-1))

        # ---- Empty output ----
        if len(all_feat) == 0:

            data.x = torch.zeros(0, C, device=device)
            data.pos = torch.zeros(0, 3, device=device)
            data.edge_index = torch.zeros(0, 2, device=device, dtype=torch.long)
            data.batch = torch.zeros(0, device=device, dtype=torch.long)
            return data

        x_out = F.relu(self.bn_emit(torch.cat(all_feat)))                # [S, C]
        pos_out = torch.cat(all_pos)                                   # [S, 3]
        spike_gc = torch.cat(all_global_cell)                          # [S]
        spike_bins = torch.cat(all_bin)                                # [S]
        spike_batch = torch.cat(all_batch_idx)                         # [S]
        S = x_out.size(0)

        # ---- Build edges per sample, merge with offsets ----
        edge_out = self._build_batched_edges(
            edge_index, batch, cell_idx, spike_gc, spike_bins,
            spike_batch, B, M, S
        )

        data.x = x_out
        data.pos = pos_out
        data.edge_index = edge_out
        data.batch = spike_batch

        return data

    # ------------------------------------------------------------------
    # Edge building
    # ------------------------------------------------------------------

    def _build_batched_edges(self, edge_index, node_batch, cell_idx,
                             spike_gc, spike_bins, spike_batch, B, M, S):
        """
        Build output edges for all samples. Each sample's edges are built
        independently, then merged with spike-index offsets.

        Returns [E', 2] in global spike-index space.
        """
        device = spike_gc.device
        edges_list = []

        # Precompute per-sample spike offsets for reindexing
        spike_counts = torch.zeros(B, device=device, dtype=torch.long)
        spike_counts.scatter_add_(0, spike_batch, torch.ones(S, device=device, dtype=torch.long))
        spike_offsets = torch.zeros(B, device=device, dtype=torch.long)
        spike_offsets[1:] = spike_counts[:-1].cumsum(0)

        for b in range(B):
            # Spikes for this sample
            sp_mask = spike_batch == b
            sp_indices = sp_mask.nonzero(as_tuple=True)[0]
            S_b = sp_indices.size(0)

            if S_b == 0:
                continue

            local_cells = spike_gc[sp_mask] % M                       # [S_b]
            local_bins = spike_bins[sp_mask]                           # [S_b]

            # Map from local cell -> latest local spike index
            cell_to_latest = torch.full((M,), -1, device=device, dtype=torch.long)
            local_spike_idx = torch.arange(S_b, device=device)
            cell_to_latest.scatter_(0, local_cells, local_spike_idx)
            cell_has_spike = cell_to_latest >= 0

            offset = spike_offsets[b]
            sample_edges = []

            # 1. Coarsened input edges
            if edge_index.numel() > 0:
                node_mask = node_batch == b
                node_indices = node_mask.nonzero(as_tuple=True)[0]
                if node_indices.numel() > 0:
                    src, dst = edge_index[:, 0], edge_index[:, 1]
                    e_mask = node_mask[src] & node_mask[dst]
                    if e_mask.any():
                        e_src_cell = cell_idx[src[e_mask]]
                        e_dst_cell = cell_idx[dst[e_mask]]
                        keep = (e_src_cell != e_dst_cell) & cell_has_spike[e_src_cell] & cell_has_spike[e_dst_cell]
                        if keep.any():
                            pairs = torch.unique(
                                torch.stack([e_src_cell[keep], e_dst_cell[keep]], dim=1), dim=0
                            )
                            sample_edges.append(torch.stack([
                                cell_to_latest[pairs[:, 0]],
                                cell_to_latest[pairs[:, 1]],
                            ], dim=1))

            # 2. Grid adjacency (causal)
            if self.grid_adj.numel() > 0:
                adj_src = self.grid_adj[:, 0]
                adj_dst = self.grid_adj[:, 1]
                both_active = cell_has_spike[adj_src] & cell_has_spike[adj_dst]
                if both_active.any():
                    ls = cell_to_latest[adj_src[both_active]]
                    ld = cell_to_latest[adj_dst[both_active]]
                    causal = local_bins[ls] <= local_bins[ld]
                    if causal.any():
                        sample_edges.append(torch.stack([ls[causal], ld[causal]], dim=1))

            # 3. Temporal self-edges
            if S_b > 1:
                sort_key = local_cells.long() * (self.num_bins + 1) + local_bins
                sorted_order = sort_key.argsort()
                sorted_cells = local_cells[sorted_order]
                same_cell = sorted_cells[:-1] == sorted_cells[1:]
                if same_cell.any():
                    sample_edges.append(torch.stack([
                        sorted_order[:-1][same_cell],
                        sorted_order[1:][same_cell],
                    ], dim=1))

            # 4. Self-loops
            self_idx = torch.arange(S_b, device=device)
            sample_edges.append(torch.stack([self_idx, self_idx], dim=1))

            # Merge, dedup, offset to global spike space
            all_e = torch.cat(sample_edges, dim=0)
            all_e = torch.unique(all_e, dim=0)
            edges_list.append(all_e + offset)

        if len(edges_list) == 0:
            return torch.zeros(0, 2, device=device, dtype=torch.long)

        return torch.cat(edges_list, dim=0)

    # ------------------------------------------------------------------
    # Inference (event-by-event, single sample)
    # ------------------------------------------------------------------

    def _coarsen_edges_inference(self, edge_index, cell_idx, fired_mask):
        """Coarsen input edges through grid routing for inference."""
        if edge_index.numel() == 0:
            return torch.zeros(0, 2, device=fired_mask.device, dtype=torch.long)

        src_cell = cell_idx[edge_index[:, 0]]
        dst_cell = cell_idx[edge_index[:, 1]]
        keep = (src_cell != dst_cell) & fired_mask[src_cell] & fired_mask[dst_cell]

        if not keep.any():
            return torch.zeros(0, 2, device=fired_mask.device, dtype=torch.long)

        coarse = torch.stack([src_cell[keep], dst_cell[keep]], dim=1)
        return torch.unique(coarse, dim=0)

    def _history_edges(self, current_fired):
        """Connect newly-fired cells to recently-fired grid neighbours."""
        if self.grid_adj.numel() == 0:
            return torch.zeros(0, 2, device=current_fired.device, dtype=torch.long)

        recent = self._fired_history.any(dim=0)
        adj_src = self.grid_adj[:, 0]
        adj_dst = self.grid_adj[:, 1]
        keep = current_fired[adj_src] & recent[adj_dst]

        if not keep.any():
            return torch.zeros(0, 2, device=current_fired.device, dtype=torch.long)

        return torch.stack([adj_src[keep], adj_dst[keep]], dim=1)

    def init_state(self, device, dtype=torch.float32):
        """Initialise persistent state for event-by-event inference."""
        C = self.out_channels
        self._membrane = torch.zeros(self.num_neurons, C, device=device, dtype=dtype)
        self._current_time = 0.0
        self._fired_history = torch.zeros(
            self.fired_history_len, self.num_neurons, device=device, dtype=torch.bool
        )
        self._history_write = 0
        self._pending_feat = []
        self._pending_pos = []
        self._pending_edges = []
        self._pending_cell_ids = []

    @torch.no_grad()
    def step(self, x_event, pos_event, edge_index, timestamp):
        """
        Process a micro-batch of events (inference, single sample).

        Args:
            x_event:    [B, C_in] or [C_in]
            pos_event:  [B, 2+] or [2+]
            edge_index: [E, 2]
            timestamp:  float
        Returns:
            n_fired: int
        """
        if x_event.dim() == 1:
            x_event = x_event.unsqueeze(0)
            pos_event = pos_event.unsqueeze(0)

        device = x_event.device

        # Decay
        dt = timestamp - self._current_time
        self._current_time = timestamp
        if dt > 0:
            decay = torch.exp(-dt / self.tau.abs().clamp(min=1e-3)).squeeze(-1)
            self._membrane *= decay.unsqueeze(-1)

        # Route & integrate
        cell_idx = self._route_to_grid(pos_event)
        messages = self.lin_integrate(x_event)
        self._membrane.scatter_add_(
            0, cell_idx.unsqueeze(-1).expand_as(messages), messages
        )

        # BN (eval mode) + threshold
        membrane_normed = self.bn(self._membrane)
        potential = membrane_normed.norm(dim=-1)
        fired_mask = potential > self.threshold.abs()

        n_fired = int(fired_mask.sum().item())
        if n_fired == 0:
            return 0

        fired_idx = fired_mask.nonzero(as_tuple=True)[0]

        # Emit
        spike_feat = F.relu(self.lin_emit(membrane_normed[fired_idx]))
        spike_xy = self.neuron_pos[fired_idx]
        fire_time = torch.full(
            (spike_xy.size(0), 1), timestamp, device=device, dtype=spike_xy.dtype
        )
        spike_pos = torch.cat([spike_xy, fire_time], dim=1)

        # Edges
        step_edges = self._coarsen_edges_inference(edge_index, cell_idx, fired_mask)
        history_edges = self._history_edges(fired_mask)
        parts = [e for e in (step_edges, history_edges) if e.numel() > 0]
        combined = (
            torch.unique(torch.cat(parts, dim=0), dim=0) if parts
            else torch.zeros(0, 2, device=device, dtype=torch.long)
        )

        # Store pending
        self._pending_feat.append(spike_feat)
        self._pending_pos.append(spike_pos)
        self._pending_edges.append(combined)
        self._pending_cell_ids.append(fired_idx)

        # Update history & reset
        self._fired_history[self._history_write] = fired_mask
        self._history_write = (self._history_write + 1) % self.fired_history_len
        self._membrane[fired_mask] = 0.0

        return n_fired

    def flush(self):
        """
        Collect all pending spikes into a single graph.

        Returns:
            x:          [S, C]
            pos:        [S, 3]   (x, y, t_fire)
            edge_index: [E, 2]
        """
        device = self._membrane.device
        C = self.out_channels

        if len(self._pending_feat) == 0:
            return (
                torch.zeros(0, C, device=device),
                torch.zeros(0, 3, device=device),
                torch.zeros(0, 2, device=device, dtype=torch.long),
            )

        x_out = torch.cat(self._pending_feat, dim=0)
        pos_out = torch.cat(self._pending_pos, dim=0)
        all_cell_ids = torch.cat(self._pending_cell_ids, dim=0)
        S = x_out.size(0)

        # Cell -> local spike index
        cell_to_local = torch.full(
            (self.num_neurons,), -1, device=device, dtype=torch.long
        )
        cell_to_local.scatter_(0, all_cell_ids, torch.arange(S, device=device))

        if len(self._pending_edges) > 0:
            all_e = torch.unique(torch.cat(self._pending_edges, dim=0), dim=0)
            local_src = cell_to_local[all_e[:, 0]]
            local_dst = cell_to_local[all_e[:, 1]]
            valid = (local_src >= 0) & (local_dst >= 0)
            edge_out = torch.stack([local_src[valid], local_dst[valid]], dim=1)
        else:
            edge_out = torch.zeros(0, 2, device=device, dtype=torch.long)

        # Self-loops
        self_idx = torch.arange(S, device=device)
        self_loops = torch.stack([self_idx, self_idx], dim=1)
        edge_out = torch.unique(torch.cat([edge_out, self_loops], dim=0), dim=0)

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
