import torch
from torch.nn import Module


class MyGraphPooling(Module):
    def __init__(self, pool_size: list):
        super(MyGraphPooling, self).__init__()
        self.pool_size = torch.tensor(pool_size, dtype=torch.long)

        self.pool_temporal = False
        self.only_pos = False
        self.self_loop = True
        self.average_positions = False
        self.to_original_dim = False

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

        # 1) quantize spatial (and optionally temporal) coords
        qpos = torch.div(pos, self.pool_size.to(x.device), rounding_mode='floor').long()
        if self.pool_temporal:
            qpos = qpos[:, :2]

        # 2) prepend batch id so different graphs don't mix
        key = torch.cat([batch.unsqueeze(1), qpos], dim=1)

        # 3) unique & inverse
        unique_keys, inv = torch.unique(key, dim=0, return_inverse=True)
        new_batch = unique_keys[:, 0]
        uniq_qpos = unique_keys[:, 1:]

        # 4) restore to original scale if requested
        if self.to_original_dim:
            uniq_qpos = uniq_qpos * self.pool_size

        # 5) optionally average the exact positions
        if self.average_positions:
            summed = torch.zeros((uniq_qpos.size(0), 3),
                                 dtype=pos.dtype, device=pos.device)
            uniq_qpos = summed.scatter_reduce(0,
                                               inv.unsqueeze(1).expand(-1, 3),
                                               pos,
                                               reduce='mean',
                                               include_self=False)

        # 6) pool node features (amax)
        pooled_x = torch.zeros((uniq_qpos.size(0), x.size(1)),
                               dtype=x.dtype, device=x.device)
        pooled_x = pooled_x.scatter_reduce(0,
                                           inv.unsqueeze(1).expand(-1, x.size(1)),
                                           x,
                                           reduce='amax',
                                           include_self=False)

        data.x = pooled_x
        data.pos = uniq_qpos
        data.batch = new_batch

        if self.only_pos:
            data.edge_index = torch.zeros(0, 2, device=x.device, dtype=torch.long)
            return data

        # 7) remap edges using the inverse index
        ei = inv[edge_index]
        mask = ei[:, 0] != ei[:, 1]
        ei = ei[mask]
        ei = torch.unique(ei, dim=0)

        # 8) re-add self loops
        if self.self_loop:
            loop = torch.stack([torch.arange(uniq_qpos.size(0), device=ei.device),
                                torch.arange(uniq_qpos.size(0), device=ei.device)], dim=1)
            ei = torch.cat([ei, loop], dim=0)

        data.edge_index = ei
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(pool_size={self.pool_size})"
