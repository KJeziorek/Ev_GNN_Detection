import torch
import torch.nn as nn

from utils.data import GraphData

class GraphPooling(nn.Module):
    def __init__(self, pool_size: list,
                 to_original_dim: bool = False,
                 average_positions: bool = False,
                 pool_temporal: bool = False):
        super().__init__()
        self.pool_size = torch.tensor(pool_size, dtype=torch.long)

        self.to_original_dim = to_original_dim
        self.average_positions = average_positions
        self.pool_temporal = pool_temporal

    def forward(self,
                data: GraphData
               ) -> GraphData: 
        
        x = data.x                          # [N, F]
        pos = data.pos                      # [N, 3]
        edge_index = data.edge_index        # [E, 2]
        batch = data.batch                  # [N]

        # 1) quantize spatial (and optionally temporal) coords
        qpos = torch.div(pos, self.pool_size.to(x.device), rounding_mode='floor').long()

        if self.pool_temporal:
            qpos = qpos[:, :2]    # only spatial
        
        # 2) prepend batch id so different graphs don't mix
        key = torch.cat([batch.unsqueeze(1), qpos], dim=1)
        
        # 3) unique & inverse
        unique_keys, inv = torch.unique(key, dim=0, return_inverse=True)
        # unique_keys is [M, 1 + 2 or 3] → [:,0] is batch, rest is quantized pos
        new_batch = unique_keys[:, 0]
        uniq_qpos = unique_keys[:, 1:]
        
        # 4) restore to original scale if requested
        if self.to_original_dim:
            uniq_qpos = uniq_qpos * self.pool_size.to(x.device)
        
        # 5) optionally average the exact positions
        if self.average_positions:
            summed = torch.zeros((uniq_qpos.size(0), 3),
                                 dtype=pos.dtype, device=pos.device)
            uniq_qpos = summed.scatter_reduce(0,
                                               inv.unsqueeze(1).expand(-1, 3),
                                               pos,
                                               reduce='mean',
                                               include_self=False)
        
        # 6) pool your node features (amax here, sum or mean easily swapped)
        pooled_x = torch.zeros((uniq_qpos.size(0), x.size(1)),
                               dtype=x.dtype, device=x.device)
        pooled_x = pooled_x.scatter_reduce(0,
                                           inv.unsqueeze(1).expand(-1, x.size(1)),
                                           x,
                                           reduce='amax',
                                           include_self=False)
        
        # 7) remap edges using the inverse index
        #    inv has length N, so inv[old_node_idx] → new_node_idx
        ei = inv[edge_index]
        mask = ei[:, 0] != ei[:, 1]
        ei = ei[mask]
        ei = torch.unique(ei, dim=0)
        
        # 8) re-add self loops
        loop = torch.stack([torch.arange(uniq_qpos.size(0), device=ei.device),
                            torch.arange(uniq_qpos.size(0), device=ei.device)], dim=1)
        ei = torch.cat([ei, loop], dim=0)
        
        data.x = pooled_x
        data.pos = uniq_qpos
        data.edge_index = ei
        data.batch = new_batch

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(pool_size={self.pool_size})"