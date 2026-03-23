from torch.nn import BatchNorm1d
from utils.data import GraphData


class BatchNorm(BatchNorm1d):
    def forward(self, data: GraphData):
        data.x = super().forward(data.x)
        return data