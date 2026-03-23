from torch.nn import Linear
from utils.data import GraphData


class LinearX(Linear):
    def forward(self, data: GraphData):
        data.x = super().forward(data.x)
        return data