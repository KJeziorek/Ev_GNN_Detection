import torch
import torch.nn as nn
from torch.autograd import Function


def quantize_tensor(tensor,
                    scale,
                    zero_point,
                    num_bits=8,
                    signed=False):
    """Quantize tensor: float -> int."""
    if signed:
        qmin = -2 ** (num_bits - 1)
        qmax = 2 ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** num_bits - 1

    q_tensor = torch.round(tensor / (scale + 1e-8) + zero_point)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    return q_tensor


def dequantize_tensor(tensor, scale, zero_point):
    """Dequantize tensor: int -> float."""
    return (tensor - zero_point) * scale


class BaseObserver(nn.Module):
    """
    Common interface for all quantization observers.

    Subclasses must implement update() to track activation statistics.
    All observers share the same quantize/dequantize interface so layers
    are agnostic to which observer type is used.
    """

    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits
        self.register_buffer('scale', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('zero_point', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('min', torch.tensor(float('inf'), requires_grad=False))
        self.register_buffer('max', torch.tensor(float('-inf'), requires_grad=False))

    def update(self, tensor: torch.Tensor):
        """Update running statistics with a new batch of data."""
        raise NotImplementedError

    def calcScaleZeroPoint(self):
        """Asymmetric affine quantization: maps [min, max] -> [0, 2^n - 1]."""
        qmin = 0.
        qmax = 2 ** self.num_bits - 1
        scale = (self.max - self.min) / (qmax - qmin)
        zero_point = qmin - self.min / scale
        zero_point = torch.round(zero_point)
        zero_point = torch.clamp(zero_point, qmin, qmax)
        return scale, zero_point

    def quantize_tensor(self, tensor: torch.Tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, self.num_bits)

    def dequantize_tensor(self, tensor_quant: torch.Tensor):
        return dequantize_tensor(tensor_quant, self.scale, self.zero_point)


class MinMaxObserver(BaseObserver):
    """
    Tracks the global min/max across all calibration batches.

    Conservative: a single outlier expands the range permanently.
    Good baseline when the activation distribution is stable.
    """

    def update(self, tensor: torch.Tensor):
        with torch.no_grad():
            batch_min = tensor.min().item()
            batch_max = tensor.max().item()
            self.min = torch.tensor(min(self.min.item(), batch_min), device=tensor.device)
            self.max = torch.tensor(max(self.max.item(), batch_max), device=tensor.device)
            if self.max > self.min:
                self.scale, self.zero_point = self.calcScaleZeroPoint()


class MovingAverageObserver(BaseObserver):
    """
    Exponential moving average (EMA) of per-batch min/max.

    Less sensitive to outlier batches than MinMax. Use when activation
    statistics vary across calibration batches and you want a smoothed range.

    Args:
        momentum: EMA decay weight for new batches (0 < momentum <= 1).
                  Higher = more weight on recent batches.
    """

    def __init__(self, num_bits: int = 8, momentum: float = 0.1):
        super().__init__(num_bits)
        self.momentum = momentum

    def update(self, tensor: torch.Tensor):
        with torch.no_grad():
            batch_min = tensor.min().item()
            batch_max = tensor.max().item()

            if self.min.item() == float('inf'):
                # First batch: initialise directly
                self.min = torch.tensor(batch_min, device=tensor.device)
                self.max = torch.tensor(batch_max, device=tensor.device)
            else:
                self.min = torch.tensor(
                    (1 - self.momentum) * self.min.item() + self.momentum * batch_min,
                    device=tensor.device,
                )
                self.max = torch.tensor(
                    (1 - self.momentum) * self.max.item() + self.momentum * batch_max,
                    device=tensor.device,
                )

            if self.max > self.min:
                self.scale, self.zero_point = self.calcScaleZeroPoint()


class PercentileObserver(BaseObserver):
    """
    Clips outliers using per-batch percentile-based range.

    Tracks the min of per-batch lower percentiles and max of per-batch upper
    percentiles across calibration. Reduces the impact of rare outliers at the
    cost of slightly more quantization error for inliers.

    Args:
        percentile: Symmetric clip percentile in (0, 100].
                    E.g. 99.9 clips the bottom 0.05% and top 0.05% per batch.
    """

    def __init__(self, num_bits: int = 8, percentile: float = 99.9):
        super().__init__(num_bits)
        self.percentile = percentile

    def update(self, tensor: torch.Tensor):
        with torch.no_grad():
            flat = tensor.detach().float().flatten()
            p_low = (100.0 - self.percentile) / 2.0
            p_high = 100.0 - p_low
            batch_min = torch.quantile(flat, p_low / 100.0).item()
            batch_max = torch.quantile(flat, p_high / 100.0).item()

            # Track min/max of the per-batch clipped ranges
            self.min = torch.tensor(min(self.min.item(), batch_min), device=tensor.device)
            self.max = torch.tensor(max(self.max.item(), batch_max), device=tensor.device)

            if self.max > self.min:
                self.scale, self.zero_point = self.calcScaleZeroPoint()


# Backward-compatible alias
Observer = MinMaxObserver


class FakeQuantize(Function):
    """
    Straight-through estimator for simulating quantization during calibration.
    Forward: quantize then dequantize (introduces rounding error).
    Backward: pass gradient through unchanged.
    """

    @staticmethod
    def forward(ctx, x, observer):
        x = observer.quantize_tensor(x)
        x = observer.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
