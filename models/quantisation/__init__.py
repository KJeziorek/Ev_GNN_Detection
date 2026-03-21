from .observer import (
    BaseObserver,
    MinMaxObserver,
    MovingAverageObserver,
    PercentileObserver,
    Observer,          # backward-compatible alias for MinMaxObserver
    FakeQuantize,
    quantize_tensor,
    dequantize_tensor,
)
from .quant_config import (
    ObserverType,
    QuantizationConfig,
    minmax_config,
    moving_avg_config,
    percentile_config,
)

__all__ = [
    "BaseObserver",
    "MinMaxObserver",
    "MovingAverageObserver",
    "PercentileObserver",
    "Observer",
    "FakeQuantize",
    "quantize_tensor",
    "dequantize_tensor",
    "ObserverType",
    "QuantizationConfig",
    "minmax_config",
    "moving_avg_config",
    "percentile_config",
]
