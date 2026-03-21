from dataclasses import dataclass, field
from enum import Enum


class ObserverType(Enum):
    MINMAX = "minmax"
    MOVING_AVERAGE = "moving_average"
    PERCENTILE = "percentile"


@dataclass
class QuantizationConfig:
    """
    Configuration that fully describes a quantization scheme.

    Pass this to layer.calibrate(config) to select the observer type and
    bit-width. The layer creates its observers from this config, so the
    layer itself stays observer-agnostic.

    Args:
        num_bits:      Bit-width for activations and weights (default 8).
        observer_type: Which statistical method to use (default MinMax).
        momentum:      EMA decay for MovingAverageObserver (ignored otherwise).
        percentile:    Clip percentile for PercentileObserver (ignored otherwise).
    """
    num_bits: int = 8
    observer_type: ObserverType = ObserverType.MINMAX
    momentum: float = 0.1
    percentile: float = 99.9

    def make_observer(self):
        """Instantiate a fresh observer configured by this spec."""
        from models.quantisation.observer import (
            MinMaxObserver, MovingAverageObserver, PercentileObserver,
        )
        if self.observer_type == ObserverType.MINMAX:
            return MinMaxObserver(num_bits=self.num_bits)
        if self.observer_type == ObserverType.MOVING_AVERAGE:
            return MovingAverageObserver(num_bits=self.num_bits, momentum=self.momentum)
        if self.observer_type == ObserverType.PERCENTILE:
            return PercentileObserver(num_bits=self.num_bits, percentile=self.percentile)
        raise ValueError(f"Unknown observer type: {self.observer_type}")


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def minmax_config(num_bits: int = 8) -> QuantizationConfig:
    """Standard min-max observer — safe default for most layers."""
    return QuantizationConfig(num_bits=num_bits, observer_type=ObserverType.MINMAX)


def moving_avg_config(num_bits: int = 8, momentum: float = 0.1) -> QuantizationConfig:
    """EMA observer — useful when activations vary across calibration batches."""
    return QuantizationConfig(num_bits=num_bits, observer_type=ObserverType.MOVING_AVERAGE,
                              momentum=momentum)


def percentile_config(num_bits: int = 8, percentile: float = 99.9) -> QuantizationConfig:
    """Percentile-clipping observer — reduces outlier sensitivity."""
    return QuantizationConfig(num_bits=num_bits, observer_type=ObserverType.PERCENTILE,
                              percentile=percentile)
