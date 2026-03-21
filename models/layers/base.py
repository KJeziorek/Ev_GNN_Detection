from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.quantisation.quant_config import QuantizationConfig


class QuantizableLayer(nn.Module):
    """
    Base class for layers that support float → calibrate → quantize transitions.

    Lifecycle
    ---------
    1. Construct the layer — pure float, zero quantization overhead.
    2. Call layer.calibrate()  (optionally passing a QuantizationConfig).
       Run your calibration data through the model; observers accumulate stats.
    3. Call layer.quantize()   — scales are fixed, weights are quantized,
       all subsequent forward passes use integer arithmetic.
    4. Optionally call layer.float_mode() to return to float inference
       (observers/scales are preserved but not used).

    Subclasses must implement
    -------------------------
    _setup_observers(config)              — create observers via config.make_observer()
    _finalize_quantization(**kwargs)      — compute scales, quantize weights/biases
    forward_float(...)                    — standard float forward
    forward_calib(...)                    — calibration forward (updates observers)
    forward_quant(...)                    — integer-arithmetic forward
    """

    _STATE_FLOAT: int = 0
    _STATE_CALIB: int = 1
    _STATE_QUANT: int = 2

    def _init_quant(self, quant_config: Optional[QuantizationConfig] = None) -> None:
        """
        Initialise quantization state. Call once at the end of __init__
        *after* super().__init__() has been called.
        """
        self._quant_config: Optional[QuantizationConfig] = quant_config
        self.register_buffer(
            '_quant_state',
            torch.tensor(self._STATE_FLOAT, dtype=torch.long, requires_grad=False),
        )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def is_float(self) -> bool:
        return self._quant_state.item() == self._STATE_FLOAT

    @property
    def is_calib(self) -> bool:
        return self._quant_state.item() == self._STATE_CALIB

    @property
    def is_quant(self) -> bool:
        return self._quant_state.item() == self._STATE_QUANT

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def float_mode(self) -> QuantizableLayer:
        """Return to float inference (observers and scales are kept but ignored)."""
        self._quant_state.fill_(self._STATE_FLOAT)
        return self

    def calibrate(self, config: Optional[QuantizationConfig] = None) -> QuantizableLayer:
        """
        Switch to calibration mode.

        Args:
            config: Override the quantization config supplied at construction.
                    If neither this nor the constructor config is set, defaults
                    to 8-bit MinMax.
        """
        if config is not None:
            self._quant_config = config
        if self._quant_config is None:
            self._quant_config = QuantizationConfig()  # 8-bit MinMax default
        self._setup_observers(self._quant_config)
        self._quant_state.fill_(self._STATE_CALIB)
        return self

    def quantize(self, **kwargs) -> QuantizableLayer:
        """
        Finalise quantization. Computes scales from observer statistics,
        quantizes weights and biases, and switches to integer-arithmetic mode.

        Any keyword arguments are forwarded to _finalize_quantization().
        Common kwargs (defined by subclasses):
            observer_input  — override the input observer
            observer_output — override the output observer
        """
        self._finalize_quantization(**kwargs)
        self._quant_state.fill_(self._STATE_QUANT)
        return self

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _setup_observers(self, config: QuantizationConfig) -> None:
        """Create/reset observers from config. Called by calibrate()."""
        raise NotImplementedError(f"{type(self).__name__} must implement _setup_observers()")

    def _finalize_quantization(self, **kwargs) -> None:
        """Compute scales and quantize weights. Called by quantize()."""
        raise NotImplementedError(f"{type(self).__name__} must implement _finalize_quantization()")

    def forward_float(self, *args, **kwargs):
        raise NotImplementedError

    def forward_calib(self, *args, **kwargs):
        raise NotImplementedError

    def forward_quant(self, *args, **kwargs):
        raise NotImplementedError
