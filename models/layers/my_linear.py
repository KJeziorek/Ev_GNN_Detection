from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.base import QuantizableLayer
from models.quantisation.observer import FakeQuantize, quantize_tensor
from models.quantisation.quant_config import QuantizationConfig


class MyLinear(QuantizableLayer):
    """
    Linear layer with three operating modes: float, calibrate, and quantize.

    Construction creates a plain float layer — no quantization overhead.
    Call calibrate() to start collecting statistics, then quantize() to
    fix the integer-arithmetic parameters.

    Args:
        input_dim:    Input feature size.
        output_dim:   Output feature size.
        bias:         Whether to include a bias term.
        quant_config: Optional quantization config (observer type + bit-width).
                      Can also be supplied or overridden at calibrate() time.

    Example::

        layer = MyLinear(16, 64)                          # float only
        layer.calibrate(percentile_config(num_bits=8))    # switch observer type
        for batch in calib_loader:
            layer(batch)                                  # accumulates stats
        layer.quantize()                                  # fix integer weights
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 4,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self._init_quant(quant_config)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        # Float layer — only thing created at construction time
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        # Observers — created lazily in _setup_observers()
        self.observer_input: Optional[nn.Module] = None
        self.observer_weight: Optional[nn.Module] = None
        self.observer_output: Optional[nn.Module] = None

        # Quantization parameters populated during quantize()
        self.num_bits_obs = 32
        self.register_buffer('m', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_in', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_w', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_out', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_m', torch.tensor(1.0, requires_grad=False))

        # Whether this is the first layer (needs to quantize raw inputs)
        self.first_layer: bool = False

    # ------------------------------------------------------------------
    # QuantizableLayer hooks
    # ------------------------------------------------------------------

    def _setup_observers(self, config: QuantizationConfig) -> None:
        self.observer_input = config.make_observer()
        self.observer_weight = config.make_observer()
        self.observer_output = config.make_observer()

    def _finalize_quantization(
        self,
        observer_input=None,
        observer_output=None,
    ) -> None:
        """Compute fixed-point scales and quantize weights/bias."""
        if observer_input is not None:
            self.observer_input = observer_input
        if observer_output is not None:
            self.observer_output = observer_output

        # Snap scales to 32-bit fixed point
        self.qscale_in.copy_((2 ** self.num_bits_obs * self.observer_input.scale).round())
        self.observer_input.scale = self.qscale_in / (2 ** self.num_bits_obs)

        self.qscale_w.copy_((2 ** self.num_bits_obs * self.observer_weight.scale).round())
        self.observer_weight.scale = self.qscale_w / (2 ** self.num_bits_obs)

        self.qscale_out.copy_((2 ** self.num_bits_obs * self.observer_output.scale).round())
        self.observer_output.scale = self.qscale_out / (2 ** self.num_bits_obs)

        # Combined scale m = (s_w * s_in) / s_out
        qscale_m = (self.observer_weight.scale * self.observer_input.scale) / self.observer_output.scale
        self.qscale_m.copy_((2 ** self.num_bits_obs * qscale_m).round())
        self.m.copy_(self.qscale_m / (2 ** self.num_bits_obs))

        # Quantize weights
        self.linear.weight = nn.Parameter(self.observer_weight.quantize_tensor(self.linear.weight))
        self.linear.weight = nn.Parameter(self.linear.weight - self.observer_weight.zero_point)

        # Quantize bias (32-bit signed, scale = s_w * s_in)
        if self.bias:
            self.linear.bias = nn.Parameter(
                quantize_tensor(
                    self.linear.bias,
                    scale=self.observer_input.scale * self.observer_weight.scale,
                    zero_point=0,
                    num_bits=32,
                    signed=True,
                )
            )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_calib:
            return self.forward_calib(x)
        if self.is_quant:
            return self.forward_quant(x)
        return self.forward_float(x)

    def forward_float(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def forward_calib(self, x: torch.Tensor) -> torch.Tensor:
        if self.first_layer:
            self.observer_input.update(x)
            x = FakeQuantize.apply(x, self.observer_input)

        self.observer_weight.update(self.linear.weight.data)
        if self.bias:
            x = F.linear(x, FakeQuantize.apply(self.linear.weight, self.observer_weight),
                         self.linear.bias)
        else:
            x = F.linear(x, FakeQuantize.apply(self.linear.weight, self.observer_weight))

        self.observer_output.update(x)
        x = FakeQuantize.apply(x, self.observer_output)
        return x

    def forward_quant(self, x: torch.Tensor) -> torch.Tensor:
        num_bits = self._quant_config.num_bits
        if self.first_layer:
            x = self.observer_input.quantize_tensor(x)
        x = x - self.observer_input.zero_point
        x = self.linear(x)
        x = (x * self.m).round() + self.observer_output.zero_point
        x = torch.clamp(x, 0, 2 ** num_bits - 1)
        return x

    # ------------------------------------------------------------------
    # Parameter export
    # ------------------------------------------------------------------

    def get_parameters(self, file_name: str) -> None:
        """Export quantized parameters in C-array and LUT format."""
        with open(file_name, 'w') as f:
            f.write(f"Input scale ({self.num_bits_obs} bit):\n {int(self.qscale_in)}\n")
            f.write(f"Input zero point:\n {int(self.observer_input.zero_point)}\n")
            f.write(f"Weight scale ({self.num_bits_obs} bit):\n {int(self.qscale_w)}\n")
            f.write(f"Weight zero point:\n {int(self.observer_weight.zero_point)}\n")
            f.write(f"Output scale ({self.num_bits_obs} bit):\n {int(self.qscale_out)}\n")
            f.write(f"Output zero point:\n {int(self.observer_output.zero_point)}\n")
            f.write(f"M scale ({self.num_bits_obs} bit):\n {int(self.qscale_m)}\n")

            weight = torch.flip(self.linear.weight, [0]).T
            weight = weight.detach().cpu().numpy().astype(np.int32).tolist()
            f.write(f"Weight ({self._quant_config.num_bits} bit):\n")
            for idx, w in enumerate(weight):
                f.write(f"weights_conv[{idx}] = {str(w).replace('[', '{').replace(']', '}')}" + ";\n")

            if self.bias:
                bias = torch.flip(self.linear.bias, [0])
                bias = bias.detach().cpu().numpy().astype(np.int32).tolist()
                f.write(f"\nBias ({self._quant_config.num_bits} bit):\n"
                        f" {str(bias).replace('[', '{').replace(']', '}')}" + ";\n")

            input_range = list(range(int(self.observer_input.min), int(self.observer_input.max + 1)))
            output_range = (
                self.observer_input.quantize_tensor(
                    torch.tensor(input_range, device=self.linear.weight.device)
                ) - self.observer_input.zero_point
            )
            output_range = output_range.detach().cpu().numpy().astype(np.int32).tolist()
            f.write(f"Input range:\n {input_range}\n")
            f.write(f"Output range:\n {output_range}\n")

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"bias={self.bias}, config={self._quant_config})")
