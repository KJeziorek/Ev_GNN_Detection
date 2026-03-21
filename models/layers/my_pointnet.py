from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.base import QuantizableLayer
from models.quantisation.observer import FakeQuantize, quantize_tensor
from models.quantisation.quant_config import QuantizationConfig


class MyPointNetConv(QuantizableLayer):
    """
    PointNet-style graph convolution with three operating modes:
    float, calibrate, and quantize.

    Construction creates a plain float layer. Call calibrate() to collect
    activation statistics, then quantize() to produce integer-arithmetic
    inference.

    Args:
        input_dim:    Concatenated input size (node features + positional diff).
        output_dim:   Output feature size.
        bias:         Bias for the global_nn aggregation layer.
        quant_config: Optional quantization config (observer type + bit-width).
                      Can be supplied or overridden at calibrate() time.
        first_layer:  If True, raw float inputs are quantized on the fly during
                      quantized inference; otherwise only the positional diff
                      is re-quantized (node features already quantized upstream).

    Example::

        conv = MyPointNetConv(16, 64, first_layer=True)
        conv.calibrate(moving_avg_config(num_bits=8))
        for batch in calib_loader:
            conv(x, pos, edge_index)
        conv.quantize()
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        first_layer: bool = False,
    ):
        super().__init__()
        self._init_quant(quant_config)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.first_layer = first_layer

        # Float layers — only things created at construction time
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.global_nn = nn.Linear(output_dim, output_dim, bias=bias)

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

        # Quantized linear created during _finalize_quantization()
        self.qlinear: Optional[nn.Linear] = None

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
        """Compute fixed-point scales, quantize weights/bias into self.qlinear."""
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

        device = self.linear.weight.device
        with torch.no_grad():
            self.qlinear = nn.Linear(self.input_dim, self.output_dim, bias=True).to(device)
            quantized_weight = self.observer_weight.quantize_tensor(self.linear.weight)
            self.qlinear.weight.copy_(quantized_weight - self.observer_weight.zero_point)
            quantized_bias = quantize_tensor(
                self.linear.bias,
                scale=self.observer_weight.scale * self.observer_input.scale,
                zero_point=0,
                num_bits=32,
                signed=True,
            )
            self.qlinear.bias.copy_(quantized_bias)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_calib:
            return self.forward_calib(x, pos, edge_index)
        if self.is_quant:
            return self.forward_quant(x, pos, edge_index)
        return self.forward_float(x, pos, edge_index)

    def forward_float(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]
        x_j = x[edge_index[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)
        msg = self.linear(msg)
        out = self._scatter_amax(msg, edge_index)
        return self.global_nn(out)

    def forward_calib(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]
        x_j = x[edge_index[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)

        # Observe and fake-quantize input messages
        self.observer_input.update(msg)
        msg = FakeQuantize.apply(msg, self.observer_input)

        # Observe and fake-quantize weights
        self.observer_weight.update(self.linear.weight.data)
        W_q = FakeQuantize.apply(self.linear.weight, self.observer_weight)
        msg = F.linear(msg, W_q, self.linear.bias)

        # Observe output — include pos diff range for next-layer input quantization
        self.observer_output.update(msg)
        self.observer_output.update(pos_j - pos_i)
        msg = FakeQuantize.apply(msg, self.observer_output)

        return self._scatter_amax(msg, edge_index)

    def forward_quant(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        num_bits = self._quant_config.num_bits
        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]

        if self.first_layer:
            x_j = x[edge_index[:, 1]]
            msg = torch.cat((x_j, pos_j - pos_i), dim=1)
            msg = self.observer_input.quantize_tensor(msg)
        else:
            # Features already quantized upstream; only re-quantize pos diff
            pos_diff = self.observer_input.quantize_tensor(pos_j - pos_i)
            msg = torch.cat((x[edge_index[:, 1]], pos_diff), dim=1)

        msg = msg - self.observer_input.zero_point
        msg = self.qlinear(msg)
        msg = (msg * self.m).round() + self.observer_output.zero_point
        msg = torch.clamp(msg, 0, 2 ** num_bits - 1)

        return self._scatter_amax(msg, edge_index)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scatter_amax(
        self,
        msg: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Max-pool messages into per-node features."""
        unique_nodes, indices = torch.unique(edge_index[:, 0], return_inverse=True)
        expanded = indices.unsqueeze(1).expand(-1, self.output_dim)
        out = torch.zeros((unique_nodes.size(0), self.output_dim),
                          dtype=msg.dtype, device=msg.device)
        out = out.scatter_reduce(0, expanded, msg, reduce="amax", include_self=False)
        return out

    # ------------------------------------------------------------------
    # Parameter export
    # ------------------------------------------------------------------

    def get_parameters(self, file_name: str) -> None:
        """Export quantized parameters in C-array and binary .mem format."""
        num_bits = self._quant_config.num_bits
        bias = torch.flip(self.qlinear.bias, [0]).detach().cpu().numpy().astype(np.int32).tolist()
        weight = torch.flip(self.qlinear.weight, [1]).detach().cpu().numpy().astype(np.int32).tolist()

        with open(file_name, 'w') as f:
            f.write(f"Input scale ({self.num_bits_obs} bit):\n {int(self.qscale_in)}\n")
            f.write(f"Input zero point:\n {int(self.observer_input.zero_point)}\n")
            f.write(f"Weight scale ({self.num_bits_obs} bit):\n {int(self.qscale_w)}\n")
            f.write(f"Weight zero point:\n {int(self.observer_weight.zero_point)}\n")
            f.write(f"Output scale ({self.num_bits_obs} bit):\n {int(self.qscale_out)}\n")
            f.write(f"Output zero point:\n {int(self.observer_output.zero_point)}\n")
            f.write(f"M scale ({self.num_bits_obs} bit):\n {int(self.qscale_m)}\n")

            f.write(f"Weight ({num_bits} bit):\n")
            for idx, w in enumerate(weight):
                f.write(f"weights_conv[{idx}] = {str(w).replace('[', '{').replace(']', '}')}" + ";\n")

            f.write(f"\nBias ({num_bits} bit):\n")
            f.write(f"bias_conv = {str(bias).replace('[', '{').replace(']', '}')}" + ";\n")

            input_range = list(range(int(self.observer_input.min), int(self.observer_input.max + 1)))
            output_range = (
                self.observer_input.quantize_tensor(
                    torch.tensor(input_range, device=self.linear.weight.device)
                ) - self.observer_input.zero_point
            )
            output_range = output_range.detach().cpu().numpy().astype(np.int32).tolist()
            f.write(f"Input range:\n {input_range}\n")
            f.write(f"Output range:\n {output_range}\n")

        with open(file_name.replace('.txt', '.mem'), 'w') as f:
            zp = self.observer_weight.zero_point.to(torch.int32).item()
            for idx, we in enumerate(weight):
                bin_vec = [np.binary_repr(w + zp, width=9)[1:] for w in we]
                bin_vec.append(np.binary_repr(bias[len(bias) - idx - 1], width=32))
                hex_val = hex(int(''.join(bin_vec), 2))
                f.write(f"{hex_val[2:]}\n")

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"first_layer={self.first_layer}, config={self._quant_config})")
