"""Explicit multi-scale kernel decomposition module."""

import torch
from torch import nn


class EMKDModule(nn.Module):
    """Fuse local and large-receptive-field convolutional features.

    The implementation follows the module used by EMKD-DSCNet: a 1x1 input
    projection is followed by parallel 1x1, 3x3, and dilated 7x7 branches.
    Their outputs are summed and fused by a 1x1 convolution, batch
    normalization, and ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1x1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        # Keep the original attribute name for checkpoint compatibility.
        self.convnxn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=9,
            dilation=3,
            bias=False,
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        local_1x1 = self.conv1x1(x)
        local_3x3 = self.conv3x3(x)
        large_7x7 = self.convnxn(x)
        return self.conv_out(local_1x1 + local_3x3 + large_7x7)


__all__ = ["EMKDModule"]
