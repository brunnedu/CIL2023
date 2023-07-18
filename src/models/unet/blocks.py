import typing as t

import torch
import torch.nn as nn

from src.models.attention_layers import LazyAttentionGate2D


class DownBlock(nn.Module):
    """
        Node that takes one input, performs:
            b = block(input)
            p = pool(b)
        and returns both b and p

        pool (optional): if no pool is provided, then the identity function is used (=> b == p)

        Usually the input will be from one level above,
        b will be passed on to the same level
        and p will be passed on to a lower level.
    """

    def __init__(self, block: nn.Module, pool: nn.Module = None):
        super().__init__()

        self.block = block
        self.pool = pool

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        b = self.block(x)
        p = self.pool(b) if self.pool else b

        return b, p


class UpBlock(nn.Module):
    """
        Node that takes two inputs
         - b : Tensor (will be upscaled first)
         - s : List[Tensor]
        that it concatenates and applies convolutions on.

        nr_channels: The number of channels on this layer
        up_mode:
            - upconv: use ConvTranspose2d to scale up image
            - upsample: use bilinear interpolation to scale up image

        Usually b is from one level below and
        - UNet++: s are all outputs from the same level
        - UNet: s is previous output wrapped in a list
    """

    def __init__(self, nr_channels: int, up_mode: str = 'upconv', activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()

        if up_mode == 'upconv':
            self.up = nn.LazyConvTranspose2d(nr_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.layer = nn.Sequential(
            nn.LazyConv2d(nr_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nr_channels),
            activation,

            nn.Conv2d(nr_channels, nr_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nr_channels),
            activation
        )

    def forward(self, b, s):
        x = torch.cat([self.up(b), *s], dim=1)
        return self.layer(x)


class AttentionGateUpBlock(nn.Module):
    """
        Like UpBlock but will apply an attention gate on the skip connection

        Node that takes two inputs
         - b : Tensor (will be upscaled first)
         - s : List[Tensor]
        that it concatenates and applies convolutions on.

        nr_channels: The number of channels on this layer
        up_mode:
            - upconv: use ConvTranspose2d to scale up image
            - upsample: use bilinear interpolation to scale up image
        ag_... see documentation of LazyAttentionGate2D

        Usually b is from one level below and
        - UNet++: s are all outputs from the same level
        - UNet: s is previous output wrapped in a list
    """

    def __init__(self, nr_channels: int, up_mode: str = 'upconv', ag_batch_norm: bool = False,
                 ag_bias_wx: bool = False):
        super().__init__()

        if up_mode == 'upconv':
            self.up = nn.LazyConvTranspose2d(nr_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.ag = LazyAttentionGate2D(nr_channels, batch_norm=ag_batch_norm, bias_wx=ag_bias_wx)

        self.layer = nn.Sequential(
            nn.LazyConv2d(nr_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nr_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(nr_channels, nr_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nr_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, b: torch.Tensor, s: t.List[torch.Tensor]) -> torch.Tensor:
        s = torch.cat(s, dim=1)
        b = self.up(b)

        gated = self.ag(b, s)

        x = torch.cat([gated, b], dim=1)
        return self.layer(x)
