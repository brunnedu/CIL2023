import torch
import typing
from torch import nn

from src.models import UNet, ABackbone
from src.models.eresunet.blocks import DCMBlock, EResUpBlock


class EResUNet(nn.Module):
    """
    E-Res U-Net is an improved U-Net model for segmentation of muscle images.

    Adapted from `E-Res U-Net: An improved U-Net model for segmentation of muscle images´
    <https://www.sciencedirect.com/science/article/pii/S0957417421010198>`
    """

    def __init__(self, backbone: ABackbone):
        super().__init__()

        self.net = UNet(
            backbone=backbone,
            bottom_ctor=DCMBlock,
            up_block_ctor=lambda ci: EResUpBlock(ci)
        )

    def forward(self, x):
        return self.net(x)
