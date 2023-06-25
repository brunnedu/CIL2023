import torch.nn as nn
import typing

from src.models import ABackbone
from src.models.dlinknet.blocks import DLinkDilateBlock


class DLinkNet(nn.Module):
    """
    D-LinkNet adapted from https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge/tree/master

    For original LinkNet see https://arxiv.org/pdf/1707.03718.pdf
    """

    def __init__(self, backbone: ABackbone, up_block_ctor: typing.Callable[[int, int], nn.Module]):
        super(DLinkNet, self).__init__()

        self.backbone = backbone
        channels = self.backbone.get_channels()
        self.dblock = DLinkDilateBlock(channels[-1])

        ups = []
        for i in range(len(channels)):
            in_channels = channels[i]
            out_channels = channels[i-1] if i > 0 else channels[i]
            layer = up_block_ctor(in_channels, out_channels)
            ups.append(layer)

        self.ups = nn.ModuleList(ups)

        # final
        self.final = nn.Sequential(
            nn.LazyConvTranspose2d(out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        outs, x = self.backbone(x)

        b = self.dblock(outs[-1])  # dilation
        for s, up in zip(reversed(outs[:-1]), reversed(self.ups)):
            b = up(b, s)

        return self.final(b)



