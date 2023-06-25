import torch.nn as nn
import typing

from src.models import ABackbone
from src.models.dinknet.blocks import DinkDilateBlock


class DinkNet(nn.Module):

    def __init__(self, backbone: ABackbone, up_block_ctor: typing.Callable[[int, int], nn.Module]):
        super(DinkNet, self).__init__()

        self.backbone = backbone
        channels = self.backbone.get_channels()

        # dilation
        self.dblock = DinkDilateBlock(channels[-1])

        # up
        ups = []
        for i in range(len(channels)):
            in_channels = channels[i]
            if i > 0:
                out_channels = channels[i-1]
            else:
                out_channels = channels[i]

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



