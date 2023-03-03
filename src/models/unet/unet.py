
import torch.nn as nn

import typing as t

from models.unet.backbones import ABackbone


class UNet(nn.Module):
    def __init__(self, backbone : ABackbone, up_block : t.Callable):
        super().__init__()

        # down nodes / encoder
        self.backbone = backbone
        
        # layers
        channels = self.backbone.get_channels()
        ups = []
        for ci in channels[:-1]:
            layer = up_block(ci)
            ups.append(layer)

        self.ups = nn.ModuleList(ups)

        self.final = nn.Sequential(
            nn.LazyConv2d(1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        outs, x = self.backbone(x)

        b = outs[-1]
        for s,up in zip(reversed(outs[:-1]),self.ups):
            b = up(b, [s])

        return self.final(b)