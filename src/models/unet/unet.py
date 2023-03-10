
import torch.nn as nn

import typing as t

from models.unet.backbones import ABackbone


class UNet(nn.Module):
    '''
        Basic UNet architecture
        Parameters:
            - backbone: the "leftmost" nodes of the UNet, defines the depth and the amount of channels per level
            - up_block_ctor: function that is fed the amount of channels and returns an up node of the UNet
            - final: the final layers of the UNet
    '''
    def __init__(self, backbone : ABackbone, 
                 up_block_ctor : t.Callable[[int], nn.Module], 
                 final : nn.Module = None):
        super().__init__()

        # down nodes / encoder
        self.backbone = backbone
        
        # layers
        channels = self.backbone.get_channels()
        ups = []
        for ci in channels[:-1]:
            layer = up_block_ctor(ci)
            ups.append(layer)

        self.ups = nn.ModuleList(ups)

        if final:
            self.final = final
        else:
            self.final = nn.Sequential(
                nn.LazyConvTranspose2d(1, kernel_size=2, stride=2),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

    def forward(self, x):
        outs, x = self.backbone(x)

        b = outs[-1]
        for s,up in zip(reversed(outs[:-1]),reversed(self.ups)):
            b = up(b, [s])

        return self.final(b)