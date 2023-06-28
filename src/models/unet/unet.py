import torch
import torch.nn as nn
import torch.nn.functional as F

import typing as t

from .backbones import ABackbone


class UNet(nn.Module):
    """
        Basic UNet architecture
        Parameters:
            - backbone: the "leftmost" nodes of the UNet, defines the depth and the amount of channels per level
            - up_block_ctor: function that is fed the amount of channels and returns an up node of the UNet
            - final: the final layers of the UNet
            - bottom: constructor of the middle-piece of the bottom-most layer (if left as None => net will be V shaped)
            - multiscale_final: does the final layer receive all outputs of the layers below (bilinearly upsampled) 
                                or just the final (most coarse grained) output?
    """

    def __init__(self, backbone: ABackbone,
                 up_block_ctor: t.Callable[[int], nn.Module],
                 final: t.Optional[nn.Module] = None,
                 bottom_ctor: t.Optional[t.Callable[[int], nn.Module]] = None,
                 multiscale_final: bool = False):
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

        self.bottom = None
        if bottom_ctor:
            self.bottom = bottom_ctor(channels[-1])

        self.multiscale_final = multiscale_final

    def forward(self, x):
        outs, x = self.backbone(x)

        b = outs[-1]

        if self.bottom is not None:
            b = self.bottom(b)

        bs = [b]
        for s, up in zip(reversed(outs[:-1]), reversed(self.ups)):
            b = up(b, [s])
            bs.append(b)

        if self.multiscale_final:
            h, w = b.shape[2], b.shape[3]
            bs = [F.interpolate(b, (h, w), mode='bilinear') for b in bs]
            bs = torch.cat(bs, dim=1)
            out = self.final(bs)
        else:
            out = self.final(b)
        return out
