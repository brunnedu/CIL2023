import torch
import torch.nn as nn

import typing as t

from models.unet.backbones import ABackbone

class UNetPPLayer(nn.Module):
    def __init__(self, channels : t.List[int], up_block : t.Callable):
        super().__init__()

        nodes = [up_block(ci) for ci in channels]
        self.nodes = nn.ModuleList(nodes)

    def forward(self, state): 
        # state is a list of list of previous outputs
        # state[0] = all outputs of layer 0
        # state[0][-1] = the output of layer 0 at the deepest level

        outputs = []
        for i,node in enumerate(self.nodes):
            b = state[-1][i+1] # last layer, one level below
            s = [l[i] for l in state[1:]] # all previous layers, same level

            o = node(b, s)
            outputs.append(o)

        return outputs   

class UNetPP(nn.Module):
    def __init__(self, backbone : ABackbone):
        super().__init__()

        # down nodes / encoder
        self.backbone = backbone
        
        # layers
        channels = self.backbone.get_channels()
        layers = []
        for i in reversed(range(1, len(channels))):
            layer = UNetPPLayer(channels[:i])
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        self.final = nn.Sequential(
            nn.LazyConv2d(1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        state = []
        
        outs, x = self.backbone(x)
        state.append(outs)

        for layer in self.layers:
            outs = layer(state)
            state.append(outs)

        y = torch.cat([l[0] for l in state], dim=1)
        return self.final(y)