import torch
import torch.nn as nn

import typing as t

from .backbones import ABackbone

class UNetPPLayer(nn.Module):
    '''
        Layer (diagonal from top left to bottom right) of the UNet++

        Parameters:
            - channels: the number of output channels on each level
            - up_block_ctor: function that is fed the amount of channels and returns an up node of the UNet++
    '''
    def __init__(self, channels : t.List[int], 
                 up_block_ctor : t.Callable[[int], nn.Module]):
        super().__init__()

        nodes = [up_block_ctor(ci) for ci in channels]
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
    '''
        UNet++ architecture
        Parameters:
            - backbone: the "leftmost" nodes of the UNet, defines the depth and the amount of channels per level
            - up_block_ctor: function that is fed the amount of channels and returns an up node of the UNet++
            - final: the final layers of the UNet++
    '''
    def __init__(self, backbone : ABackbone, 
                 up_block_ctor : t.Callable[[int], nn.Module], 
                 final : nn.Module = None):
        super().__init__()

        # down nodes / encoder
        self.backbone = backbone
        
        # layers
        channels = self.backbone.get_channels()
        layers = []
        for i in reversed(range(1, len(channels))):
            layer = UNetPPLayer(channels[:i], up_block_ctor)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if final:
            self.final = final
        else:
            self.final = nn.Sequential(
                nn.LazyConvTranspose2d(1, kernel_size=2, stride=2),
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