import torch.nn as nn
import torchvision

from abc import ABC, abstractmethod
import typing as t

from .blocks import DownBlock

class ABackbone(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def get_channels(self) -> t.List[int]:
        '''
            Returns the number of channels for each level on the backbone
        '''
        pass


class Resnet18Backbone(ABackbone):
    def __init__(self):
        super().__init__()
        
        resnet = torchvision.models.resnet.resnet18(pretrained=True)
        children = list(resnet.children())

        modules = []
        
        modules.append(DownBlock( # first 4 resnet layers are a bit special
            nn.Sequential(*children[:3]),
            children[3]
        ))

        for block in children:
            if isinstance(block, nn.Sequential): # all blocks are of type sequential
                modules.append(DownBlock(block))

        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        outputs = [] # all outputs between bottlenecks

        for layer in self.layers:
            o,x = layer(x)
            outputs.append(o)

        return outputs, x

    def get_channels(self):
        return [64, 64, 128, 256, 512] # see Resnet18 architecture