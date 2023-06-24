import torch.nn as nn
import torchvision

from abc import ABC, abstractmethod
import typing as t

from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

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

class GenericBackbone(ABackbone):
    ''' 
        A generic backbone that replicates a module for each layer and adjusts the number of channels
        
        Parameters:
            - channels: list of integers that represents the output channel on each level, 
                        e.g. [4, 16, 32, 32, 64] (=> network with depth 5)
            - down_block_ctor: function that constructs a down block when given the amount of output channels
                        e.g. lambda ci: DownBlock(LazyConvBlock(ci), MaxPool)
    '''
    def __init__(self, channels : t.List[int], 
                 down_block_ctor : t.Callable[[int], DownBlock]):
        super().__init__()

        self.channels = channels
        self.layers = nn.ModuleList([down_block_ctor(ci) for ci in channels])

    def forward(self, x):
        outputs = [] # outputs of all layers

        for layer in self.layers:
            o,x = layer(x)
            outputs.append(o)

        return outputs, x

    def get_channels(self):
        return self.channels
    
class SimpleBackbone(GenericBackbone):
    def __init__(self, channels: t.List[int]):
        down_block_ctor = lambda ci : DownBlock(
            block=nn.Sequential(
                nn.LazyConv2d(ci, kernel_size=3, padding=1),
                nn.BatchNorm2d(ci),
                nn.ReLU(inplace=True),

                nn.Conv2d(ci, ci, kernel_size=3, padding=1),
                nn.BatchNorm2d(ci),
                nn.ReLU(inplace=True),
            ),
            pool=nn.MaxPool2d(kernel_size=2, stride=2)
        )

        super().__init__(channels, down_block_ctor)

RESNET_CHANNELS_SMALL = [64, 64, 128, 256, 512]
RESNET_CHANNELS_LARGE = [64, 256, 512, 1024, 2048]

class ResnetBackbone(ABackbone):
    def __init__(self, resnet, channels):
        super().__init__()
        self.channels = channels
        
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
        return self.channels # see Resnet34- architecture

class Resnet18Backbone(ResnetBackbone):
    def __init__(self):
        super().__init__(torchvision.models.resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), RESNET_CHANNELS_SMALL)
    
class Resnet34Backbone(ResnetBackbone):
    def __init__(self):
        super().__init__(torchvision.models.resnet.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1), RESNET_CHANNELS_SMALL)

class Resnet50Backbone(ResnetBackbone):
    def __init__(self):
        super().__init__(torchvision.models.resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1), RESNET_CHANNELS_LARGE)

class Resnet101Backbone(ResnetBackbone):
    def __init__(self):
        super().__init__(torchvision.models.resnet.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1), RESNET_CHANNELS_LARGE)

class Resnet152Backbone(ResnetBackbone):
    def __init__(self):
        super().__init__(torchvision.models.resnet.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1), RESNET_CHANNELS_LARGE)