import torch
import torch.nn as nn

from abc import ABC, abstractmethod
import typing as t

from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights, resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights, \
    EfficientNet_B5_Weights, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, efficientnet_b5

from .blocks import DownBlock


class ABackbone(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def get_channels(self) -> t.List[int]:
        """
            Returns the number of channels for each level on the backbone
        """
        pass


class GenericBackbone(ABackbone):
    """
        A generic backbone that replicates a module for each layer and adjusts the number of channels

        Parameters:
            - channels: list of integers that represents the output channel on each level,
                        e.g. [4, 16, 32, 32, 64] (=> network with depth 5)
            - down_block_ctor: function that constructs a down block when given the amount of output channels
                        e.g. lambda ci: DownBlock(LazyConvBlock(ci), MaxPool)
    """

    def __init__(self, channels: t.List[int],
                 down_block_ctor: t.Callable[[int], DownBlock]):
        super().__init__()

        self.channels = channels
        self.layers = nn.ModuleList([down_block_ctor(ci) for ci in channels])

    def forward(self, x):
        outputs = []  # outputs of all layers

        for layer in self.layers:
            o, x = layer(x)
            outputs.append(o)

        return outputs, x

    def get_channels(self):
        return self.channels


class SimpleBackbone(GenericBackbone):
    def __init__(self, channels: t.List[int]):
        def down_block_ctor(ci):
            return DownBlock(
                block=nn.Sequential(
                    nn.Conv2d(ci, ci, kernel_size=3, padding=1),
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
    def __init__(self, resnet, channels: t.List[int], in_channels: int = 3):
        super().__init__()
        self.channels = channels

        children = list(resnet.children())

        modules = []
        first_conv = children[0] if in_channels == 3 else nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)
        modules.append(DownBlock( # first 4 resnet layers are a bit special
            nn.Sequential(first_conv, *children[1:3]),
            children[3]
        ))

        for block in children:
            if isinstance(block, nn.Sequential):  # all blocks are of type sequential
                modules.append(DownBlock(block))

        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        outputs = []  # all outputs between bottlenecks

        for layer in self.layers:
            o, x = layer(x)
            outputs.append(o)

        return outputs, x

    def get_channels(self):
        return self.channels


class Resnet18Backbone(ResnetBackbone):
    def __init__(self, in_channels: int = 3):
        super().__init__(resnet18(weights=ResNet18_Weights.DEFAULT), RESNET_CHANNELS_SMALL, in_channels)


class Resnet34Backbone(ResnetBackbone):
    def __init__(self, in_channels: int = 3):
        super().__init__(resnet34(weights=ResNet34_Weights.DEFAULT), RESNET_CHANNELS_SMALL, in_channels)


class Resnet50Backbone(ResnetBackbone):
    def __init__(self, in_channels: int = 3):
        super().__init__(resnet50(weights=ResNet50_Weights.DEFAULT), RESNET_CHANNELS_LARGE, in_channels)


class Resnet101Backbone(ResnetBackbone):
    def __init__(self, in_channels: int = 3):
        super().__init__(resnet101(weights=ResNet101_Weights.DEFAULT), RESNET_CHANNELS_LARGE, in_channels)


class Resnet152Backbone(ResnetBackbone):
    def __init__(self, in_channels: int = 3):
        super().__init__(resnet152(weights=ResNet152_Weights.DEFAULT), RESNET_CHANNELS_LARGE, in_channels)



#########################
#### EfficientNet V2 ####
#########################

class ConcatChannelsDownBlock(nn.Module):
    """ 
        Downblock that applies all children sequentially and passes the final output down.
        The "level"-output is the concatenation of all children's outputs on the channel dimension
    """
    def __init__(self, children: t.List[nn.Module]):
        super().__init__()

        self.layers = nn.ModuleList(children)

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return torch.cat(outs, dim=1), x

class EfficientNetBackbone(ABackbone):
    """ 
        Backbone based on the efficientnet architecture, 
        we choose to group blocks together (child_group_idx) of which either only the final output or all outputs stacked channel-wise
        (if concat_group_channels is true) will be returned.
        The grouping can be useful e.g. if two consecutive blocks work on the same image size. 
    """
    def __init__(self, efficientnet, child_channels: t.List[int], child_group_idx: t.List[int], concat_group_channels:bool = False, in_channels: int = 3):
        super().__init__()
        
        children = list(efficientnet.features.children())
        assert len(child_group_idx) == len(child_channels)
        assert len(child_group_idx) == len(children)

        if in_channels != 3:
            block0 = list(children[0].children())
            conv0 : nn.Conv2d = block0[0]
            children[0] = nn.Sequential(
                nn.Conv2d(in_channels, conv0.out_channels, conv0.kernel_size, conv0.stride, conv0.padding, conv0.dilation, bias=conv0.bias),
                *block0[1:]
            )

        modules = []
        channels = []

        current_group_idx = child_group_idx[0]
        current_group = []
        current_channels = []
        for child,group,channel in zip(children, child_group_idx, child_channels):
            if group != current_group_idx:
                if concat_group_channels:
                    modules.append(ConcatChannelsDownBlock(current_group))
                    channels.append(sum(current_channels))
                else:
                    modules.append(DownBlock(nn.Sequential(*current_group)))
                    channels.append(current_channels[-1])
                current_group = []
                current_channels = []

            current_group.append(child)
            current_group_idx = group
            current_channels.append(channel)
        
        # add the last group
        if concat_group_channels:
            modules.append(ConcatChannelsDownBlock(current_group))
            channels.append(sum(current_channels))
        else:
            modules.append(DownBlock(nn.Sequential(*current_group)))
            channels.append(current_channels[-1])

        self.channels = channels
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        outputs = []  # all outputs between bottlenecks

        for layer in self.layers:
            o, x = layer(x)
            outputs.append(o)

        return outputs, x

    def get_channels(self):
        return self.channels
    
class EfficientNetV2_S_Backbone(EfficientNetBackbone):
    def __init__(self, concat_group_channels: bool = False, in_channels: int = 3):
        super().__init__(
            efficientnet = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT), 
            child_channels = [24, 24, 48, 64, 128, 160, 256, 1280], 
            child_group_idx = [0, 0, 1, 2, 3, 3, 4, 4],
            concat_group_channels = concat_group_channels,
            in_channels = in_channels
        )

class EfficientNetV2_M_Backbone(EfficientNetBackbone):
    def __init__(self, concat_group_channels: bool = False, in_channels: int = 3):
        super().__init__(
            efficientnet = efficientnet_v2_m(EfficientNet_V2_M_Weights.DEFAULT), 
            child_channels = [24, 24, 48, 80, 160, 176, 304, 512, 1280], 
            child_group_idx = [0, 0, 1, 2, 3, 3, 4, 4, 4],
            concat_group_channels = concat_group_channels,
            in_channels = in_channels
        )

class EfficientNetV2_L_Backbone(EfficientNetBackbone):
    def __init__(self, concat_group_channels: bool = False, in_channels: int = 3):
        super().__init__(
            efficientnet = efficientnet_v2_l(EfficientNet_V2_L_Weights.DEFAULT), 
            child_channels = [32, 32, 64, 96, 192, 224, 384, 640, 1280], 
            child_group_idx = [0, 0, 1, 2, 3, 3, 4, 4, 4],
            concat_group_channels = concat_group_channels,
            in_channels = in_channels
        )

class EfficientNet_B5_Backbone(EfficientNetBackbone):
    def __init__(self, concat_group_channels: bool = False, in_channels: int = 3):
        super().__init__(
            efficientnet = efficientnet_b5(EfficientNet_B5_Weights.DEFAULT),
            child_channels = [48, 24, 40, 64, 128, 176, 304, 512, 2048],
            child_group_idx = [0, 0, 1, 2, 3, 3, 4, 4, 4],
            concat_group_channels = concat_group_channels,
            in_channels = in_channels
        )