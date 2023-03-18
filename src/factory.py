
import typing as t

from src.metrics.continuous import *
from src.models.unet.blocks import UpBlock
from src.models.unet.backbones import Resnet18Backbone
from src.models.unet.unet import UNet
from src.models.unet.unet_pp import UNetPP

from src.pinject import Proxy, Injector


def build_model(data : t.Dict):
    return model_injector()(data)

def build_criterion(data : t.Dict):
    return criterion_injector()(data)

def complete_injector():
    i = Injector()
    i['model'] = model_injector()
    i['criterion'] = criterion_injector()
    return i

##### MODELS ######
def model_injector():
    def unets(i):
        unet = Proxy(UNet)
        backbones(unet)
        upblocks(unet)
        i['UNet'] = unet

        unetpp = Proxy(UNetPP)
        backbones(unetpp)
        upblocks(unetpp)
        i['UNet++'] = unetpp

    def backbones(i):
        b = Injector()
        b['Resnet18'] = Proxy(Resnet18Backbone)
        i['backbone'] = b

    def upblocks(i):
        b = Injector()
        b['UpBlock'] = Proxy(lambda **kwargs: (lambda ci: UpBlock(ci, **kwargs)))
        i['up_block_ctor'] = b

    i = Injector()
    unets(i)
    return i


##### CRITERION ######
def criterion_injector():
    i = Injector()

    i['FocalLoss'] = Proxy(FocalLoss)
    i['DiceLoss'] = Proxy(DiceLoss)
    i['JaccardLoss'] = Proxy(JaccardLoss)

    return i
