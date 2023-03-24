from src.models.unet.unet import UNet
from src.models.unet.unet_pp import UNetPP
from src.models.unet.backbones import Resnet18Backbone
from src.models.unet.blocks import UpBlock
from src.metrics.continuous import DiceLoss, JaccardLoss, FocalLoss, BinaryF1Score
from src.transforms import AUG_TRANSFORM
import albumentations as A

from torch.optim import Adam

TRAIN_CONFIG = {
        'experiment_id': 'train_unetpp_lr1e-3_upconv_aug',  # should be changed for every run
    'resume_from_checkpoint': False,
    'dataset_kwargs': {
        'data_dir': 'data/training',
        'add_data_dir': 'data/data_2022',  # specify to use additional data
        'hist_equalization': False,
        'aug_transform': AUG_TRANSFORM,
    },
    'model_cls': UNetPP,
    'backbone_cls': Resnet18Backbone,
    'model_kwargs': {
        'up_block_ctor': lambda ci: UpBlock(ci, up_mode='upconv'),
    },
    'optimizer_cls': Adam,
    'optimizer_kwargs': {
        'lr': 1e-3,
        'weight_decay': 0,
    },
    'train_model_kwargs': {
        'accuracy_fn': BinaryF1Score(alpha=100.0),  # OneMinusLossScore(FocalLoss())
        'criterion': FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
        'val_frac': 0.1,
        'num_epochs': 100,
        'batch_size': 32,
        'num_workers': 4,  # set to 0 if multiprocessing leads to issues
        'log_frequency': 10,
        'seed': 0,
        'save_models': True,
    }
}
