from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from src.models import UNet, UNetPP, UpBlock, LUNet, MAUNet, DLinkNet, DLinkUpBlock, SegmentationEnsemble
from src.models import Resnet18Backbone, Resnet34Backbone, Resnet50Backbone, Resnet101Backbone, Resnet152Backbone
from src.models import EfficientNetV2_S_Backbone, EfficientNetV2_M_Backbone, EfficientNetV2_L_Backbone, EfficientNet_B5_Backbone
from src.models import MfidFinal, VotenetFinal
from src.metrics import DiceLoss, JaccardLoss, FocalLoss, BinaryF1Score, PatchAccuracy, PatchF1Score, \
    TopologyPreservingLoss, UncertaintyMSELoss, OneMinusLossScore, UncertaintyMSEOnlyLoss
from src.transforms import AUG_TRANSFORM
import albumentations as A

from torch.optim import Adam
import torch
from torch import nn
from torchvision import transforms

PREDICT_USING_PATCHES = False # works better with more context => false
IS_REFINEMENT = False
MODEL_RES = 224  # Adjust resolution based on the model you're using
# Default: 224 (ResNets etc.)
# EfficientNetV2 S: 384, M: 480, L: 480
# EfficientNetB5: 456

# Automatically calculated - do not modify =============================================================================
TRAIN_AUG_TRANSFORM = A.Compose([
    A.RandomCrop(height=MODEL_RES, width=MODEL_RES) if PREDICT_USING_PATCHES else A.Resize(height=MODEL_RES, width=MODEL_RES),
    AUG_TRANSFORM
])

VAL_AUG_TRANSFORM = A.CenterCrop(height=MODEL_RES, width=MODEL_RES) \
    if PREDICT_USING_PATCHES else A.Resize(height=MODEL_RES, width=MODEL_RES)

RUN_AUG_TRANSFORM = None if PREDICT_USING_PATCHES else transforms.Resize(MODEL_RES)
# ======================================================================================================================

UNET_MODEL_CONFIG = {
    'model_cls': UNetPP,
    'backbone_cls': Resnet50Backbone,
    'model_kwargs': {
        'up_block_ctor': lambda ci: UpBlock(ci, up_mode='upconv'),
        'final': VotenetFinal()
    },
    'backbone_kwargs': {
        'in_channels': 3
    }
}

MODEL_CONFIG = UNET_MODEL_CONFIG

PL_WRAPPER_KWARGS = {
    'loss_fn': UncertaintyMSELoss(),
    # TopologyPreservingLoss(nr_of_iterations=50, weight_cldice=0.5, smooth=1.0)
    'val_metrics': {
        'acc': OneMinusLossScore(UncertaintyMSELoss()),
        'mse': UncertaintyMSEOnlyLoss()
    },
    'optimizer_cls': Adam,
    'optimizer_kwargs': {
        'lr': 1e-3,
        'weight_decay': 0,
    },
    'lr_scheduler_cls': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'lr_scheduler_kwargs': {
        'mode': 'min',
        'factor': 0.2,
        'patience': 10,
    },
}

TRAIN_CONFIG = {
    'experiment_id': 'test_run',  # should be changed for every run
    'resume_from_checkpoint': False,  # set full experiment id (including timestamp) to resume from checkpoint
    'train_dataset_kwargs': {
        'data_dir': 'data/data1k',  # use our data for training
        'hist_equalization': False,
        'aug_transform': TRAIN_AUG_TRANSFORM,
        'include_low_quality_mask': IS_REFINEMENT,
        'groundtruth_subfolder': 'transformed/angle',
    },
    'val_dataset_kwargs': {
        'data_dir': 'data/training',  # use original training data for validation
        'hist_equalization': False,
        'aug_transform': VAL_AUG_TRANSFORM,
        'include_low_quality_mask': IS_REFINEMENT,
        'groundtruth_subfolder': 'transformed/angle',
    },
    'model_config': MODEL_CONFIG,
    'pl_wrapper_kwargs': PL_WRAPPER_KWARGS,
    'pl_trainer_kwargs': {
        'max_epochs': 100,
        'log_every_n_steps': 1,
        'callbacks': [
            # (EarlyStopping, {'monitor': 'val_loss', 'mode': 'min', 'patience': 1}),
            # (LearningRateMonitor, {'logging_interval': 'epoch'}),
        ]
    },
    'train_pl_wrapper_kwargs': {
        'batch_size': 8,
        'num_workers_dl': 2,  # set to 0 if multiprocessing leads to issues
        'seed': 0,
        'save_checkpoints': True,
    }
}

RUN_CONFIG = {
    'experiment_id': 'test_run_2023-07-15_20-41-48',
    'dataset_kwargs': {
        'data_dir': 'data/training',
        'hist_equalization': False,
        'transform': RUN_AUG_TRANSFORM,
        'include_low_quality_mask': IS_REFINEMENT
    },
    'use_patches': PREDICT_USING_PATCHES,
    'patches_config': {
        'size': (224, 224),
        'subdivisions': (2, 2)  # keep in mind original images are 400 x 400
    },
    'select_channel': 1,
    'model_config': MODEL_CONFIG,
    'pl_wrapper_kwargs': PL_WRAPPER_KWARGS
}
