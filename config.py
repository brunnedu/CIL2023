from src.models import UNet, UNetPP, Resnet18Backbone, UpBlock, LUNet, DLinkNet, Resnet34Backbone
from src.metrics import DiceLoss, JaccardLoss, FocalLoss, BinaryF1Score, PatchAccuracy, PatchF1Score
from src.models.dlinknet.blocks import DLinkUpBlock
from src.transforms import AUG_TRANSFORM, AUG_PATCHES_TRANSFORM, RUN_TRANSFORM, RUN_PATCHES_TRANSFORM
import albumentations as A

from torch.optim import Adam
import torch

PREDICT_USING_PATCHES = False

MODEL_CONFIG = {
    'model_cls': DLinkNet,
    'backbone_cls': Resnet34Backbone,
    'model_kwargs': {
        'up_block_ctor': lambda ci, co: DLinkUpBlock(ci, co),
    },
}

PL_WRAPPER_KWARGS = {
    'loss_fn': FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
    'val_metrics': {
        'acc': PatchAccuracy(patch_size=16, cutoff=0.25),
        'binaryf1score': BinaryF1Score(alpha=100.0),  # can add as many additional metrics as desired
        'patchf1score': PatchF1Score(),
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
    'experiment_id': 'dinknet50_e100_d5kclean_no_patches',  # should be changed for every run
    'resume_from_checkpoint': False,  # set full experiment id (including timestamp) to resume from checkpoint
    'train_dataset_kwargs': {
        'data_dir': 'data/data5k_cleaned',  # use our data for training
        'hist_equalization': False,
        'aug_transform': AUG_PATCHES_TRANSFORM if PREDICT_USING_PATCHES else AUG_TRANSFORM,
    },
    'val_dataset_kwargs': {
        'data_dir': 'data/training',  # use original training data for validation
        'hist_equalization': False,
        'aug_transform': A.RandomCrop(height=224, width=224) if PREDICT_USING_PATCHES else A.Resize(height=224, width=224),
    },
    'model_config': MODEL_CONFIG,
    'pl_wrapper_kwargs': PL_WRAPPER_KWARGS,
    'pl_trainer_kwargs': {
        'max_epochs': 100,
        'log_every_n_steps': 1,
    },
    'train_pl_wrapper_kwargs': {
        'batch_size': 32,
        'num_workers_dl': 2,  # set to 0 if multiprocessing leads to issues
        'seed': 0,
        'save_checkpoints': True,
    }
}

RUN_CONFIG = {
    'experiment_id': '',
    'dataset_kwargs': {
        'data_dir': 'data/test',
        'hist_equalization': False,
        'transform': RUN_PATCHES_TRANSFORM if PREDICT_USING_PATCHES else RUN_TRANSFORM,
    },
    'use_patches': PREDICT_USING_PATCHES,
    'patches_config': {
        'size': (224, 224),
        'subdivisions': (4, 4)  # keep in mind original images are 400 x 400
    },
    'model_config': MODEL_CONFIG,
    'pl_wrapper_kwargs': PL_WRAPPER_KWARGS
}
