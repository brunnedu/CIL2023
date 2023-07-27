import os

from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from src.models import UNet, UNetPP, UpBlock, LUNet, MAUNet, DLinkNet, DLinkUpBlock, SegmentationEnsemble, \
    PatchPredictionModel, EResUNet
from src.models import Resnet18Backbone, Resnet34Backbone, Resnet50Backbone, Resnet101Backbone, Resnet152Backbone
from src.models import EfficientNetV2_S_Backbone, EfficientNetV2_M_Backbone, EfficientNetV2_L_Backbone, EfficientNet_B5_Backbone
from src.models import MfidFinal
from src.metrics import DiceLoss, JaccardLoss, FocalLoss, BinaryF1Score, PatchAccuracy, PatchF1Score, \
    TopologyPreservingLoss, ThresholdBinaryF1Score
from src.transforms import AUG_TRANSFORM
import albumentations as A

from torch.optim import Adam
import torch
from torch import nn
from torchvision import transforms

PREDICT_USING_PATCHES = True
IS_REFINEMENT = False
INCLUDE_FLOW_INTERSECTION_DEADEND = False
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
    'model_cls': EResUNet,
    'backbone_cls': Resnet18Backbone,
    'model_kwargs': {},
    'backbone_kwargs': {
        'in_channels': 4 if IS_REFINEMENT else 3,
        # 'concat_group_channels': True # only for efficientnet backbones
    }
}

ENSEMBLE_MODEL_CONFIG = {
    'model_cls': SegmentationEnsemble,
    'model_kwargs': {
        'final_layer': nn.Sequential(  # define final layer to combine submodel outputs, if None the outputs will be averaged
            nn.LazyConv2d(1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ),
    },
}

PATCHES_MODEL_CONFIG = {
    'model_cls': PatchPredictionModel,  # set PREDICT_USING_PATCHES = False and MODEL_RES = 400 when using this model
    'model_kwargs': {
        'base_model_config': UNET_MODEL_CONFIG,  # can use any model config here
        'patches_config': {
            'patch_size': (224, 224),
            'subdivisions': (2, 2)  # keep in mind original images are 400 x 400
        },
    },

}

MODEL_CONFIG = ENSEMBLE_MODEL_CONFIG

PL_WRAPPER_KWARGS = {
    'loss_fn': FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
    # TopologyPreservingLoss(nr_of_iterations=50, weight_cldice=0.5, smooth=1.0)
    'val_metrics': {
        'acc': PatchF1Score(patch_size=16, cutoff=0.25),
        'binaryf1score': BinaryF1Score(alpha=100.0),  # can add as many additional metrics as desired
        'patchaccuracy': PatchAccuracy(patch_size=16, cutoff=0.25),
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
        'patience': 5,
    },
}

ENSEMBLE_EXPERIMENT_IDS = [  # specify experiment ids of models to ensemble
    'FINAL_unetpp_r50_upconv_patches_adam5e4_seed0_leakyrelu_2023-07-18_21-04-23',
    'FINAL_unetpp_r50_upconv_patches_adam5e4_seed0_2023-07-17_11-09-58',
]

TRAIN_CONFIG = {
    'experiment_id': 'ENSEMBLE_Test',  # should be changed for every run
    'resume_from_checkpoint': False,  # set full experiment id (including timestamp) to resume from checkpoint
    'train_dataset_kwargs': {
        # 'pred_dirs': [f"/cluster/scratch/brunnedu/out/{exp_id}/data5k" for exp_id in ENSEMBLE_EXPERIMENT_IDS],
        'pred_dirs': [f"/cluster/scratch/brunnedu/out/{exp_id}/data5k" for exp_id in ENSEMBLE_EXPERIMENT_IDS],
        'data_dir': 'data/data5k',  # use our data for training
        # 'data_dir': '/cluster/scratch/{ETHZ_NAME}/data30k',  # use (renamed) 30k dataset on scratch on cluster
        'aug_transform': TRAIN_AUG_TRANSFORM,
        'include_low_quality_mask': IS_REFINEMENT,
        # 'include_fid': INCLUDE_FLOW_INTERSECTION_DEADEND,
    },
    'val_dataset_kwargs': {
        # 'pred_dirs': [f"/cluster/scratch/brunnedu/out/{exp_id}/training" for exp_id in ENSEMBLE_EXPERIMENT_IDS],
        'pred_dirs': [f"out/{exp_id}/training" for exp_id in ENSEMBLE_EXPERIMENT_IDS],
        'data_dir': 'data/training',  # use original training data for validation
        'aug_transform': VAL_AUG_TRANSFORM,
        'include_low_quality_mask': IS_REFINEMENT,
        # 'include_fid': INCLUDE_FLOW_INTERSECTION_DEADEND,
    },
    'model_config': MODEL_CONFIG,
    'pl_wrapper_kwargs': PL_WRAPPER_KWARGS,
    'pl_trainer_kwargs': { 
        'max_epochs': 100,
        'log_every_n_steps': 10,
        'callbacks': [
            (LearningRateMonitor, {'logging_interval': 'epoch'}),
            # (EarlyStopping, {'monitor': 'val_acc', 'mode': 'max', 'patience': 20}),
        ]
    },
    'train_pl_wrapper_kwargs': {
        'batch_size': 32,
        'num_workers_dl': 2,  # set to 0 if multiprocessing leads to issues
        'seed': 0,
        'save_checkpoints': True,
    }
}

RUN_CONFIG = {
    'experiment_id': 'MODIFY_THIS',
    'dataset_kwargs': {
        'pred_dirs': [f"/cluster/scratch/brunnedu/out/{exp_id}/test" for exp_id in ENSEMBLE_EXPERIMENT_IDS],
        'data_dir': 'data/test',
        'transform': RUN_AUG_TRANSFORM,
        'include_low_quality_mask': IS_REFINEMENT
    },
    'use_patches': PREDICT_USING_PATCHES,
    'patches_config': {
        'size': (224, 224),
        'subdivisions': (2, 2)  # keep in mind original images are 400 x 400
    },
    'select_channel': 0 if INCLUDE_FLOW_INTERSECTION_DEADEND else None,
    'model_config': MODEL_CONFIG,
    'pl_wrapper_kwargs': PL_WRAPPER_KWARGS,
    'eval_metrics': [
        FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
        PatchF1Score(patch_size=16, cutoff=0.25),
        ThresholdBinaryF1Score(cutoff=0.25),
        PatchAccuracy(patch_size=16, cutoff=0.25),
    ],
}
