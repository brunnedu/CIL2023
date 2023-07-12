from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from src.models import UNet, UNetPP, UpBlock, LUNet, MAUNet, DLinkNet, DLinkUpBlock, SegmentationEnsemble, \
    PatchPredictionModel
from src.models import Resnet18Backbone, Resnet34Backbone, Resnet50Backbone, Resnet101Backbone, Resnet152Backbone
from src.models import EfficientNetV2_S_Backbone, EfficientNetV2_M_Backbone, EfficientNetV2_L_Backbone, EfficientNet_B5_Backbone
from src.models import MfidFinal
from src.metrics import DiceLoss, JaccardLoss, FocalLoss, BinaryF1Score, PatchAccuracy, PatchF1Score, \
    TopologyPreservingLoss
from src.transforms import AUG_TRANSFORM
import albumentations as A

from torch.optim import Adam
import torch
from torch import nn
from torchvision import transforms

PREDICT_USING_PATCHES = True
IS_REFINEMENT = False
INCLUDE_FLOW_INTERSECTION_DEADEND = True
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
    'model_cls': UNet,
    'backbone_cls': Resnet50Backbone,
    'model_kwargs': {
        'up_block_ctor': lambda ci: UpBlock(ci, up_mode='upconv'),
        'out_channels': 4 if INCLUDE_FLOW_INTERSECTION_DEADEND else 1,
        'final': MfidFinal(64),
        'multiscale_final': True
    },
    'backbone_kwargs': {
        'in_channels': 4 if IS_REFINEMENT else 3,
        # 'concat_group_channels': True # only for efficientnet backbones
    }
}

ENSEMBLE_MODEL_CONFIG = {
    'model_cls': SegmentationEnsemble,
    'model_kwargs': {
        'experiment_ids': [  # specify experiment ids of models to ensemble
            'unet_basic_r18_upsample_patches_focalloss_test_run_2023-06-28_02-10-44',
            'unet_basic_r18_upconv_patches_focalloss_test_run_2023-06-28_00-32-07',
            'unetpp_basic_r18_upsample_patches_focalloss_test_run_2023-06-28_02-43-56',
            'unetpp_basic_r18_upconv_patches_focalloss_test_run_2023-06-28_03-25-38',
        ],
        'freeze_submodels': True,  # decide whether to further train submodels
        'final_layer': nn.Sequential(  # define final layer to combine submodel outputs, if None the outputs will be averaged
            nn.LazyConv2d(1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        ),
        'last': False,  # if True, the submodels will be loaded from the last epoch instead of the best one
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

MODEL_CONFIG = UNET_MODEL_CONFIG

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

TRAIN_CONFIG = {
    'experiment_id': 'test_run',  # should be changed for every run
    'resume_from_checkpoint': False,  # set full experiment id (including timestamp) to resume from checkpoint
    'train_dataset_kwargs': {
        'data_dir': 'data/data1k',  # use our data for training
        'hist_equalization': False,
        'aug_transform': TRAIN_AUG_TRANSFORM,
        'include_low_quality_mask': IS_REFINEMENT,
        'include_fid': INCLUDE_FLOW_INTERSECTION_DEADEND,
    },
    'val_dataset_kwargs': {
        'data_dir': 'data/training',  # use original training data for validation
        'hist_equalization': False,
        'aug_transform': VAL_AUG_TRANSFORM,
        'include_low_quality_mask': IS_REFINEMENT,
        'include_fid': INCLUDE_FLOW_INTERSECTION_DEADEND,
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
    'experiment_id': 'test_run_2023-06-23_15-34-04',
    'dataset_kwargs': {
        'data_dir': 'data/test',
        'hist_equalization': False,
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
    'pl_wrapper_kwargs': PL_WRAPPER_KWARGS
}
