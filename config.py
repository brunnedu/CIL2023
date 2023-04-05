import torch
from pytorch_lightning.callbacks import EarlyStopping

from src.models import UNet, UNetPP, Resnet18Backbone, UpBlock
from src.metrics import DiceLoss, JaccardLoss, FocalLoss, BinaryF1Score, PatchAccuracy
from src.transforms import AUG_TRANSFORM

from torch.optim import Adam

TRAIN_CONFIG = {
    'experiment_id': 'test_run',  # should be changed for every run
    'resume_from_checkpoint': False,  # currently not working with pytorch lightning
    'dataset_kwargs': {
        'data_dir': 'data/training',
        'add_data_dir': None,  # specify to use additional data
        'hist_equalization': False,
        'aug_transform': AUG_TRANSFORM,
    },
    'model_config': {
        'model_cls': UNet,
        'backbone_cls': Resnet18Backbone,
        'model_kwargs': {
            'up_block_ctor': lambda ci: UpBlock(ci, up_mode='upconv'),
        },
    },
    'pl_wrapper_kwargs': {
        'loss_fn': FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
        'val_metrics': {
            'acc': PatchAccuracy(patch_size=16, cutoff=0.25),  # should always use PatchAccuracy as accuracy function
            'binaryf1score': BinaryF1Score(alpha=100.0),  # can add as many additional metrics as desired
        },
        'optimizer_cls': Adam,
        'optimizer_kwargs': {
            'lr': 1e-3,
            'weight_decay': 0,
        },
        'lr_scheduler_cls': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'lr_scheduler_kwargs': {
            'monitor': 'val_loss',
            'mode': 'min',
            'factor': 0.2,
            'patience': 10,
        },
    },
    'pl_trainer_kwargs': {
        'max_epochs': 100,
        'log_every_n_steps': 50,
        'callbacks': [  # list of callbacks (callback_cls, callback_kwargs)
            (EarlyStopping, {'monitor': 'val_acc', 'patience': 10, 'mode': 'max'}),
        ]
    },
    'train_pl_wrapper_kwargs': {
        'val_frac': 0.1,
        'batch_size': 64,
        'num_workers_dl': 4,  # set to 0 if multiprocessing leads to issues
        'seed': 0,
        'save_checkpoints': True,  # turn off to save storage space
    }
}
