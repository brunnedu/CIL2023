import os
import pickle
import datetime

import optuna
import torch
from optuna.pruners import MedianPruner
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim import Adam

from src.dataset import SatelliteDataset
from src.metrics import FocalLoss, PatchAccuracy, BinaryF1Score, PatchF1Score
from src.models import UNet, Resnet18Backbone, UNetPP
from src.train import train_pl_wrapper
from src.transforms import AUG_TRANSFORM
from src.utils import ensure_dir, up_block_ctor_conv, create_optuna_config, PyTorchLightningPruningCallback
from src.wrapper import PLWrapper


OPTUNA_CONFIG = {
    'experiment_id': 'optuna_test_run',  # should be changed for every run
    'optuna': {
        'pruner_cls': MedianPruner,  # prunes unpromising trials if they perform worse than the median of the previous trials
        'pruner_kwargs': {
            'n_startup_trials': 5,
            'n_warmup_steps': 10,
        },
        'study_kwargs': {
            'direction': 'maximize',  # whether to minimize or maximize the objective
            'monitor': 'val_acc',  # value to optimize
        },
        'optimize_kwargs': {
            'n_trials': 50,
            'timeout': 24*3600,
        },
    },
    'dataset_kwargs': {
        'data_dir': 'data/training',
        'add_data_dir': 'data/data_2022',  # specify to use additional data
        'hist_equalization': False,
        'aug_transform': AUG_TRANSFORM,
    },
    'model_config': {
        'model_cls': UNetPP,
        'backbone_cls': Resnet18Backbone,
        'model_kwargs': {
            'up_block_ctor': up_block_ctor_conv,
        },
    },
    'pl_wrapper_kwargs': {
        'loss_fn': FocalLoss(alpha=0.25, gamma=2.0, bce_reduction='none'),
        'val_metrics': {
            'acc': PatchAccuracy(patch_size=16, cutoff=0.25),  # should always use PatchAccuracy as accuracy function
            'binaryf1score': BinaryF1Score(alpha=100.0),  # can add as many additional metrics as desired
            'patchf1score': PatchF1Score(),
        },
        'optimizer_cls': Adam,
        'optimizer_kwargs': {
            'lr': ('float-log', 1e-4, 1e-2),
            'weight_decay': ('float-log', 1e-6, 1e-3),
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
        ]
    },
    'train_pl_wrapper_kwargs': {
        'val_frac': 0.1,
        'batch_size': ('int-log', 4, 6),
        'num_workers_dl': 2,  # set to 0 if multiprocessing leads to issues
        'seed': 0,
        'save_checkpoints': False,  # saving checkpoints with optuna is not recommended
    }
}


def objective(trial: optuna.trial.Trial) -> float:

    config = create_optuna_config(OPTUNA_CONFIG, trial)

    # initialize dataset
    dataset = SatelliteDataset(**config['dataset_kwargs'])

    # initialize model
    model_config = config['model_config']
    model = model_config['model_cls'](backbone=model_config['backbone_cls'](), **model_config['model_kwargs'])

    # initialize pytorch lightning wrapper for model
    pl_wrapper = PLWrapper(
        model=model,
        **config['pl_wrapper_kwargs'],
    )

    # add pytorch lightning pruning callback
    config['pl_trainer_kwargs']['callbacks'].append((PyTorchLightningPruningCallback, {'trial': trial, 'monitor': OPTUNA_CONFIG['optuna']['study_kwargs']['monitor']}))

    trainer = train_pl_wrapper(
        experiment_id=experiment_id,
        pl_wrapper=pl_wrapper,
        dataset=dataset,
        pl_trainer_kwargs=config['pl_trainer_kwargs'],
        **config['train_pl_wrapper_kwargs'],
    )

    # objective function returns validation accuracy
    return trainer.callback_metrics[OPTUNA_CONFIG['optuna']['study_kwargs']['monitor']].item()


if __name__ == "__main__":
    # run optuna hyperparameter optimization
    experiment_id = f"{OPTUNA_CONFIG['experiment_id']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # create new directory for experiment and copy config to it
    experiment_dir = os.path.join("out", experiment_id)
    ensure_dir(experiment_dir)
    # store optuna config using pickle
    with open(os.path.join('out', experiment_id, 'optuna_config.pickle'), 'wb') as handle:
        # Dump the dictionary using pickle.dump()
        pickle.dump(OPTUNA_CONFIG, handle)

    # start optuna study
    pruner = OPTUNA_CONFIG['optuna']['pruner_cls'](**OPTUNA_CONFIG['optuna']['pruner_kwargs']) if OPTUNA_CONFIG['optuna']['pruner_cls'] else optuna.pruners.NopPruner()

    study = optuna.create_study(pruner=pruner, **OPTUNA_CONFIG['optuna']['study_kwargs'])
    study.optimize(objective, **OPTUNA_CONFIG['optuna']['optimize_kwargs'])

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # store study using pickle
    with open(os.path.join('out', experiment_id, 'study.pickle'), 'wb') as handle:
        # Dump the dictionary using pickle.dump()
        pickle.dump(study, handle)
