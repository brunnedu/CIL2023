import json
import logging
import os
import sys
import importlib

import click
import datetime
import shutil

import torch

from src.dataset import SatelliteDataset, SatelliteDatasetRun
from src.models.unet.unet import UNet
from src.train import train_pl_wrapper
from src.run import run_pl_wrapper

from src.mask_to_submission import masks_to_submission
from src.utils import create_logger, ensure_dir, init_model, get_config, init_wrapper, load_segmentation_masks

# import the train config
from config import TRAIN_CONFIG, RUN_CONFIG


@click.group()
def cli():
    pass


@cli.command()
def train():
    """
        Trains a model on all the provided data.
        Configuration dictionary `TRAIN_CONFIG` is expected to be in config.py in same directory.
    """

    experiment_id = TRAIN_CONFIG['experiment_id']
    if not TRAIN_CONFIG['resume_from_checkpoint']:
        experiment_id = f"{experiment_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # create new directory for experiment and copy config to it
    experiment_dir = os.path.join('out', experiment_id)
    ensure_dir(experiment_dir)
    shutil.copy('config.py', os.path.join('out', experiment_id, 'config.py'))

    # initialize dataset
    ds_train = SatelliteDataset(**TRAIN_CONFIG['train_dataset_kwargs'])
    ds_val = SatelliteDataset(**TRAIN_CONFIG['val_dataset_kwargs'])

    pl_wrapper = init_wrapper(TRAIN_CONFIG)

    train_pl_wrapper(
        experiment_id=experiment_id,
        pl_wrapper=pl_wrapper,
        train_dataset=ds_train,
        val_dataset=ds_val,
        pl_trainer_kwargs=TRAIN_CONFIG['pl_trainer_kwargs'],
        resume_from_checkpoint=TRAIN_CONFIG['resume_from_checkpoint'],
        **TRAIN_CONFIG['train_pl_wrapper_kwargs'],
    )


def _run(output_path, use_last_ckpt=False, no_auto_config=False):
    """
        Runs a model on all the provided data and saves the output

        Additionally, (if make_submission flag is set), the submission.csv will be generated which conforms to the
        format required on the kaggle competition.
    """
    experiment_id = RUN_CONFIG['experiment_id']

    # try to retrieve original RUN_CONFIG from experiment directory
    # but always use RUN_CONFIG from CIL2023/config.py for dataset_kwargs and patches_config
    orig_config_path = os.path.join("out", experiment_id, "config.py")
    if os.path.isfile(os.path.join("out", experiment_id, "config.py")) and not no_auto_config:
        print(f"Original RUN_CONFIG successfully retrieved from experiment directory.")
        config = get_config(experiment_id)
        config = config.RUN_CONFIG
    else:
        print(f"Could not retrieve original RUN_CONFIG from {orig_config_path}. Will use CIL2023/config.py instead.")
        config = RUN_CONFIG

    # initialize dataset
    dataset = SatelliteDatasetRun(**RUN_CONFIG['dataset_kwargs'])

    run_pl_wrapper(
        experiment_id=experiment_id,
        dataset=dataset,
        patches_config=RUN_CONFIG['patches_config'] if config['use_patches'] else None,
        out_dir=output_path,
        use_last_ckpt=use_last_ckpt,
        select_channel=RUN_CONFIG['select_channel']
    )


@cli.command()
@click.option('-o', '--output_path', default=None,
              help='Where the outputs should be stored')
@click.option('-l', '--use_last_ckpt', is_flag=True,
              help='Use last checkpoint for inference')
@click.option('-a', '--no_auto_config', is_flag=True,
              help='Dont try to retrieve original RUN_CONFIG from experiment directory')
def run(output_path, use_last_ckpt, no_auto_config):
    """
        Runs a model on all the provided data and saves the output
    """
    _run(output_path, use_last_ckpt, no_auto_config)


@cli.command()
def prepare_for_refinement():
    """
        Runs the model on all data (including training data) and generates /lowqualitymask for each of the datasets
    """
    paths = [
        'data/training',
        'data/test',
        'data/data1k',
        'data/data5k',
        'data/data30k',
    ]

    for path in paths:
        RUN_CONFIG['dataset_kwargs']['data_dir'] = path
        _run(os.path.join(path, 'lowqualitymask'))


# Make sure to execute run first!
@cli.command()
# Unique ID for this experiment, make sure to use the full name (including the timestamp)
@click.argument('experiment_id', required=True)
@click.option('-t', '--foreground_threshold', default=0.25,
              help='The foreground threshold that should be used when generating a submission')
@click.option('-l', '--use_last_ckpt', is_flag=True, help='Use last checkpoint for submission')
def submission(experiment_id, foreground_threshold, use_last_ckpt):
    """
        Generates the submission.csv (according to the format specified on kaggle) and also generates each individual mask.
    """
    logger = create_logger(experiment_id)

    experiment_dir = os.path.join('out', experiment_id)
    run_dir = os.path.join(experiment_dir, 'run_last' if use_last_ckpt else 'run')

    if not os.path.exists(run_dir):
        logger.error(f'Cannot generate submission before executing run as the outputs do not exist yet!')
        return

    image_filenames = [os.path.join(run_dir, name) for name in os.listdir(run_dir)]

    submission_str = f'submission{int(foreground_threshold * 100)}_last' if use_last_ckpt else f'submission{int(foreground_threshold * 100)}'
    submission_dir = os.path.join(experiment_dir, submission_str)
    os.makedirs(submission_dir, exist_ok=True)

    submission_filename = os.path.join(submission_dir, f'{submission_str}.csv')
    masks_to_submission(
        submission_filename,
        submission_dir,
        foreground_threshold,
        *image_filenames
    )


@cli.command()
@click.argument('test_data_dir', type=click.Path(exists=True), required=True, default=os.path.join('data', 'test500'))
def evaluate(test_data_dir):
    """
        Generates statistics for predictions vs ground truths
    """
    experiment_id = RUN_CONFIG['experiment_id']
    # Generate Predictions
    output_path = os.path.join('out', experiment_id, os.path.basename(test_data_dir))
    if not os.path.exists(output_path):
        print("No predictions found, generating them now")
        RUN_CONFIG['dataset_kwargs']['data_dir'] = test_data_dir
        _run(output_path)
    else:
        print("Predictions found, skipping generation")

    # Evaluate Predictions
    print('Evaluating Predictions')
    predictions = load_segmentation_masks(output_path)
    groundtruths = load_segmentation_masks(os.path.join(test_data_dir, 'groundtruth'))

    results = {}
    eval_metrics = RUN_CONFIG['eval_metrics']
    for metric in eval_metrics:
        evals = torch.stack([metric(pred, gt) for pred, gt in zip(predictions, groundtruths)])
        results[str(metric)] = {}
        results[str(metric)]['mean'] = evals.mean().item()
        results[str(metric)]['std'] = evals.std().item()

    # Save statistics
    print('Saving Statistics')
    with open(os.path.join(output_path, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    cli()
