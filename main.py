import logging
import os
import sys

import click
import datetime
import shutil

from src.dataset import SatelliteDataset, SatelliteDatasetRun
from src.models.unet.unet import UNet
from src.train import train_pl_wrapper
from src.run import run_pl_wrapper

from src.mask_to_submission import masks_to_submission
from src.utils import create_logger, ensure_dir

# import the train config
from config import TRAIN_CONFIG, RUN_CONFIG
from src.wrapper import PLWrapper


@click.group()
def cli():
    pass


@cli.command()
def train():
    """
        Trains a model on all the provided data.
        Configuration dictionary `TRAIN_CONFIG` is expected to be in config.py in same directory.
    """
    config = TRAIN_CONFIG

    experiment_id = config['experiment_id']
    if not config['resume_from_checkpoint']:
        experiment_id = f"{experiment_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # create new directory for experiment and copy config to it
    experiment_dir = os.path.join('out', experiment_id)
    ensure_dir(experiment_dir)
    shutil.copy('config.py', os.path.join('out', experiment_id, 'config.py'))

    # initialize dataset
    ds_train = SatelliteDataset(**config['train_dataset_kwargs'])
    ds_val = SatelliteDataset(**config['val_dataset_kwargs'])

    # initialize model
    model_config = config['model_config']

    # models with backbone require separate initialization of backbone
    if 'backbone_cls' in model_config and model_config['backbone_cls'] is not None:
        model_config['model_kwargs']['backbone'] = model_config['backbone_cls']()

    model = model_config['model_cls'](**model_config['model_kwargs'])

    # initialize pytorch lightning wrapper for model
    pl_wrapper = PLWrapper(
        model=model,
        **config['pl_wrapper_kwargs'],
    )

    train_pl_wrapper(
        experiment_id=experiment_id,
        pl_wrapper=pl_wrapper,
        train_dataset=ds_train,
        val_dataset=ds_val,
        pl_trainer_kwargs=config['pl_trainer_kwargs'],
        resume_from_checkpoint=config['resume_from_checkpoint'],
        **config['train_pl_wrapper_kwargs'],
    )


@cli.command()
def run():
    """
        Runs a model on all the provided data and saves the output

        Additionally, (if make_submission flag is set), the submission.csv will be generated which conforms to the
        format required on the kaggle competition.
    """
    config = RUN_CONFIG
    experiment_id = config['experiment_id']

    # initialize dataset
    dataset = SatelliteDatasetRun(**config['dataset_kwargs'])

    # initialize model
    model_config = config['model_config']
    model = model_config['model_cls'](backbone=model_config['backbone_cls'](), **model_config['model_kwargs'])

    # initialize pytorch lightning wrapper for model
    pl_wrapper = PLWrapper(
        model=model,
        **config['pl_wrapper_kwargs'],
    )

    run_pl_wrapper(
        experiment_id=experiment_id,
        pl_wrapper=pl_wrapper,
        dataset=dataset,
        patches_config=config['patches_config'] if config['use_patches'] else None
    )


# Make sure to execute run first!
@cli.command()
# Unique ID for this experiment, make sure to use the full name (including the timestamp)
@click.argument('experiment_id', required=True)
@click.option('-t', '--foreground_threshold', default=0.5,
              help='The foreground threshold that should be used when generating a submission')
def submission(experiment_id, foreground_threshold):
    """
        Generates the submission.csv (according to the format specified on kaggle) and also generates each individual mask.
    """
    logger = create_logger(experiment_id)

    experiment_dir = f'./out/{experiment_id}'
    run_dir = os.path.join(experiment_dir, 'run')

    if not os.path.exists(run_dir):
        logger.error(f'Cannot generate submission before executing run as the outputs do not exist yet!')
        return

    image_filenames = [os.path.join(run_dir, name) for name in os.listdir(run_dir)]

    submission_dir = os.path.join(experiment_dir, f'submission{int(foreground_threshold * 100)}')
    os.makedirs(submission_dir, exist_ok=True)

    submission_filename = os.path.join(submission_dir, f'submission{int(foreground_threshold * 100)}.csv')
    masks_to_submission(
        submission_filename,
        submission_dir,
        foreground_threshold,
        *image_filenames
    )


@cli.command()
@click.argument('ground_truth_dir', type=click.Path(exists=True), required=True)
@click.argument('predicted_dir', type=click.Path(exists=True), required=True)
def evaluate(ground_truth_dir, predicted_dir):
    """
        Generates statistics for predictions vs ground truths
    """

    # Prepare Data

    # Execute

    # Generate Statistics


if __name__ == '__main__':
    cli()
