import os
import click
import datetime
import json
import shutil

from torch.optim import Adam

from src.dataset import SatelliteDataset, SatelliteDatasetRun
from src.metrics.continuous import *
from src.models.unet.backbones import *
from src.models.unet.blocks import *
from src.models.unet.unet import UNet
from src.train import train_model
from src.run import run_model
from src.factory import *

from src.mask_to_submission_old import masks_to_submission  # TODO: use the 2023 mask_to_submission script once released
from src.utils import create_logger

# import the train config
from config import TRAIN_CONFIG


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

    logger = create_logger(experiment_id)
    logger.info(f'Full Experiment ID: {experiment_id}')

    # copy current config to experiment folder
    shutil.copy('config.py', f'./out/{experiment_id}/config.py')

    # initialize objects
    dataset = SatelliteDataset(**config['dataset_kwargs'])

    # backbone gets instantiated separately
    model = config['model_cls'](backbone=config['backbone_cls'](), **config['model_kwargs'])
    optimizer = config['optimizer_cls'](model.parameters(), **config['optimizer_kwargs'])

    train_model(
        experiment_id=experiment_id,
        resume_from_checkpoint=config['resume_from_checkpoint'],
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        fix_seed=True,
        logger=logger,
        **config['train_model_kwargs'],
    )


@cli.command()
# Directory that contains a subdirectory "images" which contains aerial images
@click.argument('data_dir', type=click.Path(exists=True), required=True)
# Unique ID for this experiment, make sure to use the full name (including the timestamp)
@click.argument('experiment_id', required=True)
def run(data_dir, experiment_id):
    """
        Runs a model on all the provided data and saves the output

        Additionally, (if make_submission flag is set), the submission.csv will be generated which conforms to the
        format required on the kaggle competition.
    """
    logger = create_logger(experiment_id)

    with open(f'./out/{experiment_id}/config.json') as json_file:
        config = json.load(json_file)

    dataset = SatelliteDatasetRun(data_dir)
    model = build_model(config['model'])

    run_model(
        experiment_id=experiment_id,
        model=model,
        dataset=dataset,
        log_frequency=10,
        logger=logger
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


@cli.command()
def conf():
    """
        Gives all the configuration options
    """

    schema = complete_injector().hierarchical_schema()
    print(schema)

    with open('CONFIG_SCHEMA.md', 'w') as f:
        f.write(schema)


if __name__ == '__main__':
    cli()
