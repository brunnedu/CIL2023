import os
import click
import datetime;

from torch.optim import Adam

from src.dataset import SatelliteDataset, SatelliteDatasetRun
from src.metrics.continuous import *
from src.models.unet.backbones import *
from src.models.unet.blocks import *
from src.models.unet.unet import UNet
from src.train import train_model
from src.run import run_model

from src.mask_to_submission_old import masks_to_submission # TODO: use the 2023 mask_to_submission script once released
from src.utils import create_logger 

@click.group()
def cli():
    pass

@cli.command()
# Directory where the original data is located. Has to contain two subdirectories "images" & "groundtruth". Both subdirectories should contain files with matching names.
@click.argument('data_dir', type=click.Path(exists=True), required=True)
@click.option('-a', '--add_data_dir', 
              help='''Directory where additional data is located. Has to contain two subdirectories "images" & "groundtruth". 
                      Both subdirectories should contain files with matching names.''')
@click.option('-id', '--experiment_id', 
              help='Unique ID for this experiment')
@click.option('-r', '--resume_from_checkpoint', is_flag=True,
              help='Will resume from the last checkpoint of the corresponding experiment if set to true')
def train(data_dir, add_data_dir, experiment_id, resume_from_checkpoint):
    '''
        Trains a model on all the provided data
    '''
    # TODO: add support for different data augmentation, models, loss functions etc
    logger = create_logger(experiment_id)

    dataset = SatelliteDataset(data_dir, add_data_dir)

    model = UNet(Resnet18Backbone(), lambda ci: UpBlock(ci, up_mode='upsample'))
    criterion = FocalLoss()
    accuracy_fn = BinaryF1Score(alpha=100.0) # OneMinusLossScore(FocalLoss())
    optimizer = Adam(model.parameters(), lr=0.001)

    if experiment_id is None:
        experiment_id = datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S')

    train_model(
        experiment_id=experiment_id,
        model=model,
        dataset=dataset,
        criterion=criterion,
        accuracy_fn=accuracy_fn,
        val_frac=0.15,
        optimizer=optimizer,
        num_epochs=5,
        batch_size=8,
        num_workers=0,
        resume_from_checkpoint=resume_from_checkpoint,
        log_frequency=1,
        fix_seed=True,
        seed=140499,
        logger=logger
    )

@cli.command()
# Directory that contains a subdirectory "images" which contains aerial images
@click.argument('data_dir', type=click.Path(exists=True), required=True)
@click.argument('experiment_id', required=True)
def run(data_dir, experiment_id):
    '''
        Runs a model on all the provided data and saves the output

        Additionally (if make_submission flag is set), the submission.csv will be generated which conforms to the format required on the kaggle competition.
    '''
    logger = create_logger(experiment_id)

    dataset = SatelliteDatasetRun(data_dir)
    model = UNet(Resnet18Backbone(), lambda ci: UpBlock(ci, up_mode='upsample'))

    run_model(
        experiment_id=experiment_id,
        model=model,
        dataset=dataset,
        log_frequency=10,
        logger=logger
    )

# Make sure to execute run first!
@cli.command()
@click.argument('experiment_id', required=True)
@click.option('-t', '--foreground_threshold', default=0.5, 
              help='The foreground threshold that should be used when generating a submission')
def submission(experiment_id, foreground_threshold):
    '''
        Generates the submission.csv (according to the format specified on kaggle) and also generates each individual mask.
    '''
    logger = create_logger(experiment_id)

    experiment_dir = f'./out/{experiment_id}'
    run_dir = os.path.join(experiment_dir, 'run')
    
    if not os.path.exists(run_dir):
        logger.error(f'Cannot generate submission before executing run as the outputs do not exist yet!')
        return
    
    image_filenames = [os.path.join(run_dir, name) for name in os.listdir(run_dir)]

    submission_dir = os.path.join(experiment_dir, f'submission{int(foreground_threshold*100)}')
    os.makedirs(submission_dir, exist_ok=True)

    submission_filename=os.path.join(submission_dir, f'submission{int(foreground_threshold*100)}.csv')
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
    '''
        Generates statistics for predictions vs ground truths
    '''

    # Prepare Data

    # Execute

    # Generate Statistics


if __name__ == '__main__':
    cli()