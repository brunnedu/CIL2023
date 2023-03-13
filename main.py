import click
import datetime;

from torch.optim import Adam

from src.dataset import SatelliteDataset
from src.metrics.continuous import *
from src.models.unet.backbones import *
from src.models.unet.blocks import *
from src.models.unet.unet import UNet
from src.train import train_model

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

    dataset = SatelliteDataset(data_dir, add_data_dir)

    model = UNet(Resnet18Backbone(), lambda ci: UpBlock(ci, up_mode='upsample'))
    criterion = FocalLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    if experiment_id is None:
        experiment_id = datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S')

    train_model(
        experiment_id=experiment_id,
        model=model,
        dataset=dataset,
        criterion=criterion,
        val_frac=0.15,
        optimizer=optimizer,
        num_epochs=5,
        batch_size=8,
        num_workers=0,
        resume_from_checkpoint=resume_from_checkpoint,
        log_frequency=1,
        fix_seed=True,
        seed=140499,
    )

@cli.command()
@click.argument('data_path', type=click.Path(exists=True), required=True)
@click.argument('model_path', type=click.Path(exists=True), required=True)
@click.option('-n', '--noisy_data', is_flag=True, help='Should the noisy dataset be included in the training?')
def evaluate(data_path, model_path, noisy_data):
    '''
        Runs a model for all provided data and generates statistics
    '''
    click.echo(f'{data_path}, {model_path}, {noisy_data}')

    # Prepare Data

    # Execute

    # Generate Statistics


if __name__ == '__main__':
    cli()