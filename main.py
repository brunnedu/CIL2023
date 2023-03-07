import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument('data_path', type=click.Path(exists=True), required=True)
@click.option('-n', '--noisy_data', is_flag=True, help='Should the noisy dataset be included in the training?')
def train(data_path, noisy_data):
    '''
        Trains a model on all the provided data
    '''
    click.echo(f'{data_path}, {noisy_data}') # TODO: remove this line
    
    # Prepare Data

    # Do Training
    

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