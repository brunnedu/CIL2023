import os
import typing as t

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

import pytorch_lightning as pl


def predict(
        image: torch.Tensor,
        original_size: t.Tuple[int],
        pl_wrapper: pl.LightningModule,
        device: str
) -> torch.Tensor:
    assert image.shape[0] == 1
    output = pl_wrapper(image.to(device))
    output = F.interpolate(output, original_size, mode='bilinear')
    return output


def predict_patches(
        image: torch.Tensor,
        patch_size: t.Tuple[int],
        subdivisions: t.Tuple[int],
        pl_wrapper: pl.LightningModule,
        device: str
) -> torch.Tensor:
    N, C, H, W = image.shape
    assert N == 1
    assert all(400 % subdiv == 0 for subdiv in subdivisions)  # make sure all input pixels are considered

    py, px = patch_size
    sy, sx = subdivisions

    stride_y = (H - py) / (sy - 1)
    stride_x = (W - px) / (sx - 1)

    positions = [(round(y * stride_y), round(x * stride_x)) for y in range(sy) for x in range(sx)]

    images = [image[0, :, y:y + py, x:x + px] for y, x in positions]
    images = torch.stack(images, dim=0).to(device)
    outputs = pl_wrapper(images)

    # compute the average over all generated patches
    output_total = torch.zeros((1, 1, H, W)).to(device)
    output_cnt = torch.zeros((1, 1, H, W)).to(device)
    for i, (y, x) in enumerate(positions):
        output_total[0, :, y:y + py, x:x + py] += outputs[i]
        output_cnt[0, :, y:y + py, x:x + py] += 1.0

    output = output_total / output_cnt
    return output


def run_pl_wrapper(
        experiment_id: str,
        dataset: Dataset,
        pl_wrapper: pl.LightningModule,
        patches_config: t.Optional[t.Dict]
) -> float:
    """
    Run the model on every element of a dataset, upscale to the original size and then save the resulting image
    """

    # create data loader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size must be 1!
    tb_logger = pl.loggers.TensorBoardLogger("tb_logs/", name=experiment_id)
    tb_logger.experiment.add_text('experiment_id', experiment_id)

    experiment_dir = os.path.join('out', experiment_id)
    checkpoint_file = [f for f in os.listdir(experiment_dir) if 'model-epoch=' in f][0]
    print(f'Loading {checkpoint_file}')

    if patches_config:
        print('Predicting Using Patches')
    else:
        print('Predicting Using Downscaled Image')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    # TODO: make this work with pl load_from_checkpoint (issue: plwrapper has module as hyperparameter)
    pl_wrapper.load_state_dict(
        torch.load(os.path.join(experiment_dir, checkpoint_file), map_location=torch.device(device))['state_dict'],
        device)
    pl_wrapper = pl_wrapper.to(device)
    pl_wrapper = pl_wrapper.eval()

    out_dir = f'./out/{experiment_id}/run'
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i, (names, original_sizes, images) in enumerate(dataloader):
            if patches_config:
                output = predict_patches(images, patches_config['size'], patches_config['subdivisions'], pl_wrapper,
                                         device)
            else:
                output = predict(images, original_sizes[0], pl_wrapper, device)

            output = output[0].to('cpu').squeeze().numpy() * 255
            cv2.imwrite(os.path.join(out_dir, names[0]), output)

            if i % 10 == 0:
                print(i)
