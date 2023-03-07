import os

import torchvision
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.io import ImageReadMode

from src.transforms import RESNET_RESIZE


class SatelliteDataset(Dataset):
    """
    Dataset for loading satellite images and corresponding roadmap mask labels
    """
    def __init__(
            self,
            data_dir: str = 'data/training',
            transform: nn.Module = None,
    ):
        """
        Parameters
        ----------
        data_dir
            Directory where the data is located. Has to contain two subdirectories "images/" & "groundtruth/".
        transform
            A torchvision transform that is applied to every loaded image.
        """
        self.data_dir = data_dir
        self.img_paths = sorted([os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images'))])
        self.mask_paths = sorted([os.path.join(self.data_dir, 'groundtruth', f) for f in os.listdir(os.path.join(self.data_dir, 'groundtruth'))])
        self.transform = transform if transform else RESNET_RESIZE

        assert len(self.img_paths) == len(self.mask_paths), "number of satellite images doesn't match number of road masks"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)
        mask = torchvision.io.read_image(mask_path, mode=ImageReadMode.GRAY)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
