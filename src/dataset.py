import os
from typing import Optional

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode

from albumentations.core.composition import BaseCompose

from src.transforms import AUG_TRANSFORM


class SatelliteDataset(Dataset):
    """
    Dataset for loading satellite images and corresponding roadmap mask labels
    """
    def __init__(
            self,
            data_dir: str = 'data/training',
            add_data_dir: Optional[str] = None,
            aug_transform: Optional[BaseCompose] = AUG_TRANSFORM,
    ):
        """
        Parameters
        ----------
        data_dir
            Directory where the original data is located. Has to contain two subdirectories "images" & "groundtruth".
            Both subdirectories should contain files with matching names.
        add_data_dir
            Directory where additional data is located. Has to contain two subdirectories "images" & "groundtruth".
            Both subdirectories should contain files with matching names.
        aug_transform
            An albumentation transform that is applied to both the satellite image and the road mask .
        """
        self.data_dir = data_dir
        self.add_data_dir = add_data_dir
        self.img_paths = sorted([os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images'))])
        self.mask_paths = sorted([os.path.join(self.data_dir, 'groundtruth', f) for f in os.listdir(os.path.join(self.data_dir, 'groundtruth'))])
        self.aug_transform = aug_transform

        if self.add_data_dir:
            # add additional data
            self.img_paths += sorted(
                [os.path.join(self.add_data_dir, 'images', f) for f in
                 os.listdir(os.path.join(self.add_data_dir, 'images'))])
            self.mask_paths += sorted(
                [os.path.join(self.add_data_dir, 'groundtruth', f) for f in
                 os.listdir(os.path.join(self.add_data_dir, 'groundtruth'))])

        assert len(self.img_paths) == len(self.mask_paths), "number of satellite images doesn't match number of road masks"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)/255
        mask = torchvision.io.read_image(mask_path, mode=ImageReadMode.GRAY)/255

        if self.aug_transform is not None:
            transformed = self.aug_transform(image=img.permute(1, 2, 0).numpy(), mask=mask.permute(1, 2, 0).numpy())
            img, mask = torch.from_numpy(transformed['image']).permute(2, 0, 1), torch.from_numpy(
                transformed['mask']).permute(2, 0, 1)

        return img, mask
