import os
from typing import Optional

import torchvision
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode

from src.transforms import RESNET_RESIZE


class SatelliteDataset(Dataset):
    """
    Dataset for loading satellite images and corresponding roadmap mask labels
    """
    def __init__(
            self,
            data_dir: str = 'data/training',
            add_data_dir: Optional[str] = None,
            transform: Optional[nn.Module] = RESNET_RESIZE,
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
        transform
            A torchvision transform that is applied to every loaded image.
        """
        self.data_dir = data_dir
        self.add_data_dir = add_data_dir
        self.img_paths = sorted([os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images'))])
        self.mask_paths = sorted([os.path.join(self.data_dir, 'groundtruth', f) for f in os.listdir(os.path.join(self.data_dir, 'groundtruth'))])
        self.transform = transform

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

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask


class SatelliteDatasetRun(Dataset):
    """
    Dataset for loading satellite images and their names and original size (but without labels)
    """
    def __init__(
            self,
            data_dir: str = 'data/test',
            transform: Optional[nn.Module] = RESNET_RESIZE,
    ):
        """
        Parameters
        ----------
        data_dir
            Directory where the original data is located. Has to contain two subdirectories "images" & "groundtruth".
            Both subdirectories should contain files with matching names.
        transform
            A torchvision transform that is applied to every loaded image.
        """
        self.data_dir = data_dir
        self.img_names = os.listdir(os.path.join(self.data_dir, 'images'))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)

        img = torchvision.io.read_image(img_path, mode=ImageReadMode.RGB)/255
        _,oh,ow = img.shape
        original_size = (oh,ow)

        if self.transform is not None:
            img = self.transform(img)

        return img_name, original_size, img