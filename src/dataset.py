import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SatelliteDataset(Dataset):
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.img_paths = sorted([os.path.join(self.data_dir, 'images', f) for f in os.listdir(os.path.join(self.data_dir, 'images'))])
        self.mask_paths = sorted([os.path.join(self.data_dir, 'groundtruth', f) for f in os.listdir(os.path.join(self.data_dir, 'groundtruth'))])
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
