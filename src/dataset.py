import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SatelliteDataset(Dataset):
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()

    def __len__(self):
        return 144  # assuming there are 144 images in total

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, 'images', f'satimage_{idx}.png')
        mask_path = os.path.join(self.data_dir, 'groundtruth', f'satimage_{idx}.png')

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
