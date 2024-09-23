import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = Path(current_dir) / 'fundus_v2'
sys.path.append(str(subdirectory_path))

import torch
from torch.utils.data import DataLoader
from fundus_v2 import image_dataset

class FundusImageLoader:
    def __init__(self, image_dir, csv_file, batch_size=16, shuffle=True, transform=None, train=True, num_augmentations=50):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.train = train
        self.num_augmentations = num_augmentations
        self.dataloader = self._create_dataloader()

    def get_loader(self):
        return self.dataloader

    def __iter__(self):
        return iter(self.dataloader)

    def _create_dataloader(self):
        dataset = image_dataset.FundusDataset(
            image_dir=self.image_dir,
            csv_file=self.csv_file,
            transform=self.transform,
            test=not self.train,
            num_augmentations=self.num_augmentations
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
