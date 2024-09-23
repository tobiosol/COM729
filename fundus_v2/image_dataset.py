import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from fundus_v2 import image_preprocessor, image_transforms
import fundus_v2.proj_util as proj_util


class FundusDataset(Dataset):
    def __init__(self, image_dir: str, csv_file: str, transform=None, test: bool = False, num_augmentations: int = 50):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(csv_file, index_col=0)
        self.labels_df.index = self.labels_df.index.astype(str)
        self.transform = transform
        self.test = test
        self.num_augmentations = num_augmentations
        self.image_paths = proj_util.load_images_from_folder(self.image_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_preprocessor = image_preprocessor.FundusImageProcessor()
        self.image_augmentor = image_transforms.FundusImageTransforms()

    def __len__(self) -> int:
        return len(self.image_paths) * (self.num_augmentations + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_idx = idx // (self.num_augmentations + 1)
        image_path = self.image_paths[original_idx]
        
        image = self._load_and_preprocess_image(image_path)
        label = self._get_labels(image_path)

        if self.transform:
            image = self.transform(image)

        image_tensor = self._prepare_image_tensor(image)
        return image_tensor.to(self.device), label

    def _load_and_preprocess_image(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        # if self.test:
        #     image = self.image_preprocessor.preprocess(image=image)
        return image
    
    @staticmethod
    def _prepare_image_tensor(image: Image.Image) -> torch.Tensor:
        # return torch.tensor(np.array(image), dtype=torch.float32).view(-1)  # Flatten to 1D
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).squeeze()
        image_tensor = image_tensor.repeat(1, 1, 1)  # Repeat along the batch dimension
        return image_tensor

    def _get_labels(self, image_filename: str) -> torch.Tensor:
        index = os.path.splitext(os.path.basename(image_filename))[0]
        row = self.labels_df.loc[index].values.astype(np.int64)
        label_index = np.argmax(row)
        return torch.tensor(label_index, dtype=torch.long).to(self.device)