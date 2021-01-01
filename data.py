from typing import Tuple
import PIL
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage
from torch.utils.data.dataloader import DataLoader
from augmentations import get_augmentations
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
import os
import numpy as np
import torch

path = Path("../data/")


def list_files(path: Path):
    return [o for o in path.iterdir()]


def get_label(file_path: Path):
    return file_path.stem.split('.')[0]


class DogsandCatsDataset(Dataset):
    def __init__(self, files, transform=None) -> None:
        super().__init__()
        self.files = files
        self.transform = transform
        self.labels = {'cat': 0, 'dog': 1}

    def __getitem__(self, index) -> Tuple[PILImage, int]:
        file_path = self.files[index]
        label = self.labels[get_label(file_path)]
        image = Image.open(file_path)
        image = np.array(image)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.files)


class DogsandCatsDataModule(LightningDataModule):
    def __init__(
        self,
        path: str = None,
        aug_p: float = 0.5,
        val_pct: float = 0.2,
        img_sz: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
        fold_id: int = 0,
        splits: int = 5
    ):

        super().__init__()
        self.path = Path(path)
        self.aug_p = aug_p
        self.val_pct = val_pct
        self.img_sz = img_sz
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold_id = fold_id
        self.splits = splits

    def prepare_data(self):
        # only called on 1 GPU/TPU in distributed
        files = np.array(list_files(self.path/'train'))
        kf = KFold(n_splits=self.splits, random_state=2020, shuffle=True)
        splits = kf.split(files)
        train_idxs, validation_idxs = list(splits)[self.fold_id]
        self.train_files = files[train_idxs]
        self.valid_files = files[validation_idxs]
        self.train_transform, self.test_transform = get_augmentations(
            p=self.aug_p, image_size=self.img_sz
        )

    def train_dataloader(self):
        train_dataset = DogsandCatsDataset(
            files=self.train_files, transform=self.train_transform
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,

        )

    def val_dataloader(self):
        valid_dataset = DogsandCatsDataset(
            files=self.valid_files, transform=self.test_transform
        )
        return DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,

        )


if __name__ == "__main__":
    # Test Cassava Test Module
    path = Path("../data")

    dm = DogsandCatsDataModule(path)
    dm.prepare_data()
    xb, yb = next(iter(dm.train_dataloader()))
    print(xb.shape, yb.shape)
