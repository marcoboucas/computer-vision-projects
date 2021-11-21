"""Birds data loader."""

import os
from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import image
from torch.utils.data import Dataset

from config import Config


class BirdsDatasetTypes(Enum):
    """Datasets types."""

    TRAIN = "train"
    TEST = "test"


def generate_file_dataset(dataset: BirdsDatasetTypes) -> pd.DataFrame:
    """Generate the dataset with all files (links)."""
    searched_path = os.path.join(Config.BIRDS_DATA_DIRECTORY, dataset)
    dataset = []
    for root, _, files in os.walk(searched_path):

        dataset.extend(
            [{"label": os.path.basename(root), "file": os.path.join(root, file)} for file in files]
        )
        if len(dataset) > 1000:
            break
    return pd.DataFrame.from_records(dataset)


class BirdsDataset(Dataset):
    """Birds dataset."""

    def __init__(self, dataset: BirdsDatasetTypes, transform) -> None:
        self.df = generate_file_dataset(dataset)
        self.target_to_label = self.df["label"].unique()
        self.label_to_target = {k: i for i, k in enumerate(self.target_to_label)}
        self.transform = transform

    def __len__(self) -> int:
        """Length of the dataset."""
        return self.df.shape[0]

    def __getitem__(self, index) -> np.ndarray:
        row = self.df.iloc[index]
        path, label = row["file"], row["label"]
        img = np.transpose(image.imread(path), (0, 1, 2))
        return self.transform(img), self._label_to_target(label)

    def _label_to_target(self, label: str) -> int:
        """Label to target."""
        return self.label_to_target[label]

    def _target_to_label(self, target: int) -> str:
        """Target to label."""
        return self.target_to_label[target]
