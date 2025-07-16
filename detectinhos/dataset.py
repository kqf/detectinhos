from pathlib import Path
from typing import Callable, TypeVar

import cv2
import numpy as np
import torch

from detectinhos.batch import BatchElement
from detectinhos.data import Sample
from detectinhos.vanilla import to_targets

T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
)


def do_nothing(x: np.ndarray, y: T) -> tuple[np.ndarray, T]:
    return x, y


DatasetAugmentation = Callable[[np.ndarray, T], tuple[np.ndarray, T]]


def load_rgb(image_path: Path | str) -> np.ndarray:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        labels: list[Sample],
        transform: DatasetAugmentation = do_nothing,
    ) -> None:
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> BatchElement[torch.Tensor]:
        sample = self.labels[index]
        raw_image = load_rgb(sample.file_name)
        raw_targets = to_targets(sample, {})
        image, targets = self.transform(
            raw_image,
            raw_targets,
        )
        return BatchElement(
            file=sample.file_name,
            image=image,
            true=targets,
        )
