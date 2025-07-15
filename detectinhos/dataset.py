from typing import Callable, TypeVar

import numpy as np
import torch

from detectinhos.batch import BatchElement
from detectinhos.data import Sample, load_rgb
from detectinhos.vanilla import to_targets

T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
)


def do_nothing(x: np.ndarray, y: T) -> tuple[np.ndarray, T]:
    return x, y


DatasetAugmentation = Callable[[np.ndarray, T], tuple[np.ndarray, T]]


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        labels: list[Sample],
        mapping: dict[str, int],
        transform: DatasetAugmentation = do_nothing,
    ) -> None:
        self.mapping = mapping
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> BatchElement[torch.Tensor]:
        sample = self.labels[index]
        image = load_rgb(sample.file_name)
        targets = to_targets(sample, self.mapping)
        image_t, targets_t = self.transform(image, targets)
        return BatchElement(
            file=sample.file_name,
            image=image_t,
            true=targets_t,
        )
