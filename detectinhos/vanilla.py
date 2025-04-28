from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Optional, TypeVar

import cv2
import numpy as np
import torch

from detectinhos.batch import BatchElement
from detectinhos.sample import Sample, read_dataset
from detectinhos.sublosses import WeightedLoss


def load_rgb(image_path: Path | str) -> np.array:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
    Optional[WeightedLoss],
)


@dataclass
class DetectionTargets(Generic[T]):
    boxes: T
    classes: T


def to_targets(
    sample: Sample,
    mapping: dict[str, int],
) -> DetectionTargets[np.ndarray]:
    bboxes = []
    label_ids = []

    for label in sample.annotations:
        bboxes.append(label.bbox)

        label_id = mapping.get(label.label, 0)
        label_ids.append([label_id])

    return DetectionTargets(
        boxes=np.array(bboxes),
        classes=np.array(label_ids),
    )


TRANSFORM_TYPE = Callable[[np.ndarray, T], tuple[np.ndarray, T]]


class DetectionDataset(torch.data.Dataset):
    def __init__(
        self,
        labels: Path | str,
        mapping: dict[str, int],
        transform: TRANSFORM_TYPE = lambda x, y: (x, y),
    ) -> None:
        self.mapping = mapping
        self.transform = transform
        self.labels = read_dataset(labels)

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
            targets=targets_t,
        )
