from dataclasses import dataclass, fields
from typing import Callable, Generic, List, Protocol, TypeVar

import torch
from torch.nn.utils.rnn import pad_sequence

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    @classmethod
    def is_dataclass(cls) -> bool:
        ...


# A single element in the batch
@dataclass
class BatchElement(Generic[T]):
    file: str
    image: torch.Tensor
    targets: HasBoxesAndClasses[T]


# Stacked BatchElements along batch dimension
@dataclass
class Batch(Generic[T]):
    files: list[str]
    image: torch.Tensor
    targets: HasBoxesAndClasses[T]


def detection_collate(
    batch: List[BatchElement],
    to_targets: Callable[..., HasBoxesAndClasses],
) -> Batch:
    images = torch.stack([sample.image for sample in batch])
    targets = {
        field.name: pad_sequence(
            [torch.tensor(getattr(e.targets, field.name)) for e in batch],
            batch_first=True,
            padding_value=0,
        )
        for field in fields(batch[0].targets)
    }
    files = [sample.file for sample in batch]
    return Batch(files, images, to_targets(**targets))
