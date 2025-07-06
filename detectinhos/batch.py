from dataclasses import dataclass, fields
from typing import Callable, Generic, List, Optional, Protocol, TypeVar

import torch
from torch.nn.utils.rnn import pad_sequence

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    def __getitem__(self, idx) -> "HasBoxesAndClasses": ...


# A single element in the batch
@dataclass
class BatchElement(Generic[T]):
    file: str
    image: torch.Tensor
    true: HasBoxesAndClasses[T]


# Stacked BatchElements along batch dimension
@dataclass
class Batch(Generic[T]):
    files: list[str]
    image: T
    # Can be optional when we are doing inference
    true: Optional[HasBoxesAndClasses[T]] = None
    # Is optional before forward pass
    pred: Optional[HasBoxesAndClasses[T]] = None


def apply_eval(
    batch: Batch[T],
    model: torch.nn.Module,
) -> HasBoxesAndClasses[T]:
    original_mode = model.training
    model.eval()
    predicted = model(batch.image)
    model.train(original_mode)
    return predicted


def detection_collate(
    batch: List[BatchElement],
    to_targets: Callable[..., HasBoxesAndClasses],
) -> Batch:
    images = torch.stack([torch.Tensor(sample.image) for sample in batch])
    targets = {
        field.name: pad_sequence(
            [torch.Tensor(getattr(e.true, field.name)) for e in batch],
            batch_first=True,
            padding_value=0,
        )
        for field in fields(batch[0].true)
    }
    files = [sample.file for sample in batch]
    return Batch(files, images, to_targets(**targets))


OT = TypeVar("OT")


def on_batch(
    batch: Batch,
    pipeline: Callable[[HasBoxesAndClasses[torch.Tensor]], OT],
) -> list[OT]:
    if batch.pred is None:
        raise ValueError("Cannot perform inference: batch.pred is empty.")

    return [pipeline(batch.pred[i]) for i, _ in enumerate(batch.files)]
