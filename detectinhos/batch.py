from dataclasses import dataclass, fields
from typing import Generic, List, Optional, Sequence, TypeVar

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from detectinhos.tasks.interface import HasBoxesAndClasses

T = TypeVar("T")


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


def apply_eval(
    batch: Batch[T],
    model: torch.nn.Module,
) -> HasBoxesAndClasses[T]:
    original_mode = model.training
    model.eval()
    predicted = model(batch.image)
    model.train(original_mode)
    return predicted


def pad(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return pad_sequence(
        tensors,
        batch_first=True,
        padding_value=float("nan"),
    )


def to_batch(
    x: Sequence[HasBoxesAndClasses[np.ndarray]],
) -> HasBoxesAndClasses[torch.Tensor]:
    if not x:
        raise ValueError("Received an empty sequence of data")

    cls = type(x[0])
    field_names = [f.name for f in fields(x[0])]

    batched = {
        name: pad([torch.Tensor(getattr(t, name)) for t in x])
        for name in field_names
    }
    return cls(**batched)


def detection_collate(
    batch: list[BatchElement],
) -> Batch:
    images = torch.stack([torch.Tensor(sample.image) for sample in batch])
    files = [sample.file for sample in batch]
    return Batch(files, images, to_batch([b.true for b in batch]))


def un_batch(
    x: HasBoxesAndClasses[torch.Tensor],
) -> List[HasBoxesAndClasses[np.ndarray]]:
    cls = type(x)
    fnames = [f.name for f in fields(x)]  # assumes x is a dataclass
    batch_size = x.bbox.shape[0]

    result: List[HasBoxesAndClasses[np.ndarray]] = []
    for i in range(batch_size):
        valid = ~torch.isnan(x.bbox[i]).any(dim=-1)
        sample = {
            name: getattr(x, name)[i][valid].cpu().numpy() for name in fnames
        }
        result.append(cls(**sample))
    return result
