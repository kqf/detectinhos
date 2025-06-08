from dataclasses import dataclass, fields
from typing import Callable, Generic, List, Optional, Protocol, TypeVar

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from detectinhos.sample import Sample
from detectinhos.vanilla import to_annotations, to_numpy

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    def __getitem__(self, idx) -> "HasBoxesAndClasses":
        ...


# A single element in the batch
@dataclass
class BatchElement(Generic[T]):
    file: str
    image: torch.Tensor
    true: HasBoxesAndClasses[T]


# Stacked BatchElements along batch dimension
@dataclass
class Batch:
    files: list[str]
    image: torch.Tensor
    true: Optional[HasBoxesAndClasses[torch.Tensor]] = None
    pred: Optional[HasBoxesAndClasses[torch.Tensor]] = None

    def pred_to_samples(self, select_valid_indices: Callable) -> list[Sample]:
        if self.pred is None:
            return []

        output = []
        for batch_id, file in enumerate(self.files):
            pred = self.pred[batch_id]
            valid = select_valid_indices(pred)
            output.append(
                Sample(
                    file_name=file,
                    annotations=to_annotations(pred[valid]),
                )
            )
        return output

    def pred_to_numpy(
        self,
        select_valid_indices: Callable,
    ) -> list[HasBoxesAndClasses[np.ndarray]]:
        if self.pred is None:
            return []

        output = []
        for batch_id, _ in enumerate(self.files):
            pred = self.pred[batch_id]
            valid = select_valid_indices(pred)
            output.append(to_numpy(pred[valid]))
        return output


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
