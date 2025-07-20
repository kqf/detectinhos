from functools import partial
from operator import itemgetter
from typing import Generic, Protocol, TypeVar

import numpy as np
import torch
from toolz.functoolz import compose

from detectinhos.batch import Batch, apply_eval
from detectinhos.sample import Annotation, Sample

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    def __getitem__(self, idx) -> "HasBoxesAndClasses": ...


def generic_infer_on_rgb(
    image: np.ndarray,
    model: torch.nn.Module,
    priors: torch.Tensor,
    to_sample,
    to_numpy,
    decode,
    file: str = "",
):
    def to_batch(image, file="fake.png") -> Batch:
        return Batch(
            files=[file],
            image=torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0),
        )

    # On RGB
    sample = compose(
        compose(
            to_sample,
            itemgetter(0),
            to_numpy,
            partial(decode, priors=priors),
        ),
        partial(apply_eval, model=model),
        to_batch,
    )(image)
    sample.file_name = file
    return sample


def generic_infer_on_batch(
    batch: Batch,
    priors: torch.Tensor,
    to_numpy,
    to_sample,
    decode,
) -> tuple[
    list[Sample[Annotation]],
    list[Sample[Annotation]],
]:
    if batch.pred is None:
        raise IOError("First must run the inference")

    return (
        [to_sample(a) for a in to_numpy(batch.true)],
        [
            to_sample(a)
            for a in to_numpy(
                decode(
                    batch.pred,
                    priors=priors,
                )
            )
        ],
    )
