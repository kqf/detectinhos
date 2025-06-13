from functools import partial
from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
import torch
from toolz.functoolz import compose
from torchvision.ops import nms

from detectinhos.batch import Batch
from detectinhos.encode import decode as decode_boxes
from detectinhos.sample import Annotation

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    def __getitem__(self, idx) -> "HasBoxesAndClasses":
        ...


def decode(
    y_pred: HasBoxesAndClasses[torch.Tensor],
    filename: str,
    anchors: torch.Tensor,
    variances: tuple[float, float] = (0.1, 0.2),
    nms_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> torch.Tensor:
    confidence = torch.nn.functional.softmax(y_pred.classes, dim=-1)
    boxes_pred = decode_boxes(
        y_pred.boxes,
        anchors,
        variances,
    )
    # NB: Convention it's desired to start class_ids from 0,
    # 0 is for background it's not included
    score = confidence[:, 1:]

    valid_index = torch.where((score > confidence_threshold).any(-1))[0]

    # NMS doesn't accept fp16 inputs
    boxes_cand = boxes_pred[valid_index].float()
    probs_cand, _ = score[valid_index].float().max(dim=-1)

    # do NMS
    keep = nms(boxes_cand, probs_cand, nms_threshold)
    return y_pred[valid_index[keep]]


OT = TypeVar("OT")


def on_batch(
    batch: Batch,
    pipeline: Callable[[HasBoxesAndClasses[torch.Tensor], str], OT],
) -> list[OT]:
    if batch.pred is None:
        raise ValueError("Cannot perform inference: batch.pred is empty.")

    return [
        pipeline(batch.pred[i], file) for i, file in enumerate(batch.files)
    ]


def on_image(
    image: np.ndarray,
    to_batch: Callable,
    model,
    to_sample: Callable = lambda x: x,
) -> list[Annotation]:
    batch = to_batch(image)
    batch.pred = model(batch.image.unsqueeze(0))
    samples = on_batch(
        batch,
        pipeline=compose(
            to_sample,
            partial(
                decode,
                anchors=model.priors,
            ),
        ),
    )
    return samples[0].annotations
