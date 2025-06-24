from typing import Callable, Generic, Protocol, TypeVar

import torch
from torchvision.ops import nms

from detectinhos.batch import Batch
from detectinhos.encode import decode as decode_boxes

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    def __getitem__(self, idx) -> "HasBoxesAndClasses":
        ...


def decode(
    y_pred: HasBoxesAndClasses[torch.Tensor],
    anchors: torch.Tensor,
    variances: tuple[float, float] = (0.1, 0.2),
    nms_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> torch.Tensor:
    confidence = torch.nn.functional.softmax(y_pred.classes, dim=-1)
    # TODO: Fix the mutations of boxes
    y_pred.boxes = decode_boxes(
        y_pred.boxes,
        anchors,
        variances,
    )
    # NB: Convention it's desired to start class_ids from 0,
    # 0 is for background it's not included
    score = confidence[:, 1:]

    valid_index = torch.where((score > confidence_threshold).any(-1))[0]

    # NMS doesn't accept fp16 inputs
    boxes_cand = y_pred.boxes[valid_index].float()
    probs_cand, _ = score[valid_index].float().max(dim=-1)

    # do NMS
    keep = nms(boxes_cand, probs_cand, nms_threshold)
    return y_pred[valid_index[keep]]


OT = TypeVar("OT")


def on_batch(
    batch: Batch,
    pipeline: Callable[[HasBoxesAndClasses[torch.Tensor]], OT],
) -> list[OT]:
    if batch.pred is None:
        raise ValueError("Cannot perform inference: batch.pred is empty.")

    return [pipeline(batch.pred[i]) for i, _ in enumerate(batch.files)]
