from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
import torch
from torchvision.ops import nms

from detectinhos.encode import decode
from detectinhos.sample import Annotation, Sample

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    def to_samples(self, batch_id: int, valid: torch.Tensor) -> Sample:
        ...


def pred_to_labels(
    y_pred: HasBoxesAndClasses[torch.Tensor],
    anchors: torch.Tensor,
    variances: tuple[float, float] = (0.1, 0.2),
    nms_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> list[Sample]:
    confidence = torch.nn.functional.softmax(y_pred.classes, dim=-1)
    total: list[Sample] = []
    for batch_id, y_pred_boxes in enumerate(y_pred.boxes):
        boxes_pred = decode(
            y_pred_boxes,
            anchors,
            variances,
        )
        # NB: Convention it's desired to start class_ids from 0,
        # 0 is for background it's not included
        score = confidence[batch_id][:, 1:]

        valid_index = torch.where((score > confidence_threshold).any(-1))[0]

        # NMS doesn't accept fp16 inputs
        boxes_cand = boxes_pred[valid_index].float()
        probs_cand, _ = score[valid_index].float().max(dim=-1)

        # do NMS
        keep = nms(boxes_cand, probs_cand, nms_threshold)
        valid = valid_index[keep]

        total.append(y_pred.to_samples(batch_id, valid))
    return total


def infer(
    image: np.ndarray,
    to_batch: Callable,
    model,
) -> list[Annotation]:
    batch = to_batch(image)
    batch.y_pred = model(batch.image.unsqueeze(0))
    samples = pred_to_labels(batch.y_pred, model.priors)
    return samples[0].annotations
