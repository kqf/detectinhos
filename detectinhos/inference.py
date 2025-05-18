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

    @classmethod
    def is_dataclass(cls) -> bool:
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
        print(score)

        valid_index = torch.where((score > confidence_threshold).any(-1))[0]
        print(valid_index, "<")

        # NMS doesn't accept fp16 inputs
        boxes_cand = boxes_pred[valid_index].float()
        probs_cand, _ = score[valid_index].float().max(dim=-1)

        # do NMS
        keep = nms(boxes_cand, probs_cand, nms_threshold)
        valid = valid_index[keep]

        probs_pred, label_pred = score[valid].float().max(dim=-1)
        label_pred_ = label_pred.cpu().detach().numpy()
        probs_pred_ = probs_pred.cpu().detach().numpy()
        boxes_pred_ = boxes_pred[valid].cpu().detach().numpy()
        predictions = zip(
            boxes_pred_.tolist(),
            label_pred_.reshape(-1, 1).tolist(),
            probs_pred_.reshape(-1, 1).tolist(),
        )
        total.append(
            Sample(
                file_name="inference",
                annotations=[
                    Annotation(
                        bbox=box,
                        label=label,
                        score=score,
                    )
                    for box, label, score in predictions
                ],
            )
        )
    return total


def infer(
    image: np.ndarray,
    to_batch: Callable[np.ndarray, torch.Tensor],  # type: ignore
    model,
) -> list[Annotation]:
    y_pred = model(to_batch(image).unsqueeze(0))
    samples = pred_to_labels(y_pred, model.priors)
    return samples[0].annotations
