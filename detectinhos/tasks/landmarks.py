from dataclasses import dataclass
from functools import partial
from typing import Callable, TypeVar

import numpy as np
import torch
from dataclasses_json import dataclass_json

from detectinhos.encode import decode as decode_boxes, encode
from detectinhos.sample import Sample
from detectinhos.sublosses import (
    WeightedLoss,
    masked_loss,
    retina_confidence_loss,
)
from detectinhos.tasks.standard import Annotation, DetectionTargets


@dataclass_json
@dataclass
class AnnotationWithLandmarks(Annotation):
    landmarks: list[float]


T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
    WeightedLoss,
)


@dataclass
class DetectionTargetsWithLandmarks(DetectionTargets[T]):
    landmarks: T


def to_sample(
    predicted: DetectionTargetsWithLandmarks[np.ndarray],
    inverse_mapping: dict[int, str],
    file_name: str = "",
) -> Sample:
    predictions = zip(
        predicted.boxes.tolist(),
        predicted.classes.reshape(-1).tolist(),
        predicted.scores.reshape(-1).tolist(),
        predicted.landmarks.reshape(-1).tolist(),
    )
    return Sample(
        file_name=file_name,
        annotations=[
            AnnotationWithLandmarks(
                bbox=box,
                label=inverse_mapping[label],
                score=score,
                landmarks=landmarks,
            )
            for box, label, score, landmarks in predictions
        ],
    )


def to_targets(
    sample: Sample[AnnotationWithLandmarks],
    mapping: dict[str, int],
) -> DetectionTargetsWithLandmarks[np.ndarray]:
    bboxes = []
    label_ids = []
    scores = []
    landmarks = []

    for label in sample.annotations:
        bboxes.append(label.bbox)

        label_id = mapping.get(label.label, 0)
        label_ids.append([label_id])
        scores.append([label.score])
        landmarks.append(label.landmarks)

    return DetectionTargetsWithLandmarks(
        boxes=np.array(bboxes),
        classes=np.array(label_ids, dtype=np.int64),
        scores=np.array(scores, dtype=np.float32),
        landmarks=np.array(landmarks),
    )


def build_targets(
    mapping: dict[int, str],
) -> tuple[
    Callable[[DetectionTargetsWithLandmarks[np.ndarray]], Sample],
    Callable[[Sample], DetectionTargetsWithLandmarks[np.ndarray]],
]:
    inverse_mapping = {v: k for k, v in mapping.items()}
    return (
        partial(to_sample, inverse_mapping=inverse_mapping),
        partial(to_targets, mapping=mapping),
    )


TASK = DetectionTargetsWithLandmarks(
    scores=WeightedLoss(
        loss=None,
        # NB: drop the background class
        dec_pred=lambda logits, _: torch.nn.functional.softmax(logits, dim=-1)[
            ..., 1:
        ].max(dim=-1)[0],
    ),
    classes=WeightedLoss(
        loss=retina_confidence_loss,
        weight=2.0,
        enc_pred=lambda x, _: x.reshape(-1, 2),
        enc_true=lambda x, _: x,
        # NB: drop the background class, labels += 1
        dec_pred=lambda logits, _: (
            torch.nn.functional.softmax(logits, dim=-1)[..., 1:].max(dim=-1)[1]
            + 1
        ).float(),
        needs_negatives=True,
    ),
    boxes=WeightedLoss(
        loss=masked_loss(torch.nn.SmoothL1Loss()),
        weight=1.0,
        enc_pred=lambda x, _: x,
        enc_true=partial(encode, variances=[0.1, 0.2]),
        dec_pred=partial(decode_boxes, variances=[0.1, 0.2]),
        needs_negatives=False,
    ),
    # TODO: Make sure the loss is correct
    landmarks=WeightedLoss(
        loss=masked_loss(torch.nn.SmoothL1Loss()),
        weight=1.0,
        enc_pred=lambda x, _: x,
        enc_true=partial(encode, variances=[0.1, 0.2]),
        dec_pred=partial(decode_boxes, variances=[0.1, 0.2]),
        needs_negatives=False,
    ),
)
