from dataclasses import dataclass
from functools import partial
from typing import Generic, Optional, TypeVar

import numpy as np
import torch

from detectinhos.encode import decode as decode_boxes, encode
from detectinhos.inference import generic_infer_on_batch, generic_infer_on_rgb
from detectinhos.loss import decode
from detectinhos.sample import Annotation, Sample
from detectinhos.sublosses import (
    WeightedLoss,
    masked_loss,
    retina_confidence_loss,
)

T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
    WeightedLoss,
)


@dataclass
class DetectionTargets(Generic[T]):
    scores: T  # [B, N]
    boxes: T
    classes: T


def to_numpy(
    x: DetectionTargets[torch.Tensor],
) -> list[DetectionTargets[np.ndarray]]:
    result: list[DetectionTargets[np.ndarray]] = []
    for boxes, classes, scores in zip(x.boxes, x.classes, x.scores):
        valid = ~torch.isnan(boxes).any(dim=-1)  # remove NaN padded rows
        result.append(
            DetectionTargets(
                boxes=boxes[valid].cpu().numpy(),
                classes=classes[valid].cpu().numpy().reshape(-1),
                scores=scores[valid].cpu().numpy().reshape(-1),
            )
        )
    return result


def to_sample(
    predicted: Optional[DetectionTargets[np.ndarray]],
    inverse_mapping: dict[int, str],
    file_name: str = "",
) -> Sample:
    predictions = (
        zip(
            predicted.boxes.tolist(),
            predicted.classes.tolist(),
            predicted.scores.tolist(),
        )
        if predicted
        else []
    )
    return Sample(
        file_name=file_name,
        annotations=[
            Annotation(
                bbox=box,
                label=inverse_mapping[label],
                score=score,
            )
            for box, label, score in predictions
        ],
    )


def to_targets(
    sample: Sample,
    mapping: dict[str, int],
) -> DetectionTargets[np.ndarray]:
    bboxes = []
    label_ids = []
    scores = []

    for label in sample.annotations:
        bboxes.append(label.bbox)

        label_id = mapping.get(label.label, 0)
        label_ids.append([label_id])
        scores.append([label.score])

    return DetectionTargets(
        boxes=np.array(bboxes),
        classes=np.array(label_ids, dtype=np.int64),
        scores=np.array(scores, dtype=np.float32),
    )


TASK = DetectionTargets(
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
)


decode_vanilla = partial(
    decode,
    sublosses=TASK,
)


def build_inference_on_rgb(
    model: torch.nn.Module,
    priors: torch.Tensor,
    inverse_mapping: dict[int, str],
    confidence_threshold=0.5,
    nms_threshold=0.4,
):
    return partial(
        generic_infer_on_rgb,
        model=model,
        priors=priors,
        to_sample=partial(
            to_sample,
            inverse_mapping=inverse_mapping,
        ),
        to_numpy=to_numpy,
        decode=partial(
            decode_vanilla,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        ),
    )


def build_inference_on_batch(
    priors: torch.Tensor,
    inverse_mapping: dict[int, str],
    confidence_threshold=0.5,
    nms_threshold=0.4,
):
    return partial(
        generic_infer_on_batch,
        priors=priors,
        to_numpy=to_numpy,
        to_sample=partial(to_sample, inverse_mapping=inverse_mapping),
        decode=partial(
            decode_vanilla,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        ),
    )
