from collections import defaultdict
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Callable, Generic, TypeVar

import numpy as np
import torch
from dataclasses_json import dataclass_json

from detectinhos.encode import decode as decode_boxes, encode
from detectinhos.sample import RelativeXYXY, Sample
from detectinhos.sublosses import (
    WeightedLoss,
    masked_loss,
    retina_confidence_loss,
)


@dataclass_json
@dataclass
class Annotation:
    bbox: RelativeXYXY
    label: str
    score: float

    @classmethod
    def from_numpy(cls, bbox, label, score, inverse_mapping) -> "Annotation":
        return Annotation(
            bbox=bbox.tolist(),
            label=inverse_mapping[label.item()],
            score=score.item(),
        )

    def to_labels(self, mapping: dict[str, int]) -> dict[str, Any]:
        return dict(
            bbox=list(self.bbox),
            label=mapping[self.label],
            score=self.score,
        )


T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
    WeightedLoss,
)


# TODO: Add the pure vanilla tests
@dataclass
class DetectionTargets(Generic[T]):
    score: T
    bbox: T
    label: T


def to_sample(
    predicted: DetectionTargets[np.ndarray],
    inverse_mapping: dict[int, str],
    file_name: str = "",
    to_annotation: Callable = Annotation.from_numpy,
) -> Sample:
    as_dict = {f.name: getattr(predicted, f.name) for f in fields(predicted)}

    # list of dicts of detections
    predictions = [
        {name: values[i] for name, values in as_dict.items()}
        for i in range(len(next(iter(as_dict.values()))))
    ]

    return Sample(
        file_name=file_name,
        annotations=[
            to_annotation(
                **pred,
                inverse_mapping=inverse_mapping,
            )
            for pred in predictions
        ],
    )


def to_targets(
    sample: Sample,
    mapping: dict[str, int],
    to_targets=DetectionTargets,
) -> DetectionTargets[np.ndarray]:
    fields = defaultdict(list)
    for annotation in sample.annotations:
        for k, v in annotation.to_labels(mapping).items():
            fields[k].append(v)
    targets = {k: np.stack(v) for k, v in fields.items()}
    return DetectionTargets(**targets)


def build_targets(
    mapping: dict[int, str],
) -> tuple[
    Callable[[DetectionTargets[np.ndarray]], Sample],
    Callable[[Sample], DetectionTargets[np.ndarray]],
]:
    inverse_mapping = {v: k for k, v in mapping.items()}
    return (
        partial(to_sample, inverse_mapping=inverse_mapping),
        partial(to_targets, mapping=mapping),
    )


TASK = DetectionTargets(
    score=WeightedLoss(
        loss=None,
        # NB: drop the background class
        dec_pred=lambda logits, _: torch.nn.functional.softmax(logits, dim=-1)[
            ..., 1:
        ].max(dim=-1)[0],
    ),
    label=WeightedLoss(
        loss=retina_confidence_loss,
        weight=2.0,
        enc_pred=lambda x, _: x.reshape(-1, 2),
        # TODO: Should't we convert it on the batch level?
        enc_true=lambda x, _: x.long(),
        # NB: drop the background class, labels += 1
        dec_pred=lambda logits, _: (
            torch.nn.functional.softmax(logits, dim=-1)[..., 1:].max(dim=-1)[1]
            + 1
        ).float(),
        needs_negatives=True,
    ),
    bbox=WeightedLoss(
        loss=masked_loss(torch.nn.SmoothL1Loss()),
        weight=1.0,
        enc_pred=lambda x, _: x,
        enc_true=partial(encode, variances=[0.1, 0.2]),
        dec_pred=partial(decode_boxes, variances=[0.1, 0.2]),
        needs_negatives=False,
    ),
)
