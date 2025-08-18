from dataclasses import fields
from functools import partial
from operator import itemgetter
from typing import Generic, Protocol, TypeVar

import numpy as np
import torch
from toolz.functoolz import compose
from torchvision.ops import nms

from detectinhos.batch import Batch, apply_eval, pad, un_batch
from detectinhos.sample import Sample
from detectinhos.sublosses import WeightedLoss
from detectinhos.tasks.standard import Annotation

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    score: T
    bbox: T
    label: T


def decode(
    pred: HasBoxesAndClasses[torch.Tensor],
    sublosses: HasBoxesAndClasses[WeightedLoss],
    priors: torch.Tensor,
    nms_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> HasBoxesAndClasses[torch.Tensor]:
    n_batches = pred.bbox.shape[0]
    decoded_fields: dict[str, list[torch.Tensor]] = {
        f.name: [] for f in fields(sublosses)
    }

    for b in range(n_batches):
        # Decode boxes and scores
        boxes = sublosses.bbox.dec_pred(pred.bbox[b], priors)
        scores = sublosses.score.dec_pred(pred.score[b], priors)

        mask = scores > confidence_threshold
        keep = nms(boxes[mask], scores[mask], iou_threshold=nms_threshold)

        for field in fields(sublosses):
            name = field.name
            subloss = getattr(sublosses, name)
            raw = getattr(pred, name)[b]
            decoded = subloss.dec_pred(raw, priors)
            filtered = decoded[mask][keep]
            decoded_fields[name].append(filtered)

    output_cls = type(pred)
    return output_cls(**{k: pad(v) for k, v in decoded_fields.items()})


def infer_on_rgb(
    image: np.ndarray,
    model: torch.nn.Module,
    to_sample,
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
            un_batch,
            decode,
        ),
        partial(apply_eval, model=model),
        to_batch,
    )(image)
    sample.file_name = file
    return sample


def true2sample(
    true: HasBoxesAndClasses,
    to_sample,
) -> list[Sample[Annotation]]:
    return list(map(to_sample, un_batch(true)))
