from dataclasses import fields
from functools import partial
from operator import itemgetter
from typing import Generic, Protocol, TypeVar

import numpy as np
import torch
from toolz.functoolz import compose
from torch.nn.utils.rnn import pad_sequence
from torchvision.ops import nms

from detectinhos.batch import Batch, apply_eval, un_batch
from detectinhos.sample import Annotation, Sample
from detectinhos.sublosses import WeightedLoss

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    scores: T
    boxes: T
    classes: T

    def __getitem__(self, idx) -> "HasBoxesAndClasses": ...


def pad(sequence):
    return pad_sequence(
        sequence,
        batch_first=True,
        padding_value=float("nan"),
    )  # [B, N, 4]


def decode(
    pred: HasBoxesAndClasses[torch.Tensor],
    sublosses: HasBoxesAndClasses[WeightedLoss],
    priors: torch.Tensor,
    nms_threshold: float = 0.4,
    confidence_threshold: float = 0.5,
) -> HasBoxesAndClasses[torch.Tensor]:
    n_batches = pred.boxes.shape[0]
    decoded_fields: dict[str, list[torch.Tensor]] = {
        f.name: [] for f in fields(sublosses)
    }

    for b in range(n_batches):
        # Decode boxes and scores
        boxes = sublosses.boxes.dec_pred(pred.boxes[b], priors)
        scores = sublosses.scores.dec_pred(pred.scores[b], priors)

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


def generic_infer_on_rgb(
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
