from dataclasses import dataclass
from functools import partial
from operator import itemgetter, methodcaller
from typing import Generic, Optional, TypeVar

import numpy as np
import torch
from toolz.functoolz import compose
from torch.nn.utils.rnn import pad_sequence
from torchvision.ops import nms

from detectinhos.batch import Batch, apply_eval
from detectinhos.encode import decode as decode_boxes
from detectinhos.sample import Annotation, Sample
from detectinhos.sublosses import WeightedLoss

T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
    Optional[WeightedLoss],
)


@dataclass
class DetectionTargets(Generic[T]):
    scores: T  # [B, N]
    boxes: T
    classes: T


P = TypeVar(
    "P",
    np.ndarray,
    torch.Tensor,
)


def pad(sequence):
    return pad_sequence(
        sequence,
        batch_first=True,
        padding_value=float("nan"),
    )  # [B, N, 4]


@dataclass
class DetectionPredictions(DetectionTargets[P]):
    def decode(
        self,
        anchors: torch.Tensor,
        variances: tuple[float, float] = (0.1, 0.2),
        nms_threshold: float = 0.4,
        confidence_threshold: float = 0.5,
    ) -> DetectionTargets[torch.Tensor]:
        boxes_list = []
        classes_list = []
        scores_list = []

        B = self.boxes.shape[0]

        for b in range(B):
            logits = self.classes[b]  # [A, C]
            confidence = torch.nn.functional.softmax(logits, dim=-1)
            scores, labels = confidence[..., 1:].max(dim=-1)  # drop bg class
            labels += 1  # shift to match class IDs

            decoded_boxes = decode_boxes(self.boxes[b], anchors, variances)

            mask = scores > confidence_threshold
            boxes_b = decoded_boxes[mask]
            labels_b = labels[mask]
            scores_b = scores[mask]

            # Apply class-agnostic NMS
            keep = nms(boxes_b, scores_b, nms_threshold)

            boxes_list.append(boxes_b[keep])
            classes_list.append(labels_b[keep].float())
            scores_list.append(scores_b[keep])

        return DetectionTargets(
            boxes=pad(boxes_list),  # [B, N, 4]
            classes=pad(classes_list),  # [B, N]
            scores=pad(scores_list),  # [B, N]
        )


def to_numpy(
    x: Optional[DetectionTargets[torch.Tensor]],
) -> list[DetectionTargets[np.ndarray]]:
    result: list[DetectionTargets[np.ndarray]] = []
    if x is None:
        return result
    scores = x.scores if x.scores is not None else torch.ones_like(x.classes)
    for boxes, classes, scores in zip(x.boxes, x.classes, scores):
        valid = ~torch.isnan(boxes).any(dim=-1)  # remove NaN padded rows
        result.append(
            DetectionTargets(
                boxes=boxes[valid].cpu().numpy(),
                classes=classes[valid].cpu().numpy(),
                scores=scores[valid].cpu().numpy(),
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
            predicted.scores.tolist() if predicted.scores is not None else [],
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


def infer_on_rgb(
    image: np.ndarray,
    model: torch.nn.Module,
    inverse_mapping: dict[int, str],
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
            partial(to_sample, inverse_mapping=inverse_mapping),
            itemgetter(0),
            to_numpy,
            methodcaller(
                "decode",
                anchors=model.priors,
            ),
        ),
        partial(apply_eval, model=model),
        to_batch,
    )(image)
    sample.file_name = file
    return sample


def infer_on_batch(
    batch: Batch,
    priors: torch.Tensor,
    inverse_mapping: dict[int, str],
) -> tuple[
    list[Sample[Annotation]],
    list[Sample[Annotation]],
]:
    if batch.pred is None:
        raise IOError("First must run the inference")

    return (
        [
            to_sample(a, inverse_mapping=inverse_mapping)
            for a in to_numpy(batch.true)  # type: ignore
        ],
        [
            to_sample(a, inverse_mapping=inverse_mapping)
            for a in to_numpy(
                batch.pred.decode(  # type: ignore
                    priors,
                    variances=[0.1, 0.2],
                    confidence_threshold=0.01,
                    nms_threshold=2.0,
                )
            )
        ],
    )
