from dataclasses import dataclass
from functools import partial
from operator import itemgetter, methodcaller
from typing import Callable, Generic, Optional, TypeVar

import numpy as np
import torch
from toolz.functoolz import compose
from torch.nn.utils.rnn import pad_sequence

from detectinhos.batch import Batch, BatchElement, apply_eval
from detectinhos.data import Annotation, Sample, load_rgb
from detectinhos.encode import decode as decode_boxes
from detectinhos.sublosses import WeightedLoss

T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
    Optional[WeightedLoss],
)


@dataclass
class DetectionTargets(Generic[T]):
    boxes: T  # [B, N, 4]
    classes: T  # [B, N]
    scores: Optional[T] = None  # [B, N]


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
        """
        Decode predictions to padded DetectionTargets with shape:
        - boxes: [B, N, 4]
        - classes: [B, N]
        - scores: [B, N]
        where N is max number of detections in the batch.
        """
        boxes_list = []
        classes_list = []
        scores_list = []

        B = self.boxes.shape[0]

        for b in range(B):
            logits = self.classes[b]  # [A, C]
            confidence = torch.nn.functional.softmax(logits, dim=-1)
            scores, labels = confidence[:, 1:].max(dim=-1)  # drop bg class
            labels += 1  # shift to match class IDs

            decoded_boxes = decode_boxes(self.boxes[b], anchors, variances)

            mask = scores > confidence_threshold
            boxes_b = decoded_boxes[mask]
            labels_b = labels[mask]
            scores_b = scores[mask]

            boxes_list.append(boxes_b)
            classes_list.append(labels_b)
            scores_list.append(scores_b)

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
    scores = (
        x.scores
        if x.scores is not None
        else torch.ones_like(x.classes[..., 0])
    )
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
                label=label,
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


TRANSFORM_TYPE = Callable[[np.ndarray, T], tuple[np.ndarray, T]]


def do_nothing(x: np.ndarray, y: T) -> tuple[np.ndarray, T]:
    return x, y


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        labels: list[Sample],
        mapping: dict[str, int],
        transform: TRANSFORM_TYPE = do_nothing,
    ) -> None:
        self.mapping = mapping
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> BatchElement[torch.Tensor]:
        sample = self.labels[index]
        image = load_rgb(sample.file_name)
        targets = to_targets(sample, self.mapping)
        image_t, targets_t = self.transform(image, targets)
        return BatchElement(
            file=sample.file_name,
            image=image_t,
            true=targets_t,
        )


def infer_on_rgb(image: np.ndarray, model: torch.nn.Module, file: str = ""):
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
            to_numpy,
            itemgetter(0),
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
) -> tuple[
    list[Sample[Annotation]],
    list[Sample[Annotation]],
]:
    if batch.pred is None:
        raise IOError("First must run the inference")

    return (
        [to_sample(a) for a in to_numpy(batch.true)],  # type: ignore
        [
            to_sample(a)
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
