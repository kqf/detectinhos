from dataclasses import dataclass
from functools import partial
from typing import Callable, Generic, Optional, TypeVar

import numpy as np
import torch
from toolz.functoolz import compose

from detectinhos.batch import Batch, BatchElement, apply_eval, on_batch
from detectinhos.data import Annotation, Sample, load_rgb
from detectinhos.inference import decode
from detectinhos.sublosses import WeightedLoss

T = TypeVar(
    "T",
    np.ndarray,
    torch.Tensor,
    Optional[WeightedLoss],
)


@dataclass
class DetectionTargets(Generic[T]):
    boxes: T
    classes: T
    scores: Optional[T] = None

    def __getitem__(self, idx):
        if isinstance(self.boxes, WeightedLoss) or isinstance(
            self.classes, WeightedLoss
        ):
            raise RuntimeError("You should not call this on losses")

        return DetectionTargets(
            boxes=self.boxes[idx],
            classes=self.classes[idx],
            scores=self.scores[idx] if self.scores is not None else None,
        )


def to_numpy(
    x: DetectionTargets[torch.Tensor],
    file_name: str = "",
) -> DetectionTargets[np.ndarray]:
    # Convert boxes and classes to torch tensors if they aren't already
    boxes = torch.as_tensor(x.boxes)
    classes = torch.as_tensor(x.classes)

    confidence = torch.nn.functional.softmax(classes, dim=-1)
    score = confidence[..., 1:]

    probs_pred, label_pred = score.float().max(dim=-1)

    return DetectionTargets(
        classes=label_pred.cpu().detach().numpy(),
        scores=probs_pred.cpu().detach().numpy(),
        boxes=boxes.cpu().detach().numpy(),
    )


def to_sample(
    predicted: DetectionTargets[np.ndarray],
    file_name: str = "",
) -> Sample:
    predictions = zip(
        predicted.boxes.tolist(),
        predicted.classes.tolist(),
        predicted.scores.tolist() if predicted.scores is not None else [],
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
        partial(
            on_batch,
            pipeline=compose(
                to_sample,
                to_numpy,
                partial(decode, anchors=model.priors),
            ),
        ),
        partial(apply_eval, model=model),
        to_batch,
    )(image)[0]

    sample.file_name = file
    return sample


def infer_on_batch(batch: Batch, priors: torch.Tensor) -> torch.Tensor:
    batch.pred = on_batch(
        batch=batch,
        pipeline=compose(
            to_numpy,
            partial(
                decode,
                anchors=priors,
                variances=[0.1, 0.2],
                confidence_threshold=0.01,
                nms_threshold=2.0,
            ),
        ),
    )  # type: ignore
    return batch
