from typing import Generic, Protocol, TypeVar

import numpy as np
from mean_average_precision import MetricBuilder

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T
    scores: T


class Batch(Protocol):
    true: list[HasBoxesAndClasses[np.ndarray]]
    pred: list[HasBoxesAndClasses[np.ndarray]]


def to_table(batch: Batch) -> list[tuple[np.ndarray, np.ndarray]]:
    total = []
    for true_, pred_ in zip(batch.true, batch.pred):
        pred_sample = np.concatenate(
            (
                pred_.boxes,
                pred_.classes.reshape(-1, 1),
                pred_.scores.reshape(-1, 1),
            ),
            axis=1,
        )
        true_sample = np.zeros((true_.classes.shape[0], 7), dtype=np.float32)
        true_sample[:, :4] = true_.boxes
        # While calculating mAP, always start with 0
        # We don't calculate the metrics for background class
        true_sample[:, 4] = true_.classes - 1
        total.append((pred_sample, true_sample))
    return total


class MeanAveragePrecision:
    def __init__(
        self,
        num_classes: int,
    ):
        # NB: Convention, while calculating mAP, always start with 0
        # We don't calculate the metrics for background class
        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d",
            async_mode=False,
            num_classes=num_classes - 1,
        )

    def add(self, batch: Batch) -> None:
        outputs = to_table(batch=batch)
        for perimage in outputs:
            self.metric_fn.add(*perimage)

    def value(self, iou_thresholds: float = 0.5) -> dict[str, float]:
        return self.metric_fn.value(iou_thresholds=0.5)
