from typing import List, Tuple

import numpy as np
from mean_average_precision import MetricBuilder

from detectinhos.sample import Sample
from detectinhos.tasks.standard import Annotation


# TODO: For some reason this is imortant, fix me
def to_bbox(bbox):
    return bbox[0] * 640, bbox[1] * 480, bbox[2] * 640, bbox[3] * 480


def to_table(
    true: List[Sample[Annotation]],
    pred: List[Sample[Annotation]],
    mapping: dict[str, int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    total = []

    for strue, spred in zip(true, pred):
        # Format predictions
        pred_arr = np.array(
            [
                list(to_bbox(ann.bbox)) + [mapping[ann.label] - 1, ann.score]
                for ann in spred.annotations
            ],
            dtype=np.float32,
        )

        # Format ground truth
        true_arr = np.zeros((len(strue.annotations), 7), dtype=np.float32)
        for i, ann in enumerate(strue.annotations):
            true_arr[i, :4] = to_bbox(ann.bbox)
            # shift to zero-indexed class
            true_arr[i, 4] = mapping[ann.label] - 1

        total.append((pred_arr, true_arr))

    return total


class MeanAveragePrecision:
    def __init__(self, num_classes: int, mapping: dict[str, int]):
        # Convention: skip background class (index 0)
        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d",
            async_mode=False,
            num_classes=num_classes - 1,
        )
        self.mapping = mapping

    def add(self, true: List[Sample], pred: List[Sample]) -> None:
        for pred_arr, true_arr in to_table(true, pred, mapping=self.mapping):
            self.metric_fn.add(pred_arr, true_arr)

    def value(self, iou_thresholds: float = 0.5) -> dict[str, float]:
        return self.metric_fn.value(iou_thresholds=iou_thresholds)

    def reset(self) -> None:
        self.metric_fn.reset()
