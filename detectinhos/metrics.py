from typing import Callable

import numpy as np
from mean_average_precision import MetricBuilder

from detectinhos.batch import Batch


def prepare_outputs(
    batch: Batch,
    inference: Callable,
) -> list[tuple[np.ndarray, np.ndarray]]:
    total = []
    for true_, pred_ in zip(
        batch.true,
        batch.pred_to_numpy(inference),
    ):  # type: ignore
        pred_sample = np.concatenate(
            (
                pred_.boxes,
                pred_.classes,
                pred_.scores,
            ),
            axis=1,
        )

        true_sample = np.zeros((true_.shape[0], 7), dtype=np.float32)
        true_sample[:, :4] = true_.boxes
        # While calculating mAP, always start with 0
        # We don't calculate the metrics for background class
        true_sample[:, 4] = true_.score - 1
        total.append((pred_sample, true_sample))
    return total


class MeanAveragePrecision:
    def __init__(
        self,
        num_classes: int,
        inference: Callable,
    ):
        # NB: Convention, while calculating mAP, always start with 0
        # We don't calculate the metrics for background class
        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d",
            async_mode=False,
            num_classes=num_classes - 1,
        )
        self.inference = inference

    def add(self, batch: Batch) -> None:
        outputs = prepare_outputs(
            batch=batch,
            inference=self.inference,
        )
        for perimage in outputs:
            self.metric_fn.add(*perimage)
