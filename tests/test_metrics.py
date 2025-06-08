from typing import Callable

import numpy as np
import pytest
import torch

from detectinhos.batch import Batch
from detectinhos.encode import decode
from detectinhos.inference import infer_on_batch
from detectinhos.metrics import MeanAveragePrecision
from detectinhos.vanilla import DetectionTargets, to_numpy


@pytest.fixture
def batch(
    image: np.ndarray,
    boxes_true: torch.Tensor,
    boxes_pred: torch.Tensor,
    classes_true: torch.Tensor,
    classes_pred: torch.Tensor,
) -> Batch:
    return Batch(
        files=["fake.png"],
        image=torch.from_numpy(image),
        true=DetectionTargets(
            boxes=boxes_true,
            classes=classes_true,
        ),
        pred=DetectionTargets(
            boxes=boxes_pred,
            classes=classes_pred,
        ),
    )


@pytest.fixture
def inference(pred, sample_anchors):
    n_good_predictions = pred.shape[0]

    def pred_to_labels(pred: DetectionTargets) -> torch.Tensor:
        pred.boxes = decode(pred.boxes, sample_anchors, variances=[0.1, 0.2])
        return torch.arange(n_good_predictions)

    def infer(pred: Batch) -> torch.Tensor:
        return infer_on_batch(
            batch=pred,
            select_valid_indices=pred_to_labels,
            to_sample=to_numpy,  # type: ignore
        )

    return infer


def test_mean_average_precision_add(
    batch: Batch,
    inference: Callable[..., torch.Tensor],
):
    map_metric = MeanAveragePrecision(num_classes=2, inference=inference)
    map_metric.add(batch)
    assert map_metric.value()["mAP"] == pytest.approx(0.5)
