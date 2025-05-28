from typing import Callable

import numpy as np
import pytest
import torch

from detectinhos.batch import Batch
from detectinhos.metrics import MeanAveragePrecision
from detectinhos.vanilla import DetectionTargets


def batch(true: np.ndarray, pred: np.ndarray) -> Batch:
    return Batch(
        files=["fake.png"],
        image=torch.rand(480, 640, 3, dtype=torch.float32),
        true=DetectionTargets(
            boxes=torch.Tensor(true[:, :4]).unsqueeze(0),
            # We start class_ids from 1
            classes=torch.Tensor(true[:, 4]).unsqueeze(0) + 1,
        ),
        pred=DetectionTargets(
            boxes=torch.Tensor(pred[:, :4]).unsqueeze(0),
            classes=torch.Tensor(pred[:, 4:6]).unsqueeze(0),
        ),
    )


@pytest.fixture
def inference():
    def inference(pred: DetectionTargets) -> torch.Tensor:
        return torch.arange(pred.classes.shape[0])

    return inference


def test_mean_average_precision_add(
    batch: Batch,
    inference: Callable[..., torch.Tensor],
):
    map_metric = MeanAveragePrecision(num_classes=2, inference=inference)
    map_metric.add(batch)
    assert map_metric.value()["mAP"] == pytest.approx(0.5)
