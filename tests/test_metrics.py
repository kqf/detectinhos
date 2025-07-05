from functools import partial
from typing import Callable

import numpy as np
import pytest
import torch

from detectinhos.batch import Batch
from detectinhos.metrics import MeanAveragePrecision
from detectinhos.vanilla import DetectionTargets, infer_on_batch


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
    return partial(infer_on_batch, priors=sample_anchors)


# @pytest.mark.xfail
def test_mean_average_precision_add(
    batch: Batch,
    inference: Callable[..., torch.Tensor],
):
    map_metric = MeanAveragePrecision(num_classes=2)
    map_metric.add(inference(batch))
    assert map_metric.value()["mAP"] == pytest.approx(0.5)
