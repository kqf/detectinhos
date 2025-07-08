import numpy as np
import pytest
import torch

from detectinhos.batch import Batch
from detectinhos.metrics import MeanAveragePrecision
from detectinhos.vanilla import (
    DetectionPredictions,
    DetectionTargets,
    infer_on_batch,
)


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
        true=DetectionTargets(  # type: ignore
            boxes=boxes_true,
            classes=classes_true,
        ),
        pred=DetectionPredictions(  # type: ignore
            boxes=boxes_pred,
            classes=classes_pred,
        ),
    )


# @pytest.mark.xfail
def test_mean_average_precision_add(
    batch: Batch,
    sample_anchors: torch.Tensor,
):
    mapping = {"background": 0, "apple": 1}
    inverse_mapping = {v: k for k, v in mapping.items()}
    map_metric = MeanAveragePrecision(num_classes=2, mapping=mapping)
    map_metric.add(
        *infer_on_batch(
            batch,
            priors=sample_anchors,
            inverse_mapping=inverse_mapping,
        )
    )
    assert map_metric.value()["mAP"] == pytest.approx(0.5)
