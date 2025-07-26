import numpy as np
import pytest
import torch

from detectinhos.batch import Batch
from detectinhos.metrics import MeanAveragePrecision
from detectinhos.vanilla import (
    DetectionTargets,
    build_inference_on_batch,
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
            scores=classes_true,
        ),
        pred=DetectionTargets(  # type: ignore
            boxes=boxes_pred,
            classes=classes_pred,
            scores=classes_pred,
        ),
    )


# @pytest.mark.xfail
def test_mean_average_precision_add(
    batch: Batch,
    sample_anchors: torch.Tensor,
):
    mapping = {"background": 0, "apple": 1}
    inverse_mapping = {v: k for k, v in mapping.items()}
    infer_on_batch = build_inference_on_batch(
        inverse_mapping=inverse_mapping,
        priors=sample_anchors,
        confidence_threshold=0.01,
        nms_threshold=2.0,
    )
    map_metric = MeanAveragePrecision(num_classes=2, mapping=mapping)
    map_metric.add(*infer_on_batch(batch))
    assert map_metric.value()["mAP"] == pytest.approx(0.5)
