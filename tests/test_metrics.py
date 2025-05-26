import numpy as np
import pytest
import torch

from detectinhos.batch import Batch
from detectinhos.metrics import MeanAveragePrecision
from detectinhos.vanilla import DetectionTargets


@pytest.fixture
def true():
    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    return np.array(
        [
            [439, 157, 556, 241, 0, 0, 0],
            [437, 246, 518, 351, 0, 0, 0],
            [515, 306, 595, 375, 0, 0, 0],
            [407, 386, 531, 476, 0, 0, 0],
            [544, 419, 621, 476, 0, 0, 0],
            [609, 297, 636, 392, 0, 0, 0],
        ]
    )


@pytest.fixture
def pred():
    # [xmin, ymin, xmax, ymax, class_id, confidence]
    return np.array(
        [
            [429, 219, 528, 247, 0, 0.460851],
            [433, 260, 506, 336, 0, 0.269833],
            [518, 314, 603, 369, 0, 0.462608],
            [592, 310, 634, 388, 0, 0.298196],
            [403, 384, 517, 461, 0, 0.382881],
            [405, 429, 519, 470, 0, 0.369369],
            [433, 272, 499, 341, 0, 0.272826],
            [413, 390, 515, 459, 0, 0.619459],
        ]
    )


@pytest.fixture
def batch(true, pred) -> Batch:
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


def test_mean_average_precision_add(batch, inference):
    map_metric = MeanAveragePrecision(num_classes=2, inference=inference)
    map_metric.add(batch)
    assert map_metric.value()["mAP"] == pytest.approx(0.5)
