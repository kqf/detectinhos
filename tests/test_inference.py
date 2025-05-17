from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from detectinhos.inference import infer, pred_to_labels
from detectinhos.vanilla import DetectionTargets


@pytest.fixture
def image():
    return np.random.rand(480, 640, 3).astype(np.float32)


@pytest.fixture
def model(batch_size=1, n_anchors=10, n_classes=2):
    mock_model = Mock()
    mock_model.priors = torch.rand(n_anchors, 4)
    classes = torch.rand(batch_size, n_anchors, n_classes)
    classes[:, :, 1] = 0.6
    mock_model.return_value = DetectionTargets(
        classes=classes,
        boxes=torch.rand(batch_size, n_anchors, 4),
    )
    return mock_model


def test_infer(image, model):
    def to_batch(image):
        return torch.from_numpy(image).permute(2, 0, 1).float()

    annotations = infer(image, to_batch, model)
    assert len(annotations) == 1


def test_pred_to_labels():
    @dataclass
    class DummyPred:
        boxes = torch.tensor(
            [
                [
                    [0.1, 0.1, 0.2, 0.2],
                    [0.2, 0.2, 0.3, 0.3],
                    [0.3, 0.3, 0.4, 0.4],
                ]
            ]
        )
        classes = torch.tensor(
            [
                [
                    [0.1, 2.0],
                    [0.1, 0.05],
                    [0.1, 3.0],
                ]
            ]
        )

    dummy_y_pred = DummyPred()
    dummy_anchors = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.4, 0.4],
        ]
    )
    samples = pred_to_labels(
        dummy_y_pred,
        dummy_anchors,
    )
    assert len(samples) == 3
