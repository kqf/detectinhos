from unittest.mock import Mock

import numpy as np
import pytest
import torch

from detectinhos.inference import infer
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
