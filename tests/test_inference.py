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
def model():
    mock_model = Mock()
    mock_model.priors = torch.rand(1, 10, 4)
    classes = torch.rand(1, 10, 2)
    classes[:, :, 1] = 0.6
    mock_model.return_value = DetectionTargets(
        classes=classes,
        boxes=torch.rand(1, 10, 4),
    )
    return mock_model


def test_infer(image, model):
    def to_batch(image):
        return torch.from_numpy(image).permute(2, 0, 1).float()

    annotations = infer(image, to_batch, model)
    assert len(annotations) == 0
