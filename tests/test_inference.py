from unittest.mock import Mock

import numpy as np
import pytest
import torch

from detectinhos.inference import infer


@pytest.fixture
def image():
    return np.random.rand(480, 640, 3).astype(np.float32)


@pytest.fixture
def model():
    mock_model = Mock()
    mock_model.priors = Mock()
    mock_model.return_value = torch.rand(1, 10)  # Mock model output
    return mock_model


def test_infer(image, model):
    def to_batch(image):
        return torch.from_numpy(image).permute(2, 0, 1).float()

    annotations = infer(image, to_batch, model)
    assert len(annotations) == 0
