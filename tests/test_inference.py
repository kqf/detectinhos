from unittest.mock import Mock

import numpy as np
import pytest
import torch

from detectinhos.vanilla import DetectionPredictions, infer_on_rgb


@pytest.fixture
def image():
    return np.random.rand(480, 640, 3).astype(np.float32)


@pytest.fixture
def model():
    mock_model = Mock()
    mock_model.priors = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.2, 0.2, 0.3, 0.3],
            [0.3, 0.3, 0.4, 0.4],
        ]
    )
    mock_model.return_value = DetectionPredictions(
        boxes=torch.tensor(
            [
                [
                    [0.1, 0.1, 0.2, 0.2],
                    [0.2, 0.2, 0.3, 0.3],
                    [0.3, 0.3, 0.4, 0.4],
                ]
            ]
        ),
        classes=torch.tensor(
            [
                [
                    [0.0, 0.8],
                    [0.9, 0.00],
                    [0.0, 0.9],
                ]
            ]
        ),
    )

    return mock_model


def test_infer(image, model):
    sample = infer_on_rgb(image, model)
    assert len(sample.annotations) == 2
