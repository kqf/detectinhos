from unittest.mock import Mock

import numpy as np
import pytest
import torch
from toolz.functoolz import compose

from detectinhos.batch import Batch
from detectinhos.inference import infer
from detectinhos.vanilla import DetectionTargets, to_numpy, to_sample


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
    mock_model.return_value = DetectionTargets(
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
    def to_batch(image) -> Batch:
        return Batch(
            files=["fake.png"],
            image=torch.from_numpy(image).permute(2, 0, 1).float(),
        )

    annotations = infer(
        image,
        to_batch,
        model,
        compose(to_sample, to_numpy),
    )
    assert len(annotations) == 2
