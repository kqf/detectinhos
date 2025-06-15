from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from toolz.functoolz import compose

from detectinhos.batch import Batch, apply_eval
from detectinhos.inference import decode, on_batch
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


def infer_on_rgb(image: np.ndarray, model: torch.nn.Module, file: str = ""):
    def to_batch(image, file="fake.png") -> Batch:
        return Batch(
            files=[file],
            image=torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0),
        )

    # On RGB
    sample = compose(
        partial(
            on_batch,
            pipeline=compose(
                to_sample,
                to_numpy,
                partial(decode, anchors=model.priors),
            ),
        ),
        partial(apply_eval, model=model),
        to_batch,
    )(image)[0]

    sample.file_name = file
    return sample


def test_infer(image, model):
    sample = infer_on_rgb(image, model)
    assert len(sample.annotations) == 2
