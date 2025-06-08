from unittest.mock import Mock

import numpy as np
import pytest
import torch

from detectinhos.batch import Batch
from detectinhos.inference import infer, pred_to_labels
from detectinhos.vanilla import DetectionTargets, to_sample


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
        to_sample,
    )
    assert len(annotations) == 2


def test_pred_to_labels(image, model):
    y_pred = model(image)
    # sourcery skip: no-loop-in-tests
    indices = pred_to_labels(
        y_pred[0],
        model.priors,
    )
    assert indices.detach().cpu().numpy().tolist() == [2, 0]
