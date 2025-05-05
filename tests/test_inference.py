from unittest.mock import Mock

import numpy as np
import pytest
import torch

from detectinhos.inference import infer
from detectinhos.sample import Annotation


@pytest.fixture
def image():
    return np.random.rand(480, 640, 3).astype(np.float32)


@pytest.fixture
def to_image():
    return Mock(return_value=torch.rand(3, 480, 640))


@pytest.fixture
def model():
    mock_model = Mock()
    mock_model.priors = Mock()
    mock_model.return_value = torch.rand(1, 10)  # Mock model output
    return mock_model


@pytest.fixture
def mock_pred_to_labels():
    mock_annotations = [Annotation(label="test", bbox=(0, 0, 1, 1))]
    mock_samples = [Mock(annotations=mock_annotations)]
    mock_pred_to_labels = Mock()
    mock_pred_to_labels.return_value = mock_samples
    return mock_pred_to_labels


def test_infer(image, to_image, model, mock_pred_to_labels):
    annotations = infer(image, to_image, model)
    assert len(annotations) == 1
    assert annotations[0].label == "test"
