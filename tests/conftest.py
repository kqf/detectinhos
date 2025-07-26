import json
import pathlib

import cv2
import numpy as np
import pytest
import torch

from detectinhos.anchors import anchors
from detectinhos.encode import encode


@pytest.fixture
def true():
    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd] ~
    # Convert to relative coordinates (default resolution is 640x480)
    width, height = 640, 480
    data = np.array(
        [
            [439, 157, 556, 241, 0, 0, 0],
            [437, 246, 518, 351, 0, 0, 0],
            [515, 306, 595, 375, 0, 0, 0],
            [407, 386, 531, 476, 0, 0, 0],
            [544, 419, 621, 476, 0, 0, 0],
            [609, 297, 636, 392, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    data[:, [0, 2]] /= width  # x-coordinates
    data[:, [1, 3]] /= height  # y-coordinates
    return data


@pytest.fixture
def pred():
    # [xmin, ymin, xmax, ymax, class_id, confidence] ~
    # Convert to relative coordinates (default resolution is 640x480)
    width, height = 640, 480
    data = np.array(
        [
            [429, 219, 528, 247, 0, 0.460851],
            [433, 260, 506, 336, 0, 0.269833],
            [518, 314, 603, 369, 0, 0.462608],
            [592, 310, 634, 388, 0, 0.298196],
            [403, 384, 517, 461, 0, 0.382881],
            [405, 429, 519, 470, 0, 0.369369],
            [433, 272, 499, 341, 0, 0.272826],
            [413, 390, 515, 459, 0, 0.619459],
        ],
        dtype=np.float32,
    )
    data[:, [0, 2]] /= width  # x-coordinates
    data[:, [1, 3]] /= height  # y-coordinates
    return data


@pytest.fixture
def annotations(tmp_path, image, true) -> pathlib.Path:
    fname = str(tmp_path / "image.png")
    cv2.imwrite(fname, image)

    # Use true labels to populate the annotations
    example = [
        {
            "file_name": fname,
            "annotations": [
                {
                    "label": "apple",
                    "bbox": true[i][:4].tolist(),
                }
                for i in range(true.shape[0])
            ],
        },
    ]
    ofile = tmp_path / "annotations.json"
    with open(ofile, "w") as f:
        json.dump(example, f, indent=2)
    return ofile


@pytest.fixture
def boxes_true(true) -> torch.Tensor:
    return torch.Tensor(true[:, :4]).unsqueeze(0)


@pytest.fixture
def classes_true(true) -> torch.Tensor:
    return torch.Tensor(true[:, 4]).unsqueeze(0) + 1


@pytest.fixture
def boxes_pred(pred, sample_anchors) -> torch.Tensor:
    total = torch.zeros((sample_anchors.shape[0], 4), dtype=torch.float32)
    total[: pred.shape[0]] = torch.Tensor(pred[:, :4])
    return encode(
        total,
        sample_anchors,
        variances=[0.1, 0.2],
    ).unsqueeze(0)


@pytest.fixture
def classes_pred(pred, sample_anchors) -> torch.Tensor:
    total = torch.zeros((sample_anchors.shape[0], 2), dtype=torch.float32)
    # Everything is background
    total[:, 0] = 1.0
    # Except for the predictions
    total[: pred.shape[0]] = torch.Tensor(pred[:, 4:6])
    total[: pred.shape[0], 0] = 1 - total[: pred.shape[0], 1]
    # Convert probabilities to logits
    eps = 1e-7
    total = torch.clamp(total, eps, 1.0)  # ensure log safety
    logits = torch.log(total)
    return logits.unsqueeze(0)


@pytest.fixture
def image(resolution: tuple[int, int] = (480, 640)) -> np.ndarray:
    return np.random.randint(0, 255, resolution + (3,), dtype=np.uint8)


@pytest.fixture
def sample_anchors(image) -> torch.Tensor:
    return anchors(
        min_sizes=[[16, 32], [64, 128], [256, 512]],
        steps=[8, 16, 32],
        clip=False,
        resolution=image.shape[:2],
    )
