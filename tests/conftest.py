import json
import pathlib

import cv2
import numpy as np
import pytest
import torch

from detectinhos.anchors import anchors


@pytest.fixture
def annotations(tmp_path) -> pathlib.Path:
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    fname = str(tmp_path / "image.png")
    cv2.imwrite(fname, image)
    h, w, _ = image.shape
    example = [
        {
            "file_name": fname,
            "annotations": [
                {
                    "label": "person",
                    "bbox": [229 / w, 130 / h, 371 / w, 400 / h],
                    "landmarks": [
                        [488.906 / w, 373.643 / h],
                        [542.089 / w, 376.442 / h],
                        [515.031 / w, 412.83 / h],
                        [485.174 / w, 425.893 / h],
                        [538.357 / w, 431.491 / h],
                    ],
                },
                {
                    "label": "person",
                    "bbox": [0.14, 0.5, 0.35, 1.0],
                    "landmarks": [
                        [488.906 / w, 373.643 / h],
                        [542.089 / w, 376.442 / h],
                        [515.031 / w, 412.83 / h],
                        [485.174 / w, 425.893 / h],
                        [538.357 / w, 431.491 / h],
                    ],
                },
            ],
        },
    ]
    ofile = tmp_path / "annotations.json"
    with open(ofile, "w") as f:
        json.dump(example, f, indent=2)
    return ofile


@pytest.fixture
def true():
    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    return np.array(
        [
            [439, 157, 556, 241, 0, 0, 0],
            [437, 246, 518, 351, 0, 0, 0],
            [515, 306, 595, 375, 0, 0, 0],
            [407, 386, 531, 476, 0, 0, 0],
            [544, 419, 621, 476, 0, 0, 0],
            [609, 297, 636, 392, 0, 0, 0],
        ]
    )


@pytest.fixture
def pred():
    # [xmin, ymin, xmax, ymax, class_id, confidence]
    return np.array(
        [
            [429, 219, 528, 247, 0, 0.460851],
            [433, 260, 506, 336, 0, 0.269833],
            [518, 314, 603, 369, 0, 0.462608],
            [592, 310, 634, 388, 0, 0.298196],
            [403, 384, 517, 461, 0, 0.382881],
            [405, 429, 519, 470, 0, 0.369369],
            [433, 272, 499, 341, 0, 0.272826],
            [413, 390, 515, 459, 0, 0.619459],
        ]
    )


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
    return total.unsqueeze(0)
    # return encode(
    #     total,
    #     sample_anchors,
    #     variances=[0.1, 0.2],
    # ).unsqueeze(0)


@pytest.fixture
def classes_pred(pred, sample_anchors) -> torch.Tensor:
    total = torch.zeros((sample_anchors.shape[0], 2), dtype=torch.float32)
    # Everything is background
    total[:, 0] = 1.0
    # Except for the predictions
    total[: pred.shape[0]] = torch.Tensor(pred[:, 4:6])
    return total.unsqueeze(0)


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
