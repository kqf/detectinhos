import json
import pathlib

import cv2
import numpy as np
import pytest


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
