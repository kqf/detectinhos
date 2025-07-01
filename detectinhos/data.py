import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import cv2
import numpy as np
from dacite import Config, from_dict
from dataclasses_json import dataclass_json

RelativeXYXY = tuple[float, float, float, float]


@dataclass_json
@dataclass
class Annotation:
    bbox: RelativeXYXY
    label: str
    score: float = float("nan")


T = TypeVar("T", bound=Annotation)


@dataclass
class Sample(Generic[T]):
    file_name: str
    annotations: list[T]


def to_sample(
    entry: dict[str, Any],
    sample_type: type[Sample[T]],
) -> Sample[T]:
    return from_dict(
        data_class=sample_type,
        data=entry,
        config=Config(cast=[tuple]),
    )


def read_dataset(
    path: Path | str,
    sample_type: type[Sample[T]] = Sample[T],
) -> list[Sample[T]]:
    with open(path) as f:
        df = json.load(f)
    samples = [to_sample(x, sample_type) for x in df]
    return [s for s in samples if s.annotations]


def load_rgb(image_path: Path | str) -> np.ndarray:
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
