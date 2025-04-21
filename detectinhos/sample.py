import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dacite import Config, from_dict
from dataclasses_json import dataclass_json

RelativeXYXY = tuple[float, float, float, float]


@dataclass_json
@dataclass
class Annotation:
    bbox: RelativeXYXY
    landmarks: list
    label: str
    score: float = float("nan")


@dataclass_json
@dataclass
class Sample:
    file_name: str
    annotations: list[Annotation]

    def flatten(self) -> tuple:
        return tuple(zip(*[(a.bbox, a.landmarks) for a in self.annotations]))


def to_sample(entry: dict[str, Any]) -> Sample:
    return from_dict(
        data_class=Sample,
        data=entry,
        config=Config(cast=[tuple]),
    )


def remove_invalid_boxes(sample: Sample) -> Sample:
    # Keep only annotations with valid bounding boxes
    cleaned = [
        annotation
        for annotation in sample.annotations
        if annotation.bbox[0] < annotation.bbox[2]
        and annotation.bbox[1] < annotation.bbox[3]
    ]
    sample.annotations = cleaned
    return sample


def read_dataset(
    path: Path | str,
    clean: Callable = remove_invalid_boxes,
) -> list[Sample]:
    with open(path) as f:
        df = json.load(f)
    samples = [clean(to_sample(x)) for x in df]
    return [s for s in samples if s.annotations]
