from functools import partial
from typing import Callable

import numpy as np
import pytest
import torch

from detectinhos.batch import detection_collate
from detectinhos.dataset import DetectionDataset
from detectinhos.inference import (
    decode as decode_generic,
    infer_on_rgb,
    true2sample,
)
from detectinhos.loss import DetectionLoss
from detectinhos.metrics import MeanAveragePrecision
from detectinhos.sample import Annotation, Sample, read_dataset
from detectinhos.vanilla import (
    TASK,
    DetectionTargets,
    to_sample,
    to_targets,
)


def approx(x):
    return pytest.approx(x, abs=0.001)


class DedetectionModel(torch.nn.Module):
    def __init__(
        self,
        n_clases: int,
        classes: torch.Tensor,
        boxes: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_clases = n_clases
        self.classes = classes
        self.boxes = boxes

    def forward(self, images: torch.Tensor) -> DetectionTargets:
        # Expand classes_pred to shape [batch_size, n_anchors, n_clases]
        batch_size = images.shape[0]
        boxes = self.boxes.expand(batch_size, -1, 4).clone()
        classes = self.classes.expand(batch_size, -1, -1).clone()
        return DetectionTargets(
            # Return the same tensor twice, one for scores another for labels
            scores=classes,
            classes=classes,
            boxes=boxes,
        )


@pytest.fixture
def build_model(
    classes_pred,
    boxes_pred,
) -> Callable[[int], DedetectionModel]:
    def build_model(n_clases: int) -> DedetectionModel:
        return DedetectionModel(
            n_clases=n_clases,
            classes=classes_pred,
            boxes=boxes_pred,
        )

    return build_model


@pytest.mark.parametrize(
    "batch_size",
    [
        4,
    ],
)
@pytest.mark.parametrize(
    "resolution",
    [
        (480, 640),
    ],
)
def test_training_loop(
    batch_size,
    annotations,
    build_model,
    sample_anchors,
    resolution,
):
    mapping = {"background": 0, "apple": 1}
    inverse_mapping = {v: k for k, v in mapping.items()}

    dataloader = torch.utils.data.DataLoader(
        DetectionDataset(
            labels=read_dataset(annotations, Sample[Annotation]) * 8,
            to_targets=partial(to_targets, mapping=mapping),
        ),
        batch_size=batch_size,
        num_workers=1,
        collate_fn=partial(
            detection_collate,
            to_targets=DetectionTargets,
        ),
    )

    model = build_model(
        n_clases=2,
    )
    loss = DetectionLoss(
        priors=sample_anchors,
        sublosses=TASK,
    )
    decode = partial(
        decode_generic,
        sublosses=TASK,
        priors=sample_anchors,
        confidence_threshold=0.2,
        nms_threshold=1.0,
    )
    to_samples = partial(
        true2sample,
        to_sample=partial(
            to_sample,
            inverse_mapping=inverse_mapping,
        ),
    )

    decode_image = partial(
        decode_generic,
        sublosses=TASK,
        priors=sample_anchors,
        confidence_threshold=0.5,
        nms_threshold=0.4,
    )

    to_samples = partial(
        true2sample,
        to_sample=partial(
            to_sample,
            inverse_mapping=inverse_mapping,
        ),
    )

    map_metric = MeanAveragePrecision(num_classes=2, mapping=mapping)
    # sourcery skip: no-loop-in-tests
    for batch in dataloader:
        batch.pred = model(batch.image)
        batch.true.classes = batch.true.classes.long()
        losses = loss(batch.pred, batch.true)
        map_metric.add(
            true=to_samples(batch.true), pred=to_samples(decode(batch.pred))
        )
        # Test 1: Check forward pass and loss
        assert "loss" in losses

    # Test 2: Check mAP metric is calculated correctly
    assert map_metric.value()["mAP"] == pytest.approx(0.5)

    # Now check the inference after training
    infer_on_rgb_vanilla = partial(
        infer_on_rgb,
        model=model,
        to_sample=partial(
            to_sample,
            inverse_mapping=inverse_mapping,
        ),
        decode=decode_image,
    )

    # Test 3: Now check the inference works
    sample = infer_on_rgb_vanilla(np.random.randint(0, 255, resolution + (3,)))
    assert len(sample.annotations) == 1
    assert sample.annotations[0].label == "apple"
    assert sample.annotations[0].score == approx(0.62)
    assert sample.annotations[0].bbox == approx([0.645, 0.813, 0.805, 0.956])
