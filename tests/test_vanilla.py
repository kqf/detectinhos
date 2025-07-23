from functools import partial
from typing import Callable

import numpy as np
import pytest
import torch

from detectinhos.anchors import anchors
from detectinhos.batch import detection_collate
from detectinhos.dataset import DetectionDataset
from detectinhos.loss import DetectionLoss
from detectinhos.sample import Annotation, Sample, read_dataset
from detectinhos.vanilla import (
    VANILLA_TASK,
    DetectionTargets,
    build_inference_on_rgb,
    to_targets,
)


class DedetectionModel(torch.nn.Module):
    anchors: torch.Tensor

    def __init__(
        self,
        anchors: torch.Tensor,
        n_clases: int,
        classes: torch.Tensor,
        boxes: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("anchors", anchors)
        self.anchors: torch.Tensor = anchors
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
    batch_size,
    classes_pred,
    boxes_pred,
) -> Callable[[torch.Tensor, int], DedetectionModel]:
    def build_model(anchors: torch.Tensor, n_clases: int) -> DedetectionModel:
        return DedetectionModel(
            anchors=anchors,
            n_clases=n_clases,
            classes=classes_pred,
            boxes=boxes_pred,
        )

    return build_model


# TODO: Do we need other tests at all?
@pytest.mark.parametrize(
    "batch_size",
    [
        4,
    ],
)
def test_vanilla(batch_size, annotations, build_model, resolution=(480, 640)):
    dataloader = torch.utils.data.DataLoader(
        DetectionDataset(
            labels=read_dataset(annotations, Sample[Annotation]) * 8,
            to_targets=partial(to_targets, mapping={"person": 1}),
        ),
        batch_size=batch_size,
        num_workers=1,
        collate_fn=partial(
            detection_collate,
            to_targets=DetectionTargets,
        ),
    )

    model = build_model(
        anchors=anchors(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            resolution=resolution,
        ),
        n_clases=2,
    )
    loss = DetectionLoss(
        priors=model.anchors,
        sublosses=VANILLA_TASK,
    )

    # sourcery skip: no-loop-in-tests
    for batch in dataloader:
        batch.pred = model(batch.image)
        batch.true.classes = batch.true.classes.long()
        losses = loss(batch.pred, batch.true)
        assert "loss" in losses

    # Now check the inference after training
    infer_on_rgb = build_inference_on_rgb(
        model,
        priors=model.anchors,
        inverse_mapping={0: "background", 1: "apple"},
    )

    sample = infer_on_rgb(np.random.randint(0, 255, resolution + (3,)))
    assert len(sample.annotations) > 0
