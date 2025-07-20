from functools import partial

import torch

from detectinhos.anchors import anchors
from detectinhos.batch import detection_collate
from detectinhos.dataset import DetectionDataset
from detectinhos.loss import DetectionLoss
from detectinhos.sample import Annotation, Sample, read_dataset
from detectinhos.vanilla import (
    VANILLA_TASK,
    DetectionTargets,
    to_targets,
)


class DedetectionModel(torch.nn.Module):
    def __init__(self, anchors: torch.Tensor, n_clases: int):
        super().__init__()
        self.register_buffer("anchors", anchors)
        self.anchors: torch.Tensor = anchors
        self.n_clases = n_clases

    def forward(self, images: torch.Tensor) -> DetectionTargets:
        batch = images.shape[0]
        num_anchors = self.anchors.shape[0]
        return DetectionTargets(
            classes=torch.rand((batch, num_anchors, self.n_clases)),
            boxes=torch.rand((batch, num_anchors, 4)),
            scores=torch.empty((batch, num_anchors)),
        )


def test_vanilla(annotations, resolution=(480, 640)):
    dataloader = torch.utils.data.DataLoader(
        DetectionDataset(
            labels=read_dataset(annotations, Sample[Annotation]) * 8,
            to_targets=partial(to_targets, mapping={"person": 1}),
        ),
        batch_size=4,
        num_workers=1,
        collate_fn=partial(
            detection_collate,
            to_targets=DetectionTargets,
        ),
    )

    model = DedetectionModel(
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
