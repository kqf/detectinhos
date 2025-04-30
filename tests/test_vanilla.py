from functools import partial

import torch

from detectinhos.anchors import anchors
from detectinhos.batch import detection_collate
from detectinhos.sample import read_dataset
from detectinhos.vanilla import DetectionDataset, DetectionTargets


class DedetectionModel(torch.nn.Module):
    def __init__(self, anchors: torch.Tensor, n_clases: int):
        super().__init__()
        self.register_buffer("anchors", anchors)
        self.anchors: torch.Tensor = anchors
        self.n_clases = n_clases

    def forward(self, images: torch.Tensor) -> DetectionTargets:
        b_size = images.shape[0]
        num_anchors = self.anchors.shape[0]
        return DetectionTargets(
            classes=torch.rand((b_size, num_anchors, self.n_clases)),
            boxes=torch.rand((b_size, num_anchors, 4)),
        )


def test_vanilla(annotations, resolution=(480, 640)):
    dataloader = torch.utils.data.DataLoader(
        DetectionDataset(
            labels=read_dataset(annotations) * 8,
            mapping={"person": 1},
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

    # sourcery skip: no-loop-in-tests
    for batch in dataloader:
        y_pred: DetectionTargets = model(batch.image)
        print(y_pred)
