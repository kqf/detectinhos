from functools import partial

import torch

from detectinhos.batch import detection_collate
from detectinhos.sample import read_dataset
from detectinhos.vanilla import DetectionDataset, DetectionTargets


def test_vanilla(annotations):
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
    # sourcery skip: no-loop-in-tests
    for batch in dataloader:
        print(batch.image.shape)
