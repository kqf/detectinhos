from dataclasses import fields
from functools import partial
from typing import Callable, Generic, Protocol, Tuple, TypeVar

import torch
from torch import nn

from detectinhos.sublosses import WeightedLoss

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    @classmethod
    def is_dataclass(cls) -> bool: ...


def match(
    y_pred: HasBoxesAndClasses[torch.Tensor],
    y_true: HasBoxesAndClasses[torch.Tensor],
    anchors: torch.Tensor,
    negpos_ratio: int,
    overalp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # match_local(
    #         y_true.classes,
    #         y_true.boxes,
    #         self.priors,
    #         confidences=y_pred.classes,
    #         negpos_ratio=self.negpos_ratio,
    #         overalp=self.matching_overlap,
    #     )
    return torch.rand(1, 1, 1), torch.rand(1, 1, 1)


def select(y_pred, y_true, anchors, use_negatives, positives, negatives):
    b, a, o = torch.where(positives)
    if not use_negatives:
        return y_pred[b, a], y_true[b, o], anchors[a]

    # TODO: Fix this logic
    conf_pos = y_pred[b, a]
    targets_pos = y_true[b, o].view(-1)
    b, a = torch.where(negatives)
    conf_neg = y_pred[b, a]
    targets_neg = torch.zeros_like(conf_neg[:, 0], dtype=torch.long)
    conf_all = torch.cat([conf_pos, conf_neg], dim=0)
    targets_all = torch.cat([targets_pos, targets_neg], dim=0).long()
    return conf_all, targets_all, anchors[a]


MATCHING_TYPE = Callable[
    [HasBoxesAndClasses[torch.Tensor], HasBoxesAndClasses[torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor],
]


# TODO: Make it generic wrt WeightedLoss
class DetectionLoss(nn.Module):
    def __init__(
        self,
        priors: torch.Tensor,
        sublosses: HasBoxesAndClasses[WeightedLoss],
        match: MATCHING_TYPE = partial(
            match,
            negpos_ratio=7,
            overalp=0.35,
        ),
    ) -> None:
        super().__init__()
        self.sublosses = sublosses
        self.match = match
        self.register_buffer("priors", priors)

    def forward(
        self,
        y_pred: HasBoxesAndClasses[torch.Tensor],
        y_true: HasBoxesAndClasses[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        positives, negatives = self.match(
            y_pred=y_pred,
            y_true=y_true,
        )

        losses = {}
        for field in fields(self.sublosses):
            name = field.name
            subloss: WeightedLoss = getattr(self.sublosses, name)
            y_pred_, y_true_, anchor_ = select(
                getattr(y_pred, name),
                getattr(y_true, name),
                self.priors,
                use_negatives=subloss.needs_negatives,
                positives=positives,
                negatives=negatives,
            )
            losses[name] = subloss(y_pred_, y_true_, anchor_)

        losses["loss"] = torch.stack(tuple(losses.values())).sum()
        return losses
