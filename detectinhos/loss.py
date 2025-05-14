from dataclasses import fields
from functools import partial
from typing import Callable, Generic, Protocol, Tuple, TypeVar

import torch
from torch import nn

from detectinhos.matching import match
from detectinhos.sublosses import WeightedLoss

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    boxes: T
    classes: T

    @classmethod
    def is_dataclass(cls) -> bool:
        ...


def select(y_pred, y_true, anchors, use_negatives, positives, negatives):
    b, a, o = torch.where(positives)
    pred_pos = y_pred[b, a]
    true_pos = y_true[b, o].view(-1)
    anch_pos = anchors[a]

    if not use_negatives:
        return pred_pos, true_pos, anch_pos

    b, a = torch.where(negatives)
    pred_neg = y_pred[b, a]
    true_neg = torch.zeros_like(pred_neg[:, 0], dtype=torch.long)
    anch_neg = anchors[a]

    pred_all = torch.cat([pred_pos, pred_neg], dim=0)
    true_all = torch.cat([true_pos, true_neg], dim=0).long()
    anch_all = torch.cat([anch_pos, anch_neg], dim=0)
    return pred_all, true_all, anch_all


MATCHING_TYPE = Callable[
    [
        HasBoxesAndClasses[torch.Tensor],
        HasBoxesAndClasses[torch.Tensor],
        torch.Tensor,
    ],
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
            y_pred,
            y_true,
            self.priors,
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
