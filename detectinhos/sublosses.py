import functools
from dataclasses import dataclass
from typing import Callable, Union

import torch

LOSS_FUNCTION_TYPE = Union[
    torch.nn.Module,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]


@dataclass
class WeightedLoss:
    loss: LOSS_FUNCTION_TYPE
    weight: float = 1.0
    enc_pred: Callable = lambda x, _: x
    enc_true: Callable = lambda x, _: x
    needs_negatives: bool = False

    def __call__(self, y_pred, y_true, anchors):
        y_pred_encoded = self.enc_pred(y_pred, anchors)
        y_true_encoded = self.enc_true(y_true, anchors)
        return self.weight * self.loss(y_pred_encoded, y_true_encoded)


def masked_loss(loss_function: LOSS_FUNCTION_TYPE) -> LOSS_FUNCTION_TYPE:
    @functools.wraps(loss_function)
    def f(pred: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(data)
        data_masked = data[mask]
        pred_masked = pred[mask]
        loss = loss_function(data_masked, pred_masked)
        if data_masked.numel() == 0:
            loss = torch.nan_to_num(loss, 0)
        return loss / max(data_masked.shape[0], 1)

    return f
