from typing import Callable

import numpy as np
import torch

from detectinhos.sample import Annotation, Sample


def pred_to_labels(y_pred, priors) -> list[Sample]:
    return []


def infer(
    image: np.ndarray,
    to_image: Callable[np.ndarray, torch.Tensor],  # type: ignore
    model,
) -> list[Annotation]:
    y_pred = model(to_image(image).unsqueeze(0))
    samples = pred_to_labels(y_pred, model.priors)
    return samples[0].annotations
