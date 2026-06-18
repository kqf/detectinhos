import torch

from detectinhos.loss import DetectionLoss
from detectinhos.sample import Sample


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss: DetectionLoss,
        *args,
        **kwargs,
    ):
        self.model = model
        self.loss = loss

    def fit(self, samples: list[Sample]):
        print(self.model)
        print("Training the model")
        return self
