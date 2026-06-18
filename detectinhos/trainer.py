import torch

from detectinhos.sample import Sample


class Trainer:
    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        self.model = model

    def fit(self, samples: list[Sample]):
        print(self.model)
        print("Training the model")
        return self
