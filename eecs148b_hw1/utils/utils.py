import torch


class Functional:
    @staticmethod
    def ReLU(x: torch.Tensor) -> torch.Tensor:
        return x * (x > 0).to(x.dtype)