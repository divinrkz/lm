import torch
import torch.nn as nn

from .linear import Linear
from eecs148b_hw1.utils.functional import Functional as F

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: str = None, dtype: torch.dtype = None):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.ReLU(x)
        x = self.fc2(x)
        return x