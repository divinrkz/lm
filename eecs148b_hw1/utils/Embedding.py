import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self._init_weights()

    def _init_weights(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + (self.bias if self.bias is not None else 0)
