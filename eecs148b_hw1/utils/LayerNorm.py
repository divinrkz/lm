import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        mu = x.mean(dim=-1, keepdim=True)
        x = x.to(torch.float32)
        sigma = x.var(dim=-1, keepdim=True, correction=0)
        x = (x - mu) / (sigma + self.eps).sqrt() * self.weight + self.bias
        
        return x.to(in_dtype)
