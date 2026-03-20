import torch
import torch.nn as nn

from eecs148b_hw1.models.attention import MultiHeadAttention
from eecs148b_hw1.utils.layer_norm import LayerNorm
from eecs148b_hw1.models.ffn import FFN


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x