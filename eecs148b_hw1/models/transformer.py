import torch
import torch.nn as nn

from eecs148b_hw1.models.embedding import Embedding
from eecs148b_hw1.models.positional_encoding import SinusoidalPositionalEncoding
from eecs148b_hw1.utils.layer_norm import LayerNorm
from eecs148b_hw1.models.linear import Linear
from eecs148b_hw1.models.transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = SinusoidalPositionalEncoding(d_model, context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_final = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        x = x + self.position_embeddings(positions)
       
        for layer in self.layers:
            x = layer(x)
       
        x = self.ln_final(x)
        logits = self.lm_head(x)
       
        return logits