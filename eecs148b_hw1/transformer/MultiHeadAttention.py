import torch
import torch.nn as nn
from .attention import scaled_dot_product_attention
from eecs148b_hw1.utils.Linear import Linear

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = Linear(num_heads * self.d_k, d_model)
        self.k_proj = Linear(num_heads * self.d_k, d_model)
        self.v_proj = Linear(num_heads * self.d_v, d_model)
        self.output_proj = Linear(d_model, num_heads * self.d_v)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_v)

        # Attention expects (..., seq_len, d_k); keep heads as a batch-like dimension.
        q = q.transpose(1, 2)  # (batch, heads, seq, d_k)
        k = k.transpose(1, 2)  # (batch, heads, seq, d_k)
        v = v.transpose(1, 2)  # (batch, heads, seq, d_v)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, self.num_heads, seq_len, seq_len)
        
        attn = scaled_dot_product_attention(q, k, v, mask=causal_mask)  # (batch, heads, seq, d_v)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_v)
        
        return self.output_proj(attn)