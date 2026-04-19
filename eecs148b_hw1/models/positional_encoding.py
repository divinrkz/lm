import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, device: str = None, dtype: torch.dtype = None):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model, device=device, dtype=dtype)
        position = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)
        i = torch.arange(0, d_model // 2, device=device, dtype=torch.float32)
        div_term = torch.exp(2 * i * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        return self.pe[token_positions]
