import math

from jaxtyping import Float, Bool
from typing import Optional
import torch
from eecs148b_hw1.utils.utils import Functional as F

type Q = Float[Tensor, "batch_size ... seq_len d_k"]
type V = Float[Tensor, "batch_size ... seq_len d_v"]
type Mask = Bool[Tensor, "... seq_len seq_len"]


def masked_fill(tensor: torch.Tensor, mask: Mask, value: float):
    tensor = tensor.clone()
    tensor[~mask] = value
    return tensor

def scaled_dot_product_attention(queries: Q, keys: Q, values: V, mask: Mask | None = None):
   scores = queries @ keys.mT
   norm_scores = scores / math.sqrt(queries.shape[-1])
   if mask is not None: 
    masked_scores = masked_fill(norm_scores, mask, -math.inf)
   weights = F.softmax(masked_scores, dim=-1)
   weights = weights.nan_to_num(0.0)
   return weights @ values