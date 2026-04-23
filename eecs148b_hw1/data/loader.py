import torch
import numpy as np


def get_batch(x, batch_size, context_length, device):
    n = len(x)
    if context_length <= 0:
        raise ValueError("context_length must be > 0")
    if n <= context_length:
        raise ValueError(f"dataset too small (len={n}) for context_length={context_length}")

    starts = np.random.randint(0, n - context_length, size=batch_size)

    stride = x.strides[0]
    windows = np.lib.stride_tricks.as_strided(
        x,
        shape=(n - context_length, context_length + 1),
        strides=(stride, stride),
    )
    batch = windows[starts] 

    inputs = torch.as_tensor(batch[:, :-1].astype(np.int64, copy=False))
    targets = torch.as_tensor(batch[:, 1:].astype(np.int64, copy=False))
    
    return inputs.to(device), targets.to(device)