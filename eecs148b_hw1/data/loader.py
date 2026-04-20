import torch
import numpy as np


def get_batch(x, batch_size, context_length, device):
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)

    inputs = torch.stack([torch.from_numpy(np.array(x[i:i+context_length]).astype(np.int64)) for i in starts])
    targets = torch.stack([torch.from_numpy(np.array(x[i+1: i+context_length+1]).astype(np.int64)) for i in starts])

    return inputs.to(device), targets.to(device)