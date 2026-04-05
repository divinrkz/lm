import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1).mean()