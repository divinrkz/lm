import torch


class Functional:
    @staticmethod
    def ReLU(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        max_x = torch.max(x, dim=dim, keepdim=True).values
        num = torch.exp(x - max_x)   
        denom = num.sum(dim=dim, keepdim=True)
        return num / denom