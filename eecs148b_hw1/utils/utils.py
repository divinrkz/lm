import torch


class Functional:
    @staticmethod
    def ReLU(x: torch.Tensor) -> torch.Tensor:
        return x * (x > 0).to(x.dtype)
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        max_x = torch.max(x, dim=dim, keepdim=True).values
        num = torch.exp(x - max_x)   
        denom = num.sum(dim=dim, keepdim=True)
        return num / denom