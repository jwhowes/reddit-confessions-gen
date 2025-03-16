from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(d_model).normal_(mean=1.0, std=sqrt(1 / d_model)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        super(SwiGLU, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.hidden = nn.Linear(d_model, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )
