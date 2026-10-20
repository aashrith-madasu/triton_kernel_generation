import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Transformer-style MLP with a pointwise-heavy tail."""

    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.scale = nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, x):
        # h = F.gelu(F.gelu(F.gelu(self.fc1(x))))         # gelu decomposes into ~5 pointwise ops
        h = x
        for _ in range(10):
            h = F.gelu(h)
            
        return x + torch.tanh(h) * self.scale   # residual + gated scale