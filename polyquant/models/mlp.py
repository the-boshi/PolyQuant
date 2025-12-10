from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 512, 256, 128), dropout: float = 0.05):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits shape [B]
        return self.net(x).squeeze(-1)
