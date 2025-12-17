# polyquant/losses/weighted_bce.py
from __future__ import annotations

import torch
import torch.nn as nn


class PnLWeightedBCEWithLogits(nn.Module):
    """
    PnL-weighted BCE for BUY 1 share at price p:
      profit = y - p
      weight w(y,p) = p if y==0 else (1-p)

    Loss:
      E[ w(y,p) * BCE(y, q) ]
    where q = sigmoid(logits)

    Notes:
    - y expected in {0,1} float tensor
    - price expected in [0,1]
    - returns scalar
    """
    def __init__(self, min_weight: float = 1e-3):
        super().__init__()
        self.min_weight = float(min_weight)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, y: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        # shapes: logits [B] or [B,1], y [B], price [B]
        logits = logits.view(-1)
        y = y.view(-1)
        price = price.view(-1)

        # w = y*(1-p) + (1-y)*p
        w = y * (1.0 - price) + (1.0 - y) * price
        w = torch.clamp(w, min=self.min_weight)

        per_ex = self.bce(logits, y)
        return (per_ex * w).sum() / w.sum()
