from __future__ import annotations
import math
import torch

@torch.no_grad()
def profit_metrics_threshold(
    logits: torch.Tensor,   # [B] or [B,1]
    y: torch.Tensor,        # [B] in {0,1}
    price: torch.Tensor,    # [B] in [0,1]
    log_usdc_size: torch.Tensor | None = None,  # optional
    tau: float = 0.0,
    use_size: bool = False,
):
    """
    Offline paper-trading metric for: profit = (y - p) * qty
    Policy: take if q - p > tau, where q = sigmoid(logits).
    If use_size=True, qty = expm1(log_usdc_size) (approx USDC size). Else qty=1.
    """
    q = torch.sigmoid(logits.view(-1))
    y = y.view(-1)
    p = price.view(-1)

    pred_edge = q - p
    take = pred_edge > tau

    if use_size:
        if log_usdc_size is None:
            raise ValueError("use_size=True but log_usdc_size is None")
        qty = torch.expm1(log_usdc_size.view(-1)).clamp_min(0.0)
    else:
        qty = torch.ones_like(p)

    realized = (y - p) * qty

    pnl = realized[take].sum()
    cost = (p * qty)[take].sum()  # capital deployed
    n_take = take.sum()
    n = y.numel()
    hit = (y[take] > 0.5).float().mean() if n_take > 0 else torch.tensor(float("nan"), device=y.device)

    roi = pnl / cost if cost > 0 else torch.tensor(float("nan"), device=y.device)

    return {
        "pnl": float(pnl),
        "cost": float(cost),
        "roi": float(roi),
        "hit_rate": float(hit),
        "take_rate": float(n_take) / max(n, 1),
        "avg_pnl_per_trade_taken": float(pnl / n_take) if n_take > 0 else float("nan"),
    }
