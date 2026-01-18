"""
Shared evaluation metrics for all training scripts.

All metrics take raw logits (before sigmoid) and compute standardized metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import wandb


@dataclass
class MetricsAccumulator:
    """
    Accumulates batch-level metrics for later aggregation.

    Usage:
        acc = MetricsAccumulator()
        for batch in loader:
            acc.update(logits, y, price)
        metrics = acc.compute()
    """

    # Core metrics
    total_loss: float = 0.0
    total_misclass: float = 0.0
    total_mae_edge: float = 0.0
    total_n: int = 0

    # Baseline metric: misclassification when predicting based on price threshold
    total_baseline_misclass: float = 0.0

    # Track whether price was provided (for edge/profitability metrics)
    _has_price: bool = False

    # For AUC computation
    all_probs: list = field(default_factory=list)
    all_labels: list = field(default_factory=list)

    # Profitability accumulators (for tau=0, i.e., take when pred_edge > 0)
    pnl_sum: float = 0.0
    take_sum: int = 0
    win_sum: int = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        price: torch.Tensor | None = None,
        loss: float | None = None,
        edge: torch.Tensor | None = None,
        store_for_auc: bool = True,
    ):
        """
        Update accumulators with a batch of predictions.

        Args:
            logits: Model outputs before sigmoid, shape [B] or [B, 1]
            y: Ground truth labels {0, 1}, shape [B]
            price: Prices in [0, 1], shape [B]. If None, edge/profitability metrics are skipped.
            loss: Pre-computed loss value (if None, not accumulated)
            edge: True edge = y - price (if None, computed from y and price)
            store_for_auc: Whether to store predictions for AUC computation
        """
        logits = logits.view(-1)
        y = y.view(-1)

        n = logits.size(0)
        probs = torch.sigmoid(logits)

        # Misclassification
        preds = (probs > 0.5).to(y.dtype)
        misclass = (preds != y).float().mean()

        # Accumulate core metrics
        if loss is not None:
            self.total_loss += float(loss) * n
        self.total_misclass += float(misclass) * n
        self.total_n += n

        # Store for AUC
        if store_for_auc:
            self.all_probs.append(probs.detach().cpu())
            self.all_labels.append(y.detach().cpu())

        # Edge and profitability metrics (only if price is provided)
        if price is not None:
            self._has_price = True
            price = price.view(-1)
            pred_edge = probs - price

            # Baseline misclassification: predict success if price > 0.5, failure otherwise
            baseline_preds = (price > 0.5).to(y.dtype)
            baseline_misclass = (baseline_preds != y).float().mean()
            self.total_baseline_misclass += float(baseline_misclass) * n

            # Edge MAE
            if edge is None:
                true_edge = y.float() - price
            else:
                true_edge = edge.view(-1)
            mae_edge = torch.abs(pred_edge - true_edge).mean()
            self.total_mae_edge += float(mae_edge) * n

            # Profitability metrics (tau=0: take when pred_edge > 0)
            realized = y.float() - price  # profit per unit
            take = pred_edge > 0
            if take.any():
                self.pnl_sum += float(realized[take].sum())
                self.take_sum += int(take.sum().item())
                self.win_sum += int((y[take] > 0.5).sum().item())

    def compute(self, prefix: str = "") -> dict[str, float]:
        """
        Compute final metrics from accumulated values.

        Args:
            prefix: Optional prefix for metric keys (e.g., "val/" or "train/")

        Returns:
            Dictionary of metric names to values
        """
        if self.total_n == 0:
            out = {
                f"{prefix}bce": math.nan,
                f"{prefix}misclass": math.nan,
            }
            if self._has_price:
                out[f"{prefix}mae_edge"] = math.nan
                out[f"{prefix}baseline_misclass"] = math.nan
                out[f"{prefix}pnl"] = math.nan
                out[f"{prefix}pnl_norm"] = math.nan
                out[f"{prefix}take_rate"] = math.nan
                out[f"{prefix}hit_rate"] = math.nan
                out[f"{prefix}n_take"] = 0
            return out

        out = {
            f"{prefix}misclass": self.total_misclass / self.total_n,
        }

        # Only include BCE if loss was provided
        if self.total_loss > 0:
            out[f"{prefix}bce"] = self.total_loss / self.total_n

        # Edge and profitability metrics (only if price was provided)
        if self._has_price:
            out[f"{prefix}mae_edge"] = self.total_mae_edge / self.total_n
            out[f"{prefix}baseline_misclass"] = self.total_baseline_misclass / self.total_n

            # Profitability metrics (tau=0)
            pnl = self.pnl_sum
            n_take = self.take_sum

            pnl_norm = (pnl / n_take) if n_take > 0 else math.nan
            take_rate = (n_take / self.total_n) if self.total_n > 0 else math.nan
            hit_rate = (self.win_sum / n_take) if n_take > 0 else math.nan

            out[f"{prefix}pnl"] = pnl
            out[f"{prefix}pnl_norm"] = pnl_norm
            out[f"{prefix}take_rate"] = take_rate
            out[f"{prefix}hit_rate"] = hit_rate
            out[f"{prefix}n_take"] = n_take

        # Compute AUC if possible
        if self.all_probs and self.all_labels:
            try:
                from sklearn.metrics import roc_auc_score
                all_probs = torch.cat(self.all_probs).numpy()
                all_labels = torch.cat(self.all_labels).numpy()
                # Filter out NaN values
                valid_mask = ~(np.isnan(all_probs) | np.isnan(all_labels))
                all_probs = all_probs[valid_mask]
                all_labels = all_labels[valid_mask]
                if len(all_probs) > 0 and len(set(all_labels)) > 1:
                    out[f"{prefix}auc"] = float(roc_auc_score(all_labels, all_probs))
            except ImportError:
                pass

        return out

    def reset(self):
        """Reset all accumulators."""
        self.total_loss = 0.0
        self.total_misclass = 0.0
        self.total_mae_edge = 0.0
        self.total_baseline_misclass = 0.0
        self.total_n = 0
        self._has_price = False
        self.all_probs = []
        self.all_labels = []
        self.pnl_sum = 0.0
        self.take_sum = 0
        self.win_sum = 0


@torch.no_grad()
def compute_batch_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    price: torch.Tensor,
) -> dict[str, float]:
    """
    Compute metrics for a single batch.

    This is a convenience function for cases where you don't need accumulation.

    Args:
        logits: Model outputs before sigmoid, shape [B] or [B, 1]
        y: Ground truth labels {0, 1}, shape [B]
        price: Prices in [0, 1], shape [B]

    Returns:
        Dictionary of metric names to values
    """
    acc = MetricsAccumulator()
    acc.update(logits, y, price, store_for_auc=False)
    return acc.compute()

def log_metrics_to_wandb(
    metrics: dict[str, float],
    step: int,
    prefix: str = "val",
    commit: bool = False,
) -> None:
    """
    Log metrics dictionary to Weights & Biases.

    Filters out NaN values and non-float values before logging.

    Args:
        metrics: Dictionary from MetricsAccumulator.compute()
        step: Global training step
        prefix: Prefix for metric keys (e.g., "val", "train", "test")
        commit: Whether to commit the log (advance wandb step)
    """
    log_dict = {}
    for k, v in metrics.items():
        if v is None:
            continue
        if not isinstance(v, (int, float)):
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        log_dict[f"{prefix}/{k}"] = v

    if log_dict:
        wandb.log(log_dict, step=step, commit=commit)


def log_train_metrics_to_wandb(
    loss: float,
    logits: torch.Tensor,
    y: torch.Tensor,
    price: torch.Tensor,
    edge: torch.Tensor,
    lr: float,
    steps_per_sec: float,
    step: int,
) -> None:
    """
    Log training batch metrics to Weights & Biases.

    Computes misclass and mae_edge from the batch and logs uniform training stats.

    Args:
        loss: Training loss value
        logits: Model outputs before sigmoid
        y: Ground truth labels
        price: Prices
        edge: True edge values
        lr: Current learning rate
        steps_per_sec: Training speed
        step: Global training step
    """
    with torch.no_grad():
        logits = logits.view(-1)
        y = y.view(-1)
        price = price.view(-1)
        edge = edge.view(-1)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(y.dtype)
        misclass = (preds != y).float().mean()

        pred_edge = probs - price
        mae_edge = torch.abs(pred_edge - edge).mean()

    wandb.log({
        "train/bce": loss,
        "train/misclass": float(misclass),
        "train/mae_edge": float(mae_edge),
        "train/lr": lr,
        "train/steps_per_sec": steps_per_sec,
    }, step=step)


def log_train_metrics_simple_to_wandb(
    loss: float,
    logits: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    steps_per_sec: float,
    step: int,
) -> None:
    """
    Log training batch metrics to Weights & Biases (simple version without price/edge).

    Use this for models that don't have price data (e.g., transformers on sequences).

    Args:
        loss: Training loss value
        logits: Model outputs before sigmoid
        y: Ground truth labels
        lr: Current learning rate
        steps_per_sec: Training speed
        step: Global training step
    """
    with torch.no_grad():
        logits = logits.view(-1)
        y = y.view(-1)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(y.dtype)
        misclass = (preds != y).float().mean()

    wandb.log({
        "train/bce": loss,
        "train/misclass": float(misclass),
        "train/lr": lr,
        "train/steps_per_sec": steps_per_sec,
    }, step=step)