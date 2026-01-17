#!/usr/bin/env python
"""
Training script for the TradeTransformer model with per-trade edge prediction.

Uses the sequence dataset with per-token predictions.
Loss: PnL-weighted BCE
Metrics: accuracy, mae_edge, pnl_norm, take_rate, hit_rate for various thresholds
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from polyquant.config import load_paths
from polyquant.data.datasets.sequence_dataset import MarketWindowDataset
from polyquant.models.transformer import create_small_trade_transformer, create_base_trade_transformer
from polyquant.utils import load_checkpoint, save_checkpoint


# ===========================
# HYPERPARAMETERS / PATHS
# ===========================

PATHS = load_paths(__file__)
RUNS_DIR = PATHS.runs_dir
CKPT_ROOT = PATHS.checkpoints_dir

# Dataset paths (relative to project root)
INDEX_PATH = PATHS.root / "data" / "sequences" / "index.parquet"

# DataLoader
BATCH_SIZE = 32
NUM_WORKERS = 4
SEQ_LEN = 1024
CAP_TRADES = 4096
MIN_PREFIX = 20

# Minimum context before computing loss (skip predictions for first N tokens)
MIN_CONTEXT = 50

# Training
MAX_STEPS = 100_000
LOG_EVERY_STEPS = 10
VAL_EVERY_STEPS = 200
VAL_MAX_BATCHES = 50
CHECKPOINT_EVERY_STEPS = 5_000

# Optimizer
LR = 3e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 1000

# Regularization
GRAD_CLIP_NORM = 1.0
AMP_ENABLED = True
DROPOUT = 0.15

# Policy thresholds for profitability metrics
TAUS = [0.00, 0.005, 0.01, 0.02, 0.05]


# ===========================
# DATA LOADERS
# ===========================

def make_loaders(index_path: Path) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    train_ds = MarketWindowDataset(
        index_path=str(index_path),
        split="train",
        L=SEQ_LEN,
        cap_trades=CAP_TRADES,
        min_prefix=MIN_PREFIX,
        pf_cache=64,
    )
    val_ds = MarketWindowDataset(
        index_path=str(index_path),
        split="val",
        L=SEQ_LEN,
        cap_trades=CAP_TRADES,
        min_prefix=MIN_PREFIX,
        pf_cache=32,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    return train_loader, val_loader


# ===========================
# LEARNING RATE SCHEDULE
# ===========================

def get_lr(step: int, warmup_steps: int, max_steps: int, lr_max: float, lr_min: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    decay_steps = max_steps - warmup_steps
    progress = (step - warmup_steps) / max(decay_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ===========================
# LOSS FUNCTIONS
# ===========================

class SequenceBCE:
    """BCE for sequences with masking and min context."""

    def __init__(self, min_context: int = 0):
        self.min_context = min_context

    def __call__(
        self,
        logits: torch.Tensor,  # (B, L)
        y: torch.Tensor,       # (B,) market outcome
        mask: torch.Tensor,    # (B, L) valid tokens
    ) -> torch.Tensor:
        """Compute masked BCE loss, skipping first min_context tokens."""
        B, L = logits.shape
        device = logits.device

        # Expand y to (B, L) - same outcome for all trades in a market
        y_expanded = y.float().unsqueeze(1).expand(B, L)

        # BCE per token
        bce = F.binary_cross_entropy_with_logits(logits, y_expanded, reduction="none")

        # Create context mask: only include positions >= min_context
        context_mask = torch.arange(L, device=device) >= self.min_context  # (L,)
        effective_mask = mask & context_mask.unsqueeze(0)  # (B, L)

        # Apply mask and compute mean
        bce_masked = bce * effective_mask.float()
        return bce_masked.sum() / (effective_mask.float().sum() + 1e-8)


# ===========================
# EVALUATION
# ===========================

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int, min_context: int = 0) -> dict:
    """
    Evaluate on validation set with profitability metrics.

    Args:
        model: The model to evaluate
        loader: Validation data loader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate
        min_context: Skip first N tokens (no predictions without sufficient context)

    Returns:
        dict with bce, misclass, mae_edge, and per-threshold pnl/take_rate/hit_rate
    """
    model.eval()
    loss_fn = SequenceBCE(min_context=min_context)

    # Core metrics accumulators
    total_loss = 0.0
    total_mis = 0.0
    total_mae_edge = 0.0
    total_n = 0

    # Policy metrics
    pnl_sum = {t: 0.0 for t in TAUS}
    take_sum = {t: 0 for t in TAUS}
    win_sum = {t: 0 for t in TAUS}
    tot_trades = 0

    batches = 0
    for x, u, mask, y in loader:
        x = x.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # x[:, :, 0] is p_yes (price)
        price = x[:, :, 0]  # (B, L)

        logits = model(x, u, mask)  # (B, L)
        loss = loss_fn(logits, y, mask)

        probs = torch.sigmoid(logits)  # (B, L)

        # Expand y for comparison
        B, L = logits.shape
        y_expanded = y.float().unsqueeze(1).expand(B, L)

        # Create context mask: skip first min_context tokens
        context_mask = torch.arange(L, device=device) >= min_context  # (L,)
        effective_mask = mask & context_mask.unsqueeze(0)  # (B, L)

        # Only compute metrics on valid tokens with sufficient context
        valid_mask = effective_mask.flatten()
        probs_flat = probs.flatten()[valid_mask]
        y_flat = y_expanded.flatten()[valid_mask]
        price_flat = price.flatten()[valid_mask]

        n = int(valid_mask.sum())
        if n == 0:
            continue

        # Misclassification
        preds = (probs_flat > 0.5).float()
        mis = (preds != y_flat).float().mean()

        # Edge metrics
        pred_edge = probs_flat - price_flat
        true_edge = y_flat - price_flat
        mae_edge = torch.abs(pred_edge - true_edge).mean()

        total_loss += float(loss) * n
        total_mis += float(mis) * n
        total_mae_edge += float(mae_edge) * n
        total_n += n
        tot_trades += n

        # Profitability metrics (policy: take if pred_edge > tau)
        realized = y_flat - price_flat  # profit per share

        for t in TAUS:
            take = pred_edge > t
            if take.any():
                pnl_sum[t] += float(realized[take].sum())
                take_cnt = int(take.sum())
                take_sum[t] += take_cnt
                win_sum[t] += int((y_flat[take] > 0.5).sum())

        batches += 1
        if batches >= max_batches:
            break

    if total_n == 0:
        out = {"bce": float("nan"), "misclass": float("nan"), "mae_edge": float("nan")}
        for t in TAUS:
            key = f"tau_{t:.3f}".replace(".", "p")
            out[f"pnl/{key}"] = float("nan")
            out[f"pnl_norm/{key}"] = float("nan")
            out[f"take_rate/{key}"] = float("nan")
            out[f"hit_rate/{key}"] = float("nan")
        return out

    out = {
        "bce": total_loss / total_n,
        "misclass": total_mis / total_n,
        "mae_edge": total_mae_edge / total_n,
    }

    for t in TAUS:
        key = f"tau_{t:.3f}".replace(".", "p")
        n_take = take_sum[t]
        pnl = pnl_sum[t]

        pnl_norm = (pnl / n_take) if n_take > 0 else float("nan")
        take_rate = (n_take / tot_trades) if tot_trades > 0 else float("nan")
        hit_rate = (win_sum[t] / n_take) if n_take > 0 else float("nan")

        out[f"pnl/{key}"] = pnl
        out[f"pnl_norm/{key}"] = pnl_norm
        out[f"take_rate/{key}"] = take_rate
        out[f"hit_rate/{key}"] = hit_rate

    return out


# ===========================
# MAIN TRAINING
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Train TradeTransformer for edge prediction")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume from",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base"],
        help="Model size: small (~1.7M) or base (~8M)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no GPU was detected.")

    device = torch.device("cuda")
    print(f"[INFO] Using device: {device}")

    # Data
    print(f"[INFO] Loading data from {INDEX_PATH}")
    train_loader, val_loader = make_loaders(INDEX_PATH)
    print(f"[INFO] Train markets: {len(train_loader.dataset)}, Val markets: {len(val_loader.dataset)}")

    # Model
    if args.model_size == "small":
        model = create_small_trade_transformer(
            d_input=10,
            max_seq_len=SEQ_LEN,
            dropout=DROPOUT,
        )
    else:
        model = create_base_trade_transformer(
            d_input=10,
            max_seq_len=SEQ_LEN,
            dropout=DROPOUT,
        )
    model = model.to(device)
    print(f"[INFO] Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    # Loss function (skip first MIN_CONTEXT tokens)
    loss_fn = SequenceBCE(min_context=MIN_CONTEXT)

    # Directories
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    # Resume or new run
    if args.resume:
        ckpt_path = Path(args.resume).resolve()
        ckpt_dir = ckpt_path.parent
        run_name, global_step, epoch = load_checkpoint(
            ckpt_path, model, optimizer, None, scaler, device
        )
    else:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_trade_transformer"
        global_step = 0
        epoch = 0
        ckpt_dir = CKPT_ROOT / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Starting new run: {run_name}")

    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    writer.add_text(
        "run_info",
        f"run_name={run_name}, model_size={args.model_size}, seq_len={SEQ_LEN}, "
        f"batch_size={BATCH_SIZE}, min_context={MIN_CONTEXT}, resumed_from={args.resume or 'fresh'}"
    )

    # Training loop
    model.train()
    t0 = time.time()
    running_loss = 0.0
    running_mis = 0.0
    running_n = 0

    while global_step < MAX_STEPS:
        epoch += 1
        print(f"[EPOCH] {epoch}, starting at step {global_step}")

        for x, u, mask, y in train_loader:
            if global_step >= MAX_STEPS:
                break

            global_step += 1

            # Update learning rate
            lr = get_lr(global_step, WARMUP_STEPS, MAX_STEPS, LR, LR_MIN)
            set_lr(optimizer, lr)

            x = x.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # x[:, :, 0] is p_yes (price)
            price = x[:, :, 0]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(x, u, mask)  # (B, L)
                loss = loss_fn(logits, y, mask)

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

            # Running metrics
            with torch.no_grad():
                B, L = logits.shape
                y_expanded = y.float().unsqueeze(1).expand(B, L)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                valid_mask = mask.flatten()
                preds_flat = preds.flatten()[valid_mask]
                y_flat = y_expanded.flatten()[valid_mask]

                n = int(valid_mask.sum())
                mis = (preds_flat != y_flat).float().mean() if n > 0 else 0.0

            running_loss += float(loss) * n
            running_mis += float(mis) * n
            running_n += n

            # Logging
            if global_step % LOG_EVERY_STEPS == 0 and running_n > 0:
                avg_loss = running_loss / running_n
                avg_mis = running_mis / running_n
                elapsed = time.time() - t0
                steps_per_sec = global_step / max(elapsed, 1e-6)

                writer.add_scalar("train/bce", avg_loss, global_step)
                writer.add_scalar("train/misclass", avg_mis, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step)

                print(
                    f"[STEP {global_step:5d}] "
                    f"loss={avg_loss:.4f} mis={avg_mis:.3f} "
                    f"lr={lr:.2e} ({steps_per_sec:.1f} steps/s)"
                )

                # Reset running metrics
                running_loss = 0.0
                running_mis = 0.0
                running_n = 0

            # Validation
            if global_step % VAL_EVERY_STEPS == 0:
                val_metrics = evaluate(model, val_loader, device, VAL_MAX_BATCHES, MIN_CONTEXT)

                for k, v in val_metrics.items():
                    if isinstance(v, float) and not math.isnan(v):
                        writer.add_scalar(f"val/{k}", v, global_step)

                # Print key metrics
                pnl_01 = val_metrics.get("pnl_norm/tau_0p010", float("nan"))
                take_01 = val_metrics.get("take_rate/tau_0p010", float("nan"))
                print(
                    f"[VAL   {global_step:5d}] "
                    f"bce={val_metrics['bce']:.4f} mis={val_metrics['misclass']:.3f} "
                    f"mae_edge={val_metrics['mae_edge']:.4f} "
                    f"pnl_norm@0.01={pnl_01:.4f} take_rate@0.01={take_01:.3f}"
                )

                model.train()

            # Checkpoint
            if global_step % CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    run_name=run_name,
                    step=global_step,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    scaler=scaler,
                )

    # Final checkpoint
    save_checkpoint(
        ckpt_dir=ckpt_dir,
        run_name=run_name,
        step=global_step,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
    )
    writer.close()
    print(f"[DONE] Finished at step {global_step}, epoch {epoch}")


if __name__ == "__main__":
    main()
