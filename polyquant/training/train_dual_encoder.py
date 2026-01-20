#!/usr/bin/env python
"""
Training script for the DualEncoderTransformer model.

Uses the DualSequenceDataset with market and user sequences.
Loss: BCE on final trade outcome
Metrics: accuracy, AUC-ROC, PnL-based metrics
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from polyquant.config import load_paths
from polyquant.data.datasets.dual_sequence_dataset import DualSequenceDataset
from polyquant.metrics import MetricsAccumulator, log_metrics_to_wandb
from polyquant.models.dual_encoder import (
    create_base_dual_encoder,
    create_small_dual_encoder,
)
from polyquant.utils import load_checkpoint, save_checkpoint


# ===========================
# HYPERPARAMETERS / PATHS
# ===========================

PATHS = load_paths(__file__)
RUNS_DIR = PATHS.runs_dir
CKPT_ROOT = PATHS.checkpoints_dir

# Dataset paths (relative to project root)
MARKET_INDEX_PATH = PATHS.root / "data" / "sequences" / "index.parquet"
USER_INDEX_PATH = PATHS.root / "data" / "user_sequences_store" / "index.parquet"

# DataLoader
BATCH_SIZE = 512
NUM_WORKERS = 8
L_MARKET = 1024
L_USER = 128
CAP_TRADES = 4096
MIN_PREFIX = 20

# Training
MAX_STEPS = 100_000
LOG_EVERY_EPOCH_FRACTION = 0.05  # Log every 5% of an epoch
VAL_EVERY_STEPS = 100
VAL_MAX_BATCHES = 100
CHECKPOINT_EVERY_STEPS = 500

# Optimizer
LR = 3e-6
LR_MIN = 1e-8
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 2000

# Regularization
GRAD_CLIP_NORM = 1.0
AMP_ENABLED = True
DROPOUT = 0.1


# ===========================
# DATA LOADERS
# ===========================


def make_loaders(
    market_index_path: Path,
    user_index_path: Path,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    train_ds = DualSequenceDataset(
        market_index_path=str(market_index_path),
        user_index_path=str(user_index_path),
        split="train",
        L_market=L_MARKET,
        L_user=L_USER,
        cap_trades=CAP_TRADES,
        min_prefix=MIN_PREFIX,
        pf_cache=64,
    )
    val_ds = DualSequenceDataset(
        market_index_path=str(market_index_path),
        user_index_path=str(user_index_path),
        split="val",
        L_market=L_MARKET,
        L_user=L_USER,
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
# BATCH PROCESSING
# ===========================


def get_target_from_batch(
    market_y: torch.Tensor,  # (B, L_market)
    market_mask: torch.Tensor,  # (B, L_market)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the target (last valid token's y value) from the batch.

    Returns:
        y: (B,) - target labels (binary)
        valid: (B,) - True for samples with known outcome (y != 0.5)
    """
    B = market_y.shape[0]
    device = market_y.device

    # Get last valid token index
    last_idx = market_mask.sum(dim=1) - 1  # (B,)
    last_idx = last_idx.clamp(min=0)

    # Extract y at last position
    batch_idx = torch.arange(B, device=device)
    y = market_y[batch_idx, last_idx]  # (B,)

    # Valid if y is not 0.5 (resolved)
    valid = y != 0.5

    # Convert y to binary (0.5 -> 0 for loss computation, but masked out)
    y_binary = (y > 0.5).float()

    return y_binary, valid


def get_price_from_batch(
    market_x: torch.Tensor,  # (B, L_market, D_market)
    market_mask: torch.Tensor,  # (B, L_market)
) -> torch.Tensor:
    """Extract price of the target trade (last valid token)."""
    B = market_x.shape[0]
    device = market_x.device

    last_idx = market_mask.sum(dim=1) - 1
    last_idx = last_idx.clamp(min=0)

    batch_idx = torch.arange(B, device=device)
    # price is first column in market_x
    price = market_x[batch_idx, last_idx, 0]  # (B,)

    return price


def get_outcome_index_from_batch(
    market_x: torch.Tensor,  # (B, L_market, D_market)
    market_mask: torch.Tensor,  # (B, L_market)
) -> torch.Tensor:
    """Extract outcome_index of the target trade (last valid token).

    outcome_index is at index 2 in market_x features:
    [price, p_yes, outcome_index, dp_yes_clip, ...]
    """
    B = market_x.shape[0]
    device = market_x.device

    last_idx = market_mask.sum(dim=1) - 1
    last_idx = last_idx.clamp(min=0)

    batch_idx = torch.arange(B, device=device)
    outcome_index = market_x[batch_idx, last_idx, 2]  # (B,)

    return outcome_index


# ===========================
# EVALUATION
# ===========================


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> dict:
    """
    Evaluate on validation set using MetricsAccumulator.

    Returns:
        dict with bce, misclass, mae_edge, pnl metrics, auc
    """
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = MetricsAccumulator()

    batches = 0
    for batch in loader:
        market_x, market_mask, market_y, user_x, user_mask, user_y = batch

        market_x = market_x.to(device, non_blocking=True)
        market_mask = market_mask.to(device, non_blocking=True)
        market_y = market_y.to(device, non_blocking=True)
        user_x = user_x.to(device, non_blocking=True)
        user_mask = user_mask.to(device, non_blocking=True)
        user_y = user_y.to(device, non_blocking=True)

        # Concatenate user_y as 4th feature for user encoder
        user_x_with_y = torch.cat([user_x, user_y.unsqueeze(-1)], dim=-1)

        # Get target, validity, price, and outcome_index
        y, valid = get_target_from_batch(market_y, market_mask)
        price = get_price_from_batch(market_x, market_mask)
        outcome_index = get_outcome_index_from_batch(market_x, market_mask)

        if valid.sum() == 0:
            batches += 1
            if batches >= max_batches:
                break
            continue

        # Forward pass
        logits = model(market_x, market_mask, user_x_with_y, user_mask)

        # Filter to valid samples only
        valid_idx = valid.nonzero(as_tuple=True)[0]
        logits_valid = logits[valid_idx]
        y_valid = y[valid_idx]
        price_valid = price[valid_idx]
        outcome_index_valid = outcome_index[valid_idx]

        loss = loss_fn(logits_valid, y_valid)

        metrics.update(
            logits_valid,
            y_valid,
            price_valid,
            loss=float(loss),
            outcome_index=outcome_index_valid,
        )

        batches += 1
        if batches >= max_batches:
            break

    return metrics.compute()


# ===========================
# MAIN TRAINING
# ===========================


def main():
    parser = argparse.ArgumentParser(description="Train DualEncoderTransformer")
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
        help="Model size: small (~3M) or base (~12M)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no GPU was detected.")

    device = torch.device("cuda")
    print(f"[INFO] Using device: {device}")

    # Data
    print("[DEBUG] Loading data...")
    print(f"       Market index: {MARKET_INDEX_PATH}")
    print(f"       User index: {USER_INDEX_PATH}")
    train_loader, val_loader = make_loaders(MARKET_INDEX_PATH, USER_INDEX_PATH)
    print(
        f"[INFO] Train markets: {len(train_loader.dataset)}, Val markets: {len(val_loader.dataset)}"
    )

    # Model
    print("[DEBUG] Creating model...")
    d_user_input = 4  # price, p_yes, outcome_index, y

    if args.model_size == "small":
        model = create_small_dual_encoder(
            d_user_input=d_user_input,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
            dropout=DROPOUT,
        )
    else:
        model = create_base_dual_encoder(
            d_user_input=d_user_input,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
            dropout=DROPOUT,
        )

    model = model.to(device)
    print(f"[INFO] Model parameters: {model.count_parameters():,}")

    # Optimizer
    print("[DEBUG] Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
    )

    # Warmup + Cosine Annealing scheduler
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
            return LR_MIN / LR + (1 - LR_MIN / LR) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    # Directories
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    # Resume or new run
    if args.resume:
        print(f"[DEBUG] Resuming from checkpoint: {args.resume}")
        ckpt_path = Path(args.resume).resolve()
        ckpt_dir = ckpt_path.parent
        run_name, global_step, epoch = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, device
        )
        print(f"[DEBUG] Checkpoint loaded: step={global_step}, epoch={epoch}")
    else:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_dual_encoder"
        global_step = 0
        epoch = 0
        ckpt_dir = CKPT_ROOT / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Starting new run: {run_name}")

    # Initialize wandb
    print("[DEBUG] Initializing Weights & Biases...")
    wandb.init(
        project="polyquant",
        id=run_name,
        name=run_name,
        dir=str(RUNS_DIR),
        config={
            "model": "DualEncoderTransformer",
            "model_size": args.model_size,
            "batch_size": BATCH_SIZE,
            "l_market": L_MARKET,
            "l_user": L_USER,
            "max_steps": MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "lr": LR,
            "lr_min": LR_MIN,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "amp_enabled": AMP_ENABLED,
            "resumed_from": args.resume or "fresh",
        },
        resume="must" if args.resume else "never",
    )
    print("[DEBUG] Weights & Biases initialized")

    # Training loop
    print("[DEBUG] Starting training loop...")
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Calculate steps per epoch for fractional logging
    steps_per_epoch = len(train_loader)
    log_every_steps = max(1, int(steps_per_epoch * LOG_EVERY_EPOCH_FRACTION))
    print(f"[INFO] Steps per epoch: {steps_per_epoch}, logging every {log_every_steps} steps ({LOG_EVERY_EPOCH_FRACTION:.0%} of epoch)")

    model.train()
    t0 = time.time()
    last_log_time = t0

    while global_step < MAX_STEPS:
        epoch += 1
        print(f"[EPOCH] {epoch}, starting at step {global_step}", flush=True)

        for batch in train_loader:
            if global_step >= MAX_STEPS:
                break

            global_step += 1

            market_x, market_mask, market_y, user_x, user_mask, user_y = batch

            market_x = market_x.to(device, non_blocking=True)
            market_mask = market_mask.to(device, non_blocking=True)
            market_y = market_y.to(device, non_blocking=True)
            user_x = user_x.to(device, non_blocking=True)
            user_mask = user_mask.to(device, non_blocking=True)
            user_y = user_y.to(device, non_blocking=True)

            # Concatenate user_y as 4th feature
            user_x_with_y = torch.cat([user_x, user_y.unsqueeze(-1)], dim=-1)

            # Get target, validity, price, and outcome_index
            y, valid = get_target_from_batch(market_y, market_mask)
            price = get_price_from_batch(market_x, market_mask)
            outcome_index = get_outcome_index_from_batch(market_x, market_mask)

            if valid.sum() == 0:
                if global_step % log_every_steps == 0:
                    print(f"[STEP {global_step}] SKIPPED - no valid samples", flush=True)
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(market_x, market_mask, user_x_with_y, user_mask)
                # Masked BCE loss
                bce = loss_fn(logits, y)
                loss = (bce * valid.float()).sum() / valid.float().sum()

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Train logging
            if global_step % log_every_steps == 0:
                elapsed = time.time() - t0
                steps_per_sec = global_step / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]

                with torch.no_grad():
                    # Filter to valid samples
                    valid_idx = valid.nonzero(as_tuple=True)[0]
                    logits_valid = logits[valid_idx]
                    y_valid = y[valid_idx]
                    price_valid = price[valid_idx]
                    outcome_index_valid = outcome_index[valid_idx]

                    probs = torch.sigmoid(logits_valid)
                    preds = (probs > 0.5).to(y_valid.dtype)
                    misclass = (preds != y_valid).float().mean()

                    pred_edge = probs - price_valid
                    # Correct edge: YES bet (oi=1) wins if y=1, NO bet (oi=0) wins if y=0
                    trade_won = (y_valid * outcome_index_valid + (1 - y_valid) * (1 - outcome_index_valid))
                    true_edge = trade_won - price_valid
                    mae_edge = torch.abs(pred_edge - true_edge).mean()

                print(f"[STEP {global_step}] loss={float(loss):.4f} misclass={float(misclass):.4f} lr={lr:.6f}", flush=True)
                wandb.log(
                    {
                        "train/bce": float(loss),
                        "train/misclass": float(misclass),
                        "train/mae_edge": float(mae_edge),
                        "train/lr": lr,
                        "train/steps_per_sec": steps_per_sec,
                    },
                    step=global_step,
                )

                now = time.time()
                dt = now - last_log_time
                last_log_time = now

            # Validation
            if global_step % VAL_EVERY_STEPS == 0:
                val_metrics = evaluate(model, val_loader, device, VAL_MAX_BATCHES)
                log_metrics_to_wandb(val_metrics, step=global_step, prefix="val")
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
                    scheduler=scheduler,
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
        scheduler=scheduler,
        scaler=scaler,
    )

    wandb.finish()
    print(f"[DONE] Finished at step {global_step}, epoch {epoch}")


if __name__ == "__main__":
    main()
