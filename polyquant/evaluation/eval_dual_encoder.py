#!/usr/bin/env python
"""
Evaluation script for DualEncoder models on the test set.

Usage:
    python -m polyquant.evaluation.eval_dual_encoder --checkpoint path/to/checkpoint.pt

The script will:
1. Load the checkpoint
2. Run evaluation on the test set
3. Save results to a JSON file in the checkpoint directory
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from polyquant.config import load_paths
from polyquant.data.datasets.dual_sequence_dataset import DualSequenceDataset
from polyquant.metrics import MetricsAccumulator
from polyquant.models.dual_encoder import (
    create_base_dual_encoder,
    create_small_dual_encoder,
)
from polyquant.evaluation.eval_utils import (
    load_model_from_checkpoint,
    save_results_to_file,
    print_results_header,
    print_results_summary,
)


# ===========================
# CONFIGURATION
# ===========================

PATHS = load_paths(__file__)

# Dataset paths
MARKET_INDEX_PATH = PATHS.root / "data" / "sequences" / "index.parquet"
USER_INDEX_PATH = PATHS.root / "data" / "user_sequences_store" / "index.parquet"

# Model configuration (must match training)
L_MARKET = 1024
L_USER = 128
D_USER_INPUT = 4  # price, p_yes, outcome_index, y
DROPOUT = 0.1

# DataLoader settings
BATCH_SIZE = 32
NUM_WORKERS = 4
CAP_TRADES = 4096
MIN_PREFIX = 20


# ===========================
# DATA LOADING
# ===========================


def make_test_loader(
    market_index_path: Path,
    user_index_path: Path,
) -> DataLoader:
    """Create test set data loader."""
    test_ds = DualSequenceDataset(
        market_index_path=str(market_index_path),
        user_index_path=str(user_index_path),
        split="test",
        L_market=L_MARKET,
        L_user=L_USER,
        cap_trades=CAP_TRADES,
        min_prefix=MIN_PREFIX,
        pf_cache=32,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )
    return test_loader


# ===========================
# BATCH PROCESSING
# ===========================


def get_target_from_batch(
    market_y: torch.Tensor,
    market_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the target (last valid token's y value) from the batch.

    Returns:
        y: (B,) - target labels (binary)
        valid: (B,) - True for samples with known outcome (y != 0.5)
    """
    B = market_y.shape[0]
    device = market_y.device

    last_idx = market_mask.sum(dim=1) - 1
    last_idx = last_idx.clamp(min=0)

    batch_idx = torch.arange(B, device=device)
    y = market_y[batch_idx, last_idx]

    valid = y != 0.5
    y_binary = (y > 0.5).float()

    return y_binary, valid


def get_price_from_batch(
    market_x: torch.Tensor,
    market_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract price of the target trade (last valid token)."""
    B = market_x.shape[0]
    device = market_x.device

    last_idx = market_mask.sum(dim=1) - 1
    last_idx = last_idx.clamp(min=0)

    batch_idx = torch.arange(B, device=device)
    price = market_x[batch_idx, last_idx, 0]

    return price


def get_outcome_index_from_batch(
    market_x: torch.Tensor,
    market_mask: torch.Tensor,
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
    outcome_index = market_x[batch_idx, last_idx, 2]

    return outcome_index


# ===========================
# EVALUATION
# ===========================


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Evaluate model on the full test set.

    Returns:
        dict with metrics from MetricsAccumulator
    """
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = MetricsAccumulator()

    n_batches = 0
    skipped_batches = 0

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

        # Get target, validity, and price
        y, valid = get_target_from_batch(market_y, market_mask)
        price = get_price_from_batch(market_x, market_mask)
        outcome_index = get_outcome_index_from_batch(market_x, market_mask)

        if valid.sum() == 0:
            skipped_batches += 1
            n_batches += 1
            continue

        # Forward pass
        logits = model(market_x, market_mask, user_x_with_y, user_mask)

        # Filter to valid samples only (known outcome)
        valid_idx = valid.nonzero(as_tuple=True)[0]
        logits_valid = logits[valid_idx]
        y_valid = y[valid_idx]
        price_valid = price[valid_idx]
        outcome_index_valid = outcome_index[valid_idx]

        # Also filter out samples where model produced NaN
        # (can happen with short sequences due to transformer attention issues)
        non_nan_mask = ~logits_valid.isnan()
        if non_nan_mask.sum() == 0:
            skipped_batches += 1
            n_batches += 1
            continue

        logits_valid = logits_valid[non_nan_mask]
        y_valid = y_valid[non_nan_mask]
        price_valid = price_valid[non_nan_mask]
        outcome_index_valid = outcome_index_valid[non_nan_mask]

        loss = loss_fn(logits_valid, y_valid)

        metrics.update(
            logits_valid,
            y_valid,
            price_valid,
            loss=float(loss),
            outcome_index=outcome_index_valid,
        )

        n_batches += 1
        if n_batches % 100 == 0:
            print(f"  Processed {n_batches} batches, {metrics.total_n:,} valid samples...")

    print(f"  Total: {n_batches} batches, {metrics.total_n:,} valid samples")
    if skipped_batches > 0:
        print(f"  Skipped {skipped_batches} batches (no valid samples)")

    return metrics.compute()


# ===========================
# MAIN
# ===========================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DualEncoder checkpoint on test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pt) file",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "base"],
        help="Model size: small (~3M) or base (~12M)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON (default: checkpoint_dir/eval_test.json)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = checkpoint_path.parent / f"eval_test_{checkpoint_path.stem}.json"

    # Device
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU (this will be slow)")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[INFO] Using device: {device}")

    print_results_header("DualEncoder", checkpoint_path)

    # Create model
    print(f"[INFO] Creating {args.model_size} model...")
    if args.model_size == "small":
        model = create_small_dual_encoder(
            d_user_input=D_USER_INPUT,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
            dropout=DROPOUT,
        )
    else:
        model = create_base_dual_encoder(
            d_user_input=D_USER_INPUT,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
            dropout=DROPOUT,
        )

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path.name}")
    checkpoint_info = load_model_from_checkpoint(checkpoint_path, model, device)
    print(f"  Run: {checkpoint_info['run_name']}")
    print(f"  Step: {checkpoint_info['step']}")
    print(f"  Epoch: {checkpoint_info['epoch']}")

    # Create test loader
    print("[INFO] Creating test data loader...")
    print(f"  Market index: {MARKET_INDEX_PATH}")
    print(f"  User index: {USER_INDEX_PATH}")
    test_loader = make_test_loader(MARKET_INDEX_PATH, USER_INDEX_PATH)
    print(f"  Test markets: {len(test_loader.dataset)}")

    # Evaluate
    print("[INFO] Running evaluation on test set...")
    metrics = evaluate(model, test_loader, device)

    # Print and save results
    print_results_summary(metrics)

    extra_info = {
        "model_size": args.model_size,
        "l_market": L_MARKET,
        "l_user": L_USER,
        "d_user_input": D_USER_INPUT,
        "n_params": model.count_parameters(),
    }
    save_results_to_file(output_path, checkpoint_info, metrics, "dual_encoder", extra_info)


if __name__ == "__main__":
    main()
