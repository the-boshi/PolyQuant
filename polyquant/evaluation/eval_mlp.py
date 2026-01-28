#!/usr/bin/env python
"""
Evaluation script for MLP models on the test set.

Usage:
    python -m polyquant.evaluation.eval_mlp --checkpoint path/to/checkpoint.pt

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
from polyquant.data.schema import load_schema
from polyquant.data.normalize import load_feature_scaler
from polyquant.data.datasets.tabular import TabularParquetIterable
from polyquant.metrics import MetricsAccumulator
from polyquant.models.mlp import MLP
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
DATASET_ROOT = PATHS.dataset_root
SCALER_PATH = PATHS.scaler_path

# Model configuration (must match training)
HIDDEN_DIMS = (512, 512, 256, 128)
DROPOUT = 0.05

# Default checkpoint
DEFAULT_CHECKPOINT = r"C:\Users\nimro\PolyQuant\checkpoints\20260123_104407_mlp\step_0002000.pt"

# DataLoader settings
BATCH_SIZE = 4096
NUM_WORKERS = 4


# ===========================
# DATA LOADING
# ===========================


def make_test_loader(schema, scaler) -> DataLoader:
    """Create test set data loader."""
    test_ds = TabularParquetIterable(
        split_dir=DATASET_ROOT / "test",
        feature_cols=schema.feature_cols,
        scaler=scaler,
        batch_size=BATCH_SIZE,
        shuffle_files=False,
        shuffle_rowgroup=False,
        seed=789,
        shuffle_buffer=BATCH_SIZE * 2,  # Minimal buffer, no real shuffling
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return test_loader


# ===========================
# EVALUATION
# ===========================


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Evaluate model on the full test set.

    Returns:
        dict with all metrics from MetricsAccumulator
    """
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = MetricsAccumulator()

    n_batches = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        price = batch["price"].to(device, non_blocking=True)
        edge = batch["edge"].to(device, non_blocking=True)
        outcome_index = batch["outcome_index"].to(device, non_blocking=True)

        logits = model(x).view(-1)
        loss = loss_fn(logits, y)

        metrics.update(logits, y, price, loss=float(loss), edge=edge, outcome_index=outcome_index)

        n_batches += 1
        if n_batches % 100 == 0:
            print(f"  Processed {n_batches} batches, {metrics.total_n:,} samples...")

    print(f"  Total: {n_batches} batches, {metrics.total_n:,} samples")
    return metrics.compute()


# ===========================
# MAIN
# ===========================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLP checkpoint on test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to checkpoint (.pt) file",
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

    print_results_header("MLP", checkpoint_path)

    # Load schema and scaler
    print("[INFO] Loading schema and scaler...")
    schema = load_schema(DATASET_ROOT)
    scaler = load_feature_scaler(SCALER_PATH, schema.feature_cols, schema.no_scale_cols)

    # Create model
    print("[INFO] Creating model...")
    model = MLP(
        in_dim=len(schema.feature_cols),
        hidden=HIDDEN_DIMS,
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
    test_loader = make_test_loader(schema, scaler)

    # Evaluate
    print("[INFO] Running evaluation on test set...")
    metrics = evaluate(model, test_loader, device)

    # Print and save results
    print_results_summary(metrics)

    extra_info = {
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "n_features": len(schema.feature_cols),
    }
    save_results_to_file(output_path, checkpoint_info, metrics, "mlp", extra_info)


if __name__ == "__main__":
    main()
