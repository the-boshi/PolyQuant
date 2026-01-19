#!/usr/bin/env python
"""
Evaluation script for Transformer models (no user encoding) on the test set.

Usage:
    python -m polyquant.evaluation.eval_transformer --checkpoint path/to/checkpoint.pt

The script will:
1. Load the checkpoint
2. Run evaluation on the test set
3. Save results to a JSON file in the checkpoint directory
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from polyquant.config import load_paths
from polyquant.data.datasets.sequence_dataset import MarketWindowDataset
from polyquant.models.transformer import create_base_transformer_no_user
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
INDEX_PATH = PATHS.root / "data" / "sequences" / "index.parquet"

# Model configuration (must match training)
SEQ_LEN = 1024
D_INPUT = 10

# DataLoader settings
BATCH_SIZE = 64
NUM_WORKERS = 4
CAP_TRADES = 8192
MIN_PREFIX = 20


# ===========================
# DATA LOADING
# ===========================


def make_test_loader(index_path: Path) -> DataLoader:
    """Create test set data loader."""
    test_ds = MarketWindowDataset(
        index_path=str(index_path),
        split="test",
        L=SEQ_LEN,
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
# EVALUATION
# ===========================


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Evaluate model on the full test set.

    Returns:
        dict with metrics (bce, accuracy, auc)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0
    all_probs = []
    all_labels = []

    n_batches = 0
    for x, u, mask, y in loader:
        x = x.to(device, non_blocking=True)
        # u not used in no-user model
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x, mask)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        correct = (preds == y).sum()

        n = x.size(0)
        total_loss += float(loss) * n
        total_correct += int(correct)
        total_n += n

        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())

        n_batches += 1
        if n_batches % 100 == 0:
            print(f"  Processed {n_batches} batches, {total_n:,} samples...")

    print(f"  Total: {n_batches} batches, {total_n:,} samples")

    if total_n == 0:
        return {"bce": float("nan"), "accuracy": float("nan")}

    metrics = {
        "bce": total_loss / total_n,
        "accuracy": total_correct / total_n,
        "misclass": 1.0 - (total_correct / total_n),
        "n_samples": total_n,
    }

    # Compute AUC if sklearn is available
    try:
        from sklearn.metrics import roc_auc_score

        all_probs_np = torch.cat(all_probs).numpy()
        all_labels_np = torch.cat(all_labels).numpy()
        if len(set(all_labels_np)) > 1:
            metrics["auc"] = roc_auc_score(all_labels_np, all_probs_np)
    except ImportError:
        print("[WARN] sklearn not available, skipping AUC computation")

    return metrics


# ===========================
# MAIN
# ===========================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Transformer checkpoint on test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
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

    print_results_header("Transformer", checkpoint_path)

    # Create model
    print("[INFO] Creating model...")
    model = create_base_transformer_no_user(
        d_input=D_INPUT,
        max_seq_len=SEQ_LEN,
    )

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path.name}")
    checkpoint_info = load_model_from_checkpoint(checkpoint_path, model, device)
    print(f"  Run: {checkpoint_info['run_name']}")
    print(f"  Step: {checkpoint_info['step']}")
    print(f"  Epoch: {checkpoint_info['epoch']}")

    # Create test loader
    print("[INFO] Creating test data loader...")
    print(f"  Index path: {INDEX_PATH}")
    test_loader = make_test_loader(INDEX_PATH)
    print(f"  Test markets: {len(test_loader.dataset)}")

    # Evaluate
    print("[INFO] Running evaluation on test set...")
    metrics = evaluate(model, test_loader, device)

    # Print and save results
    print_results_summary(metrics)

    extra_info = {
        "seq_len": SEQ_LEN,
        "d_input": D_INPUT,
        "n_params": model.count_parameters(),
    }
    save_results_to_file(output_path, checkpoint_info, metrics, "transformer", extra_info)


if __name__ == "__main__":
    main()
