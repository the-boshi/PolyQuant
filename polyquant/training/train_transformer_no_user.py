#!/usr/bin/env python
"""
Training script for the MarketTransformer without user embeddings.

Uses the sequence dataset (MarketWindowDataset) for binary outcome prediction.
Loss: regular BCE (unweighted)
Logging: Weights & Biases
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
from polyquant.data.datasets.sequence_dataset import MarketWindowDataset
from polyquant.models.transformer import (
    create_base_transformer_no_user,
    create_small_transformer_no_user,
)
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
CAP_TRADES = 8192
MIN_PREFIX = 20

# Training
MAX_STEPS = 200_000
LOG_EVERY_STEPS = 10
VAL_EVERY_STEPS = 100
VAL_MAX_BATCHES = 50
CHECKPOINT_EVERY_STEPS = 1_000

# Optimizer
LR = 1e-4
LR_MIN = 1e-6
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

# Regularization
GRAD_CLIP_NORM = 1.0
AMP_ENABLED = True
DROPOUT = 0.1


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
# EVALUATION
# ===========================

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> dict:
    """
    Evaluate on validation set.

    Returns:
        dict with bce, accuracy, auc (if sklearn available)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0
    all_probs = []
    all_labels = []

    batches = 0
    for x, u, mask, y in loader:
        x = x.to(device, non_blocking=True)
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

        all_probs.append(probs.detach().cpu())
        all_labels.append(y.detach().cpu())

        batches += 1
        if batches >= max_batches:
            break

    if total_n == 0:
        return {"bce": float("nan"), "accuracy": float("nan")}

    metrics = {
        "bce": total_loss / total_n,
        "accuracy": total_correct / total_n,
    }

    # Compute AUC if sklearn is available
    try:
        from sklearn.metrics import roc_auc_score

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()
        if len(set(all_labels)) > 1:
            metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ImportError:
        pass

    return metrics


# ===========================
# MAIN TRAINING
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Train MarketTransformer without user embeddings")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume from",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
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
        model = create_small_transformer_no_user(
            d_input=10,
            max_seq_len=SEQ_LEN,
            dropout=DROPOUT,
        )
    else:
        model = create_base_transformer_no_user(
            d_input=10,
            max_seq_len=SEQ_LEN,
            dropout=DROPOUT,
        )
    model = model.to(device)
    print(f"[INFO] Model parameters: {model.count_parameters():,}")

    # Optimizer (manual LR schedule)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    # Directories
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    # Resume or new run
    if args.resume:
        ckpt_path = Path(args.resume).resolve()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        wandb_run_id = ckpt.get("wandb_run_id")
        ckpt_dir = ckpt_path.parent
        run_name, global_step, epoch = load_checkpoint(
            ckpt_path, model, optimizer, None, scaler, device
        )
        if wandb_run_id is None:
            print("[WARN] Checkpoint has no wandb_run_id; a new W&B run will be created.")
    else:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_transformer_no_user"
        global_step = 0
        epoch = 0
        ckpt_dir = CKPT_ROOT / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Starting new run: {run_name}")
        wandb_run_id = None

    wandb.init(
        project="polyquant",
        name=run_name,
        dir=str(RUNS_DIR),
        id=wandb_run_id,
        config={
            "model_size": args.model_size,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "cap_trades": CAP_TRADES,
            "min_prefix": MIN_PREFIX,
            "max_steps": MAX_STEPS,
            "lr": LR,
            "lr_min": LR_MIN,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "dropout": DROPOUT,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "amp_enabled": AMP_ENABLED,
            "resumed_from": args.resume or "fresh",
        },
        resume="allow" if args.resume else None,
    )

    # Training loop
    model.train()
    t0 = time.time()
    running_loss = 0.0
    running_correct = 0
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
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(x, mask)
                loss = F.binary_cross_entropy_with_logits(logits, y.float())

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

            # Running metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                correct = (preds == y).sum()

            running_loss += float(loss) * x.size(0)
            running_correct += int(correct)
            running_n += x.size(0)

            # Logging
            if global_step % LOG_EVERY_STEPS == 0:
                avg_loss = running_loss / max(running_n, 1)
                avg_acc = running_correct / max(running_n, 1)
                elapsed = time.time() - t0
                steps_per_sec = global_step / max(elapsed, 1e-6)

                wandb.log(
                    {
                        "train/bce": avg_loss,
                        "train/accuracy": avg_acc,
                        "train/lr": lr,
                        "train/steps_per_sec": steps_per_sec,
                    },
                    step=global_step,
                )

                print(
                    f"[STEP {global_step:5d}] "
                    f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                    f"lr={lr:.2e} ({steps_per_sec:.1f} steps/s)"
                )

                # Reset running metrics
                running_loss = 0.0
                running_correct = 0
                running_n = 0

            # Validation
            if global_step % VAL_EVERY_STEPS == 0:
                val_metrics = evaluate(model, val_loader, device, VAL_MAX_BATCHES)

                val_log = {
                    f"val/{k}": v
                    for k, v in val_metrics.items()
                    if v is not None and isinstance(v, float) and not math.isnan(v)
                }
                if val_log:
                    wandb.log(val_log, step=global_step)

                print(
                    f"[VAL   {global_step:5d}] "
                    f"bce={val_metrics['bce']:.4f} acc={val_metrics['accuracy']:.3f} "
                    f"auc={val_metrics.get('auc', float('nan')):.3f}"
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
                    extra={"wandb_run_id": wandb.run.id},
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
        extra={"wandb_run_id": wandb.run.id},
    )
    wandb.finish()
    print(f"[DONE] Finished at step {global_step}, epoch {epoch}")


if __name__ == "__main__":
    main()
