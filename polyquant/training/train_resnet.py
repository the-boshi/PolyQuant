#!/usr/bin/env python
import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so polyquant can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import wandb
from polyquant.config import load_paths
from polyquant.data.schema import load_schema
from polyquant.data.normalize import load_feature_scaler
from polyquant.data.datasets.tabular import make_loaders
from polyquant.utils import load_checkpoint, save_checkpoint
from polyquant.models.resnet import ResNet1D, ResNetMLP
from polyquant.metrics import MetricsAccumulator, log_metrics_to_wandb, log_train_metrics_to_wandb

# ===========================
# HYPERPARAMETERS / PATHS
# ===========================

PATHS = load_paths(__file__)
DATASET_ROOT = PATHS.dataset_root
SCALER_PATH = PATHS.scaler_path
RUNS_DIR = PATHS.runs_dir
CKPT_ROOT = PATHS.checkpoints_dir

BATCH_SIZE = 512  # Reduced from 4096 for GPU memory
NUM_WORKERS = 12

MAX_STEPS = 1_000_000
WARMUP_STEPS = 5000  # Longer warmup for stability
LOG_EVERY_EPOCH_FRACTION = 0.05  # Log every 5% of an epoch
VAL_EVERY_STEPS = 500
VAL_MAX_BATCHES = 100
CHECKPOINT_EVERY_STEPS = 10_000

LR = 3e-5  # Reduced from 1e-4 for stability
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-2  # Increased from 1e-3 for regularization

HIDDEN_DIMS = (256, 256, 512, 1024, 512, 512, 1024, 2048, 4096, 1024, 512, 256, 128, 32)  # ResNetMLP hidden layer sizes
DROPOUT = 0.15  # Slightly increased for regularization

GRAD_CLIP_NORM = 0.5  # Tighter gradient clipping
AMP_ENABLED = False


# ===========================
# EVAL
# ===========================

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int):
    """
    Evaluation on a (possibly truncated) stream of batches.

    Returns dict with:
      - bce: BCE loss
      - misclass: 0/1 error rate at threshold 0.5
      - mae_edge: mean absolute error of predicted vs true edge
      - profitability metrics
    """
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = MetricsAccumulator()

    batches = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        price = batch["price"].to(device, non_blocking=True)
        edge = batch["edge"].to(device, non_blocking=True)

        logits = model(x).view(-1)
        loss = loss_fn(logits, y)

        metrics.update(logits, y, price, loss=float(loss), edge=edge)

        batches += 1
        if batches >= max_batches:
            break

    return metrics.compute()


# ===========================
# MAIN TRAINING
# ===========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume from",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but no GPU was detected.")

    device = torch.device("cuda")
    print(f"[INFO] Using device: {device}")

    print("[DEBUG] Loading schema...")
    schema = load_schema(DATASET_ROOT)
    print("[DEBUG] Schema loaded")

    print("[DEBUG] Loading feature scaler...")
    feature_scaler = load_feature_scaler(SCALER_PATH, schema.feature_cols, schema.no_scale_cols)
    print("[DEBUG] Feature scaler loaded")

    print("[DEBUG] Creating data loaders...")
    train_loader, val_loader = make_loaders(schema, feature_scaler)
    print("[DEBUG] Data loaders created")

    print("[DEBUG] Creating model...")
    model = ResNetMLP(
        in_dim=len(schema.feature_cols),
        hidden=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(device)

    print("[DEBUG] Model created")

    print("[DEBUG] Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    print("[DEBUG] Optimizer created")

    # Warmup + Cosine Annealing scheduler
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            # Linear warmup from 0 to LR
            return step / WARMUP_STEPS
        else:
            # Cosine annealing from LR to LR_MIN
            progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
            return LR_MIN / LR + (1 - LR_MIN / LR) * 0.5 * (1 + math.cos(math.pi * progress))

    print("[DEBUG] Creating scheduler and scaler...")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    print("[DEBUG] Scheduler and scaler created")

    # run / checkpoint dirs
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    # resume or new run
    if args.resume:
        print(f"[DEBUG] Resuming from checkpoint: {args.resume}")
        ckpt_path = Path(args.resume).resolve()
        ckpt_dir = ckpt_path.parent
        run_name, global_step, epoch = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, device
        )
        print(f"[DEBUG] Checkpoint loaded: step={global_step}, epoch={epoch}")
    else:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_resnet"
        global_step = 0
        epoch = 0
        ckpt_dir = CKPT_ROOT / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Starting new run: {run_name}")

    print("[DEBUG] Initializing Weights & Biases...")
    wandb.init(
        project="polyquant",
        id=run_name,  # Use run_name as unique ID so resume works
        name=run_name,
        dir=str(RUNS_DIR),
        config={
            "model": "ResNetMLP",
            "batch_size": BATCH_SIZE,
            "max_steps": MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "lr": LR,
            "lr_min": LR_MIN,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "amp_enabled": AMP_ENABLED,
            "resumed_from": args.resume or "fresh",
        },
        resume="must" if args.resume else "never",  # Force resume or force new
    )
    print("[DEBUG] Weights & Biases initialized")

    print("[DEBUG] Starting training loop...")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Track steps per epoch dynamically (for IterableDataset without __len__)
    steps_per_epoch = None
    log_every_steps = 100  # Default for first epoch

    model.train()
    t0 = time.time()
    last_log_time = t0

    while global_step < MAX_STEPS:
        epoch += 1
        epoch_start_step = global_step
        print(f"[EPOCH] {epoch}, starting at step {global_step}")

        for batch in train_loader:
            if global_step >= MAX_STEPS:
                break

            global_step += 1

            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            price = batch["price"].to(device, non_blocking=True)
            edge = batch["edge"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(x)
                loss = loss_fn(logits.view(-1), y)

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # TRAIN LOGGING
            if global_step % log_every_steps == 0:
                elapsed = time.time() - t0
                steps_per_sec = global_step / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]

                log_train_metrics_to_wandb(
                    loss=loss.detach().item(),
                    logits=logits,
                    y=y,
                    price=price,
                    edge=edge,
                    lr=lr,
                    steps_per_sec=steps_per_sec,
                    step=global_step,
                )

                now = time.time()
                dt = now - last_log_time
                last_log_time = now
                #print(
                #    f"[STEP {global_step}] "
                #    f"loss={float(loss):.4f} mis={float(mis):.3f} mae_edge={float(mae_edge):.4f} "
                #    f"lr={lr:.2e} ({steps_per_sec:.1f} steps/s, dt={dt:.1f}s)"
                #)

            # VALIDATION (separate from checkpointing)
            if global_step % VAL_EVERY_STEPS == 0:
                val_metrics = evaluate(model, val_loader, device, VAL_MAX_BATCHES)
                log_metrics_to_wandb(val_metrics, step=global_step, prefix="val")
                model.train()

            # CHECKPOINT (strictly every CHECKPOINT_EVERY_STEPS)
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

        # Update steps_per_epoch after first epoch completes
        epoch_steps = global_step - epoch_start_step
        if steps_per_epoch is None and epoch_steps > 0:
            steps_per_epoch = epoch_steps
            log_every_steps = max(1, int(steps_per_epoch * LOG_EVERY_EPOCH_FRACTION))
            print(f"[INFO] Steps per epoch: {steps_per_epoch}, logging every {log_every_steps} steps ({LOG_EVERY_EPOCH_FRACTION:.0%} of epoch)")

        # loop over train_loader ends; while-loop will restart a new pass

    # final checkpoint
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
