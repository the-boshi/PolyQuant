#!/usr/bin/env python
import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from polyquant.data.schema import load_schema
from polyquant.data.normalize import load_feature_scaler
from polyquant.data.datasets.tabular import TabularParquetIterable
from polyquant.models.mlp import MLP


# ===========================
# HYPERPARAMETERS / PATHS
# ===========================

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "features_dataset"
SCALER_PATH = DATASET_ROOT / "train_scaler.json"

RUNS_REL = Path("runs")
CHECKPOINTS_REL = Path("checkpoints")

BATCH_SIZE = 16384
NUM_WORKERS = 4

MAX_STEPS = 50_000
LOG_EVERY_STEPS = 1
VAL_EVERY_STEPS = 5
VAL_MAX_BATCHES = 100
CHECKPOINT_EVERY_STEPS = 1_000

LR = 3e-4
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-2

HIDDEN_DIMS = [512, 512, 256, 128]
DROPOUT = 0.05

GRAD_CLIP_NORM = 1.0
AMP_ENABLED = True


# ===========================
# CHECKPOINT HELPERS
# ===========================

def save_checkpoint(
    ckpt_dir: Path,
    run_name: str,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"

    payload = {
        "run_name": run_name,
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, ckpt_path)
    print(f"[CKPT] Saved checkpoint: {ckpt_path}")
    return ckpt_path


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
):
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])

    if "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and ckpt["scheduler"] is not None and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt and ckpt["scaler"] is not None and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])

    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)
    run_name = ckpt.get("run_name", ckpt_path.parent.name)

    print(f"[CKPT] Loaded checkpoint from {ckpt_path} (step={step}, epoch={epoch}, run={run_name})")
    return run_name, step, epoch


# ===========================
# DATA HELPERS
# ===========================

def make_loaders(schema, scaler):
    train_ds = TabularParquetIterable(
        split_dir=DATASET_ROOT / "train",
        feature_cols=schema.feature_cols,
        scaler=scaler,
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        shuffle_rowgroup=True,
        seed=123,
        shuffle_buffer=500_000,
    )
    val_ds = TabularParquetIterable(
        split_dir=DATASET_ROOT / "val",
        feature_cols=schema.feature_cols,
        scaler=scaler,
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        shuffle_rowgroup=True,
        seed=456,
        shuffle_buffer=200_000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


# ===========================
# EVAL
# ===========================

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    total_loss = 0.0
    total_mis = 0.0
    total_mae_edge = 0.0
    total_n = 0

    batches = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        price = batch["price"].to(device, non_blocking=True)
        edge = batch["edge"].to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).to(y.dtype)
        mis = (preds != y).float().mean()

        pred_edge = probs - price
        mae_edge = torch.abs(pred_edge - edge).mean()

        n = x.size(0)
        total_loss += float(loss) * n
        total_mis += float(mis) * n
        total_mae_edge += float(mae_edge) * n
        total_n += n

        batches += 1
        if batches >= max_batches:
            break

    if total_n == 0:
        return {"bce": math.nan, "misclass": math.nan, "mae_edge": math.nan}

    return {
        "bce": total_loss / total_n,
        "misclass": total_mis / total_n,
        "mae_edge": total_mae_edge / total_n,
    }


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

    schema = load_schema(DATASET_ROOT)
    scaler = load_feature_scaler(SCALER_PATH, schema.feature_cols, schema.no_scale_cols)

    train_loader, val_loader = make_loaders(schema, scaler)

    model = MLP(
        in_dim=len(schema.feature_cols),
        hidden=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_STEPS,
        eta_min=LR_MIN,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)

    # run / checkpoint dirs
    RUNS_DIR = ROOT / RUNS_REL
    CKPT_ROOT = ROOT / CHECKPOINTS_REL
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    # resume or new run
    if args.resume:
        ckpt_path = Path(args.resume).resolve()
        ckpt_dir = ckpt_path.parent
        run_name, global_step, epoch = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, device
        )
    else:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        global_step = 0
        epoch = 0
        ckpt_dir = CKPT_ROOT / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Starting new run: {run_name}")

    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    writer.add_text("run_info", f"run_name={run_name}, resumed_from={args.resume or 'fresh'}")

    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    model.train()
    t0 = time.time()
    last_log_time = t0

    while global_step < MAX_STEPS:
        epoch += 1
        print(f"[EPOCH] {epoch}, starting at step {global_step}")

        for batch in train_loader:
            if global_step >= MAX_STEPS:
                break

            global_step += 1

            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # TRAIN LOGGING
            if global_step % LOG_EVERY_STEPS == 0:
                elapsed = time.time() - t0
                steps_per_sec = global_step / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]

                # misclass + mae_edge on this batch
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).to(y.dtype)
                    mis = (preds != y).float().mean()

                    price = batch["price"].to(device, non_blocking=True)
                    edge = batch["edge"].to(device, non_blocking=True)
                    pred_edge = probs - price
                    mae_edge = torch.abs(pred_edge - edge).mean()

                writer.add_scalar("train/bce", float(loss), global_step)
                writer.add_scalar("train/misclass", float(mis), global_step)
                writer.add_scalar("train/mae_edge", float(mae_edge), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step)

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
                writer.add_scalar("val/bce", val_metrics["bce"], global_step)
                writer.add_scalar("val/misclass", val_metrics["misclass"], global_step)
                writer.add_scalar("val/mae_edge", val_metrics["mae_edge"], global_step)
                print(
                    f"[VAL step {global_step}] "
                    f"bce={val_metrics['bce']:.4f} "
                    f"mis={val_metrics['misclass']:.3f} "
                    f"mae_edge={val_metrics['mae_edge']:.4f}"
                )
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
    writer.close()
    print(f"[DONE] Finished at step {global_step}, epoch {epoch}")


if __name__ == "__main__":
    main()
