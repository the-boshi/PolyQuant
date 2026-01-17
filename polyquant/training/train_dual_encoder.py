#!/usr/bin/env python
"""
Training script for the Dual-Encoder Transformer.

Uses:
- Market sequence: trades in the current market (causal attention)
- User sequence: user's historical trades (bidirectional attention)

Predicts edge for each trade position using both market and user context.
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
from polyquant.data.datasets.dual_encoder_dataset import DualEncoderDataset
from polyquant.models.dual_encoder import (
    create_small_dual_encoder,
    create_medium_dual_encoder,
    create_base_dual_encoder,
    create_small_dual_encoder_cross_attn,
    create_medium_dual_encoder_cross_attn,
)
from polyquant.utils import load_checkpoint, save_checkpoint


# ===========================
# HYPERPARAMETERS / PATHS
# ===========================

PATHS = load_paths(__file__)
RUNS_DIR = PATHS.runs_dir
CKPT_ROOT = PATHS.checkpoints_dir

# Dataset paths
INDEX_PATH = PATHS.root / "data" / "sequences" / "index.parquet"
USER_SEQ_DIR = PATHS.root / "data" / "user_sequences"

# DataLoader
BATCH_SIZE = 64  # Increased for smoother gradients
NUM_WORKERS = 4
L_MARKET = 1024
L_USER = 64
CAP_TRADES = 4096
MIN_PREFIX = 20

# Minimum context before computing loss
MIN_CONTEXT = 50

# Training
MAX_STEPS = 100_000
LOG_EVERY_STEPS = 10
VAL_EVERY_STEPS = 200
VAL_MAX_BATCHES = 50
CHECKPOINT_EVERY_STEPS = 5_000
GRAD_ACCUM_STEPS = 2  # Effective batch = 128

# Optimizer
LR = 1e-4  # Reduced for stability
LR_MIN = 1e-6
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000

# Label smoothing to handle noisy labels
LABEL_SMOOTHING = 0.05

# Feature normalization stats (computed from training data)
# Market features: [p_yes, dp_yes_clip, log_dt, log_usdc_size, user_recent_pnl_asinh,
#                  user_avg_size_log, user_days_active_log, user_hist_pnl_asinh,
#                  user_hist_winrate, user_pnl_std_log]
MARKET_MEAN = torch.tensor([0.637, 0.0, 2.4, 2.5, 0.0, 4.4, 3.3, 1.1, 0.48, 3.5])
MARKET_STD = torch.tensor([0.23, 0.04, 3.1, 1.7, 2.1, 1.5, 1.4, 5.1, 0.24, 1.8])

# User features: [price, log_usdc_size, outcome, edge]
USER_MEAN = torch.tensor([0.47, 2.3, 0.48, 0.01])
USER_STD = torch.tensor([0.30, 1.6, 0.50, 0.40])

# Regularization
GRAD_CLIP_NORM = 1.0
AMP_ENABLED = True
DROPOUT = 0.15

# Policy thresholds for profitability metrics
TAUS = [0.00, 0.005, 0.01, 0.02, 0.05]


# ===========================
# DATA LOADERS
# ===========================

def make_loaders(index_path: Path, user_seq_dir: Path) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    train_ds = DualEncoderDataset(
        index_path=str(index_path),
        user_seq_dir=str(user_seq_dir),
        split="train",
        L_market=L_MARKET,
        L_user=L_USER,
        cap_trades=CAP_TRADES,
        min_prefix=MIN_PREFIX,
        pf_cache=64,
    )
    val_ds = DualEncoderDataset(
        index_path=str(index_path),
        user_seq_dir=str(user_seq_dir),
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
# LOSS FUNCTION
# ===========================

class DualEncoderLoss:
    """
    BCE loss for dual encoder output.

    Since DualEncoderDataset returns ONE user history per market (for the last trade),
    we predict edge for the last position only.
    """

    def __init__(self, use_pnl_weight: bool = False, min_weight: float = 1e-3,
                 label_smoothing: float = 0.0):
        self.use_pnl_weight = use_pnl_weight
        self.min_weight = min_weight
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        logits: torch.Tensor,  # (B, L_market) - but we only use last valid position
        y: torch.Tensor,       # (B,) market outcome
        price: torch.Tensor,   # (B, L_market) p_yes per trade
        mask: torch.Tensor,    # (B, L_market) valid tokens
    ) -> torch.Tensor:
        """Compute loss on last valid position per sample."""
        B, L = logits.shape

        # Find last valid position for each sample
        last_valid_idx = mask.long().cumsum(dim=1).argmax(dim=1)  # (B,)

        # Gather logits at last valid position
        batch_idx = torch.arange(B, device=logits.device)
        last_logits = logits[batch_idx, last_valid_idx]  # (B,)

        # Apply label smoothing: y_smooth = y * (1 - eps) + 0.5 * eps
        y_float = y.float()
        if self.label_smoothing > 0:
            y_float = y_float * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        if self.use_pnl_weight:
            last_prices = price[batch_idx, last_valid_idx]
            w = y.float() * (1.0 - last_prices) + (1.0 - y.float()) * last_prices
            w = torch.clamp(w, min=self.min_weight)
            bce = F.binary_cross_entropy_with_logits(last_logits, y_float, reduction="none")
            return (bce * w).sum() / (w.sum() + 1e-8)
        else:
            # Simple unweighted BCE
            return F.binary_cross_entropy_with_logits(last_logits, y_float)


# ===========================
# EVALUATION
# ===========================

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int,
             market_mean: torch.Tensor, market_std: torch.Tensor,
             user_mean: torch.Tensor, user_std: torch.Tensor) -> dict:
    """
    Evaluate on validation set with profitability metrics.
    """
    model.eval()
    loss_fn = DualEncoderLoss(use_pnl_weight=False)

    total_loss = 0.0
    total_mis = 0.0
    total_mae_edge = 0.0
    total_n = 0

    pnl_sum = {t: 0.0 for t in TAUS}
    take_sum = {t: 0 for t in TAUS}
    win_sum = {t: 0 for t in TAUS}

    batches = 0
    for market_x, market_mask, user_x, user_mask, y in loader:
        market_x = market_x.to(device, non_blocking=True)
        market_mask = market_mask.to(device, non_blocking=True)
        user_x = user_x.to(device, non_blocking=True)
        user_mask = user_mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        price = market_x[:, :, 0].clone()  # p_yes is first feature (before normalization)

        # Normalize features
        market_x = (market_x - market_mean) / market_std
        user_x = (user_x - user_mean) / user_std

        logits = model(market_x, market_mask, user_x, user_mask)  # (B, L_market)
        loss = loss_fn(logits, y, price, market_mask)

        B, L = logits.shape

        # Get predictions at last valid position
        last_valid_idx = market_mask.long().cumsum(dim=1).argmax(dim=1)
        batch_idx = torch.arange(B, device=device)

        last_logits = logits[batch_idx, last_valid_idx]
        last_prices = price[batch_idx, last_valid_idx]

        probs = torch.sigmoid(last_logits)  # (B,)
        y_float = y.float()

        # Misclassification
        preds = (probs > 0.5).float()
        mis = (preds != y_float).float().mean()

        # Edge
        pred_edge = probs - last_prices
        true_edge = y_float - last_prices
        mae_edge = torch.abs(pred_edge - true_edge).mean()

        n = B
        total_loss += float(loss) * n
        total_mis += float(mis) * n
        total_mae_edge += float(mae_edge) * n
        total_n += n

        # Profitability
        realized = y_float - last_prices

        for t in TAUS:
            take = pred_edge > t
            if take.any():
                pnl_sum[t] += float(realized[take].sum())
                take_cnt = int(take.sum())
                take_sum[t] += take_cnt
                win_sum[t] += int((y_float[take] > 0.5).sum())

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
        take_rate = (n_take / total_n) if total_n > 0 else float("nan")
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
        default="medium-cross",
        choices=["small", "medium", "base", "cross-attn", "medium-cross"],
        help="Model size/type",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    print(f"[INFO] Using device: {device}")

    # Check if user sequences exist
    if not USER_SEQ_DIR.exists():
        raise RuntimeError(
            f"User sequences not found at {USER_SEQ_DIR}. "
            "Run polyquant_features/build_user_sequences.py first."
        )

    # Data
    print(f"[INFO] Loading data...")
    print(f"  Market index: {INDEX_PATH}")
    print(f"  User sequences: {USER_SEQ_DIR}")
    train_loader, val_loader = make_loaders(INDEX_PATH, USER_SEQ_DIR)
    print(f"[INFO] Train markets: {len(train_loader.dataset)}, Val markets: {len(val_loader.dataset)}")

    # Model
    if args.model_size == "small":
        model = create_small_dual_encoder(
            d_market_input=10,
            d_user_input=4,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
            dropout=DROPOUT,
        )
    elif args.model_size == "medium":
        model = create_medium_dual_encoder(
            d_market_input=10,
            d_user_input=4,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
            dropout=DROPOUT,
        )
    elif args.model_size == "base":
        model = create_base_dual_encoder(
            d_market_input=10,
            d_user_input=4,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
            dropout=DROPOUT,
        )
    elif args.model_size == "cross-attn":
        model = create_small_dual_encoder_cross_attn(
            d_market_input=10,
            d_user_input=4,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
            dropout=DROPOUT,
        )
    else:  # medium-cross
        model = create_medium_dual_encoder_cross_attn(
            d_market_input=10,
            d_user_input=4,
            max_market_len=L_MARKET,
            max_user_len=L_USER,
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

    # Loss with label smoothing for robustness
    loss_fn = DualEncoderLoss(use_pnl_weight=False, label_smoothing=LABEL_SMOOTHING)

    # Feature normalization tensors (move to device)
    market_mean = MARKET_MEAN.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, 10)
    market_std = MARKET_STD.to(device).unsqueeze(0).unsqueeze(0)
    user_mean = USER_MEAN.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
    user_std = USER_STD.to(device).unsqueeze(0).unsqueeze(0)

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
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_dual_encoder"
        global_step = 0
        epoch = 0
        ckpt_dir = CKPT_ROOT / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Starting new run: {run_name}")

    writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
    writer.add_text(
        "run_info",
        f"run_name={run_name}, model={args.model_size}, L_market={L_MARKET}, "
        f"L_user={L_USER}, batch={BATCH_SIZE}, resumed={args.resume or 'fresh'}"
    )

    # Training loop
    model.train()
    t0 = time.time()
    running_loss = 0.0
    running_mis = 0.0
    running_n = 0

    accum_step = 0  # Track gradient accumulation

    while global_step < MAX_STEPS:
        epoch += 1
        print(f"[EPOCH] {epoch}, starting at step {global_step}")

        for market_x, market_mask, user_x, user_mask, y in train_loader:
            if global_step >= MAX_STEPS:
                break

            market_x = market_x.to(device, non_blocking=True)
            market_mask = market_mask.to(device, non_blocking=True)
            user_x = user_x.to(device, non_blocking=True)
            user_mask = user_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Store raw price before normalization
            price = market_x[:, :, 0].clone()

            # Normalize features
            market_x = (market_x - market_mean) / market_std
            user_x = (user_x - user_mean) / user_std

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(market_x, market_mask, user_x, user_mask)
                loss = loss_fn(logits, y, price, market_mask)
                # Scale loss for gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            accum_step += 1

            # Only update weights after accumulating enough gradients
            if accum_step % GRAD_ACCUM_STEPS == 0:
                global_step += 1

                lr = get_lr(global_step, WARMUP_STEPS, MAX_STEPS, LR, LR_MIN)
                set_lr(optimizer, lr)

                if GRAD_CLIP_NORM > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Running metrics (compute on each batch for logging)
            with torch.no_grad():
                B, L = logits.shape
                last_valid_idx = market_mask.long().cumsum(dim=1).argmax(dim=1)
                batch_idx = torch.arange(B, device=device)

                last_logits = logits[batch_idx, last_valid_idx]
                probs = torch.sigmoid(last_logits)
                preds = (probs > 0.5).float()
                y_float = y.float()

                mis = (preds != y_float).float().mean()
                n = B

            # Undo the accumulation scaling for logging
            running_loss += float(loss) * GRAD_ACCUM_STEPS * n
            running_mis += float(mis) * n
            running_n += n

            # Logging (only after optimizer step)
            if accum_step % GRAD_ACCUM_STEPS == 0 and global_step % LOG_EVERY_STEPS == 0 and running_n > 0:
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

                running_loss = 0.0
                running_mis = 0.0
                running_n = 0

            # Validation
            if global_step % VAL_EVERY_STEPS == 0:
                val_metrics = evaluate(model, val_loader, device, VAL_MAX_BATCHES,
                                       market_mean, market_std, user_mean, user_std)

                for k, v in val_metrics.items():
                    if isinstance(v, float) and not math.isnan(v):
                        writer.add_scalar(f"val/{k}", v, global_step)

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
