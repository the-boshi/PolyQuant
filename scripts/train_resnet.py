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
from polyquant.losses.weighted_bce import PnLWeightedBCEWithLogits

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
LOG_EVERY_STEPS = 100
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
def evaluate(model, loader, device, max_batches: int, debug: bool = False, debug_file: str = None, global_step: int = 0):
    """
    Evaluation on a (possibly truncated) stream of batches.

    Returns:
      - bce: PnL-weighted BCE (your training loss)
      - misclass: 0/1 error rate at threshold 0.5
      - mae_edge: mean absolute error of (sigmoid(logit) - price) vs true edge
      - plus profitability metrics for several thresholds tau on predicted edge:
          take if (q - price) > tau
        reported as pnl/cost/roi/take_rate/hit_rate.
    """
    model.eval()
    loss_fn = PnLWeightedBCEWithLogits(min_weight=1e-3)

    # core metrics accumulators
    total_loss = 0.0
    total_unweighted_bce = 0.0  # standard BCE without PnL weighting
    total_mis = 0.0
    total_mae_edge = 0.0
    total_n = 0

    # Debug accumulators
    if debug:
        all_batch_losses = []
        all_logit_stats = []
        all_weight_stats = []
        all_unweighted_bce = []
        all_prob_stats = []

    # policy thresholds
    taus = [0.00, 0.005, 0.01, 0.02, 0.05]

    pnl_sum = {t: 0.0 for t in taus}
    cost_sum = {t: 0.0 for t in taus}
    take_sum = {t: 0 for t in taus}
    win_sum = {t: 0 for t in taus}     # count of y==1 among taken
    tot_sum = 0                        # total examples seen (for take_rate)

    batches = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)          # 0/1
        price = batch["price"].to(device, non_blocking=True)  # p in [0,1]
        edge = batch["edge"].to(device, non_blocking=True)

        logits = model(x).view(-1)

        # loss (scalar) - weighted BCE
        loss = loss_fn(logits, y, price)
        
        # unweighted BCE (standard)
        unweighted_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        total_unweighted_bce += float(unweighted_bce) * x.size(0)

        # Debug: compute components separately
        if debug:
            bce_per_ex = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            w = y * (1.0 - price) + (1.0 - y) * price
            w_clamped = torch.clamp(w, min=1e-3)

            all_batch_losses.append(float(loss))
            all_logit_stats.append({
                "min": float(logits.min()),
                "max": float(logits.max()),
                "mean": float(logits.mean()),
                "std": float(logits.std()),
            })
            all_weight_stats.append({
                "min": float(w.min()),
                "max": float(w.max()),
                "mean": float(w.mean()),
                "sum": float(w_clamped.sum()),
            })
            all_unweighted_bce.append(float(bce_per_ex.mean()))

            # Also track probability stats
            probs_debug = torch.sigmoid(logits)
            all_prob_stats.append({
                "min": float(probs_debug.min()),
                "max": float(probs_debug.max()),
                "mean": float(probs_debug.mean()),
                "std": float(probs_debug.std()),
            })

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
        tot_sum += n

        # profitability metrics (policy: take if q - p > tau; buy 1 share)
        realized = (y - price)   # profit per share

        for t in taus:
            take = pred_edge > t
            if take.any():
                pnl_sum[t] += float(realized[take].sum())
                take_cnt = int(take.sum().item())
                take_sum[t] += take_cnt
                win_sum[t] += int((y[take] > 0.5).sum().item())

        batches += 1
        if batches >= max_batches:
            break

    # Print debug summary
    if debug and batches > 0:
        import numpy as np
        losses = np.array(all_batch_losses)
        unweighted = np.array(all_unweighted_bce)
        logit_means = np.array([s["mean"] for s in all_logit_stats])
        logit_stds = np.array([s["std"] for s in all_logit_stats])
        logit_mins = np.array([s["min"] for s in all_logit_stats])
        logit_maxs = np.array([s["max"] for s in all_logit_stats])
        weight_means = np.array([s["mean"] for s in all_weight_stats])
        prob_means = np.array([s["mean"] for s in all_prob_stats])
        prob_mins = np.array([s["min"] for s in all_prob_stats])
        prob_maxs = np.array([s["max"] for s in all_prob_stats])

        debug_lines = [
            "",
            "=" * 60,
            f"DEBUG: Validation BCE Analysis (step={global_step})",
            "=" * 60,
            f"Batches evaluated: {batches}",
            f"Total samples: {total_n}",
            "",
            "Per-batch WEIGHTED BCE (what we log):",
            f"  min={losses.min():.4f}, max={losses.max():.4f}, mean={losses.mean():.4f}, std={losses.std():.4f}",
            "",
            "Per-batch UNWEIGHTED BCE (standard BCE):",
            f"  min={unweighted.min():.4f}, max={unweighted.max():.4f}, mean={unweighted.mean():.4f}",
            "",
            "Logit statistics:",
            f"  mean of means: {logit_means.mean():.4f}",
            f"  mean of stds:  {logit_stds.mean():.4f}",
            f"  global min:    {logit_mins.min():.4f}",
            f"  global max:    {logit_maxs.max():.4f}",
            "",
            "Probability statistics (sigmoid of logits):",
            f"  mean of means: {prob_means.mean():.4f}",
            f"  global min:    {prob_mins.min():.4f}",
            f"  global max:    {prob_maxs.max():.4f}",
            "",
            "Weight statistics (w = y*(1-p) + (1-y)*p):",
            f"  mean of means: {weight_means.mean():.4f}",
            "",
            f"Final accumulated BCE: {total_loss / total_n:.4f}",
            "=" * 60,
            "",
        ]

        debug_text = "\n".join(debug_lines)
        print(debug_text)

        # Write to file
        if debug_file:
            with open(debug_file, "a") as f:
                f.write(debug_text + "\n")

    if total_n == 0:
        out = {"bce": math.nan, "misclass": math.nan, "mae_edge": math.nan}
        for t in taus:
            key = f"tau_{t:.3f}".replace(".", "p")
            out[f"pnl/{key}"] = math.nan
            out[f"pnl_norm/{key}"] = math.nan
            out[f"take_rate/{key}"] = math.nan
            out[f"hit_rate/{key}"] = math.nan
            out[f"n_take/{key}"] = 0
        return out

    out = {
        "bce": total_loss / total_n,
        "bce_unweighted": total_unweighted_bce / total_n,
        "misclass": total_mis / total_n,
        "mae_edge": total_mae_edge / total_n,
    }

    for t in taus:
        key = f"tau_{t:.3f}".replace(".", "p")  # e.g., tau_0p010
        pnl = pnl_sum[t]
        cost = cost_sum[t]
        n_take = take_sum[t]

        pnl_norm = (pnl / n_take) if n_take > 0 else math.nan
        take_rate = (n_take / tot_sum) if tot_sum > 0 else math.nan
        hit_rate = (win_sum[t] / n_take) if n_take > 0 else math.nan

        out[f"pnl/{key}"] = pnl
        out[f"pnl_norm/{key}"] = pnl_norm
        out[f"take_rate/{key}"] = take_rate
        out[f"hit_rate/{key}"] = hit_rate
        out[f"n_take/{key}"] = n_take

    return out


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
    loss_fn = PnLWeightedBCEWithLogits(min_weight=1e-3)

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

            price = batch["price"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                logits = model(x)
                loss = loss_fn(logits, y, price)

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

                wandb.log({
                    "train/bce": loss.detach().item(),
                    "train/misclass": mis.detach().item(),
                    "train/mae_edge": mae_edge.detach().item(),
                    "train/lr": lr,
                    "train/steps_per_sec": steps_per_sec,
                }, step=global_step)

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
                # Debug every validation, write to wandb run directory
                debug_file = str(Path(wandb.run.dir) / "val_bce_debug.log")
                val_metrics = evaluate(model, val_loader, device, VAL_MAX_BATCHES,
                                       debug=True, debug_file=debug_file, global_step=global_step)

                val_log = {
                    f"val/{k}": v
                    for k, v in val_metrics.items()
                    if v is not None and isinstance(v, float) and not math.isnan(v)
                }
                if val_log:
                    wandb.log(val_log, step=global_step)

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
    wandb.finish()
    print(f"[DONE] Finished at step {global_step}, epoch {epoch}")


if __name__ == "__main__":
    main()
