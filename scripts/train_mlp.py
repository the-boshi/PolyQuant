#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from polyquant.data.schema import load_schema
from polyquant.data.normalize import load_feature_scaler
from polyquant.data.datasets.tabular import TabularParquetIterable
from polyquant.models.mlp import MLP

# =========================
# Hyperparameters / Config
# =========================

BATCH_SIZE = 8192
NUM_WORKERS = 2

LR = 3e-4
WEIGHT_DECAY = 1e-2
HIDDEN_DIMS = (512, 512, 256, 128)
DROPOUT = 0.05

MAX_STEPS = 100000
LOG_EVERY_STEPS = 1
EVAL_EVERY_STEPS = 10
VAL_MAX_BATCHES = 200

SAVE_EVERY_EVAL = True
GRAD_CLIP_NORM = 0.0  # 0 disables, e.g. 1.0 to enable

SEED = 123
SHUFFLE_FILES_TRAIN = True
SHUFFLE_ROWGROUP_TRAIN = True

# Paths (relative to repo root)
DATASET_REL = Path("data") / "features_dataset"
SCALER_NAME = "train_scaler.json"
CHECKPOINTS_REL = Path("checkpoints")
RUNS_REL = Path("runs") / "mlp"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    n_examples = 0
    loss_sum = 0.0
    mae_edge_sum = 0.0
    miscls_sum = 0.0

    batches = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        price = batch["price"].to(device, non_blocking=True)
        edge = batch["edge"].to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        pred = (logits > 0).to(y.dtype)
        miscls = torch.mean((pred != y).to(torch.float32))

        p = torch.sigmoid(logits)
        pred_edge = p - price
        mae_edge = torch.mean(torch.abs(pred_edge - edge))

        bsz = x.shape[0]
        n_examples += bsz
        loss_sum += float(loss) * bsz
        mae_edge_sum += float(mae_edge) * bsz
        miscls_sum += float(miscls) * bsz

        batches += 1
        if batches >= max_batches:
            break

    model.train()
    denom = max(n_examples, 1)
    return {
        "bce": loss_sum / denom,
        "mae_edge": mae_edge_sum / denom,
        "misclass": miscls_sum / denom,
        "n_examples": int(n_examples),
        "n_batches": int(batches),
    }


def main():
    # ---- hard requirement ----
    assert torch.cuda.is_available(), (
        "CUDA is not available. Fix your PyTorch install / driver. "
        "Try: python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\""
    )

    device = torch.device("cuda")
    set_seed(SEED)

    # Prefer TF32 on Ampere+ for speed (safe for this task)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ROOT = repo_root()
    DATASET_ROOT = ROOT / DATASET_REL
    SCALER_PATH = DATASET_ROOT / SCALER_NAME

    CKPT_ROOT = ROOT / CHECKPOINTS_REL
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # per-run dirs
    runs_dir = ROOT / RUNS_REL / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = CKPT_ROOT / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(runs_dir))


    # pin_memory only matters for CUDA
    pin_memory = True

    schema = load_schema(DATASET_ROOT)
    scaler = load_feature_scaler(SCALER_PATH, schema.feature_cols, schema.no_scale_cols)

    train_ds = TabularParquetIterable(
        split_dir=DATASET_ROOT / "train",
        feature_cols=schema.feature_cols,
        scaler=scaler,
        batch_size=BATCH_SIZE,
        shuffle_files=SHUFFLE_FILES_TRAIN,
        shuffle_rowgroup=SHUFFLE_ROWGROUP_TRAIN,
        seed=SEED,
    )
    val_ds = TabularParquetIterable(
        split_dir=DATASET_ROOT / "val",
        feature_cols=schema.feature_cols,
        scaler=scaler,
        batch_size=BATCH_SIZE,
        shuffle_files=False,
        shuffle_rowgroup=False,
        seed=SEED + 1,
    )

    # Dataset yields pre-batched dicts -> batch_size=None
    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    model = MLP(in_dim=len(schema.feature_cols), hidden=HIDDEN_DIMS, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    # New AMP API
    amp = torch.amp.GradScaler(enabled=True)

    meta = {
        "run_name": run_name,
        "device": "cuda",
        "dataset_root": str(DATASET_ROOT),
        "scaler_path": str(SCALER_PATH),
        "feature_dim": len(schema.feature_cols),
        "feature_cols": schema.feature_cols,
        "no_scale_cols": schema.no_scale_cols,
        "hparams": {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dims": list(HIDDEN_DIMS),
            "dropout": DROPOUT,
            "max_steps": MAX_STEPS,
            "log_every_steps": LOG_EVERY_STEPS,
            "eval_every_steps": EVAL_EVERY_STEPS,
            "val_max_batches": VAL_MAX_BATCHES,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "seed": SEED,
        },
    }
    (runs_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    step = 0
    t_log = time()

    try:

        # ---- log step 0 BEFORE training ----
        batch0 = next(iter(train_loader))
        x0 = batch0["x"].to(device)
        y0 = batch0["y"].to(device)
        price0 = batch0["price"].to(device)
        edge0 = batch0["edge"].to(device)

        with torch.no_grad():
            logits0 = model(x0)
            loss0 = loss_fn(logits0, y0)
            pred0 = (logits0 > 0).to(y0.dtype)
            mis0 = torch.mean((pred0 != y0).float())

            p0 = torch.sigmoid(logits0)
            mae_edge0 = torch.mean(torch.abs((p0 - price0) - edge0))

        writer.add_scalar("train/bce", float(loss0), 0)
        writer.add_scalar("train/misclass", float(mis0), 0)
        writer.add_scalar("train/mae_edge", float(mae_edge0), 0)

        print(f"[step 0] bce={float(loss0):.6f} misclass={float(mis0):.6f} mae_edge={float(mae_edge0):.6f}")

        for batch in train_loader:
            step += 1

            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(x)
                loss = loss_fn(logits, y)

            amp.scale(loss).backward()

            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            amp.step(opt)
            amp.update()

            with torch.no_grad():
                pred = (logits > 0).to(y.dtype)
                train_miscls = torch.mean((pred != y).to(torch.float32))

            if step % LOG_EVERY_STEPS == 0:
                dt = time() - t_log
                t_log = time()

                writer.add_scalar("train/bce", float(loss), step)
                writer.add_scalar("train/misclass", float(train_miscls), step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
                writer.add_scalar("train/steps_per_sec", LOG_EVERY_STEPS / max(dt, 1e-6), step)

                print(
                    f"step={step} "
                    f"train_bce={float(loss):.6f} "
                    f"train_miscls={float(train_miscls):.6f} "
                    f"steps_per_sec={LOG_EVERY_STEPS / max(dt, 1e-6):.2f}"
                )

            if step % EVAL_EVERY_STEPS == 0:
                metrics = evaluate(model, val_loader, device, max_batches=VAL_MAX_BATCHES)
                writer.add_scalar("val/bce", metrics["bce"], step)
                writer.add_scalar("val/mae_edge", metrics["mae_edge"], step)
                writer.add_scalar("val/misclass", metrics["misclass"], step)

                print(
                    f"[val] step={step} "
                    f"bce={metrics['bce']:.6f} "
                    f"mae_edge={metrics['mae_edge']:.6f} "
                    f"misclass={metrics['misclass']:.6f} "
                    f"(batches={metrics['n_batches']}, examples={metrics['n_examples']})"
                )

                if SAVE_EVERY_EVAL:
                    ckpt = {
                        "step": step,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "feature_cols": schema.feature_cols,
                        "scaler_path": str(SCALER_PATH),
                        "run_name": run_name,
                    }
                    ckpt_path = ckpt_dir / f"mlp_step_{step:07d}.pt"
                    torch.save(ckpt, ckpt_path)
                    print(f"saved: {ckpt_path}")

            if step >= MAX_STEPS:
                break

    finally:
        ckpt = {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "feature_cols": schema.feature_cols,
            "scaler_path": str(SCALER_PATH),
            "run_name": run_name,
        }
        ckpt_path = ckpt_dir / f"mlp_final_{step:07d}.pt"
        torch.save(ckpt, ckpt_path)
        writer.close()
        print(f"saved final: {ckpt_path}")
        print(f"tensorboard logdir: {runs_dir}")


if __name__ == "__main__":
    main()
