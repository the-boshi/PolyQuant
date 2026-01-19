"""Shared utilities for evaluation scripts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict:
    """
    Load model weights from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint .pt file
        model: Instantiated model to load weights into
        device: Device to load the model onto

    Returns:
        Checkpoint metadata dict (run_name, step, epoch, etc.)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return {
        "run_name": ckpt.get("run_name", checkpoint_path.parent.name),
        "step": ckpt.get("step", 0),
        "epoch": ckpt.get("epoch", 0),
        "checkpoint_path": str(checkpoint_path),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 1000:
                lines.append(f"  {key}: {value:.6e}")
            else:
                lines.append(f"  {key}: {value:.6f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def save_results_to_file(
    output_path: Path,
    checkpoint_info: dict,
    metrics: dict[str, Any],
    model_type: str,
    extra_info: dict | None = None,
) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        output_path: Path to save the results JSON
        checkpoint_info: Checkpoint metadata (run_name, step, etc.)
        metrics: Evaluation metrics dict
        model_type: Type of model ("resnet", "transformer", "dual_encoder")
        extra_info: Additional info to include (model config, etc.)
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "checkpoint": checkpoint_info,
        "metrics": metrics,
    }

    if extra_info:
        results["extra"] = extra_info

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[SAVED] Results saved to: {output_path}")


def print_results_header(model_type: str, checkpoint_path: Path):
    """Print a header for the evaluation run."""
    print("=" * 70)
    print(f"  EVALUATION: {model_type}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Run: {checkpoint_path.parent.name}")
    print("=" * 70)


def print_results_summary(metrics: dict[str, float]):
    """Print a summary of evaluation results."""
    print("\n" + "-" * 40)
    print("  RESULTS")
    print("-" * 40)
    print(format_metrics(metrics))
    print("-" * 40 + "\n")
