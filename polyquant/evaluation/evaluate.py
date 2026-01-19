#!/usr/bin/env python
"""
Unified evaluation script for running evaluation on any model type.

Usage:
    python -m polyquant.evaluation.evaluate --model-type resnet --checkpoint path/to/checkpoint.pt
    python -m polyquant.evaluation.evaluate --model-type transformer --checkpoint path/to/checkpoint.pt
    python -m polyquant.evaluation.evaluate --model-type dual_encoder --checkpoint path/to/checkpoint.pt --model-size small

This script dispatches to the appropriate model-specific evaluation script.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for all model types"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["resnet", "transformer", "dual_encoder"],
        help="Type of model to evaluate",
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
        help="Model size for dual_encoder (ignored for other models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    # Build the command for the specific evaluation script
    script_map = {
        "resnet": "polyquant.evaluation.eval_resnet",
        "transformer": "polyquant.evaluation.eval_transformer",
        "dual_encoder": "polyquant.evaluation.eval_dual_encoder",
    }

    script_module = script_map[args.model_type]
    cmd = [sys.executable, "-m", script_module, "--checkpoint", args.checkpoint]

    if args.model_type == "dual_encoder":
        cmd.extend(["--model-size", args.model_size])

    if args.output:
        cmd.extend(["--output", args.output])

    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
