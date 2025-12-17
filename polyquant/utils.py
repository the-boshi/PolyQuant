from pathlib import Path
import torch
import torch.nn as nn

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
