from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    dataset_root: Path
    scaler_path: Path
    runs_dir: Path
    checkpoints_dir: Path

def _repo_root(from_file: str | Path, config_name: str = "config.json", max_depth: int = 4) -> Path:
    p = Path(from_file).resolve()

    # Check the file's directory and up to `max_depth` parents
    candidates = [p.parent] + list(p.parents[:max_depth])

    for parent in candidates:
        if (parent / config_name).exists():
            return parent

    raise FileNotFoundError(
        f"Could not find {config_name} within {max_depth} parents of {from_file}. "
        f"Tried: {[str(c / config_name) for c in candidates]}"
    )

def load_paths(from_file: str | Path, config_name: str = "config.json") -> PathsConfig:
    root = _repo_root(from_file, config_name)
    cfg_path = root / config_name

    raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    def p(x: str) -> Path:
        px = Path(x)
        return px if px.is_absolute() else (root / px)

    return PathsConfig(
        root=root,
        dataset_root=p(raw["dataset_root"]),
        scaler_path=p(raw["scaler_path"]),
        runs_dir=p(raw["runs_dir"]),
        checkpoints_dir=p(raw["checkpoints_dir"]),
    )
