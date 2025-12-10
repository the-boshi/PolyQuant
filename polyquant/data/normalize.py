from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class FeatureScaler:
    feature_cols: List[str]
    mean: np.ndarray  # shape [F], float32
    std: np.ndarray   # shape [F], float32

    def transform_np(self, x: np.ndarray) -> np.ndarray:
        # x: [N, F] float32
        return (x - self.mean) / self.std


def load_feature_scaler(scaler_path: Path, feature_cols: List[str], no_scale_cols: List[str]) -> FeatureScaler:
    obj = json.loads(scaler_path.read_text(encoding="utf-8"))
    stats: Dict[str, Dict[str, float]] = obj["stats"]

    mean = np.zeros(len(feature_cols), dtype=np.float32)
    std = np.ones(len(feature_cols), dtype=np.float32)
    no_scale = set(no_scale_cols)

    for i, c in enumerate(feature_cols):
        if c in no_scale:
            continue
        s = stats.get(c)
        if s is None:
            # if it's not in stats, don't scale it
            continue
        m = float(s["mean"])
        sd = float(s["std"])
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        mean[i] = m
        std[i] = sd

    return FeatureScaler(feature_cols=feature_cols, mean=mean, std=std)
