#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from polyquant.data.schema import load_schema


@dataclass
class RunningMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations

    def update_batch(self, x: np.ndarray) -> None:
        x = x.astype(np.float64, copy=False)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return

        bn = int(x.size)
        bmean = float(x.mean())
        # batch M2
        bm2 = float(((x - bmean) ** 2).sum())

        if self.n == 0:
            self.n, self.mean, self.m2 = bn, bmean, bm2
            return

        n = self.n
        mean = self.mean
        m2 = self.m2

        delta = bmean - mean
        new_n = n + bn
        new_mean = mean + delta * (bn / new_n)
        new_m2 = m2 + bm2 + (delta * delta) * (n * bn / new_n)

        self.n, self.mean, self.m2 = new_n, new_mean, new_m2

    def finalize(self) -> Tuple[float, float, int]:
        if self.n <= 1:
            return self.mean, 1.0, self.n
        var = self.m2 / self.n  # population variance for scaling
        std = math.sqrt(var) if var > 0 else 1.0
        return self.mean, std, self.n


def main():
    dataset_root = Path(r"data\features_full")
    out_path = dataset_root / "train_scaler.json"

    schema = load_schema(dataset_root)
    train_dir = dataset_root / "train"
    files = sorted(train_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train parquet files in: {train_dir}")

    moments: Dict[str, RunningMoments] = {c: RunningMoments() for c in schema.scale_cols}

    total_rows = 0
    for fp in tqdm(files, desc="train parquets"):
        pf = pq.ParquetFile(str(fp))
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=schema.scale_cols)
            total_rows += table.num_rows

            for c in schema.scale_cols:
                arr = table.column(c).to_numpy(zero_copy_only=False)
                moments[c].update_batch(arr)

    scaler = {
        "dataset_root": str(dataset_root),
        "total_rows_scanned": int(total_rows),
        "scale_cols": schema.scale_cols,
        "no_scale_cols": schema.no_scale_cols,
        "stats": {},
    }

    for c, rm in moments.items():
        mean, std, n = rm.finalize()
        scaler["stats"][c] = {"mean": float(mean), "std": float(std), "count": int(n)}

    out_path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
