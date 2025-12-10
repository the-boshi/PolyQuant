from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pyarrow.parquet as pq


ID_COLS = ["trade_uid", "user_id", "market_id", "timestamp"]
LABEL_COLS = ["label_y"]          # created in your split script
AUX_COLS = ["edge", "price"]      # keep for metrics / pred_edge = p - price


def _infer_columns_from_parquet(split_dir: Path) -> List[str]:
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {split_dir}")
    schema = pq.ParquetFile(str(files[0])).schema_arrow
    return list(schema.names)


@dataclass(frozen=True)
class DatasetSchema:
    feature_cols: List[str]       # model inputs
    scale_cols: List[str]         # standardized (train-only mean/std)
    no_scale_cols: List[str]      # pass through as-is
    label_col: str                # "label_y"
    price_col: str                # "price"
    edge_col: str                 # "edge"


def load_schema(dataset_root: Path) -> DatasetSchema:
    """
    Figures out which columns are features, and which should be normalized.
    Uses parquet schema from train split (authoritative).
    """
    train_dir = dataset_root / "train"
    cols = _infer_columns_from_parquet(train_dir)

    # sanity
    for c in ["price", "edge"]:
        if c not in cols:
            raise RuntimeError(f"Required column missing from dataset: {c}")
    if "label_y" not in cols:
        # still usable, but training script will need to compute label_y = clip(edge+price,0,1) > 0.5
        pass

    drop = set(ID_COLS) | set(LABEL_COLS) | {"edge"}  # do not feed label/edge as features

    feature_cols = [c for c in cols if c not in drop]
    REDUNDANT = {"market_time_since_first_trade"}  # or {"seconds_since_market_open"}
    feature_cols = [c for c in feature_cols if c not in REDUNDANT]

    # columns we do NOT standardize
    no_scale_cols = []
    for c in feature_cols:
        if c.endswith("_missing"):
            no_scale_cols.append(c)
    # keep these raw/bounded
    for c in ["price", "outcome_index", "time_of_day_sin", "time_of_day_cos"]:
        if c in feature_cols and c not in no_scale_cols:
            no_scale_cols.append(c)

    scale_cols = [c for c in feature_cols if c not in set(no_scale_cols)]

    return DatasetSchema(
        feature_cols=feature_cols,
        scale_cols=scale_cols,
        no_scale_cols=no_scale_cols,
        label_col="label_y",
        price_col="price",
        edge_col="edge",
    )
