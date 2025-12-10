#!/usr/bin/env python3
"""
Split Polymarket trade-level parquet features into train/val/test folders (Split A),
based on market end_date (resolve time) from SQLite.

- Reads markets.end_date (ISO like "2021-03-15T00:00:00Z") from polymarket.db
- Sorts markets by end_ts, splits by market count: 70/15/15
- Streams input parquet row-groups and writes rows to:
    data/train/*.parquet
    data/val/*.parquet
    data/test/*.parquet
- Adds missing flags and clips sentinel -1 time-since columns:
    time_since_user_last_trade
    time_since_last_market_trade
    time_since_user_last_trade_in_market
  For each: col := max(col, 0), and add col_missing (0/1)
- Adds label column for BCE route:
    label_y = clip(edge + price, 0, 1)

Run:
  python make_splits.py
"""

import os
import re
import json
import math
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_BASE = r"C:\Users\nimro\PolyQuant-features"
DEFAULT_IN_DIR = rf"{DEFAULT_BASE}\outputs\features"
DEFAULT_DB_PATH = rf"{DEFAULT_BASE}\sql\polymarket.db"
DEFAULT_OUT_DIR = rf"{DEFAULT_BASE}\data"

# Columns that use sentinel -1 meaning "missing"
SENTINEL_NEG1_COLS = [
    "time_since_user_last_trade",
    "time_since_last_market_trade",
    "time_since_user_last_trade_in_market",
]


def parse_iso_z_to_epoch_seconds(s: str) -> int:
    # Example: "2021-03-15T00:00:00Z"
    # Convert Z -> +00:00 so datetime.fromisoformat can parse it.
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def load_market_end_ts(db_path: Path) -> dict:
    """
    Returns: dict market_id -> end_ts (int epoch seconds)
    Skips NULL end_date.
    """
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("""
            SELECT condition_id, end_date
            FROM markets
            WHERE end_date IS NOT NULL
        """)
        rows = cur.fetchall()
    finally:
        con.close()

    out = {}
    bad = 0
    for condition_id, end_date in rows:
        if condition_id is None or end_date is None:
            continue
        try:
            out[str(condition_id)] = parse_iso_z_to_epoch_seconds(str(end_date))
        except Exception:
            bad += 1

    if len(out) == 0:
        raise RuntimeError("No markets with non-null parsable end_date found in DB.")

    if bad:
        print(f"[warn] Failed to parse end_date for {bad} markets; those markets will be ignored.")
    return out


def make_split_mapping(market_end_ts: dict, train_frac=0.70, val_frac=0.15):
    """
    Split by market count after sorting by end_ts ascending.
    Returns:
      market_to_split: dict market_id -> "train"/"val"/"test"
      split_table_df:  DataFrame [market_id, end_ts, split]
    """
    items = sorted(market_end_ts.items(), key=lambda kv: kv[1])
    n = len(items)
    n_train = int(math.floor(train_frac * n))
    n_val = int(math.floor(val_frac * n))
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    market_to_split = {}
    for mid, _ in train_items:
        market_to_split[mid] = "train"
    for mid, _ in val_items:
        market_to_split[mid] = "val"
    for mid, _ in test_items:
        market_to_split[mid] = "test"

    df = pd.DataFrame(
        [(mid, ts, market_to_split[mid]) for mid, ts in items],
        columns=["market_id", "end_ts", "split"],
    )

    meta = {
        "n_markets_total": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": 1.0 - train_frac - val_frac,
        "train_end_ts_max": int(max(ts for _, ts in train_items)) if train_items else None,
        "val_end_ts_max": int(max(ts for _, ts in val_items)) if val_items else None,
        "test_end_ts_max": int(max(ts for _, ts in test_items)) if test_items else None,
    }

    return market_to_split, df, meta


class SplitParquetWriters:
    """
    Manages rolling parquet outputs per split with max rows per file.
    """
    def __init__(self, out_root: Path, rows_per_file: int = 2_000_000, compression: str = "snappy"):
        self.out_root = out_root
        self.rows_per_file = int(rows_per_file)
        self.compression = compression

        self.writers = {"train": None, "val": None, "test": None}
        self.schemas = {"train": None, "val": None, "test": None}
        self.file_index = {"train": 0, "val": 0, "test": 0}
        self.rows_in_file = {"train": 0, "val": 0, "test": 0}
        self.total_rows = {"train": 0, "val": 0, "test": 0}

        for s in ["train", "val", "test"]:
            (self.out_root / s).mkdir(parents=True, exist_ok=True)

    def _open_new(self, split: str, schema: pa.Schema):
        if self.writers[split] is not None:
            self.writers[split].close()

        idx = self.file_index[split]
        path = self.out_root / split / f"part-{idx:06d}.parquet"
        self.writers[split] = pq.ParquetWriter(
            where=str(path),
            schema=schema,
            compression=self.compression,
            use_dictionary=True,
        )
        self.schemas[split] = schema
        self.rows_in_file[split] = 0
        self.file_index[split] += 1

    def write_df(self, split: str, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return

        table = pa.Table.from_pandas(df, preserve_index=False)

        if self.writers[split] is None:
            self._open_new(split, table.schema)
        else:
            # Enforce schema consistency
            if not table.schema.equals(self.schemas[split], check_metadata=False):
                table = table.cast(self.schemas[split])

        # Roll file if needed
        if self.rows_in_file[split] + table.num_rows > self.rows_per_file:
            self._open_new(split, self.schemas[split])

        self.writers[split].write_table(table)
        self.rows_in_file[split] += table.num_rows
        self.total_rows[split] += table.num_rows

    def close(self):
        for s, w in self.writers.items():
            if w is not None:
                w.close()
                self.writers[s] = None


def transform_batch_df(df: pd.DataFrame, market_to_split: dict) -> dict:
    """
    Input df: a pandas DataFrame for one parquet chunk/row-group.
    Output: dict split -> df_for_split (with transforms applied)
    """
    if "market_id" not in df.columns:
        raise RuntimeError("Input parquet is missing required column: market_id")
    if "edge" not in df.columns or "price" not in df.columns:
        raise RuntimeError("Input parquet is missing required columns: edge and/or price")

    # Assign split by market_id mapping
    split_series = df["market_id"].astype(str).map(market_to_split)

    # Drop rows with unknown market_id (e.g., missing end_date)
    keep = split_series.notna()
    if not keep.any():
        return {"train": None, "val": None, "test": None}

    df = df.loc[keep].copy()
    split_series = split_series.loc[keep]

    # Sentinel -1 transforms + missing flags
    for col in SENTINEL_NEG1_COLS:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            missing = (x < 0) | x.isna()
            df[col] = np.where(missing.values, 0.0, x.values).astype(np.float32)
            df[f"{col}_missing"] = missing.astype(np.int8).values

    # Label for BCE route: y = edge + price (should be ~0/1)
    y = pd.to_numeric(df["edge"], errors="coerce").values + pd.to_numeric(df["price"], errors="coerce").values
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    df["label_y"] = y

    # Split into three dfs
    out = {}
    for s in ["train", "val", "test"]:
        m = (split_series.values == s)
        if m.any():
            out[s] = df.loc[m].copy()
        else:
            out[s] = None
    return out


def main():
    base_dir = Path(os.environ.get("POLYQUANT_BASE", DEFAULT_BASE))
    in_dir = Path(os.environ.get("POLYQUANT_IN_DIR", DEFAULT_IN_DIR))
    db_path = Path(os.environ.get("POLYQUANT_DB_PATH", DEFAULT_DB_PATH))
    out_dir = Path(os.environ.get("POLYQUANT_OUT_DIR", DEFAULT_OUT_DIR))

    if not in_dir.exists():
        raise FileNotFoundError(f"Input parquet dir not found: {in_dir}")
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading markets end_ts from DB...")
    market_end_ts = load_market_end_ts(db_path)

    print("[2/4] Building split mapping (Split A: by market end_ts)...")
    market_to_split, split_df, split_meta = make_split_mapping(market_end_ts, train_frac=0.70, val_frac=0.15)

    # Save mapping for reproducibility
    split_df_path = out_dir / "market_splits.parquet"
    split_meta_path = out_dir / "market_splits_meta.json"
    split_df.to_parquet(split_df_path, index=False)
    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(f"  wrote: {split_df_path}")
    print(f"  wrote: {split_meta_path}")
    print(f"  markets: train={split_meta['n_train']}, val={split_meta['n_val']}, test={split_meta['n_test']}")

    # Input files
    files = sorted(in_dir.glob("*.parquet"))
    if len(files) == 0:
        raise RuntimeError(f"No parquet files found in: {in_dir}")

    print("[3/4] Streaming parquet files and writing split datasets...")
    writers = SplitParquetWriters(out_root=out_dir, rows_per_file=2_000_000, compression="snappy")

    total_files = len(files)
    total_rowgroups = 0

    try:
        for fi, path in enumerate(files, 1):
            pf = pq.ParquetFile(str(path))
            rg = pf.num_row_groups
            total_rowgroups += rg

            print(f"  [{fi}/{total_files}] {path.name} (row_groups={rg})")

            for rgi in range(rg):
                table = pf.read_row_group(rgi)  # read all columns
                df = table.to_pandas(split_blocks=True, self_destruct=True)

                split_dfs = transform_batch_df(df, market_to_split)

                for s in ["train", "val", "test"]:
                    if split_dfs[s] is not None and len(split_dfs[s]) > 0:
                        writers.write_df(s, split_dfs[s])

    finally:
        writers.close()

    print("[4/4] Done.")
    print("Rows written:")
    print(f"  train: {writers.total_rows['train']}")
    print(f"  val:   {writers.total_rows['val']}")
    print(f"  test:  {writers.total_rows['test']}")
    print(f"Output root: {out_dir}")


if __name__ == "__main__":
    main()
