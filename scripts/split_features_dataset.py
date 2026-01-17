#!/usr/bin/env python3
"""
Split the raw features parquet chunks into train/val/test folders
based on market_id -> split mapping from market_meta.parquet.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def main():
    project_root = Path(__file__).resolve().parent.parent
    features_dir = project_root / "data" / "features"
    market_meta_path = project_root / "data" / "market_meta.parquet"
    output_dir = project_root / "data" / "features_full"

    # Load market_id -> split and market_id -> y (outcome) mappings
    print(f"Loading market metadata from {market_meta_path}")
    meta_table = pq.read_table(market_meta_path, columns=["market_id", "split", "y"])
    meta_df = meta_table.to_pandas()
    market_to_split = dict(zip(meta_df["market_id"], meta_df["split"]))
    market_to_y = dict(zip(meta_df["market_id"], meta_df["y"]))
    print(f"  Loaded {len(market_to_split)} market -> split mappings")

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Get all feature chunk files
    chunk_files = sorted(features_dir.glob("features_chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No feature chunks found in {features_dir}")
    print(f"Found {len(chunk_files)} feature chunk files")

    # Process each chunk and split by market_id
    split_counts = {"train": 0, "val": 0, "test": 0}
    unknown_markets = set()

    for chunk_path in tqdm(chunk_files, desc="Processing chunks"):
        table = pq.read_table(chunk_path)
        df = table.to_pandas()

        # Map market_id to split and label_y (market outcome)
        df["_split"] = df["market_id"].map(market_to_split)
        df["label_y"] = df["market_id"].map(market_to_y).astype("float32")

        # Track unknown markets
        unknown_mask = df["_split"].isna()
        if unknown_mask.any():
            unknown_markets.update(df.loc[unknown_mask, "market_id"].unique())

        # Write each split
        for split in ["train", "val", "test"]:
            split_df = df[df["_split"] == split].drop(columns=["_split"])
            if len(split_df) == 0:
                continue

            split_counts[split] += len(split_df)
            out_path = output_dir / split / chunk_path.name
            pq.write_table(pa.Table.from_pandas(split_df, preserve_index=False), out_path)

    # Copy market_meta to output dir
    shutil.copy(market_meta_path, output_dir / "market_meta.parquet")

    # Summary
    print("\n=== Split Summary ===")
    for split, count in split_counts.items():
        print(f"  {split}: {count:,} rows")
    print(f"  Total: {sum(split_counts.values()):,} rows")

    if unknown_markets:
        print(f"\nWarning: {len(unknown_markets)} markets not found in market_meta (skipped)")

    print(f"\nOutput written to: {output_dir}")
    print("Next step: run 'python scripts/compute_norm_stats.py' after updating the dataset path")


if __name__ == "__main__":
    main()
