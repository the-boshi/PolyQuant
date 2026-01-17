#!/usr/bin/env python
r"""
Script to convert absolute paths in index.parquet to relative paths.

The index.parquet file contains a 'path' column with machine-specific absolute paths
(e.g., C:\Users\nimro\PolyQuant\data\sequences\train\shard_0000.parquet).

This script converts them to paths relative to the PolyQuant root directory
(e.g., data/sequences/train/shard_0000.parquet).

Usage:
    python scripts/fix_index_paths.py [--dry-run]

Options:
    --dry-run   Show what would be changed without modifying the file
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# Path to the index.parquet file (relative to this script's location)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INDEX_PATH = PROJECT_ROOT / "data" / "sequences" / "index.parquet"


def normalize_path_to_relative(abs_path: str) -> str:
    """
    Convert an absolute path to a relative path starting from the PolyQuant folder.

    Examples:
        C:\\Users\\nimro\\PolyQuant\\data\\sequences\\train\\shard_0000.parquet
        -> data/sequences/train/shard_0000.parquet

        E:\\Roy_Data\\Projects\\Technion\\deep-project\\PolyQuant\\data\\sequences\\test\\shard_0010.parquet
        -> data/sequences/test/shard_0010.parquet
    """
    # Normalize slashes
    path_str = abs_path.replace("\\", "/")

    # Find the PolyQuant marker in the path (case-insensitive search)
    pattern = r"(?i)polyquant[/\\]"
    match = re.search(pattern, path_str)

    if match:
        # Take everything after "PolyQuant/"
        relative_path = path_str[match.end():]
        return relative_path

    # If PolyQuant is not found, return the original path
    # (it might already be relative)
    return path_str


def fix_index_paths(index_path: Path, dry_run: bool = False) -> None:
    """
    Read the index.parquet file, convert paths to relative, and save.

    Args:
        index_path: Path to the index.parquet file
        dry_run: If True, only show what would be changed
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    print(f"Reading index from: {index_path}")
    table = pq.read_table(index_path)
    df = table.to_pandas()

    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "path" not in df.columns:
        raise ValueError("Column 'path' not found in index.parquet")

    # Show sample of original paths
    unique_paths = df["path"].unique()
    print(f"\nUnique paths ({len(unique_paths)} total):")
    for p in unique_paths[:5]:
        print(f"  {p}")
    if len(unique_paths) > 5:
        print(f"  ... and {len(unique_paths) - 5} more")

    # Convert paths
    original_paths = df["path"].tolist()
    new_paths = [normalize_path_to_relative(p) for p in original_paths]

    # Count how many paths were changed
    changed = sum(1 for o, n in zip(original_paths, new_paths) if o != n)
    print(f"\nPaths to be updated: {changed}/{len(original_paths)}")

    # Show sample of conversions
    unique_new = set(new_paths)
    print(f"\nConverted paths ({len(unique_new)} unique):")
    for p in list(unique_new)[:5]:
        print(f"  {p}")
    if len(unique_new) > 5:
        print(f"  ... and {len(unique_new) - 5} more")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Create backup
    backup_path = index_path.with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M%S}.parquet")
    print(f"\nCreating backup: {backup_path}")
    shutil.copy2(index_path, backup_path)

    # Update the DataFrame and write back
    df["path"] = new_paths
    new_table = pa.Table.from_pandas(df, preserve_index=False)

    print(f"Writing updated index to: {index_path}")
    pq.write_table(new_table, index_path, compression="zstd")

    print("\n[DONE] Index paths updated successfully.")
    print(f"Backup saved to: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert absolute paths in index.parquet to relative paths."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying the file",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=INDEX_PATH,
        help=f"Path to index.parquet (default: {INDEX_PATH})",
    )
    args = parser.parse_args()

    fix_index_paths(args.index_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
