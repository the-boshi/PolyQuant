#!/usr/bin/env python
"""
Build user trade history sequences.

For each trade in the dataset, we need access to that user's previous N trades
(across all markets). This script builds an index that maps:
  (user_hash, timestamp) -> user's previous K trades

Output structure:
  data/user_sequences/
    user_trades.parquet       # All trades sorted by (user_hash, timestamp)
    user_index.parquet        # Index: user_hash -> (start_row, num_trades)

At training time, for a trade by user U at time T:
1. Look up user U's row range in user_index
2. Binary search to find trades before time T
3. Take the last K trades as context
"""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------
# CONFIG
# ----------------------------
# Paths relative to project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Feature files are in data/features_dataset_downsampled/{train,val,test}/part_*.parquet
FEATURE_GLOBS = [
    str(_PROJECT_ROOT / "data" / "features_dataset_downsampled" / "train" / "part_*.parquet"),
    str(_PROJECT_ROOT / "data" / "features_dataset_downsampled" / "val" / "part_*.parquet"),
    str(_PROJECT_ROOT / "data" / "features_dataset_downsampled" / "test" / "part_*.parquet"),
]
OUT_DIR = _PROJECT_ROOT / "data" / "user_sequences"

# How many user trades to keep as context
USER_SEQ_LEN = 64

# User vocab for hashing
USER_VOCAB = 2_000_000

ROW_GROUP_SIZE = 65536


def build_user_trades(out_dir: Path) -> None:
    """
    Build a single parquet with all trades sorted by (user_hash, timestamp).

    Columns:
      - user_hash: int64
      - timestamp: int64
      - market_id: string (for debugging, can drop later)
      - price: float32 (p_yes)
      - log_usdc_size: float32
      - outcome: int8 (0 or 1, the market outcome)
      - edge: float32 (realized edge/pnl per share)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA enable_progress_bar=true;")

    out_path = out_dir / "user_trades.parquet"

    # Combine all splits into one query
    union_parts = " UNION ALL ".join([
        f"SELECT * FROM read_parquet('{glob}')" for glob in FEATURE_GLOBS
    ])

    # Build user trades table sorted by user then time
    query = f"""
    COPY (
        SELECT
            (ABS(hash(user_id)) % {USER_VOCAB})::BIGINT AS user_hash,
            timestamp::BIGINT AS timestamp,
            market_id,

            -- Trade features (simplified for user history)
            CAST(price AS REAL) AS price,
            CAST(log_usdc_size AS REAL) AS log_usdc_size,

            -- Outcome (label_y is the market outcome)
            CAST(COALESCE(label_y, -1) AS TINYINT) AS outcome,

            -- Edge (realized profit per share)
            CAST(COALESCE(edge, 0) AS REAL) AS edge

        FROM ({union_parts})
        ORDER BY user_hash, timestamp
    )
    TO '{str(out_path)}'
    (FORMAT PARQUET,
     COMPRESSION 'ZSTD',
     ROW_GROUP_SIZE {ROW_GROUP_SIZE});
    """

    print("[INFO] Building user_trades.parquet (sorted by user_hash, timestamp)...")
    con.execute(query)
    print(f"[DONE] Wrote {out_path}")

    con.close()


def build_user_index(out_dir: Path) -> None:
    """
    Build an index mapping user_hash -> (start_row, num_trades).

    This allows O(1) lookup of where a user's trades start in user_trades.parquet.
    """
    user_trades_path = out_dir / "user_trades.parquet"

    print("[INFO] Building user_index.parquet...")

    # Read user_hash column to build index
    pf = pq.ParquetFile(user_trades_path)

    index_data = {
        "user_hash": [],
        "start_row": [],
        "num_trades": [],
    }

    current_user = None
    current_start = 0
    current_count = 0
    global_row = 0

    for batch in pf.iter_batches(batch_size=500_000, columns=["user_hash"]):
        user_hashes = batch["user_hash"].to_pylist()

        for uh in user_hashes:
            if current_user is None:
                current_user = uh
                current_start = global_row
                current_count = 1
            elif uh == current_user:
                current_count += 1
            else:
                # Flush previous user
                index_data["user_hash"].append(current_user)
                index_data["start_row"].append(current_start)
                index_data["num_trades"].append(current_count)

                # Start new user
                current_user = uh
                current_start = global_row
                current_count = 1

            global_row += 1

    # Flush last user
    if current_user is not None:
        index_data["user_hash"].append(current_user)
        index_data["start_row"].append(current_start)
        index_data["num_trades"].append(current_count)

    # Write index
    index_path = out_dir / "user_index.parquet"
    pq.write_table(
        pa.table(index_data),
        index_path,
        compression="zstd",
    )

    print(f"[DONE] Wrote {index_path} with {len(index_data['user_hash']):,} unique users")


def main():
    print("[STEP 1] Build user_trades.parquet")
    build_user_trades(OUT_DIR)

    print("[STEP 2] Build user_index.parquet")
    build_user_index(OUT_DIR)

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
