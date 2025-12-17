#!/usr/bin/env python
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ===========================
# CONFIG
# ===========================

ROOT = Path(__file__).resolve().parents[1]
IN_ROOT = ROOT / "data" / "features_dataset"
OUT_ROOT = ROOT / "data" / "features_dataset_downsampled"

PRICE_COL = "price"
TRADE_UID_COL = "trade_uid"

THRESH = 0.96
BUCKET_W = 0.01
MAX_PER_BUCKET = 250_000

# Streaming knobs
SCAN_BATCH_ROWS = 200_000     # pyarrow scanner batch size
WRITE_BATCH_ROWS = 250_000    # rows per written parquet file


# ===========================
# HELPERS
# ===========================

@dataclass(frozen=True)
class BucketSpec:
    lo: float
    hi: float  # exclusive
    name: str  # for logging


def make_buckets() -> Tuple[BucketSpec, ...]:
    # 1-cent buckets strictly above 0.96
    # (0.96,0.97],(0.97,0.98],... but we implement as [lo, hi) for simplicity.
    # We'll treat price==1.0 as belonging to the last bucket via clamping.
    buckets = []
    x = THRESH
    while x < 1.0 - 1e-12:
        lo = x
        hi = min(1.0, x + BUCKET_W)
        buckets.append(BucketSpec(lo=lo, hi=hi, name=f"{lo:.2f}-{hi:.2f}"))
        x = hi
    return tuple(buckets)


BUCKETS = make_buckets()


def bucket_index(price: float) -> int:
    # Only called when price > THRESH
    # Clamp price==1.0 into last bucket.
    p = min(max(price, THRESH + 1e-12), 1.0 - 1e-12)
    idx = int((p - THRESH) / BUCKET_W)
    if idx < 0:
        idx = 0
    if idx >= len(BUCKETS):
        idx = len(BUCKETS) - 1
    return idx


import hashlib
import struct

def stable_u01_from_trade_uid(uids):
    """
    Deterministic pseudo-random numbers in [0,1) from trade_uid strings.
    Uses blake2b (8-byte digest) -> uint64 -> float in [0,1).
    """
    out = []
    for s in uids:
        if s is None:
            s = ""
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
        u64 = struct.unpack("<Q", h)[0]
        out.append(u64 / 2**64)
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_metadata_files(in_root: Path, out_root: Path) -> None:
    for name in ["features_config.json", "market_splits.parquet", "market_splits_meta.json", "train_scaler.json"]:
        src = in_root / name
        if src.exists():
            dst = out_root / name
            shutil.copy2(src, dst)


# ===========================
# PASS 1: COUNT BUCKETS
# ===========================

def count_high_price_buckets(split_dir: Path) -> tuple[Dict[int, int], int]:
    dataset = ds.dataset(str(split_dir), format="parquet")
    scanner = dataset.scanner(columns=[PRICE_COL], batch_size=SCAN_BATCH_ROWS)

    counts = {i: 0 for i in range(len(BUCKETS))}
    total = 0
    total_hi = 0

    for batch in scanner.to_batches():
        prices = batch.column(0).to_numpy(zero_copy_only=False)
        total += len(prices)

        # filter high prices
        for p in prices:
            if p > THRESH:
                total_hi += 1
                counts[bucket_index(float(p))] += 1

    print(f"[COUNT] {split_dir.name}: total={total:,}, price>{THRESH}={total_hi:,}")
    for i, b in enumerate(BUCKETS):
        print(f"        bucket {i} ({b.name}): {counts[i]:,}")
    return counts, total


# ===========================
# PASS 2: FILTER + WRITE
# ===========================

def downsample_split(
    split_in: Path,
    split_out: Path,
    counts: Dict[int, int],
    max_per_bucket: int,
) -> None:

    ensure_dir(split_out)

    # keep probability per bucket
    keep_prob = {}
    for i in range(len(BUCKETS)):
        c = counts.get(i, 0)
        keep_prob[i] = 1.0 if c <= max_per_bucket else (max_per_bucket / float(c))
    print("[PROB] keep probabilities:")
    for i, b in enumerate(BUCKETS):
        print(f"       bucket {i} ({b.name}): p_keep={keep_prob[i]:.6f}")

    dataset = ds.dataset(str(split_in), format="parquet")
    # Read all columns (keep schema)
    scanner = dataset.scanner(batch_size=SCAN_BATCH_ROWS)

    writer_idx = 0
    buffer_tables = []
    buffer_rows = 0
    total_in = 0
    total_out = 0
    total_hi_out = 0

    for batch in scanner.to_batches():
        total_in += batch.num_rows

        # We need price + trade_uid to decide keep; then filter entire batch.
        price_arr = batch.column(batch.schema.get_field_index(PRICE_COL))
        uid_arr = batch.column(batch.schema.get_field_index(TRADE_UID_COL))

        prices = price_arr.to_numpy(zero_copy_only=False)
        uids = uid_arr.to_pylist()

        # deterministic random per row in [0,1)
        r = stable_u01_from_trade_uid(uids)

        keep_mask = []
        for p, rv in zip(prices, r):
            pf = float(p)
            if pf <= THRESH:
                keep_mask.append(True)
            else:
                bi = bucket_index(pf)
                kp = keep_prob[bi]
                # keep if rv < kp
                keep = (float(rv) < kp)
                keep_mask.append(keep)

        mask = pa.array(keep_mask, type=pa.bool_())
        out_batch = batch.filter(mask)

        if out_batch.num_rows > 0:
            # stats
            out_prices = out_batch.column(out_batch.schema.get_field_index(PRICE_COL)).to_numpy(zero_copy_only=False)
            total_out += out_batch.num_rows
            total_hi_out += int((out_prices > THRESH).sum())

            buffer_tables.append(out_batch)
            buffer_rows += out_batch.num_rows

            if buffer_rows >= WRITE_BATCH_ROWS:
                table = pa.Table.from_batches(buffer_tables)
                out_path = split_out / f"part_{writer_idx:05d}.parquet"
                pq.write_table(table, out_path)
                writer_idx += 1
                buffer_tables = []
                buffer_rows = 0

    # flush
    if buffer_rows > 0:
        table = pa.Table.from_batches(buffer_tables)
        out_path = split_out / f"part_{writer_idx:05d}.parquet"
        pq.write_table(table, out_path)

    print(f"[DONE] {split_in.name}: in={total_in:,} out={total_out:,} (high_price_out={total_hi_out:,})")


# ===========================
# MAIN
# ===========================

def main():
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"Input dataset root not found: {IN_ROOT}")

    ensure_dir(OUT_ROOT)
    copy_metadata_files(IN_ROOT, OUT_ROOT)

    # process splits
    for split in ["train", "val", "test"]:
        split_in = IN_ROOT / split
        split_out = OUT_ROOT / split
        if not split_in.exists():
            print(f"[SKIP] Missing split dir: {split_in}")
            continue

        counts, total_in = count_high_price_buckets(split_in)

        max_per_bucket = int(0.005 * total_in)   # 0.5%
        print(f"[INFO] {split} max_per_bucket = {max_per_bucket:,}")

        downsample_split(
            split_in,
            split_out,
            counts,
            max_per_bucket=max_per_bucket,
        )

    # write downsample meta
    meta = {
        "threshold": THRESH,
        "bucket_width": BUCKET_W,
        "max_per_bucket": MAX_PER_BUCKET,
        "buckets": [b.name for b in BUCKETS],
    }
    with (OUT_ROOT / "downsample_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] Wrote {OUT_ROOT / 'downsample_meta.json'}")


if __name__ == "__main__":
    main()
