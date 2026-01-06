from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------
# CONFIG
# ----------------------------
FEATURE_GLOB = r"C:\Users\nimro\PolyQuant\data\features\features_chunk_*.parquet"
META_PATH    = r"C:\Users\nimro\PolyQuant\data\market_meta.parquet"
OUT_DIR      = Path(r"C:\Users\nimro\PolyQuant\data\sequences")

# Sharding (keep file count modest, ensure each market maps to exactly one shard)
NUM_SHARDS = {"train": 256, "val": 64, "test": 64}

# Deterministic-ish in DuckDB: user_hash = abs(hash(user_id)) % USER_VOCAB
# If you need strict cross-version determinism, we can switch to a Python crc32 rewrite later.
USER_VOCAB = 2_000_000

ROW_GROUP_SIZE = 16_384
BATCH_SIZE_INDEX = 200_000

# Feature transforms
DP_CLIP = 0.2
PNL_ASINH_SCALE = 50.0


# ----------------------------
# Columns you requested
# ----------------------------
# meta (kept minimal; trade_uid and raw user_id are intentionally NOT kept in the training store)
# because strings slow everything down. Keep them in your original features if you need debugging.
#
# trade data (sequence token):
#   p_yes, dp_yes, dt_trade, log_usdc_size, user_hash
#
# user data:
#   user_recent_pnl_last20, user_avg_size_before, user_days_active_before,
#   user_historical_pnl_before, user_historical_winrate_before, user_pnl_std_before

# Output schema (all numeric except market_id):
# market_id, timestamp,
# p_yes, dp_yes_clip, log_dt, log_usdc_size, user_hash,
# user_recent_pnl_asinh, user_avg_size_log, user_days_active_log,
# user_hist_pnl_asinh, user_hist_winrate, user_pnl_std_log,
# y

# ----------------------------
# Phase 1: shard to temp
# ----------------------------
def shard_to_temp(tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA enable_progress_bar=true;")

    # Register meta once
    con.execute(f"CREATE TEMP VIEW meta AS SELECT market_id, split, y FROM read_parquet('{META_PATH}');")

    for split, nshards in NUM_SHARDS.items():
        out_split_tmp = tmp_dir / split
        out_split_tmp.mkdir(parents=True, exist_ok=True)

        # We write partitioned-by-shard into tmp_dir/split/shard=K/part-*.parquet
        # Then phase 2 consolidates each shard folder into one file.
        query = f"""
        COPY (
          SELECT
            t.market_id,
            t.timestamp,

            CAST(t.p_yes AS REAL) AS p_yes,

            -- clip dp_yes
            CAST(GREATEST(-{DP_CLIP}, LEAST({DP_CLIP}, t.dp_yes)) AS REAL) AS dp_yes_clip,

            -- log_dt = log1p(max(dt_trade,0))
            CAST(LN(1 + GREATEST(t.dt_trade, 0)) AS REAL) AS log_dt,

            CAST(t.log_usdc_size AS REAL) AS log_usdc_size,

            -- user hash for embedding
            (ABS(hash(t.user_id)) % {USER_VOCAB})::UBIGINT AS user_hash,

            -- user stats transforms
            CAST(ASINH(t.user_recent_pnl_last20 / {PNL_ASINH_SCALE}) AS REAL) AS user_recent_pnl_asinh,
            CAST(LN(1 + GREATEST(t.user_avg_size_before, 0)) AS REAL) AS user_avg_size_log,
            CAST(LN(1 + GREATEST(t.user_days_active_before, 0)) AS REAL) AS user_days_active_log,

            CAST(ASINH(t.user_historical_pnl_before / {PNL_ASINH_SCALE}) AS REAL) AS user_hist_pnl_asinh,
            CAST(GREATEST(0, LEAST(1, t.user_historical_winrate_before)) AS REAL) AS user_hist_winrate,
            CAST(LN(1 + GREATEST(t.user_pnl_std_before, 0)) AS REAL) AS user_pnl_std_log,

            m.y AS y,

            -- shard by market_id so a market never spans shards
            (ABS(hash(t.market_id)) % {nshards})::INTEGER AS shard

          FROM read_parquet('{FEATURE_GLOB}') t
          JOIN meta m USING (market_id)
          WHERE m.split = '{split}'

          -- critical: makes each market contiguous after consolidation
          ORDER BY t.market_id, t.timestamp
        )
        TO '{str(out_split_tmp)}'
        (FORMAT PARQUET,
         PARTITION_BY (shard),
         COMPRESSION 'ZSTD',
         ROW_GROUP_SIZE {ROW_GROUP_SIZE});
        """
        con.execute(query)

    con.close()


# ----------------------------
# Phase 2: consolidate each shard into one parquet
# ----------------------------
def consolidate_shards(tmp_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA enable_progress_bar=true;")

    for split, nshards in NUM_SHARDS.items():
        (out_dir / split).mkdir(parents=True, exist_ok=True)

        for shard in range(nshards):
            shard_folder = tmp_dir / split / f"shard={shard}"
            if not shard_folder.exists():
                continue

            src_glob = str(shard_folder / "*.parquet")
            out_path = out_dir / split / f"shard_{shard:04d}.parquet"

            # merge parts -> one shard file, keep sorted order
            con.execute(f"""
            COPY (
              SELECT *
              FROM read_parquet('{src_glob}')
              ORDER BY market_id, timestamp
            )
            TO '{str(out_path)}'
            (FORMAT PARQUET,
             COMPRESSION 'ZSTD',
             ROW_GROUP_SIZE {ROW_GROUP_SIZE});
            """)

        print(f"[INFO] consolidated split={split}")

    con.close()


# ----------------------------
# Phase 3: build index.parquet (streaming)
# ----------------------------
def build_index(out_dir: Path) -> None:
    rows = {
        "split": [],
        "market_id": [],
        "path": [],
        "start": [],
        "length": [],
        "min_ts": [],
        "max_ts": [],
        "y": [],
    }

    for split in NUM_SHARDS.keys():
        split_dir = out_dir / split
        shard_files = sorted(split_dir.glob("shard_*.parquet"))
        print(f"[INFO] indexing split={split} shards={len(shard_files)}")

        for shp in shard_files:
            pf = pq.ParquetFile(shp)

            # stream columns needed for index
            prev_mid = None
            prev_y = None
            run_start_global = 0
            run_len = 0
            run_min_ts = None
            run_max_ts = None
            global_row = 0

            for batch in pf.iter_batches(batch_size=BATCH_SIZE_INDEX, columns=["market_id", "timestamp", "y"]):
                t = pa.Table.from_batches([batch])
                mids = t["market_id"].to_pylist()
                tss = t["timestamp"].to_pylist()
                ys = t["y"].to_pylist()

                for mid, ts, y in zip(mids, tss, ys):
                    if prev_mid is None:
                        prev_mid = mid
                        prev_y = y
                        run_start_global = global_row
                        run_len = 1
                        run_min_ts = ts
                        run_max_ts = ts
                    elif mid == prev_mid:
                        run_len += 1
                        if ts < run_min_ts:
                            run_min_ts = ts
                        if ts > run_max_ts:
                            run_max_ts = ts
                    else:
                        # flush previous market run
                        rows["split"].append(split)
                        rows["market_id"].append(str(prev_mid))
                        rows["path"].append(str(shp))
                        rows["start"].append(int(run_start_global))
                        rows["length"].append(int(run_len))
                        rows["min_ts"].append(int(run_min_ts))
                        rows["max_ts"].append(int(run_max_ts))
                        rows["y"].append(int(prev_y))

                        # start new run
                        prev_mid = mid
                        prev_y = y
                        run_start_global = global_row
                        run_len = 1
                        run_min_ts = ts
                        run_max_ts = ts

                    global_row += 1

            # flush last run
            if prev_mid is not None and run_len > 0:
                rows["split"].append(split)
                rows["market_id"].append(str(prev_mid))
                rows["path"].append(str(shp))
                rows["start"].append(int(run_start_global))
                rows["length"].append(int(run_len))
                rows["min_ts"].append(int(run_min_ts))
                rows["max_ts"].append(int(run_max_ts))
                rows["y"].append(int(prev_y))

    index_path = out_dir / "index.parquet"
    pq.write_table(pa.table(rows), index_path, compression="zstd")
    print(f"[DONE] wrote index: {index_path} rows={len(rows['market_id'])}")


def main():
    tmp_dir = OUT_DIR / "_tmp_sharded"
    final_dir = OUT_DIR

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 1) shard to temp
    print("[STEP 1] shard to temp")
    shard_to_temp(tmp_dir)

    # 2) consolidate shards into /train /val /test
    print("[STEP 2] consolidate shards")
    consolidate_shards(tmp_dir, final_dir)

    # remove temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 3) build index
    print("[STEP 3] build index")
    build_index(final_dir)


if __name__ == "__main__":
    main()
