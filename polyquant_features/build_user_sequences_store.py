from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------
# CONFIG
# ----------------------------
FEATURE_GLOB = r"C:\Users\nimro\PolyQuant\data\features\features_chunk_*.parquet"
INDEX_PATH = r"C:\Users\nimro\PolyQuant\data\sequences\index.parquet"
OUT_DIR = Path(r"C:\Users\nimro\PolyQuant\data\user_sequences_store")
PROJECT_ROOT = OUT_DIR.parent.parent

# Sharding (user_id -> shard)
NUM_SHARDS = {"train": 256, "val": 64, "test": 64}

ROW_GROUP_SIZE = 16_384
BATCH_SIZE_INDEX = 200_000


# ----------------------------
# Phase 1: shard to temp
# ----------------------------
def shard_to_temp(tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA enable_progress_bar=true;")

    # Index provides split, market outcome, and resolve time (max_ts)
    con.execute(
        f"""
        CREATE TEMP VIEW meta AS
        SELECT market_id, split, y, max_ts
        FROM read_parquet('{INDEX_PATH}');
        """
    )

    for split, nshards in NUM_SHARDS.items():
        out_split_tmp = tmp_dir / split
        out_split_tmp.mkdir(parents=True, exist_ok=True)

        query = f"""
        COPY (
          SELECT
            t.user_id,
            t.timestamp,
            t.market_id,
            CAST(t.price AS REAL) AS price,
            CAST(t.p_yes AS REAL) AS p_yes,
            CAST(t.outcome_index AS TINYINT) AS outcome_index,
            CAST(m.y AS TINYINT) AS y,
            CAST(m.max_ts AS BIGINT) AS market_resolve_time,

            -- time since user's previous trade (seconds)
            CAST(COALESCE(t.timestamp - LAG(t.timestamp) OVER (
                PARTITION BY t.user_id ORDER BY t.timestamp
            ), 0) AS BIGINT) AS dt_user,


            -- shard by user_id so a user's trades never span shards
            (ABS(hash(t.user_id)) % {nshards})::INTEGER AS shard

          FROM read_parquet('{FEATURE_GLOB}') t
          JOIN meta m USING (market_id)
          WHERE m.split = '{split}'

          -- critical: makes each user contiguous after consolidation
          ORDER BY t.user_id, t.timestamp
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

            con.execute(
                f"""
                COPY (
                  SELECT *
                  FROM read_parquet('{src_glob}')
                  ORDER BY user_id, timestamp
                )
                TO '{str(out_path)}'
                (FORMAT PARQUET,
                 COMPRESSION 'ZSTD',
                 ROW_GROUP_SIZE {ROW_GROUP_SIZE});
                """
            )

        print(f"[INFO] consolidated split={split}")

    con.close()


# ----------------------------
# Phase 3: build user index.parquet (streaming)
# ----------------------------
def build_index(out_dir: Path) -> None:
    rows = {
        "split": [],
        "user_id": [],
        "path": [],
        "start": [],
        "length": [],
        "min_ts": [],
        "max_ts": [],
    }

    for split in NUM_SHARDS.keys():
        split_dir = out_dir / split
        shard_files = sorted(split_dir.glob("shard_*.parquet"))
        print(f"[INFO] indexing split={split} shards={len(shard_files)}")

        for shp in shard_files:
            pf = pq.ParquetFile(shp)

            prev_uid = None
            run_start_global = 0
            run_len = 0
            run_min_ts = None
            run_max_ts = None
            global_row = 0

            for batch in pf.iter_batches(
                batch_size=BATCH_SIZE_INDEX,
                columns=["user_id", "timestamp"],
            ):
                t = pa.Table.from_batches([batch])
                uids = t["user_id"].to_pylist()
                tss = t["timestamp"].to_pylist()

                for uid, ts in zip(uids, tss):
                    if prev_uid is None:
                        prev_uid = uid
                        run_start_global = global_row
                        run_len = 1
                        run_min_ts = ts
                        run_max_ts = ts
                    elif uid == prev_uid:
                        run_len += 1
                        if ts < run_min_ts:
                            run_min_ts = ts
                        if ts > run_max_ts:
                            run_max_ts = ts
                    else:
                        rows["split"].append(split)
                        rows["user_id"].append(str(prev_uid))
                        rel_path = Path(shp).resolve().relative_to(PROJECT_ROOT).as_posix()
                        rows["path"].append(rel_path)
                        rows["start"].append(int(run_start_global))
                        rows["length"].append(int(run_len))
                        rows["min_ts"].append(int(run_min_ts))
                        rows["max_ts"].append(int(run_max_ts))

                        prev_uid = uid
                        run_start_global = global_row
                        run_len = 1
                        run_min_ts = ts
                        run_max_ts = ts

                    global_row += 1

            if prev_uid is not None and run_len > 0:
                rows["split"].append(split)
                rows["user_id"].append(str(prev_uid))
                rel_path = Path(shp).resolve().relative_to(PROJECT_ROOT).as_posix()
                rows["path"].append(rel_path)
                rows["start"].append(int(run_start_global))
                rows["length"].append(int(run_len))
                rows["min_ts"].append(int(run_min_ts))
                rows["max_ts"].append(int(run_max_ts))

    index_path = out_dir / "index.parquet"
    pq.write_table(pa.table(rows), index_path, compression="zstd")
    print(f"[DONE] wrote index: {index_path} rows={len(rows['user_id'])}")


def main():
    tmp_dir = OUT_DIR / "_tmp_sharded"
    final_dir = OUT_DIR

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP 1] shard to temp")
    shard_to_temp(tmp_dir)

    print("[STEP 2] consolidate shards")
    consolidate_shards(tmp_dir, final_dir)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[STEP 3] build index")
    build_index(final_dir)


if __name__ == "__main__":
    main()
