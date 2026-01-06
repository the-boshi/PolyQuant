from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def init_rows_buffer() -> dict[str, list]:
    # keep keys aligned with compute_row_features()
    return {
        "trade_uid": [],
        "user_id": [],
        "market_id": [],
        "timestamp": [],
        "edge": [],

        "price": [],
        "p_yes": [],
        "dp_yes": [],
        "dt_trade": [],
        
        "log_usdc_size": [],
        "outcome_index": [],
        "seconds_since_market_open": [],
        "time_of_day_sin": [],
        "time_of_day_cos": [],

        "user_time_since_first_trade": [],
        "market_time_since_first_trade": [],

        "user_total_trades_before": [],
        "user_total_volume_before": [],
        "user_days_active_before": [],
        "user_avg_size_before": [],
        "user_std_size_before": [],
        "user_historical_pnl_before": [],
        "user_historical_winrate_before": [],
        "user_recent_pnl_last20": [],
        "user_activity_burstiness": [],
        "user_pnl_std_before": [],
        "time_since_user_last_trade": [],

        "market_total_trades_before": [],
        "market_total_volume_before": [],
        "market_volume_last_1h": [],
        "num_unique_traders_last_1h": [],
        "market_volatility_1h": [],
        "mean_trade_size_1h": [],
        "size_ratio_market": [],
        "last_price_yes": [],
        "last_price_no": [],
        "mid_price": [],
        "price_deviation": [],
        "signed_price_deviation": [],
        "time_since_last_market_trade": [],

        "user_usdc_last_1h_in_market": [],
        "user_trades_last_1h_in_market": [],
        "user_share_of_volume_1h_in_market": [],

        "user_total_usdc_in_market": [],
        "user_trade_count_in_market": [],
        "user_share_of_volume_in_market": [],
        "time_since_user_last_trade_in_market": [],
    }


def append_row(rows: dict[str, list], feats: dict[str, Any]):
    for k, v in feats.items():
        rows[k].append(v)


def write_parquet_chunk(output_dir: Path, rows: dict[str, list], chunk_idx: int):
    if not rows["trade_uid"]:
        return
    table = pa.table(rows)
    out_path = output_dir / f"features_chunk_{chunk_idx:05d}.parquet"
    pq.write_table(table, out_path)
    print(f"[INFO] Wrote {len(rows['trade_uid'])} rows to {out_path}")


@dataclass
class Checkpoint:
    trades_processed: int
    chunk_idx: int
    last_ts: int
    last_trade_uid: str

    user_states: Any
    market_states: Any
    user_market_lifetime: Any

    pending_pnls: Any
    pending_seq: int


def save_checkpoint(output_dir: Path, ckpt: Checkpoint):
    ckpt_path = output_dir / f"state_checkpoint_{ckpt.trades_processed:09d}.pkl"
    with ckpt_path.open("wb") as f:
        pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved checkpoint at {ckpt_path}")


def load_latest_checkpoint(output_dir: Path) -> Optional[Checkpoint]:
    ckpts = sorted(output_dir.glob("state_checkpoint_*.pkl"))
    if not ckpts:
        return None
    latest = ckpts[-1]
    with latest.open("rb") as f:
        obj = pickle.load(f)
    print(f"[INFO] Loaded checkpoint {latest}")
    return obj
