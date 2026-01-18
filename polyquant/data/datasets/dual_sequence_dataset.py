from __future__ import annotations

import bisect
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import torch


@dataclass(frozen=True)
class MarketRow:
    path: str
    start: int
    length: int
    y: int
    max_ts: int


@dataclass(frozen=True)
class UserRow:
    path: str
    start: int
    length: int


def _resolve_shard_path(index_path: str, shard_path: str) -> str:
    """
    Resolve shard path relative to the index.parquet location.

    If shard_path is absolute, return it as-is.
    If shard_path is relative (e.g., 'data/sequences/train/shard_0000.parquet'),
    resolve it relative to the project root (parent of index file's 'data' folder).
    """
    shard_p = Path(shard_path)

    if shard_p.is_absolute() and shard_p.exists():
        return str(shard_p)

    index_dir = Path(index_path).resolve().parent

    candidate = index_dir / shard_path
    if candidate.exists():
        return str(candidate)

    current = index_dir
    for _ in range(5):
        project_root = current
        candidate = project_root / shard_path
        if candidate.exists():
            return str(candidate)
        current = current.parent

    return shard_path


class _ParquetFileLRU:
    """Tiny LRU cache for ParquetFile handles."""
    def __init__(self, capacity: int = 32):
        self.capacity = int(capacity)
        self.cache: "OrderedDict[str, pq.ParquetFile]" = OrderedDict()

    def get(self, path: str) -> pq.ParquetFile:
        pf = self.cache.get(path)
        if pf is not None:
            self.cache.move_to_end(path)
            return pf
        pf = pq.ParquetFile(path)
        self.cache[path] = pf
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return pf


class DualSequenceDataset(torch.utils.data.Dataset):
    """
    Dual-encoder dataset with market and user sequences.

    Returns:
      market_x: (L_market, D_market) float32
      market_mask: (L_market,) bool
      market_y: (L_market,) float32 (masked to 0.5 before resolution)
      user_x: (L_user, D_user) float32
      user_mask: (L_user,) bool
      user_y: (L_user,) float32 (masked to 0.5 before resolution)
    """

    def __init__(
        self,
        market_index_path: str,
        user_index_path: str,
        split: str,
        L_market: int = 1024,
        L_user: int = 64,
        cap_trades: int | None = None,
        min_prefix: int = 20,
        pf_cache: int = 64,
    ):
        self.L_market = int(L_market)
        self.L_user = int(L_user)
        self.min_prefix = int(min_prefix)
        self.cap_trades = int(cap_trades) if cap_trades is not None else max(4 * self.L_market, 8192)
        self.cap_trades = max(self.cap_trades, self.L_market)

        # Load market index
        market_idx = pq.read_table(market_index_path).to_pandas()
        market_idx = market_idx[market_idx["split"] == split].copy()
        market_idx = market_idx[market_idx["length"] >= 10].copy()

        if len(market_idx) == 0:
            raise ValueError(f"No markets found for split={split} in {market_index_path}")

        self.market_rows: List[MarketRow] = [
            MarketRow(
                path=_resolve_shard_path(market_index_path, str(r.path)),
                start=int(r.start),
                length=int(r.length),
                y=int(r.y),
                max_ts=int(r.max_ts),
            )
            for r in tqdm(market_idx.itertuples(index=False), total=len(market_idx), desc=f"Loading {split} markets")
        ]

        # Load user index
        user_idx = pq.read_table(user_index_path).to_pandas()
        user_idx = user_idx[user_idx["split"] == split].copy()

        self.user_rows: Dict[str, UserRow] = {}
        for r in tqdm(user_idx.itertuples(index=False), total=len(user_idx), desc=f"Loading {split} users"):
            self.user_rows[str(r.user_id)] = UserRow(
                path=_resolve_shard_path(user_index_path, str(r.path)),
                start=int(r.start),
                length=int(r.length),
            )

        # Market columns (must exist in sequence shards)
        self.market_float_cols = [
            "price",
            "p_yes",
            "outcome_index",
            "dp_yes_clip",
            "log_dt",
            "log_usdc_size",
            "user_recent_pnl_asinh",
            "user_avg_size_log",
            "user_days_active_log",
            "user_hist_pnl_asinh",
            "user_hist_winrate",
            "user_pnl_std_log",
        ]
        self.market_ts_col = ["timestamp"]
        self.market_user_col = ["user_id"]
        self.market_cols = self.market_ts_col + self.market_user_col + self.market_float_cols

        # User columns (from user_sequences_store shards)
        self.user_float_cols = ["price", "p_yes", "outcome_index"]
        self.user_ts_col = ["timestamp"]
        self.user_y_col = ["y"]
        self.user_resolve_col = ["market_resolve_time"]
        self.user_cols = self.user_ts_col + self.user_float_cols + self.user_y_col + self.user_resolve_col

        self._pf_cache_cap = int(pf_cache)
        self._market_pf = _ParquetFileLRU(capacity=self._pf_cache_cap)
        self._user_pf = _ParquetFileLRU(capacity=self._pf_cache_cap)

    def __len__(self) -> int:
        return len(self.market_rows)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_market_pf"] = None
        state["_user_pf"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._market_pf = _ParquetFileLRU(capacity=getattr(self, "_pf_cache_cap", 64))
        self._user_pf = _ParquetFileLRU(capacity=getattr(self, "_pf_cache_cap", 64))

    @staticmethod
    def _col_np(tab: pa.Table, name: str, dtype) -> np.ndarray:
        arr = tab[name].combine_chunks().to_numpy(zero_copy_only=False)
        return arr.astype(dtype, copy=False)

    def _read_market_range(self, pf: pq.ParquetFile, start: int, length: int) -> pa.Table:
        return pf.read(columns=self.market_cols).slice(start, length)

    def _read_user_range(self, pf: pq.ParquetFile, start: int, length: int) -> pa.Table:
        return pf.read(columns=self.user_cols).slice(start, length)

    def __getitem__(self, i: int):
        row = self.market_rows[i]
        pf = self._market_pf.get(row.path)

        N_total = row.length
        D_market = len(self.market_float_cols)
        D_user = len(self.user_float_cols)

        if N_total <= 0:
            return (
                torch.zeros(self.L_market, D_market, dtype=torch.float32),
                torch.zeros(self.L_market, dtype=torch.bool),
                torch.full((self.L_market,), 0.5, dtype=torch.float32),
                torch.zeros(self.L_user, D_user, dtype=torch.float32),
                torch.zeros(self.L_user, dtype=torch.bool),
                torch.full((self.L_user,), 0.5, dtype=torch.float32),
            )

        # Sample cutoff n in [min_prefix-1, N_total-1]
        lo = max(self.min_prefix - 1, 0)
        hi = N_total - 1
        n = random.randint(lo, hi) if hi >= lo else hi

        # Read at most cap_trades ending at n (market-local indices)
        read_start_local = max(0, n - self.cap_trades + 1)
        read_end_local = n + 1
        read_len = read_end_local - read_start_local

        # Convert to file-global offsets
        file_start = row.start + read_start_local
        tab = self._read_market_range(pf, file_start, read_len)

        # Convert market columns -> numpy
        floats = [self._col_np(tab, c, np.float32) for c in self.market_float_cols]
        x_np = np.stack(floats, axis=1).astype(np.float32, copy=False)
        timestamps = self._col_np(tab, "timestamp", np.int64)
        user_ids = tab["user_id"].to_pylist()

        # Take last L_market from the read window (ending at n)
        end = x_np.shape[0] - 1
        start = max(0, end - self.L_market + 1)
        x_np = x_np[start : end + 1]
        timestamps = timestamps[start : end + 1]
        user_ids = user_ids[start : end + 1]

        # y masking based on market resolve time
        y_val = float(row.y)
        y_market = np.where(timestamps < row.max_ts, 0.5, y_val).astype(np.float32, copy=False)

        # User lookup based on last trade in the market window
        last_timestamp = int(timestamps[-1])
        last_user_id = str(user_ids[-1])

        # Pad market sequence
        pad = self.L_market - x_np.shape[0]
        if pad > 0:
            x_np = np.concatenate([np.zeros((pad, D_market), np.float32), x_np], axis=0)
            y_market = np.concatenate([np.full((pad,), 0.5, np.float32), y_market], axis=0)
            market_mask = np.zeros((self.L_market,), dtype=np.bool_)
            market_mask[pad:] = True
        else:
            market_mask = np.ones((self.L_market,), dtype=np.bool_)

        # Build user sequence
        user_x = np.zeros((self.L_user, D_user), dtype=np.float32)
        user_y = np.full((self.L_user,), 0.5, dtype=np.float32)
        user_mask = np.zeros((self.L_user,), dtype=np.bool_)

        user_row = self.user_rows.get(last_user_id)
        if user_row is not None and user_row.length > 0:
            user_pf = self._user_pf.get(user_row.path)
            user_tab = self._read_user_range(user_pf, user_row.start, user_row.length)

            user_ts = self._col_np(user_tab, "timestamp", np.int64)
            insert_pos = bisect.bisect_left(user_ts, last_timestamp)

            if insert_pos > 0:
                actual_start = max(0, insert_pos - self.L_user)
                actual_end = insert_pos
                user_slice = user_tab.slice(actual_start, actual_end - actual_start)

                ux = np.stack(
                    [self._col_np(user_slice, c, np.float32) for c in self.user_float_cols],
                    axis=1,
                ).astype(np.float32, copy=False)
                uts = self._col_np(user_slice, "timestamp", np.int64)
                uy = self._col_np(user_slice, "y", np.int64).astype(np.float32, copy=False)
                resolve_ts = self._col_np(user_slice, "market_resolve_time", np.int64)
                uy_masked = np.where(uts < resolve_ts, 0.5, uy).astype(np.float32, copy=False)

                pad_u = self.L_user - ux.shape[0]
                if pad_u > 0:
                    user_x = np.concatenate([np.zeros((pad_u, D_user), np.float32), ux], axis=0)
                    user_y = np.concatenate([np.full((pad_u,), 0.5, np.float32), uy_masked], axis=0)
                    user_mask = np.zeros((self.L_user,), dtype=np.bool_)
                    user_mask[pad_u:] = True
                else:
                    user_x = ux
                    user_y = uy_masked
                    user_mask = np.ones((self.L_user,), dtype=np.bool_)

        return (
            torch.from_numpy(x_np),
            torch.from_numpy(market_mask),
            torch.from_numpy(y_market),
            torch.from_numpy(user_x),
            torch.from_numpy(user_mask),
            torch.from_numpy(user_y),
        )
