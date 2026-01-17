"""
Dataset for dual-encoder training with user history context.

Returns both:
- Market sequence: trades in the current market
- User sequence: user's historical trades (from user_sequences data)

For each sampled trade position, we look up the user's previous trades
and include them as additional context.
"""
from __future__ import annotations

import bisect
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


@dataclass(frozen=True)
class MarketRow:
    path: str
    start: int
    length: int
    y: int


def _resolve_path(base_path: str, relative_path: str) -> str:
    """Resolve relative path from base."""
    rel_p = Path(relative_path)
    if rel_p.is_absolute() and rel_p.exists():
        return str(rel_p)

    base_dir = Path(base_path).resolve().parent
    current = base_dir
    for _ in range(6):
        candidate = current / relative_path
        if candidate.exists():
            return str(candidate)
        current = current.parent

    return relative_path


class _ParquetFileLRU:
    """LRU cache for ParquetFile handles."""
    def __init__(self, capacity: int = 32):
        self.capacity = int(capacity)
        self.cache: OrderedDict[str, pq.ParquetFile] = OrderedDict()

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


class UserHistoryStore:
    """
    Manages access to user trade history.

    Uses:
    - user_trades.parquet: All trades sorted by (user_hash, timestamp)
    - user_index.parquet: Maps user_hash -> (start_row, num_trades)
    """

    def __init__(self, user_seq_dir: str, max_user_len: int = 64):
        self.user_seq_dir = Path(user_seq_dir)
        self.max_user_len = max_user_len

        # Load user index into memory (small)
        index_path = self.user_seq_dir / "user_index.parquet"
        idx_table = pq.read_table(index_path)

        self.user_hash_to_range: Dict[int, Tuple[int, int]] = {}
        for uh, start, count in zip(
            idx_table["user_hash"].to_pylist(),
            idx_table["start_row"].to_pylist(),
            idx_table["num_trades"].to_pylist(),
        ):
            self.user_hash_to_range[uh] = (start, count)

        # Memory-map or lazy-load user trades
        self.trades_path = self.user_seq_dir / "user_trades.parquet"
        self._pf: Optional[pq.ParquetFile] = None
        self._trades_cache: Optional[pa.Table] = None

        # Columns to read for user history (must match build_user_sequences.py output)
        self.user_cols = ["timestamp", "price", "log_usdc_size", "outcome", "edge"]

    def _load_trades(self):
        """Load full user trades table (one-time, ~1-2GB for large datasets)."""
        if self._trades_cache is None:
            # Read only needed columns
            self._trades_cache = pq.read_table(
                self.trades_path,
                columns=["user_hash", "timestamp"] + self.user_cols[1:],  # timestamp already included
            )
            # Also load timestamps for binary search
            self._timestamps = self._trades_cache["timestamp"].to_numpy()

    def get_user_history(
        self,
        user_hash: int,
        before_timestamp: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get user's trades before a given timestamp.

        Returns:
            features: (L_user, D_user) float32 array - [price, log_usdc_size, outcome, edge]
            mask: (L_user,) bool array
        """
        self._load_trades()

        # Check if user exists
        if user_hash not in self.user_hash_to_range:
            # Unknown user - return empty
            return (
                np.zeros((self.max_user_len, 4), dtype=np.float32),
                np.zeros((self.max_user_len,), dtype=np.bool_),
            )

        start_row, num_trades = self.user_hash_to_range[user_hash]
        end_row = start_row + num_trades

        # Binary search to find trades before the given timestamp
        user_timestamps = self._timestamps[start_row:end_row]
        insert_pos = bisect.bisect_left(user_timestamps, before_timestamp)

        if insert_pos == 0:
            # No trades before this timestamp
            return (
                np.zeros((self.max_user_len, 4), dtype=np.float32),
                np.zeros((self.max_user_len,), dtype=np.bool_),
            )

        # Take last max_user_len trades before this timestamp
        actual_start = start_row + max(0, insert_pos - self.max_user_len)
        actual_end = start_row + insert_pos
        actual_len = actual_end - actual_start

        # Slice the table
        user_slice = self._trades_cache.slice(actual_start, actual_len)

        # Extract features: price, log_usdc_size, outcome, edge
        features = np.stack([
            user_slice["price"].to_numpy().astype(np.float32),
            user_slice["log_usdc_size"].to_numpy().astype(np.float32),
            user_slice["outcome"].to_numpy().astype(np.float32),
            user_slice["edge"].to_numpy().astype(np.float32),
        ], axis=1)  # (actual_len, 4)

        # Pad to max_user_len (left-pad with zeros)
        pad = self.max_user_len - actual_len
        if pad > 0:
            features = np.concatenate([
                np.zeros((pad, 4), dtype=np.float32),
                features,
            ], axis=0)
            mask = np.zeros((self.max_user_len,), dtype=np.bool_)
            mask[pad:] = True
        else:
            mask = np.ones((self.max_user_len,), dtype=np.bool_)

        return features, mask


class DualEncoderDataset(torch.utils.data.Dataset):
    """
    Dataset for dual-encoder training.

    Returns:
        market_x: (L_market, D_market) float32 - market trade features
        market_mask: (L_market,) bool - valid market positions
        user_x: (L_user, D_user) float32 - user history features
        user_mask: (L_user,) bool - valid user history positions
        y: () int64 - market outcome
    """

    def __init__(
        self,
        index_path: str,
        user_seq_dir: str,
        split: str,
        L_market: int = 1024,
        L_user: int = 64,
        cap_trades: int = 4096,
        min_prefix: int = 20,
        pf_cache: int = 64,
    ):
        self.L_market = int(L_market)
        self.L_user = int(L_user)
        self.min_prefix = int(min_prefix)
        self.cap_trades = int(cap_trades) if cap_trades else max(4 * L_market, 8192)
        self.cap_trades = max(self.cap_trades, L_market)

        # Load market index
        idx = pq.read_table(index_path).to_pandas()
        idx = idx[idx["split"] == split].copy()
        idx = idx[idx["length"] >= 10].copy()

        if len(idx) == 0:
            raise ValueError(f"No markets found for split={split}")

        self.rows: List[MarketRow] = [
            MarketRow(
                path=_resolve_path(index_path, str(r.path)),
                start=int(r.start),
                length=int(r.length),
                y=int(r.y),
            )
            for r in idx.itertuples(index=False)
        ]

        # Market columns
        self.float_cols = [
            "p_yes", "dp_yes_clip", "log_dt", "log_usdc_size",
            "user_recent_pnl_asinh", "user_avg_size_log", "user_days_active_log",
            "user_hist_pnl_asinh", "user_hist_winrate", "user_pnl_std_log",
        ]
        self.int_cols = ["user_hash"]
        self.ts_col = ["timestamp"]
        self.cols = self.ts_col + self.float_cols + self.int_cols

        self._pf_cache_cap = int(pf_cache)
        self._pf = _ParquetFileLRU(capacity=self._pf_cache_cap)

        # User history store
        self._user_store = UserHistoryStore(user_seq_dir, max_user_len=L_user)

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_pf"] = None
        state["_user_store"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pf = _ParquetFileLRU(capacity=getattr(self, "_pf_cache_cap", 64))
        # Recreate user store - will reload lazily
        from polyquant.config import load_paths
        # Note: Need to handle this properly in production
        self._user_store = None

    @staticmethod
    def _col_np(tab: pa.Table, name: str, dtype) -> np.ndarray:
        arr = tab[name].combine_chunks().to_numpy(zero_copy_only=False)
        return arr.astype(dtype, copy=False)

    def _read_range(self, pf: pq.ParquetFile, start: int, length: int) -> pa.Table:
        return pf.read(columns=self.cols).slice(start, length)

    def __getitem__(self, i: int):
        row = self.rows[i]
        pf = self._pf.get(row.path)

        N_total = row.length
        D_market = len(self.float_cols)

        if N_total <= 0:
            return (
                torch.zeros(self.L_market, D_market, dtype=torch.float32),
                torch.zeros(self.L_market, dtype=torch.bool),
                torch.zeros(self.L_user, 4, dtype=torch.float32),
                torch.zeros(self.L_user, dtype=torch.bool),
                torch.tensor(row.y, dtype=torch.long),
            )

        # Sample a cutoff position
        lo = max(self.min_prefix - 1, 0)
        hi = N_total - 1
        n = random.randint(lo, hi) if hi >= lo else hi

        # Read market trades up to position n
        read_start_local = max(0, n - self.cap_trades + 1)
        read_end_local = n + 1
        read_len = read_end_local - read_start_local

        file_start = row.start + read_start_local
        tab = self._read_range(pf, file_start, read_len)

        # Extract market features
        floats = [self._col_np(tab, c, np.float32) for c in self.float_cols]
        x_np = np.stack(floats, axis=1).astype(np.float32, copy=False)

        # Get timestamp and user_hash of the LAST trade (for user history lookup)
        timestamps = self._col_np(tab, "timestamp", np.int64)
        user_hashes = self._col_np(tab, "user_hash", np.int64)

        # Take last L_market trades
        end = x_np.shape[0] - 1
        start = max(0, end - self.L_market + 1)
        x_np = x_np[start:end + 1]

        # Get user info for the last trade
        last_timestamp = int(timestamps[-1])
        last_user_hash = int(user_hashes[-1])

        # Pad market sequence
        pad = self.L_market - x_np.shape[0]
        if pad > 0:
            x_np = np.concatenate([np.zeros((pad, D_market), np.float32), x_np], axis=0)
            market_mask = np.zeros((self.L_market,), dtype=np.bool_)
            market_mask[pad:] = True
        else:
            market_mask = np.ones((self.L_market,), dtype=np.bool_)

        # Get user history (trades by this user before this timestamp)
        if self._user_store is not None:
            user_x, user_mask = self._user_store.get_user_history(
                last_user_hash, last_timestamp
            )
        else:
            # Fallback if user store not available
            user_x = np.zeros((self.L_user, 4), dtype=np.float32)
            user_mask = np.zeros((self.L_user,), dtype=np.bool_)

        return (
            torch.from_numpy(x_np),
            torch.from_numpy(market_mask),
            torch.from_numpy(user_x),
            torch.from_numpy(user_mask),
            torch.tensor(row.y, dtype=torch.long),
        )


class DualEncoderPerTradeDataset(torch.utils.data.Dataset):
    """
    Dataset that returns user history for EACH trade in the market sequence.

    This is more expensive but allows per-trade predictions with individual user context.

    Returns:
        market_x: (L_market, D_market) float32
        market_mask: (L_market,) bool
        user_x: (L_market, L_user, D_user) float32 - user history per trade position
        user_mask: (L_market, L_user) bool
        y: () int64
    """

    def __init__(
        self,
        index_path: str,
        user_seq_dir: str,
        split: str,
        L_market: int = 512,
        L_user: int = 32,
        cap_trades: int = 2048,
        min_prefix: int = 20,
        pf_cache: int = 64,
    ):
        # Note: Smaller defaults due to memory requirements
        self.L_market = int(L_market)
        self.L_user = int(L_user)
        self.min_prefix = int(min_prefix)
        self.cap_trades = int(cap_trades) if cap_trades else max(4 * L_market, 4096)

        # Load market index
        idx = pq.read_table(index_path).to_pandas()
        idx = idx[idx["split"] == split].copy()
        idx = idx[idx["length"] >= 10].copy()

        if len(idx) == 0:
            raise ValueError(f"No markets found for split={split}")

        self.rows: List[MarketRow] = [
            MarketRow(
                path=_resolve_path(index_path, str(r.path)),
                start=int(r.start),
                length=int(r.length),
                y=int(r.y),
            )
            for r in idx.itertuples(index=False)
        ]

        self.float_cols = [
            "p_yes", "dp_yes_clip", "log_dt", "log_usdc_size",
            "user_recent_pnl_asinh", "user_avg_size_log", "user_days_active_log",
            "user_hist_pnl_asinh", "user_hist_winrate", "user_pnl_std_log",
        ]
        self.int_cols = ["user_hash"]
        self.ts_col = ["timestamp"]
        self.cols = self.ts_col + self.float_cols + self.int_cols

        self._pf_cache_cap = int(pf_cache)
        self._pf = _ParquetFileLRU(capacity=self._pf_cache_cap)

        self._user_store = UserHistoryStore(user_seq_dir, max_user_len=L_user)

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_pf"] = None
        state["_user_store"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pf = _ParquetFileLRU(capacity=getattr(self, "_pf_cache_cap", 64))
        self._user_store = None

    @staticmethod
    def _col_np(tab: pa.Table, name: str, dtype) -> np.ndarray:
        arr = tab[name].combine_chunks().to_numpy(zero_copy_only=False)
        return arr.astype(dtype, copy=False)

    def _read_range(self, pf: pq.ParquetFile, start: int, length: int) -> pa.Table:
        return pf.read(columns=self.cols).slice(start, length)

    def __getitem__(self, i: int):
        row = self.rows[i]
        pf = self._pf.get(row.path)

        N_total = row.length
        D_market = len(self.float_cols)

        if N_total <= 0:
            return (
                torch.zeros(self.L_market, D_market, dtype=torch.float32),
                torch.zeros(self.L_market, dtype=torch.bool),
                torch.zeros(self.L_market, self.L_user, 4, dtype=torch.float32),
                torch.zeros(self.L_market, self.L_user, dtype=torch.bool),
                torch.tensor(row.y, dtype=torch.long),
            )

        # Sample cutoff
        lo = max(self.min_prefix - 1, 0)
        hi = N_total - 1
        n = random.randint(lo, hi) if hi >= lo else hi

        read_start_local = max(0, n - self.cap_trades + 1)
        read_len = n + 1 - read_start_local
        file_start = row.start + read_start_local
        tab = self._read_range(pf, file_start, read_len)

        # Extract features
        floats = [self._col_np(tab, c, np.float32) for c in self.float_cols]
        x_np = np.stack(floats, axis=1).astype(np.float32, copy=False)
        timestamps = self._col_np(tab, "timestamp", np.int64)
        user_hashes = self._col_np(tab, "user_hash", np.int64)

        # Take last L_market
        end = x_np.shape[0] - 1
        start_idx = max(0, end - self.L_market + 1)
        x_np = x_np[start_idx:end + 1]
        timestamps = timestamps[start_idx:end + 1]
        user_hashes = user_hashes[start_idx:end + 1]

        actual_len = x_np.shape[0]

        # Pad market
        pad = self.L_market - actual_len
        if pad > 0:
            x_np = np.concatenate([np.zeros((pad, D_market), np.float32), x_np], axis=0)
            timestamps = np.concatenate([np.zeros((pad,), np.int64), timestamps], axis=0)
            user_hashes = np.concatenate([np.zeros((pad,), np.int64), user_hashes], axis=0)
            market_mask = np.zeros((self.L_market,), dtype=np.bool_)
            market_mask[pad:] = True
        else:
            market_mask = np.ones((self.L_market,), dtype=np.bool_)

        # Get user history for EACH trade position
        user_x = np.zeros((self.L_market, self.L_user, 4), dtype=np.float32)
        user_mask = np.zeros((self.L_market, self.L_user), dtype=np.bool_)

        if self._user_store is not None:
            for pos in range(self.L_market):
                if market_mask[pos]:
                    ts = int(timestamps[pos])
                    uh = int(user_hashes[pos])
                    ux, um = self._user_store.get_user_history(uh, ts)
                    user_x[pos] = ux
                    user_mask[pos] = um

        return (
            torch.from_numpy(x_np),
            torch.from_numpy(market_mask),
            torch.from_numpy(user_x),
            torch.from_numpy(user_mask),
            torch.tensor(row.y, dtype=torch.long),
        )
