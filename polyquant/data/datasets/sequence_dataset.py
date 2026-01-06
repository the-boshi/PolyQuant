from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


class MarketWindowDataset(torch.utils.data.Dataset):
    """
    Returns:
      x:    (L, D) float32
      u:    (L,)   int64 user_hash
      mask: (L,)   bool
      y:    ()     int64
    """

    def __init__(
        self,
        index_path: str,
        split: str,
        L: int = 512,
        cap_trades: int | None = None,
        min_prefix: int = 20,
        pf_cache: int = 64,
        include_ts: bool = False,
    ):
        self.L = int(L)
        self.min_prefix = int(min_prefix)
        self.cap_trades = int(cap_trades) if cap_trades is not None else max(4 * self.L, 8192)
        self.cap_trades = max(self.cap_trades, self.L)

        # Load index (one row per market)
        idx = pq.read_table(index_path).to_pandas()
        idx = idx[idx["split"] == split].copy()
        idx = idx[idx["length"] >= 10].copy()

        if len(idx) == 0:
            raise ValueError(f"No markets found for split={split} in {index_path}")

        self.rows: List[MarketRow] = [
            MarketRow(path=str(r.path), start=int(r.start), length=int(r.length), y=int(r.y))
            for r in idx.itertuples(index=False)
        ]

        # Columns stored in shard parquet (from our build script)
        # market_id,timestamp,p_yes,dp_yes_clip,log_dt,log_usdc_size,user_hash,
        # user_recent_pnl_asinh,user_avg_size_log,user_days_active_log,
        # user_hist_pnl_asinh,user_hist_winrate,user_pnl_std_log,y
        self.include_ts = bool(include_ts)

        self.float_cols = [
            "p_yes",
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
        self.int_cols = ["user_hash"]
        self.ts_col = ["timestamp"] if self.include_ts else []

        self.cols = self.ts_col + self.float_cols + self.int_cols  # y comes from index row

        self._pf_cache_cap = int(pf_cache)
        self._pf = _ParquetFileLRU(capacity=self._pf_cache_cap)

    def __len__(self) -> int:
        return len(self.rows)

    def _read_range(self, pf: pq.ParquetFile, start: int, length: int) -> pa.Table:
        # ParquetFile.read() then slice is ok because we wrote small row groups.
        return pf.read(columns=self.cols).slice(start, length)

    def __getstate__(self):
        # Called when DataLoader (spawn) pickles the Dataset for workers.
        state = dict(self.__dict__)
        # Drop non-picklable objects (pyarrow ParquetFile handles)
        state["_pf"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate per-worker cache lazily
        self._pf = _ParquetFileLRU(capacity=getattr(self, "_pf_cache_cap", 64))


    @staticmethod
    def _col_np(tab: pa.Table, name: str, dtype) -> np.ndarray:
        arr = tab[name].combine_chunks().to_numpy(zero_copy_only=False)
        return arr.astype(dtype, copy=False)

    def __getitem__(self, i: int):
        row = self.rows[i]
        pf = self._pf.get(row.path)

        N_total = row.length
        if N_total <= 0:
            D = len(self.float_cols) + (1 if self.include_ts else 0)
            return (
                torch.zeros(self.L, D, dtype=torch.float32),
                torch.zeros(self.L, dtype=torch.long),
                torch.zeros(self.L, dtype=torch.bool),
                torch.tensor(row.y, dtype=torch.long),
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
        tab = self._read_range(pf, file_start, read_len)

        # Convert columns -> numpy
        # Floats
        floats = [self._col_np(tab, c, np.float32) for c in (self.ts_col + self.float_cols)]
        x_np = np.stack(floats, axis=1).astype(np.float32, copy=False)

        # Ints
        u_np = self._col_np(tab, "user_hash", np.int64)

        # Take last L from the read window (ending at n)
        # tab corresponds to [read_start_local .. n], so the end is always the last row.
        end = x_np.shape[0] - 1
        start = max(0, end - self.L + 1)
        x_np = x_np[start : end + 1]
        u_np = u_np[start : end + 1]

        D = x_np.shape[1]
        pad = self.L - x_np.shape[0]
        if pad > 0:
            x_np = np.concatenate([np.zeros((pad, D), np.float32), x_np], axis=0)
            u_np = np.concatenate([np.zeros((pad,), np.int64), u_np], axis=0)
            mask = np.zeros((self.L,), dtype=bool)
            mask[pad:] = True
        else:
            mask = np.ones((self.L,), dtype=bool)

        return (
            torch.from_numpy(x_np),
            torch.from_numpy(u_np),
            torch.from_numpy(mask),
            torch.tensor(row.y, dtype=torch.long),
        )
