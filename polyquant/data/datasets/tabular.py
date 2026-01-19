from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from polyquant.data.normalize import FeatureScaler
from polyquant.config import load_paths
from torch.utils.data import DataLoader

BATCH_SIZE = 16384
NUM_WORKERS = 4

PATHS = load_paths(__file__)
DATASET_ROOT = PATHS.dataset_root
SCALER_PATH = PATHS.scaler_path
RUNS_DIR = PATHS.runs_dir
CKPT_ROOT = PATHS.checkpoints_dir

@dataclass(frozen=True)
class TabularBatch:
    x: torch.Tensor       # [B, F] float32
    y: torch.Tensor       # [B] float32 (0/1)
    price: torch.Tensor   # [B] float32
    edge: torch.Tensor    # [B] float32
    log_usdc_size: torch.Tensor  # [B] float32

class TabularParquetIterable(IterableDataset):
    def __init__(
        self,
        split_dir,
        feature_cols,
        scaler: Optional[FeatureScaler],
        batch_size: int,
        shuffle_files: bool = False,
        shuffle_rowgroup: bool = False,   # kept for compatibility, but less relevant now
        seed: int = 123,
        shuffle_buffer: int = 500_000,
    ):
        """
        Iterable over shuffled trade-level batches from parquet files.

        - Files can be shuffled (shuffle_files).
        - Trades are globally shuffled via an in-memory buffer of size `shuffle_buffer`.
        """
        super().__init__()
        self.split_dir = Path(split_dir)
        self.feature_cols = list(feature_cols)
        self.scaler = scaler
        self.batch_size = int(batch_size)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_rowgroup = bool(shuffle_rowgroup)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)

        self.files = sorted(self.split_dir.glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found in: {self.split_dir}")

        # Required non-feature columns for training/eval
        self.label_col = "label_y"
        self.price_col = "price"
        self.edge_col = "edge"
        self.size_col = "log_usdc_size"
        self.outcome_index_col = "outcome_index"

    def _iter_files_for_worker(self) -> List[Path]:
        files = list(self.files)

        wi = get_worker_info()
        base_seed = self.seed
        if wi is not None:
            base_seed += wi.id  # different order per worker

        if self.shuffle_files:
            rng = random.Random(base_seed)
            rng.shuffle(files)

        if wi is None:
            return files

        # shard files by worker id
        return files[wi.id :: wi.num_workers]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self._iter_files_for_worker()
        cols_to_read = self.feature_cols + [
            self.label_col,
            self.price_col,
            self.edge_col,
            self.size_col,
            self.outcome_index_col,
        ]

        wi = get_worker_info()
        base_seed = self.seed if wi is None else self.seed + wi.id
        rng = random.Random(base_seed)

        buffer: List[Dict[str, np.ndarray]] = []

        def maybe_flush_buffer(final: bool = False):
            """
            Shuffle the buffer and yield batches for roughly half of it
            (or all of it if final=True), keeping the rest to mix with future rows.
            """
            nonlocal buffer

            if not buffer:
                return

            rng.shuffle(buffer)

            if final:
                to_emit = len(buffer)
            else:
                # emit about half to keep memory bounded
                to_emit = len(buffer) // 2
                if to_emit == 0:
                    return

            # emit in batches
            i = 0
            while i < to_emit:
                batch_rows = buffer[i : min(i + self.batch_size, to_emit)]
                if not batch_rows:
                    break

                # stack into tensors
                x_list = [row["x"] for row in batch_rows]
                y_list = [row["y"] for row in batch_rows]
                p_list = [row["price"] for row in batch_rows]
                e_list = [row["edge"] for row in batch_rows]
                s_list = [row["log_usdc_size"] for row in batch_rows]
                o_list = [row["outcome_index"] for row in batch_rows]

                xb = torch.from_numpy(np.stack(x_list).astype(np.float32, copy=False))
                yb = torch.from_numpy(np.asarray(y_list, dtype=np.float32))
                pb = torch.from_numpy(np.asarray(p_list, dtype=np.float32))
                eb = torch.from_numpy(np.asarray(e_list, dtype=np.float32))
                sb = torch.from_numpy(np.asarray(s_list, dtype=np.float32))
                ob = torch.from_numpy(np.asarray(o_list, dtype=np.float32))

                yield {
                    "x": xb,
                    "y": yb,
                    "price": pb,
                    "edge": eb,
                    "log_usdc_size": sb,
                    "outcome_index": ob,
                }

                i += self.batch_size

            # keep tail of buffer (those not emitted)
            buffer = buffer[to_emit:]

        # main streaming loop
        for fp in files:
            pf = pq.ParquetFile(str(fp))
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=cols_to_read)

                # Build X matrix [n, F]
                xs = []
                for c in self.feature_cols:
                    arr = table.column(c).to_numpy(zero_copy_only=False)
                    xs.append(arr.astype(np.float32, copy=False))
                x = np.column_stack(xs).astype(np.float32, copy=False)

                if self.scaler is not None:
                    x = self.scaler.transform_np(x)

                # labels and aux
                y_raw = (
                    table.column(self.label_col)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.float32, copy=False)
                )
                y = (y_raw > 0.5).astype(np.float32, copy=False)

                price = (
                    table.column(self.price_col)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.float32, copy=False)
                )
                edge = (
                    table.column(self.edge_col)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.float32, copy=False)
                )
                log_usdc = (
                    table.column(self.size_col)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.float32, copy=False)
                )
                outcome_index = (
                    table.column(self.outcome_index_col)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.float32, copy=False)
                )

                n = x.shape[0]
                indices = np.arange(n)

                if self.shuffle_rowgroup and n > 1:
                    np.random.default_rng(base_seed + rg).shuffle(indices)

                # push rows into buffer
                for idx in indices:
                    row = {
                        "x": x[idx].copy(),  # 1D np.array
                        "y": y[idx],
                        "price": price[idx],
                        "edge": edge[idx],
                        "log_usdc_size": log_usdc[idx],
                        "outcome_index": outcome_index[idx],
                    }
                    buffer.append(row)

                    if len(buffer) >= self.shuffle_buffer:
                        # flush about half of buffer
                        for batch in maybe_flush_buffer(final=False):
                            yield batch

        # end of all files: flush everything
        for batch in maybe_flush_buffer(final=True):
            yield batch

def make_loaders(schema, scaler):
    train_ds = TabularParquetIterable(
        split_dir=DATASET_ROOT / "train",
        feature_cols=schema.feature_cols,
        scaler=scaler,
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        shuffle_rowgroup=True,
        seed=123,
        shuffle_buffer=500_000,
    )
    val_ds = TabularParquetIterable(
        split_dir=DATASET_ROOT / "val",
        feature_cols=schema.feature_cols,
        scaler=scaler,
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        shuffle_rowgroup=True,
        seed=456,
        shuffle_buffer=200_000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader