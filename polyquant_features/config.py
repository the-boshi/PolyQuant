from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def ts_utc(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


@dataclass(frozen=True)
class Config:
    db_path: Path
    output_dir: Path

    chunk_rows: int = 2_000_000
    checkpoint_every: int = 20_000_000

    # Start from 2023-01-01 UTC
    start_ts: int = ts_utc(2023, 1, 1)
