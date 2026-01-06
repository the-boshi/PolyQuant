from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Tuple


def parse_iso_to_ts(s: str | None) -> int | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.astimezone(timezone.utc).timestamp())


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return {row[1] for row in cur.fetchall()}  # row[1] = name


def ensure_pragmas(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")


def ensure_indexes(conn: sqlite3.Connection):
    # for trades scan and effective_end_ts computation
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_condition_ts ON trades(condition_id, timestamp);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts_uid ON trades(timestamp, trade_uid);")
    # markets join
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_condition ON markets(condition_id);")
    conn.commit()


def ensure_effective_end_ts(conn: sqlite3.Connection):
    cols = table_columns(conn, "markets")
    if "effective_end_ts" not in cols:
        conn.execute("ALTER TABLE markets ADD COLUMN effective_end_ts INTEGER;")
        conn.commit()

    # fill NULLs from last trade timestamp
    conn.executescript("""
    DROP TABLE IF EXISTS tmp_market_last_trade;
    CREATE TEMP TABLE tmp_market_last_trade AS
    SELECT condition_id, MAX(timestamp) AS effective_end_ts
    FROM trades
    GROUP BY condition_id;

    CREATE INDEX IF NOT EXISTS idx_tmp_market_last_trade
    ON tmp_market_last_trade(condition_id);

    UPDATE markets
    SET effective_end_ts = (
      SELECT t.effective_end_ts
      FROM tmp_market_last_trade t
      WHERE t.condition_id = markets.condition_id
    )
    WHERE effective_end_ts IS NULL;
    """)
    conn.commit()


def load_market_start_ts(conn: sqlite3.Connection) -> Dict[str, Optional[int]]:
    cols = table_columns(conn, "markets")
    # accept different schemas
    if "start_date" in cols:
        start_col = "start_date"
        is_iso = True
    elif "start_time" in cols:
        start_col = "start_time"
        is_iso = False  # assume unix int
    else:
        start_col = None

    out: Dict[str, Optional[int]] = {}
    cur = conn.cursor()
    if start_col is None:
        cur.execute("SELECT condition_id FROM markets;")
        for (cid,) in cur.fetchall():
            out[cid] = None
        return out

    cur.execute(f"SELECT condition_id, {start_col} FROM markets;")
    for cid, v in cur.fetchall():
        if v is None:
            out[cid] = None
        else:
            out[cid] = parse_iso_to_ts(v) if is_iso else int(v)
    return out


def load_market_end_ts(conn: sqlite3.Connection) -> Dict[str, Optional[int]]:
    cols = table_columns(conn, "markets")

    # prefer effective_end_ts; fallback to end_date/end_time if present
    fallback_col = None
    fallback_is_iso = False
    if "end_date" in cols:
        fallback_col = "end_date"
        fallback_is_iso = True
    elif "end_time" in cols:
        fallback_col = "end_time"
        fallback_is_iso = False

    out: Dict[str, Optional[int]] = {}
    cur = conn.cursor()

    if fallback_col is None:
        cur.execute("SELECT condition_id, effective_end_ts FROM markets;")
        for cid, eff in cur.fetchall():
            out[cid] = int(eff) if eff is not None else None
        return out

    cur.execute(f"SELECT condition_id, effective_end_ts, {fallback_col} FROM markets;")
    for cid, eff, fb in cur.fetchall():
        if eff is not None:
            out[cid] = int(eff)
        elif fb is None:
            out[cid] = None
        else:
            out[cid] = parse_iso_to_ts(fb) if fallback_is_iso else int(fb)
    return out


def trades_cursor(
    conn: sqlite3.Connection,
    start_ts: int,
    resume_after: Optional[Tuple[int, str]] = None,
):
    """
    Iterates trades in (timestamp, trade_uid) order.

    resume_after = (last_ts, last_trade_uid) to continue strictly after it.
    """
    if resume_after is None:
        return conn.execute(
            """
            SELECT trade_uid, proxy_wallet, condition_id, timestamp, price, size, usdc_size, outcome_index, pnl_raw
            FROM trades
            WHERE timestamp >= ?
            ORDER BY timestamp, trade_uid;
            """,
            (start_ts,),
        )

    last_ts, last_uid = resume_after
    return conn.execute(
        """
        SELECT trade_uid, proxy_wallet, condition_id, timestamp, price, size, usdc_size, outcome_index, pnl_raw
        FROM trades
        WHERE timestamp >= ?
          AND (timestamp > ? OR (timestamp = ? AND trade_uid > ?))
        ORDER BY timestamp, trade_uid;
        """,
        (start_ts, last_ts, last_ts, last_uid),
    )
