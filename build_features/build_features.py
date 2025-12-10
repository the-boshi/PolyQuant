#!/usr/bin/env python
import sqlite3
import math
import pickle
import heapq
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ===========================
# CONFIG
# ===========================

DB_PATH = Path(r"C:\Users\nimro\PolyQuant-features\data\polymarket.db")          # path to your SQLite DB
OUTPUT_DIR = Path(r"C:\Users\nimro\PolyQuant-features\outputs\features") # directory for parquet + checkpoints
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_ROWS = 1_000_000                   # how many feature rows per parquet file
CHECKPOINT_EVERY = 5_000_000             # save rolling state every N trades

# Start from 2023-01-01 UTC
START_TS_2023 = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())


# ===========================
# User / Market State Classes
# ===========================

class UserState:
    __slots__ = (
        "first_ts",
        "last_ts",
        "days_active_count",
        "last_active_day",
        "total_trades",
        "total_volume",

        "size_n",
        "size_mean",
        "size_m2",

        "pnl_n",
        "pnl_sum",
        "pnl_sum_sq",
        "wins",
        "losses",

        "recent_pnl",
        "intertrade_intervals",
    )

    def __init__(self, ts: int):
        day = ts // 86400
        self.first_ts = ts
        self.last_ts = ts
        self.days_active_count = 1
        self.last_active_day = day

        self.total_trades = 0
        self.total_volume = 0.0

        self.size_n = 0
        self.size_mean = 0.0
        self.size_m2 = 0.0

        self.pnl_n = 0
        self.pnl_sum = 0.0
        self.pnl_sum_sq = 0.0
        self.wins = 0
        self.losses = 0

        self.recent_pnl = deque(maxlen=20)
        self.intertrade_intervals = deque(maxlen=20)


def update_welford(n: int, mean: float, m2: float, x: float):
    n_new = n + 1
    delta = x - mean
    mean_new = mean + delta / n_new
    m2_new = m2 + delta * (x - mean_new)
    return n_new, mean_new, m2_new


def variance_from_welford(n: int, m2: float) -> float:
    if n < 2:
        return 0.0
    return m2 / n


class MarketState:
    """
    Per-market rolling state:
    - lifetime trades, volume
    - first and last trade ts
    - last YES/NO prices
    - 1h window for volume, volatility, user-in-market stats
    """

    __slots__ = (
        "first_ts",
        "last_ts",
        "total_trades",
        "total_volume",

        "last_price_yes",
        "last_price_no",

        "w1h_deque",          # (ts, user_id, abs_usdc, abs_size, price)
        "sum_usdc_1h",
        "sum_size_1h",
        "sum_price_1h",
        "sum_price2_1h",
        "count_1h",

        "user_usdc_1h",       # user_id -> usdc in last 1h
        "user_trades_1h",     # user_id -> trades count in last 1h
    )

    def __init__(self):
        self.first_ts = None
        self.last_ts = None
        self.total_trades = 0
        self.total_volume = 0.0

        self.last_price_yes = None
        self.last_price_no = None

        self.w1h_deque = deque()
        self.sum_usdc_1h = 0.0
        self.sum_size_1h = 0.0
        self.sum_price_1h = 0.0
        self.sum_price2_1h = 0.0
        self.count_1h = 0

        self.user_usdc_1h = {}
        self.user_trades_1h = {}

    def trim_window_1h(self, current_ts: int):
        cutoff = current_ts - 3600
        dq = self.w1h_deque
        u_usdc = self.user_usdc_1h
        u_trades = self.user_trades_1h

        while dq and dq[0][0] < cutoff:
            ts_old, user_id_old, abs_usdc_old, abs_size_old, price_old = dq.popleft()

            self.sum_usdc_1h -= abs_usdc_old
            self.sum_size_1h -= abs_size_old
            self.sum_price_1h -= price_old
            self.sum_price2_1h -= price_old * price_old
            self.count_1h -= 1

            u_usdc[user_id_old] = u_usdc.get(user_id_old, 0.0) - abs_usdc_old
            if u_usdc[user_id_old] <= 0:
                u_usdc.pop(user_id_old, None)

            u_trades[user_id_old] = u_trades.get(user_id_old, 0) - 1
            if u_trades[user_id_old] <= 0:
                u_trades.pop(user_id_old, None)

    def add_trade_1h(self, ts: int, user_id: str, abs_usdc: float, abs_size: float, price: float):
        self.w1h_deque.append((ts, user_id, abs_usdc, abs_size, price))
        self.sum_usdc_1h += abs_usdc
        self.sum_size_1h += abs_size
        self.sum_price_1h += price
        self.sum_price2_1h += price * price
        self.count_1h += 1

        self.user_usdc_1h[user_id] = self.user_usdc_1h.get(user_id, 0.0) + abs_usdc
        self.user_trades_1h[user_id] = self.user_trades_1h.get(user_id, 0) + 1

    def get_volatility_1h(self) -> float:
        if self.count_1h < 2:
            return 0.0
        mean = self.sum_price_1h / self.count_1h
        var = (self.sum_price2_1h / self.count_1h) - (mean * mean)
        if var < 0:
            var = 0.0
        return math.sqrt(var)

    def get_mean_trade_size_1h(self) -> float:
        if self.count_1h == 0:
            return 0.0
        return self.sum_size_1h / self.count_1h


# ===========================
# Helpers
# ===========================

def parse_iso_to_ts(s: str | None) -> int | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def load_market_start_times(conn: sqlite3.Connection) -> dict[str, int | None]:
    cur = conn.cursor()
    cur.execute("SELECT condition_id, start_date FROM markets;")
    rows = cur.fetchall()
    market_start_ts: dict[str, int | None] = {}
    for cond_id, start_date in rows:
        ts = parse_iso_to_ts(start_date) if start_date else None
        market_start_ts[cond_id] = ts
    return market_start_ts


def load_market_end_times(conn: sqlite3.Connection) -> dict[str, int | None]:
    cur = conn.cursor()
    cur.execute("SELECT condition_id, end_date FROM markets;")
    rows = cur.fetchall()
    market_end_ts: dict[str, int | None] = {}
    for cond_id, end_date in rows:
        ts = parse_iso_to_ts(end_date) if end_date else None
        market_end_ts[cond_id] = ts
    return market_end_ts


def time_of_day_features(ts: int):
    seconds_in_day = 86400
    frac = (ts % seconds_in_day) / seconds_in_day
    angle = 2 * math.pi * frac
    return math.sin(angle), math.cos(angle)


def init_rows_buffer():
    return {
        "trade_uid": [],
        "user_id": [],
        "market_id": [],
        "timestamp": [],
        "edge": [],

        "price": [],
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


def write_parquet_chunk(rows: dict, chunk_idx: int):
    if not rows["trade_uid"]:
        return
    table = pa.table(rows)
    out_path = OUTPUT_DIR / f"features_chunk_{chunk_idx:05d}.parquet"
    pq.write_table(table, out_path)
    print(f"[INFO] Wrote {len(rows['trade_uid'])} rows to {out_path}")


def save_checkpoint(
    user_states,
    market_states,
    user_market_lifetime,
    trades_processed: int,
    chunk_idx: int,
    pending_pnls,
    pending_seq: int,
):
    ckpt_path = OUTPUT_DIR / f"state_checkpoint_{trades_processed:09d}.pkl"
    obj = {
        "trades_processed": trades_processed,
        "chunk_idx": chunk_idx,
        "user_states": user_states,
        "market_states": market_states,
        "user_market_lifetime": user_market_lifetime,
        "pending_pnls": pending_pnls,
        "pending_seq": pending_seq,
    }
    with ckpt_path.open("wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved checkpoint at {ckpt_path}")


def load_latest_checkpoint():
    ckpts = sorted(OUTPUT_DIR.glob("state_checkpoint_*.pkl"))
    if not ckpts:
        return None
    latest = ckpts[-1]
    with latest.open("rb") as f:
        obj = pickle.load(f)
    print(f"[INFO] Loaded checkpoint {latest}")
    return obj


# ===========================
# Main
# ===========================

def build_features_all():
    conn = sqlite3.connect(DB_PATH)
    market_start_ts = load_market_start_times(conn)
    market_end_ts = load_market_end_times(conn)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            trade_uid,
            proxy_wallet,
            condition_id,
            timestamp,
            price,
            size,
            usdc_size,
            outcome_index,
            pnl_raw
        FROM trades
        WHERE timestamp >= ?
        ORDER BY timestamp, trade_uid;
        """,
        (START_TS_2023,),
    )

    # Try to resume from latest checkpoint
    ckpt = load_latest_checkpoint()

    if ckpt is not None:
        user_states = ckpt["user_states"]
        market_states = ckpt["market_states"]
        user_market_lifetime = ckpt["user_market_lifetime"]
        trades_processed = ckpt["trades_processed"]
        chunk_idx = ckpt["chunk_idx"]
        pending_pnls = ckpt.get("pending_pnls", [])
        pending_seq = ckpt.get("pending_seq", 0)
        # skip already-processed trades in cursor
        skipped = 0
        while skipped < trades_processed:
            row = cur.fetchone()
            if row is None:
                break
            skipped += 1
        if skipped > 0:
            print(f"[INFO] Skipped {skipped} already-processed trades from cursor")
        next_checkpoint = ((trades_processed // CHECKPOINT_EVERY) + 1) * CHECKPOINT_EVERY
    else:
        user_states: dict[str, UserState] = {}
        market_states: dict[str, MarketState] = {}
        user_market_lifetime: dict[tuple[str, str], dict] = {}
        trades_processed = 0
        chunk_idx = 0
        pending_pnls: list[tuple[int, int, str, float]] = []
        pending_seq = 0
        next_checkpoint = CHECKPOINT_EVERY

    rows = init_rows_buffer()
    rows_in_chunk = 0

    for (
        trade_uid,
        proxy_wallet,
        condition_id,
        ts,
        price,
        size,
        usdc_size,
        outcome_index,
        pnl_raw,
    ) in cur:

        trades_processed += 1

        # Realize PnLs for markets that have closed by this trade's timestamp
        while pending_pnls and pending_pnls[0][0] <= ts:
            _, _, user_id_flush, edge_flush = heapq.heappop(pending_pnls)
            u_state_flush = user_states.get(user_id_flush)
            if u_state_flush is None:
                continue

            u_state_flush.pnl_n += 1
            u_state_flush.pnl_sum += edge_flush
            u_state_flush.pnl_sum_sq += edge_flush * edge_flush
            if edge_flush > 0:
                u_state_flush.wins += 1
            elif edge_flush < 0:
                u_state_flush.losses += 1

            u_state_flush.recent_pnl.append(edge_flush)

        if size is None or size == 0 or pnl_raw is None or price is None:
            continue

        user_id = proxy_wallet
        market_id = condition_id
        abs_usdc = abs(usdc_size) if usdc_size is not None else 0.0
        abs_size = abs(size)

        # --- get/create states ---
        u_state = user_states.get(user_id)
        if u_state is None:
            u_state = UserState(ts)
            user_states[user_id] = u_state

        m_state = market_states.get(market_id)
        if m_state is None:
            m_state = MarketState()
            market_states[market_id] = m_state
        if m_state.first_ts is None:
            m_state.first_ts = ts

        um_key = (user_id, market_id)
        um_state = user_market_lifetime.get(um_key)
        if um_state is None:
            um_state = {
                "total_usdc": 0.0,
                "trade_count": 0,
                "last_ts": None,
            }
            user_market_lifetime[um_key] = um_state

        # --- label: edge = pnl_raw / size (assumes BUY-only) ---
        edge = pnl_raw / size

        # schedule this trade's PnL to be realized at market close
        end_ts = market_end_ts.get(market_id)
        if end_ts is not None:
            heapq.heappush(pending_pnls, (end_ts, pending_seq, user_id, edge))
            pending_seq += 1

        # --- trade_local ---
        log_usdc_size = math.log1p(abs_usdc)
        start_ts = market_start_ts.get(market_id)
        if start_ts is None:
            seconds_since_open = 0.0
        else:
            seconds_since_open = max(0, ts - start_ts)
        tod_sin, tod_cos = time_of_day_features(ts)

        # --- user_global BEFORE updating ---
        user_total_trades_before = u_state.total_trades
        user_total_volume_before = u_state.total_volume
        user_days_active_before = u_state.days_active_count

        user_time_since_first_trade = float(ts - u_state.first_ts)

        if u_state.size_n > 0:
            user_avg_size_before = u_state.size_mean
            size_var = variance_from_welford(u_state.size_n, u_state.size_m2)
            user_std_size_before = math.sqrt(size_var)
        else:
            user_avg_size_before = 0.0
            user_std_size_before = 0.0

        if u_state.pnl_n > 0:
            user_historical_pnl_before = u_state.pnl_sum
            pnl_var = variance_from_welford(u_state.pnl_n, u_state.pnl_sum_sq)
            user_pnl_std_before = math.sqrt(pnl_var)
            total_outcomes = u_state.wins + u_state.losses
            if total_outcomes > 0:
                user_historical_winrate_before = u_state.wins / total_outcomes
            else:
                user_historical_winrate_before = 0.0
        else:
            user_historical_pnl_before = 0.0
            user_pnl_std_before = 0.0
            user_historical_winrate_before = 0.0

        user_recent_pnl_last20 = sum(u_state.recent_pnl) if u_state.recent_pnl else 0.0

        if len(u_state.intertrade_intervals) >= 2:
            ivals = list(u_state.intertrade_intervals)
            m_iv = sum(ivals) / len(ivals)
            v_iv = sum((x - m_iv) ** 2 for x in ivals) / len(ivals)
            user_activity_burstiness = v_iv
        else:
            user_activity_burstiness = 0.0

        time_since_user_last_trade = (
            ts - u_state.last_ts if u_state.total_trades > 0 else -1.0
        )

        # --- market_features BEFORE updating ---
        market_total_trades_before = m_state.total_trades
        market_total_volume_before = m_state.total_volume

        market_time_since_first_trade = (
            float(ts - m_state.first_ts) if m_state.first_ts is not None else 0.0
        )

        m_state.trim_window_1h(ts)

        market_volume_last_1h = m_state.sum_usdc_1h
        num_unique_traders_last_1h = len(m_state.user_usdc_1h)
        market_volatility_1h = m_state.get_volatility_1h()
        mean_trade_size_1h = m_state.get_mean_trade_size_1h()

        if mean_trade_size_1h > 0:
            size_ratio_market = abs_size / mean_trade_size_1h
        else:
            size_ratio_market = 0.0

        # Convert current trade into YES probability space
        if outcome_index == 1:  # YES
            curr_yes_price = price
        else:  # NO
            curr_yes_price = 1.0 - price

        # Determine mid-price (use last seen YES price if available, else current)
        if m_state.last_price_yes is not None:
            mid_price = m_state.last_price_yes
        else:
            mid_price = curr_yes_price

        # Price deviation in YES-probability space
        price_deviation = curr_yes_price - mid_price
        signed_price_deviation = price_deviation

        if m_state.last_ts is not None:
            time_since_last_market_trade = ts - m_state.last_ts
        else:
            time_since_last_market_trade = -1.0

        # --- user_in_market (1h) BEFORE updating ---
        user_usdc_last_1h_in_market = m_state.user_usdc_1h.get(user_id, 0.0)
        user_trades_last_1h_in_market = m_state.user_trades_1h.get(user_id, 0)
        if market_volume_last_1h > 0:
            user_share_of_volume_1h_in_market = (
                user_usdc_last_1h_in_market / market_volume_last_1h
            )
        else:
            user_share_of_volume_1h_in_market = 0.0

        # --- user_market_lifetime BEFORE updating ---
        user_total_usdc_in_market_before = um_state["total_usdc"]
        user_trade_count_in_market_before = um_state["trade_count"]
        if um_state["last_ts"] is not None:
            dt_um = ts - um_state["last_ts"]
            # avoid 0 → treat same-second as 1s gap
            time_since_user_last_trade_in_market = float(max(dt_um, 1))
        else:
            time_since_user_last_trade_in_market = -1.0

        if market_total_volume_before > 0:
            user_share_of_volume_in_market_before = (
                user_total_usdc_in_market_before / market_total_volume_before
            )
        else:
            user_share_of_volume_in_market_before = 0.0

        # --- append row ---
        rows["trade_uid"].append(trade_uid)
        rows["user_id"].append(user_id)
        rows["market_id"].append(market_id)
        rows["timestamp"].append(ts)
        rows["edge"].append(edge)

        rows["price"].append(price)
        rows["log_usdc_size"].append(log_usdc_size)
        rows["outcome_index"].append(outcome_index)
        rows["seconds_since_market_open"].append(float(seconds_since_open))
        rows["time_of_day_sin"].append(tod_sin)
        rows["time_of_day_cos"].append(tod_cos)

        rows["user_time_since_first_trade"].append(user_time_since_first_trade)
        rows["market_time_since_first_trade"].append(market_time_since_first_trade)

        rows["user_total_trades_before"].append(user_total_trades_before)
        rows["user_total_volume_before"].append(user_total_volume_before)
        rows["user_days_active_before"].append(user_days_active_before)
        rows["user_avg_size_before"].append(user_avg_size_before)
        rows["user_std_size_before"].append(user_std_size_before)
        rows["user_historical_pnl_before"].append(user_historical_pnl_before)
        rows["user_historical_winrate_before"].append(user_historical_winrate_before)
        rows["user_recent_pnl_last20"].append(user_recent_pnl_last20)
        rows["user_activity_burstiness"].append(user_activity_burstiness)
        rows["user_pnl_std_before"].append(user_pnl_std_before)
        rows["time_since_user_last_trade"].append(float(time_since_user_last_trade))

        rows["market_total_trades_before"].append(market_total_trades_before)
        rows["market_total_volume_before"].append(market_total_volume_before)
        rows["market_volume_last_1h"].append(market_volume_last_1h)
        rows["num_unique_traders_last_1h"].append(num_unique_traders_last_1h)
        rows["market_volatility_1h"].append(market_volatility_1h)
        rows["mean_trade_size_1h"].append(mean_trade_size_1h)
        rows["size_ratio_market"].append(size_ratio_market)
        rows["last_price_yes"].append(m_state.last_price_yes or 0.0)
        rows["last_price_no"].append(m_state.last_price_no or 0.0)
        rows["mid_price"].append(mid_price)
        rows["price_deviation"].append(price_deviation)
        rows["signed_price_deviation"].append(signed_price_deviation)
        rows["time_since_last_market_trade"].append(float(time_since_last_market_trade))

        rows["user_usdc_last_1h_in_market"].append(user_usdc_last_1h_in_market)
        rows["user_trades_last_1h_in_market"].append(user_trades_last_1h_in_market)
        rows["user_share_of_volume_1h_in_market"].append(user_share_of_volume_1h_in_market)

        rows["user_total_usdc_in_market"].append(user_total_usdc_in_market_before)
        rows["user_trade_count_in_market"].append(user_trade_count_in_market_before)
        rows["user_share_of_volume_in_market"].append(user_share_of_volume_in_market_before)
        rows["time_since_user_last_trade_in_market"].append(
            float(time_since_user_last_trade_in_market)
        )

        rows_in_chunk += 1

        # --- now update states with THIS trade ---

        # user days active
        current_day = ts // 86400
        if u_state.last_active_day != current_day:
            u_state.days_active_count += 1
            u_state.last_active_day = current_day

        # user trades & volume
        u_state.total_trades += 1
        u_state.total_volume += abs_usdc

        # size stats
        u_state.size_n, u_state.size_mean, u_state.size_m2 = update_welford(
            u_state.size_n, u_state.size_mean, u_state.size_m2, abs_size
        )

        # intertrade intervals
        if u_state.total_trades > 1:
            interval = ts - u_state.last_ts
            u_state.intertrade_intervals.append(interval)

        u_state.last_ts = ts

        # market lifetime
        if m_state.total_trades == 0 and m_state.first_ts is None:
            m_state.first_ts = ts
        m_state.total_trades += 1
        m_state.total_volume += abs_usdc
        m_state.last_ts = ts

        # update last_price_yes / last_price_no (assuming outcome_index 1 == YES, 0 == NO)
        if outcome_index == 1:
            m_state.last_price_yes = price
        elif outcome_index == 0:
            m_state.last_price_no = price

        # 1h window
        m_state.add_trade_1h(ts, user_id, abs_usdc, abs_size, price)

        # user_market_lifetime
        um_state["total_usdc"] += abs_usdc
        um_state["trade_count"] += 1
        um_state["last_ts"] = ts

        # --- write chunk if needed ---
        if rows_in_chunk >= CHUNK_ROWS:
            write_parquet_chunk(rows, chunk_idx)
            chunk_idx += 1
            rows = init_rows_buffer()
            rows_in_chunk = 0

        # --- checkpoint if needed ---
        if trades_processed >= next_checkpoint:
            save_checkpoint(
                user_states,
                market_states,
                user_market_lifetime,
                trades_processed,
                chunk_idx,
                pending_pnls,
                pending_seq,
            )
            next_checkpoint += CHECKPOINT_EVERY

    conn.close()

    # write remaining rows
    if rows_in_chunk > 0:
        write_parquet_chunk(rows, chunk_idx)

    print(f"[INFO] Done. Total trades processed: {trades_processed}")


if __name__ == "__main__":
    build_features_all()
