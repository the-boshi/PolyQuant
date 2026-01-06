from __future__ import annotations

import heapq
import math
from typing import Dict, Optional, Tuple

from .states import MarketState, UserState, update_welford, variance_from_welford


def time_of_day_features(ts: int):
    seconds_in_day = 86400
    frac = (ts % seconds_in_day) / seconds_in_day
    angle = 2 * math.pi * frac
    return math.sin(angle), math.cos(angle)


def flush_pending_pnls(
    pending_pnls: list[tuple[int, int, str, float]],
    up_to_ts: int,
    user_states: Dict[str, UserState],
):
    """
    pending_pnls heap items: (end_ts, seq, user_id, pnl_value)
    Flush all with end_ts <= up_to_ts, updating user pnl stats causally.
    """
    while pending_pnls and pending_pnls[0][0] <= up_to_ts:
        _, _, user_id, pnl = heapq.heappop(pending_pnls)
        u = user_states.get(user_id)
        if u is None:
            continue

        u.pnl_n += 1
        u.pnl_sum += pnl
        u.pnl_sum_sq += pnl * pnl
        if pnl > 0:
            u.wins += 1
        elif pnl < 0:
            u.losses += 1
        u.recent_pnl.append(pnl)


def schedule_pnl(
    pending_pnls: list[tuple[int, int, str, float]],
    pending_seq: int,
    end_ts: Optional[int],
    user_id: str,
    pnl_raw: Optional[float],
) -> int:
    if end_ts is not None and pnl_raw is not None:
        heapq.heappush(pending_pnls, (end_ts, pending_seq, user_id, float(pnl_raw)))
        return pending_seq + 1
    return pending_seq


def compute_row_features(
    *,
    trade_uid: str,
    user_id: str,
    market_id: str,
    ts: int,
    price: float,
    size: float,
    usdc_size: Optional[float],
    outcome_index: int,
    pnl_raw: Optional[float],
    market_start_ts: Optional[int],
    u: UserState,
    m: MarketState,
    um: dict,
) -> dict:
    """
    Compute features using states BEFORE applying this trade updates.
    """
    abs_usdc = abs(usdc_size) if usdc_size is not None else 0.0
    abs_size = abs(size)

    # label-ish diagnostic
    edge = float(pnl_raw) / float(size) if (pnl_raw is not None and size != 0) else 0.0

    log_usdc_size = math.log1p(abs_usdc)

    # market open time
    seconds_since_open = float(max(0, ts - market_start_ts)) if market_start_ts is not None else 0.0
    tod_sin, tod_cos = time_of_day_features(ts)

    # user (global) BEFORE
    user_total_trades_before = u.total_trades
    user_total_volume_before = u.total_volume
    user_days_active_before = u.days_active_count
    user_time_since_first_trade = float(ts - u.first_ts)

    if u.size_n > 0:
        user_avg_size_before = u.size_mean
        user_std_size_before = math.sqrt(variance_from_welford(u.size_n, u.size_m2))
    else:
        user_avg_size_before = 0.0
        user_std_size_before = 0.0

    if u.pnl_n > 0:
        user_historical_pnl_before = u.pnl_sum
        user_pnl_std_before = math.sqrt(variance_from_welford(u.pnl_n, u.pnl_sum_sq))
        denom = u.wins + u.losses
        user_historical_winrate_before = (u.wins / denom) if denom > 0 else 0.0
    else:
        user_historical_pnl_before = 0.0
        user_pnl_std_before = 0.0
        user_historical_winrate_before = 0.0

    user_recent_pnl_last20 = sum(u.recent_pnl) if u.recent_pnl else 0.0

    if len(u.intertrade_intervals) >= 2:
        ivals = list(u.intertrade_intervals)
        m_iv = sum(ivals) / len(ivals)
        v_iv = sum((x - m_iv) ** 2 for x in ivals) / len(ivals)
        user_activity_burstiness = v_iv
    else:
        user_activity_burstiness = 0.0

    time_since_user_last_trade = float(ts - u.last_ts) if u.total_trades > 0 else -1.0

    # market BEFORE
    market_total_trades_before = m.total_trades
    market_total_volume_before = m.total_volume
    market_time_since_first_trade = float(ts - m.first_ts) if m.first_ts is not None else 0.0

    m.trim_window_1h(ts)
    market_volume_last_1h = m.sum_usdc_1h
    num_unique_traders_last_1h = len(m.user_usdc_1h)
    market_volatility_1h = m.volatility_1h()
    mean_trade_size_1h = m.mean_trade_size_1h()
    size_ratio_market = (abs_size / mean_trade_size_1h) if mean_trade_size_1h > 0 else 0.0

    # YES probability space
    p_yes = price if outcome_index == 1 else (1.0 - price)

    # time since previous trade in this market (BEFORE update)
    dt_trade = float(ts - m.last_ts) if m.last_ts is not None else -1.0

    # delta in implied YES prob vs previous known YES prob
    prev_p_yes = float(m.last_price_yes) if m.last_price_yes is not None else p_yes
    dp_yes = float(p_yes - prev_p_yes)

    mid_price = m.last_price_yes if m.last_price_yes is not None else p_yes
    price_deviation = p_yes - mid_price
    signed_price_deviation = price_deviation

    time_since_last_market_trade = float(ts - m.last_ts) if m.last_ts is not None else -1.0

    # user in market (1h) BEFORE
    user_usdc_last_1h_in_market = m.user_usdc_1h.get(user_id, 0.0)
    user_trades_last_1h_in_market = m.user_trades_1h.get(user_id, 0)
    user_share_of_volume_1h_in_market = (
        user_usdc_last_1h_in_market / market_volume_last_1h
        if market_volume_last_1h > 0 else 0.0
    )

    # user-market lifetime BEFORE
    user_total_usdc_in_market_before = um["total_usdc"]
    user_trade_count_in_market_before = um["trade_count"]
    if um["last_ts"] is not None:
        time_since_user_last_trade_in_market = float(max(ts - um["last_ts"], 1))
    else:
        time_since_user_last_trade_in_market = -1.0

    user_share_of_volume_in_market_before = (
        user_total_usdc_in_market_before / market_total_volume_before
        if market_total_volume_before > 0 else 0.0
    )

    return {
        "trade_uid": trade_uid,
        "user_id": user_id,
        "market_id": market_id,
        "timestamp": ts,
        "edge": edge,

        "price": float(price),

        "p_yes": p_yes,
        "dt_trade": dt_trade,
        "dp_yes": dp_yes,

        "log_usdc_size": float(log_usdc_size),
        "outcome_index": int(outcome_index),
        "seconds_since_market_open": float(seconds_since_open),
        "time_of_day_sin": float(tod_sin),
        "time_of_day_cos": float(tod_cos),

        "user_time_since_first_trade": float(user_time_since_first_trade),
        "market_time_since_first_trade": float(market_time_since_first_trade),

        "user_total_trades_before": int(user_total_trades_before),
        "user_total_volume_before": float(user_total_volume_before),
        "user_days_active_before": int(user_days_active_before),
        "user_avg_size_before": float(user_avg_size_before),
        "user_std_size_before": float(user_std_size_before),
        "user_historical_pnl_before": float(user_historical_pnl_before),
        "user_historical_winrate_before": float(user_historical_winrate_before),
        "user_recent_pnl_last20": float(user_recent_pnl_last20),
        "user_activity_burstiness": float(user_activity_burstiness),
        "user_pnl_std_before": float(user_pnl_std_before),
        "time_since_user_last_trade": float(time_since_user_last_trade),

        "market_total_trades_before": int(market_total_trades_before),
        "market_total_volume_before": float(market_total_volume_before),
        "market_volume_last_1h": float(market_volume_last_1h),
        "num_unique_traders_last_1h": int(num_unique_traders_last_1h),
        "market_volatility_1h": float(market_volatility_1h),
        "mean_trade_size_1h": float(mean_trade_size_1h),
        "size_ratio_market": float(size_ratio_market),
        "last_price_yes": float(m.last_price_yes or 0.0),
        "last_price_no": float(m.last_price_no or 0.0),
        "mid_price": float(mid_price),
        "price_deviation": float(price_deviation),
        "signed_price_deviation": float(signed_price_deviation),
        "time_since_last_market_trade": float(time_since_last_market_trade),

        "user_usdc_last_1h_in_market": float(user_usdc_last_1h_in_market),
        "user_trades_last_1h_in_market": int(user_trades_last_1h_in_market),
        "user_share_of_volume_1h_in_market": float(user_share_of_volume_1h_in_market),

        "user_total_usdc_in_market": float(user_total_usdc_in_market_before),
        "user_trade_count_in_market": int(user_trade_count_in_market_before),
        "user_share_of_volume_in_market": float(user_share_of_volume_in_market_before),
        "time_since_user_last_trade_in_market": float(time_since_user_last_trade_in_market),
    }


def apply_trade_updates(
    *,
    ts: int,
    price: float,
    outcome_index: int,
    abs_usdc: float,
    abs_size: float,
    user_id: str,
    u: UserState,
    m: MarketState,
    um: dict,
):
    # user days active
    day = ts // 86400
    if u.last_active_day != day:
        u.days_active_count += 1
        u.last_active_day = day

    # user trades/volume
    u.total_trades += 1
    u.total_volume += abs_usdc

    # size stats
    u.size_n, u.size_mean, u.size_m2 = update_welford(u.size_n, u.size_mean, u.size_m2, abs_size)

    # intertrade intervals
    if u.total_trades > 1:
        u.intertrade_intervals.append(ts - u.last_ts)
    u.last_ts = ts

    # market lifetime
    if m.first_ts is None:
        m.first_ts = ts
    m.total_trades += 1
    m.total_volume += abs_usdc
    m.last_ts = ts

    # last prices
    if outcome_index == 1:
        m.last_price_yes = price
    else:
        m.last_price_no = price

    # 1h window
    m.add_trade_1h(ts, user_id, abs_usdc, abs_size, price)

    # user-market lifetime
    um["total_usdc"] += abs_usdc
    um["trade_count"] += 1
    um["last_ts"] = ts
