from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


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


@dataclass
class UserState:
    first_ts: int
    last_ts: int
    days_active_count: int
    last_active_day: int

    total_trades: int
    total_volume: float

    size_n: int
    size_mean: float
    size_m2: float

    # PnL realized when markets close (causal)
    pnl_n: int
    pnl_sum: float
    pnl_sum_sq: float
    wins: int
    losses: int

    recent_pnl: Deque[float]
    intertrade_intervals: Deque[int]

    @staticmethod
    def create(ts: int) -> "UserState":
        day = ts // 86400
        return UserState(
            first_ts=ts,
            last_ts=ts,
            days_active_count=1,
            last_active_day=day,
            total_trades=0,
            total_volume=0.0,
            size_n=0,
            size_mean=0.0,
            size_m2=0.0,
            pnl_n=0,
            pnl_sum=0.0,
            pnl_sum_sq=0.0,
            wins=0,
            losses=0,
            recent_pnl=deque(maxlen=20),
            intertrade_intervals=deque(maxlen=20),
        )


class MarketState:
    __slots__ = (
        "first_ts",
        "last_ts",
        "total_trades",
        "total_volume",
        "last_price_yes",
        "last_price_no",
        "w1h_deque",
        "sum_usdc_1h",
        "sum_size_1h",
        "sum_price_1h",
        "sum_price2_1h",
        "count_1h",
        "user_usdc_1h",
        "user_trades_1h",
    )

    def __init__(self):
        self.first_ts: Optional[int] = None
        self.last_ts: Optional[int] = None
        self.total_trades = 0
        self.total_volume = 0.0

        self.last_price_yes: Optional[float] = None
        self.last_price_no: Optional[float] = None

        # (ts, user_id, abs_usdc, abs_size, price)
        self.w1h_deque = deque()
        self.sum_usdc_1h = 0.0
        self.sum_size_1h = 0.0
        self.sum_price_1h = 0.0
        self.sum_price2_1h = 0.0
        self.count_1h = 0

        self.user_usdc_1h: Dict[str, float] = {}
        self.user_trades_1h: Dict[str, int] = {}

    def trim_window_1h(self, current_ts: int):
        cutoff = current_ts - 3600
        dq = self.w1h_deque
        u_usdc = self.user_usdc_1h
        u_trades = self.user_trades_1h

        while dq and dq[0][0] < cutoff:
            ts_old, user_old, abs_usdc_old, abs_size_old, price_old = dq.popleft()

            self.sum_usdc_1h -= abs_usdc_old
            self.sum_size_1h -= abs_size_old
            self.sum_price_1h -= price_old
            self.sum_price2_1h -= price_old * price_old
            self.count_1h -= 1

            u_usdc[user_old] = u_usdc.get(user_old, 0.0) - abs_usdc_old
            if u_usdc[user_old] <= 0:
                u_usdc.pop(user_old, None)

            u_trades[user_old] = u_trades.get(user_old, 0) - 1
            if u_trades[user_old] <= 0:
                u_trades.pop(user_old, None)

    def add_trade_1h(self, ts: int, user_id: str, abs_usdc: float, abs_size: float, price: float):
        self.w1h_deque.append((ts, user_id, abs_usdc, abs_size, price))
        self.sum_usdc_1h += abs_usdc
        self.sum_size_1h += abs_size
        self.sum_price_1h += price
        self.sum_price2_1h += price * price
        self.count_1h += 1

        self.user_usdc_1h[user_id] = self.user_usdc_1h.get(user_id, 0.0) + abs_usdc
        self.user_trades_1h[user_id] = self.user_trades_1h.get(user_id, 0) + 1

    def volatility_1h(self) -> float:
        if self.count_1h < 2:
            return 0.0
        mean = self.sum_price_1h / self.count_1h
        var = (self.sum_price2_1h / self.count_1h) - (mean * mean)
        if var < 0:
            var = 0.0
        return math.sqrt(var)

    def mean_trade_size_1h(self) -> float:
        if self.count_1h == 0:
            return 0.0
        return self.sum_size_1h / self.count_1h
