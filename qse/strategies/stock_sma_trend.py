"""SMA trend-following stock strategy with position sizing."""

from __future__ import annotations

import numpy as np

from qse.interfaces.strategy import Strategy, StrategySignals
from qse.schema.strategy import StrategyParams
from qse.strategies.position_sizing import (
    _compute_share_sizes_for_target,
)


class StockSmaTrendStrategy(Strategy):
    """
    Canonical trend-following strategy:

    - Long when SMA(short_window) > SMA(long_window)
    - Flat or short when SMA(short_window) <= SMA(long_window) (configurable)
    - Position size scaled to target an expected daily P&L (e.g. $500–$1000)
      for a typical one-day move in the underlying.
    """

    def __init__(self, short_window: int = 12, long_window: int = 38) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("window sizes must be positive")
        self.short_window = short_window
        self.long_window = long_window

    def _rolling_mean(self, path: np.ndarray, window: int) -> np.ndarray:
        window = max(1, min(window, len(path)))
        kernel = np.ones(window) / window
        result = np.convolve(path, kernel, mode="same")
        return result[: len(path)]

    def generate_signals(self, price_paths, features, params: StrategyParams) -> StrategySignals:
        closes = np.asarray(price_paths, dtype=float)  # shape: [n_paths, n_steps]
        n_paths, n_steps = closes.shape

        short_w = int(params.params.get("short_window", self.short_window))
        long_w = int(params.params.get("long_window", self.long_window))
        if short_w <= 0 or long_w <= 0:
            raise ValueError("window sizes must be positive")
        if short_w >= long_w:
            long_w = short_w + 5

        long_w = min(long_w, n_steps)
        short_w = min(short_w, n_steps)

        short_ma = np.vstack([self._rolling_mean(row, short_w) for row in closes])
        long_ma = np.vstack([self._rolling_mean(row, long_w) for row in closes])

        # Directional signal: 1 long, 0 flat (you could allow -1 short via config)
        raw_direction = np.where(short_ma > long_ma, 1, 0).astype(np.int8)

        target_profit = float(params.params.get("target_profit_usd", 750.0))  # between 500–1000
        expected_move_pct = float(params.params.get("expected_daily_move_pct", 0.02))  # 2%
        max_position_usd = float(params.params.get("max_position_usd", 50_000.0))

        shares_per_path = _compute_share_sizes_for_target(
            closes=closes,
            target_profit_usd=target_profit,
            expected_daily_move_pct=expected_move_pct,
            max_position_usd=max_position_usd,
        )  # shape: [n_paths]

        # Broadcast shares into [n_paths, n_steps]
        position_sizes = shares_per_path[:, None].astype(np.int32)
        signals_stock = raw_direction * position_sizes

        signals_option = np.zeros_like(signals_stock, dtype=np.int32)

        return StrategySignals(
            signals_stock=signals_stock,
            signals_option=signals_option,
            option_spec=None,
            features_used=[],
        )
