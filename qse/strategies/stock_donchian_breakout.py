"""Donchian channel breakout stock strategy with position sizing."""

from __future__ import annotations

import numpy as np

from qse.interfaces.strategy import Strategy, StrategySignals
from qse.schema.strategy import StrategyParams
from qse.strategies.position_sizing import (
    _compute_share_sizes_for_target,
)


class StockDonchianBreakoutStrategy(Strategy):
    """
    Classic Donchian breakout (Turtle-style):

    - Enter long on new N-day highs (entry_lookback)
    - Exit on new M-day lows (exit_lookback)
    - Uses simple state machine to persist positions between triggers.
    """

    def __init__(self, entry_lookback: int = 20, exit_lookback: int = 10) -> None:
        if entry_lookback <= 0 or exit_lookback <= 0:
            raise ValueError("lookbacks must be positive")
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback

    def _build_direction(self, closes: np.ndarray, entry_lb: int, exit_lb: int) -> np.ndarray:
        n_paths, n_steps = closes.shape
        direction = np.zeros_like(closes, dtype=np.int8)

        for i in range(n_paths):
            in_position = False
            for t in range(n_steps):
                # Use history up to but excluding current bar
                entry_start = max(0, t - entry_lb)
                exit_start = max(0, t - exit_lb)

                highest_high = closes[i, entry_start:t].max() if t > entry_start else closes[i, t]
                lowest_low = closes[i, exit_start:t].min() if t > exit_start else closes[i, t]

                if not in_position and closes[i, t] > highest_high:
                    in_position = True
                elif in_position and closes[i, t] < lowest_low:
                    in_position = False

                direction[i, t] = 1 if in_position else 0

        return direction

    def generate_signals(self, price_paths, features, params: StrategyParams) -> StrategySignals:
        closes = np.asarray(price_paths, dtype=float)

        entry_lb = int(params.params.get("entry_lookback", self.entry_lookback))
        exit_lb = int(params.params.get("exit_lookback", self.exit_lookback))
        if entry_lb <= 0 or exit_lb <= 0:
            raise ValueError("lookbacks must be positive")

        raw_direction = self._build_direction(closes, entry_lb, exit_lb)

        target_profit = float(params.params.get("target_profit_usd", 750.0))
        expected_move_pct = float(params.params.get("expected_daily_move_pct", 0.025))
        max_position_usd = float(params.params.get("max_position_usd", 50_000.0))

        shares_per_path = _compute_share_sizes_for_target(
            closes=closes,
            target_profit_usd=target_profit,
            expected_daily_move_pct=expected_move_pct,
            max_position_usd=max_position_usd,
        )

        signals_stock = raw_direction * shares_per_path[:, None].astype(np.int32)
        signals_option = np.zeros_like(signals_stock, dtype=np.int32)

        features_used: list[str] = []

        return StrategySignals(
            signals_stock=signals_stock,
            signals_option=signals_option,
            option_spec=None,
            features_used=features_used,
        )
