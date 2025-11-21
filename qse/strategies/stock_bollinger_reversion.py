"""Bollinger Band mean-reversion stock strategy with position sizing."""

from __future__ import annotations

import numpy as np

from qse.features.technical import compute_bollinger_bands
from qse.interfaces.strategy import Strategy, StrategySignals
from qse.schema.strategy import StrategyParams
from qse.strategies.position_sizing import (
    _compute_share_sizes_for_target,
)


class StockBollingerReversionStrategy(Strategy):
    """
    Canonical Bollinger Band mean reversion:

    - Enter long when close < lower band (20, 2σ by default)
    - Exit when close > middle band
    - Position sizing targets a daily P&L using expected move sizing helper.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        if period <= 0:
            raise ValueError("Bollinger period must be positive")
        if num_std <= 0:
            raise ValueError("num_std must be positive")
        self.period = period
        self.num_std = num_std

    def _build_direction(self, closes: np.ndarray, lower: np.ndarray, middle: np.ndarray) -> np.ndarray:
        """Stateful walk to hold positions between entry/exit triggers."""

        n_paths, n_steps = closes.shape
        direction = np.zeros_like(closes, dtype=np.int8)

        for i in range(n_paths):
            in_position = False
            for t in range(n_steps):
                if not in_position and closes[i, t] < lower[i, t]:
                    in_position = True
                elif in_position and closes[i, t] > middle[i, t]:
                    in_position = False
                direction[i, t] = 1 if in_position else 0

        return direction

    def generate_signals(self, price_paths, features, params: StrategyParams) -> StrategySignals:
        closes = np.asarray(price_paths, dtype=float)
        n_paths, _ = closes.shape

        period = int(params.params.get("period", self.period))
        num_std = float(params.params.get("num_std", self.num_std))
        if period <= 0 or num_std <= 0:
            raise ValueError("period and num_std must be positive")

        if isinstance(features, dict) and {"bb_upper", "bb_middle", "bb_lower"} <= set(features.keys()):
            upper = np.asarray(features["bb_upper"], dtype=float)
            middle = np.asarray(features["bb_middle"], dtype=float)
            lower = np.asarray(features["bb_lower"], dtype=float)
        else:
            upper, middle, lower = compute_bollinger_bands(
                closes, period=period, num_std=num_std, fillna=True
            )

        raw_direction = self._build_direction(closes, lower, middle)

        target_profit = float(params.params.get("target_profit_usd", 500.0))
        expected_move_pct = float(params.params.get("expected_daily_move_pct", 0.02))
        max_position_usd = float(params.params.get("max_position_usd", 50_000.0))

        shares_per_path = _compute_share_sizes_for_target(
            closes=closes,
            target_profit_usd=target_profit,
            expected_daily_move_pct=expected_move_pct,
            max_position_usd=max_position_usd,
        )

        signals_stock = raw_direction * shares_per_path[:, None].astype(np.int32)
        signals_option = np.zeros_like(signals_stock, dtype=np.int32)

        features_used = []
        if isinstance(features, dict) and {"bb_upper", "bb_middle", "bb_lower"} <= set(features.keys()):
            features_used = ["bb_upper", "bb_middle", "bb_lower"]

        return StrategySignals(
            signals_stock=signals_stock,
            signals_option=signals_option,
            option_spec=None,
            features_used=features_used,
        )
