"""RSI-based mean reversion stock strategy with position sizing."""

from __future__ import annotations

import numpy as np

from qse.interfaces.strategy import Strategy, StrategySignals
from qse.schema.strategy import StrategyParams
from qse.strategies.position_sizing import (
    _compute_share_sizes_for_target,
)


class StockRsiReversionStrategy(Strategy):
    """
    Canonical RSI(14) mean reversion:

    - Long when RSI < oversold_threshold (e.g., 30)
    - Flat when RSI >= exit_threshold (e.g., 50)
    - Position size targets daily P&L based on expected daily move.
    """

    def generate_signals(self, price_paths, features, params: StrategyParams) -> StrategySignals:
        closes = np.asarray(price_paths, dtype=float)
        n_paths, n_steps = closes.shape

        if not isinstance(features, dict) or "rsi" not in features:
            raise ValueError("RSI feature 'rsi' is required for StockRsiReversionStrategy")

        rsi = np.asarray(features["rsi"], dtype=float)
        if rsi.shape != closes.shape:
            raise ValueError("RSI feature must match price_paths shape [n_paths, n_steps]")

        oversold = float(params.params.get("oversold_threshold", 30.0))
        exit_level = float(params.params.get("exit_threshold", 50.0))

        raw_direction = np.zeros_like(closes, dtype=np.int8)

        # Long when oversold
        raw_direction[rsi < oversold] = 1
        # Flat when recovered
        raw_direction[rsi >= exit_level] = 0

        target_profit = float(params.params.get("target_profit_usd", 750.0))
        expected_move_pct = float(params.params.get("expected_daily_move_pct", 0.02))
        max_position_usd = float(params.params.get("max_position_usd", 50_000.0))

        shares_per_path = _compute_share_sizes_for_target(
            closes=closes,
            target_profit_usd=target_profit,
            expected_daily_move_pct=expected_move_pct,
            max_position_usd=max_position_usd,
        )

        position_sizes = shares_per_path[:, None].astype(np.int32)
        signals_stock = raw_direction * position_sizes
        signals_option = np.zeros_like(signals_stock, dtype=np.int32)

        return StrategySignals(
            signals_stock=signals_stock,
            signals_option=signals_option,
            option_spec=None,
            features_used=["rsi"],
        )
