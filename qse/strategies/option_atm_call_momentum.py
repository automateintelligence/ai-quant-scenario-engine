"""ATM long call momentum strategy with position sizing."""

from __future__ import annotations

import numpy as np

from qse.interfaces.strategy import Strategy, StrategySignals
from qse.models.options import OptionSpec
from qse.schema.strategy import StrategyParams
from qse.strategies.position_sizing import (
    _compute_contract_sizes_for_target,
)


class OptionAtmCallMomentumStrategy(Strategy):
    """
    Canonical long-call momentum strategy:

    - Long ATM call when price > SMA(short_window) and SMA(short_window) > SMA(long_window)
    - Flat otherwise
    - Contracts sized to target daily P&L based on expected underlying move and assumed delta.
    """

    def __init__(self, option_spec: OptionSpec) -> None:
        self.option_spec = option_spec

    def _rolling_mean(self, path: np.ndarray, window: int) -> np.ndarray:
        window = max(1, min(window, len(path)))
        kernel = np.ones(window) / window
        result = np.convolve(path, kernel, mode="same")
        return result[: len(path)]

    def generate_signals(self, price_paths, features, params: StrategyParams) -> StrategySignals:
        closes = np.asarray(price_paths, dtype=float)
        n_paths, n_steps = closes.shape

        short_w = int(params.params.get("short_window", 20))
        long_w = int(params.params.get("long_window", 50))
        if short_w <= 0 or long_w <= 0:
            raise ValueError("window sizes must be positive")
        if short_w >= long_w:
            long_w = short_w + 5

        long_w = min(long_w, n_steps)
        short_w = min(short_w, n_steps)

        short_ma = np.vstack([self._rolling_mean(row, short_w) for row in closes])
        long_ma = np.vstack([self._rolling_mean(row, long_w) for row in closes])

        # Momentum condition: price > short_ma and short_ma > long_ma
        momentum = (closes > short_ma) & (short_ma > long_ma)

        # Optional: IV rank filter if provided
        iv_ok = np.ones_like(closes, dtype=bool)
        features_used: list[str] = []
        if isinstance(features, dict) and "iv_rank" in features:
            iv_rank = np.asarray(features["iv_rank"], dtype=float)
            iv_ok = iv_rank < float(params.params.get("max_iv_rank", 60.0))
            features_used.append("iv_rank")

        raw_direction = np.where(momentum & iv_ok, 1, 0).astype(np.int8)

        target_profit = float(params.params.get("target_profit_usd", 750.0))
        expected_move_pct = float(params.params.get("expected_daily_move_pct", 0.02))
        max_position_usd = float(params.params.get("max_position_usd", 25_000.0))
        assumed_delta = float(params.params.get("assumed_delta", 0.5))

        contracts_per_path = _compute_contract_sizes_for_target(
            closes=closes,
            target_profit_usd=target_profit,
            expected_daily_move_pct=expected_move_pct,
            assumed_delta=assumed_delta,
            max_position_usd=max_position_usd,
        )

        position_sizes = contracts_per_path[:, None].astype(np.int32)
        signals_option = raw_direction * position_sizes
        signals_stock = np.zeros_like(signals_option, dtype=np.int32)

        return StrategySignals(
            signals_stock=signals_stock,
            signals_option=signals_option,
            option_spec=self.option_spec,
            features_used=features_used,
        )
