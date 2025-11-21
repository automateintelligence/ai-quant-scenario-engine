"""ATM long put RSI-based strategy with position sizing."""

from __future__ import annotations

import numpy as np

from qse.interfaces.strategy import Strategy, StrategySignals
from qse.models.options import OptionSpec
from qse.schema.strategy import StrategyParams
from qse.strategies.position_sizing import (
    _compute_contract_sizes_for_target,
)


class OptionAtmPutRsiStrategy(Strategy):
    """
    Canonical long-put strategy:

    - Buy ATM put when RSI is deeply oversold (e.g., < 25)
    - Flat otherwise
    - Position sizing based on expected one-day move and assumed put delta.
    """

    def __init__(self, option_spec: OptionSpec) -> None:
        # option_spec.option_type should be "put", strike ~ ATM, DTE ~ 14–21
        self.option_spec = option_spec

    def generate_signals(self, price_paths, features, params: StrategyParams) -> StrategySignals:
        closes = np.asarray(price_paths, dtype=float)
        n_paths, n_steps = closes.shape

        if not isinstance(features, dict) or "rsi" not in features:
            raise ValueError("RSI feature 'rsi' is required for OptionAtmPutRsiStrategy")

        rsi = np.asarray(features["rsi"], dtype=float)
        if rsi.shape != closes.shape:
            raise ValueError("RSI feature must match price_paths shape [n_paths, n_steps]")

        oversold = float(params.params.get("oversold_threshold", 25.0))

        raw_direction = np.where(rsi < oversold, 1, 0).astype(np.int8)

        target_profit = float(params.params.get("target_profit_usd", 750.0))
        expected_move_pct = float(params.params.get("expected_daily_move_pct", 0.03))  # larger moves on selloffs
        max_position_usd = float(params.params.get("max_position_usd", 25_000.0))
        assumed_delta = float(params.params.get("assumed_delta", 0.5))  # ATM put ~ -0.5

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
            features_used=["rsi"],
        )
