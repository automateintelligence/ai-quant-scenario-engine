"""Baseline call option strategy mirroring stock signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qse.interfaces.strategy import Strategy, StrategySignals
from qse.models.options import OptionSpec
from qse.schema.strategy import StrategyParams


class OptionCallStrategy(Strategy):
    def __init__(self, option_spec: OptionSpec) -> None:
        self.option_spec = option_spec

    def _get_feature(self, features, name: str) -> tuple[np.ndarray | None, bool]:
        if isinstance(features, dict) and name in features:
            return np.asarray(features[name], dtype=float), True
        if isinstance(features, pd.DataFrame) and name in features.columns:
            return features[name].to_numpy(), True
        return None, False

    def generate_signals(self, price_paths, features, params: StrategyParams) -> StrategySignals:
        closes = np.asarray(price_paths, dtype=float)
        momentum_window = int(params.params.get("momentum_window", 3))
        if momentum_window <= 0:
            momentum_window = 1

        # Rolling momentum: price above rolling mean -> long call else flat
        kernel = np.ones(momentum_window) / momentum_window
        rolling = np.vstack([np.convolve(row, kernel, mode="same") for row in closes])
        signal = np.where(closes >= rolling, 1, 0).astype(np.int8)

        features_used: list[str] = []
        rsi, have_rsi = self._get_feature(features, "rsi")
        if have_rsi:
            min_rsi = float(params.params.get("min_rsi", 50.0))
            if rsi.shape != signal.shape:
                rsi = np.broadcast_to(rsi, signal.shape)
            signal = np.where(rsi >= min_rsi, signal, 0).astype(np.int8)
            features_used.append("rsi")

        return StrategySignals(
            signals_stock=np.zeros_like(signal, dtype=np.int8),
            signals_option=signal,
            option_spec=self.option_spec,
            features_used=features_used,
        )
