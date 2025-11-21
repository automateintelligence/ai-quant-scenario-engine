"""Baseline stock strategy using dual moving averages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qse.interfaces.strategy import Strategy, StrategySignals
from qse.schema.strategy import StrategyParams


class StockBasicStrategy(Strategy):
    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("window sizes must be positive")
        self.short_window = short_window
        self.long_window = long_window

    def _rolling_mean(self, path: np.ndarray, window: int) -> np.ndarray:
        window = max(1, min(window, len(path)))
        kernel = np.ones(window) / window
        result = np.convolve(path, kernel, mode="same")
        # np.convolve preserves length when kernel <= path length; clipping guards
        return result[: len(path)]

    def _feature_or_fallback(self, features, name: str, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
        if isinstance(features, dict) and name in features:
            arr = np.asarray(features[name], dtype=float)
            if arr.shape == fallback.shape:
                return arr, True
        if isinstance(features, pd.DataFrame) and name in features.columns:
            arr = features[name].to_numpy().reshape(fallback.shape)
            return arr, True
        return fallback, False

    def generate_signals(self, price_paths, features, params: StrategyParams) -> StrategySignals:
        closes = np.asarray(price_paths, dtype=float)
        short_w = int(params.params.get("short_window", self.short_window))
        long_w = int(params.params.get("long_window", self.long_window))
        if short_w <= 0 or long_w <= 0:
            raise ValueError("window sizes must be positive")
        if short_w >= long_w:
            long_w = short_w + 5
        short_label = f"sma_{short_w}"
        long_label = f"sma_{long_w}"

        short_len = min(short_w, closes.shape[1])
        long_len = min(long_w, closes.shape[1])

        short_ma_fallback = np.vstack([self._rolling_mean(row, short_len) for row in closes])
        long_ma_fallback = np.vstack([self._rolling_mean(row, long_len) for row in closes])

        short_ma, used_short = self._feature_or_fallback(features, short_label, short_ma_fallback)
        long_ma, used_long = self._feature_or_fallback(features, long_label, long_ma_fallback)

        raw_signal = np.where(short_ma > long_ma, 1, -1).astype(np.int8)
        signals_stock = raw_signal
        signals_option = np.zeros_like(signals_stock, dtype=np.int8)

        features_used: list[str] = []
        if used_short:
            features_used.append(short_label)
        if used_long and long_label not in features_used:
            features_used.append(long_label)

        return StrategySignals(
            signals_stock=signals_stock,
            signals_option=signals_option,
            option_spec=None,
            features_used=features_used,
        )
