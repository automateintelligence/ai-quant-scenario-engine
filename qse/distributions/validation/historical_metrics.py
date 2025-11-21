"""Historical metric calculator (US6a AS5, T161)."""

from __future__ import annotations

import numpy as np

from qse.distributions.validation.clustering_calc import autocorr_squared_returns
from qse.distributions.validation.drawdown_calc import max_drawdown
from qse.distributions.validation.extreme_moves import extreme_move_frequencies
from qse.distributions.validation.volatility_calc import annualized_volatility


def compute_historical_metrics(returns: np.ndarray, window: int = 252) -> dict:
    if returns is None or len(returns) == 0:
        return {"annualized_vol": 0.0, "acf_sq_lag1": 0.0, "max_drawdown": 0.0, "extremes": {}}

    arr = np.asarray(returns)
    if len(arr) <= window:
        segments = [arr]
    else:
        segments = [arr[i - window : i] for i in range(window, len(arr) + 1)]

    vol_values = [annualized_volatility(seg) for seg in segments]
    acf_values = [autocorr_squared_returns(seg, lag=1) for seg in segments]
    extremes = [extreme_move_frequencies(seg) for seg in segments]
    drawdowns = []
    for seg in segments:
        prices = np.exp(np.cumsum(seg))
        drawdowns.append(max_drawdown(prices))

    avg_extremes: dict[str, float] = {}
    if extremes:
        keys = extremes[0].keys()
        for key in keys:
            avg_extremes[key] = float(np.mean([ex[key] for ex in extremes]))

    metrics = {
        "annualized_vol": float(np.mean(vol_values)),
        "acf_sq_lag1": float(np.mean(acf_values)),
        "max_drawdown": float(np.mean(drawdowns)),
        "extremes": avg_extremes,
    }

    return metrics


__all__ = ["compute_historical_metrics"]


__all__ = ["compute_historical_metrics"]
