"""Historical metric calculator (US6a AS5, T161)."""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.distributions.validation.volatility_calc import annualized_volatility
from quant_scenario_engine.distributions.validation.clustering_calc import autocorr_squared_returns
from quant_scenario_engine.distributions.validation.extreme_moves import extreme_move_frequencies


def compute_historical_metrics(returns: np.ndarray) -> dict:
    return {
        "annualized_vol": annualized_volatility(returns),
        "acf_sq_lag1": autocorr_squared_returns(returns, lag=1),
        "extremes": extreme_move_frequencies(returns),
    }


__all__ = ["compute_historical_metrics"]
