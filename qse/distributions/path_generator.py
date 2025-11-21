"""Path generation utilities for regime-driven simulation (FR-015, FR-025)."""

from __future__ import annotations

import numpy as np

from qse.distributions.regime_loader import RegimeParams


def generate_price_paths_from_regime(
    s0: float,
    regime: RegimeParams,
    trade_horizon: int,
    bars_per_day: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate price paths using a simple lognormal model parameterized by regime.

    - Mean return per step = mean_daily_return / bars_per_day
    - Vol per step = daily_vol / sqrt(bars_per_day)
    - Steps = trade_horizon * bars_per_day
    """
    if trade_horizon <= 0 or bars_per_day <= 0:
        raise ValueError("trade_horizon and bars_per_day must be positive")

    steps = trade_horizon * bars_per_day
    step_mu = regime.mean_daily_return / bars_per_day
    step_sigma = regime.daily_vol / (bars_per_day ** 0.5)
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=step_mu, scale=step_sigma, size=(n_paths, steps))
    log_price = np.log(s0) + np.cumsum(log_returns, axis=1)
    return np.exp(log_price)


__all__ = ["generate_price_paths_from_regime"]
