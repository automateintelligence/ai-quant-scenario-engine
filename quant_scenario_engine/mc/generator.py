"""Monte Carlo price path generator."""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.exceptions import DistributionFitError
from quant_scenario_engine.interfaces.distribution import ReturnDistribution


def generate_price_paths(
    s0: float,
    distribution: ReturnDistribution,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate price paths from log-return samples.

    Applies log-return → price transform and rejects non-positive/overflow paths.
    """

    log_returns = distribution.sample(n_paths=n_paths, n_steps=n_steps, seed=seed)
    if log_returns.shape != (n_paths, n_steps):
        raise DistributionFitError("Distribution returned unexpected shape")

    cumsum = log_returns.cumsum(axis=1)
    prices = s0 * np.exp(cumsum)

    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise DistributionFitError("Generated paths contain non-positive or non-finite values")

    return prices

