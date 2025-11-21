"""Monte Carlo price path generator."""

from __future__ import annotations

import numpy as np

from qse.exceptions import DistributionFitError
from qse.interfaces.distribution import ReturnDistribution

try:  # Optional acceleration
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


def _transform_prices_numpy(log_returns: np.ndarray, s0: float) -> np.ndarray:
    cumsum = log_returns.cumsum(axis=1)
    return s0 * np.exp(cumsum)


if njit:
    @njit(cache=True)
    def _transform_prices_jit(log_returns: np.ndarray, s0: float) -> np.ndarray:  # pragma: no cover - compiled
        n_paths, n_steps = log_returns.shape
        out = np.empty((n_paths, n_steps), dtype=np.float64)
        for i in range(n_paths):
            acc = 0.0
            for j in range(n_steps):
                acc += log_returns[i, j]
                out[i, j] = s0 * np.exp(acc)
        return out
else:
    _transform_prices_jit = None


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

    log_returns = np.asarray(log_returns, dtype=float)
    if _transform_prices_jit is not None:
        prices = _transform_prices_jit(log_returns, s0)
    else:
        prices = _transform_prices_numpy(log_returns, s0)

    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise DistributionFitError("Generated paths contain non-positive or non-finite values")

    return prices
