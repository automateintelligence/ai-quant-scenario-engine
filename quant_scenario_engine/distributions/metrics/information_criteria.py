"""Information criteria helpers (AIC/BIC)."""

from __future__ import annotations

import numpy as np


def aic(log_likelihood: float, k: int) -> float:
    return float(2 * k - 2 * log_likelihood)


def bic(log_likelihood: float, k: int, n: int) -> float:
    return float(k * np.log(max(n, 1)) - 2 * log_likelihood)


__all__ = ["aic", "bic"]
