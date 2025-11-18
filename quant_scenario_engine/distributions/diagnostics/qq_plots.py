"""QQ plot helper returning quantile pairs (no plotting dependency)."""

from __future__ import annotations

import numpy as np


def compute_qq_pairs(empirical: np.ndarray, model_samples: np.ndarray, quantiles: np.ndarray | None = None):
    """
    Compute QQ pairs for given empirical data and model samples.

    Returns (q, emp_q, model_q) where q are quantile levels.
    """
    if quantiles is None:
        quantiles = np.linspace(0.01, 0.99, 99)
    emp_q = np.quantile(empirical, quantiles)
    model_q = np.quantile(model_samples, quantiles)
    return quantiles, emp_q, model_q


__all__ = ["compute_qq_pairs"]
