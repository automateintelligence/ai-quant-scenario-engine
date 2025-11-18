"""Excess kurtosis diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis


def compare_kurtosis(empirical: np.ndarray, model_samples: np.ndarray) -> dict:
    emp_k = float(kurtosis(empirical, fisher=True))
    model_k = float(kurtosis(model_samples, fisher=True))
    meets_threshold = model_k >= 1.0
    return {"empirical": emp_k, "model": model_k, "meets_threshold": meets_threshold}


__all__ = ["compare_kurtosis"]
