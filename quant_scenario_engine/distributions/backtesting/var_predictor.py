"""Static VaR predictor from fitted model samples (US6a AS4, T153)."""

from __future__ import annotations

import numpy as np


def predict_var_from_samples(model_samples: np.ndarray, level: float) -> float:
    """Return VaR at given level from model return samples."""
    return float(np.quantile(model_samples, 1.0 - level))


__all__ = ["predict_var_from_samples"]
