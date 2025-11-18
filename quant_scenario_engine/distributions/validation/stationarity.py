"""Stationarity/min-sample validation helpers for distribution fitters (US6a AS1)."""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.exceptions import DistributionFitError

MIN_SAMPLES = {
    "laplace": 60,
    "student_t": 60,
    "garch_t": 252,
}


def ensure_min_samples(returns: np.ndarray, model_name: str) -> None:
    """Raise if returns are insufficient for the requested model."""
    needed = MIN_SAMPLES.get(model_name, 60)
    if returns is None or len(returns) < needed:
        raise DistributionFitError(f"Insufficient samples for {model_name}: need >= {needed}, got {len(returns) if returns is not None else 0}")


__all__ = ["ensure_min_samples", "MIN_SAMPLES"]
