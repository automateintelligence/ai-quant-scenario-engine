"""Validation utilities for distributions."""

from quant_scenario_engine.distributions.validation.stationarity import ensure_min_samples, MIN_SAMPLES
# Re-export legacy helpers from validation.py for backward compatibility
from quant_scenario_engine.distributions.validation_core import (  # type: ignore
    validate_returns,
    validate_params_bounds,
    enforce_convergence,
    heavy_tail_status,
    fallback_to_laplace,
)

__all__ = [
    "ensure_min_samples",
    "MIN_SAMPLES",
    "validate_returns",
    "validate_params_bounds",
    "enforce_convergence",
    "heavy_tail_status",
    "fallback_to_laplace",
]
