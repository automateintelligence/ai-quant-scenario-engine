"""Insufficient data handling helpers for distribution audits."""

from __future__ import annotations

from typing import Optional

from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.distributions.validation.stationarity import MIN_SAMPLES
from quant_scenario_engine.utils.logging import get_logger

log = get_logger(__name__, component="distribution_errors")


def has_minimum_samples(model_name: str, sample_count: int, min_required: Optional[int] = None) -> bool:
    """Return True when the sample count satisfies the minimum requirement."""

    required = min_required if min_required is not None else MIN_SAMPLES.get(model_name, 60)
    return sample_count >= required


def handle_insufficient_data(
    model_name: str,
    sample_count: int,
    *,
    min_required: Optional[int] = None,
    symbol: Optional[str] = None,
) -> FitResult:
    """Create a FitResult entry for models skipped due to insufficient data."""

    required = min_required if min_required is not None else MIN_SAMPLES.get(model_name, 60)
    message = (
        f"Insufficient data for {model_name}: need ≥{required}, got {sample_count}"
    )
    extra = {
        "model": model_name,
        "required": required,
        "n_samples": sample_count,
        "status": "SKIPPED_INSUFFICIENT_DATA",
    }
    if symbol:
        extra["symbol"] = symbol
    log.warning("Skipping model due to insufficient data", extra=extra)

    return FitResult(
        model_name=model_name,
        log_likelihood=float("nan"),
        aic=float("inf"),
        bic=float("inf"),
        params={},
        n=sample_count,
        converged=False,
        heavy_tailed=None,
        fit_success=False,
        warnings=[message],
        error=message,
        fit_message="skipped_insufficient_data",
    )


__all__ = ["handle_insufficient_data", "has_minimum_samples"]
