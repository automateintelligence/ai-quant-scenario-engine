"""Aggregate tail diagnostics report."""

from __future__ import annotations

from quant_scenario_engine.distributions.diagnostics.kurtosis import compare_kurtosis
from quant_scenario_engine.distributions.diagnostics.tail_metrics import tail_error


def build_tail_report(empirical, model_samples) -> dict:
    """Return aggregated tail diagnostics."""
    errors = tail_error(empirical, model_samples)
    kurt = compare_kurtosis(empirical, model_samples)
    return {"tail_errors": errors, "kurtosis": kurt}


__all__ = ["build_tail_report"]
