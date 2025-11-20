"""Convergence failure helpers for distribution audits."""

from __future__ import annotations

from typing import Iterable, Optional

from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.utils.logging import get_logger

log = get_logger(__name__, component="distribution_errors")


def record_convergence_failure(
    model_name: str,
    *,
    error: Exception | str,
    n_samples: int,
    stage: str = "fit",
    warnings: Optional[Iterable[str]] = None,
) -> FitResult:
    """Log diagnostics for a failed model fit and return a placeholder FitResult."""

    message = str(error)
    warning_list = list(warnings or [])
    if message not in warning_list:
        warning_list.append(message)

    log.warning(
        "Model failed to converge",
        extra={
            "model": model_name,
            "stage": stage,
            "n_samples": n_samples,
            "status": "FAILED",
            "error": message,
        },
    )

    return FitResult(
        model_name=model_name,
        log_likelihood=float("nan"),
        aic=float("inf"),
        bic=float("inf"),
        params={},
        n=n_samples,
        converged=False,
        heavy_tailed=None,
        fit_success=False,
        warnings=warning_list,
        error=message,
        fit_message=f"FAILED_{stage}",
    )


__all__ = ["record_convergence_failure"]
