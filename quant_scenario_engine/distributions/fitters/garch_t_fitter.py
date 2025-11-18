"""GARCH-T fitter placeholder with explicit failure messaging."""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.exceptions import DistributionFitError


class GarchTFitter:
    name = "garch_t"
    k = 4  # omega, alpha, beta, nu (rough estimate for criteria)

    def fit(self, returns: np.ndarray) -> FitResult:
        raise DistributionFitError(
            "GARCH-T fitter not implemented yet; cannot proceed without proper ARCH parameters"
        )

    def sample(self, n_paths: int, n_steps: int):
        raise DistributionFitError("GARCH-T sampler not implemented")


__all__ = ["GarchTFitter"]
