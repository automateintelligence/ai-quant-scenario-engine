"""Laplace fitter wrapper for audit pipeline."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, laplace

from quant_scenario_engine.distributions.metrics.information_criteria import aic as calc_aic, bic as calc_bic
from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.exceptions import DistributionFitError
from quant_scenario_engine.interfaces.distribution import DistributionMetadata


class LaplaceFitter:
    name = "laplace"
    k = 2

    def __init__(self) -> None:
        self.params: dict[str, float] | None = None
        self._loglik: float | None = None

    def fit(self, returns: np.ndarray) -> FitResult:
        warnings: list[str] = []
        try:
            loc, scale = laplace.fit(returns)
            loglik = float(laplace.logpdf(returns, loc=loc, scale=scale).sum())
            self._loglik = loglik
            excess_kurt = float(kurtosis(returns, fisher=True))
            metadata = DistributionMetadata(
                estimator="mle",
                loglik=loglik,
                aic=calc_aic(loglik, self.k),
                bic=calc_bic(loglik, self.k, len(returns)),
                fit_status="success",
                min_samples=len(returns),
            )
            params = {"loc": float(loc), "scale": float(scale)}
            self.params = params
            return FitResult(
                model_name=self.name,
                log_likelihood=loglik,
                aic=metadata.aic or calc_aic(loglik, self.k),
                bic=metadata.bic or calc_bic(loglik, self.k, len(returns)),
                params=params,
                n=len(returns),
                converged=True,
                heavy_tailed=excess_kurt >= 1.0,
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"fit_failed: {exc}")
            raise DistributionFitError(f"Laplace fit failed: {exc}") from exc

    def sample(self, n_paths: int, n_steps: int, seed: int | None = None) -> np.ndarray:
        if not self.params:
            raise DistributionFitError("LaplaceFitter.sample called before fit")
        rng = np.random.default_rng(seed)
        return rng.laplace(self.params["loc"], self.params["scale"], size=(n_paths, n_steps))

    def log_likelihood(self) -> float:
        if self._loglik is None:
            raise DistributionFitError("LaplaceFitter.log_likelihood called before fit")
        return self._loglik


__all__ = ["LaplaceFitter"]
