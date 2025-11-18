"""Student-T fitter wrapper for audit pipeline."""

from __future__ import annotations

import numpy as np
from scipy import stats

from quant_scenario_engine.distributions.metrics.information_criteria import aic as calc_aic, bic as calc_bic
from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.exceptions import DistributionFitError


class StudentTFitter:
    name = "student_t"
    k = 3

    def __init__(self) -> None:
        self.params: dict[str, float] | None = None
        self._loglik: float | None = None

    def fit(self, returns: np.ndarray) -> FitResult:
        warnings: list[str] = []
        try:
            df, loc, scale = stats.t.fit(returns)
            loglik = float(stats.t.logpdf(returns, df=df, loc=loc, scale=scale).sum())
            self._loglik = loglik
            params = {"df": float(df), "loc": float(loc), "scale": float(scale)}
            self.params = params
            return FitResult(
                model_name=self.name,
                log_likelihood=loglik,
                aic=calc_aic(loglik, self.k),
                bic=calc_bic(loglik, self.k, len(returns)),
                params=params,
                n=len(returns),
                converged=True,
                heavy_tailed=True,
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"fit_failed: {exc}")
            raise DistributionFitError(f"Student-T fit failed: {exc}") from exc

    def sample(self, n_paths: int, n_steps: int, seed: int | None = None) -> np.ndarray:
        if not self.params:
            raise DistributionFitError("StudentTFitter.sample called before fit")
        rng = np.random.default_rng(seed)
        return rng.standard_t(self.params["df"], size=(n_paths, n_steps)) * self.params["scale"] + self.params["loc"]

    def log_likelihood(self) -> float:
        if self._loglik is None:
            raise DistributionFitError("StudentTFitter.log_likelihood called before fit")
        return self._loglik


__all__ = ["StudentTFitter"]
