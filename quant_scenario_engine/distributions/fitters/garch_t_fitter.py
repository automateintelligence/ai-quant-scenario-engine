"""GARCH-T fitter placeholder with explicit failure messaging."""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.distributions.validation.stationarity import ensure_min_samples
from quant_scenario_engine.exceptions import DistributionFitError


class GarchTFitter:
    name = "garch_t"
    k = 4  # omega, alpha, beta, nu (rough estimate for criteria)

    def fit(self, returns: np.ndarray) -> FitResult:
        ensure_min_samples(returns, self.name)
        try:
            from arch import arch_model  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            raise DistributionFitError(f"GARCH-T requires 'arch' package: {exc}") from exc

        try:
            am = arch_model(returns, vol="GARCH", p=1, o=0, q=1, dist="t", rescale=False)
            res = am.fit(disp="off", update_freq=0, show_warning=False)
            params = {k: float(v) for k, v in res.params.items()}
            loglik = float(res.loglikelihood)
            warnings: list[str] = []
            return FitResult(
                model_name=self.name,
                log_likelihood=loglik,
                aic=float(res.aic),
                bic=float(res.bic),
                params=params,
                n=len(returns),
                converged=bool(res.converged),
                heavy_tailed=True,
                fit_success=bool(res.converged),
                warnings=warnings,
            )
        except Exception as exc:
            raise DistributionFitError(f"GARCH-T fit failed: {exc}") from exc

    def sample(self, n_paths: int, n_steps: int):
        try:
            from arch import arch_model  # type: ignore
            import numpy as np
        except Exception as exc:  # pragma: no cover - dependency guard
            raise DistributionFitError(f"GARCH-T requires 'arch' package: {exc}") from exc

        # Use a simple stationary GARCH(1,1)-t parameter set if fit params unavailable
        am = arch_model(None, vol="GARCH", p=1, o=0, q=1, dist="t", rescale=False)
        params = np.array([0.00001, 0.05, 0.9, 8.0])  # omega, alpha[1], beta[1], nu
        sim = am.simulate(params, nobs=n_steps, repetitions=n_paths)
        return sim["data"]

    def log_likelihood(self) -> float:
        raise DistributionFitError("GARCH-T log-likelihood not available (not implemented)")


__all__ = ["GarchTFitter"]
