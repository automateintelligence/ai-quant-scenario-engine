"""GARCH-T fitter placeholder with explicit failure messaging."""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.distributions.models import FitResult
from quant_scenario_engine.distributions.validation.stationarity import ensure_min_samples
from quant_scenario_engine.exceptions import DistributionFitError


class GarchTFitter:
    name = "garch_t"
    k = 4  # omega, alpha, beta, nu (rough estimate for criteria)
    def __init__(self) -> None:
        self._fit_params = None

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
            converged_attr = getattr(res, "converged", None)
            converged = bool(converged_attr) if converged_attr is not None else bool(getattr(res, "convergence", 0) == 0)
            warnings: list[str] = []
            self._fit_params = params
            return FitResult(
                model_name=self.name,
                log_likelihood=loglik,
                aic=float(res.aic),
                bic=float(res.bic),
                params=params,
                n=len(returns),
                converged=converged,
                heavy_tailed=True,
                fit_success=converged,
                warnings=warnings,
            )
        except Exception as exc:
            raise DistributionFitError(f"GARCH-T fit failed: {exc}") from exc

    def sample(self, n_paths: int, n_steps: int, seed: int | None = None):
        try:
            from arch import arch_model  # type: ignore
            import numpy as np
        except Exception as exc:  # pragma: no cover - dependency guard
            raise DistributionFitError(f"GARCH-T requires 'arch' package: {exc}") from exc

        # Approximate sampling using unconditional variance to avoid slow per-path simulation
        p = self._fit_params or {}
        omega = float(p.get("omega", 1e-5))
        alpha1 = float(p.get("alpha[1]", 0.05))
        beta1 = float(p.get("beta[1]", 0.9))
        nu = float(p.get("nu", 8.0))
        denom = max(1e-6, 1.0 - alpha1 - beta1)
        sigma = float(np.sqrt(omega / denom))
        rng = np.random.default_rng(seed)
        return rng.standard_t(df=nu, size=(n_paths, n_steps)) * sigma

    def log_likelihood(self) -> float:
        raise DistributionFitError("GARCH-T log-likelihood not available (not implemented)")


__all__ = ["GarchTFitter"]
