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
        self._scale_factor = 1.0  # Rescaling factor from arch model

    def fit(self, returns: np.ndarray) -> FitResult:
        ensure_min_samples(returns, self.name)
        try:
            from arch import arch_model  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            raise DistributionFitError(f"GARCH-T requires 'arch' package: {exc}") from exc

        try:
            # Use rescaling for numerical stability (arch auto-rescales to ~unit variance)
            am = arch_model(
                returns,
                mean="Zero",  # Log returns should have near-zero mean
                vol="GARCH",
                p=1,
                o=0,
                q=1,
                dist="t",
                rescale=True,  # Enable auto-rescaling for numerical stability
            )

            # Fit with robust optimization settings
            res = am.fit(
                update_freq=0,
                disp="off",
                show_warning=False,
                options={
                    "maxiter": 1000,  # More iterations for convergence
                    "ftol": 1e-8,     # Function tolerance
                },
            )

            # Extract parameters and scale factor
            params = {k: float(v) for k, v in res.params.items()}
            loglik = float(res.loglikelihood)

            # Store the rescaling factor to convert back to original scale
            self._scale_factor = float(res.scale)

            # Check convergence
            converged_attr = getattr(res, "converged", None)
            converged = bool(converged_attr) if converged_attr is not None else bool(getattr(res, "convergence", 0) == 0)

            # Validate fitted parameters (detect degenerate solutions)
            warnings: list[str] = []
            omega = params.get("omega", 0.0)
            alpha1 = params.get("alpha[1]", 0.0)
            beta1 = params.get("beta[1]", 0.0)
            nu = params.get("nu", 5.0)

            # Check for degenerate GARCH parameters
            persistence = alpha1 + beta1
            if persistence < 0.1:
                warnings.append(
                    f"Low GARCH persistence (α+β={persistence:.4f}). "
                    "Model may not capture volatility clustering."
                )
                # Downgrade fit_success but don't fail completely
                converged = False

            if persistence >= 1.0:
                warnings.append(
                    f"Non-stationary GARCH (α+β={persistence:.4f} ≥ 1). "
                    "Variance process is explosive."
                )
                converged = False

            if omega <= 0 or alpha1 < 0 or beta1 < 0:
                warnings.append(
                    f"Invalid parameter signs: ω={omega:.6f}, α={alpha1:.6f}, β={beta1:.6f}"
                )
                converged = False

            if nu <= 2.0:
                warnings.append(
                    f"Student-t degrees of freedom too low (ν={nu:.2f} ≤ 2). "
                    "Variance may be undefined."
                )

            # Store fitted params for sampling
            self._fit_params = params

            return FitResult(
                model_name=self.name,
                log_likelihood=loglik,
                aic=float(res.aic),
                bic=float(res.bic),
                params=params,
                n=len(returns),
                converged=converged,
                heavy_tailed=True,  # Student-t always heavy-tailed if nu < 30
                fit_success=converged and not warnings,
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

        if not self._fit_params:
            raise DistributionFitError("GarchTFitter.sample called before fit")

        # Approximate sampling using unconditional variance
        # NOTE: This doesn't capture time-varying volatility (conditional GARCH dynamics)
        # but is much faster than per-path recursive simulation
        p = self._fit_params
        omega = float(p.get("omega", 1e-5))
        alpha1 = float(p.get("alpha[1]", 0.05))
        beta1 = float(p.get("beta[1]", 0.9))
        nu = float(p.get("nu", 8.0))

        # Check for degenerate parameters
        persistence = alpha1 + beta1
        if persistence < 0.01:
            # Degenerate GARCH: fall back to i.i.d. Student-t with empirical volatility
            # Use omega as the variance estimate (since α+β ≈ 0 means σ² ≈ ω)
            sigma_rescaled = float(np.sqrt(max(omega, 1e-8)))
        elif persistence >= 1.0:
            # Non-stationary: no unconditional variance exists
            # Use bounded approximation to avoid explosion
            sigma_rescaled = float(np.sqrt(omega / 0.01))  # Cap persistence at 0.99
        else:
            # Standard unconditional variance formula: σ² = ω / (1 - α - β)
            # Note: omega, alpha, beta are in rescaled space
            denom = 1.0 - persistence
            sigma_rescaled = float(np.sqrt(omega / denom))

        # Convert back to original scale
        sigma_original = sigma_rescaled * self._scale_factor

        # Generate i.i.d. Student-t samples with unconditional volatility
        rng = np.random.default_rng(seed)
        return rng.standard_t(df=nu, size=(n_paths, n_steps)) * sigma_original

    def log_likelihood(self) -> float:
        raise DistributionFitError("GARCH-T log-likelihood not available (not implemented)")


__all__ = ["GarchTFitter"]
