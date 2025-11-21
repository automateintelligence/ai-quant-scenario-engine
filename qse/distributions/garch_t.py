"""GARCH(1,1)-t distribution (optional, US7/advanced features)."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis
from numpy.random import PCG64, Generator

from qse.distributions.stationarity import check_stationarity
from qse.distributions.validation import enforce_convergence, validate_returns
from qse.exceptions import DependencyError, DistributionFitError
from qse.interfaces.distribution import DistributionMetadata, ReturnDistribution


class GarchTDistribution(ReturnDistribution):
    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._params = None
        self._last_return: float | None = None

    def fit(self, returns: np.ndarray, min_samples: int = 252) -> None:
        validate_returns(returns, min_samples)
        stationarity = check_stationarity(returns)
        if stationarity.recommendation.startswith("difference"):
            returns = np.diff(returns)

        try:
            from arch.univariate import arch_model  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise DependencyError("arch library is required for GarchTDistribution") from exc

        model = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        res = model.fit(disp="off", show_warning=False)

        self._model = model
        self._params = res.params
        self._last_return = float(returns[-1])

        enforce_convergence(res.params.to_dict())
        self.metadata = DistributionMetadata(
            estimator="mle",
            loglik=float(res.loglikelihood),
            aic=float(res.aic),
            bic=float(res.bic),
            fit_status="success",
            min_samples=min_samples,
            excess_kurtosis=float(kurtosis(res.std_resid, fisher=True)),
            heavy_tail_warning=False,
        )

    def sample(self, n_paths: int, n_steps: int, seed: int | None = None) -> np.ndarray:
        if self._model is None or self._params is None or self._last_return is None:
            raise DistributionFitError("Model not fit")

        rng = Generator(PCG64(seed)) if seed is not None else None
        outputs = np.empty((n_paths, n_steps), dtype=float)

        for i in range(n_paths):
            try:
                if rng is not None:
                    np.random.seed(int(rng.integers(0, 2**32 - 1)))
                sim = self._model.simulate(
                    self._params,
                    nobs=n_steps,
                    burn=0,
                )
            except Exception as exc:  # pragma: no cover - library-layer errors
                raise DistributionFitError(f"Failed to simulate GARCH-T: {exc}") from exc

            sim_data = getattr(sim, "data", None)
            if sim_data is None and hasattr(sim, "simulated_data"):
                sim_data = sim.simulated_data
            data_arr = np.asarray(sim_data).reshape(-1)
            outputs[i] = data_arr[-n_steps:]

        return outputs
