"""Normal distribution model (US7)."""

from __future__ import annotations

import numpy as np
from numpy.random import PCG64, Generator
from scipy.stats import kurtosis, norm

from qse.distributions.stationarity import check_stationarity
from qse.distributions.validation import (
    enforce_convergence,
    heavy_tail_status,
    validate_params_bounds,
    validate_returns,
)
from qse.exceptions import DistributionFitError
from qse.interfaces.distribution import DistributionMetadata, ReturnDistribution


class NormalDistribution(ReturnDistribution):
    def __init__(self) -> None:
        super().__init__()
        self.loc: float | None = None
        self.scale: float | None = None

    def fit(self, returns: np.ndarray, min_samples: int = 60) -> None:
        validate_returns(returns, min_samples)
        stationarity = check_stationarity(returns)
        if stationarity.recommendation.startswith("difference"):
            returns = np.diff(returns)

        loc, scale = norm.fit(returns)
        self.loc, self.scale = float(loc), float(scale)
        enforce_convergence({"loc": self.loc, "scale": self.scale})
        validate_params_bounds({"scale": self.scale}, {"scale": (1e-9, 10.0)})
        excess_kurt = float(kurtosis(returns, fisher=True))
        # normal has kurtosis 0; allow heavy_tail_status to warn
        _, warn = heavy_tail_status(excess_kurt)
        self.metadata = DistributionMetadata(
            estimator="mle",
            loglik=float(norm.logpdf(returns, loc=loc, scale=scale).sum()),
            aic=float(2 * 2 - 2 * norm.logpdf(returns, loc=loc, scale=scale).sum()),
            bic=float(len(returns) * np.log(len(returns)) - 2 * norm.logpdf(returns, loc=loc, scale=scale).sum()),
            fit_status="success",
            min_samples=min_samples,
            excess_kurtosis=excess_kurt,
            heavy_tail_warning=warn,
        )

    def sample(self, n_paths: int, n_steps: int, seed: int | None = None) -> np.ndarray:
        if self.loc is None or self.scale is None:
            raise DistributionFitError("Model not fit")
        rng = Generator(PCG64(seed)) if seed is not None else np.random.default_rng()
        return rng.normal(self.loc, self.scale, size=(n_paths, n_steps))
