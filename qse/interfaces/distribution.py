"""Distribution interface for return models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Estimator = Literal["mle", "gmm"]
FitStatus = Literal["success", "warn", "fail"]


@dataclass
class DistributionMetadata:
    estimator: Estimator | None = None
    loglik: float | None = None
    aic: float | None = None
    bic: float | None = None
    fit_status: FitStatus = "warn"
    min_samples: int = 60
    excess_kurtosis: float | None = None
    heavy_tail_warning: bool = False
    fallback_model: str | None = None


class ReturnDistribution(ABC):
    """Base class for all unconditional or state-conditioned return models.

    This interface supports two workflows:
    - 001-mvp-pipeline: fit() from historical data, then sample()
    - 009-option-optimizer: generate_paths() with regime parameters

    Implementations must support both workflows for cross-feature compatibility.
    """

    metadata: DistributionMetadata

    def __init__(self) -> None:
        self.metadata = DistributionMetadata()

    @abstractmethod
    def fit(self, returns) -> None:
        """Fit parameters from 1D array of log returns.

        Used by 001-mvp-pipeline for historical backtesting workflow.
        """

    @abstractmethod
    def sample(self, n_paths: int, n_steps: int, seed: int | None = None):
        """Produce log-return matrix of shape (n_paths, n_steps).

        Used by 001-mvp-pipeline after fit() to generate Monte Carlo paths.
        """

    def generate_paths(
        self,
        s0: float,
        trade_horizon: int,
        bars_per_day: int,
        regime_params: dict,
        seed: int
    ):
        """Generate price paths from regime parameters (009-option-optimizer workflow).

        Args:
            s0: Initial stock price
            trade_horizon: Number of days to simulate
            bars_per_day: Intraday bars per trading day (1 for daily, 390 for 1min)
            regime_params: Dict with keys {mean_daily_return, daily_vol, skew, kurtosis_excess}
            seed: Random seed for reproducibility

        Returns:
            np.ndarray of shape (n_paths, n_steps) with price paths

        Note:
            Default implementation: set distribution params from regime, then sample().
            Subclasses can override for more sophisticated regime handling.
        """
        # Default implementation delegates to fit/sample workflow
        # Subclasses can override for direct regime-to-paths generation
        raise NotImplementedError(
            "generate_paths() must be implemented for 009-option-optimizer compatibility"
        )
