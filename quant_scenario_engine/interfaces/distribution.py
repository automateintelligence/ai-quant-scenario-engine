"""Distribution interface for return models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

Estimator = Literal["mle", "gmm"]
FitStatus = Literal["success", "warn", "fail"]


@dataclass
class DistributionMetadata:
    estimator: Optional[Estimator] = None
    loglik: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    fit_status: FitStatus = "warn"
    min_samples: int = 60


class ReturnDistribution(ABC):
    """Base class for all unconditional or state-conditioned return models."""

    metadata: DistributionMetadata

    def __init__(self) -> None:
        self.metadata = DistributionMetadata()

    @abstractmethod
    def fit(self, returns) -> None:
        """Fit parameters from 1D array of log returns."""

    @abstractmethod
    def sample(self, n_paths: int, n_steps: int, seed: Optional[int] = None):
        """Produce log-return matrix of shape (n_paths, n_steps)."""

