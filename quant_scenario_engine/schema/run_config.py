"""Run configuration schema and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from quant_scenario_engine.exceptions import ConfigValidationError

CovEstimator = Literal["sample", "ledoit_wolf", "shrinkage_delta"]
VarMethod = Literal["parametric", "historical"]


@dataclass(slots=True)
class RunConfig:
    n_paths: int
    n_steps: int
    seed: int
    distribution_model: str
    data_source: str
    covariance_estimator: CovEstimator = "sample"
    var_method: VarMethod = "historical"
    lookback_window: int = 252
    max_workers: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ConfigValidationError("n_paths must be > 0")
        if self.n_steps <= 0:
            raise ConfigValidationError("n_steps must be > 0")
        if self.seed is None:
            raise ConfigValidationError("seed is required for reproducibility")
        if self.covariance_estimator not in {"sample", "ledoit_wolf", "shrinkage_delta"}:
            raise ConfigValidationError("invalid covariance_estimator")
        if self.var_method not in {"parametric", "historical"}:
            raise ConfigValidationError("invalid var_method")
        if self.lookback_window <= 0:
            raise ConfigValidationError("lookback_window must be positive")
        if self.max_workers is not None and self.max_workers <= 0:
            raise ConfigValidationError("max_workers must be positive when set")

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return cls(**data)

    def to_dict(self) -> dict:
        return {
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "seed": self.seed,
            "distribution_model": self.distribution_model,
            "data_source": self.data_source,
            "covariance_estimator": self.covariance_estimator,
            "var_method": self.var_method,
            "lookback_window": self.lookback_window,
            "max_workers": self.max_workers,
        }

