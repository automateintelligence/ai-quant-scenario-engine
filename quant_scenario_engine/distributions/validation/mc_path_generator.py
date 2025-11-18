"""Monte Carlo path generator for audit validation (US6a AS5, T156)."""

from __future__ import annotations

import numpy as np

from quant_scenario_engine.interfaces.distribution import ReturnDistribution


def generate_paths(distribution: ReturnDistribution, n_paths: int = 10_000, n_steps: int = 252, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Distribution.sample uses its own RNG; rely on provided seed for determinism when supported
    paths = distribution.sample(n_paths=n_paths, n_steps=n_steps, seed=seed)
    return paths


__all__ = ["generate_paths"]
