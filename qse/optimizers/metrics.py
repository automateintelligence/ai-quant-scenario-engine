"""Adaptive CI computation for optimizer (FR-032/FR-033)."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

from qse.optimizer.metrics import AdaptiveCISettings, next_path_count


def compute_ci(samples: Iterable[float], alpha: float = 0.05) -> Tuple[float, float]:
    """Return (mean, halfwidth) using normal approximation."""
    arr = np.fromiter(samples, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    z = 1.96 if math.isclose(alpha, 0.05, rel_tol=1e-3) else 1.96
    halfwidth = z * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, halfwidth


def adaptive_path_plan(
    epnl_samples: Iterable[float],
    pop_samples: Iterable[float],
    current_paths: int,
    settings: AdaptiveCISettings = AdaptiveCISettings(),
):
    """Compute CI half-widths and decide next path count."""
    _, epnl_hw = compute_ci(epnl_samples)
    _, pop_hw = compute_ci(pop_samples)
    next_paths, status = next_path_count(current_paths, epnl_hw, pop_hw, settings)
    diagnostics = {
        "epnl_ci_halfwidth": epnl_hw,
        "pop_ci_halfwidth": pop_hw,
        "path_status": status,
        "next_paths": next_paths,
    }
    return next_paths, diagnostics


__all__ = ["compute_ci", "adaptive_path_plan"]
