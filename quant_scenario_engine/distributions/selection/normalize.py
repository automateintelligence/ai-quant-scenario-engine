"""AIC normalization helper (US6a AS6, T164)."""

from __future__ import annotations

import numpy as np


def normalize_aic(aic_values):
    vals = np.array(list(aic_values), dtype=float)
    aic_min, aic_max = float(np.min(vals)), float(np.max(vals))
    denom = max(1e-12, aic_max - aic_min)
    return [(v - aic_min) / denom for v in vals]


__all__ = ["normalize_aic"]
