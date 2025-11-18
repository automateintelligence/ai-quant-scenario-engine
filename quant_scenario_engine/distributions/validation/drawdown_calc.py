"""Maximum drawdown calculator (US6a AS5, T159)."""

from __future__ import annotations

import numpy as np


def max_drawdown(price_paths: np.ndarray) -> float:
    peak = np.maximum.accumulate(price_paths)
    dd = (price_paths - peak) / peak
    return float(dd.min())


__all__ = ["max_drawdown"]
