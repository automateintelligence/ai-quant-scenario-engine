"""Volatility clustering metric (US6a AS5, T158)."""

from __future__ import annotations

import numpy as np


def autocorr_squared_returns(returns: np.ndarray, lag: int = 1) -> float:
    sq = returns**2
    if len(sq) <= lag:
        return 0.0
    a = sq[lag:] - sq.mean()
    b = sq[:-lag] - sq.mean()
    denom = np.sum((sq - sq.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


__all__ = ["autocorr_squared_returns"]
