"""Annualized volatility calculator (US6a AS5, T157)."""

from __future__ import annotations

import numpy as np


def annualized_volatility(returns: np.ndarray) -> float:
    daily_vol = float(np.std(returns, ddof=1))
    return daily_vol * (252.0 ** 0.5)


__all__ = ["annualized_volatility"]
