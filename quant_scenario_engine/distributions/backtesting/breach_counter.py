"""Breach counter for VaR backtesting (US6a AS4, T154)."""

from __future__ import annotations

import numpy as np


def count_breaches(returns: np.ndarray, var_level: float) -> tuple[int, list[bool]]:
    breaches_bool = (returns < var_level).tolist()
    return int(sum(breaches_bool)), breaches_bool


__all__ = ["count_breaches"]
