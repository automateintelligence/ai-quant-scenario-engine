"""Extreme move counter (US6a AS5, T160)."""

from __future__ import annotations

import numpy as np


def extreme_move_frequencies(returns: np.ndarray, thresholds=(0.03, 0.05)) -> dict:
    freqs = {}
    for th in thresholds:
        freqs[f"gt_{int(th*100)}pct"] = float(np.mean(np.abs(returns) > th))
    return freqs


__all__ = ["extreme_move_frequencies"]
