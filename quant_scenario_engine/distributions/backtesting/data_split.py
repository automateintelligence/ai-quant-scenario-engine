"""Train/test splitter for distribution audit (US6a AS4, T150)."""

from __future__ import annotations

import numpy as np


def train_test_split(returns: np.ndarray, train_fraction: float = 0.7, min_train: int = 50):
    if returns is None or len(returns) < min_train + 10:
        raise ValueError("Not enough samples to split returns for backtesting")
    n = len(returns)
    n_train = max(min_train, int(train_fraction * n))
    r_train = returns[:n_train]
    r_test = returns[n_train:]
    return r_train, r_test


__all__ = ["train_test_split"]
