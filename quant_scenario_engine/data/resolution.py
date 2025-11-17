"""Resolution tier selection utilities."""

from __future__ import annotations

from typing import Literal

Resolution = Literal["1d", "5m", "1m"]


def select_resolution(for_use: str) -> Resolution:
    """Choose resolution tier based on use case.

    - distribution_fitting -> daily
    - backtesting -> 5m
    - live -> 1m
    """

    mapping = {
        "distribution": "1d",
        "backtest": "5m",
        "live": "1m",
    }
    return mapping.get(for_use, "1d")  # default to safest tier

