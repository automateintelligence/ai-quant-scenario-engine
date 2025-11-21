"""Regime entity for 009-option-optimizer.

Represents expected underlying stock behavior, influencing option spread exploration.
Future: 001-mvp-pipeline may import for regime-based distribution selection (US7 Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RegimeLabel = Literal[
    "neutral",
    "strong-neutral",
    "low-volatility",
    "volatility-dir-uncertain",
    "mild-bearish",
    "strong-bearish",
    "mild-bullish",
    "strong-bullish",
]

RegimeMode = Literal["table", "calibrated", "explicit"]
RegimeSource = Literal["config", "calibrated", "override"]


@dataclass
class Regime:
    """Qualitative market outlook mapped to quantitative distribution parameters.

    Three modes of operation:
    1. table: Look up pre-defined params from config.yml regime table
    2. calibrated: Bootstrap from historical bars matching regime characteristics
    3. explicit: User provides exact statistical parameters via CLI overrides

    Used by 009-option-optimizer to generate Monte Carlo paths reflecting trader's
    expected stock behavior (e.g., "strong-bullish" → mean_daily_return=0.02).
    """

    label: RegimeLabel
    mode: RegimeMode
    mean_daily_return: float
    daily_vol: float
    skew: float
    kurtosis_excess: float
    source: RegimeSource = "config"

    def __post_init__(self) -> None:
        if self.label not in {
            "neutral",
            "strong-neutral",
            "low-volatility",
            "volatility-dir-uncertain",
            "mild-bearish",
            "strong-bearish",
            "mild-bullish",
            "strong-bullish",
        }:
            raise ValueError(f"Unknown regime label: {self.label}")

        if self.mode not in {"table", "calibrated", "explicit"}:
            raise ValueError(f"Invalid regime mode: {self.mode}")

        # Sanity checks on parameters
        if abs(self.mean_daily_return) > 0.10:  # 10% daily return is extreme
            raise ValueError(f"mean_daily_return {self.mean_daily_return} looks implausible")

        if self.daily_vol <= 0 or self.daily_vol > 0.30:  # 30% daily vol is extreme
            raise ValueError(f"daily_vol {self.daily_vol} looks implausible")

        if abs(self.skew) > 5:
            raise ValueError(f"skew {self.skew} looks implausible")

        if abs(self.kurtosis_excess) > 50:
            raise ValueError(f"kurtosis_excess {self.kurtosis_excess} looks implausible")

    def compound_for_horizon(self, trade_horizon: int) -> dict:
        """Compound daily parameters over multi-day horizon.

        For trade_horizon > 1, return distribution should compound returns
        across days, not scale linearly (i.e., 3-day mean ≠ 3 × daily_mean).

        Returns:
            Dict with horizon-adjusted params for distribution generation
        """
        # Simple compounding model (can be refined with autocorrelation)
        return {
            "mean_return": self.mean_daily_return * trade_horizon,
            "vol": self.daily_vol * (trade_horizon ** 0.5),  # Sqrt-time scaling
            "skew": self.skew,  # Skew/kurtosis don't scale simply
            "kurtosis_excess": self.kurtosis_excess,
        }
