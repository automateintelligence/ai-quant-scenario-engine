"""Regime loader supporting table, calibrated, and explicit modes (FR-013, FR-014)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RegimeParams:
    mean_daily_return: float
    daily_vol: float
    skew: float
    kurtosis_excess: float
    source: str
    mode: str


class RegimeNotFoundError(ValueError):
    """Raised when a requested regime label is not present or cannot be resolved."""


def _read_table_entry(regimes_cfg: Dict[str, Any], regime: str) -> Dict[str, float]:
    if regime not in regimes_cfg:
        available = ", ".join(sorted(regimes_cfg.keys())) if regimes_cfg else "none"
        raise RegimeNotFoundError(f"Unknown regime '{regime}'. Available: {available}")
    entry = regimes_cfg[regime]
    required = ["mean_daily_return", "daily_vol", "skew", "kurtosis_excess"]
    missing = [k for k in required if k not in entry]
    if missing:
        raise RegimeNotFoundError(f"Regime '{regime}' missing fields: {', '.join(missing)}")
    return entry


def load_regime_params(
    regime: str,
    regimes_cfg: Dict[str, Any],
    mode: str = "table",
    overrides: Dict[str, float] | None = None,
) -> RegimeParams:
    """
    Load regime parameters according to mode.

    Modes:
    - table: read from config.regimes[regime]
    - calibrated: placeholder hook; falls back to table values for now (until calibration infra exists)
    - explicit: use overrides directly
    """

    normalized_mode = mode.lower()
    if normalized_mode == "explicit":
        if not overrides:
            raise RegimeNotFoundError("Explicit mode requires overrides for regime parameters")
        entry = overrides
        source = "explicit_override"
    elif normalized_mode == "calibrated":
        # Calibration pipeline not yet implemented; use table values as fallback
        entry = _read_table_entry(regimes_cfg, regime)
        source = "calibrated_fallback_table"
    else:  # default table
        entry = _read_table_entry(regimes_cfg, regime)
        source = "table"

    return RegimeParams(
        mean_daily_return=float(entry["mean_daily_return"]),
        daily_vol=float(entry["daily_vol"]),
        skew=float(entry["skew"]),
        kurtosis_excess=float(entry["kurtosis_excess"]),
        source=source,
        mode=normalized_mode,
    )


__all__ = ["RegimeParams", "RegimeNotFoundError", "load_regime_params"]
