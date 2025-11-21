"""Configuration validation helpers (FR-002, FR-014)."""

from __future__ import annotations

from typing import Dict, Any

from qse.distributions.regime_loader import RegimeNotFoundError


def ensure_valid_regime(regime: str, regimes_cfg: Dict[str, Any]) -> None:
    """Raise when regime label is unknown."""
    if regime not in regimes_cfg:
        available = ", ".join(sorted(regimes_cfg.keys())) if regimes_cfg else "none"
        raise RegimeNotFoundError(f"Unknown regime '{regime}'. Available: {available}")


def ensure_valid_regime_mode(mode: str) -> str:
    """Normalize and validate regime_mode."""
    normalized = mode.lower()
    if normalized not in {"table", "calibrated", "explicit"}:
        raise ValueError("regime_mode must be one of: table, calibrated, explicit")
    return normalized


__all__ = ["ensure_valid_regime", "ensure_valid_regime_mode"]
