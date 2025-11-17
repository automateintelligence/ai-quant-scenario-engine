"""Screening result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from quant_scenario_engine.simulation.metrics import MetricsReport


@dataclass
class SymbolScreenResult:
    symbol: str
    metrics_unconditional: MetricsReport
    metrics_conditional: Optional[MetricsReport] = None
    episode_count: int | None = None
    rank_metric: float | None = None
    low_confidence: bool = False
    comparison: Optional[dict[str, Any]] = None


__all__ = ["SymbolScreenResult"]
