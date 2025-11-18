"""Shared models for distribution audit (US6a)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FitResult:
    model_name: str
    log_likelihood: float
    aic: float
    bic: float
    params: Dict[str, float]
    n: int
    converged: bool
    heavy_tailed: bool | None = None
    fit_success: bool = True
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    fit_message: Optional[str] = None


__all__ = ["FitResult"]
