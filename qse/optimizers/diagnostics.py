"""Diagnostics helpers for optimizer outputs (FR-054/FR-055/FR-075)."""

from __future__ import annotations

from typing import Dict


def empty_result_diagnostics(rejections: Dict[str, int], stage_counts: Dict[str, int], hints: str | None = None) -> Dict:
    """Build diagnostics when no candidates survive."""
    return {
        "stage_counts": stage_counts,
        "rejection_breakdown": rejections,
        "hints": hints or "No candidates survived filters; consider relaxing constraints.",
    }


def adaptive_diagnostics(epnl_ci_halfwidth: float, pop_ci_halfwidth: float, status: str, path_count: int) -> Dict:
    """Diagnostics for adaptive path adjustments."""
    return {
        "epnl_ci_halfwidth": epnl_ci_halfwidth,
        "pop_ci_halfwidth": pop_ci_halfwidth,
        "path_status": status,
        "path_count": path_count,
    }


__all__ = ["empty_result_diagnostics", "adaptive_diagnostics"]
