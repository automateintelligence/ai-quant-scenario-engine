"""Composite scoring for distribution audit (US6a AS6, T165)."""

from __future__ import annotations

def composite_score(aic_norm: float, tail_error: float, var_penalty: float, cluster_error: float, weights=(0.2, 0.4, 0.3, 0.1)) -> float:
    w1, w2, w3, w4 = weights
    return (
        w1 * (-aic_norm)
        + w2 * (-tail_error)
        + w3 * (-var_penalty)
        + w4 * (-cluster_error)
    )


__all__ = ["composite_score"]
