"""Kupiec unconditional coverage test (US6a AS4, T151)."""

from __future__ import annotations

import math


def kupiec_pvalue(n_obs: int, n_breaches: int, alpha: float) -> float:
    """Return p-value for Kupiec unconditional coverage test."""
    if n_obs <= 0:
        return 1.0
    pi = n_breaches / n_obs
    if pi == 0 or pi == 1:
        return 0.0
    lr_stat = -2 * (
        math.log((1 - alpha) ** (n_obs - n_breaches) * alpha ** n_breaches)
        - math.log((1 - pi) ** (n_obs - n_breaches) * pi ** n_breaches)
    )
    # asymptotic chi-square with 1 dof: pvalue = exp(-lr/2)
    return math.exp(-lr_stat / 2)


__all__ = ["kupiec_pvalue"]
