"""Christoffersen independence test (US6a AS4, T152)."""

from __future__ import annotations

import math
from typing import Sequence


def christoffersen_pvalue(breaches: Sequence[bool]) -> float:
    """Independence test for breach indicator sequence. Returns p-value."""
    if not breaches:
        return 1.0
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(breaches[:-1], breaches[1:]):
        if not prev and not curr:
            n00 += 1
        elif not prev and curr:
            n01 += 1
        elif prev and not curr:
            n10 += 1
        else:
            n11 += 1

    def _p(x, y):
        denom = (x + y) or 1
        return y / denom

    pi0 = _p(n00, n01)
    pi1 = _p(n10, n11)
    pi = _p(n00 + n10, n01 + n11)

    # Likelihood ratio
    def _ll(a, b):
        if a + b == 0:
            return 0.0
        return (a * math.log(max(1e-12, 1 - b)) + b * math.log(max(1e-12, b)))

    ll_ind = _ll(n00 + n10, n01 + n11)
    ll_dep = _ll(n00, pi0) + _ll(n01, 1 - pi0) + _ll(n10, pi1) + _ll(n11, 1 - pi1)
    lr = -2 * (ll_ind - ll_dep)

    # asymptotic chi-square with 1 dof: pvalue = exp(-lr/2)
    return math.exp(-lr / 2)


__all__ = ["christoffersen_pvalue"]
