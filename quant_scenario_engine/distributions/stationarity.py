"""Stationarity test utilities (placeholders)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  # pragma: no cover - optional import fallback
    adfuller = None  # type: ignore
    kpss = None  # type: ignore


@dataclass
class StationarityResult:
    adf_pvalue: Optional[float]
    kpss_pvalue: Optional[float]
    recommendation: str


def check_stationarity(series: np.ndarray, alpha: float = 0.05) -> StationarityResult:
    adf_pvalue = None
    kpss_pvalue = None
    recommendation = "unknown"

    if adfuller:
        adf_pvalue = float(adfuller(series, autolag="AIC")[1])
    if kpss:
        kpss_pvalue = float(kpss(series, nlags="auto")[1])

    if adf_pvalue is not None and adf_pvalue < alpha:
        recommendation = "difference_if_needed"
    elif kpss_pvalue is not None and kpss_pvalue < alpha:
        recommendation = "difference"
    else:
        recommendation = "stationary"

    return StationarityResult(adf_pvalue=adf_pvalue, kpss_pvalue=kpss_pvalue, recommendation=recommendation)

