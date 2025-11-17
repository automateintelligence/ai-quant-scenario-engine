"""AR detection utilities using ACF/PACF heuristics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from statsmodels.tsa.stattools import acf, pacf
except Exception:  # pragma: no cover - optional
    acf = None  # type: ignore
    pacf = None  # type: ignore


@dataclass
class ARDetectionResult:
    order_suggestion: int
    acf_values: list
    pacf_values: list


def detect_ar_process(series: np.ndarray, max_lag: int = 10) -> ARDetectionResult:
    acf_values = []
    pacf_values = []
    if acf:
        acf_values = acf(series, nlags=max_lag).tolist()
    if pacf:
        pacf_values = pacf(series, nlags=max_lag, method="ywunbiased").tolist()

    order = 0
    for i, value in enumerate(pacf_values[1:], start=1):
        if abs(value) > 0.2:
            order = i
            break

    return ARDetectionResult(order_suggestion=order, acf_values=acf_values, pacf_values=pacf_values)

