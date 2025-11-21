"""PyVollib option pricer wrapper (optional, US7)."""

from __future__ import annotations

import numpy as np

from qse.exceptions import DependencyError, PricingError
from qse.interfaces.pricing import OptionPricer
from qse.models.options import OptionSpec


class PyVollibPricer(OptionPricer):
    def __init__(self) -> None:
        try:  # type: ignore
            from py_vollib.black_scholes_merton import black_scholes_merton  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise DependencyError("py_vollib is required for PyVollibPricer") from exc

    def price(self, path_slice: np.ndarray, option_spec: OptionSpec) -> np.ndarray:
        try:
            from py_vollib.black_scholes_merton import black_scholes_merton
            from py_vollib.black_scholes_merton import greek  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise DependencyError("py_vollib is required for PyVollibPricer") from exc

        s = np.asarray(path_slice, dtype=float)
        if s.ndim != 1 or s.size == 0:
            raise PricingError("path_slice must be 1-D and non-empty")
        if np.any(s <= 0):
            raise PricingError("Non-positive spot encountered")

        # Assume zero carry/borrow for MVP
        r = float(option_spec.risk_free_rate)
        q = 0.0
        sigma = float(option_spec.implied_vol)
        t = max(option_spec.maturity_days, 1) / 252.0

        try:
            prices = np.array(
                [
                    black_scholes_merton(option_spec.option_type[0], float(spot), float(option_spec.strike), t, r, sigma, q)
                    for spot in s
                ]
            ) * option_spec.contracts
        except Exception as exc:  # pragma: no cover - library errors
            raise PricingError(f"py_vollib pricing failed: {exc}") from exc

        return prices
