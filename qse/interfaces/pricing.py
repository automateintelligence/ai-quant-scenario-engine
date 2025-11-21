"""Option pricer interface.

Shared between 001-mvp-pipeline and 009-option-optimizer for consistent pricing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from qse.models.options import OptionSpec


@dataclass
class Greeks:
    """Option Greeks bundle."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float | None = None


class OptionPricer(ABC):
    """Base option pricer interface.

    Shared between:
    - 001-mvp-pipeline: Stock vs option comparison (US1)
    - 009-option-optimizer: Multi-leg option strategy scoring (US1-US8)

    Implementations:
    - BlackScholesPricer (baseline, European exercise)
    - BjerksundStenslandPricer (default for 009, American exercise)
    - HestonPricer (advanced, stochastic volatility - future)
    """

    @abstractmethod
    def price(self, path_slice: np.ndarray, option_spec: OptionSpec) -> np.ndarray:
        """Return mark-to-market price series for the provided path slice.

        Args:
            path_slice: Underlying price path of shape (n_steps,)
            option_spec: Option specification (strike, maturity, IV, etc.)

        Returns:
            Option price series of shape (n_steps,)
        """

    @abstractmethod
    def greeks(
        self,
        underlying: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend: float,
        iv: float,
        option_type: str
    ) -> Greeks:
        """Compute option Greeks at a single point.

        Args:
            underlying: Current underlying price
            strike: Strike price
            maturity: Time to maturity in years
            rate: Risk-free rate (annualized)
            dividend: Dividend yield (annualized)
            iv: Implied volatility (annualized)
            option_type: 'call' or 'put'

        Returns:
            Greeks bundle with delta, gamma, theta, vega, rho

        Note:
            Used by 009-option-optimizer for position Greek calculations.
            001-mvp-pipeline may use for analysis but not required for US1-US6.
        """

