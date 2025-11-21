"""Shared option leg representation for both 001 and 009.

Harmonizes:
- 001-mvp-pipeline: OptionSpec (single-leg wrapper)
- 009-option-optimizer: CandidateStructure.legs[] (multi-leg structures)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]
Side = Literal["long", "short"]


@dataclass(slots=True)
class OptionLeg:
    """Single option leg specification.

    Used by:
    - 001 OptionSpec: Wraps single leg for stock-vs-option comparison
    - 009 CandidateStructure: Multiple legs for spreads/condors/strangles

    Design decision: Quantity is always positive, sign encoded in 'side' field.
    This avoids sign convention confusion (contracts=-2 vs side='short', quantity=2).
    """

    option_type: OptionType  # 'call' or 'put'
    strike: float  # Always numeric (resolve "atm", "+0.05" before creating leg)
    maturity_days: int  # Days to expiration at entry
    implied_vol: float  # Annualized IV
    side: Side  # 'long' (buy) or 'short' (sell)
    quantity: int  # Always positive, sign in 'side'
    risk_free_rate: float = 0.02  # Annualized
    dividend_yield: float = 0.0  # Annualized

    def __post_init__(self) -> None:
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity_days <= 0:
            raise ValueError("maturity_days must be positive")
        if self.implied_vol <= 0 or self.implied_vol >= 5:
            raise ValueError("implied_vol must be between 0 and 5")
        if self.side not in {"long", "short"}:
            raise ValueError("side must be 'long' or 'short'")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive (sign encoded in 'side')")
        if self.risk_free_rate < -1:
            raise ValueError("risk_free_rate looks invalid")

    def signed_quantity(self) -> int:
        """Return signed quantity (positive for long, negative for short)."""
        return self.quantity if self.side == "long" else -self.quantity

    def notional(self, underlying_price: float) -> float:
        """Return notional exposure (unsigned)."""
        return abs(self.signed_quantity()) * underlying_price * 100  # Standard contract size
